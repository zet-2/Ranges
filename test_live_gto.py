#!/usr/bin/env python3
"""Focused deterministic tests for the owned-simulator live GTO router."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

import live_gto as live
from gto_oracle import (
    Action,
    ActionKind,
    ActionValue,
    ComboPolicy,
    EngineResponseError,
    OracleCache,
    PlayerRange,
    Position,
    SolveResult,
    SolverMetadata,
    WeightedCombo,
)
from preflop_observation import ObservationProvenance, ObservedPreflopState


BINARY_DIGEST = "d" * 64


def make_hu_handoff(
    *,
    hand_id: str = "owned-sim-hand-17",
    survivors: tuple[str, ...] = ("BB", "BTN"),
) -> ObservedPreflopState:
    folded = frozenset(
        position
        for position in ("UTG", "HJ", "CO", "BTN", "SB", "BB")
        if position not in survivors
    )
    return ObservedPreflopState(
        actor=None,
        contributions=(
            ("UTG", Decimal(0)),
            ("HJ", Decimal(0)),
            ("CO", Decimal(0)),
            ("BTN", Decimal("2.5")),
            ("SB", Decimal("0.5")),
            ("BB", Decimal("2.5")),
        ),
        folded=folded,
        initial_stacks=tuple(
            (position, Decimal(100))
            for position in ("UTG", "HJ", "CO", "BTN", "SB", "BB")
        ),
        terminal=True,
        provenance=ObservationProvenance(
            source="preflop_to_flop_transition",
            hand_id=hand_id,
        ),
    )


def make_state(**changes) -> live.LiveDecisionState:
    state = live.LiveDecisionState(
        hand_id="owned-sim-hand-17",
        street="TURN",
        board=("2c", "7d", "Ts", "Jh"),
        hero_combo=("As", "Ad"),
        hero_position="BB",
        villain_position="BTN",
        hero_is_oop=True,
        active_villains=1,
        pot_bb=Decimal("9.25"),
        hero_stack_bb=Decimal("30.125"),
        villain_stack_bb=Decimal("18.375"),
        hero_current_bet_bb=Decimal(0),
        villain_current_bet_bb=Decimal(0),
        amount_to_call_bb=Decimal(0),
        legal_actions=("Check", "Bet"),
        street_root_confirmed=True,
        preflop_observation=make_hu_handoff(),
    )
    return replace(state, **changes)


def make_ranges(
    *,
    injected: bool = False,
    approximate: bool = False,
) -> live.RangeBundle:
    return live.RangeBundle(
        oop=PlayerRange(
            Position.OOP,
            (
                WeightedCombo(("As", "Ad")),
                WeightedCombo(("Kc", "Kh")),
            ),
        ),
        ip=PlayerRange(
            Position.IP,
            (
                WeightedCombo(("Qc", "Qh")),
                WeightedCombo(("9c", "9h")),
            ),
        ),
        profile_id="test-profile:BB.defend_vs_BTN.open",
        hero_combo_injected=injected,
        approximate=approximate,
    )


def make_result(
    spec,
    *,
    digest: str = BINARY_DIGEST,
    converged: bool = True,
) -> SolveResult:
    """Return complete combo-specific policies for any small fake spec."""

    weighted_combos = (
        spec.oop_range.combos
        if spec.acting_player is Position.OOP
        else spec.ip_range.combos
    )
    total_weight = sum(
        (combo.weight for combo in weighted_combos), start=Decimal(0)
    )
    remaining_reach = Decimal(1)
    policies = []
    for index, combo in enumerate(weighted_combos):
        if index == len(weighted_combos) - 1:
            reach = remaining_reach
        else:
            reach = combo.weight / total_weight
            remaining_reach -= reach
        values = []
        action_count = len(spec.tree.modeled_actions)
        remaining_frequency = Decimal(1)
        for action_index, action in enumerate(spec.tree.modeled_actions):
            if action_count == 2:
                frequency = (
                    Decimal("0.25")
                    if action.kind is ActionKind.CHECK
                    else Decimal("0.75")
                )
            elif action_index == action_count - 1:
                frequency = remaining_frequency
            else:
                frequency = Decimal(1) / Decimal(action_count)
                remaining_frequency -= frequency
            ev = Decimal("1.0") + Decimal(action_index) / Decimal(10)
            values.append(ActionValue(action, frequency, ev))
        policies.append(
            ComboPolicy(
                combo.cards,
                reach,
                Decimal("0.5"),
                tuple(values),
            )
        )
    return SolveResult.for_spec(
        spec,
        tuple(policies),
        SolverMetadata(
            solver_name=spec.parameters.solver_name,
            solver_version=spec.parameters.solver_commit,
            iterations=240,
            elapsed_seconds=Decimal("0.08"),
            exploitability=Decimal("0.1"),
            converged=converged,
            extra=(
                ("binary_sha256", digest),
                ("execution_context", "owned_simulator"),
                ("exploitability_pct_of_pot", "0.2"),
            ),
        ),
    )


class StubRangeProvider:
    def __init__(self, bundle=None):
        self.bundle = bundle or make_ranges()
        self.calls = []

    def ranges_for(self, state):
        self.calls.append(state)
        return self.bundle


class SpyEngineFactory:
    """Factory/client fake with one aggregate call log."""

    class Client:
        def __init__(self, owner):
            self.owner = owner

        @property
        def binary_sha256(self):
            self.owner.digest_reads += 1
            return self.owner.digest

        def solve(self, spec):
            self.owner.solve_calls.append(spec)
            return self.owner.result_factory(spec)

    def __init__(self, *, digest=BINARY_DIGEST, result_factory=None):
        self.digest = digest
        self.result_factory = result_factory or (
            lambda spec: make_result(spec, digest=self.digest)
        )
        self.factory_calls = []
        self.digest_reads = 0
        self.solve_calls = []

    def __call__(self, binary, **kwargs):
        self.factory_calls.append((Path(binary), kwargs))
        return self.Client(self)


class RangeExpansionTests(unittest.TestCase):
    def test_expands_pairs_suited_offsuit_ranges_weights_and_blockers(self):
        broad = live.expand_range_text("AA, AKs, AKo")
        self.assertEqual(22, len(broad))  # 6 + 4 + 12

        ladders = live.expand_range_text("TT-88, A5s-A2s")
        self.assertEqual(34, len(ladders))  # 3*6 + 4*4

        weighted = live.expand_range_text("AKs:0.25, AsKs:0.75")
        by_combo = {combo.cards: combo.weight for combo in weighted}
        exact = WeightedCombo(("As", "Ks")).cards
        self.assertEqual(4, len(weighted))
        self.assertEqual(Decimal("0.75"), by_combo[exact])
        self.assertEqual(
            {Decimal("0.25"), Decimal("0.75")}, set(by_combo.values())
        )

        blocked = live.expand_range_text("AA", dead_cards=("As",))
        self.assertEqual(3, len(blocked))
        self.assertTrue(all("As" not in combo.cards for combo in blocked))

    def test_rejects_invalid_range_notation_and_empty_blocked_range(self):
        for text in ("", "AAs", "AKx", "AKs:0", "AKs:NaN"):
            with self.subTest(text=text), self.assertRaises(live.LiveGTORangeError):
                live.expand_range_text(text)
        with self.assertRaises(live.LiveGTORangeError):
            live.expand_range_text("AsKs", dead_cards=("As",))

    def test_position_provider_swaps_roles_and_injects_hero_into_ip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ranges.json"
            path.write_text(
                json.dumps(
                    {
                        "ranges": {
                            "6max": {
                                "BB": {"defend_vs_open": "AA"},
                                "BTN": {"open": "KK"},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            state = make_state(
                hero_combo=("Qc", "Qh"),
                hero_position="BTN",
                villain_position="BB",
                hero_is_oop=False,
                action_history=("CHECK",),
            )
            ranges = live.PositionChartRangeProvider(path).ranges_for(state)

        hero_combo = WeightedCombo(state.hero_combo).cards
        self.assertNotIn(hero_combo, {combo.cards for combo in ranges.oop.combos})
        self.assertIn(hero_combo, {combo.cards for combo in ranges.ip.combos})
        self.assertTrue(ranges.hero_combo_injected)
        self.assertIn("BB.defend_vs_open_vs_BTN.open", ranges.profile_id)
        self.assertTrue(ranges.approximate)
        self.assertIn("static position charts", ranges.provenance)
        self.assertTrue(
            any("action path" in note for note in ranges.approximations)
        )

    def test_position_provider_labels_the_chart_entry_actually_used(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ranges.json"
            path.write_text(
                json.dumps(
                    {
                        "ranges": {
                            "6max": {
                                "BB": {"open": "AA"},
                                "BTN": {"defend_vs_open": "KK"},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            state = make_state(
                hero_combo=("As", "Ad"),
                hero_position="BB",
                villain_position="BTN",
            )
            ranges = live.PositionChartRangeProvider(path).ranges_for(state)

        self.assertIn("BB.open_vs_BTN.defend_vs_open", ranges.profile_id)
        self.assertIn("BB.open and BTN.defend_vs_open", ranges.provenance)

    def test_position_charts_require_verified_hu_preflop_handoff(self):
        provider = live.PositionChartRangeProvider("unused-ranges.json")

        with self.assertRaisesRegex(
            live.LiveGTORangeError,
            "handoff is missing",
        ):
            provider.ranges_for(make_state(preflop_observation=None))


class ConfigurationTests(unittest.TestCase):
    def test_live_use_requires_explicit_owned_simulator_opt_in(self):
        with self.assertRaisesRegex(
            live.LiveGTOConfigurationError, "GTO_OWNED_SIMULATOR_ACK"
        ):
            live.LiveGTOConfig(enabled=True)

        with tempfile.TemporaryDirectory() as directory:
            env = {
                "GTO_LIVE_ENABLED": "1",
                "GTO_OWNED_SIMULATOR_ACK": "yes",
                "GTO_ENGINE_PATH": "bin/engine",
                "GTO_CACHE_PATH": "cache/live.sqlite3",
                "GTO_RANGE_DATA_PATH": "ranges.json",
                "GTO_RANGE_SOURCE": "blueprint",
                "PREFLOP_BLUEPRINT_CACHE_PATH": "cache/preflop",
                "PREFLOP_BLUEPRINT_MATCH_MODE": "abstract",
                "PREFLOP_BLUEPRINT_SIZE_TOLERANCE_PCT": "10",
                "PREFLOP_BLUEPRINT_MAX_STACK_ERROR_PCT": "20",
                "GTO_TURN_TIMEOUT_SECONDS": "0.75",
                "GTO_FLOP_CACHE_ONLY": "true",
            }
            with mock.patch.dict(os.environ, env, clear=True):
                config = live.LiveGTOConfig.from_env(directory)
        root = Path(directory)
        self.assertTrue(config.enabled)
        self.assertTrue(config.owned_simulator_acknowledged)
        self.assertEqual(root / "bin/engine", config.engine_path)
        self.assertEqual(root / "cache/live.sqlite3", config.cache_path)
        self.assertEqual(root / "ranges.json", config.range_data_path)
        self.assertEqual("blueprint", config.range_source)
        self.assertEqual(root / "cache/preflop", config.blueprint_cache_path)
        self.assertEqual("abstract", config.blueprint_match_mode)
        self.assertEqual(Decimal(10), config.blueprint_size_tolerance_pct)
        self.assertEqual(Decimal(20), config.blueprint_max_stack_error_pct)
        self.assertEqual(Decimal("0.75"), config.turn_timeout_seconds)
        self.assertTrue(config.flop_cache_only)

    def test_rake_configuration_rejects_nonfinite_values(self):
        for changes in (
            {"rake_rate_pct": Decimal("NaN")},
            {"rake_rate_pct": Decimal("Infinity")},
            {"rake_cap_bb": Decimal("NaN")},
            {"rake_cap_bb": Decimal("Infinity")},
        ):
            with self.subTest(changes=changes), self.assertRaises(
                live.LiveGTOConfigurationError
            ):
                live.LiveGTOConfig(**changes)

    def test_disabled_router_does_not_touch_ranges_or_engine(self):
        provider = StubRangeProvider()
        engine = SpyEngineFactory()
        outcome = live.LiveGTORouter(
            live.LiveGTOConfig(enabled=False),
            range_provider=provider,
            engine_factory=engine,
        ).evaluate(make_state())
        self.assertEqual(live.LiveGTOStatus.DISABLED, outcome.status)
        self.assertEqual([], provider.calls)
        self.assertEqual([], engine.factory_calls)
        self.assertEqual(0, engine.digest_reads)
        self.assertEqual([], engine.solve_calls)


class EligibilityRoutingTests(unittest.TestCase):
    def test_unsupported_nodes_have_zero_range_and_engine_access(self):
        with tempfile.TemporaryDirectory() as directory:
            config = live.LiveGTOConfig(
                enabled=True,
                owned_simulator_acknowledged=True,
                cache_path=Path(directory) / "never-created.sqlite3",
            )
            cases = (
                (
                    make_state(street="PREFLOP", board=()),
                    "preflop",
                ),
                (
                    make_state(active_villains=2),
                    "exactly one active villain",
                ),
                (
                    make_state(hero_is_oop=False),
                    "empty action history",
                ),
                (
                    make_state(
                        amount_to_call_bb=Decimal("1.5"),
                        legal_actions=("Fold", "Call", "Raise"),
                    ),
                    "not an untouched root node",
                ),
                (
                    make_state(villain_current_bet_bb=Decimal("1.5")),
                    "not an untouched root node",
                ),
                (
                    make_state(preflop_observation=None),
                    "handoff is missing",
                ),
                (
                    make_state(
                        preflop_observation=make_hu_handoff(
                            survivors=("BB", "BTN", "SB")
                        )
                    ),
                    "was not heads-up",
                ),
                (
                    make_state(
                        preflop_observation=None,
                        preflop_mapping_error=(
                            "postflop hand is not heads-up: found 3 live players"
                        ),
                    ),
                    "verified heads-up preflop handoff is unavailable",
                ),
                (
                    make_state(
                        preflop_observation=make_hu_handoff(
                            hand_id="previous-hand"
                        )
                    ),
                    "different hand",
                ),
            )
            for state, reason in cases:
                provider = StubRangeProvider()
                engine = SpyEngineFactory()
                with self.subTest(reason=reason):
                    outcome = live.LiveGTORouter(
                        config,
                        range_provider=provider,
                        engine_factory=engine,
                    ).evaluate(state)
                    self.assertEqual(live.LiveGTOStatus.UNSUPPORTED, outcome.status)
                    self.assertIn(reason, outcome.reason)
                    self.assertEqual([], provider.calls)
                    self.assertEqual([], engine.factory_calls)
                    self.assertEqual(0, engine.digest_reads)
                    self.assertEqual([], engine.solve_calls)
            self.assertFalse(config.cache_path.exists())


class DescendantNodeMappingTests(unittest.TestCase):
    def setUp(self):
        self.config = live.LiveGTOConfig(
            enabled=True,
            owned_simulator_acknowledged=True,
            chip_scale=100,
            bet_size_pct=Decimal("50"),
        )

    def test_ip_after_oop_check_targets_ip_check_bet_node(self):
        state = make_state(
            hero_combo=("Qc", "Qh"),
            hero_is_oop=False,
            action_history=("CHECK",),
            street_root_confirmed=False,
        )
        spec = live.build_live_spec(state, make_ranges(), self.config)

        self.assertEqual(Position.IP, spec.acting_player)
        self.assertEqual((Action(ActionKind.CHECK),), spec.tree.action_history)
        self.assertEqual(0, spec.tree.facing_bet)
        self.assertEqual(
            {ActionKind.CHECK, ActionKind.BET},
            set(spec.tree.legal_action_kinds),
        )

    def test_ip_facing_first_oop_bet_uses_exact_path_and_raise_to(self):
        state = make_state(
            hero_combo=("Qc", "Qh"),
            hero_is_oop=False,
            hero_current_bet_bb=Decimal(0),
            villain_current_bet_bb=Decimal("2"),
            amount_to_call_bb=Decimal("2"),
            legal_actions=("Fold", "Call", "Raise"),
            street_root_confirmed=False,
            action_history=("BET",),
            observed_bet_to_bb=Decimal("2"),
        )
        spec = live.build_live_spec(state, make_ranges(), self.config)

        self.assertEqual(Position.IP, spec.acting_player)
        self.assertEqual((Action(ActionKind.BET, 200),), spec.tree.action_history)
        self.assertEqual(200, spec.tree.facing_bet)
        self.assertEqual("200c", spec.parameters.bet_sizes.turn.oop.bet)
        self.assertEqual("200c", spec.parameters.bet_sizes.turn.ip.bet)
        self.assertEqual(
            {
                Action(ActionKind.FOLD),
                Action(ActionKind.CALL),
                Action(ActionKind.RAISE, 500),
            },
            set(spec.tree.modeled_actions),
        )

    def test_oop_facing_ip_bet_uses_check_bet_path(self):
        state = make_state(
            hero_current_bet_bb=Decimal(0),
            villain_current_bet_bb=Decimal("2"),
            amount_to_call_bb=Decimal("2"),
            legal_actions=("Fold", "Call", "Raise"),
            street_root_confirmed=False,
            action_history=("CHECK", "BET"),
            observed_bet_to_bb=Decimal("2"),
        )
        spec = live.build_live_spec(state, make_ranges(), self.config)

        self.assertEqual(Position.OOP, spec.acting_player)
        self.assertEqual(
            (Action(ActionKind.CHECK), Action(ActionKind.BET, 200)),
            spec.tree.action_history,
        )
        self.assertEqual(200, spec.tree.facing_bet)

    def test_facing_raise_is_clamped_to_all_in_like_upstream(self):
        state = make_state(
            hero_combo=("Qc", "Qh"),
            hero_is_oop=False,
            hero_stack_bb=Decimal("4"),
            villain_stack_bb=Decimal("6"),
            hero_current_bet_bb=Decimal(0),
            villain_current_bet_bb=Decimal("2"),
            amount_to_call_bb=Decimal("2"),
            legal_actions=("Fold", "Call", "Raise"),
            street_root_confirmed=False,
            action_history=("BET",),
            observed_bet_to_bb=Decimal("2"),
        )
        spec = live.build_live_spec(state, make_ranges(), self.config)
        self.assertIn(
            Action(ActionKind.ALL_IN, 400),
            spec.tree.modeled_actions,
        )
        self.assertNotIn(Action(ActionKind.RAISE, 500), spec.tree.modeled_actions)

    def test_all_in_only_no_wager_control_cannot_create_ordinary_bet(self):
        state = make_state(legal_actions=("Check", "All-in"))

        spec = live.build_live_spec(state, make_ranges(), self.config)

        self.assertEqual(
            {
                Action(ActionKind.CHECK),
                Action(ActionKind.ALL_IN, 1838),
            },
            set(spec.tree.modeled_actions),
        )
        self.assertNotIn(ActionKind.BET, spec.tree.legal_action_kinds)

    def test_all_in_only_facing_control_cannot_create_ordinary_raise(self):
        state = make_state(
            hero_combo=("Qc", "Qh"),
            hero_is_oop=False,
            hero_current_bet_bb=Decimal(0),
            villain_current_bet_bb=Decimal("2"),
            amount_to_call_bb=Decimal("2"),
            legal_actions=("Fold", "Call", "All-in"),
            street_root_confirmed=False,
            action_history=("BET",),
            observed_bet_to_bb=Decimal("2"),
        )

        spec = live.build_live_spec(state, make_ranges(), self.config)

        self.assertEqual(
            {
                Action(ActionKind.FOLD),
                Action(ActionKind.CALL),
                Action(ActionKind.ALL_IN, 1838),
            },
            set(spec.tree.modeled_actions),
        )
        self.assertNotIn(ActionKind.RAISE, spec.tree.legal_action_kinds)

    def test_descendant_action_labels_use_call_and_raise_semantics(self):
        self.assertEqual(
            ("Call", "2 BB"),
            live._action_label(
                Action(ActionKind.CALL),
                100,
                amount_to_call_bb=Decimal("2"),
            ),
        )
        self.assertEqual(
            ("Raise", "4 BB"),
            live._action_label(
                Action(ActionKind.ALL_IN, 400),
                100,
                amount_to_call_bb=Decimal("2"),
            ),
        )
        self.assertEqual(
            ("Bet", "4 BB"),
            live._action_label(
                Action(ActionKind.ALL_IN, 400),
                100,
                amount_to_call_bb=Decimal(0),
            ),
        )


class SupportedMappingAndCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cache_path = Path(self.temporary.name) / "nested" / "live.sqlite3"
        self.config = live.LiveGTOConfig(
            enabled=True,
            owned_simulator_acknowledged=True,
            engine_path=Path(self.temporary.name) / "fake-engine",
            cache_path=self.cache_path,
            chip_scale=100,
            bet_size_pct=Decimal("50"),
            turn_timeout_seconds=Decimal("0.75"),
            flop_cache_only=False,
            rake_rate_pct=Decimal("4.5"),
            rake_cap_bb=Decimal("0.25"),
        )

    def test_supported_state_maps_exactly_and_miss_then_hit_is_deterministic(self):
        state = make_state()
        ranges = make_ranges(injected=True)
        provider = StubRangeProvider(ranges)
        engine = SpyEngineFactory()
        router = live.LiveGTORouter(
            self.config,
            range_provider=provider,
            engine_factory=engine,
        )

        fresh = router.evaluate(state)
        cached = router.evaluate(state)

        self.assertEqual(live.LiveGTOStatus.SOLVED, fresh.status)
        self.assertEqual("GTO fresh", fresh.source)
        self.assertFalse(fresh.cache_hit)
        self.assertEqual(live.LiveGTOStatus.SOLVED, cached.status)
        self.assertEqual("GTO cache", cached.source)
        self.assertTrue(cached.cache_hit)
        self.assertEqual(1, len(engine.solve_calls))
        self.assertEqual(2, engine.digest_reads)
        self.assertEqual(2, len(engine.factory_calls))
        self.assertEqual(2, len(provider.calls))

        spec = fresh.spec
        self.assertIsNotNone(spec)
        self.assertEqual(live.Street.TURN, spec.street)
        self.assertEqual(state.board, spec.board)
        self.assertEqual(Position.OOP, spec.acting_player)
        self.assertEqual(ranges.oop, spec.oop_range)
        self.assertEqual(ranges.ip, spec.ip_range)
        self.assertEqual(925, spec.tree.pot)
        self.assertEqual(1838, spec.tree.effective_stack)
        self.assertEqual(0, spec.tree.facing_bet)
        self.assertEqual(Decimal("4.5"), spec.tree.rake_rate_pct)
        self.assertEqual(25, spec.tree.rake_cap)
        aggressive = next(
            action
            for action in spec.tree.modeled_actions
            if action.kind is not ActionKind.CHECK
        )
        self.assertEqual(ActionKind.BET, aggressive.kind)
        self.assertEqual(463, aggressive.amount)

        binary, kwargs = engine.factory_calls[0]
        self.assertEqual(self.config.engine_path, binary)
        self.assertFalse(kwargs["offline_only_acknowledged"])
        self.assertTrue(kwargs["owned_simulator_acknowledged"])
        self.assertEqual(Decimal("0.75"), kwargs["timeout_seconds"])

        # Full mixed policy is visible and the same hand/spec gets the same
        # selected action and roll regardless of fresh versus cached source.
        self.assertIn(
            "* **GTO mix:** Bet 4.63 BB 75.0% | Check 25.0%",
            fresh.analysis,
        )
        self.assertIn("observed Hero combo added", fresh.analysis)
        def selected_lines(analysis):
            selected = {}
            for line in analysis.splitlines():
                if line.startswith(
                    ("**Action:**", "**Size:**", "* **Stable roll:**")
                ):
                    key, value = line.replace("*", "").split(":", 1)
                    selected[key.strip()] = value.strip()
            return selected

        fresh_lines = selected_lines(fresh.analysis)
        cached_lines = selected_lines(cached.analysis)
        self.assertEqual(fresh_lines, cached_lines)

        hero_combo = WeightedCombo(state.hero_combo).cards
        seed = f"{state.hand_id}:{spec.cache_key}:{','.join(hero_combo)}"
        roll = Decimal(
            int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:8], "big")
        ) / Decimal(2**64)
        expected_action = "Bet" if roll < Decimal("0.75") else "Check"
        expected_size = "4.63 BB" if expected_action == "Bet" else "0"
        self.assertEqual(expected_action, fresh_lines["Action"])
        self.assertEqual(expected_size, fresh_lines["Size"])
        self.assertEqual(
            f"{float(roll) * 100:.2f}%", fresh_lines["Stable roll"]
        )

        with OracleCache(self.cache_path) as cache:
            self.assertEqual(1, cache.result_count())

    def test_flop_cache_only_miss_never_solves_but_cached_node_is_used(self):
        state = make_state(street="FLOP", board=("2c", "7d", "Ts"))
        ranges = make_ranges()
        provider = StubRangeProvider(ranges)
        engine = SpyEngineFactory()
        config = replace(self.config, flop_cache_only=True)
        router = live.LiveGTORouter(
            config,
            range_provider=provider,
            engine_factory=engine,
        )

        miss = router.evaluate(state)
        self.assertEqual(live.LiveGTOStatus.CACHE_MISS, miss.status)
        self.assertIn("cache-only", miss.reason)
        self.assertEqual([], engine.solve_calls)

        spec = live.build_live_spec(state, ranges, config)
        with OracleCache(self.cache_path) as cache:
            cache.put(spec, make_result(spec))

        hit = router.evaluate(state)
        self.assertEqual(live.LiveGTOStatus.SOLVED, hit.status)
        self.assertTrue(hit.cache_hit)
        self.assertEqual("GTO cache", hit.source)
        self.assertEqual([], engine.solve_calls)

    def test_unconverged_cache_hit_is_rejected_without_reusing_policy(self):
        state = make_state()
        ranges = make_ranges()
        spec = live.build_live_spec(state, ranges, self.config)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with OracleCache(self.cache_path) as cache:
            cache.put(spec, make_result(spec, converged=False))
        engine = SpyEngineFactory()

        outcome = live.LiveGTORouter(
            self.config,
            range_provider=StubRangeProvider(ranges),
            engine_factory=engine,
        ).evaluate(state)

        self.assertEqual(live.LiveGTOStatus.FAILED, outcome.status)
        self.assertIn("cached solver result stopped", outcome.reason)
        self.assertEqual([], engine.solve_calls)

    def test_corrupt_sqlite_cache_returns_failed_instead_of_raising(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_bytes(b"not a sqlite database")

        outcome = live.LiveGTORouter(
            self.config,
            range_provider=StubRangeProvider(make_ranges()),
            engine_factory=SpyEngineFactory(),
        ).evaluate(make_state())

        self.assertEqual(live.LiveGTOStatus.FAILED, outcome.status)
        self.assertIn("cache", outcome.reason)

    def test_cache_operational_failure_returns_failed_for_hybrid_fallback(self):
        with mock.patch.object(
            live,
            "OracleCache",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            outcome = live.LiveGTORouter(
                self.config,
                range_provider=StubRangeProvider(make_ranges()),
                engine_factory=SpyEngineFactory(),
            ).evaluate(make_state())

        self.assertEqual(live.LiveGTOStatus.FAILED, outcome.status)
        self.assertIn("database is locked", outcome.reason)

    def test_approximate_range_source_is_never_labeled_gto(self):
        outcome = live.LiveGTORouter(
            self.config,
            range_provider=StubRangeProvider(
                make_ranges(approximate=True)
            ),
            engine_factory=SpyEngineFactory(),
        ).evaluate(make_state())

        self.assertEqual(live.LiveGTOStatus.SOLVED, outcome.status)
        self.assertEqual("APPROXIMATE_SOLVER fresh", outcome.source)
        self.assertIn("**Approximate solver mix:**", outcome.analysis)
        self.assertNotIn("**GTO mix:**", outcome.analysis)

    def test_descendant_ip_policy_is_formatted_with_facing_sizes(self):
        state = make_state(
            hero_combo=("Qc", "Qh"),
            hero_is_oop=False,
            hero_current_bet_bb=Decimal(0),
            villain_current_bet_bb=Decimal("2"),
            amount_to_call_bb=Decimal("2"),
            legal_actions=("Fold", "Call", "Raise"),
            street_root_confirmed=False,
            action_history=("BET",),
            observed_bet_to_bb=Decimal("2"),
        )
        outcome = live.LiveGTORouter(
            self.config,
            range_provider=StubRangeProvider(make_ranges()),
            engine_factory=SpyEngineFactory(),
        ).evaluate(state)

        self.assertEqual(live.LiveGTOStatus.SOLVED, outcome.status)
        self.assertEqual(Position.IP, outcome.spec.acting_player)
        self.assertIn("Call 2 BB", outcome.analysis)
        self.assertIn("Raise 5 BB", outcome.analysis)

    def test_unreachable_hero_combo_is_explicitly_unsupported(self):
        state = make_state(
            hero_is_oop=False,
            action_history=("CHECK",),
            street_root_confirmed=False,
        )
        outcome = live.LiveGTORouter(
            self.config,
            range_provider=StubRangeProvider(make_ranges()),
            engine_factory=SpyEngineFactory(),
        ).evaluate(state)

        self.assertEqual(live.LiveGTOStatus.UNSUPPORTED, outcome.status)
        self.assertIn("unreachable", outcome.reason)

    def test_engine_unreachable_node_is_unsupported_not_failed(self):
        state = make_state(
            hero_combo=("Qc", "Qh"),
            hero_is_oop=False,
            hero_current_bet_bb=Decimal(0),
            villain_current_bet_bb=Decimal("2"),
            amount_to_call_bb=Decimal("2"),
            legal_actions=("Fold", "Call", "Raise"),
            street_root_confirmed=False,
            action_history=("BET",),
            observed_bet_to_bb=Decimal("2"),
        )

        def unreachable(_spec):
            raise EngineResponseError(
                "UNREACHABLE_NODE",
                "the current actor has no positive-reach combos at this node",
            )

        outcome = live.LiveGTORouter(
            self.config,
            range_provider=StubRangeProvider(make_ranges()),
            engine_factory=SpyEngineFactory(result_factory=unreachable),
        ).evaluate(state)

        self.assertEqual(live.LiveGTOStatus.UNSUPPORTED, outcome.status)
        self.assertIn("unreachable", outcome.reason)


if __name__ == "__main__":
    unittest.main()
