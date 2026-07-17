#!/usr/bin/env python3
"""Offline integration tests for the live six-max preflop blueprint path.

The fixture deliberately uses the production immutable blueprint models and a
small but internally coherent six-max action tree.  It never reads the network
or the real on-disk blueprint cache.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import hashlib
from types import SimpleNamespace
import unittest

import live_gto as live
from preflop_blueprint import (
    BlueprintAction,
    BlueprintManifest,
    BlueprintNetworkDisabledError,
    BlueprintNode,
    BlueprintSpot,
    CANONICAL_HAND_CLASSES,
    hand_class_combo_count,
)
from preflop_history import ActionStep
from preflop_observation import ObservationProvenance, ObservedPreflopState


POSITIONS = ("UTG", "HJ", "CO", "BTN", "SB", "BB")
GENERATED_AT = "2026-06-09T19:18:20.659Z"


def _weights(**values: str) -> tuple[tuple[str, Decimal], ...]:
    return tuple((hand_class, Decimal(value)) for hand_class, value in values.items())


def _action(
    label: str,
    *,
    kind: str,
    weights: tuple[tuple[str, Decimal], ...],
) -> BlueprintAction:
    combos = sum(
        (
            Decimal(hand_class_combo_count(hand_class)) * frequency
            for hand_class, frequency in weights
        ),
        Decimal(0),
    )
    return BlueprintAction(
        label=label,
        kind=kind,
        size_pct=Decimal(label[:-1]) if kind == "raise" else None,
        combos=combos,
        weights=weights,
        evs=None,
    )


def _node(history: str, *actions: BlueprintAction) -> BlueprintNode:
    totals = {hand_class: Decimal(0) for hand_class in CANONICAL_HAND_CLASSES}
    for action in actions:
        for hand_class, value in action.weights:
            totals[hand_class] += value
    residual = {
        hand_class: Decimal(1) - value
        for hand_class, value in totals.items()
        if value < 1
    }
    completed = list(actions)
    if residual:
        fold_index = next(
            (index for index, action in enumerate(completed) if action.kind == "fold"),
            None,
        )
        if fold_index is None:
            completed.append(
                _action("Fold", kind="fold", weights=tuple(residual.items()))
            )
        else:
            fold = completed[fold_index]
            merged = dict(fold.weights)
            for hand_class, value in residual.items():
                merged[hand_class] = merged.get(hand_class, Decimal(0)) + value
            completed[fold_index] = _action(
                "Fold", kind="fold", weights=tuple(merged.items())
            )
    payload_digest = hashlib.sha256(history.encode("ascii")).hexdigest()
    return BlueprintNode(
        game="nl",
        version=2,
        stack=100,
        history=history,
        actor=history.split("_")[-1],
        actions=tuple(completed),
        continuing_combos=sum(
            (action.combos for action in completed if action.kind != "fold"),
            Decimal(0),
        ),
        response_sha256=payload_digest,
    )


def _coherent_nodes() -> dict[str, BlueprintNode]:
    """Return a root mix plus BTN-open/SB-call/BB-fold HU handoff."""

    nodes = {
        "UTG": _node(
            "UTG",
            _action(
                "Fold",
                kind="fold",
                weights=_weights(AKs="0.25", **{"72o": "1"}),
            ),
            _action(
                "60%",
                kind="raise",
                weights=_weights(AA="1", AKs="0.75"),
            ),
        ),
        "UTG_Fold_HJ": _node(
            "UTG_Fold_HJ",
            _action("Fold", kind="fold", weights=_weights(AA="1")),
        ),
        "UTG_60%_HJ": _node(
            "UTG_60%_HJ",
            _action(
                "Call",
                kind="call",
                weights=_weights(AKs="0.5", QQ="0.25"),
            ),
        ),
        "UTG_Fold_HJ_Fold_CO": _node(
            "UTG_Fold_HJ_Fold_CO",
            _action("Fold", kind="fold", weights=_weights(AA="1")),
        ),
        "UTG_Fold_HJ_Fold_CO_Fold_BTN": _node(
            "UTG_Fold_HJ_Fold_CO_Fold_BTN",
            _action(
                "60%",
                kind="raise",
                weights=_weights(AA="0.8", KQo="0.4"),
            ),
        ),
        "UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB": _node(
            "UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB",
            _action(
                "Call",
                kind="call",
                weights=_weights(QQ="0.5", AKs="0.25"),
            ),
        ),
        "UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB_Call_BB": _node(
            "UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB_Call_BB",
            _action("Fold", kind="fold", weights=_weights(AA="1")),
        ),
    }
    return nodes


def _abstract_reraise_nodes() -> dict[str, BlueprintNode]:
    """UTG opens, four players fold, BB 3-bets, and UTG acts again."""

    histories = (
        "UTG",
        "UTG_60%_HJ",
        "UTG_60%_HJ_Fold_CO",
        "UTG_60%_HJ_Fold_CO_Fold_BTN",
        "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold_SB",
        "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold_SB_Fold_BB",
        "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold_SB_Fold_BB_182%_UTG",
    )
    nodes = {
        histories[0]: _node(
            histories[0],
            _action("60%", kind="raise", weights=_weights(AA="1")),
        ),
        histories[1]: _node(
            histories[1],
            _action("Fold", kind="fold", weights=_weights(AA="1")),
        ),
        histories[2]: _node(
            histories[2],
            _action("Fold", kind="fold", weights=_weights(AA="1")),
        ),
        histories[3]: _node(
            histories[3],
            _action("Fold", kind="fold", weights=_weights(AA="1")),
        ),
        histories[4]: _node(
            histories[4],
            _action("Fold", kind="fold", weights=_weights(AA="1")),
        ),
        histories[5]: _node(
            histories[5],
            _action("182%", kind="raise", weights=_weights(AA="1")),
        ),
        histories[6]: _node(
            histories[6],
            _action("50%", kind="raise", weights=_weights(AA="1")),
        ),
    }
    return nodes


class StrictFakeStore:
    """Minimal offline cache facade that fails on every unknown artifact."""

    def __init__(
        self,
        *,
        nodes: dict[str, BlueprintNode] | None = None,
        spots: tuple[BlueprintSpot, ...] | None = None,
    ) -> None:
        self.nodes = nodes if nodes is not None else _coherent_nodes()
        self._spots = spots or tuple(
            BlueprintSpot(history, history.count("_") // 2 + 1)
            for history in self.nodes
        )
        self.calls: list[tuple[object, ...]] = []

    def manifest(self) -> BlueprintManifest:
        self.calls.append(("manifest",))
        return BlueprintManifest(
            game="nl",
            version=2,
            generated_at=GENERATED_AT,
            stacks=(100,),
            positions=POSITIONS,
        )

    def spots(self, stack: int) -> tuple[BlueprintSpot, ...]:
        self.calls.append(("spots", stack))
        if stack != 100:
            raise BlueprintNetworkDisabledError(f"stack {stack} spots are not cached")
        return self._spots

    def node(self, stack: int, history: str) -> BlueprintNode:
        self.calls.append(("node", stack, history))
        if stack != 100 or history not in self.nodes:
            raise BlueprintNetworkDisabledError(
                f"node {stack}:{history} is not cached"
            )
        return self.nodes[history]


class MissingCacheStore(StrictFakeStore):
    def manifest(self) -> BlueprintManifest:
        raise BlueprintNetworkDisabledError("blueprint manifest is not cached")


class EngineMustNotRun:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("the postflop engine must not run for preflop policy")


def _config(**changes) -> live.LiveGTOConfig:
    base = live.LiveGTOConfig(
        enabled=True,
        owned_simulator_acknowledged=True,
        range_source="blueprint",
        rake_rate_pct=Decimal(5),
        rake_cap_bb=Decimal("0.5"),
    )
    return replace(base, **changes)


def _ordered_amounts(
    values: dict[str, Decimal] | None = None,
    *,
    default: Decimal,
) -> tuple[tuple[str, Decimal], ...]:
    overrides = values or {}
    return tuple((position, overrides.get(position, default)) for position in POSITIONS)


def _observation(
    *,
    actor: str | None,
    contributions: dict[str, Decimal],
    folded: set[str],
    terminal: bool,
    initial_stacks: dict[str, Decimal] | None = None,
    hand_id: str = "fixture-1",
) -> ObservedPreflopState:
    return ObservedPreflopState(
        actor=actor,
        contributions=_ordered_amounts(contributions, default=Decimal(0)),
        folded=frozenset(folded),
        initial_stacks=_ordered_amounts(initial_stacks, default=Decimal(100)),
        terminal=terminal,
        provenance=ObservationProvenance(source="offline_test", hand_id=hand_id),
    )


def _root_observation(
    *,
    initial_stacks: dict[str, Decimal] | None = None,
    sb: Decimal = Decimal("0.5"),
) -> ObservedPreflopState:
    return _observation(
        actor="UTG",
        contributions={"SB": sb, "BB": Decimal(1)},
        folded=set(),
        terminal=False,
        initial_stacks=initial_stacks,
    )


def _preflop_state(
    observation: ObservedPreflopState | None = None,
    **changes,
) -> live.LiveDecisionState:
    state = live.LiveDecisionState(
        hand_id="fixture-preflop-1",
        street="PREFLOP",
        board=(),
        hero_combo=("As", "Ks"),
        hero_position="UTG",
        villain_position="BB",
        hero_is_oop=False,
        active_villains=5,
        pot_bb=Decimal("1.5"),
        hero_stack_bb=Decimal(100),
        villain_stack_bb=Decimal(99),
        hero_current_bet_bb=Decimal(0),
        villain_current_bet_bb=Decimal(1),
        amount_to_call_bb=Decimal(1),
        legal_actions=("Fold", "Call", "Raise"),
        street_root_confirmed=True,
        preflop_observation=observation or _root_observation(),
    )
    return replace(state, **changes)


def _terminal_observation() -> ObservedPreflopState:
    return _observation(
        actor=None,
        contributions={
            "BTN": Decimal("2.5"),
            "SB": Decimal("2.5"),
            "BB": Decimal(1),
        },
        folded={"UTG", "HJ", "CO", "BB"},
        terminal=True,
        hand_id="fixture-postflop-1",
    )


def _postflop_state(**changes) -> live.LiveDecisionState:
    state = live.LiveDecisionState(
        hand_id="fixture-postflop-1",
        street="FLOP",
        board=("2c", "7d", "Th"),
        hero_combo=("As", "Ks"),
        hero_position="SB",
        villain_position="BTN",
        hero_is_oop=True,
        active_villains=1,
        pot_bb=Decimal(5),
        hero_stack_bb=Decimal("97.5"),
        villain_stack_bb=Decimal("97.5"),
        hero_current_bet_bb=Decimal(0),
        villain_current_bet_bb=Decimal(0),
        amount_to_call_bb=Decimal(0),
        legal_actions=("Check", "Bet"),
        street_root_confirmed=True,
        preflop_observation=_terminal_observation(),
    )
    return replace(state, **changes)


class PreflopRouterIntegrationTests(unittest.TestCase):
    def test_cached_mixed_policy_bypasses_engine_and_is_deterministic(self):
        store = StrictFakeStore()
        provider = live.BlueprintRangeProvider(_config(), store=store)
        engine = EngineMustNotRun()
        router = live.LiveGTORouter(
            _config(), range_provider=provider, engine_factory=engine
        )
        state = _preflop_state()

        first = router.evaluate(state)
        second = router.evaluate(state)

        self.assertEqual(live.LiveGTOStatus.SOLVED, first.status)
        self.assertEqual(first.analysis, second.analysis)
        self.assertIn("AKs", first.analysis)
        self.assertIn("25.0%", first.analysis)
        self.assertIn("75.0%", first.analysis)
        self.assertEqual("verified preflop blueprint cache", first.source)
        self.assertEqual(0, engine.calls)

    def test_exact_mode_rejects_nonuniform_stacks_and_rake_mismatch(self):
        nonuniform = _root_observation(
            initial_stacks={"BB": Decimal(90), "UTG": Decimal(100)}
        )
        cases = (
            (
                _config(),
                _preflop_state(nonuniform),
                "uniform-stack blueprint",
            ),
            (
                _config(rake_rate_pct=Decimal(0), rake_cap_bb=Decimal(0)),
                _preflop_state(),
                "fixed at 5% rake",
            ),
        )
        for config, state, reason in cases:
            with self.subTest(reason=reason):
                provider = live.BlueprintRangeProvider(
                    config, store=StrictFakeStore()
                )
                result = live.LiveGTORouter(
                    config, range_provider=provider, engine_factory=EngineMustNotRun()
                ).evaluate(state)
                self.assertEqual(live.LiveGTOStatus.UNSUPPORTED, result.status)
                self.assertIn(reason, result.reason)

    def test_visible_call_stack_and_buttons_are_cross_checked(self):
        provider = live.BlueprintRangeProvider(_config(), store=StrictFakeStore())
        router = live.LiveGTORouter(
            _config(), range_provider=provider, engine_factory=EngineMustNotRun()
        )
        cases = (
            (
                _preflop_state(amount_to_call_bb=Decimal("0.5")),
                "call amount disagrees",
            ),
            (
                _preflop_state(hero_current_bet_bb=Decimal("0.5")),
                "current bet disagrees",
            ),
            (
                _preflop_state(legal_actions=("Fold", "Raise")),
                "buttons disagree",
            ),
        )
        for state, reason in cases:
            with self.subTest(reason=reason):
                outcome = router.evaluate(state)
                self.assertEqual(live.LiveGTOStatus.UNSUPPORTED, outcome.status)
                self.assertIn(reason, outcome.reason)

    def test_abstract_stack_bucket_has_a_hard_error_bound(self):
        config = _config(
            blueprint_match_mode="abstract",
            blueprint_max_stack_error_pct=Decimal(25),
        )
        observation = _root_observation(
            initial_stacks={"UTG": Decimal(100), "BB": Decimal(20)}
        )
        outcome = live.LiveGTORouter(
            config,
            range_provider=live.BlueprintRangeProvider(
                config, store=StrictFakeStore()
            ),
            engine_factory=EngineMustNotRun(),
        ).evaluate(_preflop_state(observation))
        self.assertEqual(live.LiveGTOStatus.UNSUPPORTED, outcome.status)
        self.assertIn("nearest 100 BB blueprint bucket", outcome.reason)

    def test_normal_raise_is_not_remapped_to_or_authorized_by_all_in(self):
        provider = live.BlueprintRangeProvider(_config(), store=StrictFakeStore())
        no_raise_button = live.LiveGTORouter(
            _config(), range_provider=provider, engine_factory=EngineMustNotRun()
        ).evaluate(
            _preflop_state(legal_actions=("Fold", "Call", "All-in"))
        )
        self.assertEqual(live.LiveGTOStatus.UNSUPPORTED, no_raise_button.status)
        self.assertIn("absent from the visible Hero buttons", no_raise_button.reason)

        root = _node(
            "UTG",
            _action(
                "3960%", kind="raise", weights=_weights(AA="1")
            ),
        )
        store = StrictFakeStore(
            nodes={"UTG": root}, spots=(BlueprintSpot("UTG", 1),)
        )
        collapsed = live.BlueprintRangeProvider(_config(), store=store)
        with self.assertRaisesRegex(live.LiveGTORangeError, "collapse into an actual all-in"):
            collapsed.preflop_policy_for(
                _preflop_state(hero_combo=("As", "Ah"))
            )

    def test_abstract_mode_records_stack_rake_and_sizing_caveats(self):
        config = _config(
            blueprint_match_mode="abstract",
            blueprint_size_tolerance_pct=Decimal(20),
            rake_rate_pct=Decimal(0),
            rake_cap_bb=Decimal(0),
        )
        initial_stacks = {
                "UTG": Decimal(95),
                "HJ": Decimal(96),
                "CO": Decimal(97),
                "BTN": Decimal(98),
                "SB": Decimal(99),
                "BB": Decimal(101),
        }
        observation = _observation(
            actor="HJ",
            contributions={
                "UTG": Decimal("2.6"),
                "SB": Decimal("0.5"),
                "BB": Decimal(1),
            },
            folded=set(),
            terminal=False,
            initial_stacks=initial_stacks,
        )
        provider = live.BlueprintRangeProvider(config, store=StrictFakeStore())

        bundle = provider.preflop_policy_for(
            _preflop_state(
                observation,
                hero_position="HJ",
                hero_stack_bb=Decimal(96),
                amount_to_call_bb=Decimal("2.6"),
            )
        )

        caveats = " | ".join(bundle.approximations)
        self.assertTrue(bundle.approximate)
        self.assertIn("abstract blueprint matching", caveats)
        self.assertIn("source rake is 5% capped at 0.5 BB", caveats)
        self.assertIn("uniform 100 BB bucket", caveats)
        self.assertIn("contribution sizes mapped", caveats)

        outcome = live.LiveGTORouter(
            config,
            range_provider=provider,
            engine_factory=EngineMustNotRun(),
        ).evaluate(
            _preflop_state(
                observation,
                hero_position="HJ",
                hero_stack_bb=Decimal(96),
                amount_to_call_bb=Decimal("2.6"),
            )
        )
        self.assertEqual(live.LiveGTOStatus.SOLVED, outcome.status)
        self.assertEqual("approximate preflop blueprint cache", outcome.source)
        self.assertIn("**Approximate blueprint mix:**", outcome.analysis)

    def test_abstract_raise_uses_observed_last_full_raise_minimum(self):
        config = _config(
            blueprint_match_mode="abstract",
            blueprint_size_tolerance_pct=Decimal(20),
        )
        nodes = _abstract_reraise_nodes()
        current_history = (
            "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold_SB_Fold_BB_182%_UTG"
        )
        store = StrictFakeStore(
            nodes=nodes,
            spots=(BlueprintSpot(current_history, 7),),
        )
        observation = _observation(
            actor="UTG",
            contributions={
                "UTG": Decimal(3),
                "SB": Decimal("0.5"),
                "BB": Decimal(15),
            },
            folded={"HJ", "CO", "BTN", "SB"},
            terminal=False,
        )
        provider = live.BlueprintRangeProvider(config, store=store)
        state = _preflop_state(
            observation,
            hero_combo=("As", "Ah"),
            hero_stack_bb=Decimal(97),
            hero_current_bet_bb=Decimal(3),
            villain_stack_bb=Decimal(85),
            villain_current_bet_bb=Decimal(15),
            amount_to_call_bb=Decimal(12),
            pot_bb=Decimal("18.5"),
        )

        with self.assertRaisesRegex(
            live.LiveGTORangeError,
            "not legal in the observed game",
        ):
            provider.preflop_policy_for(state)

    def test_abstract_raise_fails_closed_when_prior_target_was_overwritten(self):
        observation = _observation(
            actor="BB",
            contributions={
                "UTG": Decimal(15),
                "BTN": Decimal(15),
                "SB": Decimal("0.5"),
                "BB": Decimal(1),
            },
            folded={"HJ", "CO", "SB"},
            terminal=False,
        )
        resolution = SimpleNamespace(
            steps=(
                ActionStep("UTG", "60%"),
                ActionStep("BTN", "100%"),
                ActionStep("UTG", "Call"),
            )
        )

        with self.assertRaisesRegex(
            live.LiveGTORangeError,
            "cannot reconstruct an earlier observed raise target",
        ):
            live.BlueprintRangeProvider._observed_last_full_raise(
                observation,
                resolution,
            )

    def test_no_matching_and_ambiguous_paths_are_unsupported(self):
        no_match_store = StrictFakeStore(
            spots=(BlueprintSpot("UTG_60%_HJ", 2),)
        )
        no_match_provider = live.BlueprintRangeProvider(
            _config(), store=no_match_store
        )
        no_match = live.LiveGTORouter(
            _config(),
            range_provider=no_match_provider,
            engine_factory=EngineMustNotRun(),
        ).evaluate(_preflop_state())
        self.assertEqual(live.LiveGTOStatus.UNSUPPORTED, no_match.status)
        self.assertIn("no decision blueprint path", no_match.reason)

        ambiguous_store = StrictFakeStore(
            spots=(
                BlueprintSpot("UTG_60%_HJ", 2),
                BlueprintSpot("UTG_60.1%_HJ", 2),
            )
        )
        observed = _observation(
            actor="HJ",
            contributions={
                "UTG": Decimal("2.50125"),
                "SB": Decimal("0.5"),
                "BB": Decimal(1),
            },
            folded=set(),
            terminal=False,
        )
        ambiguous_state = _preflop_state(
            observed,
            hero_position="HJ",
            hero_stack_bb=Decimal(100),
        )
        ambiguous_provider = live.BlueprintRangeProvider(
            _config(), store=ambiguous_store
        )
        ambiguous = live.LiveGTORouter(
            _config(),
            range_provider=ambiguous_provider,
            engine_factory=EngineMustNotRun(),
        ).evaluate(ambiguous_state)
        self.assertEqual(live.LiveGTOStatus.UNSUPPORTED, ambiguous.status)
        self.assertIn("decision blueprint paths match", ambiguous.reason)

    def test_missing_offline_artifact_is_failed_not_silently_approximated(self):
        provider = live.BlueprintRangeProvider(_config(), store=MissingCacheStore())
        result = live.LiveGTORouter(
            _config(), range_provider=provider, engine_factory=EngineMustNotRun()
        ).evaluate(_preflop_state())

        self.assertEqual(live.LiveGTOStatus.FAILED, result.status)
        self.assertIn("not cached", result.reason)


class PostflopRangeIntegrationTests(unittest.TestCase):
    def test_terminal_hu_ranges_use_cumulative_reach_and_never_inject_hero(self):
        provider = live.BlueprintRangeProvider(_config(), store=StrictFakeStore())

        bundle = provider.ranges_for(_postflop_state())

        oop = {combo.cards: combo.weight for combo in bundle.oop.combos}
        ip = {combo.cards: combo.weight for combo in bundle.ip.combos}
        hero = live.WeightedCombo(("As", "Ks")).cards
        villain_kqo = live.WeightedCombo(("Kh", "Qd")).cards
        self.assertEqual(Decimal("0.25"), oop[hero])
        self.assertEqual(Decimal("0.4"), ip[villain_kqo])
        self.assertFalse(bundle.hero_combo_injected)
        self.assertFalse(bundle.approximate)
        self.assertTrue(
            any(
                "folded-card bunching is omitted" in caveat
                for caveat in bundle.approximations
            )
        )
        self.assertIn("SB-vs-BTN", bundle.profile_id)

    def test_turn_ranges_without_postflop_action_conditioning_are_approximate(self):
        provider = live.BlueprintRangeProvider(_config(), store=StrictFakeStore())

        bundle = provider.ranges_for(
            _postflop_state(
                street="TURN",
                board=("2c", "7d", "Th", "Jc"),
            )
        )

        self.assertTrue(bundle.approximate)
        self.assertTrue(
            any(
                "not conditioned on prior postflop actions" in caveat
                for caveat in bundle.approximations
            )
        )

    def test_zero_reach_hero_is_rejected_instead_of_injected(self):
        provider = live.BlueprintRangeProvider(_config(), store=StrictFakeStore())
        state = _postflop_state(hero_combo=("7s", "2d"))

        with self.assertRaisesRegex(live.PreflopHistoryError, "zero blueprint reach"):
            provider.ranges_for(state)


if __name__ == "__main__":
    unittest.main()
