import json
import copy
import ast
import hashlib
import os
import sqlite3
import sys
import tempfile
import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from gto_oracle import (
    Action,
    ActionKind,
    ActionValue,
    AllocationMode,
    AssessmentStatus,
    BetSizingConfig,
    ComboPolicy,
    DecisionQuery,
    EngineClient,
    EngineProcessError,
    EngineProtocolError,
    EngineResponseError,
    IllegalActionAssessment,
    OracleCache,
    OracleValidationError,
    OutOfTreeActionAssessment,
    PlayerRange,
    PlayerBetSizes,
    Position,
    ScoredAction,
    SolveResult,
    SolveParameters,
    SolveSpec,
    SolverMetadata,
    Street,
    StreetBetSizes,
    TreeConfig,
    UnsupportedGameError,
    WeightedCombo,
    assess_action,
    build_engine_request,
    canonical_json,
    parse_engine_response,
    render_weighted_range,
    validate_result_for_spec,
)


PINNED_SOLVER_COMMIT = "9d1509fe5077d019825f833eed04b16d342dfda1"


def make_parameters():
    half_pot = PlayerBetSizes("50%", "2.5x")
    return SolveParameters(
        chip_scale=100,
        chip_unit="0.01 BB",
        bet_sizes=BetSizingConfig(
            flop=StreetBetSizes(half_pot, half_pot),
            turn=StreetBetSizes(half_pot, half_pot),
            river=StreetBetSizes(half_pot, half_pot),
            flop_donk_sizes=None,
            turn_donk_sizes=None,
            river_donk_sizes=None,
        ),
        add_allin_threshold=Decimal("1.5"),
        force_allin_threshold=Decimal("0.15"),
        merging_threshold=Decimal("0.1"),
        target_exploitability_pct=Decimal("0.5"),
        max_iterations=1000,
        allocation_mode=AllocationMode.UNCOMPRESSED_F32,
        solver_name="b-inary/postflop-solver",
        solver_commit=PINNED_SOLVER_COMMIT,
    )


def make_spec(*, reverse_inputs=False):
    oop_combos = (
        WeightedCombo(("As", "Ad"), Decimal("0.5")),
        WeightedCombo(("Kc", "Kh"), Decimal(1)),
    )
    ip_combos = (
        WeightedCombo(("Qc", "Qh"), Decimal("0.25")),
        WeightedCombo(("Jc", "Jh"), Decimal("0.75")),
    )
    actions = (
        Action(ActionKind.CHECK),
        Action(ActionKind.BET, 3),
        Action(ActionKind.BET, 6),
    )
    kinds = (ActionKind.CHECK, ActionKind.BET)
    players = (Position.OOP, Position.IP)
    if reverse_inputs:
        oop_combos = tuple(reversed(oop_combos))
        ip_combos = tuple(reversed(ip_combos))
        actions = tuple(reversed(actions))
        kinds = tuple(reversed(kinds))
        players = tuple(reversed(players))
    return SolveSpec(
        street=Street.FLOP,
        board=("2c", "7d", "Ts"),
        acting_player=Position.OOP,
        oop_range=PlayerRange(Position.OOP, oop_combos),
        ip_range=PlayerRange(Position.IP, ip_combos),
        tree=TreeConfig(
            pot=9,
            effective_stack=30,
            facing_bet=0,
            legal_action_kinds=kinds,
            modeled_actions=actions,
        ),
        parameters=make_parameters(),
        players=players,
    )


def make_engine_spec():
    parameters = make_parameters()
    return SolveSpec(
        street=Street.RIVER,
        board=("Td", "9d", "6h", "Qc", "2s"),
        acting_player=Position.OOP,
        oop_range=PlayerRange(
            Position.OOP,
            (
                WeightedCombo(("Ad", "Kd"), Decimal("0.5")),
                WeightedCombo(("Ah", "Kh"), Decimal(1)),
            ),
        ),
        ip_range=PlayerRange(
            Position.IP,
            (
                WeightedCombo(("As", "Ks"), Decimal("0.25")),
                WeightedCombo(("Ac", "Kc"), Decimal("0.75")),
            ),
        ),
        tree=TreeConfig(
            pot=100,
            effective_stack=400,
            facing_bet=0,
            legal_action_kinds=(ActionKind.CHECK, ActionKind.BET),
            modeled_actions=(Action(ActionKind.CHECK), Action(ActionKind.BET, 50)),
        ),
        parameters=parameters,
    )


def make_engine_response(spec):
    request = build_engine_request(spec)
    effective_keys = (
        "chip_scale",
        "chip_unit",
        "street",
        "board",
        "oop_range",
        "ip_range",
        "starting_pot",
        "effective_stack",
        "bet_sizes",
        "rake",
        "tree_options",
        "target_exploitability_pct",
        "max_iterations",
        "allocation_mode",
    )
    combo_template = {
        "equity": 0.5,
    }
    return {
        "schema_version": 1,
        "id": spec.cache_key,
        "operation": "solve_root",
        "status": "ok",
        "provenance": {
            "solver": {
                "name": spec.parameters.solver_name,
                "algorithm": "Discounted CFR",
                "commit": spec.parameters.solver_commit,
                "abstraction": "none (suit isomorphism only)",
                "allocation_mode": spec.parameters.allocation_mode.value,
                "memory_hard_limit_bytes": 8 * 1024 * 1024 * 1024,
            },
            "offline_only_acknowledged": True,
            "effective_request": {key: request[key] for key in effective_keys},
        },
        "street": spec.street.value,
        "board": list(spec.board),
        "root_player": "OOP",
        "root_actions": [
            {"index": 0, "label": "Check", "kind": "CHECK", "amount": None},
            {"index": 1, "label": "Bet 50", "kind": "BET", "amount": 50},
        ],
        "players": [
            {
                "player": "OOP",
                "total_reachable_weight": 3,
                "average_equity": 0.5,
                "average_ev_units": 6.233333333333333,
                "combos": [
                    {
                        **combo_template,
                        "hand": "AdKd",
                        "range_weight": 0.5,
                        "normalized_weight": 1,
                        "reach_weight": 0.3333333333,
                        "equilibrium_ev_units": 5.5,
                        "root_action_frequencies": [0.25, 0.75],
                        "root_action_evs_units": [4, 6],
                    },
                    {
                        **combo_template,
                        "hand": "AhKh",
                        "range_weight": 1,
                        "normalized_weight": 2,
                        "reach_weight": 0.6666666667,
                        "equilibrium_ev_units": 6.6,
                        "root_action_frequencies": [0.8, 0.2],
                        "root_action_evs_units": [7, 5],
                    },
                ],
            },
            {
                "player": "IP",
                "total_reachable_weight": 2,
                "average_equity": 0.55,
                "average_ev_units": -5.5,
                "combos": [
                    {
                        **combo_template,
                        "hand": "AsKs",
                        "range_weight": 0.25,
                        "normalized_weight": 0.5,
                        "reach_weight": 0.25,
                        "equity": 0.4,
                        "equilibrium_ev_units": -4,
                        "root_action_frequencies": None,
                        "root_action_evs_units": None,
                    },
                    {
                        **combo_template,
                        "hand": "AcKc",
                        "range_weight": 0.75,
                        "normalized_weight": 1.5,
                        "reach_weight": 0.75,
                        "equity": 0.6,
                        "equilibrium_ev_units": -6,
                        "root_action_frequencies": None,
                        "root_action_evs_units": None,
                    },
                ],
            },
        ],
        "convergence": {
            "iterations": 800,
            "max_iterations": spec.parameters.max_iterations,
            "target_exploitability_pct": 0.5,
            "target_exploitability_units": 0.5,
            "exploitability_pct_of_pot": 0.4,
            "exploitability_units": 0.4,
            "target_reached": True,
        },
        "memory": {
            "estimated_uncompressed_bytes": 1000,
            "estimated_compressed_bytes": 500,
            "allocation_mode": spec.parameters.allocation_mode.value,
            "hard_limit_bytes": 8 * 1024 * 1024 * 1024,
        },
        "timings_ms": {
            "tree_build": 1,
            "allocation": 2,
            "solve": 20,
            "extraction": 1,
            "total": 24,
        },
    }


def make_engine_node_spec(*, acting_player=Position.IP):
    root = make_engine_spec()
    if acting_player is Position.IP:
        history = (Action(ActionKind.BET, 50),)
        facing_bet = 50
        legal_kinds = (ActionKind.FOLD, ActionKind.CALL, ActionKind.RAISE)
        actions = (
            Action(ActionKind.FOLD),
            Action(ActionKind.CALL),
            Action(ActionKind.RAISE, 125),
        )
    else:
        history = (
            Action(ActionKind.CHECK),
            Action(ActionKind.BET, 50),
        )
        facing_bet = 50
        legal_kinds = (ActionKind.FOLD, ActionKind.CALL, ActionKind.RAISE)
        actions = (
            Action(ActionKind.FOLD),
            Action(ActionKind.CALL),
            Action(ActionKind.RAISE, 125),
        )
    return replace(
        root,
        acting_player=acting_player,
        tree=TreeConfig(
            pot=root.tree.pot,
            effective_stack=root.tree.effective_stack,
            facing_bet=facing_bet,
            legal_action_kinds=legal_kinds,
            modeled_actions=actions,
            action_history=history,
            rake_rate_pct=root.tree.rake_rate_pct,
            rake_cap=root.tree.rake_cap,
        ),
    )


def make_engine_node_response(spec):
    request = build_engine_request(spec)
    base = make_engine_response(make_engine_spec())
    action_labels = {
        ActionKind.FOLD: "Fold",
        ActionKind.CALL: "Call",
        ActionKind.RAISE: "Raise to 125",
    }
    node_actions = [
        {
            "index": index,
            "label": action_labels[action.kind],
            "kind": action.kind.value,
            "amount": action.amount,
        }
        for index, action in enumerate(
            (
                Action(ActionKind.FOLD),
                Action(ActionKind.CALL),
                Action(ActionKind.RAISE, 125),
            )
        )
    ]
    acting_range = (
        spec.ip_range if spec.acting_player is Position.IP else spec.oop_range
    )
    combo = acting_range.combos[0]
    hand = "".join(reversed(combo.cards))
    response = {
        key: value
        for key, value in base.items()
        if key
        not in {
            "root_player",
            "root_actions",
            "players",
        }
    }
    response.update(
        {
            "id": spec.cache_key,
            "operation": "solve_node",
            "street": spec.street.value,
            "board": list(spec.board),
            "current_player": spec.acting_player.value,
            "node_actions": node_actions,
            "node_total_reachable_weight": 0.3,
            "policies": [
                {
                    "hand": hand,
                    "input_range_weight": float(combo.weight),
                    "path_weight": float(combo.weight) * 0.5,
                    "joint_compatible_weight": 0.3,
                    "conditional_reach_weight": 1.0,
                    "equity": 0.55,
                    "equilibrium_ev_units": 8.4,
                    "node_action_frequencies": [0.1, 0.6, 0.3],
                    "node_action_evs_units": [0.0, 10.0, 8.0],
                }
            ],
        }
    )
    effective_keys = (
        "chip_scale",
        "chip_unit",
        "street",
        "board",
        "oop_range",
        "ip_range",
        "starting_pot",
        "effective_stack",
        "bet_sizes",
        "rake",
        "tree_options",
        "target_exploitability_pct",
        "max_iterations",
        "allocation_mode",
        "action_history",
        "expected_current_player",
        "expected_facing_bet",
        "expected_node_actions",
    )
    response["provenance"] = copy.deepcopy(base["provenance"])
    response["provenance"]["effective_request"] = {
        key: request[key] for key in effective_keys
    }
    return response


def make_metadata(*, elapsed_seconds=Decimal("1.25"), converged=True, extra=None):
    return SolverMetadata(
        solver_name="b-inary/postflop-solver",
        solver_version=PINNED_SOLVER_COMMIT,
        iterations=500,
        elapsed_seconds=elapsed_seconds,
        exploitability=Decimal("0.003"),
        converged=converged,
        extra=extra or (("algorithm", "DCFR"), ("threads", "4")),
    )


def make_result(spec, *, metadata=None):
    return SolveResult.for_spec(
        spec,
        (
            ComboPolicy(
                ("As", "Ad"),
                Decimal("0.4"),
                Decimal("0.55"),
                (
                    ActionValue(
                        Action(ActionKind.CHECK), Decimal("0.20"), Decimal("1.2")
                    ),
                    ActionValue(
                        Action(ActionKind.BET, 3),
                        Decimal("0.70"),
                        Decimal("1.5"),
                    ),
                    ActionValue(
                        Action(ActionKind.BET, 6),
                        Decimal("0.10"),
                        Decimal("1.4"),
                    ),
                ),
            ),
            ComboPolicy(
                ("Kh", "Kc"),
                Decimal("0.6"),
                Decimal("0.48"),
                (
                    ActionValue(
                        Action(ActionKind.CHECK), Decimal("0.75"), Decimal("1.6")
                    ),
                    ActionValue(
                        Action(ActionKind.BET, 3),
                        Decimal("0.20"),
                        Decimal("1.3"),
                    ),
                    ActionValue(
                        Action(ActionKind.BET, 6),
                        Decimal("0.05"),
                        Decimal("0.9"),
                    ),
                ),
            ),
        ),
        metadata or make_metadata(),
    )


class ModelValidationTests(unittest.TestCase):
    def test_postflop_hu_spec_is_valid_and_immutable(self):
        spec = make_spec()
        self.assertEqual(Street.FLOP, spec.street)
        self.assertEqual((Position.OOP, Position.IP), spec.players)
        with self.assertRaises((AttributeError, TypeError)):
            spec.board = ("2c", "7d", "Th")

    def test_preflop_and_multiway_are_typed_unsupported(self):
        spec = make_spec()
        with self.assertRaises(UnsupportedGameError):
            SolveSpec(
                street=Street.PREFLOP,
                board=(),
                acting_player=spec.acting_player,
                oop_range=spec.oop_range,
                ip_range=spec.ip_range,
                tree=spec.tree,
                parameters=spec.parameters,
            )
        with self.assertRaises(UnsupportedGameError):
            SolveSpec(
                street=spec.street,
                board=spec.board,
                acting_player=spec.acting_player,
                oop_range=spec.oop_range,
                ip_range=spec.ip_range,
                tree=spec.tree,
                parameters=spec.parameters,
                players=(Position.OOP, Position.IP, Position.IP),
            )

    def test_cards_ranges_and_tree_are_strictly_validated(self):
        spec = make_spec()
        with self.assertRaises(OracleValidationError):
            WeightedCombo(("As", "As"), Decimal(1))
        with self.assertRaises(OracleValidationError):
            PlayerRange(
                Position.OOP,
                (
                    WeightedCombo(("As", "Ad")),
                    WeightedCombo(("Ad", "As")),
                ),
            )
        with self.assertRaises(OracleValidationError):
            SolveSpec(
                street=Street.FLOP,
                board=("2c", "7d", "As"),
                acting_player=Position.OOP,
                oop_range=spec.oop_range,
                ip_range=spec.ip_range,
                tree=spec.tree,
                parameters=spec.parameters,
            )
        with self.assertRaises(OracleValidationError):
            TreeConfig(
                pot=9,
                effective_stack=30,
                facing_bet=0,
                legal_action_kinds=(ActionKind.CALL,),
                modeled_actions=(Action(ActionKind.CALL),),
            )
        with self.assertRaises(OracleValidationError):
            Action(ActionKind.BET, Decimal(0))
        with self.assertRaises(OracleValidationError):
            Action(ActionKind.CHECK, Decimal(1))

    def test_tree_derives_and_enforces_the_minimum_full_raise_to(self):
        spec = make_engine_node_spec()
        self.assertEqual(100, spec.tree.minimum_raise_to)
        reraised = TreeConfig(
            pot=100,
            effective_stack=400,
            facing_bet=75,
            legal_action_kinds=(
                ActionKind.FOLD,
                ActionKind.CALL,
                ActionKind.RAISE,
            ),
            modeled_actions=(
                Action(ActionKind.FOLD),
                Action(ActionKind.CALL),
                Action(ActionKind.RAISE, 200),
            ),
            action_history=(
                Action(ActionKind.BET, 50),
                Action(ActionKind.RAISE, 125),
            ),
        )
        self.assertEqual(200, reraised.minimum_raise_to)
        with self.assertRaisesRegex(
            OracleValidationError,
            "below the minimum full raise-to",
        ):
            replace(
                spec.tree,
                modeled_actions=(
                    Action(ActionKind.FOLD),
                    Action(ActionKind.CALL),
                    Action(ActionKind.RAISE, 60),
                ),
            )

    def test_each_combo_requires_a_complete_normalized_policy(self):
        spec = make_spec()
        with self.assertRaises(OracleValidationError):
            ComboPolicy(
                ("As", "Ad"),
                Decimal("0.5"),
                Decimal("0.5"),
                (
                    ActionValue(
                        Action(ActionKind.CHECK), Decimal("0.4"), Decimal(1)
                    ),
                    ActionValue(
                        Action(ActionKind.BET, 3),
                        Decimal("0.5"),
                        Decimal(2),
                    ),
                ),
            )

    def test_result_requires_exact_acting_range_combo_coverage(self):
        spec = make_spec()
        complete = make_result(spec)
        missing = SolveResult(
            spec.cache_key,
            (complete.combo_policies[0],),
            make_metadata(),
        )
        with self.assertRaises(OracleValidationError):
            validate_result_for_spec(spec, missing)
        with self.assertRaises(OracleValidationError):
            SolveResult(
                spec.cache_key,
                (complete.combo_policies[0], complete.combo_policies[0]),
                make_metadata(),
            )

    def test_descendant_result_allows_only_positive_reach_acting_subset(self):
        spec = make_engine_node_spec()
        actions = spec.tree.modeled_actions
        policy = ComboPolicy(
            spec.ip_range.combos[0].cards,
            Decimal(1),
            Decimal("0.5"),
            tuple(
                ActionValue(
                    action,
                    Decimal(1) if index == 0 else Decimal(0),
                    Decimal(index),
                )
                for index, action in enumerate(actions)
            ),
        )
        result = SolveResult.for_spec(spec, (policy,), make_metadata())
        self.assertEqual((policy,), result.combo_policies)

        outside = replace(policy, private_combo=("9c", "9h"))
        with self.assertRaisesRegex(OracleValidationError, "out-of-range"):
            SolveResult.for_spec(spec, (outside,), make_metadata())
        with self.assertRaisesRegex(OracleValidationError, "positive-reach"):
            SolveResult.for_spec(
                spec,
                (replace(policy, reach_weight=Decimal(0)),),
                make_metadata(),
            )

    def test_result_requires_pinned_solver_and_normalized_reach(self):
        spec = make_spec()
        complete = make_result(spec)
        for metadata in (
            replace(complete.metadata, solver_name="unexpected/solver"),
            replace(complete.metadata, solver_version="a" * 40),
        ):
            result = SolveResult(
                spec.cache_key,
                complete.combo_policies,
                metadata,
            )
            with self.subTest(metadata=metadata):
                with self.assertRaises(OracleValidationError):
                    validate_result_for_spec(spec, result)

        bad_reach = SolveResult(
            spec.cache_key,
            (
                replace(complete.combo_policies[0], reach_weight=Decimal("0.3")),
                complete.combo_policies[1],
            ),
            complete.metadata,
        )
        with self.assertRaises(OracleValidationError):
            validate_result_for_spec(spec, bad_reach)
        with self.assertRaises(OracleValidationError):
            SolveResult.for_spec(
                spec,
                (
                    complete.combo_policies[0],
                    ComboPolicy(
                        ("9c", "9h"),
                        Decimal("0.1"),
                        Decimal("0.5"),
                        complete.combo_policies[0].action_values,
                    ),
                ),
                make_metadata(),
            )


class CanonicalIdentityTests(unittest.TestCase):
    def test_equivalent_unordered_inputs_have_same_json_and_hash(self):
        first = make_spec()
        second = make_spec(reverse_inputs=True)
        self.assertEqual(first, second)
        self.assertEqual(first.canonical_json, second.canonical_json)
        self.assertEqual(first.cache_key, second.cache_key)
        parsed = json.loads(first.canonical_json)
        self.assertEqual(9, parsed["tree"]["pot"])
        self.assertEqual(100, parsed["parameters"]["chip_scale"])
        self.assertIsNone(
            parsed["parameters"]["bet_sizes"]["flop_donk_sizes"]
        )

    def test_meaningful_state_change_changes_hash(self):
        first = make_spec()
        second = SolveSpec(
            street=first.street,
            board=("2c", "7d", "Js"),
            acting_player=first.acting_player,
            oop_range=first.oop_range,
            ip_range=first.ip_range,
            tree=first.tree,
            parameters=first.parameters,
        )
        self.assertNotEqual(first.cache_key, second.cache_key)

    def test_node_history_actor_and_modeled_actions_each_change_hash(self):
        spec = make_engine_node_spec()
        variants = (
            replace(
                spec,
                tree=replace(
                    spec.tree,
                    action_history=(Action(ActionKind.BET, 51),),
                ),
            ),
            replace(spec, acting_player=Position.OOP),
            replace(
                spec,
                tree=replace(
                    spec.tree,
                    modeled_actions=(
                        Action(ActionKind.FOLD),
                        Action(ActionKind.CALL),
                        Action(ActionKind.RAISE, 150),
                    ),
                ),
            ),
        )
        for changed in variants:
            with self.subTest(changed=changed):
                self.assertNotEqual(spec.cache_key, changed.cache_key)

    def test_solver_quality_and_provenance_fields_change_hash(self):
        spec = make_spec()
        params = spec.parameters
        variants = (
            replace(params, chip_scale=10),
            replace(params, chip_unit="0.1 BB"),
            replace(params, add_allin_threshold=Decimal("1.6")),
            replace(params, force_allin_threshold=Decimal("0.2")),
            replace(params, merging_threshold=Decimal("0.2")),
            replace(params, target_exploitability_pct=Decimal("0.25")),
            replace(params, max_iterations=2000),
            replace(params, allocation_mode=AllocationMode.COMPRESSED_I16),
            replace(params, solver_name="fork/postflop-solver"),
            replace(params, solver_commit="a" * 40),
        )
        for variant in variants:
            with self.subTest(variant=variant):
                changed = replace(spec, parameters=variant)
                self.assertNotEqual(spec.cache_key, changed.cache_key)

    def test_every_street_position_size_and_donk_changes_hash(self):
        spec = make_spec()
        sizing = spec.parameters.bet_sizes
        for street_name in ("flop", "turn", "river"):
            for position_name in ("oop", "ip"):
                for field_name, replacement in (
                    ("bet", "33%, 75%"),
                    ("raise_sizes", "3x"),
                ):
                    street = getattr(sizing, street_name)
                    player = getattr(street, position_name)
                    changed_player = replace(player, **{field_name: replacement})
                    changed_street = replace(
                        street,
                        **{position_name: changed_player},
                    )
                    changed_sizing = replace(
                        sizing,
                        **{street_name: changed_street},
                    )
                    changed = replace(
                        spec,
                        parameters=replace(
                            spec.parameters,
                            bet_sizes=changed_sizing,
                        ),
                    )
                    with self.subTest(
                        street=street_name,
                        position=position_name,
                        field=field_name,
                    ):
                        self.assertNotEqual(spec.cache_key, changed.cache_key)

        for field_name in ("turn_donk_sizes", "river_donk_sizes"):
            changed_keys = []
            for replacement in ("", "50%"):
                changed_sizing = replace(sizing, **{field_name: replacement})
                changed = replace(
                    spec,
                    parameters=replace(spec.parameters, bet_sizes=changed_sizing),
                )
                with self.subTest(field=field_name, replacement=replacement):
                    self.assertNotEqual(spec.cache_key, changed.cache_key)
                changed_keys.append(changed.cache_key)
            self.assertNotEqual(changed_keys[0], changed_keys[1])

    def test_engine_chip_values_must_be_integral_units(self):
        spec = make_spec()
        with self.assertRaises(OracleValidationError):
            Action(ActionKind.BET, Decimal(3))
        with self.assertRaises(OracleValidationError):
            TreeConfig(
                pot=Decimal(9),
                effective_stack=30,
                facing_bet=0,
                legal_action_kinds=spec.tree.legal_action_kinds,
                modeled_actions=spec.tree.modeled_actions,
            )

    def test_canonical_json_rejects_floats(self):
        with self.assertRaises(OracleValidationError):
            canonical_json({"unsafe": 0.1})


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.spec = make_spec()
        self.result = make_result(self.spec)

    def test_frequency_regret_and_near_optimal(self):
        assessment = assess_action(
            self.spec,
            self.result,
            Action(ActionKind.BET, 6),
            private_combo=("As", "Ad"),
            ev_tolerance=Decimal("0.1"),
        )
        self.assertIsInstance(assessment, ScoredAction)
        self.assertEqual(AssessmentStatus.SCORED, assessment.status)
        self.assertEqual(Decimal("0.1"), assessment.oracle_mass)
        self.assertEqual(("Ad", "As"), assessment.private_combo)
        self.assertEqual(Decimal("0.4"), assessment.reach_weight)
        self.assertEqual(Decimal("0.55"), assessment.equity)
        self.assertEqual(Decimal("0.1"), assessment.ev_regret)
        self.assertTrue(assessment.near_optimal)

        check = assess_action(
            self.spec,
            self.result,
            DecisionQuery(("As", "Ad"), Action(ActionKind.CHECK)),
            ev_tolerance=Decimal("0.29"),
        )
        self.assertEqual(Decimal("0.3"), check.ev_regret)
        self.assertFalse(check.near_optimal)

        other_combo = assess_action(
            self.spec,
            self.result,
            DecisionQuery(("Kc", "Kh"), Action(ActionKind.CHECK)),
        )
        self.assertEqual(Decimal("0.75"), other_combo.oracle_mass)
        self.assertEqual(Decimal(0), other_combo.ev_regret)
        self.assertTrue(other_combo.near_optimal)

    def test_illegal_and_out_of_tree_are_distinct_types(self):
        illegal = assess_action(
            self.spec,
            self.result,
            Action(ActionKind.CALL),
            private_combo=("As", "Ad"),
        )
        self.assertIsInstance(illegal, IllegalActionAssessment)
        self.assertEqual(AssessmentStatus.ILLEGAL, illegal.status)

        out_of_tree = assess_action(
            self.spec,
            self.result,
            Action(ActionKind.BET, 4),
            private_combo=("As", "Ad"),
        )
        self.assertIsInstance(out_of_tree, OutOfTreeActionAssessment)
        self.assertEqual(AssessmentStatus.OUT_OF_TREE, out_of_tree.status)
        self.assertEqual(3, len(out_of_tree.modeled_actions))

        over_stack = assess_action(
            self.spec,
            self.result,
            Action(ActionKind.BET, 31),
            private_combo=("As", "Ad"),
        )
        self.assertIsInstance(over_stack, IllegalActionAssessment)

    def test_undersized_raise_is_illegal_but_short_all_in_remains_legal(self):
        spec = make_engine_node_spec()
        result = parse_engine_response(
            spec,
            json.dumps(make_engine_node_response(spec)),
        )
        undersized = assess_action(
            spec,
            result,
            Action(ActionKind.RAISE, 60),
            private_combo=spec.ip_range.combos[0].cards,
        )
        self.assertIsInstance(undersized, IllegalActionAssessment)
        self.assertIn("minimum full raise-to 100", undersized.reason)

        short_all_in = Action(ActionKind.ALL_IN, 75)
        short_spec = replace(
            spec,
            tree=TreeConfig(
                pot=spec.tree.pot,
                effective_stack=75,
                facing_bet=50,
                legal_action_kinds=(
                    ActionKind.FOLD,
                    ActionKind.CALL,
                    ActionKind.ALL_IN,
                ),
                modeled_actions=(
                    Action(ActionKind.FOLD),
                    Action(ActionKind.CALL),
                    short_all_in,
                ),
                action_history=(Action(ActionKind.BET, 50),),
            ),
        )
        policy = ComboPolicy(
            short_spec.ip_range.combos[0].cards,
            Decimal(1),
            Decimal("0.5"),
            tuple(
                ActionValue(
                    action,
                    Decimal(1) if action == short_all_in else Decimal(0),
                    Decimal(1),
                )
                for action in short_spec.tree.modeled_actions
            ),
        )
        short_result = SolveResult.for_spec(
            short_spec,
            (policy,),
            make_metadata(),
        )
        assessment = assess_action(
            short_spec,
            short_result,
            short_all_in,
            private_combo=policy.private_combo,
        )
        self.assertIsInstance(assessment, ScoredAction)

    def test_mismatched_result_is_rejected(self):
        changed = SolveSpec(
            street=self.spec.street,
            board=("2c", "7d", "Js"),
            acting_player=self.spec.acting_player,
            oop_range=self.spec.oop_range,
            ip_range=self.spec.ip_range,
            tree=self.spec.tree,
            parameters=self.spec.parameters,
        )
        with self.assertRaises(OracleValidationError):
            assess_action(
                changed,
                self.result,
                Action(ActionKind.CHECK),
                private_combo=("As", "Ad"),
            )

    def test_exact_private_combo_is_required(self):
        with self.assertRaises(OracleValidationError):
            assess_action(self.spec, self.result, Action(ActionKind.CHECK))
        with self.assertRaises(OracleValidationError):
            assess_action(
                self.spec,
                self.result,
                Action(ActionKind.CHECK),
                private_combo=("9c", "9h"),
            )

    def test_unconverged_results_require_explicit_diagnostic_opt_in(self):
        unconverged = make_result(
            self.spec,
            metadata=make_metadata(converged=False),
        )
        decision = DecisionQuery(("As", "Ad"), Action(ActionKind.CHECK))
        with self.assertRaisesRegex(OracleValidationError, "unconverged"):
            assess_action(self.spec, unconverged, decision)
        assessment = assess_action(
            self.spec,
            unconverged,
            decision,
            allow_unconverged=True,
        )
        self.assertIsInstance(assessment, ScoredAction)


class EngineClientProtocolTests(unittest.TestCase):
    def setUp(self):
        self.spec = make_engine_spec()
        self.response = make_engine_response(self.spec)

    def test_request_has_explicit_ranges_units_and_complete_tree(self):
        request = build_engine_request(self.spec)
        self.assertEqual(self.spec.cache_key, request["id"])
        self.assertEqual(100, request["chip_scale"])
        self.assertEqual("0.01 BB", request["chip_unit"])
        self.assertEqual("uncompressed_f32", request["allocation_mode"])
        self.assertEqual(
            "AdKd:0.5,AhKh:1",
            render_weighted_range(self.spec.oop_range),
        )
        self.assertEqual("2.5x", request["bet_sizes"]["river"]["ip"]["raise"])
        self.assertIsNone(request["tree_options"]["turn_donk_sizes"])
        self.assertTrue(request["offline_only_acknowledged"])
        self.assertNotIn("owned_simulator_acknowledged", request)

        simulator_request = build_engine_request(
            self.spec,
            offline_only_acknowledged=False,
            owned_simulator_acknowledged=True,
        )
        self.assertFalse(simulator_request["offline_only_acknowledged"])
        self.assertTrue(simulator_request["owned_simulator_acknowledged"])

    def test_strict_fixture_response_builds_combo_specific_result(self):
        result = parse_engine_response(self.spec, json.dumps(self.response))
        self.assertEqual(2, len(result.combo_policies))
        first = next(
            policy
            for policy in result.combo_policies
            if policy.private_combo == ("Kd", "Ad")
        )
        by_action = {value.action: value for value in first.action_values}
        self.assertEqual(
            Decimal("0.75"),
            by_action[Action(ActionKind.BET, 50)].frequency,
        )
        self.assertEqual(Decimal(6), by_action[Action(ActionKind.BET, 50)].ev)
        self.assertEqual(self.spec.parameters.solver_commit, result.metadata.solver_version)
        self.assertEqual(
            "0.4",
            dict(result.metadata.extra)["exploitability_pct_of_pot"],
        )
        self.assertEqual(
            "offline",
            dict(result.metadata.extra)["execution_context"],
        )
        extra = dict(result.metadata.extra)
        self.assertEqual("1000", extra["estimated_uncompressed_bytes"])
        self.assertEqual("500", extra["estimated_compressed_bytes"])
        self.assertEqual("24", extra["timing_total_ms"])

    def test_owned_simulator_context_is_explicit_and_pinned(self):
        response = copy.deepcopy(self.response)
        response["provenance"]["offline_only_acknowledged"] = False
        response["provenance"]["owned_simulator_acknowledged"] = True
        result = parse_engine_response(
            self.spec,
            json.dumps(response),
            offline_only_acknowledged=False,
            owned_simulator_acknowledged=True,
        )
        self.assertEqual(
            "owned_simulator",
            dict(result.metadata.extra)["execution_context"],
        )
        with self.assertRaises(EngineProtocolError):
            parse_engine_response(self.spec, json.dumps(response))

    def test_ip_combo_vectors_and_player_aggregates_are_fully_validated(self):
        mutations = []
        missing_ip_combo = copy.deepcopy(self.response)
        missing_ip_combo["players"][1]["combos"].pop()
        mutations.append(missing_ip_combo)

        invalid_ip_card = copy.deepcopy(self.response)
        invalid_ip_card["players"][1]["combos"][0]["hand"] = "ZZZZ"
        mutations.append(invalid_ip_card)

        wrong_ip_range_weight = copy.deepcopy(self.response)
        wrong_ip_range_weight["players"][1]["combos"][0]["range_weight"] = 0.4
        mutations.append(wrong_ip_range_weight)

        bad_ip_normalized = copy.deepcopy(self.response)
        bad_ip_normalized["players"][1]["combos"][0]["normalized_weight"] = -1
        mutations.append(bad_ip_normalized)

        bad_ip_reach = copy.deepcopy(self.response)
        bad_ip_reach["players"][1]["combos"][0]["reach_weight"] = 0.4
        mutations.append(bad_ip_reach)

        bad_ip_equity = copy.deepcopy(self.response)
        bad_ip_equity["players"][1]["combos"][0]["equity"] = 2
        mutations.append(bad_ip_equity)

        bad_ip_ev = copy.deepcopy(self.response)
        bad_ip_ev["players"][1]["combos"][0]["equilibrium_ev_units"] = -40
        mutations.append(bad_ip_ev)

        nonnull_ip_vectors = copy.deepcopy(self.response)
        nonnull_ip_vectors["players"][1]["combos"][0][
            "root_action_frequencies"
        ] = [1, 0]
        mutations.append(nonnull_ip_vectors)

        bad_total = copy.deepcopy(self.response)
        bad_total["players"][0]["total_reachable_weight"] = 4
        mutations.append(bad_total)

        bad_average_equity = copy.deepcopy(self.response)
        bad_average_equity["players"][0]["average_equity"] = 0.8
        mutations.append(bad_average_equity)

        bad_average_ev = copy.deepcopy(self.response)
        bad_average_ev["players"][0]["average_ev_units"] = 20
        mutations.append(bad_average_ev)

        bad_oop_equilibrium = copy.deepcopy(self.response)
        bad_oop_equilibrium["players"][0]["combos"][0][
            "equilibrium_ev_units"
        ] = 9
        mutations.append(bad_oop_equilibrium)

        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(EngineProtocolError):
                    parse_engine_response(self.spec, json.dumps(mutation))

    def test_convergence_formulas_and_status_are_consistent(self):
        mutations = []
        wrong_target_units = copy.deepcopy(self.response)
        wrong_target_units["convergence"]["target_exploitability_units"] = 0.7
        mutations.append(wrong_target_units)

        wrong_exploitability_pct = copy.deepcopy(self.response)
        wrong_exploitability_pct["convergence"]["exploitability_pct_of_pot"] = 0.9
        mutations.append(wrong_exploitability_pct)

        false_below_target = copy.deepcopy(self.response)
        false_below_target["convergence"]["target_reached"] = False
        mutations.append(false_below_target)

        true_above_target = copy.deepcopy(self.response)
        true_above_target["convergence"]["exploitability_units"] = 0.8
        true_above_target["convergence"]["exploitability_pct_of_pot"] = 0.8
        mutations.append(true_above_target)

        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(EngineProtocolError):
                    parse_engine_response(self.spec, json.dumps(mutation))

    def test_nonfinite_json_and_f64_conversion_are_rejected(self):
        for constant in ("NaN", "Infinity", "-Infinity"):
            payload = json.dumps(self.response).replace(
                '"total": 24', f'"total": {constant}'
            )
            with self.subTest(constant=constant):
                with self.assertRaises(EngineProtocolError):
                    parse_engine_response(self.spec, payload)

        for threshold in (Decimal("1e10000"), Decimal("1e-10000")):
            changed = replace(
                self.spec,
                parameters=replace(
                    self.spec.parameters,
                    add_allin_threshold=threshold,
                ),
            )
            with self.subTest(threshold=threshold):
                with self.assertRaises(OracleValidationError):
                    build_engine_request(changed)

    def test_engine_client_requires_exactly_one_execution_context(self):
        with self.assertRaisesRegex(OracleValidationError, "exactly one"):
            EngineClient(
                "/missing/engine",
                offline_only_acknowledged=False,
            )
        with self.assertRaisesRegex(OracleValidationError, "exactly one"):
            EngineClient(
                "/missing/engine",
                offline_only_acknowledged=True,
                owned_simulator_acknowledged=True,
            )
        client = EngineClient(
            "/missing/engine",
            offline_only_acknowledged=False,
            owned_simulator_acknowledged=True,
        )
        self.assertEqual("owned_simulator", client.execution_context)

    def test_id_commit_provenance_and_root_actions_are_pinned(self):
        mutations = []
        wrong_id = copy.deepcopy(self.response)
        wrong_id["id"] = "other-request"
        mutations.append(wrong_id)

        wrong_commit = copy.deepcopy(self.response)
        wrong_commit["provenance"]["solver"]["commit"] = "a" * 40
        mutations.append(wrong_commit)

        wrong_provenance = copy.deepcopy(self.response)
        wrong_provenance["provenance"]["effective_request"]["max_iterations"] = 999
        mutations.append(wrong_provenance)

        wrong_action = copy.deepcopy(self.response)
        wrong_action["root_actions"][1]["amount"] = 51
        mutations.append(wrong_action)

        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(EngineProtocolError):
                    parse_engine_response(self.spec, json.dumps(mutation))

    def test_duplicate_combos_and_bad_reach_sum_are_rejected(self):
        duplicate = copy.deepcopy(self.response)
        duplicate["players"][0]["combos"].append(
            copy.deepcopy(duplicate["players"][0]["combos"][0])
        )
        with self.assertRaises(EngineProtocolError):
            parse_engine_response(self.spec, json.dumps(duplicate))

        bad_reach = copy.deepcopy(self.response)
        bad_reach["players"][0]["combos"][0]["reach_weight"] = 0.2
        with self.assertRaises(EngineProtocolError):
            parse_engine_response(self.spec, json.dumps(bad_reach))

    def test_blocker_impossible_root_combo_is_rejected_before_execution(self):
        invalid = replace(
            self.spec,
            oop_range=PlayerRange(
                Position.OOP,
                (
                    WeightedCombo(("As", "Ks")),
                    WeightedCombo(("Ad", "Kd")),
                ),
            ),
            ip_range=PlayerRange(
                Position.IP,
                (WeightedCombo(("As", "Ks")),),
            ),
        )
        with self.assertRaisesRegex(
            OracleValidationError,
            "blocker-impossible.*zero compatible opponent mass",
        ):
            build_engine_request(invalid)
        with patch(
            "gto_oracle.engine_client._run_engine_process",
        ) as run:
            with self.assertRaisesRegex(
                OracleValidationError,
                "blocker-impossible",
            ):
                EngineClient(
                    "/missing/engine",
                    offline_only_acknowledged=True,
                    timeout_seconds=1,
                ).solve(invalid)
        run.assert_not_called()

    def test_client_hashes_executed_binary_into_metadata(self):
        binary_bytes = b"offline solver fixture binary"
        expected_digest = hashlib.sha256(binary_bytes).hexdigest()
        completed = SimpleNamespace(
            stdout=json.dumps(self.response),
            stderr="",
            returncode=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "solver-fixture"
            binary.write_bytes(binary_bytes)
            binary.chmod(0o700)
            with patch(
                "gto_oracle.engine_client._run_engine_process",
                return_value=completed,
            ) as run:
                result = EngineClient(
                    binary,
                    offline_only_acknowledged=True,
                    timeout_seconds=1,
                ).solve(self.spec)
            self.assertEqual(binary.resolve(), run.call_args.args[0])
        self.assertEqual(
            expected_digest,
            dict(result.metadata.extra)["binary_sha256"],
        )

    def test_relative_binary_is_resolved_before_hashing_and_execution(self):
        cwd_binary_bytes = b"working-directory engine"
        path_binary_bytes = b"PATH engine must not execute"
        completed = SimpleNamespace(
            stdout=json.dumps(self.response),
            stderr="",
            returncode=0,
        )
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            working = root / "working"
            path_directory = root / "path"
            working.mkdir()
            path_directory.mkdir()
            cwd_binary = working / "solver"
            path_binary = path_directory / "solver"
            cwd_binary.write_bytes(cwd_binary_bytes)
            path_binary.write_bytes(path_binary_bytes)
            cwd_binary.chmod(0o700)
            path_binary.chmod(0o700)
            try:
                os.chdir(working)
                with patch.dict(
                    os.environ,
                    {
                        "PATH": (
                            f"{path_directory}{os.pathsep}"
                            f"{os.environ.get('PATH', '')}"
                        )
                    },
                ), patch(
                    "gto_oracle.engine_client._run_engine_process",
                    return_value=completed,
                ) as run:
                    client = EngineClient(
                        "./solver",
                        offline_only_acknowledged=True,
                        timeout_seconds=1,
                    )
                    result = client.solve(self.spec)
            finally:
                os.chdir(original_cwd)

        expected_binary = cwd_binary.resolve()
        self.assertEqual(expected_binary, client.binary)
        self.assertEqual(expected_binary, run.call_args.args[0])
        self.assertTrue(run.call_args.args[0].is_absolute())
        self.assertEqual(
            hashlib.sha256(cwd_binary_bytes).hexdigest(),
            dict(result.metadata.extra)["binary_sha256"],
        )

    def test_process_output_limits_are_enforced_while_collecting(self):
        for stream, limit_name in (
            ("stdout", "MAX_RESPONSE_BYTES"),
            ("stderr", "MAX_STDERR_BYTES"),
        ):
            with self.subTest(stream=stream), tempfile.TemporaryDirectory() as directory:
                binary = Path(directory) / f"oversized-{stream}"
                binary.write_text(
                    f"#!{sys.executable}\n"
                    "import sys\n"
                    "sys.stdin.buffer.read()\n"
                    f"sys.{stream}.buffer.write(b'x' * 4096)\n",
                    encoding="utf-8",
                )
                binary.chmod(0o700)
                with patch(
                    f"gto_oracle.engine_client.{limit_name}",
                    1024,
                ):
                    with self.assertRaisesRegex(
                        EngineProcessError,
                        f"{stream} exceeded the 1024-byte safety limit",
                    ):
                        EngineClient(
                            binary,
                            offline_only_acknowledged=True,
                            timeout_seconds=2,
                        ).solve(self.spec)

    def test_error_and_malformed_responses_are_typed(self):
        error = {
            "schema_version": 1,
            "id": self.spec.cache_key,
            "operation": "solve_root",
            "status": "error",
            "error": {"code": "MEMORY_LIMIT", "message": "too large"},
        }
        with self.assertRaises(EngineResponseError) as raised:
            parse_engine_response(self.spec, json.dumps(error))
        self.assertEqual("MEMORY_LIMIT", raised.exception.code)
        with self.assertRaises(EngineProtocolError):
            parse_engine_response(self.spec, "{not-json")
        with self.assertRaises(EngineProtocolError):
            parse_engine_response(
                self.spec,
                '{"schema_version":1,"schema_version":1}',
            )

    def test_empty_non_root_specs_are_rejected_before_execution(self):
        with self.assertRaises(OracleValidationError):
            build_engine_request(replace(self.spec, acting_player=Position.IP))


class EngineNodeProtocolTests(unittest.TestCase):
    def setUp(self):
        self.spec = make_engine_node_spec()
        self.response = make_engine_node_response(self.spec)

    def test_node_request_adds_exact_expectations_without_changing_legacy_root(self):
        root_request = build_engine_request(make_engine_spec())
        self.assertEqual("solve_root", root_request["operation"])
        for key in (
            "action_history",
            "expected_current_player",
            "expected_facing_bet",
            "expected_node_actions",
        ):
            self.assertNotIn(key, root_request)

        request = build_engine_request(self.spec)
        self.assertEqual("solve_node", request["operation"])
        self.assertEqual(
            [{"kind": "BET", "amount": 50}],
            request["action_history"],
        )
        self.assertEqual("IP", request["expected_current_player"])
        self.assertEqual(50, request["expected_facing_bet"])
        self.assertEqual(
            [
                {"kind": "CALL", "amount": None},
                {"kind": "FOLD", "amount": None},
                {"kind": "RAISE", "amount": 125},
            ],
            request["expected_node_actions"],
        )
        self.assertEqual(
            request["action_history"],
            self.response["provenance"]["effective_request"]["action_history"],
        )

    def test_ip_node_builds_positive_reach_subset_with_action_evs(self):
        result = parse_engine_response(self.spec, json.dumps(self.response))
        self.assertEqual(1, len(result.combo_policies))
        policy = result.combo_policies[0]
        self.assertEqual(("Kc", "Ac"), policy.private_combo)
        self.assertEqual(Decimal(1), policy.reach_weight)
        by_action = {value.action: value for value in policy.action_values}
        self.assertEqual(
            Decimal("0.6"),
            by_action[Action(ActionKind.CALL)].frequency,
        )
        self.assertEqual(
            Decimal(8),
            by_action[Action(ActionKind.RAISE, 125)].ev,
        )
        extra = dict(result.metadata.extra)
        self.assertEqual("solve_node", extra["protocol_operation"])
        self.assertEqual("0.3", extra["node_total_reachable_weight"])

    def test_oop_descendant_actor_is_supported(self):
        spec = make_engine_node_spec(acting_player=Position.OOP)
        response = make_engine_node_response(spec)
        result = parse_engine_response(spec, json.dumps(response))
        self.assertEqual(("Kd", "Ad"), result.combo_policies[0].private_combo)
        self.assertEqual("OOP", response["current_player"])

    def test_node_owned_simulator_context_is_exact(self):
        response = copy.deepcopy(self.response)
        response["provenance"]["offline_only_acknowledged"] = False
        response["provenance"]["owned_simulator_acknowledged"] = True
        result = parse_engine_response(
            self.spec,
            json.dumps(response),
            offline_only_acknowledged=False,
            owned_simulator_acknowledged=True,
        )
        self.assertEqual(
            "owned_simulator",
            dict(result.metadata.extra)["execution_context"],
        )
        with self.assertRaises(EngineProtocolError):
            parse_engine_response(self.spec, json.dumps(response))

    def test_engine_client_selects_solve_node_on_the_subprocess_wire(self):
        binary_bytes = b"node-capable local solver fixture"
        completed = SimpleNamespace(
            stdout=json.dumps(self.response),
            stderr="",
            returncode=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "solver-node-fixture"
            binary.write_bytes(binary_bytes)
            binary.chmod(0o700)
            with patch(
                "gto_oracle.engine_client._run_engine_process",
                return_value=completed,
            ) as run:
                result = EngineClient(
                    binary,
                    offline_only_acknowledged=True,
                    timeout_seconds=1,
                ).solve(self.spec)
        wire_request = json.loads(run.call_args.args[1])
        self.assertEqual("solve_node", wire_request["operation"])
        self.assertEqual(
            [{"kind": "BET", "amount": 50}],
            wire_request["action_history"],
        )
        self.assertEqual(
            hashlib.sha256(binary_bytes).hexdigest(),
            dict(result.metadata.extra)["binary_sha256"],
        )

    def test_node_envelope_actor_actions_and_provenance_are_pinned(self):
        mutations = []

        wrong_operation = copy.deepcopy(self.response)
        wrong_operation["operation"] = "solve_root"
        mutations.append(wrong_operation)

        wrong_actor = copy.deepcopy(self.response)
        wrong_actor["current_player"] = "OOP"
        mutations.append(wrong_actor)

        wrong_action = copy.deepcopy(self.response)
        wrong_action["node_actions"][2]["amount"] = 126
        mutations.append(wrong_action)

        wrong_history = copy.deepcopy(self.response)
        wrong_history["provenance"]["effective_request"]["action_history"][0][
            "amount"
        ] = 51
        mutations.append(wrong_history)

        wrong_expected_actor = copy.deepcopy(self.response)
        wrong_expected_actor["provenance"]["effective_request"][
            "expected_current_player"
        ] = "OOP"
        mutations.append(wrong_expected_actor)

        wrong_facing = copy.deepcopy(self.response)
        wrong_facing["provenance"]["effective_request"][
            "expected_facing_bet"
        ] = 49
        mutations.append(wrong_facing)

        wrong_expected_actions = copy.deepcopy(self.response)
        wrong_expected_actions["provenance"]["effective_request"][
            "expected_node_actions"
        ][2]["amount"] = 126
        mutations.append(wrong_expected_actions)

        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(EngineProtocolError):
                    parse_engine_response(self.spec, json.dumps(mutation))

    def test_node_weights_vectors_and_evs_are_strictly_validated(self):
        mutations = []

        wrong_input = copy.deepcopy(self.response)
        wrong_input["policies"][0]["input_range_weight"] = 0.5
        mutations.append(wrong_input)

        impossible_path = copy.deepcopy(self.response)
        impossible_path["policies"][0]["path_weight"] = 0.8
        mutations.append(impossible_path)

        bad_joint = copy.deepcopy(self.response)
        bad_joint["policies"][0]["joint_compatible_weight"] = 0.2
        mutations.append(bad_joint)

        bad_reach = copy.deepcopy(self.response)
        bad_reach["policies"][0]["conditional_reach_weight"] = 0.5
        mutations.append(bad_reach)

        bad_total = copy.deepcopy(self.response)
        bad_total["node_total_reachable_weight"] = 0.4
        mutations.append(bad_total)

        bad_frequency_vector = copy.deepcopy(self.response)
        bad_frequency_vector["policies"][0]["node_action_frequencies"].pop()
        mutations.append(bad_frequency_vector)

        bad_ev_vector = copy.deepcopy(self.response)
        bad_ev_vector["policies"][0]["node_action_evs_units"].pop()
        mutations.append(bad_ev_vector)

        bad_equilibrium_ev = copy.deepcopy(self.response)
        bad_equilibrium_ev["policies"][0]["equilibrium_ev_units"] = 9.0
        mutations.append(bad_equilibrium_ev)

        duplicate = copy.deepcopy(self.response)
        duplicate["policies"].append(copy.deepcopy(duplicate["policies"][0]))
        duplicate["policies"][0]["joint_compatible_weight"] = 0.15
        duplicate["policies"][0]["conditional_reach_weight"] = 0.5
        duplicate["policies"][1]["joint_compatible_weight"] = 0.15
        duplicate["policies"][1]["conditional_reach_weight"] = 0.5
        mutations.append(duplicate)

        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(EngineProtocolError):
                    parse_engine_response(self.spec, json.dumps(mutation))


class SQLiteCacheTests(unittest.TestCase):
    def test_transactional_idempotent_roundtrip(self):
        spec = make_spec()
        result = make_result(spec)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oracle.sqlite3"
            with OracleCache(path) as cache:
                self.assertIsNone(cache.get(spec))
                cache.put(spec, result)
                cache.put(spec, result)
                self.assertEqual(1, cache.result_count())
                self.assertEqual(2, cache.combo_policy_count())
                self.assertEqual(6, cache.action_value_count())
                self.assertEqual(result, cache.get(spec))

            connection = sqlite3.connect(path)
            try:
                self.assertEqual(
                    1,
                    connection.execute("SELECT COUNT(*) FROM oracle_results").fetchone()[0],
                )
                self.assertEqual(
                    6,
                    connection.execute(
                        "SELECT COUNT(*) FROM oracle_action_values"
                    ).fetchone()[0],
                )
                self.assertEqual(
                    2,
                    connection.execute(
                        "SELECT COUNT(DISTINCT combo_cards) FROM oracle_action_values"
                    ).fetchone()[0],
                )
            finally:
                connection.close()

    def test_descendant_subset_policy_roundtrips_without_schema_changes(self):
        spec = make_engine_node_spec()
        result = parse_engine_response(
            spec,
            json.dumps(make_engine_node_response(spec)),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "node-oracle.sqlite3"
            with OracleCache(path) as cache:
                cache.put(spec, result)
                loaded = cache.get(spec)
                self.assertEqual(result, loaded)
                self.assertEqual(1, cache.result_count())
                self.assertEqual(1, cache.combo_policy_count())
                self.assertEqual(3, cache.action_value_count())

    def test_replacing_same_spec_updates_metadata_and_actions_atomically(self):
        spec = make_spec()
        first = make_result(spec)
        second = make_result(
            spec,
            metadata=make_metadata(elapsed_seconds=Decimal("2.0")),
        )
        with OracleCache() as cache:
            cache.put(spec, first)
            cache.put(spec, second)
            loaded = cache.get(spec)
            self.assertEqual(Decimal("2"), loaded.metadata.elapsed_seconds)
            self.assertEqual(1, cache.result_count())
            self.assertEqual(2, cache.combo_policy_count())
            self.assertEqual(6, cache.action_value_count())

    def test_binary_digest_qualifies_cache_hits(self):
        spec = make_spec()
        digest = "a" * 64
        result = make_result(
            spec,
            metadata=make_metadata(
                extra=(("binary_sha256", digest),),
            ),
        )
        with OracleCache() as cache:
            cache.put(spec, result)
            self.assertEqual(
                result,
                cache.get(spec, expected_binary_sha256=digest),
            )
            self.assertIsNone(
                cache.get(spec, expected_binary_sha256="b" * 64)
            )
            with self.assertRaises(OracleValidationError):
                cache.get(spec, expected_binary_sha256="not-a-digest")

    def test_execution_context_qualifies_cache_hits(self):
        spec = make_spec()
        result = make_result(
            spec,
            metadata=make_metadata(
                extra=(("execution_context", "offline"),),
            ),
        )
        with OracleCache() as cache:
            cache.put(spec, result)
            self.assertEqual(
                result,
                cache.get(spec, expected_execution_context="offline"),
            )
            self.assertIsNone(
                cache.get(spec, expected_execution_context="owned_simulator")
            )
            with self.assertRaises(OracleValidationError):
                cache.get(spec, expected_execution_context="unknown")


class OfflineIsolationTests(unittest.TestCase):
    def test_live_assistant_does_not_import_gto_oracle(self):
        source = Path("poker_assistant.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
        self.assertFalse(
            any(
                module == "gto_oracle" or module.startswith("gto_oracle.")
                for module in imported_modules
            )
        )


if __name__ == "__main__":
    unittest.main()
