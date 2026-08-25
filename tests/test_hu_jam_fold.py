"""Tests for the explicit raked HU jam/fold reference harness."""

from __future__ import annotations

import copy
from decimal import Decimal
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from hu_jam_fold import (
    ARTIFACT_TYPE,
    BehaviorStrategy,
    JamFoldValidationError,
    RakeProfile,
    load_game_artifact,
    load_solution,
    main,
    measure_best_responses,
    parse_game_payload,
    solve_game,
    verify_solution_payload,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_GAME = ROOT / "tests" / "fixtures" / "hu_jam_fold_two_type_raked_v1.json"


def game_payload() -> dict:
    return json.loads(REFERENCE_GAME.read_text(encoding="utf-8"))


class RakeSemanticsTests(unittest.TestCase):
    def test_no_flop_no_drop_keeps_fold_terminals_zero_sum(self):
        game = load_game_artifact(REFERENCE_GAME)
        sb_fold = game.profile.sb_fold_payoffs()
        bb_fold = game.profile.bb_fold_payoffs()
        self.assertEqual(sb_fold, (Decimal("-0.5"), Decimal("0.5")))
        self.assertEqual(bb_fold, (Decimal("1"), Decimal("-1")))

    def test_called_jam_deducts_capped_floor_rounded_rake(self):
        game = load_game_artifact(REFERENCE_GAME)
        sb, bb = game.profile.called_jam_payoffs(Decimal("0.5"))
        self.assertEqual(sb, Decimal("-0.1"))
        self.assertEqual(bb, Decimal("-0.1"))
        self.assertEqual(sb + bb, Decimal("-0.2"))

        cap_profile = RakeProfile(
            rate_pct=Decimal("20"),
            cap_bb=Decimal("0.5"),
            chip_unit_bb=Decimal("0.01"),
            no_flop_no_drop=True,
        )
        self.assertEqual(cap_profile.amount(Decimal("100"), flop_seen=True), Decimal("0.5"))

    def test_fold_terminal_can_be_raked_only_when_profile_explicitly_allows_it(self):
        payload = game_payload()
        payload["profile"]["rake"]["no_flop_no_drop"] = False
        game = parse_game_payload(payload)
        sb, bb = game.profile.sb_fold_payoffs()
        self.assertEqual(sb, Decimal("-0.5"))
        self.assertEqual(bb, Decimal("0.43"))
        self.assertEqual(sb + bb, Decimal("-0.07"))


class ArtifactValidationTests(unittest.TestCase):
    def test_reference_artifact_is_canonical_and_stable(self):
        first = load_game_artifact(REFERENCE_GAME)
        second = parse_game_payload(first.to_payload())
        self.assertEqual(first, second)
        self.assertEqual(len(first.fingerprint), 64)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.sb_types, ("strong", "weak"))
        self.assertEqual(first.bb_types, ("strong", "weak"))

    def test_rejects_unknown_fields_floats_and_duplicate_pairs(self):
        unknown = game_payload()
        unknown["profile"]["surprise"] = True
        with self.assertRaisesRegex(JamFoldValidationError, "keys mismatch"):
            parse_game_payload(unknown)

        binary_float = game_payload()
        binary_float["profile"]["effective_stack_bb"] = 2.0
        with self.assertRaisesRegex(JamFoldValidationError, "binary float"):
            parse_game_payload(binary_float)

        duplicate = game_payload()
        duplicate["private_model"]["deals"].append(
            copy.deepcopy(duplicate["private_model"]["deals"][0])
        )
        with self.assertRaisesRegex(JamFoldValidationError, "repeat a type pair"):
            parse_game_payload(duplicate)

        boolean_version = game_payload()
        boolean_version["schema_version"] = True
        with self.assertRaisesRegex(JamFoldValidationError, "schema_version"):
            parse_game_payload(boolean_version)

    def test_loader_rejects_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            duplicate = Path(directory) / "duplicate.json"
            duplicate.write_text(
                '{"schema_version":1,"schema_version":1}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(JamFoldValidationError, "duplicate JSON key"):
                load_game_artifact(duplicate)

    def test_rejects_ambiguous_or_misaligned_rake_profiles(self):
        zero_cap = game_payload()
        zero_cap["profile"]["rake"]["cap_bb"] = "0"
        with self.assertRaisesRegex(JamFoldValidationError, "positive explicit cap"):
            parse_game_payload(zero_cap)

        misaligned = game_payload()
        misaligned["profile"]["rake"]["chip_unit_bb"] = "0.03"
        with self.assertRaisesRegex(JamFoldValidationError, "align"):
            parse_game_payload(misaligned)

    def test_float64_certificate_rejects_numeric_underflow_and_overflow(self):
        underflow = game_payload()
        underflow["private_model"]["deals"][0]["weight"] = "1e-10000"
        with self.assertRaisesRegex(JamFoldValidationError, "underflows"):
            parse_game_payload(underflow)

        overflow = game_payload()
        overflow["profile"]["effective_stack_bb"] = "1e10000"
        with self.assertRaisesRegex(JamFoldValidationError, "float64"):
            parse_game_payload(overflow)


class BestResponseTests(unittest.TestCase):
    def test_exact_enumeration_detects_a_bad_strategy_for_both_players(self):
        game = load_game_artifact(REFERENCE_GAME)
        strategy = BehaviorStrategy(
            sb_jam=(("strong", 0.0), ("weak", 1.0)),
            bb_call=(("strong", 0.0), ("weak", 0.0)),
        )
        report = measure_best_responses(game, strategy)
        self.assertGreater(report.sb_deviation_gain_bb, 0.7)
        self.assertGreater(report.bb_deviation_gain_bb, 0.0)
        self.assertEqual(report.expected_rake_bb, 0.0)

    def test_solver_reaches_measured_bilateral_target_on_reference_game(self):
        game = load_game_artifact(REFERENCE_GAME)
        solution = solve_game(
            game,
            target_epsilon_bb=0.0005,
            max_iterations=10_000,
            min_iterations=2_000,
            check_every=1_000,
        )
        self.assertTrue(solution.converged)
        self.assertEqual(solution.iterations, 2_000)
        self.assertLessEqual(solution.report.sb_deviation_gain_bb, 0.0005)
        self.assertLessEqual(solution.report.bb_deviation_gain_bb, 0.0005)
        self.assertLessEqual(solution.report.epsilon_bb, 0.0005)

        sb = dict(solution.strategy.sb_jam)
        bb = dict(solution.strategy.bb_call)
        self.assertGreater(sb["strong"], 0.999)
        self.assertAlmostEqual(sb["weak"], 0.27, delta=0.02)
        self.assertGreater(bb["strong"], 0.999)
        self.assertAlmostEqual(bb["weak"], 0.69, delta=0.02)
        self.assertGreater(solution.report.expected_rake_bb, 0.1)

    def test_unreachable_bb_infoset_has_no_false_deviation_gain(self):
        game = load_game_artifact(REFERENCE_GAME)
        fold_all = BehaviorStrategy(
            sb_jam=(("strong", 0.0), ("weak", 0.0)),
            bb_call=(("strong", 1.0), ("weak", 1.0)),
        )
        report = measure_best_responses(game, fold_all)
        self.assertEqual(report.bb_deviation_gain_bb, 0.0)


class SolutionManifestTests(unittest.TestCase):
    def setUp(self):
        self.game = load_game_artifact(REFERENCE_GAME)
        self.solution = solve_game(
            self.game,
            target_epsilon_bb=0.0005,
            max_iterations=10_000,
            min_iterations=2_000,
            check_every=1_000,
        )
        self.payload = self.solution.to_payload()

    def test_manifest_is_explicitly_restricted_and_reverifiable(self):
        self.assertEqual(self.payload["artifact_type"], ARTIFACT_TYPE)
        self.assertFalse(self.payload["claim"]["full_hunl"])
        self.assertTrue(self.payload["claim"]["rake_aware"])
        self.assertEqual(
            self.payload["claim"]["certificate_method"],
            "bilateral_full_best_response_enumeration",
        )
        self.assertEqual(
            self.payload["claim"]["certificate_numeric"], "ieee754_float64"
        )
        verified = verify_solution_payload(
            self.game, self.payload, max_epsilon_bb=0.0005
        )
        self.assertTrue(verified.converged)
        self.assertAlmostEqual(
            verified.report.epsilon_bb, self.solution.report.epsilon_bb
        )

    def test_tampered_strategy_or_certificate_fails_closed(self):
        strategy = copy.deepcopy(self.payload)
        strategy["strategy"]["sb"][0]["jam"] = "0"
        strategy["strategy"]["sb"][0]["fold"] = "1"
        with self.assertRaisesRegex(JamFoldValidationError, "fingerprint"):
            verify_solution_payload(self.game, strategy)

        certificate = copy.deepcopy(self.payload)
        certificate["claim"]["achieved_epsilon_bb"] = "0"
        with self.assertRaisesRegex(JamFoldValidationError, "mismatch"):
            verify_solution_payload(self.game, certificate)

        scope = copy.deepcopy(self.payload)
        scope["claim"]["full_hunl"] = True
        with self.assertRaisesRegex(JamFoldValidationError, "full_hunl=false"):
            verify_solution_payload(self.game, scope)

    def test_strict_loader_rejects_an_unconverged_but_diagnostic_manifest(self):
        weak = solve_game(
            self.game,
            target_epsilon_bb=0.0,
            max_iterations=1,
            min_iterations=1,
            check_every=1,
        )
        self.assertFalse(weak.converged)
        with self.assertRaisesRegex(JamFoldValidationError, "did not reach"):
            verify_solution_payload(self.game, weak.to_payload())
        diagnostic = verify_solution_payload(
            self.game,
            weak.to_payload(),
            require_target_reached=False,
        )
        self.assertFalse(diagnostic.converged)

    def test_target_comparison_does_not_reuse_manifest_value_tolerance(self):
        near_miss = copy.deepcopy(self.payload)
        near_miss["claim"]["target_epsilon_bb"] = str(
            self.solution.report.epsilon_bb - 5e-11
        )
        near_miss["claim"]["target_reached"] = False
        diagnostic = verify_solution_payload(
            self.game,
            near_miss,
            require_target_reached=False,
        )
        self.assertFalse(diagnostic.converged)

    def test_cli_solve_and_verify_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "solution.json"
            stdout = io.StringIO()
            with mock.patch("sys.stdout", stdout):
                status = main(
                    [
                        "solve",
                        "--game",
                        str(REFERENCE_GAME),
                        "--output",
                        str(output),
                        "--target-epsilon-bb",
                        "0.0005",
                        "--max-iterations",
                        "10000",
                        "--min-iterations",
                        "2000",
                        "--check-every",
                        "1000",
                    ]
                )
            self.assertEqual(status, 0)
            self.assertTrue(output.exists())
            loaded = load_solution(self.game, output, max_epsilon_bb=0.0005)
            self.assertTrue(loaded.converged)

            with mock.patch("sys.stdout", io.StringIO()):
                status = main(
                    [
                        "verify",
                        "--game",
                        str(REFERENCE_GAME),
                        "--solution",
                        str(output),
                        "--max-epsilon-bb",
                        "0.0005",
                    ]
                )
            self.assertEqual(status, 0)


if __name__ == "__main__":
    unittest.main()
