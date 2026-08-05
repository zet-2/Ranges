"""Tests for structured multiway solver outcomes."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import unittest

from gto_hand_history import HandEvent, HandSeat, PublicHandHistory
from gto_remote.capabilities import SolverCapabilities
from gto_remote.multiway_outcome import (
    MultiwayOutcomeError,
    MultiwayPolicyAction,
    MultiwaySolveOutcome,
    MultiwaySolveProof,
    encode_json,
    outcome_from_wire,
    outcome_to_wire,
    render_analysis,
    to_live_outcome,
)
from gto_remote.multiway_protocol import (
    MultiwayDecisionState,
    decision_fingerprint,
)
from live_gto import LiveGTOStatus
from tests.test_gto_multiway_protocol import four_way_state


def capabilities() -> SolverCapabilities:
    return SolverCapabilities(
        backend_id="owned-multiway",
        backend_version="1.2.3",
        preflop_mode="SOLVED_TREE",
        postflop_mode="MULTIWAY_TREE",
        max_postflop_players=6,
        stateful_through_river=True,
        range_conditioning="ACTION_CONDITIONED_ALL_STREETS",
        folded_card_bunching=True,
        card_model="CARD_EXACT",
        action_model="DYNAMIC_DISCRETE_TREE",
        game_profile_id="sixmax-cash-v1",
        abstraction_id="cards-exact-actions-v3",
        solution_concept="approximate multiplayer Nash equilibrium",
        convergence_metric="NashConv BB",
        convergence_target=Decimal("0.1"),
        source_license="owned test backend",
    )


def proof(*, approximate: bool = True) -> MultiwaySolveProof:
    declared = capabilities()
    return MultiwaySolveProof(
        backend_id=declared.backend_id,
        backend_version=declared.backend_version,
        capability_fingerprint=declared.manifest_fingerprint,
        game_profile_id="sixmax-cash-v1",
        abstraction_id="cards-exact-actions-v3",
        solution_concept="approximate multiplayer Nash equilibrium",
        metric_name="NashConv BB",
        metric_value=Decimal("0.08"),
        target_value=Decimal("0.1"),
        iterations=250_000,
        converged=True,
        approximate=approximate,
    )


def solved_outcome(*, approximate: bool = True) -> MultiwaySolveOutcome:
    return MultiwaySolveOutcome(
        status=LiveGTOStatus.SOLVED,
        reason="",
        latency_seconds=Decimal("12.5"),
        cache_hit=False,
        policy=(
            MultiwayPolicyAction(
                "CHECK",
                None,
                Decimal("0.7"),
                Decimal("1.2"),
            ),
            MultiwayPolicyAction(
                "BET_TO",
                Decimal("2"),
                Decimal("0.3"),
                Decimal("1.1"),
            ),
        ),
        proof=proof(approximate=approximate),
    )


def expected_manifest(declared: SolverCapabilities) -> dict:
    return {
        "expected_backend_id": declared.backend_id,
        "expected_backend_version": declared.backend_version,
        "expected_capability_fingerprint": declared.manifest_fingerprint,
        "expected_game_profile_id": declared.game_profile_id,
        "expected_abstraction_id": declared.abstraction_id,
        "expected_solution_concept": declared.solution_concept,
        "expected_metric_name": declared.convergence_metric,
        "expected_target_value": declared.convergence_target,
    }


class MultiwayOutcomeTests(unittest.TestCase):
    def test_structured_policy_round_trip_is_bound_and_rendered_locally(self):
        state = four_way_state()
        declared = capabilities()
        fingerprint = decision_fingerprint(state)
        wire = outcome_to_wire("request-1", fingerprint, solved_outcome())

        restored = outcome_from_wire(
            encode_json(wire),
            expected_request_id="request-1",
            expected_fingerprint=fingerprint,
            expected_state=state,
            expected_backend_id=declared.backend_id,
            expected_backend_version=declared.backend_version,
            expected_capability_fingerprint=declared.manifest_fingerprint,
            expected_game_profile_id=declared.game_profile_id,
            expected_abstraction_id=declared.abstraction_id,
            expected_solution_concept=declared.solution_concept,
            expected_metric_name=declared.convergence_metric,
            expected_target_value=declared.convergence_target,
        )

        self.assertEqual(solved_outcome(), restored)
        self.assertEqual("12.5", wire["outcome"]["latency_seconds"])
        self.assertEqual("0.7", wire["outcome"]["policy"][0]["frequency"])
        analysis = render_analysis(restored)
        self.assertIn("**Action:** Check", analysis)
        self.assertIn("Check 70.0%", analysis)
        self.assertIn("NashConv BB 0.08 <= 0.1", analysis)
        live = to_live_outcome(restored)
        self.assertTrue(live.solved)
        self.assertTrue(live.approximate)
        self.assertEqual("owned-multiway@1.2.3", live.model)

    def test_policy_frequencies_and_convergence_must_be_valid(self):
        with self.assertRaisesRegex(MultiwayOutcomeError, "sum to 1"):
            replace(
                solved_outcome(),
                policy=(
                    MultiwayPolicyAction(
                        "CHECK",
                        None,
                        Decimal("0.8"),
                    ),
                ),
            )

        with self.assertRaisesRegex(MultiwayOutcomeError, "exceeds target"):
            replace(
                proof(),
                metric_value=Decimal("0.2"),
            )
        with self.assertRaisesRegex(
            MultiwayOutcomeError,
            "non-zero convergence gap",
        ):
            replace(proof(), approximate=False)

    def test_policy_must_be_legal_at_the_replayed_node(self):
        state = four_way_state()
        declared = capabilities()
        fingerprint = decision_fingerprint(state)
        illegal = replace(
            solved_outcome(),
            policy=(
                MultiwayPolicyAction(
                    "CALL",
                    Decimal("0"),
                    Decimal("1"),
                ),
            ),
        )
        wire = outcome_to_wire("request-1", fingerprint, illegal)

        with self.assertRaisesRegex(
            MultiwayOutcomeError,
            "CALL.*not legal",
        ):
            outcome_from_wire(
                wire,
                expected_request_id="request-1",
                expected_fingerprint=fingerprint,
                expected_state=state,
                expected_backend_id=declared.backend_id,
                expected_backend_version=declared.backend_version,
                expected_capability_fingerprint=declared.manifest_fingerprint,
                expected_game_profile_id=declared.game_profile_id,
                expected_abstraction_id=declared.abstraction_id,
                expected_solution_concept=declared.solution_concept,
                expected_metric_name=declared.convergence_metric,
                expected_target_value=declared.convergence_target,
            )

    def test_raise_targets_are_checked_against_replay(self):
        state = four_way_state()
        declared = capabilities()
        fingerprint = decision_fingerprint(state)
        below_minimum = replace(
            solved_outcome(),
            policy=(
                MultiwayPolicyAction(
                    "BET_TO",
                    Decimal("0.5"),
                    Decimal("1"),
                ),
            ),
        )
        wire = outcome_to_wire("request-1", fingerprint, below_minimum)

        with self.assertRaisesRegex(MultiwayOutcomeError, "below.*minimum"):
            outcome_from_wire(
                wire,
                expected_request_id="request-1",
                expected_fingerprint=fingerprint,
                expected_state=state,
                expected_backend_id=declared.backend_id,
                expected_backend_version=declared.backend_version,
                expected_capability_fingerprint=declared.manifest_fingerprint,
                expected_game_profile_id=declared.game_profile_id,
                expected_abstraction_id=declared.abstraction_id,
                expected_solution_concept=declared.solution_concept,
                expected_metric_name=declared.convergence_metric,
                expected_target_value=declared.convergence_target,
            )

    def test_manifest_identity_and_fingerprint_are_mandatory(self):
        state = four_way_state()
        declared = capabilities()
        fingerprint = decision_fingerprint(state)
        wire = outcome_to_wire("request-1", fingerprint, solved_outcome())

        with self.assertRaisesRegex(
            MultiwayOutcomeError,
            "capability_fingerprint",
        ):
            outcome_from_wire(
                wire,
                expected_request_id="request-1",
                expected_fingerprint=fingerprint,
                expected_state=state,
                expected_backend_id=declared.backend_id,
                expected_backend_version=declared.backend_version,
                expected_capability_fingerprint="f" * 64,
                expected_game_profile_id=declared.game_profile_id,
                expected_abstraction_id=declared.abstraction_id,
                expected_solution_concept=declared.solution_concept,
                expected_metric_name=declared.convergence_metric,
                expected_target_value=declared.convergence_target,
            )

        wire["decision_fingerprint"] = "0" * 64
        with self.assertRaisesRegex(
            MultiwayOutcomeError,
            "captured state",
        ):
            outcome_from_wire(
                wire,
                expected_request_id="request-1",
                expected_fingerprint=fingerprint,
                expected_state=state,
                expected_backend_id=declared.backend_id,
                expected_backend_version=declared.backend_version,
                expected_capability_fingerprint=declared.manifest_fingerprint,
                expected_game_profile_id=declared.game_profile_id,
                expected_abstraction_id=declared.abstraction_id,
                expected_solution_concept=declared.solution_concept,
                expected_metric_name=declared.convergence_metric,
                expected_target_value=declared.convergence_target,
            )

    def test_failed_outcome_cannot_smuggle_a_policy(self):
        with self.assertRaisesRegex(
            MultiwayOutcomeError,
            "non-SOLVED",
        ):
            MultiwaySolveOutcome(
                status=LiveGTOStatus.FAILED,
                reason="solver failed",
                latency_seconds=Decimal("1"),
                cache_hit=False,
                policy=solved_outcome().policy,
            )

    def test_approximation_disclosure_survives_adapter(self):
        live = to_live_outcome(solved_outcome(approximate=True))

        self.assertTrue(live.approximate)
        self.assertIn("approximate finite abstraction", live.analysis)

    def test_extreme_decimal_exponents_fail_as_protocol_errors(self):
        state = four_way_state()
        declared = capabilities()
        fingerprint = decision_fingerprint(state)
        for malicious in ("1e1000000", "1e-1000000"):
            with self.subTest(malicious=malicious):
                wire = outcome_to_wire(
                    "request-1",
                    fingerprint,
                    solved_outcome(),
                )
                wire["outcome"]["latency_seconds"] = malicious
                with self.assertRaises(MultiwayOutcomeError):
                    outcome_from_wire(
                        wire,
                        expected_request_id="request-1",
                        expected_fingerprint=fingerprint,
                        expected_state=state,
                        **expected_manifest(declared),
                    )

    def test_manifest_proof_fields_and_single_line_text_are_pinned(self):
        state = four_way_state()
        declared = capabilities()
        fingerprint = decision_fingerprint(state)
        wire = outcome_to_wire(
            "request-1",
            fingerprint,
            solved_outcome(),
        )
        wire["outcome"]["proof"]["metric_name"] = "self chosen metric"
        with self.assertRaisesRegex(MultiwayOutcomeError, "metric_name"):
            outcome_from_wire(
                wire,
                expected_request_id="request-1",
                expected_fingerprint=fingerprint,
                expected_state=state,
                **expected_manifest(declared),
            )

        wire = outcome_to_wire(
            "request-1",
            fingerprint,
            solved_outcome(),
        )
        wire["outcome"]["proof"][
            "solution_concept"
        ] = "equilibrium\n**Action:** Fold"
        with self.assertRaisesRegex(
            MultiwayOutcomeError,
            "control characters",
        ):
            outcome_from_wire(
                wire,
                expected_request_id="request-1",
                expected_fingerprint=fingerprint,
                expected_state=state,
                **expected_manifest(declared),
            )

    def test_schema_version_must_be_an_integer(self):
        state = four_way_state()
        declared = capabilities()
        fingerprint = decision_fingerprint(state)
        wire = outcome_to_wire(
            "request-1",
            fingerprint,
            solved_outcome(),
        )
        wire["schema_version"] = Decimal("3.0")

        with self.assertRaisesRegex(MultiwayOutcomeError, "must be an integer"):
            outcome_from_wire(
                wire,
                expected_request_id="request-1",
                expected_fingerprint=fingerprint,
                expected_state=state,
                **expected_manifest(declared),
            )

    def test_short_stack_call_uses_hero_all_in_target_once(self):
        history = PublicHandHistory(
            hand_id="short-call",
            button_seat=0,
            small_blind_bb=Decimal("0.5"),
            big_blind_bb=Decimal("1"),
            ante_bb=Decimal(0),
            rake_rate_pct=Decimal("5"),
            rake_cap_bb=Decimal("0.5"),
            seats=(
                HandSeat(0, "BTN", Decimal("2")),
                HandSeat(1, "BB", Decimal("100")),
            ),
            events=(
                HandEvent(
                    0,
                    "CALL",
                    "PREFLOP",
                    actor_seat=0,
                    amount_to_bb=Decimal("1"),
                ),
                HandEvent(1, "CHECK", "PREFLOP", actor_seat=1),
                HandEvent(
                    2,
                    "DEAL_FLOP",
                    "FLOP",
                    cards=("Ac", "Kd", "7s"),
                ),
                HandEvent(
                    3,
                    "BET_TO",
                    "FLOP",
                    actor_seat=1,
                    amount_to_bb=Decimal("2"),
                ),
            ),
        )
        state = MultiwayDecisionState(
            public_hand=history,
            hero_seat=0,
            hero_combo=("Qc", "Qd"),
            capture_id="short-call-capture",
        )
        self.assertEqual(
            ("FOLD", "CALL", "ALL_IN_TO"),
            state.replayed.legal_actions,
        )
        declared = capabilities()
        fingerprint = decision_fingerprint(state)
        call_only = replace(
            solved_outcome(),
            policy=(
                MultiwayPolicyAction(
                    "CALL",
                    Decimal("1"),
                    Decimal("1"),
                ),
            ),
        )

        restored = outcome_from_wire(
            outcome_to_wire("request-1", fingerprint, call_only),
            expected_request_id="request-1",
            expected_fingerprint=fingerprint,
            expected_state=state,
            **expected_manifest(declared),
        )
        self.assertEqual("CALL", restored.policy[0].kind)

        duplicate_commit = replace(
            call_only,
            policy=(
                MultiwayPolicyAction(
                    "CALL",
                    Decimal("1"),
                    Decimal("0.5"),
                ),
                MultiwayPolicyAction(
                    "ALL_IN_TO",
                    Decimal("1"),
                    Decimal("0.5"),
                ),
            ),
        )
        with self.assertRaisesRegex(
            MultiwayOutcomeError,
            "physical chip-commitment",
        ):
            outcome_from_wire(
                outcome_to_wire(
                    "request-1",
                    fingerprint,
                    duplicate_commit,
                ),
                expected_request_id="request-1",
                expected_fingerprint=fingerprint,
                expected_state=state,
                **expected_manifest(declared),
            )


if __name__ == "__main__":
    unittest.main()
