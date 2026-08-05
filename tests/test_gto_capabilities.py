"""Tests for audited GTO capability manifests."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import json
import unittest

from gto_remote.capabilities import (
    NATIVE_ROUTER_CAPABILITIES,
    SolverCapabilities,
    SolverCapabilitiesError,
    capabilities_for_router,
    parse_capabilities_json,
)
from gto_remote.server import EvaluationService
from live_gto import LiveDecisionState
from tests.test_gto_hand_history import heads_up_to_turn_history
from gto_hand_history import HandEvent


class _Router:
    def evaluate(self, state):  # pragma: no cover - manifest test only
        raise AssertionError("not called")


class SolverCapabilityTests(unittest.TestCase):
    def test_native_router_never_claims_full_six_max(self):
        capabilities = capabilities_for_router(_Router())
        wire = capabilities.to_wire()

        self.assertIs(capabilities, NATIVE_ROUTER_CAPABILITIES)
        self.assertFalse(capabilities.full_six_max_ready)
        self.assertFalse(wire["full_six_max_ready"])
        self.assertEqual(2, wire["max_postflop_players"])
        self.assertTrue(wire["stateful_through_river"])
        self.assertEqual(
            "ACTION_CONDITIONED_ALL_STREETS",
            wire["range_conditioning"],
        )
        self.assertEqual(
            "DYNAMIC_DISCRETE_TREE",
            wire["action_model"],
        )
        self.assertIn("multiway", " ".join(wire["full_six_max_gaps"]))
        self.assertIs(
            EvaluationService(_Router()).capabilities,
            NATIVE_ROUTER_CAPABILITIES,
        )

    def test_a_complete_declared_backend_has_no_capability_gaps(self):
        complete = SolverCapabilities(
            backend_id="licensed-or-owned-six-max",
            backend_version="1.0",
            preflop_mode="SOLVED_TREE",
            postflop_mode="MULTIWAY_TREE",
            max_postflop_players=6,
            stateful_through_river=True,
            range_conditioning="ACTION_CONDITIONED_ALL_STREETS",
            folded_card_bunching=True,
            card_model="CARD_EXACT",
            action_model="CONTINUOUS_NO_LIMIT",
            game_profile_id="sixmax-complete",
            abstraction_id="card-exact-continuous",
            solution_concept="multiplayer Nash equilibrium",
            convergence_metric="declared abstraction NashConv",
            convergence_target=Decimal(0),
            source_license="private owned source",
        )

        self.assertTrue(complete.full_six_max_ready)
        self.assertEqual((), complete.full_six_max_gaps())
        self.assertTrue(complete.to_wire()["full_six_max_ready"])

    def test_fixed_action_tree_cannot_claim_full_six_max(self):
        fixed = replace(
            self._complete_capabilities(),
            action_model="FIXED_DISCRETE_TREE",
        )

        self.assertFalse(fixed.full_six_max_ready)
        self.assertIn("action model", " ".join(fixed.full_six_max_gaps()))

    def test_placeholder_solution_concept_cannot_claim_readiness_or_exactness(self):
        undeclared = replace(
            self._complete_capabilities(),
            solution_concept="not configured",
        )
        history = heads_up_to_turn_history()
        state = LiveDecisionState(
            hand_id=history.hand_id,
            street="TURN",
            board=("2c", "7d", "Jh", "As"),
            hero_combo=("Qc", "Qd"),
            hero_position="BB",
            villain_position="BTN",
            hero_is_oop=True,
            active_villains=1,
            pot_bb=Decimal("11.5"),
            hero_stack_bb=Decimal("94.5"),
            villain_stack_bb=Decimal("94.5"),
            hero_current_bet_bb=Decimal(0),
            villain_current_bet_bb=Decimal(0),
            amount_to_call_bb=Decimal(0),
            legal_actions=("CHECK", "BET"),
            street_root_confirmed=True,
            public_hand=history,
        )

        self.assertFalse(undeclared.full_six_max_ready)
        self.assertIn(
            "solution concept",
            " ".join(undeclared.full_six_max_gaps()),
        )
        self.assertIn(
            "solution concept",
            " ".join(undeclared.exactness_gaps_for_state(state)),
        )

    def test_state_coverage_is_derived_from_replayed_public_hand(self):
        capabilities = self._complete_capabilities()
        history = heads_up_to_turn_history()
        state = LiveDecisionState(
            hand_id=history.hand_id,
            street="TURN",
            board=("2c", "7d", "Jh", "As"),
            hero_combo=("Qc", "Qd"),
            hero_position="BB",
            villain_position="BTN",
            hero_is_oop=True,
            active_villains=1,
            pot_bb=Decimal("11.5"),
            hero_stack_bb=Decimal("94.5"),
            villain_stack_bb=Decimal("94.5"),
            hero_current_bet_bb=Decimal(0),
            villain_current_bet_bb=Decimal(0),
            amount_to_call_bb=Decimal(0),
            legal_actions=("CHECK", "BET"),
            street_root_confirmed=True,
            public_hand=history,
        )

        self.assertEqual((), capabilities.support_gaps_for_state(state))
        self.assertEqual((), capabilities.exactness_gaps_for_state(state))

        no_bunching = replace(capabilities, folded_card_bunching=False)
        self.assertIn(
            "folded-card bunching",
            " ".join(no_bunching.exactness_gaps_for_state(state)),
        )

        hu_only = replace(
            capabilities,
            postflop_mode="HU_SUBGAME",
            max_postflop_players=2,
        )
        multiway_without_transcript = replace(
            state,
            public_hand=None,
            active_villains=2,
        )
        self.assertIn(
            "at most 2",
            " ".join(
                hu_only.support_gaps_for_state(multiway_without_transcript)
            ),
        )

    def test_hu_river_still_requires_the_peak_multiway_postflop_capability(self):
        original = heads_up_to_turn_history()
        four_way_then_hu = replace(
            original,
            hand_id="four-way-flop-hu-turn",
            events=(
                HandEvent(0, "FOLD", "PREFLOP", actor_seat=0),
                HandEvent(1, "FOLD", "PREFLOP", actor_seat=1),
                HandEvent(
                    2,
                    "CALL",
                    "PREFLOP",
                    actor_seat=2,
                    amount_to_bb=Decimal("1"),
                ),
                HandEvent(
                    3,
                    "CALL",
                    "PREFLOP",
                    actor_seat=3,
                    amount_to_bb=Decimal("1"),
                ),
                HandEvent(
                    4,
                    "CALL",
                    "PREFLOP",
                    actor_seat=4,
                    amount_to_bb=Decimal("1"),
                ),
                HandEvent(5, "CHECK", "PREFLOP", actor_seat=5),
                HandEvent(
                    6,
                    "DEAL_FLOP",
                    "FLOP",
                    cards=("Ac", "Kd", "7s"),
                ),
                HandEvent(7, "CHECK", "FLOP", actor_seat=4),
                HandEvent(8, "CHECK", "FLOP", actor_seat=5),
                HandEvent(
                    9,
                    "BET_TO",
                    "FLOP",
                    actor_seat=2,
                    amount_to_bb=Decimal("1"),
                ),
                HandEvent(
                    10,
                    "CALL",
                    "FLOP",
                    actor_seat=3,
                    amount_to_bb=Decimal("1"),
                ),
                HandEvent(11, "FOLD", "FLOP", actor_seat=4),
                HandEvent(12, "FOLD", "FLOP", actor_seat=5),
                HandEvent(13, "DEAL_TURN", "TURN", cards=("2c",)),
            ),
        )
        replayed = four_way_then_hu.replay()
        self.assertEqual(2, len(replayed.live_seats))

        state = LiveDecisionState(
            hand_id=four_way_then_hu.hand_id,
            street="TURN",
            board=replayed.board,
            hero_combo=("Qc", "Qd"),
            hero_position="CO",
            villain_position="BTN",
            hero_is_oop=False,
            active_villains=1,
            pot_bb=replayed.pot_bb,
            hero_stack_bb=replayed.stack_map[2],
            villain_stack_bb=replayed.stack_map[3],
            hero_current_bet_bb=Decimal(0),
            villain_current_bet_bb=Decimal(0),
            amount_to_call_bb=Decimal(0),
            legal_actions=replayed.legal_actions,
            street_root_confirmed=True,
            public_hand=four_way_then_hu,
        )
        hu_only = replace(
            self._complete_capabilities(),
            postflop_mode="HU_SUBGAME",
            max_postflop_players=2,
        )

        gaps = hu_only.support_gaps_for_state(state)

        self.assertIn("hand path reached 4", " ".join(gaps))
        self.assertIn("4-player path", " ".join(gaps))

    @staticmethod
    def _complete_capabilities() -> SolverCapabilities:
        return SolverCapabilities(
            backend_id="licensed-or-owned-six-max",
            backend_version="1.0",
            preflop_mode="SOLVED_TREE",
            postflop_mode="MULTIWAY_TREE",
            max_postflop_players=6,
            stateful_through_river=True,
            range_conditioning="ACTION_CONDITIONED_ALL_STREETS",
            folded_card_bunching=True,
            card_model="CARD_EXACT",
            action_model="CONTINUOUS_NO_LIMIT",
            game_profile_id="sixmax-complete",
            abstraction_id="card-exact-continuous",
            solution_concept="multiplayer Nash equilibrium",
            convergence_metric="declared abstraction NashConv",
            convergence_target=Decimal(0),
            source_license="private owned source",
        )

    def test_malformed_or_untyped_manifests_are_rejected(self):
        with self.assertRaises(SolverCapabilitiesError):
            SolverCapabilities(
                backend_id="bad id",
                backend_version="1",
                preflop_mode="NONE",
                postflop_mode="NONE",
                max_postflop_players=0,
                stateful_through_river=False,
                range_conditioning="NONE",
                folded_card_bunching=False,
                card_model="ABSTRACT_BUCKETS",
                action_model="FIXED_DISCRETE_TREE",
                game_profile_id="invalid",
                abstraction_id="invalid",
                solution_concept="not configured",
                convergence_metric="none",
                convergence_target=Decimal(0),
                source_license="test",
            )

        class BadRouter:
            capabilities = {"full_six_max_ready": True}

        with self.assertRaises(SolverCapabilitiesError):
            capabilities_for_router(BadRouter())

    def test_manifest_json_rejects_duplicate_keys(self):
        wire = json.dumps(self._complete_capabilities().to_wire())
        duplicate = '{"backend_id":"shadow",' + wire[1:]

        with self.assertRaisesRegex(
            SolverCapabilitiesError,
            "duplicate JSON key 'backend_id'",
        ):
            parse_capabilities_json(duplicate)


if __name__ == "__main__":
    unittest.main()
