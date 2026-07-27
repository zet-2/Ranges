from __future__ import annotations

import unittest

from gto_remote.capabilities import (
    NATIVE_ROUTER_CAPABILITIES,
    SolverCapabilities,
    SolverCapabilitiesError,
    capabilities_for_router,
)
from gto_remote.server import EvaluationService


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
            action_model="DYNAMIC_DISCRETE_TREE",
            convergence_metric="declared abstraction NashConv",
            source_license="private owned source",
        )

        self.assertTrue(complete.full_six_max_ready)
        self.assertEqual((), complete.full_six_max_gaps())
        self.assertTrue(complete.to_wire()["full_six_max_ready"])

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
                convergence_metric="none",
                source_license="test",
            )

        class BadRouter:
            capabilities = {"full_six_max_ready": True}

        with self.assertRaises(SolverCapabilitiesError):
            capabilities_for_router(BadRouter())


if __name__ == "__main__":
    unittest.main()
