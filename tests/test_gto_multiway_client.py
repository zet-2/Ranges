"""Tests for the authenticated multiway-v3 client."""

from __future__ import annotations

from decimal import Decimal
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from gto_remote.capabilities import SolverCapabilities
from gto_remote.client import RemoteGTOClientConfig, RemoteGTORequestError
from gto_remote.multiway_client import RemoteMultiwayClient
from gto_remote.multiway_outcome import (
    MultiwayPolicyAction,
    MultiwaySolveOutcome,
    MultiwaySolveProof,
    encode_json,
    outcome_to_wire,
)
from gto_remote.multiway_protocol import (
    decision_fingerprint,
    parse_evaluate_request,
)
from live_gto import LiveGTOStatus
from tests.test_gto_multiway_protocol import four_way_state
from tests.test_gto_remote_client import AUTH_TOKEN, FakeOpener, FakeResponse, REQUEST_ID


def capabilities() -> SolverCapabilities:
    return SolverCapabilities(
        backend_id="remote-multiway",
        backend_version="2",
        preflop_mode="SOLVED_TREE",
        postflop_mode="MULTIWAY_TREE",
        max_postflop_players=6,
        stateful_through_river=True,
        range_conditioning="ACTION_CONDITIONED_ALL_STREETS",
        folded_card_bunching=True,
        card_model="CARD_EXACT",
        action_model="DYNAMIC_DISCRETE_TREE",
        game_profile_id="sixmax-v1",
        abstraction_id="test-exact",
        solution_concept="multiplayer Nash approximation",
        convergence_metric="NashConv BB",
        convergence_target=Decimal("0.1"),
        source_license="owned test",
    )


def response_body() -> bytes:
    declared = capabilities()
    state = four_way_state()
    outcome = MultiwaySolveOutcome(
        status=LiveGTOStatus.SOLVED,
        reason="",
        latency_seconds=Decimal("0.2"),
        cache_hit=True,
        policy=(
            MultiwayPolicyAction(
                "CHECK",
                None,
                Decimal("1"),
            ),
        ),
        proof=MultiwaySolveProof(
            backend_id=declared.backend_id,
            backend_version=declared.backend_version,
            capability_fingerprint=declared.manifest_fingerprint,
            game_profile_id="sixmax-v1",
            abstraction_id="test-exact",
            solution_concept="multiplayer Nash approximation",
            metric_name="NashConv BB",
            metric_value=Decimal("0.05"),
            target_value=Decimal("0.1"),
            iterations=1000,
            converged=True,
            approximate=True,
        ),
    )
    return encode_json(
        outcome_to_wire(
            REQUEST_ID,
            decision_fingerprint(state),
            outcome,
        )
    )


def client(opener) -> RemoteMultiwayClient:
    return RemoteMultiwayClient(
        RemoteGTOClientConfig(
            endpoint="https://solver.example/v1/evaluate",
            bearer_token=AUTH_TOKEN,
            timeout_seconds="4",
        ),
        capabilities(),
        opener=opener,
    )


class RemoteMultiwayClientTests(unittest.TestCase):
    def test_posts_v3_and_accepts_only_manifest_bound_structured_policy(self):
        response = FakeResponse(response_body())
        opener = FakeOpener(response=response)

        result = client(opener).evaluate(
            four_way_state(),
            request_id=REQUEST_ID,
        )

        self.assertEqual(LiveGTOStatus.SOLVED, result.status)
        self.assertIn("**Action:** Check", result.analysis)
        self.assertEqual("remote-multiway@2", result.model)
        request, timeout = opener.calls[0]
        self.assertEqual(4.0, timeout)
        request_id, state, fingerprint = parse_evaluate_request(request.data)
        self.assertEqual(REQUEST_ID, request_id)
        self.assertEqual(four_way_state(), state)
        self.assertEqual(decision_fingerprint(state), fingerprint)
        self.assertTrue(response.closed)

    def test_rejects_non_multiway_state_before_network(self):
        opener = FakeOpener(response=FakeResponse(response_body()))
        from tests.test_gto_remote import sample_state

        with self.assertRaises(RemoteGTORequestError):
            client(opener).request(sample_state(), request_id=REQUEST_ID)
        self.assertEqual([], opener.calls)

    def test_rejects_unsupported_state_before_network(self):
        opener = FakeOpener(response=FakeResponse(response_body()))
        limited = replace(
            capabilities(),
            postflop_mode="HU_SUBGAME",
            max_postflop_players=2,
        )
        remote = RemoteMultiwayClient(
            client(opener).config,
            limited,
            opener=opener,
        )

        with self.assertRaisesRegex(
            RemoteGTORequestError,
            "hand path reached 4",
        ):
            remote.request(four_way_state(), request_id=REQUEST_ID)
        self.assertEqual([], opener.calls)

    def test_abstract_manifest_cannot_accept_an_exact_claim(self):
        body = json.loads(response_body())
        body["outcome"]["proof"]["metric_value"] = "0"
        body["outcome"]["proof"]["approximate"] = False
        opener = FakeOpener(
            response=FakeResponse(encode_json(body))
        )

        result = client(opener).evaluate(
            four_way_state(),
            request_id=REQUEST_ID,
        )

        self.assertEqual(LiveGTOStatus.FAILED, result.status)
        self.assertIn("labelled an uncovered game exact", result.reason)
        self.assertIn("finite discrete abstraction", result.reason)

    def test_from_env_pins_a_complete_local_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capabilities.json"
            path.write_text(
                json.dumps(capabilities().to_wire()),
                encoding="utf-8",
            )
            remote = RemoteMultiwayClient.from_env(
                environment={
                    "GTO_REMOTE_ENDPOINT": (
                        "https://solver.example/v1/evaluate"
                    ),
                    "GTO_REMOTE_ENABLED": "1",
                    "GTO_REMOTE_AUTH_TOKEN": AUTH_TOKEN,
                    "GTO_REMOTE_CAPABILITIES_PATH": str(path),
                },
                opener=FakeOpener(response=FakeResponse(response_body())),
            )

        self.assertEqual(
            capabilities().manifest_fingerprint,
            remote.capabilities.manifest_fingerprint,
        )

    def test_disabled_from_env_does_not_require_a_manifest(self):
        remote = RemoteMultiwayClient.from_env(
            environment={
                "GTO_REMOTE_ENABLED": "0",
            },
            opener=FakeOpener(),
        )

        self.assertFalse(remote.config.enabled)
        self.assertFalse(remote.capabilities.full_six_max_ready)


if __name__ == "__main__":
    unittest.main()
