from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from gto_remote.client import RemoteGTOClient, RemoteGTOClientConfig
from gto_remote.capabilities import SolverCapabilities
from gto_remote.multiway_outcome import (
    MultiwayPolicyAction,
    MultiwaySolveOutcome,
    MultiwaySolveProof,
    outcome_from_wire as multiway_outcome_from_wire,
)
from gto_remote.multiway_protocol import (
    decision_fingerprint as multiway_decision_fingerprint,
)
from gto_remote.protocol import (
    RemoteProtocolError,
    build_evaluate_request,
    decision_fingerprint,
    decision_state_from_wire,
    decision_state_to_wire,
    encode_json,
    outcome_from_wire,
    outcome_to_wire,
    parse_evaluate_request,
)
from gto_remote.server import (
    BusyError,
    EvaluationService,
    IdempotencyConflictError,
    ServerConfig,
    UnsupportedSchemaError,
    create_server,
)
from live_gto import LiveDecisionState, LiveGTOOutcome, LiveGTOStatus
from preflop_observation import ObservationProvenance, ObservedPreflopState
from test_gto_multiway_protocol import four_way_state


POSITIONS = ("UTG", "HJ", "CO", "BTN", "SB", "BB")


def sample_state(*, hand_id: str = "hand-1") -> LiveDecisionState:
    contributions = tuple(
        (position, Decimal("2") if position == "BTN" else Decimal(0))
        for position in POSITIONS
    )
    initial_stacks = tuple((position, Decimal("100")) for position in POSITIONS)
    observation = ObservedPreflopState(
        actor=None,
        contributions=contributions,
        folded=frozenset({"UTG", "HJ", "CO", "SB"}),
        initial_stacks=initial_stacks,
        terminal=True,
        provenance=ObservationProvenance(
            source="terminal_preflop_history",
            preflop_index=2,
            flop_index=3,
            hand_id=hand_id,
        ),
    )
    return LiveDecisionState(
        hand_id=hand_id,
        street="FLOP",
        board=("2c", "7d", "Jh"),
        hero_combo=("As", "Kd"),
        hero_position="BTN",
        villain_position="BB",
        hero_is_oop=False,
        active_villains=1,
        pot_bb=Decimal("4.5"),
        hero_stack_bb=Decimal("98"),
        villain_stack_bb=Decimal("98"),
        hero_current_bet_bb=Decimal(0),
        villain_current_bet_bb=Decimal(0),
        amount_to_call_bb=Decimal(0),
        legal_actions=("CHECK", "BET"),
        street_root_confirmed=True,
        action_history=("CHECK",),
        observed_bet_to_bb=Decimal(0),
        preflop_observation=observation,
    )


class ProtocolTests(unittest.TestCase):
    def test_state_round_trip_includes_preflop_observation(self) -> None:
        original = sample_state()
        wire = decision_state_to_wire(original)
        restored = decision_state_from_wire(wire)

        self.assertEqual(restored, original)
        self.assertEqual(wire["pot_bb"], "4.5")
        self.assertEqual(
            wire["preflop_observation"]["initial_stacks"]["BB"],
            "100",
        )

    def test_request_and_fingerprint_are_stable(self) -> None:
        state = sample_state()
        request = build_evaluate_request("request-1", state)
        request_id, restored, fingerprint = parse_evaluate_request(
            encode_json(request)
        )

        self.assertEqual(request_id, "request-1")
        self.assertEqual(restored, state)
        self.assertEqual(fingerprint, decision_fingerprint(state))
        self.assertRegex(fingerprint, r"^[0-9a-f]{64}$")

    def test_duplicate_keys_and_float_chip_amounts_are_rejected(self) -> None:
        with self.assertRaisesRegex(RemoteProtocolError, "duplicate JSON key"):
            parse_evaluate_request(
                b'{"schema_version":2,"schema_version":2,'
                b'"request_id":"r","state":{}}'
            )

        request = build_evaluate_request("request-1", sample_state())
        request["state"]["pot_bb"] = 4.5
        with self.assertRaisesRegex(RemoteProtocolError, "decimal JSON string"):
            parse_evaluate_request(encode_json(request))

    def test_outcome_round_trip_is_minimal_and_bound_to_request(self) -> None:
        state = sample_state()
        fingerprint = decision_fingerprint(state)
        outcome = LiveGTOOutcome(
            LiveGTOStatus.SOLVED,
            "",
            0.25,
            analysis="BET 50%",
            source="verified cache",
            model="test-solver",
            cache_hit=True,
            approximate=False,
            spec_key="abc123",
        )
        wire = outcome_to_wire("request-1", fingerprint, outcome)
        restored = outcome_from_wire(
            encode_json(wire),
            expected_request_id="request-1",
            expected_fingerprint=fingerprint,
        )

        self.assertEqual(restored.status, LiveGTOStatus.SOLVED)
        self.assertEqual(restored.analysis, "BET 50%")
        self.assertEqual(restored.effective_spec_key, "abc123")
        self.assertNotIn("result", wire["outcome"])
        self.assertNotIn("spec", wire["outcome"])


class _Router:
    def __init__(self, *, block: bool = False) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = block
        self.calls = 0
        self.config = SimpleNamespace(
            engine_path=Path(sys.executable),
            enabled=True,
            owned_simulator_acknowledged=True,
        )

    def evaluate(self, state: LiveDecisionState) -> LiveGTOOutcome:
        self.calls += 1
        self.entered.set()
        if self.block:
            self.release.wait(timeout=2)
        return LiveGTOOutcome(
            LiveGTOStatus.SOLVED,
            "",
            0.01,
            analysis=f"CHECK {state.hand_id}",
            source="test",
            model="test",
            spec_key="spec-key",
        )


class EvaluationServiceTests(unittest.TestCase):
    def test_server_rejects_unsafe_bearer_tokens(self) -> None:
        for token in ("short", "x" * 31, "x" * 31 + "\n"):
            with self.subTest(token=repr(token)):
                with self.assertRaises(ValueError):
                    ServerConfig(bearer_token=token)

    def test_replays_same_request_and_rejects_conflicting_reuse(self) -> None:
        router = _Router()
        service = EvaluationService(router, cache_entries=2)
        state = sample_state()
        fingerprint = decision_fingerprint(state)

        first, first_cached = service.evaluate(
            "request-1",
            fingerprint,
            state,
        )
        second, second_cached = service.evaluate(
            "request-1",
            fingerprint,
            state,
        )

        self.assertFalse(first_cached)
        self.assertTrue(second_cached)
        self.assertEqual(first, second)
        self.assertEqual(router.calls, 1)
        with self.assertRaises(IdempotencyConflictError):
            service.evaluate(
                "request-1",
                decision_fingerprint(sample_state(hand_id="different")),
                sample_state(hand_id="different"),
            )

    def test_default_native_router_does_not_advertise_or_dispatch_v3(self) -> None:
        router = _Router()
        service = EvaluationService(router)
        state = four_way_state()

        self.assertEqual((2,), service.supported_schema_versions)
        with self.assertRaisesRegex(UnsupportedSchemaError, "schema 3"):
            service.evaluate(
                "request-v3",
                multiway_decision_fingerprint(state),
                state,
            )
        self.assertEqual(0, router.calls)

    def test_second_solve_fails_fast_while_slot_is_busy(self) -> None:
        router = _Router(block=True)
        service = EvaluationService(router)
        first_state = sample_state()
        result: list[bytes] = []

        worker = threading.Thread(
            target=lambda: result.append(
                service.evaluate(
                    "request-1",
                    decision_fingerprint(first_state),
                    first_state,
                )[0]
            )
        )
        worker.start()
        self.assertTrue(router.entered.wait(timeout=1))
        second_state = sample_state(hand_id="hand-2")
        with self.assertRaises(BusyError):
            service.evaluate(
                "request-2",
                decision_fingerprint(second_state),
                second_state,
            )
        router.release.set()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(result), 1)

    def test_service_preserves_a_structured_four_way_v3_policy(self) -> None:
        capabilities = SolverCapabilities(
            backend_id="service-multiway",
            backend_version="1",
            preflop_mode="SOLVED_TREE",
            postflop_mode="MULTIWAY_TREE",
            max_postflop_players=6,
            stateful_through_river=True,
            range_conditioning="ACTION_CONDITIONED_ALL_STREETS",
            folded_card_bunching=True,
            card_model="CARD_EXACT",
            action_model="DYNAMIC_DISCRETE_TREE",
            game_profile_id="sixmax-test",
            abstraction_id="card-exact",
            solution_concept="multiplayer Nash approximation",
            convergence_metric="NashConv BB",
            convergence_target=Decimal("0.1"),
            source_license="owned test",
        )

        class MultiwayRouter:
            supported_schema_versions = (2, 3)

            def __init__(self):
                self.capabilities = capabilities
                self.calls = 0

            def evaluate(self, state):
                self.calls += 1
                return MultiwaySolveOutcome(
                    status=LiveGTOStatus.SOLVED,
                    reason="",
                    latency_seconds=Decimal("0.1"),
                    cache_hit=False,
                    policy=(
                        MultiwayPolicyAction(
                            "CHECK",
                            None,
                            Decimal("1"),
                        ),
                    ),
                    proof=MultiwaySolveProof(
                        backend_id=capabilities.backend_id,
                        backend_version=capabilities.backend_version,
                        capability_fingerprint=(
                            capabilities.manifest_fingerprint
                        ),
                        game_profile_id="sixmax-test",
                        abstraction_id="card-exact",
                        solution_concept="multiplayer Nash approximation",
                        metric_name="NashConv BB",
                        metric_value=Decimal("0.05"),
                        target_value=Decimal("0.1"),
                        iterations=100,
                        converged=True,
                        approximate=True,
                    ),
                )

        router = MultiwayRouter()
        service = EvaluationService(router)
        state = four_way_state()
        fingerprint = multiway_decision_fingerprint(state)

        body, cached = service.evaluate("request-v3", fingerprint, state)
        restored = multiway_outcome_from_wire(
            body,
            expected_request_id="request-v3",
            expected_fingerprint=fingerprint,
            expected_state=state,
            expected_backend_id=capabilities.backend_id,
            expected_backend_version=capabilities.backend_version,
            expected_capability_fingerprint=capabilities.manifest_fingerprint,
            expected_game_profile_id=capabilities.game_profile_id,
            expected_abstraction_id=capabilities.abstraction_id,
            expected_solution_concept=capabilities.solution_concept,
            expected_metric_name=capabilities.convergence_metric,
            expected_target_value=capabilities.convergence_target,
        )

        self.assertFalse(cached)
        self.assertEqual(LiveGTOStatus.SOLVED, restored.status)
        self.assertEqual("CHECK", restored.policy[0].kind)
        self.assertEqual(1, router.calls)


class HTTPServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.token = "t" * 32
        self.router = _Router()
        try:
            self.server = create_server(
                ServerConfig(
                    bearer_token=self.token,
                    bind_host="127.0.0.1",
                    port=0,
                ),
                self.router,
            )
        except PermissionError:
            self.skipTest("the test sandbox does not permit loopback sockets")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _post(self, request_id: str, state: LiveDecisionState, *, token: str):
        body = encode_json(build_evaluate_request(request_id, state))
        request = Request(
            self.base_url + "/v1/evaluate",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Idempotency-Key": request_id,
                "X-Request-ID": request_id,
            },
        )
        return urlopen(request, timeout=2)

    def test_requires_auth_and_replays_completed_request(self) -> None:
        with self.assertRaises(HTTPError) as caught:
            self._post("request-1", sample_state(), token="wrong")
        self.assertEqual(caught.exception.code, 401)

        with self._post("request-1", sample_state(), token=self.token) as first:
            first_body = first.read()
            self.assertEqual(first.headers["Idempotency-Replayed"], "false")
        with self._post("request-1", sample_state(), token=self.token) as second:
            second_body = second.read()
            self.assertEqual(second.headers["Idempotency-Replayed"], "true")

        self.assertEqual(first_body, second_body)
        self.assertEqual(self.router.calls, 1)

    def test_health_ready_checks_router_and_engine(self) -> None:
        with urlopen(self.base_url + "/health/ready", timeout=2) as response:
            body = json.loads(response.read())
        self.assertEqual(body["status"], "ready")
        self.assertTrue(body["engine_available"])

    def test_remote_client_and_server_round_trip(self) -> None:
        client = RemoteGTOClient(
            RemoteGTOClientConfig(
                endpoint=self.base_url + "/v1/evaluate",
                bearer_token=self.token,
                timeout_seconds=2,
                allow_insecure_http=True,
            )
        )

        outcome = client.request(sample_state(), request_id="request-client-1")

        self.assertEqual(outcome.status, LiveGTOStatus.SOLVED)
        self.assertEqual(outcome.analysis, "CHECK hand-1")
        self.assertEqual(outcome.source, "test")
        self.assertEqual(self.router.calls, 1)


if __name__ == "__main__":
    unittest.main()
