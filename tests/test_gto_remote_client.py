"""Automated tests for the authenticated, bounded remote GTO transport."""

from __future__ import annotations

from decimal import Decimal
from email.message import Message
from io import BytesIO
import json
import socket
import unittest
from urllib.error import HTTPError, URLError

from gto_remote.client import (
    RemoteGTOClient,
    RemoteGTOClientConfig,
    RemoteGTOConfigurationError,
    RemoteGTOConnectionError,
    RemoteGTOHTTPError,
    RemoteGTORequestError,
    RemoteGTORequestTooLargeError,
    RemoteGTOResponseProtocolError,
    RemoteGTOResponseTooLargeError,
    RemoteGTOTimeoutError,
)
import poker_assistant as app
from gto_remote.protocol import (
    decision_fingerprint,
    encode_json,
    outcome_to_wire,
    parse_evaluate_request,
)
from live_gto import (
    LiveDecisionState,
    LiveGTOConfig,
    LiveGTOConfigurationError,
    LiveGTOOutcome,
    LiveGTOStatus,
    LiveGTORouter,
)
from preflop_observation import ObservationProvenance, ObservedPreflopState


REQUEST_ID = "solve-17"
POSITIONS = ("UTG", "HJ", "CO", "BTN", "SB", "BB")
AUTH_TOKEN = "t" * 32


def make_state() -> LiveDecisionState:
    observation = ObservedPreflopState(
        actor=None,
        contributions=tuple(
            (position, Decimal("2.5") if position in {"BTN", "BB"} else Decimal(0))
            for position in POSITIONS
        ),
        folded=frozenset({"UTG", "HJ", "CO", "SB"}),
        initial_stacks=tuple((position, Decimal(100)) for position in POSITIONS),
        terminal=True,
        provenance=ObservationProvenance(
            source="preflop_to_flop_transition",
            preflop_index=4,
            flop_index=5,
            hand_id="hand-17",
        ),
    )
    return LiveDecisionState(
        hand_id="hand-17",
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
        preflop_observation=observation,
    )


def make_response(request_id: str = REQUEST_ID) -> bytes:
    outcome = LiveGTOOutcome(
        status=LiveGTOStatus.SOLVED,
        reason="",
        latency_seconds=0.125,
        analysis="Raise 25% / call 75%",
        source="remote GTO",
        model="postflop-solver",
        cache_hit=False,
        approximate=False,
        spec_key="a" * 64,
    )
    return encode_json(
        outcome_to_wire(
            request_id,
            decision_fingerprint(make_state()),
            outcome,
        )
    )


class FakeResponse:
    def __init__(self, body: bytes, *, status=200, headers=None):
        self.status = status
        self.headers = Message()
        self.headers["Content-Type"] = "application/json; charset=utf-8"
        self.headers["X-Request-ID"] = REQUEST_ID
        for name, value in (headers or {}).items():
            if name in self.headers:
                self.headers.replace_header(name, value)
            else:
                self.headers[name] = value
        self.stream = BytesIO(body)
        self.closed = False

    def read(self, amount=-1):
        return self.stream.read(amount)

    def close(self):
        self.closed = True


class FakeOpener:
    def __init__(self, *, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        if self.error is not None:
            raise self.error
        return self.response


def make_client(opener, **changes):
    values = {
        "endpoint": "https://solver.example/v1/evaluate",
        "bearer_token": AUTH_TOKEN,
        "timeout_seconds": "7.5",
    }
    values.update(changes)
    return RemoteGTOClient(RemoteGTOClientConfig(**values), opener=opener)


class RemoteGTOClientTests(unittest.TestCase):
    def test_posts_shared_protocol_with_bearer_and_request_identity(self):
        response = FakeResponse(make_response())
        opener = FakeOpener(response=response)
        result = make_client(opener).evaluate(make_state(), request_id=REQUEST_ID)

        self.assertEqual(LiveGTOStatus.SOLVED, result.status)
        self.assertEqual("a" * 64, result.spec_key)
        request, timeout = opener.calls[0]
        self.assertEqual("POST", request.get_method())
        self.assertEqual(7.5, timeout)
        self.assertEqual(
            f"Bearer {AUTH_TOKEN}",
            request.get_header("Authorization"),
        )
        self.assertEqual(REQUEST_ID, request.get_header("X-request-id"))
        self.assertEqual(REQUEST_ID, request.get_header("Idempotency-key"))
        parsed_id, parsed_state, fingerprint = parse_evaluate_request(request.data)
        self.assertEqual(REQUEST_ID, parsed_id)
        self.assertEqual(make_state(), parsed_state)
        self.assertEqual(decision_fingerprint(make_state()), fingerprint)
        self.assertTrue(response.closed)

    def test_from_env_and_config_repr_do_not_expose_token(self):
        opener = FakeOpener(response=FakeResponse(make_response()))
        client = RemoteGTOClient.from_env(
            environment={
                "GTO_REMOTE_ENDPOINT": "https://solver.example/v1/evaluate",
                "GTO_REMOTE_ENABLED": "1",
                "GTO_REMOTE_AUTH_TOKEN": AUTH_TOKEN,
                "GTO_REMOTE_TIMEOUT_SECONDS": "9",
                "GTO_REMOTE_MAX_REQUEST_BYTES": "4096",
                "GTO_REMOTE_MAX_RESPONSE_BYTES": "8192",
            },
            opener=opener,
        )
        self.assertNotIn(AUTH_TOKEN, repr(client.config))
        self.assertEqual(9.0, client._timeout_seconds)
        self.assertEqual(4096, client.config.max_request_bytes)

    def test_disabled_and_transport_failures_return_fail_closed_outcomes(self):
        opener = FakeOpener(error=AssertionError("network must not open"))
        disabled = RemoteGTOClient(
            RemoteGTOClientConfig(endpoint="", bearer_token="", enabled=False),
            opener=opener,
        )
        outcome = disabled.evaluate(make_state())
        self.assertEqual(LiveGTOStatus.DISABLED, outcome.status)
        self.assertEqual([], opener.calls)

        failed = make_client(
            FakeOpener(error=URLError("dns failed"))
        ).evaluate(make_state(), request_id=REQUEST_ID)
        self.assertEqual(LiveGTOStatus.FAILED, failed.status)
        self.assertIn("connection failed", failed.reason)
        self.assertNotIn(AUTH_TOKEN, failed.reason)

    def test_rejects_unsafe_configuration_state_and_request_id(self):
        invalid = (
            {"endpoint": "http://solver.example/v1/evaluate"},
            {
                "endpoint": "http://solver.example/v1/evaluate",
                "allow_insecure_http": True,
            },
            {"endpoint": "https://solver.example/wrong"},
            {"endpoint": "https://user@solver.example/v1/evaluate"},
            {"bearer_token": "bad token"},
            {"timeout_seconds": "NaN"},
            {"max_response_bytes": 0},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises(RemoteGTOConfigurationError):
                    make_client(FakeOpener(), **changes)
        loopback = make_client(
            FakeOpener(),
            endpoint="http://127.0.0.1:8787/v1/evaluate",
            allow_insecure_http=True,
        )
        self.assertEqual(
            "http://127.0.0.1:8787/v1/evaluate",
            loopback.config.endpoint,
        )
        with self.assertRaises(RemoteGTORequestError):
            make_client(FakeOpener()).request(object(), request_id=REQUEST_ID)
        with self.assertRaises(RemoteGTORequestError):
            make_client(FakeOpener()).request(make_state(), request_id="bad id")

    def test_enforces_request_and_response_limits_before_parsing(self):
        opener = FakeOpener()
        with self.assertRaises(RemoteGTORequestTooLargeError):
            make_client(opener, max_request_bytes=10).request(
                make_state(), request_id=REQUEST_ID
            )
        self.assertEqual([], opener.calls)

        response = FakeResponse(b"x" * 33)
        with self.assertRaises(RemoteGTOResponseTooLargeError):
            make_client(
                FakeOpener(response=response),
                max_response_bytes=32,
            ).request(make_state(), request_id=REQUEST_ID)
        self.assertTrue(response.closed)

        declared = FakeResponse(b"{}", headers={"Content-Length": "999"})
        with self.assertRaises(RemoteGTOResponseTooLargeError):
            make_client(
                FakeOpener(response=declared),
                max_response_bytes=64,
            ).request(make_state(), request_id=REQUEST_ID)
        self.assertEqual(0, declared.stream.tell())

    def test_maps_timeout_connection_and_http_failures(self):
        with self.assertRaises(RemoteGTOTimeoutError):
            make_client(
                FakeOpener(error=URLError(socket.timeout("late")))
            ).request(make_state(), request_id=REQUEST_ID)
        with self.assertRaises(RemoteGTOConnectionError):
            make_client(
                FakeOpener(error=URLError("dns failed"))
            ).request(make_state(), request_id=REQUEST_ID)

        headers = Message()
        headers["Content-Type"] = "application/json"
        error = HTTPError(
            "https://solver.example/v1/evaluate",
            503,
            "Unavailable",
            headers,
            BytesIO(b'{"error":"busy"}'),
        )
        with self.assertRaises(RemoteGTOHTTPError) as raised:
            make_client(FakeOpener(error=error)).request(
                make_state(), request_id=REQUEST_ID
            )
        self.assertEqual(503, raised.exception.status_code)
        self.assertNotIn(AUTH_TOKEN, str(raised.exception))

    def test_rejects_bad_http_or_protocol_identity(self):
        wrong_type = FakeResponse(
            make_response(),
            headers={"Content-Type": "text/plain"},
        )
        with self.assertRaises(RemoteGTOResponseProtocolError):
            make_client(FakeOpener(response=wrong_type)).request(
                make_state(), request_id=REQUEST_ID
            )

        wrong_header = FakeResponse(
            make_response(),
            headers={"X-Request-ID": "other"},
        )
        with self.assertRaises(RemoteGTOResponseProtocolError):
            make_client(FakeOpener(response=wrong_header)).request(
                make_state(), request_id=REQUEST_ID
            )

        wrong_body = json.loads(make_response())
        wrong_body["decision_fingerprint"] = "b" * 64
        with self.assertRaises(RemoteGTOResponseProtocolError):
            make_client(
                FakeOpener(response=FakeResponse(encode_json(wrong_body)))
            ).request(make_state(), request_id=REQUEST_ID)


class RouterSelectionTests(unittest.TestCase):
    def test_app_selects_local_or_remote_transport_explicitly(self):
        live_config = LiveGTOConfig(
            enabled=True,
            owned_simulator_acknowledged=True,
        )
        remote = app.create_live_gto_router(
            live_config,
            execution_mode="remote",
            environment={
                "GTO_REMOTE_ENABLED": "1",
                "GTO_REMOTE_ENDPOINT": "https://solver.example/v1/evaluate",
                "GTO_REMOTE_AUTH_TOKEN": AUTH_TOKEN,
            },
        )
        self.assertIsInstance(remote, RemoteGTOClient)
        self.assertIsInstance(
            app.create_live_gto_router(
                live_config,
                execution_mode="local",
                environment={},
            ),
            LiveGTORouter,
        )
        with self.assertRaises(LiveGTOConfigurationError):
            app.create_live_gto_router(
                live_config,
                execution_mode="automatic",
                environment={},
            )


if __name__ == "__main__":
    unittest.main()
