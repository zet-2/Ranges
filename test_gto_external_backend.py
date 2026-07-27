from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from gto_remote.capabilities import SolverCapabilities
from gto_remote.external_backend import (
    ExternalBackendConfig,
    ExternalBackendConfigurationError,
    ExternalSolverBackend,
)
from live_gto import LiveDecisionState, LiveGTOStatus
from test_gto_hand_history import heads_up_to_turn_history


SUCCESS_ADAPTER = r"""
import hashlib
import json
import os
import sys

request = json.load(sys.stdin)
state = request["state"]
canonical = json.dumps(
    state,
    ensure_ascii=True,
    allow_nan=False,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
fingerprint = hashlib.sha256(canonical).hexdigest()
secret_is_absent = "GTO_REMOTE_AUTH_TOKEN" not in os.environ
response = {
    "schema_version": request["schema_version"],
    "request_id": request["request_id"],
    "decision_fingerprint": fingerprint,
    "outcome": {
        "status": "SOLVED" if secret_is_absent else "FAILED",
        "reason": "" if secret_is_absent else "server secret leaked",
        "latency_seconds": 0,
        "analysis": "**Action:** Check\n**Size:** 0",
        "source": "owned external adapter",
        "model": "test-six-max",
        "cache_hit": False,
        "approximate": False,
        "spec_key": "a" * 64,
    },
}
json.dump(response, sys.stdout, separators=(",", ":"))
"""


def complete_capabilities() -> SolverCapabilities:
    return SolverCapabilities(
        backend_id="owned-six-max",
        backend_version="1",
        preflop_mode="SOLVED_TREE",
        postflop_mode="MULTIWAY_TREE",
        max_postflop_players=6,
        stateful_through_river=True,
        range_conditioning="ACTION_CONDITIONED_ALL_STREETS",
        folded_card_bunching=True,
        card_model="CARD_EXACT",
        action_model="DYNAMIC_DISCRETE_TREE",
        convergence_metric="declared abstraction NashConv",
        source_license="owned test adapter",
    )


def bound_state() -> LiveDecisionState:
    history = heads_up_to_turn_history()
    return LiveDecisionState(
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


def process_config(code: str = SUCCESS_ADAPTER, **changes) -> ExternalBackendConfig:
    values = {
        "command": (str(Path(sys.executable).resolve()), "-c", code),
        "timeout_seconds": Decimal("2"),
        "max_request_bytes": 1024 * 1024,
        "max_response_bytes": 1024 * 1024,
    }
    values.update(changes)
    return ExternalBackendConfig(**values)


class ExternalBackendTests(unittest.TestCase):
    def test_declared_player_limit_is_enforced_without_a_transcript(self):
        hu_capabilities = replace(
            complete_capabilities(),
            preflop_mode="NONE",
            postflop_mode="HU_SUBGAME",
            max_postflop_players=2,
            stateful_through_river=False,
            range_conditioning="NONE",
            folded_card_bunching=False,
        )
        state = replace(
            bound_state(),
            public_hand=None,
            active_villains=2,
        )
        outcome = ExternalSolverBackend(
            process_config(),
            hu_capabilities,
        ).evaluate(state)

        self.assertEqual(LiveGTOStatus.UNSUPPORTED, outcome.status)
        self.assertIn("at most 2", outcome.reason)

    def test_full_backend_requires_and_consumes_a_bound_public_hand(self):
        backend = ExternalSolverBackend(
            process_config(),
            complete_capabilities(),
        )
        missing = bound_state()
        missing = LiveDecisionState(
            **{
                field: getattr(missing, field)
                for field in missing.__dataclass_fields__
                if field != "public_hand"
            }
        )

        unsupported = backend.evaluate(missing)
        self.assertEqual(LiveGTOStatus.UNSUPPORTED, unsupported.status)
        self.assertIn("complete public_hand", unsupported.reason)

        with patch.dict(
            os.environ,
            {"GTO_REMOTE_AUTH_TOKEN": "must-not-reach-child"},
        ):
            solved = backend.evaluate(bound_state())
        self.assertEqual(LiveGTOStatus.SOLVED, solved.status)
        self.assertEqual("a" * 64, solved.spec_key)
        self.assertNotIn("leaked", solved.reason)

    def test_timeout_bad_identity_and_nonzero_exit_fail_closed(self):
        sleeping = ExternalSolverBackend(
            process_config(
                "import time; time.sleep(2)",
                timeout_seconds=Decimal("0.05"),
            ),
            complete_capabilities(),
        ).evaluate(bound_state())
        self.assertEqual(LiveGTOStatus.FAILED, sleeping.status)
        self.assertIn("deadline", sleeping.reason)

        bad_identity_code = SUCCESS_ADAPTER.replace(
            "fingerprint = hashlib.sha256(canonical).hexdigest()",
            'fingerprint = "b" * 64',
        )
        bad_identity = ExternalSolverBackend(
            process_config(bad_identity_code),
            complete_capabilities(),
        ).evaluate(bound_state())
        self.assertEqual(LiveGTOStatus.FAILED, bad_identity.status)
        self.assertIn("fingerprint", bad_identity.reason)

        nonzero = ExternalSolverBackend(
            process_config("raise SystemExit(7)"),
            complete_capabilities(),
        ).evaluate(bound_state())
        self.assertEqual(LiveGTOStatus.FAILED, nonzero.status)
        self.assertIn("status 7", nonzero.reason)

    def test_configuration_is_no_shell_absolute_and_bounded(self):
        invalid = (
            {"command": ("python3", "-c", "pass")},
            {"command": (str(Path(sys.executable).resolve()), "", "pass")},
            {"timeout_seconds": "NaN"},
            {"max_response_bytes": 10},
            {"environment": (("BAD-NAME", "value"),)},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises(ExternalBackendConfigurationError):
                    process_config(**changes)

    def test_capability_wire_manifest_round_trip(self):
        capabilities = complete_capabilities()
        self.assertEqual(
            capabilities,
            SolverCapabilities.from_wire(
                json.loads(json.dumps(capabilities.to_wire()))
            ),
        )
        contradictory = capabilities.to_wire()
        contradictory["full_six_max_ready"] = False
        with self.assertRaisesRegex(
            ValueError,
            "contradicts",
        ):
            SolverCapabilities.from_wire(contradictory)


if __name__ == "__main__":
    unittest.main()
