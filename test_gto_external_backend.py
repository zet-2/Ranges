from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import gto_remote.external_backend as external_backend_module
from gto_remote.capabilities import SolverCapabilities
from gto_remote.external_backend import (
    ExternalBackendConfig,
    ExternalBackendConfigurationError,
    ExternalSolverBackend,
)
from gto_remote.multiway_outcome import MultiwaySolveOutcome
from gto_remote.multiway_outcome import (
    outcome_from_wire as multiway_outcome_from_wire,
)
from gto_remote.multiway_protocol import (
    decision_fingerprint as multiway_decision_fingerprint,
)
from gto_remote.server import EvaluationService
from live_gto import LiveDecisionState, LiveGTOStatus
from test_gto_hand_history import heads_up_to_turn_history
from test_gto_multiway_protocol import four_way_state


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
        action_model="CONTINUOUS_NO_LIMIT",
        game_profile_id="sixmax-test-v1",
        abstraction_id="card-exact-test",
        solution_concept="multiplayer Nash approximation",
        convergence_metric="NashConv BB",
        convergence_target=Decimal("0.1"),
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
        # Production external solve servers are Linux. Tests run on macOS too,
        # where process groups are explicitly only a development safeguard.
        "allow_best_effort_process_cleanup": not sys.platform.startswith(
            "linux"
        ),
    }
    values.update(changes)
    return ExternalBackendConfig(**values)


def multiway_adapter_code() -> str:
    declared = complete_capabilities()
    return f"""
import hashlib
import json
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
response = {{
    "schema_version": 3,
    "request_id": request["request_id"],
    "decision_fingerprint": fingerprint,
    "outcome": {{
        "status": "SOLVED",
        "reason": "",
        "latency_seconds": "0",
        "cache_hit": False,
        "policy": [
            {{
                "kind": "CHECK",
                "amount_to_bb": None,
                "frequency": "0.7",
                "ev_bb": "1.2",
            }},
            {{
                "kind": "BET_TO",
                "amount_to_bb": "2",
                "frequency": "0.3",
                "ev_bb": "1.1",
            }},
        ],
        "proof": {{
            "backend_id": {declared.backend_id!r},
            "backend_version": {declared.backend_version!r},
            "capability_fingerprint": {declared.manifest_fingerprint!r},
            "game_profile_id": "sixmax-test-v1",
            "abstraction_id": "card-exact-test",
            "solution_concept": "multiplayer Nash approximation",
            "metric_name": "NashConv BB",
            "metric_value": "0.08",
            "target_value": "0.1",
            "iterations": 1000,
            "converged": True,
            "approximate": True,
        }},
    }},
}}
json.dump(response, sys.stdout, separators=(",", ":"))
"""


class ExternalBackendTests(unittest.TestCase):
    def test_linux_adopted_child_cleanup_repeats_for_exposed_generations(self):
        child_sets = (
            {10, 20},
            {10, 21},
            {10, 21},
            {10},
            {10},
            {10},
        )
        with (
            patch.object(
                external_backend_module,
                "_linux_direct_children",
                side_effect=child_sets,
            ),
            patch.object(
                external_backend_module.os,
                "kill",
            ) as kill,
            patch.object(
                external_backend_module.os,
                "waitpid",
                return_value=(0, 0),
            ),
            patch.object(external_backend_module.time, "sleep"),
        ):
            external_backend_module._kill_and_reap_adopted_linux_children(
                frozenset({10}),
            )

        killed_pids = {
            call.args[0]
            for call in kill.call_args_list
        }
        self.assertEqual({20, 21}, killed_pids)

    def test_genuine_four_way_v3_request_returns_structured_policy(self):
        backend = ExternalSolverBackend(
            process_config(multiway_adapter_code()),
            complete_capabilities(),
        )
        outcome = backend.evaluate(four_way_state())

        self.assertIsInstance(outcome, MultiwaySolveOutcome)
        self.assertEqual(LiveGTOStatus.SOLVED, outcome.status)
        self.assertEqual(2, len(outcome.policy))
        self.assertEqual("CHECK", outcome.policy[0].kind)
        self.assertEqual("sixmax-test-v1", outcome.proof.game_profile_id)

        state = four_way_state()
        fingerprint = multiway_decision_fingerprint(state)
        body, cached = EvaluationService(backend).evaluate(
            "four-way-service",
            fingerprint,
            state,
        )
        restored = multiway_outcome_from_wire(
            body,
            expected_request_id="four-way-service",
            expected_fingerprint=fingerprint,
            expected_state=state,
            expected_backend_id=complete_capabilities().backend_id,
            expected_backend_version=complete_capabilities().backend_version,
            expected_capability_fingerprint=(
                complete_capabilities().manifest_fingerprint
            ),
            expected_game_profile_id=(
                complete_capabilities().game_profile_id
            ),
            expected_abstraction_id=(
                complete_capabilities().abstraction_id
            ),
            expected_solution_concept=(
                complete_capabilities().solution_concept
            ),
            expected_metric_name=(
                complete_capabilities().convergence_metric
            ),
            expected_target_value=(
                complete_capabilities().convergence_target
            ),
        )
        self.assertFalse(cached)
        self.assertEqual(outcome.policy, restored.policy)

    def test_declared_player_limit_is_enforced_without_a_transcript(self):
        hu_capabilities = replace(
            complete_capabilities(),
            preflop_mode="FIXED_BLUEPRINT",
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
        self.assertIn("schema v3", outcome.reason)

    def test_multiway_public_hand_cannot_downgrade_to_v2(self):
        multiway = four_way_state()
        replayed = multiway.replayed
        v2_projection = LiveDecisionState(
            hand_id=multiway.hand_id,
            street=replayed.street,
            board=replayed.board,
            hero_combo=multiway.hero_combo,
            hero_position=multiway.hero_position,
            villain_position="BB",
            hero_is_oop=True,
            active_villains=3,
            pot_bb=replayed.pot_bb,
            hero_stack_bb=replayed.stack_map[multiway.hero_seat],
            villain_stack_bb=replayed.stack_map[5],
            hero_current_bet_bb=(
                replayed.street_contribution_map[multiway.hero_seat]
            ),
            villain_current_bet_bb=replayed.street_contribution_map[5],
            amount_to_call_bb=replayed.amount_to_call_bb,
            legal_actions=("CHECK", "BET", "ALL-IN"),
            street_root_confirmed=True,
            public_hand=multiway.public_hand,
        )

        outcome = ExternalSolverBackend(
            process_config(),
            complete_capabilities(),
        ).evaluate(v2_projection)

        self.assertEqual(LiveGTOStatus.UNSUPPORTED, outcome.status)
        self.assertIn("schema v3", outcome.reason)

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
        self.assertEqual("external owned-six-max", solved.source)
        self.assertEqual("owned-six-max@1", solved.model)

    def test_an_uncovered_backend_cannot_label_its_answer_exact(self):
        no_bunching = replace(
            complete_capabilities(),
            folded_card_bunching=False,
        )
        exact_claim = ExternalSolverBackend(
            process_config(),
            no_bunching,
        ).evaluate(bound_state())

        self.assertEqual(LiveGTOStatus.FAILED, exact_claim.status)
        self.assertIn("labelled an uncovered game exact", exact_claim.reason)
        self.assertIn("folded-card bunching", exact_claim.reason)

        approximate_code = SUCCESS_ADAPTER.replace(
            '"approximate": False',
            '"approximate": True',
        )
        disclosed = ExternalSolverBackend(
            process_config(approximate_code),
            no_bunching,
        ).evaluate(bound_state())

        self.assertEqual(LiveGTOStatus.SOLVED, disclosed.status)
        self.assertTrue(disclosed.approximate)

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

    def test_output_limits_are_enforced_while_the_adapter_is_running(self):
        excessive_stdout = ExternalSolverBackend(
            process_config(
                "import sys,time; "
                "sys.stdout.write('x' * 2048); sys.stdout.flush(); "
                "time.sleep(2)",
                timeout_seconds=Decimal("1"),
                max_response_bytes=1024,
            ),
            complete_capabilities(),
        ).evaluate(bound_state())

        self.assertEqual(LiveGTOStatus.FAILED, excessive_stdout.status)
        self.assertIn("response exceeds", excessive_stdout.reason)
        self.assertNotIn("deadline", excessive_stdout.reason)

    @unittest.skipUnless(
        sys.platform.startswith("linux"),
        "detached-process containment requires Linux subreaper semantics",
    )
    def test_adapter_descendants_are_killed_when_the_leader_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "orphan-survived"
            child_code = (
                "import os,pathlib,time; os.setsid(); time.sleep(0.25); "
                f"pathlib.Path({str(marker)!r}).write_text('alive')"
            )
            adapter_code = (
                "import subprocess,sys; "
                f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
                "raise SystemExit(7)"
            )
            outcome = ExternalSolverBackend(
                process_config(adapter_code),
                complete_capabilities(),
            ).evaluate(bound_state())
            time.sleep(0.35)

            self.assertEqual(LiveGTOStatus.FAILED, outcome.status)
            self.assertFalse(marker.exists())

    @unittest.skipIf(
        sys.platform.startswith("linux"),
        "Linux uses strong subreaper containment",
    )
    def test_non_linux_cleanup_requires_explicit_unsafe_acknowledgement(self):
        with self.assertRaisesRegex(
            ExternalBackendConfigurationError,
            "strong detached-process cleanup requires Linux",
        ):
            ExternalBackendConfig(
                command=(str(Path(sys.executable).resolve()), "-c", "pass"),
                timeout_seconds=Decimal("1"),
                max_request_bytes=1024,
                max_response_bytes=1024,
            )

    def test_configuration_is_no_shell_absolute_and_bounded(self):
        invalid = (
            {"command": ("python3", "-c", "pass")},
            {"command": (str(Path(sys.executable).resolve()), "", "pass")},
            {"timeout_seconds": "NaN"},
            {"max_response_bytes": 10},
            {"environment": (("BAD-NAME", "value"),)},
            {"allow_best_effort_process_cleanup": "yes"},
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
