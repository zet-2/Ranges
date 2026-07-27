"""Strict subprocess adapter for an owned or properly licensed solver backend.

There is no public, documented MonkerSolver server API to embed directly.
Instead of coupling the OCR client to one proprietary executable, the solve
server can launch a local adapter that implements the same versioned JSON
request/response protocol as ``/v1/evaluate``.

The adapter is never invoked through a shell.  It receives no remote bearer
token and no inherited application environment unless values are explicitly
listed in ``GTO_EXTERNAL_ENV_JSON``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from types import SimpleNamespace
from typing import Mapping

from gto_hand_history import replay_public_hand
from live_gto import LiveDecisionState, LiveGTOOutcome, LiveGTOStatus

from .capabilities import SolverCapabilities, SolverCapabilitiesError
from .protocol import (
    RemoteProtocolError,
    build_evaluate_request,
    decision_fingerprint,
    encode_json,
    outcome_from_wire,
)


DEFAULT_TIMEOUT_SECONDS = Decimal("300")
DEFAULT_MAX_REQUEST_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_RESPONSE_BYTES = 1024 * 1024
MAX_STDERR_BYTES = 64 * 1024
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


class ExternalBackendError(RuntimeError):
    """The external adapter could not safely evaluate a decision."""


class ExternalBackendConfigurationError(ExternalBackendError):
    """External command, manifest, or limits are invalid."""


class ExternalBackendTimeoutError(ExternalBackendError):
    """The configured adapter exceeded its hard deadline."""


class ExternalBackendProtocolError(ExternalBackendError):
    """The adapter output did not match the bound remote protocol."""


def _positive_decimal(value: object, field_name: str) -> Decimal:
    if isinstance(value, bool):
        raise ExternalBackendConfigurationError(
            f"{field_name} must be a finite positive decimal"
        )
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ExternalBackendConfigurationError(
            f"{field_name} must be a decimal"
        ) from error
    if not result.is_finite() or result <= 0:
        raise ExternalBackendConfigurationError(
            f"{field_name} must be finite and positive"
        )
    return result


def _bounded_integer(
    value: object,
    field_name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExternalBackendConfigurationError(
            f"{field_name} must be an integer"
        )
    if not minimum <= value <= maximum:
        raise ExternalBackendConfigurationError(
            f"{field_name} must be between {minimum} and {maximum}"
        )
    return value


@dataclass(frozen=True, slots=True)
class ExternalBackendConfig:
    """One no-shell adapter command and its process safety limits."""

    command: tuple[str, ...]
    timeout_seconds: Decimal | int | str = DEFAULT_TIMEOUT_SECONDS
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES
    working_directory: Path | None = None
    environment: tuple[tuple[str, str], ...] = field(
        default_factory=tuple,
        repr=False,
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.command, tuple)
            or not 1 <= len(self.command) <= 64
            or any(
                not isinstance(part, str)
                or not part
                or "\x00" in part
                or len(part.encode("utf-8")) > 16_384
                for part in self.command
            )
        ):
            raise ExternalBackendConfigurationError(
                "command must contain 1..64 bounded non-empty strings"
            )
        executable = Path(self.command[0])
        if not executable.is_absolute():
            raise ExternalBackendConfigurationError(
                "external adapter executable path must be absolute"
            )
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ExternalBackendConfigurationError(
                f"external adapter is missing or not executable: {executable}"
            )
        timeout = _positive_decimal(self.timeout_seconds, "timeout_seconds")
        if timeout > Decimal(86_400):
            raise ExternalBackendConfigurationError(
                "timeout_seconds cannot exceed 86400"
            )
        _bounded_integer(
            self.max_request_bytes,
            "max_request_bytes",
            minimum=1_024,
            maximum=64 * 1024 * 1024,
        )
        _bounded_integer(
            self.max_response_bytes,
            "max_response_bytes",
            minimum=1_024,
            maximum=64 * 1024 * 1024,
        )
        if self.working_directory is not None:
            if (
                not isinstance(self.working_directory, Path)
                or not self.working_directory.is_absolute()
                or not self.working_directory.is_dir()
            ):
                raise ExternalBackendConfigurationError(
                    "working_directory must be an existing absolute directory"
                )
        if not isinstance(self.environment, tuple):
            raise ExternalBackendConfigurationError(
                "environment must be a tuple of key/value pairs"
            )
        seen: set[str] = set()
        for name, value in self.environment:
            if (
                not isinstance(name, str)
                or not _ENV_NAME_RE.fullmatch(name)
                or name in seen
            ):
                raise ExternalBackendConfigurationError(
                    "external environment names must be unique safe identifiers"
                )
            if (
                not isinstance(value, str)
                or "\x00" in value
                or len(value.encode("utf-8")) > 64 * 1024
            ):
                raise ExternalBackendConfigurationError(
                    f"external environment value for {name} is invalid"
                )
            seen.add(name)

    @property
    def timeout_float(self) -> float:
        converted = float(_positive_decimal(self.timeout_seconds, "timeout_seconds"))
        if not math.isfinite(converted):
            raise ExternalBackendConfigurationError("timeout_seconds is too large")
        return converted

    @classmethod
    def from_env(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> tuple["ExternalBackendConfig", SolverCapabilities]:
        env = os.environ if environment is None else environment
        raw_command = env.get("GTO_EXTERNAL_COMMAND_JSON", "").strip()
        if not raw_command:
            raise ExternalBackendConfigurationError(
                "GTO_EXTERNAL_COMMAND_JSON is required"
            )
        try:
            command_value = json.loads(raw_command)
        except json.JSONDecodeError as error:
            raise ExternalBackendConfigurationError(
                "GTO_EXTERNAL_COMMAND_JSON must be a JSON array"
            ) from error
        if (
            not isinstance(command_value, list)
            or any(not isinstance(part, str) for part in command_value)
        ):
            raise ExternalBackendConfigurationError(
                "GTO_EXTERNAL_COMMAND_JSON must be an array of strings"
            )

        raw_environment = env.get("GTO_EXTERNAL_ENV_JSON", "{}").strip()
        try:
            environment_value = json.loads(raw_environment)
        except json.JSONDecodeError as error:
            raise ExternalBackendConfigurationError(
                "GTO_EXTERNAL_ENV_JSON must be a JSON object"
            ) from error
        if (
            not isinstance(environment_value, dict)
            or any(
                not isinstance(name, str) or not isinstance(value, str)
                for name, value in environment_value.items()
            )
        ):
            raise ExternalBackendConfigurationError(
                "GTO_EXTERNAL_ENV_JSON must map strings to strings"
            )

        raw_workdir = env.get("GTO_EXTERNAL_WORKDIR", "").strip()
        workdir = Path(raw_workdir).expanduser().resolve() if raw_workdir else None
        try:
            max_request = int(
                env.get(
                    "GTO_EXTERNAL_MAX_REQUEST_BYTES",
                    str(DEFAULT_MAX_REQUEST_BYTES),
                )
            )
            max_response = int(
                env.get(
                    "GTO_EXTERNAL_MAX_RESPONSE_BYTES",
                    str(DEFAULT_MAX_RESPONSE_BYTES),
                )
            )
        except ValueError as error:
            raise ExternalBackendConfigurationError(
                "external byte limits must be integers"
            ) from error

        manifest_path_raw = env.get(
            "GTO_EXTERNAL_CAPABILITIES_PATH",
            "",
        ).strip()
        if not manifest_path_raw:
            raise ExternalBackendConfigurationError(
                "GTO_EXTERNAL_CAPABILITIES_PATH is required"
            )
        manifest_path = Path(manifest_path_raw).expanduser().resolve()
        if not manifest_path.is_file():
            raise ExternalBackendConfigurationError(
                f"capability manifest is missing: {manifest_path}"
            )
        try:
            manifest_bytes = manifest_path.read_bytes()
        except OSError as error:
            raise ExternalBackendConfigurationError(
                f"cannot read capability manifest: {error}"
            ) from error
        if len(manifest_bytes) > 256 * 1024:
            raise ExternalBackendConfigurationError(
                "capability manifest exceeds 262144 bytes"
            )
        try:
            manifest_value = json.loads(manifest_bytes)
            capabilities = SolverCapabilities.from_wire(manifest_value)
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            SolverCapabilitiesError,
        ) as error:
            raise ExternalBackendConfigurationError(
                f"capability manifest is invalid: {error}"
            ) from error

        return (
            cls(
                command=tuple(command_value),
                timeout_seconds=env.get(
                    "GTO_EXTERNAL_TIMEOUT_SECONDS",
                    str(DEFAULT_TIMEOUT_SECONDS),
                ),
                max_request_bytes=max_request,
                max_response_bytes=max_response,
                working_directory=workdir,
                environment=tuple(sorted(environment_value.items())),
            ),
            capabilities,
        )


def _adapter_environment(config: ExternalBackendConfig) -> dict[str, str]:
    # Deliberately do not inherit GTO_REMOTE_AUTH_TOKEN, API keys, or the
    # server's general environment.  A licensed adapter receives only values
    # explicitly provisioned for it.
    result: dict[str, str] = {}
    for name in ("PATH", "LANG", "LC_ALL", "TZ"):
        value = os.environ.get(name)
        if value:
            result[name] = value
    result["PYTHONUNBUFFERED"] = "1"
    result.update(dict(config.environment))
    return result


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover - solve server target is Linux.
            process.kill()
    except (OSError, ProcessLookupError):
        process.kill()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover - SIGKILL should win.
        pass


class ExternalSolverBackend:
    """Router-compatible adapter around one explicitly configured executable."""

    def __init__(
        self,
        process_config: ExternalBackendConfig,
        capabilities: SolverCapabilities,
        *,
        enabled: bool = True,
        owned_simulator_acknowledged: bool = True,
    ) -> None:
        if not isinstance(process_config, ExternalBackendConfig):
            raise ExternalBackendConfigurationError(
                "process_config must be ExternalBackendConfig"
            )
        if not isinstance(capabilities, SolverCapabilities):
            raise ExternalBackendConfigurationError(
                "capabilities must be SolverCapabilities"
            )
        if not isinstance(enabled, bool) or not isinstance(
            owned_simulator_acknowledged,
            bool,
        ):
            raise ExternalBackendConfigurationError(
                "enabled and owned_simulator_acknowledged must be boolean"
            )
        self.process_config = process_config
        self.capabilities = capabilities
        self.config = SimpleNamespace(
            engine_path=Path(process_config.command[0]),
            enabled=enabled,
            owned_simulator_acknowledged=owned_simulator_acknowledged,
        )

    @classmethod
    def from_env(
        cls,
        live_config: object,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> "ExternalSolverBackend":
        process_config, capabilities = ExternalBackendConfig.from_env(environment)
        return cls(
            process_config,
            capabilities,
            enabled=bool(getattr(live_config, "enabled", False)),
            owned_simulator_acknowledged=bool(
                getattr(live_config, "owned_simulator_acknowledged", False)
            ),
        )

    def _preflight_reason(self, state: LiveDecisionState) -> str:
        if not isinstance(state, LiveDecisionState):
            return "external backend requires LiveDecisionState"
        if self.capabilities.full_six_max_ready and state.public_hand is None:
            return (
                "the declared full six-max backend requires a complete "
                "public_hand transcript"
            )
        if state.street == "PREFLOP":
            if self.capabilities.preflop_mode == "NONE":
                return "external backend does not support preflop"
            return ""
        if self.capabilities.postflop_mode == "NONE":
            return "external backend does not support postflop"
        players = state.active_villains + 1
        if state.public_hand is not None:
            replayed = replay_public_hand(state.public_hand)
            players = len(replayed.live_seats)
        if players > self.capabilities.max_postflop_players:
            return (
                f"external backend supports at most "
                f"{self.capabilities.max_postflop_players} postflop players; "
                f"the hand has {players}"
            )
        if players > 2 and self.capabilities.postflop_mode != "MULTIWAY_TREE":
            return "external backend is not a multiway postflop solver"
        return ""

    def request(self, state: LiveDecisionState) -> LiveGTOOutcome:
        """Invoke the adapter once and strictly bind its response."""

        reason = self._preflight_reason(state)
        if reason:
            return LiveGTOOutcome(
                status=LiveGTOStatus.UNSUPPORTED,
                reason=reason,
                latency_seconds=0.0,
                source=f"external {self.capabilities.backend_id}",
                model=self.capabilities.backend_id,
            )
        fingerprint = decision_fingerprint(state)
        request_id = f"backend-{fingerprint[:32]}"
        request_body = build_evaluate_request(request_id, state)
        payload = encode_json(request_body)
        if len(payload) > self.process_config.max_request_bytes:
            raise ExternalBackendProtocolError(
                "external backend request exceeds its configured byte limit"
            )

        started = time.monotonic()
        try:
            process = subprocess.Popen(
                self.process_config.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=(
                    str(self.process_config.working_directory)
                    if self.process_config.working_directory is not None
                    else None
                ),
                env=_adapter_environment(self.process_config),
                start_new_session=(os.name == "posix"),
            )
        except OSError as error:
            raise ExternalBackendError(
                f"could not start external adapter: {error}"
            ) from error
        try:
            try:
                stdout, stderr = process.communicate(
                    input=payload,
                    timeout=self.process_config.timeout_float,
                )
            except subprocess.TimeoutExpired as error:
                _terminate_process(process)
                raise ExternalBackendTimeoutError(
                    "external adapter exceeded its configured deadline"
                ) from error
        finally:
            if process.poll() is None:
                _terminate_process(process)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()

        if len(stdout) > self.process_config.max_response_bytes:
            raise ExternalBackendProtocolError(
                "external adapter response exceeds its configured byte limit"
            )
        if len(stderr) > MAX_STDERR_BYTES:
            raise ExternalBackendProtocolError(
                "external adapter stderr exceeds its configured byte limit"
            )
        if process.returncode != 0:
            # Stderr may contain captured state or licensed-backend details.
            # Keep it server-local and return only the exit status.
            raise ExternalBackendError(
                f"external adapter exited with status {process.returncode}"
            )
        try:
            outcome = outcome_from_wire(
                stdout,
                expected_request_id=request_id,
                expected_fingerprint=fingerprint,
            )
        except RemoteProtocolError as error:
            raise ExternalBackendProtocolError(str(error)) from error
        latency = time.monotonic() - started
        if outcome.latency_seconds > latency + 1:
            raise ExternalBackendProtocolError(
                "external adapter reported impossible latency"
            )
        return outcome

    def evaluate(self, state: LiveDecisionState) -> LiveGTOOutcome:
        """Fail closed for server routing and preserve typed solver outcomes."""

        if not self.config.enabled:
            return LiveGTOOutcome(
                status=LiveGTOStatus.DISABLED,
                reason="external solver backend is disabled",
                latency_seconds=0.0,
                source=f"external {self.capabilities.backend_id}",
                model=self.capabilities.backend_id,
            )
        if not self.config.owned_simulator_acknowledged:
            return LiveGTOOutcome(
                status=LiveGTOStatus.DISABLED,
                reason="owned-simulator acknowledgement is missing",
                latency_seconds=0.0,
                source=f"external {self.capabilities.backend_id}",
                model=self.capabilities.backend_id,
            )
        started = time.monotonic()
        try:
            return self.request(state)
        except (ExternalBackendError, RemoteProtocolError, ValueError) as error:
            return LiveGTOOutcome(
                status=LiveGTOStatus.FAILED,
                reason=f"external solver backend failed: {error}",
                latency_seconds=time.monotonic() - started,
                source=f"external {self.capabilities.backend_id}",
                model=self.capabilities.backend_id,
            )


__all__ = [
    "DEFAULT_MAX_REQUEST_BYTES",
    "DEFAULT_MAX_RESPONSE_BYTES",
    "DEFAULT_TIMEOUT_SECONDS",
    "ExternalBackendConfig",
    "ExternalBackendConfigurationError",
    "ExternalBackendError",
    "ExternalBackendProtocolError",
    "ExternalBackendTimeoutError",
    "ExternalSolverBackend",
]
