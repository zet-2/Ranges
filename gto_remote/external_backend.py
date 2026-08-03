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

import ctypes
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Mapping

from live_gto import LiveDecisionState, LiveGTOOutcome, LiveGTOStatus

from .capabilities import (
    SolverCapabilities,
    SolverCapabilitiesError,
    parse_capabilities_json,
)
from .multiway_outcome import (
    MultiwayOutcomeError,
    MultiwaySolveOutcome,
    outcome_from_wire as multiway_outcome_from_wire,
)
from .multiway_protocol import (
    MultiwayDecisionState,
    MultiwayProtocolError,
    build_evaluate_request as build_multiway_evaluate_request,
    decision_fingerprint as multiway_decision_fingerprint,
    encode_json as encode_multiway_json,
)
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
_PROCESS_IO_CHUNK_BYTES = 64 * 1024
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_PR_SET_CHILD_SUBREAPER = 36
_PROCESS_TREE_LOCK = threading.Lock()
_LINUX_SUBREAPER_ENABLED = False


class ExternalBackendError(RuntimeError):
    """The external adapter could not safely evaluate a decision."""


class ExternalBackendConfigurationError(ExternalBackendError):
    """External command, manifest, or limits are invalid."""


class ExternalBackendTimeoutError(ExternalBackendError):
    """The configured adapter exceeded its hard deadline."""


class ExternalBackendProtocolError(ExternalBackendError):
    """The adapter output did not match the bound remote protocol."""


class ExternalBackendContainmentError(ExternalBackendError):
    """The adapter process tree could not be proven terminated."""


def _environment_flag(
    environment: Mapping[str, str],
    name: str,
    default: str,
) -> bool:
    raw = environment.get(name, default).strip().lower()
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    raise ExternalBackendConfigurationError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off"
    )


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
    allow_best_effort_process_cleanup: bool = False

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
        if not isinstance(self.allow_best_effort_process_cleanup, bool):
            raise ExternalBackendConfigurationError(
                "allow_best_effort_process_cleanup must be boolean"
            )
        if (
            not sys.platform.startswith("linux")
            and not self.allow_best_effort_process_cleanup
        ):
            raise ExternalBackendConfigurationError(
                "strong detached-process cleanup requires Linux; "
                "GTO_EXTERNAL_ALLOW_BEST_EFFORT_PROCESS_CLEANUP=1 is an "
                "unsafe development-only acknowledgement"
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
            capabilities = parse_capabilities_json(manifest_bytes)
        except SolverCapabilitiesError as error:
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
                allow_best_effort_process_cleanup=_environment_flag(
                    env,
                    "GTO_EXTERNAL_ALLOW_BEST_EFFORT_PROCESS_CLEANUP",
                    "0",
                ),
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


def _enable_linux_subreaper() -> None:
    """Make this dedicated server adopt orphaned adapter descendants."""

    global _LINUX_SUBREAPER_ENABLED
    if _LINUX_SUBREAPER_ENABLED:
        return
    if not sys.platform.startswith("linux"):
        raise ExternalBackendContainmentError(
            "Linux subreaper containment is unavailable on this OS"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = getattr(libc, "prctl", None)
    if prctl is None:
        raise ExternalBackendContainmentError(
            "Linux prctl is unavailable for adapter containment"
        )
    result = prctl(
        ctypes.c_int(_PR_SET_CHILD_SUBREAPER),
        ctypes.c_ulong(1),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise ExternalBackendContainmentError(
            "could not enable Linux child-subreaper containment: "
            f"errno {error_number}"
        )
    existing_children = _linux_direct_children()
    if existing_children:
        raise ExternalBackendContainmentError(
            "external solve server already owns child processes; "
            "dedicated process-tree containment cannot establish a clean "
            "baseline"
        )
    _LINUX_SUBREAPER_ENABLED = True


def _linux_direct_children() -> set[int]:
    """Return every direct child of every thread in this process."""

    task_root = Path("/proc/self/task")
    try:
        task_paths = tuple(task_root.iterdir())
    except OSError as error:
        raise ExternalBackendContainmentError(
            "cannot inspect Linux process children for containment"
        ) from error
    children: set[int] = set()
    for task_path in task_paths:
        try:
            raw = (task_path / "children").read_text(
                encoding="ascii",
                errors="strict",
            )
        except FileNotFoundError:
            # A worker thread may exit between listing and reading.
            continue
        except OSError as error:
            raise ExternalBackendContainmentError(
                "cannot inspect Linux process children for containment"
            ) from error
        for value in raw.split():
            try:
                child_pid = int(value)
            except ValueError as error:
                raise ExternalBackendContainmentError(
                    "Linux child-process metadata is malformed"
                ) from error
            if child_pid > 0:
                children.add(child_pid)
    return children


def _kill_and_reap_adopted_linux_children(
    baseline_children: frozenset[int],
    *,
    deadline_seconds: float = 5.0,
) -> None:
    """Kill descendants adopted after their adapter ancestry exits.

    ``PR_SET_CHILD_SUBREAPER`` reparents even double-forked or ``setsid()``
    descendants to this dedicated solve-server process. Killing one adopted
    parent may expose another generation, so cleanup repeats until the direct
    child set is stable and equal to its pre-request baseline.
    """

    deadline = time.monotonic() + deadline_seconds
    stable_empty_scans = 0
    while True:
        adopted = _linux_direct_children() - baseline_children
        for child_pid in adopted:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as error:
                raise ExternalBackendContainmentError(
                    "could not terminate an adopted adapter descendant"
                ) from error
        for child_pid in adopted:
            try:
                os.waitpid(child_pid, os.WNOHANG)
            except (ChildProcessError, ProcessLookupError):
                pass

        remaining = _linux_direct_children() - baseline_children
        if not remaining:
            stable_empty_scans += 1
            if stable_empty_scans >= 2:
                return
        else:
            stable_empty_scans = 0
        if time.monotonic() >= deadline:
            raise ExternalBackendContainmentError(
                "adapter descendants did not terminate before the "
                "containment deadline"
            )
        time.sleep(0.01)


def _terminate_process(
    process: subprocess.Popen[bytes],
    *,
    baseline_children: frozenset[int] = frozenset(),
) -> None:
    if os.name == "posix":
        # The leader starts in a separate process group. On Linux the server
        # is also a subreaper, so descendants that call setsid()/double-fork
        # are adopted and killed below after the original group is gone.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            if process.poll() is None:
                process.kill()
    elif process.poll() is None:  # pragma: no cover - solve server target is Linux.
        try:
            process.kill()
        except OSError:
            pass
    if process.poll() is None:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - SIGKILL should win.
            raise ExternalBackendContainmentError(
                "adapter process leader did not terminate after SIGKILL"
            )
    if sys.platform.startswith("linux"):
        _kill_and_reap_adopted_linux_children(baseline_children)


def _run_adapter_process(
    config: ExternalBackendConfig,
    payload: bytes,
) -> subprocess.CompletedProcess[bytes]:
    """Serialize process-tree supervision around one adapter exchange."""

    with _PROCESS_TREE_LOCK:
        return _run_adapter_process_serialized(config, payload)


def _run_adapter_process_serialized(
    config: ExternalBackendConfig,
    payload: bytes,
) -> subprocess.CompletedProcess[bytes]:
    """Exchange one request while bounding both output pipes during reads."""

    linux_containment = sys.platform.startswith("linux")
    if linux_containment:
        _enable_linux_subreaper()
        baseline_children = frozenset()
        if _linux_direct_children():
            # No non-adapter children are permitted in this dedicated server.
            # If a previous cleanup failed, do not let its descendants become
            # the baseline for a later request.
            _kill_and_reap_adopted_linux_children(baseline_children)
    else:
        if not config.allow_best_effort_process_cleanup:
            raise ExternalBackendContainmentError(
                "strong detached-process cleanup requires Linux"
            )
        baseline_children = frozenset()

    try:
        process = subprocess.Popen(
            config.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            cwd=(
                str(config.working_directory)
                if config.working_directory is not None
                else None
            ),
            env=_adapter_environment(config),
            start_new_session=(os.name == "posix"),
        )
    except OSError as error:
        raise ExternalBackendError(
            f"could not start external adapter: {error}"
        ) from error
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None

    selector = None
    try:
        selector = selectors.DefaultSelector()
        request_offset = 0
        stdout = bytearray()
        stderr = bytearray()
        buffers = {"stdout": stdout, "stderr": stderr}
        limits = {
            "stdout": config.max_response_bytes,
            "stderr": MAX_STDERR_BYTES,
        }
        deadline = time.monotonic() + config.timeout_float
    except BaseException:
        _terminate_process(
            process,
            baseline_children=baseline_children,
        )
        if selector is not None:
            selector.close()
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                stream.close()
            except OSError:
                pass
        raise

    def close_registered(stream) -> None:
        assert selector is not None
        try:
            selector.unregister(stream)
        except (KeyError, ValueError):
            pass
        try:
            stream.close()
        except OSError:
            pass

    try:
        assert selector is not None
        for name, stream in (
            ("stdout", process.stdout),
            ("stderr", process.stderr),
        ):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        os.set_blocking(process.stdin.fileno(), False)
        if payload:
            try:
                while request_offset < len(payload):
                    written = os.write(
                        process.stdin.fileno(),
                        payload[request_offset:],
                    )
                    if written <= 0:
                        break
                    request_offset += written
            except BlockingIOError:
                pass
            except BrokenPipeError:
                close_registered(process.stdin)
            if request_offset == len(payload):
                close_registered(process.stdin)
            elif not process.stdin.closed:
                selector.register(
                    process.stdin,
                    selectors.EVENT_WRITE,
                    "stdin",
                )
        else:
            process.stdin.close()

        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ExternalBackendTimeoutError(
                    "external adapter exceeded its configured deadline"
                )
            events = selector.select(min(0.05, remaining))
            if not events:
                if process.poll() is not None:
                    # A one-shot adapter must not leave descendants holding the
                    # protocol pipes open after its leader exits.
                    _terminate_process(
                        process,
                        baseline_children=baseline_children,
                    )
                continue
            for key, _ in events:
                stream = key.fileobj
                name = key.data
                if name == "stdin":
                    try:
                        written = os.write(
                            stream.fileno(),
                            payload[request_offset:],
                        )
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        close_registered(stream)
                        continue
                    request_offset += written
                    if request_offset == len(payload):
                        close_registered(stream)
                    continue

                try:
                    chunk = os.read(stream.fileno(), _PROCESS_IO_CHUNK_BYTES)
                except BlockingIOError:
                    continue
                if not chunk:
                    close_registered(stream)
                    continue
                output = buffers[name]
                output.extend(chunk)
                if len(output) > limits[name]:
                    label = "response" if name == "stdout" else "stderr"
                    raise ExternalBackendProtocolError(
                        f"external adapter {label} exceeds its configured "
                        "byte limit"
                    )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ExternalBackendTimeoutError(
                "external adapter exceeded its configured deadline"
            )
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as error:
            raise ExternalBackendTimeoutError(
                "external adapter exceeded its configured deadline"
            ) from error
        return subprocess.CompletedProcess(
            config.command,
            returncode,
            stdout=bytes(stdout),
            stderr=bytes(stderr),
        )
    finally:
        _terminate_process(
            process,
            baseline_children=baseline_children,
        )
        assert selector is not None
        selector.close()
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                stream.close()
            except OSError:
                pass


class ExternalSolverBackend:
    """Router-compatible adapter around one explicitly configured executable."""

    supported_schema_versions = (2, 3)

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

    def _preflight_reason(
        self,
        state: LiveDecisionState | MultiwayDecisionState,
    ) -> str:
        if not isinstance(state, (LiveDecisionState, MultiwayDecisionState)):
            return (
                "external backend requires LiveDecisionState or "
                "MultiwayDecisionState"
            )
        if isinstance(state, LiveDecisionState):
            players, peak_postflop_players, _ = (
                self.capabilities._state_player_profile(state)
            )
            street = str(state.street).strip().upper()
            multiway_path = (
                players > 2
                if street == "PREFLOP"
                else peak_postflop_players > 2
            )
            if multiway_path:
                return (
                    "multiway decisions require transcript-first schema v3; "
                    "schema v2 cannot carry a structured policy and proof"
                )
        gaps = self.capabilities.support_gaps_for_state(state)
        return gaps[0] if gaps else ""

    def request(
        self,
        state: LiveDecisionState | MultiwayDecisionState,
    ) -> LiveGTOOutcome | MultiwaySolveOutcome:
        """Invoke the adapter once and strictly bind its response."""

        reason = self._preflight_reason(state)
        if reason:
            if isinstance(state, MultiwayDecisionState):
                return MultiwaySolveOutcome(
                    status=LiveGTOStatus.UNSUPPORTED,
                    reason=reason,
                    latency_seconds=Decimal(0),
                    cache_hit=False,
                )
            return LiveGTOOutcome(
                status=LiveGTOStatus.UNSUPPORTED,
                reason=reason,
                latency_seconds=0.0,
                source=f"external {self.capabilities.backend_id}",
                model=self.capabilities.backend_id,
            )
        is_multiway = isinstance(state, MultiwayDecisionState)
        if is_multiway:
            fingerprint = multiway_decision_fingerprint(state)
            request_id = f"backend-v3-{fingerprint[:29]}"
            request_body = build_multiway_evaluate_request(request_id, state)
            payload = encode_multiway_json(request_body)
        else:
            fingerprint = decision_fingerprint(state)
            request_id = f"backend-{fingerprint[:32]}"
            request_body = build_evaluate_request(request_id, state)
            payload = encode_json(request_body)
        if len(payload) > self.process_config.max_request_bytes:
            raise ExternalBackendProtocolError(
                "external backend request exceeds its configured byte limit"
            )

        started = time.monotonic()
        completed = _run_adapter_process(self.process_config, payload)
        if completed.returncode != 0:
            # Stderr may contain captured state or licensed-backend details.
            # Keep it server-local and return only the exit status.
            raise ExternalBackendError(
                f"external adapter exited with status {completed.returncode}"
            )
        stdout = completed.stdout
        try:
            if is_multiway:
                assert isinstance(state, MultiwayDecisionState)
                outcome = multiway_outcome_from_wire(
                    stdout,
                    expected_request_id=request_id,
                    expected_fingerprint=fingerprint,
                    expected_state=state,
                    expected_backend_id=self.capabilities.backend_id,
                    expected_backend_version=self.capabilities.backend_version,
                    expected_capability_fingerprint=(
                        self.capabilities.manifest_fingerprint
                    ),
                    expected_game_profile_id=(
                        self.capabilities.game_profile_id
                    ),
                    expected_abstraction_id=(
                        self.capabilities.abstraction_id
                    ),
                    expected_solution_concept=(
                        self.capabilities.solution_concept
                    ),
                    expected_metric_name=(
                        self.capabilities.convergence_metric
                    ),
                    expected_target_value=(
                        self.capabilities.convergence_target
                    ),
                )
            else:
                outcome = outcome_from_wire(
                    stdout,
                    expected_request_id=request_id,
                    expected_fingerprint=fingerprint,
                )
        except (RemoteProtocolError, MultiwayOutcomeError) as error:
            raise ExternalBackendProtocolError(str(error)) from error
        latency = time.monotonic() - started
        reported_latency = float(outcome.latency_seconds)
        if reported_latency > latency + 1:
            raise ExternalBackendProtocolError(
                "external adapter reported impossible latency"
            )
        approximate = (
            bool(outcome.proof and outcome.proof.approximate)
            if isinstance(outcome, MultiwaySolveOutcome)
            else outcome.approximate
        )
        if outcome.solved and not approximate:
            exactness_gaps = self.capabilities.exactness_gaps_for_state(state)
            if exactness_gaps:
                raise ExternalBackendProtocolError(
                    "external adapter labelled an uncovered game exact: "
                    + "; ".join(exactness_gaps)
                )
        if isinstance(outcome, MultiwaySolveOutcome):
            return outcome
        # Backend identity is deployment configuration, not untrusted child
        # process prose.  The adapter may still describe its policy in the
        # analysis, but cannot impersonate a different solver in audit logs.
        return replace(
            outcome,
            source=f"external {self.capabilities.backend_id}",
            model=(
                f"{self.capabilities.backend_id}@"
                f"{self.capabilities.backend_version}"
            ),
        )

    @staticmethod
    def _typed_failure(
        state: object,
        status: LiveGTOStatus,
        reason: str,
        latency_seconds: float,
        *,
        source: str = "",
        model: str = "",
    ) -> LiveGTOOutcome | MultiwaySolveOutcome:
        if isinstance(state, MultiwayDecisionState):
            return MultiwaySolveOutcome(
                status=status,
                reason=reason,
                latency_seconds=Decimal(str(latency_seconds)),
                cache_hit=False,
            )
        return LiveGTOOutcome(
            status=status,
            reason=reason,
            latency_seconds=latency_seconds,
            source=source,
            model=model,
        )

    def evaluate(
        self,
        state: LiveDecisionState | MultiwayDecisionState,
    ) -> LiveGTOOutcome | MultiwaySolveOutcome:
        """Fail closed for server routing and preserve typed solver outcomes."""

        if not self.config.enabled:
            return self._typed_failure(
                state,
                LiveGTOStatus.DISABLED,
                "external solver backend is disabled",
                0.0,
                source=f"external {self.capabilities.backend_id}",
                model=self.capabilities.backend_id,
            )
        if not self.config.owned_simulator_acknowledged:
            return self._typed_failure(
                state,
                LiveGTOStatus.DISABLED,
                "owned-simulator acknowledgement is missing",
                0.0,
                source=f"external {self.capabilities.backend_id}",
                model=self.capabilities.backend_id,
            )
        started = time.monotonic()
        try:
            return self.request(state)
        except (
            ExternalBackendError,
            RemoteProtocolError,
            MultiwayProtocolError,
            MultiwayOutcomeError,
            ValueError,
        ) as error:
            return self._typed_failure(
                state,
                LiveGTOStatus.FAILED,
                f"external solver backend failed: {error}",
                time.monotonic() - started,
                source=f"external {self.capabilities.backend_id}",
                model=self.capabilities.backend_id,
            )


__all__ = [
    "DEFAULT_MAX_REQUEST_BYTES",
    "DEFAULT_MAX_RESPONSE_BYTES",
    "DEFAULT_TIMEOUT_SECONDS",
    "ExternalBackendConfig",
    "ExternalBackendConfigurationError",
    "ExternalBackendContainmentError",
    "ExternalBackendError",
    "ExternalBackendProtocolError",
    "ExternalBackendTimeoutError",
    "ExternalSolverBackend",
]
