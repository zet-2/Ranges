"""Small authenticated HTTP front end for :class:`live_gto.LiveGTORouter`.

The endpoint is synchronous by design: the OCR machine submits one captured
decision and receives one minimal outcome.  Exactly one call may enter the
router at a time, because a solver process can consume the configured memory
limit by itself.  Concurrent calls fail quickly with HTTP 429.

TLS is intentionally terminated by a local reverse proxy such as Caddy.  The
default bind address is loopback, and non-loopback binding requires an explicit
environment acknowledgement.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hmac
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import signal
import socket
import threading
from typing import Any
from urllib.parse import urlsplit

from live_gto import LiveGTOConfig, LiveGTORouter

from .capabilities import capabilities_for_router
from .external_backend import (
    ExternalBackendConfigurationError,
    ExternalSolverBackend,
)
from .protocol import (
    PROTOCOL_SCHEMA_VERSION,
    RemoteProtocolError,
    encode_json,
    outcome_to_wire,
    parse_evaluate_request,
)


DEFAULT_PORT = 8787
DEFAULT_MAX_REQUEST_BYTES = 256 * 1024
DEFAULT_CACHE_ENTRIES = 512
MIN_BEARER_TOKEN_BYTES = 32
MAX_BEARER_TOKEN_BYTES = 16 * 1024
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class ConfigurationError(ValueError):
    """The API cannot start safely with its current environment."""


class BusyError(RuntimeError):
    """The one solver slot is already occupied."""


class RequestInProgressError(BusyError):
    """An identical idempotent request is already running."""


class IdempotencyConflictError(RuntimeError):
    """A request ID was reused for a different captured state."""


@dataclass(frozen=True, slots=True)
class ServerConfig:
    bearer_token: str
    bind_host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES
    idempotency_cache_entries: int = DEFAULT_CACHE_ENTRIES
    socket_timeout_seconds: float = 15.0
    source_url: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.bearer_token, str):
            raise ConfigurationError("GTO_REMOTE_AUTH_TOKEN must be text")
        token_size = len(self.bearer_token.encode("utf-8"))
        if (
            not MIN_BEARER_TOKEN_BYTES <= token_size <= MAX_BEARER_TOKEN_BYTES
            or any(
                character.isspace()
                or ord(character) < 33
                or ord(character) == 127
                for character in self.bearer_token
            )
        ):
            raise ConfigurationError(
                "GTO_REMOTE_AUTH_TOKEN must contain 32..16384 "
                "non-whitespace UTF-8 bytes"
            )
        if not isinstance(self.port, int) or not 0 <= self.port <= 65_535:
            raise ConfigurationError("GTO_API_PORT must be 0..65535")
        if not 1_024 <= self.max_request_bytes <= 16 * 1024 * 1024:
            raise ConfigurationError(
                "GTO_API_MAX_REQUEST_BYTES must be between 1024 and 16777216"
            )
        if not 1 <= self.idempotency_cache_entries <= 100_000:
            raise ConfigurationError(
                "GTO_API_IDEMPOTENCY_CACHE_ENTRIES must be 1..100000"
            )
        if not 1 <= self.socket_timeout_seconds <= 300:
            raise ConfigurationError(
                "GTO_API_SOCKET_TIMEOUT_SECONDS must be 1..300"
            )

    @classmethod
    def from_env(cls) -> "ServerConfig":
        token = os.getenv(
            "GTO_REMOTE_AUTH_TOKEN",
            os.getenv(
                "GTO_REMOTE_BEARER_TOKEN",
                os.getenv("GTO_API_BEARER_TOKEN", ""),
            ),
        )
        bind_host = os.getenv("GTO_API_BIND", "127.0.0.1").strip()
        if not bind_host:
            raise ConfigurationError("GTO_API_BIND cannot be empty")
        allow_nonloopback = _env_flag("GTO_API_ALLOW_NONLOOPBACK")
        if not _is_loopback(bind_host) and not allow_nonloopback:
            raise ConfigurationError(
                "non-loopback GTO_API_BIND requires "
                "GTO_API_ALLOW_NONLOOPBACK=1 and a TLS reverse proxy"
            )
        return cls(
            bearer_token=token,
            bind_host=bind_host,
            port=_env_integer("GTO_API_PORT", DEFAULT_PORT),
            max_request_bytes=_env_integer(
                "GTO_API_MAX_REQUEST_BYTES",
                DEFAULT_MAX_REQUEST_BYTES,
            ),
            idempotency_cache_entries=_env_integer(
                "GTO_API_IDEMPOTENCY_CACHE_ENTRIES",
                DEFAULT_CACHE_ENTRIES,
            ),
            socket_timeout_seconds=_env_float(
                "GTO_API_SOCKET_TIMEOUT_SECONDS",
                15.0,
            ),
            source_url=os.getenv("GTO_API_SOURCE_URL", "").strip(),
        )


@dataclass(frozen=True, slots=True)
class _CachedResponse:
    fingerprint: str
    body: bytes


def _env_flag(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _env_integer(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be an integer") from error


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        return float(raw)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be a number") from error


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return bool(
            socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        ) and all(
            address[4][0].startswith("127.") or address[4][0] == "::1"
            for address in socket.getaddrinfo(
                host,
                None,
                type=socket.SOCK_STREAM,
            )
        )
    except socket.gaierror:
        return False


class EvaluationService:
    """Serialize router access and retain recent idempotent responses."""

    def __init__(
        self,
        router: object,
        *,
        cache_entries: int = DEFAULT_CACHE_ENTRIES,
    ) -> None:
        if not 1 <= cache_entries <= 100_000:
            raise ValueError("cache_entries must be 1..100000")
        self.router = router
        self.capabilities = capabilities_for_router(router)
        self.cache_entries = cache_entries
        self._solver_slot = threading.Lock()
        self._state_lock = threading.Lock()
        self._cache: OrderedDict[str, _CachedResponse] = OrderedDict()
        self._in_flight: dict[str, str] = {}

    @property
    def busy(self) -> bool:
        return self._solver_slot.locked()

    def evaluate(
        self,
        request_id: str,
        fingerprint: str,
        state,
    ) -> tuple[bytes, bool]:
        """Return ``(response_body, cache_hit)`` or fail without queueing."""

        with self._state_lock:
            cached = self._cache.get(request_id)
            if cached is not None:
                if cached.fingerprint != fingerprint:
                    raise IdempotencyConflictError(
                        "request_id was already used for a different decision"
                    )
                self._cache.move_to_end(request_id)
                return cached.body, True
            running_fingerprint = self._in_flight.get(request_id)
            if running_fingerprint is not None:
                if running_fingerprint != fingerprint:
                    raise IdempotencyConflictError(
                        "request_id is running with a different decision"
                    )
                raise RequestInProgressError("this request is already running")

        if not self._solver_slot.acquire(blocking=False):
            raise BusyError("the solver is busy")
        try:
            # Recheck after acquiring the slot: a competing thread may have
            # completed while this request was waiting for the state mutex.
            with self._state_lock:
                cached = self._cache.get(request_id)
                if cached is not None:
                    if cached.fingerprint != fingerprint:
                        raise IdempotencyConflictError(
                            "request_id was already used for a different decision"
                        )
                    self._cache.move_to_end(request_id)
                    return cached.body, True
                self._in_flight[request_id] = fingerprint

            outcome = self.router.evaluate(state)
            body = encode_json(outcome_to_wire(request_id, fingerprint, outcome))
            with self._state_lock:
                self._cache[request_id] = _CachedResponse(fingerprint, body)
                self._cache.move_to_end(request_id)
                while len(self._cache) > self.cache_entries:
                    self._cache.popitem(last=False)
            return body, False
        finally:
            with self._state_lock:
                self._in_flight.pop(request_id, None)
            self._solver_slot.release()


class GTOHTTPServer(ThreadingHTTPServer):
    """Threaded transport; expensive work remains single-flight."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address,
        handler_class,
        *,
        api_config: ServerConfig,
        service: EvaluationService,
    ) -> None:
        self.api_config = api_config
        self.service = service
        super().__init__(server_address, handler_class)


class GTORequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "GTO-Remote/1"
    sys_version = ""

    @property
    def api_server(self) -> GTOHTTPServer:
        return self.server  # type: ignore[return-value]

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(
            self.api_server.api_config.socket_timeout_seconds
        )

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        target = urlsplit(self.path)
        if target.query:
            self._api_error(
                HTTPStatus.BAD_REQUEST,
                "QUERY_NOT_ALLOWED",
                "query parameters are not supported",
            )
            return
        path = target.path
        if path == "/health/live":
            self._json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "schema_version": PROTOCOL_SCHEMA_VERSION,
                },
            )
            return
        if path == "/health/ready":
            router_config = getattr(self.api_server.service.router, "config", None)
            engine_path = getattr(router_config, "engine_path", None)
            engine_ready = bool(
                engine_path
                and Path(engine_path).is_file()
                and os.access(engine_path, os.X_OK)
            )
            enabled = bool(getattr(router_config, "enabled", False))
            acknowledged = bool(
                getattr(router_config, "owned_simulator_acknowledged", False)
            )
            ready = engine_ready and enabled and acknowledged
            self._json(
                HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE,
                {
                    "status": "ready" if ready else "not_ready",
                    "engine_available": engine_ready,
                    "gto_enabled": enabled,
                    "owned_simulator_acknowledged": acknowledged,
                    "solver_busy": self.api_server.service.busy,
                    "backend_id": self.api_server.service.capabilities.backend_id,
                    "full_six_max_ready": (
                        self.api_server.service.capabilities.full_six_max_ready
                    ),
                },
            )
            return
        if path == "/v1/about":
            if not self._authorized():
                return
            self._json(
                HTTPStatus.OK,
                {
                    "schema_version": PROTOCOL_SCHEMA_VERSION,
                    "license": "AGPL-3.0-or-later",
                    "source_url": self.api_server.api_config.source_url,
                    "max_concurrent_evaluations": 1,
                    "solver_capabilities": (
                        self.api_server.service.capabilities.to_wire()
                    ),
                },
            )
            return
        self._api_error(HTTPStatus.NOT_FOUND, "NOT_FOUND", "unknown endpoint")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        # Each expensive evaluation is one self-contained exchange. Closing
        # afterwards also prevents an unread rejected request body from being
        # interpreted as a second request on the same connection.
        self.close_connection = True
        target = urlsplit(self.path)
        if target.query:
            self._api_error(
                HTTPStatus.BAD_REQUEST,
                "QUERY_NOT_ALLOWED",
                "query parameters are not supported",
            )
            return
        path = target.path
        if path != "/v1/evaluate":
            self._api_error(HTTPStatus.NOT_FOUND, "NOT_FOUND", "unknown endpoint")
            return
        if not self._authorized():
            return
        if self.headers.get("Transfer-Encoding") is not None:
            self.close_connection = True
            self._api_error(
                HTTPStatus.BAD_REQUEST,
                "TRANSFER_ENCODING_NOT_ALLOWED",
                "Transfer-Encoding is not supported; send Content-Length",
            )
            return
        content_type = self.headers.get("Content-Type", "")
        if content_type.split(";", 1)[0].strip().lower() != "application/json":
            self._api_error(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "CONTENT_TYPE",
                "Content-Type must be application/json",
            )
            return
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self._api_error(
                HTTPStatus.LENGTH_REQUIRED,
                "LENGTH_REQUIRED",
                "Content-Length is required",
            )
            return
        try:
            content_length = int(raw_length)
        except ValueError:
            self._api_error(
                HTTPStatus.BAD_REQUEST,
                "INVALID_LENGTH",
                "Content-Length must be an integer",
            )
            return
        if content_length <= 0:
            self._api_error(
                HTTPStatus.BAD_REQUEST,
                "EMPTY_BODY",
                "request body cannot be empty",
            )
            return
        if content_length > self.api_server.api_config.max_request_bytes:
            self.close_connection = True
            self._api_error(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "REQUEST_TOO_LARGE",
                "request body exceeds the configured safety limit",
            )
            return
        try:
            body = self.rfile.read(content_length)
        except (OSError, TimeoutError):
            self.close_connection = True
            self._api_error(
                HTTPStatus.REQUEST_TIMEOUT,
                "READ_TIMEOUT",
                "request body was not received in time",
            )
            return
        if len(body) != content_length:
            self.close_connection = True
            self._api_error(
                HTTPStatus.BAD_REQUEST,
                "TRUNCATED_BODY",
                "request body ended before Content-Length",
            )
            return
        try:
            request_id, state, fingerprint = parse_evaluate_request(body)
        except RemoteProtocolError as error:
            self._api_error(
                HTTPStatus.BAD_REQUEST,
                "INVALID_REQUEST",
                str(error),
            )
            return
        request_id_header = self.headers.get("X-Request-ID", "")
        if not _REQUEST_ID_RE.fullmatch(request_id_header):
            self._api_error(
                HTTPStatus.BAD_REQUEST,
                "INVALID_REQUEST_ID_HEADER",
                "X-Request-ID must contain 1..128 safe ASCII characters",
            )
            return
        if not hmac.compare_digest(request_id_header, request_id):
            self._api_error(
                HTTPStatus.BAD_REQUEST,
                "REQUEST_ID_MISMATCH",
                "X-Request-ID must equal request_id",
                extra_headers={"X-Request-ID": request_id},
            )
            return
        idempotency_key = self.headers.get("Idempotency-Key", "")
        if not _REQUEST_ID_RE.fullmatch(idempotency_key):
            self._api_error(
                HTTPStatus.BAD_REQUEST,
                "INVALID_IDEMPOTENCY_KEY",
                "Idempotency-Key must contain 1..128 URL-safe characters",
            )
            return
        if not hmac.compare_digest(idempotency_key, request_id):
            self._api_error(
                HTTPStatus.BAD_REQUEST,
                "IDEMPOTENCY_MISMATCH",
                "Idempotency-Key must equal request_id",
                extra_headers={"X-Request-ID": request_id},
            )
            return
        try:
            response_body, cached = self.api_server.service.evaluate(
                request_id,
                fingerprint,
                state,
            )
        except IdempotencyConflictError as error:
            self._api_error(
                HTTPStatus.CONFLICT,
                "IDEMPOTENCY_CONFLICT",
                str(error),
                extra_headers={"X-Request-ID": request_id},
            )
            return
        except RequestInProgressError as error:
            self._api_error(
                HTTPStatus.TOO_MANY_REQUESTS,
                "REQUEST_IN_PROGRESS",
                str(error),
                extra_headers={
                    "Retry-After": "1",
                    "X-Request-ID": request_id,
                },
            )
            return
        except BusyError as error:
            self._api_error(
                HTTPStatus.TOO_MANY_REQUESTS,
                "SOLVER_BUSY",
                str(error),
                extra_headers={
                    "Retry-After": "1",
                    "X-Request-ID": request_id,
                },
            )
            return
        except Exception:
            # Do not send exception text or captured state over the wire.
            self._api_error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "INTERNAL_ERROR",
                "evaluation failed unexpectedly",
                extra_headers={"X-Request-ID": request_id},
            )
            return
        self._send(
            HTTPStatus.OK,
            response_body,
            extra_headers={
                "Idempotency-Replayed": "true" if cached else "false",
                "X-Request-ID": request_id,
            },
        )

    def _authorized(self) -> bool:
        authorization = self.headers.get("Authorization", "")
        prefix = "Bearer "
        supplied = authorization[len(prefix) :] if authorization.startswith(prefix) else ""
        if supplied and hmac.compare_digest(
            supplied,
            self.api_server.api_config.bearer_token,
        ):
            return True
        self._api_error(
            HTTPStatus.UNAUTHORIZED,
            "UNAUTHORIZED",
            "a valid bearer token is required",
            extra_headers={"WWW-Authenticate": "Bearer"},
        )
        return False

    def _api_error(
        self,
        status: HTTPStatus,
        code: str,
        message: str,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._json(
            status,
            {
                "schema_version": PROTOCOL_SCHEMA_VERSION,
                "error": {"code": code, "message": message},
            },
            extra_headers=extra_headers,
        )

    def _json(
        self,
        status: HTTPStatus,
        value: Any,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._send(status, encode_json(value), extra_headers=extra_headers)

    def _send(
        self,
        status: HTTPStatus,
        body: bytes,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if extra_headers:
            for name, value in extra_headers.items():
                self.send_header(name, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def log_message(self, format: str, *args: Any) -> None:
        # BaseHTTPRequestHandler logs only method/path/status; request bodies
        # and Authorization headers are never logged.
        super().log_message(format, *args)


def create_server(
    api_config: ServerConfig,
    router: object,
) -> GTOHTTPServer:
    service = EvaluationService(
        router,
        cache_entries=api_config.idempotency_cache_entries,
    )
    return GTOHTTPServer(
        (api_config.bind_host, api_config.port),
        GTORequestHandler,
        api_config=api_config,
        service=service,
    )


def main() -> int:
    api_config = ServerConfig.from_env()
    base_dir = Path(os.getenv("GTO_SERVER_BASE_DIR", ".")).expanduser().resolve()
    live_config = LiveGTOConfig.from_env(base_dir)
    if not live_config.enabled:
        raise ConfigurationError("GTO_LIVE_ENABLED=1 is required on the solve server")
    if not live_config.owned_simulator_acknowledged:
        raise ConfigurationError(
            "GTO_OWNED_SIMULATOR_ACK=1 is required on the solve server"
        )
    backend_name = os.getenv("GTO_SERVER_BACKEND", "native").strip().lower()
    if backend_name == "native":
        if not live_config.engine_path.is_file() or not os.access(
            live_config.engine_path,
            os.X_OK,
        ):
            raise ConfigurationError(
                f"GTO engine is missing or not executable: {live_config.engine_path}"
            )
        router = LiveGTORouter(live_config)
    elif backend_name == "external":
        try:
            router = ExternalSolverBackend.from_env(live_config)
        except ExternalBackendConfigurationError as error:
            raise ConfigurationError(
                f"external solver backend is invalid: {error}"
            ) from error
    else:
        raise ConfigurationError(
            "GTO_SERVER_BACKEND must be 'native' or 'external'"
        )
    server = create_server(api_config, router)

    def terminate(_signum, _frame) -> None:
        # Raising in the main thread unwinds serve_forever without the
        # same-thread shutdown deadlock documented by socketserver.
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
