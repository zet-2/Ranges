"""Authenticated, bounded HTTP transport for the remote live-GTO protocol."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
import ipaddress
import math
import os
import re
import socket
import time
from typing import Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, OpenerDirector, Request, build_opener
import uuid

from live_gto import LiveDecisionState, LiveGTOOutcome, LiveGTOStatus

from .protocol import (
    RemoteProtocolError,
    build_evaluate_request,
    decision_fingerprint,
    encode_json,
    outcome_from_wire,
)


EVALUATE_PATH = "/v1/evaluate"
DEFAULT_MAX_REQUEST_BYTES = 256 * 1024
DEFAULT_MAX_RESPONSE_BYTES = 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024
_MIN_TOKEN_BYTES = 32
_MAX_TOKEN_BYTES = 16 * 1024
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


class RemoteGTOClientError(RuntimeError):
    """Base class for remote transport errors."""


class RemoteGTOConfigurationError(RemoteGTOClientError):
    """Remote transport configuration is unsafe or malformed."""


class RemoteGTORequestError(RemoteGTOClientError):
    """The request identity or live state violates the protocol."""


class RemoteGTORequestTooLargeError(RemoteGTOClientError):
    """The encoded request exceeds the configured byte limit."""


class RemoteGTOResponseTooLargeError(RemoteGTOClientError):
    """The response exceeds the configured byte limit."""


class RemoteGTOTimeoutError(RemoteGTOClientError):
    """The remote network operation timed out."""


class RemoteGTOConnectionError(RemoteGTOClientError):
    """The evaluator could not be reached."""


class RemoteGTOHTTPError(RemoteGTOClientError):
    """The evaluator returned a non-success HTTP status."""

    def __init__(
        self,
        status_code: int,
        reason: str,
        request_id: str,
        response_body: bytes = b"",
    ) -> None:
        self.status_code = status_code
        self.reason = reason
        self.request_id = request_id
        self.response_body = response_body
        super().__init__(f"remote evaluator returned HTTP {status_code} {reason}")


class RemoteGTOResponseProtocolError(RemoteGTOClientError):
    """The HTTP success response is not bound to the submitted decision."""


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RemoteGTOConfigurationError(
            f"{field_name} must be a positive integer"
        )
    return value


def _positive_timeout(value: object) -> float:
    if isinstance(value, bool):
        raise RemoteGTOConfigurationError(
            "timeout_seconds must be finite and positive"
        )
    try:
        timeout = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise RemoteGTOConfigurationError(
            "timeout_seconds must be a decimal number"
        ) from error
    if not timeout.is_finite() or timeout <= 0:
        raise RemoteGTOConfigurationError(
            "timeout_seconds must be finite and positive"
        )
    converted = float(timeout)
    if not math.isfinite(converted):
        raise RemoteGTOConfigurationError("timeout_seconds is too large")
    return converted


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
    raise RemoteGTOConfigurationError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off"
    )


def _environment_integer(
    environment: Mapping[str, str],
    name: str,
    default: int,
) -> int:
    raw = environment.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise RemoteGTOConfigurationError(f"{name} must be an integer") from error
    return _positive_integer(value, name)


@dataclass(frozen=True, slots=True)
class RemoteGTOClientConfig:
    """Remote endpoint and safety limits; the bearer token is repr-redacted."""

    endpoint: str
    bearer_token: str = field(repr=False)
    enabled: bool = True
    timeout_seconds: Decimal | int | str = Decimal(300)
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES
    allow_insecure_http: bool = False

    @classmethod
    def from_env(
        cls,
        environment: Mapping[str, str] | None = None,
        *,
        default_enabled: bool = False,
    ) -> "RemoteGTOClientConfig":
        """Load the explicit ``GTO_REMOTE_*`` environment contract."""

        env = os.environ if environment is None else environment
        enabled = _environment_flag(
            env,
            "GTO_REMOTE_ENABLED",
            "1" if default_enabled else "0",
        )
        endpoint = env.get("GTO_REMOTE_ENDPOINT", "").strip()
        token = env.get(
            "GTO_REMOTE_AUTH_TOKEN",
            env.get(
                "GTO_API_BEARER_TOKEN",
                env.get("GTO_REMOTE_BEARER_TOKEN", ""),
            ),
        )
        if enabled and not endpoint:
            raise RemoteGTOConfigurationError(
                "GTO_REMOTE_ENDPOINT cannot be empty"
            )
        if enabled and not token:
            raise RemoteGTOConfigurationError(
                "GTO_REMOTE_AUTH_TOKEN cannot be empty"
            )
        return cls(
            endpoint=endpoint,
            bearer_token=token,
            enabled=enabled,
            timeout_seconds=env.get("GTO_REMOTE_TIMEOUT_SECONDS", "300").strip(),
            max_request_bytes=_environment_integer(
                env,
                "GTO_REMOTE_MAX_REQUEST_BYTES",
                DEFAULT_MAX_REQUEST_BYTES,
            ),
            max_response_bytes=_environment_integer(
                env,
                "GTO_REMOTE_MAX_RESPONSE_BYTES",
                DEFAULT_MAX_RESPONSE_BYTES,
            ),
            allow_insecure_http=_environment_flag(
                env,
                "GTO_REMOTE_ALLOW_INSECURE_HTTP",
                "0",
            ),
        )


class _NoRedirectHandler(HTTPRedirectHandler):
    """Do not forward the bearer token to any redirect target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _validate_config(config: RemoteGTOClientConfig) -> None:
    if not isinstance(config, RemoteGTOClientConfig):
        raise RemoteGTOConfigurationError(
            "config must be RemoteGTOClientConfig"
        )
    if not isinstance(config.enabled, bool):
        raise RemoteGTOConfigurationError("enabled must be boolean")
    if not isinstance(config.allow_insecure_http, bool):
        raise RemoteGTOConfigurationError(
            "allow_insecure_http must be boolean"
        )
    _positive_timeout(config.timeout_seconds)
    _positive_integer(config.max_request_bytes, "max_request_bytes")
    _positive_integer(config.max_response_bytes, "max_response_bytes")
    if not config.enabled:
        return
    if not isinstance(config.endpoint, str):
        raise RemoteGTOConfigurationError("endpoint must be a URL string")
    parts = urlsplit(config.endpoint)
    schemes = {"https", "http"} if config.allow_insecure_http else {"https"}
    if parts.scheme.lower() not in schemes or not parts.hostname:
        raise RemoteGTOConfigurationError(
            "endpoint must be an absolute HTTPS URL"
        )
    if parts.scheme.lower() == "http":
        hostname = parts.hostname.lower()
        try:
            loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            loopback = hostname == "localhost"
        if not loopback:
            raise RemoteGTOConfigurationError(
                "insecure HTTP is permitted only on a loopback endpoint "
                "reached through a trusted tunnel"
            )
    if parts.username is not None or parts.password is not None:
        raise RemoteGTOConfigurationError(
            "endpoint must not contain URL credentials"
        )
    if parts.query or parts.fragment or parts.path.rstrip("/") != EVALUATE_PATH:
        raise RemoteGTOConfigurationError(
            f"endpoint must use exactly {EVALUATE_PATH} without query or fragment"
        )
    token = config.bearer_token
    if not isinstance(token, str) or not token:
        raise RemoteGTOConfigurationError("bearer_token cannot be empty")
    token_size = len(token.encode("utf-8"))
    if not _MIN_TOKEN_BYTES <= token_size <= _MAX_TOKEN_BYTES or any(
        character.isspace() or ord(character) < 33 or ord(character) == 127
        for character in token
    ):
        raise RemoteGTOConfigurationError(
            "bearer_token must contain 32..16384 non-whitespace UTF-8 bytes"
        )


def _validate_request_id(value: object) -> str:
    if not isinstance(value, str) or not _REQUEST_ID_RE.fullmatch(value):
        raise RemoteGTORequestError(
            "request_id must contain 1-128 safe ASCII identifier characters"
        )
    return value


def _content_length(headers: Mapping[str, str]) -> int | None:
    raw = headers.get("Content-Length")
    if raw is None:
        return None
    try:
        length = int(raw)
    except (TypeError, ValueError) as error:
        raise RemoteGTOResponseProtocolError(
            "response Content-Length is invalid"
        ) from error
    if length < 0:
        raise RemoteGTOResponseProtocolError(
            "response Content-Length is invalid"
        )
    return length


def _read_limited(response, limit: int, request_id: str) -> bytes:
    length = _content_length(response.headers)
    if length is not None and length > limit:
        raise RemoteGTOResponseTooLargeError(
            f"response {request_id} exceeds the {limit}-byte safety limit"
        )
    body = bytearray()
    while True:
        remaining = limit - len(body)
        chunk = response.read(min(_READ_CHUNK_BYTES, remaining + 1))
        if not chunk:
            break
        if not isinstance(chunk, bytes):
            raise RemoteGTOResponseProtocolError(
                "HTTP response stream returned non-bytes"
            )
        body.extend(chunk)
        if len(body) > limit:
            raise RemoteGTOResponseTooLargeError(
                f"response {request_id} exceeds the {limit}-byte safety limit"
            )
    return bytes(body)


def _json_content_type(headers: Mapping[str, str]) -> bool:
    media_type = headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    return media_type == "application/json" or media_type.endswith("+json")


class RemoteGTOClient:
    """POST one ``LiveDecisionState`` to the authenticated evaluator."""

    def __init__(
        self,
        config: RemoteGTOClientConfig,
        *,
        opener: OpenerDirector | None = None,
        request_id_factory: Callable[[], str] | None = None,
    ) -> None:
        _validate_config(config)
        if opener is not None and not callable(getattr(opener, "open", None)):
            raise RemoteGTOConfigurationError("opener must provide open()")
        if request_id_factory is not None and not callable(request_id_factory):
            raise RemoteGTOConfigurationError(
                "request_id_factory must be callable"
            )
        self.config = config
        self._timeout_seconds = _positive_timeout(config.timeout_seconds)
        self._opener = opener or build_opener(_NoRedirectHandler())
        self._request_id_factory = request_id_factory or (
            lambda: str(uuid.uuid4())
        )

    @classmethod
    def from_env(
        cls,
        live_config: object | None = None,
        *,
        environment: Mapping[str, str] | None = None,
        opener: OpenerDirector | None = None,
        request_id_factory: Callable[[], str] | None = None,
    ) -> "RemoteGTOClient":
        live_enabled = (
            bool(getattr(live_config, "enabled", False))
            if live_config is not None
            else False
        )
        config = RemoteGTOClientConfig.from_env(
            environment,
            default_enabled=live_enabled,
        )
        if live_config is not None and not live_enabled and config.enabled:
            config = replace(config, enabled=False)
        return cls(
            config,
            opener=opener,
            request_id_factory=request_id_factory,
        )

    def request(
        self,
        state: LiveDecisionState,
        *,
        request_id: str | None = None,
    ) -> LiveGTOOutcome:
        """Send one request, raising typed errors on any unsafe failure."""

        if not self.config.enabled:
            raise RemoteGTOConfigurationError("remote GTO is disabled")
        request_id = _validate_request_id(
            self._request_id_factory() if request_id is None else request_id
        )
        try:
            request_body = build_evaluate_request(request_id, state)
            fingerprint = decision_fingerprint(state)
            payload = encode_json(request_body)
        except RemoteProtocolError as error:
            raise RemoteGTORequestError(str(error)) from error
        if len(payload) > self.config.max_request_bytes:
            raise RemoteGTORequestTooLargeError(
                f"request {request_id} exceeds the "
                f"{self.config.max_request_bytes}-byte safety limit"
            )
        request = Request(
            self.config.endpoint,
            data=payload,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.config.bearer_token}",
                "Content-Type": "application/json",
                "User-Agent": "Ranges-RemoteGTO/1",
                "Idempotency-Key": request_id,
                "X-Request-ID": request_id,
            },
            method="POST",
        )

        try:
            response = self._opener.open(request, timeout=self._timeout_seconds)
            try:
                status = getattr(response, "status", None)
                if (
                    isinstance(status, bool)
                    or not isinstance(status, int)
                    or not 200 <= status < 300
                ):
                    body = _read_limited(
                        response,
                        self.config.max_response_bytes,
                        request_id,
                    )
                    raise RemoteGTOHTTPError(
                        status if isinstance(status, int) else 0,
                        "unexpected status",
                        request_id,
                        body,
                    )
                if response.headers.get("X-Request-ID") != request_id:
                    raise RemoteGTOResponseProtocolError(
                        "HTTP X-Request-ID does not match the request"
                    )
                if not _json_content_type(response.headers):
                    raise RemoteGTOResponseProtocolError(
                        "HTTP response Content-Type must be JSON"
                    )
                encoding = response.headers.get("Content-Encoding", "identity")
                if encoding.lower().strip() not in {"", "identity"}:
                    raise RemoteGTOResponseProtocolError(
                        "compressed HTTP responses are not accepted"
                    )
                body = _read_limited(
                    response,
                    self.config.max_response_bytes,
                    request_id,
                )
            finally:
                response.close()
        except HTTPError as error:
            try:
                body = _read_limited(
                    error,
                    self.config.max_response_bytes,
                    request_id,
                )
            finally:
                error.close()
            raise RemoteGTOHTTPError(
                error.code,
                str(error.reason),
                request_id,
                body,
            ) from error
        except (socket.timeout, TimeoutError) as error:
            raise RemoteGTOTimeoutError(
                f"remote evaluator exceeded {self._timeout_seconds} seconds"
            ) from error
        except URLError as error:
            if isinstance(error.reason, (socket.timeout, TimeoutError)):
                raise RemoteGTOTimeoutError(
                    f"remote evaluator exceeded {self._timeout_seconds} seconds"
                ) from error
            raise RemoteGTOConnectionError(
                f"remote evaluator connection failed: {error.reason}"
            ) from error
        except OSError as error:
            raise RemoteGTOConnectionError(
                f"remote evaluator connection failed: {error}"
            ) from error

        try:
            return outcome_from_wire(
                body,
                expected_request_id=request_id,
                expected_fingerprint=fingerprint,
            )
        except RemoteProtocolError as error:
            raise RemoteGTOResponseProtocolError(str(error)) from error

    def evaluate(
        self,
        state: LiveDecisionState,
        *,
        request_id: str | None = None,
    ) -> LiveGTOOutcome:
        """Fail closed for router integration without leaking transport errors."""

        if not self.config.enabled:
            return LiveGTOOutcome(
                status=LiveGTOStatus.DISABLED,
                reason="remote GTO is disabled",
                latency_seconds=0.0,
                source="remote GTO",
            )
        started = time.monotonic()
        try:
            return self.request(state, request_id=request_id)
        except RemoteGTOClientError as error:
            return LiveGTOOutcome(
                status=LiveGTOStatus.FAILED,
                reason=f"remote GTO request failed: {error}",
                latency_seconds=time.monotonic() - started,
                source="remote GTO",
            )
