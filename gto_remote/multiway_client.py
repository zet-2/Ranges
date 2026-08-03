"""Authenticated HTTP client for transcript-first multiway protocol v3."""

from __future__ import annotations

import os
from pathlib import Path
from decimal import Decimal
import socket
import time
from typing import Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import OpenerDirector, Request

from live_gto import LiveGTOOutcome, LiveGTOStatus

from .capabilities import (
    SolverCapabilities,
    SolverCapabilitiesError,
    parse_capabilities_json,
)
from .client import (
    RemoteGTOClient,
    RemoteGTOClientConfig,
    RemoteGTOClientError,
    RemoteGTOConfigurationError,
    RemoteGTOConnectionError,
    RemoteGTOHTTPError,
    RemoteGTORequestError,
    RemoteGTORequestTooLargeError,
    RemoteGTOResponseProtocolError,
    RemoteGTOTimeoutError,
    _json_content_type,
    _read_limited,
    _validate_request_id,
)
from .multiway_outcome import (
    MultiwayOutcomeError,
    MultiwaySolveOutcome,
    outcome_from_wire,
    to_live_outcome,
)
from .multiway_protocol import (
    MultiwayDecisionState,
    MultiwayProtocolError,
    build_evaluate_request,
    decision_fingerprint,
    encode_json,
)


class RemoteMultiwayClient(RemoteGTOClient):
    """Pin one expected backend manifest and submit only schema-v3 states."""

    def __init__(
        self,
        config: RemoteGTOClientConfig,
        expected_capabilities: SolverCapabilities,
        *,
        opener: OpenerDirector | None = None,
        request_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if not isinstance(expected_capabilities, SolverCapabilities):
            raise RemoteGTOConfigurationError(
                "expected_capabilities must be SolverCapabilities"
            )
        super().__init__(
            config,
            opener=opener,
            request_id_factory=request_id_factory,
        )
        self.capabilities = expected_capabilities

    @staticmethod
    def _capabilities_from_env(
        environment: Mapping[str, str],
        *,
        required: bool,
    ) -> SolverCapabilities:
        raw_path = environment.get(
            "GTO_REMOTE_CAPABILITIES_PATH",
            "",
        ).strip()
        if not raw_path:
            if required:
                raise RemoteGTOConfigurationError(
                    "GTO_REMOTE_CAPABILITIES_PATH is required"
                )
            return SolverCapabilities(
                backend_id="remote-multiway-unconfigured",
                backend_version="0",
                preflop_mode="NONE",
                postflop_mode="NONE",
                max_postflop_players=0,
                stateful_through_river=False,
                range_conditioning="NONE",
                folded_card_bunching=False,
                card_model="ABSTRACT_BUCKETS",
                action_model="FIXED_DISCRETE_TREE",
                game_profile_id="not-configured",
                abstraction_id="not-configured",
                solution_concept="not configured",
                convergence_metric="not configured",
                convergence_target=Decimal(0),
                source_license="not configured",
            )
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise RemoteGTOConfigurationError(
                f"remote capability manifest is missing: {path}"
            )
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise RemoteGTOConfigurationError(
                f"cannot read remote capability manifest: {error}"
            ) from error
        if len(payload) > 256 * 1024:
            raise RemoteGTOConfigurationError(
                "remote capability manifest exceeds 262144 bytes"
            )
        try:
            return parse_capabilities_json(payload)
        except SolverCapabilitiesError as error:
            raise RemoteGTOConfigurationError(
                f"remote capability manifest is invalid: {error}"
            ) from error

    @classmethod
    def from_env(
        cls,
        live_config: object | None = None,
        *,
        environment: Mapping[str, str] | None = None,
        opener: OpenerDirector | None = None,
        request_id_factory: Callable[[], str] | None = None,
    ) -> "RemoteMultiwayClient":
        env = os.environ if environment is None else environment
        live_enabled = (
            bool(getattr(live_config, "enabled", False))
            if live_config is not None
            else False
        )
        config = RemoteGTOClientConfig.from_env(
            env,
            default_enabled=live_enabled,
        )
        if live_config is not None and not live_enabled and config.enabled:
            from dataclasses import replace

            config = replace(config, enabled=False)
        capabilities = cls._capabilities_from_env(
            env,
            required=config.enabled,
        )
        return cls(
            config,
            capabilities,
            opener=opener,
            request_id_factory=request_id_factory,
        )

    def request_structured(
        self,
        state: MultiwayDecisionState,
        *,
        request_id: str | None = None,
    ) -> MultiwaySolveOutcome:
        """Submit a v3 decision and return a fully validated mixed policy."""

        if not self.config.enabled:
            raise RemoteGTOConfigurationError("remote multiway GTO is disabled")
        if not isinstance(state, MultiwayDecisionState):
            raise RemoteGTORequestError(
                "remote multiway GTO requires MultiwayDecisionState"
            )
        try:
            support_gaps = self.capabilities.support_gaps_for_state(state)
        except SolverCapabilitiesError as error:
            raise RemoteGTORequestError(
                f"cannot validate remote capabilities: {error}"
            ) from error
        if support_gaps:
            raise RemoteGTORequestError(
                "remote backend does not support this decision: "
                + "; ".join(support_gaps)
            )
        request_id = _validate_request_id(
            self._request_id_factory() if request_id is None else request_id
        )
        try:
            request_body = build_evaluate_request(request_id, state)
            fingerprint = decision_fingerprint(state)
            payload = encode_json(request_body)
        except MultiwayProtocolError as error:
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
                "User-Agent": "Ranges-RemoteMultiwayGTO/1",
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
            outcome = outcome_from_wire(
                body,
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
                expected_abstraction_id=self.capabilities.abstraction_id,
                expected_solution_concept=(
                    self.capabilities.solution_concept
                ),
                expected_metric_name=self.capabilities.convergence_metric,
                expected_target_value=(
                    self.capabilities.convergence_target
                ),
            )
        except (MultiwayProtocolError, MultiwayOutcomeError) as error:
            raise RemoteGTOResponseProtocolError(str(error)) from error
        if outcome.solved:
            assert outcome.proof is not None
            if not outcome.proof.approximate:
                exactness_gaps = self.capabilities.exactness_gaps_for_state(
                    state
                )
                if exactness_gaps:
                    raise RemoteGTOResponseProtocolError(
                        "remote backend labelled an uncovered game exact: "
                        + "; ".join(exactness_gaps)
                    )
        return outcome

    def request(
        self,
        state: MultiwayDecisionState,
        *,
        request_id: str | None = None,
    ) -> LiveGTOOutcome:
        return to_live_outcome(
            self.request_structured(state, request_id=request_id)
        )

    def evaluate(
        self,
        state: MultiwayDecisionState,
        *,
        request_id: str | None = None,
    ) -> LiveGTOOutcome:
        if not self.config.enabled:
            return LiveGTOOutcome(
                status=LiveGTOStatus.DISABLED,
                reason="remote multiway GTO is disabled",
                latency_seconds=0.0,
                source="remote multiway GTO",
            )
        started = time.monotonic()
        try:
            return self.request(state, request_id=request_id)
        except RemoteGTOClientError as error:
            return LiveGTOOutcome(
                status=LiveGTOStatus.FAILED,
                reason=f"remote multiway GTO request failed: {error}",
                latency_seconds=time.monotonic() - started,
                source="remote multiway GTO",
            )


__all__ = ["RemoteMultiwayClient"]
