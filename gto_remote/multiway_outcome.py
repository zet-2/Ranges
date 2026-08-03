"""Structured, auditable responses for transcript-first multiway solving.

Multiway protocol v3 never accepts free-form prose as the solver contract.
The backend returns a complete mixed policy plus convergence evidence bound to
the exact decision fingerprint and capability manifest.  Human-readable advice
is rendered locally only after legality, frequency, identity, and convergence
checks succeed.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, DecimalException, InvalidOperation
import json
import re
from typing import Any, Mapping

from live_gto import LiveGTOOutcome, LiveGTOStatus

from .multiway_protocol import (
    MAX_REQUEST_ID_BYTES,
    MultiwayDecisionState,
    MultiwayProtocolError,
    PROTOCOL_SCHEMA_VERSION,
)


MAX_RESPONSE_BYTES = 1024 * 1024
MAX_POLICY_ACTIONS = 32
MAX_TEXT_BYTES = 4096
MAX_ABS_DECIMAL = Decimal("10000000")
MAX_LATENCY_SECONDS = Decimal("86400")
MAX_SIGNIFICANT_DIGITS = 28
MAX_DECIMAL_PLACES = 12
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+/-]{0,255}$")
_ACTION_KINDS = frozenset(
    {"FOLD", "CHECK", "CALL", "BET_TO", "RAISE_TO", "ALL_IN_TO"}
)
_RESPONSE_KEYS = frozenset(
    {"schema_version", "request_id", "decision_fingerprint", "outcome"}
)
_OUTCOME_KEYS = frozenset(
    {"status", "reason", "latency_seconds", "cache_hit", "policy", "proof"}
)
_ACTION_KEYS = frozenset({"kind", "amount_to_bb", "frequency", "ev_bb"})
_PROOF_KEYS = frozenset(
    {
        "backend_id",
        "backend_version",
        "capability_fingerprint",
        "game_profile_id",
        "abstraction_id",
        "solution_concept",
        "metric_name",
        "metric_value",
        "target_value",
        "iterations",
        "converged",
        "approximate",
    }
)


class MultiwayOutcomeError(MultiwayProtocolError):
    """A solver response is malformed, unbound, illegal, or unauditable."""


def _exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    field: str,
) -> None:
    actual = frozenset(value)
    if actual == expected:
        return
    details: list[str] = []
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        details.append("missing " + ", ".join(missing))
    if unexpected:
        details.append("unexpected " + ", ".join(unexpected))
    raise MultiwayOutcomeError(
        f"{field} schema mismatch: {'; '.join(details)}"
    )


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MultiwayOutcomeError(f"{field} must be an object")
    return value


def _array(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise MultiwayOutcomeError(f"{field} must be an array")
    return value


def _string(value: Any, field: str, *, max_bytes: int = MAX_TEXT_BYTES) -> str:
    if not isinstance(value, str):
        raise MultiwayOutcomeError(f"{field} must be a string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise MultiwayOutcomeError(f"{field} must be valid UTF-8") from error
    if len(encoded) > max_bytes:
        raise MultiwayOutcomeError(f"{field} exceeds {max_bytes} UTF-8 bytes")
    return value


def _single_line(
    value: Any,
    field: str,
    *,
    max_bytes: int = MAX_TEXT_BYTES,
) -> str:
    text = _string(value, field, max_bytes=max_bytes)
    if any(ord(character) < 32 or ord(character) == 127 for character in text):
        raise MultiwayOutcomeError(f"{field} cannot contain control characters")
    return text


def _identifier(value: Any, field: str) -> str:
    identifier = _single_line(value, field, max_bytes=256)
    if not _IDENTIFIER_RE.fullmatch(identifier):
        raise MultiwayOutcomeError(
            f"{field} must be a safe non-empty identifier"
        )
    return identifier


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise MultiwayOutcomeError(f"{field} must be boolean")
    return value


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MultiwayOutcomeError(f"{field} must be an integer")
    if value < minimum:
        raise MultiwayOutcomeError(f"{field} must be at least {minimum}")
    return value


def _bounded_decimal(
    result: Decimal,
    field: str,
    *,
    nonnegative: bool,
    maximum: Decimal,
) -> Decimal:
    if not result.is_finite():
        raise MultiwayOutcomeError(f"{field} must be finite")
    if nonnegative and result < 0:
        raise MultiwayOutcomeError(f"{field} must be non-negative")
    try:
        if abs(result) > maximum:
            raise MultiwayOutcomeError(
                f"{field} exceeds the protocol safety bound"
            )
        if result == 0:
            return Decimal(0)
        parts = result.as_tuple()
        if (
            len(parts.digits) > MAX_SIGNIFICANT_DIGITS
            or parts.exponent < -MAX_DECIMAL_PLACES
        ):
            raise MultiwayOutcomeError(
                f"{field} exceeds decimal precision limits"
            )
        return result.normalize()
    except (DecimalException, OverflowError, ValueError) as error:
        raise MultiwayOutcomeError(
            f"{field} exceeds decimal safety limits"
        ) from error


def _decimal(
    value: Any,
    field: str,
    *,
    nonnegative: bool = False,
    maximum: Decimal = MAX_ABS_DECIMAL,
) -> Decimal:
    if not isinstance(value, str) or not value.strip():
        raise MultiwayOutcomeError(f"{field} must be a decimal JSON string")
    try:
        result = Decimal(value)
    except (DecimalException, InvalidOperation, ValueError) as error:
        raise MultiwayOutcomeError(f"{field} is not a decimal") from error
    return _bounded_decimal(
        result,
        field,
        nonnegative=nonnegative,
        maximum=maximum,
    )


def _decimal_text(
    value: Decimal,
    field: str,
    *,
    nonnegative: bool = False,
    maximum: Decimal = MAX_ABS_DECIMAL,
) -> str:
    if not isinstance(value, Decimal):
        raise MultiwayOutcomeError(f"{field} must be a finite Decimal")
    normalized = _bounded_decimal(
        value,
        field,
        nonnegative=nonnegative,
        maximum=maximum,
    )
    try:
        return "0" if normalized == 0 else format(normalized, "f")
    except (DecimalException, OverflowError, ValueError) as error:
        raise MultiwayOutcomeError(
            f"{field} cannot be formatted safely"
        ) from error


def _optional_decimal(
    value: Any,
    field: str,
    *,
    nonnegative: bool = False,
) -> Decimal | None:
    if value is None:
        return None
    return _decimal(value, field, nonnegative=nonnegative)


def _optional_decimal_text(
    value: Decimal | None,
    field: str,
    *,
    nonnegative: bool = False,
) -> str | None:
    if value is None:
        return None
    return _decimal_text(value, field, nonnegative=nonnegative)


@dataclass(frozen=True, slots=True)
class MultiwayPolicyAction:
    kind: str
    amount_to_bb: Decimal | None
    frequency: Decimal
    ev_bb: Decimal | None = None

    def __post_init__(self) -> None:
        if self.kind not in _ACTION_KINDS:
            raise MultiwayOutcomeError(f"unsupported policy action {self.kind!r}")
        _decimal_text(self.frequency, "policy.frequency", nonnegative=True)
        if self.frequency > 1:
            raise MultiwayOutcomeError("policy.frequency cannot exceed 1")
        _optional_decimal_text(
            self.amount_to_bb,
            "policy.amount_to_bb",
            nonnegative=True,
        )
        _optional_decimal_text(self.ev_bb, "policy.ev_bb")
        needs_amount = self.kind in {
            "CALL",
            "BET_TO",
            "RAISE_TO",
            "ALL_IN_TO",
        }
        if needs_amount != (self.amount_to_bb is not None):
            qualifier = "requires" if needs_amount else "forbids"
            raise MultiwayOutcomeError(
                f"{self.kind} {qualifier} amount_to_bb"
            )


@dataclass(frozen=True, slots=True)
class MultiwaySolveProof:
    backend_id: str
    backend_version: str
    capability_fingerprint: str
    game_profile_id: str
    abstraction_id: str
    solution_concept: str
    metric_name: str
    metric_value: Decimal
    target_value: Decimal
    iterations: int
    converged: bool
    approximate: bool

    def __post_init__(self) -> None:
        for field, value in (
            ("backend_id", self.backend_id),
            ("backend_version", self.backend_version),
            ("game_profile_id", self.game_profile_id),
            ("abstraction_id", self.abstraction_id),
        ):
            _identifier(value, f"proof.{field}")
        for field, value in (
            ("solution_concept", self.solution_concept),
            ("metric_name", self.metric_name),
        ):
            if not _single_line(
                value,
                f"proof.{field}",
                max_bytes=256,
            ).strip():
                raise MultiwayOutcomeError(f"proof.{field} cannot be empty")
        if not _SHA256_RE.fullmatch(self.capability_fingerprint):
            raise MultiwayOutcomeError(
                "proof.capability_fingerprint must be lowercase SHA-256"
            )
        _decimal_text(
            self.metric_value,
            "proof.metric_value",
            nonnegative=True,
        )
        _decimal_text(
            self.target_value,
            "proof.target_value",
            nonnegative=True,
        )
        _integer(self.iterations, "proof.iterations", minimum=1)
        _boolean(self.converged, "proof.converged")
        _boolean(self.approximate, "proof.approximate")
        if self.converged and self.metric_value > self.target_value:
            raise MultiwayOutcomeError(
                "proof says converged but metric_value exceeds target_value"
            )
        if not self.approximate and self.metric_value != 0:
            raise MultiwayOutcomeError(
                "a non-zero convergence gap must be disclosed as approximate"
            )


@dataclass(frozen=True, slots=True)
class MultiwaySolveOutcome:
    status: LiveGTOStatus
    reason: str
    latency_seconds: Decimal
    cache_hit: bool
    policy: tuple[MultiwayPolicyAction, ...] = ()
    proof: MultiwaySolveProof | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, LiveGTOStatus):
            raise MultiwayOutcomeError("outcome.status must be LiveGTOStatus")
        _single_line(self.reason, "outcome.reason")
        _decimal_text(
            self.latency_seconds,
            "outcome.latency_seconds",
            nonnegative=True,
            maximum=MAX_LATENCY_SECONDS,
        )
        _boolean(self.cache_hit, "outcome.cache_hit")
        if not isinstance(self.policy, tuple):
            raise MultiwayOutcomeError("outcome.policy must be a tuple")
        if len(self.policy) > MAX_POLICY_ACTIONS:
            raise MultiwayOutcomeError("outcome.policy contains too many actions")
        if any(not isinstance(action, MultiwayPolicyAction) for action in self.policy):
            raise MultiwayOutcomeError(
                "outcome.policy must contain MultiwayPolicyAction values"
            )
        solved = self.status is LiveGTOStatus.SOLVED
        if solved:
            if not self.policy or self.proof is None:
                raise MultiwayOutcomeError(
                    "SOLVED outcome requires policy and proof"
                )
            total = sum(
                (action.frequency for action in self.policy),
                Decimal(0),
            )
            if abs(total - Decimal(1)) > Decimal("0.000001"):
                raise MultiwayOutcomeError(
                    "SOLVED policy frequencies must sum to 1"
                )
            if not self.proof.converged:
                raise MultiwayOutcomeError(
                    "SOLVED outcome requires a converged proof"
                )
            if self.reason:
                raise MultiwayOutcomeError(
                    "SOLVED outcome reason must be empty"
                )
        elif self.policy or self.proof is not None:
            raise MultiwayOutcomeError(
                "non-SOLVED outcome cannot contain policy or proof"
            )

    @property
    def solved(self) -> bool:
        return self.status is LiveGTOStatus.SOLVED


def _action_to_wire(action: MultiwayPolicyAction) -> dict[str, Any]:
    return {
        "kind": action.kind,
        "amount_to_bb": _optional_decimal_text(
            action.amount_to_bb,
            "policy.amount_to_bb",
            nonnegative=True,
        ),
        "frequency": _decimal_text(
            action.frequency,
            "policy.frequency",
            nonnegative=True,
        ),
        "ev_bb": _optional_decimal_text(action.ev_bb, "policy.ev_bb"),
    }


def _action_from_wire(value: Any, field: str) -> MultiwayPolicyAction:
    raw = _object(value, field)
    _exact_keys(raw, _ACTION_KEYS, field)
    return MultiwayPolicyAction(
        kind=_string(raw["kind"], f"{field}.kind", max_bytes=32),
        amount_to_bb=_optional_decimal(
            raw["amount_to_bb"],
            f"{field}.amount_to_bb",
            nonnegative=True,
        ),
        frequency=_decimal(
            raw["frequency"],
            f"{field}.frequency",
            nonnegative=True,
        ),
        ev_bb=_optional_decimal(raw["ev_bb"], f"{field}.ev_bb"),
    )


def _proof_to_wire(proof: MultiwaySolveProof) -> dict[str, Any]:
    return {
        "backend_id": proof.backend_id,
        "backend_version": proof.backend_version,
        "capability_fingerprint": proof.capability_fingerprint,
        "game_profile_id": proof.game_profile_id,
        "abstraction_id": proof.abstraction_id,
        "solution_concept": proof.solution_concept,
        "metric_name": proof.metric_name,
        "metric_value": _decimal_text(
            proof.metric_value,
            "proof.metric_value",
            nonnegative=True,
        ),
        "target_value": _decimal_text(
            proof.target_value,
            "proof.target_value",
            nonnegative=True,
        ),
        "iterations": proof.iterations,
        "converged": proof.converged,
        "approximate": proof.approximate,
    }


def _proof_from_wire(value: Any) -> MultiwaySolveProof:
    raw = _object(value, "outcome.proof")
    _exact_keys(raw, _PROOF_KEYS, "outcome.proof")
    return MultiwaySolveProof(
        backend_id=_string(raw["backend_id"], "proof.backend_id", max_bytes=256),
        backend_version=_string(
            raw["backend_version"],
            "proof.backend_version",
            max_bytes=256,
        ),
        capability_fingerprint=_string(
            raw["capability_fingerprint"],
            "proof.capability_fingerprint",
            max_bytes=64,
        ),
        game_profile_id=_string(
            raw["game_profile_id"],
            "proof.game_profile_id",
            max_bytes=256,
        ),
        abstraction_id=_string(
            raw["abstraction_id"],
            "proof.abstraction_id",
            max_bytes=256,
        ),
        solution_concept=_string(
            raw["solution_concept"],
            "proof.solution_concept",
            max_bytes=256,
        ),
        metric_name=_string(
            raw["metric_name"],
            "proof.metric_name",
            max_bytes=256,
        ),
        metric_value=_decimal(
            raw["metric_value"],
            "proof.metric_value",
            nonnegative=True,
        ),
        target_value=_decimal(
            raw["target_value"],
            "proof.target_value",
            nonnegative=True,
        ),
        iterations=_integer(raw["iterations"], "proof.iterations", minimum=1),
        converged=_boolean(raw["converged"], "proof.converged"),
        approximate=_boolean(raw["approximate"], "proof.approximate"),
    )


def outcome_to_wire(
    request_id: str,
    fingerprint: str,
    outcome: MultiwaySolveOutcome,
) -> dict[str, Any]:
    if not isinstance(outcome, MultiwaySolveOutcome):
        raise MultiwayOutcomeError(
            "outcome must be MultiwaySolveOutcome"
        )
    if not _SHA256_RE.fullmatch(fingerprint):
        raise MultiwayOutcomeError(
            "decision_fingerprint must be lowercase SHA-256"
        )
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "request_id": _string(
            request_id,
            "request_id",
            max_bytes=MAX_REQUEST_ID_BYTES,
        ),
        "decision_fingerprint": fingerprint,
        "outcome": {
            "status": outcome.status.value,
            "reason": outcome.reason,
            "latency_seconds": _decimal_text(
                outcome.latency_seconds,
                "outcome.latency_seconds",
                nonnegative=True,
                maximum=MAX_LATENCY_SECONDS,
            ),
            "cache_hit": outcome.cache_hit,
            "policy": [_action_to_wire(action) for action in outcome.policy],
            "proof": (
                _proof_to_wire(outcome.proof)
                if outcome.proof is not None
                else None
            ),
        },
    }


def _loads_strict(payload: bytes | str) -> Any:
    if isinstance(payload, bytes):
        if len(payload) > MAX_RESPONSE_BYTES:
            raise MultiwayOutcomeError(
                f"response exceeds {MAX_RESPONSE_BYTES} bytes"
            )
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise MultiwayOutcomeError("response must be UTF-8") from error
    elif isinstance(payload, str):
        text = payload
        try:
            if len(text.encode("utf-8")) > MAX_RESPONSE_BYTES:
                raise MultiwayOutcomeError(
                    f"response exceeds {MAX_RESPONSE_BYTES} bytes"
                )
        except UnicodeEncodeError as error:
            raise MultiwayOutcomeError("response must be valid UTF-8") from error
    else:
        raise MultiwayOutcomeError("response must be bytes or text")
    if not text.strip():
        raise MultiwayOutcomeError("response cannot be empty")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise MultiwayOutcomeError(f"duplicate JSON key {key!r}")
            result[key] = item
        return result

    def reject_constant(value: str) -> None:
        raise MultiwayOutcomeError(
            f"non-finite JSON constant {value!r} is forbidden"
        )

    try:
        return json.loads(
            text,
            parse_float=Decimal,
            parse_int=int,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except MultiwayOutcomeError:
        raise
    except (json.JSONDecodeError, InvalidOperation, ValueError) as error:
        raise MultiwayOutcomeError(
            f"response is not strict JSON: {error}"
        ) from error


def _validate_policy_legality(
    state: MultiwayDecisionState,
    policy: tuple[MultiwayPolicyAction, ...],
) -> None:
    replayed = state.replayed
    legal = set(replayed.legal_actions)
    high = max(replayed.street_contribution_map.values(), default=Decimal(0))
    hero_current = replayed.street_contribution_map[state.hero_seat]
    hero_all_in_to = hero_current + replayed.stack_map[state.hero_seat]
    seen: set[tuple[str, Decimal | None]] = set()
    seen_commit_targets: set[Decimal] = set()
    for action in policy:
        identity = (action.kind, action.amount_to_bb)
        if identity in seen:
            raise MultiwayOutcomeError(
                "policy contains a duplicate action/amount"
            )
        seen.add(identity)
        if action.kind not in legal:
            raise MultiwayOutcomeError(
                f"policy action {action.kind} is not legal at the replayed node"
            )
        amount = action.amount_to_bb
        if action.kind == "CALL":
            call_target = min(high, hero_all_in_to)
            if amount != call_target:
                raise MultiwayOutcomeError(
                    "CALL amount_to_bb must equal Hero's replayed call target"
                )
            assert amount is not None
            if amount in seen_commit_targets:
                raise MultiwayOutcomeError(
                    "policy duplicates one physical chip-commitment branch"
                )
            seen_commit_targets.add(amount)
        elif action.kind in {"BET_TO", "RAISE_TO"}:
            assert amount is not None
            minimum = replayed.minimum_raise_to_bb
            if minimum is None or amount < minimum:
                raise MultiwayOutcomeError(
                    f"{action.kind} amount_to_bb is below the replayed minimum"
                )
            if amount >= hero_all_in_to:
                raise MultiwayOutcomeError(
                    f"{action.kind} must remain below Hero's all-in target"
                )
        elif action.kind == "ALL_IN_TO":
            if amount != hero_all_in_to:
                raise MultiwayOutcomeError(
                    "ALL_IN_TO amount_to_bb must equal Hero's all-in target"
                )
            assert amount is not None
            if amount in seen_commit_targets:
                raise MultiwayOutcomeError(
                    "policy duplicates one physical chip-commitment branch"
                )
            seen_commit_targets.add(amount)


def outcome_from_wire(
    payload: bytes | str | Mapping[str, Any],
    *,
    expected_request_id: str,
    expected_fingerprint: str,
    expected_state: MultiwayDecisionState,
    expected_backend_id: str,
    expected_backend_version: str,
    expected_capability_fingerprint: str,
    expected_game_profile_id: str,
    expected_abstraction_id: str,
    expected_solution_concept: str,
    expected_metric_name: str,
    expected_target_value: Decimal,
) -> MultiwaySolveOutcome:
    value = _loads_strict(payload) if isinstance(payload, (bytes, str)) else payload
    response = _object(value, "response")
    _exact_keys(response, _RESPONSE_KEYS, "response")
    if _integer(
        response["schema_version"],
        "response.schema_version",
        minimum=1,
    ) != PROTOCOL_SCHEMA_VERSION:
        raise MultiwayOutcomeError(
            "response schema_version differs from the request"
        )
    if _string(
        response["request_id"],
        "request_id",
        max_bytes=MAX_REQUEST_ID_BYTES,
    ) != expected_request_id:
        raise MultiwayOutcomeError("response request_id differs from the request")
    fingerprint = _string(
        response["decision_fingerprint"],
        "decision_fingerprint",
        max_bytes=64,
    )
    if fingerprint != expected_fingerprint:
        raise MultiwayOutcomeError(
            "response decision_fingerprint differs from the captured state"
        )

    raw = _object(response["outcome"], "outcome")
    _exact_keys(raw, _OUTCOME_KEYS, "outcome")
    status_text = _string(raw["status"], "outcome.status", max_bytes=32)
    try:
        status = LiveGTOStatus(status_text)
    except ValueError as error:
        raise MultiwayOutcomeError(
            f"unsupported outcome.status {status_text!r}"
        ) from error
    policy_values = _array(raw["policy"], "outcome.policy")
    if len(policy_values) > MAX_POLICY_ACTIONS:
        raise MultiwayOutcomeError("outcome.policy contains too many actions")
    policy = tuple(
        _action_from_wire(value, f"outcome.policy[{index}]")
        for index, value in enumerate(policy_values)
    )
    proof = None if raw["proof"] is None else _proof_from_wire(raw["proof"])
    outcome = MultiwaySolveOutcome(
        status=status,
        reason=_single_line(raw["reason"], "outcome.reason"),
        latency_seconds=_decimal(
            raw["latency_seconds"],
            "outcome.latency_seconds",
            nonnegative=True,
            maximum=MAX_LATENCY_SECONDS,
        ),
        cache_hit=_boolean(raw["cache_hit"], "outcome.cache_hit"),
        policy=policy,
        proof=proof,
    )
    if outcome.solved:
        assert proof is not None
        if proof.backend_id != expected_backend_id:
            raise MultiwayOutcomeError("proof backend_id differs from manifest")
        if proof.backend_version != expected_backend_version:
            raise MultiwayOutcomeError(
                "proof backend_version differs from manifest"
            )
        if proof.capability_fingerprint != expected_capability_fingerprint:
            raise MultiwayOutcomeError(
                "proof capability_fingerprint differs from manifest"
            )
        if proof.game_profile_id != expected_game_profile_id:
            raise MultiwayOutcomeError(
                "proof game_profile_id differs from manifest"
            )
        if proof.abstraction_id != expected_abstraction_id:
            raise MultiwayOutcomeError(
                "proof abstraction_id differs from manifest"
            )
        if proof.solution_concept != expected_solution_concept:
            raise MultiwayOutcomeError(
                "proof solution_concept differs from manifest"
            )
        if proof.metric_name != expected_metric_name:
            raise MultiwayOutcomeError(
                "proof metric_name differs from manifest"
            )
        if proof.target_value != expected_target_value:
            raise MultiwayOutcomeError(
                "proof target_value differs from manifest"
            )
        _validate_policy_legality(expected_state, policy)
    return outcome


def _display_action(action: MultiwayPolicyAction) -> tuple[str, str]:
    label = {
        "FOLD": "Fold",
        "CHECK": "Check",
        "CALL": "Call",
        "BET_TO": "Bet to",
        "RAISE_TO": "Raise to",
        "ALL_IN_TO": "All-in to",
    }[action.kind]
    if action.amount_to_bb is None:
        return label, "0 BB"
    return label, f"{format(action.amount_to_bb, 'f')} BB"


def _escape_markdown(value: str) -> str:
    return re.sub(r"([\\`*_\[\]{}()#+.!|>\-])", r"\\\1", value)


def render_analysis(outcome: MultiwaySolveOutcome) -> str:
    """Render a validated mixed policy for the existing terminal UI."""

    if not outcome.solved:
        return ""
    dominant = max(
        outcome.policy,
        key=lambda action: (action.frequency, action.kind),
    )
    action_label, size_label = _display_action(dominant)
    mix = "; ".join(
        f"{_display_action(action)[0]} "
        f"{format(action.frequency * Decimal(100), 'f')}%"
        + (
            f" @ {format(action.amount_to_bb, 'f')} BB"
            if action.amount_to_bb is not None
            else ""
        )
        for action in sorted(
            outcome.policy,
            key=lambda item: (-item.frequency, item.kind),
        )
    )
    assert outcome.proof is not None
    proof = outcome.proof
    disclosure = "approximate finite abstraction" if proof.approximate else "exact declared game"
    solution_concept = _escape_markdown(proof.solution_concept)
    metric_name = _escape_markdown(proof.metric_name)
    return (
        f"**Action:** {action_label}\n"
        f"**Size:** {size_label}\n"
        f"* **Mix:** {mix}\n"
        f"* **Solution:** {disclosure}; {solution_concept}\n"
        f"* **Convergence:** {metric_name} "
        f"{format(proof.metric_value, 'f')} <= "
        f"{format(proof.target_value, 'f')}"
    )


def to_live_outcome(outcome: MultiwaySolveOutcome) -> LiveGTOOutcome:
    """Adapt a verified v3 result to the current application display model."""

    proof = outcome.proof
    return LiveGTOOutcome(
        status=outcome.status,
        reason=outcome.reason.replace("[", r"\[").replace("]", r"\]"),
        latency_seconds=float(outcome.latency_seconds),
        analysis=render_analysis(outcome),
        source=(
            f"multiway {proof.game_profile_id}"
            if proof is not None
            else "multiway solver"
        ),
        model=(
            f"{proof.backend_id}@{proof.backend_version}"
            if proof is not None
            else ""
        ),
        cache_hit=outcome.cache_hit,
        approximate=bool(proof and proof.approximate),
    )


def encode_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise MultiwayOutcomeError(
            f"value is not strict JSON: {error}"
        ) from error


__all__ = [
    "MAX_POLICY_ACTIONS",
    "MAX_RESPONSE_BYTES",
    "MultiwayOutcomeError",
    "MultiwayPolicyAction",
    "MultiwaySolveOutcome",
    "MultiwaySolveProof",
    "encode_json",
    "outcome_from_wire",
    "outcome_to_wire",
    "render_analysis",
    "to_live_outcome",
]
