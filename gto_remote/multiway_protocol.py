"""Strict, transcript-first request protocol for multiway solving.

Protocol v2 transports a heads-up-shaped decision snapshot.  This additive v3
boundary instead transports only the canonical inputs that cannot be derived
from public action history:

* the complete :class:`gto_hand_history.PublicHandHistory`,
* Hero's physical seat and private cards, and
* an opaque capture identity used to bind the request to one observation.

Street, board, pot, stacks, contributions, acting seat, legal actions, and all
other public facts come exclusively from replaying ``public_hand``.  They are
intentionally absent from the wire schema, so a request cannot contain two
contradictory versions of the same public state.

This module defines requests only.  A structured v3 outcome contract belongs in
a separate boundary once the solver proof requirements are finalized.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
from typing import Any, Mapping

from gto_hand_history import (
    PublicHandHistory,
    PublicHandHistoryError,
    ReplayedHandState,
    public_hand_from_wire,
    public_hand_to_wire,
    replay_public_hand,
)

from .protocol import RemoteProtocolError


PROTOCOL_SCHEMA_VERSION = 3
MAX_REQUEST_BYTES = 256 * 1024
MAX_REQUEST_ID_BYTES = 128
MAX_CAPTURE_ID_BYTES = 128

_CARD_RE = re.compile(r"^[2-9TJQKA][cdhs]$")
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_REQUEST_KEYS = frozenset({"schema_version", "request_id", "state"})
_STATE_KEYS = frozenset(
    {"capture_id", "hero_seat", "hero_combo", "public_hand"}
)


class MultiwayProtocolError(RemoteProtocolError):
    """A multiway request violates the v3 wire contract."""


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
    raise MultiwayProtocolError(
        f"{field} schema mismatch: {'; '.join(details)}"
    )


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MultiwayProtocolError(f"{field} must be an object")
    return value


def _array(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise MultiwayProtocolError(f"{field} must be an array")
    return value


def _string(value: Any, field: str, *, max_bytes: int) -> str:
    if not isinstance(value, str):
        raise MultiwayProtocolError(f"{field} must be a string")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise MultiwayProtocolError(f"{field} must be valid UTF-8 text") from error
    if size > max_bytes:
        raise MultiwayProtocolError(
            f"{field} exceeds {max_bytes} UTF-8 bytes"
        )
    return value


def _identifier(value: Any, field: str, *, max_bytes: int) -> str:
    identifier = _string(value, field, max_bytes=max_bytes)
    if not identifier:
        raise MultiwayProtocolError(f"{field} cannot be empty")
    if any(ord(character) < 32 or ord(character) == 127 for character in identifier):
        raise MultiwayProtocolError(f"{field} cannot contain control characters")
    return identifier


def _request_id(value: Any) -> str:
    identifier = _identifier(
        value,
        "request_id",
        max_bytes=MAX_REQUEST_ID_BYTES,
    )
    if not _REQUEST_ID_RE.fullmatch(identifier):
        raise MultiwayProtocolError(
            "request_id must use safe ASCII identifier characters"
        )
    return identifier


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MultiwayProtocolError(f"{field} must be an integer")
    return value


def _card(value: Any, field: str) -> str:
    card = _string(value, field, max_bytes=2)
    if not _CARD_RE.fullmatch(card):
        raise MultiwayProtocolError(
            f"{field} must use canonical notation such as 'As' or 'Td'"
        )
    return card


def _validated_public_hand(value: Any) -> PublicHandHistory:
    try:
        return public_hand_from_wire(value)
    except PublicHandHistoryError as error:
        raise MultiwayProtocolError(
            f"state.public_hand is invalid: {error}"
        ) from error


def _public_hand_wire(history: PublicHandHistory) -> dict[str, Any]:
    try:
        return public_hand_to_wire(history)
    except PublicHandHistoryError as error:
        raise MultiwayProtocolError(
            f"state.public_hand is invalid: {error}"
        ) from error


@dataclass(frozen=True, slots=True)
class MultiwayDecisionState:
    """Canonical private-plus-public inputs for one Hero decision.

    ``replayed`` is the sole source of public decision facts.  In particular,
    this class has no singular-villain fields and stores no duplicated pot,
    stack, board, street, or legal-action snapshot.
    """

    public_hand: PublicHandHistory
    hero_seat: int
    hero_combo: tuple[str, str]
    capture_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.public_hand, PublicHandHistory):
            raise MultiwayProtocolError(
                "state.public_hand must be PublicHandHistory"
            )
        hero_seat = _integer(self.hero_seat, "state.hero_seat")
        capture_id = _identifier(
            self.capture_id,
            "state.capture_id",
            max_bytes=MAX_CAPTURE_ID_BYTES,
        )
        if capture_id != self.capture_id:  # Defensive if normalization is added.
            raise AssertionError("capture identity changed during validation")
        if not isinstance(self.hero_combo, tuple) or len(self.hero_combo) != 2:
            raise MultiwayProtocolError(
                "state.hero_combo must be a two-card tuple"
            )
        hero_combo = tuple(
            _card(card, f"state.hero_combo[{index}]")
            for index, card in enumerate(self.hero_combo)
        )
        if len(set(hero_combo)) != 2:
            raise MultiwayProtocolError("state.hero_combo cannot repeat a card")

        try:
            replayed = replay_public_hand(self.public_hand)
        except PublicHandHistoryError as error:
            raise MultiwayProtocolError(
                f"state.public_hand is invalid: {error}"
            ) from error

        occupied_seats = {seat.seat for seat in self.public_hand.seats}
        if hero_seat not in occupied_seats:
            raise MultiwayProtocolError(
                "state.hero_seat is not occupied in public_hand"
            )
        if replayed.actor_seat is None:
            raise MultiwayProtocolError(
                "state.public_hand is not at an active player decision"
            )
        if replayed.actor_seat != hero_seat:
            raise MultiwayProtocolError(
                "state.public_hand does not have action on Hero"
            )
        if not replayed.legal_actions:
            raise MultiwayProtocolError(
                "state.public_hand has no legal Hero actions"
            )
        if set(hero_combo) & set(replayed.board):
            raise MultiwayProtocolError(
                "state.hero_combo cannot repeat a public board card"
            )

    @property
    def replayed(self) -> ReplayedHandState:
        """Return all public facts derived from the canonical transcript."""

        try:
            return replay_public_hand(self.public_hand)
        except PublicHandHistoryError as error:  # Defensive after construction.
            raise MultiwayProtocolError(
                f"state.public_hand is invalid: {error}"
            ) from error

    @property
    def hand_id(self) -> str:
        return self.public_hand.hand_id

    @property
    def hero_position(self) -> str:
        for seat in self.public_hand.seats:
            if seat.seat == self.hero_seat:
                return seat.position
        raise AssertionError("validated Hero seat disappeared")

    @property
    def hero_stack_bb(self) -> Decimal:
        return self.replayed.stack_map[self.hero_seat]

    @property
    def hero_current_bet_bb(self) -> Decimal:
        return self.replayed.street_contribution_map[self.hero_seat]


def decision_state_to_wire(state: MultiwayDecisionState) -> dict[str, Any]:
    """Serialize a v3 state without duplicating replay-derived public facts."""

    if not isinstance(state, MultiwayDecisionState):
        raise MultiwayProtocolError("state must be MultiwayDecisionState")
    wire = {
        "capture_id": _identifier(
            state.capture_id,
            "state.capture_id",
            max_bytes=MAX_CAPTURE_ID_BYTES,
        ),
        "hero_seat": _integer(state.hero_seat, "state.hero_seat"),
        "hero_combo": [
            _card(card, f"state.hero_combo[{index}]")
            for index, card in enumerate(state.hero_combo)
        ],
        "public_hand": _public_hand_wire(state.public_hand),
    }
    # Apply the receive-side checks before allowing an invalid local object onto
    # the wire.  This also guarantees that every Decimal is emitted as text.
    decision_state_from_wire(wire)
    return wire


def decision_state_from_wire(value: Any) -> MultiwayDecisionState:
    """Validate and reconstruct a v3 decision state from JSON values."""

    raw = _object(value, "state")
    _exact_keys(raw, _STATE_KEYS, "state")
    hero_combo_raw = _array(raw["hero_combo"], "state.hero_combo")
    if len(hero_combo_raw) != 2:
        raise MultiwayProtocolError(
            "state.hero_combo must contain two cards"
        )
    hero_combo = tuple(
        _card(card, f"state.hero_combo[{index}]")
        for index, card in enumerate(hero_combo_raw)
    )
    return MultiwayDecisionState(
        public_hand=_validated_public_hand(raw["public_hand"]),
        hero_seat=_integer(raw["hero_seat"], "state.hero_seat"),
        hero_combo=hero_combo,  # type: ignore[arg-type]
        capture_id=_identifier(
            raw["capture_id"],
            "state.capture_id",
            max_bytes=MAX_CAPTURE_ID_BYTES,
        ),
    )


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise MultiwayProtocolError(
            f"value is not strict JSON: {error}"
        ) from error


def decision_fingerprint(state: MultiwayDecisionState) -> str:
    """Return a stable SHA-256 identity for the complete v3 capture."""

    return hashlib.sha256(
        _canonical_json(decision_state_to_wire(state))
    ).hexdigest()


def build_evaluate_request(
    request_id: str,
    state: MultiwayDecisionState,
) -> dict[str, Any]:
    """Build and size-check one schema-v3 evaluation request."""

    request = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "request_id": _request_id(request_id),
        "state": decision_state_to_wire(state),
    }
    if len(encode_json(request)) > MAX_REQUEST_BYTES:
        raise MultiwayProtocolError(
            f"encoded request exceeds {MAX_REQUEST_BYTES} bytes"
        )
    return request


def _loads_strict(payload: bytes | str) -> Any:
    if isinstance(payload, bytes):
        if len(payload) > MAX_REQUEST_BYTES:
            raise MultiwayProtocolError(
                f"JSON payload exceeds {MAX_REQUEST_BYTES} bytes"
            )
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise MultiwayProtocolError(
                "JSON payload must be UTF-8"
            ) from error
    elif isinstance(payload, str):
        try:
            payload_size = len(payload.encode("utf-8"))
        except UnicodeEncodeError as error:
            raise MultiwayProtocolError(
                "JSON payload must be valid UTF-8 text"
            ) from error
        if payload_size > MAX_REQUEST_BYTES:
            raise MultiwayProtocolError(
                f"JSON payload exceeds {MAX_REQUEST_BYTES} bytes"
            )
        text = payload
    else:
        raise MultiwayProtocolError(
            "JSON payload must be bytes or text"
        )
    if not text.strip():
        raise MultiwayProtocolError("JSON payload cannot be empty")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise MultiwayProtocolError(f"duplicate JSON key {key!r}")
            result[key] = item
        return result

    def reject_constant(value: str) -> None:
        raise MultiwayProtocolError(
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
    except MultiwayProtocolError:
        raise
    except (json.JSONDecodeError, InvalidOperation, ValueError) as error:
        raise MultiwayProtocolError(
            f"payload is not strict JSON: {error}"
        ) from error


def parse_evaluate_request(
    payload: bytes | str,
) -> tuple[str, MultiwayDecisionState, str]:
    """Parse a v3 request into ``(request_id, state, fingerprint)``."""

    request = _object(_loads_strict(payload), "request")
    _exact_keys(request, _REQUEST_KEYS, "request")
    schema_version = _integer(
        request["schema_version"],
        "schema_version",
    )
    if schema_version != PROTOCOL_SCHEMA_VERSION:
        raise MultiwayProtocolError(
            f"unsupported schema_version {schema_version}; "
            f"expected {PROTOCOL_SCHEMA_VERSION}"
        )
    request_id = _request_id(request["request_id"])
    state = decision_state_from_wire(request["state"])
    return request_id, state, decision_fingerprint(state)


def encode_json(value: Any) -> bytes:
    """Encode a v3 request as compact strict UTF-8 JSON."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise MultiwayProtocolError(
            f"value is not strict JSON: {error}"
        ) from error


__all__ = [
    "MAX_CAPTURE_ID_BYTES",
    "MAX_REQUEST_BYTES",
    "MAX_REQUEST_ID_BYTES",
    "MultiwayDecisionState",
    "MultiwayProtocolError",
    "PROTOCOL_SCHEMA_VERSION",
    "build_evaluate_request",
    "decision_fingerprint",
    "decision_state_from_wire",
    "decision_state_to_wire",
    "encode_json",
    "parse_evaluate_request",
]
