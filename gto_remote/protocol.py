"""Strict wire protocol shared by the OCR client and remote GTO server.

The Mac sends a compact :class:`live_gto.LiveDecisionState`, not a SolveSpec.
Range reconstruction, cache selection, and solver execution consequently stay
on the solve server and use one version of the live routing code.

Decimal chip values are JSON strings.  This deliberately avoids silently
changing a captured amount through binary floating-point serialization.
Unknown fields, duplicate JSON keys, non-finite numbers, and malformed cards
are rejected before they reach the solver.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import re
from typing import Any, Mapping

from gto_hand_history import (
    PublicHandHistory,
    PublicHandHistoryError,
    public_hand_from_wire,
    public_hand_to_wire,
    replay_public_hand,
)
from live_gto import (
    LiveDecisionState,
    LiveGTOOutcome,
    LiveGTOStatus,
)
from preflop_observation import (
    ObservationProvenance,
    ObservedPreflopState,
    PROVIDER_POSITIONS,
)


PROTOCOL_SCHEMA_VERSION = 2
MAX_REQUEST_ID_BYTES = 128
MAX_HAND_ID_BYTES = 256
MAX_TEXT_BYTES = 4_096
MAX_ACTIONS = 16
MAX_ABS_BB = Decimal("10000000")
_CARD_RE = re.compile(r"^[2-9TJQKA][cdhs]$")
_REQUEST_KEYS = frozenset({"schema_version", "request_id", "state"})
_STATE_KEYS = frozenset(
    {
        "hand_id",
        "street",
        "board",
        "hero_combo",
        "hero_position",
        "villain_position",
        "hero_is_oop",
        "active_villains",
        "pot_bb",
        "hero_stack_bb",
        "villain_stack_bb",
        "hero_current_bet_bb",
        "villain_current_bet_bb",
        "amount_to_call_bb",
        "legal_actions",
        "street_root_confirmed",
        "action_history",
        "observed_bet_to_bb",
        "mapping_error",
        "preflop_observation",
        "preflop_mapping_error",
        "public_hand",
    }
)
_OBSERVATION_KEYS = frozenset(
    {
        "actor",
        "contributions",
        "folded",
        "initial_stacks",
        "terminal",
        "provenance",
        "all_in",
    }
)
_PROVENANCE_KEYS = frozenset(
    {"source", "preflop_index", "flop_index", "hand_id"}
)
_OUTCOME_KEYS = frozenset(
    {
        "status",
        "reason",
        "latency_seconds",
        "analysis",
        "source",
        "model",
        "cache_hit",
        "approximate",
        "spec_key",
    }
)
_RESPONSE_KEYS = frozenset(
    {"schema_version", "request_id", "decision_fingerprint", "outcome"}
)


class RemoteProtocolError(ValueError):
    """A remote request or response violates the versioned wire contract."""


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], field: str) -> None:
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
    raise RemoteProtocolError(f"{field} schema mismatch: {'; '.join(details)}")


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RemoteProtocolError(f"{field} must be an object")
    return value


def _array(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise RemoteProtocolError(f"{field} must be an array")
    return value


def _string(value: Any, field: str, *, max_bytes: int = MAX_TEXT_BYTES) -> str:
    if not isinstance(value, str):
        raise RemoteProtocolError(f"{field} must be a string")
    if len(value.encode("utf-8")) > max_bytes:
        raise RemoteProtocolError(f"{field} exceeds {max_bytes} UTF-8 bytes")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise RemoteProtocolError(f"{field} must be boolean")
    return value


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RemoteProtocolError(f"{field} must be an integer")
    return value


def _optional_index(value: Any, field: str) -> int | None:
    if value is None:
        return None
    result = _integer(value, field)
    if result < 0:
        raise RemoteProtocolError(f"{field} must be non-negative")
    return result


def _decimal_text(value: Decimal) -> str:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise RemoteProtocolError("decision state contains a non-finite Decimal")
    if value < 0:
        raise RemoteProtocolError("decision state chip amounts must be non-negative")
    if value > MAX_ABS_BB:
        raise RemoteProtocolError("decision state chip amount exceeds the safety bound")
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def _decimal(value: Any, field: str, *, nonnegative: bool = True) -> Decimal:
    if not isinstance(value, str) or not value.strip():
        raise RemoteProtocolError(f"{field} must be a decimal JSON string")
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise RemoteProtocolError(f"{field} is not a decimal") from error
    if not result.is_finite():
        raise RemoteProtocolError(f"{field} must be finite")
    if nonnegative and result < 0:
        raise RemoteProtocolError(f"{field} must be non-negative")
    if abs(result) > MAX_ABS_BB:
        raise RemoteProtocolError(f"{field} exceeds the protocol safety bound")
    return Decimal(0) if result == 0 else result.normalize()


def _card(value: Any, field: str) -> str:
    card = _string(value, field, max_bytes=2)
    if not _CARD_RE.fullmatch(card):
        raise RemoteProtocolError(
            f"{field} must use canonical notation such as 'As' or 'Td'"
        )
    return card


def _strings(
    value: Any,
    field: str,
    *,
    max_items: int = MAX_ACTIONS,
    item_max_bytes: int = 64,
) -> tuple[str, ...]:
    items = _array(value, field)
    if len(items) > max_items:
        raise RemoteProtocolError(f"{field} contains too many items")
    return tuple(
        _string(item, f"{field}[{index}]", max_bytes=item_max_bytes)
        for index, item in enumerate(items)
    )


def _amount_map_to_wire(values: tuple[tuple[str, Decimal], ...]) -> dict[str, str]:
    amount_map = dict(values)
    if set(amount_map) != set(PROVIDER_POSITIONS) or len(values) != len(amount_map):
        raise RemoteProtocolError(
            "preflop amount maps must contain each six-max position exactly once"
        )
    return {position: _decimal_text(amount_map[position]) for position in PROVIDER_POSITIONS}


def _amount_map_from_wire(value: Any, field: str) -> tuple[tuple[str, Decimal], ...]:
    amounts = _object(value, field)
    expected = frozenset(PROVIDER_POSITIONS)
    _exact_keys(amounts, expected, field)
    return tuple(
        (
            position,
            _decimal(amounts[position], f"{field}.{position}"),
        )
        for position in PROVIDER_POSITIONS
    )


def _position_set_from_wire(value: Any, field: str) -> frozenset[str]:
    positions = _strings(
        value,
        field,
        max_items=len(PROVIDER_POSITIONS),
        item_max_bytes=3,
    )
    if len(positions) != len(set(positions)):
        raise RemoteProtocolError(f"{field} cannot repeat a position")
    unknown = set(positions) - set(PROVIDER_POSITIONS)
    if unknown:
        raise RemoteProtocolError(f"{field} has unknown positions: {sorted(unknown)}")
    return frozenset(positions)


def _observation_to_wire(
    observation: ObservedPreflopState | None,
) -> dict[str, Any] | None:
    if observation is None:
        return None
    if not isinstance(observation, ObservedPreflopState):
        raise RemoteProtocolError(
            "preflop_observation must be ObservedPreflopState or None"
        )
    actor = observation.actor
    if actor is not None and actor not in PROVIDER_POSITIONS:
        raise RemoteProtocolError("preflop observation actor is not canonical")
    provenance = observation.provenance
    if not isinstance(provenance, ObservationProvenance):
        raise RemoteProtocolError("preflop observation provenance is invalid")
    return {
        "actor": actor,
        "contributions": _amount_map_to_wire(observation.contributions),
        "folded": sorted(observation.folded),
        "initial_stacks": _amount_map_to_wire(observation.initial_stacks),
        "terminal": _boolean(
            observation.terminal,
            "preflop_observation.terminal",
        ),
        "provenance": {
            "source": _string(provenance.source, "provenance.source"),
            "preflop_index": provenance.preflop_index,
            "flop_index": provenance.flop_index,
            "hand_id": _string(
                provenance.hand_id,
                "provenance.hand_id",
                max_bytes=MAX_HAND_ID_BYTES,
            ),
        },
        "all_in": sorted(observation.all_in),
    }


def _observation_from_wire(value: Any) -> ObservedPreflopState | None:
    if value is None:
        return None
    observation = _object(value, "state.preflop_observation")
    _exact_keys(observation, _OBSERVATION_KEYS, "state.preflop_observation")
    actor_value = observation["actor"]
    if actor_value is None:
        actor = None
    else:
        actor = _string(actor_value, "state.preflop_observation.actor", max_bytes=3)
        if actor not in PROVIDER_POSITIONS:
            raise RemoteProtocolError(
                "state.preflop_observation.actor is not a canonical position"
            )
    folded = _position_set_from_wire(
        observation["folded"], "state.preflop_observation.folded"
    )
    all_in = _position_set_from_wire(
        observation["all_in"], "state.preflop_observation.all_in"
    )
    if folded & all_in:
        raise RemoteProtocolError(
            "state.preflop_observation folded and all_in positions overlap"
        )
    provenance_value = _object(
        observation["provenance"],
        "state.preflop_observation.provenance",
    )
    _exact_keys(
        provenance_value,
        _PROVENANCE_KEYS,
        "state.preflop_observation.provenance",
    )
    provenance = ObservationProvenance(
        source=_string(
            provenance_value["source"],
            "state.preflop_observation.provenance.source",
        ),
        preflop_index=_optional_index(
            provenance_value["preflop_index"],
            "state.preflop_observation.provenance.preflop_index",
        ),
        flop_index=_optional_index(
            provenance_value["flop_index"],
            "state.preflop_observation.provenance.flop_index",
        ),
        hand_id=_string(
            provenance_value["hand_id"],
            "state.preflop_observation.provenance.hand_id",
            max_bytes=MAX_HAND_ID_BYTES,
        ),
    )
    return ObservedPreflopState(
        actor=actor,
        contributions=_amount_map_from_wire(
            observation["contributions"],
            "state.preflop_observation.contributions",
        ),
        folded=folded,
        initial_stacks=_amount_map_from_wire(
            observation["initial_stacks"],
            "state.preflop_observation.initial_stacks",
        ),
        terminal=_boolean(
            observation["terminal"],
            "state.preflop_observation.terminal",
        ),
        provenance=provenance,
        all_in=all_in,
    )


def _public_hand_to_wire(
    public_hand: PublicHandHistory | None,
) -> dict[str, Any] | None:
    if public_hand is None:
        return None
    try:
        return public_hand_to_wire(public_hand)
    except PublicHandHistoryError as error:
        raise RemoteProtocolError(f"state.public_hand is invalid: {error}") from error


def _public_hand_from_wire(value: Any) -> PublicHandHistory | None:
    if value is None:
        return None
    try:
        return public_hand_from_wire(value)
    except PublicHandHistoryError as error:
        raise RemoteProtocolError(f"state.public_hand is invalid: {error}") from error


def _validate_public_hand_binding(state: LiveDecisionState) -> None:
    """Bind an optional lossless transcript to the submitted Hero decision."""

    public_hand = state.public_hand
    if public_hand is None:
        return
    try:
        replayed = replay_public_hand(public_hand)
    except PublicHandHistoryError as error:  # Defensive; parsing already replays.
        raise RemoteProtocolError(f"state.public_hand is invalid: {error}") from error
    if public_hand.hand_id != state.hand_id:
        raise RemoteProtocolError(
            "state.public_hand belongs to a different hand_id"
        )
    if replayed.street != state.street:
        raise RemoteProtocolError(
            "state.public_hand street differs from the captured decision"
        )
    if replayed.board != state.board:
        raise RemoteProtocolError(
            "state.public_hand board differs from the captured decision"
        )
    matching_hero_seats = [
        seat.seat
        for seat in public_hand.seats
        if seat.position == state.hero_position
    ]
    if len(matching_hero_seats) != 1:
        raise RemoteProtocolError(
            "state.hero_position does not identify one public_hand seat"
        )
    hero_seat = matching_hero_seats[0]
    if replayed.actor_seat != hero_seat:
        raise RemoteProtocolError(
            "state.public_hand does not have action on Hero"
        )
    if len(replayed.live_seats - {hero_seat}) != state.active_villains:
        raise RemoteProtocolError(
            "state.public_hand active players differ from active_villains"
        )
    if replayed.amount_to_call_bb != state.amount_to_call_bb:
        raise RemoteProtocolError(
            "state.public_hand call amount differs from the captured decision"
        )
    observed_actions = {
        action.strip().upper().replace("_TO", "").replace("_", "-")
        for action in state.legal_actions
    }
    known_actions = {"FOLD", "CHECK", "CALL", "BET", "RAISE", "ALL-IN"}
    if not observed_actions or not observed_actions <= known_actions:
        raise RemoteProtocolError(
            "state.legal_actions are empty or unsupported"
        )
    replay_actions = set(replayed.legal_actions)
    allowed_actions: set[str] = set()
    for action in replay_actions:
        allowed_actions.add(
            {
                "BET_TO": "BET",
                "RAISE_TO": "RAISE",
                "ALL_IN_TO": "ALL-IN",
            }.get(action, action)
        )
    if "ALL_IN_TO" in replay_actions:
        # A poker UI may expose a shove through its ordinary bet/raise control
        # instead of a distinct ALL-IN button.
        hero_current = replayed.street_contribution_map[hero_seat]
        hero_all_in_target = hero_current + replayed.stack_map[hero_seat]
        table_high_bet = hero_current + replayed.amount_to_call_bb
        if replayed.amount_to_call_bb == 0:
            allowed_actions.add("BET")
        elif hero_all_in_target > table_high_bet:
            allowed_actions.add("RAISE")
    if not observed_actions <= allowed_actions:
        raise RemoteProtocolError(
            "state.legal_actions contradict the replayed public_hand"
        )
    required_passive = (
        {"FOLD", "CALL"}
        if replayed.amount_to_call_bb > 0
        else {"CHECK"}
    )
    if not required_passive <= observed_actions:
        raise RemoteProtocolError(
            "state.legal_actions omit a required passive control"
        )


def decision_state_to_wire(state: LiveDecisionState) -> dict[str, Any]:
    """Serialize a live state without lossy floating-point conversion."""

    if not isinstance(state, LiveDecisionState):
        raise RemoteProtocolError("state must be LiveDecisionState")
    wire = {
        "hand_id": _string(
            state.hand_id,
            "state.hand_id",
            max_bytes=MAX_HAND_ID_BYTES,
        ),
        "street": _string(state.street, "state.street", max_bytes=16).upper(),
        "board": [_card(card, "state.board card") for card in state.board],
        "hero_combo": [
            _card(card, "state.hero_combo card") for card in state.hero_combo
        ],
        "hero_position": _string(
            state.hero_position,
            "state.hero_position",
            max_bytes=16,
        ),
        "villain_position": _string(
            state.villain_position,
            "state.villain_position",
            max_bytes=16,
        ),
        "hero_is_oop": _boolean(state.hero_is_oop, "state.hero_is_oop"),
        "active_villains": _integer(
            state.active_villains,
            "state.active_villains",
        ),
        "pot_bb": _decimal_text(state.pot_bb),
        "hero_stack_bb": _decimal_text(state.hero_stack_bb),
        "villain_stack_bb": _decimal_text(state.villain_stack_bb),
        "hero_current_bet_bb": _decimal_text(state.hero_current_bet_bb),
        "villain_current_bet_bb": _decimal_text(state.villain_current_bet_bb),
        "amount_to_call_bb": _decimal_text(state.amount_to_call_bb),
        "legal_actions": list(state.legal_actions),
        "street_root_confirmed": _boolean(
            state.street_root_confirmed,
            "state.street_root_confirmed",
        ),
        "action_history": list(state.action_history),
        "observed_bet_to_bb": _decimal_text(state.observed_bet_to_bb),
        "mapping_error": _string(state.mapping_error, "state.mapping_error"),
        "preflop_observation": _observation_to_wire(state.preflop_observation),
        "preflop_mapping_error": _string(
            state.preflop_mapping_error,
            "state.preflop_mapping_error",
        ),
        "public_hand": _public_hand_to_wire(state.public_hand),
    }
    # Apply the same schema checks in both directions so an invalid internal
    # object is rejected on the Mac before any network request is attempted.
    decision_state_from_wire(wire)
    return wire


def decision_state_from_wire(value: Any) -> LiveDecisionState:
    """Validate and reconstruct a live decision state from JSON values."""

    state = _object(value, "state")
    _exact_keys(state, _STATE_KEYS, "state")
    street = _string(state["street"], "state.street", max_bytes=16).upper()
    if street not in {"PREFLOP", "FLOP", "TURN", "RIVER"}:
        raise RemoteProtocolError("state.street is unsupported")
    board = tuple(
        _card(card, f"state.board[{index}]")
        for index, card in enumerate(_array(state["board"], "state.board"))
    )
    expected_board = {"PREFLOP": 0, "FLOP": 3, "TURN": 4, "RIVER": 5}[street]
    if len(board) != expected_board:
        raise RemoteProtocolError(
            f"state.board must contain {expected_board} cards for {street}"
        )
    hero_combo = tuple(
        _card(card, f"state.hero_combo[{index}]")
        for index, card in enumerate(
            _array(state["hero_combo"], "state.hero_combo")
        )
    )
    if len(hero_combo) != 2:
        raise RemoteProtocolError("state.hero_combo must contain two cards")
    if len(set((*board, *hero_combo))) != len(board) + 2:
        raise RemoteProtocolError("state board and Hero cards cannot repeat")
    active_villains = _integer(
        state["active_villains"],
        "state.active_villains",
    )
    if not 0 <= active_villains <= 5:
        raise RemoteProtocolError("state.active_villains must be between 0 and 5")
    legal_actions = _strings(state["legal_actions"], "state.legal_actions")
    action_history = _strings(
        state["action_history"],
        "state.action_history",
        max_items=2,
    )
    restored = LiveDecisionState(
        hand_id=_string(
            state["hand_id"],
            "state.hand_id",
            max_bytes=MAX_HAND_ID_BYTES,
        ),
        street=street,
        board=board,
        hero_combo=hero_combo,  # type: ignore[arg-type]
        hero_position=_string(
            state["hero_position"],
            "state.hero_position",
            max_bytes=16,
        ),
        villain_position=_string(
            state["villain_position"],
            "state.villain_position",
            max_bytes=16,
        ),
        hero_is_oop=_boolean(state["hero_is_oop"], "state.hero_is_oop"),
        active_villains=active_villains,
        pot_bb=_decimal(state["pot_bb"], "state.pot_bb"),
        hero_stack_bb=_decimal(
            state["hero_stack_bb"],
            "state.hero_stack_bb",
        ),
        villain_stack_bb=_decimal(
            state["villain_stack_bb"],
            "state.villain_stack_bb",
        ),
        hero_current_bet_bb=_decimal(
            state["hero_current_bet_bb"],
            "state.hero_current_bet_bb",
        ),
        villain_current_bet_bb=_decimal(
            state["villain_current_bet_bb"],
            "state.villain_current_bet_bb",
        ),
        amount_to_call_bb=_decimal(
            state["amount_to_call_bb"],
            "state.amount_to_call_bb",
        ),
        legal_actions=legal_actions,
        street_root_confirmed=_boolean(
            state["street_root_confirmed"],
            "state.street_root_confirmed",
        ),
        action_history=action_history,
        observed_bet_to_bb=_decimal(
            state["observed_bet_to_bb"],
            "state.observed_bet_to_bb",
        ),
        mapping_error=_string(state["mapping_error"], "state.mapping_error"),
        preflop_observation=_observation_from_wire(
            state["preflop_observation"]
        ),
        preflop_mapping_error=_string(
            state["preflop_mapping_error"],
            "state.preflop_mapping_error",
        ),
        public_hand=_public_hand_from_wire(state["public_hand"]),
    )
    _validate_public_hand_binding(restored)
    return restored


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
        raise RemoteProtocolError(f"value is not strict JSON: {error}") from error


def decision_fingerprint(state: LiveDecisionState) -> str:
    """Return a stable SHA-256 identity for the complete captured decision."""

    return hashlib.sha256(
        _canonical_json(decision_state_to_wire(state))
    ).hexdigest()


def build_evaluate_request(
    request_id: str,
    state: LiveDecisionState,
) -> dict[str, Any]:
    request_id = _string(
        request_id,
        "request_id",
        max_bytes=MAX_REQUEST_ID_BYTES,
    )
    if not request_id:
        raise RemoteProtocolError("request_id cannot be empty")
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "request_id": request_id,
        "state": decision_state_to_wire(state),
    }


def _loads_strict(payload: bytes | str) -> Any:
    if isinstance(payload, bytes):
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RemoteProtocolError("JSON payload must be UTF-8") from error
    elif isinstance(payload, str):
        text = payload
    else:
        raise RemoteProtocolError("JSON payload must be bytes or text")
    if not text.strip():
        raise RemoteProtocolError("JSON payload cannot be empty")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RemoteProtocolError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise RemoteProtocolError(f"non-finite JSON constant {value!r} is forbidden")

    try:
        return json.loads(
            text,
            parse_float=Decimal,
            parse_int=int,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except RemoteProtocolError:
        raise
    except (json.JSONDecodeError, InvalidOperation, ValueError) as error:
        raise RemoteProtocolError(f"payload is not strict JSON: {error}") from error


def parse_evaluate_request(
    payload: bytes | str,
) -> tuple[str, LiveDecisionState, str]:
    """Parse a request and return ``(request_id, state, fingerprint)``."""

    request = _object(_loads_strict(payload), "request")
    _exact_keys(request, _REQUEST_KEYS, "request")
    schema_version = _integer(request["schema_version"], "schema_version")
    if schema_version != PROTOCOL_SCHEMA_VERSION:
        raise RemoteProtocolError(
            f"unsupported schema_version {schema_version}; "
            f"expected {PROTOCOL_SCHEMA_VERSION}"
        )
    request_id = _string(
        request["request_id"],
        "request_id",
        max_bytes=MAX_REQUEST_ID_BYTES,
    )
    if not request_id:
        raise RemoteProtocolError("request_id cannot be empty")
    state = decision_state_from_wire(request["state"])
    return request_id, state, decision_fingerprint(state)


def outcome_to_wire(
    request_id: str,
    fingerprint: str,
    outcome: LiveGTOOutcome,
) -> dict[str, Any]:
    """Return the intentionally small, auditable remote response."""

    if not isinstance(outcome, LiveGTOOutcome):
        raise RemoteProtocolError("outcome must be LiveGTOOutcome")
    latency = float(outcome.latency_seconds)
    if not math.isfinite(latency) or latency < 0:
        raise RemoteProtocolError("outcome latency must be finite and non-negative")
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise RemoteProtocolError("decision fingerprint must be lowercase SHA-256")
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
            "latency_seconds": latency,
            "analysis": outcome.analysis,
            "source": outcome.source,
            "model": outcome.model,
            "cache_hit": outcome.cache_hit,
            "approximate": outcome.approximate,
            "spec_key": outcome.effective_spec_key,
        },
    }


def outcome_from_wire(
    payload: bytes | str | Mapping[str, Any],
    *,
    expected_request_id: str,
    expected_fingerprint: str,
) -> LiveGTOOutcome:
    """Strictly validate a response before the OCR process uses its advice."""

    value = _loads_strict(payload) if isinstance(payload, (bytes, str)) else payload
    response = _object(value, "response")
    _exact_keys(response, _RESPONSE_KEYS, "response")
    schema_version = _integer(response["schema_version"], "schema_version")
    if schema_version != PROTOCOL_SCHEMA_VERSION:
        raise RemoteProtocolError("response schema_version differs from the request")
    request_id = _string(
        response["request_id"],
        "request_id",
        max_bytes=MAX_REQUEST_ID_BYTES,
    )
    if request_id != expected_request_id:
        raise RemoteProtocolError("response request_id differs from the request")
    fingerprint = _string(
        response["decision_fingerprint"],
        "decision_fingerprint",
        max_bytes=64,
    )
    if fingerprint != expected_fingerprint:
        raise RemoteProtocolError(
            "response decision_fingerprint differs from the captured state"
        )
    outcome_value = _object(response["outcome"], "outcome")
    _exact_keys(outcome_value, _OUTCOME_KEYS, "outcome")
    raw_status = _string(outcome_value["status"], "outcome.status", max_bytes=32)
    try:
        status = LiveGTOStatus(raw_status)
    except ValueError as error:
        raise RemoteProtocolError(f"unsupported outcome.status {raw_status!r}") from error
    raw_latency = outcome_value["latency_seconds"]
    if isinstance(raw_latency, bool) or not isinstance(raw_latency, (int, Decimal)):
        raise RemoteProtocolError("outcome.latency_seconds must be a JSON number")
    latency = float(raw_latency)
    if not math.isfinite(latency) or latency < 0:
        raise RemoteProtocolError(
            "outcome.latency_seconds must be finite and non-negative"
        )
    return LiveGTOOutcome(
        status=status,
        reason=_string(outcome_value["reason"], "outcome.reason"),
        latency_seconds=latency,
        analysis=_string(outcome_value["analysis"], "outcome.analysis", max_bytes=1_000_000),
        source=_string(outcome_value["source"], "outcome.source"),
        model=_string(outcome_value["model"], "outcome.model"),
        cache_hit=_boolean(outcome_value["cache_hit"], "outcome.cache_hit"),
        approximate=_boolean(
            outcome_value["approximate"],
            "outcome.approximate",
        ),
        spec_key=_string(outcome_value["spec_key"], "outcome.spec_key", max_bytes=128),
    )


def encode_json(value: Any) -> bytes:
    """Encode a request or response as compact UTF-8 JSON."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RemoteProtocolError(f"value is not strict JSON: {error}") from error
