"""Canonical, replayable public NLHE hand history for remote solving.

The existing live router intentionally accepts only a small decision snapshot.
That is sufficient for a conservative heads-up subgame, but a solver that keeps
ranges conditioned from preflop through river needs every public action.

This module defines that lossless public boundary.  It contains no OCR code and
no solver code: the Mac is responsible for producing events, while the server
replays and validates them before any backend may consume the hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
from typing import Any, Mapping


STREETS: tuple[str, ...] = ("PREFLOP", "FLOP", "TURN", "RIVER")
ACTION_KINDS = frozenset(
    {"FOLD", "CHECK", "CALL", "BET_TO", "RAISE_TO", "ALL_IN_TO"}
)
DEAL_KINDS = frozenset({"DEAL_FLOP", "DEAL_TURN", "DEAL_RIVER"})
EVENT_KINDS = ACTION_KINDS | DEAL_KINDS
SIX_MAX_POSITIONS: tuple[str, ...] = ("UTG", "HJ", "CO", "BTN", "SB", "BB")

MAX_EVENTS = 512
MAX_HAND_ID_BYTES = 256
MAX_POSITION_BYTES = 16
MAX_ABS_BB = Decimal("10000000")

_CARD_RE = re.compile(r"^[2-9TJQKA][cdhs]$")
_POSITION_RE = re.compile(r"^[A-Z][A-Z0-9_-]{0,15}$")
_HISTORY_KEYS = frozenset(
    {
        "variant",
        "hand_id",
        "button_seat",
        "small_blind_bb",
        "big_blind_bb",
        "ante_bb",
        "rake_rate_pct",
        "rake_cap_bb",
        "seats",
        "events",
    }
)
_SEAT_KEYS = frozenset({"seat", "position", "starting_stack_bb"})
_EVENT_KEYS = frozenset(
    {"sequence", "kind", "street", "actor_seat", "amount_to_bb", "cards"}
)


class PublicHandHistoryError(ValueError):
    """A public hand transcript is malformed or violates NLHE rules."""


@dataclass(frozen=True, slots=True)
class HandSeat:
    """One occupied physical seat at the beginning of a hand."""

    seat: int
    position: str
    starting_stack_bb: Decimal


@dataclass(frozen=True, slots=True)
class HandEvent:
    """One public voluntary action or board-deal event.

    ``amount_to_bb`` is the actor's cumulative contribution on the current
    street after the action.  It is required for CALL/BET_TO/RAISE_TO/
    ALL_IN_TO and must be ``None`` for FOLD/CHECK/deal events.
    """

    sequence: int
    kind: str
    street: str
    actor_seat: int | None = None
    amount_to_bb: Decimal | None = None
    cards: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PublicHandHistory:
    """Complete public information needed to replay one NLHE hand."""

    hand_id: str
    button_seat: int
    small_blind_bb: Decimal
    big_blind_bb: Decimal
    ante_bb: Decimal
    rake_rate_pct: Decimal
    rake_cap_bb: Decimal
    seats: tuple[HandSeat, ...]
    events: tuple[HandEvent, ...] = ()
    variant: str = "NLHE"

    def replay(self) -> "ReplayedHandState":
        return replay_public_hand(self)


@dataclass(frozen=True, slots=True)
class ReplayedHandState:
    """Immutable public state after replaying all supplied events."""

    hand_id: str
    street: str
    board: tuple[str, ...]
    actor_seat: int | None
    pot_bb: Decimal
    amount_to_call_bb: Decimal
    minimum_raise_to_bb: Decimal | None
    stacks: tuple[tuple[int, Decimal], ...]
    street_contributions: tuple[tuple[int, Decimal], ...]
    total_contributions: tuple[tuple[int, Decimal], ...]
    folded: frozenset[int]
    all_in: frozenset[int]
    live_seats: frozenset[int]
    legal_actions: tuple[str, ...]
    round_closed: bool
    terminal: bool
    next_sequence: int

    @property
    def stack_map(self) -> dict[int, Decimal]:
        return dict(self.stacks)

    @property
    def street_contribution_map(self) -> dict[int, Decimal]:
        return dict(self.street_contributions)

    @property
    def total_contribution_map(self) -> dict[int, Decimal]:
        return dict(self.total_contributions)


def _decimal(value: object, field: str, *, allow_zero: bool = True) -> Decimal:
    if isinstance(value, bool) or isinstance(value, float):
        raise PublicHandHistoryError(
            f"{field} must be an exact decimal, never bool/float"
        )
    if isinstance(value, Decimal):
        result = value
    else:
        try:
            result = Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError) as error:
            raise PublicHandHistoryError(f"{field} is not a decimal") from error
    if not result.is_finite():
        raise PublicHandHistoryError(f"{field} must be finite")
    if result < 0 or (not allow_zero and result == 0):
        qualifier = "positive" if not allow_zero else "non-negative"
        raise PublicHandHistoryError(f"{field} must be {qualifier}")
    if result > MAX_ABS_BB:
        raise PublicHandHistoryError(f"{field} exceeds the safety bound")
    return Decimal(0) if result == 0 else result.normalize()


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PublicHandHistoryError(f"{field} must be an integer")
    return value


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
    raise PublicHandHistoryError(
        f"{field} schema mismatch: {'; '.join(details)}"
    )


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PublicHandHistoryError(f"{field} must be an object")
    return value


def _array(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise PublicHandHistoryError(f"{field} must be an array")
    return value


def _string(value: Any, field: str, *, max_bytes: int) -> str:
    if not isinstance(value, str):
        raise PublicHandHistoryError(f"{field} must be a string")
    if len(value.encode("utf-8")) > max_bytes:
        raise PublicHandHistoryError(f"{field} exceeds {max_bytes} UTF-8 bytes")
    return value


def _card(value: Any, field: str) -> str:
    card = _string(value, field, max_bytes=2)
    if not _CARD_RE.fullmatch(card):
        raise PublicHandHistoryError(
            f"{field} must use canonical notation such as 'As' or 'Td'"
        )
    return card


def _decimal_text(value: object, field: str, *, allow_zero: bool = True) -> str:
    normalized = _decimal(value, field, allow_zero=allow_zero)
    return format(normalized, "f")


class _Replay:
    def __init__(self, history: PublicHandHistory) -> None:
        self.history = history
        self._validate_header()
        self.ordered_seats = tuple(sorted(seat.seat for seat in history.seats))
        self.seat_by_id = {seat.seat: seat for seat in history.seats}
        self.remaining = {
            seat.seat: _decimal(
                seat.starting_stack_bb,
                f"seat {seat.seat} starting_stack_bb",
                allow_zero=False,
            )
            for seat in history.seats
        }
        self.street_contribution = {
            seat.seat: Decimal(0) for seat in history.seats
        }
        self.total_contribution = {
            seat.seat: Decimal(0) for seat in history.seats
        }
        self.folded: set[int] = set()
        self.all_in: set[int] = set()
        self.board: list[str] = []
        self.street = "PREFLOP"
        self.highest = Decimal(0)
        self.last_full_raise = self.big_blind
        self.acted_since_full_raise: set[int] = set()
        self.pending: list[int] = []
        self.round_closed = False
        self.terminal = False

        self.small_blind_seat, self.big_blind_seat = self._blind_seats()
        self._post_forced_bets()
        self.pending = self._ordered_eligible_after(self.big_blind_seat)
        self._normalize_round()

    def _validate_header(self) -> None:
        history = self.history
        if history.variant != "NLHE":
            raise PublicHandHistoryError("variant must be NLHE")
        if (
            not isinstance(history.hand_id, str)
            or not history.hand_id
            or len(history.hand_id.encode("utf-8")) > MAX_HAND_ID_BYTES
        ):
            raise PublicHandHistoryError(
                f"hand_id must contain 1..{MAX_HAND_ID_BYTES} UTF-8 bytes"
            )
        if not isinstance(history.seats, tuple) or not 2 <= len(history.seats) <= 6:
            raise PublicHandHistoryError("a hand must contain 2..6 occupied seats")
        seat_ids: set[int] = set()
        positions: set[str] = set()
        for index, seat in enumerate(history.seats):
            if not isinstance(seat, HandSeat):
                raise PublicHandHistoryError(f"seats[{index}] must be HandSeat")
            seat_id = _integer(seat.seat, f"seats[{index}].seat")
            if not 0 <= seat_id <= 9 or seat_id in seat_ids:
                raise PublicHandHistoryError("seat IDs must be unique integers 0..9")
            position = seat.position
            if (
                not isinstance(position, str)
                or not _POSITION_RE.fullmatch(position)
                or position in positions
            ):
                raise PublicHandHistoryError(
                    "positions must be unique canonical uppercase labels"
                )
            _decimal(
                seat.starting_stack_bb,
                f"seats[{index}].starting_stack_bb",
                allow_zero=False,
            )
            seat_ids.add(seat_id)
            positions.add(position)
        button = _integer(history.button_seat, "button_seat")
        if button not in seat_ids:
            raise PublicHandHistoryError("button_seat is not occupied")
        if len(history.seats) == 6:
            if positions != set(SIX_MAX_POSITIONS):
                raise PublicHandHistoryError(
                    "a six-max hand requires UTG/HJ/CO/BTN/SB/BB exactly once"
                )
            ordered = sorted(seat_ids)
            button_index = ordered.index(button)
            clockwise = ordered[button_index:] + ordered[:button_index]
            actual = {
                seat.seat: seat.position
                for seat in history.seats
            }
            expected = dict(zip(clockwise, ("BTN", "SB", "BB", "UTG", "HJ", "CO")))
            if actual != expected:
                raise PublicHandHistoryError(
                    "six-max positions disagree with button/physical seat order"
                )
        self.small_blind = _decimal(
            history.small_blind_bb,
            "small_blind_bb",
            allow_zero=False,
        )
        self.big_blind = _decimal(
            history.big_blind_bb,
            "big_blind_bb",
            allow_zero=False,
        )
        if self.small_blind > self.big_blind:
            raise PublicHandHistoryError(
                "small_blind_bb cannot exceed big_blind_bb"
            )
        self.ante = _decimal(history.ante_bb, "ante_bb")
        self.rake_rate = _decimal(history.rake_rate_pct, "rake_rate_pct")
        if self.rake_rate > 100:
            raise PublicHandHistoryError("rake_rate_pct cannot exceed 100")
        self.rake_cap = _decimal(history.rake_cap_bb, "rake_cap_bb")
        if not isinstance(history.events, tuple) or len(history.events) > MAX_EVENTS:
            raise PublicHandHistoryError(
                f"events must be a tuple with at most {MAX_EVENTS} items"
            )

    def _next_seat(self, seat: int) -> int:
        index = self.ordered_seats.index(seat)
        return self.ordered_seats[(index + 1) % len(self.ordered_seats)]

    def _blind_seats(self) -> tuple[int, int]:
        if len(self.ordered_seats) == 2:
            small = self.history.button_seat
            return small, self._next_seat(small)
        small = self._next_seat(self.history.button_seat)
        return small, self._next_seat(small)

    def _ordered_after(self, seat: int, candidates: set[int]) -> list[int]:
        result: list[int] = []
        current = self._next_seat(seat)
        while current != seat:
            if current in candidates:
                result.append(current)
            current = self._next_seat(current)
        if seat in candidates:
            result.append(seat)
        return result

    def _eligible(self) -> set[int]:
        return (
            set(self.ordered_seats)
            - self.folded
            - self.all_in
        )

    def _live(self) -> set[int]:
        return set(self.ordered_seats) - self.folded

    def _ordered_eligible_after(self, seat: int) -> list[int]:
        return self._ordered_after(seat, self._eligible())

    def _commit_delta(
        self,
        seat: int,
        amount: Decimal,
        *,
        label: str,
        live_on_street: bool = True,
    ) -> None:
        if amount < 0:
            raise PublicHandHistoryError(f"{label} cannot be negative")
        paid = min(amount, self.remaining[seat])
        self.remaining[seat] -= paid
        if live_on_street:
            self.street_contribution[seat] += paid
        self.total_contribution[seat] += paid
        if self.remaining[seat] == 0:
            self.all_in.add(seat)

    def _post_forced_bets(self) -> None:
        if self.ante:
            for seat in self.ordered_seats:
                # Antes are dead money.  They reduce stacks and increase the
                # pot, but are not live chips toward a preflop call/raise.
                self._commit_delta(
                    seat,
                    self.ante,
                    label="ante",
                    live_on_street=False,
                )
        self._commit_delta(
            self.small_blind_seat,
            self.small_blind,
            label="small blind",
        )
        self._commit_delta(
            self.big_blind_seat,
            self.big_blind,
            label="big blind",
        )
        self.highest = max(self.street_contribution.values())

    def _normalize_round(self) -> None:
        live = self._live()
        if len(live) <= 1:
            self.pending = []
            self.round_closed = True
            self.terminal = True
            return
        eligible = self._eligible()
        self.pending = [seat for seat in self.pending if seat in eligible]
        if not eligible:
            self.pending = []
            self.round_closed = True
            return
        if len(eligible) == 1:
            only = next(iter(eligible))
            if self.street_contribution[only] >= self.highest:
                self.pending = []
                self.round_closed = True
                return
        if not self.pending:
            self.round_closed = True

    @property
    def actor(self) -> int | None:
        return None if self.round_closed or not self.pending else self.pending[0]

    def _require_action_shape(self, event: HandEvent) -> int:
        if self.terminal:
            raise PublicHandHistoryError(
                f"event {event.sequence} appears after the hand ended"
            )
        if self.round_closed:
            raise PublicHandHistoryError(
                f"event {event.sequence} is an action after the betting round closed"
            )
        if event.street != self.street:
            raise PublicHandHistoryError(
                f"event {event.sequence} street is {event.street}, expected {self.street}"
            )
        if event.cards:
            raise PublicHandHistoryError("action events cannot contain board cards")
        if event.actor_seat != self.actor:
            raise PublicHandHistoryError(
                f"event {event.sequence} actor is {event.actor_seat}, "
                f"expected {self.actor}"
            )
        assert event.actor_seat is not None
        return event.actor_seat

    def _amount_to(self, event: HandEvent) -> Decimal:
        if event.amount_to_bb is None:
            raise PublicHandHistoryError(
                f"{event.kind} requires amount_to_bb"
            )
        return _decimal(
            event.amount_to_bb,
            f"events[{event.sequence}].amount_to_bb",
        )

    def _passive_complete(self, actor: int) -> None:
        self.acted_since_full_raise.add(actor)
        if not self.pending or self.pending[0] != actor:
            raise AssertionError("pending actor invariant failed")
        self.pending.pop(0)
        self._normalize_round()

    def _full_aggression(self, actor: int, raise_size: Decimal) -> None:
        self.last_full_raise = raise_size
        self.acted_since_full_raise = {actor}
        self.pending = self._ordered_after(actor, self._eligible() - {actor})
        self._normalize_round()

    def _short_all_in_aggression(self, actor: int) -> None:
        self.acted_since_full_raise.add(actor)
        candidates = {
            seat
            for seat in self._eligible()
            if seat != actor and self.street_contribution[seat] < self.highest
        }
        self.pending = self._ordered_after(actor, candidates)
        self._normalize_round()

    def _pay_to(self, actor: int, target: Decimal) -> None:
        current = self.street_contribution[actor]
        if target < current:
            raise PublicHandHistoryError(
                f"seat {actor} amount-to {target} is below its current {current}"
            )
        delta = target - current
        if delta > self.remaining[actor]:
            raise PublicHandHistoryError(
                f"seat {actor} amount-to {target} exceeds its stack"
            )
        self.remaining[actor] -= delta
        self.street_contribution[actor] = target
        self.total_contribution[actor] += delta
        if self.remaining[actor] == 0:
            self.all_in.add(actor)

    def _apply_action(self, event: HandEvent) -> None:
        actor = self._require_action_shape(event)
        current = self.street_contribution[actor]
        facing = self.highest - current
        if facing < 0:
            raise AssertionError("actor contribution exceeds table high bet")

        if event.kind in {"FOLD", "CHECK"}:
            if event.amount_to_bb is not None:
                raise PublicHandHistoryError(
                    f"{event.kind} must not contain amount_to_bb"
                )
            if event.kind == "FOLD":
                if facing == 0:
                    raise PublicHandHistoryError("cannot fold when checking is free")
                self.folded.add(actor)
            elif facing != 0:
                raise PublicHandHistoryError(
                    f"seat {actor} cannot check facing {facing} BB"
                )
            self._passive_complete(actor)
            return

        target = self._amount_to(event)
        max_target = current + self.remaining[actor]

        if event.kind == "CALL":
            if facing <= 0:
                raise PublicHandHistoryError("CALL requires a wager to call")
            expected = min(self.highest, max_target)
            if target != expected:
                raise PublicHandHistoryError(
                    f"CALL amount-to must be {expected}, got {target}"
                )
            self._pay_to(actor, target)
            self._passive_complete(actor)
            return

        if event.kind == "BET_TO":
            if self.highest != 0:
                raise PublicHandHistoryError("BET_TO requires an unopened street")
            if target <= current or target > max_target:
                raise PublicHandHistoryError("BET_TO target is outside the actor stack")
            if target - current < self.big_blind:
                raise PublicHandHistoryError(
                    "BET_TO must be at least one big blind; use ALL_IN_TO for a short shove"
                )
            self._pay_to(actor, target)
            self.highest = target
            self._full_aggression(actor, target)
            return

        if event.kind == "RAISE_TO":
            if self.highest <= 0:
                raise PublicHandHistoryError("RAISE_TO requires an existing wager")
            if actor in self.acted_since_full_raise:
                raise PublicHandHistoryError(
                    "betting was not reopened for this actor"
                )
            if target <= self.highest or target > max_target:
                raise PublicHandHistoryError(
                    "RAISE_TO must exceed the high bet without exceeding the stack"
                )
            raise_size = target - self.highest
            if raise_size < self.last_full_raise:
                raise PublicHandHistoryError(
                    f"minimum full raise increment is {self.last_full_raise} BB"
                )
            self._pay_to(actor, target)
            self.highest = target
            self._full_aggression(actor, raise_size)
            return

        if event.kind != "ALL_IN_TO":
            raise PublicHandHistoryError(f"unsupported action kind {event.kind}")
        if target != max_target:
            raise PublicHandHistoryError(
                f"ALL_IN_TO must use the actor's full {max_target} BB"
            )
        if target <= current:
            raise PublicHandHistoryError("actor is already all-in")
        if target <= self.highest:
            expected = min(self.highest, max_target)
            if target != expected:
                raise PublicHandHistoryError(
                    f"all-in call amount-to must be {expected}"
                )
            self._pay_to(actor, target)
            self._passive_complete(actor)
            return
        if actor in self.acted_since_full_raise:
            raise PublicHandHistoryError(
                "betting was not reopened for this all-in raise"
            )
        previous_highest = self.highest
        raise_size = target - previous_highest
        self._pay_to(actor, target)
        self.highest = target
        if raise_size >= self.last_full_raise:
            self._full_aggression(actor, raise_size)
        else:
            self._short_all_in_aggression(actor)

    def _apply_deal(self, event: HandEvent) -> None:
        if event.actor_seat is not None or event.amount_to_bb is not None:
            raise PublicHandHistoryError(
                "deal events cannot contain actor_seat or amount_to_bb"
            )
        if not self.round_closed:
            raise PublicHandHistoryError(
                f"{event.kind} appears before the betting round closed"
            )
        if self.terminal:
            raise PublicHandHistoryError(
                f"{event.kind} appears after the hand ended by folds"
            )
        expected = {
            "PREFLOP": ("DEAL_FLOP", "FLOP", 3),
            "FLOP": ("DEAL_TURN", "TURN", 1),
            "TURN": ("DEAL_RIVER", "RIVER", 1),
        }.get(self.street)
        if expected is None:
            raise PublicHandHistoryError("no board may be dealt after RIVER")
        expected_kind, next_street, card_count = expected
        if event.kind != expected_kind or event.street != next_street:
            raise PublicHandHistoryError(
                f"expected {expected_kind} for {next_street}"
            )
        if len(event.cards) != card_count:
            raise PublicHandHistoryError(
                f"{event.kind} requires {card_count} board card(s)"
            )
        cards = tuple(
            _card(card, f"events[{event.sequence}].cards")
            for card in event.cards
        )
        if len(set((*self.board, *cards))) != len(self.board) + len(cards):
            raise PublicHandHistoryError("board cards cannot repeat")

        self.board.extend(cards)
        self.street = next_street
        self.street_contribution = {
            seat: Decimal(0) for seat in self.ordered_seats
        }
        self.highest = Decimal(0)
        self.last_full_raise = self.big_blind
        self.acted_since_full_raise = set()
        self.round_closed = False
        self.pending = self._ordered_eligible_after(self.history.button_seat)
        self._normalize_round()

    def apply(self, event: HandEvent, expected_sequence: int) -> None:
        if not isinstance(event, HandEvent):
            raise PublicHandHistoryError(
                f"events[{expected_sequence}] must be HandEvent"
            )
        sequence = _integer(event.sequence, f"events[{expected_sequence}].sequence")
        if sequence != expected_sequence:
            raise PublicHandHistoryError(
                f"event sequence must be contiguous from zero; expected "
                f"{expected_sequence}, got {sequence}"
            )
        if event.kind not in EVENT_KINDS:
            raise PublicHandHistoryError(
                f"events[{sequence}].kind is unsupported"
            )
        if event.street not in STREETS:
            raise PublicHandHistoryError(
                f"events[{sequence}].street is unsupported"
            )
        if event.kind in ACTION_KINDS:
            self._apply_action(event)
        else:
            self._apply_deal(event)
        if self.street == "RIVER" and self.round_closed:
            self.terminal = True

    def _legal_actions(self) -> tuple[str, ...]:
        actor = self.actor
        if actor is None:
            return ()
        current = self.street_contribution[actor]
        facing = self.highest - current
        legal: list[str] = ["FOLD", "CALL"] if facing > 0 else ["CHECK"]
        max_target = current + self.remaining[actor]
        can_aggress = (
            actor not in self.acted_since_full_raise
            and max_target > self.highest
        )
        if can_aggress:
            if self.highest == 0:
                if max_target - current >= self.big_blind:
                    legal.append("BET_TO")
            elif max_target - self.highest >= self.last_full_raise:
                legal.append("RAISE_TO")
            legal.append("ALL_IN_TO")
        elif facing > 0 and self.remaining[actor] <= facing:
            legal.append("ALL_IN_TO")
        return tuple(legal)

    def freeze(self, next_sequence: int) -> ReplayedHandState:
        actor = self.actor
        amount_to_call = (
            Decimal(0)
            if actor is None
            else max(
                Decimal(0),
                self.highest - self.street_contribution[actor],
            )
        )
        minimum_raise_to = None
        if (
            actor is not None
            and actor not in self.acted_since_full_raise
            and self.street_contribution[actor] + self.remaining[actor]
            > self.highest
        ):
            candidate = (
                self.big_blind
                if self.highest == 0
                else self.highest + self.last_full_raise
            )
            if (
                self.street_contribution[actor] + self.remaining[actor]
                >= candidate
            ):
                minimum_raise_to = candidate
        return ReplayedHandState(
            hand_id=self.history.hand_id,
            street=self.street,
            board=tuple(self.board),
            actor_seat=actor,
            pot_bb=sum(self.total_contribution.values(), Decimal(0)).normalize(),
            amount_to_call_bb=amount_to_call.normalize(),
            minimum_raise_to_bb=(
                minimum_raise_to.normalize()
                if minimum_raise_to is not None
                else None
            ),
            stacks=tuple(
                (seat, self.remaining[seat].normalize())
                for seat in self.ordered_seats
            ),
            street_contributions=tuple(
                (seat, self.street_contribution[seat].normalize())
                for seat in self.ordered_seats
            ),
            total_contributions=tuple(
                (seat, self.total_contribution[seat].normalize())
                for seat in self.ordered_seats
            ),
            folded=frozenset(self.folded),
            all_in=frozenset(self.all_in),
            live_seats=frozenset(self._live()),
            legal_actions=self._legal_actions(),
            round_closed=self.round_closed,
            terminal=self.terminal,
            next_sequence=next_sequence,
        )


def replay_public_hand(history: PublicHandHistory) -> ReplayedHandState:
    """Validate and replay a transcript from forced bets to its current node."""

    if not isinstance(history, PublicHandHistory):
        raise PublicHandHistoryError("history must be PublicHandHistory")
    replay = _Replay(history)
    for index, event in enumerate(history.events):
        replay.apply(event, index)
    return replay.freeze(len(history.events))


def public_hand_to_wire(history: PublicHandHistory) -> dict[str, Any]:
    """Serialize and fully validate one public hand transcript."""

    replay_public_hand(history)
    return {
        "variant": history.variant,
        "hand_id": history.hand_id,
        "button_seat": history.button_seat,
        "small_blind_bb": _decimal_text(
            history.small_blind_bb,
            "small_blind_bb",
            allow_zero=False,
        ),
        "big_blind_bb": _decimal_text(
            history.big_blind_bb,
            "big_blind_bb",
            allow_zero=False,
        ),
        "ante_bb": _decimal_text(history.ante_bb, "ante_bb"),
        "rake_rate_pct": _decimal_text(
            history.rake_rate_pct,
            "rake_rate_pct",
        ),
        "rake_cap_bb": _decimal_text(history.rake_cap_bb, "rake_cap_bb"),
        "seats": [
            {
                "seat": seat.seat,
                "position": seat.position,
                "starting_stack_bb": _decimal_text(
                    seat.starting_stack_bb,
                    f"seat {seat.seat} starting_stack_bb",
                    allow_zero=False,
                ),
            }
            for seat in history.seats
        ],
        "events": [
            {
                "sequence": event.sequence,
                "kind": event.kind,
                "street": event.street,
                "actor_seat": event.actor_seat,
                "amount_to_bb": (
                    None
                    if event.amount_to_bb is None
                    else _decimal_text(
                        event.amount_to_bb,
                        f"events[{event.sequence}].amount_to_bb",
                    )
                ),
                "cards": list(event.cards),
            }
            for event in history.events
        ],
    }


def public_hand_from_wire(value: Any) -> PublicHandHistory:
    """Parse strict JSON values and replay them before returning."""

    raw = _object(value, "public_hand")
    _exact_keys(raw, _HISTORY_KEYS, "public_hand")
    seats_raw = _array(raw["seats"], "public_hand.seats")
    seats: list[HandSeat] = []
    for index, item in enumerate(seats_raw):
        seat_value = _object(item, f"public_hand.seats[{index}]")
        _exact_keys(seat_value, _SEAT_KEYS, f"public_hand.seats[{index}]")
        starting_stack_raw = seat_value["starting_stack_bb"]
        if not isinstance(starting_stack_raw, str):
            raise PublicHandHistoryError(
                f"public_hand.seats[{index}].starting_stack_bb "
                "must be a decimal JSON string"
            )
        seats.append(
            HandSeat(
                seat=_integer(
                    seat_value["seat"],
                    f"public_hand.seats[{index}].seat",
                ),
                position=_string(
                    seat_value["position"],
                    f"public_hand.seats[{index}].position",
                    max_bytes=MAX_POSITION_BYTES,
                ),
                starting_stack_bb=_decimal(
                    starting_stack_raw,
                    f"public_hand.seats[{index}].starting_stack_bb",
                    allow_zero=False,
                ),
            )
        )

    events_raw = _array(raw["events"], "public_hand.events")
    if len(events_raw) > MAX_EVENTS:
        raise PublicHandHistoryError(
            f"public_hand.events exceeds {MAX_EVENTS} items"
        )
    events: list[HandEvent] = []
    for index, item in enumerate(events_raw):
        event_value = _object(item, f"public_hand.events[{index}]")
        _exact_keys(event_value, _EVENT_KEYS, f"public_hand.events[{index}]")
        actor_raw = event_value["actor_seat"]
        actor = (
            None
            if actor_raw is None
            else _integer(actor_raw, f"public_hand.events[{index}].actor_seat")
        )
        amount_raw = event_value["amount_to_bb"]
        if amount_raw is None:
            amount = None
        else:
            if not isinstance(amount_raw, str):
                raise PublicHandHistoryError(
                    f"public_hand.events[{index}].amount_to_bb "
                    "must be a decimal JSON string or null"
                )
            amount = _decimal(
                amount_raw,
                f"public_hand.events[{index}].amount_to_bb",
            )
        cards = tuple(
            _card(card, f"public_hand.events[{index}].cards[{card_index}]")
            for card_index, card in enumerate(
                _array(
                    event_value["cards"],
                    f"public_hand.events[{index}].cards",
                )
            )
        )
        events.append(
            HandEvent(
                sequence=_integer(
                    event_value["sequence"],
                    f"public_hand.events[{index}].sequence",
                ),
                kind=_string(
                    event_value["kind"],
                    f"public_hand.events[{index}].kind",
                    max_bytes=16,
                ),
                street=_string(
                    event_value["street"],
                    f"public_hand.events[{index}].street",
                    max_bytes=16,
                ),
                actor_seat=actor,
                amount_to_bb=amount,
                cards=cards,
            )
        )

    decimal_fields = (
        ("small_blind_bb", False),
        ("big_blind_bb", False),
        ("ante_bb", True),
        ("rake_rate_pct", True),
        ("rake_cap_bb", True),
    )
    parsed_decimals: dict[str, Decimal] = {}
    for field, allow_zero in decimal_fields:
        raw_value = raw[field]
        if not isinstance(raw_value, str):
            raise PublicHandHistoryError(
                f"public_hand.{field} must be a decimal JSON string"
            )
        parsed_decimals[field] = _decimal(
            raw_value,
            f"public_hand.{field}",
            allow_zero=allow_zero,
        )

    history = PublicHandHistory(
        variant=_string(
            raw["variant"],
            "public_hand.variant",
            max_bytes=16,
        ),
        hand_id=_string(
            raw["hand_id"],
            "public_hand.hand_id",
            max_bytes=MAX_HAND_ID_BYTES,
        ),
        button_seat=_integer(raw["button_seat"], "public_hand.button_seat"),
        small_blind_bb=parsed_decimals["small_blind_bb"],
        big_blind_bb=parsed_decimals["big_blind_bb"],
        ante_bb=parsed_decimals["ante_bb"],
        rake_rate_pct=parsed_decimals["rake_rate_pct"],
        rake_cap_bb=parsed_decimals["rake_cap_bb"],
        seats=tuple(seats),
        events=tuple(events),
    )
    replay_public_hand(history)
    return history


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
        raise PublicHandHistoryError(
            f"public hand is not strict JSON: {error}"
        ) from error


def public_hand_fingerprint(history: PublicHandHistory) -> str:
    """Return a stable identity for the complete public hand path."""

    return hashlib.sha256(
        _canonical_json(public_hand_to_wire(history))
    ).hexdigest()


__all__ = [
    "ACTION_KINDS",
    "DEAL_KINDS",
    "EVENT_KINDS",
    "HandEvent",
    "HandSeat",
    "PublicHandHistory",
    "PublicHandHistoryError",
    "ReplayedHandState",
    "public_hand_fingerprint",
    "public_hand_from_wire",
    "public_hand_to_wire",
    "replay_public_hand",
]
