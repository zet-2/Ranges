"""Strict adapters from live table snapshots to preflop blueprint observations.

This module intentionally does not import :mod:`poker_assistant`.  Importing that
module initializes API clients and GUI/hotkey state, while the reconstruction
below only needs a small, duck-typed subset of its snapshot objects.

Two observations are supported:

* :func:`current_decision` reads a six-handed preflop frame where action is on
  Hero.
* :func:`terminal_from_history` anchors reconstruction to the current hand,
  compares its latest usable preflop frame with its first postflop capture, and
  requires that capture to be a flop.  Chips already committed on the flop are
  subtracted so they are not mistaken for preflop contributions.

Anything that cannot be reconstructed unambiguously fails closed with
:class:`PreflopObservationError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Iterable, Mapping, Sequence


PROVIDER_POSITIONS: tuple[str, ...] = ("UTG", "HJ", "CO", "BTN", "SB", "BB")
POSITION_ALIASES: Mapping[str, str] = {"MP": "HJ"}
_OCCUPIED_STATUSES = frozenset({"ACTIVE", "FOLDED", "ALL_IN"})
_LIVE_STATUSES = frozenset({"ACTIVE", "ALL_IN"})


class PreflopObservationError(ValueError):
    """A live snapshot cannot be mapped to one unique blueprint observation."""


def canonical_position(value: object) -> str:
    """Return a PokerStudy six-max position, mapping the app's MP to HJ."""

    position = str(value).strip().upper()
    position = POSITION_ALIASES.get(position, position)
    if position not in PROVIDER_POSITIONS:
        raise PreflopObservationError(f"unknown six-max position: {value!r}")
    return position


def _decimal(value: object, *, field: str) -> Decimal:
    if isinstance(value, Decimal):
        result = value
    else:
        try:
            result = Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise PreflopObservationError(f"{field} is not a decimal: {value!r}") from exc
    if not result.is_finite():
        raise PreflopObservationError(f"{field} must be finite")
    return result


def _ordered_amounts(values: Mapping[str, Decimal]) -> tuple[tuple[str, Decimal], ...]:
    return tuple((position, values[position]) for position in PROVIDER_POSITIONS)


@dataclass(frozen=True)
class ObservationProvenance:
    """Where an observation came from, without retaining mutable snapshots."""

    source: str
    preflop_index: int | None = None
    flop_index: int | None = None
    hand_id: str = ""


@dataclass(frozen=True)
class ObservedPreflopState:
    """Immutable public state suitable for deterministic blueprint matching."""

    actor: str | None
    contributions: tuple[tuple[str, Decimal], ...]
    folded: frozenset[str]
    initial_stacks: tuple[tuple[str, Decimal], ...]
    terminal: bool
    provenance: ObservationProvenance
    all_in: frozenset[str] = frozenset()

    @property
    def source(self) -> str:
        return self.provenance.source

    @property
    def contribution_map(self) -> dict[str, Decimal]:
        return dict(self.contributions)

    @property
    def initial_stack_map(self) -> dict[str, Decimal]:
        return dict(self.initial_stacks)

    @property
    def live_positions(self) -> frozenset[str]:
        return frozenset(PROVIDER_POSITIONS) - self.folded

    def to_history_observation(self):
        """Return the resolver's matcher view without importing app code.

        ``preflop_history`` owns the betting-state machine and intentionally
        carries fewer fields.  The lazy import keeps this adapter independent
        while preserving every field relevant to node matching.  Rich live
        provenance remains available on this object.
        """

        from preflop_history import ObservedPreflopState as HistoryObservation

        return HistoryObservation.create(
            actor=self.actor,
            contributions=self.contribution_map,
            folded=self.folded,
            survivors=self.live_positions if self.terminal else None,
            all_in=self.all_in,
        )


@dataclass(frozen=True)
class _PlayerView:
    seat: int
    position: str
    stack: Decimal
    current_bet: Decimal
    status: str
    is_hero: bool


@dataclass(frozen=True)
class _Layout:
    dealer: int
    players: tuple[_PlayerView, ...]

    @property
    def by_position(self) -> dict[str, _PlayerView]:
        return {player.position: player for player in self.players}

    @property
    def seat_to_position(self) -> tuple[tuple[int, str], ...]:
        return tuple(sorted((player.seat, player.position) for player in self.players))


def _attribute(obj: object, name: str, *, context: str) -> object:
    try:
        return getattr(obj, name)
    except AttributeError as exc:
        raise PreflopObservationError(f"{context} is missing {name}") from exc


def _street(snapshot: object) -> str:
    meta = _attribute(snapshot, "meta_info", context="snapshot")
    value = _attribute(meta, "current_street", context="snapshot.meta_info")
    return str(value).strip().upper()


def _layout(snapshot: object) -> _Layout:
    raw_dealer = _attribute(snapshot, "dealer_seat_index", context="snapshot")
    try:
        dealer = int(raw_dealer)
    except (TypeError, ValueError) as exc:
        raise PreflopObservationError("dealer_seat_index must be an integer") from exc
    if dealer < 0:
        raise PreflopObservationError("dealer seat is unknown")

    raw_players = _attribute(snapshot, "players", context="snapshot")
    if isinstance(raw_players, (str, bytes)):
        raise PreflopObservationError("snapshot.players must be a sequence")

    players: list[_PlayerView] = []
    seen_seats: set[int] = set()
    seen_positions: set[str] = set()
    try:
        iterator = iter(raw_players)
    except TypeError as exc:
        raise PreflopObservationError("snapshot.players must be iterable") from exc

    for raw_player in iterator:
        status = str(
            _attribute(raw_player, "status", context="player")
        ).strip().upper()
        # Empty and sitting-out seats are not part of a six-handed blueprint.
        if status in {"EMPTY", "SITTING_OUT"}:
            continue
        if status not in _OCCUPIED_STATUSES:
            raise PreflopObservationError(f"unsupported occupied player status: {status!r}")

        raw_seat = _attribute(raw_player, "seat_index", context="player")
        try:
            seat = int(raw_seat)
        except (TypeError, ValueError) as exc:
            raise PreflopObservationError("player.seat_index must be an integer") from exc
        if seat < 0 or seat in seen_seats:
            raise PreflopObservationError(f"duplicate or invalid occupied seat: {seat}")

        position = canonical_position(_attribute(raw_player, "name", context="player"))
        if position in seen_positions:
            raise PreflopObservationError(
                f"duplicate canonical position after aliases: {position}"
            )

        stack = _decimal(
            _attribute(raw_player, "stack_size", context="player"),
            field=f"{position} stack",
        )
        current_bet = _decimal(
            _attribute(raw_player, "current_bet", context="player"),
            field=f"{position} current bet",
        )
        if stack < 0 or current_bet < 0:
            raise PreflopObservationError(
                f"{position} stack and current bet must be nonnegative"
            )

        seen_seats.add(seat)
        seen_positions.add(position)
        players.append(
            _PlayerView(
                seat=seat,
                position=position,
                stack=stack,
                current_bet=current_bet,
                status=status,
                is_hero=bool(_attribute(raw_player, "is_hero", context="player")),
            )
        )

    if len(players) != 6 or seen_positions != set(PROVIDER_POSITIONS):
        raise PreflopObservationError(
            "a blueprint observation requires six unique occupied positions"
        )
    if dealer not in seen_seats:
        raise PreflopObservationError("dealer seat is not occupied")

    # Position labels are vision-derived.  Validate them against the physical
    # dealer/seat order before they can select a solver node.
    ordered_seats = sorted(seen_seats)
    dealer_index = ordered_seats.index(dealer)
    action_order = ordered_seats[dealer_index:] + ordered_seats[:dealer_index]
    expected_by_seat = dict(
        zip(action_order, ("BTN", "SB", "BB", "UTG", "HJ", "CO"))
    )
    actual_by_seat = {player.seat: player.position for player in players}
    if actual_by_seat != expected_by_seat:
        raise PreflopObservationError(
            "position labels disagree with the dealer/seat order"
        )
    return _Layout(dealer=dealer, players=tuple(players))


def _snapshot_hand_id(snapshot: object) -> str:
    value = getattr(snapshot, "hand_id", "")
    return "" if value is None else str(value)


def current_decision(snapshot: object) -> ObservedPreflopState:
    """Build the visible preflop state when action is confirmed on Hero."""

    if _street(snapshot) != "PREFLOP":
        raise PreflopObservationError("current decision snapshot must be PREFLOP")
    layout = _layout(snapshot)
    heroes = [player for player in layout.players if player.is_hero]
    if len(heroes) != 1:
        raise PreflopObservationError("snapshot must contain exactly one Hero")
    hero = heroes[0]

    raw_actor = _attribute(snapshot, "action_on_seat_index", context="snapshot")
    try:
        actor_seat = int(raw_actor)
    except (TypeError, ValueError) as exc:
        raise PreflopObservationError("action_on_seat_index must be an integer") from exc
    if actor_seat != hero.seat:
        raise PreflopObservationError("action is not confirmed on Hero")

    contributions = {
        player.position: player.current_bet for player in layout.players
    }
    initial_stacks = {
        player.position: player.stack + player.current_bet for player in layout.players
    }
    folded = frozenset(
        player.position for player in layout.players if player.status == "FOLDED"
    )
    all_in = frozenset(
        player.position for player in layout.players if player.status == "ALL_IN"
    )
    return ObservedPreflopState(
        actor=hero.position,
        contributions=_ordered_amounts(contributions),
        folded=folded,
        initial_stacks=_ordered_amounts(initial_stacks),
        terminal=False,
        provenance=ObservationProvenance(
            source="current_preflop_decision",
            hand_id=_snapshot_hand_id(snapshot),
        ),
        all_in=all_in,
    )


def _history_snapshots(history: object) -> list[object]:
    if history is None:
        return []
    if hasattr(history, "snapshots"):
        raw = getattr(history, "snapshots")
    elif hasattr(history, "meta_info"):
        raw = [history]
    else:
        raw = history
    if isinstance(raw, (str, bytes)):
        raise PreflopObservationError("history must contain snapshot objects")
    try:
        return list(raw)
    except TypeError as exc:
        raise PreflopObservationError("history must be iterable") from exc


def _latest_valid_preflop(
    snapshots: Sequence[object],
    *,
    anchor_index: int,
    hand_id: str,
) -> tuple[int, object, _Layout]:
    """Find the nearest usable preflop frame for the anchored current hand."""

    saw_preflop = False
    failures: list[str] = []
    for index in range(anchor_index, -1, -1):
        snapshot = snapshots[index]
        if _snapshot_hand_id(snapshot) != hand_id:
            continue
        if _street(snapshot) != "PREFLOP":
            continue
        saw_preflop = True
        try:
            return index, snapshot, _layout(snapshot)
        except PreflopObservationError as exc:
            failures.append(str(exc))
    if saw_preflop:
        detail = failures[-1] if failures else "invalid snapshot"
        raise PreflopObservationError(
            f"no valid PREFLOP snapshot for current hand {hand_id!r}: {detail}"
        )
    raise PreflopObservationError(
        f"history has no PREFLOP snapshot for current hand {hand_id!r}"
    )


def _first_flop_after(
    snapshots: Sequence[object],
    preflop_index: int,
    *,
    anchor_index: int,
    hand_id: str,
) -> tuple[int, object]:
    for index in range(preflop_index + 1, anchor_index + 1):
        snapshot = snapshots[index]
        if _snapshot_hand_id(snapshot) != hand_id:
            continue
        street = _street(snapshot)
        if street == "PREFLOP":
            continue
        if street == "FLOP":
            return index, snapshot
        if street in {"TURN", "RIVER"}:
            raise PreflopObservationError(
                f"first postflop capture is {street}; preflop investment is unknowable"
            )
        raise PreflopObservationError(f"unsupported street in history: {street!r}")
    raise PreflopObservationError(
        f"current hand {hand_id!r} has no FLOP snapshot after PREFLOP"
    )


def terminal_from_history(
    history: object,
    current_snapshot: object | None = None,
    *,
    require_heads_up: bool = True,
) -> ObservedPreflopState:
    """Infer terminal preflop contributions for the anchored current hand.

    ``history`` may be a ``HandHistory``-like object exposing ``snapshots`` or
    any iterable of snapshots.  ``current_snapshot`` is appended for the common
    live path where history contains earlier frames but not the frame currently
    being evaluated.  Its hand ID (or the final history snapshot's hand ID when
    it is omitted) is authoritative: frames from any other hand are never used.
    """

    snapshots = _history_snapshots(history)
    if current_snapshot is not None and (
        not snapshots or snapshots[-1] is not current_snapshot
    ):
        snapshots.append(current_snapshot)

    if not snapshots:
        raise PreflopObservationError("history has no current snapshot")
    anchor_index = len(snapshots) - 1
    anchored_snapshot = snapshots[anchor_index]
    current_hand_id = _snapshot_hand_id(anchored_snapshot)
    if not current_hand_id:
        raise PreflopObservationError(
            "current snapshot must have a non-empty hand_id"
        )

    pre_index, pre_snapshot, pre_layout = _latest_valid_preflop(
        snapshots,
        anchor_index=anchor_index,
        hand_id=current_hand_id,
    )
    flop_index, flop_snapshot = _first_flop_after(
        snapshots,
        pre_index,
        anchor_index=anchor_index,
        hand_id=current_hand_id,
    )
    flop_layout = _layout(flop_snapshot)

    if pre_layout.dealer != flop_layout.dealer:
        raise PreflopObservationError("dealer changed between PREFLOP and FLOP")
    if pre_layout.seat_to_position != flop_layout.seat_to_position:
        raise PreflopObservationError(
            "seat-to-position mapping changed between PREFLOP and FLOP"
        )
    pre_by_position = pre_layout.by_position
    flop_by_position = flop_layout.by_position
    initial_stacks: dict[str, Decimal] = {}
    contributions: dict[str, Decimal] = {}
    folded: set[str] = set()
    live: set[str] = set()
    all_in: set[str] = set()

    for position in PROVIDER_POSITIONS:
        pre_player = pre_by_position[position]
        flop_player = flop_by_position[position]
        initial = pre_player.stack + pre_player.current_bet
        # A flop bet is still part of the player's displayed total and must not
        # be counted as money invested preflop.
        remaining_at_flop = flop_player.stack + flop_player.current_bet
        contribution = initial - remaining_at_flop
        if not initial.is_finite() or not contribution.is_finite():
            raise PreflopObservationError(
                f"non-finite inferred contribution for {position}"
            )
        if initial < 0 or contribution < 0:
            raise PreflopObservationError(
                f"negative inferred preflop contribution for {position}"
            )
        initial_stacks[position] = initial
        contributions[position] = contribution

        # If no chips remain after subtracting a flop wager, the player was
        # already all-in at the preflop terminal node.  A player who shoved on
        # the flop still has their entire post-preflop stack in current_bet and
        # therefore is not misclassified here.
        if flop_player.status == "ALL_IN" and remaining_at_flop == 0 and initial > 0:
            all_in.add(position)

    highest_contribution = max(contributions.values(), default=Decimal(0))
    contribution_tolerance = Decimal("0.05")
    for position in PROVIDER_POSITIONS:
        flop_player = flop_by_position[position]
        contribution = contributions[position]
        if flop_player.status == "FOLDED":
            # A genuine preflop fold must have invested less than the final
            # wager.  If it invested the full amount, it saw the flop and its
            # visible fold happened postflop; retain it as a preflop survivor so
            # a false HU handoff cannot be manufactured.
            if contribution + contribution_tolerance < highest_contribution:
                folded.add(position)
            else:
                live.add(position)
        elif flop_player.status in _LIVE_STATUSES:
            if (
                flop_player.status != "ALL_IN"
                and contribution + contribution_tolerance < highest_contribution
            ):
                raise PreflopObservationError(
                    f"active {position} did not match the terminal preflop wager"
                )
            live.add(position)
        else:  # _layout already rejects this; retain the invariant explicitly.
            raise PreflopObservationError(
                f"unsupported FLOP status for {position}: {flop_player.status!r}"
            )

    if live & folded or live | folded != set(PROVIDER_POSITIONS):
        raise PreflopObservationError("FLOP statuses do not partition the table")
    if require_heads_up and len(live) != 2:
        raise PreflopObservationError(
            f"postflop hand is not heads-up: found {len(live)} live players"
        )

    return ObservedPreflopState(
        actor=None,
        contributions=_ordered_amounts(contributions),
        folded=frozenset(folded),
        initial_stacks=_ordered_amounts(initial_stacks),
        terminal=True,
        provenance=ObservationProvenance(
            source="preflop_to_flop_transition",
            preflop_index=pre_index,
            flop_index=flop_index,
            hand_id=current_hand_id,
        ),
        all_in=frozenset(all_in),
    )


__all__ = [
    "ObservedPreflopState",
    "ObservationProvenance",
    "POSITION_ALIASES",
    "PROVIDER_POSITIONS",
    "PreflopObservationError",
    "canonical_position",
    "current_decision",
    "terminal_from_history",
]
