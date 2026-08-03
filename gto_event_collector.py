"""Fail-closed conversion from complete table observations to public events.

The recorder is deliberately independent from the GUI and OCR provider.  It
accepts snapshot-like objects, requires an initial preflop frame before any
voluntary action, and allows at most one action or one board deal between
observations.  If a frame was skipped, the transcript becomes unavailable
instead of fabricating an action path.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from enum import Enum
import os
import threading
from typing import Mapping

from gto_hand_history import (
    HandEvent,
    HandSeat,
    PublicHandHistory,
    PublicHandHistoryError,
    ReplayedHandState,
    replay_public_hand,
)


_OCCUPIED_STATUSES = frozenset({"ACTIVE", "FOLDED", "ALL_IN"})
_POSITION_ALIASES = {"MP": "HJ"}
_STREET_INDEX = {"PREFLOP": 0, "FLOP": 1, "TURN": 2, "RIVER": 3}


class PublicEventCollectionError(ValueError):
    """Observed frames cannot prove one contiguous public action path."""


class PublicEventProbeStatus(str, Enum):
    """Outcome of a non-mutating candidate observation."""

    REJECTED = "rejected"
    ANCHORED = "anchored"
    DUPLICATE = "duplicate"
    ADVANCED = "advanced"
    NEW_HAND_ANCHORED = "new_hand_anchored"


def _decimal(value: object, field: str) -> Decimal:
    if isinstance(value, bool):
        raise PublicEventCollectionError(f"{field} must be a decimal")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise PublicEventCollectionError(f"{field} must be a decimal") from error
    if not result.is_finite() or result < 0:
        raise PublicEventCollectionError(
            f"{field} must be finite and non-negative"
        )
    return Decimal(0) if result == 0 else result.normalize()


@dataclass(frozen=True, slots=True)
class PublicEventCollectorConfig:
    small_blind_bb: Decimal = Decimal("0.5")
    big_blind_bb: Decimal = Decimal("1")
    ante_bb: Decimal = Decimal(0)
    rake_rate_pct: Decimal = Decimal("5")
    rake_cap_bb: Decimal = Decimal("0.5")
    amount_tolerance_bb: Decimal = Decimal("0.05")

    def __post_init__(self) -> None:
        small = _decimal(self.small_blind_bb, "small_blind_bb")
        big = _decimal(self.big_blind_bb, "big_blind_bb")
        if small <= 0 or big <= 0 or small > big:
            raise PublicEventCollectionError(
                "blinds must be positive and small blind cannot exceed big blind"
            )
        _decimal(self.ante_bb, "ante_bb")
        rake = _decimal(self.rake_rate_pct, "rake_rate_pct")
        if rake > 100:
            raise PublicEventCollectionError("rake_rate_pct cannot exceed 100")
        _decimal(self.rake_cap_bb, "rake_cap_bb")
        tolerance = _decimal(
            self.amount_tolerance_bb,
            "amount_tolerance_bb",
        )
        if tolerance > Decimal("0.5"):
            raise PublicEventCollectionError(
                "amount_tolerance_bb cannot exceed 0.5 BB"
            )

    @classmethod
    def from_env(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> "PublicEventCollectorConfig":
        env = os.environ if environment is None else environment
        return cls(
            small_blind_bb=_decimal(
                env.get("GTO_TABLE_SMALL_BLIND_BB", "0.5"),
                "GTO_TABLE_SMALL_BLIND_BB",
            ),
            big_blind_bb=_decimal(
                env.get("GTO_TABLE_BIG_BLIND_BB", "1"),
                "GTO_TABLE_BIG_BLIND_BB",
            ),
            ante_bb=_decimal(
                env.get("GTO_TABLE_ANTE_BB", "0"),
                "GTO_TABLE_ANTE_BB",
            ),
            rake_rate_pct=_decimal(
                env.get("GTO_RAKE_RATE_PCT", "5"),
                "GTO_RAKE_RATE_PCT",
            ),
            rake_cap_bb=_decimal(
                env.get("GTO_RAKE_CAP_BB", "0.5"),
                "GTO_RAKE_CAP_BB",
            ),
            amount_tolerance_bb=_decimal(
                env.get("GTO_EVENT_AMOUNT_TOLERANCE_BB", "0.05"),
                "GTO_EVENT_AMOUNT_TOLERANCE_BB",
            ),
        )


@dataclass(frozen=True, slots=True)
class _PlayerObservation:
    seat: int
    position: str
    stack: Decimal
    current_bet: Decimal
    status: str
    visible_action: str
    is_dealer: bool


@dataclass(frozen=True, slots=True)
class _SnapshotObservation:
    hand_id: str
    street: str
    board: tuple[str, ...]
    pot: Decimal
    dealer_seat: int
    action_on_seat: int
    players: tuple[_PlayerObservation, ...]

    @property
    def player_map(self) -> dict[int, _PlayerObservation]:
        return {player.seat: player for player in self.players}


@dataclass(frozen=True, slots=True)
class PublicEventProbe:
    """A transactional recorder candidate produced without changing state.

    Accepted probes are tied to ``base_version``.  They can be committed only
    while the recorder remains at that exact version, which prevents a delayed
    decoder result from overwriting a newer accepted observation.
    """

    status: PublicEventProbeStatus
    base_version: int
    candidate_hand_id: str
    replaces_hand: bool
    history: PublicHandHistory | None = None
    error: str = ""
    base_event_count: int = 0
    _snapshot: _SnapshotObservation | None = None
    _owner_token: object | None = None

    @property
    def accepted(self) -> bool:
        return self.status is not PublicEventProbeStatus.REJECTED

    @property
    def events_added(self) -> int:
        if not self.accepted or self.history is None:
            return 0
        return max(0, len(self.history.events) - self.base_event_count)


def _attribute(value: object, name: str, context: str) -> object:
    try:
        return getattr(value, name)
    except AttributeError as error:
        raise PublicEventCollectionError(
            f"{context} is missing {name}"
        ) from error


def _canonical_position(value: object) -> str:
    position = str(value or "").strip().upper()
    return _POSITION_ALIASES.get(position, position)


def _canonical_action(value: object) -> str:
    text = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    if not text:
        return ""
    if "ALL" in text and "IN" in text:
        return "ALL_IN"
    for action in ("RAISE", "BET", "CALL", "CHECK", "FOLD"):
        if action in text:
            return action
    return ""


def _snapshot(snapshot: object) -> _SnapshotObservation:
    hand_id = str(_attribute(snapshot, "hand_id", "snapshot") or "").strip()
    if not hand_id:
        raise PublicEventCollectionError("snapshot.hand_id cannot be empty")
    meta = _attribute(snapshot, "meta_info", "snapshot")
    street = str(
        _attribute(meta, "current_street", "snapshot.meta_info")
    ).strip().upper()
    if street not in _STREET_INDEX:
        raise PublicEventCollectionError(f"unsupported street {street!r}")
    board_state = _attribute(snapshot, "board_state", "snapshot")
    raw_board = _attribute(
        board_state,
        "community_cards",
        "snapshot.board_state",
    )
    try:
        board = tuple(str(card) for card in raw_board)
    except TypeError as error:
        raise PublicEventCollectionError("board must be iterable") from error
    expected_board = {"PREFLOP": 0, "FLOP": 3, "TURN": 4, "RIVER": 5}[street]
    if len(board) != expected_board:
        raise PublicEventCollectionError(
            f"{street} snapshot must contain {expected_board} board cards"
        )
    pot = _decimal(
        _attribute(board_state, "total_pot", "snapshot.board_state"),
        "snapshot pot",
    )
    try:
        dealer = int(_attribute(snapshot, "dealer_seat_index", "snapshot"))
        action_on = int(_attribute(snapshot, "action_on_seat_index", "snapshot"))
    except (TypeError, ValueError) as error:
        raise PublicEventCollectionError(
            "dealer/action seat indices must be integers"
        ) from error

    raw_players = _attribute(snapshot, "players", "snapshot")
    players: list[_PlayerObservation] = []
    try:
        iterator = iter(raw_players)
    except TypeError as error:
        raise PublicEventCollectionError("snapshot.players must be iterable") from error
    for raw_player in iterator:
        status = str(
            _attribute(raw_player, "status", "player")
        ).strip().upper()
        if status in {"EMPTY", "SITTING_OUT"}:
            continue
        if status not in _OCCUPIED_STATUSES:
            raise PublicEventCollectionError(
                f"unsupported occupied status {status!r}"
            )
        try:
            seat = int(_attribute(raw_player, "seat_index", "player"))
        except (TypeError, ValueError) as error:
            raise PublicEventCollectionError(
                "player.seat_index must be an integer"
            ) from error
        players.append(
            _PlayerObservation(
                seat=seat,
                position=_canonical_position(
                    _attribute(raw_player, "name", "player")
                ),
                stack=_decimal(
                    _attribute(raw_player, "stack_size", "player"),
                    f"seat {seat} stack",
                ),
                current_bet=_decimal(
                    _attribute(raw_player, "current_bet", "player"),
                    f"seat {seat} current bet",
                ),
                status=status,
                visible_action=_canonical_action(
                    getattr(raw_player, "visible_action", "")
                ),
                is_dealer=bool(getattr(raw_player, "is_dealer", False)),
            )
        )
    if not 2 <= len(players) <= 6:
        raise PublicEventCollectionError(
            "six-max recording requires 2..6 occupied seats"
        )
    if len({player.seat for player in players}) != len(players):
        raise PublicEventCollectionError("occupied seat IDs must be unique")
    if len({player.position for player in players}) != len(players):
        raise PublicEventCollectionError("position labels must be unique")
    if dealer not in {player.seat for player in players}:
        raise PublicEventCollectionError("dealer seat is not occupied")
    dealer_flags = {player.seat for player in players if player.is_dealer}
    if dealer_flags and dealer_flags != {dealer}:
        raise PublicEventCollectionError(
            "dealer flag disagrees with dealer_seat_index"
        )
    return _SnapshotObservation(
        hand_id=hand_id,
        street=street,
        board=board,
        pot=pot,
        dealer_seat=dealer,
        action_on_seat=action_on,
        players=tuple(sorted(players, key=lambda player: player.seat)),
    )


def _close(left: Decimal, right: Decimal, tolerance: Decimal) -> bool:
    return abs(left - right) <= tolerance


class PublicHandEventRecorder:
    """Build one complete transcript from contiguous full-table observations."""

    def __init__(
        self,
        config: PublicEventCollectorConfig | None = None,
    ) -> None:
        self.config = config or PublicEventCollectorConfig.from_env()
        if not isinstance(self.config, PublicEventCollectorConfig):
            raise PublicEventCollectionError(
                "config must be PublicEventCollectorConfig"
            )
        self.history: PublicHandHistory | None = None
        self.last_snapshot: _SnapshotObservation | None = None
        self.error: str = ""
        self._version = 0
        self._probe_owner = object()
        self._lock = threading.RLock()

    @property
    def complete(self) -> bool:
        with self._lock:
            return self.history is not None and not self.error

    @property
    def version(self) -> int:
        """Return the monotonic state version used by transactional probes."""

        with self._lock:
            return self._version

    def reset(self) -> None:
        with self._lock:
            self.history = None
            self.last_snapshot = None
            self.error = ""
            self._version += 1

    def invalidate_gap(self, reason: str) -> None:
        """Make the current transcript unavailable after an explicit input gap."""

        detail = str(reason or "").strip()
        if not detail:
            raise PublicEventCollectionError("capture gap reason cannot be empty")
        with self._lock:
            self.error = f"capture gap: {detail}"
            self._version += 1

    def _initial_history(
        self,
        snapshot: _SnapshotObservation,
    ) -> PublicHandHistory:
        if snapshot.street != "PREFLOP" or snapshot.board:
            raise PublicEventCollectionError(
                "recording must start on a preflop frame before any voluntary action"
            )
        if any(player.visible_action for player in snapshot.players):
            raise PublicEventCollectionError(
                "first frame has a visible voluntary-action overlay and "
                "cannot prove an untouched forced-bet state"
            )
        seats = tuple(
            HandSeat(
                seat=player.seat,
                position=player.position,
                # The visible stack is behind the live blind.  Antes are dead
                # and therefore are not part of current_bet.
                starting_stack_bb=(
                    player.stack
                    + player.current_bet
                    + self.config.ante_bb
                ),
            )
            for player in snapshot.players
        )
        history = PublicHandHistory(
            hand_id=snapshot.hand_id,
            button_seat=snapshot.dealer_seat,
            small_blind_bb=self.config.small_blind_bb,
            big_blind_bb=self.config.big_blind_bb,
            ante_bb=self.config.ante_bb,
            rake_rate_pct=self.config.rake_rate_pct,
            rake_cap_bb=self.config.rake_cap_bb,
            seats=seats,
        )
        replayed = replay_public_hand(history)
        if not self._matches(replayed, snapshot):
            raise PublicEventCollectionError(
                "first frame is not the untouched forced-bet state; "
                "recording began after a voluntary action or OCR is inconsistent"
            )
        return history

    def _matches(
        self,
        replayed: ReplayedHandState,
        snapshot: _SnapshotObservation,
    ) -> bool:
        tolerance = self.config.amount_tolerance_bb
        if (
            replayed.hand_id != snapshot.hand_id
            or replayed.street != snapshot.street
            or replayed.board != snapshot.board
        ):
            return False
        observed = snapshot.player_map
        if set(observed) != set(replayed.stack_map):
            return False
        if snapshot.action_on_seat >= 0 and (
            replayed.actor_seat != snapshot.action_on_seat
        ):
            return False
        for seat, player in observed.items():
            if not _close(replayed.stack_map[seat], player.stack, tolerance):
                return False
            if not _close(
                replayed.street_contribution_map[seat],
                player.current_bet,
                tolerance,
            ):
                return False
            if seat in replayed.folded:
                if player.status != "FOLDED":
                    return False
            elif player.status == "FOLDED":
                return False
            if seat in replayed.all_in:
                if player.status != "ALL_IN":
                    return False
            elif player.status == "ALL_IN":
                return False
        if not _close(replayed.pot_bb, snapshot.pot, tolerance):
            return False
        return True

    def _new_explicit_action(
        self,
        replayed: ReplayedHandState,
        snapshot: _SnapshotObservation,
    ) -> bool:
        actor = replayed.actor_seat
        if actor is None:
            return False
        current = snapshot.player_map[actor].visible_action
        if not current:
            return False
        if self.last_snapshot is None:
            return True
        previous_player = self.last_snapshot.player_map.get(actor)
        return previous_player is None or previous_player.visible_action != current

    def _infer_action(
        self,
        replayed: ReplayedHandState,
        snapshot: _SnapshotObservation,
        *,
        street_advanced: bool,
    ) -> HandEvent:
        actor = replayed.actor_seat
        if actor is None:
            raise PublicEventCollectionError(
                "no actor is pending; expected a board deal"
            )
        player = snapshot.player_map.get(actor)
        if player is None:
            raise PublicEventCollectionError("expected actor disappeared")
        previous_stack = replayed.stack_map[actor]
        previous_target = replayed.street_contribution_map[actor]
        highest = max(
            replayed.street_contribution_map.values(),
            default=Decimal(0),
        )
        facing = highest - previous_target
        tolerance = self.config.amount_tolerance_bb

        if player.status == "FOLDED" or player.visible_action == "FOLD":
            return HandEvent(
                sequence=replayed.next_sequence,
                kind="FOLD",
                street=replayed.street,
                actor_seat=actor,
            )

        if street_advanced:
            paid = previous_stack - player.stack
            if paid < -tolerance:
                raise PublicEventCollectionError(
                    "actor stack increased across a street transition"
                )
            paid = max(Decimal(0), paid)
            target = previous_target + paid
        else:
            target = player.current_bet
            expected_stack = previous_stack - (target - previous_target)
            if target + tolerance < previous_target or not _close(
                expected_stack,
                player.stack,
                tolerance,
            ):
                raise PublicEventCollectionError(
                    "actor stack/current-bet delta is not one legal action"
                )

        paid_more = target > previous_target + tolerance
        all_in = (
            player.status == "ALL_IN"
            or player.stack <= tolerance
            or player.visible_action == "ALL_IN"
        )
        if paid_more:
            if all_in:
                kind = "ALL_IN_TO"
            elif target <= highest + tolerance:
                kind = "CALL"
                target = min(target, highest)
            elif highest == 0:
                kind = "BET_TO"
            else:
                kind = "RAISE_TO"
            return HandEvent(
                sequence=replayed.next_sequence,
                kind=kind,
                street=replayed.street,
                actor_seat=actor,
                amount_to_bb=target.normalize(),
            )

        action_moved = (
            snapshot.action_on_seat >= 0
            and snapshot.action_on_seat != actor
        )
        if facing <= tolerance and (
            player.visible_action in {"CHECK", "CALL"}
            or action_moved
        ):
            return HandEvent(
                sequence=replayed.next_sequence,
                kind="CHECK",
                street=replayed.street,
                actor_seat=actor,
            )
        raise PublicEventCollectionError(
            "frame does not prove the expected actor's fold/check/call/bet/raise"
        )

    @staticmethod
    def _deal_event(
        replayed: ReplayedHandState,
        snapshot: _SnapshotObservation,
    ) -> HandEvent:
        if not replayed.round_closed:
            raise PublicEventCollectionError(
                "board advanced before the prior betting round closed"
            )
        expected_next = {
            "PREFLOP": ("FLOP", "DEAL_FLOP", 3),
            "FLOP": ("TURN", "DEAL_TURN", 1),
            "TURN": ("RIVER", "DEAL_RIVER", 1),
        }.get(replayed.street)
        if expected_next is None:
            raise PublicEventCollectionError("board advanced after river")
        next_street, kind, new_count = expected_next
        if snapshot.street != next_street:
            raise PublicEventCollectionError(
                "observation skipped one or more streets"
            )
        if snapshot.board[: len(replayed.board)] != replayed.board:
            raise PublicEventCollectionError("previous board cards changed")
        new_cards = snapshot.board[len(replayed.board) :]
        if len(new_cards) != new_count:
            raise PublicEventCollectionError(
                f"{kind} requires exactly {new_count} new card(s)"
            )
        return HandEvent(
            sequence=replayed.next_sequence,
            kind=kind,
            street=next_street,
            cards=tuple(new_cards),
        )

    def _append(
        self,
        history: PublicHandHistory,
        event: HandEvent,
    ) -> tuple[PublicHandHistory, ReplayedHandState]:
        updated = replace(history, events=(*history.events, event))
        try:
            return updated, replay_public_hand(updated)
        except PublicHandHistoryError as error:
            raise PublicEventCollectionError(
                f"inferred event is illegal: {error}"
            ) from error

    def _advance(
        self,
        history: PublicHandHistory,
        snapshot: _SnapshotObservation,
    ) -> PublicHandHistory:
        replayed = replay_public_hand(history)
        current_index = _STREET_INDEX[replayed.street]
        observed_index = _STREET_INDEX[snapshot.street]
        if observed_index < current_index or observed_index > current_index + 1:
            raise PublicEventCollectionError(
                "observation rolled back or skipped a street"
            )
        street_advanced = observed_index == current_index + 1

        if replayed.round_closed:
            if not street_advanced:
                raise PublicEventCollectionError(
                    "betting round is closed but the next board was not observed"
                )
        else:
            event = self._infer_action(
                replayed,
                snapshot,
                street_advanced=street_advanced,
            )
            history, replayed = self._append(history, event)

        if street_advanced:
            deal = self._deal_event(replayed, snapshot)
            history, replayed = self._append(history, deal)

        if not self._matches(replayed, snapshot):
            raise PublicEventCollectionError(
                "one inferred transition does not reproduce the observation; "
                "an action frame was skipped or OCR is ambiguous"
            )
        return history

    def probe(self, raw_snapshot: object) -> PublicEventProbe:
        """Evaluate one candidate without mutating or poisoning recorder state."""

        with self._lock:
            base_version = self._version
            base_event_count = (
                len(self.history.events) if self.history is not None else 0
            )
            snapshot: _SnapshotObservation | None = None
            candidate_hand_id = ""
            replaces_hand = False
            try:
                snapshot = _snapshot(raw_snapshot)
                candidate_hand_id = snapshot.hand_id
                replaces_hand = bool(
                    self.history is not None
                    and snapshot.hand_id != self.history.hand_id
                )
                if self.error and not replaces_hand:
                    raise PublicEventCollectionError(self.error)

                if self.history is None or replaces_hand:
                    history = self._initial_history(snapshot)
                    return PublicEventProbe(
                        status=(
                            PublicEventProbeStatus.NEW_HAND_ANCHORED
                            if replaces_hand
                            else PublicEventProbeStatus.ANCHORED
                        ),
                        base_version=base_version,
                        candidate_hand_id=candidate_hand_id,
                        replaces_hand=replaces_hand,
                        history=history,
                        base_event_count=(
                            0 if replaces_hand else base_event_count
                        ),
                        _snapshot=snapshot,
                        _owner_token=self._probe_owner,
                    )

                replayed = replay_public_hand(self.history)
                if self._matches(
                    replayed,
                    snapshot,
                ) and not self._new_explicit_action(
                    replayed,
                    snapshot,
                ):
                    return PublicEventProbe(
                        status=PublicEventProbeStatus.DUPLICATE,
                        base_version=base_version,
                        candidate_hand_id=candidate_hand_id,
                        replaces_hand=False,
                        history=self.history,
                        base_event_count=base_event_count,
                        _snapshot=snapshot,
                        _owner_token=self._probe_owner,
                    )

                history = self._advance(self.history, snapshot)
                return PublicEventProbe(
                    status=PublicEventProbeStatus.ADVANCED,
                    base_version=base_version,
                    candidate_hand_id=candidate_hand_id,
                    replaces_hand=False,
                    history=history,
                    base_event_count=base_event_count,
                    _snapshot=snapshot,
                    _owner_token=self._probe_owner,
                )
            except (PublicEventCollectionError, PublicHandHistoryError) as error:
                return PublicEventProbe(
                    status=PublicEventProbeStatus.REJECTED,
                    base_version=base_version,
                    candidate_hand_id=candidate_hand_id,
                    replaces_hand=replaces_hand,
                    error=str(error),
                    base_event_count=base_event_count,
                    _snapshot=snapshot,
                    _owner_token=self._probe_owner,
                )

    def commit(self, probe: PublicEventProbe) -> PublicHandHistory:
        """Atomically accept a successful probe if its base state is current."""

        with self._lock:
            if not isinstance(probe, PublicEventProbe):
                raise PublicEventCollectionError(
                    "probe must be a PublicEventProbe"
                )
            if probe._owner_token is not self._probe_owner:
                raise PublicEventCollectionError(
                    "cannot commit observation probe from another recorder"
                )
            if (
                not probe.accepted
                or probe.history is None
                or probe._snapshot is None
            ):
                raise PublicEventCollectionError(
                    "cannot commit rejected observation: "
                    f"{probe.error or 'unknown error'}"
                )
            if probe.base_version != self._version:
                raise PublicEventCollectionError(
                    "cannot commit stale observation probe"
                )

            self.history = probe.history
            self.last_snapshot = probe._snapshot
            self.error = ""
            self._version += 1
            return self.history

    def observe(self, raw_snapshot: object) -> PublicHandHistory | None:
        """Consume one frame with the legacy fail-closed mutation semantics."""

        with self._lock:
            probe = self.probe(raw_snapshot)
            if not probe.accepted:
                # Preserve the original behavior: seeing a different hand
                # resets the old transcript before a malformed/late new anchor
                # fails.
                if probe.replaces_hand:
                    self.reset()
                self.error = probe.error
                self._version += 1
                return None
            return self.commit(probe)


__all__ = [
    "PublicEventCollectionError",
    "PublicEventCollectorConfig",
    "PublicEventProbe",
    "PublicEventProbeStatus",
    "PublicHandEventRecorder",
]
