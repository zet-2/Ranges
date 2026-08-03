"""Fail-closed semantic decoder for continuous public-hand capture.

The decoder deliberately knows nothing about Gemini, credentials, or solver
backends.  A caller injects exactly one bounded frame-analysis callable and a
snapshot validator.  The surrounding :class:`PublicHistoryWorker` supplies the
hard deadline and turns decoder gaps into coordinator invalidations.

Only locally stable keyframe reasons are analyzed.  In particular, CHANGE,
PEAK, and FLUSH frames describe an in-progress visual transition and are
acknowledged as transient without spending a provider request.

Hand identity is session-local because the current vision schema has no table
hand number.  A new identity is issued only when all of these are visible:

* the dealer moved;
* a previous postflop or terminal state regressed to PREFLOP;
* the board cleared; and
* the current frame is an untouched 0.5/1 BB blind anchor.

Anything that resembles a hand boundary without proving the full conjunction
is a GAP.  Hero hole cards are intentionally not part of the boundary proof:
the same combination can be dealt again.

The schema also does not expose a general action-on seat.  This decoder never
invents one and never fabricates a villain CHECK.  The transactional public
event recorder therefore rejects any transition that cannot be proved from a
visible action overlay, contribution delta, or other explicit evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
import math
import threading
import time
import uuid
from typing import Callable, Iterable, Mapping, Protocol

from live_capture import Keyframe, KeyframeReason
from public_history_pipeline import (
    AcceptedPublicHistoryFrame,
    PublicHistoryCoordinator,
)
from public_history_worker import DecodeResult, PublicHistoryWorker


class FrameAnalyzer(Protocol):
    """One provider-specific, bounded analysis call."""

    def __call__(
        self,
        regions: Mapping[str, object],
        *,
        timeout_seconds: float,
    ) -> object:
        """Return one mutable snapshot-like object or raise."""


class SnapshotValidator(Protocol):
    """Return human-readable reasons that make a snapshot unsafe."""

    def __call__(self, snapshot: object) -> Iterable[str]:
        """Return an empty iterable only for a structurally valid snapshot."""


_TRANSIENT_REASONS = frozenset(
    {
        KeyframeReason.CHANGE,
        KeyframeReason.PEAK,
        KeyframeReason.FLUSH,
    }
)
_OCCUPIED_STATUSES = frozenset({"ACTIVE", "FOLDED", "ALL_IN"})
_LIVE_STATUSES = frozenset({"ACTIVE", "ALL_IN"})
_ALL_STATUSES = _OCCUPIED_STATUSES | frozenset({"EMPTY", "SITTING_OUT"})
_BOARD_LENGTHS = {"PREFLOP": 0, "FLOP": 3, "TURN": 4, "RIVER": 5}
_AMOUNT_TOLERANCE_BB = 0.05


@dataclass(frozen=True, slots=True)
class _HandEvidence:
    dealer_seat: int
    street: str
    board: tuple[str, ...]
    hero_cards: tuple[str, ...]
    live_players: int
    untouched_blind_anchor: bool

    @property
    def terminal(self) -> bool:
        return self.live_players <= 1

    @property
    def postflop(self) -> bool:
        return self.street in {"FLOP", "TURN", "RIVER"} and bool(self.board)


@dataclass(frozen=True, slots=True)
class ContinuousPublicHistoryReadiness:
    """One accepted bundle proven current through worker and ring cursors."""

    ready: bool
    reason: str
    accepted_frame: AcceptedPublicHistoryFrame | None = None
    consumed_through_frame_id: int = 0
    acknowledged_through_frame_id: int = 0
    observed_gap_revision: int = 0
    current_gap_revision: int = 0


def _utc_capture_timestamp(keyframe: Keyframe) -> str:
    captured_at = keyframe.frame.captured_at.astimezone(timezone.utc)
    return captured_at.isoformat().replace("+00:00", "Z")


def _error_text(prefix: str, error: BaseException) -> str:
    detail = str(error).strip()
    suffix = f": {detail}" if detail else ""
    return f"{prefix}: {type(error).__name__}{suffix}"


def _players(snapshot: object) -> list[object]:
    raw_players = getattr(snapshot, "players", None)
    if raw_players is None:
        raise ValueError("snapshot is missing players")
    try:
        return list(raw_players)
    except TypeError as error:
        raise ValueError("snapshot.players must be iterable") from error


def _float_amount(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        amount = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(amount) or amount < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return amount


def _looks_like_untouched_blind_anchor(
    snapshot: object,
    *,
    street: str,
    board: tuple[str, ...],
    players: list[object],
) -> bool:
    """Recognize only the standard normalized 0.5/1 BB forced-bet root."""

    if street != "PREFLOP" or board:
        return False
    occupied = [
        player
        for player in players
        if str(getattr(player, "status", "") or "").strip().upper()
        in _OCCUPIED_STATUSES
    ]
    if not 2 <= len(occupied) <= 6:
        return False
    if any(
        str(getattr(player, "status", "") or "").strip().upper() == "FOLDED"
        for player in occupied
    ):
        return False
    if any(
        str(getattr(player, "visible_action", "") or "").strip()
        for player in occupied
    ):
        return False
    try:
        positive_bets = sorted(
            amount
            for player in occupied
            if (
                amount := _float_amount(
                    getattr(player, "current_bet", None),
                    "player.current_bet",
                )
            )
            > _AMOUNT_TOLERANCE_BB
        )
    except ValueError:
        return False
    if len(positive_bets) != 2:
        return False
    return (
        abs(positive_bets[0] - 0.5) <= _AMOUNT_TOLERANCE_BB
        and abs(positive_bets[1] - 1.0) <= _AMOUNT_TOLERANCE_BB
    )


def _hand_evidence(snapshot: object) -> _HandEvidence:
    meta = getattr(snapshot, "meta_info", None)
    board_state = getattr(snapshot, "board_state", None)
    if meta is None or board_state is None:
        raise ValueError("snapshot is missing meta_info or board_state")
    street = str(getattr(meta, "current_street", "") or "").strip().upper()
    if street not in _BOARD_LENGTHS:
        raise ValueError(f"unsupported snapshot street {street!r}")
    try:
        board = tuple(getattr(board_state, "community_cards"))
    except (AttributeError, TypeError) as error:
        raise ValueError("snapshot board must be iterable") from error
    if len(board) != _BOARD_LENGTHS[street]:
        raise ValueError(
            f"{street} snapshot must contain {_BOARD_LENGTHS[street]} board cards"
        )
    try:
        dealer = int(getattr(snapshot, "dealer_seat_index"))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("snapshot dealer seat must be an integer") from error
    players = _players(snapshot)
    normalized_players = []
    for player in players:
        status = str(
            getattr(player, "status", "") or ""
        ).strip().upper()
        if status not in _ALL_STATUSES:
            raise ValueError(f"unsupported player status {status!r}")
        try:
            seat = int(getattr(player, "seat_index"))
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("player seat must be an integer") from error
        if seat not in range(6):
            raise ValueError("player seat must be S0-S5")
        normalized_players.append((player, seat, status))
    all_seats = [seat for _, seat, _ in normalized_players]
    if len(all_seats) != len(set(all_seats)):
        raise ValueError("snapshot player seats must be unique")

    occupied = [
        (player, seat, status)
        for player, seat, status in normalized_players
        if status in _OCCUPIED_STATUSES
    ]
    if not 2 <= len(occupied) <= 6:
        raise ValueError("snapshot must prove 2..6 occupied seats")
    occupied_seats = {seat for _, seat, _ in occupied}
    if dealer not in occupied_seats:
        raise ValueError("snapshot does not prove an occupied dealer seat")
    positions = []
    for player, _, _ in occupied:
        position = str(getattr(player, "name", "") or "").strip().upper()
        position = "HJ" if position == "MP" else position
        if not position:
            raise ValueError("occupied player position cannot be empty")
        positions.append(position)
    if len(positions) != len(set(positions)):
        raise ValueError("occupied player positions must be unique")
    dealer_flags = {
        seat
        for player, seat, _ in occupied
        if bool(getattr(player, "is_dealer", False))
    }
    if dealer_flags and dealer_flags != {dealer}:
        raise ValueError("dealer flag disagrees with dealer seat")
    heroes = [
        player
        for player, _, _ in normalized_players
        if bool(getattr(player, "is_hero", False))
    ]
    if len(heroes) != 1:
        raise ValueError("snapshot must identify exactly one Hero")
    try:
        action_on = int(getattr(snapshot, "action_on_seat_index"))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("snapshot action-on seat must be an integer") from error
    if action_on != -1 and action_on not in occupied_seats:
        raise ValueError(
            "snapshot action-on seat must be occupied or unknown"
        )

    live_players = sum(
        1
        for _, _, status in normalized_players
        if status in _LIVE_STATUSES
    )
    if live_players < 1:
        raise ValueError("snapshot must retain at least one live player")
    hero = heroes[0]
    hero_cards = tuple(getattr(hero, "hole_cards", None) or ())
    return _HandEvidence(
        dealer_seat=dealer,
        street=street,
        board=tuple(str(card) for card in board),
        hero_cards=tuple(str(card) for card in hero_cards),
        live_players=live_players,
        untouched_blind_anchor=_looks_like_untouched_blind_anchor(
            snapshot,
            street=street,
            board=board,
            players=players,
        ),
    )


class ContinuousPublicHistoryDecoder:
    """Convert stable captured keyframes into validated public observations."""

    def __init__(
        self,
        *,
        analyze_frame: FrameAnalyzer,
        validate_snapshot: SnapshotValidator,
        hand_id_factory: Callable[[int], str] | None = None,
    ) -> None:
        if not callable(analyze_frame):
            raise TypeError("analyze_frame must be callable")
        if not callable(validate_snapshot):
            raise TypeError("validate_snapshot must be callable")
        if hand_id_factory is not None and not callable(hand_id_factory):
            raise TypeError("hand_id_factory must be callable")
        session_token = uuid.uuid4().hex
        self._analyze_frame = analyze_frame
        self._validate_snapshot = validate_snapshot
        self._hand_id_factory = hand_id_factory or (
            lambda ordinal: f"continuous-{session_token}-{ordinal}"
        )
        self._hand_ordinal = 0
        self._active_hand_id = ""
        self._previous_evidence: _HandEvidence | None = None
        self._lock = threading.RLock()

    @property
    def active_hand_id(self) -> str:
        with self._lock:
            return self._active_hand_id

    @property
    def hand_ordinal(self) -> int:
        with self._lock:
            return self._hand_ordinal

    def _boundary(
        self,
        current: _HandEvidence,
    ) -> tuple[bool, str]:
        previous = self._previous_evidence
        if previous is None:
            if not current.untouched_blind_anchor:
                return (
                    False,
                    "initial observation does not prove an untouched blind "
                    "anchor",
                )
            return True, ""

        dealer_moved = current.dealer_seat != previous.dealer_seat
        regressed_to_preflop = (
            current.street == "PREFLOP"
            and not current.board
            and (previous.postflop or previous.terminal)
        )
        proven_new_hand = (
            dealer_moved
            and regressed_to_preflop
            and current.untouched_blind_anchor
        )
        if proven_new_hand:
            return True, ""

        board_rolled_back = bool(previous.board) and not current.board
        hero_cards_changed = bool(
            len(previous.hero_cards) == 2
            and len(current.hero_cards) == 2
            and set(previous.hero_cards) != set(current.hero_cards)
        )
        resembles_boundary = (
            dealer_moved
            or board_rolled_back
            or regressed_to_preflop
            or (
                current.untouched_blind_anchor
                and not previous.untouched_blind_anchor
            )
        )
        if resembles_boundary:
            missing = []
            if not dealer_moved:
                missing.append("dealer did not move")
            if not regressed_to_preflop:
                missing.append(
                    "prior postflop/terminal to PREFLOP regression is unproved"
                )
            if not current.untouched_blind_anchor:
                missing.append("untouched blind anchor is unproved")
            return False, "ambiguous hand boundary: " + ", ".join(missing)
        if hero_cards_changed:
            return (
                False,
                "ambiguous hand identity: Hero cards changed without a "
                "corroborated new blind anchor",
            )
        return False, ""

    @staticmethod
    def _set_identity_and_timestamp(
        snapshot: object,
        *,
        hand_id: str,
        timestamp: str,
    ) -> None:
        if not hasattr(snapshot, "hand_id"):
            raise ValueError("snapshot is missing hand_id")
        if not hasattr(snapshot, "timestamp"):
            raise ValueError("snapshot is missing timestamp")
        try:
            setattr(snapshot, "hand_id", hand_id)
            setattr(snapshot, "timestamp", timestamp)
        except (AttributeError, TypeError) as error:
            raise ValueError(
                "snapshot hand_id/timestamp must be mutable"
            ) from error

    def _validation_errors(self, snapshot: object) -> tuple[str, ...]:
        try:
            raw_errors = self._validate_snapshot(snapshot)
        except Exception as error:
            return (_error_text("snapshot validator failed", error),)
        if raw_errors is None:
            return ("snapshot validator returned no result",)
        if isinstance(raw_errors, str):
            raw_errors = (raw_errors,)
        try:
            return tuple(
                detail
                for error in raw_errors
                if (detail := str(error or "").strip())
            )
        except TypeError as error:
            return (_error_text("snapshot validator returned invalid errors", error),)

    def decode(
        self,
        keyframe: Keyframe,
        *,
        timeout_seconds: float,
    ) -> DecodeResult:
        """Perform at most one provider call and return one explicit outcome."""

        if not isinstance(keyframe, Keyframe):
            raise TypeError("keyframe must be a Keyframe")
        if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
            raise ValueError("timeout_seconds must be finite and positive")
        if keyframe.reason in _TRANSIENT_REASONS:
            return DecodeResult.transient(
                f"{keyframe.reason.value} keyframe is visually transitional"
            )

        try:
            snapshot = self._analyze_frame(
                keyframe.frame.regions,
                timeout_seconds=float(timeout_seconds),
            )
        except TimeoutError:
            # Preserve the worker's distinct timeout/poison semantics.
            raise
        except Exception as error:
            return DecodeResult.gap(_error_text("frame analysis failed", error))
        if snapshot is None:
            return DecodeResult.gap("frame analysis returned no snapshot")

        with self._lock:
            try:
                evidence = _hand_evidence(snapshot)
                starts_new_hand, boundary_error = self._boundary(evidence)
                if boundary_error:
                    return DecodeResult.gap(boundary_error)
                candidate_ordinal = (
                    self._hand_ordinal + 1
                    if starts_new_hand
                    else self._hand_ordinal
                )
                candidate_hand_id = (
                    str(self._hand_id_factory(candidate_ordinal) or "").strip()
                    if starts_new_hand
                    else self._active_hand_id
                )
                if not candidate_hand_id:
                    return DecodeResult.gap(
                        "hand identity factory returned an empty identifier"
                    )
                self._set_identity_and_timestamp(
                    snapshot,
                    hand_id=candidate_hand_id,
                    timestamp=_utc_capture_timestamp(keyframe),
                )
            except Exception as error:
                return DecodeResult.gap(
                    _error_text("snapshot normalization failed", error)
                )

            validation_errors = self._validation_errors(snapshot)
            vision_error = str(
                getattr(snapshot, "vision_error", "") or ""
            ).strip()
            if vision_error:
                validation_errors = (vision_error, *validation_errors)
            if validation_errors:
                return DecodeResult.gap(
                    "invalid analyzed snapshot: " + "; ".join(validation_errors)
                )

            if starts_new_hand:
                self._hand_ordinal = candidate_ordinal
                self._active_hand_id = candidate_hand_id
            self._previous_evidence = evidence
            return DecodeResult.observation(snapshot)


class ContinuousPublicHistoryRuntime:
    """One-shot lifecycle for consumer-before-producer history capture."""

    def __init__(
        self,
        *,
        capture_service: object,
        coordinator: PublicHistoryCoordinator,
        worker: PublicHistoryWorker,
    ) -> None:
        if not callable(getattr(capture_service, "start", None)):
            raise TypeError("capture_service must provide start()")
        if not callable(getattr(capture_service, "stop", None)):
            raise TypeError("capture_service must provide stop()")
        if not isinstance(coordinator, PublicHistoryCoordinator):
            raise TypeError("coordinator must be a PublicHistoryCoordinator")
        if not isinstance(worker, PublicHistoryWorker):
            raise TypeError("worker must be a PublicHistoryWorker")
        if getattr(capture_service, "ring", None) is not worker.ring:
            raise ValueError("capture service and worker must share one ring")
        if worker.coordinator is not coordinator:
            raise ValueError("worker and runtime must share one coordinator")
        self.capture_service = capture_service
        self.coordinator = coordinator
        self.worker = worker
        self._started = False
        self._stopped = False
        self._stopping = False
        self._failed = False
        self._shutdown_requested = False
        self._lock = threading.Lock()

    @property
    def is_healthy(self) -> bool:
        """Whether both producer and consumer are alive and usable."""

        with self._lock:
            if (
                not self._started
                or self._stopped
                or self._stopping
                or self._failed
                or self._shutdown_requested
            ):
                return False
        capture_running = bool(
            getattr(self.capture_service, "running", False)
        )
        capture_thread = getattr(self.capture_service, "_thread", None)
        capture_alive = bool(
            capture_thread is not None
            and callable(getattr(capture_thread, "is_alive", None))
            and capture_thread.is_alive()
        )
        capture_error = getattr(
            self.capture_service,
            "background_error",
            None,
        )
        try:
            worker_view = self.worker.view()
        except Exception:
            return False
        return bool(
            (capture_running or capture_alive)
            and capture_error is None
            and worker_view.running
            and not worker_view.background_error
        )

    @property
    def health_error(self) -> str:
        if self.is_healthy:
            return ""
        with self._lock:
            if not self._started:
                return "continuous public history was not started"
            if self._shutdown_requested or self._stopping or self._stopped:
                return "continuous public history is stopping or stopped"
            if self._failed:
                return "continuous public history lifecycle failed"
        try:
            worker_view = self.worker.view()
        except Exception as error:
            return _error_text("worker health check failed", error)
        capture_error = getattr(
            self.capture_service,
            "background_error",
            None,
        )
        if capture_error is not None:
            return _error_text(
                "continuous capture failed",
                capture_error,
            )
        if worker_view.background_error:
            return (
                "public-history worker failed: "
                + worker_view.background_error
            )
        if not worker_view.running:
            return "public-history worker is not running"
        return "continuous capture producer is not running"

    def accepted_frame_if_caught_up(
        self,
    ) -> ContinuousPublicHistoryReadiness:
        """Atomically read an accepted bundle only at the ring's current tip."""

        with self._lock:
            if not self._started:
                return ContinuousPublicHistoryReadiness(
                    False,
                    "continuous public history was not started",
                )
            if self._shutdown_requested or self._stopping or self._stopped:
                return ContinuousPublicHistoryReadiness(
                    False,
                    "continuous public history is stopping or stopped",
                )
            if self._failed:
                return ContinuousPublicHistoryReadiness(
                    False,
                    "continuous public history lifecycle failed",
                )

        def capture_error() -> str:
            error = getattr(
                self.capture_service,
                "background_error",
                None,
            )
            if error is not None:
                return _error_text("continuous capture failed", error)
            running = bool(
                getattr(self.capture_service, "running", False)
            )
            thread = getattr(self.capture_service, "_thread", None)
            alive = bool(
                thread is not None
                and callable(getattr(thread, "is_alive", None))
                and thread.is_alive()
            )
            if not (running or alive):
                return "continuous capture producer is not running"
            return ""

        producer_error = capture_error()
        if producer_error:
            return ContinuousPublicHistoryReadiness(
                False,
                producer_error,
            )

        try:
            caught_up = self.worker.read_if_caught_up(
                lambda: (
                    self.coordinator.latest_accepted_frame(),
                    self.coordinator.view(),
                )
            )
        except Exception as error:
            return ContinuousPublicHistoryReadiness(
                False,
                _error_text("caught-up history read failed", error),
            )

        def readiness(
            ready: bool,
            reason: str,
            accepted: AcceptedPublicHistoryFrame | None = None,
        ) -> ContinuousPublicHistoryReadiness:
            return ContinuousPublicHistoryReadiness(
                ready=ready,
                reason=reason,
                accepted_frame=accepted,
                consumed_through_frame_id=(
                    caught_up.consumed_through_frame_id
                ),
                acknowledged_through_frame_id=(
                    caught_up.acknowledged_through_frame_id
                ),
                observed_gap_revision=caught_up.observed_gap_revision,
                current_gap_revision=caught_up.current_gap_revision,
            )

        if not caught_up.caught_up:
            return readiness(False, caught_up.reason)
        if capture_error():
            return readiness(
                False,
                "continuous capture changed during accepted-history read",
            )
        if (
            not isinstance(caught_up.value, tuple)
            or len(caught_up.value) != 2
        ):
            return readiness(
                False,
                "caught-up history read returned an invalid result",
            )
        accepted, coordinator_view = caught_up.value
        if accepted is None:
            detail = (
                getattr(coordinator_view, "gap_reason", "")
                or getattr(coordinator_view, "last_rejection", "")
                or "no accepted frame/transcript bundle is available"
            )
            return readiness(False, detail)
        if not isinstance(accepted, AcceptedPublicHistoryFrame):
            return readiness(
                False,
                "caught-up history read returned an invalid accepted bundle",
            )
        if (
            accepted.frame_id
            != caught_up.consumed_through_frame_id
            or accepted.frame_id
            != caught_up.acknowledged_through_frame_id
        ):
            return readiness(
                False,
                "accepted frame is behind worker/ring consumption "
                f"(accepted {accepted.frame_id}, worker "
                f"{caught_up.consumed_through_frame_id}, ACK "
                f"{caught_up.acknowledged_through_frame_id})",
            )
        return readiness(True, "", accepted)

    def start(self) -> None:
        with self._lock:
            if (
                self._started
                or self._stopped
                or self._shutdown_requested
                or self._failed
            ):
                raise RuntimeError("continuous public history cannot be restarted")
            self._started = True
        try:
            self.worker.start()
        except BaseException:
            with self._lock:
                self._failed = True
                self._stopped = True
            raise
        try:
            self.capture_service.start()
        except BaseException:
            try:
                self.worker.stop(drain=False)
            finally:
                with self._lock:
                    self._failed = True
                    self._stopped = True
                    self._shutdown_requested = True
            raise

    def stop(self, *, timeout_seconds: float = 5.0) -> None:
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds cannot be negative")
        with self._lock:
            if self._stopped:
                return
            if not self._started:
                self._stopped = True
                return
            if self._stopping:
                raise RuntimeError(
                    "continuous public history stop is already in progress"
                )
            self._stopping = True
            self._shutdown_requested = True

        # Stopping or losing either side immediately revokes every previously
        # published strategy bundle.
        try:
            self.coordinator.invalidate_gap(
                "continuous public history runtime stopped"
            )
        except BaseException:
            with self._lock:
                self._stopping = False
            raise
        deadline = time.monotonic() + timeout_seconds

        def remaining() -> float:
            return max(0.0, deadline - time.monotonic())

        # The producer flushes its detector into the ring before the consumer
        # is asked to drain and stop.  A timed-out or observably live producer
        # keeps the consumer alive so a retry can finish the ordered tail.
        capture_error: BaseException | None = None
        try:
            self.capture_service.stop(timeout_seconds=remaining())
        except BaseException as error:
            capture_thread = getattr(self.capture_service, "_thread", None)
            capture_alive = bool(
                getattr(self.capture_service, "running", False)
                or (
                    capture_thread is not None
                    and callable(getattr(capture_thread, "is_alive", None))
                    and capture_thread.is_alive()
                )
            )
            if isinstance(error, TimeoutError) or capture_alive:
                with self._lock:
                    self._stopping = False
                raise
            # A terminated producer may report a terminal capture/cleanup
            # failure.  Preserve it, but still stop the consumer to avoid a
            # stranded worker.
            capture_error = error

        try:
            self.worker.stop(
                drain=True,
                timeout_seconds=remaining(),
            )
        except BaseException as worker_error:
            with self._lock:
                self._stopping = False
                if capture_error is not None:
                    self._failed = True
            if capture_error is not None:
                raise RuntimeError(
                    "capture shutdown failed with "
                    f"{type(capture_error).__name__}: {capture_error}; "
                    "worker shutdown also failed with "
                    f"{type(worker_error).__name__}: {worker_error}"
                ) from capture_error
            raise
        with self._lock:
            self._stopping = False
            self._stopped = True
            if capture_error is not None:
                self._failed = True
        if capture_error is not None:
            raise capture_error


__all__ = [
    "ContinuousPublicHistoryDecoder",
    "ContinuousPublicHistoryReadiness",
    "ContinuousPublicHistoryRuntime",
    "FrameAnalyzer",
    "SnapshotValidator",
]
