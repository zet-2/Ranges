"""Provider-neutral coordination for ordered public-hand observations.

The coordinator accepts asynchronously decoded snapshot-like candidates,
orders them by capture frame ID, and advances a transactional
``PublicHandEventRecorder`` only when a candidate exactly replays.  A rejected
candidate does not mutate the recorder.  Readiness is intentionally stricter
than decoder progress: a transcript is usable only when an accepted
observation proves the requested hand through the requested frame.
"""

from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import threading
from typing import Callable, Iterator

from gto_event_collector import (
    PublicEventProbeStatus,
    PublicHandEventRecorder,
)
from gto_hand_history import PublicHandHistory


class PublicHistoryStatus(str, Enum):
    WAITING_FOR_ANCHOR = "waiting_for_anchor"
    TRACKING = "tracking"
    GAPPED = "gapped"


class CandidateDisposition(str, Enum):
    ANCHORED = "anchored"
    NEW_HAND_ANCHORED = "new_hand_anchored"
    ADVANCED = "advanced"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"
    IGNORED_GAPPED_HAND = "ignored_gapped_hand"


@dataclass(frozen=True, slots=True)
class PublicObservationCandidate:
    """One decoded full-table observation tied to its capture frame."""

    frame_id: int
    snapshot: object

    def __post_init__(self) -> None:
        if self.frame_id <= 0:
            raise ValueError("candidate frame_id must be positive")


@dataclass(frozen=True, slots=True)
class CandidateResult:
    """Observable outcome after processing one ordered candidate."""

    frame_id: int
    disposition: CandidateDisposition
    status: PublicHistoryStatus
    hand_id: str = ""
    error: str = ""
    event_count: int = 0


@dataclass(frozen=True, slots=True)
class AcceptedPublicHistoryFrame:
    """One atomic accepted-frame/snapshot/transcript bundle.

    Consumers must not combine ``history`` from this bundle with a separately
    captured snapshot.  The coordinator publishes the three values together
    only after the snapshot has replayed exactly through ``frame_id``.
    """

    frame_id: int
    snapshot: object
    history: PublicHandHistory

    def __post_init__(self) -> None:
        if self.frame_id <= 0:
            raise ValueError("accepted frame_id must be positive")
        if self.snapshot is None:
            raise ValueError("accepted frame requires a snapshot")
        if not isinstance(self.history, PublicHandHistory):
            raise TypeError("accepted frame history must be PublicHandHistory")
        snapshot_hand_id = _raw_hand_id(self.snapshot)
        if snapshot_hand_id != self.history.hand_id:
            raise ValueError(
                "accepted snapshot and history must have the same hand_id"
            )


@dataclass(frozen=True, slots=True)
class PublicHistoryView:
    """Immutable coordinator state for diagnostics and strategy gating."""

    status: PublicHistoryStatus
    hand_id: str
    history: PublicHandHistory | None
    decoder_watermark: int
    proven_through_frame_id: int
    last_processed_frame_id: int
    pending_candidates: int
    gap_reason: str
    gap_frame_id: int | None
    last_rejection: str
    accepted_frame: AcceptedPublicHistoryFrame | None = None


@dataclass(frozen=True, slots=True)
class PublicHistoryReadiness:
    """Result of a same-hand, through-frame readiness check."""

    ready: bool
    reason: str
    status: PublicHistoryStatus
    hand_id: str
    through_frame_id: int
    decoder_watermark: int
    proven_through_frame_id: int
    history: PublicHandHistory | None = None
    accepted_frame: AcceptedPublicHistoryFrame | None = None


def _raw_hand_id(snapshot: object) -> str:
    return str(getattr(snapshot, "hand_id", "") or "").strip()


class PublicHistoryCoordinator:
    """Order decoder results and expose only a proven contiguous transcript."""

    def __init__(
        self,
        recorder_factory: Callable[[], PublicHandEventRecorder] = (
            PublicHandEventRecorder
        ),
    ) -> None:
        if not callable(recorder_factory):
            raise TypeError("recorder_factory must be callable")
        self._recorder_factory = recorder_factory
        self._recorder = self._make_recorder()
        self._status = PublicHistoryStatus.WAITING_FOR_ANCHOR
        self._active_hand_id = ""
        self._expected_hand_id = ""
        self._decoder_watermark = 0
        self._proven_through_frame_id = 0
        self._last_processed_frame_id = 0
        self._pending: dict[int, PublicObservationCandidate] = {}
        self._gap_reason = ""
        self._gap_frame_id: int | None = None
        self._gapped_hand_ids: set[str] = set()
        self._unknown_hand_before_gap = False
        self._unresolved_rejections: dict[int, str] = {}
        self._retired_hand_ids: set[str] = set()
        self._last_rejection = ""
        self._accepted_frame: AcceptedPublicHistoryFrame | None = None
        self._lock = threading.RLock()

    def _make_recorder(self) -> PublicHandEventRecorder:
        recorder = self._recorder_factory()
        if not isinstance(recorder, PublicHandEventRecorder):
            raise TypeError(
                "recorder_factory must return PublicHandEventRecorder"
            )
        return recorder

    @staticmethod
    def _copy_accepted_frame(
        accepted: AcceptedPublicHistoryFrame | None,
    ) -> AcceptedPublicHistoryFrame | None:
        if accepted is None:
            return None
        return AcceptedPublicHistoryFrame(
            frame_id=accepted.frame_id,
            snapshot=deepcopy(accepted.snapshot),
            history=accepted.history,
        )

    def _accepted_is_current_locked(self) -> bool:
        accepted = self._accepted_frame
        return bool(
            self._status is PublicHistoryStatus.TRACKING
            and accepted is not None
            and self._recorder.complete
            and self._recorder.history is not None
            and not self._pending
            and accepted.frame_id == self._proven_through_frame_id
            and self._decoder_watermark == self._proven_through_frame_id
            and accepted.history == self._recorder.history
        )

    @property
    def status(self) -> PublicHistoryStatus:
        with self._lock:
            return self._status

    @property
    def decoder_watermark(self) -> int:
        with self._lock:
            return self._decoder_watermark

    @property
    def proven_through_frame_id(self) -> int:
        with self._lock:
            return self._proven_through_frame_id

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Hold the coordinator boundary across an external commit plus ACK.

        The public-history worker uses this to ensure a concurrent readiness
        read cannot observe a newly committed transcript until the source-ring
        acknowledgement has either succeeded or a late gap has invalidated it.
        """

        with self._lock:
            yield

    def submit_candidate(
        self,
        frame_id: int,
        snapshot: object,
    ) -> None:
        """Buffer one decoder result until the watermark proves ordering."""

        # Buffer an owned copy: submit and watermark advancement are separate
        # API calls, so a caller must not be able to mutate the candidate in
        # between and change what the recorder eventually probes.
        candidate = PublicObservationCandidate(
            frame_id,
            deepcopy(snapshot),
        )
        with self._lock:
            if frame_id <= self._decoder_watermark:
                raise ValueError(
                    "candidate arrived at or behind decoder watermark"
                )
            if frame_id <= self._last_processed_frame_id:
                raise ValueError("candidate frame was already processed")
            if frame_id in self._pending:
                raise ValueError("candidate frame_id is already buffered")
            self._pending[frame_id] = candidate

    def _replace_with_waiting_recorder(self, expected_hand_id: str) -> None:
        self._recorder = self._make_recorder()
        self._status = PublicHistoryStatus.WAITING_FOR_ANCHOR
        self._active_hand_id = ""
        self._expected_hand_id = expected_hand_id
        self._proven_through_frame_id = 0
        self._gap_reason = ""
        self._gap_frame_id = None
        self._gapped_hand_ids.clear()
        self._unknown_hand_before_gap = False
        self._unresolved_rejections.clear()
        self._accepted_frame = None

    def _accepted_result(
        self,
        candidate: PublicObservationCandidate,
        disposition: CandidateDisposition,
    ) -> CandidateResult:
        history = self._recorder.history
        assert history is not None
        self._status = PublicHistoryStatus.TRACKING
        self._active_hand_id = history.hand_id
        self._expected_hand_id = history.hand_id
        self._proven_through_frame_id = candidate.frame_id
        self._gap_reason = ""
        self._gap_frame_id = None
        self._gapped_hand_ids.clear()
        self._unknown_hand_before_gap = False
        self._unresolved_rejections.clear()
        self._last_rejection = ""
        self._accepted_frame = AcceptedPublicHistoryFrame(
            frame_id=candidate.frame_id,
            snapshot=deepcopy(candidate.snapshot),
            history=history,
        )
        return CandidateResult(
            frame_id=candidate.frame_id,
            disposition=disposition,
            status=self._status,
            hand_id=history.hand_id,
            event_count=len(history.events),
        )

    def _process_waiting(
        self,
        candidate: PublicObservationCandidate,
    ) -> CandidateResult:
        raw_hand_id = _raw_hand_id(candidate.snapshot)
        if raw_hand_id and raw_hand_id in self._retired_hand_ids:
            error = (
                f"retired hand {raw_hand_id!r} cannot be anchored again"
            )
            self._last_rejection = error
            return CandidateResult(
                frame_id=candidate.frame_id,
                disposition=CandidateDisposition.REJECTED,
                status=self._status,
                hand_id=raw_hand_id,
                error=error,
            )
        probe = self._recorder.probe(candidate.snapshot)
        candidate_hand_id = probe.candidate_hand_id or _raw_hand_id(
            candidate.snapshot
        )
        if not probe.accepted:
            self._expected_hand_id = candidate_hand_id or self._expected_hand_id
            self._last_rejection = probe.error
            return CandidateResult(
                frame_id=candidate.frame_id,
                disposition=CandidateDisposition.REJECTED,
                status=self._status,
                hand_id=candidate_hand_id,
                error=probe.error,
            )

        self._recorder.commit(probe)
        return self._accepted_result(
            candidate,
            CandidateDisposition.ANCHORED,
        )

    def _process_tracking(
        self,
        candidate: PublicObservationCandidate,
    ) -> CandidateResult:
        raw_hand_id = _raw_hand_id(candidate.snapshot)
        if (
            raw_hand_id
            and raw_hand_id != self._active_hand_id
            and raw_hand_id in self._retired_hand_ids
        ):
            error = (
                f"retired hand {raw_hand_id!r} cannot replace the active hand"
            )
            self._last_rejection = error
            return CandidateResult(
                frame_id=candidate.frame_id,
                disposition=CandidateDisposition.REJECTED,
                status=self._status,
                hand_id=raw_hand_id,
                error=error,
                event_count=(
                    len(self._recorder.history.events)
                    if self._recorder.history is not None
                    else 0
                ),
            )
        probe = self._recorder.probe(candidate.snapshot)
        candidate_hand_id = probe.candidate_hand_id or _raw_hand_id(
            candidate.snapshot
        )
        if not probe.accepted:
            if probe.replaces_hand or (
                candidate_hand_id
                and candidate_hand_id != self._active_hand_id
            ):
                # The old hand is over, but this first view of the new hand is
                # not a valid forced-bet anchor.  Stop exposing the old prefix.
                if self._active_hand_id:
                    self._retired_hand_ids.add(self._active_hand_id)
                self._replace_with_waiting_recorder(candidate_hand_id)
            self._last_rejection = probe.error
            return CandidateResult(
                frame_id=candidate.frame_id,
                disposition=CandidateDisposition.REJECTED,
                status=self._status,
                hand_id=candidate_hand_id or self._active_hand_id,
                error=probe.error,
                event_count=(
                    len(self._recorder.history.events)
                    if self._recorder.history is not None
                    else 0
                ),
            )

        previous_hand_id = self._active_hand_id
        self._recorder.commit(probe)
        if (
            probe.replaces_hand
            and previous_hand_id
            and previous_hand_id != candidate_hand_id
        ):
            self._retired_hand_ids.add(previous_hand_id)
        disposition = {
            PublicEventProbeStatus.NEW_HAND_ANCHORED: (
                CandidateDisposition.NEW_HAND_ANCHORED
            ),
            PublicEventProbeStatus.DUPLICATE: CandidateDisposition.DUPLICATE,
            PublicEventProbeStatus.ADVANCED: CandidateDisposition.ADVANCED,
            PublicEventProbeStatus.ANCHORED: CandidateDisposition.ANCHORED,
        }[probe.status]
        return self._accepted_result(candidate, disposition)

    def _process_gapped(
        self,
        candidate: PublicObservationCandidate,
    ) -> CandidateResult:
        candidate_hand_id = _raw_hand_id(candidate.snapshot)
        blocked_hand_id = self._active_hand_id or self._expected_hand_id
        if (
            self._gap_frame_id is not None
            and candidate.frame_id <= self._gap_frame_id
        ):
            if candidate_hand_id:
                self._gapped_hand_ids.add(candidate_hand_id)
            else:
                self._unknown_hand_before_gap = True
        if (
            self._unknown_hand_before_gap
            or not candidate_hand_id
            or candidate_hand_id in self._gapped_hand_ids
            or candidate_hand_id in self._retired_hand_ids
            or (
                self._gap_frame_id is not None
                and candidate.frame_id <= self._gap_frame_id
            )
        ):
            return CandidateResult(
                frame_id=candidate.frame_id,
                disposition=CandidateDisposition.IGNORED_GAPPED_HAND,
                status=self._status,
                hand_id=blocked_hand_id or candidate_hand_id,
                error=self._gap_reason,
                event_count=(
                    len(self._recorder.history.events)
                    if self._recorder.history is not None
                    else 0
                ),
            )

        # Probe a different hand against a temporary recorder.  A malformed
        # candidate must not clear GAPPED; recovery is committed only after
        # the candidate proves an untouched forced-bet anchor.
        candidate_recorder = self._make_recorder()
        probe = candidate_recorder.probe(candidate.snapshot)
        if not probe.accepted:
            self._last_rejection = probe.error
            return CandidateResult(
                frame_id=candidate.frame_id,
                disposition=CandidateDisposition.REJECTED,
                status=self._status,
                hand_id=candidate_hand_id,
                error=probe.error,
            )
        candidate_recorder.commit(probe)
        if blocked_hand_id:
            self._retired_hand_ids.add(blocked_hand_id)
        self._recorder = candidate_recorder
        return self._accepted_result(
            candidate,
            CandidateDisposition.ANCHORED,
        )

    def _process_candidate(
        self,
        candidate: PublicObservationCandidate,
    ) -> CandidateResult:
        if self._status is PublicHistoryStatus.WAITING_FOR_ANCHOR:
            return self._process_waiting(candidate)
        if self._status is PublicHistoryStatus.TRACKING:
            return self._process_tracking(candidate)
        return self._process_gapped(candidate)

    def advance_decoder_watermark(
        self,
        through_frame_id: int,
    ) -> tuple[CandidateResult, ...]:
        """Process buffered candidates in frame order through a decoder ACK."""

        if through_frame_id < 0:
            raise ValueError("decoder watermark cannot be negative")
        with self._lock:
            if through_frame_id < self._decoder_watermark:
                raise ValueError("decoder watermark cannot move backwards")

            ready_ids = sorted(
                frame_id
                for frame_id in self._pending
                if frame_id <= through_frame_id
            )
            results = []
            for frame_id in ready_ids:
                candidate = self._pending.pop(frame_id)
                result = self._process_candidate(candidate)
                results.append(result)
                if result.disposition is CandidateDisposition.REJECTED:
                    self._unresolved_rejections[frame_id] = _raw_hand_id(
                        candidate.snapshot
                    )
                elif result.disposition is not (
                    CandidateDisposition.IGNORED_GAPPED_HAND
                ):
                    self._unresolved_rejections = {
                        rejected_frame: hand_id
                        for rejected_frame, hand_id
                        in self._unresolved_rejections.items()
                        if rejected_frame > frame_id
                    }
                self._last_processed_frame_id = frame_id
            self._decoder_watermark = through_frame_id
            return tuple(results)

    def invalidate_gap(
        self,
        reason: str,
        *,
        frame_id: int | None = None,
    ) -> None:
        """Explicitly invalidate the active hand after capture/decode loss."""

        detail = str(reason or "").strip()
        if not detail:
            raise ValueError("gap reason cannot be empty")
        if frame_id is not None and frame_id <= 0:
            raise ValueError("gap frame_id must be positive")
        with self._lock:
            self._recorder.invalidate_gap(detail)
            self._status = PublicHistoryStatus.GAPPED
            self._gap_reason = detail
            if frame_id is None:
                # Without an ordering coordinate, no later candidate can prove
                # whether it belongs before or after the loss.  Automatic
                # recovery would be fail-open, so require an explicit
                # coordinator reset/restart instead.
                self._unknown_hand_before_gap = True
            else:
                # Gap callbacks can arrive out of frame order.  Never let an
                # older late callback move the recovery boundary backwards.
                self._gap_frame_id = max(
                    self._gap_frame_id or frame_id,
                    frame_id,
                )
            effective_gap_frame_id = self._gap_frame_id
            blocked_hand_id = self._active_hand_id or self._expected_hand_id
            if blocked_hand_id:
                self._gapped_hand_ids.add(blocked_hand_id)
            for rejected_frame, rejected_hand_id in (
                self._unresolved_rejections.items()
            ):
                if (
                    effective_gap_frame_id is not None
                    and rejected_frame > effective_gap_frame_id
                ):
                    continue
                if rejected_hand_id:
                    self._gapped_hand_ids.add(rejected_hand_id)
                else:
                    self._unknown_hand_before_gap = True
            self._last_rejection = ""
            self._accepted_frame = None

    def view(self) -> PublicHistoryView:
        with self._lock:
            accepted = (
                self._accepted_frame
                if self._accepted_is_current_locked()
                else None
            )
            return PublicHistoryView(
                status=self._status,
                hand_id=self._active_hand_id or self._expected_hand_id,
                history=self._recorder.history,
                decoder_watermark=self._decoder_watermark,
                proven_through_frame_id=self._proven_through_frame_id,
                last_processed_frame_id=self._last_processed_frame_id,
                pending_candidates=len(self._pending),
                gap_reason=self._gap_reason,
                gap_frame_id=self._gap_frame_id,
                last_rejection=self._last_rejection,
                accepted_frame=self._copy_accepted_frame(accepted),
            )

    def readiness(
        self,
        *,
        hand_id: str,
        through_frame_id: int,
    ) -> PublicHistoryReadiness:
        """Check whether one exact hand is proven through a capture frame."""

        requested_hand = str(hand_id or "").strip()
        if not requested_hand:
            raise ValueError("readiness hand_id cannot be empty")
        if through_frame_id <= 0:
            raise ValueError("through_frame_id must be positive")

        with self._lock:
            reason = ""
            ready = False
            history: PublicHandHistory | None = None
            if self._status is PublicHistoryStatus.WAITING_FOR_ANCHOR:
                reason = "waiting for untouched preflop forced-bet anchor"
            elif self._status is PublicHistoryStatus.GAPPED:
                reason = f"public history is gapped: {self._gap_reason}"
            elif self._active_hand_id != requested_hand:
                reason = (
                    f"tracked hand {self._active_hand_id!r} does not match "
                    f"requested hand {requested_hand!r}"
                )
            elif self._decoder_watermark < through_frame_id:
                reason = (
                    "decoder has not reached requested frame "
                    f"{through_frame_id}"
                )
            elif self._proven_through_frame_id < through_frame_id:
                reason = (
                    "no accepted observation proves public history through "
                    f"frame {through_frame_id}"
                )
            elif self._proven_through_frame_id > through_frame_id:
                reason = (
                    "the retained transcript is proven only for frame "
                    f"{self._proven_through_frame_id}, not historical frame "
                    f"{through_frame_id}"
                )
            elif self._decoder_watermark > through_frame_id:
                reason = (
                    "decoder has advanced to newer frame "
                    f"{self._decoder_watermark}; frame {through_frame_id} is "
                    "no longer the latest accepted table state"
                )
            elif not self._recorder.complete or self._recorder.history is None:
                reason = self._recorder.error or "public history is unavailable"
            elif (
                self._accepted_frame is None
                or self._accepted_frame.frame_id != through_frame_id
                or self._accepted_frame.history != self._recorder.history
                or self._pending
            ):
                reason = (
                    "accepted snapshot/history pairing is unavailable for "
                    f"frame {through_frame_id}"
                )
            else:
                ready = True
                history = self._recorder.history

            return PublicHistoryReadiness(
                ready=ready,
                reason=reason,
                status=self._status,
                hand_id=self._active_hand_id or self._expected_hand_id,
                through_frame_id=through_frame_id,
                decoder_watermark=self._decoder_watermark,
                proven_through_frame_id=self._proven_through_frame_id,
                history=history,
                accepted_frame=(
                    self._copy_accepted_frame(self._accepted_frame)
                    if ready
                    else None
                ),
            )

    def latest_accepted_frame(self) -> AcceptedPublicHistoryFrame | None:
        """Return the latest internally consistent bundle under one lock."""

        with self._lock:
            if not self._accepted_is_current_locked():
                return None
            return self._copy_accepted_frame(self._accepted_frame)

    def accepted_frame_if_ready(
        self,
        *,
        hand_id: str,
        through_frame_id: int,
    ) -> AcceptedPublicHistoryFrame | None:
        """Return only the exact bundle that passes frame-exact readiness."""

        return self.readiness(
            hand_id=hand_id,
            through_frame_id=through_frame_id,
        ).accepted_frame

    def history_if_ready(
        self,
        *,
        hand_id: str,
        through_frame_id: int,
    ) -> PublicHandHistory | None:
        """Return the transcript only when ``readiness`` succeeds."""

        return self.readiness(
            hand_id=hand_id,
            through_frame_id=through_frame_id,
        ).history


__all__ = [
    "AcceptedPublicHistoryFrame",
    "CandidateDisposition",
    "CandidateResult",
    "PublicHistoryCoordinator",
    "PublicHistoryReadiness",
    "PublicHistoryStatus",
    "PublicHistoryView",
    "PublicObservationCandidate",
]
