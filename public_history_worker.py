"""Ordered bridge from captured keyframes to public-hand reconstruction.

The worker is provider-neutral.  A decoder receives one :class:`Keyframe` at a
time and must return an explicit observation, transient frame, or gap result.
No OCR/model implementation lives here.

Consumption is fail-closed:

* keyframes and capture gaps are merged by frame ID;
* a trusted observation is submitted and committed before its ring ACK;
* decoder/capture failures invalidate the coordinator before their ACK;
* the ACK uses the ring's gap-revision compare-and-set, so a late gap cannot be
  silently crossed while a decoder call is in flight.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import threading
from typing import Callable, Protocol

from live_capture import (
    CaughtUpRingRead,
    CaptureGapEvent,
    Keyframe,
    KeyframeRing,
)
from public_history_pipeline import (
    CandidateDisposition,
    CandidateResult,
    PublicHistoryCoordinator,
)


class DecodeDisposition(str, Enum):
    """Exhaustive semantic decoder outcomes."""

    OBSERVATION = "observation"
    TRANSIENT = "transient"
    GAP = "gap"


@dataclass(frozen=True, slots=True)
class DecodeResult:
    """One explicit semantic result for a keyframe."""

    disposition: DecodeDisposition
    snapshot: object | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, DecodeDisposition):
            raise TypeError("disposition must be a DecodeDisposition")
        detail = str(self.reason or "").strip()
        if self.disposition is DecodeDisposition.OBSERVATION:
            if self.snapshot is None:
                raise ValueError("observation result requires a snapshot")
            if detail:
                raise ValueError("observation result cannot include a reason")
        elif self.snapshot is not None:
            raise ValueError(
                f"{self.disposition.value} result cannot include a snapshot"
            )
        if self.disposition is DecodeDisposition.GAP and not detail:
            raise ValueError("gap result requires a reason")

    @classmethod
    def observation(cls, snapshot: object) -> "DecodeResult":
        return cls(DecodeDisposition.OBSERVATION, snapshot=snapshot)

    @classmethod
    def transient(cls, reason: str = "") -> "DecodeResult":
        return cls(DecodeDisposition.TRANSIENT, reason=str(reason or ""))

    @classmethod
    def gap(cls, reason: str) -> "DecodeResult":
        return cls(DecodeDisposition.GAP, reason=reason)


class PublicObservationDecoder(Protocol):
    """Decode one keyframe within the supplied provider timeout.

    Implementations must enforce ``timeout_seconds`` and raise
    :class:`TimeoutError` when it expires.  The worker converts that exception
    to an explicit coordinator gap.
    """

    def decode(
        self,
        keyframe: Keyframe,
        *,
        timeout_seconds: float,
    ) -> DecodeResult:
        """Return exactly one explicit outcome for ``keyframe``."""


class WorkerEventDisposition(str, Enum):
    OBSERVATION_ACCEPTED = "observation_accepted"
    OBSERVATION_IGNORED = "observation_ignored"
    OBSERVATION_REJECTED = "observation_rejected"
    TRANSIENT = "transient"
    DECODER_GAP = "decoder_gap"
    CAPTURE_GAP = "capture_gap"
    DECODER_TIMEOUT = "decoder_timeout"
    DECODER_ERROR = "decoder_error"
    COORDINATOR_ERROR = "coordinator_error"


@dataclass(frozen=True, slots=True)
class WorkerEvent:
    """Observable outcome for one consumed keyframe or gap."""

    frame_id: int
    disposition: WorkerEventDisposition
    detail: str = ""
    candidate_result: CandidateResult | None = None


@dataclass(frozen=True, slots=True)
class WorkerCycle:
    """Deterministic result of one atomic-snapshot processing cycle."""

    events: tuple[WorkerEvent, ...]
    observed_gap_revision: int
    consumed_through_frame_id: int
    acknowledged_through_frame_id: int
    acknowledgement_committed: bool
    acknowledgement_stale: bool
    fatal_error: str = ""

    @property
    def idle(self) -> bool:
        return not self.events and not self.fatal_error


@dataclass(frozen=True, slots=True)
class PublicHistoryWorkerView:
    running: bool
    consumed_through_frame_id: int
    observed_gap_revision: int
    cycles: int
    events: int
    decoder_failures: int
    decoder_poisoned: bool
    stale_acknowledgements: int
    last_error: str
    background_error: str


@dataclass(frozen=True, slots=True)
class PublicHistoryCaughtUpRead:
    """Downstream read tied to worker cursors and one atomic ring snapshot."""

    caught_up: bool
    reason: str
    consumed_through_frame_id: int
    observed_gap_revision: int
    acknowledged_through_frame_id: int
    current_gap_revision: int
    value: object | None = None


class PublicHistoryWorker:
    """Sequential transactional consumer for a :class:`KeyframeRing`."""

    def __init__(
        self,
        *,
        ring: KeyframeRing,
        coordinator: PublicHistoryCoordinator,
        decoder: PublicObservationDecoder,
        decode_timeout_seconds: float = 8.0,
        idle_wait_seconds: float = 0.1,
    ) -> None:
        if not isinstance(ring, KeyframeRing):
            raise TypeError("ring must be a KeyframeRing")
        if not isinstance(coordinator, PublicHistoryCoordinator):
            raise TypeError("coordinator must be a PublicHistoryCoordinator")
        decode = getattr(decoder, "decode", None)
        if not callable(decode):
            raise TypeError("decoder must provide a callable decode method")
        if decode_timeout_seconds <= 0:
            raise ValueError("decode_timeout_seconds must be positive")
        if idle_wait_seconds <= 0:
            raise ValueError("idle_wait_seconds must be positive")

        self.ring = ring
        self.coordinator = coordinator
        # PublicHistoryCoordinator serializes every mutation/readiness check on
        # one transaction. Holding it through the ring CAS prevents a
        # concurrent strategy read from observing a commit whose ACK was made
        # stale by a newly recorded gap.
        self.decoder = decoder
        self.decode_timeout_seconds = float(decode_timeout_seconds)
        self.idle_wait_seconds = float(idle_wait_seconds)

        self._consumed_through = ring.acknowledged_through
        self._observed_gap_revision = 0
        self._cycles = 0
        self._event_count = 0
        self._decoder_failures = 0
        self._decoder_poisoned = False
        self._stale_acknowledgements = 0
        self._last_error = ""
        self._background_error = ""

        self._process_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._started = False
        self._drain_on_stop = True

    def _invalidate(self, reason: str, frame_id: int) -> None:
        self.coordinator.invalidate_gap(reason, frame_id=frame_id)

    def _coordinator_failure(
        self,
        *,
        frame_id: int,
        error: Exception,
    ) -> WorkerEvent:
        detail = (
            f"public history coordinator failed at frame {frame_id}: "
            f"{type(error).__name__}: {error}"
        )
        try:
            self._invalidate(detail, frame_id)
        except Exception as invalidate_error:
            raise RuntimeError(
                f"{detail}; fail-closed invalidation also failed: "
                f"{type(invalidate_error).__name__}: {invalidate_error}"
            ) from invalidate_error
        self._last_error = detail
        return WorkerEvent(
            frame_id=frame_id,
            disposition=WorkerEventDisposition.COORDINATOR_ERROR,
            detail=detail,
        )

    def _process_gap_event(self, event: CaptureGapEvent) -> WorkerEvent:
        gap = event.gap
        detail = (
            f"capture gap {gap.kind.value} frames "
            f"{gap.first_frame_id}-{gap.last_frame_id}: {gap.reason}"
        )
        # The whole lost interval is unsafe.  Using only the first coordinate
        # would let a new hand anchor inside a declared multi-frame gap.
        self._invalidate(detail, gap.last_frame_id)
        self._consumed_through = max(
            self._consumed_through,
            gap.last_frame_id,
        )
        self._last_error = detail
        return WorkerEvent(
            frame_id=gap.first_frame_id,
            disposition=WorkerEventDisposition.CAPTURE_GAP,
            detail=detail,
        )

    def _process_observation(
        self,
        keyframe: Keyframe,
        snapshot: object,
    ) -> WorkerEvent:
        self.coordinator.submit_candidate(keyframe.frame_id, snapshot)
        candidate_results = self.coordinator.advance_decoder_watermark(
            keyframe.frame_id
        )
        if len(candidate_results) != 1:
            raise RuntimeError(
                "trusted observation did not produce exactly one result"
            )
        candidate = candidate_results[0]
        if candidate.disposition is CandidateDisposition.REJECTED:
            detail = (
                f"trusted observation rejected at frame {keyframe.frame_id}: "
                f"{candidate.error or 'unknown reconstruction error'}"
            )
            self._invalidate(detail, keyframe.frame_id)
            self._last_error = detail
            return WorkerEvent(
                frame_id=keyframe.frame_id,
                disposition=WorkerEventDisposition.OBSERVATION_REJECTED,
                detail=detail,
                candidate_result=candidate,
            )
        if candidate.disposition is CandidateDisposition.IGNORED_GAPPED_HAND:
            return WorkerEvent(
                frame_id=keyframe.frame_id,
                disposition=WorkerEventDisposition.OBSERVATION_IGNORED,
                detail=candidate.error,
                candidate_result=candidate,
            )
        return WorkerEvent(
            frame_id=keyframe.frame_id,
            disposition=WorkerEventDisposition.OBSERVATION_ACCEPTED,
            candidate_result=candidate,
        )

    def _decode_with_deadline(self, keyframe: Keyframe) -> DecodeResult:
        """Enforce the provider deadline without trusting decoder cooperation.

        Python cannot safely kill an arbitrary decoder thread.  The timed-out
        call is isolated in a daemon thread and the decoder instance is
        permanently poisoned, preventing an accumulation of concurrent calls.
        Decoder implementations must remain pure with respect to the ring and
        coordinator; late results are ignored.
        """

        if self._decoder_poisoned:
            raise RuntimeError(
                "decoder disabled after a previous hard timeout"
            )

        finished = threading.Event()
        outcome: list[tuple[str, object]] = []

        def invoke() -> None:
            try:
                outcome.append(
                    (
                        "result",
                        self.decoder.decode(
                            keyframe,
                            timeout_seconds=self.decode_timeout_seconds,
                        ),
                    )
                )
            except BaseException as error:
                outcome.append(("error", error))
            finally:
                finished.set()

        thread = threading.Thread(
            target=invoke,
            name=f"public-history-decode-{keyframe.frame_id}",
            daemon=True,
        )
        thread.start()
        if not finished.wait(self.decode_timeout_seconds):
            self._decoder_poisoned = True
            raise TimeoutError("decoder exceeded the enforced hard deadline")

        kind, value = outcome[0]
        if kind == "error":
            if isinstance(value, Exception):
                raise value
            raise RuntimeError(
                f"decoder aborted with {type(value).__name__}"
            )
        if not isinstance(value, DecodeResult):
            raise TypeError("decoder must return DecodeResult")
        return value

    def _process_keyframe(self, keyframe: Keyframe) -> WorkerEvent:
        try:
            result = self._decode_with_deadline(keyframe)
        except TimeoutError as error:
            self._decoder_failures += 1
            detail = (
                f"decoder timeout at frame {keyframe.frame_id} after "
                f"{self.decode_timeout_seconds:g}s"
            )
            if str(error).strip():
                detail += f": {error}"
            self._invalidate(detail, keyframe.frame_id)
            self._last_error = detail
            return WorkerEvent(
                frame_id=keyframe.frame_id,
                disposition=WorkerEventDisposition.DECODER_TIMEOUT,
                detail=detail,
            )
        except Exception as error:
            self._decoder_failures += 1
            detail = (
                f"decoder error at frame {keyframe.frame_id}: "
                f"{type(error).__name__}: {error}"
            )
            self._invalidate(detail, keyframe.frame_id)
            self._last_error = detail
            return WorkerEvent(
                frame_id=keyframe.frame_id,
                disposition=WorkerEventDisposition.DECODER_ERROR,
                detail=detail,
            )

        if result.disposition is DecodeDisposition.OBSERVATION:
            assert result.snapshot is not None
            return self._process_observation(keyframe, result.snapshot)
        if result.disposition is DecodeDisposition.TRANSIENT:
            self.coordinator.advance_decoder_watermark(keyframe.frame_id)
            return WorkerEvent(
                frame_id=keyframe.frame_id,
                disposition=WorkerEventDisposition.TRANSIENT,
                detail=str(result.reason or "").strip(),
            )

        detail = (
            f"decoder declared gap at frame {keyframe.frame_id}: "
            f"{str(result.reason).strip()}"
        )
        self._invalidate(detail, keyframe.frame_id)
        self._last_error = detail
        return WorkerEvent(
            frame_id=keyframe.frame_id,
            disposition=WorkerEventDisposition.DECODER_GAP,
            detail=detail,
        )

    def process_once(self) -> WorkerCycle:
        """Process all items in one atomic ring snapshot, in frame order."""

        current_thread = threading.current_thread()
        with self._state_lock:
            if (
                self._running
                and self._thread is not current_thread
            ):
                raise RuntimeError(
                    "process_once cannot run beside the background consumer"
                )

        with self._process_lock, self.coordinator.transaction():
            batch = self.ring.consumer_batch(
                after_frame_id=self._consumed_through,
                after_gap_revision=self._observed_gap_revision,
            )
            ordered: list[
                tuple[int, int, int, Keyframe | CaptureGapEvent]
            ] = []
            for gap_event in batch.gap_events:
                ordered.append(
                    (
                        gap_event.gap.first_frame_id,
                        0,
                        gap_event.revision,
                        gap_event,
                    )
                )
            for keyframe in batch.keyframes:
                ordered.append(
                    (keyframe.frame_id, 1, keyframe.frame_id, keyframe)
                )
            ordered.sort(key=lambda item: item[:3])

            events: list[WorkerEvent] = []
            fatal_error = ""
            processed_all_gaps = True
            for index, (_, _, _, item) in enumerate(ordered):
                if (
                    self._thread is current_thread
                    and self._stop_event.is_set()
                    and not self._drain_on_stop
                ):
                    processed_all_gaps = not any(
                        isinstance(remaining[3], CaptureGapEvent)
                        for remaining in ordered[index:]
                    )
                    break
                frame_id = (
                    item.frame_id
                    if isinstance(item, Keyframe)
                    else item.gap.first_frame_id
                )
                try:
                    if isinstance(item, CaptureGapEvent):
                        event = self._process_gap_event(item)
                    else:
                        event = self._process_keyframe(item)
                        self._consumed_through = max(
                            self._consumed_through,
                            item.frame_id,
                        )
                except Exception as error:
                    try:
                        event = self._coordinator_failure(
                            frame_id=frame_id,
                            error=error,
                        )
                        self._consumed_through = max(
                            self._consumed_through,
                            frame_id,
                        )
                    except Exception as fatal:
                        fatal_error = str(fatal)
                        self._last_error = fatal_error
                        processed_all_gaps = not any(
                            isinstance(remaining[3], CaptureGapEvent)
                            for remaining in ordered[index:]
                        )
                        break
                events.append(event)

            if processed_all_gaps:
                self._observed_gap_revision = batch.gap_revision

            acknowledgement_committed = False
            acknowledgement_stale = False
            if not fatal_error and processed_all_gaps and events:
                expected_revision = batch.gap_revision
                for _ in range(1_000):
                    acknowledgement_committed = (
                        self.ring.acknowledge_consumer_batch(
                            self._consumed_through,
                            expected_gap_revision=expected_revision,
                        )
                    )
                    if acknowledgement_committed:
                        break
                    acknowledgement_stale = True
                    late_batch = self.ring.consumer_batch(
                        after_frame_id=self._consumed_through,
                        after_gap_revision=expected_revision,
                    )
                    if not late_batch.gap_events:
                        fatal_error = (
                            "ring gap revision changed without a ledger event"
                        )
                        break
                    try:
                        for late_gap in sorted(
                            late_batch.gap_events,
                            key=lambda event: (
                                event.gap.first_frame_id,
                                event.revision,
                            ),
                        ):
                            events.append(self._process_gap_event(late_gap))
                    except Exception as error:
                        fatal_error = (
                            "late-gap fail-closed invalidation failed: "
                            f"{type(error).__name__}: {error}"
                        )
                        self._last_error = fatal_error
                        break
                    expected_revision = late_batch.gap_revision
                    self._observed_gap_revision = expected_revision
                else:
                    fatal_error = (
                        "gap stream did not stabilize before ACK"
                    )
                    self._last_error = fatal_error
                if acknowledgement_stale:
                    self._stale_acknowledgements += 1

            with self._state_lock:
                self._cycles += 1
                self._event_count += len(events)
                cycle = WorkerCycle(
                    events=tuple(events),
                    observed_gap_revision=self._observed_gap_revision,
                    consumed_through_frame_id=self._consumed_through,
                    acknowledged_through_frame_id=(
                        self.ring.acknowledged_through
                    ),
                    acknowledgement_committed=acknowledgement_committed,
                    acknowledgement_stale=acknowledgement_stale,
                    fatal_error=fatal_error,
                )
            return cycle

    def drain(self, *, max_cycles: int = 10_000) -> tuple[WorkerCycle, ...]:
        """Synchronously consume until the current ring/ledger is empty."""

        if max_cycles <= 0:
            raise ValueError("max_cycles must be positive")
        cycles: list[WorkerCycle] = []
        for _ in range(max_cycles):
            cycle = self.process_once()
            cycles.append(cycle)
            if cycle.fatal_error:
                raise RuntimeError(cycle.fatal_error)
            if cycle.idle:
                return tuple(cycles)
        raise RuntimeError("public history worker drain exceeded max_cycles")

    def _run(self) -> None:
        try:
            while True:
                if self._stop_event.is_set() and not self._drain_on_stop:
                    break
                cycle = self.process_once()
                if cycle.fatal_error:
                    raise RuntimeError(cycle.fatal_error)
                if self._stop_event.is_set() and self._drain_on_stop:
                    if cycle.idle:
                        break
                    continue
                if cycle.idle:
                    self.ring.wait_for_pending(
                        after_frame_id=self._consumed_through,
                        after_gap_revision=self._observed_gap_revision,
                        timeout_seconds=self.idle_wait_seconds,
                        stop_event=self._stop_event,
                    )
        except Exception as error:
            with self._state_lock:
                self._background_error = (
                    f"{type(error).__name__}: {error}"
                )
                self._last_error = self._background_error
        finally:
            with self._state_lock:
                self._running = False

    def start(self, *, thread_name: str = "public-history-worker") -> None:
        """Start the optional single background consumer."""

        with self._state_lock:
            if self._started:
                raise RuntimeError("public history worker cannot be restarted")
            self._started = True
            self._running = True
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name=thread_name,
                daemon=True,
            )
            self._thread.start()

    def stop(
        self,
        *,
        drain: bool = True,
        timeout_seconds: float = 5.0,
    ) -> None:
        """Stop background processing, optionally draining retained input."""

        if timeout_seconds < 0:
            raise ValueError("timeout_seconds cannot be negative")
        with self._state_lock:
            self._drain_on_stop = bool(drain)
            thread = self._thread
        if thread is None:
            if drain:
                self.drain()
            return
        self._stop_event.set()
        self.ring.wake_consumers()
        thread.join(timeout_seconds)
        if thread.is_alive():
            raise TimeoutError("public history worker did not stop in time")
        with self._state_lock:
            error = self._background_error
        if error:
            raise RuntimeError(
                f"public history worker failed: {error}"
            )

    def read_if_caught_up(
        self,
        reader: Callable[[], object],
    ) -> PublicHistoryCaughtUpRead:
        """Read coordinator state only if no ring input is awaiting this worker.

        The lock order matches :meth:`process_once`.  The ring executes
        ``reader`` while its condition remains held, so the returned value and
        the no-pending/no-unseen-gap proof describe one atomic boundary.
        """

        if not callable(reader):
            raise TypeError("reader must be callable")
        with self._process_lock, self.coordinator.transaction():
            with self._state_lock:
                running = self._running
                background_error = self._background_error
                consumed_through = self._consumed_through
                observed_gap_revision = self._observed_gap_revision

            def convert(
                ring_read: CaughtUpRingRead,
                *,
                caught_up: bool | None = None,
                reason: str | None = None,
                keep_value: bool = True,
            ) -> PublicHistoryCaughtUpRead:
                return PublicHistoryCaughtUpRead(
                    caught_up=(
                        ring_read.caught_up
                        if caught_up is None
                        else caught_up
                    ),
                    reason=(
                        ring_read.reason if reason is None else reason
                    ),
                    consumed_through_frame_id=(
                        ring_read.consumed_through_frame_id
                    ),
                    observed_gap_revision=(
                        ring_read.observed_gap_revision
                    ),
                    acknowledged_through_frame_id=(
                        ring_read.acknowledged_through_frame_id
                    ),
                    current_gap_revision=ring_read.current_gap_revision,
                    value=ring_read.value if keep_value else None,
                )

            ring_read = self.ring.read_if_caught_up(
                consumed_through_frame_id=consumed_through,
                observed_gap_revision=observed_gap_revision,
                reader=(reader if running and not background_error else lambda: None),
            )
            if background_error:
                return convert(
                    ring_read,
                    caught_up=False,
                    reason="public-history worker failed: "
                    + background_error,
                    keep_value=False,
                )
            if not running:
                return convert(
                    ring_read,
                    caught_up=False,
                    reason="public-history worker is not running",
                    keep_value=False,
                )
            with self._state_lock:
                if self._background_error:
                    return convert(
                        ring_read,
                        caught_up=False,
                        reason="public-history worker failed: "
                        + self._background_error,
                        keep_value=False,
                    )
                if not self._running:
                    return convert(
                        ring_read,
                        caught_up=False,
                        reason="public-history worker stopped during "
                        "the caught-up read",
                        keep_value=False,
                    )
            return convert(ring_read)

    def view(self) -> PublicHistoryWorkerView:
        with self._process_lock:
            with self._state_lock:
                return PublicHistoryWorkerView(
                    running=self._running,
                    consumed_through_frame_id=self._consumed_through,
                    observed_gap_revision=self._observed_gap_revision,
                    cycles=self._cycles,
                    events=self._event_count,
                    decoder_failures=self._decoder_failures,
                    decoder_poisoned=self._decoder_poisoned,
                    stale_acknowledgements=self._stale_acknowledgements,
                    last_error=self._last_error,
                    background_error=self._background_error,
                )


__all__ = [
    "DecodeDisposition",
    "DecodeResult",
    "PublicHistoryWorker",
    "PublicHistoryCaughtUpRead",
    "PublicHistoryWorkerView",
    "PublicObservationDecoder",
    "WorkerCycle",
    "WorkerEvent",
    "WorkerEventDisposition",
]
