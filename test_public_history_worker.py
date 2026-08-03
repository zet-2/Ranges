from __future__ import annotations

from datetime import datetime, timezone
import threading
import time
import unittest

from PIL import Image

from live_capture import (
    CaptureFrame,
    CaptureGap,
    CaptureGapKind,
    FrameSignature,
    Keyframe,
    KeyframeReason,
    KeyframeRing,
)
from public_history_pipeline import (
    CandidateDisposition,
    PublicHistoryCoordinator,
    PublicHistoryStatus,
)
from public_history_worker import (
    DecodeResult,
    PublicHistoryWorker,
    WorkerEventDisposition,
)
from test_gto_event_collector import snapshot_for_prefix
from test_gto_hand_history import heads_up_to_turn_history


def keyframe(frame_id: int) -> Keyframe:
    regions = {"table": Image.new("RGB", (8, 8), (frame_id, 0, 0))}
    signature = FrameSignature.from_regions(regions, sample_size=(4, 4))
    return Keyframe(
        frame=CaptureFrame(
            frame_id=frame_id,
            monotonic_ns=frame_id,
            captured_at=datetime(2026, 7, 30, tzinfo=timezone.utc),
            regions=regions,
            signature=signature,
        ),
        reason=KeyframeReason.CHANGE,
    )


def capture_gap(
    frame_id: int,
    reason: str = "source lost a frame",
) -> CaptureGap:
    return CaptureGap(
        kind=CaptureGapKind.CAPTURE_ERROR,
        first_frame_id=frame_id,
        last_frame_id=frame_id,
        detected_monotonic_ns=frame_id,
        reason=reason,
    )


class SequenceDecoder:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def decode(self, frame, *, timeout_seconds):
        self.calls.append((frame.frame_id, timeout_seconds))
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class RecordingCoordinator(PublicHistoryCoordinator):
    def __init__(self, ring):
        super().__init__()
        self.ring = ring
        self.calls = []

    def submit_candidate(self, frame_id, snapshot):
        self.calls.append(("submit", frame_id, self.ring.acknowledged_through))
        return super().submit_candidate(frame_id, snapshot)

    def advance_decoder_watermark(self, through_frame_id):
        self.calls.append(
            ("advance", through_frame_id, self.ring.acknowledged_through)
        )
        return super().advance_decoder_watermark(through_frame_id)

    def invalidate_gap(self, reason, *, frame_id=None):
        self.calls.append(
            ("invalidate", frame_id, self.ring.acknowledged_through)
        )
        return super().invalidate_gap(reason, frame_id=frame_id)


class PublicHistoryWorkerTests(unittest.TestCase):
    def setUp(self):
        self.history = heads_up_to_turn_history()

    def worker(self, ring, decoder, coordinator=None):
        return PublicHistoryWorker(
            ring=ring,
            coordinator=coordinator or PublicHistoryCoordinator(),
            decoder=decoder,
            decode_timeout_seconds=0.25,
            idle_wait_seconds=0.01,
        )

    def test_observations_are_committed_sequentially_before_ack(self):
        ring = KeyframeRing(4)
        ring.offer(keyframe(1))
        ring.offer(keyframe(2))
        coordinator = RecordingCoordinator(ring)
        decoder = SequenceDecoder(
            DecodeResult.observation(snapshot_for_prefix(self.history, 0)),
            DecodeResult.observation(snapshot_for_prefix(self.history, 1)),
        )
        worker = self.worker(ring, decoder, coordinator)

        cycle = worker.process_once()

        self.assertEqual(
            [
                WorkerEventDisposition.OBSERVATION_ACCEPTED,
                WorkerEventDisposition.OBSERVATION_ACCEPTED,
            ],
            [event.disposition for event in cycle.events],
        )
        self.assertEqual(
            [CandidateDisposition.ANCHORED, CandidateDisposition.ADVANCED],
            [
                event.candidate_result.disposition
                for event in cycle.events
            ],
        )
        self.assertTrue(cycle.acknowledgement_committed)
        self.assertEqual(2, ring.acknowledged_through)
        self.assertTrue(all(call[2] == 0 for call in coordinator.calls))
        ready = coordinator.readiness(
            hand_id=self.history.hand_id,
            through_frame_id=2,
        )
        self.assertTrue(ready.ready)

    def test_transient_advances_decoder_before_ack(self):
        ring = KeyframeRing(2)
        ring.offer(keyframe(4))
        coordinator = RecordingCoordinator(ring)
        worker = self.worker(
            ring,
            SequenceDecoder(DecodeResult.transient("animation")),
            coordinator,
        )

        cycle = worker.process_once()

        self.assertEqual(
            WorkerEventDisposition.TRANSIENT,
            cycle.events[0].disposition,
        )
        self.assertEqual(4, coordinator.decoder_watermark)
        self.assertEqual([("advance", 4, 0)], coordinator.calls)
        self.assertEqual(4, ring.acknowledged_through)

    def test_explicit_decoder_gap_invalidates_before_ack(self):
        ring = KeyframeRing(2)
        ring.offer(keyframe(3))
        coordinator = RecordingCoordinator(ring)
        worker = self.worker(
            ring,
            SequenceDecoder(DecodeResult.gap("cards obscured")),
            coordinator,
        )

        cycle = worker.process_once()

        self.assertEqual(
            WorkerEventDisposition.DECODER_GAP,
            cycle.events[0].disposition,
        )
        self.assertEqual(PublicHistoryStatus.GAPPED, coordinator.status)
        self.assertEqual([("invalidate", 3, 0)], coordinator.calls)
        self.assertEqual(3, ring.acknowledged_through)

    def test_capture_gaps_and_keyframes_are_merged_by_frame_id(self):
        ring = KeyframeRing(4)
        ring.offer(keyframe(1))
        ring.offer(keyframe(3))
        ring.record_gap(capture_gap(2))
        coordinator = RecordingCoordinator(ring)
        decoder = SequenceDecoder(
            DecodeResult.observation(snapshot_for_prefix(self.history, 0)),
            DecodeResult.observation(snapshot_for_prefix(self.history, 1)),
        )
        worker = self.worker(ring, decoder, coordinator)

        cycle = worker.process_once()

        self.assertEqual(
            [
                WorkerEventDisposition.OBSERVATION_ACCEPTED,
                WorkerEventDisposition.CAPTURE_GAP,
                WorkerEventDisposition.OBSERVATION_IGNORED,
            ],
            [event.disposition for event in cycle.events],
        )
        self.assertEqual(
            ["submit", "advance", "invalidate", "submit", "advance"],
            [call[0] for call in coordinator.calls],
        )
        self.assertEqual(PublicHistoryStatus.GAPPED, coordinator.status)
        self.assertEqual(3, ring.acknowledged_through)

    def test_rejected_trusted_observation_invalidates_hand(self):
        ring = KeyframeRing(2)
        ring.offer(keyframe(1))
        malformed = snapshot_for_prefix(self.history, 0)
        malformed.board_state.community_cards = ["Ah"]
        coordinator = RecordingCoordinator(ring)
        worker = self.worker(
            ring,
            SequenceDecoder(DecodeResult.observation(malformed)),
            coordinator,
        )

        cycle = worker.process_once()

        event = cycle.events[0]
        self.assertEqual(
            WorkerEventDisposition.OBSERVATION_REJECTED,
            event.disposition,
        )
        self.assertEqual(
            CandidateDisposition.REJECTED,
            event.candidate_result.disposition,
        )
        self.assertEqual(PublicHistoryStatus.GAPPED, coordinator.status)
        self.assertEqual(
            ["submit", "advance", "invalidate"],
            [call[0] for call in coordinator.calls],
        )
        self.assertTrue(all(call[2] == 0 for call in coordinator.calls))
        self.assertEqual(1, ring.acknowledged_through)

    def test_timeout_and_decoder_error_are_fail_closed_and_observable(self):
        for error, expected in (
            (TimeoutError("provider deadline"), WorkerEventDisposition.DECODER_TIMEOUT),
            (RuntimeError("bad response"), WorkerEventDisposition.DECODER_ERROR),
        ):
            with self.subTest(error=type(error).__name__):
                ring = KeyframeRing(2)
                ring.offer(keyframe(1))
                coordinator = RecordingCoordinator(ring)
                worker = self.worker(
                    ring,
                    SequenceDecoder(error),
                    coordinator,
                )

                cycle = worker.process_once()

                self.assertEqual(expected, cycle.events[0].disposition)
                self.assertEqual(PublicHistoryStatus.GAPPED, coordinator.status)
                self.assertEqual([("invalidate", 1, 0)], coordinator.calls)
                self.assertEqual(1, ring.acknowledged_through)
                self.assertEqual(1, worker.view().decoder_failures)
                self.assertTrue(worker.view().last_error)

    def test_coordinator_error_is_invalidated_before_ack(self):
        ring = KeyframeRing(2)
        ring.offer(keyframe(1))

        class BrokenSubmitCoordinator(RecordingCoordinator):
            def submit_candidate(inner_self, frame_id, snapshot):
                inner_self.calls.append(
                    ("submit", frame_id, ring.acknowledged_through)
                )
                raise RuntimeError("transaction unavailable")

        coordinator = BrokenSubmitCoordinator(ring)
        worker = self.worker(
            ring,
            SequenceDecoder(
                DecodeResult.observation(
                    snapshot_for_prefix(self.history, 0)
                )
            ),
            coordinator,
        )

        cycle = worker.process_once()

        self.assertEqual(
            WorkerEventDisposition.COORDINATOR_ERROR,
            cycle.events[0].disposition,
        )
        self.assertEqual(
            ["submit", "invalidate"],
            [call[0] for call in coordinator.calls],
        )
        self.assertTrue(all(call[2] == 0 for call in coordinator.calls))
        self.assertEqual(PublicHistoryStatus.GAPPED, coordinator.status)
        self.assertEqual(1, ring.acknowledged_through)

    def test_failed_fail_closed_invalidation_is_fatal_and_never_acks(self):
        ring = KeyframeRing(2)
        ring.offer(keyframe(1))

        class BrokenCoordinator(PublicHistoryCoordinator):
            def submit_candidate(inner_self, frame_id, snapshot):
                raise RuntimeError("transaction unavailable")

            def invalidate_gap(inner_self, reason, *, frame_id=None):
                raise RuntimeError("invalidation unavailable")

        worker = self.worker(
            ring,
            SequenceDecoder(
                DecodeResult.observation(
                    snapshot_for_prefix(self.history, 0)
                )
            ),
            BrokenCoordinator(),
        )

        cycle = worker.process_once()

        self.assertTrue(cycle.fatal_error)
        self.assertEqual((), cycle.events)
        self.assertFalse(cycle.acknowledgement_committed)
        self.assertEqual(0, ring.acknowledged_through)

    def test_late_gap_is_invalidated_and_retried_before_cycle_returns(self):
        ring = KeyframeRing(2)
        ring.offer(keyframe(1))

        class GapDuringDecode:
            def decode(inner_self, frame, *, timeout_seconds):
                ring.record_gap(capture_gap(1, "late gap"))
                return DecodeResult.observation(
                    snapshot_for_prefix(self.history, 0)
                )

        coordinator = RecordingCoordinator(ring)
        worker = self.worker(ring, GapDuringDecode(), coordinator)

        first = worker.process_once()

        self.assertTrue(first.acknowledgement_stale)
        self.assertTrue(first.acknowledgement_committed)
        self.assertEqual(
            WorkerEventDisposition.CAPTURE_GAP,
            first.events[-1].disposition,
        )
        self.assertEqual(PublicHistoryStatus.GAPPED, coordinator.status)
        self.assertEqual(1, ring.acknowledged_through)
        self.assertEqual(1, worker.view().stale_acknowledgements)

        second = worker.process_once()
        self.assertTrue(second.idle)

    def test_strategy_read_cannot_observe_commit_before_late_gap_invalidation(self):
        ring = KeyframeRing(2)
        ring.offer(keyframe(1))
        decode_entered = threading.Event()
        release_decode = threading.Event()

        class GapThenBlockDecoder:
            def decode(inner_self, frame, *, timeout_seconds):
                ring.record_gap(capture_gap(1, "concurrent late gap"))
                decode_entered.set()
                release_decode.wait(1)
                return DecodeResult.observation(
                    snapshot_for_prefix(self.history, 0)
                )

        coordinator = PublicHistoryCoordinator()
        worker = self.worker(ring, GapThenBlockDecoder(), coordinator)
        process_thread = threading.Thread(target=worker.process_once)
        process_thread.start()
        self.assertTrue(decode_entered.wait(0.2))
        readiness_results = []

        def read_strategy_state():
            readiness_results.append(
                coordinator.readiness(
                    hand_id=self.history.hand_id,
                    through_frame_id=1,
                )
            )

        reader = threading.Thread(target=read_strategy_state)
        reader.start()
        time.sleep(0.01)
        self.assertTrue(reader.is_alive())
        release_decode.set()
        process_thread.join(1)
        reader.join(1)

        self.assertFalse(process_thread.is_alive())
        self.assertFalse(reader.is_alive())
        self.assertEqual(1, len(readiness_results))
        self.assertFalse(readiness_results[0].ready)
        self.assertEqual(PublicHistoryStatus.GAPPED, readiness_results[0].status)

    def test_ranged_gap_blocks_recovery_inside_the_lost_interval(self):
        ring = KeyframeRing(2)
        ring.offer(keyframe(3))
        ring.record_gap(
            CaptureGap(
                kind=CaptureGapKind.CAPTURE_ERROR,
                first_frame_id=2,
                last_frame_id=4,
                detected_monotonic_ns=4,
                reason="three-frame capture loss",
                dropped_count=3,
            )
        )
        coordinator = PublicHistoryCoordinator()
        worker = self.worker(
            ring,
            SequenceDecoder(
                DecodeResult.observation(
                    snapshot_for_prefix(self.history, 0)
                )
            ),
            coordinator,
        )

        cycle = worker.process_once()

        self.assertEqual(
            [
                WorkerEventDisposition.CAPTURE_GAP,
                WorkerEventDisposition.OBSERVATION_IGNORED,
            ],
            [event.disposition for event in cycle.events],
        )
        self.assertEqual(PublicHistoryStatus.GAPPED, coordinator.status)
        self.assertEqual(4, coordinator.view().gap_frame_id)
        self.assertEqual(4, ring.acknowledged_through)

    def test_worker_enforces_hard_timeout_and_poisons_decoder(self):
        ring = KeyframeRing(2)
        ring.offer(keyframe(1))
        release = threading.Event()

        class BlockingDecoder:
            def decode(inner_self, frame, *, timeout_seconds):
                release.wait(1)
                return DecodeResult.transient()

        coordinator = PublicHistoryCoordinator()
        worker = PublicHistoryWorker(
            ring=ring,
            coordinator=coordinator,
            decoder=BlockingDecoder(),
            decode_timeout_seconds=0.02,
            idle_wait_seconds=0.01,
        )

        started = time.monotonic()
        cycle = worker.process_once()
        elapsed = time.monotonic() - started
        release.set()

        self.assertLess(elapsed, 0.2)
        self.assertEqual(
            WorkerEventDisposition.DECODER_TIMEOUT,
            cycle.events[0].disposition,
        )
        self.assertTrue(worker.view().decoder_poisoned)
        self.assertEqual(PublicHistoryStatus.GAPPED, coordinator.status)
        self.assertEqual(1, ring.acknowledged_through)

    def test_background_stop_with_drain_consumes_retained_frames(self):
        ring = KeyframeRing(4)
        ring.offer(keyframe(1))
        ring.offer(keyframe(2))
        decoder = SequenceDecoder(
            DecodeResult.transient(),
            DecodeResult.transient(),
        )
        worker = self.worker(ring, decoder)

        worker.start()
        deadline = time.monotonic() + 1
        while ring.acknowledged_through < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        worker.stop(drain=True, timeout_seconds=1)

        self.assertEqual(2, ring.acknowledged_through)
        self.assertFalse(worker.view().running)
        self.assertEqual([1, 2], [call[0] for call in decoder.calls])

    def test_non_draining_stop_wakes_idle_worker_immediately(self):
        ring = KeyframeRing(2)
        worker = PublicHistoryWorker(
            ring=ring,
            coordinator=PublicHistoryCoordinator(),
            decoder=SequenceDecoder(),
            decode_timeout_seconds=0.25,
            idle_wait_seconds=10,
        )
        worker.start()
        time.sleep(0.01)

        started = time.monotonic()
        worker.stop(drain=False, timeout_seconds=0.2)

        self.assertLess(time.monotonic() - started, 0.2)
        self.assertFalse(worker.view().running)

    def test_non_draining_stop_finishes_current_decode_not_entire_batch(self):
        ring = KeyframeRing(4)
        ring.offer(keyframe(1))
        ring.offer(keyframe(2))
        entered = threading.Event()
        release = threading.Event()

        class SlowFirstDecoder:
            def __init__(inner_self):
                inner_self.calls = []

            def decode(inner_self, frame, *, timeout_seconds):
                inner_self.calls.append(frame.frame_id)
                entered.set()
                release.wait(1)
                return DecodeResult.transient()

        decoder = SlowFirstDecoder()
        worker = self.worker(ring, decoder)
        worker.start()
        self.assertTrue(entered.wait(0.2))
        stop_errors = []

        def stop_worker():
            try:
                worker.stop(drain=False, timeout_seconds=1)
            except Exception as error:
                stop_errors.append(error)

        stopper = threading.Thread(target=stop_worker)
        stopper.start()
        time.sleep(0.01)
        release.set()
        stopper.join(1)

        self.assertFalse(stopper.is_alive())
        self.assertEqual([], stop_errors)
        self.assertEqual([1], decoder.calls)
        self.assertEqual(1, ring.acknowledged_through)
        self.assertEqual([2], [item.frame_id for item in ring.pending()])

    def test_synchronous_drain_is_idempotent(self):
        ring = KeyframeRing(2)
        ring.offer(keyframe(1))
        worker = self.worker(
            ring,
            SequenceDecoder(DecodeResult.transient()),
        )

        first = worker.drain()
        second = worker.drain()

        self.assertEqual(2, len(first))
        self.assertEqual(1, len(second))
        self.assertTrue(first[-1].idle)
        self.assertTrue(second[-1].idle)
        self.assertEqual(1, ring.acknowledged_through)


class DecodeResultTests(unittest.TestCase):
    def test_outcomes_are_explicit_and_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "requires a snapshot"):
            DecodeResult.observation(None)
        with self.assertRaisesRegex(ValueError, "cannot include a snapshot"):
            DecodeResult(
                disposition=DecodeResult.transient().disposition,
                snapshot=object(),
            )
        with self.assertRaisesRegex(ValueError, "requires a reason"):
            DecodeResult.gap(" ")


if __name__ == "__main__":
    unittest.main()
