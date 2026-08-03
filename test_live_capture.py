from __future__ import annotations

from datetime import datetime, timedelta, timezone
import threading
import time
import unittest

from PIL import Image, ImageDraw

from live_capture import (
    CaptureFrame,
    CaptureGap,
    CaptureGapKind,
    ContinuousCaptureService,
    FrameSignature,
    Keyframe,
    KeyframeDetector,
    KeyframeReason,
    KeyframeRing,
)


BASE_TIME = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)


def solid(value: int, size: tuple[int, int] = (96, 64)) -> Image.Image:
    return Image.new("RGB", size, (value, value, value))


def marked(
    value: int,
    *,
    box: tuple[int, int, int, int] = (12, 12, 72, 48),
    size: tuple[int, int] = (96, 64),
) -> Image.Image:
    image = solid(0, size)
    ImageDraw.Draw(image).rectangle(box, fill=(value, value, value))
    return image


def frame(
    frame_id: int,
    value: int,
    *,
    monotonic_ns: int | None = None,
    extra_regions: dict[str, Image.Image] | None = None,
) -> CaptureFrame:
    regions = {"table": marked(value)}
    if extra_regions:
        regions.update(extra_regions)
    signature = FrameSignature.from_regions(regions, sample_size=(24, 16))
    return CaptureFrame(
        frame_id=frame_id,
        monotonic_ns=(
            frame_id * 100_000_000
            if monotonic_ns is None
            else monotonic_ns
        ),
        captured_at=BASE_TIME + timedelta(milliseconds=frame_id * 100),
        regions=regions,
        signature=signature,
    )


def keyframe(frame_id: int) -> Keyframe:
    return Keyframe(frame(frame_id, frame_id * 20), KeyframeReason.CHANGE)


class FrameSignatureTests(unittest.TestCase):
    def test_signature_is_deterministic_and_detects_local_change(self):
        first = FrameSignature.from_regions(
            {"seat": marked(40), "board": solid(10)},
            sample_size=(24, 16),
        )
        identical = FrameSignature.from_regions(
            {"board": solid(10), "seat": marked(40)},
            sample_size=(24, 16),
        )
        changed = FrameSignature.from_regions(
            {"seat": marked(220), "board": solid(10)},
            sample_size=(24, 16),
        )

        self.assertEqual(first.digest, identical.digest)
        self.assertNotEqual(first.digest, changed.digest)
        by_name = {item.name: item for item in first.compare(changed)}
        self.assertEqual(0.0, by_name["board"].pixel_fraction)
        self.assertGreater(by_name["seat"].pixel_fraction, 0.1)

    def test_region_addition_is_a_full_change(self):
        first = FrameSignature.from_regions({"board": solid(10)})
        second = FrameSignature.from_regions(
            {"board": solid(10), "seat": solid(20)}
        )

        changes = {item.name: item for item in first.compare(second)}
        self.assertEqual(1.0, changes["seat"].pixel_fraction)
        self.assertEqual(1.0, changes["seat"].edge_fraction)


class KeyframeDetectorTests(unittest.TestCase):
    def detector(self, **overrides) -> KeyframeDetector:
        config = {
            "pixel_fraction_threshold": 0.05,
            "edge_fraction_threshold": 0.05,
            "stable_frames": 2,
            "heartbeat_interval_ns": None,
        }
        config.update(overrides)
        return KeyframeDetector(**config)

    def test_selects_first_change_peak_and_stable_frame(self):
        detector = self.detector()

        self.assertEqual(
            [KeyframeReason.BASELINE],
            [item.reason for item in detector.observe(frame(1, 0))],
        )
        changed = detector.observe(frame(2, 50))
        self.assertEqual([2], [item.frame_id for item in changed])
        self.assertEqual(KeyframeReason.CHANGE, changed[0].reason)

        self.assertEqual((), detector.observe(frame(3, 220)))
        self.assertEqual((), detector.observe(frame(4, 220)))
        completed = detector.observe(frame(5, 220))

        self.assertEqual([3, 5], [item.frame_id for item in completed])
        self.assertEqual(
            [KeyframeReason.PEAK, KeyframeReason.STABLE],
            [item.reason for item in completed],
        )
        self.assertIn("table", completed[0].changed_regions)

    def test_ignored_region_change_does_not_create_keyframe(self):
        detector = self.detector(ignored_regions=("timer",))
        baseline = frame(1, 20, extra_regions={"timer": marked(20)})
        timer_changed = frame(2, 20, extra_regions={"timer": marked(240)})

        detector.observe(baseline)
        self.assertEqual((), detector.observe(timer_changed))

    def test_heartbeat_preserves_unchanged_scene(self):
        detector = self.detector(heartbeat_interval_ns=500_000_000)
        detector.observe(frame(1, 20, monotonic_ns=0))
        self.assertEqual(
            (),
            detector.observe(frame(2, 20, monotonic_ns=400_000_000)),
        )
        heartbeat = detector.observe(
            frame(3, 20, monotonic_ns=500_000_000)
        )

        self.assertEqual(1, len(heartbeat))
        self.assertEqual(KeyframeReason.HEARTBEAT, heartbeat[0].reason)

    def test_flush_retains_peak_from_unfinished_burst(self):
        detector = self.detector(stable_frames=3)
        detector.observe(frame(1, 0))
        detector.observe(frame(2, 40))
        detector.observe(frame(3, 220))
        flushed = detector.flush()

        self.assertEqual([3], [item.frame_id for item in flushed])
        self.assertEqual(KeyframeReason.PEAK, flushed[0].reason)

    def test_rejects_out_of_order_frames(self):
        detector = self.detector()
        detector.observe(frame(2, 0))
        with self.assertRaisesRegex(ValueError, "increasing frame IDs"):
            detector.observe(frame(1, 20))

    def test_zero_threshold_does_not_mark_identical_frames_changed(self):
        detector = KeyframeDetector(
            pixel_fraction_threshold=0,
            edge_fraction_threshold=0,
            stable_frames=1,
            heartbeat_interval_ns=None,
        )

        first = detector.observe(frame(1, 20, monotonic_ns=0))
        second = detector.observe(
            frame(2, 20, monotonic_ns=10_000_000)
        )

        self.assertEqual(KeyframeReason.BASELINE, first[0].reason)
        self.assertEqual((), second)


class KeyframeRingTests(unittest.TestCase):
    def test_acknowledged_frames_are_released_before_new_offer(self):
        ring = KeyframeRing(2)
        self.assertIsNone(ring.offer(keyframe(1)))
        self.assertIsNone(ring.offer(keyframe(2)))
        ring.acknowledge(1)
        self.assertIsNone(ring.offer(keyframe(3)))

        self.assertEqual([2, 3], [item.frame_id for item in ring.pending()])
        self.assertEqual(1, ring.acknowledged_through)
        self.assertEqual((), ring.gaps())

    def test_unacknowledged_overflow_is_explicit_and_does_not_evict(self):
        ring = KeyframeRing(2)
        ring.offer(keyframe(1))
        ring.offer(keyframe(2))

        gap = ring.offer(keyframe(3))

        self.assertIsNotNone(gap)
        assert gap is not None
        self.assertEqual(CaptureGapKind.QUEUE_OVERFLOW, gap.kind)
        self.assertEqual((3, 3), (gap.first_frame_id, gap.last_frame_id))
        self.assertEqual([1, 2], [item.frame_id for item in ring.pending()])
        self.assertEqual((gap,), ring.gaps())

    def test_adjacent_overflow_gaps_are_coalesced_without_losing_count(self):
        ring = KeyframeRing(1)
        ring.offer(keyframe(1))
        ring.offer(keyframe(2))
        ring.offer(keyframe(3))

        gaps = ring.gaps()
        self.assertEqual(1, len(gaps))
        self.assertEqual((2, 3), (gaps[0].first_frame_id, gaps[0].last_frame_id))
        self.assertEqual(2, gaps[0].dropped_count)

    def test_ack_watermark_cannot_move_backwards(self):
        ring = KeyframeRing(2)
        ring.acknowledge(4)
        with self.assertRaisesRegex(ValueError, "cannot move backwards"):
            ring.acknowledge(3)

    def test_frame_already_covered_by_ack_is_not_retained(self):
        ring = KeyframeRing(2)
        ring.acknowledge(2)
        self.assertIsNone(ring.offer(keyframe(1)))
        self.assertEqual((), ring.pending())

    def test_consumer_batch_atomically_exposes_keyframes_and_gap_ledger(self):
        ring = KeyframeRing(3)
        ring.offer(keyframe(1))
        ring.record_gap(
            CaptureGap(
                kind=CaptureGapKind.CAPTURE_ERROR,
                first_frame_id=2,
                last_frame_id=2,
                detected_monotonic_ns=2,
                reason="capture failed",
            )
        )

        batch = ring.consumer_batch()

        self.assertEqual([1], [item.frame_id for item in batch.keyframes])
        self.assertEqual(1, batch.gap_revision)
        self.assertEqual(
            [(1, 2)],
            [
                (event.revision, event.gap.first_frame_id)
                for event in batch.gap_events
            ],
        )

    def test_consumer_ack_fails_if_a_gap_arrived_after_snapshot(self):
        ring = KeyframeRing(3)
        ring.offer(keyframe(1))
        batch = ring.consumer_batch()
        ring.record_gap(
            CaptureGap(
                kind=CaptureGapKind.CAPTURE_ERROR,
                first_frame_id=1,
                last_frame_id=1,
                detected_monotonic_ns=2,
                reason="late capture failure",
            )
        )

        acknowledged = ring.acknowledge_consumer_batch(
            1,
            expected_gap_revision=batch.gap_revision,
        )

        self.assertFalse(acknowledged)
        self.assertEqual(0, ring.acknowledged_through)
        self.assertEqual([1], [item.frame_id for item in ring.pending()])

    def test_gap_ledger_keeps_coalesced_extensions_as_new_events(self):
        ring = KeyframeRing(1)
        ring.offer(keyframe(1))
        ring.offer(keyframe(2))
        first = ring.consumer_batch()
        ring.offer(keyframe(3))

        second = ring.consumer_batch(
            after_gap_revision=first.gap_revision,
        )

        self.assertEqual(1, len(first.gap_events))
        self.assertEqual(1, len(second.gap_events))
        self.assertEqual(3, second.gap_events[0].gap.first_frame_id)
        self.assertEqual((2, 3), (
            ring.gaps()[0].first_frame_id,
            ring.gaps()[0].last_frame_id,
        ))

    def test_wait_for_pending_wakes_for_offer(self):
        ring = KeyframeRing(2)

        def publish():
            ring.offer(keyframe(1))

        publisher = threading.Thread(target=publish)
        publisher.start()
        pending = ring.wait_for_pending(timeout_seconds=1)
        publisher.join()

        self.assertEqual([1], [item.frame_id for item in pending])


class FakeClock:
    def __init__(self, *, stop_after_waits: int | None = None):
        self.now_ns = 0
        self.waits = 0
        self.stop_after_waits = stop_after_waits

    def monotonic_ns(self) -> int:
        return self.now_ns

    def utc_now(self) -> datetime:
        return BASE_TIME + timedelta(microseconds=self.now_ns / 1_000)

    def wait(self, event: threading.Event, timeout_seconds: float) -> bool:
        if event.is_set():
            return True
        self.now_ns += round(timeout_seconds * 1_000_000_000)
        self.waits += 1
        return bool(
            self.stop_after_waits is not None
            and self.waits >= self.stop_after_waits
        )


class SequenceSource:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = 0

    def capture(self):
        self.calls += 1
        value = next(self.values)
        if isinstance(value, Exception):
            raise value
        return {"table": marked(value)}


class ContinuousCaptureServiceTests(unittest.TestCase):
    def detector(self) -> KeyframeDetector:
        return KeyframeDetector(
            pixel_fraction_threshold=0.05,
            edge_fraction_threshold=0.05,
            stable_frames=1,
            heartbeat_interval_ns=None,
        )

    def test_capture_once_uses_injected_source_and_clock(self):
        source_image = marked(30)

        class Source:
            def capture(self):
                return {"table": source_image}

        clock = FakeClock()
        service = ContinuousCaptureService(
            Source(),
            detector=self.detector(),
            ring=KeyframeRing(4),
            clock=clock,
        )
        result = service.capture_once()
        source_image.paste((255, 255, 255), (0, 0, 96, 64))

        self.assertIsNotNone(result.frame)
        assert result.frame is not None
        self.assertEqual(1, result.frame.frame_id)
        self.assertEqual(BASE_TIME, result.frame.captured_at)
        self.assertEqual((0, 0, 0), result.frame.regions["table"].getpixel((0, 0)))
        self.assertEqual([1], [item.frame_id for item in service.ring.pending()])
        self.assertEqual((), result.gaps)

    def test_capture_failure_records_gap_and_next_frame_keeps_order(self):
        source = SequenceSource([RuntimeError("screen denied"), 40])
        clock = FakeClock()
        service = ContinuousCaptureService(
            source,
            detector=self.detector(),
            ring=KeyframeRing(4),
            clock=clock,
        )

        failed = service.capture_once()
        succeeded = service.capture_once()

        self.assertIsNone(failed.frame)
        self.assertEqual(CaptureGapKind.CAPTURE_ERROR, failed.gaps[0].kind)
        self.assertEqual(1, failed.gaps[0].first_frame_id)
        self.assertIsNotNone(succeeded.frame)
        assert succeeded.frame is not None
        self.assertEqual(2, succeeded.frame.frame_id)
        self.assertEqual([2], [item.frame_id for item in service.ring.pending()])

    def test_service_overflow_surfaces_gap_in_cycle(self):
        source = SequenceSource([0, 200])
        service = ContinuousCaptureService(
            source,
            detector=self.detector(),
            ring=KeyframeRing(1),
            clock=FakeClock(),
        )
        first = service.capture_once()
        second = service.capture_once()

        self.assertEqual((), first.gaps)
        self.assertEqual(1, len(second.gaps))
        self.assertEqual(
            CaptureGapKind.QUEUE_OVERFLOW,
            second.gaps[0].kind,
        )
        self.assertEqual([1], [item.frame_id for item in service.ring.pending()])

    def test_background_service_uses_clock_schedule_and_stops_cleanly(self):
        source = SequenceSource([0, 60, 220])
        clock = FakeClock(stop_after_waits=3)
        service = ContinuousCaptureService(
            source,
            detector=KeyframeDetector(
                pixel_fraction_threshold=0.05,
                edge_fraction_threshold=0.05,
                stable_frames=3,
                heartbeat_interval_ns=None,
            ),
            ring=KeyframeRing(8),
            clock=clock,
            fps=10,
        )

        service.start(thread_name="capture-test")
        thread = service._thread
        assert thread is not None
        thread.join(timeout=1)
        service.stop(timeout_seconds=1)

        self.assertFalse(service.running)
        self.assertEqual(3, source.calls)
        self.assertEqual(300_000_000, clock.now_ns)
        self.assertEqual([1, 2, 3], [
            item.frame_id for item in service.ring.pending()
        ])

    def test_service_cannot_be_restarted(self):
        source = SequenceSource([0])
        clock = FakeClock(stop_after_waits=1)
        service = ContinuousCaptureService(source, clock=clock)
        service.start()
        thread = service._thread
        assert thread is not None
        thread.join(timeout=1)
        with self.assertRaisesRegex(RuntimeError, "cannot be restarted"):
            service.start()

    def test_sync_capture_cannot_switch_to_background_and_closes_on_stop(self):
        class CloseableSource(SequenceSource):
            def __init__(self):
                super().__init__([0])
                self.owner = None
                self.closed = False

            def capture(self):
                self.owner = threading.get_ident()
                return super().capture()

            def close(self):
                if threading.get_ident() != self.owner:
                    raise RuntimeError("wrong close thread")
                self.closed = True

        source = CloseableSource()
        service = ContinuousCaptureService(
            source,
            detector=self.detector(),
            clock=FakeClock(),
        )
        service.capture_once()

        with self.assertRaisesRegex(
            RuntimeError,
            "cannot start after synchronous capture",
        ):
            service.start()
        service.stop()

        self.assertTrue(source.closed)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            service.capture_once()

    def test_record_gap_accepts_external_gap(self):
        ring = KeyframeRing(2)
        gap = CaptureGap(
            kind=CaptureGapKind.CAPTURE_ERROR,
            first_frame_id=4,
            last_frame_id=4,
            detected_monotonic_ns=10,
            reason="capture failed: OSError",
        )
        self.assertEqual(gap, ring.record_gap(gap))
        self.assertEqual((gap,), ring.gaps())

    def test_out_of_order_external_gaps_do_not_corrupt_frame_ranges(self):
        ring = KeyframeRing(2)
        later = CaptureGap(
            kind=CaptureGapKind.CAPTURE_ERROR,
            first_frame_id=10,
            last_frame_id=10,
            detected_monotonic_ns=20,
            reason="capture failed",
        )
        earlier = CaptureGap(
            kind=CaptureGapKind.CAPTURE_ERROR,
            first_frame_id=1,
            last_frame_id=1,
            detected_monotonic_ns=10,
            reason="capture failed",
        )

        ring.record_gap(later)
        ring.record_gap(earlier)

        self.assertEqual(
            [(1, 1, 1), (10, 10, 1)],
            [
                (gap.first_frame_id, gap.last_frame_id, gap.dropped_count)
                for gap in ring.gaps()
            ],
        )

    def test_historical_gap_does_not_make_default_wait_spin(self):
        ring = KeyframeRing(2)
        ring.record_gap(
            CaptureGap(
                kind=CaptureGapKind.CAPTURE_ERROR,
                first_frame_id=1,
                last_frame_id=1,
                detected_monotonic_ns=10,
                reason="capture failed",
            )
        )
        started = time.monotonic()

        pending = ring.wait_for_pending(timeout_seconds=0.03)

        self.assertEqual((), pending)
        self.assertGreaterEqual(time.monotonic() - started, 0.02)

    def test_coalesced_gap_advances_wait_revision(self):
        ring = KeyframeRing(1)
        ring.offer(keyframe(1))
        ring.offer(keyframe(2))
        revision = ring.gap_revision

        def publish_coalesced_gap():
            time.sleep(0.02)
            ring.offer(keyframe(3))

        publisher = threading.Thread(target=publish_coalesced_gap)
        publisher.start()
        started = time.monotonic()
        pending = ring.wait_for_pending(
            after_frame_id=1,
            after_gap_revision=revision,
            timeout_seconds=0.2,
        )
        elapsed = time.monotonic() - started
        publisher.join(timeout=1)

        self.assertEqual((), pending)
        self.assertEqual(revision + 1, ring.gap_revision)
        self.assertLess(elapsed, 0.15)

    def test_gap_snapshot_binds_history_and_revision_atomically(self):
        ring = KeyframeRing(1)
        ring.offer(keyframe(1))
        ring.offer(keyframe(2))

        snapshot = ring.gap_snapshot()

        self.assertEqual(ring.gap_revision, snapshot.revision)
        self.assertEqual(ring.gaps(), snapshot.gaps)


if __name__ == "__main__":
    unittest.main()
