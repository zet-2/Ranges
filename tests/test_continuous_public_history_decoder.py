"""Tests for continuous public-history semantic decoding."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import os
from types import SimpleNamespace
import unittest
from unittest import mock

from PIL import Image

from continuous_public_history_decoder import (
    ContinuousPublicHistoryDecoder,
    ContinuousPublicHistoryRuntime,
)
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
    AcceptedPublicHistoryFrame,
    PublicHistoryCoordinator,
    PublicHistoryStatus,
)
from public_history_worker import (
    DecodeDisposition,
    DecodeResult,
    PublicHistoryWorker,
    WorkerEventDisposition,
)
from tests.test_gto_event_collector import snapshot_for_prefix
from tests.test_gto_hand_history import heads_up_to_turn_history


POSITIONS = ("UTG", "HJ", "CO", "BTN", "SB", "BB")
CAPTURED_AT = datetime(2026, 7, 30, 12, 34, 56, tzinfo=timezone.utc)


def keyframe(
    frame_id: int,
    reason: KeyframeReason = KeyframeReason.HEARTBEAT,
) -> Keyframe:
    regions = {
        "table": Image.new("RGB", (8, 8), (frame_id % 255, 0, 0))
    }
    return Keyframe(
        frame=CaptureFrame(
            frame_id=frame_id,
            monotonic_ns=frame_id,
            captured_at=CAPTURED_AT + timedelta(seconds=frame_id),
            regions=regions,
            signature=FrameSignature.from_regions(
                regions,
                sample_size=(4, 4),
            ),
        ),
        reason=reason,
    )


def table_snapshot(
    *,
    dealer: int = 3,
    street: str = "PREFLOP",
    board: tuple[str, ...] = (),
    hero_cards: tuple[str, str] = ("Ah", "Kd"),
    blind_seats: tuple[int, int] = (4, 5),
    live_seats: tuple[int, ...] = tuple(range(6)),
):
    players = []
    for seat in range(6):
        current_bet = (
            0.5
            if seat == blind_seats[0]
            else 1.0
            if seat == blind_seats[1]
            else 0.0
        )
        status = "ACTIVE" if seat in live_seats else "FOLDED"
        players.append(
            SimpleNamespace(
                seat_index=seat,
                name=POSITIONS[seat],
                username=f"P{seat}",
                stack_size=100.0 - current_bet,
                current_bet=current_bet,
                status=status,
                is_hero=seat == 4,
                is_dealer=seat == dealer,
                visible_action="",
                hole_cards=list(hero_cards) if seat == 4 else None,
            )
        )
    return SimpleNamespace(
        hand_id="provider-random-id",
        timestamp="decode-time",
        meta_info=SimpleNamespace(current_street=street),
        board_state=SimpleNamespace(
            community_cards=list(board),
            total_pot=sum(player.current_bet for player in players),
        ),
        dealer_seat_index=dealer,
        action_on_seat_index=-1,
        players=players,
        vision_error="",
    )


def enrich_recorder_snapshot(snapshot):
    snapshot.timestamp = "decode-time"
    snapshot.vision_error = ""
    for player in snapshot.players:
        player.is_hero = player.seat_index == 4
        player.hole_cards = ["Ah", "Kd"] if player.is_hero else None
        player.username = f"P{player.seat_index}"
    # The live schema proves only Hero or unknown as action-on.  Unknown must
    # never be expanded to the recorder's expected villain actor.
    snapshot.action_on_seat_index = -1
    return snapshot


class SequenceAnalyzer:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, regions, *, timeout_seconds):
        self.calls.append((regions, timeout_seconds))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class ContinuousPublicHistoryDecoderTests(unittest.TestCase):
    def decoder(self, analyzer, *, validator=lambda snapshot: ()):
        return ContinuousPublicHistoryDecoder(
            analyze_frame=analyzer,
            validate_snapshot=validator,
            hand_id_factory=lambda ordinal: f"session-hand-{ordinal}",
        )

    def test_transitional_keyframes_do_not_spend_analysis_calls(self):
        analyzer = SequenceAnalyzer()
        decoder = self.decoder(analyzer)

        results = [
            decoder.decode(keyframe(index, reason), timeout_seconds=0.5)
            for index, reason in enumerate(
                (
                    KeyframeReason.CHANGE,
                    KeyframeReason.PEAK,
                    KeyframeReason.FLUSH,
                ),
                1,
            )
        ]

        self.assertEqual([], analyzer.calls)
        self.assertTrue(
            all(
                result.disposition is DecodeDisposition.TRANSIENT
                for result in results
            )
        )

    def test_one_analysis_per_stable_frame_uses_capture_time_and_stable_id(self):
        first = table_snapshot()
        second = deepcopy(first)
        analyzer = SequenceAnalyzer(first, second)
        decoder = self.decoder(analyzer)

        decoded_first = decoder.decode(
            keyframe(1, KeyframeReason.BASELINE),
            timeout_seconds=0.75,
        )
        decoded_second = decoder.decode(
            keyframe(2, KeyframeReason.HEARTBEAT),
            timeout_seconds=0.75,
        )

        self.assertEqual(2, len(analyzer.calls))
        self.assertEqual([0.75, 0.75], [call[1] for call in analyzer.calls])
        self.assertEqual(
            "session-hand-1",
            decoded_first.snapshot.hand_id,
        )
        self.assertEqual(
            decoded_first.snapshot.hand_id,
            decoded_second.snapshot.hand_id,
        )
        self.assertEqual(
            (CAPTURED_AT + timedelta(seconds=1))
            .isoformat()
            .replace("+00:00", "Z"),
            decoded_first.snapshot.timestamp,
        )
        self.assertEqual(
            (CAPTURED_AT + timedelta(seconds=2))
            .isoformat()
            .replace("+00:00", "Z"),
            decoded_second.snapshot.timestamp,
        )
        self.assertEqual(1, decoder.hand_ordinal)

    def test_new_hand_requires_rotated_dealer_regression_and_blind_anchor(self):
        first = table_snapshot(dealer=3)
        postflop = table_snapshot(
            dealer=3,
            street="FLOP",
            board=("2c", "7d", "Jh"),
        )
        # The same Hero combo is legal in a later hand and is not an identity
        # discriminator.
        new_anchor = table_snapshot(
            dealer=4,
            blind_seats=(5, 0),
            hero_cards=("Ah", "Kd"),
        )
        analyzer = SequenceAnalyzer(first, postflop, new_anchor)
        decoder = self.decoder(analyzer)

        results = [
            decoder.decode(keyframe(frame_id), timeout_seconds=1)
            for frame_id in (1, 2, 3)
        ]

        self.assertTrue(
            all(
                result.disposition is DecodeDisposition.OBSERVATION
                for result in results
            )
        )
        self.assertEqual(
            ["session-hand-1", "session-hand-1", "session-hand-2"],
            [result.snapshot.hand_id for result in results],
        )

    def test_partial_hand_boundary_is_a_gap_not_a_new_identity(self):
        first = table_snapshot(dealer=3)
        unexplained_rotation = table_snapshot(
            dealer=4,
            blind_seats=(5, 0),
        )
        decoder = self.decoder(
            SequenceAnalyzer(first, unexplained_rotation)
        )

        accepted = decoder.decode(keyframe(1), timeout_seconds=1)
        ambiguous = decoder.decode(keyframe(2), timeout_seconds=1)

        self.assertEqual(
            DecodeDisposition.OBSERVATION,
            accepted.disposition,
        )
        self.assertEqual(DecodeDisposition.GAP, ambiguous.disposition)
        self.assertIn("ambiguous hand boundary", ambiguous.reason)
        self.assertEqual(1, decoder.hand_ordinal)

    def test_first_blind_frame_with_action_overlay_is_not_an_anchor(self):
        overlaid = table_snapshot()
        overlaid.players[0].visible_action = "FOLD"
        decoder = self.decoder(SequenceAnalyzer(overlaid))

        result = decoder.decode(keyframe(1), timeout_seconds=1)

        self.assertEqual(DecodeDisposition.GAP, result.disposition)
        self.assertIn("untouched blind anchor", result.reason)
        self.assertEqual(0, decoder.hand_ordinal)

    def test_malformed_one_player_frame_cannot_manufacture_terminal_boundary(self):
        malformed = table_snapshot(dealer=3, live_seats=(3,))
        for player in malformed.players:
            if player.seat_index != 3:
                player.status = "SITTING_OUT"
                player.current_bet = 0.0
        next_anchor = table_snapshot(
            dealer=4,
            blind_seats=(5, 0),
        )
        decoder = self.decoder(
            SequenceAnalyzer(malformed, next_anchor)
        )

        rejected = decoder.decode(keyframe(1), timeout_seconds=1)
        accepted = decoder.decode(keyframe(2), timeout_seconds=1)

        self.assertEqual(DecodeDisposition.GAP, rejected.disposition)
        self.assertIn("2..6 occupied", rejected.reason)
        self.assertEqual(
            DecodeDisposition.OBSERVATION,
            accepted.disposition,
        )
        self.assertEqual("session-hand-1", accepted.snapshot.hand_id)
        self.assertEqual(1, decoder.hand_ordinal)

    def test_provider_validation_and_vision_failures_are_explicit_gaps(self):
        for snapshot, validator, expected in (
            (
                table_snapshot(),
                lambda candidate: ("six seats are inconsistent",),
                "six seats are inconsistent",
            ),
            (
                table_snapshot(),
                lambda candidate: (),
                "provider parse failed",
            ),
            (
                table_snapshot(),
                lambda candidate: None,
                "validator returned no result",
            ),
        ):
            with self.subTest(expected=expected):
                if expected == "provider parse failed":
                    snapshot.vision_error = expected
                decoder = self.decoder(
                    SequenceAnalyzer(snapshot),
                    validator=validator,
                )

                result = decoder.decode(keyframe(1), timeout_seconds=1)

                self.assertEqual(DecodeDisposition.GAP, result.disposition)
                self.assertIn(expected, result.reason)
                self.assertEqual(0, decoder.hand_ordinal)

        error_decoder = self.decoder(
            SequenceAnalyzer(RuntimeError("provider unavailable"))
        )
        error = error_decoder.decode(keyframe(1), timeout_seconds=1)
        self.assertEqual(DecodeDisposition.GAP, error.disposition)
        self.assertIn("provider unavailable", error.reason)

    def test_declared_timeout_propagates_to_worker_timeout_path(self):
        decoder = self.decoder(
            SequenceAnalyzer(TimeoutError("request deadline"))
        )

        with self.assertRaisesRegex(TimeoutError, "request deadline"):
            decoder.decode(keyframe(1), timeout_seconds=0.25)

    def test_worker_and_coordinator_receive_stable_identity_and_atomic_pair(self):
        target = heads_up_to_turn_history()
        snapshots = [
            enrich_recorder_snapshot(snapshot_for_prefix(target, count))
            for count in (0, 1)
        ]
        analyzer = SequenceAnalyzer(*snapshots)
        decoder = self.decoder(analyzer)
        ring = KeyframeRing(4)
        ring.offer(keyframe(1, KeyframeReason.BASELINE))
        ring.offer(keyframe(2, KeyframeReason.HEARTBEAT))
        coordinator = PublicHistoryCoordinator()
        worker = PublicHistoryWorker(
            ring=ring,
            coordinator=coordinator,
            decoder=decoder,
            decode_timeout_seconds=0.25,
            idle_wait_seconds=0.01,
        )

        cycle = worker.process_once()

        self.assertEqual(
            [
                WorkerEventDisposition.OBSERVATION_ACCEPTED,
                WorkerEventDisposition.OBSERVATION_ACCEPTED,
            ],
            [event.disposition for event in cycle.events],
        )
        accepted = coordinator.latest_accepted_frame()
        self.assertIsNotNone(accepted)
        self.assertEqual(2, accepted.frame_id)
        self.assertIsNot(snapshots[1], accepted.snapshot)
        self.assertEqual("session-hand-1", accepted.history.hand_id)
        snapshots[1].hand_id = "mutated-original"
        accepted.snapshot.hand_id = "mutated-reader-copy"
        fresh_copy = coordinator.latest_accepted_frame()
        self.assertEqual("session-hand-1", fresh_copy.snapshot.hand_id)
        readiness = coordinator.readiness(
            hand_id=accepted.history.hand_id,
            through_frame_id=2,
        )
        self.assertTrue(readiness.ready)
        self.assertIsNot(accepted, readiness.accepted_frame)
        self.assertEqual(
            "session-hand-1",
            readiness.accepted_frame.snapshot.hand_id,
        )
        coordinator.invalidate_gap("later capture loss", frame_id=3)
        self.assertIsNone(coordinator.latest_accepted_frame())

    def test_newer_transient_watermark_makes_older_bundle_unavailable(self):
        target = heads_up_to_turn_history()
        coordinator = PublicHistoryCoordinator()
        coordinator.submit_candidate(
            1,
            snapshot_for_prefix(target, 0),
        )
        coordinator.advance_decoder_watermark(1)
        self.assertIsNotNone(coordinator.latest_accepted_frame())

        coordinator.advance_decoder_watermark(2)

        self.assertIsNone(coordinator.latest_accepted_frame())
        self.assertIsNone(coordinator.view().accepted_frame)
        readiness = coordinator.readiness(
            hand_id=target.hand_id,
            through_frame_id=1,
        )
        self.assertFalse(readiness.ready)
        self.assertIsNone(readiness.accepted_frame)
        self.assertIn("newer frame 2", readiness.reason)

    def test_coordinator_owns_candidate_before_async_watermark_advance(self):
        target = heads_up_to_turn_history()
        coordinator = PublicHistoryCoordinator()
        original = snapshot_for_prefix(target, 0)
        coordinator.submit_candidate(1, original)
        original.board_state.community_cards = ["Ah"]

        result = coordinator.advance_decoder_watermark(1)[0]

        self.assertEqual("anchored", result.disposition.value)
        accepted = coordinator.latest_accepted_frame()
        self.assertEqual([], accepted.snapshot.board_state.community_cards)

    def test_late_older_gap_cannot_move_recovery_boundary_backwards(self):
        target = heads_up_to_turn_history()
        coordinator = PublicHistoryCoordinator()
        coordinator.submit_candidate(1, snapshot_for_prefix(target, 0))
        coordinator.advance_decoder_watermark(1)
        coordinator.invalidate_gap("newer loss", frame_id=10)
        coordinator.invalidate_gap("older late callback", frame_id=5)
        self.assertEqual(10, coordinator.view().gap_frame_id)

        next_hand = replace(target, hand_id="next-hand")
        coordinator.submit_candidate(
            7,
            snapshot_for_prefix(next_hand, 0),
        )
        result = coordinator.advance_decoder_watermark(7)[0]

        self.assertEqual(
            PublicHistoryStatus.GAPPED,
            result.status,
        )
        self.assertEqual(PublicHistoryStatus.GAPPED, coordinator.status)
        self.assertIsNone(coordinator.latest_accepted_frame())

    def test_unproved_villain_check_is_rejected_and_gaps_the_hand(self):
        target = heads_up_to_turn_history()
        snapshots = [
            enrich_recorder_snapshot(
                snapshot_for_prefix(
                    target,
                    count,
                    include_action_overlay=(count != 8),
                )
            )
            # Prefix 8 is the invisible villain CHECK.  It is observationally
            # identical to prefix 7, so it remains a duplicate.  Prefix 9 then
            # proves a different player bet while the recorder still expects
            # that check, forcing the required fail-closed rejection.
            for count in range(10)
        ]
        decoder = self.decoder(SequenceAnalyzer(*snapshots))
        ring = KeyframeRing(16)
        for frame_id in range(1, 11):
            ring.offer(
                keyframe(
                    frame_id,
                    (
                        KeyframeReason.BASELINE
                        if frame_id == 1
                        else KeyframeReason.HEARTBEAT
                    ),
                )
            )
        coordinator = PublicHistoryCoordinator()
        worker = PublicHistoryWorker(
            ring=ring,
            coordinator=coordinator,
            decoder=decoder,
            decode_timeout_seconds=0.25,
            idle_wait_seconds=0.01,
        )

        cycle = worker.process_once()

        self.assertEqual(
            WorkerEventDisposition.OBSERVATION_REJECTED,
            cycle.events[-1].disposition,
        )
        self.assertIn(
            "does not prove the expected actor",
            cycle.events[-1].detail,
        )
        self.assertEqual(PublicHistoryStatus.GAPPED, coordinator.status)
        self.assertIsNone(coordinator.latest_accepted_frame())


class ContinuousPublicHistoryRuntimeTests(unittest.TestCase):
    def test_stop_before_start_permanently_prevents_start(self):
        calls = []
        ring = KeyframeRing(2)
        coordinator = PublicHistoryCoordinator()

        class RecordingWorker(PublicHistoryWorker):
            def start(inner_self, **kwargs):
                calls.append("worker.start")

            def stop(inner_self, **kwargs):
                calls.append("worker.stop")

        worker = RecordingWorker(
            ring=ring,
            coordinator=coordinator,
            decoder=SimpleNamespace(
                decode=lambda frame, timeout_seconds: DecodeResult.transient()
            ),
        )

        class CaptureService:
            def __init__(inner_self):
                inner_self.ring = ring

            def start(inner_self):
                calls.append("capture.start")

            def stop(inner_self, **kwargs):
                calls.append("capture.stop")

        runtime = ContinuousPublicHistoryRuntime(
            capture_service=CaptureService(),
            coordinator=coordinator,
            worker=worker,
        )

        runtime.stop()
        with self.assertRaisesRegex(RuntimeError, "cannot be restarted"):
            runtime.start()
        runtime.stop()

        self.assertEqual([], calls)

    def test_capture_background_error_is_unhealthy(self):
        ring = KeyframeRing(2)
        coordinator = PublicHistoryCoordinator()

        class HealthyWorker(PublicHistoryWorker):
            def __init__(inner_self, **kwargs):
                super().__init__(**kwargs)
                inner_self.fake_running = False

            def start(inner_self, **kwargs):
                inner_self.fake_running = True

            def stop(inner_self, **kwargs):
                inner_self.fake_running = False

            def view(inner_self):
                return SimpleNamespace(
                    running=inner_self.fake_running,
                    background_error=None,
                )

        worker = HealthyWorker(
            ring=ring,
            coordinator=coordinator,
            decoder=SimpleNamespace(
                decode=lambda frame, timeout_seconds: DecodeResult.transient()
            ),
        )

        class CaptureService:
            def __init__(inner_self):
                inner_self.ring = ring
                inner_self.running = False
                inner_self.background_error = None

            def start(inner_self):
                inner_self.running = True

            def stop(inner_self, **kwargs):
                inner_self.running = False

        capture = CaptureService()
        runtime = ContinuousPublicHistoryRuntime(
            capture_service=capture,
            coordinator=coordinator,
            worker=worker,
        )
        runtime.start()
        self.assertTrue(runtime.is_healthy)

        capture.background_error = RuntimeError("source cleanup broke")

        self.assertFalse(runtime.is_healthy)
        self.assertIn("source cleanup broke", runtime.health_error)
        runtime.stop()

    def test_terminal_capture_cleanup_error_still_stops_worker(self):
        calls = []
        ring = KeyframeRing(2)
        coordinator = PublicHistoryCoordinator()

        class RecordingWorker(PublicHistoryWorker):
            def start(inner_self, **kwargs):
                calls.append("worker.start")

            def stop(inner_self, **kwargs):
                calls.append("worker.stop")

        worker = RecordingWorker(
            ring=ring,
            coordinator=coordinator,
            decoder=SimpleNamespace(
                decode=lambda frame, timeout_seconds: DecodeResult.transient()
            ),
        )

        class CaptureService:
            def __init__(inner_self):
                inner_self.ring = ring
                inner_self.running = False

            def start(inner_self):
                inner_self.running = True

            def stop(inner_self, **kwargs):
                inner_self.running = False
                raise RuntimeError("terminal capture cleanup failure")

        runtime = ContinuousPublicHistoryRuntime(
            capture_service=CaptureService(),
            coordinator=coordinator,
            worker=worker,
        )
        runtime.start()

        with self.assertRaisesRegex(
            RuntimeError,
            "terminal capture cleanup failure",
        ):
            runtime.stop()

        self.assertEqual(
            ["worker.start", "worker.stop"],
            calls,
        )
        runtime.stop()
        self.assertEqual(
            ["worker.start", "worker.stop"],
            calls,
        )

    def test_live_capture_error_keeps_worker_alive_for_stop_retry(self):
        calls = []
        ring = KeyframeRing(2)
        coordinator = PublicHistoryCoordinator()

        class RecordingWorker(PublicHistoryWorker):
            def start(inner_self, **kwargs):
                calls.append("worker.start")

            def stop(inner_self, **kwargs):
                calls.append("worker.stop")

        worker = RecordingWorker(
            ring=ring,
            coordinator=coordinator,
            decoder=SimpleNamespace(
                decode=lambda frame, timeout_seconds: DecodeResult.transient()
            ),
        )

        class CaptureService:
            def __init__(inner_self):
                inner_self.ring = ring
                inner_self.running = False
                inner_self.first_stop = True

            def start(inner_self):
                inner_self.running = True

            def stop(inner_self, **kwargs):
                if inner_self.first_stop:
                    inner_self.first_stop = False
                    raise RuntimeError("producer is still live")
                inner_self.running = False

        runtime = ContinuousPublicHistoryRuntime(
            capture_service=CaptureService(),
            coordinator=coordinator,
            worker=worker,
        )
        runtime.start()

        with self.assertRaisesRegex(RuntimeError, "still live"):
            runtime.stop()
        self.assertEqual(["worker.start"], calls)

        runtime.stop()
        self.assertEqual(["worker.start", "worker.stop"], calls)

    def test_lifecycle_starts_consumer_before_producer_and_drains_after_stop(self):
        calls = []
        ring = KeyframeRing(2)
        coordinator = PublicHistoryCoordinator()
        target = heads_up_to_turn_history()
        coordinator.submit_candidate(1, snapshot_for_prefix(target, 0))
        coordinator.advance_decoder_watermark(1)
        self.assertIsNotNone(coordinator.latest_accepted_frame())

        class RecordingWorker(PublicHistoryWorker):
            def start(inner_self, **kwargs):
                calls.append("worker.start")

            def stop(inner_self, *, drain=True, timeout_seconds=5.0):
                calls.append(
                    f"worker.stop(drain={drain},timeout={timeout_seconds:g})"
                )

        worker = RecordingWorker(
            ring=ring,
            coordinator=coordinator,
            decoder=SimpleNamespace(
                decode=lambda frame, timeout_seconds: DecodeResult.transient()
            ),
            decode_timeout_seconds=0.25,
            idle_wait_seconds=0.01,
        )

        class CaptureService:
            def __init__(inner_self):
                inner_self.ring = ring

            def start(inner_self):
                calls.append("capture.start")

            def stop(inner_self, *, timeout_seconds=5.0):
                calls.append(f"capture.stop(timeout={timeout_seconds:g})")

        runtime = ContinuousPublicHistoryRuntime(
            capture_service=CaptureService(),
            coordinator=coordinator,
            worker=worker,
        )

        runtime.start()
        runtime.stop(timeout_seconds=0.5)

        self.assertEqual(["worker.start", "capture.start"], calls[:2])
        self.assertTrue(calls[2].startswith("capture.stop(timeout="))
        self.assertTrue(
            calls[3].startswith("worker.stop(drain=True,timeout=")
        )
        capture_timeout = float(
            calls[2].removeprefix("capture.stop(timeout=").removesuffix(")")
        )
        worker_timeout = float(
            calls[3]
            .removeprefix("worker.stop(drain=True,timeout=")
            .removesuffix(")")
        )
        self.assertGreaterEqual(capture_timeout, worker_timeout)
        self.assertLessEqual(capture_timeout, 0.5)
        self.assertIsNone(coordinator.latest_accepted_frame())
        self.assertEqual(PublicHistoryStatus.GAPPED, coordinator.status)

    def test_producer_stop_timeout_keeps_worker_alive_and_allows_retry(self):
        calls = []
        ring = KeyframeRing(2)
        coordinator = PublicHistoryCoordinator()

        class RecordingWorker(PublicHistoryWorker):
            def start(inner_self, **kwargs):
                calls.append("worker.start")

            def stop(inner_self, *, drain=True, timeout_seconds=5.0):
                calls.append(("worker.stop", drain, timeout_seconds))

        worker = RecordingWorker(
            ring=ring,
            coordinator=coordinator,
            decoder=SimpleNamespace(
                decode=lambda frame, timeout_seconds: DecodeResult.transient()
            ),
            decode_timeout_seconds=0.25,
            idle_wait_seconds=0.01,
        )

        class TimeoutOnceCapture:
            def __init__(inner_self):
                inner_self.ring = ring
                inner_self.stop_calls = 0

            def start(inner_self):
                calls.append("capture.start")

            def stop(inner_self, *, timeout_seconds=5.0):
                inner_self.stop_calls += 1
                calls.append(
                    (
                        "capture.stop",
                        inner_self.stop_calls,
                        timeout_seconds,
                    )
                )
                if inner_self.stop_calls == 1:
                    raise TimeoutError("producer still joining")

        capture = TimeoutOnceCapture()
        runtime = ContinuousPublicHistoryRuntime(
            capture_service=capture,
            coordinator=coordinator,
            worker=worker,
        )
        runtime.start()

        with self.assertRaisesRegex(TimeoutError, "still joining"):
            runtime.stop(timeout_seconds=0.5)
        self.assertFalse(
            any(
                isinstance(call, tuple) and call[0] == "worker.stop"
                for call in calls
            )
        )

        runtime.stop(timeout_seconds=0.5)

        worker_stops = [
            call
            for call in calls
            if isinstance(call, tuple) and call[0] == "worker.stop"
        ]
        self.assertEqual(1, len(worker_stops))
        self.assertTrue(worker_stops[0][1])
        self.assertGreaterEqual(worker_stops[0][2], 0)
        self.assertLessEqual(worker_stops[0][2], 0.5)
        self.assertEqual(PublicHistoryStatus.GAPPED, coordinator.status)


class PublicHistoryVisionHookTests(unittest.TestCase):
    @staticmethod
    def _app_snapshot(
        app,
        *,
        street="PREFLOP",
        board=(),
        hero_cards=("Ah", "Kd"),
    ):
        raw = table_snapshot(
            street=street,
            board=tuple(board),
            hero_cards=tuple(hero_cards),
        )
        return app.GameSnapshot(
            hand_id="flow-hand",
            timestamp="2026-07-30T12:00:00Z",
            meta_info=app.MetaInfo(current_street=street),
            board_state=app.BoardState(
                community_cards=list(board),
                total_pot=raw.board_state.total_pot,
            ),
            dealer_seat_index=raw.dealer_seat_index,
            action_on_seat_index=4,
            players=[
                app.Player(
                    seat_index=player.seat_index,
                    name=player.name,
                    username=player.username,
                    stack_size=player.stack_size,
                    current_bet=player.current_bet,
                    status=player.status,
                    is_hero=player.is_hero,
                    is_dealer=player.is_dealer,
                    visible_action=player.visible_action,
                    hole_cards=player.hole_cards,
                )
                for player in raw.players
            ],
            last_action_context=app.LastActionContext(
                amount_to_call=0.0,
                hero_action_options=["Check", "Bet"],
            ),
        )

    @staticmethod
    def _caught_up_runtime():
        target = heads_up_to_turn_history()
        coordinator = PublicHistoryCoordinator()
        coordinator.submit_candidate(
            1,
            snapshot_for_prefix(target, 0),
        )
        coordinator.advance_decoder_watermark(1)
        ring = KeyframeRing(4)
        ring.acknowledge(1)

        class PassiveWorker(PublicHistoryWorker):
            def start(inner_self, **kwargs):
                with inner_self._state_lock:
                    inner_self._started = True
                    inner_self._running = True

            def stop(inner_self, **kwargs):
                with inner_self._state_lock:
                    inner_self._running = False

        worker = PassiveWorker(
            ring=ring,
            coordinator=coordinator,
            decoder=SimpleNamespace(
                decode=lambda frame, timeout_seconds: DecodeResult.transient()
            ),
        )

        class CaptureService:
            def __init__(inner_self):
                inner_self.ring = ring
                inner_self.running = False
                inner_self.background_error = None

            def start(inner_self):
                inner_self.running = True

            def stop(inner_self, **kwargs):
                inner_self.running = False

        runtime = ContinuousPublicHistoryRuntime(
            capture_service=CaptureService(),
            coordinator=coordinator,
            worker=worker,
        )
        runtime.start()
        accepted = coordinator.latest_accepted_frame()
        return runtime, ring, accepted

    def _assert_pending_ledger_blocks_pairing_and_token(
        self,
        *,
        mutate_ring,
        expected_reason,
    ):
        import poker_assistant as app

        runtime, ring, accepted = self._caught_up_runtime()
        self.assertIsNotNone(accepted)
        baseline = runtime.accepted_frame_if_caught_up()
        self.assertTrue(baseline.ready, baseline.reason)
        self.assertEqual(accepted, baseline.accepted_frame)
        mutate_ring(ring)
        token = app.AcceptedMultiwayDecisionToken(
            frame_id=accepted.frame_id,
            public_hand=accepted.history,
            decision_fingerprint="stale-token",
        )
        try:
            with self.assertRaisesRegex(
                app.MultiwayProtocolError,
                expected_reason,
            ):
                app.accepted_multiway_strategy_input(
                    app.GameSnapshot(),
                    runtime,
                )

            ready, reason = app.continuous_multiway_token_status(
                token,
                runtime,
            )
            self.assertFalse(ready)
            self.assertRegex(reason, expected_reason)
        finally:
            runtime.stop()

    def test_pending_keyframe_blocks_pairing_and_token_status(self):
        self._assert_pending_ledger_blocks_pairing_and_token(
            mutate_ring=lambda ring: ring.offer(keyframe(2)),
            expected_reason="pending keyframe 2",
        )

    def test_unseen_gap_blocks_pairing_and_token_status(self):
        self._assert_pending_ledger_blocks_pairing_and_token(
            mutate_ring=lambda ring: ring.record_gap(
                CaptureGap(
                    kind=CaptureGapKind.CAPTURE_ERROR,
                    first_frame_id=2,
                    last_frame_id=2,
                    detected_monotonic_ns=2,
                    reason="pending capture failure",
                )
            ),
            expected_reason="unseen revision",
        )

    def test_dedicated_client_makes_one_request_with_per_call_deadline(self):
        os.environ.setdefault("GEMINI_API_KEY", "test-key")
        os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
        import poker_assistant as app

        provider_calls = []

        class Models:
            def generate_content(inner_self, **kwargs):
                provider_calls.append(kwargs)
                return SimpleNamespace(text='{"ok":true}')

        dedicated_client = SimpleNamespace(models=Models())
        parsed = SimpleNamespace(vision_error="")
        original_config = app.types.GenerateContentConfig(
            max_output_tokens=10
        )
        captures = {"buttons": object(), "table": object()}

        with (
            mock.patch.object(
                app.genai,
                "Client",
                return_value=dedicated_client,
            ) as client_constructor,
            mock.patch.object(
                app,
                "detect_locally_dealt_seats",
                return_value={0, 1},
            ),
            mock.patch.object(
                app,
                "detect_hero_action_buttons",
                return_value=False,
            ),
            mock.patch.object(
                app,
                "build_vision_request",
                return_value=(["vision"], original_config),
            ),
            mock.patch.object(
                app,
                "parse_response",
                return_value=parsed,
            ) as parse,
        ):
            analyzer = app.create_public_history_frame_analyzer()
            result = analyzer(captures, timeout_seconds=1.5)

        self.assertIs(parsed, result)
        client_constructor.assert_called_once()
        self.assertEqual(1, len(provider_calls))
        call = provider_calls[0]
        self.assertEqual(["vision"], call["contents"])
        self.assertEqual(1400, call["config"].http_options.timeout)
        self.assertEqual(
            1,
            call["config"].http_options.retry_options.attempts,
        )
        self.assertIsNone(original_config.http_options)
        parse.assert_called_once_with(
            '{"ok":true}',
            {0, 1},
            hero_turn_confirmed=False,
        )

    def test_multiway_getter_returns_only_one_accepted_bundle(self):
        import poker_assistant as app

        target = heads_up_to_turn_history()
        public_hand = replace(target, events=target.events[:4])
        raw = snapshot_for_prefix(target, 4)
        players = [
            app.Player(
                seat_index=player.seat_index,
                name=player.name,
                username=f"P{player.seat_index}",
                stack_size=player.stack_size,
                current_bet=player.current_bet,
                status=player.status,
                is_hero=player.seat_index == 4,
                is_dealer=player.is_dealer,
                visible_action=player.visible_action,
                hole_cards=(
                    ["Ah", "Kd"] if player.seat_index == 4 else None
                ),
            )
            for player in raw.players
        ]
        snapshot = app.GameSnapshot(
            hand_id=public_hand.hand_id,
            timestamp="2026-07-30T12:00:00Z",
            meta_info=app.MetaInfo(current_street="PREFLOP"),
            board_state=app.BoardState(
                community_cards=[],
                total_pot=raw.board_state.total_pot,
            ),
            dealer_seat_index=raw.dealer_seat_index,
            action_on_seat_index=4,
            players=players,
            last_action_context=app.LastActionContext(
                amount_to_call=2.0,
                hero_action_options=["Fold", "Call", "Raise"],
            ),
        )
        accepted = AcceptedPublicHistoryFrame(
            frame_id=17,
            snapshot=snapshot,
            history=public_hand,
        )
        accepted_holder = {"value": accepted}
        coordinator = SimpleNamespace(
            latest_accepted_frame=lambda: accepted_holder["value"],
        )
        runtime = SimpleNamespace(
            coordinator=coordinator,
            is_healthy=True,
            health_error="",
            accepted_frame_if_caught_up=lambda: SimpleNamespace(
                ready=accepted_holder["value"] is not None,
                reason=(
                    ""
                    if accepted_holder["value"] is not None
                    else "no accepted frame"
                ),
                accepted_frame=accepted_holder["value"],
            ),
        )

        selected_snapshot, selected_history, selected_token = (
            app.accepted_multiway_strategy_input(snapshot, runtime)
        )

        self.assertIsNot(snapshot, selected_snapshot)
        self.assertEqual(snapshot.timestamp, selected_snapshot.timestamp)
        self.assertEqual(
            public_hand.hand_id,
            selected_snapshot.hand_id,
        )
        self.assertEqual(accepted.frame_id, selected_token.frame_id)
        self.assertIs(public_hand, selected_token.public_hand)
        self.assertEqual((selected_snapshot,), selected_history.snapshots)
        self.assertIs(public_hand, selected_history.public_hand)

        mismatches = {
            "dealer differs": lambda candidate: setattr(
                candidate,
                "dealer_seat_index",
                2,
            ),
            "seat positions differ": lambda candidate: setattr(
                candidate.players[0],
                "name",
                "BTN",
            ),
            "Hero combo differs": lambda candidate: setattr(
                candidate.players[4],
                "hole_cards",
                ["Qh", "Kd"],
            ),
        }
        for expected, mutate in mismatches.items():
            with self.subTest(expected=expected):
                mismatched = deepcopy(snapshot)
                mutate(mismatched)
                with self.assertRaisesRegex(
                    app.MultiwayProtocolError,
                    expected,
                ):
                    app.accepted_multiway_strategy_input(
                        mismatched,
                        runtime,
                    )

        evaluator = mock.Mock(return_value=SimpleNamespace(ok=True))
        accepted_holder["value"] = None
        with self.assertRaisesRegex(
            app.MultiwayProtocolError,
            "before solver request",
        ):
            app.evaluate_continuous_multiway_strategy(
                selected_snapshot,
                selected_history,
                selected_token,
                runtime=runtime,
                evaluator=evaluator,
            )
        evaluator.assert_not_called()

        # A semantically identical accepted heartbeat may advance frame_id
        # without revoking the decision token.
        accepted_holder["value"] = AcceptedPublicHistoryFrame(
            frame_id=18,
            snapshot=deepcopy(snapshot),
            history=public_hand,
        )
        heartbeat_result = app.evaluate_continuous_multiway_strategy(
            selected_snapshot,
            selected_history,
            selected_token,
            runtime=runtime,
            evaluator=evaluator,
        )
        self.assertTrue(heartbeat_result.ok)

        def invalidate_during_evaluate(*args, **kwargs):
            accepted_holder["value"] = None
            return SimpleNamespace(ok=True)

        with self.assertRaisesRegex(
            app.MultiwayProtocolError,
            "ADVICE DISCARDED",
        ):
            app.evaluate_continuous_multiway_strategy(
                selected_snapshot,
                selected_history,
                selected_token,
                runtime=runtime,
                evaluator=invalidate_during_evaluate,
            )

        prepared = app.prepare_strategy_state(
            selected_snapshot,
            selected_history,
        )
        state = app.build_multiway_solver_state(
            selected_snapshot,
            selected_history,
            prepared,
        )
        self.assertIs(public_hand, state.public_hand)

        wrong_dealer = deepcopy(selected_snapshot)
        wrong_dealer.dealer_seat_index = 2
        with self.assertRaisesRegex(
            app.MultiwayProtocolError,
            "dealer differs",
        ):
            app.build_multiway_solver_state(
                wrong_dealer,
                selected_history,
                app.prepare_strategy_state(
                    wrong_dealer,
                    selected_history,
                ),
            )

        wrong_position = deepcopy(selected_snapshot)
        wrong_position.players[0].name = "BTN"
        with self.assertRaisesRegex(
            app.MultiwayProtocolError,
            "position differs",
        ):
            app.build_multiway_solver_state(
                wrong_position,
                selected_history,
                app.prepare_strategy_state(
                    wrong_position,
                    selected_history,
                ),
            )

        class FakeRemoteMultiwayClient(app.RemoteMultiwayClient):
            def __init__(inner_self):
                inner_self.calls = 0

            def evaluate(inner_self, state):
                inner_self.calls += 1
                raise AssertionError("legacy history reached remote transport")

        legacy_history = app.HandHistory(
            hand_id=public_hand.hand_id,
            snapshots=[selected_snapshot],
        )
        legacy_history.public_event_recorder.history = public_hand
        remote = FakeRemoteMultiwayClient()
        rejected = app.evaluate_strategy_backend(
            selected_snapshot,
            legacy_history,
            backend="GTO",
            router=remote,
        )
        self.assertEqual(0, remote.calls)
        self.assertIn(
            "legacy manual history is forbidden",
            rejected.final_analysis,
        )

    def test_multiway_getter_invalidates_unhealthy_runtime(self):
        import poker_assistant as app

        invalidations = []
        coordinator = SimpleNamespace(
            invalidate_gap=lambda reason: invalidations.append(reason),
        )
        runtime = SimpleNamespace(
            coordinator=coordinator,
            is_healthy=False,
            health_error="public-history worker failed: poisoned",
        )

        with self.assertRaisesRegex(
            app.MultiwayProtocolError,
            "worker failed",
        ):
            app.accepted_multiway_strategy_input(
                app.GameSnapshot(),
                runtime,
            )

        self.assertEqual(
            ["public-history worker failed: poisoned"],
            invalidations,
        )

    def test_v3_pairing_uses_raw_combo_before_legacy_card_repair(self):
        import poker_assistant as app

        previous = self._app_snapshot(
            app,
            street="FLOP",
            board=("2c", "7d", "Jh"),
            hero_cards=("Ah", "Kd"),
        )
        current = self._app_snapshot(
            app,
            street="TURN",
            board=("2c", "7d", "Jh", "9s"),
            hero_cards=("Qh", "Qs"),
        )
        history = app.HandHistory(
            hand_id=previous.hand_id,
            snapshots=[previous],
        )
        seen_raw = []

        def reject_after_observing(candidate):
            seen_raw.append(deepcopy(candidate))
            raise app.MultiwayProtocolError("test pairing stop")

        original_history = app.current_history
        original_recorder = app.recorder
        app.current_history = history
        app.recorder = None
        try:
            with (
                mock.patch.object(
                    app,
                    "analyze_state",
                    return_value=(
                        current,
                        {"buttons": object()},
                        0.01,
                        0.02,
                    ),
                ),
                mock.patch.object(
                    app,
                    "detect_hero_action_buttons",
                    return_value=True,
                ),
                mock.patch.object(
                    app,
                    "capture_validation_regions",
                    return_value={"frame": object()},
                ),
                mock.patch.object(
                    app,
                    "table_state_change_reasons",
                    return_value=[],
                ),
                mock.patch.object(
                    app,
                    "validate_snapshot_candidate",
                    return_value=[],
                ),
                mock.patch.object(
                    app,
                    "accepted_multiway_strategy_input",
                    side_effect=reject_after_observing,
                ),
                mock.patch.object(app, "STRATEGY_BACKEND", "GTO_MULTIWAY"),
                mock.patch.object(
                    app,
                    "PUBLIC_HISTORY_DECODER_ENABLED",
                    True,
                ),
                mock.patch.object(app, "display_results"),
            ):
                app.run_analysis_flow("strategy")
        finally:
            app.current_history = original_history
            app.recorder = original_recorder

        self.assertEqual(1, len(seen_raw))
        raw_hero = next(
            player for player in seen_raw[0].players if player.is_hero
        )
        repaired_hero = next(
            player for player in current.players if player.is_hero
        )
        self.assertEqual(["Qh", "Qs"], raw_hero.hole_cards)
        self.assertEqual(["Ah", "Kd"], repaired_hero.hole_cards)

    def test_display_boundary_rechecks_token_after_final_pixel_guard(self):
        import poker_assistant as app

        current = self._app_snapshot(app)
        public_hand = heads_up_to_turn_history()
        token = app.AcceptedMultiwayDecisionToken(
            frame_id=7,
            public_hand=public_hand,
            decision_fingerprint=("decision",),
        )
        accepted_history = app.AcceptedMultiwayStrategyHistory(
            snapshots=(deepcopy(current),),
            public_hand=public_hand,
        )
        result = SimpleNamespace(
            final_analysis="**Action:** Bet",
            metrics={"ok": True},
            hand_rank="High Card",
            latency_seconds=0.1,
            source="test solver",
            prompt="test prompt",
        )
        events = []

        def capture(*args, **kwargs):
            events.append("pixel_capture")
            return {"frame": object()}

        def token_status(*args, **kwargs):
            events.append("token_check")
            return False, "accepted public history changed"

        def display(*args, **kwargs):
            events.append("display")

        original_history = app.current_history
        original_recorder = app.recorder
        app.current_history = app.HandHistory()
        app.recorder = None
        try:
            with (
                mock.patch.object(
                    app,
                    "analyze_state",
                    return_value=(
                        current,
                        {"buttons": object()},
                        0.01,
                        0.02,
                    ),
                ),
                mock.patch.object(
                    app,
                    "detect_hero_action_buttons",
                    return_value=True,
                ),
                mock.patch.object(
                    app,
                    "capture_validation_regions",
                    side_effect=capture,
                ),
                mock.patch.object(
                    app,
                    "table_state_change_reasons",
                    return_value=[],
                ),
                mock.patch.object(
                    app,
                    "validate_snapshot_candidate",
                    return_value=[],
                ),
                mock.patch.object(
                    app,
                    "accepted_multiway_strategy_input",
                    return_value=(
                        deepcopy(current),
                        accepted_history,
                        token,
                    ),
                ),
                mock.patch.object(
                    app,
                    "evaluate_continuous_multiway_strategy",
                    return_value=result,
                ),
                mock.patch.object(
                    app,
                    "continuous_multiway_token_status",
                    side_effect=token_status,
                ),
                mock.patch.object(app, "STRATEGY_BACKEND", "GTO_MULTIWAY"),
                mock.patch.object(
                    app,
                    "PUBLIC_HISTORY_DECODER_ENABLED",
                    True,
                ),
                mock.patch("builtins.open", mock.mock_open()),
                mock.patch.object(
                    app,
                    "display_results",
                    side_effect=display,
                ) as displayed,
            ):
                app.run_analysis_flow("strategy")
        finally:
            app.current_history = original_history
            app.recorder = original_recorder

        self.assertEqual(
            ["pixel_capture", "pixel_capture", "token_check", "display"],
            events,
        )
        displayed_analysis = displayed.call_args.args[1]
        self.assertIn("ADVICE DISCARDED", displayed_analysis)
        self.assertNotEqual(result.final_analysis, displayed_analysis)


if __name__ == "__main__":
    unittest.main()
