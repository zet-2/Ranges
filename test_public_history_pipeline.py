from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import unittest

from gto_hand_history import HandEvent
from public_history_pipeline import (
    CandidateDisposition,
    PublicHistoryCoordinator,
    PublicHistoryStatus,
)
from test_gto_event_collector import snapshot_for_prefix
from test_gto_hand_history import heads_up_to_turn_history


class PublicHistoryCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.target = heads_up_to_turn_history()
        self.coordinator = PublicHistoryCoordinator()

    def submit_and_advance(self, frame_id, snapshot):
        self.coordinator.submit_candidate(frame_id, snapshot)
        results = self.coordinator.advance_decoder_watermark(frame_id)
        self.assertEqual(1, len(results))
        return results[0]

    def test_starts_waiting_and_late_frame_does_not_create_prefix(self):
        result = self.submit_and_advance(
            1,
            snapshot_for_prefix(self.target, 1),
        )

        self.assertEqual(CandidateDisposition.REJECTED, result.disposition)
        self.assertEqual(
            PublicHistoryStatus.WAITING_FOR_ANCHOR,
            self.coordinator.status,
        )
        view = self.coordinator.view()
        self.assertIsNone(view.history)
        self.assertEqual(1, view.decoder_watermark)
        self.assertEqual(0, view.proven_through_frame_id)
        readiness = self.coordinator.readiness(
            hand_id=self.target.hand_id,
            through_frame_id=1,
        )
        self.assertFalse(readiness.ready)
        self.assertIn("waiting for", readiness.reason)

    def test_malformed_anchor_can_be_corrected_for_the_same_unseen_hand(self):
        malformed = snapshot_for_prefix(self.target, 0)
        malformed.board_state.community_cards = ["Ah"]
        first = self.submit_and_advance(1, malformed)

        corrected = self.submit_and_advance(
            2,
            snapshot_for_prefix(self.target, 0),
        )

        self.assertEqual(CandidateDisposition.REJECTED, first.disposition)
        self.assertEqual(CandidateDisposition.ANCHORED, corrected.disposition)
        self.assertEqual(PublicHistoryStatus.TRACKING, self.coordinator.status)

    def test_anchor_tracks_and_is_ready_only_for_same_proven_frame(self):
        result = self.submit_and_advance(
            3,
            snapshot_for_prefix(self.target, 0),
        )

        self.assertEqual(CandidateDisposition.ANCHORED, result.disposition)
        self.assertEqual(PublicHistoryStatus.TRACKING, result.status)
        ready = self.coordinator.readiness(
            hand_id=self.target.hand_id,
            through_frame_id=3,
        )
        self.assertTrue(ready.ready)
        self.assertEqual(self.target.hand_id, ready.history.hand_id)

        wrong_hand = self.coordinator.readiness(
            hand_id="other-hand",
            through_frame_id=3,
        )
        self.assertFalse(wrong_hand.ready)
        self.assertIn("does not match", wrong_hand.reason)

    def test_decoder_watermark_and_accepted_proof_are_separate(self):
        self.submit_and_advance(1, snapshot_for_prefix(self.target, 0))
        self.coordinator.advance_decoder_watermark(5)

        readiness = self.coordinator.readiness(
            hand_id=self.target.hand_id,
            through_frame_id=5,
        )

        self.assertFalse(readiness.ready)
        self.assertEqual(5, readiness.decoder_watermark)
        self.assertEqual(1, readiness.proven_through_frame_id)
        self.assertIn("no accepted observation", readiness.reason)

    def test_out_of_order_decoder_results_commit_in_frame_order(self):
        self.coordinator.submit_candidate(
            3,
            snapshot_for_prefix(self.target, 1),
        )
        self.coordinator.submit_candidate(
            1,
            snapshot_for_prefix(self.target, 0),
        )

        results = self.coordinator.advance_decoder_watermark(3)

        self.assertEqual([1, 3], [result.frame_id for result in results])
        self.assertEqual(
            [CandidateDisposition.ANCHORED, CandidateDisposition.ADVANCED],
            [result.disposition for result in results],
        )
        view = self.coordinator.view()
        self.assertEqual(3, view.proven_through_frame_id)
        self.assertEqual(self.target.events[:1], view.history.events)

    def test_rejected_noisy_candidate_does_not_poison_later_valid_frame(self):
        self.submit_and_advance(1, snapshot_for_prefix(self.target, 0))

        rejected = self.submit_and_advance(
            2,
            snapshot_for_prefix(self.target, 2),
        )
        self.assertEqual(CandidateDisposition.REJECTED, rejected.disposition)
        after_rejection = self.coordinator.view()
        self.assertEqual(PublicHistoryStatus.TRACKING, after_rejection.status)
        self.assertEqual((), after_rejection.history.events)
        self.assertEqual(1, after_rejection.proven_through_frame_id)

        accepted = self.submit_and_advance(
            3,
            snapshot_for_prefix(self.target, 1),
        )
        self.assertEqual(CandidateDisposition.ADVANCED, accepted.disposition)
        self.assertEqual("", self.coordinator.view().last_rejection)
        self.assertEqual(
            self.target.events[:1],
            self.coordinator.view().history.events,
        )
        self.assertTrue(
            self.coordinator.readiness(
                hand_id=self.target.hand_id,
                through_frame_id=3,
            ).ready
        )

    def test_duplicate_candidate_advances_proven_frame_without_new_event(self):
        self.submit_and_advance(1, snapshot_for_prefix(self.target, 0))
        duplicate = self.submit_and_advance(
            4,
            snapshot_for_prefix(self.target, 0),
        )

        self.assertEqual(CandidateDisposition.DUPLICATE, duplicate.disposition)
        self.assertEqual((), self.coordinator.view().history.events)
        self.assertTrue(
            self.coordinator.readiness(
                hand_id=self.target.hand_id,
                through_frame_id=4,
            ).ready
        )

    def test_tracks_ordered_multiway_path_without_forcing_heads_up(self):
        multiway = replace(
            self.target,
            hand_id="multiway-hand",
            events=(
                HandEvent(0, "FOLD", "PREFLOP", actor_seat=0),
                HandEvent(1, "FOLD", "PREFLOP", actor_seat=1),
                HandEvent(
                    2,
                    "CALL",
                    "PREFLOP",
                    actor_seat=2,
                    amount_to_bb=Decimal("1"),
                ),
                HandEvent(
                    3,
                    "CALL",
                    "PREFLOP",
                    actor_seat=3,
                    amount_to_bb=Decimal("1"),
                ),
                HandEvent(
                    4,
                    "CALL",
                    "PREFLOP",
                    actor_seat=4,
                    amount_to_bb=Decimal("1"),
                ),
                HandEvent(5, "CHECK", "PREFLOP", actor_seat=5),
                HandEvent(
                    6,
                    "DEAL_FLOP",
                    "FLOP",
                    cards=("Ac", "Kd", "7s"),
                ),
            ),
        )

        for event_count in range(len(multiway.events) + 1):
            self.submit_and_advance(
                event_count + 1,
                snapshot_for_prefix(multiway, event_count),
            )

        view = self.coordinator.view()
        self.assertEqual(PublicHistoryStatus.TRACKING, view.status)
        self.assertEqual(multiway, view.history)
        self.assertTrue(
            self.coordinator.readiness(
                hand_id="multiway-hand",
                through_frame_id=len(multiway.events) + 1,
            ).ready
        )

    def test_explicit_gap_blocks_same_hand_and_new_anchor_recovers(self):
        self.submit_and_advance(1, snapshot_for_prefix(self.target, 0))
        self.coordinator.invalidate_gap(
            "unacknowledged keyframe overflow",
            frame_id=2,
        )

        view = self.coordinator.view()
        self.assertEqual(PublicHistoryStatus.GAPPED, view.status)
        self.assertIn("overflow", view.gap_reason)
        self.assertEqual(2, view.gap_frame_id)
        self.assertFalse(
            self.coordinator.readiness(
                hand_id=self.target.hand_id,
                through_frame_id=1,
            ).ready
        )

        ignored = self.submit_and_advance(
            3,
            snapshot_for_prefix(self.target, 1),
        )
        self.assertEqual(
            CandidateDisposition.IGNORED_GAPPED_HAND,
            ignored.disposition,
        )
        self.assertEqual(PublicHistoryStatus.GAPPED, self.coordinator.status)

        next_hand = replace(self.target, hand_id="next-hand")
        recovered = self.submit_and_advance(
            5,
            snapshot_for_prefix(next_hand, 0),
        )
        self.assertEqual(CandidateDisposition.ANCHORED, recovered.disposition)
        self.assertEqual(PublicHistoryStatus.TRACKING, recovered.status)
        self.assertTrue(
            self.coordinator.readiness(
                hand_id="next-hand",
                through_frame_id=5,
            ).ready
        )

    def test_valid_new_hand_anchor_replaces_tracking_hand(self):
        self.submit_and_advance(1, snapshot_for_prefix(self.target, 0))
        next_hand = replace(self.target, hand_id="next-hand")

        result = self.submit_and_advance(
            2,
            snapshot_for_prefix(next_hand, 0),
        )

        self.assertEqual(
            CandidateDisposition.NEW_HAND_ANCHORED,
            result.disposition,
        )
        self.assertEqual("next-hand", self.coordinator.view().hand_id)
        self.assertEqual((), self.coordinator.view().history.events)

    def test_gap_while_waiting_cannot_recover_same_hand(self):
        self.submit_and_advance(1, snapshot_for_prefix(self.target, 1))
        self.coordinator.invalidate_gap("decoder response lost", frame_id=2)

        ignored = self.submit_and_advance(
            3,
            snapshot_for_prefix(self.target, 0),
        )

        self.assertEqual(
            CandidateDisposition.IGNORED_GAPPED_HAND,
            ignored.disposition,
        )
        self.assertEqual(PublicHistoryStatus.GAPPED, self.coordinator.status)

    def test_malformed_candidate_cannot_clear_a_gap_or_reanchor_same_hand(self):
        self.submit_and_advance(1, snapshot_for_prefix(self.target, 0))
        self.coordinator.invalidate_gap("decoder response lost", frame_id=2)
        malformed = snapshot_for_prefix(self.target, 0)
        malformed.hand_id = ""

        ignored = self.submit_and_advance(3, malformed)

        self.assertEqual(
            CandidateDisposition.IGNORED_GAPPED_HAND,
            ignored.disposition,
        )
        self.assertEqual(PublicHistoryStatus.GAPPED, self.coordinator.status)
        same_hand = self.submit_and_advance(
            4,
            snapshot_for_prefix(self.target, 0),
        )
        self.assertEqual(
            CandidateDisposition.IGNORED_GAPPED_HAND,
            same_hand.disposition,
        )
        self.assertEqual(PublicHistoryStatus.GAPPED, self.coordinator.status)

    def test_malformed_different_hand_cannot_clear_a_gap(self):
        self.submit_and_advance(1, snapshot_for_prefix(self.target, 0))
        self.coordinator.invalidate_gap("decoder response lost", frame_id=2)
        different = replace(self.target, hand_id="different-hand")
        malformed = snapshot_for_prefix(different, 0)
        malformed.board_state.community_cards = ["Ah"]

        rejected = self.submit_and_advance(3, malformed)

        self.assertEqual(CandidateDisposition.REJECTED, rejected.disposition)
        self.assertEqual(PublicHistoryStatus.GAPPED, self.coordinator.status)
        original = self.submit_and_advance(
            4,
            snapshot_for_prefix(self.target, 0),
        )
        self.assertEqual(
            CandidateDisposition.IGNORED_GAPPED_HAND,
            original.disposition,
        )

    def test_candidate_before_gap_cannot_clear_later_gap(self):
        self.submit_and_advance(1, snapshot_for_prefix(self.target, 0))
        next_hand = replace(self.target, hand_id="next-before-gap")
        self.coordinator.submit_candidate(
            3,
            snapshot_for_prefix(next_hand, 0),
        )
        self.coordinator.invalidate_gap("capture frame lost", frame_id=4)

        before_gap = self.coordinator.advance_decoder_watermark(3)

        self.assertEqual(1, len(before_gap))
        self.assertEqual(
            CandidateDisposition.IGNORED_GAPPED_HAND,
            before_gap[0].disposition,
        )
        self.assertEqual(PublicHistoryStatus.GAPPED, self.coordinator.status)
        after_gap_same_hand = self.submit_and_advance(
            5,
            snapshot_for_prefix(next_hand, 0),
        )
        self.assertEqual(
            CandidateDisposition.IGNORED_GAPPED_HAND,
            after_gap_same_hand.disposition,
        )
        self.assertEqual(PublicHistoryStatus.GAPPED, self.coordinator.status)

    def test_unknown_pre_gap_candidate_blocks_automatic_recovery(self):
        self.submit_and_advance(1, snapshot_for_prefix(self.target, 0))
        next_hand = replace(self.target, hand_id="unidentified-before-gap")
        unidentified = snapshot_for_prefix(next_hand, 0)
        unidentified.hand_id = ""
        self.coordinator.submit_candidate(3, unidentified)
        self.coordinator.invalidate_gap("capture frame lost", frame_id=4)
        self.coordinator.advance_decoder_watermark(3)

        later = self.submit_and_advance(
            5,
            snapshot_for_prefix(next_hand, 0),
        )

        self.assertEqual(
            CandidateDisposition.IGNORED_GAPPED_HAND,
            later.disposition,
        )
        self.assertEqual(PublicHistoryStatus.GAPPED, self.coordinator.status)

    def test_processed_unknown_candidate_is_tainted_by_a_later_gap(self):
        self.submit_and_advance(1, snapshot_for_prefix(self.target, 0))
        next_hand = replace(self.target, hand_id="unknown-then-gap")
        unidentified = snapshot_for_prefix(next_hand, 0)
        unidentified.hand_id = ""
        rejected = self.submit_and_advance(3, unidentified)
        self.assertEqual(CandidateDisposition.REJECTED, rejected.disposition)

        self.coordinator.invalidate_gap("capture frame lost", frame_id=4)
        later = self.submit_and_advance(
            5,
            snapshot_for_prefix(next_hand, 0),
        )

        self.assertEqual(
            CandidateDisposition.IGNORED_GAPPED_HAND,
            later.disposition,
        )
        self.assertEqual(PublicHistoryStatus.GAPPED, self.coordinator.status)

    def test_unordered_gap_cannot_be_cleared_by_a_pending_new_hand(self):
        self.submit_and_advance(1, snapshot_for_prefix(self.target, 0))
        next_hand = replace(self.target, hand_id="pending-at-unknown-gap")
        self.coordinator.submit_candidate(
            3,
            snapshot_for_prefix(next_hand, 0),
        )

        self.coordinator.invalidate_gap("loss without frame identity")
        result = self.coordinator.advance_decoder_watermark(3)

        self.assertEqual(1, len(result))
        self.assertEqual(
            CandidateDisposition.IGNORED_GAPPED_HAND,
            result[0].disposition,
        )
        self.assertEqual(PublicHistoryStatus.GAPPED, self.coordinator.status)

    def test_late_first_view_of_new_hand_stops_exposing_old_history(self):
        self.submit_and_advance(1, snapshot_for_prefix(self.target, 0))
        next_hand = replace(self.target, hand_id="next-hand")

        result = self.submit_and_advance(
            2,
            snapshot_for_prefix(next_hand, 1),
        )

        self.assertEqual(CandidateDisposition.REJECTED, result.disposition)
        view = self.coordinator.view()
        self.assertEqual(PublicHistoryStatus.WAITING_FOR_ANCHOR, view.status)
        self.assertEqual("next-hand", view.hand_id)
        self.assertIsNone(view.history)
        old_readiness = self.coordinator.readiness(
            hand_id=self.target.hand_id,
            through_frame_id=1,
        )
        self.assertFalse(old_readiness.ready)

    def test_malformed_first_view_with_new_hand_id_hides_old_history(self):
        self.submit_and_advance(1, snapshot_for_prefix(self.target, 0))
        next_hand = replace(self.target, hand_id="next-hand")
        malformed = snapshot_for_prefix(next_hand, 0)
        malformed.board_state.community_cards = ["Ah"]

        result = self.submit_and_advance(2, malformed)

        self.assertEqual(CandidateDisposition.REJECTED, result.disposition)
        view = self.coordinator.view()
        self.assertEqual(PublicHistoryStatus.WAITING_FOR_ANCHOR, view.status)
        self.assertEqual("next-hand", view.hand_id)
        self.assertIsNone(view.history)
        stale_old = self.submit_and_advance(
            3,
            snapshot_for_prefix(self.target, 0),
        )
        self.assertEqual(
            CandidateDisposition.REJECTED,
            stale_old.disposition,
        )
        self.assertEqual(
            PublicHistoryStatus.WAITING_FOR_ANCHOR,
            self.coordinator.status,
        )

    def test_retired_hand_id_cannot_replace_a_newer_hand(self):
        self.submit_and_advance(1, snapshot_for_prefix(self.target, 0))
        next_hand = replace(self.target, hand_id="newer-hand")
        self.submit_and_advance(2, snapshot_for_prefix(next_hand, 0))

        rollback = self.submit_and_advance(
            3,
            snapshot_for_prefix(self.target, 0),
        )

        self.assertEqual(CandidateDisposition.REJECTED, rollback.disposition)
        view = self.coordinator.view()
        self.assertEqual(PublicHistoryStatus.TRACKING, view.status)
        self.assertEqual("newer-hand", view.hand_id)

    def test_candidate_at_or_behind_watermark_is_rejected(self):
        self.coordinator.advance_decoder_watermark(4)

        with self.assertRaisesRegex(ValueError, "behind decoder watermark"):
            self.coordinator.submit_candidate(
                4,
                snapshot_for_prefix(self.target, 0),
            )

    def test_watermark_cannot_move_backwards(self):
        self.coordinator.advance_decoder_watermark(4)
        with self.assertRaisesRegex(ValueError, "cannot move backwards"):
            self.coordinator.advance_decoder_watermark(3)

    def test_history_if_ready_never_returns_unproven_prefix(self):
        self.submit_and_advance(1, snapshot_for_prefix(self.target, 0))
        self.assertIsNotNone(
            self.coordinator.history_if_ready(
                hand_id=self.target.hand_id,
                through_frame_id=1,
            )
        )
        self.assertIsNone(
            self.coordinator.history_if_ready(
                hand_id=self.target.hand_id,
                through_frame_id=2,
            )
        )

    def test_old_frame_never_receives_a_newer_transcript(self):
        self.submit_and_advance(1, snapshot_for_prefix(self.target, 0))
        self.submit_and_advance(3, snapshot_for_prefix(self.target, 1))

        old = self.coordinator.readiness(
            hand_id=self.target.hand_id,
            through_frame_id=1,
        )

        self.assertFalse(old.ready)
        self.assertIsNone(old.history)
        self.assertIn("historical frame 1", old.reason)


if __name__ == "__main__":
    unittest.main()
