"""Tests for fail-closed public hand-event collection."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import threading
from types import SimpleNamespace
import unittest

from gto_event_collector import (
    PublicEventCollectionError,
    PublicEventCollectorConfig,
    PublicEventProbeStatus,
    PublicHandEventRecorder,
)
from gto_hand_history import PublicHandHistory, replay_public_hand
from tests.test_gto_hand_history import heads_up_to_turn_history


def snapshot_for_prefix(
    target: PublicHandHistory,
    event_count: int,
    *,
    include_action_overlay: bool = True,
):
    prefix = replace(target, events=target.events[:event_count])
    state = replay_public_hand(prefix)
    last_event = target.events[event_count - 1] if event_count else None
    visible_by_seat = {}
    if (
        include_action_overlay
        and last_event is not None
        and last_event.actor_seat is not None
    ):
        visible_by_seat[last_event.actor_seat] = {
            "BET_TO": "BET",
            "RAISE_TO": "RAISE",
            "ALL_IN_TO": "ALL-IN",
        }.get(last_event.kind, last_event.kind)
    seat_by_id = {seat.seat: seat for seat in target.seats}
    players = []
    for seat_id in sorted(seat_by_id):
        if seat_id in state.folded:
            status = "FOLDED"
        elif seat_id in state.all_in:
            status = "ALL_IN"
        else:
            status = "ACTIVE"
        players.append(
            SimpleNamespace(
                seat_index=seat_id,
                name=seat_by_id[seat_id].position,
                stack_size=float(state.stack_map[seat_id]),
                current_bet=float(state.street_contribution_map[seat_id]),
                status=status,
                visible_action=visible_by_seat.get(seat_id, ""),
                is_dealer=seat_id == target.button_seat,
            )
        )
    return SimpleNamespace(
        hand_id=target.hand_id,
        meta_info=SimpleNamespace(current_street=state.street),
        board_state=SimpleNamespace(
            community_cards=list(state.board),
            total_pot=float(state.pot_bb),
        ),
        dealer_seat_index=target.button_seat,
        action_on_seat_index=(
            state.actor_seat if state.actor_seat is not None else -1
        ),
        players=players,
    )


class PublicEventRecorderTests(unittest.TestCase):
    def test_snapshot_with_too_few_players_fails_closed_without_raising(self):
        target = heads_up_to_turn_history()
        partial = snapshot_for_prefix(target, 0)
        partial.players = partial.players[:1]
        recorder = PublicHandEventRecorder()

        self.assertIsNone(recorder.observe(partial))
        self.assertFalse(recorder.complete)
        self.assertIn("2..6 occupied seats", recorder.error)

    def test_unproven_all_in_status_is_not_accepted(self):
        target = heads_up_to_turn_history()
        initial = snapshot_for_prefix(target, 0)
        initial.players[0].status = "ALL_IN"
        recorder = PublicHandEventRecorder()

        self.assertIsNone(recorder.observe(initial))
        self.assertIn("untouched forced-bet state", recorder.error)

    def test_application_history_exposes_only_the_gap_free_transcript(self):
        from poker_assistant import HandHistory

        target = heads_up_to_turn_history()
        application_history = HandHistory()
        for event_count in range(len(target.events) + 1):
            application_history.add_snapshot(
                snapshot_for_prefix(target, event_count)
            )

        self.assertEqual(target, application_history.public_hand)
        self.assertEqual("", application_history.public_hand_error)

    def test_reconstructs_a_complete_preflop_to_turn_path(self):
        target = heads_up_to_turn_history()
        recorder = PublicHandEventRecorder(
            PublicEventCollectorConfig(
                small_blind_bb=target.small_blind_bb,
                big_blind_bb=target.big_blind_bb,
                ante_bb=target.ante_bb,
                rake_rate_pct=target.rake_rate_pct,
                rake_cap_bb=target.rake_cap_bb,
            )
        )

        for event_count in range(len(target.events) + 1):
            result = recorder.observe(snapshot_for_prefix(target, event_count))
            self.assertIsNotNone(result, recorder.error)

        self.assertTrue(recorder.complete)
        self.assertEqual(target, recorder.history)
        self.assertEqual("", recorder.error)

    def test_duplicate_frames_do_not_duplicate_events(self):
        target = heads_up_to_turn_history()
        recorder = PublicHandEventRecorder()
        initial = snapshot_for_prefix(target, 0)
        recorder.observe(initial)
        recorder.observe(initial)
        recorder.observe(snapshot_for_prefix(target, 1))
        recorder.observe(snapshot_for_prefix(target, 1))

        self.assertTrue(recorder.complete)
        self.assertEqual(target.events[:1], recorder.history.events)

    def test_skipped_or_late_frames_fail_closed(self):
        target = heads_up_to_turn_history()

        skipped = PublicHandEventRecorder()
        skipped.observe(snapshot_for_prefix(target, 0))
        self.assertIsNone(skipped.observe(snapshot_for_prefix(target, 2)))
        self.assertFalse(skipped.complete)
        self.assertIn("skipped", skipped.error)

        late = PublicHandEventRecorder()
        self.assertIsNone(late.observe(snapshot_for_prefix(target, 1)))
        self.assertFalse(late.complete)
        self.assertIn("untouched forced-bet state", late.error)

    def test_a_check_requires_an_overlay_or_proven_actor_change(self):
        target = heads_up_to_turn_history()
        recorder = PublicHandEventRecorder()
        for event_count in range(8):
            overlay = event_count != 8
            recorder.observe(
                snapshot_for_prefix(
                    target,
                    event_count,
                    include_action_overlay=overlay,
                )
            )
        # Prefix 7 is the flop deal. Prefix 8 is BB's check and has no chip
        # delta. Remove both overlay and actor transition evidence.
        no_check = snapshot_for_prefix(
            target,
            8,
            include_action_overlay=False,
        )
        no_check.action_on_seat_index = -1
        recorder.observe(no_check)
        self.assertEqual(7, len(recorder.history.events))

        # The following BTN bet now proves that an intermediate action was
        # skipped, so the recorder refuses the transcript.
        self.assertIsNone(recorder.observe(snapshot_for_prefix(target, 9)))
        self.assertIn("expected actor", recorder.error)

    def test_antes_are_dead_money_not_part_of_the_call_target(self):
        target = replace(
            heads_up_to_turn_history(),
            ante_bb=Decimal("0.1"),
        )
        # Existing voluntary amount-to values remain 2.5 BB: the replay treats
        # antes as dead money, not as live preflop contributions.
        state = replay_public_hand(replace(target, events=target.events[:4]))
        self.assertEqual(Decimal("2.5"), state.street_contribution_map[3])
        self.assertEqual(Decimal("4.6"), state.pot_bb)

        recorder = PublicHandEventRecorder(
            PublicEventCollectorConfig(ante_bb=Decimal("0.1"))
        )
        self.assertIsNotNone(recorder.observe(snapshot_for_prefix(target, 0)))
        self.assertEqual(target.seats, recorder.history.seats)

    def test_rejected_probe_does_not_poison_or_mutate_recorder(self):
        target = heads_up_to_turn_history()
        recorder = PublicHandEventRecorder()
        recorder.observe(snapshot_for_prefix(target, 0))
        initial_history = recorder.history
        initial_version = recorder.version

        rejected = recorder.probe(snapshot_for_prefix(target, 2))

        self.assertEqual(PublicEventProbeStatus.REJECTED, rejected.status)
        self.assertFalse(rejected.accepted)
        self.assertIn("skipped", rejected.error)
        self.assertEqual(initial_history, recorder.history)
        self.assertEqual(initial_version, recorder.version)
        self.assertEqual("", recorder.error)
        self.assertTrue(recorder.complete)

        accepted = recorder.probe(snapshot_for_prefix(target, 1))
        recorder.commit(accepted)
        self.assertEqual(target.events[:1], recorder.history.events)
        self.assertEqual("", recorder.error)

    def test_probe_commit_rejects_stale_or_rejected_transaction(self):
        target = heads_up_to_turn_history()
        recorder = PublicHandEventRecorder()
        first = recorder.probe(snapshot_for_prefix(target, 0))
        stale = recorder.probe(snapshot_for_prefix(target, 0))
        recorder.commit(first)

        with self.assertRaisesRegex(
            PublicEventCollectionError,
            "stale observation probe",
        ):
            recorder.commit(stale)

        rejected = recorder.probe(snapshot_for_prefix(target, 2))
        with self.assertRaisesRegex(
            PublicEventCollectionError,
            "cannot commit rejected",
        ):
            recorder.commit(rejected)

        foreign = PublicHandEventRecorder().probe(
            snapshot_for_prefix(target, 0)
        )
        with self.assertRaisesRegex(
            PublicEventCollectionError,
            "another recorder",
        ):
            recorder.commit(foreign)

    def test_concurrent_commits_allow_exactly_one_probe(self):
        target = heads_up_to_turn_history()
        recorder = PublicHandEventRecorder()
        probes = (
            recorder.probe(snapshot_for_prefix(target, 0)),
            recorder.probe(snapshot_for_prefix(target, 0)),
        )
        barrier = threading.Barrier(3)
        outcomes = []
        outcomes_lock = threading.Lock()

        def commit(probe):
            barrier.wait()
            try:
                recorder.commit(probe)
                result = "accepted"
            except PublicEventCollectionError as error:
                result = str(error)
            with outcomes_lock:
                outcomes.append(result)

        threads = [
            threading.Thread(target=commit, args=(probe,))
            for probe in probes
        ]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=2)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(1, outcomes.count("accepted"))
        self.assertEqual(1, sum("stale" in outcome for outcome in outcomes))

    def test_probe_reports_duplicate_advance_and_new_hand_anchor(self):
        target = heads_up_to_turn_history()
        recorder = PublicHandEventRecorder()

        anchor = recorder.probe(snapshot_for_prefix(target, 0))
        self.assertEqual(PublicEventProbeStatus.ANCHORED, anchor.status)
        self.assertEqual(0, anchor.events_added)
        recorder.commit(anchor)

        duplicate = recorder.probe(snapshot_for_prefix(target, 0))
        self.assertEqual(PublicEventProbeStatus.DUPLICATE, duplicate.status)
        self.assertEqual(0, duplicate.events_added)
        recorder.commit(duplicate)

        advance = recorder.probe(snapshot_for_prefix(target, 1))
        self.assertEqual(PublicEventProbeStatus.ADVANCED, advance.status)
        self.assertEqual(1, advance.events_added)
        recorder.commit(advance)

        next_hand = replace(target, hand_id="next-hand")
        replacement = recorder.probe(snapshot_for_prefix(next_hand, 0))
        self.assertEqual(
            PublicEventProbeStatus.NEW_HAND_ANCHORED,
            replacement.status,
        )
        self.assertTrue(replacement.replaces_hand)
        recorder.commit(replacement)
        self.assertEqual("next-hand", recorder.history.hand_id)
        self.assertEqual((), recorder.history.events)

    def test_explicit_gap_invalidates_until_reset_or_new_hand(self):
        target = heads_up_to_turn_history()
        recorder = PublicHandEventRecorder()
        recorder.observe(snapshot_for_prefix(target, 0))
        retained_history = recorder.history

        recorder.invalidate_gap("keyframe queue overflow")

        self.assertFalse(recorder.complete)
        self.assertEqual(retained_history, recorder.history)
        self.assertIn("capture gap", recorder.error)
        rejected = recorder.probe(snapshot_for_prefix(target, 1))
        self.assertEqual(PublicEventProbeStatus.REJECTED, rejected.status)
        self.assertIn("capture gap", rejected.error)

        next_hand = replace(target, hand_id="recovered")
        recovery = recorder.probe(snapshot_for_prefix(next_hand, 0))
        self.assertEqual(
            PublicEventProbeStatus.NEW_HAND_ANCHORED,
            recovery.status,
        )
        recorder.commit(recovery)
        self.assertTrue(recorder.complete)
        self.assertEqual("recovered", recorder.history.hand_id)

    def test_invalidate_gap_requires_reason(self):
        with self.assertRaisesRegex(
            PublicEventCollectionError,
            "reason cannot be empty",
        ):
            PublicHandEventRecorder().invalidate_gap("")


if __name__ == "__main__":
    unittest.main()
