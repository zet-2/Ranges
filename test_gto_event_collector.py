from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace
import unittest

from gto_event_collector import (
    PublicEventCollectorConfig,
    PublicHandEventRecorder,
)
from gto_hand_history import PublicHandHistory, replay_public_hand
from test_gto_hand_history import heads_up_to_turn_history


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


if __name__ == "__main__":
    unittest.main()
