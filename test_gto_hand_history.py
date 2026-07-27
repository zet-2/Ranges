from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import unittest

from gto_hand_history import (
    HandEvent,
    HandSeat,
    PublicHandHistory,
    PublicHandHistoryError,
    public_hand_fingerprint,
    public_hand_from_wire,
    public_hand_to_wire,
    replay_public_hand,
)
from gto_remote.protocol import (
    PROTOCOL_SCHEMA_VERSION,
    RemoteProtocolError,
    build_evaluate_request,
    decision_state_from_wire,
    decision_state_to_wire,
)
from live_gto import LiveDecisionState


def six_max_seats() -> tuple[HandSeat, ...]:
    # With button on physical seat 3, clockwise order is:
    # BTN(3), SB(4), BB(5), UTG(0), HJ(1), CO(2).
    positions = {
        0: "UTG",
        1: "HJ",
        2: "CO",
        3: "BTN",
        4: "SB",
        5: "BB",
    }
    return tuple(
        HandSeat(seat, positions[seat], Decimal("100"))
        for seat in range(6)
    )


def heads_up_to_turn_history() -> PublicHandHistory:
    events = (
        HandEvent(0, "FOLD", "PREFLOP", actor_seat=0),
        HandEvent(1, "FOLD", "PREFLOP", actor_seat=1),
        HandEvent(2, "FOLD", "PREFLOP", actor_seat=2),
        HandEvent(
            3,
            "RAISE_TO",
            "PREFLOP",
            actor_seat=3,
            amount_to_bb=Decimal("2.5"),
        ),
        HandEvent(4, "FOLD", "PREFLOP", actor_seat=4),
        HandEvent(
            5,
            "CALL",
            "PREFLOP",
            actor_seat=5,
            amount_to_bb=Decimal("2.5"),
        ),
        HandEvent(6, "DEAL_FLOP", "FLOP", cards=("2c", "7d", "Jh")),
        HandEvent(7, "CHECK", "FLOP", actor_seat=5),
        HandEvent(
            8,
            "BET_TO",
            "FLOP",
            actor_seat=3,
            amount_to_bb=Decimal("3"),
        ),
        HandEvent(
            9,
            "CALL",
            "FLOP",
            actor_seat=5,
            amount_to_bb=Decimal("3"),
        ),
        HandEvent(10, "DEAL_TURN", "TURN", cards=("As",)),
    )
    return PublicHandHistory(
        hand_id="hand-complete-1",
        button_seat=3,
        small_blind_bb=Decimal("0.5"),
        big_blind_bb=Decimal("1"),
        ante_bb=Decimal(0),
        rake_rate_pct=Decimal("5"),
        rake_cap_bb=Decimal("0.5"),
        seats=six_max_seats(),
        events=events,
    )


class PublicHandReplayTests(unittest.TestCase):
    def test_replays_every_action_and_preserves_range_conditioning_path(self):
        state = replay_public_hand(heads_up_to_turn_history())

        self.assertEqual("TURN", state.street)
        self.assertEqual(("2c", "7d", "Jh", "As"), state.board)
        self.assertEqual(5, state.actor_seat)
        self.assertEqual(Decimal("11.5"), state.pot_bb)
        self.assertEqual(Decimal(0), state.amount_to_call_bb)
        self.assertEqual(frozenset({3, 5}), state.live_seats)
        self.assertEqual(frozenset({0, 1, 2, 4}), state.folded)
        self.assertEqual(Decimal("94.5"), state.stack_map[3])
        self.assertEqual(Decimal("94.5"), state.stack_map[5])
        self.assertEqual(("CHECK", "BET_TO", "ALL_IN_TO"), state.legal_actions)
        self.assertFalse(state.round_closed)
        self.assertFalse(state.terminal)
        self.assertEqual(11, state.next_sequence)

    def test_supports_multiway_paths_instead_of_forcing_hu(self):
        events = (
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
            HandEvent(6, "DEAL_FLOP", "FLOP", cards=("Ac", "Kd", "7s")),
        )
        history = replace(heads_up_to_turn_history(), events=events)
        state = history.replay()

        self.assertEqual(frozenset({2, 3, 4, 5}), state.live_seats)
        self.assertEqual(4, state.actor_seat)
        self.assertEqual(Decimal("4"), state.pot_bb)

    def test_rejects_missing_out_of_turn_and_illegal_actions(self):
        original = heads_up_to_turn_history()
        cases = (
            (
                replace(
                    original,
                    events=(
                        replace(original.events[0], sequence=1),
                        *original.events[1:],
                    ),
                ),
                "contiguous",
            ),
            (
                replace(
                    original,
                    events=(
                        replace(original.events[0], actor_seat=1),
                        *original.events[1:],
                    ),
                ),
                "expected 0",
            ),
            (
                replace(
                    original,
                    events=(
                        *original.events[:3],
                        replace(
                            original.events[3],
                            amount_to_bb=Decimal("1.5"),
                        ),
                        *original.events[4:],
                    ),
                ),
                "minimum full raise",
            ),
            (
                replace(
                    original,
                    events=(
                        *original.events[:10],
                        replace(original.events[10], cards=("2c",)),
                    ),
                ),
                "cannot repeat",
            ),
        )
        for history, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(PublicHandHistoryError, message):
                    replay_public_hand(history)

    def test_short_all_in_does_not_reopen_betting(self):
        short_seats = tuple(
            replace(seat, starting_stack_bb=Decimal("3.5"))
            if seat.seat == 2
            else seat
            for seat in six_max_seats()
        )
        events = (
            HandEvent(0, "FOLD", "PREFLOP", actor_seat=0),
            HandEvent(
                1,
                "RAISE_TO",
                "PREFLOP",
                actor_seat=1,
                amount_to_bb=Decimal("3"),
            ),
            HandEvent(
                2,
                "ALL_IN_TO",
                "PREFLOP",
                actor_seat=2,
                amount_to_bb=Decimal("3.5"),
            ),
            HandEvent(3, "FOLD", "PREFLOP", actor_seat=3),
            HandEvent(4, "FOLD", "PREFLOP", actor_seat=4),
            HandEvent(5, "FOLD", "PREFLOP", actor_seat=5),
            # HJ already made the last full raise.  CO's +0.5 short shove lets
            # HJ call/fold, but does not grant another raise.
            HandEvent(
                6,
                "RAISE_TO",
                "PREFLOP",
                actor_seat=1,
                amount_to_bb=Decimal("8"),
            ),
        )
        history = replace(
            heads_up_to_turn_history(),
            seats=short_seats,
            events=events,
        )
        with self.assertRaisesRegex(
            PublicHandHistoryError,
            "not reopened",
        ):
            history.replay()


class PublicHandWireTests(unittest.TestCase):
    def test_strict_wire_round_trip_and_fingerprint(self):
        history = heads_up_to_turn_history()
        wire = public_hand_to_wire(history)
        restored = public_hand_from_wire(wire)

        self.assertEqual(history, restored)
        self.assertEqual("2.5", wire["events"][3]["amount_to_bb"])
        self.assertEqual(
            public_hand_fingerprint(history),
            public_hand_fingerprint(restored),
        )
        self.assertRegex(public_hand_fingerprint(history), r"^[0-9a-f]{64}$")

    def test_wire_rejects_float_amounts_and_unknown_fields(self):
        wire = public_hand_to_wire(heads_up_to_turn_history())
        wire["events"][3]["amount_to_bb"] = 2.5
        with self.assertRaisesRegex(
            PublicHandHistoryError,
            "decimal JSON string",
        ):
            public_hand_from_wire(wire)

        wire = public_hand_to_wire(heads_up_to_turn_history())
        wire["unexpected"] = True
        with self.assertRaisesRegex(PublicHandHistoryError, "unexpected"):
            public_hand_from_wire(wire)

    def test_remote_decision_binds_to_the_full_replayed_hand(self):
        history = heads_up_to_turn_history()
        state = LiveDecisionState(
            hand_id=history.hand_id,
            street="TURN",
            board=("2c", "7d", "Jh", "As"),
            hero_combo=("Qc", "Qd"),
            hero_position="BB",
            villain_position="BTN",
            hero_is_oop=True,
            active_villains=1,
            pot_bb=Decimal("11.5"),
            hero_stack_bb=Decimal("94.5"),
            villain_stack_bb=Decimal("94.5"),
            hero_current_bet_bb=Decimal(0),
            villain_current_bet_bb=Decimal(0),
            amount_to_call_bb=Decimal(0),
            legal_actions=("CHECK", "BET"),
            street_root_confirmed=True,
            public_hand=history,
        )
        wire = decision_state_to_wire(state)
        restored = decision_state_from_wire(wire)
        request = build_evaluate_request("full-hand-request", state)

        self.assertEqual(state, restored)
        self.assertEqual(history.hand_id, wire["public_hand"]["hand_id"])
        self.assertEqual(PROTOCOL_SCHEMA_VERSION, request["schema_version"])

        for changed, message in (
            (replace(state, hand_id="other"), "different hand_id"),
            (replace(state, street="RIVER"), "board must contain"),
            (replace(state, hero_position="BTN"), "action on Hero"),
            (replace(state, active_villains=2), "active players"),
            (replace(state, amount_to_call_bb=Decimal("1")), "call amount"),
            (
                replace(state, legal_actions=("FOLD", "CALL")),
                "legal_actions contradict",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(RemoteProtocolError, message):
                    decision_state_to_wire(changed)


if __name__ == "__main__":
    unittest.main()
