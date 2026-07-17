#!/usr/bin/env python3
"""Deterministic tests for live preflop observation reconstruction."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
import unittest

from preflop_observation import (
    PreflopObservationError,
    canonical_position,
    current_decision,
    terminal_from_history,
)


POSITIONS = ("UTG", "MP", "CO", "BTN", "SB", "BB")


def player(
    seat: int,
    position: str,
    *,
    stack: object = 100,
    bet: object = 0,
    status: str = "ACTIVE",
    hero: bool = False,
):
    return SimpleNamespace(
        seat_index=seat,
        name=position,
        stack_size=stack,
        current_bet=bet,
        status=status,
        is_hero=hero,
    )


def snapshot(
    street: str,
    players,
    *,
    dealer: int = 3,
    actor: int = -1,
    hand_id: str = "hand-17",
):
    return SimpleNamespace(
        hand_id=hand_id,
        meta_info=SimpleNamespace(current_street=street),
        players=list(players),
        dealer_seat_index=dealer,
        action_on_seat_index=actor,
    )


def six_players(
    *,
    stacks=None,
    bets=None,
    statuses=None,
    hero_position="CO",
):
    stacks = stacks or {}
    bets = bets or {}
    statuses = statuses or {}
    result = []
    for seat, position in enumerate(POSITIONS):
        result.append(
            player(
                seat,
                position,
                stack=stacks.get(position, 100),
                bet=bets.get(position, 0),
                status=statuses.get(position, "ACTIVE"),
                hero=position == hero_position,
            )
        )
    return result


class CurrentDecisionTests(unittest.TestCase):
    def test_current_decision_is_strict_and_maps_mp_to_hj(self):
        players = six_players(
            stacks={"UTG": "98", "CO": "97.5", "SB": "99.5", "BB": "99"},
            bets={"UTG": "2", "CO": "2.5", "SB": ".5", "BB": "1"},
            statuses={"UTG": "FOLDED"},
        )
        state = current_decision(snapshot("PREFLOP", players, actor=2))

        self.assertEqual(state.actor, "CO")
        self.assertFalse(state.terminal)
        self.assertEqual(state.folded, frozenset({"UTG"}))
        self.assertEqual(state.contribution_map["HJ"], Decimal("0"))
        self.assertEqual(state.contribution_map["CO"], Decimal("2.5"))
        self.assertEqual(state.initial_stack_map["UTG"], Decimal("100"))
        self.assertEqual(state.initial_stack_map["CO"], Decimal("100.0"))
        self.assertEqual(state.source, "current_preflop_decision")

        matcher_state = state.to_history_observation()
        self.assertEqual(matcher_state.actor, "CO")
        self.assertEqual(matcher_state.contribution_map, state.contribution_map)
        self.assertEqual(matcher_state.folded, state.folded)

    def test_current_decision_requires_action_on_hero(self):
        with self.assertRaisesRegex(PreflopObservationError, "not confirmed on Hero"):
            current_decision(snapshot("PREFLOP", six_players(), actor=1))

    def test_aliases_and_duplicate_canonical_positions(self):
        self.assertEqual(canonical_position("mp"), "HJ")
        self.assertEqual(canonical_position(" hj "), "HJ")
        players = six_players()
        players[1].name = "HJ"
        players[2].name = "MP"
        with self.assertRaisesRegex(PreflopObservationError, "duplicate canonical"):
            current_decision(snapshot("PREFLOP", players, actor=2))

    def test_position_labels_must_match_dealer_and_seat_order(self):
        with self.assertRaisesRegex(PreflopObservationError, "dealer/seat order"):
            current_decision(
                snapshot("PREFLOP", six_players(), dealer=0, actor=2)
            )

    def test_current_all_in_positions_are_preserved(self):
        players = six_players(
            stacks={"UTG": 0},
            bets={"UTG": 100, "SB": ".5", "BB": 1},
            statuses={"UTG": "ALL_IN"},
        )
        state = current_decision(snapshot("PREFLOP", players, actor=2))
        self.assertEqual(frozenset({"UTG"}), state.all_in)
        self.assertEqual(
            frozenset({"UTG"}), state.to_history_observation().all_in
        )


class TerminalObservationTests(unittest.TestCase):
    def make_heads_up_transition(self):
        pre = snapshot(
            "PREFLOP",
            six_players(
                stacks={"UTG": 98, "MP": 98, "CO": 98, "BTN": 98, "SB": 97, "BB": 97},
                bets={"UTG": 2, "MP": 2, "CO": 2, "BTN": 2, "SB": 3, "BB": 3},
            ),
        )
        flop = snapshot(
            "FLOP",
            six_players(
                stacks={"UTG": 98, "MP": 98, "CO": 98, "BTN": 98, "SB": 95, "BB": 97},
                bets={"SB": 2, "BB": 0},
                statuses={
                    "UTG": "FOLDED",
                    "MP": "FOLDED",
                    "CO": "FOLDED",
                    "BTN": "FOLDED",
                    "SB": "ACTIVE",
                    "BB": "ACTIVE",
                },
            ),
        )
        return pre, flop

    def test_successful_heads_up_terminal_subtracts_flop_bet(self):
        pre, flop = self.make_heads_up_transition()
        history = SimpleNamespace(snapshots=[pre])
        state = terminal_from_history(history, flop)

        self.assertTrue(state.terminal)
        self.assertIsNone(state.actor)
        self.assertEqual(state.live_positions, frozenset({"SB", "BB"}))
        # SB has 95 behind plus a 2 BB flop bet, so only 3 BB were preflop.
        self.assertEqual(state.contribution_map["SB"], Decimal("3"))
        self.assertEqual(state.contribution_map["BB"], Decimal("3"))
        self.assertEqual(state.contribution_map["UTG"], Decimal("2"))
        self.assertEqual(state.initial_stack_map["HJ"], Decimal("100"))
        self.assertEqual(state.provenance.preflop_index, 0)
        self.assertEqual(state.provenance.flop_index, 1)

        matcher_state = state.to_history_observation()
        self.assertEqual(matcher_state.survivors, frozenset({"SB", "BB"}))

    def test_current_snapshot_cannot_reuse_an_earlier_hand_transition(self):
        old_pre, old_flop = self.make_heads_up_transition()
        old_pre.hand_id = old_flop.hand_id = "old-hand"
        _, current_flop = self.make_heads_up_transition()
        current_flop.hand_id = "current-hand"

        with self.assertRaisesRegex(
            PreflopObservationError,
            "no PREFLOP snapshot for current hand 'current-hand'",
        ):
            terminal_from_history([old_pre, old_flop], current_flop)

    def test_current_snapshot_selects_only_its_exact_hand_transition(self):
        old_pre, old_flop = self.make_heads_up_transition()
        old_pre.hand_id = old_flop.hand_id = "old-hand"
        current_pre, current_flop = self.make_heads_up_transition()
        current_pre.hand_id = current_flop.hand_id = "current-hand"

        state = terminal_from_history(
            [old_pre, old_flop, current_pre],
            current_flop,
        )

        self.assertEqual(state.provenance.hand_id, "current-hand")
        self.assertEqual(state.provenance.preflop_index, 2)
        self.assertEqual(state.provenance.flop_index, 3)

    def test_missing_preflop_is_rejected(self):
        _, flop = self.make_heads_up_transition()
        with self.assertRaisesRegex(PreflopObservationError, "no PREFLOP"):
            terminal_from_history([flop])

    def test_first_postflop_capture_turn_is_rejected(self):
        pre, flop = self.make_heads_up_transition()
        flop.meta_info.current_street = "TURN"
        with self.assertRaisesRegex(PreflopObservationError, "first postflop capture is TURN"):
            terminal_from_history([pre, flop])

    def test_dealer_mismatch_is_rejected(self):
        pre, flop = self.make_heads_up_transition()
        flop.dealer_seat_index = 4
        with self.assertRaisesRegex(PreflopObservationError, "dealer/seat order"):
            terminal_from_history([pre, flop])

    def test_seat_position_mapping_mismatch_is_rejected(self):
        pre, flop = self.make_heads_up_transition()
        flop.players[0].name, flop.players[1].name = (
            flop.players[1].name,
            flop.players[0].name,
        )
        with self.assertRaisesRegex(PreflopObservationError, "position labels"):
            terminal_from_history([pre, flop])

    def test_negative_inferred_contribution_is_rejected(self):
        pre, flop = self.make_heads_up_transition()
        # Initial UTG total was 100; 101 still present at the flop is impossible.
        flop.players[0].stack_size = 101
        with self.assertRaisesRegex(PreflopObservationError, "negative inferred"):
            terminal_from_history([pre, flop])

    def test_multiway_is_rejected_by_default_but_can_be_observed(self):
        pre, flop = self.make_heads_up_transition()
        pre.players[0].stack_size = 97
        pre.players[0].current_bet = 3
        flop.players[0].stack_size = 97
        flop.players[0].status = "ACTIVE"
        with self.assertRaisesRegex(PreflopObservationError, "not heads-up"):
            terminal_from_history([pre, flop])

        state = terminal_from_history([pre, flop], require_heads_up=False)
        self.assertEqual(state.live_positions, frozenset({"UTG", "SB", "BB"}))

    def test_postflop_fold_cannot_be_relabelled_as_preflop_fold(self):
        pre, flop = self.make_heads_up_transition()
        # BTN matched the final 3 BB preflop price and therefore saw the flop.
        # Its FOLDED status on the first captured flop frame must not manufacture
        # a heads-up preflop handoff.
        pre.players[3].stack_size = 97
        pre.players[3].current_bet = 3
        flop.players[3].stack_size = 97
        with self.assertRaisesRegex(PreflopObservationError, "not heads-up"):
            terminal_from_history([pre, flop])

        observed = terminal_from_history([pre, flop], require_heads_up=False)
        self.assertIn("BTN", observed.live_positions)

    def test_hand_id_mismatch_is_rejected(self):
        pre, flop = self.make_heads_up_transition()
        flop.hand_id = "different-hand"
        with self.assertRaisesRegex(
            PreflopObservationError,
            "no PREFLOP snapshot for current hand 'different-hand'",
        ):
            terminal_from_history([pre, flop])

    def test_terminal_reconstruction_requires_a_current_hand_id(self):
        pre, flop = self.make_heads_up_transition()
        pre.hand_id = flop.hand_id = ""
        with self.assertRaisesRegex(PreflopObservationError, "non-empty hand_id"):
            terminal_from_history([pre, flop])


if __name__ == "__main__":
    unittest.main()
