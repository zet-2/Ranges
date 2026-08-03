#!/usr/bin/env python3
"""Deterministic regression tests for local poker facts and prompt inputs."""

import json
import os
import threading
import time
import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest import mock

from PIL import Image, ImageDraw


os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

import poker_assistant as app


def player(
    seat,
    position,
    *,
    hero=False,
    status="ACTIVE",
    bet=0.0,
    stack=100.0,
    action="",
):
    return app.Player(
        seat_index=seat,
        name=position,
        username="Hero" if hero else f"Villain{seat}",
        stack_size=stack,
        current_bet=bet,
        status=status,
        is_hero=hero,
        hole_cards=["6d", "4h"] if hero else None,
        visible_action=action,
    )


def snapshot(players, *, street="FLOP", board=None, dealer=0, pot=4.0):
    return app.GameSnapshot(
        meta_info=app.MetaInfo(current_street=street),
        board_state=app.BoardState(community_cards=board or [], total_pot=pot),
        dealer_seat_index=dealer,
        players=players,
        last_action_context=app.LastActionContext(hero_action_options=["Check", "Bet"]),
    )


def parallel_response(**overrides):
    """Build one complete strict FAST response for validator tests."""
    fields = {
        "Seen Hero": "6d 4h",
        "Seen Board": "Kc 7c 5d",
        "Seen Pot": "4 BB",
        "Seen Call": "0 BB",
        "Seen Stack": "100 BB",
        "Seen Live Seats": "S0:A",
        "Seen Effective Stack": "100 BB",
        "Seen Dealer": "S0",
        "Options": "Check, Bet",
        "Action": "Check",
        "Size": "0",
        "Why": "Keep the pot controlled while realizing the draw.",
    }
    fields.update(overrides)
    return "\n".join(
        f"**{label}:** {fields[label]}"
        for label in app.PARALLEL_STRATEGY_LABELS
        if label in fields
    )


class HandEvaluatorTests(unittest.TestCase):
    def test_screenshot_hand_is_open_ended_draw(self):
        result = app.HandEvaluator.evaluate(["6d", "4h"], ["Kc", "7c", "5d"])
        self.assertIn("High Card (K-high)", result)
        self.assertIn("Open-ended straight draw", result)
        self.assertIn("3 or 8", result)
        self.assertIn("8 nominal outs", result)

    def test_gutshot_and_double_gutshot(self):
        gutshot = app.HandEvaluator.evaluate(["6d", "4h"], ["Kc", "7c", "8d"])
        double_gutshot = app.HandEvaluator.evaluate(["5c", "Jh"], ["7c", "8d", "9s"])
        self.assertIn("Gutshot straight draw (5", gutshot)
        self.assertIn("Double-gutshot straight draw", double_gutshot)
        self.assertIn("6 or T", double_gutshot)

    def test_flush_and_combo_draws_are_hero_owned(self):
        flush_draw = app.HandEvaluator.evaluate(["Ah", "Jh"], ["2h", "7h", "Kc"])
        board_only = app.HandEvaluator.evaluate(["Ah", "Kd"], ["2c", "5c", "8c", "Jc"])
        combo = app.HandEvaluator.evaluate(["6c", "4c"], ["Kc", "7c", "5d"])
        self.assertIn("Flush draw (9 nominal outs)", flush_draw)
        self.assertNotIn("Flush draw", board_only)
        self.assertIn("Combo draw", combo)

    def test_made_straight_and_river_suppress_straight_draw(self):
        made = app.HandEvaluator.evaluate(["6d", "4h"], ["7c", "5d", "3s"])
        river = app.HandEvaluator.evaluate(["6d", "4h"], ["Kc", "7c", "5d", "Qs", "2h"])
        self.assertEqual("Straight (7-high)", made)
        self.assertNotIn("draw", river.lower())

    def test_board_texture(self):
        dynamic = app.HandEvaluator.board_texture(["Kc", "7c", "5d"])
        dry = app.HandEvaluator.board_texture(["Ks", "7d", "2c"])
        self.assertIn("two-tone", dynamic)
        self.assertIn("semi-connected", dynamic)
        self.assertIn("dynamic", dynamic)
        self.assertIn("rainbow", dry)
        self.assertIn("disconnected", dry)
        self.assertIn("dry", dry)


class StrategyContextTests(unittest.TestCase):
    def test_relative_position_uses_live_action_order(self):
        hero_co = player(5, "CO", hero=True)
        button = player(0, "BTN")
        big_blind = player(2, "BB")
        state = snapshot([hero_co, button], dealer=0)
        self.assertEqual("OOP", app.relative_postflop_position(state, hero_co, [button]))

        state.players = [hero_co, big_blind, button]
        self.assertEqual(
            "SANDWICHED",
            app.relative_postflop_position(state, hero_co, [big_blind, button]),
        )

        hero_mp = player(4, "MP", hero=True)
        state.players = [hero_mp, big_blind]
        self.assertEqual("IP", app.relative_postflop_position(state, hero_mp, [big_blind]))

    def test_multiway_context_and_prompt_facts(self):
        hero = player(4, "SB", hero=True)
        villains = [player(2, "BB"), player(5, "CO")]
        state = snapshot([*villains, hero], board=["Kc", "7c", "5d"], dealer=0)
        history = app.HandHistory(snapshots=[state], hand_id="test")
        details = app.HandEvaluator.evaluate_details(hero.hole_cards, state.board_state.community_cards)
        context = app.build_strategy_context(state, history, details)
        metrics = {"final_pot": 3.78, "spr": 9.4}
        prompt = app.generate_strategy_prompt_fast(
            "{}", "No reliable betting actions observed.", "", state, metrics,
            "No wager to call; Check is available", details["summary"], context,
        )

        self.assertEqual("MULTIWAY", context.pot_type)
        self.assertEqual(2, context.opponents_in_hand)
        self.assertIn("Open-ended straight draw", prompt)
        self.assertIn("MULTIWAY", prompt)
        self.assertIn("C-BET TERMINOLOGY ALLOWED: UNKNOWN", prompt)
        self.assertIn("NOT a remaining-action count", prompt)
        self.assertNotIn("Can act:", prompt)
        self.assertNotIn("with any 2 cards", prompt)
        self.assertNotIn('"Nothing" hand', prompt)

    def test_all_in_villain_still_defines_relative_position(self):
        hero = player(4, "BB", hero=True)
        villain = player(0, "BTN", status="ALL_IN")
        state = snapshot(
            [villain, hero],
            board=["Kc", "7c", "5d"],
            dealer=0,
        )
        history = app.HandHistory(snapshots=[state], hand_id="all-in-position")
        details = app.HandEvaluator.evaluate_details(
            hero.hole_cards,
            state.board_state.community_cards,
        )

        context = app.build_strategy_context(state, history, details)

        self.assertEqual(1, context.opponents_in_hand)
        self.assertEqual(0, context.opponents_eligible_to_act)
        self.assertEqual("OOP", context.relative_position)

    def test_response_sanitizes_unsupported_cbet_claim(self):
        context = app.StrategyContext(
            opponents_in_hand=1,
            opponents_eligible_to_act=1,
            opponents_before_hero=0,
            opponents_after_hero=1,
            pot_type="HEADS-UP",
            relative_position="OOP",
            action_flow="One opponent is positioned after Hero",
            board_texture="dry",
            made_hand="High Card",
            draws="NONE",
            preflop_aggressor="UNKNOWN",
            hero_was_pfa=None,
            current_street_aggressor="NONE",
            legal_actions="Check, Bet",
        )
        result = app.sanitize_strategy_response("Make a c-bet as a continuation bet.", context)
        self.assertNotIn("c-bet", result.lower())
        self.assertNotIn("continuation bet", result.lower())
        initiative = app.sanitize_strategy_response(
            "**Action:** Check\n**Size:** 0\n**Why:** We have the initiative.",
            context,
        )
        self.assertNotIn("we have the initiative", initiative.lower())

        illegal = app.sanitize_strategy_response(
            "**Action:** Fold\n**Size:** 0\n**Why:** cautious.", context
        )
        self.assertTrue(illegal.startswith("Strategy Error:"))

        composite = app.sanitize_strategy_response(
            "**Action:** Check/Fold\n**Size:** 0\n**Why:** cautious.", context
        )
        self.assertTrue(composite.startswith("Strategy Error:"))
        missing = app.sanitize_strategy_response("Check seems best.", context)
        self.assertTrue(missing.startswith("Strategy Error:"))

    def test_ip_counts_do_not_become_still_to_act_claims(self):
        hero = player(4, "CO", hero=True)
        villains = [player(0, "SB"), player(3, "MP")]
        state = snapshot([*villains, hero], board=["6h", "Jc", "8h"], dealer=5)
        history = app.HandHistory(snapshots=[state])
        details = app.HandEvaluator.evaluate_details(
            hero.hole_cards, state.board_state.community_cards
        )
        context = app.build_strategy_context(state, history, details)

        self.assertEqual((2, 0), (
            context.opponents_before_hero,
            context.opponents_after_hero,
        ))
        self.assertEqual("IP", context.relative_position)
        sanitized = app.sanitize_strategy_response(
            "**Action:** Check\n**Size:** 0\n**Why:** two others still to act.",
            context,
        )
        self.assertNotIn("still to act", sanitized.lower())
        self.assertIn("0 eligible opponents positioned after Hero", sanitized)

        sonnet_style = app.sanitize_strategy_response(
            "**Action:** Check\n**Size:** 0\n**Why:** With 3 opponents still able to act, "
            "the bettor plus two live hands behind create risk.",
            context,
        )
        self.assertNotIn("3 opponents still able to act", sonnet_style)
        self.assertNotIn("two live hands behind", sonnet_style)
        self.assertEqual(2, sonnet_style.count("0 eligible opponents positioned after Hero"))

    def test_preflop_co_has_only_three_positions_after_hero(self):
        hero = player(4, "CO", hero=True)
        villains = [
            player(2, "MP"),
            player(5, "BTN"),
            player(0, "SB"),
            player(1, "BB"),
        ]
        state = snapshot([*villains, hero], street="PREFLOP", board=[], dealer=5)
        details = app.HandEvaluator.evaluate_details(hero.hole_cards, [])
        context = app.build_strategy_context(
            state, app.HandHistory(snapshots=[state]), details
        )
        self.assertEqual(1, context.opponents_before_hero)
        self.assertEqual(3, context.opponents_after_hero)

    def test_turn_four_out_board_pair_call_is_overridden(self):
        hero = player(4, "UTG", hero=True, stack=61.3)
        hero.hole_cards = ["As", "4d"]
        state = snapshot(
            [player(0, "CO"), player(2, "SB"), hero],
            street="TURN",
            board=["8d", "6c", "5c", "5d"],
            dealer=1,
            pot=12.0,
        )
        state.last_action_context.amount_to_call = 6.0
        state.last_action_context.hero_action_options = ["Fold", "Call", "Raise"]
        details = app.HandEvaluator.evaluate_details(
            hero.hole_cards, state.board_state.community_cards
        )
        context = app.build_strategy_context(
            state, app.HandHistory(snapshots=[state]), details
        )
        guarded = app.apply_deterministic_strategy_guard(
            "**Action:** Call\n**Size:** 6 BB\n**Why:** Four outs have 32% equity.",
            state,
            context,
            details,
            {"pot_odds_pct": 33.3},
        )
        self.assertIn("**Action:** Fold", guarded)
        self.assertIn("8.7%", guarded)

    def test_multiway_flop_high_card_gutshot_call_is_overridden(self):
        hero = player(4, "UTG", hero=True, stack=62.3)
        hero.hole_cards = ["As", "4d"]
        state = snapshot(
            [player(0, "CO"), player(2, "SB"), player(3, "BB"), hero],
            street="FLOP",
            board=["8d", "6c", "5c"],
            dealer=1,
            pot=3.76,
        )
        state.last_action_context.amount_to_call = 1.0
        state.last_action_context.hero_action_options = ["Fold", "Call", "Raise"]
        details = app.HandEvaluator.evaluate_details(
            hero.hole_cards, state.board_state.community_cards
        )
        context = app.build_strategy_context(
            state, app.HandHistory(snapshots=[state]), details
        )
        guarded = app.apply_deterministic_strategy_guard(
            "**Action:** Call\n**Size:** 1 BB\n**Why:** A-high is premium.",
            state,
            context,
            details,
            {"pot_odds_pct": 21.0},
        )
        self.assertIn("**Action:** Fold", guarded)
        self.assertIn("16.5%", guarded)

    def test_missing_dealer_keeps_action_order_unknown(self):
        hero = player(4, "?", hero=True)
        villains = [player(0, "?"), player(3, "?")]
        state = snapshot([*villains, hero], board=["6h", "Jc", "8h"], dealer=-1)
        details = app.HandEvaluator.evaluate_details(
            hero.hole_cards, state.board_state.community_cards
        )
        context = app.build_strategy_context(
            state, app.HandHistory(snapshots=[state]), details
        )
        self.assertIsNone(context.opponents_before_hero)
        self.assertIsNone(context.opponents_after_hero)
        self.assertEqual("UNKNOWN", context.relative_position)
        self.assertIn("dealer is not confirmed", context.action_flow)


class VisionWireFormatTests(unittest.TestCase):
    @staticmethod
    def captures(buttons=None):
        return {
            "seat1": Image.new("RGB", (283, 135), (120, 10, 10)),
            "seat2": Image.new("RGB", (234, 146), (10, 120, 10)),
            "seat3": Image.new("RGB", (309, 165), (10, 10, 120)),
            "seat4": Image.new("RGB", (314, 119), (120, 120, 10)),
            "hero": Image.new("RGB", (263, 167), (120, 10, 120)),
            "seat6": Image.new("RGB", (340, 125), (10, 120, 120)),
            "board": Image.new("RGB", (346, 155), (80, 90, 100)),
            "buttons": buttons or FreshnessAndButtonTests.action_buttons(),
        }

    def test_compact_payload_materializes_the_existing_snapshot_model(self):
        payload = {
            "n": ["Dealer", "Folded", "", "Allin", "Hero", "Away"],
            "s": [90, 80, 0, 0, 95, 70],
            "w": [0, 0, 0, 12, 1, 0],
            "x": ["A", "F", "E", "I", "A", "S"],
            "d": 0,
            "v": ["CHECK", "FOLD", "", "BET", "", ""],
            "h": ["9c", "5h"],
            "b": ["6h", "Jc", "8h"],
            "p": 15.5,
            "o": ["FOLD", "CALL", "RAISE"],
            "c": 11,
        }

        state = app.parse_response(json.dumps(payload), hero_turn_confirmed=True)
        by_seat = {p.seat_index: p for p in state.players}

        self.assertEqual(["9c", "5h"], by_seat[4].hole_cards)
        self.assertEqual("ACTIVE", by_seat[0].status)
        self.assertEqual("FOLDED", by_seat[1].status)
        self.assertEqual("EMPTY", by_seat[2].status)
        self.assertEqual("ALL_IN", by_seat[3].status)
        self.assertEqual("SITTING_OUT", by_seat[5].status)
        self.assertEqual(["6h", "Jc", "8h"], state.board_state.community_cards)
        self.assertEqual(15.5, state.board_state.total_pot)
        self.assertEqual(["Fold", "Call", "Raise"], state.last_action_context.hero_action_options)
        self.assertEqual(11, state.last_action_context.amount_to_call)

    def test_all_in_chip_bubble_cannot_become_a_remaining_stack(self):
        payload = {
            "n": ["Andy", "mikeCase860", "Villain", "", "Hero", "Dealer"],
            # Gemini copied the external 21.6 BB bubble into the stack array.
            "s": [76, 21.6, 150.5, 0, 78.8, 111.3],
            "w": [1, 0, 1, 0, 1, 1],
            "x": ["A", "I", "A", "F", "A", "A"],
            "d": 5,
            "v": ["", "", "", "FOLD", "", ""],
            "h": ["Ac", "9c"],
            "b": [],
            "p": 27.1,
            "o": ["FOLD", "CALL", "RAISE"],
            "c": 20.6,
        }

        state = app.parse_response(
            json.dumps(payload),
            hero_turn_confirmed=True,
        )
        all_in = next(player for player in state.players if player.seat_index == 1)

        self.assertEqual("ALL_IN", all_in.status)
        self.assertEqual(0.0, all_in.stack_size)
        self.assertEqual(21.6, all_in.current_bet)

    def test_mosaic_request_adds_lossless_card_detail_and_structured_json(self):
        captures = self.captures()
        mosaic = app.build_vision_mosaic(captures)
        self.assertEqual((1080, 600), mosaic.size)
        self.assertEqual((1000, 600), app.build_vision_card_detail(captures).size)

        with mock.patch.object(app, "VISION_LAYOUT", "mosaic"):
            contents, config = app.build_vision_request(captures)

        self.assertEqual(4, len(contents))
        self.assertEqual("image/png", contents[3].inline_data.mime_type)
        self.assertEqual("application/json", config.response_mime_type)
        self.assertEqual(400, config.max_output_tokens)
        self.assertIsNotNone(config.response_json_schema)
        self.assertNotIn("pattern", config.response_json_schema["properties"]["h"]["items"])
        self.assertIn("Contributed:", contents[0])
        self.assertIn(
            "literal Pot:",
            config.response_json_schema["properties"]["p"]["description"],
        )

        pot_contents, pot_config = app.build_total_pot_reread_request(captures)
        self.assertEqual(2, len(pot_contents))
        self.assertEqual("image/png", pot_contents[1].inline_data.mime_type)
        self.assertEqual(40, pot_config.max_output_tokens)
        self.assertEqual(
            ["p"],
            pot_config.response_json_schema["required"],
        )
        self.assertEqual(219, app.SEAT_ZONES["board"]["top"])

    def test_wager_always_gets_a_board_only_ai_pot_reread(self):
        cases = [
            # Gemini copied `Contributed: 2 BB`; parsing raised it to the bet.
            ("contributed", 2, 5.67, 11.3, ["Jc", "5d", "4c"]),
            # Gemini copied the plausible previous-pot chip pile.
            ("previous pot", 3.78, 2, 5.78, ["Jh", "6s", "9h", "Ks", "3h"]),
            # The same confusion also occurs before community cards are dealt.
            ("preflop contributed", 1.5, 20.6, 27.1, []),
        ]
        for label, initial_pot, call_amount, expected_pot, board in cases:
            with self.subTest(label=label):
                payload = {
                    "n": [
                        "",
                        "mdk54",
                        "jimmyd123128",
                        "Sir-bubba-20",
                        "biba287",
                        "",
                    ],
                    "s": [0, 51, 114.8, 68, 97, 0],
                    "w": [0, 0, 0, call_amount, 0, 0],
                    "x": ["E", "F", "A", "A", "A", "E"],
                    "d": 2,
                    "v": ["", "", "", "", "", ""],
                    "h": ["4s", "3s"],
                    "b": board,
                    "p": initial_pot,
                    "o": ["FOLD", "CALL", "RAISE"],
                    "c": call_amount,
                }
                client = mock.Mock()
                client.models.generate_content.side_effect = [
                    SimpleNamespace(text=json.dumps(payload), usage_metadata=None),
                    SimpleNamespace(
                        text=json.dumps({"p": expected_pot}),
                        usage_metadata=None,
                    ),
                ]

                with mock.patch.object(app, "gemini_client", client):
                    state, _ = app.analyze_captures(
                        self.captures(),
                        hero_turn_confirmed=True,
                    )

                self.assertEqual(expected_pot, state.board_state.total_pot)
                self.assertEqual(2, client.models.generate_content.call_count)
                focused_call = (
                    client.models.generate_content.call_args_list[1].kwargs
                )
                self.assertEqual(2, len(focused_call["contents"]))
                self.assertIn("Contributed:", focused_call["contents"][0])
                self.assertFalse(
                    app.displayed_pot_conflicts_with_call(state)
                )
                self.assertFalse(
                    any(
                        "pot must exceed the call amount" in error
                        for error in app.validate_snapshot_candidate(
                            state,
                            require_hero_hand=True,
                        )
                    )
                )

    def test_gemini_request_error_is_preserved_for_the_retry_panel(self):
        captures = self.captures()
        client = mock.Mock()
        client.models.generate_content.side_effect = ValueError(
            "400 INVALID_ARGUMENT: unsupported schema keyword"
        )

        with mock.patch.object(app, "gemini_client", client):
            state, _ = app.analyze_captures(
                captures,
                hero_turn_confirmed=True,
            )

        errors = app.validate_snapshot_candidate(
            state,
            require_hero_hand=True,
        )
        self.assertEqual(1, len(errors))
        self.assertIn("400 INVALID_ARGUMENT", errors[0])
        self.assertNotIn("six distinct seats", errors[0])

    def test_strategy_preflight_skips_gemini_without_large_buttons(self):
        captures = self.captures(
            buttons=Image.new("RGB", (483, 141), (35, 38, 42))
        )
        client = mock.Mock()
        with (
            mock.patch.object(app, "capture_regions", return_value=captures),
            mock.patch.object(app, "gemini_client", client),
        ):
            state, _, _, vision_time = app.analyze_state(
                1, require_hero_turn=True
            )

        client.models.generate_content.assert_not_called()
        self.assertEqual([], state.players)
        self.assertEqual(0.0, vision_time)

    def test_parallel_suit_disagreement_is_rejected_without_mutation(self):
        state = snapshot(
            [player(0, "BTN"), player(4, "SB", hero=True)],
            street="PREFLOP",
            board=[],
            dealer=0,
        )
        hero = next(p for p in state.players if p.is_hero)
        hero.hole_cards = ["6h", "2c"]
        raw = parallel_response(
            **{
                "Seen Hero": "6d 2c",
                "Seen Board": "Preflop",
            }
        )

        error = app.validate_parallel_observation(raw, state)

        self.assertIn("disagreed on Hero's cards", error)
        self.assertEqual(["6h", "2c"], hero.hole_cards)

    def test_facing_a_bet_restores_fundamental_fold_and_call_options(self):
        payload = {
            "n": ["V0", "V1", "V2", "V3", "Hero", "V5"],
            "s": [90, 90, 90, 90, 95, 90],
            "w": [3, 1, 1, 0, 1, 3],
            "x": ["A", "A", "A", "A", "A", "A"],
            "d": 5,
            "v": ["CALL", "", "", "", "", ""],
            "h": ["6d", "2c"],
            "b": [],
            "p": 7.5,
            "o": ["CALL"],
            "c": 2,
        }

        state = app.parse_response(json.dumps(payload), hero_turn_confirmed=True)

        self.assertEqual(
            ["Fold", "Call"], state.last_action_context.hero_action_options
        )


class FastValidationTests(unittest.TestCase):
    @staticmethod
    def valid_state(*, board=None, pot=4.0):
        state = snapshot(
            [
                player(0, "BTN"),
                player(1, "OUT", status="EMPTY", stack=0),
                player(2, "OUT", status="EMPTY", stack=0),
                player(3, "OUT", status="EMPTY", stack=0),
                player(4, "SB", hero=True),
                player(5, "OUT", status="EMPTY", stack=0),
            ],
            board=["Kc", "7c", "5d"] if board is None else board,
            dealer=0,
            pot=pot,
        )
        state.action_on_seat_index = 4
        return state

    def test_parallel_prompt_contains_all_fields_once_and_in_order(self):
        prompt = app.build_parallel_strategy_prompt()
        positions = []
        for label in app.PARALLEL_STRATEGY_LABELS:
            marker = f"**{label}:**"
            self.assertEqual(1, prompt.count(marker), label)
            positions.append(prompt.index(marker))
        self.assertEqual(sorted(positions), positions)

    def test_every_parallel_field_is_required(self):
        state = self.valid_state()
        complete = parallel_response().splitlines()
        for label in app.PARALLEL_STRATEGY_LABELS:
            raw = "\n".join(
                line for line in complete
                if not line.startswith(f"**{label}:**")
            )
            with self.subTest(label=label):
                self.assertIn(
                    "missing a required field",
                    app.validate_parallel_observation(raw, state),
                )

    def test_parallel_observation_rejects_every_material_disagreement(self):
        state = self.valid_state()
        cases = {
            "Hero": ({"Seen Hero": "6h 4d"}, "Hero's cards"),
            "Board": ({"Seen Board": "Kc 7c 6d"}, "the board"),
            "Pot": ({"Seen Pot": "4.5 BB"}, "the pot"),
            "Call": ({"Seen Call": "0.5 BB"}, "call amount"),
            "Stack": ({"Seen Stack": "101 BB"}, "Hero's stack"),
            "Live seats": ({"Seen Live Seats": "S1:A"}, "live seats"),
            "Live status": ({"Seen Live Seats": "S0:I"}, "live seats"),
            "Effective": (
                {"Seen Effective Stack": "99 BB"},
                "effective stack",
            ),
            "Dealer": ({"Seen Dealer": "S1"}, "the dealer"),
            "Options": ({"Options": "Fold, Call"}, "legal actions"),
        }
        for label, (overrides, expected) in cases.items():
            with self.subTest(label=label):
                self.assertIn(
                    expected,
                    app.validate_parallel_observation(
                        parallel_response(**overrides), state
                    ),
                )

    def test_preflop_board_and_complete_observation_are_valid(self):
        state = self.valid_state(board=[])
        raw = parallel_response(**{"Seen Board": "Preflop"})
        self.assertEqual("", app.validate_parallel_observation(raw, state))
        self.assertEqual("", app.validate_parallel_candidate(raw, state))

    def test_strict_bb_amount_rejects_signs_units_and_trailing_text(self):
        accepted = {
            "0": 0.0,
            "4": 4.0,
            "4 BB": 4.0,
            "[4.5 BB]": 4.5,
        }
        for value, expected in accepted.items():
            with self.subTest(value=value):
                self.assertEqual(expected, app.strict_bb_amount(value))
        for value in (
            "-5 BB", "+5 BB", "5 chips", "1e2 BB", "NaN", "5 BB now"
        ):
            with self.subTest(value=value):
                self.assertIsNone(app.strict_bb_amount(value))

    def test_bet_and_raise_sizes_are_bounded(self):
        state = self.valid_state()
        self.assertIn(
            "invalid BB size",
            app.validate_parallel_action(
                parallel_response(Action="Bet", Size="-5 chips"), state
            ),
        )
        self.assertIn(
            "below the legal minimum",
            app.validate_parallel_action(
                parallel_response(Action="Bet", Size="0.5 BB"), state
            ),
        )
        self.assertIn(
            "outside Hero's stack",
            app.validate_parallel_action(
                parallel_response(Action="Bet", Size="101 BB"), state
            ),
        )
        self.assertEqual(
            "",
            app.validate_parallel_action(
                parallel_response(Action="Bet", Size="1 BB"), state
            ),
        )

        hero = next(player for player in state.players if player.is_hero)
        villain = next(player for player in state.players if player.seat_index == 0)
        hero.current_bet = 1
        villain.current_bet = 3
        state.last_action_context.amount_to_call = 2
        state.last_action_context.hero_action_options = ["Fold", "Call", "Raise"]
        raise_fields = {
            "Seen Call": "2 BB",
            "Options": "Fold, Call, Raise",
            "Action": "Raise",
        }
        self.assertIn(
            "below the legal minimum",
            app.validate_parallel_action(
                parallel_response(**raise_fields, Size="4 BB"), state
            ),
        )
        self.assertEqual(
            "",
            app.validate_parallel_action(
                parallel_response(**raise_fields, Size="5 BB"), state
            ),
        )

    def test_snapshot_candidate_rejects_partial_duplicate_and_negative_state(self):
        state = self.valid_state()
        self.assertEqual(
            [], app.validate_snapshot_candidate(state, require_hero_hand=True)
        )

        state.board_state.community_cards = ["Kc", "7c"]
        state.players[0].stack_size = -1
        hero = next(player for player in state.players if player.is_hero)
        hero.hole_cards = ["Kc", "4h"]
        errors = app.validate_snapshot_candidate(state, require_hero_hand=True)
        self.assertTrue(any("0, 3, 4, or 5" in error for error in errors))
        self.assertTrue(any("same card" in error for error in errors))
        self.assertTrue(any("non-negative" in error for error in errors))

    def test_displayed_pot_must_exceed_call_amount_on_every_street(self):
        state = self.valid_state(pot=4.0)
        state.last_action_context = app.LastActionContext(
            amount_to_call=4.0,
            hero_action_options=["Fold", "Call", "Raise"],
        )

        errors = app.validate_snapshot_candidate(state, require_hero_hand=True)
        self.assertTrue(
            any("pot must exceed the call amount" in error for error in errors)
        )

        state.board_state.total_pot = 5.0
        self.assertFalse(
            any(
                "pot must exceed the call amount" in error
                for error in app.validate_snapshot_candidate(
                    state,
                    require_hero_hand=True,
                )
            )
        )

        state.board_state.community_cards = []
        state.board_state.total_pot = 4.0
        self.assertTrue(
            any(
                "pot must exceed the call amount" in error
                for error in app.validate_snapshot_candidate(
                    state,
                    require_hero_hand=True,
                )
            )
        )

    def test_fast_claude_call_has_timeout_no_retry_and_no_temperature(self):
        client = mock.Mock()
        configured = client.with_options.return_value
        configured.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=parallel_response())]
        )
        captures = VisionWireFormatTests.captures()

        with mock.patch.object(app, "anthropic_client", client):
            analysis, _, _ = app.request_parallel_strategy(captures, "")

        client.with_options.assert_called_once_with(
            timeout=app.CLAUDE_FAST_TIMEOUT_SECONDS,
            max_retries=0,
        )
        kwargs = configured.messages.create.call_args.kwargs
        self.assertNotIn("temperature", kwargs)
        self.assertIn("**Action:** Check", analysis)

    def test_gemini_deadline_respects_provider_minimum(self):
        self.assertGreaterEqual(app.GEMINI_TIMEOUT_MS, 10000)
        self.assertGreater(
            app.FAST_REQUEST_TIMEOUT_SECONDS,
            app.GEMINI_TIMEOUT_MS / 1000,
        )

    def test_coach_amount_uses_the_same_stack_and_minimum_checks(self):
        state = self.valid_state()
        valid = "**Action:** Check\n**Amount:** 0\n"
        malformed = "**Action:** Bet\n**Amount:** -5 chips\n"
        too_large = "**Action:** Bet\n**Amount:** 101 BB\n"

        self.assertEqual(valid, app.validate_strategy_amount(valid, state))
        self.assertIn("invalid BB size", app.validate_strategy_amount(malformed, state))
        self.assertIn(
            "outside Hero's stack",
            app.validate_strategy_amount(too_large, state),
        )

    def test_coach_action_must_match_visible_legal_controls(self):
        state = self.valid_state()
        state.last_action_context.hero_action_options = ["Check", "All-in"]
        illegal_bet = "**Action:** Bet\n**Amount:** 4 BB\n"
        self.assertIn(
            "not independently confirmed as legal",
            app.validate_strategy_amount(illegal_bet, state),
        )

        state.last_action_context = app.LastActionContext(
            amount_to_call=2,
            hero_action_options=["Fold", "Call", "All-in"],
        )
        illegal_raise = "**Action:** Raise\n**Amount:** 4 BB\n"
        self.assertIn(
            "not independently confirmed as legal",
            app.validate_strategy_amount(illegal_raise, state),
        )
        explicit_all_in = "**Action:** Raise\n**Amount:** 100 BB\n"
        self.assertEqual(
            explicit_all_in,
            app.validate_strategy_amount(explicit_all_in, state),
        )


class ParsingAndHistoryTests(unittest.TestCase):
    def test_amount_to_call_is_incremental(self):
        response = {
            "seats": [
                {
                    "seat_index": 0,
                    "name": "Villain",
                    "stack_size_bb": 94,
                    "current_bet_bb": 6,
                    "has_cards": True,
                    "is_dealer": True,
                },
                {
                    "seat_index": 4,
                    "name": "Hero",
                    "stack_size_bb": 98,
                    "current_bet_bb": 2,
                    "has_cards": True,
                    "hole_cards": ["As", "Kd"],
                },
            ],
            "board_cards": [],
            "total_pot_bb": 10,
            "hero_context": {"is_turn": True, "action_options": ["Fold", "Call", "Raise"]},
        }
        state = app.parse_response(json.dumps(response))
        self.assertEqual(4.0, state.last_action_context.amount_to_call)
        self.assertEqual(["Fold", "Call", "Raise"], state.last_action_context.hero_action_options)

    def test_preflop_aggressor_updates_only_on_observed_raise(self):
        baseline = snapshot(
            [player(2, "BB", bet=1), player(4, "MP", hero=True), player(5, "CO")],
            street="PREFLOP",
            board=[],
        )
        opened = snapshot(
            [player(2, "BB", bet=1), player(4, "MP", hero=True), player(5, "CO", bet=3)],
            street="PREFLOP",
            board=[],
        )
        history = app.HandHistory(snapshots=[baseline, opened])
        label, hero_was_pfa = app.infer_preflop_aggressor(history, opened)
        self.assertEqual("CO", label)
        self.assertFalse(hero_was_pfa)

    def test_action_history_matches_players_by_seat(self):
        first = snapshot(
            [player(4, "SB", hero=True), player(5, "CO")],
            street="FLOP",
            board=["Kc", "7c", "5d"],
        )
        second = snapshot(
            [player(5, "CO", bet=2.5), player(4, "SB", hero=True)],
            street="FLOP",
            board=["Kc", "7c", "5d"],
        )
        history = app.HandHistory(snapshots=[first, second], hand_id="test")
        summary = app.parse_action_history(history.to_json())
        self.assertIn("CO", summary)
        self.assertIn("bets 2.5 BB", summary)
        self.assertNotIn("checks", summary.lower())

    def test_visible_bet_and_numeric_delta_are_not_duplicated(self):
        first = snapshot(
            [player(5, "CO"), player(4, "SB", hero=True)],
            street="FLOP",
            board=["Kc", "7c", "5d"],
        )
        second = snapshot(
            [
                player(5, "CO", bet=2.5, action="BET"),
                player(4, "SB", hero=True),
            ],
            street="FLOP",
            board=["Kc", "7c", "5d"],
        )
        summary = app.parse_action_history(
            app.HandHistory(snapshots=[first, second]).to_json()
        )
        self.assertEqual(1, summary.lower().count("bets"))

    def test_repeated_raise_overlay_does_not_hide_new_bet_delta(self):
        first = snapshot(
            [
                player(5, "CO", bet=2.0, action="RAISE"),
                player(4, "SB", hero=True),
            ],
            street="PREFLOP",
            board=[],
        )
        second = snapshot(
            [
                player(5, "CO", bet=5.0, action="RAISE"),
                player(4, "SB", hero=True),
            ],
            street="PREFLOP",
            board=[],
        )
        summary = app.parse_action_history(
            app.HandHistory(snapshots=[first, second]).to_json()
        )
        self.assertIn("raises to 5 BB", summary)

    def test_pot_total_is_not_double_counted(self):
        state = snapshot(
            [
                player(0, "SB", bet=0.5),
                player(1, "BB", bet=1.0),
                player(3, "MP", bet=1.0),
                player(4, "CO", hero=True, bet=1.0),
            ],
            street="PREFLOP",
            board=[],
            pot=3.5,
        )
        history = app.HandHistory(snapshots=[state])
        self.assertEqual(3.5, app.strategy_pot_for_street(state, history))

    def test_one_old_pot_ocr_spike_is_not_carried_forward(self):
        old = snapshot(
            [player(4, "CO", hero=True)],
            street="FLOP",
            board=["6h", "Jc", "8h"],
            pot=45.0,
        )
        current = snapshot(
            [player(4, "CO", hero=True)],
            street="FLOP",
            board=["6h", "Jc", "8h"],
            pot=4.5,
        )
        history = app.HandHistory(snapshots=[old, current])
        self.assertEqual(4.5, app.strategy_pot_for_street(current, history))

    def test_corroborated_pot_preserves_exact_decimal(self):
        first = snapshot(
            [player(4, "CO", hero=True)],
            board=["6h", "Jc", "8h"],
            pot=3.78,
        )
        second = snapshot(
            [player(4, "CO", hero=True)],
            board=["6h", "Jc", "8h"],
            pot=3.78,
        )
        current = snapshot(
            [player(4, "CO", hero=True)],
            board=["6h", "Jc", "8h"],
            pot=3.5,
        )
        history = app.HandHistory(snapshots=[first, second, current])
        self.assertEqual(3.78, app.strategy_pot_for_street(current, history))

    def test_fold_and_position_are_stable_within_hand(self):
        prior = snapshot(
            [
                player(1, "BB", status="ACTIVE"),
                player(2, "UTG", status="FOLDED"),
                player(3, "MP", status="ACTIVE"),
                player(4, "CO", hero=True),
            ],
            board=["6h", "Jc", "8h"],
            dealer=5,
        )
        prior.players[2].username = "SuperDog782"
        current = snapshot(
            [
                player(1, "OUT", status="EMPTY", stack=0),
                player(2, "BB", status="ACTIVE"),
                player(3, "UTG", status="ACTIVE"),
                player(4, "CO", hero=True),
            ],
            street="TURN",
            board=["6h", "Jc", "8h", "3s"],
            dealer=5,
        )
        current.players[2].username = "WrongNeighbor"
        history = app.HandHistory(snapshots=[prior])

        app.reconcile_snapshot_with_history(current, history)

        by_seat = {p.seat_index: p for p in current.players}
        self.assertEqual("FOLDED", by_seat[1].status)
        self.assertEqual("FOLDED", by_seat[2].status)
        self.assertEqual("UTG", by_seat[2].name)
        self.assertEqual("MP", by_seat[3].name)
        self.assertEqual("SuperDog782", by_seat[3].username)

    def test_empty_preflop_seats_cannot_join_or_gain_positions_postflop(self):
        prior = snapshot(
            [
                player(0, "OUT", status="EMPTY", stack=0),
                player(1, "OUT", status="EMPTY", stack=0),
                player(2, "OUT", status="EMPTY", stack=0),
                player(3, "BTN", status="ACTIVE"),
                player(4, "SB", hero=True),
                player(5, "BB", status="ACTIVE"),
            ],
            street="PREFLOP",
            board=[],
            dealer=3,
            pot=7.0,
        )
        current = snapshot(
            [
                player(0, "BTN", status="FOLDED"),
                player(1, "SB", status="FOLDED"),
                player(2, "BB", status="FOLDED"),
                player(3, "SB", status="FOLDED"),
                player(4, "BB", hero=True),
                player(5, "SB", status="ACTIVE"),
            ],
            street="FLOP",
            board=["Qd", "2h", "Jc"],
            dealer=3,
            pot=11.3,
        )

        app.reconcile_snapshot_with_history(
            current,
            app.HandHistory(snapshots=[prior], hand_id="sparse-table"),
        )

        by_seat = {p.seat_index: p for p in current.players}
        for seat in (0, 1, 2):
            self.assertEqual("OUT", by_seat[seat].name)
            self.assertEqual("SITTING_OUT", by_seat[seat].status)
        self.assertEqual("BTN", by_seat[3].name)
        self.assertEqual("SB", by_seat[4].name)
        self.assertEqual("BB", by_seat[5].name)
        dealt_positions = [
            by_seat[seat].name for seat in (3, 4, 5)
        ]
        self.assertEqual(len(dealt_positions), len(set(dealt_positions)))

    def test_positive_local_cards_override_an_old_false_fold(self):
        prior = snapshot(
            [player(1, "BB", status="FOLDED"), player(4, "CO", hero=True)],
            street="FLOP",
            board=["6h", "Jc", "8h"],
            dealer=5,
        )
        current = snapshot(
            [player(1, "BB", status="ACTIVE"), player(4, "CO", hero=True)],
            street="TURN",
            board=["6h", "Jc", "8h", "3s"],
            dealer=5,
        )
        current.players[0].cards_confirmed_locally = True
        app.reconcile_snapshot_with_history(
            current, app.HandHistory(snapshots=[prior])
        )
        self.assertEqual("ACTIVE", current.players[0].status)

    def test_positive_local_cards_do_not_cancel_all_in(self):
        prior = snapshot(
            [player(1, "BB", status="ALL_IN"), player(4, "CO", hero=True)],
            board=["6h", "Jc", "8h"],
            dealer=5,
        )
        current = snapshot(
            [player(1, "BB", status="ACTIVE"), player(4, "CO", hero=True)],
            street="TURN",
            board=["6h", "Jc", "8h", "3s"],
            dealer=5,
        )
        current.players[0].cards_confirmed_locally = True
        app.reconcile_snapshot_with_history(
            current, app.HandHistory(snapshots=[prior])
        )
        self.assertEqual("ALL_IN", current.players[0].status)

    def test_impossible_zero_stack_ocr_is_restored_from_previous_street(self):
        prior = snapshot(
            [player(2, "SB", status="ACTIVE", stack=238), player(4, "UTG", hero=True)],
            street="TURN",
            board=["8d", "6c", "5c", "5d"],
            dealer=1,
            pot=12.0,
        )
        current = snapshot(
            [player(2, "SB", status="ACTIVE", stack=0), player(4, "UTG", hero=True)],
            street="RIVER",
            board=["8d", "6c", "5c", "5d", "4h"],
            dealer=1,
            pot=24.4,
        )
        current.players[0].cards_confirmed_locally = True

        app.reconcile_snapshot_with_history(
            current, app.HandHistory(snapshots=[prior])
        )
        self.assertEqual(238, current.players[0].stack_size)
        self.assertEqual("ACTIVE", current.players[0].status)

    def test_preaction_controls_do_not_create_hero_turn(self):
        response = {
            "seats": [
                {
                    "seat_index": 4,
                    "name": "Hero",
                    "stack_size_bb": 99,
                    "has_cards": True,
                    "hole_cards": ["9c", "5h"],
                }
            ],
            "board_cards": [],
            "total_pot_bb": 3.5,
            "hero_context": {
                "is_turn": True,
                "action_options": ["Check", "Check/Fold", "Call Any"],
            },
        }
        state = app.parse_response(
            json.dumps(response), hero_turn_confirmed=False
        )
        self.assertEqual(-1, state.action_on_seat_index)
        self.assertEqual([], state.last_action_context.hero_action_options)

    def test_visible_action_is_not_used_as_username(self):
        response = {
            "seats": [
                {
                    "seat_index": 3,
                    "name": "Check",
                    "stack_size_bb": 99,
                    "has_cards": True,
                }
            ],
            "hero_context": {"is_turn": False, "action_options": []},
        }
        state = app.parse_response(json.dumps(response))
        villain = state.players[0]
        self.assertEqual("CHECK", villain.visible_action)
        self.assertTrue(villain.username.startswith("Unknown_"))

    def test_new_hand_detected_after_hero_was_out(self):
        previous = snapshot(
            [player(4, "CO", hero=True)], street="PREFLOP", board=[], dealer=0
        )
        previous.players[0].hole_cards = None
        current = snapshot(
            [player(4, "MP", hero=True)], street="PREFLOP", board=[], dealer=1
        )
        history = app.HandHistory(snapshots=[previous])
        self.assertTrue(history.is_new_hand(current))

    def test_same_hole_cards_in_reverse_order_are_same_hand(self):
        previous = snapshot([player(4, "CO", hero=True)], dealer=5)
        previous.players[0].hole_cards = ["9c", "5h"]
        current = snapshot([player(4, "CO", hero=True)], dealer=5)
        current.players[0].hole_cards = ["5h", "9c"]
        history = app.HandHistory(snapshots=[previous])
        self.assertFalse(history.is_new_hand(current))

    def test_dealer_move_starts_new_preflop_hand_even_with_same_cards(self):
        previous = snapshot(
            [player(4, "CO", hero=True)],
            street="PREFLOP",
            board=[],
            dealer=5,
        )
        current = snapshot(
            [player(4, "MP", hero=True)],
            street="PREFLOP",
            board=[],
            dealer=0,
        )
        current.players[0].hole_cards = list(previous.players[0].hole_cards)
        history = app.HandHistory(snapshots=[previous])
        self.assertTrue(history.is_new_hand(current))

    def test_partial_hole_card_ocr_does_not_start_new_hand(self):
        previous = snapshot([player(4, "CO", hero=True)], dealer=5)
        previous.players[0].hole_cards = ["9c", "5h"]
        current = snapshot([player(4, "CO", hero=True)], dealer=5)
        current.players[0].hole_cards = ["9c"]
        history = app.HandHistory(snapshots=[previous])
        self.assertFalse(history.is_new_hand(current))

    def test_hole_card_suit_glitch_does_not_reset_continuing_board(self):
        previous = snapshot(
            [player(4, "UTG", hero=True)],
            street="FLOP",
            board=["6s", "9c", "3s"],
            dealer=1,
        )
        previous.players[0].hole_cards = ["9d", "3h"]
        current = snapshot(
            [player(4, "UTG", hero=True)],
            street="TURN",
            board=["6s", "9c", "3s", "4d"],
            dealer=1,
        )
        current.players[0].hole_cards = ["9d", "3d"]
        history = app.HandHistory(snapshots=[previous])
        self.assertFalse(history.is_new_hand(current))

    def test_hole_card_glitch_does_not_reset_preflop_to_flop_transition(self):
        previous = snapshot(
            [player(4, "SB", hero=True)],
            street="PREFLOP",
            board=[],
            dealer=3,
        )
        previous.players[0].hole_cards = ["As", "4d"]
        current = snapshot(
            [player(4, "SB", hero=True)],
            street="FLOP",
            board=["8d", "6c", "5c"],
            dealer=3,
        )
        current.players[0].hole_cards = ["As", "4h"]
        history = app.HandHistory(snapshots=[previous])

        self.assertFalse(history.is_new_hand(current))
        self.assertTrue(
            app.preserve_hero_cards_on_continuing_board(current, history)
        )
        self.assertEqual(["As", "4d"], current.players[0].hole_cards)

    def test_continuing_board_restores_validated_hero_cards_before_duplicate_check(self):
        previous = snapshot(
            [player(4, "BTN", hero=True)],
            street="TURN",
            board=["9h", "Ts", "3h", "Td"],
            dealer=4,
        )
        previous.players[0].hole_cards = ["3s", "7c"]
        current = snapshot(
            [player(4, "BTN", hero=True)],
            street="RIVER",
            board=["9h", "Ts", "3h", "Td", "7s"],
            dealer=4,
        )
        current.players[0].hole_cards = ["3s", "7s"]
        history = app.HandHistory(snapshots=[previous])

        self.assertTrue(app.preserve_hero_cards_on_continuing_board(current, history))
        self.assertEqual(["3s", "7c"], current.players[0].hole_cards)
        errors = app.validate_snapshot_candidate(current, require_hero_hand=True)
        self.assertNotIn("the same card appears more than once", errors)


class LocalCardBackDetectionTests(unittest.TestCase):
    @staticmethod
    def card_back_crop():
        crop = Image.new("RGB", (240, 140), (40, 45, 50))
        draw = ImageDraw.Draw(crop)
        draw.rounded_rectangle((45, 7, 103, 34), radius=5, fill=(153, 54, 63))
        draw.rounded_rectangle((106, 7, 164, 34), radius=5, fill=(153, 54, 63))
        return crop

    def test_detects_paired_red_card_backs_but_not_single_red_ui(self):
        self.assertTrue(app.detect_opponent_card_backs(self.card_back_crop()))

        negative = Image.new("RGB", (240, 140), (40, 45, 50))
        ImageDraw.Draw(negative).rectangle((45, 7, 164, 12), fill=(180, 40, 50))
        self.assertFalse(app.detect_opponent_card_backs(negative))

    def test_local_evidence_corrects_only_matching_opponent(self):
        folded = player(1, "CO", status="FOLDED")
        other = player(0, "BTN", status="FOLDED")
        state = snapshot([other, folded, player(4, "SB", hero=True)])
        captures = {
            "seat1": Image.new("RGB", (240, 140), (40, 45, 50)),
            "seat2": self.card_back_crop(),
        }

        corrected = app.apply_local_card_back_evidence(state, captures)

        self.assertEqual([1], corrected)
        self.assertEqual("ACTIVE", folded.status)
        self.assertEqual("FOLDED", other.status)

    def test_parse_applies_local_evidence_before_empty_and_positions(self):
        response = {
            "seats": [
                {
                    "seat_index": 1,
                    "name": "Kran77",
                    "stack_size_bb": 104,
                    "has_cards": False,
                    "is_folded": True,
                    "is_empty": True,
                    "is_dealer": True,
                },
                {
                    "seat_index": 4,
                    "name": "Hero",
                    "stack_size_bb": 99,
                    "has_cards": True,
                    "hole_cards": ["3c", "8h"],
                },
            ],
            "board_cards": ["5h", "8s", "7s", "3d", "8c"],
            "total_pot_bb": 2.36,
            "hero_context": {
                "is_turn": True,
                "action_options": ["Check", "Bet"],
            },
        }

        state = app.parse_response(json.dumps(response), locally_dealt_seats={1})
        villain = next(p for p in state.players if p.seat_index == 1)
        hero = next(p for p in state.players if p.is_hero)
        details = app.HandEvaluator.evaluate_details(
            hero.hole_cards,
            state.board_state.community_cards,
        )
        context = app.build_strategy_context(
            state,
            app.HandHistory(snapshots=[state]),
            details,
        )

        self.assertEqual("ACTIVE", villain.status)
        self.assertNotEqual("OUT", villain.name)
        self.assertEqual(1, context.opponents_in_hand)
        self.assertEqual("HEADS-UP", context.pot_type)

    def test_local_card_absence_overrides_false_active_flags(self):
        response = {
            "seats": [
                {
                    "seat_index": index,
                    "name": f"Villain{index}",
                    "stack_size_bb": 100,
                    "has_cards": True,
                }
                for index in range(6)
            ],
            "hero_context": {"is_turn": True, "action_options": ["Check", "Bet"]},
        }
        response["seats"][4]["hole_cards"] = ["9s", "9c"]

        state = app.parse_response(
            json.dumps(response),
            locally_dealt_seats={0, 2},
            hero_turn_confirmed=True,
        )
        by_seat = {player.seat_index: player for player in state.players}

        self.assertEqual("ACTIVE", by_seat[0].status)
        self.assertEqual("FOLDED", by_seat[1].status)
        self.assertEqual("ACTIVE", by_seat[2].status)
        self.assertEqual("FOLDED", by_seat[3].status)
        self.assertEqual("ACTIVE", by_seat[4].status)
        self.assertEqual("FOLDED", by_seat[5].status)
        self.assertEqual(app.HERO_USERNAME, by_seat[4].username)

    def test_hero_username_cannot_be_assigned_to_an_opponent(self):
        response = {
            "seats": [
                {
                    "seat_index": 3,
                    "name": app.HERO_USERNAME,
                    "stack_size_bb": 100,
                    "has_cards": True,
                },
                {
                    "seat_index": 4,
                    "name": "WrongHero",
                    "stack_size_bb": 60,
                    "has_cards": True,
                    "hole_cards": ["As", "4d"],
                },
            ],
        }
        state = app.parse_response(json.dumps(response), locally_dealt_seats={3})
        by_seat = {player.seat_index: player for player in state.players}
        self.assertEqual("Unknown_S3", by_seat[3].username)
        self.assertEqual(app.HERO_USERNAME, by_seat[4].username)

    def test_locally_dealt_zero_stack_opponent_is_all_in(self):
        response = {
            "seats": [
                {
                    "seat_index": 1,
                    "name": "Villain",
                    "stack_size_bb": 0,
                    "current_bet_bb": 12,
                    "has_cards": True,
                },
                {
                    "seat_index": 4,
                    "name": "WrongHeroName",
                    "stack_size_bb": 90,
                    "has_cards": True,
                    "hole_cards": ["Ah", "Kd"],
                },
            ],
        }
        state = app.parse_response(json.dumps(response), locally_dealt_seats={1})
        villain = next(player for player in state.players if player.seat_index == 1)
        self.assertEqual("ALL_IN", villain.status)

    def test_pot_is_repaired_to_visible_bet_lower_bound(self):
        response = {
            "seats": [
                {"seat_index": 0, "name": "A", "stack": 80, "bet": 3},
                {"seat_index": 2, "name": "B", "stack": 80, "bet": 15},
                {
                    "seat_index": 4,
                    "name": "Hero",
                    "stack": 60,
                    "bet": 3,
                    "hole_cards": ["9s", "9c"],
                },
                {"seat_index": 5, "name": "C", "stack": 80, "bet": 1},
            ],
            "total_pot_bb": 3,
        }
        state = app.parse_response(json.dumps(response))
        self.assertEqual(22, state.board_state.total_pot)


class FreshnessAndButtonTests(unittest.TestCase):
    @staticmethod
    def action_buttons(labels=("Check", "Bet")):
        image = Image.new("RGB", (483, 141), (35, 38, 42))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((162, 74, 311, 133), radius=8, fill=(220, 220, 220))
        draw.rounded_rectangle((323, 74, 472, 133), radius=8, fill=(220, 220, 220))
        draw.text((205, 96), labels[0], fill=(30, 30, 30))
        draw.text((375, 96), labels[1], fill=(30, 30, 30))
        return image

    def base_captures(self):
        return {
            "board": Image.new("RGB", (346, 155), (55, 59, 63)),
            "hero": Image.new("RGB", (263, 167), (45, 49, 53)),
            "buttons": self.action_buttons(),
        }

    def test_large_buttons_are_actionable_but_small_checkboxes_are_not(self):
        self.assertTrue(app.detect_hero_action_buttons(self.action_buttons()))
        checkboxes = Image.new("RGB", (483, 141), (35, 38, 42))
        draw = ImageDraw.Draw(checkboxes)
        for y in (10, 50, 90):
            draw.rectangle((10, y, 220, y + 22), fill=(75, 78, 82))
        self.assertFalse(app.detect_hero_action_buttons(checkboxes))

    def test_real_button_geometry_near_upper_boundary_is_actionable(self):
        image = Image.new("RGB", (483, 141), (35, 38, 42))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((162, 39, 311, 100), radius=8, fill=(210, 210, 210))
        draw.rounded_rectangle((323, 39, 472, 100), radius=8, fill=(210, 210, 210))
        self.assertTrue(app.detect_hero_action_buttons(image))

    def test_identical_state_and_timer_only_change_are_fresh(self):
        before = self.base_captures()
        after = {name: image.copy() for name, image in before.items()}
        ImageDraw.Draw(after["hero"]).rectangle(
            (0, 150, 262, 166), fill=(220, 190, 40)
        )
        self.assertEqual([], app.table_state_change_reasons(before, after))

    def test_board_change_and_disappearing_buttons_are_stale(self):
        before = self.base_captures()
        after = {name: image.copy() for name, image in before.items()}
        ImageDraw.Draw(after["board"]).rectangle(
            (20, 25, 190, 125), fill=(235, 235, 235)
        )
        after["buttons"] = Image.new("RGB", (483, 141), (35, 38, 42))
        reasons = app.table_state_change_reasons(before, after)
        self.assertIn("board or pot changed", reasons)
        self.assertIn("Hero action buttons disappeared", reasons)

    def test_hero_card_change_is_stale(self):
        before = self.base_captures()
        after = {name: image.copy() for name, image in before.items()}
        ImageDraw.Draw(after["hero"]).rectangle(
            (70, 25, 205, 95), fill=(235, 235, 235)
        )
        self.assertIn(
            "Hero cards changed",
            app.table_state_change_reasons(before, after),
        )

    def test_stack_or_timer_animation_below_cards_is_fresh(self):
        before = self.base_captures()
        after = {name: image.copy() for name, image in before.items()}
        ImageDraw.Draw(after["hero"]).rectangle(
            (35, 112, 230, 138), fill=(230, 190, 35)
        )
        self.assertNotIn(
            "Hero cards changed",
            app.table_state_change_reasons(before, after),
        )

    def test_dimmed_action_buttons_keep_the_same_decision_alive(self):
        before = self.base_captures()
        after = {name: image.copy() for name, image in before.items()}
        after["buttons"] = before["buttons"].point(lambda value: int(value * 0.4))

        self.assertTrue(app.detect_hero_action_buttons(before["buttons"]))
        self.assertFalse(app.detect_hero_action_buttons(after["buttons"]))
        reasons = app.table_state_change_reasons(before, after)
        self.assertNotIn("Hero action buttons disappeared", reasons)
        self.assertNotIn("Hero action controls changed", reasons)

    def test_same_button_geometry_with_redrawn_labels_is_stale(self):
        before = self.base_captures()
        after = {name: image.copy() for name, image in before.items()}
        after["buttons"] = self.action_buttons(("Fold", "Call"))
        self.assertIn(
            "Hero action controls changed",
            app.table_state_change_reasons(before, after),
        )

    def test_amount_only_button_redraw_is_stale(self):
        before = self.base_captures()
        before["buttons"] = self.action_buttons(("Check", "Call 1"))
        after = {name: image.copy() for name, image in before.items()}
        after["buttons"] = self.action_buttons(("Check", "Call 2"))
        self.assertIn(
            "Hero action controls changed",
            app.table_state_change_reasons(before, after),
        )


class PresentationUXTests(unittest.TestCase):
    @staticmethod
    def multiway_river():
        return snapshot(
            [
                player(0, "CO", status="FOLDED"),
                player(1, "BTN"),
                player(2, "SB"),
                player(3, "BB"),
                player(4, "UTG", hero=True),
                player(5, "MP", status="FOLDED"),
            ],
            street="RIVER",
            board=["5h", "Jd", "8s", "7d", "6s"],
            pot=27.1,
        )

    def test_multiway_solver_limit_is_a_clear_wait_state(self):
        recommendation, details, style = app.strategy_display_content(
            self.multiway_river(),
            (
                "Strategy Error: GTO_HU unsupported: the live state does not "
                "identify exactly one HU villain"
            ),
        )

        self.assertEqual("WAIT FOR HEADS-UP (3 LEFT)", recommendation)
        self.assertIn("exactly 1 opponent", details)
        self.assertIn("Press J again when only 1 opponent remains", details)
        self.assertNotIn("live state", details)
        self.assertEqual("bold black on cyan", style)

    def test_preflop_solver_limit_explains_current_coverage(self):
        state = self.multiway_river()
        state.meta_info.current_street = "PREFLOP"
        state.board_state.community_cards = []

        recommendation, details, style = app.strategy_display_content(
            state,
            (
                "Strategy Error: GTO_HU unsupported: the configured range "
                "source has no preflop mixed policy"
            ),
        )

        self.assertEqual("POSTFLOP GTO ONLY", recommendation)
        self.assertIn("Use P for the preflop chart", details)
        self.assertEqual("bold black on cyan", style)

    def test_successful_solver_action_keeps_action_first(self):
        recommendation, details, style = app.strategy_display_content(
            self.multiway_river(),
            (
                "**Action:** Bet\n"
                "**Size:** 13.5 BB\n"
                "**Why:** Solver-selected action.\n"
                "* **GTO mix:** Bet 70% | Check 30%"
            ),
        )

        self.assertEqual("Bet 13.5 BB", recommendation)
        self.assertIn("GTO mix", details)
        self.assertEqual("bold white on green", style)

    def test_failure_reason_is_hidden_from_timing_header(self):
        self.assertEqual(
            "",
            app.visible_strategy_source(
                "GTO_HU unsupported: the live state does not identify exactly "
                "one HU villain"
            ),
        )
        self.assertEqual(
            "APPROXIMATE_SOLVER fresh",
            app.visible_strategy_source("APPROXIMATE_SOLVER fresh"),
        )

    def test_stale_button_reason_is_user_facing(self):
        self.assertEqual(
            "available action or amount changed",
            app.friendly_stale_reasons(["Hero action controls changed"]),
        )

    def test_incomplete_multiway_history_is_explained_without_solver_jargon(self):
        recommendation, details, style = app.strategy_display_content(
            self.multiway_river(),
            (
                "Strategy Error: GTO unavailable: complete multiway "
                "public_hand is unavailable: continuous history has not "
                "reached this decision"
            ),
        )

        self.assertEqual("FULL HISTORY NOT READY", recommendation)
        self.assertIn("every action", details)
        self.assertIn("no action was calculated", details)
        self.assertNotIn("public_hand", details)
        self.assertEqual("bold black on yellow", style)


class StrategyBackendIntegrationTests(unittest.TestCase):
    @staticmethod
    def heads_up_state(*, pot=4.0, hero_stack=100.0, villain_stack=80.0):
        state = snapshot(
            [
                player(0, "BTN", stack=villain_stack),
                player(4, "BB", hero=True, stack=hero_stack),
            ],
            board=["Kc", "7c", "5d"],
            dealer=0,
            pot=pot,
        )
        state.action_on_seat_index = 4
        state.hand_id = "backend-integration"
        return state

    def test_default_backend_is_strict_gto(self):
        self.assertEqual("GTO", app.DEFAULT_STRATEGY_BACKEND)
        self.assertIn("GTO_HU", app.VALID_STRATEGY_BACKENDS)
        self.assertIn("GTO_MULTIWAY", app.VALID_STRATEGY_BACKENDS)

    def test_multiway_v3_builder_uses_only_the_replayable_transcript(self):
        from test_gto_multiway_protocol import four_way_flop_history

        public_hand = four_way_flop_history()
        replayed = public_hand.replay()
        players = []
        seat_by_id = {seat.seat: seat for seat in public_hand.seats}
        for seat in sorted(seat_by_id):
            players.append(
                app.Player(
                    seat_index=seat,
                    name=seat_by_id[seat].position,
                    username="Hero" if seat == 4 else f"Villain{seat}",
                    stack_size=float(replayed.stack_map[seat]),
                    current_bet=float(
                        replayed.street_contribution_map[seat]
                    ),
                    status=(
                        "FOLDED"
                        if seat in replayed.folded
                        else "ALL_IN"
                        if seat in replayed.all_in
                        else "ACTIVE"
                    ),
                    is_hero=seat == 4,
                    hole_cards=["Qc", "Qd"] if seat == 4 else None,
                    is_dealer=seat == public_hand.button_seat,
                )
            )
        state = app.GameSnapshot(
            hand_id=public_hand.hand_id,
            timestamp="2026-07-30T12:00:00Z",
            meta_info=app.MetaInfo(current_street=replayed.street),
            board_state=app.BoardState(
                community_cards=list(replayed.board),
                total_pot=float(replayed.pot_bb),
            ),
            dealer_seat_index=public_hand.button_seat,
            action_on_seat_index=4,
            players=players,
            last_action_context=app.LastActionContext(
                amount_to_call=float(replayed.amount_to_call_bb),
                hero_action_options=["Check", "Bet"],
            ),
        )
        history = app.HandHistory(
            hand_id=public_hand.hand_id,
            snapshots=[state],
        )
        history.public_event_recorder.history = public_hand
        prepared = app.prepare_strategy_state(state, history)

        routed = app.build_multiway_solver_state(
            state,
            history,
            prepared,
        )

        self.assertEqual(4, routed.hero_seat)
        self.assertEqual(("Qc", "Qd"), routed.hero_combo)
        self.assertEqual(frozenset({2, 3, 4, 5}), routed.replayed.live_seats)
        self.assertEqual(Decimal("4"), routed.replayed.pot_bb)
        self.assertFalse(hasattr(routed, "villain_position"))

    def test_solved_gto_bypasses_claude_and_preserves_source_and_metrics(self):
        state = self.heads_up_state()
        history = app.HandHistory(snapshots=[state], hand_id=state.hand_id)
        analysis = (
            "**Action:** Bet\n"
            "**Size:** 2 BB\n"
            "**Why:** Stable local solver mix.\n"
            "* **GTO mix:** Bet 2 BB 75.0% | Check 25.0%"
        )
        outcome = SimpleNamespace(
            solved=True,
            status=app.LiveGTOStatus.SOLVED,
            reason="",
            analysis=analysis,
            source="GTO cache",
            spec=SimpleNamespace(cache_key="a" * 64),
        )
        router = mock.Mock()
        router.evaluate.return_value = outcome
        failing_claude = mock.Mock()
        failing_claude.with_options.side_effect = AssertionError(
            "Claude must not be touched for a solved GTO node"
        )
        failing_claude.messages.create.side_effect = AssertionError(
            "Claude must not be touched for a solved GTO node"
        )

        result = app.evaluate_strategy_backend(
            state,
            history,
            mode="COACH",
            backend="GTO",
            router=router,
            client=failing_claude,
        )

        router.evaluate.assert_called_once()
        routed_state = router.evaluate.call_args.args[0]
        self.assertTrue(routed_state.hero_is_oop)
        self.assertEqual(1, routed_state.active_villains)
        self.assertTrue(routed_state.street_root_confirmed)
        failing_claude.with_options.assert_not_called()
        failing_claude.messages.create.assert_not_called()
        self.assertEqual("GTO", result.mode)
        self.assertEqual("b-inary/postflop-solver", result.model)
        self.assertEqual("GTO cache", result.source)
        self.assertEqual(analysis, result.final_analysis)
        self.assertEqual(4.0, result.metrics["final_pot"])
        self.assertEqual(80.0, result.metrics["eff_stack"])
        self.assertEqual(1, result.metrics["opponents_in_hand"])
        self.assertEqual("OOP", result.metrics["relative_position"])
        prompt = json.loads(result.prompt)
        self.assertEqual("GTO", prompt["backend"])
        self.assertEqual("user_controlled_simulator_live", prompt["usage"])
        self.assertEqual("a" * 64, prompt["spec_key"])

    def test_approximate_solver_source_is_not_relabelled_as_gto(self):
        state = self.heads_up_state()
        history = app.HandHistory(snapshots=[state], hand_id=state.hand_id)
        outcome = SimpleNamespace(
            solved=True,
            status=app.LiveGTOStatus.SOLVED,
            reason="",
            analysis=(
                "**Action:** Check\n"
                "**Size:** 0\n"
                "**Why:** Approximate local solver mix."
            ),
            source="APPROXIMATE_SOLVER fresh",
            model="b-inary/postflop-solver",
            spec=SimpleNamespace(cache_key="b" * 64),
        )
        router = mock.Mock()
        router.evaluate.return_value = outcome
        claude = mock.Mock()

        result = app.evaluate_strategy_backend(
            state,
            history,
            mode="COACH",
            backend="HYBRID",
            router=router,
            client=claude,
        )

        self.assertEqual("APPROXIMATE_SOLVER", result.mode)
        self.assertEqual("APPROXIMATE_SOLVER fresh", result.source)
        self.assertEqual(
            "APPROXIMATE_SOLVER",
            json.loads(result.prompt)["backend"],
        )
        claude.messages.create.assert_not_called()

    def test_gto_hu_accepts_labelled_approximation_without_claude(self):
        state = self.heads_up_state()
        history = app.HandHistory(snapshots=[state], hand_id=state.hand_id)
        outcome = SimpleNamespace(
            solved=True,
            status=app.LiveGTOStatus.SOLVED,
            reason="",
            analysis=(
                "**Action:** Check\n"
                "**Size:** 0\n"
                "**Why:** Approximate local solver mix."
            ),
            source="APPROXIMATE_SOLVER fresh",
            approximate=True,
            model="b-inary/postflop-solver",
            spec=SimpleNamespace(cache_key="d" * 64),
        )
        router = mock.Mock()
        router.evaluate.return_value = outcome
        claude = mock.Mock()

        with mock.patch.object(
            app,
            "evaluate_strategy_snapshot",
            side_effect=AssertionError("GTO_HU must never invoke Claude"),
        ) as evaluate_claude:
            result = app.evaluate_strategy_backend(
                state,
                history,
                mode="COACH",
                backend="GTO_HU",
                router=router,
                client=claude,
            )

        evaluate_claude.assert_not_called()
        claude.with_options.assert_not_called()
        claude.messages.create.assert_not_called()
        self.assertEqual("APPROXIMATE_SOLVER", result.mode)
        self.assertEqual("APPROXIMATE_SOLVER fresh", result.source)
        self.assertEqual(
            "APPROXIMATE_SOLVER",
            json.loads(result.prompt)["backend"],
        )

    def test_multiway_mode_never_falls_back_to_the_local_hu_router(self):
        state = self.heads_up_state()
        history = app.HandHistory(snapshots=[state], hand_id=state.hand_id)
        local_router = mock.Mock(spec=app.LiveGTORouter)

        result = app.evaluate_strategy_backend(
            state,
            history,
            backend="GTO_MULTIWAY",
            router=local_router,
        )

        local_router.evaluate.assert_not_called()
        self.assertEqual("GTO_MULTIWAY", result.mode)
        self.assertIn("requires GTO_EXECUTION_MODE=remote", result.error)

    def test_gto_hu_unsupported_node_does_not_fall_back_to_claude(self):
        state = self.heads_up_state()
        history = app.HandHistory(snapshots=[state], hand_id=state.hand_id)
        outcome = SimpleNamespace(
            solved=False,
            status=app.LiveGTOStatus.UNSUPPORTED,
            reason="the solver requires exactly one active villain",
        )
        router = mock.Mock()
        router.evaluate.return_value = outcome

        with mock.patch.object(
            app,
            "evaluate_strategy_snapshot",
            side_effect=AssertionError("GTO_HU must never invoke Claude"),
        ) as evaluate_claude:
            result = app.evaluate_strategy_backend(
                state,
                history,
                backend="GTO_HU",
                router=router,
            )

        evaluate_claude.assert_not_called()
        self.assertEqual("GTO_HU", result.mode)
        self.assertIn("GTO_HU unsupported", result.error)

    def test_gto_hu_charts_use_same_hand_position_continuity(self):
        hand_id = "live-position-continuity"
        preflop = snapshot(
            [
                player(0, "MP", status="FOLDED", stack=250.4),
                player(1, "CO", bet=2.0, stack=157.3),
                player(2, "BTN", bet=2.0, stack=70.4),
                player(3, "SB", status="FOLDED", bet=0.5, stack=250.6),
                player(4, "BB", hero=True, stack=68.5),
                player(5, "UTG", bet=1.0, stack=80.2),
            ],
            street="PREFLOP",
            board=[],
            dealer=2,
            pot=6.5,
        )
        flop = snapshot(
            [
                player(0, "MP", status="FOLDED", stack=250.4),
                player(1, "CO", stack=157.3),
                player(2, "BTN", stack=70.4),
                player(3, "SB", status="FOLDED", stack=250.6),
                player(4, "BB", hero=True, stack=67.5),
                player(5, "UTG", stack=80.2),
            ],
            street="FLOP",
            board=["8s", "2s", "7h"],
            dealer=2,
            pot=8.03,
        )
        river = snapshot(
            [
                player(0, "MP", status="FOLDED", stack=0),
                player(1, "CO", status="FOLDED", stack=153.3),
                player(2, "BTN", stack=43.2),
                player(3, "SB", status="FOLDED", stack=250.6),
                player(4, "BB", hero=True, stack=40.3),
                player(5, "UTG", status="FOLDED", stack=75.2),
            ],
            street="RIVER",
            board=["8s", "2s", "7h", "Td", "4c"],
            dealer=2,
            pot=67.1,
        )
        for state in (preflop, flop, river):
            state.hand_id = hand_id
            state.action_on_seat_index = 4
        history = app.HandHistory(
            snapshots=[preflop, flop, river],
            hand_id=hand_id,
        )
        outcome = SimpleNamespace(
            solved=True,
            status=app.LiveGTOStatus.SOLVED,
            reason="",
            analysis=(
                "**Action:** Check\n"
                "**Size:** 0\n"
                "**Why:** Approximate chart-seeded solver mix."
            ),
            source="APPROXIMATE_SOLVER fresh",
            approximate=True,
            model="b-inary/postflop-solver",
            spec=SimpleNamespace(cache_key="e" * 64),
        )
        router = mock.Mock()
        router.config = SimpleNamespace(range_source="charts")
        router.evaluate.return_value = outcome

        with mock.patch.object(
            app,
            "evaluate_strategy_snapshot",
            side_effect=AssertionError("GTO_HU must never invoke Claude"),
        ):
            result = app.evaluate_strategy_backend(
                river,
                history,
                backend="GTO_HU",
                router=router,
            )

        routed = router.evaluate.call_args.args[0]
        self.assertEqual("", routed.preflop_mapping_error)
        self.assertIsNotNone(routed.preflop_observation)
        self.assertFalse(routed.preflop_observation.terminal)
        self.assertEqual(
            app.POSITION_ONLY_HANDOFF_SOURCE,
            routed.preflop_observation.provenance.source,
        )
        self.assertEqual(
            frozenset({"UTG", "CO", "BTN", "BB"}),
            routed.preflop_observation.live_positions,
        )
        self.assertEqual("APPROXIMATE_SOLVER", result.mode)

    def test_strict_gto_rejects_approximate_result_without_calling_claude(self):
        state = self.heads_up_state()
        history = app.HandHistory(snapshots=[state], hand_id=state.hand_id)
        outcome = SimpleNamespace(
            solved=True,
            status=app.LiveGTOStatus.SOLVED,
            reason="",
            analysis=(
                "**Action:** Check\n"
                "**Size:** 0\n"
                "**Why:** Approximate local solver mix."
            ),
            source="APPROXIMATE_SOLVER fresh",
            approximate=True,
            model="b-inary/postflop-solver",
            spec=SimpleNamespace(cache_key="c" * 64),
        )
        router = mock.Mock()
        router.evaluate.return_value = outcome
        claude = mock.Mock()

        with mock.patch.object(
            app,
            "evaluate_strategy_snapshot",
            side_effect=AssertionError("strict GTO must not invoke Claude"),
        ) as evaluate_claude:
            result = app.evaluate_strategy_backend(
                state,
                history,
                mode="COACH",
                backend="GTO",
                router=router,
                client=claude,
            )

        evaluate_claude.assert_not_called()
        claude.with_options.assert_not_called()
        claude.messages.create.assert_not_called()
        self.assertEqual("GTO", result.mode)
        self.assertEqual(
            "GTO failed: strict GTO mode rejected an explicitly approximate "
            "solver result",
            result.error,
        )
        self.assertEqual(f"Strategy Error: {result.error}", result.final_analysis)

    def test_hybrid_unsupported_calls_claude_once_and_labels_fallback(self):
        state = self.heads_up_state()
        history = app.HandHistory(snapshots=[state], hand_id=state.hand_id)
        outcome = SimpleNamespace(
            solved=False,
            status=app.LiveGTOStatus.UNSUPPORTED,
            reason="the current bridge has no IP root policy",
            analysis="",
            source="",
            spec=None,
        )
        router = mock.Mock()
        router.evaluate.return_value = outcome
        claude = mock.Mock()
        claude.messages.create.return_value = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="text",
                    text=(
                        "**Action:** Check\n"
                        "**Size:** 0\n"
                        "**Why:** Keep the pot controlled."
                    ),
                )
            ]
        )

        result = app.evaluate_strategy_backend(
            state,
            history,
            mode="COACH",
            backend="HYBRID",
            router=router,
            client=claude,
        )

        router.evaluate.assert_called_once()
        claude.messages.create.assert_called_once()
        self.assertEqual(app.CLAUDE_MODEL, claude.messages.create.call_args.kwargs["model"])
        self.assertIn("**Action:** Check", result.final_analysis)
        self.assertIn(
            "**Backend:** Claude fallback — GTO unsupported:",
            result.final_analysis,
        )
        self.assertEqual(4.0, result.metrics["final_pot"])
        self.assertEqual(
            f"COACH {app.CLAUDE_MODEL} fallback — GTO unsupported: "
            "the current bridge has no IP root policy",
            result.source,
        )

    def test_same_street_progress_is_not_treated_as_a_solver_root(self):
        previous = self.heads_up_state(pot=4.0, hero_stack=100.0)
        current = self.heads_up_state(pot=6.0, hero_stack=98.0)
        history = app.HandHistory(snapshots=[previous], hand_id=previous.hand_id)

        self.assertFalse(app.live_gto_street_root_confirmed(current, history))
        self.assertTrue(app.live_gto_street_root_confirmed(previous, history))


class LiveGTODecisionPathTests(unittest.TestCase):
    @staticmethod
    def build_state(state, history):
        state.action_on_seat_index = 4
        state.hand_id = "mapped-live-node"
        for prior in history.snapshots:
            prior.hand_id = state.hand_id
        prepared = app.prepare_strategy_state(state, history)
        return app.build_live_gto_state(state, history, prepared)

    @staticmethod
    def six_max_players(*, flop=False):
        positions = ("UTG", "MP", "CO", "BTN", "SB", "BB")
        result = []
        for seat, position in enumerate(positions):
            is_hero = seat == 4
            if flop:
                survives = seat in {4, 5}
                result.append(
                    player(
                        seat,
                        position,
                        hero=is_hero,
                        status="ACTIVE" if survives else "FOLDED",
                        bet=0,
                        stack=97 if survives else 98,
                    )
                )
            else:
                contribution = 3 if seat in {4, 5} else 2
                result.append(
                    player(
                        seat,
                        position,
                        hero=is_hero,
                        status="ACTIVE",
                        bet=contribution,
                        stack=100 - contribution,
                    )
                )
        return result

    def test_build_live_state_propagates_current_six_max_preflop_observation(self):
        players = [
            player(0, "UTG", status="FOLDED"),
            player(1, "MP", status="FOLDED"),
            player(2, "CO", status="FOLDED"),
            player(3, "BTN", status="FOLDED"),
            player(4, "SB", hero=True, stack=99.5, bet=0.5),
            player(5, "BB", stack=99, bet=1),
        ]
        state = snapshot(
            players, street="PREFLOP", board=[], dealer=3, pot=1.5
        )
        state.hand_id = "preflop-bridge"
        state.action_on_seat_index = 4
        state.last_action_context = app.LastActionContext(
            amount_to_call=0.5,
            hero_action_options=["Fold", "Call", "Raise"],
        )
        history = app.HandHistory(snapshots=[state], hand_id=state.hand_id)
        prepared = app.prepare_strategy_state(state, history)

        mapped = app.build_live_gto_state(state, history, prepared)

        self.assertEqual("SB", mapped.preflop_observation.actor)
        self.assertEqual(
            frozenset({"UTG", "HJ", "CO", "BTN"}),
            mapped.preflop_observation.folded,
        )
        self.assertEqual(
            Decimal("100.0"),
            mapped.preflop_observation.initial_stack_map["SB"],
        )
        self.assertEqual("", mapped.preflop_mapping_error)

    def test_build_live_state_propagates_terminal_preflop_handoff(self):
        preflop = snapshot(
            self.six_max_players(),
            street="PREFLOP",
            board=[],
            dealer=3,
            pot=14,
        )
        flop = snapshot(
            self.six_max_players(flop=True),
            street="FLOP",
            board=["Kc", "7d", "3s"],
            dealer=3,
            pot=18,
        )
        for state in (preflop, flop):
            state.hand_id = "terminal-bridge"
            state.action_on_seat_index = 4
        flop.last_action_context = app.LastActionContext(
            amount_to_call=0,
            hero_action_options=["Check", "Bet"],
        )
        history = app.HandHistory(
            snapshots=[preflop, flop], hand_id=flop.hand_id
        )
        prepared = app.prepare_strategy_state(flop, history)

        mapped = app.build_live_gto_state(flop, history, prepared)

        self.assertTrue(mapped.preflop_observation.terminal)
        self.assertEqual(
            frozenset({"SB", "BB"}), mapped.preflop_observation.live_positions
        )
        self.assertEqual(
            Decimal(3), mapped.preflop_observation.contribution_map["SB"]
        )
        self.assertEqual("", mapped.preflop_mapping_error)

    def test_ip_no_wager_maps_to_oop_check(self):
        state = snapshot(
            [
                player(0, "BB", stack=80),
                player(4, "BTN", hero=True, stack=100),
            ],
            board=["Kc", "7c", "5d"],
            dealer=4,
            pot=8,
        )
        history = app.HandHistory(snapshots=[state])
        mapped = self.build_state(state, history)

        self.assertFalse(mapped.hero_is_oop)
        self.assertEqual(("CHECK",), mapped.action_history)
        self.assertEqual(8, mapped.pot_bb)
        self.assertEqual(100, mapped.hero_stack_bb)
        self.assertEqual(80, mapped.villain_stack_bb)
        self.assertEqual("", mapped.mapping_error)

    def test_ip_facing_first_bet_reconstructs_inclusive_root(self):
        state = snapshot(
            [
                player(0, "BB", stack=78, bet=2, action="BET"),
                player(4, "BTN", hero=True, stack=100),
            ],
            board=["Kc", "7c", "5d"],
            dealer=4,
            pot=10,
        )
        state.last_action_context = app.LastActionContext(
            amount_to_call=2,
            hero_action_options=["Fold", "Call", "Raise"],
        )
        history = app.HandHistory(snapshots=[state])
        mapped = self.build_state(state, history)

        self.assertFalse(mapped.hero_is_oop)
        self.assertEqual(("BET",), mapped.action_history)
        self.assertEqual(Decimal("2"), mapped.observed_bet_to_bb)
        self.assertEqual(Decimal("8"), mapped.pot_bb)
        self.assertEqual(Decimal("100"), mapped.hero_stack_bb)
        self.assertEqual(Decimal("80"), mapped.villain_stack_bb)
        self.assertEqual("", mapped.mapping_error)

    def test_oop_facing_ip_bet_requires_and_uses_prior_root(self):
        prior = snapshot(
            [
                player(0, "BTN", stack=80),
                player(4, "BB", hero=True, stack=100),
            ],
            board=["Kc", "7c", "5d"],
            dealer=0,
            pot=8,
        )
        prior.action_on_seat_index = 4
        current = snapshot(
            [
                player(0, "BTN", stack=78, bet=2, action="BET"),
                player(4, "BB", hero=True, stack=100),
            ],
            board=["Kc", "7c", "5d"],
            dealer=0,
            pot=10,
        )
        current.last_action_context = app.LastActionContext(
            amount_to_call=2,
            hero_action_options=["Fold", "Call", "Raise"],
        )
        history = app.HandHistory(snapshots=[prior, current])
        mapped = self.build_state(current, history)

        self.assertTrue(mapped.hero_is_oop)
        self.assertEqual(("CHECK", "BET"), mapped.action_history)
        self.assertEqual(Decimal("8"), mapped.pot_bb)
        self.assertEqual(Decimal("100"), mapped.hero_stack_bb)
        self.assertEqual(Decimal("80"), mapped.villain_stack_bb)
        self.assertEqual("", mapped.mapping_error)

    def test_oop_facing_ip_bet_without_prior_root_is_refused(self):
        current = snapshot(
            [
                player(0, "BTN", stack=78, bet=2),
                player(4, "BB", hero=True, stack=100),
            ],
            board=["Kc", "7c", "5d"],
            dealer=0,
            pot=10,
        )
        current.last_action_context = app.LastActionContext(
            amount_to_call=2,
            hero_action_options=["Fold", "Call", "Raise"],
        )
        history = app.HandHistory(snapshots=[current])
        mapped = self.build_state(current, history)

        self.assertEqual((), mapped.action_history)
        self.assertIn("prior Hero-check root", mapped.mapping_error)

    def test_prior_hero_bet_then_raise_is_not_mislabeled_check_bet(self):
        current = snapshot(
            [
                player(0, "BTN", stack=75, bet=5, action="RAISE"),
                player(4, "BB", hero=True, stack=98, bet=2),
            ],
            board=["Kc", "7c", "5d"],
            dealer=0,
            pot=15,
        )
        current.last_action_context = app.LastActionContext(
            amount_to_call=3,
            hero_action_options=["Fold", "Call", "Raise"],
        )
        history = app.HandHistory(snapshots=[current])
        mapped = self.build_state(current, history)

        self.assertEqual((), mapped.action_history)
        self.assertIn("unsupported prior call, bet, or raise", mapped.mapping_error)

    def test_replay_1784233445_remains_truthfully_three_way(self):
        preflop = snapshot(
            [
                player(0, "SB", status="ACTIVE", bet=1.0, stack=180.2),
                player(1, "OUT", status="EMPTY", stack=0),
                player(2, "BB", status="ACTIVE", bet=4.0, stack=62.9),
                player(3, "OUT", status="EMPTY", stack=0),
                player(4, "BTN", hero=True, stack=94.4),
                player(5, "OUT", status="EMPTY", stack=0),
            ],
            street="PREFLOP",
            board=[],
            dealer=4,
            pot=5.0,
        )
        flop = snapshot(
            [
                player(0, "SB", status="ACTIVE", stack=177.2),
                player(1, "UTG", status="FOLDED"),
                player(2, "BB", status="ACTIVE", stack=62.9),
                player(3, "CO", status="FOLDED"),
                player(4, "BTN", hero=True, stack=91.4),
                player(5, "SB", status="FOLDED", stack=63.9),
            ],
            street="FLOP",
            board=["Kh", "Jc", "5h"],
            dealer=4,
            pot=11.3,
        )
        for state in (preflop, flop):
            state.hand_id = "1784233445"
            next(p for p in state.players if p.is_hero).hole_cards = ["Kc", "7c"]
        flop.action_on_seat_index = 4
        flop.last_action_context = app.LastActionContext(
            amount_to_call=0,
            hero_action_options=["Check"],
        )
        history = app.HandHistory(
            snapshots=[preflop],
            hand_id=preflop.hand_id,
        )

        app.reconcile_snapshot_with_history(flop, history)
        history.add_snapshot(flop)
        prepared = app.prepare_strategy_state(flop, history)
        mapped = app.build_live_gto_state(flop, history, prepared)

        self.assertEqual(2, mapped.active_villains)
        self.assertIn("exactly one HU villain", mapped.mapping_error)
        router = app.LiveGTORouter(
            app.LiveGTOConfig(
                enabled=True,
                owned_simulator_acknowledged=True,
            ),
            range_provider=mock.Mock(),
        )
        outcome = router.evaluate(mapped)
        self.assertEqual(app.LiveGTOStatus.UNSUPPORTED, outcome.status)
        self.assertIn("exactly one HU villain", outcome.reason)


class FlowFreshnessGateTests(unittest.TestCase):
    def setUp(self):
        self.original_history = app.current_history
        self.original_recorder = app.recorder
        self.original_backend = app.STRATEGY_BACKEND
        app.current_history = app.HandHistory()
        app.recorder = None
        # This class verifies the explicit Claude flow and must not inherit the
        # process/user strategy backend.
        app.STRATEGY_BACKEND = "CLAUDE"

    def tearDown(self):
        app.current_history = self.original_history
        app.recorder = self.original_recorder
        app.STRATEGY_BACKEND = self.original_backend

    @staticmethod
    def captures():
        return FreshnessAndButtonTests().base_captures()

    @staticmethod
    def actionable_state():
        state = snapshot(
            [
                player(0, "BTN"),
                player(1, "OUT", status="EMPTY", stack=0),
                player(2, "OUT", status="EMPTY", stack=0),
                player(3, "OUT", status="EMPTY", stack=0),
                player(4, "SB", hero=True),
                player(5, "OUT", status="EMPTY", stack=0),
            ],
            board=["Kc", "7c", "5d"],
            dealer=0,
            pot=4.0,
        )
        state.action_on_seat_index = 4
        state.hand_id = "flow-test"
        return state

    def test_change_after_vision_skips_claude(self):
        baseline = self.captures()
        changed = {name: image.copy() for name, image in baseline.items()}
        ImageDraw.Draw(changed["board"]).rectangle(
            (20, 25, 190, 125), fill=(235, 235, 235)
        )
        client = mock.Mock()

        with (
            mock.patch.object(
                app, "analyze_state",
                return_value=(self.actionable_state(), baseline, 0.01, 0.02),
            ),
            mock.patch.object(app, "capture_validation_regions", return_value=changed),
            mock.patch.object(app, "anthropic_client", client),
            mock.patch.object(app, "PROMPT_MODE", "COACH"),
            mock.patch.object(app, "save_stale_debug_capture"),
            mock.patch.object(app, "display_stale_result") as stale_display,
            mock.patch.object(app, "display_results") as normal_display,
        ):
            app.run_analysis_flow("strategy")

        client.messages.create.assert_not_called()
        stale_display.assert_called_once()
        normal_display.assert_not_called()
        self.assertEqual([], app.current_history.snapshots)

    def test_change_after_claude_discards_response(self):
        baseline = self.captures()
        changed = {name: image.copy() for name, image in baseline.items()}
        ImageDraw.Draw(changed["board"]).rectangle(
            (20, 25, 190, 125), fill=(235, 235, 235)
        )
        client = mock.Mock()
        client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(
                type="text",
                text="**Action:** Check\n**Size:** 0\n**Why:** Keep the pot small.",
            )]
        )

        with (
            mock.patch.object(
                app, "analyze_state",
                return_value=(self.actionable_state(), baseline, 0.01, 0.02),
            ),
            mock.patch.object(
                app, "capture_validation_regions", side_effect=[baseline, changed]
            ),
            mock.patch.object(app, "anthropic_client", client),
            mock.patch.object(app, "PROMPT_MODE", "COACH"),
            mock.patch.object(app, "save_stale_debug_capture"),
            mock.patch.object(app, "display_stale_result") as stale_display,
            mock.patch.object(app, "display_results") as normal_display,
        ):
            app.run_analysis_flow("strategy")

        client.messages.create.assert_called_once()
        self.assertEqual(
            app.CLAUDE_MODEL,
            client.messages.create.call_args.kwargs["model"],
        )
        self.assertNotIn("temperature", client.messages.create.call_args.kwargs)
        stale_display.assert_called_once()
        normal_display.assert_not_called()

    def test_fast_mode_sends_verified_text_state_to_haiku(self):
        baseline = self.captures()
        state = self.actionable_state()
        hero = next(p for p in state.players if p.is_hero)
        client = mock.Mock()
        fast_client = mock.Mock()
        client.with_options.return_value = fast_client
        fast_client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(
                type="text",
                text=(
                    "**Action:** Check\n"
                    "**Size:** 0\n"
                    "**Why:** Keep the pot controlled while realizing the draw."
                ),
            )]
        )

        with (
            mock.patch.object(app, "PROMPT_MODE", "FAST"),
            mock.patch.object(
                app, "analyze_state", return_value=(state, baseline, 0.01, 1.7)
            ) as vision,
            mock.patch.object(app, "anthropic_client", client),
            mock.patch.object(app, "request_parallel_strategy") as strategy,
            mock.patch.object(
                app, "capture_validation_regions", return_value=baseline
            ),
            mock.patch.object(app, "display_results") as normal_display,
        ):
            app.run_analysis_flow("strategy")

        self.assertEqual(["6d", "4h"], hero.hole_cards)
        vision.assert_called_once()
        strategy.assert_not_called()
        client.with_options.assert_called_once()
        self.assertEqual(
            app.CLAUDE_FAST_MODEL,
            fast_client.messages.create.call_args.kwargs["model"],
        )
        prompt = fast_client.messages.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("Board: Kc 7c 5d", prompt)
        displayed_analysis = normal_display.call_args.args[1]
        self.assertIn("**Action:** Check", displayed_analysis)
        self.assertFalse(displayed_analysis.startswith("Strategy Error:"))

    def test_fast_no_longer_runs_a_second_visual_card_read(self):
        baseline = self.captures()
        state = self.actionable_state()
        hero = next(player for player in state.players if player.is_hero)
        original_cards = list(hero.hole_cards)
        client = mock.Mock()
        fast_client = mock.Mock()
        client.with_options.return_value = fast_client
        fast_client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(
                type="text",
                text="**Action:** Check\n**Size:** 0\n**Why:** Realize equity.",
            )]
        )

        with (
            mock.patch.object(app, "PROMPT_MODE", "FAST"),
            mock.patch.object(
                app, "analyze_state", return_value=(state, baseline, 0.01, 1.7)
            ),
            mock.patch.object(app, "anthropic_client", client),
            mock.patch.object(app, "request_parallel_strategy") as visual_strategy,
            mock.patch.object(
                app, "capture_validation_regions", return_value=baseline
            ),
            mock.patch.object(app, "display_results") as normal_display,
        ):
            app.run_analysis_flow("strategy")

        visual_strategy.assert_not_called()
        self.assertEqual(1, len(app.current_history.snapshots))
        self.assertEqual(original_cards, hero.hole_cards)
        displayed_analysis = normal_display.call_args.args[1]
        self.assertNotIn("disagreed", displayed_analysis)

    def test_fast_provider_timeout_is_reported(self):
        baseline = self.captures()
        state = self.actionable_state()
        client = mock.Mock()
        fast_client = mock.Mock()
        client.with_options.return_value = fast_client
        fast_client.messages.create.side_effect = TimeoutError("request timed out")

        with (
            mock.patch.object(app, "PROMPT_MODE", "FAST"),
            mock.patch.object(
                app, "analyze_state", return_value=(state, baseline, 0.01, 0.05)
            ),
            mock.patch.object(app, "anthropic_client", client),
            mock.patch.object(app, "capture_validation_regions", return_value=baseline),
            mock.patch.object(app, "display_results") as normal_display,
        ):
            app.run_analysis_flow("strategy")

        self.assertEqual(1, len(app.current_history.snapshots))
        self.assertIn("timed out", normal_display.call_args.args[1])


class AnalysisHotkeyLockTests(unittest.TestCase):
    def test_claude_hotkey_is_disabled_in_gto_only_mode(self):
        with (
            mock.patch.object(app, "STRATEGY_BACKEND", "GTO_HU"),
            mock.patch.object(app, "PROMPT_MODE", "FAST"),
            mock.patch.object(app.console, "print") as output,
        ):
            app.on_press(SimpleNamespace(char="m"))
            self.assertEqual("FAST", app.PROMPT_MODE)

        self.assertIn("GTO-only mode", output.call_args.args[0])

    def test_repeated_analysis_request_is_ignored_while_worker_runs(self):
        started = threading.Event()
        release = threading.Event()

        def blocked_analysis(mode):
            started.set()
            release.wait(timeout=1)

        with mock.patch.object(app, "run_analysis_flow", side_effect=blocked_analysis) as run:
            app.start_analysis_flow("strategy")
            self.assertTrue(started.wait(timeout=1))
            app.start_analysis_flow("strategy")
            run.assert_called_once_with(mode="strategy")
            release.set()

        for _ in range(100):
            if not app.analysis_lock.locked():
                break
            time.sleep(0.001)
        self.assertFalse(app.analysis_lock.locked())


if __name__ == "__main__":
    unittest.main()
