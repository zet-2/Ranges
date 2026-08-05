"""Tests for PokerBench benchmark loading, scoring, and reporting."""

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("POKER_ASSISTANT_OFFLINE", "1")

import poker_assistant as app
import pokerbench_benchmark as bench


class MoveParsingTests(unittest.TestCase):
    def test_source_move_normalization(self):
        cases = [
            ("call", "preflop", bench.Move("CALL")),
            ("3.0 bb", "preflop", bench.Move("RAISE", bench.Decimal("3.0"))),
            ("Bet 24", "postflop", bench.Move("BET", bench.Decimal("24"))),
            ("Raise 29", "postflop", bench.Move("RAISE", bench.Decimal("29"))),
            ("all-in", "postflop", bench.Move("ALL_IN")),
        ]
        for raw, split, expected in cases:
            with self.subTest(raw=raw, split=split):
                self.assertEqual(expected, bench.parse_move(raw, split=split))

    def test_model_response_normalization(self):
        cases = [
            ('{"action":"CALL","amount":0}', bench.Move("CALL")),
            ('{"action":"RAISE","amount":29}', bench.Move("RAISE", bench.Decimal("29"))),
            ("Action: CHECK\nSize: 0", bench.Move("CHECK")),
            ("**Action:** Raise\n**Size:** 29 BB", bench.Move("RAISE", bench.Decimal("29"))),
            ("Bet 24", bench.Move("BET", bench.Decimal("24"))),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(expected, bench.parse_model_move(raw))

    def test_model_parser_rejects_ambiguous_or_invalid_output(self):
        invalid = [
            "Call or fold",
            "Action: CALL\nAction: FOLD\nSize: 0",
            "Action: CHECK\nSize: 1",
            "Action: RAISE\nSize: -2",
            "Action: RAISE 10\nSize: 20",
            "Bet 0",
            '{"action":"BET","amount":0}',
            '{"action":"CHECK","amount":1}',
            "",
        ]
        for raw in invalid:
            with self.subTest(raw=raw):
                with self.assertRaises(bench.PokerBenchDataError):
                    bench.parse_model_move(raw)

    def test_available_moves_uses_literal_parser(self):
        moves = bench.parse_available_moves(
            "['Fold', 'Call', 'Raise 29']",
            split="postflop",
        )
        self.assertEqual(
            (
                bench.Move("FOLD"),
                bench.Move("CALL"),
                bench.Move("RAISE", bench.Decimal("29")),
            ),
            moves,
        )
        with self.assertRaises(bench.PokerBenchDataError):
                bench.parse_available_moves(
                    "__import__('os').system('echo unsafe')",
                    split="postflop",
                )
        with self.assertRaises(bench.PokerBenchDataError):
            bench.parse_move("0 bb", split="preflop")


class AdapterTests(unittest.TestCase):
    POST_HEADERS = [
        "", "preflop_action", "board_flop", "board_turn", "board_river",
        "aggressor_position", "postflop_action", "evaluation_at",
        "available_moves", "pot_size", "hero_position", "holding",
        "correct_decision",
    ]

    PRE_HEADERS = [
        "", "prev_line", "hero_pos", "hero_holding", "correct_decision",
        "num_players", "num_bets", "available_moves", "pot_size",
    ]

    def _write_csv(self, directory, name, headers, rows):
        path = Path(directory) / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_postflop_adapter_does_not_leak_future_cards(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_csv(
                directory,
                "post.csv",
                self.POST_HEADERS,
                [{
                    "": "7",
                    "preflop_action": "HJ/2.0bb/BB/call",
                    "board_flop": "Ks7h2d",
                    "board_turn": "Jc",
                    "board_river": "7c",
                    "aggressor_position": "OOP",
                    "postflop_action": "OOP_CHECK",
                    "evaluation_at": "Flop",
                    "available_moves": "['Check', 'Bet 5']",
                    "pot_size": "6",
                    "hero_position": "IP",
                    "holding": "8h8c",
                    "correct_decision": "Check",
                }],
            )
            cases, rejected = bench.load_postflop_cases(path)
        self.assertEqual([], rejected)
        self.assertEqual(1, len(cases))
        prompt = cases[0].prompt
        self.assertIn("Ks 7h 2d", prompt)
        self.assertNotIn("Jc", prompt)
        self.assertNotIn("7c", prompt)

    def test_preflop_adapter_cleans_known_whitespace_noise(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_csv(
                directory,
                "pre.csv",
                self.PRE_HEADERS,
                [{
                    "": "3",
                    "prev_line": "U TG/2.0 bb/BTN/fold",
                    "hero_pos": "B B",
                    "hero_holding": "9 d5d",
                    "correct_decision": "check",
                    "num_players": "2",
                    "num_bets": "1",
                    "available_moves": "['check', 'fold']",
                    "pot_size": "4.5",
                }],
            )
            cases, rejected = bench.load_preflop_cases(path)
        self.assertEqual([], rejected)
        self.assertEqual(["9d", "5d"], cases[0].metadata["holding"])
        self.assertEqual("BB", cases[0].metadata["hero_position"])

    def test_malformed_row_is_quarantined(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_csv(
                directory,
                "post.csv",
                self.POST_HEADERS,
                [{
                    "": "9",
                    "preflop_action": "HJ/2.0bb/BB/call",
                    "board_flop": "KsKs2d",
                    "board_turn": "Jc",
                    "board_river": "7c",
                    "aggressor_position": "OOP",
                    "postflop_action": "OOP_CHECK",
                    "evaluation_at": "Flop",
                    "available_moves": "['Check']",
                    "pot_size": "6",
                    "hero_position": "IP",
                    "holding": "8h8c",
                    "correct_decision": "Check",
                }],
            )
            cases, rejected = bench.load_postflop_cases(path)
        self.assertEqual([], cases)
        self.assertEqual(1, len(rejected))
        self.assertIn("duplicate", rejected[0]["error"])

    def test_cross_street_duplicate_and_action_board_mismatch_are_quarantined(self):
        rows = [
            {
                "": "10",
                "preflop_action": "HJ/2.0bb/BB/call",
                "board_flop": "Ks7h2d",
                "board_turn": "Ks",
                "board_river": "7c",
                "aggressor_position": "OOP",
                "postflop_action": "OOP_CHECK/dealcards/Ks/OOP_CHECK",
                "evaluation_at": "Turn",
                "available_moves": "['Check']",
                "pot_size": "6",
                "hero_position": "IP",
                "holding": "8h8c",
                "correct_decision": "Check",
            },
            {
                "": "11",
                "preflop_action": "HJ/2.0bb/BB/call",
                "board_flop": "Ks7h2d",
                "board_turn": "Jc",
                "board_river": "7c",
                "aggressor_position": "OOP",
                "postflop_action": "OOP_CHECK/dealcards/Qc/OOP_CHECK",
                "evaluation_at": "Turn",
                "available_moves": "['Check']",
                "pot_size": "6",
                "hero_position": "IP",
                "holding": "8h8c",
                "correct_decision": "Check",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_csv(
                directory, "post.csv", self.POST_HEADERS, rows
            )
            cases, rejected = bench.load_postflop_cases(path)
        self.assertEqual([], cases)
        self.assertEqual(2, len(rejected))
        self.assertIn("duplicate", rejected[0]["error"])
        self.assertIn("disagree", rejected[1]["error"])


class FakeCompleter:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        response = self.responses[len(self.calls) - 1]
        if isinstance(response, Exception):
            raise response
        return bench.Completion(
            text=response,
            latency_seconds=0.25,
            input_tokens=20,
            output_tokens=5,
        )


class RunnerAndCacheTests(unittest.TestCase):
    def _case(self, index, target, legal):
        return bench.PokerBenchCase(
            case_id=f"case-{index}",
            split="postflop",
            source_index=index,
            street="FLOP",
            prompt=f"prompt-{index}",
            target=target,
            legal_moves=tuple(legal),
            metadata={},
        )

    def test_cache_key_is_stable_and_configuration_sensitive(self):
        case = self._case(1, bench.Move("CHECK"), [bench.Move("CHECK")])
        first = bench.cache_key(case, model="model-a", max_tokens=80)
        same = bench.cache_key(case, model="model-a", max_tokens=80)
        changed = bench.cache_key(case, model="model-b", max_tokens=80)
        self.assertEqual(first, same)
        self.assertNotEqual(first, changed)

    def test_anthropic_request_uses_structured_output_without_temperature(self):
        class Messages:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(
                    content=[SimpleNamespace(
                        type="text",
                        text='{"action":"CHECK","amount":0}',
                    )],
                    usage=SimpleNamespace(input_tokens=10, output_tokens=8),
                    model="claude-sonnet-5-20260701",
                    stop_reason="end_turn",
                )

        completer = object.__new__(bench.AnthropicCompleter)
        completer.client = SimpleNamespace(messages=Messages())
        completion = completer.complete(
            "prompt", model="claude-sonnet-5", max_tokens=80
        )
        request = completer.client.messages.calls[0]
        self.assertNotIn("temperature", request)
        self.assertEqual(
            "json_schema", request["output_config"]["format"]["type"]
        )
        self.assertEqual({"type": "disabled"}, request["thinking"])
        self.assertEqual("end_turn", completion.stop_reason)

    def test_runner_resumes_from_cache_and_scores_all_cases(self):
        cases = [
            self._case(1, bench.Move("CHECK"), [bench.Move("CHECK")]),
            self._case(
                2,
                bench.Move("BET", bench.Decimal("24")),
                [bench.Move("CHECK"), bench.Move("BET", bench.Decimal("24"))],
            ),
        ]
        completer = FakeCompleter([
            "Action: CHECK\nSize: 0",
            "Action: BET\nSize: 20",
        ])
        with tempfile.TemporaryDirectory() as directory:
            cache = bench.JsonlCache(Path(directory) / "cache.jsonl")
            first = bench.run_cases(
                cases,
                model="test-model",
                completer=completer,
                cache=cache,
            )
            second = bench.run_cases(
                cases,
                model="test-model",
                completer=completer,
                cache=bench.JsonlCache(Path(directory) / "cache.jsonl"),
            )
        self.assertEqual(2, len(completer.calls))
        self.assertTrue(all(result["cached"] for result in second))
        report = bench.summarize(first, model="test-model", quarantined=[])
        self.assertEqual(1.0, report["overall"]["action_agreement"])
        self.assertEqual(0.5, report["overall"]["exact_decision_agreement"])
        self.assertEqual(0.5, report["overall"]["legal_move_rate"])

    def test_provider_failure_does_not_inflate_denominator_or_enter_cache(self):
        case = self._case(1, bench.Move("CALL"), [bench.Move("FOLD"), bench.Move("CALL")])
        completer = FakeCompleter([RuntimeError("timeout")])
        with tempfile.TemporaryDirectory() as directory:
            cache = bench.JsonlCache(Path(directory) / "cache.jsonl")
            results = bench.run_cases(
                [case], model="test-model", completer=completer, cache=cache
            )
        report = bench.summarize(results, model="test-model", quarantined=[])
        self.assertEqual(0.0, report["overall"]["coverage_rate"])
        self.assertEqual(0.0, report["overall"]["action_agreement"])
        self.assertEqual({}, cache.entries)

    def test_provider_error_circuit_breaker_stops_further_calls(self):
        cases = [
            self._case(index, bench.Move("CHECK"), [bench.Move("CHECK")])
            for index in range(5)
        ]
        completer = FakeCompleter([
            RuntimeError("rate limited"),
            RuntimeError("rate limited"),
        ])
        with tempfile.TemporaryDirectory() as directory:
            results = bench.run_cases(
                cases,
                model="test-model",
                completer=completer,
                cache=bench.JsonlCache(Path(directory) / "cache.jsonl"),
                max_consecutive_provider_errors=2,
            )
        self.assertEqual(2, len(completer.calls))
        self.assertEqual(
            [
                "PROVIDER_ERROR",
                "PROVIDER_ERROR",
                "ABORTED_PROVIDER_ERRORS",
                "ABORTED_PROVIDER_ERRORS",
                "ABORTED_PROVIDER_ERRORS",
            ],
            [result["status"] for result in results],
        )

    def test_malformed_cache_completion_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.jsonl"
            path.write_text(json.dumps({
                "schema_version": bench.CACHE_SCHEMA_VERSION,
                "key": "bad",
                "completion": {},
            }) + "\n", encoding="utf-8")
            cache = bench.JsonlCache(path)
        self.assertEqual(1, cache.corrupt_lines)
        self.assertEqual({}, cache.entries)

    def test_subset_selection_preserves_source_proportions(self):
        cases = []
        for index in range(90):
            cases.append(self._case(
                index, bench.Move("CHECK"), [bench.Move("CHECK")]
            ))
        for index in range(90, 100):
            cases.append(bench.PokerBenchCase(
                case_id=f"case-{index}",
                split="preflop",
                source_index=index,
                street="PREFLOP",
                prompt=f"prompt-{index}",
                target=bench.Move("CALL"),
                legal_moves=(bench.Move("CALL"),),
                metadata={},
            ))
        selected = bench.select_cases(cases, limit=20, seed=17)
        self.assertEqual(18, sum(case.split == "postflop" for case in selected))
        self.assertEqual(2, sum(case.split == "preflop" for case in selected))
        self.assertEqual(
            [case.case_id for case in selected],
            [case.case_id for case in bench.select_cases(cases, limit=20, seed=17)],
        )


class StrategyEntryPointTests(unittest.TestCase):
    class FakeMessages:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="**Action:** Check\n**Size:** 0\n**Why:** Free option.")]
            )

    class FakeClient:
        def __init__(self):
            self.messages = StrategyEntryPointTests.FakeMessages()
            self.options = []

        def with_options(self, **kwargs):
            self.options.append(kwargs)
            return self

    def test_structured_entry_point_bypasses_vision_and_preserves_input(self):
        snapshot = app.GameSnapshot(
            hand_id="offline-test",
            meta_info=app.MetaInfo(current_street="PREFLOP"),
            board_state=app.BoardState(total_pot=2.0),
            dealer_seat_index=3,
            action_on_seat_index=4,
            players=[
                app.Player(
                    seat_index=4,
                    name="SB",
                    username="Hero",
                    stack_size=99.5,
                    current_bet=0.5,
                    status="ACTIVE",
                    is_hero=True,
                    hole_cards=["As", "Kd"],
                ),
                app.Player(
                    seat_index=5,
                    name="BB",
                    username="Villain",
                    stack_size=99.0,
                    current_bet=1.0,
                    status="ACTIVE",
                ),
            ],
            last_action_context=app.LastActionContext(
                amount_to_call=0.0,
                hero_action_options=["Check", "Raise 3 BB"],
            ),
        )
        history = app.HandHistory(hand_id="offline-test", snapshots=[snapshot])
        before = history.to_json()
        client = self.FakeClient()

        result = app.evaluate_strategy_snapshot(
            snapshot,
            history,
            mode="FAST",
            client=client,
            action_history_override="No prior voluntary action.",
        )

        self.assertEqual(before, history.to_json())
        self.assertEqual("FAST", result.mode)
        self.assertEqual(app.CLAUDE_FAST_MODEL, result.model)
        self.assertIn("**Action:** Check", result.final_analysis)
        self.assertEqual(1, len(client.messages.calls))
        self.assertEqual(app.CLAUDE_FAST_MODEL, client.messages.calls[0]["model"])
        self.assertIn("No prior voluntary action", result.prompt)


if __name__ == "__main__":
    unittest.main()
