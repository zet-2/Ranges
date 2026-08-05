"""Tests for the GTO oracle benchmark and validation harness."""

import json
import itertools
import tempfile
import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import gto_oracle_benchmark as bench
from gto_oracle import (
    Action,
    ActionKind,
    ActionValue,
    ComboPolicy,
    OracleCache,
    SolveResult,
    SolverMetadata,
)


FAKE_BINARY_SHA256 = "f" * 64


def fake_result(spec):
    combos = (
        spec.oop_range.combos
        if spec.acting_player is bench.Position.OOP
        else spec.ip_range.combos
    )
    total = sum((combo.weight for combo in combos), Decimal(0))
    policies = []
    for index, combo in enumerate(combos):
        first_is_best = index % 2 == 0
        values = []
        action_count = len(spec.tree.modeled_actions)
        for action_index, action in enumerate(spec.tree.modeled_actions):
            preferred = action_index == (0 if first_is_best else 1)
            values.append(
                ActionValue(
                    action,
                    (
                        Decimal(1)
                        if action_count == 1
                        else Decimal("0.8")
                        if preferred
                        else Decimal("0.2") / Decimal(action_count - 1)
                    ),
                    Decimal(10) if preferred else Decimal(7),
                )
            )
        policies.append(
            ComboPolicy(
                private_combo=combo.cards,
                reach_weight=combo.weight / total,
                equity=Decimal("0.5"),
                action_values=tuple(values),
            )
        )
    return SolveResult.for_spec(
        spec,
        policies,
        SolverMetadata(
            solver_name=spec.parameters.solver_name,
            solver_version=spec.parameters.solver_commit,
            iterations=10,
            elapsed_seconds=Decimal("0.01"),
            exploitability=Decimal("0.001"),
            converged=True,
            extra=(("binary_sha256", FAKE_BINARY_SHA256),),
        ),
    )


class FakeEngine:
    binary_sha256 = FAKE_BINARY_SHA256

    def __init__(self):
        self.calls = []

    def solve(self, spec):
        self.calls.append(spec.cache_key)
        return fake_result(spec)


class FakeCompleter:
    def __init__(self):
        self.calls = []

    def complete(self, prompt, *, model, max_tokens, actions):
        self.calls.append(
            {
                "prompt": prompt,
                "model": model,
                "max_tokens": max_tokens,
                "actions": tuple(actions),
            }
        )
        index = 0 if "haiku" in model else 1
        return bench.Completion(
            text=json.dumps({"action_index": index}),
            latency_seconds=0.25,
            input_tokens=40,
            output_tokens=5,
            response_model=model + "-resolved",
            stop_reason="end_turn",
        )


class CaseAndPromptTests(unittest.TestCase):
    def test_strict_json_and_completion_reject_nonfinite_numbers(self):
        with self.assertRaisesRegex(bench.BenchmarkError, "non-finite"):
            bench._strict_json_loads('{"value":NaN}', field="fixture")
        with self.assertRaisesRegex(bench.BenchmarkError, "latency"):
            bench.Completion('{"action_index":0}', float("inf"))

    def test_demo_suite_round_trips_as_strict_case_json(self):
        cases = bench.demo_cases()
        self.assertGreaterEqual(len(cases), 2)
        self.assertEqual({"RIVER", "TURN"}, {case.spec.street.value for case in cases})
        self.assertGreaterEqual(
            sum(len(case.spec.oop_range.combos) for case in cases),
            7,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            bench.write_case_file(path, cases)
            loaded = bench.load_case_file(path)
        self.assertEqual(
            [case.spec.cache_key for case in cases],
            [case.spec.cache_key for case in loaded],
        )

    def test_representative_suite_is_stratified_and_round_trips(self):
        cases = bench.representative_validation_cases()
        self.assertEqual(22, len(cases))
        self.assertEqual(len(cases), len({case.case_id for case in cases}))
        self.assertEqual(
            len(cases),
            len({case.spec.cache_key for case in cases}),
        )
        self.assertEqual(
            {"FLOP", "TURN", "RIVER"},
            {case.spec.street.value for case in cases},
        )
        self.assertEqual(
            {0, 1, 2},
            {len(case.spec.tree.action_history) for case in cases},
        )
        self.assertTrue(
            {200, 400, 800}.issubset(
                {case.spec.tree.effective_stack for case in cases}
            )
        )
        self.assertTrue(any(case.spec.tree.rake_rate_pct > 0 for case in cases))
        self.assertEqual(
            {
                bench.AllocationMode.UNCOMPRESSED_F32,
                bench.AllocationMode.COMPRESSED_I16,
            },
            {case.spec.parameters.allocation_mode for case in cases},
        )
        self.assertTrue(
            any(
                case.spec.parameters.bet_sizes.turn_donk_sizes is None
                and case.spec.parameters.bet_sizes.river_donk_sizes == ""
                for case in cases
            )
        )
        self.assertTrue(
            any(
                case.spec.parameters.bet_sizes.river_donk_sizes == "25%"
                for case in cases
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "representative.json"
            bench.write_case_file(path, cases)
            loaded = bench.load_case_file(path)
        self.assertEqual(
            [case.spec.cache_key for case in cases],
            [case.spec.cache_key for case in loaded],
        )

    def test_stress_suite_adds_tight_and_two_size_flops(self):
        representative = bench.representative_validation_cases()
        stress = bench.stress_validation_cases()
        self.assertEqual(len(representative) + 2, len(stress))
        self.assertEqual(
            "0.1",
            str(stress[-2].spec.parameters.target_exploitability_pct),
        )
        self.assertEqual(
            {
                Action(ActionKind.CHECK),
                Action(ActionKind.BET, 33),
                Action(ActionKind.BET, 75),
            },
            set(stress[-1].spec.tree.modeled_actions),
        )

    def test_case_file_rejects_live_usage_and_unknown_fields(self):
        payload = bench.case_file_data([bench.demo_case()])
        payload["usage"] = "live"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(bench.BenchmarkError, "offline_post_session"):
                bench.load_case_file(path)

        payload = bench.case_file_data([bench.demo_case()])
        payload["unexpected"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(bench.BenchmarkError, "unexpected"):
                bench.load_case_file(path)

    def test_prompt_contains_state_but_not_oracle_policy_or_ev(self):
        case = bench.demo_case()
        result = fake_result(case.spec)
        decision = bench.PreparedDecision(
            case,
            result,
            result.combo_policies[0].private_combo,
        )
        prompt = bench.build_prompt(decision)
        self.assertIn("OFFLINE POST-SESSION", prompt)
        self.assertIn(
            "Decision node: ROOT; acting player (Hero): OOP.",
            prompt,
        )
        self.assertIn(
            "Ordered same-street action history from the tree root: []",
            prompt,
        )
        self.assertIn("ALL_IN_TO 10", prompt)
        self.assertIn("Exact OOP range", prompt)
        self.assertIn('"river_donk_sizes":null', prompt)
        self.assertIn('"force_allin_threshold":"0.15"', prompt)
        self.assertNotIn("oracle_mass", prompt)
        self.assertNotIn("action_ev", prompt)
        self.assertNotIn("frequency", prompt.lower())

    def test_prompt_identity_tracks_actor_history_and_every_tree_input(self):
        source = bench.demo_cases()[1]
        base_spec = source.spec
        parameters = base_spec.parameters
        sizing = parameters.bet_sizes

        def with_player_size(
            street_name,
            position_name,
            field_name,
            value,
        ):
            street = getattr(sizing, street_name)
            player = getattr(street, position_name)
            changed_player = replace(player, **{field_name: value})
            changed_street = replace(
                street,
                **{position_name: changed_player},
            )
            changed_sizing = replace(
                sizing,
                **{street_name: changed_street},
            )
            return replace(
                base_spec,
                parameters=replace(parameters, bet_sizes=changed_sizing),
            )

        variants = {}
        for street_name in ("flop", "turn", "river"):
            for position_name in ("oop", "ip"):
                variants[f"{street_name}_{position_name}_bet"] = with_player_size(
                    street_name,
                    position_name,
                    "bet",
                    "67%",
                )
                variants[
                    f"{street_name}_{position_name}_raise"
                ] = with_player_size(
                    street_name,
                    position_name,
                    "raise_sizes",
                    "3x",
                )

        variants.update(
            {
                "turn_donk": replace(
                    base_spec,
                    parameters=replace(
                        parameters,
                        bet_sizes=replace(sizing, turn_donk_sizes=""),
                    ),
                ),
                "river_donk": replace(
                    base_spec,
                    parameters=replace(
                        parameters,
                        bet_sizes=replace(
                            sizing,
                            river_donk_sizes="25%",
                        ),
                    ),
                ),
                "add_allin_threshold": replace(
                    base_spec,
                    parameters=replace(
                        parameters,
                        add_allin_threshold=Decimal("1.75"),
                    ),
                ),
                "force_allin_threshold": replace(
                    base_spec,
                    parameters=replace(
                        parameters,
                        force_allin_threshold=Decimal("0.2"),
                    ),
                ),
                "merging_threshold": replace(
                    base_spec,
                    parameters=replace(
                        parameters,
                        merging_threshold=Decimal("0.2"),
                    ),
                ),
                "pot": replace(
                    base_spec,
                    tree=replace(base_spec.tree, pot=101),
                ),
                "effective_stack": replace(
                    base_spec,
                    tree=replace(base_spec.tree, effective_stack=450),
                ),
                "rake_rate": replace(
                    base_spec,
                    tree=replace(
                        base_spec.tree,
                        rake_rate_pct=Decimal("4"),
                    ),
                ),
                "rake_cap": replace(
                    base_spec,
                    tree=replace(base_spec.tree, rake_cap=10),
                ),
                "modeled_action": replace(
                    base_spec,
                    tree=replace(
                        base_spec.tree,
                        modeled_actions=(
                            Action(ActionKind.CHECK),
                            Action(ActionKind.BET, 60),
                        ),
                    ),
                ),
                "legal_action_kind": replace(
                    base_spec,
                    tree=replace(
                        base_spec.tree,
                        legal_action_kinds=(
                            ActionKind.CHECK,
                            ActionKind.ALL_IN,
                        ),
                        modeled_actions=(
                            Action(ActionKind.CHECK),
                            Action(ActionKind.ALL_IN, 400),
                        ),
                    ),
                ),
                "descendant_actor_and_history": replace(
                    base_spec,
                    acting_player=bench.Position.IP,
                    tree=replace(
                        base_spec.tree,
                        action_history=(Action(ActionKind.CHECK),),
                    ),
                ),
                "target_exploitability": replace(
                    base_spec,
                    parameters=replace(
                        parameters,
                        target_exploitability_pct=Decimal("0.02"),
                    ),
                ),
                "max_iterations": replace(
                    base_spec,
                    parameters=replace(
                        parameters,
                        max_iterations=20_001,
                    ),
                ),
                "allocation_mode": replace(
                    base_spec,
                    parameters=replace(
                        parameters,
                        allocation_mode=bench.AllocationMode.COMPRESSED_I16,
                    ),
                ),
                "solver_identity": replace(
                    base_spec,
                    parameters=replace(
                        parameters,
                        solver_commit="a" * 40,
                    ),
                ),
            }
        )

        def prompt_for(spec):
            case = bench.OracleBenchmarkCase(
                "prompt-integrity",
                "constant description",
                spec,
            )
            result = fake_result(spec)
            return bench.build_prompt(
                bench.PreparedDecision(
                    case,
                    result,
                    result.combo_policies[0].private_combo,
                )
            )

        base_prompt = prompt_for(base_spec)
        changed_prompts = []
        for name, spec in variants.items():
            with self.subTest(input=name):
                prompt = prompt_for(spec)
                self.assertNotEqual(base_prompt, prompt)
                changed_prompts.append(prompt)
                if name == "descendant_actor_and_history":
                    self.assertIn(
                        "Decision node: DESCENDANT; acting player (Hero): IP.",
                        prompt,
                    )
                    self.assertIn(
                        'history from the tree root: ["CHECK"]',
                        prompt,
                    )
        self.assertEqual(len(changed_prompts), len(set(changed_prompts)))

    def test_uniform_limited_selection_is_reproducible_and_records_probability(self):
        cases = bench.demo_cases()
        results = {case.spec.cache_key: fake_result(case.spec) for case in cases}
        first = bench.prepare_decisions(cases, results, limit=3, seed=91)
        same = bench.prepare_decisions(cases, results, limit=3, seed=91)
        different = bench.prepare_decisions(cases, results, limit=3, seed=92)
        self.assertEqual(
            [decision.decision_id for decision in first],
            [decision.decision_id for decision in same],
        )
        self.assertNotEqual(
            [decision.decision_id for decision in first],
            [decision.decision_id for decision in different],
        )
        population_size = sum(
            len(result.combo_policies) for result in results.values()
        )
        expected_probability = Decimal(3) / Decimal(population_size)
        self.assertTrue(
            all(
                decision.selection_inclusion_probability
                == expected_probability
                for decision in first
            )
        )
        self.assertTrue(
            all(
                decision.selection_population_size == population_size
                for decision in first
            )
        )

    def test_limited_reach_estimator_is_unbiased_over_uniform_subsets(self):
        population = (
            (Decimal("0.8"), Decimal(0)),
            (Decimal("0.1"), Decimal(1)),
            (Decimal("0.1"), Decimal(1)),
        )
        inclusion_probability = Decimal(2) / Decimal(3)

        def row(weight, regret):
            return {
                "status": "SCORED",
                "reach_weight": str(weight),
                "selection_inclusion_probability": str(
                    inclusion_probability
                ),
                "selection_population_size": 3,
                "selection_population_reach_weight": "1",
                "ev_regret_pot_fraction": str(regret),
                "oracle_mass": "0.5",
                "cached": False,
                "latency_seconds": 0.1,
                "legal": True,
                "in_tree": True,
                "near_optimal": regret == 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }

        estimates = []
        for subset in itertools.combinations(population, 2):
            summary = bench.aggregate_results(
                [row(weight, regret) for weight, regret in subset]
            )
            self.assertEqual(
                "horvitz_thompson_known_population_reach",
                summary["reach_weighted_estimator"],
            )
            estimates.append(
                summary[
                    "reach_weighted_mean_ev_regret_pot_fraction"
                ]
            )
        self.assertAlmostEqual(0.2, sum(estimates) / len(estimates), places=12)


class ResponseAndScoringTests(unittest.TestCase):
    def setUp(self):
        self.case = bench.demo_cases()[1]
        self.result = fake_result(self.case.spec)
        self.decision = bench.PreparedDecision(
            self.case,
            self.result,
            self.result.combo_policies[0].private_combo,
        )

    def test_structured_schema_forces_one_exact_action_index(self):
        schema = bench.response_output_config(self.case.spec.tree.modeled_actions)
        action_schema = schema["format"]["schema"]["properties"]["action_index"]
        self.assertEqual([0, 1], action_schema["enum"])
        self.assertFalse(schema["format"]["schema"]["additionalProperties"])

    def test_parser_is_json_only_and_supports_auditable_direct_actions(self):
        actions = self.case.spec.tree.modeled_actions
        self.assertEqual(actions[1], bench.parse_model_action('{"action_index":1}', actions))
        self.assertEqual(
            Action(ActionKind.CHECK),
            bench.parse_model_action('{"action":"CHECK","amount":0}', actions),
        )
        with self.assertRaises(bench.ModelActionError):
            bench.parse_model_action("Action: CHECK", actions)
        with self.assertRaises(bench.ModelActionError):
            bench.parse_model_action('{"action_index":9}', actions)

    def test_scores_regret_and_distinguishes_illegal_from_out_of_tree(self):
        scored = bench.score_completion(
            self.decision,
            model="model",
            completion=bench.Completion('{"action_index":1}', 0.1),
            cached=False,
            ev_tolerance=Decimal(0),
        )
        self.assertEqual("SCORED", scored["status"])
        self.assertIsNotNone(scored["ev_regret_pot_fraction"])

        illegal = bench.score_completion(
            self.decision,
            model="model",
            completion=bench.Completion('{"action":"FOLD","amount":null}', 0.1),
            cached=False,
            ev_tolerance=Decimal(0),
        )
        self.assertEqual("ILLEGAL", illegal["status"])
        self.assertFalse(illegal["legal"])

        out_of_tree = bench.score_completion(
            self.decision,
            model="model",
            completion=bench.Completion('{"action":"BET","amount":25}', 0.1),
            cached=False,
            ev_tolerance=Decimal(0),
        )
        self.assertEqual("OUT_OF_TREE", out_of_tree["status"])
        self.assertTrue(out_of_tree["legal"])
        self.assertFalse(out_of_tree["in_tree"])

    def test_anthropic_adapter_uses_dynamic_json_schema_without_temperature(self):
        class Messages:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(
                    content=[SimpleNamespace(type="text", text='{"action_index":0}')],
                    usage=SimpleNamespace(input_tokens=10, output_tokens=4),
                    model="claude-sonnet-5-resolved",
                    stop_reason="end_turn",
                )

        completer = object.__new__(bench.AnthropicCompleter)
        completer.client = SimpleNamespace(messages=Messages())
        completion = completer.complete(
            "prompt",
            model="claude-sonnet-5",
            max_tokens=64,
            actions=self.case.spec.tree.modeled_actions,
        )
        request = completer.client.messages.calls[0]
        self.assertNotIn("temperature", request)
        self.assertEqual("json_schema", request["output_config"]["format"]["type"])
        self.assertNotIn("name", request["output_config"]["format"])
        self.assertEqual([0, 1], request["output_config"]["format"]["schema"]["properties"]["action_index"]["enum"])
        self.assertEqual({"type": "disabled"}, request["thinking"])
        self.assertEqual("claude-sonnet-5-resolved", completion.response_model)


class RunnerAndReportTests(unittest.TestCase):
    def test_solve_only_report_never_requires_model_completions(self):
        cases = bench.demo_cases()
        results = {
            case.spec.cache_key: fake_result(case.spec)
            for case in cases
        }
        cache_hits = {
            case.spec.cache_key: index % 2 == 0
            for index, case in enumerate(cases)
        }

        report = bench.build_oracle_validation_report(
            cases,
            results,
            cache_hits,
            engine_binary="/opt/gto-oracle-engine",
            engine_timeout_seconds=600.0,
            suite_name="demo",
        )

        self.assertTrue(report["run_complete"])
        self.assertEqual(len(cases), report["case_count"])
        self.assertEqual(sum(cache_hits.values()), report["cache_hits"])
        self.assertEqual("demo", report["suite"])
        self.assertEqual(
            {"RIVER": 2, "TURN": 1},
            report["coverage"]["streets"],
        )
        self.assertEqual("0.03", report["solver_elapsed_seconds_total"])
        self.assertEqual(
            {
                "estimated_uncompressed_bytes": None,
                "estimated_compressed_bytes": None,
                "hard_limit_bytes": None,
                "allocation_mode": None,
            },
            report["cases"][0]["memory"],
        )
        self.assertNotIn("models", report)
        self.assertNotIn("provider_calls_enabled", report)
        self.assertTrue(all(case["converged"] for case in report["cases"]))

    def test_changed_engine_binary_invalidates_oracle_cache(self):
        case = bench.demo_case()
        current = fake_result(case.spec)
        stale = SolveResult.for_spec(
            case.spec,
            current.combo_policies,
            SolverMetadata(
                solver_name=current.metadata.solver_name,
                solver_version=current.metadata.solver_version,
                iterations=current.metadata.iterations,
                elapsed_seconds=current.metadata.elapsed_seconds,
                exploitability=current.metadata.exploitability,
                converged=current.metadata.converged,
                extra=(("binary_sha256", "a" * 64),),
            ),
        )
        engine = FakeEngine()
        with tempfile.TemporaryDirectory() as directory:
            with OracleCache(Path(directory) / "oracle.sqlite3") as cache:
                cache.put(case.spec, stale)
                results, hits = bench.load_or_solve(
                    [case],
                    oracle_cache=cache,
                    engine=engine,
                )
                refreshed = cache.get(
                    case.spec,
                    expected_binary_sha256=FAKE_BINARY_SHA256,
                )

        self.assertEqual([case.spec.cache_key], engine.calls)
        self.assertFalse(hits[case.spec.cache_key])
        self.assertEqual(current, results[case.spec.cache_key])
        self.assertEqual(current, refreshed)

    def test_nonconverged_oracle_is_rejected_even_from_cache(self):
        case = bench.demo_case()
        converged = fake_result(case.spec)
        nonconverged = SolveResult.for_spec(
            case.spec,
            converged.combo_policies,
            SolverMetadata(
                solver_name=case.spec.parameters.solver_name,
                solver_version=case.spec.parameters.solver_commit,
                iterations=case.spec.parameters.max_iterations,
                elapsed_seconds=Decimal("0.1"),
                exploitability=Decimal("1"),
                converged=False,
                extra=(("binary_sha256", FAKE_BINARY_SHA256),),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            with OracleCache(Path(directory) / "oracle.sqlite3") as cache:
                cache.put(case.spec, nonconverged)
                with self.assertRaisesRegex(bench.BenchmarkError, "did not reach"):
                    bench.load_or_solve(
                        [case],
                        oracle_cache=cache,
                        engine=FakeEngine(),
                    )

    def test_oracle_and_model_caches_resume_and_models_are_paired(self):
        cases = bench.demo_cases()
        fake_engine = FakeEngine()
        fake_completer = FakeCompleter()
        models = [bench.DEFAULT_FAST_MODEL, bench.DEFAULT_COACH_MODEL]
        with tempfile.TemporaryDirectory() as directory:
            oracle_path = Path(directory) / "oracle.sqlite3"
            response_path = Path(directory) / "responses.jsonl"
            with OracleCache(oracle_path) as cache:
                results, first_hits = bench.load_or_solve(
                    cases,
                    oracle_cache=cache,
                    engine=fake_engine,
                )
            decisions = bench.prepare_decisions(cases, results, limit=4, seed=17)
            response_cache = bench.ModelResponseCache(response_path)
            rows = []
            for model in models:
                rows.extend(
                    bench.run_model(
                        decisions,
                        model=model,
                        completer=fake_completer,
                        response_cache=response_cache,
                        allow_provider_calls=True,
                    )
                )
            first_call_count = len(fake_completer.calls)
            for model in models:
                cached_rows = bench.run_model(
                    decisions,
                    model=model,
                    completer=fake_completer,
                    response_cache=bench.ModelResponseCache(response_path),
                    allow_provider_calls=False,
                )
                self.assertTrue(all(row["cached"] for row in cached_rows))
            with OracleCache(oracle_path) as cache:
                _, second_hits = bench.load_or_solve(
                    cases,
                    oracle_cache=cache,
                    engine=fake_engine,
                )

        self.assertEqual(len(cases), len(fake_engine.calls))
        self.assertTrue(all(not hit for hit in first_hits.values()))
        self.assertTrue(all(second_hits.values()))
        self.assertEqual(first_call_count, len(fake_completer.calls))
        self.assertEqual(len(decisions) * len(models), len(rows))
        report = bench.build_report(
            rows,
            models=models,
            cases=cases,
            decisions=decisions,
            run_config={"offline_confirmed": True},
            oracle_cache_hits=first_hits,
            oracle_results=results,
        )
        self.assertTrue(report["run_complete"])
        self.assertEqual(1, len(report["paired_comparisons"]))
        self.assertEqual(
            len(decisions),
            report["paired_comparisons"][0]["paired_scored_decisions"],
        )

    def test_cache_resume_rejects_changed_resolved_model_version(self):
        class ChangedResolvedModelCompleter:
            def __init__(self):
                self.calls = 0

            def complete(self, prompt, *, model, max_tokens, actions):
                self.calls += 1
                return bench.Completion(
                    '{"action_index":0}',
                    0.1,
                    response_model="snapshot-b",
                )

        case = bench.demo_case()
        result = fake_result(case.spec)
        decisions = bench.prepare_decisions(
            [case],
            {case.spec.cache_key: result},
        )
        model = "rolling-alias"
        first_key = bench.completion_cache_key(
            decisions[0],
            model=model,
            max_tokens=64,
        )
        second_key = bench.completion_cache_key(
            decisions[1],
            model=model,
            max_tokens=64,
        )
        completer = ChangedResolvedModelCompleter()
        with tempfile.TemporaryDirectory() as directory:
            cache = bench.ModelResponseCache(
                Path(directory) / "responses.jsonl"
            )
            cache.put(
                first_key,
                bench.Completion(
                    '{"action_index":0}',
                    0.1,
                    response_model="snapshot-a",
                ),
            )
            with self.assertRaisesRegex(
                bench.BenchmarkError,
                "snapshot-b.*pinned to 'snapshot-a'",
            ):
                bench.run_model(
                    decisions,
                    model=model,
                    completer=completer,
                    response_cache=cache,
                    allow_provider_calls=True,
                )
            self.assertIsNone(cache.get(second_key))
        self.assertEqual(1, completer.calls)

    def test_provider_calls_are_disabled_by_default_path(self):
        case = bench.demo_case()
        result = fake_result(case.spec)
        decisions = bench.prepare_decisions(
            [case], {case.spec.cache_key: result}, limit=1
        )
        with tempfile.TemporaryDirectory() as directory:
            rows = bench.run_model(
                decisions,
                model=bench.DEFAULT_FAST_MODEL,
                completer=None,
                response_cache=bench.ModelResponseCache(
                    Path(directory) / "responses.jsonl"
                ),
                allow_provider_calls=False,
            )
        self.assertEqual("CACHE_MISS", rows[0]["status"])


if __name__ == "__main__":
    unittest.main()
