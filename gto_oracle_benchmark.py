#!/usr/bin/env python3
"""Offline, post-session model evaluation against the local GTO oracle.

This module is intentionally independent from ``poker_assistant.py``.  It has
no screen capture, vision, keyboard hook, or live-table integration.  A run is
accepted only after the caller explicitly confirms that the poker client is
closed.  Network/model calls require a second, separate ``--call-models`` flag.

The benchmark is heads-up postflop only because that is the exact scope of the
underlying oracle.  It evaluates a model's *selected action* for each private
combo using counterfactual EV regret and the equilibrium probability assigned
to that action.  It does not train a model and it is not a GTO certificate.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time
from typing import Any, Callable, Iterable, Sequence

from dotenv import load_dotenv

from gto_oracle import (
    Action,
    ActionKind,
    AllocationMode,
    AssessmentStatus,
    BetSizingConfig,
    EngineClient,
    OracleCache,
    OracleValidationError,
    PlayerBetSizes,
    PlayerRange,
    Position,
    ScoredAction,
    SolveParameters,
    SolveResult,
    SolveSpec,
    Street,
    StreetBetSizes,
    TreeConfig,
    WeightedCombo,
    assess_action,
    canonical_json,
)


BENCHMARK_SCHEMA_VERSION = 2
CASE_FILE_SCHEMA_VERSION = 1
RESPONSE_CACHE_SCHEMA_VERSION = 1
PROMPT_VERSION = "gto-oracle-action-v2"
DEFAULT_FAST_MODEL = "claude-haiku-4-5"
DEFAULT_COACH_MODEL = "claude-sonnet-5"
DEFAULT_ENGINE = Path(
    "/private/tmp/oracle-engine-target/release/gto-oracle-engine"
)
DEFAULT_OUTPUT_DIR = Path("benchmark_results/gto_oracle")
DEFAULT_RESPONSE_CACHE = DEFAULT_OUTPUT_DIR / "model_responses.jsonl"
DEFAULT_ORACLE_CACHE = DEFAULT_OUTPUT_DIR / "oracle.sqlite3"
PINNED_SOLVER_COMMIT = "9d1509fe5077d019825f833eed04b16d342dfda1"

SYSTEM_PROMPT = (
    "You are evaluating an already completed heads-up no-limit Hold'em hand "
    "offline. Choose the single listed action with the highest expected value "
    "for the exact private combo and supplied ranges. Never invent a size. "
    "Return only the requested JSON object."
)


class BenchmarkError(ValueError):
    """A benchmark case, model response, or local artifact is invalid."""


class ModelActionError(BenchmarkError):
    """A model response is not one unambiguous JSON poker action."""


@dataclass(frozen=True, slots=True)
class OracleBenchmarkCase:
    """One transparent HU postflop solve specification."""

    case_id: str
    description: str
    spec: SolveSpec

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", self.case_id
        ):
            raise BenchmarkError(
                "case_id must be 1-128 safe ASCII identifier characters"
            )
        if not isinstance(self.description, str):
            raise BenchmarkError("case description must be a string")
        if not isinstance(self.spec, SolveSpec):
            raise BenchmarkError("case spec must be a SolveSpec")


@dataclass(frozen=True, slots=True)
class PreparedDecision:
    """One exact private-combo decision backed by a solved policy."""

    source_case: OracleBenchmarkCase
    oracle_result: SolveResult
    private_combo: tuple[str, str]
    selection_inclusion_probability: Decimal = Decimal(1)
    selection_population_size: int = 1
    selection_population_reach_weight: Decimal | None = None

    def __post_init__(self) -> None:
        probability = self.selection_inclusion_probability
        population_reach = self.selection_population_reach_weight
        if population_reach is None:
            population_reach = self.policy.reach_weight
            object.__setattr__(
                self,
                "selection_population_reach_weight",
                population_reach,
            )
        if (
            not isinstance(probability, Decimal)
            or not probability.is_finite()
            or not Decimal(0) < probability <= Decimal(1)
        ):
            raise BenchmarkError(
                "selection inclusion probability must be a Decimal in (0, 1]"
            )
        if (
            isinstance(self.selection_population_size, bool)
            or not isinstance(self.selection_population_size, int)
            or self.selection_population_size <= 0
        ):
            raise BenchmarkError("selection population size must be positive")
        if (
            not isinstance(population_reach, Decimal)
            or not population_reach.is_finite()
            or population_reach <= 0
        ):
            raise BenchmarkError(
                "selection population reach weight must be a positive Decimal"
            )

    @property
    def decision_id(self) -> str:
        payload = {
            "case_id": self.source_case.case_id,
            "spec_key": self.source_case.spec.cache_key,
            "private_combo": self.private_combo,
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    @property
    def policy(self):
        for policy in self.oracle_result.combo_policies:
            if policy.private_combo == self.private_combo:
                return policy
        raise BenchmarkError("prepared private combo is absent from oracle result")


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    latency_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    response_model: str = ""
    stop_reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise BenchmarkError("completion text must be a string")
        if (
            isinstance(self.latency_seconds, bool)
            or not isinstance(self.latency_seconds, (int, float))
            or not math.isfinite(float(self.latency_seconds))
            or self.latency_seconds < 0
        ):
            raise BenchmarkError("completion latency must be finite and non-negative")
        for field_name, value in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise BenchmarkError(f"completion {field_name} must be a non-negative integer")
        if not isinstance(self.response_model, str) or not isinstance(
            self.stop_reason, str
        ):
            raise BenchmarkError("completion provider metadata must be strings")


def _exact_keys(value: dict[str, Any], expected: set[str], field: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise BenchmarkError(f"{field} schema mismatch: {'; '.join(details)}")


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkError(f"{field} must be an object")
    return value


def _array(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise BenchmarkError(f"{field} must be an array")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise BenchmarkError(f"{field} must be a string")
    return value


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkError(f"{field} must be an integer")
    return value


def _decimal(value: Any, field: str) -> Decimal:
    if isinstance(value, bool) or isinstance(value, float):
        raise BenchmarkError(f"{field} must be an exact JSON number or string")
    try:
        result = value if isinstance(value, Decimal) else Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as error:
        raise BenchmarkError(f"{field} must be a decimal number") from error
    if not result.is_finite():
        raise BenchmarkError(f"{field} must be finite")
    return result


def _strict_json_loads(payload: str, *, field: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise BenchmarkError(f"{field} repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        def reject_constant(value: str):
            raise BenchmarkError(
                f"{field} contains forbidden non-finite JSON constant {value!r}"
            )

        return json.loads(
            payload,
            parse_float=Decimal,
            parse_int=int,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except BenchmarkError:
        raise
    except (json.JSONDecodeError, InvalidOperation, ValueError) as error:
        raise BenchmarkError(f"{field} is not strict JSON: {error}") from error


def _action_from_data(value: Any, field: str) -> Action:
    body = _object(value, field)
    _exact_keys(body, {"kind", "amount"}, field)
    kind_text = _string(body["kind"], f"{field}.kind").upper()
    try:
        kind = ActionKind(kind_text)
    except ValueError as error:
        raise BenchmarkError(f"{field}.kind is unsupported") from error
    amount = body["amount"]
    if amount is not None:
        amount = _integer(amount, f"{field}.amount")
    try:
        return Action(kind, amount)
    except OracleValidationError as error:
        raise BenchmarkError(f"{field} is invalid: {error}") from error


def _weighted_combo_from_data(value: Any, field: str) -> WeightedCombo:
    body = _object(value, field)
    _exact_keys(body, {"cards", "weight"}, field)
    cards = _array(body["cards"], f"{field}.cards")
    if len(cards) != 2:
        raise BenchmarkError(f"{field}.cards must contain two cards")
    return WeightedCombo(
        tuple(_string(card, f"{field}.cards") for card in cards),
        _decimal(body["weight"], f"{field}.weight"),
    )


def _range_from_data(value: Any, field: str) -> PlayerRange:
    body = _object(value, field)
    _exact_keys(body, {"position", "combos"}, field)
    try:
        position = Position(_string(body["position"], f"{field}.position").upper())
    except ValueError as error:
        raise BenchmarkError(f"{field}.position must be OOP or IP") from error
    combos = tuple(
        _weighted_combo_from_data(item, f"{field}.combos[{index}]")
        for index, item in enumerate(_array(body["combos"], f"{field}.combos"))
    )
    return PlayerRange(position, combos)


def _player_sizes_from_data(value: Any, field: str) -> PlayerBetSizes:
    body = _object(value, field)
    _exact_keys(body, {"bet", "raise_sizes"}, field)
    return PlayerBetSizes(
        _string(body["bet"], f"{field}.bet"),
        _string(body["raise_sizes"], f"{field}.raise_sizes"),
    )


def _street_sizes_from_data(value: Any, field: str) -> StreetBetSizes:
    body = _object(value, field)
    _exact_keys(body, {"oop", "ip"}, field)
    return StreetBetSizes(
        _player_sizes_from_data(body["oop"], f"{field}.oop"),
        _player_sizes_from_data(body["ip"], f"{field}.ip"),
    )


def solve_spec_from_data(value: Any) -> SolveSpec:
    """Parse the complete canonical SolveSpec JSON representation."""

    body = _object(value, "spec")
    _exact_keys(
        body,
        {
            "street",
            "board",
            "acting_player",
            "oop_range",
            "ip_range",
            "tree",
            "parameters",
            "players",
            "variant",
        },
        "spec",
    )
    tree = _object(body["tree"], "spec.tree")
    _exact_keys(
        tree,
        {
            "pot",
            "effective_stack",
            "facing_bet",
            "legal_action_kinds",
            "modeled_actions",
            "action_history",
            "rake_rate_pct",
            "rake_cap",
        },
        "spec.tree",
    )
    parameters = _object(body["parameters"], "spec.parameters")
    _exact_keys(
        parameters,
        {
            "chip_scale",
            "chip_unit",
            "bet_sizes",
            "add_allin_threshold",
            "force_allin_threshold",
            "merging_threshold",
            "target_exploitability_pct",
            "max_iterations",
            "allocation_mode",
            "solver_name",
            "solver_commit",
        },
        "spec.parameters",
    )
    sizes = _object(parameters["bet_sizes"], "spec.parameters.bet_sizes")
    _exact_keys(
        sizes,
        {
            "flop",
            "turn",
            "river",
            "flop_donk_sizes",
            "turn_donk_sizes",
            "river_donk_sizes",
        },
        "spec.parameters.bet_sizes",
    )

    def optional_text(item: Any, field: str) -> str | None:
        return None if item is None else _string(item, field)

    try:
        return SolveSpec(
            street=Street(_string(body["street"], "spec.street").upper()),
            board=tuple(
                _string(card, "spec.board card")
                for card in _array(body["board"], "spec.board")
            ),
            acting_player=Position(
                _string(body["acting_player"], "spec.acting_player").upper()
            ),
            oop_range=_range_from_data(body["oop_range"], "spec.oop_range"),
            ip_range=_range_from_data(body["ip_range"], "spec.ip_range"),
            tree=TreeConfig(
                pot=_integer(tree["pot"], "spec.tree.pot"),
                effective_stack=_integer(
                    tree["effective_stack"], "spec.tree.effective_stack"
                ),
                facing_bet=_integer(tree["facing_bet"], "spec.tree.facing_bet"),
                legal_action_kinds=tuple(
                    ActionKind(_string(item, "legal action kind").upper())
                    for item in _array(
                        tree["legal_action_kinds"],
                        "spec.tree.legal_action_kinds",
                    )
                ),
                modeled_actions=tuple(
                    _action_from_data(item, f"spec.tree.modeled_actions[{index}]")
                    for index, item in enumerate(
                        _array(tree["modeled_actions"], "spec.tree.modeled_actions")
                    )
                ),
                action_history=tuple(
                    _action_from_data(item, f"spec.tree.action_history[{index}]")
                    for index, item in enumerate(
                        _array(tree["action_history"], "spec.tree.action_history")
                    )
                ),
                rake_rate_pct=_decimal(
                    tree["rake_rate_pct"], "spec.tree.rake_rate_pct"
                ),
                rake_cap=_integer(tree["rake_cap"], "spec.tree.rake_cap"),
            ),
            parameters=SolveParameters(
                chip_scale=_integer(
                    parameters["chip_scale"], "spec.parameters.chip_scale"
                ),
                chip_unit=_string(
                    parameters["chip_unit"], "spec.parameters.chip_unit"
                ),
                bet_sizes=BetSizingConfig(
                    flop=_street_sizes_from_data(
                        sizes["flop"], "spec.parameters.bet_sizes.flop"
                    ),
                    turn=_street_sizes_from_data(
                        sizes["turn"], "spec.parameters.bet_sizes.turn"
                    ),
                    river=_street_sizes_from_data(
                        sizes["river"], "spec.parameters.bet_sizes.river"
                    ),
                    flop_donk_sizes=optional_text(
                        sizes["flop_donk_sizes"],
                        "spec.parameters.bet_sizes.flop_donk_sizes",
                    ),
                    turn_donk_sizes=optional_text(
                        sizes["turn_donk_sizes"],
                        "spec.parameters.bet_sizes.turn_donk_sizes",
                    ),
                    river_donk_sizes=optional_text(
                        sizes["river_donk_sizes"],
                        "spec.parameters.bet_sizes.river_donk_sizes",
                    ),
                ),
                add_allin_threshold=_decimal(
                    parameters["add_allin_threshold"],
                    "spec.parameters.add_allin_threshold",
                ),
                force_allin_threshold=_decimal(
                    parameters["force_allin_threshold"],
                    "spec.parameters.force_allin_threshold",
                ),
                merging_threshold=_decimal(
                    parameters["merging_threshold"],
                    "spec.parameters.merging_threshold",
                ),
                target_exploitability_pct=_decimal(
                    parameters["target_exploitability_pct"],
                    "spec.parameters.target_exploitability_pct",
                ),
                max_iterations=_integer(
                    parameters["max_iterations"],
                    "spec.parameters.max_iterations",
                ),
                allocation_mode=AllocationMode(
                    _string(
                        parameters["allocation_mode"],
                        "spec.parameters.allocation_mode",
                    )
                ),
                solver_name=_string(
                    parameters["solver_name"], "spec.parameters.solver_name"
                ),
                solver_commit=_string(
                    parameters["solver_commit"], "spec.parameters.solver_commit"
                ),
            ),
            players=tuple(
                Position(_string(item, "spec.players item").upper())
                for item in _array(body["players"], "spec.players")
            ),
            variant=_string(body["variant"], "spec.variant"),
        )
    except (OracleValidationError, ValueError) as error:
        raise BenchmarkError(f"invalid SolveSpec: {error}") from error


def load_case_file(path: Path) -> list[OracleBenchmarkCase]:
    """Load one strict, auditable case file without schema defaults."""

    try:
        payload = path.read_text(encoding="utf-8")
    except OSError as error:
        raise BenchmarkError(f"cannot read case file {path}: {error}") from error
    root = _object(_strict_json_loads(payload, field="case file"), "case file")
    _exact_keys(root, {"schema_version", "usage", "cases"}, "case file")
    if _integer(root["schema_version"], "case file.schema_version") != (
        CASE_FILE_SCHEMA_VERSION
    ):
        raise BenchmarkError("unsupported case file schema_version")
    if _string(root["usage"], "case file.usage") != "offline_post_session_only":
        raise BenchmarkError("case file usage must be offline_post_session_only")
    cases = []
    seen_ids: set[str] = set()
    for index, raw_case in enumerate(_array(root["cases"], "case file.cases")):
        body = _object(raw_case, f"case file.cases[{index}]")
        _exact_keys(body, {"case_id", "description", "spec"}, f"case[{index}]")
        case = OracleBenchmarkCase(
            case_id=_string(body["case_id"], f"case[{index}].case_id"),
            description=_string(
                body["description"], f"case[{index}].description"
            ),
            spec=solve_spec_from_data(body["spec"]),
        )
        if case.case_id in seen_ids:
            raise BenchmarkError(f"duplicate case_id {case.case_id!r}")
        seen_ids.add(case.case_id)
        cases.append(case)
    if not cases:
        raise BenchmarkError("case file must contain at least one case")
    return cases


def demo_case() -> OracleBenchmarkCase:
    """Return the tiny Brown value/bluff river game used by engine tests."""

    half_pot = PlayerBetSizes("50%", "2.5x")
    parameters = SolveParameters(
        chip_scale=1,
        chip_unit="chip",
        bet_sizes=BetSizingConfig(
            flop=StreetBetSizes(half_pot, half_pot),
            turn=StreetBetSizes(half_pot, half_pot),
            river=StreetBetSizes(half_pot, half_pot),
            flop_donk_sizes=None,
            turn_donk_sizes=None,
            river_donk_sizes=None,
        ),
        add_allin_threshold=Decimal("1.5"),
        force_allin_threshold=Decimal("0.15"),
        merging_threshold=Decimal("0.1"),
        target_exploitability_pct=Decimal("0.001"),
        max_iterations=20_000,
        allocation_mode=AllocationMode.UNCOMPRESSED_F32,
        solver_name="b-inary/postflop-solver",
        solver_commit=PINNED_SOLVER_COMMIT,
    )
    spec = SolveSpec(
        street=Street.RIVER,
        board=("2s", "3h", "4d", "6c", "7c"),
        acting_player=Position.OOP,
        oop_range=PlayerRange(
            Position.OOP,
            (
                WeightedCombo(("As", "Ah"), Decimal(1)),
                WeightedCombo(("Qs", "Qh"), Decimal(1)),
            ),
        ),
        ip_range=PlayerRange(
            Position.IP,
            (WeightedCombo(("Ks", "Kh"), Decimal(1)),),
        ),
        tree=TreeConfig(
            pot=20,
            effective_stack=10,
            facing_bet=0,
            legal_action_kinds=(ActionKind.CHECK, ActionKind.ALL_IN),
            modeled_actions=(
                Action(ActionKind.CHECK),
                Action(ActionKind.ALL_IN, 10),
            ),
        ),
        parameters=parameters,
    )
    return OracleBenchmarkCase(
        case_id="brown-river-value-bluff",
        description=(
            "Tiny deterministic river smoke case: OOP holds AA or QQ versus "
            "IP's KK, with one half-pot all-in size."
        ),
        spec=spec,
    )


def demo_cases() -> list[OracleBenchmarkCase]:
    """Return a deterministic river+turn validation suite.

    This remains much smaller than a representative corpus.  It combines the
    analytic two-combo fixture, a weighted turn node, and a broader river node
    with two bet sizes so pure, mixed, and sizing decisions are all exercised.
    """

    river = demo_case()
    half_pot = PlayerBetSizes("50%", "2.5x")
    parameters = SolveParameters(
        chip_scale=100,
        chip_unit="centi-BB",
        bet_sizes=BetSizingConfig(
            flop=StreetBetSizes(half_pot, half_pot),
            turn=StreetBetSizes(half_pot, half_pot),
            river=StreetBetSizes(half_pot, half_pot),
            flop_donk_sizes=None,
            turn_donk_sizes=None,
            river_donk_sizes=None,
        ),
        add_allin_threshold=Decimal("1.5"),
        force_allin_threshold=Decimal("0.15"),
        merging_threshold=Decimal("0.1"),
        target_exploitability_pct=Decimal("0.01"),
        max_iterations=20000,
        allocation_mode=AllocationMode.UNCOMPRESSED_F32,
        solver_name="b-inary/postflop-solver",
        solver_commit=PINNED_SOLVER_COMMIT,
    )
    turn_spec = SolveSpec(
        street=Street.TURN,
        board=("Td", "9d", "6h", "Qc"),
        acting_player=Position.OOP,
        oop_range=PlayerRange(
            Position.OOP,
            (
                WeightedCombo(("Ad", "Kd"), Decimal("0.75")),
                WeightedCombo(("Ah", "Kh"), Decimal(1)),
                WeightedCombo(("Js", "Jh"), Decimal("0.8")),
                WeightedCombo(("8s", "7s"), Decimal("0.6")),
                WeightedCombo(("5d", "4d"), Decimal("0.4")),
            ),
        ),
        ip_range=PlayerRange(
            Position.IP,
            (
                WeightedCombo(("As", "Ks"), Decimal(1)),
                WeightedCombo(("Ac", "Kc"), Decimal("0.8")),
                WeightedCombo(("Jd", "Jc"), Decimal("0.75")),
                WeightedCombo(("8h", "7h"), Decimal("0.6")),
                WeightedCombo(("5s", "4s"), Decimal("0.4")),
            ),
        ),
        tree=TreeConfig(
            pot=100,
            effective_stack=400,
            facing_bet=0,
            legal_action_kinds=(ActionKind.CHECK, ActionKind.BET),
            modeled_actions=(
                Action(ActionKind.CHECK),
                Action(ActionKind.BET, 50),
            ),
        ),
        parameters=parameters,
    )
    turn = OracleBenchmarkCase(
        case_id="turn-single-size-explicit-ranges",
        description=(
            "Turn root check/bet decision with five explicit weighted combos "
            "per player and one 50%-pot bet size."
        ),
        spec=turn_spec,
    )

    two_sizes = PlayerBetSizes("33%, 75%", "2.5x")
    river_parameters = SolveParameters(
        chip_scale=100,
        chip_unit="centi-BB",
        bet_sizes=BetSizingConfig(
            flop=StreetBetSizes(two_sizes, two_sizes),
            turn=StreetBetSizes(two_sizes, two_sizes),
            river=StreetBetSizes(two_sizes, two_sizes),
            flop_donk_sizes=None,
            turn_donk_sizes=None,
            river_donk_sizes=None,
        ),
        add_allin_threshold=Decimal("1.5"),
        force_allin_threshold=Decimal("0.15"),
        merging_threshold=Decimal("0.1"),
        target_exploitability_pct=Decimal("0.01"),
        max_iterations=20000,
        allocation_mode=AllocationMode.UNCOMPRESSED_F32,
        solver_name="b-inary/postflop-solver",
        solver_commit=PINNED_SOLVER_COMMIT,
    )

    def explicit_range(position: Position, hands: str) -> PlayerRange:
        return PlayerRange(
            position,
            tuple(
                WeightedCombo((hand[:2], hand[2:]), Decimal(1))
                for hand in hands.split()
            ),
        )

    medium_river_spec = SolveSpec(
        street=Street.RIVER,
        board=("Td", "9d", "6h", "Qc", "2s"),
        acting_player=Position.OOP,
        oop_range=explicit_range(
            Position.OOP,
            (
                "QhQd QsQd AdQd QsQh AhQh AsQs KdKc KhKc KsKc AcKc "
                "KhKd KsKd AdKd KsKh AhKh AsKs AdAc AhAc AsAc AhAd AsAd AsAh"
            ),
        ),
        ip_range=explicit_range(
            Position.IP,
            (
                "9h9c 9s9c 9s9h ThTc TsTc AcTc TsTh AhTh AsTs JdJc JhJc "
                "JsJc AcJc JhJd JsJd AdJd JsJh AhJh AsJs KdQd KhQh KsQs"
            ),
        ),
        tree=TreeConfig(
            pot=100,
            effective_stack=400,
            facing_bet=0,
            legal_action_kinds=(ActionKind.CHECK, ActionKind.BET),
            modeled_actions=(
                Action(ActionKind.CHECK),
                Action(ActionKind.BET, 33),
                Action(ActionKind.BET, 75),
            ),
        ),
        parameters=river_parameters,
    )
    medium_river = OracleBenchmarkCase(
        case_id="river-two-size-explicit-ranges",
        description=(
            "River root check/33%/75% decision with 22 explicit combos per "
            "player at SPR 4."
        ),
        spec=medium_river_spec,
    )
    return [river, turn, medium_river]


def case_file_data(cases: Sequence[OracleBenchmarkCase]) -> dict[str, Any]:
    return {
        "schema_version": CASE_FILE_SCHEMA_VERSION,
        "usage": "offline_post_session_only",
        "cases": [
            {
                "case_id": case.case_id,
                "description": case.description,
                "spec": _strict_json_loads(case.spec.canonical_json, field="spec"),
            }
            for case in cases
        ],
    }


def write_case_file(path: Path, cases: Sequence[OracleBenchmarkCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            case_file_data(cases),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def action_data(action: Action) -> dict[str, Any]:
    return {"kind": action.kind.value, "amount": action.amount}


def _action_label(action: Action) -> str:
    if action.amount is None:
        return action.kind.value
    if action.kind is ActionKind.BET:
        return f"BET_TO {action.amount}"
    if action.kind is ActionKind.RAISE:
        return f"RAISE_TO {action.amount}"
    return f"ALL_IN_TO {action.amount}"


def _range_text(player_range: PlayerRange) -> str:
    return ", ".join(
        f"{combo.cards[0]}{combo.cards[1]}@{combo.weight}"
        for combo in player_range.combos
    )


def _prompt_strategy_context(spec: SolveSpec) -> dict[str, Any]:
    """Return every solver input that defines the game tree or extracted node."""

    sizing = spec.parameters.bet_sizes

    def player_sizes(value: PlayerBetSizes) -> dict[str, str]:
        return {
            "bet": value.bet,
            "raise_sizes": value.raise_sizes,
        }

    def street_sizes(value: StreetBetSizes) -> dict[str, dict[str, str]]:
        return {
            "oop": player_sizes(value.oop),
            "ip": player_sizes(value.ip),
        }

    return {
        "game": {
            "variant": spec.variant,
            "players": [player.value for player in spec.players],
            "street": spec.street.value,
            "board": list(spec.board),
        },
        "node": {
            "type": "root" if not spec.tree.action_history else "descendant",
            "acting_player": spec.acting_player.value,
            "action_history": [
                action_data(action) for action in spec.tree.action_history
            ],
            "facing_bet": spec.tree.facing_bet,
            "legal_action_kinds": [
                kind.value for kind in spec.tree.legal_action_kinds
            ],
            "modeled_actions": [
                action_data(action) for action in spec.tree.modeled_actions
            ],
        },
        "tree": {
            "starting_pot": spec.tree.pot,
            "effective_stack": spec.tree.effective_stack,
            "rake": {
                "rate_pct": spec.tree.rake_rate_pct,
                "cap": spec.tree.rake_cap,
            },
            "bet_sizes": {
                "flop": street_sizes(sizing.flop),
                "turn": street_sizes(sizing.turn),
                "river": street_sizes(sizing.river),
                "flop_donk_sizes": sizing.flop_donk_sizes,
                "turn_donk_sizes": sizing.turn_donk_sizes,
                "river_donk_sizes": sizing.river_donk_sizes,
            },
            "options": {
                "add_allin_threshold": spec.parameters.add_allin_threshold,
                "force_allin_threshold": spec.parameters.force_allin_threshold,
                "merging_threshold": spec.parameters.merging_threshold,
            },
        },
        "oracle_numeric_contract": {
            "target_exploitability_pct": (
                spec.parameters.target_exploitability_pct
            ),
            "max_iterations": spec.parameters.max_iterations,
            "allocation_mode": spec.parameters.allocation_mode.value,
            "solver_name": spec.parameters.solver_name,
            "solver_commit": spec.parameters.solver_commit,
        },
    }


def build_prompt(decision: PreparedDecision) -> str:
    """Build a policy-blind prompt: ranges and tree are shown, oracle EVs are not."""

    case = decision.source_case
    spec = case.spec
    actions = list(spec.tree.modeled_actions)
    menu = [
        {
            "action_index": index,
            "action": _action_label(action),
        }
        for index, action in enumerate(actions)
    ]
    board = " ".join(spec.board)
    combo = " ".join(decision.private_combo)
    node_type = "ROOT" if not spec.tree.action_history else "DESCENDANT"
    history = [
        _action_label(action) for action in spec.tree.action_history
    ]
    strategy_context = canonical_json(_prompt_strategy_context(spec))
    return (
        "OFFLINE POST-SESSION EVALUATION (the hand is already over).\n"
        f"Case: {case.case_id} — {case.description}\n"
        "Game: heads-up NLHE postflop.\n"
        f"Decision node: {node_type}; acting player (Hero): "
        f"{spec.acting_player.value}.\n"
        "Ordered same-street action history from the tree root: "
        f"{json.dumps(history, separators=(',', ':'))}\n"
        f"Street: {spec.street.value}\n"
        f"Board: {board}\n"
        f"Hero exact private combo: {combo}\n"
        f"Tree starting pot: {spec.tree.pot} engine units\n"
        f"Effective stack: {spec.tree.effective_stack} engine units\n"
        f"Current facing bet: {spec.tree.facing_bet} engine units\n"
        f"Unit audit label: {spec.parameters.chip_unit}; "
        f"chip_scale={spec.parameters.chip_scale}\n"
        f"Rake: {spec.tree.rake_rate_pct}% capped at {spec.tree.rake_cap} units\n"
        f"Exact OOP range (combo@weight): {_range_text(spec.oop_range)}\n"
        f"Exact IP range (combo@weight): {_range_text(spec.ip_range)}\n"
        "Exact strategy-defining tree and node inputs (JSON):\n"
        f"{strategy_context}\n"
        "The only modeled legal actions at this decision node are:\n"
        f"{json.dumps(menu, separators=(',', ':'))}\n"
        "Amounts are final BET_TO/RAISE_TO/ALL_IN_TO contributions, not chips added.\n"
        "Select exactly one listed action. Return only "
        '{"action_index":<integer>}.'
    )


def response_output_config(actions: Sequence[Action]) -> dict[str, Any]:
    if not actions:
        raise BenchmarkError("cannot build output schema without actions")
    return {
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "action_index": {
                        "type": "integer",
                        "enum": list(range(len(actions))),
                    }
                },
                "required": ["action_index"],
                "additionalProperties": False,
            },
        }
    }


def parse_model_action(text: str, modeled_actions: Sequence[Action]) -> Action:
    """Parse strict JSON, with a direct-action fallback for audit/cache fixtures."""

    if not isinstance(text, str) or not text.strip():
        raise ModelActionError("model response is empty")
    try:
        body = _object(_strict_json_loads(text, field="model response"), "model response")
    except BenchmarkError as error:
        raise ModelActionError(str(error)) from error
    if set(body) == {"action_index"}:
        try:
            index = _integer(body["action_index"], "action_index")
        except BenchmarkError as error:
            raise ModelActionError(str(error)) from error
        if not 0 <= index < len(modeled_actions):
            raise ModelActionError("action_index is outside the supplied action menu")
        return modeled_actions[index]

    # This branch makes old/manual cached JSON auditable and lets the scorer
    # distinguish poker-illegal decisions from legal sizes omitted by the tree.
    if set(body) in ({"action", "amount"}, {"kind", "amount"}):
        key = "action" if "action" in body else "kind"
        raw_kind = body[key]
        if not isinstance(raw_kind, str):
            raise ModelActionError(f"{key} must be a string")
        try:
            kind = ActionKind(raw_kind.upper())
        except ValueError as error:
            raise ModelActionError(f"unsupported action {raw_kind!r}") from error
        amount = body["amount"]
        if amount == 0 and kind not in {
            ActionKind.BET,
            ActionKind.RAISE,
            ActionKind.ALL_IN,
        }:
            amount = None
        if amount is not None and (isinstance(amount, bool) or not isinstance(amount, int)):
            raise ModelActionError("amount must be an integer or null")
        try:
            return Action(kind, amount)
        except OracleValidationError as error:
            raise ModelActionError(str(error)) from error
    raise ModelActionError(
        "model JSON must contain only action_index, or action/kind plus amount"
    )


def completion_cache_key(
    decision: PreparedDecision,
    *,
    model: str,
    max_tokens: int,
) -> str:
    prompt = build_prompt(decision)
    payload = {
        "schema_version": RESPONSE_CACHE_SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "decision_id": decision.decision_id,
        "spec_key": decision.source_case.spec.cache_key,
        "model": model,
        "max_tokens": max_tokens,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(
            SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "output_schema": response_output_config(
            decision.source_case.spec.tree.modeled_actions
        ),
        "actions": [
            action_data(action)
            for action in decision.source_case.spec.tree.modeled_actions
        ],
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class ModelResponseCache:
    """Append-only JSONL completion cache for interruption-safe paired runs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: dict[str, Completion] = {}
        self.corrupt_lines = 0
        self.stale_lines = 0
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        entry = _strict_json_loads(line, field="response cache line")
                        if not isinstance(entry, dict):
                            raise BenchmarkError("cache line must be an object")
                        if entry.get("schema_version") != RESPONSE_CACHE_SCHEMA_VERSION:
                            self.stale_lines += 1
                            continue
                        _exact_keys(
                            entry,
                            {"schema_version", "key", "completion"},
                            "response cache line",
                        )
                        key = _string(entry["key"], "response cache key")
                        completion = _object(
                            entry["completion"], "response cache completion"
                        )
                        _exact_keys(
                            completion,
                            {
                                "text",
                                "latency_seconds",
                                "input_tokens",
                                "output_tokens",
                                "response_model",
                                "stop_reason",
                            },
                            "response cache completion",
                        )
                        latency = completion["latency_seconds"]
                        if isinstance(latency, bool) or not isinstance(
                            latency, (int, Decimal)
                        ):
                            raise BenchmarkError("cached latency must be numeric")
                        self.entries[key] = Completion(
                            text=_string(completion["text"], "cached text"),
                            latency_seconds=float(latency),
                            input_tokens=_integer(
                                completion["input_tokens"], "cached input_tokens"
                            ),
                            output_tokens=_integer(
                                completion["output_tokens"], "cached output_tokens"
                            ),
                            response_model=_string(
                                completion["response_model"], "cached response_model"
                            ),
                            stop_reason=_string(
                                completion["stop_reason"], "cached stop_reason"
                            ),
                        )
                    except (BenchmarkError, OSError, ValueError):
                        self.corrupt_lines += 1

    def get(self, key: str) -> Completion | None:
        return self.entries.get(key)

    def put(self, key: str, completion: Completion) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise BenchmarkError("response cache key must be a SHA-256 digest")
        payload = {
            "schema_version": RESPONSE_CACHE_SCHEMA_VERSION,
            "key": key,
            "completion": {
                "text": completion.text,
                "latency_seconds": completion.latency_seconds,
                "input_tokens": completion.input_tokens,
                "output_tokens": completion.output_tokens,
                "response_model": completion.response_model,
                "stop_reason": completion.stop_reason,
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, allow_nan=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.entries[key] = completion


class AnthropicCompleter:
    """Minimal structured-output adapter; constructed only with --call-models."""

    def __init__(self, api_key: str, *, timeout: float, max_retries: int) -> None:
        from anthropic import Anthropic

        self.client = Anthropic(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    def complete(
        self,
        prompt: str,
        *,
        model: str,
        max_tokens: int,
        actions: Sequence[Action],
    ) -> Completion:
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": response_output_config(actions),
        }
        if re.fullmatch(r"claude-sonnet-5(?:-[A-Za-z0-9]+)?", model):
            request["thinking"] = {"type": "disabled"}
        started = time.perf_counter()
        response = self.client.messages.create(**request)
        text = "\n".join(
            block.text
            for block in response.content
            if getattr(block, "type", "") == "text"
        ).strip()
        usage = getattr(response, "usage", None)
        return Completion(
            text=text,
            latency_seconds=time.perf_counter() - started,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            response_model=str(getattr(response, "model", "") or ""),
            stop_reason=str(getattr(response, "stop_reason", "") or ""),
        )


def load_or_solve(
    cases: Sequence[OracleBenchmarkCase],
    *,
    oracle_cache: OracleCache,
    engine: Any,
) -> tuple[dict[str, SolveResult], dict[str, bool]]:
    """Resolve each distinct spec exactly once, using the transactional cache."""

    results: dict[str, SolveResult] = {}
    cache_hits: dict[str, bool] = {}
    expected_binary_sha256 = getattr(engine, "binary_sha256", None)
    if not isinstance(expected_binary_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_binary_sha256
    ):
        raise BenchmarkError(
            "oracle engine must expose a lowercase binary_sha256 digest"
        )
    expected_execution_context = getattr(engine, "execution_context", None)
    if expected_execution_context not in {None, "offline", "owned_simulator"}:
        raise BenchmarkError("oracle engine exposes an invalid execution context")
    for case in cases:
        key = case.spec.cache_key
        if key in results:
            continue
        result = oracle_cache.get(
            case.spec,
            expected_binary_sha256=expected_binary_sha256,
            expected_execution_context=expected_execution_context,
        )
        cached = result is not None
        if result is None:
            result = engine.solve(case.spec)
            if dict(result.metadata.extra).get("binary_sha256") != (
                expected_binary_sha256
            ):
                raise BenchmarkError(
                    f"oracle result binary digest mismatch for spec {key}"
                )
            if (
                expected_execution_context is not None
                and dict(result.metadata.extra).get("execution_context")
                != expected_execution_context
            ):
                raise BenchmarkError(
                    f"oracle result execution-context mismatch for spec {key}"
                )
            if not result.metadata.converged:
                raise BenchmarkError(
                    f"oracle did not reach its requested target for spec {key}"
                )
            oracle_cache.put(case.spec, result)
        elif not result.metadata.converged:
            raise BenchmarkError(
                f"cached oracle did not reach its requested target for spec {key}"
            )
        results[key] = result
        cache_hits[key] = cached
    return results, cache_hits


def prepare_decisions(
    cases: Sequence[OracleBenchmarkCase],
    results: dict[str, SolveResult],
    *,
    limit: int | None = None,
    seed: int = 17,
) -> list[PreparedDecision]:
    population = [
        PreparedDecision(case, results[case.spec.cache_key], policy.private_combo)
        for case in sorted(cases, key=lambda item: item.case_id)
        for policy in results[case.spec.cache_key].combo_policies
    ]
    if not population:
        raise BenchmarkError("no private-combo decisions were prepared")
    population_size = len(population)
    population_reach = sum(
        (decision.policy.reach_weight for decision in population),
        Decimal(0),
    )
    decisions = population
    inclusion_probability = Decimal(1)
    if limit is not None:
        if limit <= 0:
            raise BenchmarkError("decision limit must be positive")
        if limit < population_size:
            # Uniform sampling gives every decision the same known inclusion
            # probability. Aggregate reach metrics can therefore use a
            # Horvitz-Thompson correction without weighting reach twice.
            randomizer = random.Random(seed)
            decisions = randomizer.sample(population, limit)
            inclusion_probability = Decimal(limit) / Decimal(population_size)
            decisions.sort(
                key=lambda item: (
                    item.source_case.case_id,
                    item.private_combo,
                )
            )
    return [
        replace(
            decision,
            selection_inclusion_probability=inclusion_probability,
            selection_population_size=population_size,
            selection_population_reach_weight=population_reach,
        )
        for decision in decisions
    ]


def _decimal_text(value: Decimal | None) -> str | None:
    if value is None:
        return None
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def _base_result(decision: PreparedDecision, model: str) -> dict[str, Any]:
    spec = decision.source_case.spec
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "decision_id": decision.decision_id,
        "source_case_id": decision.source_case.case_id,
        "spec_key": spec.cache_key,
        "model": model,
        "street": spec.street.value,
        "board": list(spec.board),
        "acting_player": spec.acting_player.value,
        "private_combo": list(decision.private_combo),
        "reach_weight": _decimal_text(decision.policy.reach_weight),
        "selection_inclusion_probability": _decimal_text(
            decision.selection_inclusion_probability
        ),
        "selection_population_size": decision.selection_population_size,
        "selection_population_reach_weight": _decimal_text(
            decision.selection_population_reach_weight
        ),
        "pot_units": spec.tree.pot,
        "chip_scale": spec.parameters.chip_scale,
        "chip_unit": spec.parameters.chip_unit,
        "modeled_actions": [action_data(action) for action in spec.tree.modeled_actions],
    }


def _failed_result(
    decision: PreparedDecision,
    model: str,
    *,
    status: str,
    error: str,
) -> dict[str, Any]:
    row = _base_result(decision, model)
    row.update(
        {
            "status": status,
            "cached": False,
            "raw_response": "",
            "prediction": None,
            "error": error,
            "legal": False,
            "in_tree": False,
            "near_optimal": False,
            "oracle_mass": None,
            "action_ev_units": None,
            "best_ev_units": None,
            "ev_regret_units": None,
            "ev_regret_pot_fraction": None,
            "latency_seconds": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "response_model": "",
            "stop_reason": "",
        }
    )
    return row


def score_completion(
    decision: PreparedDecision,
    *,
    model: str,
    completion: Completion,
    cached: bool,
    ev_tolerance: Decimal,
) -> dict[str, Any]:
    row = _base_result(decision, model)
    spec = decision.source_case.spec
    common = {
        "cached": cached,
        "raw_response": completion.text,
        "latency_seconds": completion.latency_seconds,
        "input_tokens": completion.input_tokens,
        "output_tokens": completion.output_tokens,
        "response_model": completion.response_model,
        "stop_reason": completion.stop_reason,
    }
    try:
        action = parse_model_action(completion.text, spec.tree.modeled_actions)
    except ModelActionError as error:
        row.update(
            common
            | {
                "status": "PARSE_ERROR",
                "prediction": None,
                "error": str(error),
                "legal": False,
                "in_tree": False,
                "near_optimal": False,
                "oracle_mass": None,
                "action_ev_units": None,
                "best_ev_units": None,
                "ev_regret_units": None,
                "ev_regret_pot_fraction": None,
            }
        )
        return row

    assessment = assess_action(
        spec,
        decision.oracle_result,
        action,
        private_combo=decision.private_combo,
        ev_tolerance=ev_tolerance,
    )
    if assessment.status is AssessmentStatus.SCORED:
        assert isinstance(assessment, ScoredAction)
        regret_fraction = assessment.ev_regret / Decimal(spec.tree.pot)
        row.update(
            common
            | {
                "status": "SCORED",
                "prediction": action_data(action),
                "error": "",
                "legal": True,
                "in_tree": True,
                "near_optimal": assessment.near_optimal,
                "oracle_mass": _decimal_text(assessment.oracle_mass),
                "action_ev_units": _decimal_text(assessment.action_ev),
                "best_ev_units": _decimal_text(assessment.best_ev),
                "ev_regret_units": _decimal_text(assessment.ev_regret),
                "ev_regret_pot_fraction": _decimal_text(regret_fraction),
            }
        )
        return row

    row.update(
        common
        | {
            "status": assessment.status.value,
            "prediction": action_data(action),
            "error": assessment.reason,
            "legal": assessment.status is AssessmentStatus.OUT_OF_TREE,
            "in_tree": False,
            "near_optimal": False,
            "oracle_mass": None,
            "action_ev_units": None,
            "best_ev_units": None,
            "ev_regret_units": None,
            "ev_regret_pot_fraction": None,
        }
    )
    return row


def run_model(
    decisions: Sequence[PreparedDecision],
    *,
    model: str,
    completer: Any | None,
    response_cache: ModelResponseCache,
    allow_provider_calls: bool,
    max_tokens: int = 64,
    ev_tolerance: Decimal = Decimal(0),
    max_consecutive_provider_errors: int = 3,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    if max_tokens <= 0:
        raise BenchmarkError("max_tokens must be positive")
    if max_consecutive_provider_errors <= 0:
        raise BenchmarkError("max_consecutive_provider_errors must be positive")
    keyed_decisions = [
        (
            decision,
            completion_cache_key(
                decision,
                model=model,
                max_tokens=max_tokens,
            ),
        )
        for decision in decisions
    ]
    cached_completions = {
        key: completion
        for _, key in keyed_decisions
        if (completion := response_cache.get(key)) is not None
    }
    pinned_response_model: str | None = None
    for completion in cached_completions.values():
        candidate = completion.response_model.strip()
        if not candidate:
            raise BenchmarkError(
                f"cached completion for requested model {model!r} has no "
                "response_model and cannot be resumed safely"
            )
        if (
            pinned_response_model is not None
            and candidate != pinned_response_model
        ):
            raise BenchmarkError(
                f"response cache mixes resolved response_model versions "
                f"{pinned_response_model!r} and {candidate!r} for requested "
                f"model {model!r}; use an immutable model ID or a new cache"
            )
        pinned_response_model = candidate

    results = []
    consecutive_errors = 0
    total = len(decisions)
    for index, (decision, key) in enumerate(keyed_decisions, start=1):
        completion = cached_completions.get(key)
        cached = completion is not None
        if completion is None and not allow_provider_calls:
            results.append(
                _failed_result(
                    decision,
                    model,
                    status="CACHE_MISS",
                    error="model calls disabled; no cached response",
                )
            )
            if progress:
                progress(index, total)
            continue
        if completion is None:
            if completer is None:
                raise BenchmarkError("provider calls enabled without a completer")
            try:
                completion = completer.complete(
                    build_prompt(decision),
                    model=model,
                    max_tokens=max_tokens,
                    actions=decision.source_case.spec.tree.modeled_actions,
                )
                candidate = completion.response_model.strip()
                if not candidate:
                    raise BenchmarkError(
                        f"provider completion for requested model {model!r} "
                        "has no response_model and cannot be cached safely"
                    )
                if (
                    pinned_response_model is not None
                    and candidate != pinned_response_model
                ):
                    raise BenchmarkError(
                        f"requested model {model!r} resolved to "
                        f"{candidate!r} after this run was pinned to "
                        f"{pinned_response_model!r}; use an immutable model "
                        "ID or a new cache"
                    )
                pinned_response_model = candidate
                response_cache.put(key, completion)
                consecutive_errors = 0
            except BenchmarkError:
                raise
            except Exception as error:
                consecutive_errors += 1
                results.append(
                    _failed_result(
                        decision,
                        model,
                        status="PROVIDER_ERROR",
                        error=str(error),
                    )
                )
                if progress:
                    progress(index, total)
                if consecutive_errors >= max_consecutive_provider_errors:
                    for remaining in decisions[index:]:
                        results.append(
                            _failed_result(
                                remaining,
                                model,
                                status="ABORTED_PROVIDER_ERRORS",
                                error=(
                                    "provider circuit breaker opened after "
                                    f"{consecutive_errors} consecutive errors"
                                ),
                            )
                        )
                    if progress and index < total:
                        progress(total, total)
                    return results
                continue
        results.append(
            score_completion(
                decision,
                model=model,
                completion=completion,
                cached=cached,
                ev_tolerance=ev_tolerance,
            )
        )
        if progress:
            progress(index, total)
    return results


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def aggregate_results(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    scored = [row for row in rows if row["status"] == "SCORED"]
    statuses = Counter(row["status"] for row in rows)
    reach_total = sum(Decimal(row["reach_weight"]) for row in rows)
    reach_scored = sum(Decimal(row["reach_weight"]) for row in scored)
    regrets = [float(Decimal(row["ev_regret_pot_fraction"])) for row in scored]
    masses = [float(Decimal(row["oracle_mass"])) for row in scored]
    selection_fields = {
        "selection_inclusion_probability",
        "selection_population_size",
        "selection_population_reach_weight",
    }
    rows_with_selection = [
        selection_fields.issubset(row) for row in rows
    ]
    if any(rows_with_selection) and not all(rows_with_selection):
        raise BenchmarkError("rows mix legacy and selection-aware schemas")
    selection_aware = bool(rows) and all(rows_with_selection)
    inclusion_probabilities = (
        {
            Decimal(row["selection_inclusion_probability"])
            for row in rows
        }
        if selection_aware
        else {Decimal(1)}
    )
    population_sizes = (
        {int(row["selection_population_size"]) for row in rows}
        if selection_aware
        else {len(rows)}
    )
    population_reaches = (
        {
            Decimal(row["selection_population_reach_weight"])
            for row in rows
        }
        if selection_aware
        else {reach_total}
    )
    if (
        len(inclusion_probabilities) > 1
        or len(population_sizes) > 1
        or len(population_reaches) > 1
    ):
        raise BenchmarkError("rows contain inconsistent selection metadata")
    inclusion_probability = (
        next(iter(inclusion_probabilities))
        if inclusion_probabilities
        else Decimal(1)
    )
    population_size = next(iter(population_sizes)) if population_sizes else 0
    population_reach = (
        next(iter(population_reaches))
        if population_reaches
        else Decimal(0)
    )
    if not Decimal(0) < inclusion_probability <= Decimal(1):
        raise BenchmarkError("row inclusion probability must be in (0, 1]")
    if rows and (
        population_size < len(rows)
        or population_reach <= 0
    ):
        raise BenchmarkError("rows contain invalid selection population metadata")
    adjusted_scored_reach = sum(
        Decimal(row["reach_weight"]) / inclusion_probability
        for row in scored
    )
    weighted_regret_numerator = sum(
        Decimal(row["ev_regret_pot_fraction"])
        * Decimal(row["reach_weight"])
        / inclusion_probability
        for row in scored
    )
    # With complete sampled responses, the full population reach is known and
    # turns the numerator into an unbiased Horvitz-Thompson estimate. For
    # partial coverage, use the corresponding Hájek ratio over scored rows.
    weighted_regret_denominator = (
        population_reach
        if scored and len(scored) == len(rows)
        else adjusted_scored_reach
    )
    weighted_regret = (
        weighted_regret_numerator / weighted_regret_denominator
        if weighted_regret_denominator
        else Decimal(0)
    )
    uncached_latencies = [
        float(row["latency_seconds"])
        for row in rows
        if not row["cached"] and row["status"] not in {"CACHE_MISS", "ABORTED_PROVIDER_ERRORS"}
    ]
    return {
        "decisions": total,
        "selection_population_decisions": population_size,
        "selection_inclusion_probability": float(inclusion_probability),
        "reach_weighted_estimator": (
            "exact"
            if inclusion_probability == 1 and len(scored) == len(rows)
            else "exact_scored_rows"
            if inclusion_probability == 1
            else "horvitz_thompson_known_population_reach"
            if scored and len(scored) == len(rows)
            else "hajek_scored_rows"
        ),
        "status_counts": dict(sorted(statuses.items())),
        "coverage_rate": len(scored) / total if total else 0.0,
        "reach_weighted_coverage": (
            float(reach_scored / reach_total) if reach_total else 0.0
        ),
        "legal_action_rate": (
            sum(bool(row["legal"]) for row in rows) / total if total else 0.0
        ),
        "in_tree_rate": (
            sum(bool(row["in_tree"]) for row in rows) / total if total else 0.0
        ),
        "near_optimal_rate_on_scored": (
            sum(bool(row["near_optimal"]) for row in scored) / len(scored)
            if scored
            else 0.0
        ),
        "mean_ev_regret_pot_fraction": statistics.mean(regrets) if regrets else 0.0,
        "reach_weighted_mean_ev_regret_pot_fraction": float(weighted_regret),
        "mean_oracle_action_mass": statistics.mean(masses) if masses else 0.0,
        "cache_hits": sum(bool(row["cached"]) for row in rows),
        "input_tokens": sum(int(row["input_tokens"]) for row in rows),
        "output_tokens": sum(int(row["output_tokens"]) for row in rows),
        "uncached_latency_p50_seconds": (
            statistics.median(uncached_latencies) if uncached_latencies else 0.0
        ),
        "uncached_latency_p95_seconds": _percentile(uncached_latencies, 0.95),
    }


def paired_comparisons(
    rows: Sequence[dict[str, Any]], models: Sequence[str]
) -> list[dict[str, Any]]:
    by_model = {
        model: {row["decision_id"]: row for row in rows if row["model"] == model}
        for model in models
    }
    comparisons = []
    for model_a, model_b in itertools.combinations(models, 2):
        shared_ids = sorted(set(by_model[model_a]) & set(by_model[model_b]))
        paired = [
            (by_model[model_a][decision_id], by_model[model_b][decision_id])
            for decision_id in shared_ids
            if by_model[model_a][decision_id]["status"] == "SCORED"
            and by_model[model_b][decision_id]["status"] == "SCORED"
        ]
        deltas = [
            Decimal(second["ev_regret_pot_fraction"])
            - Decimal(first["ev_regret_pot_fraction"])
            for first, second in paired
        ]
        comparisons.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "shared_decisions": len(shared_ids),
                "paired_scored_decisions": len(paired),
                "model_a_lower_regret": sum(delta > 0 for delta in deltas),
                "model_b_lower_regret": sum(delta < 0 for delta in deltas),
                "regret_ties": sum(delta == 0 for delta in deltas),
                "mean_regret_delta_b_minus_a_pot_fraction": (
                    float(sum(deltas, Decimal(0)) / len(deltas))
                    if deltas
                    else 0.0
                ),
                "model_a_mean_oracle_mass": (
                    statistics.mean(float(Decimal(first["oracle_mass"])) for first, _ in paired)
                    if paired
                    else 0.0
                ),
                "model_b_mean_oracle_mass": (
                    statistics.mean(float(Decimal(second["oracle_mass"])) for _, second in paired)
                    if paired
                    else 0.0
                ),
            }
        )
    return comparisons


def build_report(
    rows: Sequence[dict[str, Any]],
    *,
    models: Sequence[str],
    cases: Sequence[OracleBenchmarkCase],
    decisions: Sequence[PreparedDecision],
    run_config: dict[str, Any],
    oracle_cache_hits: dict[str, bool],
    oracle_results: dict[str, SolveResult] | None = None,
) -> dict[str, Any]:
    oracle_by_spec = oracle_results or {
        decision.source_case.spec.cache_key: decision.oracle_result
        for decision in decisions
    }
    if any(case.spec.cache_key not in oracle_by_spec for case in cases):
        raise BenchmarkError("report is missing an oracle result for a source case")
    by_model = {
        model: aggregate_results([row for row in rows if row["model"] == model])
        for model in models
    }
    incomplete_statuses = {
        "CACHE_MISS",
        "PROVIDER_ERROR",
        "ABORTED_PROVIDER_ERRORS",
    }
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "offline_post_session_model_evaluation",
        "run_complete": not any(row["status"] in incomplete_statuses for row in rows),
        "models": list(models),
        "source_cases": [
            {
                "case_id": case.case_id,
                "description": case.description,
                "spec_key": case.spec.cache_key,
                "street": case.spec.street.value,
                "board": list(case.spec.board),
                "oracle_cache_hit": oracle_cache_hits[case.spec.cache_key],
                "solver": case.spec.parameters.solver_name,
                "solver_commit": case.spec.parameters.solver_commit,
                "target_exploitability_pct": str(
                    case.spec.parameters.target_exploitability_pct
                ),
                "oracle_result": {
                    "solver_version": oracle_by_spec[
                        case.spec.cache_key
                    ].metadata.solver_version,
                    "iterations": oracle_by_spec[
                        case.spec.cache_key
                    ].metadata.iterations,
                    "elapsed_seconds": str(
                        oracle_by_spec[
                            case.spec.cache_key
                        ].metadata.elapsed_seconds
                    ),
                    "exploitability_units": str(
                        oracle_by_spec[
                            case.spec.cache_key
                        ].metadata.exploitability
                    ),
                    "converged": oracle_by_spec[
                        case.spec.cache_key
                    ].metadata.converged,
                    "extra": dict(
                        oracle_by_spec[case.spec.cache_key].metadata.extra
                    ),
                },
            }
            for case in cases
        ],
        "decision_count_per_model": len(decisions),
        "run_config": run_config,
        "per_model": by_model,
        "paired_comparisons": paired_comparisons(rows, models),
        "limitations": [
            "Heads-up NLHE postflop nodes only; not six-max or multiway GTO.",
            "The oracle is conditional on supplied ranges, rake, stack, and discrete bet tree.",
            "One deterministic action is requested from each model; mixed-strategy sampling is not measured.",
            "EV regret evaluates decisions but does not train, fine-tune, or certify either model.",
            "The demo is a tiny smoke case and is not a representative model comparison dataset.",
            "Descendant cases require an exact supported same-street action history; this is not a full hand-history benchmark.",
            "This subsystem is post-session only and must never be connected to live capture or hotkeys.",
        ],
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _nonnegative_decimal(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("must be a decimal number") from error
    if not parsed.is_finite() or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _models_from_args(args: argparse.Namespace) -> list[str]:
    if args.model == "fast":
        return [args.fast_model]
    if args.model == "coach":
        return [args.coach_model]
    return list(dict.fromkeys([args.fast_model, args.coach_model]))


def _progress(label: str) -> Callable[[int, int], None]:
    def callback(index: int, total: int) -> None:
        if index == 1 or index == total or index % 10 == 0:
            print(f"\r{label}: {index}/{total}", end="", flush=True)
            if index == total:
                print()

    return callback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Claude actions with an exact local HU postflop oracle, "
            "strictly offline/post-session."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("write-demo", help="write the transparent demo case JSON")
    demo.add_argument("--output", type=Path, default=Path("benchmark_data/gto_oracle/demo.json"))

    validate = subparsers.add_parser("validate", help="strictly validate a case file")
    validate.add_argument("--cases", type=Path, required=True)

    run = subparsers.add_parser("run", help="solve, query cached/models, score, and report")
    run.add_argument(
        "--offline-confirmed",
        action="store_true",
        help="confirm the poker client is closed and all hands are already complete",
    )
    run.add_argument(
        "--call-models",
        action="store_true",
        help="explicitly allow Anthropic API calls and associated cost; otherwise cache-only",
    )
    run.add_argument(
        "--cases",
        type=Path,
        help="strict case JSON; default is the small built-in validation suite",
    )
    run.add_argument("--model", choices=("fast", "coach", "both"), default="both")
    run.add_argument(
        "--fast-model",
        default=os.getenv("CLAUDE_FAST_MODEL", DEFAULT_FAST_MODEL),
    )
    run.add_argument(
        "--coach-model",
        default=os.getenv("CLAUDE_MODEL", DEFAULT_COACH_MODEL),
    )
    run.add_argument("--limit", type=_positive_int)
    run.add_argument("--seed", type=int, default=17)
    run.add_argument("--max-tokens", type=_positive_int, default=64)
    run.add_argument("--ev-tolerance", type=_nonnegative_decimal, default=Decimal(0))
    run.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    run.add_argument("--engine-timeout", type=_positive_float, default=600.0)
    run.add_argument("--provider-timeout", type=_positive_float, default=30.0)
    run.add_argument("--provider-retries", type=int, choices=range(0, 6), default=2)
    run.add_argument("--max-consecutive-errors", type=_positive_int, default=3)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run.add_argument("--response-cache", type=Path, default=DEFAULT_RESPONSE_CACHE)
    run.add_argument("--oracle-cache", type=Path, default=DEFAULT_ORACLE_CACHE)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "write-demo":
        write_case_file(args.output, demo_cases())
        print(args.output)
        return 0
    if args.command == "validate":
        cases = load_case_file(args.cases)
        print(f"Validated {len(cases)} offline HU postflop case(s)")
        for case in cases:
            print(f"  {case.case_id}: {case.spec.cache_key}")
        return 0

    if not args.offline_confirmed:
        parser.error(
            "run requires --offline-confirmed: close the poker client and use completed hands only"
        )
    load_dotenv()
    cases = load_case_file(args.cases) if args.cases else demo_cases()
    models = _models_from_args(args)
    if args.call_models:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("ANTHROPIC_API_KEY is required with --call-models", file=sys.stderr)
            return 2
        completer: Any | None = AnthropicCompleter(
            api_key,
            timeout=args.provider_timeout,
            max_retries=args.provider_retries,
        )
    else:
        completer = None

    args.oracle_cache.parent.mkdir(parents=True, exist_ok=True)
    with OracleCache(args.oracle_cache) as oracle_cache:
        oracle_results, oracle_hits = load_or_solve(
            cases,
            oracle_cache=oracle_cache,
            engine=EngineClient(
                args.engine,
                offline_only_acknowledged=args.offline_confirmed,
                timeout_seconds=str(args.engine_timeout),
            ),
        )
    decisions = prepare_decisions(
        cases,
        oracle_results,
        limit=args.limit,
        seed=args.seed,
    )
    response_cache = ModelResponseCache(args.response_cache)
    if response_cache.corrupt_lines:
        print(f"Warning: ignored {response_cache.corrupt_lines} corrupt cache line(s)")
    if response_cache.stale_lines:
        print(f"Info: ignored {response_cache.stale_lines} stale cache line(s)")

    rows: list[dict[str, Any]] = []
    for model in models:
        rows.extend(
            run_model(
                decisions,
                model=model,
                completer=completer,
                response_cache=response_cache,
                allow_provider_calls=args.call_models,
                max_tokens=args.max_tokens,
                ev_tolerance=args.ev_tolerance,
                max_consecutive_provider_errors=args.max_consecutive_errors,
                progress=_progress(model),
            )
        )

    run_config = {
        "offline_confirmed": True,
        "provider_calls_enabled": args.call_models,
        "prompt_version": PROMPT_VERSION,
        "case_source": str(args.cases) if args.cases else "built_in_demo",
        "decision_limit": args.limit,
        "selection": "all" if args.limit is None else "uniform_without_replacement",
        "reach_weighted_estimator": (
            "exact"
            if args.limit is None
            else "horvitz_thompson_known_population_reach"
        ),
        "selection_seed": args.seed,
        "max_tokens": args.max_tokens,
        "ev_tolerance_units": str(args.ev_tolerance),
        "engine_binary": str(args.engine),
        "engine_timeout_seconds": args.engine_timeout,
        "response_cache": str(args.response_cache),
        "oracle_cache": str(args.oracle_cache),
    }
    report = build_report(
        rows,
        models=models,
        cases=cases,
        decisions=decisions,
        run_config=run_config,
        oracle_cache_hits=oracle_hits,
        oracle_results=oracle_results,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    results_path = args.output_dir / f"{stamp}_results.jsonl"
    report_path = args.output_dir / f"{stamp}_report.json"
    _write_jsonl(results_path, rows)
    _write_json(report_path, report)

    print(f"Results: {results_path}")
    print(f"Report: {report_path}")
    for model in models:
        summary = report["per_model"][model]
        print(
            f"{model}: coverage={summary['coverage_rate']:.1%}, "
            "weighted regret/pot="
            f"{summary['reach_weighted_mean_ev_regret_pot_fraction']:.6f}, "
            f"oracle mass={summary['mean_oracle_action_mass']:.3f}"
        )
    if not args.call_models:
        print("Model network calls were disabled; only cached completions were used.")
    return 0 if report["run_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
