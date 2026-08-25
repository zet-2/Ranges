"""Strict JSON artifact loading and certificate verification for jam/fold."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Mapping

from hu_jam_fold_model import (
    ALGORITHM_ID,
    ARTIFACT_TYPE,
    CERTIFICATE_METHOD,
    CERTIFICATE_NUMERIC,
    GAME_ID,
    GAME_SCHEMA_VERSION,
    MANIFEST_NUMBER_TOLERANCE,
    PROBABILITY_TOLERANCE,
    RAKE_BASIS,
    RAKE_ROUNDING,
    SOLUTION_CONCEPT,
    SOLUTION_SCHEMA_VERSION,
    TARGET_COMPARISON_TOLERANCE,
    TYPE_SEMANTICS,
    BehaviorStrategy,
    BestResponseReport,
    JamFoldGame,
    JamFoldProfile,
    JamFoldSolution,
    JamFoldValidationError,
    PrivateTypeDeal,
    RakeProfile,
    decimal_value,
    exact_keys,
    finite_float,
    mapping,
    positive_int,
)
from hu_jam_fold_solver import measure_best_responses


def _require_schema_version(
    value: object,
    expected: int,
    field_name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise JamFoldValidationError(f"{field_name} must be {expected}")


def _parse_rake(payload: object) -> RakeProfile:
    rake = mapping(payload, "profile rake")
    exact_keys(
        rake,
        {
            "rate_pct",
            "cap_bb",
            "chip_unit_bb",
            "no_flop_no_drop",
            "rounding",
            "basis",
        },
        "profile rake",
    )
    if rake["rounding"] != RAKE_ROUNDING:
        raise JamFoldValidationError(f"rake rounding must be {RAKE_ROUNDING!r}")
    if rake["basis"] != RAKE_BASIS:
        raise JamFoldValidationError(f"rake basis must be {RAKE_BASIS!r}")
    return RakeProfile(
        rate_pct=decimal_value(rake["rate_pct"], "rake rate_pct"),
        cap_bb=decimal_value(rake["cap_bb"], "rake cap_bb"),
        chip_unit_bb=decimal_value(rake["chip_unit_bb"], "rake chip_unit_bb"),
        no_flop_no_drop=rake["no_flop_no_drop"],
    )


def _parse_profile(payload: object) -> JamFoldProfile:
    profile = mapping(payload, "profile")
    exact_keys(
        profile,
        {
            "profile_id",
            "effective_stack_bb",
            "small_blind_bb",
            "big_blind_bb",
            "rake",
        },
        "profile",
    )
    return JamFoldProfile(
        profile_id=profile["profile_id"],
        effective_stack_bb=decimal_value(
            profile["effective_stack_bb"],
            "effective_stack_bb",
        ),
        small_blind_bb=decimal_value(
            profile["small_blind_bb"],
            "small_blind_bb",
        ),
        big_blind_bb=decimal_value(
            profile["big_blind_bb"],
            "big_blind_bb",
        ),
        rake=_parse_rake(profile["rake"]),
    )


def _parse_deals(payload: object) -> tuple[PrivateTypeDeal, ...]:
    if not isinstance(payload, list):
        raise JamFoldValidationError("private_model deals must be an array")
    deals: list[PrivateTypeDeal] = []
    for index, item in enumerate(payload):
        row = mapping(item, f"private_model deals[{index}]")
        exact_keys(
            row,
            {"sb_type", "bb_type", "weight", "sb_showdown_equity"},
            f"private_model deals[{index}]",
        )
        deals.append(
            PrivateTypeDeal(
                sb_type=row["sb_type"],
                bb_type=row["bb_type"],
                weight=decimal_value(row["weight"], f"deals[{index}] weight"),
                sb_showdown_equity=decimal_value(
                    row["sb_showdown_equity"],
                    f"deals[{index}] sb_showdown_equity",
                ),
            )
        )
    return tuple(deals)


def _parse_private_model(
    payload: object,
) -> tuple[object, object, tuple[PrivateTypeDeal, ...]]:
    model = mapping(payload, "private_model")
    exact_keys(
        model,
        {"model_id", "model_version", "type_semantics", "deals"},
        "private_model",
    )
    if model["type_semantics"] != TYPE_SEMANTICS:
        raise JamFoldValidationError(
            f"private_model type_semantics must be {TYPE_SEMANTICS!r}"
        )
    return model["model_id"], model["model_version"], _parse_deals(model["deals"])


def parse_game_payload(payload: object) -> JamFoldGame:
    root = mapping(payload, "game artifact")
    exact_keys(
        root,
        {"schema_version", "game", "profile", "private_model"},
        "game artifact",
    )
    _require_schema_version(
        root["schema_version"],
        GAME_SCHEMA_VERSION,
        "game schema_version",
    )
    if root["game"] != GAME_ID:
        raise JamFoldValidationError(f"game must be {GAME_ID!r}")
    model_id, model_version, deals = _parse_private_model(root["private_model"])
    return JamFoldGame(
        profile=_parse_profile(root["profile"]),
        model_id=model_id,
        model_version=model_version,
        deals=deals,
    )


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise JamFoldValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json_file(path: Path, field_name: str) -> object:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                JamFoldValidationError(f"invalid JSON constant {value}")
            ),
        )
    except JamFoldValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise JamFoldValidationError(
            f"cannot read {field_name} {path}: {error}"
        ) from error


def load_game_artifact(path: Path | str) -> JamFoldGame:
    source = Path(path)
    return parse_game_payload(_load_json_file(source, "game artifact"))


def _parse_probability(value: object, field_name: str) -> float:
    parsed = decimal_value(value, field_name)
    if not 0 <= parsed <= 1:
        raise JamFoldValidationError(f"{field_name} must be in [0, 1]")
    return float(parsed)


def _parse_strategy_rows(
    rows: object,
    *,
    field_name: str,
    action_name: str,
) -> list[tuple[str, float]]:
    if not isinstance(rows, list):
        raise JamFoldValidationError(f"strategy {field_name} must be an array")
    result: list[tuple[str, float]] = []
    for index, item in enumerate(rows):
        row = mapping(item, f"strategy {field_name}[{index}]")
        exact_keys(
            row,
            {"type", "fold", action_name},
            f"strategy {field_name}[{index}]",
        )
        fold = _parse_probability(
            row["fold"],
            f"strategy {field_name}[{index}] fold",
        )
        action = _parse_probability(
            row[action_name],
            f"strategy {field_name}[{index}] {action_name}",
        )
        if abs(fold + action - 1.0) > PROBABILITY_TOLERANCE:
            raise JamFoldValidationError(
                f"strategy {field_name}[{index}] probabilities do not sum to 1"
            )
        result.append((row["type"], action))
    return result


def _strategy_from_payload(payload: object) -> BehaviorStrategy:
    root = mapping(payload, "strategy")
    exact_keys(root, {"sb", "bb_vs_jam"}, "strategy")
    sb = _parse_strategy_rows(root["sb"], field_name="sb", action_name="jam")
    bb = _parse_strategy_rows(
        root["bb_vs_jam"],
        field_name="bb_vs_jam",
        action_name="call",
    )
    return BehaviorStrategy(tuple(sb), tuple(bb))


def _manifest_float(value: object, field_name: str) -> float:
    return finite_float(decimal_value(value, field_name), field_name)


def _assert_close(actual: float, expected: object, field_name: str) -> None:
    claimed = _manifest_float(expected, field_name)
    tolerance = MANIFEST_NUMBER_TOLERANCE * max(
        1.0,
        abs(actual),
        abs(claimed),
    )
    if abs(actual - claimed) > tolerance:
        raise JamFoldValidationError(
            f"{field_name} mismatch: claimed {claimed}, recomputed {actual}"
        )


def _solution_root(payload: object) -> Mapping[str, object]:
    root = mapping(payload, "solution")
    exact_keys(
        root,
        {
            "schema_version",
            "artifact_type",
            "game_fingerprint",
            "profile",
            "private_model",
            "claim",
            "solve",
            "strategy",
            "strategy_fingerprint",
            "utilities",
        },
        "solution",
    )
    return root


def _validate_solution_envelope(
    game: JamFoldGame,
    root: Mapping[str, object],
) -> None:
    _require_schema_version(
        root["schema_version"],
        SOLUTION_SCHEMA_VERSION,
        "solution schema_version",
    )
    if root["artifact_type"] != ARTIFACT_TYPE:
        raise JamFoldValidationError(f"artifact_type must be {ARTIFACT_TYPE!r}")
    if root["game_fingerprint"] != game.fingerprint:
        raise JamFoldValidationError(
            "solution game_fingerprint does not match artifact"
        )
    if root["profile"] != game.profile.to_payload():
        raise JamFoldValidationError("solution profile does not match artifact")
    expected_model = {
        "model_id": game.model_id,
        "model_version": game.model_version,
    }
    if root["private_model"] != expected_model:
        raise JamFoldValidationError("solution private_model does not match artifact")


def _parse_claim(payload: object) -> tuple[Mapping[str, object], float, bool]:
    claim = mapping(payload, "solution claim")
    exact_keys(
        claim,
        {
            "solution_concept",
            "certificate_method",
            "certificate_numeric",
            "game_scope",
            "full_hunl",
            "rake_aware",
            "target_epsilon_bb",
            "achieved_epsilon_bb",
            "sb_deviation_gain_bb",
            "bb_deviation_gain_bb",
            "target_reached",
        },
        "solution claim",
    )
    expected = (
        ("solution_concept", SOLUTION_CONCEPT),
        ("certificate_method", CERTIFICATE_METHOD),
        ("certificate_numeric", CERTIFICATE_NUMERIC),
        ("game_scope", "restricted_hu_jam_fold"),
    )
    for field_name, expected_value in expected:
        if claim[field_name] != expected_value:
            raise JamFoldValidationError(f"{field_name} is unsupported")
    if claim["full_hunl"] is not False:
        raise JamFoldValidationError("jam/fold solution must declare full_hunl=false")
    if claim["rake_aware"] is not True:
        raise JamFoldValidationError("jam/fold solution must declare rake_aware=true")
    reached = claim["target_reached"]
    if not isinstance(reached, bool):
        raise JamFoldValidationError("target_reached must be boolean")
    target = _manifest_float(claim["target_epsilon_bb"], "target_epsilon_bb")
    if target < 0:
        raise JamFoldValidationError("target_epsilon_bb cannot be negative")
    return claim, target, reached


def _parse_solve_metadata(payload: object) -> tuple[int, int, int]:
    solve = mapping(payload, "solution solve")
    exact_keys(
        solve,
        {"algorithm", "iterations", "min_iterations", "check_every"},
        "solution solve",
    )
    if solve["algorithm"] != ALGORITHM_ID:
        raise JamFoldValidationError("solution algorithm is unsupported")
    iterations = positive_int(solve["iterations"], "solution iterations")
    minimum = positive_int(solve["min_iterations"], "solution min_iterations")
    interval = positive_int(solve["check_every"], "solution check_every")
    if iterations < minimum:
        raise JamFoldValidationError(
            "solution iterations cannot be below min_iterations"
        )
    return iterations, minimum, interval


def _validated_strategy(
    game: JamFoldGame,
    root: Mapping[str, object],
) -> BehaviorStrategy:
    strategy = _strategy_from_payload(root["strategy"])
    strategy.validate_for(game)
    if root["strategy_fingerprint"] != strategy.fingerprint:
        raise JamFoldValidationError("solution strategy_fingerprint mismatch")
    return strategy


def _validate_report_claims(
    report: BestResponseReport,
    claim: Mapping[str, object],
    utilities_payload: object,
) -> None:
    _assert_close(
        report.epsilon_bb,
        claim["achieved_epsilon_bb"],
        "achieved_epsilon_bb",
    )
    _assert_close(
        report.sb_deviation_gain_bb,
        claim["sb_deviation_gain_bb"],
        "sb_deviation_gain_bb",
    )
    _assert_close(
        report.bb_deviation_gain_bb,
        claim["bb_deviation_gain_bb"],
        "bb_deviation_gain_bb",
    )
    utilities = mapping(utilities_payload, "solution utilities")
    fields = {
        "sb_utility_bb",
        "bb_utility_bb",
        "expected_rake_bb",
        "sb_best_response_utility_bb",
        "bb_best_response_utility_bb",
    }
    exact_keys(utilities, fields, "solution utilities")
    for field_name in fields:
        _assert_close(getattr(report, field_name), utilities[field_name], field_name)


def _validate_max_epsilon(max_epsilon_bb: float | None) -> float | None:
    if max_epsilon_bb is None:
        return None
    if (
        isinstance(max_epsilon_bb, bool)
        or not isinstance(max_epsilon_bb, (int, float))
        or not math.isfinite(float(max_epsilon_bb))
        or max_epsilon_bb < 0
    ):
        raise JamFoldValidationError(
            "max_epsilon_bb must be finite and non-negative"
        )
    return float(max_epsilon_bb)


def _validate_target_policy(
    report: BestResponseReport,
    *,
    target: float,
    claimed_reached: bool,
    require_target_reached: bool,
    max_epsilon_bb: float | None,
) -> bool:
    reached = report.epsilon_bb <= target + TARGET_COMPARISON_TOLERANCE
    if claimed_reached != reached:
        raise JamFoldValidationError("target_reached disagrees with recomputed gap")
    if require_target_reached and not reached:
        raise JamFoldValidationError(
            "solution did not reach its declared epsilon target"
        )
    maximum = _validate_max_epsilon(max_epsilon_bb)
    if (
        maximum is not None
        and report.epsilon_bb > maximum + MANIFEST_NUMBER_TOLERANCE
    ):
        raise JamFoldValidationError(
            "solution exceeds the verifier's maximum epsilon"
        )
    return reached


def verify_solution_payload(
    game: JamFoldGame,
    payload: object,
    *,
    max_epsilon_bb: float | None = None,
    require_target_reached: bool = True,
) -> JamFoldSolution:
    """Validate a manifest and independently recompute its bilateral gap."""
    root = _solution_root(payload)
    _validate_solution_envelope(game, root)
    claim, target, claimed_reached = _parse_claim(root["claim"])
    iterations, minimum, interval = _parse_solve_metadata(root["solve"])
    strategy = _validated_strategy(game, root)
    report = measure_best_responses(game, strategy)
    _validate_report_claims(report, claim, root["utilities"])
    reached = _validate_target_policy(
        report,
        target=target,
        claimed_reached=claimed_reached,
        require_target_reached=require_target_reached,
        max_epsilon_bb=max_epsilon_bb,
    )
    return JamFoldSolution(
        game=game,
        strategy=strategy,
        report=report,
        target_epsilon_bb=target,
        iterations=iterations,
        min_iterations=minimum,
        check_every=interval,
        converged=reached,
    )


def load_solution(
    game: JamFoldGame,
    path: Path | str,
    *,
    max_epsilon_bb: float | None = None,
    require_target_reached: bool = True,
) -> JamFoldSolution:
    source = Path(path)
    payload = _load_json_file(source, "solution")
    return verify_solution_payload(
        game,
        payload,
        max_epsilon_bb=max_epsilon_bb,
        require_target_reached=require_target_reached,
    )


def atomic_write_json(path: Path, payload: object, *, force: bool) -> None:
    path = path.resolve()
    if path.exists() and not force:
        raise JamFoldValidationError(
            f"refusing to overwrite existing output {path}; pass --force"
        )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError as error:
        raise JamFoldValidationError(
            f"cannot write solution {path}: {error}"
        ) from error
    finally:
        if temporary.exists():
            temporary.unlink()

