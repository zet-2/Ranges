"""Search and bilateral best-response measurement for HU jam/fold games."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from hu_jam_fold_model import (
    MANIFEST_NUMBER_TOLERANCE,
    TARGET_COMPARISON_TOLERANCE,
    BehaviorStrategy,
    BestResponseReport,
    JamFoldGame,
    JamFoldSolution,
    JamFoldValidationError,
    finite_float,
    positive_int,
)


@dataclass(frozen=True, slots=True)
class _CompiledGame:
    sb_types: tuple[str, ...]
    bb_types: tuple[str, ...]
    probabilities: np.ndarray
    sb_marginal: np.ndarray
    bb_marginal: np.ndarray
    sb_fold: float
    bb_after_sb_fold: float
    sb_after_bb_fold: float
    bb_fold: float
    sb_called: np.ndarray
    bb_called: np.ndarray


@dataclass(slots=True)
class _SearchState:
    sb_regrets: np.ndarray
    bb_regrets: np.ndarray
    sb_strategy_sum: np.ndarray
    bb_strategy_sum: np.ndarray
    total_average_weight: float = 0.0
    completed: int = 0


def _compile(game: JamFoldGame) -> _CompiledGame:
    sb_types = game.sb_types
    bb_types = game.bb_types
    sb_index = {name: index for index, name in enumerate(sb_types)}
    bb_index = {name: index for index, name in enumerate(bb_types)}
    shape = (len(sb_types), len(bb_types))
    weights = np.zeros(shape, dtype=np.float64)
    sb_called = np.zeros(shape, dtype=np.float64)
    bb_called = np.zeros(shape, dtype=np.float64)
    for deal in game.deals:
        i = sb_index[deal.sb_type]
        j = bb_index[deal.bb_type]
        weights[i, j] = finite_float(deal.weight, "private deal weight")
        sb_value, bb_value = game.profile.called_jam_payoffs(
            deal.sb_showdown_equity
        )
        sb_called[i, j] = finite_float(sb_value, "SB called-jam payoff")
        bb_called[i, j] = finite_float(bb_value, "BB called-jam payoff")
    probabilities = _normalize_weights(weights)
    sb_marginal = np.sum(probabilities, axis=1)
    bb_marginal = np.sum(probabilities, axis=0)
    if np.any(sb_marginal <= 0) or np.any(bb_marginal <= 0):
        raise JamFoldValidationError("every private type needs positive marginal mass")
    sb_fold, bb_after_sb_fold = game.profile.sb_fold_payoffs()
    sb_after_bb_fold, bb_fold = game.profile.bb_fold_payoffs()
    return _CompiledGame(
        sb_types=sb_types,
        bb_types=bb_types,
        probabilities=probabilities,
        sb_marginal=sb_marginal,
        bb_marginal=bb_marginal,
        sb_fold=finite_float(sb_fold, "SB-fold SB payoff"),
        bb_after_sb_fold=finite_float(bb_after_sb_fold, "SB-fold BB payoff"),
        sb_after_bb_fold=finite_float(sb_after_bb_fold, "BB-fold SB payoff"),
        bb_fold=finite_float(bb_fold, "BB-fold BB payoff"),
        sb_called=sb_called,
        bb_called=bb_called,
    )


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0:
        raise JamFoldValidationError("private model total weight must be positive")
    probabilities = weights / total
    if np.any((weights > 0) & (probabilities == 0)):
        raise JamFoldValidationError(
            "private deal probability underflows the float64 certificate"
        )
    return probabilities


def _strategy_arrays(
    compiled: _CompiledGame,
    strategy: BehaviorStrategy,
) -> tuple[np.ndarray, np.ndarray]:
    sb_map = dict(strategy.sb_jam)
    bb_map = dict(strategy.bb_call)
    sb_jam = np.array(
        [sb_map[name] for name in compiled.sb_types],
        dtype=np.float64,
    )
    bb_call = np.array(
        [bb_map[name] for name in compiled.bb_types],
        dtype=np.float64,
    )
    return sb_jam, bb_call


def _behavior_strategy(
    compiled: _CompiledGame,
    sb_jam: np.ndarray,
    bb_call: np.ndarray,
) -> BehaviorStrategy:
    return BehaviorStrategy(
        tuple(zip(compiled.sb_types, (float(value) for value in sb_jam))),
        tuple(zip(compiled.bb_types, (float(value) for value in bb_call))),
    )


def _sb_response_values(
    compiled: _CompiledGame,
    sb_jam: np.ndarray,
    bb_call: np.ndarray,
) -> tuple[float, float]:
    probabilities = compiled.probabilities
    jam_pair = (
        (1.0 - bb_call[np.newaxis, :]) * compiled.sb_after_bb_fold
        + bb_call[np.newaxis, :] * compiled.sb_called
    )
    current = float(
        np.sum(
            probabilities
            * (
                (1.0 - sb_jam[:, np.newaxis]) * compiled.sb_fold
                + sb_jam[:, np.newaxis] * jam_pair
            )
        )
    )
    jam_by_type = np.sum(probabilities * jam_pair, axis=1)
    fold_by_type = compiled.sb_marginal * compiled.sb_fold
    best = float(np.sum(np.maximum(fold_by_type, jam_by_type)))
    return current, best


def _bb_response_values(
    compiled: _CompiledGame,
    sb_jam: np.ndarray,
    bb_call: np.ndarray,
) -> tuple[float, float]:
    probabilities = compiled.probabilities
    after_sb_fold = float(
        np.sum(
            probabilities
            * (1.0 - sb_jam[:, np.newaxis])
            * compiled.bb_after_sb_fold
        )
    )
    jam_pair = (
        (1.0 - bb_call[np.newaxis, :]) * compiled.bb_fold
        + bb_call[np.newaxis, :] * compiled.bb_called
    )
    current = after_sb_fold + float(
        np.sum(probabilities * sb_jam[:, np.newaxis] * jam_pair)
    )
    fold_by_type = np.sum(
        probabilities * sb_jam[:, np.newaxis] * compiled.bb_fold,
        axis=0,
    )
    call_by_type = np.sum(
        probabilities * sb_jam[:, np.newaxis] * compiled.bb_called,
        axis=0,
    )
    best = after_sb_fold + float(
        np.sum(np.maximum(fold_by_type, call_by_type))
    )
    return current, best


def measure_best_responses(
    game: JamFoldGame,
    strategy: BehaviorStrategy,
) -> BestResponseReport:
    """Enumerate both best responses in the complete supplied finite game."""
    strategy.validate_for(game)
    compiled = _compile(game)
    sb_jam, bb_call = _strategy_arrays(compiled, strategy)
    sb_current, sb_best = _sb_response_values(compiled, sb_jam, bb_call)
    bb_current, bb_best = _bb_response_values(compiled, sb_jam, bb_call)
    sb_gain = max(0.0, sb_best - sb_current)
    bb_gain = max(0.0, bb_best - bb_current)
    welfare = sb_current + bb_current
    if welfare > MANIFEST_NUMBER_TOLERANCE:
        raise JamFoldValidationError(
            "recomputed player welfare is positive; rake/payoff conservation failed"
        )
    return BestResponseReport(
        sb_utility_bb=sb_current,
        bb_utility_bb=bb_current,
        expected_rake_bb=max(0.0, -welfare),
        sb_best_response_utility_bb=sb_best,
        bb_best_response_utility_bb=bb_best,
        sb_deviation_gain_bb=sb_gain,
        bb_deviation_gain_bb=bb_gain,
        epsilon_bb=max(sb_gain, bb_gain),
    )


def _regret_strategy(regrets: np.ndarray) -> np.ndarray:
    positive = np.maximum(regrets, 0.0)
    totals = np.sum(positive, axis=1, keepdims=True)
    return np.divide(
        positive,
        totals,
        out=np.full_like(positive, 0.5),
        where=totals > 0,
    )


def _new_search_state(compiled: _CompiledGame) -> _SearchState:
    sb_regrets = np.zeros((len(compiled.sb_types), 2), dtype=np.float64)
    bb_regrets = np.zeros((len(compiled.bb_types), 2), dtype=np.float64)
    return _SearchState(
        sb_regrets=sb_regrets,
        bb_regrets=bb_regrets,
        sb_strategy_sum=np.zeros_like(sb_regrets),
        bb_strategy_sum=np.zeros_like(bb_regrets),
    )


def _action_values(
    compiled: _CompiledGame,
    sb_jam: np.ndarray,
    bb_call: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = compiled.probabilities
    sb_jam_pair = (
        (1.0 - bb_call[np.newaxis, :]) * compiled.sb_after_bb_fold
        + bb_call[np.newaxis, :] * compiled.sb_called
    )
    sb_values = np.column_stack(
        (
            np.full(len(compiled.sb_types), compiled.sb_fold),
            np.sum(probabilities * sb_jam_pair, axis=1) / compiled.sb_marginal,
        )
    )
    bb_fold = (
        np.sum(
            probabilities * sb_jam[:, np.newaxis] * compiled.bb_fold,
            axis=0,
        )
        / compiled.bb_marginal
    )
    bb_call_values = (
        np.sum(
            probabilities * sb_jam[:, np.newaxis] * compiled.bb_called,
            axis=0,
        )
        / compiled.bb_marginal
    )
    return sb_values, np.column_stack((bb_fold, bb_call_values))


def _advance_search(
    compiled: _CompiledGame,
    state: _SearchState,
    iteration: int,
) -> None:
    sb_strategy = _regret_strategy(state.sb_regrets)
    bb_strategy = _regret_strategy(state.bb_regrets)
    average_weight = float(iteration)
    state.sb_strategy_sum += average_weight * sb_strategy
    state.bb_strategy_sum += average_weight * bb_strategy
    state.total_average_weight += average_weight
    sb_values, bb_values = _action_values(
        compiled,
        sb_strategy[:, 1],
        bb_strategy[:, 1],
    )
    sb_node = np.sum(sb_strategy * sb_values, axis=1)
    bb_node = np.sum(bb_strategy * bb_values, axis=1)
    state.sb_regrets = np.maximum(
        0.0,
        state.sb_regrets
        + compiled.sb_marginal[:, np.newaxis]
        * (sb_values - sb_node[:, np.newaxis]),
    )
    state.bb_regrets = np.maximum(
        0.0,
        state.bb_regrets
        + compiled.bb_marginal[:, np.newaxis]
        * (bb_values - bb_node[:, np.newaxis]),
    )
    state.completed = iteration


def _average_candidate(
    game: JamFoldGame,
    compiled: _CompiledGame,
    state: _SearchState,
) -> tuple[BehaviorStrategy, BestResponseReport]:
    sb_jam = state.sb_strategy_sum[:, 1] / state.total_average_weight
    bb_call = state.bb_strategy_sum[:, 1] / state.total_average_weight
    strategy = _behavior_strategy(compiled, sb_jam, bb_call)
    return strategy, measure_best_responses(game, strategy)


def _validate_solve_controls(
    target_epsilon_bb: float,
    max_iterations: int,
    min_iterations: int,
    check_every: int,
) -> tuple[float, int, int, int]:
    if (
        isinstance(target_epsilon_bb, bool)
        or not isinstance(target_epsilon_bb, (int, float))
        or not math.isfinite(float(target_epsilon_bb))
        or target_epsilon_bb < 0
    ):
        raise JamFoldValidationError(
            "target_epsilon_bb must be finite and non-negative"
        )
    maximum = positive_int(max_iterations, "max_iterations")
    minimum = positive_int(min_iterations, "min_iterations")
    interval = positive_int(check_every, "check_every")
    if minimum > maximum:
        raise JamFoldValidationError("min_iterations cannot exceed max_iterations")
    return float(target_epsilon_bb), maximum, minimum, interval


def _target_reached(report: BestResponseReport, target: float) -> bool:
    return report.epsilon_bb <= target + TARGET_COMPARISON_TOLERANCE


def solve_game(
    game: JamFoldGame,
    *,
    target_epsilon_bb: float = 0.001,
    max_iterations: int = 250_000,
    min_iterations: int = 2_000,
    check_every: int = 1_000,
) -> JamFoldSolution:
    """Search with RM+; certify convergence only through measured BR gaps."""
    target, maximum, minimum, interval = _validate_solve_controls(
        target_epsilon_bb,
        max_iterations,
        min_iterations,
        check_every,
    )
    compiled = _compile(game)
    state = _new_search_state(compiled)
    candidate: tuple[BehaviorStrategy, BestResponseReport] | None = None
    for iteration in range(1, maximum + 1):
        _advance_search(compiled, state, iteration)
        should_check = iteration >= minimum and (
            iteration % interval == 0 or iteration == maximum
        )
        if not should_check:
            continue
        candidate = _average_candidate(game, compiled, state)
        if _target_reached(candidate[1], target):
            break
    if candidate is None:
        candidate = _average_candidate(game, compiled, state)
    strategy, report = candidate
    return JamFoldSolution(
        game=game,
        strategy=strategy,
        report=report,
        target_epsilon_bb=target,
        iterations=state.completed,
        min_iterations=minimum,
        check_every=interval,
        converged=_target_reached(report, target),
    )

