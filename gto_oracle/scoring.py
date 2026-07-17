"""Solver-neutral decision scoring for reproducible local analysis."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import TypeAlias

from .models import (
    Action,
    ActionKind,
    DecisionQuery,
    OracleValidationError,
    SolveResult,
    SolveSpec,
    _coerce_decimal,
    validate_result_for_spec,
)


class AssessmentStatus(str, Enum):
    """Typed outcome of comparing one action with an oracle tree."""

    SCORED = "SCORED"
    ILLEGAL = "ILLEGAL"
    OUT_OF_TREE = "OUT_OF_TREE"


@dataclass(frozen=True, slots=True)
class ScoredAction:
    """Frequency and EV quality of an action represented by the oracle."""

    private_combo: tuple[str, str]
    action: Action
    reach_weight: Decimal
    equity: Decimal
    oracle_mass: Decimal
    action_ev: Decimal
    best_ev: Decimal
    ev_regret: Decimal
    near_optimal: bool
    ev_tolerance: Decimal
    status: AssessmentStatus = AssessmentStatus.SCORED


@dataclass(frozen=True, slots=True)
class IllegalActionAssessment:
    """An action impossible at the represented poker node."""

    private_combo: tuple[str, str]
    action: Action
    reason: str
    status: AssessmentStatus = AssessmentStatus.ILLEGAL


@dataclass(frozen=True, slots=True)
class OutOfTreeActionAssessment:
    """A legally possible action whose exact size was not solved."""

    private_combo: tuple[str, str]
    action: Action
    reason: str
    modeled_actions: tuple[Action, ...]
    status: AssessmentStatus = AssessmentStatus.OUT_OF_TREE


ActionAssessment: TypeAlias = (
    ScoredAction | IllegalActionAssessment | OutOfTreeActionAssessment
)


def assess_action(
    spec: SolveSpec,
    result: SolveResult,
    decision: Action | DecisionQuery,
    *,
    private_combo: tuple[str, str] | None = None,
    ev_tolerance: Decimal = Decimal(0),
    allow_unconverged: bool = False,
) -> ActionAssessment:
    """Score one exact private combo and action against its oracle policy.

    ``oracle_mass`` is the solver's probability for the exact action.
    ``ev_regret`` is ``best action EV - selected action EV``. A decision is
    near-optimal when regret is no larger than ``ev_tolerance``. Pass either a
    :class:`DecisionQuery`, or an :class:`Action` plus ``private_combo``.
    Unconverged solver output is rejected unless the caller deliberately sets
    ``allow_unconverged=True`` for diagnostic analysis.
    """

    if not isinstance(spec, SolveSpec):
        raise OracleValidationError("spec must be a SolveSpec")
    if not isinstance(result, SolveResult):
        raise OracleValidationError("result must be a SolveResult")
    if isinstance(decision, DecisionQuery):
        if private_combo is not None:
            raise OracleValidationError(
                "private_combo must not be repeated with a DecisionQuery"
            )
        query = decision
    elif isinstance(decision, Action):
        if private_combo is None:
            raise OracleValidationError(
                "private_combo is required when scoring an Action"
            )
        query = DecisionQuery(private_combo, decision)
    else:
        raise OracleValidationError("decision must be an Action or DecisionQuery")
    action = query.action
    tolerance = _coerce_decimal(ev_tolerance, "EV tolerance")
    if tolerance < 0:
        raise OracleValidationError("EV tolerance cannot be negative")
    if not isinstance(allow_unconverged, bool):
        raise OracleValidationError("allow_unconverged must be boolean")
    validate_result_for_spec(spec, result)
    if not result.metadata.converged and not allow_unconverged:
        raise OracleValidationError(
            "cannot score an unconverged solve without allow_unconverged=True"
        )
    policies_by_combo = {
        policy.private_combo: policy for policy in result.combo_policies
    }
    policy = policies_by_combo.get(query.private_combo)
    if policy is None:
        raise OracleValidationError(
            "decision private combo is not in the acting player's solved range"
        )

    tree = spec.tree
    if action.kind not in tree.legal_action_kinds:
        return IllegalActionAssessment(
            query.private_combo,
            action,
            f"{action.kind.value} is not legal while facing {tree.facing_bet} chips",
        )
    if action.amount is not None and action.amount > tree.effective_stack:
        return IllegalActionAssessment(
            query.private_combo, action, "action amount exceeds effective stack"
        )
    if action.kind is ActionKind.RAISE and action.amount <= tree.facing_bet:
        return IllegalActionAssessment(
            query.private_combo, action, "raise amount does not exceed the call"
        )
    minimum_raise_to = tree.minimum_raise_to
    if (
        action.kind is ActionKind.RAISE
        and minimum_raise_to is not None
        and action.amount < minimum_raise_to
    ):
        return IllegalActionAssessment(
            query.private_combo,
            action,
            f"raise amount is below the minimum full raise-to {minimum_raise_to}",
        )
    if action.kind is ActionKind.ALL_IN and action.amount != tree.effective_stack:
        return IllegalActionAssessment(
            query.private_combo,
            action,
            "all-in amount differs from effective stack",
        )
    if action not in tree.modeled_actions:
        return OutOfTreeActionAssessment(
            query.private_combo,
            action,
            "action is legal but its exact size is absent from the solver tree",
            tree.modeled_actions,
        )

    by_action = {value.action: value for value in policy.action_values}
    selected = by_action[action]
    best_ev = max(value.ev for value in policy.action_values)
    regret = max(Decimal(0), best_ev - selected.ev)
    return ScoredAction(
        private_combo=query.private_combo,
        action=action,
        reach_weight=policy.reach_weight,
        equity=policy.equity,
        oracle_mass=selected.frequency,
        action_ev=selected.ev,
        best_ev=best_ev,
        ev_regret=regret,
        near_optimal=regret <= tolerance,
        ev_tolerance=tolerance,
    )
