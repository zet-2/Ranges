"""Immutable models for one action-conditioned flop-to-river solve.

The legacy :class:`~gto_oracle.models.SolveSpec` intentionally describes a
single street root or a short same-street descendant.  A complete public hand
needs a different contract: the solver must begin on the flop, traverse every
public action and chance card in order, and expose both players' conditional
range weights at the final decision node.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import re

from .models import (
    Action,
    ActionKind,
    ComboPolicy,
    OracleValidationError,
    PlayerRange,
    Position,
    SolveParameters,
    SolverMetadata,
    _action_sort_key,
    _coerce_decimal,
    _coerce_engine_units,
    _coerce_enum,
    _validate_card,
)
from .serialization import canonical_json, sha256_key


@dataclass(frozen=True, slots=True)
class ContinuationAction:
    """One exact public poker action in a postflop path."""

    action: Action

    def __post_init__(self) -> None:
        if not isinstance(self.action, Action):
            raise OracleValidationError(
                "continuation action step requires an Action"
            )


@dataclass(frozen=True, slots=True)
class ContinuationDeal:
    """One exact turn or river card in a postflop path."""

    card: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "card",
            _validate_card(self.card, "continuation deal card"),
        )


ContinuationStep = ContinuationAction | ContinuationDeal


@dataclass(frozen=True, slots=True)
class ContinuationSpec:
    """A reproducible HU solve that always starts at the true flop root."""

    flop: tuple[str, str, str]
    current_board: tuple[str, ...]
    acting_player: Position
    oop_range: PlayerRange
    ip_range: PlayerRange
    starting_pot: int
    effective_stack: int
    path: tuple[ContinuationStep, ...]
    expected_total_invested: tuple[int, int]
    facing_bet: int
    legal_action_kinds: tuple[ActionKind, ...]
    modeled_actions: tuple[Action, ...]
    parameters: SolveParameters
    rake_rate_pct: Decimal = Decimal(0)
    rake_cap: int = 0
    variant: str = "NLHE"

    def __post_init__(self) -> None:
        if self.variant != "NLHE":
            raise OracleValidationError(
                "continuation solving supports only no-limit Hold'em"
            )
        flop = tuple(
            _validate_card(card, f"flop card {index}")
            for index, card in enumerate(self.flop)
        )
        if len(flop) != 3 or len(set(flop)) != 3:
            raise OracleValidationError(
                "continuation flop must contain three unique cards"
            )
        board = tuple(
            _validate_card(card, f"current board card {index}")
            for index, card in enumerate(self.current_board)
        )
        if len(board) not in {3, 4, 5}:
            raise OracleValidationError(
                "continuation current board must contain 3, 4, or 5 cards"
            )
        if len(set(board)) != len(board):
            raise OracleValidationError(
                "continuation current board cannot repeat a card"
            )
        if board[:3] != flop:
            raise OracleValidationError(
                "continuation current board must preserve the flop prefix"
            )
        object.__setattr__(self, "flop", flop)
        object.__setattr__(self, "current_board", board)

        actor = _coerce_enum(
            self.acting_player,
            Position,
            "continuation acting player",
        )
        object.__setattr__(self, "acting_player", actor)
        if not isinstance(self.oop_range, PlayerRange) or not isinstance(
            self.ip_range, PlayerRange
        ):
            raise OracleValidationError(
                "continuation ranges must be PlayerRange instances"
            )
        if self.oop_range.position is not Position.OOP:
            raise OracleValidationError(
                "continuation oop_range must be assigned to OOP"
            )
        if self.ip_range.position is not Position.IP:
            raise OracleValidationError(
                "continuation ip_range must be assigned to IP"
            )
        flop_set = set(flop)
        for player_range in (self.oop_range, self.ip_range):
            for combo in player_range.combos:
                if flop_set.intersection(combo.cards):
                    raise OracleValidationError(
                        "continuation input range overlaps the flop"
                    )

        starting_pot = _coerce_engine_units(
            self.starting_pot,
            "continuation starting pot",
        )
        effective_stack = _coerce_engine_units(
            self.effective_stack,
            "continuation effective stack",
        )
        facing_bet = _coerce_engine_units(
            self.facing_bet,
            "continuation facing bet",
        )
        rake_cap = _coerce_engine_units(
            self.rake_cap,
            "continuation rake cap",
        )
        if starting_pot <= 0 or effective_stack <= 0:
            raise OracleValidationError(
                "continuation starting pot and effective stack must be positive"
            )
        if starting_pot > 536_870_911 or effective_stack > 536_870_911:
            raise OracleValidationError(
                "continuation pot or stack exceeds the engine safe limit"
            )
        if facing_bet > effective_stack:
            raise OracleValidationError(
                "continuation facing bet cannot exceed the root effective stack"
            )
        object.__setattr__(self, "starting_pot", starting_pot)
        object.__setattr__(self, "effective_stack", effective_stack)
        object.__setattr__(self, "facing_bet", facing_bet)
        object.__setattr__(self, "rake_cap", rake_cap)

        rake_rate = _coerce_decimal(
            self.rake_rate_pct,
            "continuation rake rate percent",
        )
        if not Decimal(0) <= rake_rate <= Decimal(100):
            raise OracleValidationError(
                "continuation rake rate must be between 0 and 100"
            )
        object.__setattr__(self, "rake_rate_pct", rake_rate)

        path = tuple(self.path)
        if len(path) > 256:
            raise OracleValidationError(
                "continuation path cannot exceed 256 public steps"
            )
        if any(
            not isinstance(step, (ContinuationAction, ContinuationDeal))
            for step in path
        ):
            raise OracleValidationError(
                "continuation path contains an unsupported step"
            )
        dealt = tuple(
            step.card for step in path if isinstance(step, ContinuationDeal)
        )
        if dealt != board[3:]:
            raise OracleValidationError(
                "continuation deal steps must exactly equal the turn/river board suffix"
            )
        for step in path:
            if (
                isinstance(step, ContinuationAction)
                and step.action.amount is not None
                and step.action.amount > effective_stack
            ):
                raise OracleValidationError(
                    "continuation path action exceeds the root effective stack"
                )
        object.__setattr__(self, "path", path)

        invested = tuple(
            _coerce_engine_units(value, "continuation total invested")
            for value in self.expected_total_invested
        )
        if len(invested) != 2:
            raise OracleValidationError(
                "continuation expected_total_invested must contain OOP and IP"
            )
        if any(value > effective_stack for value in invested):
            raise OracleValidationError(
                "continuation total investment exceeds the root effective stack"
            )
        actor_index = 0 if actor is Position.OOP else 1
        implied_facing = invested[actor_index ^ 1] - invested[actor_index]
        if implied_facing < 0 or implied_facing != facing_bet:
            raise OracleValidationError(
                "continuation investments disagree with the facing-bet amount"
            )
        object.__setattr__(
            self,
            "expected_total_invested",
            invested,
        )

        kinds = tuple(
            _coerce_enum(kind, ActionKind, "continuation legal action kind")
            for kind in self.legal_action_kinds
        )
        if not kinds or len(kinds) != len(set(kinds)):
            raise OracleValidationError(
                "continuation legal action kinds must be non-empty and unique"
            )
        if facing_bet == 0:
            if (
                ActionKind.CHECK not in kinds
                or set(kinds)
                & {ActionKind.FOLD, ActionKind.CALL, ActionKind.RAISE}
            ):
                raise OracleValidationError(
                    "a no-wager continuation node requires CHECK and forbids "
                    "FOLD/CALL/RAISE"
                )
        elif (
            not {ActionKind.FOLD, ActionKind.CALL}.issubset(kinds)
            or set(kinds) & {ActionKind.CHECK, ActionKind.BET}
        ):
            raise OracleValidationError(
                "a facing-bet continuation node requires FOLD/CALL and forbids "
                "CHECK/BET"
            )
        kinds = tuple(sorted(kinds, key=lambda kind: kind.value))
        object.__setattr__(self, "legal_action_kinds", kinds)

        modeled = tuple(self.modeled_actions)
        if not modeled or any(not isinstance(action, Action) for action in modeled):
            raise OracleValidationError(
                "continuation modeled actions must contain Action instances"
            )
        if len(modeled) != len(set(modeled)):
            raise OracleValidationError(
                "continuation modeled actions cannot contain duplicates"
            )
        if {action.kind for action in modeled} != set(kinds):
            raise OracleValidationError(
                "continuation modeled actions must cover every legal kind exactly"
            )
        if any(
            action.amount is not None and action.amount > effective_stack
            for action in modeled
        ):
            raise OracleValidationError(
                "continuation modeled action exceeds the root effective stack"
            )
        object.__setattr__(
            self,
            "modeled_actions",
            tuple(sorted(modeled, key=_action_sort_key)),
        )
        if not isinstance(self.parameters, SolveParameters):
            raise OracleValidationError(
                "continuation parameters must be SolveParameters"
            )

    @property
    def canonical_json(self) -> str:
        return canonical_json(self)

    @property
    def cache_key(self) -> str:
        return sha256_key(self)


@dataclass(frozen=True, slots=True)
class ConditionalCombo:
    """One private combo's weight after the complete public path."""

    cards: tuple[str, str]
    input_range_weight: Decimal
    path_weight: Decimal
    joint_compatible_weight: Decimal
    conditional_reach_weight: Decimal

    def __post_init__(self) -> None:
        from .models import _coerce_combo_cards

        object.__setattr__(
            self,
            "cards",
            _coerce_combo_cards(self.cards, "conditional range combo"),
        )
        for field_name in (
            "input_range_weight",
            "path_weight",
            "joint_compatible_weight",
            "conditional_reach_weight",
        ):
            value = _coerce_decimal(getattr(self, field_name), field_name)
            if value <= 0:
                raise OracleValidationError(
                    f"{field_name} must be positive for an exported combo"
                )
            if (
                field_name in {"input_range_weight", "path_weight", "conditional_reach_weight"}
                and value > 1
            ):
                raise OracleValidationError(
                    f"{field_name} cannot exceed one"
                )
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True, slots=True)
class ConditionalRange:
    """A player's normalized range at the final continuation node."""

    position: Position
    combos: tuple[ConditionalCombo, ...]

    def __post_init__(self) -> None:
        position = _coerce_enum(
            self.position,
            Position,
            "conditional range position",
        )
        combos = tuple(self.combos)
        if not combos or any(not isinstance(combo, ConditionalCombo) for combo in combos):
            raise OracleValidationError(
                "conditional range must contain ConditionalCombo entries"
            )
        identities = [combo.cards for combo in combos]
        if len(identities) != len(set(identities)):
            raise OracleValidationError(
                "conditional range cannot repeat a combo"
            )
        reach_sum = sum(
            (combo.conditional_reach_weight for combo in combos),
            Decimal(0),
        )
        if abs(reach_sum - Decimal(1)) > Decimal("1e-6"):
            raise OracleValidationError(
                "conditional range reach weights must sum to one"
            )
        object.__setattr__(self, "position", position)
        object.__setattr__(
            self,
            "combos",
            tuple(sorted(combos, key=lambda combo: combo.cards)),
        )


@dataclass(frozen=True, slots=True)
class ContinuationResult:
    """Current-node policy plus both action-conditioned ranges."""

    spec_key: str
    combo_policies: tuple[ComboPolicy, ...]
    conditional_ranges: tuple[ConditionalRange, ConditionalRange]
    metadata: SolverMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.spec_key, str) or not re.fullmatch(
            r"[0-9a-f]{64}",
            self.spec_key,
        ):
            raise OracleValidationError(
                "continuation result spec key must be lowercase SHA-256"
            )
        policies = tuple(self.combo_policies)
        if not policies or any(
            not isinstance(policy, ComboPolicy) for policy in policies
        ):
            raise OracleValidationError(
                "continuation result must contain combo policies"
            )
        if len({policy.private_combo for policy in policies}) != len(policies):
            raise OracleValidationError(
                "continuation result cannot repeat a policy combo"
            )
        ranges = tuple(self.conditional_ranges)
        if len(ranges) != 2 or {item.position for item in ranges} != {
            Position.OOP,
            Position.IP,
        }:
            raise OracleValidationError(
                "continuation result requires OOP and IP conditional ranges"
            )
        if not isinstance(self.metadata, SolverMetadata):
            raise OracleValidationError(
                "continuation result metadata must be SolverMetadata"
            )
        object.__setattr__(
            self,
            "combo_policies",
            tuple(sorted(policies, key=lambda policy: policy.private_combo)),
        )
        object.__setattr__(
            self,
            "conditional_ranges",
            tuple(sorted(ranges, key=lambda item: item.position.value)),
        )

    @classmethod
    def for_spec(
        cls,
        spec: ContinuationSpec,
        combo_policies: tuple[ComboPolicy, ...],
        conditional_ranges: tuple[ConditionalRange, ConditionalRange],
        metadata: SolverMetadata,
    ) -> "ContinuationResult":
        if not isinstance(spec, ContinuationSpec):
            raise OracleValidationError(
                "continuation result requires a ContinuationSpec"
            )
        result = cls(
            spec.cache_key,
            combo_policies,
            conditional_ranges,
            metadata,
        )
        validate_continuation_result(spec, result)
        return result


def validate_continuation_result(
    spec: ContinuationSpec,
    result: ContinuationResult,
) -> None:
    """Bind a continuation response to the exact request and final node."""

    if not isinstance(spec, ContinuationSpec) or not isinstance(
        result,
        ContinuationResult,
    ):
        raise OracleValidationError(
            "continuation validation requires matching model instances"
        )
    if result.spec_key != spec.cache_key:
        raise OracleValidationError(
            "continuation result belongs to a different specification"
        )
    if result.metadata.solver_name != spec.parameters.solver_name:
        raise OracleValidationError(
            "continuation solver name differs from the specification"
        )
    if result.metadata.solver_version != spec.parameters.solver_commit:
        raise OracleValidationError(
            "continuation solver commit differs from the specification"
        )

    acting_range = (
        spec.oop_range
        if spec.acting_player is Position.OOP
        else spec.ip_range
    )
    allowed_combos = {combo.cards for combo in acting_range.combos}
    policy_combos = {policy.private_combo for policy in result.combo_policies}
    if not policy_combos or not policy_combos.issubset(allowed_combos):
        raise OracleValidationError(
            "continuation policies contain no reachable requested combos"
        )
    reach_sum = sum(
        (policy.reach_weight for policy in result.combo_policies),
        Decimal(0),
    )
    if abs(reach_sum - Decimal(1)) > Decimal("1e-6"):
        raise OracleValidationError(
            "continuation acting-player policy reaches must sum to one"
        )
    expected_actions = set(spec.modeled_actions)
    for policy in result.combo_policies:
        if {value.action for value in policy.action_values} != expected_actions:
            raise OracleValidationError(
                "continuation policy actions differ from the modeled node"
            )

    requested = {
        Position.OOP: {combo.cards for combo in spec.oop_range.combos},
        Position.IP: {combo.cards for combo in spec.ip_range.combos},
    }
    for conditional_range in result.conditional_ranges:
        if not {
            combo.cards for combo in conditional_range.combos
        }.issubset(requested[conditional_range.position]):
            raise OracleValidationError(
                "conditional range contains a combo outside its input range"
            )


__all__ = [
    "ConditionalCombo",
    "ConditionalRange",
    "ContinuationAction",
    "ContinuationDeal",
    "ContinuationResult",
    "ContinuationSpec",
    "ContinuationStep",
    "validate_continuation_result",
]
