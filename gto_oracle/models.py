"""Immutable, solver-neutral models for reproducible poker analysis.

Nothing in this module captures a screen, listens for keys, or calls a model API.
The types deliberately describe one heads-up postflop decision node so that a
solver result can be reproduced and audited after play.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
import re
from typing import Iterable


class OracleValidationError(ValueError):
    """Raised when oracle data violates a structural invariant."""


class UnsupportedGameError(OracleValidationError):
    """Raised for preflop, multiway, or non-NLHE requests not supported here."""


class Position(str, Enum):
    """The two positions in a heads-up postflop solve."""

    OOP = "OOP"
    IP = "IP"


class Street(str, Enum):
    """Hold'em streets; :class:`SolveSpec` intentionally rejects PREFLOP."""

    PREFLOP = "PREFLOP"
    FLOP = "FLOP"
    TURN = "TURN"
    RIVER = "RIVER"


class ActionKind(str, Enum):
    """Solver-neutral no-limit Hold'em action categories."""

    CHECK = "CHECK"
    BET = "BET"
    FOLD = "FOLD"
    CALL = "CALL"
    RAISE = "RAISE"
    ALL_IN = "ALL_IN"


class AllocationMode(str, Enum):
    """Memory representation requested from the pinned local engine."""

    UNCOMPRESSED_F32 = "uncompressed_f32"
    COMPRESSED_I16 = "compressed_i16"


_CARD_RE = re.compile(r"^[2-9TJQKA][cdhs]$")
_RANK_ORDER = {rank: index for index, rank in enumerate("23456789TJQKA")}
_SUIT_ORDER = {suit: index for index, suit in enumerate("cdhs")}
_SIZED_ACTIONS = frozenset(
    {ActionKind.BET, ActionKind.RAISE, ActionKind.ALL_IN}
)


def _coerce_enum(value: object, enum_type: type[Enum], field_name: str):
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value.upper())
        except ValueError as error:
            raise OracleValidationError(
                f"{field_name} has unsupported value {value!r}"
            ) from error
    raise OracleValidationError(
        f"{field_name} must be a {enum_type.__name__}, not {type(value).__name__}"
    )


def _coerce_decimal(value: object, field_name: str) -> Decimal:
    """Return a finite canonical Decimal and reject binary floating point."""

    if isinstance(value, bool) or isinstance(value, float):
        raise OracleValidationError(
            f"{field_name} must use Decimal, int, or str; floats are ambiguous"
        )
    try:
        decimal_value = value if isinstance(value, Decimal) else Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as error:
        raise OracleValidationError(f"{field_name} is not a decimal number") from error
    if not decimal_value.is_finite():
        raise OracleValidationError(f"{field_name} must be finite")
    if decimal_value == 0:
        return Decimal(0)
    return decimal_value.normalize()


def _card_sort_key(card: str) -> tuple[int, int]:
    return (_RANK_ORDER[card[0]], _SUIT_ORDER[card[1]])


def _validate_card(card: object, field_name: str) -> str:
    if not isinstance(card, str) or not _CARD_RE.fullmatch(card):
        raise OracleValidationError(
            f"{field_name} must use canonical card notation such as 'As' or 'Td'"
        )
    return card


def _coerce_combo_cards(value: object, field_name: str) -> tuple[str, str]:
    try:
        cards = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise OracleValidationError(f"{field_name} must contain two cards") from error
    if len(cards) != 2:
        raise OracleValidationError(f"{field_name} must contain two cards")
    validated = tuple(
        _validate_card(card, f"{field_name} card {index}")
        for index, card in enumerate(cards)
    )
    if validated[0] == validated[1]:
        raise OracleValidationError(f"{field_name} cannot repeat a card")
    return tuple(sorted(validated, key=_card_sort_key))  # type: ignore[return-value]


def _coerce_engine_units(value: object, field_name: str) -> int:
    """Require an integral signed-32-bit engine-unit count."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise OracleValidationError(f"{field_name} must be integral engine units")
    if value < 0 or value > 2_147_483_647:
        raise OracleValidationError(
            f"{field_name} must fit a non-negative signed 32-bit integer"
        )
    return value


def _action_sort_key(action: "Action") -> tuple[str, int]:
    return (action.kind.value, action.amount if action.amount is not None else -1)


@dataclass(frozen=True, slots=True)
class Action:
    """A node action using final BET_TO/RAISE_TO/ALL_IN_TO engine units.

    BET, RAISE, and ALL_IN require a positive amount. CHECK, FOLD, and CALL
    are unsized because their chip amount follows from the node state. Sized
    amounts are final contribution targets, never incremental chips added.
    """

    kind: ActionKind
    amount: int | None = None

    def __post_init__(self) -> None:
        kind = _coerce_enum(self.kind, ActionKind, "action kind")
        object.__setattr__(self, "kind", kind)
        if kind in _SIZED_ACTIONS:
            if self.amount is None:
                raise OracleValidationError(f"{kind.value} requires an amount")
            amount = _coerce_engine_units(self.amount, "action amount")
            if amount <= 0:
                raise OracleValidationError("action amount must be positive")
            object.__setattr__(self, "amount", amount)
        elif self.amount is not None:
            raise OracleValidationError(f"{kind.value} must not have an amount")


@dataclass(frozen=True, slots=True)
class DecisionQuery:
    """An exact private combo and action to score after a session."""

    private_combo: tuple[str, str]
    action: Action

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "private_combo",
            _coerce_combo_cards(self.private_combo, "decision private combo"),
        )
        if not isinstance(self.action, Action):
            raise OracleValidationError("decision query action must be an Action")


@dataclass(frozen=True, slots=True)
class WeightedCombo:
    """One unordered private-card combination and its range weight."""

    cards: tuple[str, str]
    weight: Decimal = Decimal(1)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "cards", _coerce_combo_cards(self.cards, "private-card combo")
        )

        weight = _coerce_decimal(self.weight, "combo weight")
        if not Decimal(0) < weight <= Decimal(1):
            raise OracleValidationError("combo weight must be greater than 0 and at most 1")
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True, slots=True)
class PlayerRange:
    """A weighted private-card range for exactly one HU position."""

    position: Position
    combos: tuple[WeightedCombo, ...]

    def __post_init__(self) -> None:
        position = _coerce_enum(self.position, Position, "range position")
        object.__setattr__(self, "position", position)
        combos = tuple(self.combos)
        if not combos:
            raise OracleValidationError("a player range cannot be empty")
        if any(not isinstance(combo, WeightedCombo) for combo in combos):
            raise OracleValidationError("range combos must be WeightedCombo instances")
        ordered = tuple(sorted(combos, key=lambda combo: combo.cards))
        identities = [combo.cards for combo in ordered]
        if len(identities) != len(set(identities)):
            raise OracleValidationError("a player range cannot repeat a combo")
        object.__setattr__(self, "combos", ordered)


def _minimum_full_raise_to(
    action_history: tuple[Action, ...],
    facing_bet: int,
) -> int | None:
    """Derive the next full raise-to target from same-street action history.

    Sized actions are final contribution targets. A short all-in may increase
    the amount to call without resetting the last full raise increment.
    """

    if facing_bet <= 0:
        return None

    current_to = 0
    last_full_raise = 0
    history_is_coherent = True
    for action in action_history:
        if action.kind not in _SIZED_ACTIONS:
            continue
        assert action.amount is not None
        if action.amount <= current_to:
            history_is_coherent = False
            break
        increment = action.amount - current_to
        if current_to == 0:
            last_full_raise = increment
        elif action.kind is not ActionKind.ALL_IN:
            if increment < last_full_raise:
                history_is_coherent = False
                break
            last_full_raise = increment
        elif increment >= last_full_raise:
            last_full_raise = increment
        current_to = action.amount

    if (
        not history_is_coherent
        or current_to < facing_bet
        or last_full_raise <= 0
    ):
        # A facing-bet node without a complete usable history can only be
        # interpreted as facing the first wager of the street.
        return facing_bet * 2
    return current_to + last_full_raise


@dataclass(frozen=True, slots=True)
class TreeConfig:
    """Configuration of one HU decision node and its discrete tree.

    ``modeled_actions`` are the branches actually solved. ``legal_action_kinds``
    describes poker legality at the node, allowing scoring to distinguish an
    impossible action from a legal bet size omitted by the solver abstraction.
    """

    pot: int
    effective_stack: int
    facing_bet: int
    legal_action_kinds: tuple[ActionKind, ...]
    modeled_actions: tuple[Action, ...]
    action_history: tuple[Action, ...] = ()
    rake_rate_pct: Decimal = Decimal(0)
    rake_cap: int = 0

    def __post_init__(self) -> None:
        pot = _coerce_engine_units(self.pot, "pot")
        stack = _coerce_engine_units(self.effective_stack, "effective stack")
        facing = _coerce_engine_units(self.facing_bet, "facing bet")
        rake_rate = _coerce_decimal(self.rake_rate_pct, "rake rate percent")
        rake_cap = _coerce_engine_units(self.rake_cap, "rake cap")
        if pot <= 0:
            raise OracleValidationError("pot must be positive")
        if stack <= 0:
            raise OracleValidationError("effective stack must be positive")
        if pot > 536_870_911 or stack > 536_870_911:
            raise OracleValidationError(
                "pot and effective stack exceed the engine's safe unit limit"
            )
        if facing < 0 or facing > stack:
            raise OracleValidationError("facing bet must be between 0 and the stack")
        if not Decimal(0) <= rake_rate <= Decimal(100):
            raise OracleValidationError("rake rate percent must be between 0 and 100")
        object.__setattr__(self, "pot", pot)
        object.__setattr__(self, "effective_stack", stack)
        object.__setattr__(self, "facing_bet", facing)
        object.__setattr__(self, "rake_rate_pct", rake_rate)
        object.__setattr__(self, "rake_cap", rake_cap)

        kinds = tuple(
            _coerce_enum(kind, ActionKind, "legal action kind")
            for kind in self.legal_action_kinds
        )
        if not kinds:
            raise OracleValidationError("legal action kinds cannot be empty")
        if len(kinds) != len(set(kinds)):
            raise OracleValidationError("legal action kinds cannot contain duplicates")
        kinds = tuple(sorted(kinds, key=lambda kind: kind.value))
        object.__setattr__(self, "legal_action_kinds", kinds)

        if facing == 0:
            invalid = set(kinds) & {
                ActionKind.FOLD,
                ActionKind.CALL,
                ActionKind.RAISE,
            }
            if invalid or ActionKind.CHECK not in kinds:
                raise OracleValidationError(
                    "when facing no bet, CHECK is required and FOLD/CALL/RAISE are illegal"
                )
        else:
            invalid = set(kinds) & {ActionKind.CHECK, ActionKind.BET}
            if invalid or not {ActionKind.FOLD, ActionKind.CALL}.issubset(kinds):
                raise OracleValidationError(
                    "when facing a bet, FOLD and CALL are required and CHECK/BET are illegal"
                )

        modeled = tuple(self.modeled_actions)
        history = tuple(self.action_history)
        if not modeled:
            raise OracleValidationError("modeled actions cannot be empty")
        if any(not isinstance(action, Action) for action in modeled + history):
            raise OracleValidationError("tree actions must be Action instances")
        if len(modeled) != len(set(modeled)):
            raise OracleValidationError("modeled actions cannot contain duplicates")
        modeled_kinds = {action.kind for action in modeled}
        if not modeled_kinds.issubset(kinds):
            raise OracleValidationError("every modeled action kind must be legally available")
        if modeled_kinds != set(kinds):
            raise OracleValidationError("every legal action kind needs a modeled branch")

        for action in modeled:
            if action.amount is not None and action.amount > stack:
                raise OracleValidationError("a modeled action cannot exceed the stack")
            if action.kind is ActionKind.RAISE and action.amount <= facing:
                raise OracleValidationError("a raise must add more chips than a call")
            if (
                action.kind is ActionKind.RAISE
                and action.amount < _minimum_full_raise_to(history, facing)
            ):
                raise OracleValidationError(
                    "a modeled non-all-in raise is below the minimum full raise-to"
                )
            if action.kind is ActionKind.ALL_IN and action.amount != stack:
                raise OracleValidationError("ALL_IN amount must equal the effective stack")
        for action in history:
            if action.amount is not None and action.amount > stack:
                raise OracleValidationError("a history action cannot exceed the stack")
            if action.kind is ActionKind.ALL_IN and action.amount != stack:
                raise OracleValidationError(
                    "a history ALL_IN amount must equal the effective stack"
                )
        object.__setattr__(
            self, "modeled_actions", tuple(sorted(modeled, key=_action_sort_key))
        )
        object.__setattr__(self, "action_history", history)

    @property
    def minimum_raise_to(self) -> int | None:
        """Return the legal full raise-to floor, excluding short all-ins."""

        return _minimum_full_raise_to(self.action_history, self.facing_bet)


def _sizing_text(value: object, field_name: str, *, optional: bool = False):
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise OracleValidationError(f"{field_name} must be a sizing string")
    value = value.strip()
    if len(value.encode("utf-8")) > 512:
        raise OracleValidationError(f"{field_name} cannot exceed 512 bytes")
    return value


@dataclass(frozen=True, slots=True)
class PlayerBetSizes:
    """Exact upstream bet and raise size strings for one position/street."""

    bet: str
    raise_sizes: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "bet", _sizing_text(self.bet, "bet sizes"))
        object.__setattr__(
            self,
            "raise_sizes",
            _sizing_text(self.raise_sizes, "raise sizes"),
        )


@dataclass(frozen=True, slots=True)
class StreetBetSizes:
    """Complete OOP/IP size menu for one postflop street."""

    oop: PlayerBetSizes
    ip: PlayerBetSizes

    def __post_init__(self) -> None:
        if not isinstance(self.oop, PlayerBetSizes) or not isinstance(
            self.ip, PlayerBetSizes
        ):
            raise OracleValidationError(
                "street bet sizes require PlayerBetSizes for both OOP and IP"
            )


@dataclass(frozen=True, slots=True)
class BetSizingConfig:
    """All size menus that define the solver's postflop action tree.

    Donk fields preserve three upstream states: ``None`` inherits the ordinary
    OOP menu, ``""`` configures no donk sizes, and a non-empty string supplies
    a custom menu.
    """

    flop: StreetBetSizes
    turn: StreetBetSizes
    river: StreetBetSizes
    flop_donk_sizes: str | None
    turn_donk_sizes: str | None
    river_donk_sizes: str | None

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, StreetBetSizes)
            for value in (self.flop, self.turn, self.river)
        ):
            raise OracleValidationError(
                "bet sizing config requires FLOP, TURN, and RIVER menus"
            )
        if self.flop_donk_sizes is not None:
            raise OracleValidationError(
                "flop donk sizes are unsupported and must be explicit None"
            )
        object.__setattr__(
            self,
            "turn_donk_sizes",
            _sizing_text(
                self.turn_donk_sizes,
                "turn donk sizes",
                optional=True,
            ),
        )
        object.__setattr__(
            self,
            "river_donk_sizes",
            _sizing_text(
                self.river_donk_sizes,
                "river donk sizes",
                optional=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class SolveParameters:
    """Pinned engine, tree, units, allocation, and convergence contract."""

    chip_scale: int
    chip_unit: str
    bet_sizes: BetSizingConfig
    add_allin_threshold: Decimal
    force_allin_threshold: Decimal
    merging_threshold: Decimal
    target_exploitability_pct: Decimal
    max_iterations: int
    allocation_mode: AllocationMode
    solver_name: str
    solver_commit: str

    def __post_init__(self) -> None:
        chip_scale = _coerce_engine_units(self.chip_scale, "chip scale")
        if not 1 <= chip_scale <= 1_000_000_000:
            raise OracleValidationError("chip scale must be between 1 and 1,000,000,000")
        if not isinstance(self.chip_unit, str) or not self.chip_unit.strip():
            raise OracleValidationError("chip unit label cannot be empty")
        chip_unit = self.chip_unit.strip()
        if len(chip_unit.encode("utf-8")) > 64 or any(
            ord(character) < 32 or 127 <= ord(character) <= 159
            for character in chip_unit
        ):
            raise OracleValidationError(
                "chip unit label must be control-free and at most 64 bytes"
            )
        if not isinstance(self.bet_sizes, BetSizingConfig):
            raise OracleValidationError("bet_sizes must be BetSizingConfig")

        add_allin = _coerce_decimal(
            self.add_allin_threshold, "add-all-in threshold"
        )
        force_allin = _coerce_decimal(
            self.force_allin_threshold, "force-all-in threshold"
        )
        merging = _coerce_decimal(self.merging_threshold, "merging threshold")
        target = _coerce_decimal(
            self.target_exploitability_pct,
            "target exploitability percent",
        )
        if min(add_allin, force_allin, merging) < 0:
            raise OracleValidationError("tree thresholds cannot be negative")
        if not Decimal(0) < target <= Decimal(100):
            raise OracleValidationError(
                "target exploitability percent must be in (0, 100]"
            )
        if isinstance(self.max_iterations, bool) or not isinstance(
            self.max_iterations, int
        ):
            raise OracleValidationError("max iterations must be an integer")
        if not 1 <= self.max_iterations <= 1_000_000:
            raise OracleValidationError(
                "max iterations must be between 1 and 1,000,000"
            )
        allocation_mode = _coerce_enum(
            self.allocation_mode,
            AllocationMode,
            "allocation mode",
        )
        if not isinstance(self.solver_name, str) or not self.solver_name.strip():
            raise OracleValidationError("pinned solver name cannot be empty")
        solver_name = self.solver_name.strip()
        if not isinstance(self.solver_commit, str) or not re.fullmatch(
            r"[0-9a-f]{40}", self.solver_commit
        ):
            raise OracleValidationError(
                "pinned solver commit must be a lowercase 40-character git SHA"
            )

        object.__setattr__(self, "chip_scale", chip_scale)
        object.__setattr__(self, "chip_unit", chip_unit)
        object.__setattr__(self, "add_allin_threshold", add_allin)
        object.__setattr__(self, "force_allin_threshold", force_allin)
        object.__setattr__(self, "merging_threshold", merging)
        object.__setattr__(self, "target_exploitability_pct", target)
        object.__setattr__(self, "allocation_mode", allocation_mode)
        object.__setattr__(self, "solver_name", solver_name)


@dataclass(frozen=True, slots=True)
class SolveSpec:
    """A reproducible HU postflop solve request."""

    street: Street
    board: tuple[str, ...]
    acting_player: Position
    oop_range: PlayerRange
    ip_range: PlayerRange
    tree: TreeConfig
    parameters: SolveParameters
    players: tuple[Position, ...] = field(
        default=(Position.OOP, Position.IP)
    )
    variant: str = "NLHE"

    def __post_init__(self) -> None:
        street = _coerce_enum(self.street, Street, "street")
        actor = _coerce_enum(self.acting_player, Position, "acting player")
        object.__setattr__(self, "street", street)
        object.__setattr__(self, "acting_player", actor)
        if street is Street.PREFLOP:
            raise UnsupportedGameError("the oracle foundation supports postflop only")
        if self.variant != "NLHE":
            raise UnsupportedGameError("only no-limit Hold'em (NLHE) is supported")

        players = tuple(
            _coerce_enum(player, Position, "player") for player in self.players
        )
        if len(players) != 2 or set(players) != {Position.OOP, Position.IP}:
            raise UnsupportedGameError(
                "only heads-up solves with exactly OOP and IP are supported"
            )
        object.__setattr__(self, "players", (Position.OOP, Position.IP))

        board = tuple(
            _validate_card(card, f"board card {index}")
            for index, card in enumerate(self.board)
        )
        required_cards = {
            Street.FLOP: 3,
            Street.TURN: 4,
            Street.RIVER: 5,
        }[street]
        if len(board) != required_cards:
            raise OracleValidationError(
                f"{street.value} requires exactly {required_cards} board cards"
            )
        if len(board) != len(set(board)):
            raise OracleValidationError("board cards cannot be repeated")
        object.__setattr__(self, "board", board)

        if not isinstance(self.oop_range, PlayerRange) or not isinstance(
            self.ip_range, PlayerRange
        ):
            raise OracleValidationError("oop_range and ip_range must be PlayerRange instances")
        if self.oop_range.position is not Position.OOP:
            raise OracleValidationError("oop_range must be assigned to OOP")
        if self.ip_range.position is not Position.IP:
            raise OracleValidationError("ip_range must be assigned to IP")
        board_set = set(board)
        for player_range in (self.oop_range, self.ip_range):
            for combo in player_range.combos:
                overlap = board_set.intersection(combo.cards)
                if overlap:
                    repeated = ", ".join(sorted(overlap, key=_card_sort_key))
                    raise OracleValidationError(
                        f"{player_range.position.value} combo overlaps board: {repeated}"
                    )
        if not isinstance(self.tree, TreeConfig):
            raise OracleValidationError("tree must be a TreeConfig instance")
        if not isinstance(self.parameters, SolveParameters):
            raise OracleValidationError("parameters must be SolveParameters")

    @property
    def canonical_json(self) -> str:
        """Return deterministic sorted JSON suitable for audit logs."""

        from .serialization import canonical_json

        return canonical_json(self)

    @property
    def cache_key(self) -> str:
        """Return the SHA-256 identity of this exact solve configuration."""

        from .serialization import sha256_key

        return sha256_key(self)


@dataclass(frozen=True, slots=True)
class ActionValue:
    """Oracle frequency and counterfactual EV for one modeled action."""

    action: Action
    frequency: Decimal
    ev: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.action, Action):
            raise OracleValidationError("action value requires an Action")
        frequency = _coerce_decimal(self.frequency, "action frequency")
        ev = _coerce_decimal(self.ev, "action EV")
        if not Decimal(0) <= frequency <= Decimal(1):
            raise OracleValidationError("action frequency must be between 0 and 1")
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "ev", ev)


@dataclass(frozen=True, slots=True)
class ComboPolicy:
    """One acting-player combo's reach, equity, and complete mixed strategy."""

    private_combo: tuple[str, str]
    reach_weight: Decimal
    equity: Decimal
    action_values: tuple[ActionValue, ...]

    def __post_init__(self) -> None:
        combo = _coerce_combo_cards(self.private_combo, "policy private combo")
        reach = _coerce_decimal(self.reach_weight, "policy reach weight")
        equity = _coerce_decimal(self.equity, "policy equity")
        if not Decimal(0) <= reach <= Decimal(1):
            raise OracleValidationError("policy reach weight must be between 0 and 1")
        if not Decimal(0) <= equity <= Decimal(1):
            raise OracleValidationError("policy equity must be between 0 and 1")
        values = tuple(self.action_values)
        if not values:
            raise OracleValidationError("combo policy needs at least one action value")
        if any(not isinstance(value, ActionValue) for value in values):
            raise OracleValidationError(
                "combo policy action_values must contain ActionValue instances"
            )
        actions = [value.action for value in values]
        if len(actions) != len(set(actions)):
            raise OracleValidationError("combo policy cannot repeat an action")
        frequency_sum = sum(
            (value.frequency for value in values),
            start=Decimal(0),
        )
        # The Rust bridge exports f32 strategy vectors; tolerate only their
        # expected round-off while still rejecting incomplete policies.
        if abs(frequency_sum - Decimal(1)) > Decimal("1e-6"):
            raise OracleValidationError("each combo policy's frequencies must sum to 1")
        object.__setattr__(self, "private_combo", combo)
        object.__setattr__(self, "reach_weight", reach)
        object.__setattr__(self, "equity", equity)
        object.__setattr__(
            self,
            "action_values",
            tuple(sorted(values, key=lambda value: _action_sort_key(value.action))),
        )


@dataclass(frozen=True, slots=True)
class SolverMetadata:
    """Auditable metadata emitted by a local solver run.

    ``exploitability`` is measured in integral engine chip units (represented
    as Decimal after the solver's f32 calculation). Percentage-of-pot
    exploitability is retained separately in ``extra`` when supplied by the
    bridge, avoiding an ambiguous unit migration in the SQLite schema.
    """

    solver_name: str
    solver_version: str
    iterations: int
    elapsed_seconds: Decimal
    exploitability: Decimal
    converged: bool
    extra: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.solver_name, str) or not self.solver_name.strip():
            raise OracleValidationError("solver name cannot be empty")
        if not isinstance(self.solver_version, str) or not self.solver_version.strip():
            raise OracleValidationError("solver version cannot be empty")
        if isinstance(self.iterations, bool) or not isinstance(self.iterations, int):
            raise OracleValidationError("iterations must be an integer")
        if self.iterations < 0:
            raise OracleValidationError("iterations cannot be negative")
        elapsed = _coerce_decimal(self.elapsed_seconds, "elapsed seconds")
        exploitability = _coerce_decimal(self.exploitability, "exploitability")
        if elapsed < 0 or exploitability < 0:
            raise OracleValidationError(
                "elapsed seconds and exploitability cannot be negative"
            )
        if not isinstance(self.converged, bool):
            raise OracleValidationError("converged must be boolean")
        object.__setattr__(self, "solver_name", self.solver_name.strip())
        object.__setattr__(self, "solver_version", self.solver_version.strip())
        object.__setattr__(self, "elapsed_seconds", elapsed)
        object.__setattr__(self, "exploitability", exploitability)

        extra = tuple(self.extra)
        for item in extra:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not all(isinstance(value, str) for value in item)
            ):
                raise OracleValidationError("metadata extra must contain string pairs")
        keys = [key for key, _ in extra]
        if len(keys) != len(set(keys)):
            raise OracleValidationError("metadata extra keys must be unique")
        object.__setattr__(self, "extra", tuple(sorted(extra)))


@dataclass(frozen=True, slots=True)
class SolveResult:
    """Per-private-combo strategies and EVs from a local solver node."""

    spec_key: str
    combo_policies: tuple[ComboPolicy, ...]
    metadata: SolverMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.spec_key, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.spec_key
        ):
            raise OracleValidationError("spec key must be a lowercase SHA-256 hex digest")
        policies = tuple(self.combo_policies)
        if not policies:
            raise OracleValidationError("solve result needs at least one combo policy")
        if any(not isinstance(policy, ComboPolicy) for policy in policies):
            raise OracleValidationError(
                "combo_policies must contain ComboPolicy instances"
            )
        combos = [policy.private_combo for policy in policies]
        if len(combos) != len(set(combos)):
            raise OracleValidationError("solve result cannot repeat a private combo")
        if not isinstance(self.metadata, SolverMetadata):
            raise OracleValidationError("metadata must be SolverMetadata")
        object.__setattr__(
            self,
            "combo_policies",
            tuple(sorted(policies, key=lambda policy: policy.private_combo)),
        )

    @classmethod
    def for_spec(
        cls,
        spec: SolveSpec,
        combo_policies: Iterable[ComboPolicy],
        metadata: SolverMetadata,
    ) -> "SolveResult":
        """Build a result already bound to a validated solve specification."""

        if not isinstance(spec, SolveSpec):
            raise OracleValidationError("spec must be a SolveSpec")
        result = cls(spec.cache_key, tuple(combo_policies), metadata)
        validate_result_for_spec(spec, result)
        return result


def validate_result_for_spec(spec: SolveSpec, result: SolveResult) -> None:
    """Validate the one-to-one contract between a spec and its result."""

    if result.spec_key != spec.cache_key:
        raise OracleValidationError("solve result belongs to a different spec")
    if result.metadata.solver_name != spec.parameters.solver_name:
        raise OracleValidationError("solve result solver name differs from the spec")
    if result.metadata.solver_version != spec.parameters.solver_commit:
        raise OracleValidationError("solve result solver commit differs from the spec")
    acting_range = (
        spec.oop_range if spec.acting_player is Position.OOP else spec.ip_range
    )
    expected_combos = {combo.cards for combo in acting_range.combos}
    result_combos = {policy.private_combo for policy in result.combo_policies}
    is_descendant = bool(spec.tree.action_history)
    coverage_matches = (
        bool(result_combos) and result_combos.issubset(expected_combos)
        if is_descendant
        else result_combos == expected_combos
    )
    if not coverage_matches:
        missing_combos = expected_combos - result_combos
        unexpected_combos = result_combos - expected_combos
        details = []
        if missing_combos and not is_descendant:
            details.append(f"missing {len(missing_combos)} acting-range combo policy(s)")
        if unexpected_combos:
            details.append(
                f"contains {len(unexpected_combos)} out-of-range combo policy(s)"
            )
        if is_descendant and not result_combos:
            details.append("contains no positive-reach acting-range combo policies")
        raise OracleValidationError("result combo coverage mismatch: " + ", ".join(details))

    if any(policy.reach_weight <= 0 for policy in result.combo_policies):
        raise OracleValidationError(
            "solve results may contain only positive-reach combo policies"
        )

    reach_sum = sum(
        (policy.reach_weight for policy in result.combo_policies),
        start=Decimal(0),
    )
    if abs(reach_sum - Decimal(1)) > Decimal("1e-6"):
        raise OracleValidationError("acting-player combo reach weights must sum to 1")

    modeled_actions = set(spec.tree.modeled_actions)
    for policy in result.combo_policies:
        result_actions = {value.action for value in policy.action_values}
        if result_actions != modeled_actions:
            missing = modeled_actions - result_actions
            unexpected = result_actions - modeled_actions
            details = []
            if missing:
                details.append(f"missing {len(missing)} modeled action(s)")
            if unexpected:
                details.append(f"contains {len(unexpected)} unexpected action(s)")
            raise OracleValidationError(
                f"result action set mismatch for {policy.private_combo}: "
                + ", ".join(details)
            )
