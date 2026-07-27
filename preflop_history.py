"""Deterministic reconstruction of PokerStudy preflop action histories.

The public blueprint describes decision nodes with histories such as::

    UTG_60%_HJ_Fold_CO_Fold_BTN_Fold_SB_Fold_BB

Position and action tokens alternate and a decision history always ends with the
player whose turn it is.  This module treats that history as a poker state
machine.  It deliberately fails closed when the visible table state can map to
zero or more than one blueprint node.

All chip arithmetic uses :class:`~decimal.Decimal`.  A percentage action follows
the convention used by the provider: first call the current wager, compute the
pot after that call, then add ``percentage * pot_after_call`` to the previous
highest contribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import Iterable, Mapping, Sequence

from preflop_blueprint import BlueprintAction, PokerStudyBlueprintStore


POSITIONS: tuple[str, ...] = ("UTG", "HJ", "CO", "BTN", "SB", "BB")
POSITION_ALIASES: Mapping[str, str] = {"MP": "HJ"}
RANKS_DESC = "AKQJT98765432"
SUITS = "shdc"
_PERCENT_ACTION = re.compile(r"^(\d+(?:\.\d+)?)%$")


def _all_hand_classes() -> tuple[str, ...]:
    classes: list[str] = []
    for first_index, first in enumerate(RANKS_DESC):
        classes.append(first + first)
        for second in RANKS_DESC[first_index + 1 :]:
            classes.append(first + second + "s")
            classes.append(first + second + "o")
    assert len(classes) == 169
    return tuple(classes)


ALL_HAND_CLASSES: tuple[str, ...] = _all_hand_classes()
ALL_HAND_CLASS_SET = frozenset(ALL_HAND_CLASSES)


class PreflopHistoryError(ValueError):
    """Base class for fail-closed history reconstruction failures."""


class InvalidHistoryError(PreflopHistoryError):
    """A provider history has invalid syntax or actor ordering."""


class IllegalActionError(PreflopHistoryError):
    """An action is not legal in the reconstructed public state."""


class NoMatchingHistoryError(PreflopHistoryError):
    """No blueprint history matches the observed table state."""


class AmbiguousHistoryError(PreflopHistoryError):
    """More than one blueprint history matches the observed table state."""


class ReachError(PreflopHistoryError):
    """Blueprint reach weights are inconsistent with the selected path."""


class ZeroHeroReachError(ReachError):
    """The observed Hero combo has zero blueprint reach at this node."""


def as_decimal(value: object, *, field: str = "value") -> Decimal:
    """Convert a JSON/vision number to Decimal without binary-float leakage."""

    if isinstance(value, Decimal):
        result = value
    else:
        try:
            result = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise PreflopHistoryError(f"{field} is not a decimal: {value!r}") from exc
    if not result.is_finite():
        raise PreflopHistoryError(f"{field} must be finite")
    return result


def canonical_position(position: str) -> str:
    value = str(position).strip().upper()
    value = POSITION_ALIASES.get(value, value)
    if value not in POSITIONS:
        raise InvalidHistoryError(f"unknown preflop position: {position!r}")
    return value


@dataclass(frozen=True)
class ParsedAction:
    label: str
    kind: str
    size_pct: Decimal | None = None


def parse_action_label(label: str) -> ParsedAction:
    raw = str(label).strip()
    named = {
        "fold": ("Fold", "fold"),
        "call": ("Call", "call"),
        "check": ("Check", "check"),
        "ai": ("AI", "all_in"),
        "allin": ("AI", "all_in"),
        "all-in": ("AI", "all_in"),
    }
    if raw.lower() in named:
        canonical, kind = named[raw.lower()]
        return ParsedAction(canonical, kind)
    match = _PERCENT_ACTION.fullmatch(raw)
    if not match:
        raise InvalidHistoryError(f"unknown preflop action: {label!r}")
    percentage = as_decimal(match.group(1), field="raise percentage")
    if percentage <= 0:
        raise InvalidHistoryError("raise percentage must be positive")
    normalized = format(percentage.normalize(), "f")
    return ParsedAction(f"{normalized}%", "raise", percentage)


@dataclass(frozen=True)
class ActionStep:
    actor: str
    action: str


@dataclass(frozen=True)
class ParsedHistory:
    raw: str
    steps: tuple[ActionStep, ...]
    decision_actor: str | None


def parse_history(history: str, *, completed: bool = False) -> ParsedHistory:
    """Parse a decision history, or an action-complete path when requested."""

    raw = str(history).strip()
    if not raw:
        raise InvalidHistoryError("preflop history is empty")
    tokens = raw.split("_")
    expected_even = completed
    if (len(tokens) % 2 == 0) != expected_even:
        suffix = "an action" if completed else "an actor"
        raise InvalidHistoryError(f"preflop history must end with {suffix}: {raw!r}")

    positions = [canonical_position(token) for token in tokens[0::2]]
    if positions[0] != "UTG":
        raise InvalidHistoryError("a six-max provider history must start at UTG")
    actions = [parse_action_label(token).label for token in tokens[1::2]]
    steps = tuple(ActionStep(actor, action) for actor, action in zip(positions, actions))
    decision_actor = None if completed else positions[-1]
    return ParsedHistory(raw=raw, steps=steps, decision_actor=decision_actor)


def _position_items(values: Mapping[str, object]) -> tuple[tuple[str, Decimal], ...]:
    converted: dict[str, Decimal] = {}
    for raw_position, raw_value in values.items():
        position = canonical_position(raw_position)
        if position in converted:
            raise PreflopHistoryError(f"duplicate observed position: {position}")
        amount = as_decimal(raw_value, field=f"contribution[{position}]")
        if amount < 0:
            raise PreflopHistoryError("observed contributions cannot be negative")
        converted[position] = amount
    return tuple((position, converted[position]) for position in POSITIONS if position in converted)


@dataclass(frozen=True)
class ObservedPreflopState:
    """Public values read from the table; omitted contributions are wildcards."""

    actor: str | None
    contributions: tuple[tuple[str, Decimal], ...]
    folded: frozenset[str] | None
    survivors: frozenset[str] | None = None
    all_in: frozenset[str] | None = None

    @classmethod
    def create(
        cls,
        *,
        actor: str | None = None,
        contributions: Mapping[str, object] | None = None,
        folded: Iterable[str] | None = None,
        survivors: Iterable[str] | None = None,
        all_in: Iterable[str] | None = None,
    ) -> "ObservedPreflopState":
        canonical_folded = (
            None if folded is None else frozenset(canonical_position(p) for p in folded)
        )
        canonical_survivors = (
            None
            if survivors is None
            else frozenset(canonical_position(p) for p in survivors)
        )
        canonical_all_in = (
            None if all_in is None else frozenset(canonical_position(p) for p in all_in)
        )
        if canonical_folded is not None and canonical_survivors is not None:
            if canonical_folded & canonical_survivors:
                raise PreflopHistoryError("a position cannot be both folded and surviving")
        if canonical_folded is not None and canonical_all_in is not None:
            if canonical_folded & canonical_all_in:
                raise PreflopHistoryError("a position cannot be both folded and all-in")
        return cls(
            actor=None if actor is None else canonical_position(actor),
            contributions=_position_items(contributions or {}),
            folded=canonical_folded,
            survivors=canonical_survivors,
            all_in=canonical_all_in,
        )

    @property
    def contribution_map(self) -> dict[str, Decimal]:
        return dict(self.contributions)


@dataclass(frozen=True)
class PreflopPublicState:
    contributions: tuple[tuple[str, Decimal], ...]
    folded: frozenset[str]
    all_in: frozenset[str]
    pending: tuple[str, ...]
    actor: str | None
    pot: Decimal
    highest_bet: Decimal
    last_full_raise: Decimal
    terminal: bool

    @property
    def contribution_map(self) -> dict[str, Decimal]:
        return dict(self.contributions)

    @property
    def live_positions(self) -> frozenset[str]:
        return frozenset(POSITIONS) - self.folded


@dataclass(frozen=True)
class PreflopResolution:
    stack: int
    history: str
    steps: tuple[ActionStep, ...]
    state: PreflopPublicState
    matched_spot_history: str
    final_action: str | None = None


@dataclass(frozen=True)
class ReachResolution:
    stack: int
    history: str
    reaches: tuple[tuple[str, tuple[tuple[str, Decimal], ...]], ...]
    artifact_nodes: tuple[str, ...]

    def for_position(self, position: str) -> dict[str, Decimal]:
        wanted = canonical_position(position)
        for candidate, weights in self.reaches:
            if candidate == wanted:
                return dict(weights)
        raise ReachError(f"reach is missing position {wanted}")


def _seat_order_after(actor: str, eligible: set[str]) -> list[str]:
    index = POSITIONS.index(actor)
    ordered = POSITIONS[index + 1 :] + POSITIONS[:index]
    return [position for position in ordered if position in eligible]


class _MutableState:
    def __init__(self, stack: Decimal) -> None:
        if stack < Decimal("1"):
            raise IllegalActionError("blueprint stack must be at least 1 BB")
        self.stack = stack
        self.contributions = {position: Decimal("0") for position in POSITIONS}
        self.contributions["SB"] = Decimal("0.5")
        self.contributions["BB"] = Decimal("1")
        self.folded: set[str] = set()
        self.all_in: set[str] = set()
        self.pending = list(POSITIONS)
        self.highest = Decimal("1")
        self.last_full_raise = Decimal("1")
        self.terminal = False

    @property
    def actor(self) -> str | None:
        return self.pending[0] if self.pending and not self.terminal else None

    def _remove_pending_actor(self, actor: str) -> None:
        if not self.pending or self.pending[0] != actor:
            raise InvalidHistoryError(f"{actor} acted out of order")
        self.pending.pop(0)

    def _after_action(self) -> None:
        live = set(POSITIONS) - self.folded
        if len(live) <= 1:
            self.pending.clear()
            self.terminal = True
            return
        self.pending = [
            position
            for position in self.pending
            if position not in self.folded and position not in self.all_in
        ]
        if not self.pending:
            self.terminal = True

    def _raise(self, actor: str, target: Decimal, *, all_in: bool) -> None:
        previous_highest = self.highest
        if target <= previous_highest:
            raise IllegalActionError(f"{actor} raise-to must exceed {previous_highest}")
        if target > self.stack:
            raise IllegalActionError(f"{actor} cannot contribute more than {self.stack} BB")
        raise_size = target - previous_highest
        if not all_in and raise_size < self.last_full_raise:
            raise IllegalActionError(
                f"{actor} raise increment {raise_size} is below minimum {self.last_full_raise}"
            )
        self.contributions[actor] = target
        self.highest = target
        if raise_size >= self.last_full_raise:
            self.last_full_raise = raise_size
        if all_in:
            self.all_in.add(actor)
        eligible = set(POSITIONS) - self.folded - self.all_in - {actor}
        self.pending = _seat_order_after(actor, eligible)

    def apply(self, actor: str, label: str) -> None:
        if self.terminal:
            raise IllegalActionError("an action appears after the preflop round ended")
        if self.actor != actor:
            raise InvalidHistoryError(f"expected {self.actor}, got {actor}")
        parsed = parse_action_label(label)
        current = self.contributions[actor]

        if parsed.kind == "fold":
            if current >= self.highest:
                raise IllegalActionError(f"{actor} cannot fold when checking is free")
            self._remove_pending_actor(actor)
            self.folded.add(actor)
        elif parsed.kind == "check":
            if current != self.highest:
                raise IllegalActionError(f"{actor} cannot check facing {self.highest - current} BB")
            self._remove_pending_actor(actor)
        elif parsed.kind == "call":
            if current > self.highest:
                raise IllegalActionError(
                    f"{actor} contribution exceeds the current highest bet"
                )
            if current < self.highest:
                if self.highest > self.stack:
                    raise IllegalActionError(f"{actor} cannot call beyond the blueprint stack")
                self.contributions[actor] = self.highest
                if self.highest == self.stack:
                    self.all_in.add(actor)
            # PokerStudy labels the unraised BB's passive option `Call` rather
            # than `Check`.  At equal contributions it is a state-machine check,
            # but the source label remains Call so the matching action weights
            # are retrieved from the correct artifact branch.
            self._remove_pending_actor(actor)
        elif parsed.kind == "raise":
            assert parsed.size_pct is not None
            call_amount = self.highest - current
            if call_amount < 0:
                raise IllegalActionError("actor contribution exceeds the current highest bet")
            pot_after_call = sum(self.contributions.values()) + call_amount
            increment = pot_after_call * parsed.size_pct / Decimal("100")
            target = self.highest + increment
            self._raise(actor, target, all_in=False)
        elif parsed.kind == "all_in":
            target = self.stack
            if target <= current:
                raise IllegalActionError(f"{actor} is already all-in")
            if target <= self.highest:
                self.contributions[actor] = target
                self.all_in.add(actor)
                self._remove_pending_actor(actor)
            else:
                self._raise(actor, target, all_in=True)
        else:  # pragma: no cover - ParsedAction is a closed set.
            raise AssertionError(parsed.kind)
        self._after_action()

    def freeze(self) -> PreflopPublicState:
        return PreflopPublicState(
            contributions=tuple((position, self.contributions[position]) for position in POSITIONS),
            folded=frozenset(self.folded),
            all_in=frozenset(self.all_in),
            pending=tuple(self.pending),
            actor=self.actor,
            pot=sum(self.contributions.values()),
            highest_bet=self.highest,
            last_full_raise=self.last_full_raise,
            terminal=self.terminal,
        )


def simulate_history(history: str, *, stack: object, completed: bool = False) -> tuple[ParsedHistory, PreflopPublicState]:
    parsed = parse_history(history, completed=completed)
    state = _MutableState(as_decimal(stack, field="stack"))
    for index, step in enumerate(parsed.steps):
        state.apply(step.actor, step.action)
        if not completed:
            # The next position token is part of the provider history.  Verify it
            # immediately instead of merely comparing the final actor.
            next_index = index + 1
            if next_index < len(parsed.steps):
                expected = parsed.steps[next_index].actor
            else:
                expected = parsed.decision_actor
            if state.actor != expected:
                raise InvalidHistoryError(
                    f"history says {expected} acts after {step.actor} {step.action}; "
                    f"legal next actor is {state.actor}"
                )
    frozen = state.freeze()
    if completed and not frozen.terminal:
        raise IllegalActionError("completed path does not end the preflop betting round")
    if not completed and frozen.actor != parsed.decision_actor:
        raise InvalidHistoryError(
            f"history ends at {parsed.decision_actor}, legal actor is {frozen.actor}"
        )
    return parsed, frozen


def _compatible(
    state: PreflopPublicState,
    observed: ObservedPreflopState,
    tolerance: Decimal | Mapping[str, Decimal],
    *,
    terminal: bool,
) -> bool:
    if observed.actor is not None and state.actor != observed.actor:
        return False
    if observed.folded is not None and state.folded != observed.folded:
        return False
    if observed.survivors is not None and state.live_positions != observed.survivors:
        return False
    if observed.all_in is not None and state.all_in != observed.all_in:
        return False
    if state.terminal != terminal:
        return False
    actual = state.contribution_map
    for position, value in observed.contributions:
        allowed = tolerance[position] if isinstance(tolerance, Mapping) else tolerance
        if abs(actual[position] - value) > allowed:
            return False
    return True


def _coerce_tolerances(
    value: object,
    observed_positions: Iterable[str],
) -> Decimal | dict[str, Decimal]:
    if not isinstance(value, Mapping):
        tolerance = as_decimal(value, field="contribution tolerance")
        if tolerance < 0:
            raise PreflopHistoryError("contribution tolerance cannot be negative")
        return tolerance
    result: dict[str, Decimal] = {}
    for raw_position in observed_positions:
        position = canonical_position(raw_position)
        if position not in value:
            raise PreflopHistoryError(f"missing contribution tolerance for {position}")
        tolerance = as_decimal(
            value[position], field=f"contribution tolerance[{position}]"
        )
        if tolerance < 0:
            raise PreflopHistoryError("contribution tolerances cannot be negative")
        result[position] = tolerance
    return result


def _tolerance_for(
    tolerance: Decimal | Mapping[str, Decimal], position: str
) -> Decimal:
    return tolerance[position] if isinstance(tolerance, Mapping) else tolerance


def _spot_history(spot: object) -> str:
    history = getattr(spot, "history", None)
    if not isinstance(history, str):
        raise InvalidHistoryError("blueprint spot is missing a string history")
    return history


def _action_label(action: BlueprintAction) -> str:
    label = getattr(action, "label", None)
    if label is None:
        # Kept for a clear integration error if an older schema used `action`.
        label = getattr(action, "action", None)
    if not isinstance(label, str):
        raise ReachError("blueprint action is missing its label")
    return parse_action_label(label).label


def _node_actions(node: object) -> Sequence[BlueprintAction]:
    actions = getattr(node, "actions", None)
    if not isinstance(actions, (tuple, list)) or not actions:
        raise ReachError("blueprint node has no actions")
    return actions


def select_blueprint_action(node: object, label: str) -> BlueprintAction:
    wanted = parse_action_label(label).label
    matches = [action for action in _node_actions(node) if _action_label(action) == wanted]
    if not matches:
        raise ReachError(f"node {getattr(node, 'history', '?')} has no action {wanted}")
    if len(matches) > 1:
        raise ReachError(f"node has duplicate action label {wanted}")
    return matches[0]


class PreflopHistoryResolver:
    """Map observed public states to unique blueprint paths and reach ranges."""

    def __init__(
        self,
        store: PokerStudyBlueprintStore,
        *,
        contribution_tolerance: object = Decimal("0.05"),
        # PokerStudy publishes weights rounded to 0.005 increments.  The same
        # small allowance used by the payload validator is needed when a
        # cumulative reach is compared with its parent node.
        reach_tolerance: object = Decimal("0.01"),
    ) -> None:
        self.store = store
        self.contribution_tolerance = as_decimal(
            contribution_tolerance, field="contribution tolerance"
        )
        self.reach_tolerance = as_decimal(reach_tolerance, field="reach tolerance")
        if self.contribution_tolerance < 0 or self.reach_tolerance < 0:
            raise PreflopHistoryError("tolerances cannot be negative")

    def resolve_decision(
        self,
        *,
        stack: int,
        expected_actor: str,
        observed_contributions: Mapping[str, object] | None = None,
        observed_folded: Iterable[str] | None = None,
        observed_all_in: Iterable[str] | None = None,
        tolerance: object | None = None,
    ) -> PreflopResolution:
        actor = canonical_position(expected_actor)
        observed = ObservedPreflopState.create(
            actor=actor,
            contributions=observed_contributions,
            folded=observed_folded,
            all_in=observed_all_in,
        )
        allowed = _coerce_tolerances(
            self.contribution_tolerance if tolerance is None else tolerance,
            (position for position, _ in observed.contributions),
        )
        matches: list[PreflopResolution] = []
        for spot in self.store.spots(stack):
            history = _spot_history(spot)
            if canonical_position(history.split("_")[-1]) != actor:
                continue
            parsed, state = simulate_history(history, stack=stack)
            if _compatible(state, observed, allowed, terminal=False):
                matches.append(
                    PreflopResolution(
                        stack=stack,
                        history=history,
                        steps=parsed.steps,
                        state=state,
                        matched_spot_history=history,
                    )
                )
        return self._require_unique(matches, observed, kind="decision")

    def resolve_completed_path(
        self, *, stack: int, decision_history: str, final_action: str
    ) -> PreflopResolution:
        canonical_action = parse_action_label(final_action).label
        node = self.store.node(stack, decision_history)
        select_blueprint_action(node, canonical_action)
        completed_history = f"{decision_history}_{canonical_action}"
        parsed, state = simulate_history(completed_history, stack=stack, completed=True)
        return PreflopResolution(
            stack=stack,
            history=completed_history,
            steps=parsed.steps,
            state=state,
            matched_spot_history=decision_history,
            final_action=canonical_action,
        )

    def resolve_hu_handoff(
        self,
        *,
        stack: int,
        observed_contributions: Mapping[str, object],
        observed_folded: Iterable[str],
        survivors: Iterable[str] | None = None,
        observed_all_in: Iterable[str] | None = None,
        tolerance: object | None = None,
    ) -> PreflopResolution:
        observed = ObservedPreflopState.create(
            contributions=observed_contributions,
            folded=observed_folded,
            survivors=survivors,
            all_in=observed_all_in,
        )
        expected_survivors = (
            observed.survivors
            if observed.survivors is not None
            else frozenset(POSITIONS) - (observed.folded or frozenset())
        )
        if len(expected_survivors) != 2:
            raise NoMatchingHistoryError("HU handoff requires exactly two surviving positions")
        return self._resolve_terminal_handoff(
            stack=stack,
            observed=observed,
            tolerance=tolerance,
            kind="HU terminal",
        )

    def resolve_terminal_handoff(
        self,
        *,
        stack: int,
        observed_contributions: Mapping[str, object],
        observed_folded: Iterable[str],
        survivors: Iterable[str] | None = None,
        observed_all_in: Iterable[str] | None = None,
        tolerance: object | None = None,
    ) -> PreflopResolution:
        """Resolve a completed preflop round with two or more survivors.

        This is used by the explicitly approximate HU projection when a hand
        reached the flop multiway but later observations prove that only Hero
        and one opponent remain.  It resolves only the preflop terminal path;
        it does not pretend the later fold was part of that preflop tree.
        """

        observed = ObservedPreflopState.create(
            contributions=observed_contributions,
            folded=observed_folded,
            survivors=survivors,
            all_in=observed_all_in,
        )
        expected_survivors = (
            observed.survivors
            if observed.survivors is not None
            else frozenset(POSITIONS) - (observed.folded or frozenset())
        )
        if not 2 <= len(expected_survivors) <= len(POSITIONS):
            raise NoMatchingHistoryError(
                "terminal handoff requires between two and six surviving positions"
            )
        return self._resolve_terminal_handoff(
            stack=stack,
            observed=observed,
            tolerance=tolerance,
            kind="terminal",
        )

    def _resolve_terminal_handoff(
        self,
        *,
        stack: int,
        observed: ObservedPreflopState,
        tolerance: object | None,
        kind: str,
    ) -> PreflopResolution:
        expected_survivors = (
            observed.survivors
            if observed.survivors is not None
            else frozenset(POSITIONS) - (observed.folded or frozenset())
        )
        allowed = _coerce_tolerances(
            self.contribution_tolerance if tolerance is None else tolerance,
            (position for position, _ in observed.contributions),
        )
        observed_amounts = observed.contribution_map
        matches: list[PreflopResolution] = []

        for spot in self.store.spots(stack):
            history = _spot_history(spot)
            parsed, before = simulate_history(history, stack=stack)
            actor = before.actor
            if actor is None:
                continue
            final_folded = observed.folded or frozenset()
            # A terminal HU round has two forms:
            #   * the last pending survivor calls/checks, or
            #   * the last pending third player folds, leaving two survivors.
            nonfold_frontier = (
                actor in expected_survivors
                and before.folded == final_folded
                and before.live_positions == expected_survivors
            )
            fold_frontier = (
                actor in final_folded
                and before.folded == final_folded - {actor}
                and before.live_positions == expected_survivors | {actor}
            )
            if not (nonfold_frontier or fold_frontier):
                continue
            before_amounts = before.contribution_map
            impossible = False
            for position, observed_amount in observed_amounts.items():
                current = before_amounts[position]
                position_tolerance = _tolerance_for(allowed, position)
                if position == actor:
                    if current > observed_amount + position_tolerance:
                        impossible = True
                        break
                elif abs(current - observed_amount) > position_tolerance:
                    impossible = True
                    break
            if impossible:
                continue

            node = self.store.node(stack, history)
            for action in _node_actions(node):
                label = _action_label(action)
                kind = parse_action_label(label).kind
                if kind == "fold" and not fold_frontier:
                    continue
                if kind != "fold" and not nonfold_frontier:
                    continue
                completed_history = f"{history}_{label}"
                try:
                    complete_parsed, state = simulate_history(
                        completed_history, stack=stack, completed=True
                    )
                except IllegalActionError:
                    continue
                if state.live_positions != expected_survivors:
                    continue
                if _compatible(state, observed, allowed, terminal=True):
                    matches.append(
                        PreflopResolution(
                            stack=stack,
                            history=completed_history,
                            steps=complete_parsed.steps,
                            state=state,
                            matched_spot_history=history,
                            final_action=label,
                        )
                    )
        return self._require_unique(matches, observed, kind=kind)

    @staticmethod
    def _require_unique(
        matches: Sequence[PreflopResolution],
        observed: ObservedPreflopState,
        *,
        kind: str,
    ) -> PreflopResolution:
        if not matches:
            raise NoMatchingHistoryError(f"no {kind} blueprint path matches {observed}")
        if len(matches) > 1:
            examples = ", ".join(match.history for match in matches[:3])
            raise AmbiguousHistoryError(
                f"{len(matches)} {kind} blueprint paths match the observation: {examples}"
            )
        return matches[0]

    def walk_reaches(
        self,
        resolution: PreflopResolution,
        *,
        hero_position: str | None = None,
        hero_cards: Sequence[str] | None = None,
    ) -> ReachResolution:
        reaches: dict[str, dict[str, Decimal]] = {
            position: {hand_class: Decimal("1") for hand_class in ALL_HAND_CLASSES}
            for position in POSITIONS
        }
        node_histories: list[str] = []
        prefix_tokens: list[str] = ["UTG"]

        for index, step in enumerate(resolution.steps):
            prefix = "_".join(prefix_tokens)
            if prefix.split("_")[-1] != step.actor:
                raise ReachError(f"reach path expected {prefix}, got actor {step.actor}")
            node = self.store.node(resolution.stack, prefix)
            node_actor = canonical_position(getattr(node, "actor", ""))
            node_history = getattr(node, "history", prefix)
            if node_actor != step.actor or node_history != prefix:
                raise ReachError(f"blueprint node identity mismatch at {prefix}")
            node_actions = _node_actions(node)
            branch_totals = {hand_class: Decimal(0) for hand_class in ALL_HAND_CLASSES}
            for candidate in node_actions:
                for hand_class, value in _weights_map(candidate).items():
                    branch_totals[hand_class] += value
            prior = reaches[step.actor]
            for hand_class in ALL_HAND_CLASSES:
                if abs(branch_totals[hand_class] - prior[hand_class]) > self.reach_tolerance:
                    raise ReachError(
                        f"action branches do not conserve reach for {step.actor} "
                        f"{hand_class}: {branch_totals[hand_class]} vs {prior[hand_class]}"
                    )

            action = select_blueprint_action(node, step.action)
            weights = _weights_map(action)
            replacement: dict[str, Decimal] = {}
            for hand_class in ALL_HAND_CLASSES:
                value = weights.get(hand_class, Decimal("0"))
                if value < 0 or value > Decimal("1") + self.reach_tolerance:
                    raise ReachError(
                        f"invalid cumulative reach {value} for {step.actor} {hand_class}"
                    )
                if value > prior[hand_class] + self.reach_tolerance:
                    raise ReachError(
                        f"cumulative reach increased for {step.actor} {hand_class}: "
                        f"{prior[hand_class]} -> {value}"
                    )
                # Provider weights are rounded, so a child can exceed its
                # parent by a tolerated sliver.  Keep the cumulative invariant
                # exact after validation: tolerated upward drift is clamped to
                # the parent, and a zero-reach class can never be resurrected.
                replacement[hand_class] = min(
                    value,
                    prior[hand_class],
                    Decimal("1"),
                )
            reaches[step.actor] = replacement
            node_histories.append(prefix)
            prefix_tokens.append(parse_action_label(step.action).label)
            if index + 1 < len(resolution.steps):
                prefix_tokens.append(resolution.steps[index + 1].actor)

        result = ReachResolution(
            stack=resolution.stack,
            history=resolution.history,
            reaches=tuple(
                (
                    position,
                    tuple((hand_class, reaches[position][hand_class]) for hand_class in ALL_HAND_CLASSES),
                )
                for position in POSITIONS
            ),
            artifact_nodes=tuple(node_histories),
        )
        if hero_position is not None or hero_cards is not None:
            if hero_position is None or hero_cards is None:
                raise ReachError("hero_position and hero_cards must be supplied together")
            if len(hero_cards) != 2:
                raise ReachError("Hero must have exactly two cards")
            hand_class = hand_class_for_cards(hero_cards[0], hero_cards[1])
            if result.for_position(hero_position)[hand_class] <= 0:
                raise ZeroHeroReachError(
                    f"Hero {canonical_position(hero_position)} {hand_class} has zero blueprint reach"
                )
        return result


def _weights_map(action: object) -> dict[str, Decimal]:
    raw = getattr(action, "weights", None)
    if isinstance(raw, Mapping):
        items = raw.items()
    elif isinstance(raw, (tuple, list)):
        items = raw
    else:
        raise ReachError("blueprint action weights are not a mapping/pair sequence")
    result: dict[str, Decimal] = {}
    for hand_class, raw_weight in items:
        name = str(hand_class)
        if name not in ALL_HAND_CLASS_SET:
            raise ReachError(f"unknown hand class in blueprint weights: {name}")
        if name in result:
            raise ReachError(f"duplicate blueprint weight for {name}")
        result[name] = as_decimal(raw_weight, field=f"weight[{name}]")
    return result


def _normalize_card(card: str) -> str:
    raw = str(card).strip()
    if len(raw) == 3 and raw[:2] == "10":
        raw = "T" + raw[2]
    if len(raw) != 2:
        raise ReachError(f"invalid card: {card!r}")
    rank = raw[0].upper()
    suit = raw[1].lower()
    if rank not in RANKS_DESC or suit not in SUITS:
        raise ReachError(f"invalid card: {card!r}")
    return rank + suit


def hand_class_for_cards(first_card: str, second_card: str) -> str:
    first = _normalize_card(first_card)
    second = _normalize_card(second_card)
    if first == second:
        raise ReachError("the same card cannot appear twice")
    rank_a, suit_a = first
    rank_b, suit_b = second
    if rank_a == rank_b:
        return rank_a + rank_b
    if RANKS_DESC.index(rank_a) < RANKS_DESC.index(rank_b):
        high, low = rank_a, rank_b
    else:
        high, low = rank_b, rank_a
    return high + low + ("s" if suit_a == suit_b else "o")


def expand_class_weights(
    class_weights: Mapping[str, object] | Sequence[tuple[str, object]],
    *,
    dead_cards: Iterable[str] = (),
) -> dict[tuple[str, str], Decimal]:
    """Expand 169-class reach weights into exact unordered card combinations."""

    items = class_weights.items() if isinstance(class_weights, Mapping) else class_weights
    weights: dict[str, Decimal] = {}
    for hand_class, raw_weight in items:
        if hand_class not in ALL_HAND_CLASS_SET:
            raise ReachError(f"unknown hand class: {hand_class}")
        if hand_class in weights:
            raise ReachError(f"duplicate hand class: {hand_class}")
        value = as_decimal(raw_weight, field=f"weight[{hand_class}]")
        if value < 0 or value > 1:
            raise ReachError(f"hand-class weight must be in [0, 1]: {hand_class}={value}")
        weights[hand_class] = value

    dead_list = [_normalize_card(card) for card in dead_cards]
    if len(dead_list) != len(set(dead_list)):
        raise ReachError("dead cards contain a duplicate")
    dead = set(dead_list)
    combos: dict[tuple[str, str], Decimal] = {}

    for hand_class, weight in weights.items():
        if weight == 0:
            continue
        if len(hand_class) == 2:  # Pair.
            rank = hand_class[0]
            candidates = [
                (rank + SUITS[left], rank + SUITS[right])
                for left in range(len(SUITS))
                for right in range(left + 1, len(SUITS))
            ]
        else:
            high, low, texture = hand_class
            if texture == "s":
                candidates = [(high + suit, low + suit) for suit in SUITS]
            else:
                candidates = [
                    (high + high_suit, low + low_suit)
                    for high_suit in SUITS
                    for low_suit in SUITS
                    if high_suit != low_suit
                ]
        for combo in candidates:
            if combo[0] in dead or combo[1] in dead:
                continue
            combos[combo] = weight
    return combos


def expand_position_reach(
    reach: ReachResolution,
    position: str,
    *,
    dead_cards: Iterable[str] = (),
) -> dict[tuple[str, str], Decimal]:
    return expand_class_weights(reach.for_position(position), dead_cards=dead_cards)


__all__ = [
    "ALL_HAND_CLASSES",
    "POSITIONS",
    "ActionStep",
    "AmbiguousHistoryError",
    "IllegalActionError",
    "InvalidHistoryError",
    "NoMatchingHistoryError",
    "ObservedPreflopState",
    "ParsedHistory",
    "PreflopHistoryError",
    "PreflopHistoryResolver",
    "PreflopPublicState",
    "PreflopResolution",
    "ReachError",
    "ReachResolution",
    "ZeroHeroReachError",
    "canonical_position",
    "expand_class_weights",
    "expand_position_reach",
    "hand_class_for_cards",
    "parse_action_label",
    "parse_history",
    "select_blueprint_action",
    "simulate_history",
]
