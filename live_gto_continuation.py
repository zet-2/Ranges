"""Build a verified flop-to-current solve from a complete public hand.

This module is the application-layer bridge between the lossless public event
recorder and the solver-neutral continuation models.  It never infers a
missing action: every path step comes from a replayed ``PublicHandHistory``.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, ROUND_HALF_UP

from gto_hand_history import (
    PublicHandHistory,
    PublicHandHistoryError,
    replay_public_hand,
)
from gto_oracle import (
    Action,
    ActionKind,
    AllocationMode,
    BetSizingConfig,
    ContinuationAction,
    ContinuationDeal,
    ContinuationSpec,
    PlayerBetSizes,
    Position,
    SolveParameters,
    StreetBetSizes,
)
from live_gto import (
    LiveDecisionState,
    LiveGTOConfig,
    PINNED_SOLVER_COMMIT,
    RangeBundle,
)


class LiveGTOContinuationError(ValueError):
    """A complete transcript cannot define one supported HU continuation."""


def _units(value: Decimal, scale: int, field: str) -> int:
    if not isinstance(value, Decimal):
        value = Decimal(str(value))
    if not value.is_finite() or value < 0:
        raise LiveGTOContinuationError(
            f"{field} must be finite and non-negative"
        )
    result = int(
        (value * scale).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    )
    if value > 0 and result <= 0:
        raise LiveGTOContinuationError(
            f"{field} rounds to zero solver units"
        )
    return result


def _canonical_live_actions(actions: tuple[str, ...]) -> set[str]:
    result = set()
    for action in actions:
        text = action.strip().upper().replace("-", "_").replace(" ", "_")
        if text == "ALLIN":
            text = "ALL_IN"
        if text:
            result.add(text)
    return result


def _continuation_sizes(config: LiveGTOConfig) -> BetSizingConfig:
    size = f"{format(config.bet_size_pct.normalize(), 'f')}%"
    standard = PlayerBetSizes(size, "2.5x")
    street = StreetBetSizes(standard, standard)
    return BetSizingConfig(
        flop=street,
        turn=street,
        river=street,
        flop_donk_sizes=None,
        # ``None`` inherits the ordinary OOP menu. A complete continuation
        # must permit an OOP lead after either player called the prior street.
        turn_donk_sizes=None,
        river_donk_sizes=None,
    )


def _current_node_actions(
    state: LiveDecisionState,
    replayed,
    *,
    hero_seat: int,
    villain_seat: int,
    config: LiveGTOConfig,
) -> tuple[int, tuple[ActionKind, ...], tuple[Action, ...]]:
    scale = config.chip_scale
    visible = _canonical_live_actions(state.legal_actions)
    actor_contribution = replayed.street_contribution_map[hero_seat]
    villain_contribution = replayed.street_contribution_map[villain_seat]
    highest = max(actor_contribution, villain_contribution)
    facing_bb = highest - actor_contribution
    if facing_bb != replayed.amount_to_call_bb:
        raise LiveGTOContinuationError(
            "replayed street contributions disagree with the call amount"
        )
    facing = _units(facing_bb, scale, "continuation facing bet")
    current_effective_to = _units(
        min(
            replayed.stack_map[hero_seat] + actor_contribution,
            replayed.stack_map[villain_seat] + villain_contribution,
        ),
        scale,
        "current effective street target",
    )
    if current_effective_to <= 0:
        raise LiveGTOContinuationError(
            "the current actor has no effective chips available"
        )

    if facing == 0:
        if "CHECK" not in visible:
            raise LiveGTOContinuationError(
                "visible controls do not authorize the replayed check"
            )
        pot = _units(replayed.pot_bb, scale, "current pot")
        requested = int(
            (
                Decimal(pot)
                * config.bet_size_pct
                / Decimal(100)
            ).quantize(Decimal(1), rounding=ROUND_HALF_UP)
        )
        requested = max(1, min(requested, current_effective_to))
        aggressive = (
            Action(ActionKind.ALL_IN, current_effective_to)
            if requested == current_effective_to
            else Action(ActionKind.BET, requested)
        )
        if aggressive.kind is ActionKind.BET and "BET" not in visible:
            raise LiveGTOContinuationError(
                "visible controls do not authorize the modeled bet"
            )
        if (
            aggressive.kind is ActionKind.ALL_IN
            and not ({"BET", "ALL_IN"} & visible)
        ):
            raise LiveGTOContinuationError(
                "visible controls do not authorize the modeled all-in"
            )
        return (
            0,
            (ActionKind.CHECK, aggressive.kind),
            (Action(ActionKind.CHECK), aggressive),
        )

    if not {"FOLD", "CALL"}.issubset(visible):
        raise LiveGTOContinuationError(
            "visible controls do not authorize fold and call"
        )
    legal_kinds: list[ActionKind] = [ActionKind.FOLD, ActionKind.CALL]
    modeled: list[Action] = [
        Action(ActionKind.FOLD),
        Action(ActionKind.CALL),
    ]
    replay_legal = set(replayed.legal_actions)
    highest_units = _units(highest, scale, "highest street contribution")
    can_aggress = bool({"RAISE_TO", "ALL_IN_TO"} & replay_legal)
    if can_aggress and current_effective_to > highest_units:
        if not ({"RAISE", "ALL_IN"} & visible):
            raise LiveGTOContinuationError(
                "visible controls do not authorize the replayed raise option"
            )
        minimum_raise = (
            _units(
                replayed.minimum_raise_to_bb,
                scale,
                "minimum raise-to",
            )
            if replayed.minimum_raise_to_bb is not None
            else current_effective_to
        )
        requested_raise = int(
            (Decimal(highest_units) * Decimal("2.5")).quantize(
                Decimal(1),
                rounding=ROUND_HALF_UP,
            )
        )
        raise_to = min(
            current_effective_to,
            max(minimum_raise, requested_raise),
        )
        aggressive_kind = (
            ActionKind.ALL_IN
            if raise_to == current_effective_to
            else ActionKind.RAISE
        )
        legal_kinds.append(aggressive_kind)
        modeled.append(Action(aggressive_kind, raise_to))
    return facing, tuple(legal_kinds), tuple(modeled)


def _flop_root(
    history: PublicHandHistory,
) -> tuple[int, object]:
    flop_indices = [
        index
        for index, event in enumerate(history.events)
        if event.kind == "DEAL_FLOP"
    ]
    if len(flop_indices) != 1:
        raise LiveGTOContinuationError(
            "complete continuation requires exactly one DEAL_FLOP event"
        )
    flop_index = flop_indices[0]
    prefix = replace(history, events=history.events[: flop_index + 1])
    try:
        replayed = replay_public_hand(prefix)
    except PublicHandHistoryError as error:
        raise LiveGTOContinuationError(
            f"flop-root replay failed: {error}"
        ) from error
    if replayed.street != "FLOP" or len(replayed.board) != 3:
        raise LiveGTOContinuationError(
            "public transcript does not reach a valid flop root"
        )
    if len(replayed.live_seats) != 2:
        raise LiveGTOContinuationError(
            "native continuation requires exactly two players at the flop root"
        )
    if replayed.actor_seat is None:
        raise LiveGTOContinuationError(
            "flop root does not identify the OOP actor"
        )
    return flop_index, replayed


def _path_from_history(
    history: PublicHandHistory,
    *,
    after_index: int,
    scale: int,
    live_seats: frozenset[int],
) -> tuple[ContinuationAction | ContinuationDeal, ...]:
    path: list[ContinuationAction | ContinuationDeal] = []
    action_kind = {
        "FOLD": ActionKind.FOLD,
        "CHECK": ActionKind.CHECK,
        "CALL": ActionKind.CALL,
        "BET_TO": ActionKind.BET,
        "RAISE_TO": ActionKind.RAISE,
        "ALL_IN_TO": ActionKind.ALL_IN,
    }
    for event in history.events[after_index + 1 :]:
        if event.kind in {"DEAL_TURN", "DEAL_RIVER"}:
            if len(event.cards) != 1:
                raise LiveGTOContinuationError(
                    f"{event.kind} must contain exactly one card"
                )
            path.append(ContinuationDeal(event.cards[0]))
            continue
        kind = action_kind.get(event.kind)
        if kind is None:
            raise LiveGTOContinuationError(
                f"unsupported postflop event {event.kind!r}"
            )
        if event.actor_seat not in live_seats:
            raise LiveGTOContinuationError(
                "postflop continuation contains an actor outside the HU hand"
            )
        amount = (
            _units(
                event.amount_to_bb,
                scale,
                f"{event.kind} amount-to",
            )
            if kind
            in {
                ActionKind.BET,
                ActionKind.RAISE,
                ActionKind.ALL_IN,
            }
            and event.amount_to_bb is not None
            else None
        )
        path.append(ContinuationAction(Action(kind, amount)))
    return tuple(path)


def flop_range_state(state: LiveDecisionState) -> LiveDecisionState:
    """Return a range-only state that cannot leak future board cards."""

    if len(state.board) < 3:
        raise LiveGTOContinuationError(
            "postflop continuation needs a complete flop"
        )
    history = state.public_hand
    if history is not None:
        flop_index, flop = _flop_root(history)
        history = replace(
            history,
            events=history.events[: flop_index + 1],
        )
        seat_by_position = {
            seat.position: seat.seat for seat in history.seats
        }
        try:
            hero_seat = seat_by_position[state.hero_position]
            villain_seat = seat_by_position[state.villain_position]
        except KeyError as error:
            raise LiveGTOContinuationError(
                "Hero or Villain position is absent from public_hand"
            ) from error
        pot = flop.pot_bb
        hero_stack = flop.stack_map[hero_seat]
        villain_stack = flop.stack_map[villain_seat]
    else:
        pot = state.pot_bb
        hero_stack = state.hero_stack_bb
        villain_stack = state.villain_stack_bb
    return replace(
        state,
        street="FLOP",
        board=tuple(state.board[:3]),
        pot_bb=pot,
        hero_stack_bb=hero_stack,
        villain_stack_bb=villain_stack,
        hero_current_bet_bb=Decimal(0),
        villain_current_bet_bb=Decimal(0),
        amount_to_call_bb=Decimal(0),
        legal_actions=("CHECK", "BET"),
        street_root_confirmed=True,
        action_history=(),
        observed_bet_to_bb=Decimal(0),
        mapping_error="",
        public_hand=history,
    )


def build_live_continuation_spec(
    state: LiveDecisionState,
    ranges: RangeBundle,
    config: LiveGTOConfig,
) -> ContinuationSpec:
    """Build an exact public-path request or raise without guessing."""

    history = state.public_hand
    if history is None:
        raise LiveGTOContinuationError(
            "action-conditioned continuation requires a complete public_hand"
        )
    try:
        replayed = replay_public_hand(history)
    except PublicHandHistoryError as error:
        raise LiveGTOContinuationError(
            f"public hand replay failed: {error}"
        ) from error
    if replayed.terminal or replayed.actor_seat is None:
        raise LiveGTOContinuationError(
            "public hand does not end at a live decision node"
        )
    if replayed.street != state.street or replayed.board != state.board:
        raise LiveGTOContinuationError(
            "public hand street/board differs from the captured decision"
        )

    flop_index, flop = _flop_root(history)
    if replayed.live_seats != flop.live_seats:
        raise LiveGTOContinuationError(
            "native continuation cannot include a postflop player elimination"
        )
    seat_by_position = {seat.position: seat.seat for seat in history.seats}
    try:
        hero_seat = seat_by_position[state.hero_position]
        villain_seat = seat_by_position[state.villain_position]
    except KeyError as error:
        raise LiveGTOContinuationError(
            "Hero or Villain position is absent from public_hand"
        ) from error
    if {hero_seat, villain_seat} != set(flop.live_seats):
        raise LiveGTOContinuationError(
            "captured Hero/Villain do not equal the flop HU seats"
        )
    if replayed.actor_seat != hero_seat:
        raise LiveGTOContinuationError(
            "public hand action is not currently on Hero"
        )
    oop_seat = flop.actor_seat
    assert oop_seat is not None
    ip_seat = next(iter(flop.live_seats - {oop_seat}))
    expected_actor = (
        Position.OOP if replayed.actor_seat == oop_seat else Position.IP
    )
    if state.hero_is_oop != (hero_seat == oop_seat):
        raise LiveGTOContinuationError(
            "captured HU position disagrees with the replayed flop order"
        )

    root_pot = _units(
        flop.pot_bb,
        config.chip_scale,
        "flop-root pot",
    )
    root_effective = _units(
        min(flop.stack_map[oop_seat], flop.stack_map[ip_seat]),
        config.chip_scale,
        "flop-root effective stack",
    )
    oop_invested = _units(
        flop.stack_map[oop_seat] - replayed.stack_map[oop_seat],
        config.chip_scale,
        "OOP postflop investment",
    )
    ip_invested = _units(
        flop.stack_map[ip_seat] - replayed.stack_map[ip_seat],
        config.chip_scale,
        "IP postflop investment",
    )
    facing, legal_kinds, modeled_actions = _current_node_actions(
        state,
        replayed,
        hero_seat=hero_seat,
        villain_seat=villain_seat,
        config=config,
    )
    parameters = SolveParameters(
        chip_scale=config.chip_scale,
        chip_unit=(
            "centi-BB"
            if config.chip_scale == 100
            else "scaled-BB"
        ),
        bet_sizes=_continuation_sizes(config),
        add_allin_threshold=Decimal(0),
        force_allin_threshold=Decimal(0),
        merging_threshold=Decimal(0),
        target_exploitability_pct=config.target_exploitability_pct,
        max_iterations=config.max_iterations,
        allocation_mode=AllocationMode.UNCOMPRESSED_F32,
        solver_name="b-inary/postflop-solver",
        solver_commit=PINNED_SOLVER_COMMIT,
    )
    return ContinuationSpec(
        flop=tuple(flop.board),  # type: ignore[arg-type]
        current_board=tuple(replayed.board),
        acting_player=expected_actor,
        oop_range=ranges.oop,
        ip_range=ranges.ip,
        starting_pot=root_pot,
        effective_stack=root_effective,
        path=_path_from_history(
            history,
            after_index=flop_index,
            scale=config.chip_scale,
            live_seats=flop.live_seats,
        ),
        expected_total_invested=(oop_invested, ip_invested),
        facing_bet=facing,
        legal_action_kinds=legal_kinds,
        modeled_actions=modeled_actions,
        parameters=parameters,
        rake_rate_pct=config.rake_rate_pct,
        rake_cap=_units(
            config.rake_cap_bb,
            config.chip_scale,
            "rake cap",
        ),
    )


__all__ = [
    "LiveGTOContinuationError",
    "build_live_continuation_spec",
    "flop_range_state",
]
