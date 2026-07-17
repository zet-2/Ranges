from __future__ import annotations

from decimal import Decimal
import unittest

from preflop_blueprint import (
    BlueprintAction,
    BlueprintNode,
    BlueprintSpot,
    hand_class_combo_count,
)
from preflop_history import (
    ALL_HAND_CLASSES,
    AmbiguousHistoryError,
    InvalidHistoryError,
    NoMatchingHistoryError,
    PreflopHistoryResolver,
    ReachError,
    ZeroHeroReachError,
    canonical_position,
    expand_class_weights,
    hand_class_for_cards,
    simulate_history,
)


def _action(label: str, weights: dict[str, object]) -> BlueprintAction:
    if label.endswith("%"):
        kind = "raise"
        size_pct = Decimal(label[:-1])
    elif label == "AI":
        kind = "allin"
        size_pct = None
    else:
        kind = label.lower()
        size_pct = None
    normalized_weights = tuple(
        (name, Decimal(str(value))) for name, value in weights.items()
    )
    combos = sum(
        (
            weight * hand_class_combo_count(name)
            for name, weight in normalized_weights
        ),
        Decimal("0"),
    )
    return BlueprintAction(
        label=label,
        kind=kind,
        size_pct=size_pct,
        combos=combos,
        weights=normalized_weights,
        evs=None,
    )


def _node(
    history: str,
    actor: str,
    *actions: BlueprintAction,
    prior: dict[str, object] | None = None,
) -> BlueprintNode:
    expected = {
        hand_class: Decimal(str(value))
        for hand_class, value in (prior or {name: 1 for name in ALL_HAND_CLASSES}).items()
    }
    totals = {hand_class: Decimal(0) for hand_class in expected}
    for action in actions:
        for hand_class, value in action.weights:
            if hand_class in totals:
                totals[hand_class] += value
    residual = {
        hand_class: expected[hand_class] - totals[hand_class]
        for hand_class in expected
        if expected[hand_class] - totals[hand_class] > 0
    }
    completed = list(actions)
    if residual:
        fold_index = next(
            (index for index, action in enumerate(completed) if action.kind == "fold"),
            None,
        )
        if fold_index is None:
            completed.append(_action("Fold", residual))
        else:
            fold = completed[fold_index]
            merged = dict(fold.weights)
            for hand_class, value in residual.items():
                merged[hand_class] = merged.get(hand_class, Decimal(0)) + value
            completed[fold_index] = _action("Fold", merged)
    return BlueprintNode(
        game="nl",
        version=2,
        stack=100,
        history=history,
        actor=actor,
        actions=tuple(completed),
        continuing_combos=sum(
            (action.combos for action in completed if action.kind != "fold"),
            Decimal("0"),
        ),
        response_sha256="0" * 64,
    )


class FakeStore:
    def __init__(self, histories: list[str], nodes: dict[str, BlueprintNode] | None = None):
        self._spots = tuple(BlueprintSpot(history=value, depth=1) for value in histories)
        self._nodes = nodes or {}

    def spots(self, stack: int):
        assert stack == 100
        return self._spots

    def node(self, stack: int, history: str):
        assert stack == 100
        return self._nodes[history]


def _single_raised_hu_fixture() -> tuple[str, FakeStore]:
    prefixes = [
        ("UTG", "UTG", "60%", {"AA": ".8", "72o": ".2"}),
        ("UTG_60%_HJ", "HJ", "Fold", {"AA": "1", "72o": "1"}),
        (
            "UTG_60%_HJ_Fold_CO",
            "CO",
            "Fold",
            {"AA": "1", "72o": "1"},
        ),
        (
            "UTG_60%_HJ_Fold_CO_Fold_BTN",
            "BTN",
            "Fold",
            {"AA": "1", "72o": "1"},
        ),
        (
            "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold_SB",
            "SB",
            "Fold",
            {"AA": "1", "72o": "1"},
        ),
        (
            "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold_SB_Fold_BB",
            "BB",
            "Call",
            {"AA": ".5", "72o": ".1"},
        ),
    ]
    nodes = {
        history: _node(history, actor, _action(action, weights))
        for history, actor, action, weights in prefixes
    }
    terminal_decision = prefixes[-1][0]
    return terminal_decision, FakeStore([terminal_decision], nodes)


def test_percentage_raise_first_calls_then_uses_pot_after_call() -> None:
    _, state = simulate_history("UTG_60%_HJ", stack=100)
    assert state.actor == "HJ"
    assert state.contribution_map["UTG"] == Decimal("2.50")
    assert state.pot == Decimal("4.00")

    history = "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold_SB_Fold_BB_182%_UTG"
    _, reraised = simulate_history(history, stack=100)
    # Before BB raises the pot is 4 BB. BB calls 1.5 first, so 182% of 5.5
    # is added to the prior 2.5 BB high-water mark.
    assert reraised.contribution_map["BB"] == Decimal("12.510")
    assert reraised.actor == "UTG"


def test_provider_call_at_unraised_bb_is_a_passive_check() -> None:
    decision = "UTG_Fold_HJ_Fold_CO_Fold_BTN_Fold_SB_Call_BB"
    _, before = simulate_history(decision, stack=100)
    assert before.actor == "BB"
    assert before.contribution_map["BB"] == Decimal("1")

    parsed, complete = simulate_history(
        f"{decision}_Call", stack=100, completed=True
    )
    assert parsed.steps[-1].action == "Call"
    assert complete.terminal
    assert complete.contribution_map["BB"] == Decimal("1")
    assert complete.live_positions == frozenset({"SB", "BB"})


def test_history_actor_order_is_validated() -> None:
    with unittest.TestCase().assertRaises(InvalidHistoryError):
        simulate_history("UTG_60%_CO", stack=100)


def test_position_alias_mp_maps_to_provider_hj() -> None:
    assert canonical_position("MP") == "HJ"
    store = FakeStore(["UTG_60%_HJ"])
    resolution = PreflopHistoryResolver(store).resolve_decision(
        stack=100,
        expected_actor="MP",
        observed_contributions={"UTG": "2.5", "SB": ".5", "BB": "1"},
        observed_folded=set(),
    )
    assert resolution.state.actor == "HJ"


def test_decision_resolution_requires_a_unique_match() -> None:
    unique = PreflopHistoryResolver(FakeStore(["UTG_60%_HJ"]))
    result = unique.resolve_decision(
        stack=100,
        expected_actor="HJ",
        observed_contributions={"UTG": "2.50"},
        observed_folded=set(),
        tolerance="0.001",
    )
    assert result.history == "UTG_60%_HJ"

    ambiguous = PreflopHistoryResolver(
        FakeStore(["UTG_60%_HJ", "UTG_61%_HJ"]), contribution_tolerance=".05"
    )
    with unittest.TestCase().assertRaises(AmbiguousHistoryError):
        ambiguous.resolve_decision(
            stack=100,
            expected_actor="HJ",
            observed_contributions={"UTG": "2.51"},
            observed_folded=set(),
        )
    with unittest.TestCase().assertRaises(NoMatchingHistoryError):
        unique.resolve_decision(
            stack=100,
            expected_actor="HJ",
            observed_contributions={"UTG": "4"},
            observed_folded=set(),
        )


def test_per_seat_tolerances_do_not_relax_blinds_with_a_large_raise() -> None:
    resolver = PreflopHistoryResolver(FakeStore(["UTG_60%_HJ"]))
    with unittest.TestCase().assertRaises(NoMatchingHistoryError):
        resolver.resolve_decision(
            stack=100,
            expected_actor="HJ",
            observed_contributions={"UTG": "2.5", "SB": ".6", "BB": "1"},
            observed_folded=set(),
            tolerance={"UTG": ".5", "SB": ".05", "BB": ".05"},
        )


def test_all_in_identity_is_part_of_node_matching() -> None:
    resolver = PreflopHistoryResolver(FakeStore(["UTG_AI_HJ"]))
    matched = resolver.resolve_decision(
        stack=100,
        expected_actor="HJ",
        observed_contributions={"UTG": 100, "SB": ".5", "BB": 1},
        observed_folded=set(),
        observed_all_in={"UTG"},
    )
    assert matched.state.all_in == frozenset({"UTG"})
    with unittest.TestCase().assertRaises(NoMatchingHistoryError):
        resolver.resolve_decision(
            stack=100,
            expected_actor="HJ",
            observed_contributions={"UTG": 100, "SB": ".5", "BB": 1},
            observed_folded=set(),
            observed_all_in=set(),
        )

def test_resolve_single_raised_heads_up_terminal_path() -> None:
    terminal_decision, store = _single_raised_hu_fixture()
    resolver = PreflopHistoryResolver(store)
    resolution = resolver.resolve_hu_handoff(
        stack=100,
        observed_contributions={
            "UTG": "2.5",
            "HJ": "0",
            "CO": "0",
            "BTN": "0",
            "SB": ".5",
            "BB": "2.5",
        },
        observed_folded={"HJ", "CO", "BTN", "SB"},
        survivors={"UTG", "BB"},
    )
    assert resolution.matched_spot_history == terminal_decision
    assert resolution.final_action == "Call"
    assert resolution.state.terminal
    assert resolution.state.live_positions == frozenset({"UTG", "BB"})


def test_hu_terminal_path_can_end_when_third_player_folds() -> None:
    decision = "UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB_Call_BB"
    store = FakeStore(
        [decision],
        {decision: _node(decision, "BB", _action("Fold", {"AA": 1}))},
    )
    resolution = PreflopHistoryResolver(store).resolve_hu_handoff(
        stack=100,
        observed_contributions={
            "UTG": 0,
            "HJ": 0,
            "CO": 0,
            "BTN": "2.5",
            "SB": "2.5",
            "BB": "1",
        },
        observed_folded={"UTG", "HJ", "CO", "BB"},
        survivors={"BTN", "SB"},
    )
    assert resolution.final_action == "Fold"
    assert resolution.state.live_positions == frozenset({"BTN", "SB"})


def test_cumulative_reach_replaces_previous_reach_instead_of_multiplying() -> None:
    paths = [
        ("UTG", "UTG", "60%", {"AA": ".8", "22": ".4"}),
        ("UTG_60%_HJ", "HJ", "Fold", {"AA": "1", "22": "1"}),
        (
            "UTG_60%_HJ_Fold_CO",
            "CO",
            "Fold",
            {"AA": "1", "22": "1"},
        ),
        (
            "UTG_60%_HJ_Fold_CO_Fold_BTN",
            "BTN",
            "Fold",
            {"AA": "1", "22": "1"},
        ),
        (
            "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold_SB",
            "SB",
            "Fold",
            {"AA": "1", "22": "1"},
        ),
        (
            "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold_SB_Fold_BB",
            "BB",
            "182%",
            {"AA": ".5", "22": ".1"},
        ),
        (
            "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold_SB_Fold_BB_182%_UTG",
            "UTG",
            "Call",
            {"AA": ".6", "22": ".1"},
        ),
    ]
    nodes = {}
    for history, actor, action, weights in paths:
        prior = None
        if history == paths[-1][0]:
            prior = {"AA": ".8", "22": ".4"}
        nodes[history] = _node(
            history, actor, _action(action, weights), prior=prior
        )
    final_decision = paths[-1][0]
    resolver = PreflopHistoryResolver(FakeStore([final_decision], nodes))
    resolution = resolver.resolve_completed_path(
        stack=100, decision_history=final_decision, final_action="Call"
    )
    reach = resolver.walk_reaches(
        resolution, hero_position="UTG", hero_cards=("As", "Ah")
    )
    utg = reach.for_position("UTG")
    assert utg["AA"] == Decimal(".6")
    assert utg["22"] == Decimal(".1")
    assert utg["AA"] != Decimal(".8") * Decimal(".6")


def test_zero_reach_hero_fails_closed_and_is_never_injected() -> None:
    terminal_decision, store = _single_raised_hu_fixture()
    resolver = PreflopHistoryResolver(store)
    resolution = resolver.resolve_completed_path(
        stack=100, decision_history=terminal_decision, final_action="Call"
    )
    with unittest.TestCase().assertRaises(ZeroHeroReachError):
        resolver.walk_reaches(
            resolution, hero_position="UTG", hero_cards=("As", "Kd")
        )


def test_tolerated_upward_drift_cannot_resurrect_zero_reach() -> None:
    paths = [
        ("UTG", "UTG", "60%", {"22": "1"}),
        ("UTG_60%_HJ", "HJ", "Fold", {"AA": "1", "22": "1"}),
        (
            "UTG_60%_HJ_Fold_CO",
            "CO",
            "Fold",
            {"AA": "1", "22": "1"},
        ),
        (
            "UTG_60%_HJ_Fold_CO_Fold_BTN",
            "BTN",
            "Fold",
            {"AA": "1", "22": "1"},
        ),
        (
            "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold_SB",
            "SB",
            "Fold",
            {"AA": "1", "22": "1"},
        ),
        (
            "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold_SB_Fold_BB",
            "BB",
            "182%",
            {"AA": "1", "22": "1"},
        ),
        (
            "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold_SB_Fold_BB_182%_UTG",
            "UTG",
            "Call",
            {"AA": ".005", "22": "1"},
        ),
    ]
    nodes = {}
    for history, actor, action, weights in paths:
        prior = {"AA": "0", "22": "1"} if history == paths[-1][0] else None
        nodes[history] = _node(
            history,
            actor,
            _action(action, weights),
            prior=prior,
        )
    final_decision = paths[-1][0]
    resolver = PreflopHistoryResolver(FakeStore([final_decision], nodes))
    resolution = resolver.resolve_completed_path(
        stack=100,
        decision_history=final_decision,
        final_action="Call",
    )

    reach = resolver.walk_reaches(resolution)
    assert reach.for_position("UTG")["AA"] == Decimal("0")
    with unittest.TestCase().assertRaises(ZeroHeroReachError):
        resolver.walk_reaches(
            resolution,
            hero_position="UTG",
            hero_cards=("As", "Ah"),
        )


def test_incomplete_action_branches_fail_reach_conservation() -> None:
    incomplete = _node(
        "UTG",
        "UTG",
        _action("60%", {"AA": ".8"}),
        prior={"AA": ".8"},
    )
    # Build a resolution directly so the malformed node reaches the reach walk.
    _, state = simulate_history("UTG_60%_HJ", stack=100)
    from preflop_history import ActionStep, PreflopResolution

    resolution = PreflopResolution(
        stack=100,
        history="UTG_60%_HJ",
        steps=(ActionStep("UTG", "60%"),),
        state=state,
        matched_spot_history="UTG_60%_HJ",
    )
    resolver = PreflopHistoryResolver(
        FakeStore(["UTG_60%_HJ"], {"UTG": incomplete})
    )
    with unittest.TestCase().assertRaises(ReachError):
        resolver.walk_reaches(resolution)


def test_exact_combo_expansion_and_dead_card_filtering() -> None:
    assert len(ALL_HAND_CLASSES) == 169
    assert hand_class_for_cards("As", "Kh") == "AKo"
    assert hand_class_for_cards("10h", "9h") == "T9s"
    assert len(expand_class_weights({"AA": 1})) == 6
    assert len(expand_class_weights({"AKs": ".5"})) == 4
    assert len(expand_class_weights({"AKo": ".5"})) == 12
    blocked = expand_class_weights({"AA": 1, "AKs": 1}, dead_cards={"As"})
    assert len(blocked) == 6  # Three AA combos plus three suited AK combos.
    assert all("As" not in combo for combo in blocked)


class PreflopHistoryTests(unittest.TestCase):
    """Expose the compact function-style cases to unittest discovery."""

    def test_percentage_raise_formula(self):
        test_percentage_raise_first_calls_then_uses_pot_after_call()

    def test_actor_order(self):
        test_history_actor_order_is_validated()

    def test_unraised_bb_call_alias(self):
        test_provider_call_at_unraised_bb_is_a_passive_check()

    def test_mp_alias(self):
        test_position_alias_mp_maps_to_provider_hj()

    def test_unique_resolution(self):
        test_decision_resolution_requires_a_unique_match()

    def test_per_seat_tolerance(self):
        test_per_seat_tolerances_do_not_relax_blinds_with_a_large_raise()

    def test_all_in_matching(self):
        test_all_in_identity_is_part_of_node_matching()

    def test_hu_call_terminal(self):
        test_resolve_single_raised_heads_up_terminal_path()

    def test_hu_fold_terminal(self):
        test_hu_terminal_path_can_end_when_third_player_folds()

    def test_cumulative_reach(self):
        test_cumulative_reach_replaces_previous_reach_instead_of_multiplying()

    def test_zero_hero_reach(self):
        test_zero_reach_hero_fails_closed_and_is_never_injected()

    def test_zero_reach_cannot_resurrect(self):
        test_tolerated_upward_drift_cannot_resurrect_zero_reach()

    def test_reach_conservation(self):
        test_incomplete_action_branches_fail_reach_conservation()

    def test_combo_expansion(self):
        test_exact_combo_expansion_and_dead_card_filtering()


if __name__ == "__main__":
    unittest.main()
