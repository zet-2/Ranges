from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from gto_oracle import (
    ActionValue,
    ComboPolicy,
    ConditionalCombo,
    ConditionalRange,
    ContinuationAction,
    ContinuationDeal,
    ContinuationResult,
    ContinuationSpec,
    PlayerRange,
    Position,
    SolverMetadata,
    WeightedCombo,
    build_continuation_request,
)
from live_gto import (
    LiveDecisionState,
    LiveGTOConfig,
    LiveGTOStatus,
    LiveGTORouter,
    RangeBundle,
)
from live_gto_continuation import (
    build_live_continuation_spec,
    flop_range_state,
)
from test_gto_hand_history import heads_up_to_turn_history


_BINARY_DIGEST = "b" * 64


def _state() -> LiveDecisionState:
    history = heads_up_to_turn_history()
    return LiveDecisionState(
        hand_id=history.hand_id,
        street="TURN",
        board=("2c", "7d", "Jh", "As"),
        hero_combo=("Qc", "Qd"),
        hero_position="BB",
        villain_position="BTN",
        hero_is_oop=True,
        active_villains=1,
        pot_bb=Decimal("11.5"),
        hero_stack_bb=Decimal("94.5"),
        villain_stack_bb=Decimal("94.5"),
        hero_current_bet_bb=Decimal(0),
        villain_current_bet_bb=Decimal(0),
        amount_to_call_bb=Decimal(0),
        legal_actions=("CHECK", "BET"),
        street_root_confirmed=True,
        public_hand=history,
    )


def _ranges() -> RangeBundle:
    return RangeBundle(
        oop=PlayerRange(
            Position.OOP,
            (WeightedCombo(("Qc", "Qd"), Decimal(1)),),
        ),
        ip=PlayerRange(
            Position.IP,
            (
                WeightedCombo(("As", "Ad"), Decimal("0.5")),
                WeightedCombo(("Kc", "Kd"), Decimal(1)),
            ),
        ),
        profile_id="continuation-test",
        hero_combo_injected=False,
        provenance="unit-test ranges",
    )


def _config(cache_path: Path = Path(":memory:")) -> LiveGTOConfig:
    return LiveGTOConfig(
        enabled=True,
        owned_simulator_acknowledged=True,
        engine_path=Path("unused-test-engine"),
        cache_path=cache_path,
        target_exploitability_pct=Decimal("100"),
        max_iterations=10,
        mix_secret=b"s" * 32,
    )


def _result_for_spec(spec: ContinuationSpec) -> ContinuationResult:
    current_dead = set(spec.current_board)
    ranges = {
        Position.OOP: spec.oop_range,
        Position.IP: spec.ip_range,
    }
    conditional_ranges = []
    for position in (Position.OOP, Position.IP):
        compatible = [
            combo
            for combo in ranges[position].combos
            if not current_dead.intersection(combo.cards)
        ]
        total = sum((combo.weight for combo in compatible), Decimal(0))
        conditional_ranges.append(
            ConditionalRange(
                position,
                tuple(
                    ConditionalCombo(
                        cards=combo.cards,
                        input_range_weight=combo.weight,
                        path_weight=combo.weight,
                        joint_compatible_weight=combo.weight,
                        conditional_reach_weight=combo.weight / total,
                    )
                    for combo in compatible
                ),
            )
        )

    acting = ranges[spec.acting_player]
    reachable = [
        combo
        for combo in acting.combos
        if not current_dead.intersection(combo.cards)
    ]
    acting_total = sum((combo.weight for combo in reachable), Decimal(0))
    action_frequency = Decimal(1) / Decimal(len(spec.modeled_actions))
    policies = tuple(
        ComboPolicy(
            combo.cards,
            combo.weight / acting_total,
            Decimal("0.5"),
            tuple(
                ActionValue(action, action_frequency, Decimal(0))
                for action in spec.modeled_actions
            ),
        )
        for combo in reachable
    )
    metadata = SolverMetadata(
        solver_name=spec.parameters.solver_name,
        solver_version=spec.parameters.solver_commit,
        iterations=1,
        elapsed_seconds=Decimal("0.01"),
        exploitability=Decimal(0),
        converged=True,
        extra=(
            ("binary_sha256", _BINARY_DIGEST),
            ("execution_context", "owned_simulator"),
            ("exploitability_pct_of_pot", "0"),
        ),
    )
    return ContinuationResult.for_spec(
        spec,
        policies,
        tuple(conditional_ranges),  # type: ignore[arg-type]
        metadata,
    )


class ContinuationBridgeTests(unittest.TestCase):
    def test_builds_exact_flop_to_turn_path_without_future_card_leak(self):
        state = _state()
        range_state = flop_range_state(state)
        spec = build_live_continuation_spec(
            state,
            _ranges(),
            _config(),
        )

        self.assertEqual(("2c", "7d", "Jh"), range_state.board)
        self.assertEqual(7, len(range_state.public_hand.events))
        self.assertEqual(Decimal("5.5"), range_state.pot_bb)
        self.assertEqual(Decimal("97.5"), range_state.hero_stack_bb)
        self.assertEqual(Decimal("97.5"), range_state.villain_stack_bb)
        self.assertEqual(("2c", "7d", "Jh"), spec.flop)
        self.assertEqual(state.board, spec.current_board)
        self.assertEqual(550, spec.starting_pot)
        self.assertEqual(9750, spec.effective_stack)
        self.assertEqual((300, 300), spec.expected_total_invested)
        self.assertEqual(Position.OOP, spec.acting_player)
        self.assertEqual(
            [
                ("action", "CHECK", None),
                ("action", "BET", 300),
                ("action", "CALL", None),
                ("deal", "As", None),
            ],
            [
                (
                    "action",
                    step.action.kind.value,
                    step.action.amount,
                )
                if isinstance(step, ContinuationAction)
                else ("deal", step.card, None)
                for step in spec.path
            ],
        )
        # AsAd is legal at the flop root. It is removed only when As is dealt,
        # which proves the range source did not see a future board card.
        self.assertIn(
            ("Ad", "As"),
            {combo.cards for combo in spec.ip_range.combos},
        )

        request = build_continuation_request(
            spec,
            offline_only_acknowledged=False,
            owned_simulator_acknowledged=True,
        )
        self.assertEqual("solve_path", request["operation"])
        self.assertEqual(list(spec.flop), request["board"])
        self.assertEqual(list(spec.current_board), request["expected_board"])
        self.assertEqual(
            ["action", "action", "action", "deal"],
            [step["type"] for step in request["path_history"]],
        )

    def test_router_uses_full_path_and_reuses_conditional_cache(self):
        class RangeProvider:
            def __init__(self) -> None:
                self.boards: list[tuple[str, ...]] = []

            def ranges_for(self, state: LiveDecisionState) -> RangeBundle:
                self.boards.append(state.board)
                return _ranges()

        class FakeEngine:
            binary_sha256 = _BINARY_DIGEST

            def __init__(self) -> None:
                self.calls = 0

            def solve_continuation(
                self,
                spec: ContinuationSpec,
            ) -> ContinuationResult:
                self.calls += 1
                return _result_for_spec(spec)

        provider = RangeProvider()
        engine = FakeEngine()
        with tempfile.TemporaryDirectory() as directory:
            config = _config(Path(directory) / "oracle.sqlite3")
            router = LiveGTORouter(
                config,
                range_provider=provider,
                engine_factory=lambda *args, **kwargs: engine,
            )
            first = router.evaluate(_state())
            second = router.evaluate(_state())

        self.assertEqual(LiveGTOStatus.SOLVED, first.status)
        self.assertEqual(LiveGTOStatus.SOLVED, second.status)
        self.assertIsInstance(first.spec, ContinuationSpec)
        self.assertIsInstance(first.result, ContinuationResult)
        self.assertIn("Range continuity", first.analysis)
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(1, engine.calls)
        self.assertEqual(
            [("2c", "7d", "Jh"), ("2c", "7d", "Jh")],
            provider.boards,
        )


if __name__ == "__main__":
    unittest.main()
