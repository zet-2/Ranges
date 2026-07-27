"""Reproducible Mac-to-server GTO continuation simulation.

The simulation deliberately uses tiny synthetic private ranges.  Its purpose
is to verify transport, public-event continuity, solver traversal, conditional
ranges, caching, and response identity without pretending to benchmark a
production six-max strategy profile.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
from types import SimpleNamespace

from gto_event_collector import PublicHandEventRecorder
from gto_hand_history import (
    HandEvent,
    HandSeat,
    PublicHandHistory,
    replay_public_hand,
)
from gto_oracle import (
    ActionValue,
    ComboPolicy,
    ConditionalCombo,
    ConditionalRange,
    ContinuationResult,
    ContinuationSpec,
    EngineClient,
    PlayerRange,
    Position,
    SolverMetadata,
    WeightedCombo,
)
from gto_remote.protocol import (
    build_evaluate_request,
    decision_fingerprint,
    encode_json,
    outcome_from_wire,
    parse_evaluate_request,
)
from gto_remote.server import EvaluationService
from live_gto import (
    LiveDecisionState,
    LiveGTOConfig,
    LiveGTORouter,
    RangeBundle,
)


_FAKE_BINARY_DIGEST = "d" * 64


def _seats() -> tuple[HandSeat, ...]:
    positions = {
        0: "UTG",
        1: "HJ",
        2: "CO",
        3: "BTN",
        4: "SB",
        5: "BB",
    }
    return tuple(
        HandSeat(seat, positions[seat], Decimal("100"))
        for seat in range(6)
    )


def _target_history(street: str) -> PublicHandHistory:
    events: tuple[HandEvent, ...] = (
        HandEvent(0, "FOLD", "PREFLOP", actor_seat=0),
        HandEvent(1, "FOLD", "PREFLOP", actor_seat=1),
        HandEvent(2, "FOLD", "PREFLOP", actor_seat=2),
        HandEvent(
            3,
            "RAISE_TO",
            "PREFLOP",
            actor_seat=3,
            amount_to_bb=Decimal("2.5"),
        ),
        HandEvent(4, "FOLD", "PREFLOP", actor_seat=4),
        HandEvent(
            5,
            "CALL",
            "PREFLOP",
            actor_seat=5,
            amount_to_bb=Decimal("2.5"),
        ),
        HandEvent(6, "DEAL_FLOP", "FLOP", cards=("2c", "7d", "Jh")),
        HandEvent(7, "CHECK", "FLOP", actor_seat=5),
        HandEvent(
            8,
            "BET_TO",
            "FLOP",
            actor_seat=3,
            amount_to_bb=Decimal("3"),
        ),
        HandEvent(
            9,
            "CALL",
            "FLOP",
            actor_seat=5,
            amount_to_bb=Decimal("3"),
        ),
        HandEvent(10, "DEAL_TURN", "TURN", cards=("4s",)),
    )
    if street == "RIVER":
        events = (
            *events,
            HandEvent(11, "CHECK", "TURN", actor_seat=5),
            HandEvent(
                12,
                "BET_TO",
                "TURN",
                actor_seat=3,
                amount_to_bb=Decimal("5.75"),
            ),
            HandEvent(
                13,
                "CALL",
                "TURN",
                actor_seat=5,
                amount_to_bb=Decimal("5.75"),
            ),
            HandEvent(14, "DEAL_RIVER", "RIVER", cards=("5c",)),
        )
    return PublicHandHistory(
        hand_id=f"simulation-{street.lower()}-1",
        button_seat=3,
        small_blind_bb=Decimal("0.5"),
        big_blind_bb=Decimal("1"),
        ante_bb=Decimal(0),
        rake_rate_pct=Decimal("5"),
        rake_cap_bb=Decimal("0.5"),
        seats=_seats(),
        events=events,
    )


def _snapshot_for_prefix(
    target: PublicHandHistory,
    event_count: int,
) -> SimpleNamespace:
    prefix = replace(target, events=target.events[:event_count])
    state = replay_public_hand(prefix)
    last_event = target.events[event_count - 1] if event_count else None
    visible_by_seat: dict[int, str] = {}
    if last_event is not None and last_event.actor_seat is not None:
        visible_by_seat[last_event.actor_seat] = {
            "BET_TO": "BET",
            "RAISE_TO": "RAISE",
            "ALL_IN_TO": "ALL-IN",
        }.get(last_event.kind, last_event.kind)
    seat_by_id = {seat.seat: seat for seat in target.seats}
    players = []
    for seat_id in sorted(seat_by_id):
        status = (
            "FOLDED"
            if seat_id in state.folded
            else "ALL_IN"
            if seat_id in state.all_in
            else "ACTIVE"
        )
        players.append(
            SimpleNamespace(
                seat_index=seat_id,
                name=seat_by_id[seat_id].position,
                stack_size=float(state.stack_map[seat_id]),
                current_bet=float(
                    state.street_contribution_map[seat_id]
                ),
                status=status,
                visible_action=visible_by_seat.get(seat_id, ""),
                is_dealer=seat_id == target.button_seat,
            )
        )
    return SimpleNamespace(
        hand_id=target.hand_id,
        meta_info=SimpleNamespace(current_street=state.street),
        board_state=SimpleNamespace(
            community_cards=list(state.board),
            total_pot=float(state.pot_bb),
        ),
        dealer_seat_index=target.button_seat,
        action_on_seat_index=(
            state.actor_seat if state.actor_seat is not None else -1
        ),
        players=players,
    )


def _collect_history(target: PublicHandHistory) -> PublicHandHistory:
    recorder = PublicHandEventRecorder()
    for event_count in range(len(target.events) + 1):
        result = recorder.observe(
            _snapshot_for_prefix(target, event_count)
        )
        if result is None:
            raise RuntimeError(
                f"event recorder failed at prefix {event_count}: "
                f"{recorder.error}"
            )
    if not recorder.complete or recorder.history != target:
        raise RuntimeError("event recorder did not reproduce the target history")
    return recorder.history


def _decision_state(history: PublicHandHistory) -> LiveDecisionState:
    replayed = replay_public_hand(history)
    if replayed.actor_seat != 5:
        raise RuntimeError("simulation expected action on BB/OOP")
    legal_actions = tuple(
        {
            "BET_TO": "BET",
            "RAISE_TO": "RAISE",
            "ALL_IN_TO": "ALL-IN",
        }.get(action, action)
        for action in replayed.legal_actions
    )
    return LiveDecisionState(
        hand_id=history.hand_id,
        street=replayed.street,
        board=replayed.board,
        hero_combo=("Qc", "Qd"),
        hero_position="BB",
        villain_position="BTN",
        hero_is_oop=True,
        active_villains=1,
        pot_bb=replayed.pot_bb,
        hero_stack_bb=replayed.stack_map[5],
        villain_stack_bb=replayed.stack_map[3],
        hero_current_bet_bb=replayed.street_contribution_map[5],
        villain_current_bet_bb=replayed.street_contribution_map[3],
        amount_to_call_bb=replayed.amount_to_call_bb,
        legal_actions=legal_actions,
        street_root_confirmed=True,
        public_hand=history,
    )


class _SimulationRanges:
    def __init__(self) -> None:
        self.boards_seen: list[tuple[str, ...]] = []
        self.public_event_counts_seen: list[int | None] = []

    def ranges_for(self, state: LiveDecisionState) -> RangeBundle:
        self.boards_seen.append(state.board)
        self.public_event_counts_seen.append(
            len(state.public_hand.events)
            if state.public_hand is not None
            else None
        )
        return RangeBundle(
            oop=PlayerRange(
                Position.OOP,
                (WeightedCombo(("Qc", "Qd"), Decimal(1)),),
            ),
            ip=PlayerRange(
                Position.IP,
                (
                    WeightedCombo(("4s", "4d"), Decimal("0.5")),
                    WeightedCombo(("As", "Ad"), Decimal("0.5")),
                    WeightedCombo(("Kc", "Kd"), Decimal(1)),
                    WeightedCombo(("Tc", "Td"), Decimal(1)),
                ),
            ),
            profile_id="simulation-only-tiny-ranges",
            hero_combo_injected=False,
            provenance="deterministic local simulation fixture",
            approximations=(
                "tiny synthetic ranges validate plumbing, not production strategy",
            ),
            approximate=True,
        )


def _fake_result(spec: ContinuationSpec) -> ContinuationResult:
    dead = set(spec.current_board)
    source_ranges = {
        Position.OOP: spec.oop_range,
        Position.IP: spec.ip_range,
    }
    conditional_ranges = []
    for position in (Position.OOP, Position.IP):
        compatible = [
            combo
            for combo in source_ranges[position].combos
            if not dead.intersection(combo.cards)
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
    acting = [
        combo
        for combo in source_ranges[spec.acting_player].combos
        if not dead.intersection(combo.cards)
    ]
    acting_total = sum((combo.weight for combo in acting), Decimal(0))
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
        for combo in acting
    )
    metadata = SolverMetadata(
        solver_name=spec.parameters.solver_name,
        solver_version=spec.parameters.solver_commit,
        iterations=1,
        elapsed_seconds=Decimal("0.001"),
        exploitability=Decimal(0),
        converged=True,
        extra=(
            ("binary_sha256", _FAKE_BINARY_DIGEST),
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


class _FakeEngine:
    binary_sha256 = _FAKE_BINARY_DIGEST

    def __init__(self) -> None:
        self.calls = 0

    def solve_continuation(
        self,
        spec: ContinuationSpec,
    ) -> ContinuationResult:
        self.calls += 1
        return _fake_result(spec)


class _CapturingRouter:
    def __init__(self, router: LiveGTORouter) -> None:
        self.router = router
        self.config = router.config
        self.last_outcome = None

    def evaluate(self, state: LiveDecisionState):
        self.last_outcome = self.router.evaluate(state)
        return self.last_outcome


def run_simulation(args: argparse.Namespace) -> dict[str, object]:
    target = _target_history(args.street)
    collected = _collect_history(target)
    mac_state = _decision_state(collected)

    request_id = f"simulation-{args.mode}-{args.street.lower()}"
    request_payload = encode_json(
        build_evaluate_request(request_id, mac_state)
    )
    parsed_id, server_state, fingerprint = parse_evaluate_request(
        request_payload
    )
    if fingerprint != decision_fingerprint(mac_state):
        raise RuntimeError("Mac/server decision fingerprints differ")

    ranges = _SimulationRanges()
    fake_engine = _FakeEngine() if args.mode == "dry-run" else None
    config = LiveGTOConfig(
        enabled=True,
        owned_simulator_acknowledged=True,
        engine_path=args.engine,
        cache_path=args.cache,
        target_exploitability_pct=args.target,
        max_iterations=args.max_iterations,
        flop_timeout_seconds=args.timeout,
        mix_secret=b"simulation-stable-mix-secret-0001",
    )
    router = LiveGTORouter(
        config,
        range_provider=ranges,
        engine_factory=(
            (lambda *unused_args, **unused_kwargs: fake_engine)
            if fake_engine is not None
            else EngineClient
        ),
    )
    capturing = _CapturingRouter(router)
    service = EvaluationService(capturing)
    response_body, service_cache_hit = service.evaluate(
        parsed_id,
        fingerprint,
        server_state,
    )
    replay_body, replay_service_cache_hit = service.evaluate(
        parsed_id,
        fingerprint,
        server_state,
    )
    if replay_body != response_body:
        raise RuntimeError("idempotent server replay changed the response")
    remote_outcome = outcome_from_wire(
        response_body,
        expected_request_id=parsed_id,
        expected_fingerprint=fingerprint,
    )
    local_outcome = capturing.last_outcome
    if local_outcome is None:
        raise RuntimeError("server router did not produce an outcome")

    conditional_counts: dict[str, int] = {}
    future_blocker_removed = None
    solver_iterations = None
    solver_elapsed_seconds = None
    exploitability_pct_of_pot = None
    estimated_uncompressed_bytes = None
    if isinstance(local_outcome.result, ContinuationResult):
        for conditional_range in local_outcome.result.conditional_ranges:
            conditional_counts[conditional_range.position.value] = len(
                conditional_range.combos
            )
        future_blocker_removed = not any(
            set(combo.cards) == {"4s", "4d"}
            for conditional_range in local_outcome.result.conditional_ranges
            for combo in conditional_range.combos
        )
        metadata = local_outcome.result.metadata
        extra = dict(metadata.extra)
        solver_iterations = metadata.iterations
        solver_elapsed_seconds = str(metadata.elapsed_seconds)
        exploitability_pct_of_pot = extra.get(
            "exploitability_pct_of_pot"
        )
        estimated_uncompressed_bytes = extra.get(
            "estimated_uncompressed_bytes"
        )

    action_lines = [
        line
        for line in remote_outcome.analysis.splitlines()
        if line.startswith("**Action:**") or line.startswith("**Size:**")
    ]
    spec = local_outcome.spec
    path_steps = len(spec.path) if isinstance(spec, ContinuationSpec) else 0
    return {
        "mode": args.mode,
        "street": args.street,
        "status": remote_outcome.status.value,
        "reason": remote_outcome.reason,
        "end_to_end_latency_seconds": remote_outcome.latency_seconds,
        "events_collected": len(collected.events),
        "request_bytes": len(request_payload),
        "decision_fingerprint": fingerprint,
        "hu_detected": server_state.active_villains == 1,
        "range_source_board_seen": [
            list(board) for board in ranges.boards_seen
        ],
        "range_source_public_event_counts_seen": (
            ranges.public_event_counts_seen
        ),
        "future_cards_hidden_from_range_source": all(
            len(board) == 3 for board in ranges.boards_seen
        )
        and all(
            count == 7
            for count in ranges.public_event_counts_seen
        ),
        "solver_path_steps": path_steps,
        "conditional_combo_counts": conditional_counts,
        "future_blocker_removed_4s4d": future_blocker_removed,
        "solver_iterations": solver_iterations,
        "solver_elapsed_seconds": solver_elapsed_seconds,
        "exploitability_pct_of_pot": exploitability_pct_of_pot,
        "estimated_uncompressed_bytes": estimated_uncompressed_bytes,
        "solver_cache_hit": remote_outcome.cache_hit,
        "service_first_cache_hit": service_cache_hit,
        "service_replay_cache_hit": replay_service_cache_hit,
        "fake_engine_calls": (
            fake_engine.calls if fake_engine is not None else None
        ),
        "spec_key": remote_outcome.effective_spec_key,
        "action": action_lines,
        "model": remote_outcome.model,
        "source": remote_outcome.source,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate the complete OCR-Mac-to-GTO-server path",
    )
    parser.add_argument(
        "--mode",
        choices=("dry-run", "real"),
        default="dry-run",
    )
    parser.add_argument(
        "--street",
        choices=("TURN", "RIVER"),
        default="TURN",
    )
    parser.add_argument(
        "--engine",
        type=Path,
        default=Path(
            "gto_oracle_engine/target/release/gto-oracle-engine"
        ),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("/private/tmp/gto-full-process-simulation.sqlite3"),
    )
    parser.add_argument(
        "--target",
        type=Decimal,
        default=Decimal("100"),
    )
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument(
        "--timeout",
        type=Decimal,
        default=Decimal("120"),
    )
    return parser.parse_args()


def main() -> int:
    summary = run_simulation(_arguments())
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "SOLVED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
