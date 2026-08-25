#!/usr/bin/env python3
"""Public API and CLI for the explicit raked HU jam/fold reference game.

The implementation is deliberately split by responsibility:

* :mod:`hu_jam_fold_model` owns rake, payoffs, types, and serialization;
* :mod:`hu_jam_fold_solver` owns RM+ search and bilateral best responses;
* :mod:`hu_jam_fold_artifact` owns strict JSON and certificate verification.

This facade preserves the original import and command-line contract. Nothing
in this harness claims to solve full continuous-action HUNL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from hu_jam_fold_artifact import (
    atomic_write_json,
    load_game_artifact,
    load_solution,
    parse_game_payload,
    verify_solution_payload,
)
from hu_jam_fold_model import (
    ALGORITHM_ID,
    ARTIFACT_TYPE,
    CERTIFICATE_METHOD,
    CERTIFICATE_NUMERIC,
    GAME_ID,
    GAME_SCHEMA_VERSION,
    RAKE_BASIS,
    RAKE_ROUNDING,
    SOLUTION_CONCEPT,
    SOLUTION_SCHEMA_VERSION,
    TYPE_SEMANTICS,
    BehaviorStrategy,
    BestResponseReport,
    JamFoldGame,
    JamFoldProfile,
    JamFoldSolution,
    JamFoldValidationError,
    PrivateTypeDeal,
    RakeProfile,
)
from hu_jam_fold_solver import measure_best_responses, solve_game


__all__ = [
    "ALGORITHM_ID",
    "ARTIFACT_TYPE",
    "CERTIFICATE_METHOD",
    "CERTIFICATE_NUMERIC",
    "GAME_ID",
    "GAME_SCHEMA_VERSION",
    "RAKE_BASIS",
    "RAKE_ROUNDING",
    "SOLUTION_CONCEPT",
    "SOLUTION_SCHEMA_VERSION",
    "TYPE_SEMANTICS",
    "BehaviorStrategy",
    "BestResponseReport",
    "JamFoldGame",
    "JamFoldProfile",
    "JamFoldSolution",
    "JamFoldValidationError",
    "PrivateTypeDeal",
    "RakeProfile",
    "load_game_artifact",
    "load_solution",
    "main",
    "measure_best_responses",
    "parse_game_payload",
    "solve_game",
    "verify_solution_payload",
]


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve and verify an explicit finite HU jam/fold game."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    solve = commands.add_parser("solve", help="search for and measure a strategy")
    solve.add_argument("--game", required=True, type=Path)
    solve.add_argument("--output", required=True, type=Path)
    solve.add_argument("--target-epsilon-bb", type=float, default=0.001)
    solve.add_argument("--max-iterations", type=int, default=250_000)
    solve.add_argument("--min-iterations", type=int, default=2_000)
    solve.add_argument("--check-every", type=int, default=1_000)
    solve.add_argument("--force", action="store_true")
    verify = commands.add_parser(
        "verify",
        help="recompute both best responses and validate a solution",
    )
    verify.add_argument("--game", required=True, type=Path)
    verify.add_argument("--solution", required=True, type=Path)
    verify.add_argument("--max-epsilon-bb", required=True, type=float)
    return parser


def _solve_command(args: argparse.Namespace, game: JamFoldGame) -> int:
    solution = solve_game(
        game,
        target_epsilon_bb=args.target_epsilon_bb,
        max_iterations=args.max_iterations,
        min_iterations=args.min_iterations,
        check_every=args.check_every,
    )
    atomic_write_json(args.output, solution.to_payload(), force=args.force)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "game_fingerprint": game.fingerprint,
                "iterations": solution.iterations,
                "target_reached": solution.converged,
                "epsilon_bb": solution.report.epsilon_bb,
                "sb_gap_bb": solution.report.sb_deviation_gain_bb,
                "bb_gap_bb": solution.report.bb_deviation_gain_bb,
            },
            sort_keys=True,
        )
    )
    return 0 if solution.converged else 2


def _verify_command(args: argparse.Namespace, game: JamFoldGame) -> int:
    verified = load_solution(
        game,
        args.solution,
        max_epsilon_bb=args.max_epsilon_bb,
        require_target_reached=True,
    )
    print(
        json.dumps(
            {
                "solution": str(args.solution.resolve()),
                "verified": True,
                "epsilon_bb": verified.report.epsilon_bb,
                "sb_gap_bb": verified.report.sb_deviation_gain_bb,
                "bb_gap_bb": verified.report.bb_deviation_gain_bb,
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_cli_parser().parse_args(argv)
    try:
        game = load_game_artifact(args.game)
        if args.command == "solve":
            return _solve_command(args, game)
        return _verify_command(args, game)
    except JamFoldValidationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
