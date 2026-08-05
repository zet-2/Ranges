"""Tests for the full GTO process simulation harness."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from gto_full_process_simulation import run_simulation


class FullProcessSimulationTests(unittest.TestCase):
    def _run(self, street: str) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as directory:
            return run_simulation(
                SimpleNamespace(
                    mode="dry-run",
                    street=street,
                    engine=Path("unused"),
                    cache=Path(directory) / "simulation.sqlite3",
                    target=Decimal("100"),
                    max_iterations=10,
                    timeout=Decimal("10"),
                )
            )

    def test_turn_process_is_lossless_and_idempotent(self):
        result = self._run("TURN")

        self.assertEqual("SOLVED", result["status"])
        self.assertEqual(11, result["events_collected"])
        self.assertEqual(4, result["solver_path_steps"])
        self.assertTrue(result["hu_detected"])
        self.assertTrue(result["future_cards_hidden_from_range_source"])
        self.assertTrue(result["future_blocker_removed_4s4d"])
        self.assertFalse(result["service_first_cache_hit"])
        self.assertTrue(result["service_replay_cache_hit"])
        self.assertEqual(1, result["fake_engine_calls"])

    def test_river_process_preserves_both_chance_steps(self):
        result = self._run("RIVER")

        self.assertEqual("SOLVED", result["status"])
        self.assertEqual(15, result["events_collected"])
        self.assertEqual(8, result["solver_path_steps"])
        self.assertTrue(result["future_cards_hidden_from_range_source"])
        self.assertTrue(result["future_blocker_removed_4s4d"])


if __name__ == "__main__":
    unittest.main()
