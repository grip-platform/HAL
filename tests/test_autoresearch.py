"""Tests for HAL AutoResearchLoop."""
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.hal.autoresearch import AutoResearchLoop, Experiment


class TestGoodhartGate(unittest.TestCase):

    def setUp(self):
        self.loop = AutoResearchLoop.__new__(AutoResearchLoop)
        self.loop._direction = "improve routing"

    def test_passes_normal_description(self):
        self.assertTrue(self.loop._goodhart_check("Reorder fallback chain to put gemini first"))

    def test_blocks_overfit_description(self):
        self.assertFalse(self.loop._goodhart_check("Hardcod gemma for this specific test case"))

    def test_blocks_magic_number(self):
        self.assertFalse(self.loop._goodhart_check("Use magic number 42 to pass rubric check"))


class TestAutoResearchLoop(unittest.TestCase):

    def _make_loop(self, score_seq):
        loop = AutoResearchLoop.__new__(AutoResearchLoop)
        loop._repo = "/tmp/fake"
        loop._budget_s = 1.0
        loop._direction = "test"
        loop.score = MagicMock(side_effect=score_seq)
        return loop

    def test_keeps_improvement(self):
        # hill_climb calls score() once for best_score, then run_experiment calls score() twice
        loop = self._make_loop([0.9, 0.9, 0.95])
        applied = []
        reverted = []
        result = loop.hill_climb([
            ("improve routing", lambda: applied.append(1), lambda: reverted.append(1)),
        ])
        self.assertEqual(len(result.experiments), 1)
        self.assertTrue(result.experiments[0].kept)
        self.assertEqual(len(reverted), 0)

    def test_reverts_regression(self):
        loop = self._make_loop([0.9, 0.9, 0.85])
        reverted = []
        result = loop.hill_climb([
            ("bad change", lambda: None, lambda: reverted.append(1)),
        ])
        self.assertFalse(result.experiments[0].kept)
        self.assertEqual(len(reverted), 1)

    def test_goodhart_blocks_experiment(self):
        # When Goodhart blocks, score() is called once for score_before (in run_experiment),
        # and once by hill_climb for initial best_score — so two calls total
        loop = self._make_loop([1.0, 1.0])
        result = loop.hill_climb([
            ("Hardcod this specific test value", lambda: None, lambda: None),
        ])
        self.assertFalse(result.experiments[0].reflection_passed)
        self.assertFalse(result.experiments[0].kept)


if __name__ == "__main__":
    unittest.main()
