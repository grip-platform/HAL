"""Tests for HAL model comparison."""
import os
import sys
import unittest

from lib.hal.model_compare import ComparisonResult, ModelComparator, ModelResult
from lib.hal.quality_scorer import QualityScore


class TestModelResult(unittest.TestCase):

    def test_success(self):
        r = ModelResult("m", "response", QualityScore(20, 20, 20, 20))
        self.assertTrue(r.success)

    def test_failure(self):
        r = ModelResult("m", "", QualityScore(0, 0, 0, 0), error="timeout")
        self.assertFalse(r.success)

    def test_to_dict(self):
        r = ModelResult("model/name", "text", QualityScore(25, 20, 15, 10))
        d = r.to_dict()
        self.assertEqual(d["model"], "model/name")
        self.assertIn("quality", d)


class TestComparisonResult(unittest.TestCase):

    def test_winner(self):
        result = ComparisonResult(query="test", results=[
            ModelResult("a", "good", QualityScore(25, 25, 20, 20)),
            ModelResult("b", "ok", QualityScore(15, 15, 10, 10)),
        ])
        self.assertEqual(result.winner, "a")

    def test_fastest(self):
        result = ComparisonResult(query="test", results=[
            ModelResult("slow", "r", QualityScore(20, 20, 20, 20), latency_ms=500),
            ModelResult("fast", "r", QualityScore(20, 20, 20, 20), latency_ms=50),
        ])
        self.assertEqual(result.fastest, "fast")

    def test_cheapest(self):
        result = ComparisonResult(query="test", results=[
            ModelResult("expensive", "r", QualityScore(20, 20, 20, 20), cost_usd=0.05),
            ModelResult("cheap", "r", QualityScore(20, 20, 20, 20), cost_usd=0.001),
        ])
        self.assertEqual(result.cheapest, "cheap")

    def test_empty_results(self):
        result = ComparisonResult(query="test")
        self.assertEqual(result.winner, "")
        self.assertEqual(result.fastest, "")

    def test_summary(self):
        result = ComparisonResult(query="test", results=[
            ModelResult("a", "resp", QualityScore(20, 20, 15, 15)),
        ])
        self.assertIn("a", result.summary)

    def test_to_dict(self):
        result = ComparisonResult(query="q", results=[
            ModelResult("m", "r", QualityScore(1, 2, 3, 4)),
        ])
        d = result.to_dict()
        self.assertIn("winner", d)
        self.assertIn("results", d)


class TestModelComparatorNoRegistry(unittest.TestCase):

    def test_no_registry(self):
        comp = ModelComparator()
        result = comp.compare("test", ["m1", "m2"])
        self.assertEqual(len(result.results), 2)
        self.assertTrue(all(not r.success for r in result.results))


if __name__ == "__main__":
    unittest.main()
