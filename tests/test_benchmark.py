"""Tests for HAL provider benchmark."""
import os
import sys
import unittest

from lib.hal.benchmark import Benchmark, BenchmarkResult, ModelScore, _BENCHMARK_PROMPTS


class TestModelScore(unittest.TestCase):

    def test_composite(self):
        score = ModelScore("test", avg_quality=80, avg_latency_ms=200, total_cost_usd=0.01)
        self.assertGreater(score.composite, 0)

    def test_to_dict(self):
        d = ModelScore("model/name", prompts_completed=5).to_dict()
        self.assertEqual(d["model"], "model/name")
        self.assertIn("composite", d)


class TestBenchmarkResult(unittest.TestCase):

    def test_leaderboard(self):
        result = BenchmarkResult(scores=[
            ModelScore("slow", avg_quality=50, avg_latency_ms=1000),
            ModelScore("fast", avg_quality=80, avg_latency_ms=100),
        ])
        self.assertEqual(result.leaderboard[0], "fast")

    def test_summary(self):
        result = BenchmarkResult(scores=[ModelScore("m1", avg_quality=75)])
        self.assertIn("m1", result.summary)

    def test_to_dict(self):
        d = BenchmarkResult(scores=[ModelScore("m1")]).to_dict()
        self.assertIn("leaderboard", d)

    def test_empty(self):
        self.assertEqual(BenchmarkResult().leaderboard, [])


class TestBenchmarkPrompts(unittest.TestCase):

    def test_has_prompts(self):
        self.assertGreater(len(_BENCHMARK_PROMPTS), 3)

    def test_categories(self):
        categories = {cat for cat, _ in _BENCHMARK_PROMPTS}
        self.assertIn("code", categories)
        self.assertIn("explain", categories)


class TestBenchmarkNoRegistry(unittest.TestCase):

    def test_run_without_registry(self):
        result = Benchmark().run(["m1", "m2"])
        self.assertEqual(len(result.scores), 2)
        self.assertTrue(all(s.errors > 0 for s in result.scores))


if __name__ == "__main__":
    unittest.main()
