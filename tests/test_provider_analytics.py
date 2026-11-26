"""Tests for HAL provider analytics."""
import os
import sys
import unittest

from lib.hal.provider_analytics import ProviderAnalytics, ProviderStats


class TestProviderStats(unittest.TestCase):

    def test_avg_latency(self):
        stats = ProviderStats(provider="groq", successful_calls=10, total_latency_ms=1000.0)
        self.assertAlmostEqual(stats.avg_latency_ms, 100.0)

    def test_error_rate(self):
        stats = ProviderStats(provider="groq", total_calls=10, successful_calls=8)
        self.assertAlmostEqual(stats.error_rate, 0.2)

    def test_error_rate_zero_calls(self):
        stats = ProviderStats(provider="groq")
        self.assertEqual(stats.error_rate, 0.0)

    def test_avg_tokens(self):
        stats = ProviderStats(provider="groq", successful_calls=5, total_tokens=2500)
        self.assertEqual(stats.avg_tokens_per_call, 500)

    def test_to_dict(self):
        stats = ProviderStats(provider="groq", total_calls=1, successful_calls=1)
        d = stats.to_dict()
        self.assertEqual(d["provider"], "groq")
        self.assertIn("avg_latency_ms", d)


class TestProviderAnalytics(unittest.TestCase):

    def test_record_and_get(self):
        analytics = ProviderAnalytics()
        analytics._stats.clear()
        analytics.record_call("groq", latency_ms=100, tokens=500, success=True)
        stats = analytics.get_stats("groq")
        self.assertEqual(stats.total_calls, 1)
        self.assertEqual(stats.successful_calls, 1)
        self.assertAlmostEqual(stats.avg_latency_ms, 100.0)

    def test_record_failure(self):
        analytics = ProviderAnalytics()
        analytics._stats.clear()
        analytics.record_call("groq", success=False)
        stats = analytics.get_stats("groq")
        self.assertEqual(stats.total_calls, 1)
        self.assertEqual(stats.successful_calls, 0)
        self.assertAlmostEqual(stats.error_rate, 1.0)

    def test_get_unknown_provider(self):
        analytics = ProviderAnalytics()
        stats = analytics.get_stats("nonexistent")
        self.assertEqual(stats.total_calls, 0)

    def test_all_stats_sorted(self):
        analytics = ProviderAnalytics()
        analytics._stats.clear()
        analytics.record_call("a", latency_ms=10, success=True)
        analytics.record_call("b", latency_ms=20, success=True)
        analytics.record_call("b", latency_ms=20, success=True)
        stats = analytics.all_stats()
        self.assertEqual(stats[0].provider, "b")

    def test_ranking_by_latency(self):
        analytics = ProviderAnalytics()
        analytics._stats.clear()
        analytics.record_call("slow", latency_ms=500, success=True)
        analytics.record_call("fast", latency_ms=50, success=True)
        ranking = analytics.ranking("latency")
        self.assertEqual(ranking[0], "fast")

    def test_ranking_by_reliability(self):
        analytics = ProviderAnalytics()
        analytics._stats.clear()
        analytics.record_call("reliable", success=True)
        analytics.record_call("reliable", success=True)
        analytics.record_call("flaky", success=True)
        analytics.record_call("flaky", success=False)
        ranking = analytics.ranking("reliability")
        self.assertEqual(ranking[0], "reliable")


if __name__ == "__main__":
    unittest.main()
