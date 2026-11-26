"""Tests for HAL usage reporter."""
import os, sys, unittest
from lib.hal.usage_reporter import UsageReporter

class TestUsageReporter(unittest.TestCase):
    def test_empty(self):
        r = UsageReporter().report()
        self.assertEqual(r.total_queries, 0)

    def test_record_and_report(self):
        rep = UsageReporter()
        rep.record("groq", "llama", 500, 0.01)
        rep.record("gemini", "flash", 300, 0.005)
        r = rep.report("daily")
        self.assertEqual(r.total_queries, 2)
        self.assertEqual(r.total_tokens, 800)
        self.assertAlmostEqual(r.total_cost_usd, 0.015)

    def test_by_provider(self):
        rep = UsageReporter()
        rep.record("groq", "llama", 100, 0.01)
        rep.record("groq", "llama", 200, 0.02)
        r = rep.report()
        self.assertAlmostEqual(r.by_provider["groq"], 0.03)

    def test_by_model(self):
        rep = UsageReporter()
        rep.record("g", "llama", 100, 0.01)
        rep.record("g", "llama", 200, 0.02)
        r = rep.report()
        self.assertEqual(r.by_model["llama"], 2)

    def test_avg(self):
        rep = UsageReporter()
        rep.record("g", "m", 1000, 0.10)
        rep.record("g", "m", 2000, 0.20)
        r = rep.report()
        self.assertEqual(r.avg_tokens_per_query, 1500)

    def test_to_dict(self):
        d = UsageReporter().report().to_dict()
        self.assertIn("total_queries", d)

    def test_reset(self):
        rep = UsageReporter()
        rep.record("g", "m", 100, 0.01)
        rep.reset()
        self.assertEqual(rep.report().total_queries, 0)

if __name__ == "__main__":
    unittest.main()
