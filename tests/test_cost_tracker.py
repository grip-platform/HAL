"""Tests for HAL cost tracker including token estimation."""
import os
import sys
import unittest

from lib.hal.cost_tracker import (
    CostTracker,
    calculate_cost,
    estimate_cost,
    estimate_tokens,
)
from lib.hal.types import TokenUsage


class TestEstimateTokens(unittest.TestCase):

    def test_empty_string(self):
        self.assertEqual(estimate_tokens(""), 0)

    def test_single_word(self):
        tokens = estimate_tokens("hello")
        self.assertGreater(tokens, 0)

    def test_typical_sentence(self):
        tokens = estimate_tokens("The quick brown fox jumps over the lazy dog")
        # 9 words / 0.75 = 12 tokens
        self.assertEqual(tokens, 12)

    def test_long_text(self):
        text = " ".join(["word"] * 1000)
        tokens = estimate_tokens(text)
        # 1000 words / 0.75 = 1333
        self.assertEqual(tokens, 1333)


class TestEstimateCost(unittest.TestCase):

    def test_groq_free_tier(self):
        cost = estimate_cost("llama-3.3-70b-versatile", "Hello world", 100)
        self.assertGreater(cost, 0)
        self.assertLess(cost, 0.001)  # Very cheap

    def test_local_is_free(self):
        cost = estimate_cost("local", "any text here", 1000)
        self.assertEqual(cost, 0.0)


class TestCalculateCost(unittest.TestCase):

    def test_claude_opus_cost(self):
        cost = calculate_cost("claude-opus-4-7", input_tokens=1000, output_tokens=500)
        # (1000/1M * 5.0) + (500/1M * 25.0) = 0.005 + 0.0125 = 0.0175
        self.assertAlmostEqual(cost, 0.0175, places=4)

    def test_local_free(self):
        cost = calculate_cost("local", input_tokens=100000, output_tokens=50000)
        self.assertEqual(cost, 0.0)


class TestCostTracker(unittest.TestCase):

    def test_record_and_snapshot(self):
        tracker = CostTracker()
        tracker.record("llama-3.3-70b-versatile", "groq",
                       input_tokens=100, output_tokens=50)
        snap = tracker.snapshot()
        self.assertEqual(snap.input_tokens, 100)
        self.assertEqual(snap.output_tokens, 50)
        self.assertGreaterEqual(snap.total_usd, 0)

    def test_per_provider(self):
        tracker = CostTracker()
        tracker.record("test-model", "groq", input_tokens=100)
        tracker.record("test-model", "gemini", input_tokens=200)
        providers = tracker.per_provider()
        self.assertIn("groq", providers)
        self.assertIn("gemini", providers)

    def test_reset(self):
        tracker = CostTracker()
        tracker.record("test", "test", input_tokens=1000)
        tracker.reset()
        self.assertEqual(tracker.total_usd, 0.0)

    def test_record_response(self):
        tracker = CostTracker()
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        cost = tracker.record_response("llama-3.3-70b-versatile", "groq", usage)
        self.assertGreaterEqual(cost, 0)


if __name__ == "__main__":
    unittest.main()
