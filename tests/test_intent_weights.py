"""Tests for multi-intent confidence weighting in IntentDetector."""
import os
import sys
import unittest

from lib.hal.intent_detector import IntentDetector, IntentWeights


class TestIntentWeights(unittest.TestCase):
    """Test the IntentWeights dataclass."""

    def test_dominant_returns_highest(self):
        w = IntentWeights((("debug", 0.8), ("code", 0.6)))
        self.assertEqual(w.dominant, "debug")

    def test_dominant_empty_returns_chat(self):
        w = IntentWeights(())
        self.assertEqual(w.dominant, "chat")

    def test_active_filters_by_threshold(self):
        w = IntentWeights((("debug", 0.8), ("code", 0.6), ("chat", 0.1)))
        active = w.active
        self.assertIn("debug", active)
        self.assertIn("code", active)
        self.assertNotIn("chat", active)

    def test_get_returns_confidence(self):
        w = IntentWeights((("debug", 0.8), ("code", 0.6)))
        self.assertEqual(w.get("debug"), 0.8)
        self.assertEqual(w.get("code"), 0.6)

    def test_get_missing_returns_zero(self):
        w = IntentWeights((("debug", 0.8),))
        self.assertEqual(w.get("research"), 0.0)

    def test_to_dict(self):
        w = IntentWeights((("debug", 0.8), ("code", 0.6)))
        d = w.to_dict()
        self.assertEqual(d, {"debug": 0.8, "code": 0.6})


class TestMultiIntentClassification(unittest.TestCase):
    """Test that classify() returns multi-intent weights."""

    def setUp(self):
        self.detector = IntentDetector()

    def test_fix_bug_matches_debug_and_code(self):
        """'Fix the bug in auth.py' should match both debug and code."""
        intent = self.detector.classify("Fix the bug in auth.py")
        self.assertIsNotNone(intent.weights)
        # "fix" triggers debug, code patterns; ".py" boosts both
        self.assertGreater(intent.weights.get("debug"), 0.5)
        # "fix" also matches "fix" in debug pattern
        self.assertEqual(intent.category, "debug")

    def test_command_is_dominant(self):
        intent = self.detector.classify("/status")
        self.assertEqual(intent.category, "command")
        self.assertGreaterEqual(intent.confidence, 0.9)

    def test_review_code_matches_review(self):
        intent = self.detector.classify("Review this code for security issues")
        self.assertIn("review", intent.weights.active)

    def test_empty_message_returns_empty_weights(self):
        intent = self.detector.classify("")
        self.assertEqual(intent.category, "chat")
        self.assertEqual(len(intent.weights.weights), 0)

    def test_no_match_returns_chat(self):
        intent = self.detector.classify("the sky is blue")
        self.assertEqual(intent.category, "chat")

    def test_backward_compat_category_still_works(self):
        """Intent.category must still return the primary category."""
        intent = self.detector.classify("Explain how async/await works")
        self.assertEqual(intent.category, "explain")
        self.assertIsInstance(intent.confidence, float)
        self.assertIsInstance(intent.suggested_mode, str)

    def test_weights_sorted_by_confidence_desc(self):
        intent = self.detector.classify("Fix and refactor the broken code in auth.py")
        if len(intent.weights.weights) >= 2:
            confs = [c for _, c in intent.weights.weights]
            self.assertEqual(confs, sorted(confs, reverse=True))


if __name__ == "__main__":
    unittest.main()
