"""Tests for HAL intent detector."""
import os
import sys
import unittest

from lib.hal.intent_detector import Intent, IntentDetector


class TestIntentDetector(unittest.TestCase):

    def setUp(self):
        self.detector = IntentDetector()

    def test_debug_intent(self):
        intent = self.detector.classify("Fix the null pointer bug in auth.py line 42")
        self.assertEqual(intent.category, "debug")
        self.assertGreater(intent.confidence, 0.7)

    def test_code_intent(self):
        intent = self.detector.classify("Write a function that sorts a list")
        self.assertEqual(intent.category, "code")

    def test_explain_intent(self):
        intent = self.detector.classify("Explain how async/await works in Python")
        self.assertEqual(intent.category, "explain")

    def test_research_intent(self):
        intent = self.detector.classify("Research the best database for this use case")
        self.assertEqual(intent.category, "research")

    def test_review_intent(self):
        intent = self.detector.classify("Review this code for security issues")
        self.assertEqual(intent.category, "review")

    def test_command_intent(self):
        intent = self.detector.classify("/status")
        self.assertEqual(intent.category, "command")
        self.assertGreater(intent.confidence, 0.9)

    def test_chat_intent(self):
        intent = self.detector.classify("Hello, how are you?")
        self.assertEqual(intent.category, "chat")

    def test_empty_message(self):
        intent = self.detector.classify("")
        self.assertEqual(intent.category, "chat")
        self.assertEqual(intent.confidence, 0.0)

    def test_code_indicator_boosts(self):
        plain = self.detector.classify("Fix the authentication issue")
        with_code = self.detector.classify("Fix the authentication issue in auth.py line 42")
        self.assertGreaterEqual(with_code.confidence, plain.confidence)

    def test_suggested_mode(self):
        intent = self.detector.classify("Write a new module")
        self.assertEqual(intent.suggested_mode, "code")

    def test_suggested_tools(self):
        intent = self.detector.classify("Fix the bug in server.py")
        self.assertIn("read", intent.suggested_tools)

    def test_classify_batch(self):
        intents = self.detector.classify_batch(["Fix bug", "Explain this", "Hello"])
        self.assertEqual(len(intents), 3)


class TestIntent(unittest.TestCase):

    def test_frozen(self):
        intent = Intent("code", 0.8, "code", ("read",))
        with self.assertRaises(AttributeError):
            intent.category = "debug"


if __name__ == "__main__":
    unittest.main()
