"""Tests for HAL smart suggestions."""
import os
import sys
import unittest

from lib.hal.suggestions import Suggestion, SuggestionEngine
from lib.hal.types import ChatMessage


def _msg(role: str, content: str) -> ChatMessage:
    return ChatMessage(role=role, content=content)


class TestSuggestionEngine(unittest.TestCase):

    def test_empty_conversation(self):
        engine = SuggestionEngine()
        suggestions = engine.suggest([])
        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0].category, "tip")

    def test_normal_conversation(self):
        history = [_msg("user", "Hello"), _msg("assistant", "Hi there")]
        engine = SuggestionEngine()
        suggestions = engine.suggest(history)
        self.assertIsInstance(suggestions, list)

    def test_long_conversation_warning(self):
        history = []
        for i in range(25):
            history.append(_msg("user", f"Question {i}"))
            history.append(_msg("assistant", f"Answer {i}"))
        engine = SuggestionEngine()
        suggestions = engine.suggest(history)
        warnings = [s for s in suggestions if s.trigger == "long_conversation"]
        self.assertGreater(len(warnings), 0)

    def test_error_pattern_detection(self):
        history = [
            _msg("user", "Fix it"),
            _msg("assistant", "Error: failed to compile the module"),
            _msg("user", "Try again"),
            _msg("assistant", "Error: another exception occurred in the build"),
        ]
        engine = SuggestionEngine()
        suggestions = engine.suggest(history)
        error_suggestions = [s for s in suggestions if s.trigger == "repeated_errors"]
        self.assertGreater(len(error_suggestions), 0)

    def test_code_without_tests(self):
        history = [
            _msg("user", "Write a function"),
            _msg("assistant", "Here:\n```python\ndef foo(): return 42\n```"),
        ]
        engine = SuggestionEngine()
        suggestions = engine.suggest(history)
        test_suggestions = [s for s in suggestions if s.trigger == "code_no_tests"]
        self.assertGreater(len(test_suggestions), 0)

    def test_high_cost_warning(self):
        engine = SuggestionEngine()
        suggestions = engine.suggest(
            [_msg("user", "Hi"), _msg("assistant", "Hello")],
            context={"total_cost_usd": 2.50},
        )
        cost_warnings = [s for s in suggestions if s.trigger == "high_cost"]
        self.assertGreater(len(cost_warnings), 0)

    def test_max_three_suggestions(self):
        # Create a scenario with many triggers
        history = []
        for i in range(25):
            history.append(_msg("user", f"Question {i}?"))
            history.append(_msg("assistant", f"Error: failed {i}\n```code```"))
        engine = SuggestionEngine()
        suggestions = engine.suggest(history, context={"total_cost_usd": 5.0})
        self.assertLessEqual(len(suggestions), 3)


class TestSuggestion(unittest.TestCase):

    def test_frozen(self):
        s = Suggestion("Do this", "tip", 0.5, "test")
        with self.assertRaises(AttributeError):
            s.text = "changed"

    def test_fields(self):
        s = Suggestion("Save your work", "warning", 0.9, "context_full")
        self.assertEqual(s.category, "warning")
        self.assertEqual(s.confidence, 0.9)


if __name__ == "__main__":
    unittest.main()
