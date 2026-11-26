"""Tests for HAL conversation insights engine."""
import os
import sys
import unittest

from lib.hal.insights import (
    InsightsEngine,
    _detect_friction,
    _extract_topics,
    _generate_suggestions,
)
from lib.hal.types import ChatMessage


def _msg(role: str, content: str) -> ChatMessage:
    return ChatMessage(role=role, content=content)


class TestInsightsEngine(unittest.TestCase):

    def test_empty_history(self):
        engine = InsightsEngine()
        report = engine.analyse([])
        self.assertEqual(report.turn_count, 0)

    def test_basic_conversation(self):
        history = [
            _msg("user", "How do I fix this bug?"),
            _msg("assistant", "You need to check the null pointer on line 42."),
            _msg("user", "Thanks, what about the tests?"),
            _msg("assistant", "Run pytest with the -v flag."),
        ]
        engine = InsightsEngine()
        report = engine.analyse(history)
        self.assertEqual(report.turn_count, 2)
        self.assertGreater(report.total_tokens_est, 0)
        self.assertEqual(report.questions_asked, 2)

    def test_code_blocks_counted(self):
        history = [
            _msg("user", "Show me an example"),
            _msg("assistant", "Here:\n```python\nx = 1\n```\nAnd:\n```python\ny = 2\n```"),
        ]
        engine = InsightsEngine()
        report = engine.analyse(history)
        self.assertEqual(report.code_blocks, 2)

    def test_summary_string(self):
        history = [_msg("user", "Hello"), _msg("assistant", "Hi")]
        report = InsightsEngine().analyse(history)
        self.assertIn("1 turns", report.summary)

    def test_to_dict(self):
        report = InsightsEngine().analyse([_msg("user", "Hi"), _msg("assistant", "Hey")])
        d = report.to_dict()
        self.assertIn("turn_count", d)
        self.assertIn("efficiency_score", d)


class TestDetectFriction(unittest.TestCase):

    def test_no_friction(self):
        history = [_msg("user", "Hello"), _msg("assistant", "Hi")]
        friction = _detect_friction(history)
        self.assertEqual(len(friction), 0)

    def test_correction_detected(self):
        history = [
            _msg("user", "Fix the auth bug"),
            _msg("assistant", "I changed the CSS."),
            _msg("user", "No not that, I said fix the auth bug"),
        ]
        friction = _detect_friction(history)
        corrections = [f for f in friction if "correction" in f.description.lower()]
        self.assertGreater(len(corrections), 0)

    def test_frustration_detected(self):
        history = [
            _msg("user", "Fix it"),
            _msg("assistant", "Done."),
            _msg("user", "Stop!!! That's wrong!!!"),
        ]
        friction = _detect_friction(history)
        high = [f for f in friction if f.severity == "high"]
        self.assertGreater(len(high), 0)


class TestExtractTopics(unittest.TestCase):

    def test_single_topic(self):
        msgs = [_msg("user", "How does the authentication system work?")]
        topics = _extract_topics(msgs)
        self.assertGreaterEqual(len(topics), 1)

    def test_topic_shift(self):
        msgs = [
            _msg("user", "Fix the authentication login bug in the auth module"),
            _msg("user", "Now update the database migration schema for postgres"),
        ]
        topics = _extract_topics(msgs)
        self.assertGreaterEqual(len(topics), 1)


class TestGenerateSuggestions(unittest.TestCase):

    def test_smooth_session(self):
        suggestions = _generate_suggestions(5, 10_000, [], 1, 500)
        self.assertIn("smoothly", suggestions[0].lower())

    def test_verbose_warning(self):
        suggestions = _generate_suggestions(5, 10_000, [], 1, 3000)
        verbose_warnings = [s for s in suggestions if "verbose" in s.lower()]
        self.assertGreater(len(verbose_warnings), 0)

    def test_topic_shift_warning(self):
        suggestions = _generate_suggestions(10, 50_000, [], 8, 500)
        shift_warnings = [s for s in suggestions if "topic" in s.lower()]
        self.assertGreater(len(shift_warnings), 0)


class TestEfficiencyScore(unittest.TestCase):

    def test_zero_tokens(self):
        report = InsightsEngine().analyse([])
        self.assertEqual(report.efficiency_score, 0.0)

    def test_bounded(self):
        history = [_msg("user", "x"), _msg("assistant", "y")]
        report = InsightsEngine().analyse(history)
        self.assertGreaterEqual(report.efficiency_score, 0.0)
        self.assertLessEqual(report.efficiency_score, 1.0)


if __name__ == "__main__":
    unittest.main()
