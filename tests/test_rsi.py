"""Tests for HAL RSI (Recursive Self-Improvement) engine."""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from lib.hal.rsi import (
    RSIEngine,
    _parse_patterns,
    _pattern_id,
    _propose_action,
)
from lib.hal.types import ChatMessage


class TestPatternId(unittest.TestCase):
    """Test stable pattern ID generation."""

    def test_deterministic(self):
        id1 = _pattern_id("user asks about models")
        id2 = _pattern_id("user asks about models")
        self.assertEqual(id1, id2)

    def test_case_insensitive(self):
        id1 = _pattern_id("User Asks About Models")
        id2 = _pattern_id("user asks about models")
        self.assertEqual(id1, id2)

    def test_different_descriptions_different_ids(self):
        id1 = _pattern_id("pattern A")
        id2 = _pattern_id("pattern B")
        self.assertNotEqual(id1, id2)

    def test_length(self):
        pid = _pattern_id("test")
        self.assertEqual(len(pid), 12)


class TestParsePatterns(unittest.TestCase):
    """Test LLM response parsing."""

    def test_valid_json_array(self):
        text = '[{"description": "user prefers concise answers", "category": "preference"}]'
        result = _parse_patterns(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["description"], "user prefers concise answers")

    def test_json_embedded_in_text(self):
        text = 'Here are the patterns:\n[{"description": "test"}]\nDone.'
        result = _parse_patterns(text)
        self.assertEqual(len(result), 1)

    def test_invalid_json(self):
        result = _parse_patterns("not json at all")
        self.assertEqual(result, [])

    def test_empty_array(self):
        result = _parse_patterns("[]")
        self.assertEqual(result, [])

    def test_filters_invalid_entries(self):
        text = '[{"description": "valid"}, {"no_description": true}, "string"]'
        result = _parse_patterns(text)
        self.assertEqual(len(result), 1)


class TestProposeAction(unittest.TestCase):
    """Test pattern-to-proposal mapping."""

    def test_error_recovery_becomes_hook(self):
        ptype, _ = _propose_action({"category": "error_recovery", "description": "test"})
        self.assertEqual(ptype, "hook")

    def test_tool_usage_becomes_skill(self):
        ptype, _ = _propose_action({"category": "tool_usage", "description": "test"})
        self.assertEqual(ptype, "skill")

    def test_unknown_category_becomes_memory(self):
        ptype, _ = _propose_action({"category": "unknown_thing", "description": "test"})
        self.assertEqual(ptype, "memory")


class TestRSIEngine(unittest.TestCase):
    """Test the RSI engine core functionality."""

    def setUp(self):
        self.registry = MagicMock()
        self.registry.list_providers.return_value = []

    @patch("lib.hal.rsi._PATTERNS_PATH", "/tmp/hal-test-rsi-patterns.json")
    def test_short_history_returns_empty(self):
        engine = RSIEngine(self.registry)
        history = [ChatMessage(role="user", content="hi")]
        proposals = engine.analyse(history)
        self.assertEqual(proposals, [])

    @patch("lib.hal.rsi._PATTERNS_PATH", "/tmp/hal-test-rsi-patterns2.json")
    def test_get_patterns_empty_initially(self):
        try:
            os.remove("/tmp/hal-test-rsi-patterns2.json")
        except FileNotFoundError:
            pass
        engine = RSIEngine(self.registry)
        self.assertEqual(engine.get_patterns(), [])

    @patch("lib.hal.rsi._PATTERNS_PATH", "/tmp/hal-test-rsi-patterns3.json")
    def test_record_and_retrieve_pattern(self):
        try:
            os.remove("/tmp/hal-test-rsi-patterns3.json")
        except FileNotFoundError:
            pass
        engine = RSIEngine(self.registry)
        engine._record_pattern({
            "description": "user asks about cost",
            "category": "query_style",
            "example": "how much did that cost?",
        })
        engine._save_patterns()
        patterns = engine.get_patterns()
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].occurrences, 1)
        self.assertEqual(patterns[0].category, "query_style")

    @patch("lib.hal.rsi._PATTERNS_PATH", "/tmp/hal-test-rsi-patterns4.json")
    def test_ysh_threshold_generates_proposal(self):
        try:
            os.remove("/tmp/hal-test-rsi-patterns4.json")
        except FileNotFoundError:
            pass
        engine = RSIEngine(self.registry)
        for _ in range(3):
            engine._record_pattern({
                "description": "user switches model frequently",
                "category": "preference",
            })
        proposals = engine._generate_proposals()
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].type, "config")
        self.assertGreaterEqual(proposals[0].confidence, 0.5)

    @patch("lib.hal.rsi._PATTERNS_PATH", "/tmp/hal-test-rsi-patterns5.json")
    def test_below_threshold_no_proposal(self):
        try:
            os.remove("/tmp/hal-test-rsi-patterns5.json")
        except FileNotFoundError:
            pass
        engine = RSIEngine(self.registry)
        engine._record_pattern({
            "description": "one-off pattern",
            "category": "workflow",
        })
        proposals = engine._generate_proposals()
        self.assertEqual(proposals, [])


if __name__ == "__main__":
    unittest.main()
