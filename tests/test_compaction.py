"""Tests for HAL smart compaction."""
import os
import sys
import unittest

from lib.hal.compaction import (
    CompactionResult,
    SmartCompactor,
    _estimate_tokens,
    _extract_code_blocks,
    _extract_decisions,
    _extract_file_refs,
    _mechanical_compact,
)
from lib.hal.types import ChatMessage


def _msg(role: str, content: str) -> ChatMessage:
    return ChatMessage(role=role, content=content)


class TestExtractCodeBlocks(unittest.TestCase):

    def test_extracts_fenced_blocks(self):
        history = [_msg("assistant", "Here:\n```python\nx = 1\n```\nDone")]
        blocks = _extract_code_blocks(history)
        self.assertEqual(len(blocks), 1)
        self.assertIn("x = 1", blocks[0])

    def test_no_blocks(self):
        history = [_msg("user", "No code here")]
        self.assertEqual(_extract_code_blocks(history), [])

    def test_multiple_blocks(self):
        history = [_msg("assistant", "```a```\ntext\n```b```")]
        blocks = _extract_code_blocks(history)
        self.assertEqual(len(blocks), 2)


class TestExtractDecisions(unittest.TestCase):

    def test_finds_decisions(self):
        history = [_msg("assistant", "We decided to use Python for the backend. The tests should pass.")]
        decisions = _extract_decisions(history)
        self.assertGreater(len(decisions), 0)
        self.assertTrue(any("decided" in d.lower() for d in decisions))

    def test_ignores_short_fragments(self):
        history = [_msg("user", "We will. Ok.")]
        decisions = _extract_decisions(history)
        self.assertEqual(len(decisions), 0)

    def test_no_decisions(self):
        history = [_msg("user", "Hello there")]
        decisions = _extract_decisions(history)
        self.assertEqual(len(decisions), 0)


class TestExtractFileRefs(unittest.TestCase):

    def test_finds_python_files(self):
        history = [_msg("assistant", "Check lib/hal/server.py and tests/test_server.py")]
        refs = _extract_file_refs(history)
        self.assertIn("lib/hal/server.py", refs)
        self.assertIn("tests/test_server.py", refs)

    def test_no_refs(self):
        history = [_msg("user", "No files mentioned")]
        self.assertEqual(_extract_file_refs(history), set())


class TestMechanicalCompact(unittest.TestCase):

    def test_includes_topic(self):
        history = [
            _msg("user", "Fix the authentication bug"),
            _msg("assistant", "I found the issue in auth.py"),
        ]
        result = _mechanical_compact(history, [], [])
        self.assertIn("authentication", result.lower())

    def test_includes_file_refs(self):
        history = [_msg("assistant", "Modified server.py and client.js")]
        result = _mechanical_compact(history, [], [])
        self.assertIn("server.py", result)

    def test_includes_last_response(self):
        history = [
            _msg("user", "Question"),
            _msg("assistant", "The final answer is 42"),
        ]
        result = _mechanical_compact(history, [], [])
        self.assertIn("42", result)


class TestSmartCompactor(unittest.TestCase):

    def test_empty_history(self):
        compactor = SmartCompactor()
        result = compactor.compact([])
        self.assertEqual(result.turns_compacted, 0)
        self.assertEqual(result.summary, "")

    def test_mechanical_compact(self):
        history = [
            _msg("user", "How does the auth module work?"),
            _msg("assistant", "The auth module in auth.py handles JWT tokens."),
        ]
        compactor = SmartCompactor()
        result = compactor.compact(history)
        self.assertEqual(result.turns_compacted, 2)
        self.assertGreater(len(result.summary), 0)

    def test_compression_ratio(self):
        long_history = [_msg("user", "x " * 500), _msg("assistant", "y " * 500)]
        result = SmartCompactor().compact(long_history)
        self.assertGreater(result.compression_ratio, 0.0)


class TestCompactionResult(unittest.TestCase):

    def test_summary_message(self):
        result = CompactionResult("Summary text", 5, 2, 1, 1000, 200)
        msg = result.summary_message
        self.assertEqual(msg.role, "system")
        self.assertIn("5 turns", msg.content)
        self.assertIn("Summary text", msg.content)

    def test_compression_ratio_zero(self):
        result = CompactionResult("", 0, 0, 0, 0, 0)
        self.assertEqual(result.compression_ratio, 0.0)


class TestEstimateTokens(unittest.TestCase):

    def test_nonempty(self):
        history = [_msg("user", "Hello world")]
        self.assertGreater(_estimate_tokens(history), 0)

    def test_empty(self):
        self.assertEqual(_estimate_tokens([]), 0)


if __name__ == "__main__":
    unittest.main()
