"""Tests for context_collapse — intelligent context compression."""
import unittest

from lib.hal.context_collapse import (
    CollapseResult,
    collapse_context,
    _estimate_tokens,
    _strip_git_sections,
    _trim_history,
)


class TestEstimateTokens(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(_estimate_tokens(""), 0)

    def test_short(self):
        self.assertEqual(_estimate_tokens("hi"), 1)

    def test_typical(self):
        result = _estimate_tokens("hello world test")
        self.assertGreater(result, 2)


class TestCollapseContext(unittest.TestCase):
    def test_no_collapse_under_budget(self):
        result = collapse_context("short system", [{"role": "user", "content": "hi"}], budget_tokens=1000)
        self.assertEqual(result.sections_collapsed, [])
        self.assertEqual(result.system_prompt, "short system")

    def test_git_stripped_first(self):
        system = "Instructions here.\nGit status:\n  M file.py\n  M other.py\nRecent commits:\n  abc123\n  def456\nMore instructions."
        result = collapse_context(system, [], budget_tokens=20)
        self.assertNotIn("file.py", result.system_prompt)
        self.assertTrue(any("git" in s for s in result.sections_collapsed))

    def test_history_trimmed(self):
        messages = [{"role": "user", "content": "msg" * 100}] * 20
        result = collapse_context("sys", messages, budget_tokens=500)
        self.assertLess(len(result.messages), 20)
        self.assertTrue(any("history" in s for s in result.sections_collapsed))

    def test_returns_collapse_result(self):
        result = collapse_context("test", [], budget_tokens=1000)
        self.assertIsInstance(result, CollapseResult)
        self.assertGreater(result.token_estimate, 0)
        self.assertGreater(result.original_tokens, 0)

    def test_aggressive_collapse_on_huge_context(self):
        huge_system = "x" * 100000
        messages = [{"role": "user", "content": "y" * 10000}] * 10
        result = collapse_context(huge_system, messages, budget_tokens=1000)
        self.assertLess(result.token_estimate, result.original_tokens, "Collapse must reduce token count")
        self.assertGreater(len(result.sections_collapsed), 0, "At least one section must be collapsed")


class TestStripGitSections(unittest.TestCase):
    def test_strips_git_status(self):
        system = "Before.\nGit status:\n  M a.py\n  M b.py\nAfter."
        collapsed: list[str] = []
        result = _strip_git_sections(system, collapsed)
        self.assertNotIn("M a.py", result)
        self.assertIn("Before.", result)
        self.assertIn("After.", result)

    def test_no_git_sections_unchanged(self):
        system = "Just normal text here."
        collapsed: list[str] = []
        result = _strip_git_sections(system, collapsed)
        self.assertEqual(result, system)


class TestTrimHistory(unittest.TestCase):
    def test_keeps_system_messages(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a3"},
        ]
        collapsed: list[str] = []
        result = _trim_history(messages, keep_turns=1, collapsed=collapsed)
        roles = [m["role"] for m in result]
        self.assertIn("system", roles)
        self.assertEqual(result[-1]["content"], "a3")

    def test_no_trim_under_limit(self):
        messages = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
        collapsed: list[str] = []
        result = _trim_history(messages, keep_turns=4, collapsed=collapsed)
        self.assertEqual(len(result), 2)
        self.assertEqual(collapsed, [])


if __name__ == "__main__":
    unittest.main()
