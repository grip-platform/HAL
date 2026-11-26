"""Tests for Phase γ W4 speculative decoding primitive (H-SPEC)."""
from __future__ import annotations

import unittest

from lib.hal.speculative import (
    DEFAULT_THRESHOLD,
    accept_ratio,
    decide_commit,
)


class TestAcceptRatio(unittest.TestCase):
    def test_empty_draft_is_zero(self):
        self.assertEqual(accept_ratio([], ["a", "b"]), 0.0)

    def test_empty_verify_is_zero(self):
        self.assertEqual(accept_ratio(["a"], []), 0.0)

    def test_perfect_match(self):
        self.assertEqual(accept_ratio(["a", "b", "c"], ["a", "b", "c"]), 1.0)

    def test_prefix_match_then_divergence(self):
        # 2 of 4 draft tokens accepted.
        self.assertEqual(accept_ratio(["a", "b", "c", "d"], ["a", "b", "x", "y"]), 0.5)

    def test_first_token_divergence(self):
        self.assertEqual(accept_ratio(["a", "b"], ["x", "b"]), 0.0)


class TestDecideCommit(unittest.TestCase):
    def test_empty_draft_never_commits(self):
        r = decide_commit([], ["a"])
        self.assertFalse(r.committed)
        self.assertEqual(r.ratio, 0.0)

    def test_perfect_match_commits(self):
        r = decide_commit(["a", "b", "c"], ["a", "b", "c"])
        self.assertTrue(r.committed)
        self.assertEqual(r.accept_count, 3)

    def test_above_threshold_commits(self):
        # 7/10 = 0.7 meets the default.
        draft = ["t"] * 10
        verify = ["t"] * 7 + ["x"] * 3
        r = decide_commit(draft, verify)
        self.assertTrue(r.committed)
        self.assertAlmostEqual(r.ratio, 0.7)

    def test_below_threshold_fallbacks(self):
        draft = ["t"] * 10
        verify = ["t"] * 5 + ["x"] * 5
        r = decide_commit(draft, verify)
        self.assertFalse(r.committed)
        self.assertAlmostEqual(r.ratio, 0.5)

    def test_custom_threshold_honoured(self):
        r = decide_commit(["a", "b"], ["a", "x"], threshold=0.3)
        self.assertTrue(r.committed)


class TestDefaults(unittest.TestCase):
    def test_default_threshold(self):
        self.assertEqual(DEFAULT_THRESHOLD, 0.7)


if __name__ == "__main__":
    unittest.main()
