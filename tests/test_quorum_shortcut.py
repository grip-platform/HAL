"""Tests for Phase γ W2 quorum shortcut (H-QUORUM)."""
from __future__ import annotations

import unittest

from lib.hal.quorum_shortcut import (
    DEFAULT_MIN_VOTES,
    DEFAULT_THRESHOLD,
    QuorumDecision,
    Vote,
    agreement_score,
    should_shortcut,
    shortcut_metadata,
)


def _votes(*signals: str) -> list[Vote]:
    return [Vote(member=f"m{i}", signal=s) for i, s in enumerate(signals)]


class TestAgreementScore(unittest.TestCase):
    def test_empty_is_zero(self):
        self.assertEqual(agreement_score([]), 0.0)

    def test_all_agree_is_one(self):
        self.assertEqual(agreement_score(_votes("T", "T", "T")), 1.0)

    def test_majority_is_fraction(self):
        self.assertAlmostEqual(agreement_score(_votes("T", "T", "F")), 2 / 3)

    def test_tie_returns_tied_count(self):
        # 2/4 = 0.5 either way.
        self.assertAlmostEqual(agreement_score(_votes("T", "T", "F", "F")), 0.5)

    def test_all_distinct_uses_top_one(self):
        self.assertAlmostEqual(
            agreement_score(_votes("A", "B", "C", "D")), 0.25,
        )


class TestShouldShortcut(unittest.TestCase):
    def test_insufficient_votes_never_shortcuts(self):
        d = should_shortcut(_votes("T", "T"), min_votes=3)
        self.assertFalse(d.shortcut)
        self.assertIn("fewer than 3 votes", d.reason)

    def test_unanimous_triple_shortcuts(self):
        d = should_shortcut(_votes("T", "T", "T"))
        self.assertTrue(d.shortcut)
        self.assertAlmostEqual(d.agreement_score, 1.0)

    def test_just_above_threshold_shortcuts(self):
        # 4/4 = 1.0 > 0.85
        d = should_shortcut(_votes("T", "T", "T", "T"), threshold=0.85)
        self.assertTrue(d.shortcut)

    def test_just_below_threshold_does_not_shortcut(self):
        # 3/4 = 0.75 < 0.85
        d = should_shortcut(_votes("T", "T", "T", "F"), threshold=0.85)
        self.assertFalse(d.shortcut)
        self.assertIn("below threshold", d.reason)

    def test_custom_threshold_honoured(self):
        # 3/4 = 0.75 should shortcut at threshold 0.70
        d = should_shortcut(_votes("T", "T", "T", "F"), threshold=0.70)
        self.assertTrue(d.shortcut)

    def test_exact_threshold_equality_shortcuts(self):
        # 3/3 = 1.0, threshold 1.0 → equality path, shortcut.
        d = should_shortcut(_votes("T", "T", "T"), threshold=1.0)
        self.assertTrue(d.shortcut)


class TestShortcutMetadata(unittest.TestCase):
    def test_shortcut_payload_shape(self):
        d = QuorumDecision(shortcut=True, agreement_score=0.92, reason="ok")
        meta = shortcut_metadata(d)
        self.assertEqual(meta["SHORTCUT"], "quorum")
        self.assertEqual(meta["agreement_score"], "0.920")
        self.assertEqual(meta["reason"], "ok")

    def test_no_shortcut_payload(self):
        d = QuorumDecision(shortcut=False, agreement_score=0.5, reason="low")
        meta = shortcut_metadata(d)
        self.assertEqual(meta["SHORTCUT"], "none")


class TestDefaults(unittest.TestCase):
    def test_default_threshold(self):
        self.assertEqual(DEFAULT_THRESHOLD, 0.85)

    def test_default_min_votes(self):
        self.assertEqual(DEFAULT_MIN_VOTES, 3)


if __name__ == "__main__":
    unittest.main()
