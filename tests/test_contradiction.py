"""Tests for Phase β W3 — contradiction detector (H-CONTRA foundation)."""
from __future__ import annotations

import unittest

from lib.hal.memory_engine import MemoryEntry
from lib.hal.contradiction import (
    ContradictionResult,
    JudgeVerdict,
    SimpleHeuristicJudge,
    detect_contradiction,
    scan,
)


def _entry(fn, content="", name="", desc=""):
    return MemoryEntry(filename=fn, name=name, description=desc, content=content)


def _yes_judge(a, b):
    return JudgeVerdict(True, 0.9, "mocked yes")


def _no_judge(a, b):
    return JudgeVerdict(False, 0.0, "mocked no")


def _selective_judge(pairs_to_flag):
    """Flag only the pairs (fn_a, fn_b) supplied."""
    flagged = set((x, y) for x, y in pairs_to_flag)

    def judge(a, b):
        key = (a.filename, b.filename)
        if key in flagged or tuple(reversed(key)) in flagged:
            return JudgeVerdict(True, 0.8)
        return JudgeVerdict(False, 0.0)

    return judge


class TestContradictionResultFlag(unittest.TestCase):
    def test_empty_contradicting_is_not_flagged(self):
        r = ContradictionResult(entry=_entry("a.md"))
        self.assertFalse(r.is_contradicted)

    def test_populated_contradicting_sets_flag(self):
        r = ContradictionResult(entry=_entry("a.md"), contradicting=[_entry("b.md")])
        self.assertTrue(r.is_contradicted)


class TestDetectContradiction(unittest.TestCase):
    def test_no_neighbours_returns_clean(self):
        r = detect_contradiction(_entry("a.md"), [], judge_fn=_yes_judge)
        self.assertFalse(r.is_contradicted)
        self.assertEqual(r.max_confidence, 0.0)

    def test_self_neighbour_skipped(self):
        self_ref = _entry("a.md")
        r = detect_contradiction(self_ref, [self_ref], judge_fn=_yes_judge)
        self.assertFalse(r.is_contradicted)

    def test_all_contradicting_collected(self):
        entry = _entry("a.md")
        neighbours = [_entry("b.md"), _entry("c.md")]
        r = detect_contradiction(entry, neighbours, judge_fn=_yes_judge)
        self.assertTrue(r.is_contradicted)
        self.assertEqual(len(r.contradicting), 2)
        self.assertAlmostEqual(r.max_confidence, 0.9)

    def test_none_contradicting_stays_clean(self):
        r = detect_contradiction(
            _entry("a.md"), [_entry("b.md"), _entry("c.md")], judge_fn=_no_judge,
        )
        self.assertFalse(r.is_contradicted)

    def test_selective_contradicting_partially_flagged(self):
        entry = _entry("a.md")
        neighbours = [_entry("b.md"), _entry("c.md"), _entry("d.md")]
        judge = _selective_judge([("a.md", "c.md")])
        r = detect_contradiction(entry, neighbours, judge_fn=judge)
        self.assertTrue(r.is_contradicted)
        self.assertEqual([n.filename for n in r.contradicting], ["c.md"])


class TestScan(unittest.TestCase):
    def test_all_entries_represented_in_map(self):
        entries = [_entry("a.md"), _entry("b.md"), _entry("c.md")]
        results = scan(
            entries,
            neighbours_fn=lambda e, es: es,
            judge_fn=_no_judge,
        )
        self.assertEqual(set(results.keys()), {"a.md", "b.md", "c.md"})
        for r in results.values():
            self.assertFalse(r.is_contradicted)

    def test_neighbours_fn_filters_scope(self):
        entries = [_entry("a.md"), _entry("b.md"), _entry("c.md")]
        # Judge would flag everything, but neighbours_fn returns empty.
        results = scan(
            entries,
            neighbours_fn=lambda e, es: [],
            judge_fn=_yes_judge,
        )
        for r in results.values():
            self.assertFalse(r.is_contradicted)

    def test_known_contradiction_fixture(self):
        """H-CONTRA regression anchor: a known pair gets tagged."""
        entries = [
            _entry("alpha.md", content="H091 is confirmed for semantic cache"),
            _entry("beta.md", content="H091 is refuted for semantic cache"),
            _entry("gamma.md", content="unrelated topic"),
        ]
        results = scan(
            entries,
            neighbours_fn=lambda e, es: [x for x in es if x.filename != e.filename],
            judge_fn=SimpleHeuristicJudge(min_overlap=0.2),
        )
        self.assertTrue(results["alpha.md"].is_contradicted)
        self.assertTrue(results["beta.md"].is_contradicted)
        self.assertFalse(results["gamma.md"].is_contradicted)


class TestSimpleHeuristicJudge(unittest.TestCase):
    def test_low_overlap_returns_no_contradiction(self):
        a = _entry("a.md", content="kangaroo jumps forest")
        b = _entry("b.md", content="semiconductor fabrication photolithography")
        self.assertFalse(SimpleHeuristicJudge()(a, b).contradicts)

    def test_polarity_flip_flags(self):
        a = _entry("a.md", content="H091 cache hit rate is confirmed for all prompts")
        b = _entry("b.md", content="H091 cache hit rate is refuted for all prompts")
        self.assertTrue(SimpleHeuristicJudge()(a, b).contradicts)

    def test_negation_flip_flags(self):
        a = _entry("a.md", content="The streaming renderer is enabled by default")
        b = _entry("b.md", content="The streaming renderer is disabled by default")
        self.assertTrue(SimpleHeuristicJudge()(a, b).contradicts)

    def test_duplicate_content_does_not_flag(self):
        a = _entry("a.md", content="same content same content")
        b = _entry("b.md", content="same content same content")
        self.assertFalse(SimpleHeuristicJudge()(a, b).contradicts)

    def test_tunable_min_overlap(self):
        judge_strict = SimpleHeuristicJudge(min_overlap=0.99)
        a = _entry("a.md", content="widget is enabled")
        b = _entry("b.md", content="widget is disabled")
        # Strict overlap requirement rejects even obvious cases.
        self.assertFalse(judge_strict(a, b).contradicts)

    def test_confidence_reflects_configuration(self):
        judge = SimpleHeuristicJudge(confidence=0.77)
        a = _entry("a.md", content="test is enabled by default")
        b = _entry("b.md", content="test is disabled by default")
        v = judge(a, b)
        self.assertAlmostEqual(v.confidence, 0.77)

    def test_rationale_populated_on_positive(self):
        a = _entry("a.md", content="X is enabled broadly")
        b = _entry("b.md", content="X is disabled broadly")
        v = SimpleHeuristicJudge()(a, b)
        self.assertIn("flip", v.rationale)

    def test_empty_content_is_clean(self):
        v = SimpleHeuristicJudge()(_entry("a.md"), _entry("b.md"))
        self.assertFalse(v.contradicts)


if __name__ == "__main__":
    unittest.main()
