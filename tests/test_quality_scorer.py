"""Tests for HAL response quality scorer."""
import os
import sys
import unittest

from lib.hal.quality_scorer import QualityScore, QualityScorer


class TestQualityScore(unittest.TestCase):

    def test_overall(self):
        score = QualityScore(20, 20, 20, 20)
        self.assertEqual(score.overall, 80)

    def test_grade_a(self):
        self.assertEqual(QualityScore(25, 25, 22, 20).grade, "A")

    def test_grade_b(self):
        self.assertEqual(QualityScore(20, 20, 20, 18).grade, "B")

    def test_grade_f(self):
        self.assertEqual(QualityScore(5, 5, 5, 5).grade, "F")

    def test_to_dict(self):
        d = QualityScore(20, 15, 10, 25).to_dict()
        self.assertEqual(d["overall"], 70)
        self.assertIn("grade", d)

    def test_frozen(self):
        s = QualityScore(1, 2, 3, 4)
        with self.assertRaises(AttributeError):
            s.relevance = 10


class TestQualityScorer(unittest.TestCase):

    def test_good_response(self):
        scorer = QualityScorer()
        score = scorer.score(
            "How do I sort a list in Python?",
            "Use the `sorted()` function:\n\n```python\nresult = sorted([3, 1, 2])\n```\n\nThis returns a new sorted list. "
            "For in-place sorting, use `list.sort()`. Both support a `key` parameter for custom ordering.",
        )
        self.assertGreater(score.overall, 60)

    def test_empty_response(self):
        scorer = QualityScorer()
        score = scorer.score("Hello?", "")
        # Relevance, completeness, conciseness = 0; structure base = 10
        self.assertLessEqual(score.overall, 15)

    def test_irrelevant_response(self):
        scorer = QualityScorer()
        score = scorer.score(
            "How do I configure nginx?",
            "The weather today is sunny with a high of 25 degrees.",
        )
        self.assertLess(score.relevance, 20)

    def test_well_structured_gets_higher(self):
        scorer = QualityScorer()
        plain = scorer.score("Question?", "Just some plain text answer without formatting.")
        structured = scorer.score("Question?", "## Answer\n\n```python\ncode\n```\n\n- Point 1\n- Point 2\n\n**Key takeaway**: formatted.")
        self.assertGreater(structured.structure, plain.structure)

    def test_concise_scores_well(self):
        scorer = QualityScorer()
        # Query 5 words, response ~30 words (ratio ~6) = ideal
        score = scorer.score(
            "How does auth work?",
            "The auth module validates JWT tokens on each request. "
            "It checks the signature, expiry, and permissions before "
            "allowing access to protected endpoints.",
        )
        self.assertGreaterEqual(score.conciseness, 20)

    def test_verbose_scores_lower(self):
        scorer = QualityScorer()
        # Very verbose response (ratio > 30)
        score = scorer.score("Short query?", "word " * 200)
        self.assertLess(score.conciseness, 20)


if __name__ == "__main__":
    unittest.main()
