"""Tests for HAL input attribution."""
import os
import sys
import unittest

from lib.hal.attribution import (
    AttributionReport,
    AttributionScore,
    InputAttributor,
    _equal_attribution,
    _parse_scores,
)


class TestEqualAttribution(unittest.TestCase):

    def test_single_section(self):
        scores = _equal_attribution({"code": "..."})
        self.assertEqual(len(scores), 1)
        self.assertAlmostEqual(scores[0].importance, 1.0)

    def test_two_sections(self):
        scores = _equal_attribution({"code": "...", "docs": "..."})
        self.assertEqual(len(scores), 2)
        self.assertAlmostEqual(scores[0].importance, 0.5)

    def test_empty_sections(self):
        scores = _equal_attribution({})
        self.assertEqual(len(scores), 0)


class TestParseScores(unittest.TestCase):

    def test_valid_json(self):
        text = '{"code": 0.7, "docs": 0.3}'
        sections = {"code": "...", "docs": "..."}
        scores = _parse_scores(text, sections)
        self.assertEqual(len(scores), 2)
        code_score = next(s for s in scores if s.section == "code")
        self.assertAlmostEqual(code_score.importance, 0.7)

    def test_json_in_text(self):
        text = 'Here are the scores: {"code": 0.6, "docs": 0.4} done'
        sections = {"code": "...", "docs": "..."}
        scores = _parse_scores(text, sections)
        self.assertEqual(len(scores), 2)

    def test_invalid_json_fallback(self):
        sections = {"code": "...", "docs": "..."}
        scores = _parse_scores("not json", sections)
        self.assertEqual(len(scores), 2)
        self.assertAlmostEqual(scores[0].importance, 0.5)

    def test_normalises_to_sum_one(self):
        text = '{"a": 2, "b": 8}'
        sections = {"a": "...", "b": "..."}
        scores = _parse_scores(text, sections)
        total = sum(s.importance for s in scores)
        self.assertAlmostEqual(total, 1.0)


class TestAttributionReport(unittest.TestCase):

    def test_dominant_section(self):
        scores = (
            AttributionScore(section="code", importance=0.7),
            AttributionScore(section="docs", importance=0.3),
        )
        report = AttributionReport(scores=scores, method="rating", total_sections=2)
        self.assertEqual(report.dominant_section, "code")

    def test_empty_report(self):
        report = AttributionReport(scores=(), method="rating", total_sections=0)
        self.assertEqual(report.dominant_section, "")

    def test_summary(self):
        scores = (AttributionScore(section="code", importance=0.8),)
        report = AttributionReport(scores=scores, method="rating", total_sections=1)
        self.assertIn("code", report.summary)
        self.assertIn("80%", report.summary)


class TestInputAttributorNoRegistry(unittest.TestCase):

    def test_fallback_without_registry(self):
        attributor = InputAttributor()
        report = attributor.attribute_sections(
            sections={"code": "x = 1", "docs": "Variable assignment"},
            response="Sets x to 1",
        )
        self.assertEqual(report.total_sections, 2)
        self.assertEqual(len(report.scores), 2)
        # Without registry, should get equal attribution
        self.assertAlmostEqual(report.scores[0].importance, 0.5)

    def test_empty_sections(self):
        attributor = InputAttributor()
        report = attributor.attribute_sections(sections={}, response="hello")
        self.assertEqual(report.total_sections, 0)


if __name__ == "__main__":
    unittest.main()
