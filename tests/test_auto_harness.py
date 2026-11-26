"""Tests for HAL auto-harness module."""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from lib.hal.auto_harness import (
    AutoHarness,
    FeatureScore,
    HarnessReport,
    _parse_features,
)


class TestFeatureScore(unittest.TestCase):

    def test_score_calculation(self):
        f = FeatureScore(
            name="test", description="desc",
            applicability=8, complexity=3, novelty=7, burden=2,
            recommendation="absorb", hal_module="registry.py", rationale="good",
        )
        # 8 + 7 - 3 - 2 + 20 = 30
        self.assertEqual(f.score, 30)

    def test_low_score(self):
        f = FeatureScore(
            name="bad", description="",
            applicability=2, complexity=9, novelty=1, burden=8,
            recommendation="skip", hal_module="", rationale="",
        )
        # 2 + 1 - 9 - 8 + 20 = 6
        self.assertEqual(f.score, 6)

    def test_to_dict(self):
        f = FeatureScore(
            name="test", description="desc",
            applicability=5, complexity=5, novelty=5, burden=5,
            recommendation="monitor", hal_module="moe.py", rationale="ok",
        )
        d = f.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertEqual(d["score"], 20)
        self.assertIn("hal_module", d)


class TestParseFeatures(unittest.TestCase):

    def test_valid_json(self):
        text = '[{"name": "feature1", "description": "desc", "applicability": 8, "complexity": 3, "novelty": 7, "burden": 2, "recommendation": "absorb", "hal_module": "registry.py", "rationale": "good"}]'
        features = _parse_features(text)
        self.assertEqual(len(features), 1)
        self.assertEqual(features[0].name, "feature1")

    def test_json_in_text(self):
        text = 'Here are features:\n[{"name": "f1", "applicability": 5}]\nDone.'
        features = _parse_features(text)
        self.assertEqual(len(features), 1)

    def test_invalid_json(self):
        features = _parse_features("not json")
        self.assertEqual(features, [])

    def test_empty_array(self):
        features = _parse_features("[]")
        self.assertEqual(features, [])

    def test_filters_bad_entries(self):
        text = '[{"name": "good"}, {"no_name": true}]'
        features = _parse_features(text)
        self.assertEqual(len(features), 1)


class TestHarnessReport(unittest.TestCase):

    def test_to_dict(self):
        report = HarnessReport(
            repo="owner/repo", scan_time=5.0,
            description="test", language="Python", stars=100,
            features=[], summary="test summary",
        )
        d = report.to_dict()
        self.assertEqual(d["repo"], "owner/repo")
        self.assertEqual(d["stars"], 100)


class TestAutoHarness(unittest.TestCase):

    def test_create_without_registry(self):
        harness = AutoHarness()
        self.assertIsNone(harness._registry)

    @patch("lib.hal.auto_harness.subprocess.run")
    def test_fetch_metadata(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"description": "test repo", "stargazerCount": 42}',
        )
        harness = AutoHarness()
        meta = harness._fetch_metadata("owner/repo")
        self.assertEqual(meta["description"], "test repo")

    @patch("lib.hal.auto_harness.subprocess.run")
    def test_fetch_metadata_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        harness = AutoHarness()
        meta = harness._fetch_metadata("owner/repo")
        self.assertEqual(meta, {})

    @patch("lib.hal.auto_harness.subprocess.run")
    def test_fetch_structure(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="README.md\nsrc\ntests\n",
        )
        harness = AutoHarness()
        structure = harness._fetch_structure("owner/repo")
        self.assertEqual(len(structure), 3)
        self.assertIn("README.md", structure)


if __name__ == "__main__":
    unittest.main()
