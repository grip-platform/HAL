"""Tests for HAL employee gate feature tracking."""
import os
import sys
import unittest

from lib.hal.employee_gate import (
    get_feature_status,
    get_implementation_score,
    list_remaining,
)


class TestGetFeatureStatus(unittest.TestCase):

    def test_returns_list(self):
        status = get_feature_status()
        self.assertIsInstance(status, list)
        self.assertGreater(len(status), 10)

    def test_has_required_fields(self):
        for feature in get_feature_status():
            self.assertIn("name", feature)
            self.assertIn("cc_gate", feature)
            self.assertIn("hal_module", feature)
            self.assertIn("status", feature)

    def test_verify_is_implemented(self):
        status = get_feature_status()
        verify = next((f for f in status if "verification" in f["name"].lower()), None)
        self.assertIsNotNone(verify)
        self.assertEqual(verify["status"], "implemented")

    def test_telemetry_is_implemented(self):
        status = get_feature_status()
        telemetry = next((f for f in status if "telemetry" in f["name"].lower()), None)
        self.assertIsNotNone(telemetry)
        self.assertEqual(telemetry["status"], "implemented")


class TestGetImplementationScore(unittest.TestCase):

    def test_score_structure(self):
        score = get_implementation_score()
        self.assertIn("total_features", score)
        self.assertIn("implemented", score)
        self.assertIn("score", score)
        self.assertIn("score_pct", score)

    def test_score_above_80_pct(self):
        score = get_implementation_score()
        self.assertGreaterEqual(score["score"], 0.80)

    def test_total_features(self):
        score = get_implementation_score()
        self.assertGreater(score["total_features"], 10)


class TestListRemaining(unittest.TestCase):

    def test_returns_list(self):
        remaining = list_remaining()
        self.assertIsInstance(remaining, list)

    def test_no_implemented_in_remaining(self):
        remaining = list_remaining()
        for f in remaining:
            self.assertNotEqual(f.status, "implemented")


if __name__ == "__main__":
    unittest.main()
