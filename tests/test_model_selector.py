"""Tests for HAL model selector."""
import os, sys, unittest
from lib.hal.model_selector import ModelSelector, Selection

class TestModelSelector(unittest.TestCase):
    def test_no_registry(self):
        s = ModelSelector().select("Fix the bug")
        self.assertEqual(s.tier, "none")
        self.assertIn("debug", s.intent)

    def test_selection_fields(self):
        s = Selection("m", "reason", "code", "mid")
        self.assertEqual(s.model, "m")

    def test_detects_intent(self):
        s = ModelSelector().select("Explain how this works")
        self.assertEqual(s.intent, "explain")

if __name__ == "__main__":
    unittest.main()
