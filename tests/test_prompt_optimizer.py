"""Tests for HAL prompt optimizer."""
import os, sys, unittest
from lib.hal.prompt_optimizer import PromptOptimizer

class TestPromptOptimizer(unittest.TestCase):
    def test_removes_verbose(self):
        r = PromptOptimizer().optimize("Please can you write a function")
        self.assertNotIn("please can you", r.text.lower())

    def test_compresses_whitespace(self):
        r = PromptOptimizer().optimize("too   many   spaces")
        self.assertNotIn("   ", r.text)

    def test_savings(self):
        r = PromptOptimizer().optimize("I would like you to please can you help me write code")
        self.assertGreater(r.savings_pct, 0)

    def test_no_change_for_clean(self):
        r = PromptOptimizer().optimize("Write a sort function")
        self.assertEqual(r.text, "Write a sort function")

    def test_empty(self):
        r = PromptOptimizer().optimize("")
        self.assertEqual(r.savings_pct, 0)

if __name__ == "__main__":
    unittest.main()
