"""Tests for HAL CLI argument parsing and command routing."""
import os
import sys
import unittest

from lib.hal.cli import _parse_args


class TestParseArgs(unittest.TestCase):
    """Test CLI argument parsing."""

    def _parse(self, argv: list) -> dict:
        """Helper: parse args with mocked sys.argv."""
        original = sys.argv
        sys.argv = ["hal"] + argv
        try:
            return _parse_args()
        finally:
            sys.argv = original

    def test_defaults(self):
        args = self._parse([])
        self.assertEqual(args["model"], "")
        self.assertEqual(args["mode"], "")
        self.assertFalse(args["serve"])
        self.assertFalse(args["status"])
        self.assertEqual(args["port"], 3850)

    def test_model_flag(self):
        args = self._parse(["--model", "groq/llama-3.3-70b"])
        self.assertEqual(args["model"], "groq/llama-3.3-70b")

    def test_mode_flag(self):
        args = self._parse(["--mode", "legal"])
        self.assertEqual(args["mode"], "legal")

    def test_serve_flag(self):
        args = self._parse(["--serve"])
        self.assertTrue(args["serve"])

    def test_serve_with_port(self):
        args = self._parse(["--serve", "--port", "8080"])
        self.assertTrue(args["serve"])
        self.assertEqual(args["port"], 8080)

    def test_status_flag(self):
        args = self._parse(["--status"])
        self.assertTrue(args["status"])

    def test_combined_flags(self):
        args = self._parse(["--model", "ollama/mistral:7b", "--mode", "legal"])
        self.assertEqual(args["model"], "ollama/mistral:7b")
        self.assertEqual(args["mode"], "legal")

    def test_unknown_args_ignored(self):
        args = self._parse(["--unknown", "value"])
        self.assertEqual(args["model"], "")


if __name__ == "__main__":
    unittest.main()
