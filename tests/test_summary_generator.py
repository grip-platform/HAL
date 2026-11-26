"""Tests for HAL summary generator."""
import os
import sys
import unittest

from lib.hal.summary_generator import SummaryGenerator
from lib.hal.types import ChatMessage


def _msg(role, content):
    return ChatMessage(role=role, content=content)


class TestSummaryGenerator(unittest.TestCase):

    def test_empty(self):
        self.assertIn("No conversation", SummaryGenerator().generate([]).text)

    def test_technical(self):
        s = SummaryGenerator().generate([_msg("user", "Fix server.py"), _msg("assistant", "Done. TODO: tests.")], "technical")
        self.assertIn("server.py", s.files_referenced)

    def test_executive(self):
        self.assertEqual(SummaryGenerator().generate([_msg("user", "Hi")], "executive").style, "executive")

    def test_actions(self):
        s = SummaryGenerator().generate([_msg("assistant", "TODO: write tests.")], "actions")
        self.assertGreater(len(s.action_items), 0)

    def test_decisions(self):
        s = SummaryGenerator().generate([_msg("assistant", "We decided to use Python.")])
        self.assertGreater(len(s.key_decisions), 0)

    def test_files(self):
        self.assertIn("auth.py", SummaryGenerator().generate([_msg("user", "Check auth.py")]).files_referenced)


if __name__ == "__main__":
    unittest.main()
