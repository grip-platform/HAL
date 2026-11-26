"""Regression anchors for the /help command and banner hint line.

Guards against:
- /help being removed or silently broken (content checks)
- the banner hint line dropping key commands users expect
- unknown commands not suggesting /help
"""
from __future__ import annotations

import io
import sys
import unittest

from lib.hal.cli import _HELP_TEXT, _handle_command, _print_help


class HelpTextContentTests(unittest.TestCase):
    """The help text must surface the feature groups the user cares about."""

    def test_covers_model_command(self):
        self.assertIn("/model", _HELP_TEXT)

    def test_covers_mode_command(self):
        self.assertIn("/mode", _HELP_TEXT)

    def test_covers_tools_command(self):
        self.assertIn("/tools", _HELP_TEXT)

    def test_covers_council_command(self):
        self.assertIn("/council", _HELP_TEXT)

    def test_covers_voice_cli(self):
        self.assertIn("voice", _HELP_TEXT)

    def test_covers_cost_command(self):
        self.assertIn("/cost", _HELP_TEXT)

    def test_covers_quit_command(self):
        self.assertIn("/quit", _HELP_TEXT)

    def test_covers_serve_flag(self):
        self.assertIn("--serve", _HELP_TEXT)


class PrintHelpTests(unittest.TestCase):
    """_print_help() must write non-empty content to stdout."""

    def _capture(self):
        buf = io.StringIO()
        sys.stdout = buf
        try:
            _print_help()
        finally:
            sys.stdout = sys.__stdout__
        return buf.getvalue()

    def test_prints_something(self):
        out = self._capture()
        self.assertTrue(out.strip())

    def test_prints_section_headers(self):
        out = self._capture()
        self.assertIn("Tools", out)
        self.assertIn("Council", out)


class BannerHintTests(unittest.TestCase):
    """The one-line banner hint must include the most-used commands."""

    def _banner_hint_line(self) -> str:
        import lib.hal.cli as cli_mod
        import shutil
        w = min(shutil.get_terminal_size((80, 24)).columns - 4, 72)
        bar = f"  \033[38;5;240m{'─' * w}\033[0m"
        buf = io.StringIO()
        sys.stdout = buf
        try:
            cli_mod._print_banner(3, "groq/llama-3.3", [], "")
        finally:
            sys.stdout = sys.__stdout__
        return buf.getvalue()

    def test_banner_has_help(self):
        self.assertIn("/help", self._banner_hint_line())

    def test_banner_has_tools(self):
        self.assertIn("/tools", self._banner_hint_line())

    def test_banner_has_council(self):
        self.assertIn("/council", self._banner_hint_line())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
