"""Tests for ``lib.hal.osa.escape`` — AppleScript + JavaScript escapers."""
from __future__ import annotations

import unittest

from lib.hal.osa.escape import applescript_escape, js_escape


class AppleScriptEscape(unittest.TestCase):
    """Minimal escape surface: backslash-doubling, quote-escaping, passthrough."""

    def test_plain_string_surrounded_by_quotes(self) -> None:
        self.assertEqual(applescript_escape("hello"), '"hello"')

    def test_empty_string_yields_empty_literal(self) -> None:
        self.assertEqual(applescript_escape(""), '""')

    def test_double_quote_is_backslash_escaped(self) -> None:
        self.assertEqual(
            applescript_escape('he said "hi"'), '"he said \\"hi\\""'
        )

    def test_backslash_is_doubled_first(self) -> None:
        # Order matters — backslash must be doubled BEFORE quotes are escaped
        # so the escape character for a quote is not itself doubled.
        self.assertEqual(applescript_escape("a\\b"), '"a\\\\b"')

    def test_backslash_then_quote_stays_correct(self) -> None:
        self.assertEqual(
            applescript_escape('a\\"b'), '"a\\\\\\"b"'
        )

    def test_newline_passes_through(self) -> None:
        # AppleScript accepts newlines inside string literals.
        self.assertEqual(applescript_escape("line1\nline2"), '"line1\nline2"')

    def test_unicode_passes_through(self) -> None:
        self.assertEqual(applescript_escape("café ☕"), '"café ☕"')

    def test_injection_attempt_rendered_inert(self) -> None:
        # The kind of string the tab_manager.py url_pattern injection
        # path would accept unescaped today.
        hostile = '"; do shell script "rm -rf /"; -- '
        escaped = applescript_escape(hostile)
        # Verify every bare double-quote in the input is now backslash-prefixed.
        import re
        bare_quotes = re.findall(r'(?<!\\)"', escaped)
        # Two bare quotes are the outer delimiters added by the escaper.
        self.assertEqual(len(bare_quotes), 2)


class JavaScriptEscape(unittest.TestCase):
    """JS-OSA escape: backslash, double-quote, and parser-breaking controls."""

    def test_plain_string_surrounded_by_quotes(self) -> None:
        self.assertEqual(js_escape("hello"), '"hello"')

    def test_double_quote_escaped(self) -> None:
        self.assertEqual(js_escape('say "hi"'), '"say \\"hi\\""')

    def test_backslash_doubled_first(self) -> None:
        self.assertEqual(js_escape("a\\b"), '"a\\\\b"')

    def test_newline_escaped_for_js_parser(self) -> None:
        """Contrast with AppleScript — JS cannot tolerate raw newlines."""
        self.assertEqual(js_escape("a\nb"), '"a\\nb"')

    def test_carriage_return_escaped(self) -> None:
        self.assertEqual(js_escape("a\rb"), '"a\\rb"')

    def test_tab_escaped(self) -> None:
        self.assertEqual(js_escape("a\tb"), '"a\\tb"')

    def test_combined_controls_escaped(self) -> None:
        self.assertEqual(js_escape("a\n\tb"), '"a\\n\\tb"')

    def test_injection_payload_rendered_inert(self) -> None:
        """Goodhart guard — a JS-injection attempt must not close the literal."""
        hostile = '"; Application("System Events").restart(); //'
        escaped = js_escape(hostile)
        import re
        bare_quotes = re.findall(r'(?<!\\)"', escaped)
        self.assertEqual(len(bare_quotes), 2, escaped)


if __name__ == "__main__":
    unittest.main()
