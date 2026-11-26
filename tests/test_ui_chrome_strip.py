"""Regression anchors for Issue #201 — UI chrome stripping.

The hypothesis under test (H-UI-CHROME-STRIP): pasting HAL's own REPL
chrome back as user input will never again reach the LLM. The regex
catalogue in :mod:`lib.hal.ui_chrome` physically prevents the model from
seeing the tokens that trigger "Limit Sentinel has switched" and similar
hallucinations (Issue #201).

Goodhart-protection: every assertion checks actual content, not just
call counts. If a pattern is weakened (e.g. the status-line regex is
made too permissive and swallows user prose, or too restrictive and
misses a chrome variant), at least one of these tests will fail.
"""
from __future__ import annotations

import importlib.util
import pathlib
import unittest

_REPO = pathlib.Path(__file__).parent.parent


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ui_chrome = _load("lib/hal/ui_chrome.py", "_wt_ui_chrome")
strip_ui_chrome = _ui_chrome.strip_ui_chrome
UI_CHROME_SYSTEM_NOTE = _ui_chrome.UI_CHROME_SYSTEM_NOTE


class StripUiChromeTests(unittest.TestCase):
    # ── Core strip behaviour ────────────────────────────────────────

    def test_status_line_only_returns_empty_after_trim(self):
        footer = "  ● 17 providers · groq/llama-3.3-70b-versatile · $0.000945 · tools: off"
        self.assertEqual(strip_ui_chrome(footer), "")

    def test_status_line_above_prose_keeps_prose(self):
        footer = (
            "  ● 17 providers · groq/llama-3.3-70b-versatile · $0.000945 · tools: off\n"
            "What is HAL?"
        )
        self.assertEqual(strip_ui_chrome(footer), "What is HAL?")

    def test_full_chrome_paste_preserves_only_user_intent(self):
        """The exact repro from Issue #201's chat paste."""
        raw = (
            "  ·····································································\n"
            "  ● 17 providers · groq/llama-3.3-70b-versatile · $0.000945 · tools: off\n"
            "❯  Why did my provider switch?\n"
            "  ·····································································"
        )
        self.assertEqual(
            strip_ui_chrome(raw),
            "Why did my provider switch?",
        )

    def test_bar_divider_is_stripped(self):
        self.assertEqual(strip_ui_chrome("  ···································"), "")

    def test_bare_prompt_marker_is_stripped(self):
        self.assertEqual(strip_ui_chrome("❯"), "")
        self.assertEqual(strip_ui_chrome("  ❯  "), "")

    def test_prompt_marker_preserves_content_after_marker(self):
        self.assertEqual(strip_ui_chrome("❯  explain the router"), "explain the router")
        self.assertEqual(strip_ui_chrome("  ❯  what is a council?"), "what is a council?")

    def test_tool_mode_banner_is_stripped(self):
        self.assertEqual(strip_ui_chrome("tool mode enabled (groq)"), "")
        self.assertEqual(strip_ui_chrome("  tool mode enabled (ollama)"), "")

    def test_providers_singular_also_matches(self):
        self.assertEqual(
            strip_ui_chrome("● 1 provider · ollama/gemma4 · $0.000000 · tools: on"),
            "",
        )

    # ── Non-chrome text MUST survive ─────────────────────────────────

    def test_user_prose_with_mid_line_middot_survives(self):
        text = "Redis vs MongoDB · which is faster for caching?"
        self.assertEqual(strip_ui_chrome(text), text)

    def test_user_prose_mentioning_circle_glyph_survives(self):
        """A line starting with `●` that is NOT the status format is kept.

        The status regex requires the full shape
        (``● N providers · ...``). A bulleted note that happens to
        start with ``●`` but not match the rest must NOT be stripped.
        """
        text = "● Note: this is a bullet, not a status footer."
        self.assertEqual(strip_ui_chrome(text), text)

    def test_user_prose_with_single_middot_survives(self):
        """One or two `·` glyphs is not a bar — keep the line."""
        text = "latency · throughput · p99"
        self.assertEqual(strip_ui_chrome(text), text)

    def test_empty_string_is_empty(self):
        self.assertEqual(strip_ui_chrome(""), "")

    def test_none_like_empty_is_idempotent(self):
        self.assertEqual(strip_ui_chrome("\n\n\n"), "")

    def test_multi_paragraph_preserves_internal_blank_line(self):
        text = "paragraph one\n\nparagraph two"
        self.assertEqual(strip_ui_chrome(text), text)

    def test_idempotent(self):
        raw = (
            "  ● 5 providers · ollama/mistral · $0.0 · tools: off\n"
            "❯ hello there"
        )
        once = strip_ui_chrome(raw)
        twice = strip_ui_chrome(once)
        self.assertEqual(once, twice)
        self.assertEqual(once, "hello there")


class ToolTraceStripTests(unittest.TestCase):
    """Regression anchors for Issue #286 — tool-call trace stripping.

    When a user copies REPL history that includes tool-call traces back
    as input, lines like "▸ Write /path..." and "✖ ToolName (err)" should
    be silently dropped — they are UI chrome, not user intent.
    """

    def test_tool_success_trace_stripped(self):
        raw = "▸ Write /tmp/hello.txt\nactual question"
        self.assertEqual(strip_ui_chrome(raw), "actual question")

    def test_tool_failure_trace_stripped(self):
        raw = "✖ Read /nonexistent (file not found)\nactual question"
        self.assertEqual(strip_ui_chrome(raw), "actual question")

    def test_multi_trace_lines_all_stripped(self):
        raw = "▸ Bash echo hello\n▸ Write /tmp/x\n✖ Read /missing (err)\nkeep this"
        self.assertEqual(strip_ui_chrome(raw), "keep this")

    def test_tool_trace_indented_stripped(self):
        raw = "  ▸ Write /tmp/file.txt\nkeep this"
        self.assertEqual(strip_ui_chrome(raw), "keep this")

    def test_mid_sentence_arrow_preserved(self):
        raw = "The arrow ▸ in typography marks a continuation."
        self.assertEqual(strip_ui_chrome(raw), raw)

    def test_mixed_session_paste(self):
        raw = (
            "● 3 providers · ollama/mistral · $0.0 · tools: on\n"
            "▸ Write /tmp/poem.md\n"
            "✖ Read /missing (not found)\n"
            "❯ what did you just write?"
        )
        self.assertEqual(strip_ui_chrome(raw), "what did you just write?")


class MutationGuardTests(unittest.TestCase):
    """Goodhart-protection: if the implementation is neutered (returns
    input unchanged, always returns empty, etc.) these tests MUST fail.

    Each test here has been constructed so a broken implementation of
    strip_ui_chrome has no chance of passing by accident.
    """

    def test_non_trivial_removal(self):
        # Returning input verbatim would pass many tests; this one forces
        # the function to actually remove something.
        raw = "● 17 providers · groq · $0.0 · tools: on\nREAL_CONTENT"
        self.assertNotIn("providers", strip_ui_chrome(raw))
        self.assertIn("REAL_CONTENT", strip_ui_chrome(raw))

    def test_non_trivial_preservation(self):
        # Always-empty would pass "strip the footer" but this catches it.
        self.assertIn("keep me", strip_ui_chrome("keep me"))

    def test_preserves_code_block_with_middots(self):
        # A real use case: user pastes a log file containing middots.
        raw = "log line · INFO · request processed"
        self.assertEqual(strip_ui_chrome(raw), raw)


class SystemNoteTests(unittest.TestCase):
    """Sanity check on the belt-and-braces system note."""

    def test_note_is_nonempty(self):
        self.assertTrue(UI_CHROME_SYSTEM_NOTE.strip())

    def test_note_mentions_key_glyphs(self):
        # If someone rewrites the note without mentioning the glyphs, the
        # model loses the prior. Catch that.
        self.assertIn("●", UI_CHROME_SYSTEM_NOTE)
        self.assertIn("❯", UI_CHROME_SYSTEM_NOTE)

    def test_note_names_the_hallucination_pattern(self):
        # The specific hallucination we're guarding against.
        self.assertIn("Limit Sentinel", UI_CHROME_SYSTEM_NOTE)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
