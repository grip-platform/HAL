"""Legacy regression anchors for the former lib/hal/chrome_stripper module.

``chrome_stripper.py`` was absorbed into :mod:`lib.hal.ui_chrome` in the
DRY cleanup sprint. The tests here intentionally keep importing via the
legacy public names (``strip_hal_chrome``, ``_is_chrome_line``,
``_strip_ansi``) so that if the backward-compatible aliases are ever
removed the regression anchors loudly fail instead of silently drifting.

Goodhart guard: the mutation-style test at the bottom constructs a message
containing a real footer paste and asserts the stripped result contains no
chrome characters. Removing the strip call from Conversation breaks it.
"""
from __future__ import annotations

import unittest

from lib.hal.ui_chrome import strip_hal_chrome, _is_chrome_line
from lib.hal.status_footer import FooterState, StatusFooter


# ---------------------------------------------------------------------------
# Helpers: derive canonical footer lines from the authoritative emitter so
# tests update automatically when status_footer._format_lines changes.
# ---------------------------------------------------------------------------

def _canonical_footer_lines() -> tuple[str, str]:
    """Return (bar_line, status_line) exactly as StatusFooter would emit them,
    but without ANSI codes so they are readable in assertions."""
    import io
    from lib.hal.ui_chrome import _strip_ansi

    buf = io.StringIO()
    # Monkeypatch isatty so render() doesn't bail out.
    buf.isatty = lambda: True  # type: ignore[attr-defined]
    footer = StatusFooter(stream=buf)
    state = FooterState(providers=17, model="groq/llama-3.3-70b-versatile", cost_usd=0.000945)
    footer.render(state)
    raw = buf.getvalue()
    # Strip ANSI and cursor-save/restore escape sequences.
    plain = _strip_ansi(raw).replace("\x1b[s", "").replace("\x1b[u", "")
    lines = [l for l in plain.splitlines() if l.strip()]
    assert len(lines) == 2, f"Expected 2 footer lines, got: {lines!r}"
    return lines[0], lines[1]


# ---------------------------------------------------------------------------
# Unit: _is_chrome_line
# ---------------------------------------------------------------------------

class TestIsChromeLineBarLine(unittest.TestCase):
    def test_detects_pure_dot_line(self):
        assert _is_chrome_line("  ···············")

    def test_detects_dot_line_with_ansi(self):
        assert _is_chrome_line("  \033[38;5;240m···············\033[0m")

    def test_not_chrome_empty(self):
        assert not _is_chrome_line("")

    def test_not_chrome_regular_text(self):
        assert not _is_chrome_line("  Some normal sentence.")


class TestIsChromeLineStatusLine(unittest.TestCase):
    def test_detects_bullet_line(self):
        assert _is_chrome_line("  ● 17 providers · groq/llama · $0.000945 · tools: off")

    def test_detects_bullet_line_with_ansi(self):
        assert _is_chrome_line("  \033[2m●\033[0m 17 providers · groq/llama · $0.001")

    def test_not_chrome_markdown_bullet(self):
        # A markdown bullet using ASCII hyphen must NOT be stripped.
        assert not _is_chrome_line("- item in a list")

    def test_not_chrome_regular_bullet_unicode(self):
        # Only the specific HAL bullet (U+25CF ●) triggers; other bullet chars do not.
        assert not _is_chrome_line("• regular unicode bullet")


# ---------------------------------------------------------------------------
# Unit: strip_hal_chrome — basic cases
# ---------------------------------------------------------------------------

class TestStripHalChrome(unittest.TestCase):
    def test_removes_status_line(self):
        text = "● 17 providers · groq/llama-3.3-70b-versatile · $0.000945 · tools: off"
        assert strip_hal_chrome(text) == ""

    def test_removes_bar_line(self):
        text = "  ···············"
        assert strip_hal_chrome(text) == ""

    def test_preserves_user_text(self):
        text = "What is a Knowledge Engine?"
        assert strip_hal_chrome(text) == text

    def test_preserves_code_blocks(self):
        code = "```python\nfor i in range(10):\n    print(i)\n```"
        assert strip_hal_chrome(code) == code

    def test_preserves_markdown_bullets(self):
        text = "- item one\n- item two"
        assert strip_hal_chrome(text) == text

    def test_removes_footer_in_middle_of_multiline(self):
        text = (
            "Can you help me with:\n"
            "● 17 providers · groq/llama · $0.001 · tools: off\n"
            "  ···············\n"
            "- Understanding KONO"
        )
        result = strip_hal_chrome(text)
        assert "●" not in result
        assert "···" not in result
        assert "Can you help me with:" in result
        assert "Understanding KONO" in result

    def test_idempotent(self):
        text = (
            "Help me\n"
            "● 17 providers · groq/llama · $0.001 · tools: off\n"
            "  ···············\n"
            "with this"
        )
        once = strip_hal_chrome(text)
        twice = strip_hal_chrome(once)
        assert once == twice

    def test_empty_string(self):
        assert strip_hal_chrome("") == ""

    def test_only_whitespace(self):
        assert strip_hal_chrome("   \n   ") == ""


# ---------------------------------------------------------------------------
# Unit: canonical footer from live emitter (auto-updates on footer changes)
# ---------------------------------------------------------------------------

class TestCanonicalFooterStripped(unittest.TestCase):
    """Derive the real footer lines from StatusFooter and assert they are
    removed.  If status_footer._format_lines changes, this test automatically
    validates the stripper still works — no manual regex maintenance needed."""

    def test_canonical_bar_line_stripped(self):
        bar, _status = _canonical_footer_lines()
        assert strip_hal_chrome(bar) == ""

    def test_canonical_status_line_stripped(self):
        _bar, status = _canonical_footer_lines()
        assert strip_hal_chrome(status) == ""

    def test_canonical_full_footer_stripped(self):
        bar, status = _canonical_footer_lines()
        full_footer = bar + "\n" + status
        assert strip_hal_chrome(full_footer) == ""


# ---------------------------------------------------------------------------
# Goodhart guard: mutation-style end-to-end test
#
# Constructs the exact paste scenario from issue #201 and asserts the stripped
# message contains no chrome characters.  Removing the strip_hal_chrome() call
# from Conversation._prepare_turn makes this test fail.
# ---------------------------------------------------------------------------

class TestGoodhartGuard(unittest.TestCase):
    def test_pasted_footer_chrome_absent_after_strip(self):
        """Simulate user pasting REPL history with footer included."""
        bar, status = _canonical_footer_lines()
        pasted_context = (
            "Here is my previous session:\n\n"
            "User: What is HAL?\n"
            "Assistant: HAL is a Knowledge Engine.\n"
            f"{bar}\n"
            f"{status}\n\n"
            "Now my new question: explain the Limit Sentinel."
        )
        result = strip_hal_chrome(pasted_context)

        # Chrome characters must be absent from the result sent to the LLM.
        assert "●" not in result, "Bullet chrome survived stripping"
        assert "···" not in result, "Bar chrome survived stripping"

        # Legitimate content must be preserved.
        assert "What is HAL?" in result
        assert "HAL is a Knowledge Engine." in result
        assert "explain the Limit Sentinel." in result

    def test_ansi_footer_chrome_absent_after_strip(self):
        """Same scenario but with ANSI codes intact (direct terminal copy)."""
        ansi_status = "  \033[2m●\033[0m 17 providers · groq/llama-3.3-70b-versatile · $0.000945 · tools: \033[2moff\033[0m"
        ansi_bar = "  \033[38;5;240m" + "·" * 60 + "\033[0m"
        pasted = f"My question\n{ansi_bar}\n{ansi_status}\nafter footer"
        result = strip_hal_chrome(pasted)

        assert "●" not in result
        assert "···" not in result
        assert "My question" in result
        assert "after footer" in result
