"""Tests for lib/hal/cli.py::_render_thinking and _print_response.

Hypothesis H-CLI-THINKING-1: After PR #125 W15 added ``ChatResponse.thinking``,
the chat REPL must render that text dimmed above the main answer block so
Haiku 4.5 extended-thinking becomes visible in the terminal. This wave is
the landing spot: a new ``_render_thinking`` helper + call-site integration
in ``_print_response``.

Covers:
- empty thinking text → no output, answer renders normally
- whitespace-only thinking → no output
- single-line thinking → dimmed block + separator
- multi-line thinking → each line wrapped + dimmed
- response without ``.thinking`` attribute → graceful degradation
- integration: ``_print_response`` calls the helper, answer still renders
"""
import io
import sys
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.hal import cli


@dataclass
class _FakeResponse:
    """Minimal stand-in for ``ChatResponse`` used by the CLI renderer."""
    text: str = ""
    thinking: str = ""
    provider: str = "anthropic"
    model: str = "claude-haiku-4-5-20251001"
    latency_ms: float = 1234.0
    cost_usd: float = 0.0
    usage: object = None


@dataclass
class _FakeCost:
    total_usd: float = 0.0


@dataclass
class _FakeConv:
    cost: _FakeCost = field(default_factory=_FakeCost)


def _capture(fn, *args, **kwargs) -> str:
    """Run a print-producing function and return the captured stdout."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue()


# ── _render_thinking unit tests ──────────────────────────────────────────


class TestRenderThinking(unittest.TestCase):

    def test_empty_string_is_noop(self):
        output = _capture(cli._render_thinking, "", 72)
        self.assertEqual(output, "")

    def test_whitespace_only_is_noop(self):
        output = _capture(cli._render_thinking, "   \n\t  \n", 72)
        self.assertEqual(output, "")

    def test_single_line_thinking_includes_label(self):
        output = _capture(cli._render_thinking, "Thinking about the problem.", 72)
        self.assertIn("[thinking]", output)
        self.assertIn("Thinking about the problem.", output)

    def test_multi_line_thinking_preserves_lines(self):
        thinking = "Line one of reasoning.\nLine two of reasoning.\nLine three."
        output = _capture(cli._render_thinking, thinking, 72)
        self.assertIn("Line one of reasoning.", output)
        self.assertIn("Line two of reasoning.", output)
        self.assertIn("Line three.", output)

    def test_thinking_is_wrapped_to_width(self):
        long_line = "word " * 60  # ~300 chars
        output = _capture(cli._render_thinking, long_line, 40)
        # Every output line (minus ANSI) should be at most width + small padding
        # Check that no line exceeds a generous wrap bound.
        plain_lines = [
            line for line in output.splitlines()
            if "[thinking]" not in line and "·" not in line
        ]
        for line in plain_lines:
            # Strip ANSI escape codes (GRAY/RESET) for length check
            stripped = line.replace(cli.GRAY, "").replace(cli.RESET, "")
            self.assertLessEqual(
                len(stripped), 50,
                f"Wrapped line exceeds width+padding: {stripped!r}",
            )

    def test_thinking_dimmed_with_gray_codes(self):
        output = _capture(cli._render_thinking, "Some reasoning.", 72)
        self.assertIn(cli.GRAY, output)
        self.assertIn(cli.RESET, output)

    def test_thinking_ends_with_separator(self):
        output = _capture(cli._render_thinking, "One line.", 20)
        # The helper prints a row of box-drawing dashes at the end
        self.assertIn("─", output)


# ── _print_response integration tests ───────────────────────────────────


class TestPrintResponseWithThinking(unittest.TestCase):

    def test_response_with_thinking_renders_both_blocks(self):
        response = _FakeResponse(
            text="The answer is 42.",
            thinking="Let me consider the options carefully.",
        )
        output = _capture(cli._print_response, response, _FakeConv())
        self.assertIn("[thinking]", output)
        self.assertIn("Let me consider the options carefully.", output)
        self.assertIn("The answer is 42.", output)

    def test_response_without_thinking_skips_thinking_block(self):
        response = _FakeResponse(text="Plain answer.", thinking="")
        output = _capture(cli._print_response, response, _FakeConv())
        self.assertNotIn("[thinking]", output)
        self.assertIn("Plain answer.", output)

    def test_response_missing_thinking_attribute_is_graceful(self):
        """Legacy responses without a ``.thinking`` attr should still render."""
        class LegacyResponse:
            text = "Legacy answer."
            provider = "legacy"
            model = "legacy-model"
            latency_ms = 100.0
            cost_usd = None
            usage = None

        output = _capture(cli._print_response, LegacyResponse(), _FakeConv())
        self.assertNotIn("[thinking]", output)
        self.assertIn("Legacy answer.", output)

    def test_thinking_renders_before_answer(self):
        response = _FakeResponse(
            text="ANSWER_MARKER",
            thinking="THINKING_MARKER",
        )
        output = _capture(cli._print_response, response, _FakeConv())
        thinking_pos = output.index("THINKING_MARKER")
        answer_pos = output.index("ANSWER_MARKER")
        self.assertLess(
            thinking_pos, answer_pos,
            "Thinking block must appear before the answer",
        )


if __name__ == "__main__":
    unittest.main()
