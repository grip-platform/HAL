"""Tests for the 11 Swiss Nihilism TUI improvements.

Each test is Goodhart-protected: it will FAIL if the implementation regresses,
not just if the test runs. Where possible, tests verify observable OUTPUT values
(specific glyphs, column content) rather than call counts.
"""
import io
import re
import sys
import unittest
from unittest.mock import MagicMock, patch

from lib.hal.cli import (
    _bar,
    _bar_with_turn,
    _clear_processing_indicator,
    _dispatch_turn,
    _handle_command,
    _render_thinking,
    _show_cost,
    _stream_plain_tool_response,
)
from lib.hal.status_footer import FooterState, StatusFooter
from lib.hal.ui_chrome import strip_ui_chrome


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _capture(fn, *args, **kwargs) -> str:
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        fn(*args, **kwargs)
    return buf.getvalue()


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def _mock_conv(turns=3, total_tokens=12345, total_usd=0.0047,
               cost_by_provider=None, context_usage_pct=0.0):
    conv = MagicMock()
    conv.model = "groq/llama-3.3-70b-versatile"
    conv.cost.total_usd = total_usd
    conv.status.return_value = {
        "turns": turns,
        "total_tokens": total_tokens,
        "total_cost_usd": total_usd,
        "cost_by_provider": cost_by_provider or {},
        "context_usage_pct": context_usage_pct,
        "context_warning": None,
    }
    return conv


def _mock_response(text="Hello, world!"):
    resp = MagicMock()
    resp.text = text
    resp.thinking = ""
    resp.latency_ms = 100
    resp.provider = "groq"
    resp.cost_usd = 0.0
    resp.usage = MagicMock(completion_tokens=10)
    return resp


# ===========================================================================
# Improvement 1: Layout fix — no cursor save/restore in StatusFooter.render()
# ===========================================================================

class TestLayoutFixNoCursorSaveRestore(unittest.TestCase):
    def test_render_output_has_no_save_cursor_escape(self):
        buf = io.StringIO()
        footer = StatusFooter(stream=buf)
        with patch.object(buf, "isatty", return_value=True):
            footer.render(FooterState(providers=3, model="gpt-4o", cost_usd=0.05))
        output = buf.getvalue()
        self.assertNotIn("\033[s", output, "Cursor save escape must not be in footer output")
        self.assertNotIn("\033[u", output, "Cursor restore escape must not be in footer output")

    def test_render_writes_exactly_two_lines(self):
        buf = io.StringIO()
        footer = StatusFooter(stream=buf)
        with patch.object(buf, "isatty", return_value=True):
            lines_written = footer.render(FooterState(providers=1, model="x"))
        self.assertEqual(lines_written, 2)


# ===========================================================================
# Improvement 2: Progressive cost — hidden <$0.01, dim $0.01-$0.10, normal +
# ===========================================================================

class TestProgressiveCost(unittest.TestCase):
    def _status_plain(self, cost_usd: float) -> str:
        buf = io.StringIO()
        footer = StatusFooter(stream=buf)
        with patch.object(buf, "isatty", return_value=True):
            footer.render(FooterState(providers=2, model="x", cost_usd=cost_usd))
        return _strip_ansi(buf.getvalue())

    def test_cost_hidden_below_one_cent(self):
        line = self._status_plain(0.009)
        self.assertNotIn("0.009", line)

    def test_cost_shown_between_one_and_ten_cents(self):
        line = self._status_plain(0.05)
        self.assertIn("$0.0500", line)

    def test_cost_shown_at_ten_cents_and_above(self):
        line = self._status_plain(0.15)
        self.assertIn("$0.1500", line)


# ===========================================================================
# Improvement 3: ─ separator bar (Swiss grid, not · dots)
# ===========================================================================

class TestSwissSeparatorBar(unittest.TestCase):
    def test_footer_bar_uses_em_dash_not_middle_dot(self):
        buf = io.StringIO()
        footer = StatusFooter(stream=buf)
        with patch.object(buf, "isatty", return_value=True):
            footer.render(FooterState(providers=1, model="x"))
        first_line = _strip_ansi(buf.getvalue().split("\n")[0])
        self.assertIn("─", first_line, "Footer bar must use ─")
        self.assertNotIn("·", first_line, "Footer bar must not use ·")

    def test_bar_helper_uses_em_dash_by_default(self):
        result = _strip_ansi(_bar())
        self.assertIn("─", result)
        self.assertNotIn("·", result)


# ===========================================================================
# Improvement 4: │ left-rail for HAL responses
# ===========================================================================

class TestHALResponseLeftRail(unittest.TestCase):
    def test_print_response_uses_pipe_left_rail(self):
        resp = _mock_response("HAL speaks.")
        conv = _mock_conv()
        stdout_buf = io.StringIO()
        with patch("lib.hal.cli._STREAMER"), \
             patch("lib.hal.cli._print_response_footer"), \
             patch("lib.hal.cli._maybe_print_cache_marker"), \
             patch("lib.hal.cli._maybe_print_citation_chip"), \
             patch("lib.hal.cli._maybe_print_cost_heatmap"), \
             patch("sys.stdout", stdout_buf):
            from lib.hal.cli import _print_response
            _print_response(resp, conv, turn=1)
        # │ is written atomically via sys.stdout.write, not via _STREAMER.render_to_stdout
        self.assertIn("│", stdout_buf.getvalue(), "_print_response must use │ left-rail")

    def test_thinking_uses_dashed_rail_not_solid(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            _render_thinking("Reasoning...", width=72)
        plain = _strip_ansi(buf.getvalue())
        for line in plain.splitlines():
            if "Reasoning" in line:
                self.assertIn("╎", line)
                self.assertNotIn("│", line)


# ===========================================================================
# Improvement 5: ╎ left-rail for thinking blocks
# ===========================================================================

class TestThinkingRail(unittest.TestCase):
    def test_render_thinking_uses_dashed_vertical(self):
        out = _capture(_render_thinking, "Considering the problem.", width=72)
        self.assertIn("╎", _strip_ansi(out))

    def test_render_thinking_ends_with_em_dash_rule(self):
        out = _capture(_render_thinking, "Thought.", width=20)
        last = [l for l in _strip_ansi(out).splitlines() if l.strip()][-1]
        self.assertIn("─", last)

    def test_render_thinking_noop_on_empty(self):
        out = _capture(_render_thinking, "", width=72) + _capture(_render_thinking, "   ", width=72)
        self.assertEqual(out, "")


# ===========================================================================
# Improvement 6: ◈ processing indicator erased with \r
# ===========================================================================

class TestProcessingIndicator(unittest.TestCase):
    def test_clear_writes_spaces_and_carriage_return(self):
        writes = []
        mock_stdout = MagicMock()
        mock_stdout.write.side_effect = writes.append
        with patch("sys.stdout", mock_stdout):
            _clear_processing_indicator()
        combined = "".join(writes)
        self.assertIn("\r", combined)
        self.assertIn(" ", combined)


# ===========================================================================
# Improvement 7: ⊘ ErrorType structured error via _dispatch_turn
# ===========================================================================

class TestStructuredErrorDisplay(unittest.TestCase):
    def test_dispatch_turn_shows_oplus_on_exception(self):
        conv = _mock_conv()
        conv.send.side_effect = ValueError("provider timeout")
        tool_state = {"enabled": False, "chat_fn": None}
        buf = io.StringIO()
        with patch("sys.stdout", buf), \
             patch("lib.hal.cli._print_response"), \
             patch("lib.hal.cli._clear_processing_indicator"):
            _dispatch_turn(conv, tool_state, "hello", turn=1)
        plain = _strip_ansi(buf.getvalue())
        self.assertIn("⊘", plain)
        self.assertIn("ValueError", plain)

    def test_dispatch_turn_does_not_use_old_x_marker(self):
        conv = _mock_conv()
        conv.send.side_effect = RuntimeError("boom")
        tool_state = {"enabled": False, "chat_fn": None}
        buf = io.StringIO()
        with patch("sys.stdout", buf), \
             patch("lib.hal.cli._print_response"), \
             patch("lib.hal.cli._clear_processing_indicator"):
            _dispatch_turn(conv, tool_state, "hi", turn=1)
        self.assertNotIn("✗", _strip_ansi(buf.getvalue()))


# ===========================================================================
# Improvement 8: _bar_with_turn embeds turn number in ╌ separator
# ===========================================================================

class TestBarWithTurn(unittest.TestCase):
    def test_turn_zero_returns_plain_bar(self):
        result = _strip_ansi(_bar_with_turn(0))
        self.assertNotRegex(result, r"\d")
        self.assertIn("╌", result)

    def test_turn_number_embedded(self):
        for n in (1, 5, 12):
            with self.subTest(n=n):
                result = _strip_ansi(_bar_with_turn(n))
                self.assertIn(str(n), result)
                self.assertIn("╌", result)

    def test_bar_padded_on_both_sides_of_number(self):
        result = _strip_ansi(_bar_with_turn(3))
        idx = result.index("3")
        self.assertIn("╌", result[:idx])
        self.assertIn("╌", result[idx + 1:])


# ===========================================================================
# Improvement 9: CYAN │ for tool responses
# ===========================================================================

class TestToolResponseRail(unittest.TestCase):
    def test_stream_plain_tool_response_uses_cyan_pipe_rail(self):
        stdout_buf = io.StringIO()
        with patch("lib.hal.cli._STREAMER"), \
             patch("sys.stdout", stdout_buf):
            _stream_plain_tool_response("Tool result.", tool_count=1, turn=2)
        raw = stdout_buf.getvalue()
        # │ and CYAN ANSI code are written atomically via sys.stdout.write per line
        self.assertIn("│", raw, "Tool response rail must use │ glyph")
        self.assertIn("\033[36m", raw, "Tool response rail must be CYAN")


# ===========================================================================
# Improvement 10: Context % in footer (shown when > 20%)
# ===========================================================================

class TestContextPercentage(unittest.TestCase):
    def _footer_plain(self, context_pct: float) -> str:
        buf = io.StringIO()
        footer = StatusFooter(stream=buf)
        with patch.object(buf, "isatty", return_value=True):
            footer.render(FooterState(providers=1, model="x", context_pct=context_pct))
        return _strip_ansi(buf.getvalue())

    def test_context_hidden_at_or_below_twenty_percent(self):
        for pct in (0.0, 10.0, 20.0):
            with self.subTest(pct=pct):
                self.assertNotIn("ctx", self._footer_plain(pct))

    def test_context_shown_above_twenty_percent(self):
        for pct in (21.0, 50.0, 85.0):
            with self.subTest(pct=pct):
                line = self._footer_plain(pct)
                self.assertIn("ctx", line)
                self.assertIn(str(int(pct)), line)


# ===========================================================================
# Improvement 11a: Arrow notation in _handle_command
# ===========================================================================

class TestArrowNotation(unittest.TestCase):
    def _run_cmd(self, cmd: str) -> str:
        conv = _mock_conv()
        registry = MagicMock()
        council = MagicMock()
        tool_state = {"enabled": False, "chat_fn": None}
        buf = io.StringIO()
        with patch("sys.stdout", buf), patch("lib.hal.cli._maybe_rebind_tools"):
            _handle_command(cmd, conv, registry, council, tool_state)
        return _strip_ansi(buf.getvalue())

    def test_model_switch_uses_arrow_notation(self):
        out = self._run_cmd("/model anthropic/claude-opus-4-7")
        self.assertIn("model →", out)
        self.assertNotIn("Switched to", out)

    def test_mode_switch_uses_arrow_notation(self):
        out = self._run_cmd("/mode legal")
        self.assertIn("mode →", out)
        self.assertNotIn("Mode:", out)

    def test_clear_uses_lowercase(self):
        out = self._run_cmd("/clear")
        self.assertIn("cleared", out)
        self.assertNotIn("Cleared", out)


# ===========================================================================
# Improvement 11b: Swiss /quit table in _show_cost
# ===========================================================================

class TestSwissQuitTable(unittest.TestCase):
    def _table(self, **kwargs) -> str:
        return _strip_ansi(_capture(_show_cost, _mock_conv(**kwargs)))

    def test_headers_present(self):
        out = self._table(turns=5, total_tokens=9876, total_usd=0.0033)
        self.assertIn("turns", out)
        self.assertIn("tokens", out)
        self.assertIn("cost", out)

    def test_turn_count_shown(self):
        self.assertIn("7", self._table(turns=7))

    def test_token_count_formatted_with_comma(self):
        self.assertIn("12,345", self._table(total_tokens=12345))

    def test_total_cost_shown(self):
        self.assertIn("0.0047", self._table(total_usd=0.0047))

    def test_provider_breakdown_shown(self):
        out = self._table(total_usd=0.005, cost_by_provider={"groq": 0.002, "anthropic": 0.003})
        self.assertIn("groq", out)
        self.assertIn("anthropic", out)

    def test_zero_cost_providers_excluded(self):
        out = self._table(total_usd=0.003, cost_by_provider={"groq": 0.003, "anthropic": 0.0})
        self.assertIn("groq", out)
        self.assertNotIn("anthropic", out)

    def test_divider_uses_em_dash(self):
        out = self._table(total_usd=0.005, cost_by_provider={"groq": 0.005})
        self.assertIn("─", out)


# ===========================================================================
# Improvement 11c: turn_count increments in _run_chat
# ===========================================================================

class TestTurnCountWiring(unittest.TestCase):
    def test_dispatch_called_with_incrementing_turns(self):
        from lib.hal.cli import _run_chat
        dispatched = []

        def fake_dispatch(conv, tool_state, user_input, turn):
            dispatched.append(turn)

        inputs = iter(["hello", "world", KeyboardInterrupt()])

        def fake_input(_prompt):
            val = next(inputs)
            if isinstance(val, KeyboardInterrupt):
                raise val
            return val

        registry = MagicMock()
        registry.list_providers.return_value = ["groq"]
        registry.routing.default_model = "groq/llama"

        with patch("lib.hal.cli._dispatch_turn", side_effect=fake_dispatch), \
             patch("builtins.input", side_effect=fake_input), \
             patch("lib.hal.cli._print_banner"), \
             patch("lib.hal.cli._render_status_footer"), \
             patch("lib.hal.cli._show_cost"), \
             patch("lib.hal.cli.Conversation"), \
             patch("lib.hal.cli.HALCouncil"), \
             patch("lib.hal.cli.StatusFooter"):
            _run_chat(registry, [], "groq/llama", "default")

        self.assertEqual(dispatched, [1, 2])


# ===========================================================================
# ui_chrome: ─ bar pattern recognised as chrome
# ===========================================================================

class TestUiChromeRecognisesNewBar(unittest.TestCase):
    def test_em_dash_bar_stripped(self):
        self.assertEqual(strip_ui_chrome("─" * 40), "")

    def test_middle_dot_bar_still_stripped(self):
        self.assertEqual(strip_ui_chrome("·" * 40), "")

    def test_short_bar_not_stripped(self):
        bar = "─" * 3
        self.assertEqual(strip_ui_chrome(bar), bar)

    def test_mixed_content_preserved(self):
        content = "─── some heading ───"
        self.assertEqual(strip_ui_chrome(content), content)


if __name__ == "__main__":
    unittest.main()
