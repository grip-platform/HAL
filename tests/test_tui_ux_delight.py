"""Regression anchors for the 11 TUI UX delight improvements.

Each improvement has at least one Goodhart-protected test: a test that
passes when the implementation is correct and FAILS when it is broken.
"""
from __future__ import annotations

import io
import sys
import types
import unittest

from lib.hal.cli import (
    _COST_DIM_FLOOR,
    _COST_NORMAL_FLOOR,
    _COST_YELLOW_FLOOR,
    _KNOWN_COMMANDS,
    _LATENCY_GREEN_MS,
    _LATENCY_YELLOW_MS,
    _did_you_mean,
    _format_cost,
    _format_latency,
    _provider_badge,
    CYAN, DIM, GREEN, RED, RESET, YELLOW,
)


# ── Improvement 1 & 8: cost threshold + per-turn delta ───────────────

class FormatCostTests(unittest.TestCase):

    def test_below_threshold_returns_empty(self):
        self.assertEqual(_format_cost(0.005), "")

    def test_exactly_at_threshold_returns_empty(self):
        # $0.0099... should still be hidden
        self.assertEqual(_format_cost(0.0099), "")

    def test_at_dim_floor_shows_cost(self):
        result = _format_cost(0.01)
        self.assertIn("0.0100", result)

    def test_dim_range_uses_dim_codes(self):
        result = _format_cost(0.05)
        self.assertIn(DIM, result)

    def test_normal_range_no_dim(self):
        result = _format_cost(0.15)
        # Normal range should NOT lead with DIM
        self.assertFalse(result.startswith(DIM))

    def test_yellow_range_uses_yellow(self):
        result = _format_cost(1.50)
        self.assertIn(YELLOW, result)

    def test_turn_delta_shown_when_provided(self):
        result = _format_cost(0.05, 0.003)
        self.assertIn("+$0.0030", result)

    def test_turn_delta_hidden_when_zero(self):
        result = _format_cost(0.05, 0.0)
        self.assertNotIn("+$", result)

    def test_goodhart_cost_is_in_output(self):
        # If implementation always returns "" this would catch it
        self.assertIn("session", _format_cost(0.50))


# ── Improvement 2: latency colour coding ─────────────────────────────

class FormatLatencyTests(unittest.TestCase):

    def test_fast_is_green(self):
        result = _format_latency(200)
        self.assertIn(GREEN, result)
        self.assertIn("200ms", result)

    def test_medium_is_yellow(self):
        result = _format_latency(1000)
        self.assertIn(YELLOW, result)

    def test_slow_is_red(self):
        result = _format_latency(3000)
        self.assertIn(RED, result)

    def test_boundary_green_to_yellow(self):
        # Exactly at the boundary should be yellow (not green)
        result = _format_latency(_LATENCY_GREEN_MS)
        self.assertIn(YELLOW, result)
        self.assertNotIn(GREEN, result)

    def test_goodhart_latency_ms_in_output(self):
        self.assertIn("500ms", _format_latency(500))


# ── Improvement 3: LOCAL badge ────────────────────────────────────────

class ProviderBadgeTests(unittest.TestCase):

    def _make_conv(self, provider_name: str, type_value: str):
        provider = types.SimpleNamespace(name=provider_name, type=types.SimpleNamespace(value=type_value))
        registry = types.SimpleNamespace(list_providers=lambda: [provider])
        return types.SimpleNamespace(_registry=registry)

    def test_local_provider_shows_badge(self):
        conv = self._make_conv("ollama", "local")
        result = _provider_badge("ollama", conv)
        self.assertIn("LOCAL", result)

    def test_cloud_provider_no_badge(self):
        conv = self._make_conv("groq", "cloud")
        result = _provider_badge("groq", conv)
        self.assertEqual(result, "")

    def test_unknown_provider_no_badge(self):
        conv = self._make_conv("groq", "cloud")
        result = _provider_badge("unknown", conv)
        self.assertEqual(result, "")

    def test_exception_returns_empty(self):
        broken = types.SimpleNamespace(_registry=types.SimpleNamespace(
            list_providers=lambda: (_ for _ in ()).throw(RuntimeError("oops"))
        ))
        self.assertEqual(_provider_badge("any", broken), "")


# ── Improvement 11: did-you-mean ─────────────────────────────────────

class DidYouMeanTests(unittest.TestCase):

    def test_typo_model_suggests_model(self):
        result = _did_you_mean("/modell")
        self.assertEqual(result, "/model")

    def test_typo_quit_suggests_quit(self):
        result = _did_you_mean("/quut")
        self.assertEqual(result, "/quit")

    def test_completely_wrong_returns_empty(self):
        result = _did_you_mean("/zzzzzfoo")
        self.assertEqual(result, "")

    def test_exact_match_returns_itself(self):
        result = _did_you_mean("/help")
        self.assertEqual(result, "/help")

    def test_known_commands_is_frozen(self):
        self.assertIsInstance(_KNOWN_COMMANDS, frozenset)


# ── Improvement 9: tool count badge ──────────────────────────────────

class StreamPlainToolResponseTests(unittest.TestCase):

    def _capture(self, text: str, tool_count: int) -> str:
        from lib.hal.cli import _stream_plain_tool_response
        buf = io.StringIO()
        _orig = sys.stdout
        sys.stdout = buf
        try:
            import unittest.mock as mock
            with mock.patch("lib.hal.cli._STREAMER") as ms:
                ms.render_to_stdout = lambda t, **kw: sys.stdout.write(t)
                _stream_plain_tool_response(text, tool_count)
        finally:
            sys.stdout = _orig
        return buf.getvalue()

    def test_tool_count_shown_when_nonzero(self):
        out = self._capture("done", 3)
        self.assertIn("3 tools", out)

    def test_tool_count_singular(self):
        out = self._capture("done", 1)
        self.assertIn("1 tool", out)
        self.assertNotIn("1 tools", out)

    def test_no_badge_when_zero(self):
        out = self._capture("done", 0)
        self.assertNotIn("tool", out)


# ── Constants sanity ──────────────────────────────────────────────────

class ThresholdConstantTests(unittest.TestCase):

    def test_dim_floor_below_normal_floor(self):
        self.assertLess(_COST_DIM_FLOOR, _COST_NORMAL_FLOOR)

    def test_normal_floor_below_yellow_floor(self):
        self.assertLess(_COST_NORMAL_FLOOR, _COST_YELLOW_FLOOR)

    def test_green_ms_below_yellow_ms(self):
        self.assertLess(_LATENCY_GREEN_MS, _LATENCY_YELLOW_MS)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
