"""Regression tests for three HAL TUI fixes.

Bug 1 — Tools on by default: _init_tool_state must probe auth at startup.
Bug 2 — ANSI leaking: render_to_stdout must receive plain text per line.
Bug 3 — /mode provider/model: must route to model switch, not mode string.
"""
from __future__ import annotations

import os
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

_HAL_HOME = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

_DETECT = "lib.hal.tool_cli_integration.detect_tool_chat_fn"
_ANSI_ESC = "\033"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _FakeUsage:
    completion_tokens: int = 10


@dataclass
class _FakeResponse:
    text: str
    thinking: str = ""
    provider: str = "groq"
    model: str = "llama-3.3-70b"
    latency_ms: float = 100.0
    usage: _FakeUsage = field(default_factory=_FakeUsage)
    cost_usd: float = 0.0


def _fake_conv():
    conv = MagicMock()
    conv.cost = MagicMock()
    conv.cost.total_usd = 0.0
    conv.model = "groq/llama-3.3-70b"
    conv.status.return_value = {}  # prevents TypeError from ctx_pct > 20
    return conv


def _disabled_tool_state():
    from lib.hal.tool_circuit_breaker import CircuitBreaker
    return {"enabled": False, "chat_fn": None, "auth": None,
            "breaker": CircuitBreaker(threshold=2)}


# ---------------------------------------------------------------------------
# Bug 1: _init_tool_state auto-detects auth at boot
# ---------------------------------------------------------------------------

class TestInitToolStateAutoDetect:
    """Tools must start enabled when credentials are available."""

    def test_enabled_when_model_has_auth(self):
        from lib.hal.cli import _init_tool_state
        mock_fn = MagicMock(return_value={})
        with patch(_DETECT, return_value=(mock_fn, "groq")):
            ts = _init_tool_state("groq/llama-3.3-70b")
        assert ts["enabled"] is True
        assert ts["chat_fn"] is mock_fn
        assert ts["auth"] == "groq"

    def test_disabled_when_no_auth(self):
        from lib.hal.cli import _init_tool_state
        with patch(_DETECT, return_value=(None, None)):
            ts = _init_tool_state("groq/llama-3.3-70b")
        assert ts["enabled"] is False
        assert ts["chat_fn"] is None

    def test_disabled_when_no_model_provided(self):
        """Bare startup with no model arg must not crash and must default off."""
        from lib.hal.cli import _init_tool_state
        with patch(_DETECT, side_effect=AssertionError("must not probe without model")):
            ts = _init_tool_state()
        assert ts["enabled"] is False

    def test_detection_exception_does_not_crash_startup(self):
        """Keychain unavailable must not prevent HAL from starting."""
        from lib.hal.cli import _init_tool_state
        with patch(_DETECT, side_effect=RuntimeError("keychain locked")):
            ts = _init_tool_state("mlx_vlm/auto")
        assert ts["enabled"] is False

    def test_circuit_breaker_always_initialised(self):
        from lib.hal.cli import _init_tool_state
        with patch(_DETECT, return_value=(None, None)):
            ts = _init_tool_state("groq/llama")
        assert "breaker" in ts


# ---------------------------------------------------------------------------
# Bug 2: ANSI prefix written atomically, plain text passed to render_to_stdout
# ---------------------------------------------------------------------------

class TestNoAnsiInRenderedText:
    """render_to_stdout must receive plain text only.

    Embedding the ANSI │ prefix in the streamed text causes chunk-boundary
    splits inside escape sequences at chunk_size=20 chars.
    """

    def test_single_line_response_plain_text(self):
        from lib.hal import cli
        resp = _FakeResponse(text="Hello from HAL.")
        conv = _fake_conv()

        with patch.object(cli, "_STREAMER") as mock_streamer:
            cli._print_response(resp, conv)

        for c in mock_streamer.render_to_stdout.call_args_list:
            text = c.args[0] if c.args else c.kwargs.get("text", "")
            assert _ANSI_ESC not in text, (
                f"ANSI escape leaked into render_to_stdout: {text!r}. "
                "The │ prefix must be written atomically via sys.stdout.write."
            )

    def test_multi_line_response_all_calls_plain(self):
        """Each line of a wrapped response must arrive as a separate clean call."""
        from lib.hal import cli
        resp = _FakeResponse(text="A long response sentence. " * 6)
        conv = _fake_conv()

        with patch.object(cli, "_STREAMER") as mock_streamer:
            cli._print_response(resp, conv)

        call_count = mock_streamer.render_to_stdout.call_count
        assert call_count >= 2, (
            f"Multi-line text must produce multiple render_to_stdout calls, got {call_count}"
        )
        for c in mock_streamer.render_to_stdout.call_args_list:
            text = c.args[0] if c.args else c.kwargs.get("text", "")
            assert _ANSI_ESC not in text, (
                f"ANSI escape in call {c}: {text!r}"
            )

    def test_tool_response_plain_text(self):
        """_stream_plain_tool_response must also pass plain text per line."""
        from lib.hal import cli

        with patch.object(cli, "_STREAMER") as mock_streamer:
            cli._stream_plain_tool_response("Tool returned: answer is 42.")

        for c in mock_streamer.render_to_stdout.call_args_list:
            text = c.args[0] if c.args else c.kwargs.get("text", "")
            assert _ANSI_ESC not in text, (
                f"ANSI escape in tool stream call: {text!r}"
            )

    def test_response_content_still_reaches_output(self):
        """Fixing ANSI must not swallow the actual response text."""
        import io
        from lib.hal import cli

        resp = _FakeResponse(text="The answer is visible.")
        conv = _fake_conv()
        sink = io.StringIO()

        with patch("sys.stdout", sink):
            with patch.dict(os.environ, {"HAL_NO_STREAM": "1"}):
                cli._print_response(resp, conv)

        assert "The answer is visible." in sink.getvalue(), (
            "Response text must still appear in output after ANSI fix."
        )


# ---------------------------------------------------------------------------
# Bug 3: /mode provider/model must route to model switch + tool rebind
# ---------------------------------------------------------------------------

class TestModeSwitchWithProviderModel:
    """/mode mlx_vlm/auto must act like /model mlx_vlm/auto."""

    def _deps(self, model="claude-haiku-4-5"):
        conv = MagicMock()
        conv.model = model
        return conv, MagicMock(), MagicMock()

    def test_mode_provider_slash_model_sets_model(self):
        from lib.hal.cli import _handle_command
        conv, registry, council = self._deps()
        ts = _disabled_tool_state()

        buf = StringIO()
        with redirect_stdout(buf):
            with patch("lib.hal.cli._maybe_rebind_tools"):
                _handle_command("/mode mlx_vlm/auto", conv, registry, council, ts)

        conv.configure.assert_called_once_with(model="mlx_vlm/auto")

    def test_mode_provider_slash_model_does_not_set_mode(self):
        from lib.hal.cli import _handle_command
        conv, registry, council = self._deps()
        ts = _disabled_tool_state()

        buf = StringIO()
        with redirect_stdout(buf):
            with patch("lib.hal.cli._maybe_rebind_tools"):
                _handle_command("/mode ollama/llama3:8b", conv, registry, council, ts)

        calls = conv.configure.call_args_list
        assert len(calls) == 1
        assert "mode" not in calls[0].kwargs, (
            f"/mode ollama/llama3:8b must configure model=, not mode=. Calls: {calls}"
        )

    def test_mode_provider_slash_model_triggers_rebind(self):
        """/mode mlx_vlm/auto must rebind tools to the new provider."""
        from lib.hal.cli import _handle_command
        conv, registry, council = self._deps()
        ts = _disabled_tool_state()

        with patch("lib.hal.cli._maybe_rebind_tools") as mock_rebind:
            buf = StringIO()
            with redirect_stdout(buf):
                _handle_command("/mode groq/llama-3.3", conv, registry, council, ts)

        mock_rebind.assert_called_once_with("groq/llama-3.3", ts)

    def test_plain_mode_name_still_configures_mode(self):
        """/mode balanced must still act as a mode selector."""
        from lib.hal.cli import _handle_command
        conv, registry, council = self._deps()
        ts = _disabled_tool_state()

        buf = StringIO()
        with redirect_stdout(buf):
            _handle_command("/mode balanced", conv, registry, council, ts)

        conv.configure.assert_called_once_with(mode="balanced")

    def test_mode_output_label_is_model_for_provider_args(self):
        """/mode mlx_vlm/auto must print 'model →', not 'mode →'."""
        from lib.hal.cli import _handle_command
        conv, registry, council = self._deps()
        ts = _disabled_tool_state()

        buf = StringIO()
        with redirect_stdout(buf):
            with patch("lib.hal.cli._maybe_rebind_tools"):
                _handle_command("/mode mlx_vlm/auto", conv, registry, council, ts)

        output = buf.getvalue()
        assert "mode →" not in output, (
            f"Output must not say 'mode →' for a provider/model arg. Got: {output!r}"
        )
        assert "model" in output


if __name__ == "__main__":
    import unittest
    unittest.main()
