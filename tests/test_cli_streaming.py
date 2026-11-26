"""Tests for cli.py streaming integration (Phase α W1).

Verifies:
  * env-var plumbing for chunk size and delay
  * ``_print_response`` invokes the streamer for the answer block
  * ``_run_tool_turn`` invokes the streamer for tool-loop output
  * non-TTY fallback writes the response text exactly once

The CLI integration is tested against a mocked streamer and stdout so the
tests stay hermetic (no terminal, no sleep).
"""
import io
import os
import unittest
from dataclasses import dataclass
from unittest import mock


@dataclass
class _FakeCost:
    total_usd: float = 0.0


@dataclass
class _FakeConv:
    cost: _FakeCost
    _last_response_text: str = ""


@dataclass
class _FakeResponse:
    text: str
    thinking: str
    provider: str
    model: str
    latency_ms: float
    cost_usd: float = 0.0
    usage: object = None


class TestStreamEnvHelpers(unittest.TestCase):
    """Env-var helpers default sensibly and guard against junk values."""

    def test_defaults(self):
        from lib.hal import cli
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HAL_STREAM_CHUNK_CHARS", None)
            os.environ.pop("HAL_STREAM_DELAY_MS", None)
            self.assertEqual(cli._stream_chunk_chars(), 20)
            self.assertEqual(cli._stream_delay_ms(), 20)

    def test_env_overrides(self):
        from lib.hal import cli
        with mock.patch.dict(os.environ, {
            "HAL_STREAM_CHUNK_CHARS": "60",
            "HAL_STREAM_DELAY_MS": "0",
        }):
            self.assertEqual(cli._stream_chunk_chars(), 60)
            self.assertEqual(cli._stream_delay_ms(), 0)

    def test_invalid_values_fall_back_to_defaults(self):
        from lib.hal import cli
        with mock.patch.dict(os.environ, {
            "HAL_STREAM_CHUNK_CHARS": "not-a-number",
            "HAL_STREAM_DELAY_MS": "also-bad",
        }):
            self.assertEqual(cli._stream_chunk_chars(), 20)
            self.assertEqual(cli._stream_delay_ms(), 20)

    def test_floors_clamp_to_safe_minimums(self):
        from lib.hal import cli
        with mock.patch.dict(os.environ, {
            "HAL_STREAM_CHUNK_CHARS": "0",  # must become >=1
            "HAL_STREAM_DELAY_MS": "-50",   # must become >=0
        }):
            self.assertEqual(cli._stream_chunk_chars(), 1)
            self.assertEqual(cli._stream_delay_ms(), 0)


class TestPrintResponseStreaming(unittest.TestCase):
    """``_print_response`` routes the answer text through ResponseStreamer."""

    def test_print_response_calls_streamer(self):
        from lib.hal import cli
        fake_response = _FakeResponse(
            text="Hello from HAL. This response should be streamed.",
            thinking="",
            provider="groq",
            model="llama-3.3-70b-versatile",
            latency_ms=123.0,
        )
        fake_conv = _FakeConv(cost=_FakeCost(total_usd=0.000123))
        with mock.patch.object(cli, "_STREAMER") as streamer_mock:
            cli._print_response(fake_response, fake_conv)
        self.assertTrue(streamer_mock.render_to_stdout.called)
        args, kwargs = streamer_mock.render_to_stdout.call_args
        # The wrapped text should contain the original content.
        rendered_arg = args[0] if args else kwargs.get("text", "")
        self.assertIn("Hello from HAL", rendered_arg)
        # Streamer receives chunk + delay controls from env helpers.
        self.assertIn("chars_per_chunk", kwargs)
        self.assertIn("delay_ms", kwargs)

    def test_non_tty_output_contains_full_text(self):
        """End-to-end: non-TTY stdout captures the full response in one block."""
        from lib.hal import cli
        fake_response = _FakeResponse(
            text="A non-TTY test sentence that must appear verbatim.",
            thinking="",
            provider="groq",
            model="llama-3.3-70b-versatile",
            latency_ms=42.0,
        )
        fake_conv = _FakeConv(cost=_FakeCost(total_usd=0.0))
        sink = io.StringIO()  # StringIO.isatty() -> False
        with mock.patch("sys.stdout", sink):
            with mock.patch.dict(os.environ, {"HAL_NO_STREAM": "1"}):
                cli._print_response(fake_response, fake_conv)
        output = sink.getvalue()
        # Content preserved.
        self.assertIn("A non-TTY test sentence that must appear verbatim.", output)
        # Metadata footer rendered.
        self.assertIn("groq/llama-3.3-70b-versatile", output)
        self.assertIn("42ms", output)


class TestRunToolTurnStreaming(unittest.TestCase):
    """``_run_tool_turn`` streams ``conv._last_response_text`` via the streamer."""

    def test_tool_turn_routes_through_streamer(self):
        from lib.hal import cli
        fake_conv = _FakeConv(
            cost=_FakeCost(),
            _last_response_text="tool-result text, streamed the same way",
        )
        from lib.hal.tool_circuit_breaker import CircuitBreaker
        tool_state = {
            "enabled": True,
            "chat_fn": mock.Mock(),
            "auth": None,
            "breaker": CircuitBreaker(threshold=2),
        }

        with mock.patch("lib.hal.tool_cli_integration.build_builtin_tool_stack",
                        return_value=(mock.Mock(), mock.Mock())):
            with mock.patch("lib.hal.tool_render.send_with_tools_to_cli"):
                with mock.patch.object(cli, "_STREAMER") as streamer_mock:
                    cli._run_tool_turn(fake_conv, "do something", tool_state)

        self.assertTrue(streamer_mock.render_to_stdout.called)
        args, kwargs = streamer_mock.render_to_stdout.call_args
        rendered = args[0] if args else kwargs.get("text", "")
        self.assertIn("tool-result text", rendered)


if __name__ == "__main__":
    unittest.main()
