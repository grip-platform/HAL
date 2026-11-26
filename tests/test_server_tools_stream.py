"""Tests for the /api/tools/stream SSE endpoint (W10 integration).

Verifies the route is registered, the SSE headers are correct, and the
endpoint emits a well-formed SSE error when no auth is available.
"""
import io
import os
import sys
import unittest
from unittest import mock

_HAL_HOME = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)


class TestRouteRegistration(unittest.TestCase):
    def test_tools_stream_route_exists(self):
        """Regression anchor — drift detection for the route dict."""
        import lib.hal.server as server_module
        with open(server_module.__file__, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn('"/api/tools/stream"', content)
        self.assertIn("_api_tools_stream", content)

    def test_helper_methods_defined(self):
        import lib.hal.server as server_module
        with open(server_module.__file__, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("def _start_sse_stream(self)", content)
        self.assertIn("def _write_sse(self", content)
        self.assertIn("def _pipe_tool_events(self", content)


class TestSSEHeaderShape(unittest.TestCase):
    def test_sse_headers_declared(self):
        """text/event-stream, no-cache, X-Accel-Buffering: no."""
        import lib.hal.server as server_module
        with open(server_module.__file__, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn('"text/event-stream"', content)
        self.assertIn('"no-cache"', content)
        self.assertIn('"X-Accel-Buffering"', content)


class TestNoAuthFlow(unittest.TestCase):
    """When detect_tool_chat_fn returns (None, None), endpoint must
    surface an SSE error event, not 500 the browser."""

    def test_missing_auth_emits_sse_error(self):
        from lib.hal.server import _make_handler

        sessions = mock.MagicMock()
        prepared_session = mock.MagicMock()
        prepared_session.id = "sess-123"
        prepared_session.conversation.model = "claude-opus-4-7"

        with mock.patch("lib.hal.server._prepare_session", return_value=prepared_session):
            HandlerClass = _make_handler(
                registry=mock.MagicMock(),
                conversation=mock.MagicMock(),
                council=mock.MagicMock(),
                sessions=sessions,
                streamer=mock.MagicMock(),
            )
            handler = HandlerClass.__new__(HandlerClass)
            handler.wfile = io.BytesIO()
            handler.send_response = mock.MagicMock()
            handler.send_header = mock.MagicMock()
            handler.end_headers = mock.MagicMock()
            handler._cors_headers = mock.MagicMock()
            handler._read_body = mock.MagicMock(return_value={"message": "hi"})
            handler._json = mock.MagicMock()

            with mock.patch(
                "lib.hal.tool_cli_integration.detect_tool_chat_fn",
                return_value=(None, None),
            ):
                handler._api_tools_stream()

            written = handler.wfile.getvalue().decode("utf-8")
            self.assertIn("event: error", written)
            self.assertIn("no auth configured", written)

    def test_empty_message_returns_400(self):
        from lib.hal.server import _make_handler

        sessions = mock.MagicMock()
        HandlerClass = _make_handler(
            registry=mock.MagicMock(),
            conversation=mock.MagicMock(),
            council=mock.MagicMock(),
            sessions=sessions,
            streamer=mock.MagicMock(),
        )
        handler = HandlerClass.__new__(HandlerClass)
        handler._read_body = mock.MagicMock(return_value={"message": ""})
        handler._json = mock.MagicMock()

        handler._api_tools_stream()
        handler._json.assert_called_once()
        call_args = handler._json.call_args
        self.assertEqual(call_args.args[1], 400)


if __name__ == "__main__":
    unittest.main()
