"""A request the server refuses before reading its body still has that body read.

Closing a socket that still holds unread request bytes makes Linux send a
reset, so the client sees "connection reset" instead of the refusal it was
sent. That bites whenever a body is still in the kernel when the refusal goes
out: a large body, or one the client sends after its headers, as Python's
http.client does. Each test drives one early-refusal path with a body waiting
in ``rfile`` and asserts the whole body was consumed. That holds on every OS,
so these tests fail on the old code wherever they run, not only on Linux.
"""
import http.client
import io
import os
import unittest
from unittest.mock import MagicMock, patch

from lib.hal import api_throttle, http_refusal, server
from lib.hal.api_throttle import ChatThrottle

BODY = b'{"message": "hi", "session_id": "s-1", "model": "test-model"}'
TAILNET_PEER = "100.64.0.7"  # a non-loopback address; no ACL tag is configured


def _headers(length):
    raw = f"Content-Length: {length}\r\nContent-Type: application/json\r\n\r\n"
    return http.client.parse_headers(io.BytesIO(raw.encode()))


def _handler(path, peer="127.0.0.1", length=len(BODY), method="POST"):
    """The real Handler class, with in-memory streams instead of a socket."""
    handler_cls = server._make_handler(
        MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
    h = handler_cls.__new__(handler_cls)
    h.path, h.command = path, method
    h.request_version = "HTTP/1.0"
    h.requestline = f"{method} {path} HTTP/1.0"
    h.headers = _headers(length)
    h.rfile, h.wfile = io.BytesIO(BODY), io.BytesIO()
    h.client_address = (peer, 40000)
    h.close_connection = True
    h.log_message = lambda *args, **kwargs: None
    return h


def _status(h):
    return int(h.wfile.getvalue().split(b" ", 2)[1])


def _reader_raising(exc):
    """An rfile whose every read fails with ``exc``."""

    class _Failing(io.RawIOBase):
        def readable(self):
            return True

        def readinto(self, buffer):
            raise exc

    return io.BufferedReader(_Failing())


class TestEarlyRefusalReadsTheBody(unittest.TestCase):

    def setUp(self):
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for name in ("HAL_GATEWAY_TS_ACL_TAG", "HAL_GATEWAY_AUTH_TOKEN"):
            os.environ.pop(name, None)

    def assert_refused_and_drained(self, h, code):
        getattr(h, f"do_{h.command}")()
        self.assertEqual(_status(h), code)
        self.assertEqual(h.rfile.tell(), len(BODY),
                         "the refused request's body was left unread")

    def test_post_unknown_route_404(self):
        self.assert_refused_and_drained(_handler("/api/no-such-route"), 404)

    def test_post_remote_peer_on_a_private_route_403(self):
        self.assert_refused_and_drained(
            _handler("/api/tools/execute", peer=TAILNET_PEER), 403)

    def test_post_remote_peer_on_v1_403(self):
        self.assert_refused_and_drained(
            _handler("/v1/messages", peer=TAILNET_PEER), 403)

    def test_post_throttled_chat_429(self):
        spent = ChatThrottle(per_min=10_000, daily_budget=0.0)
        with patch.object(api_throttle, "_INSTANCE", spent):
            self.assert_refused_and_drained(_handler("/api/chat"), 429)

    def test_get_remote_peer_on_a_private_route_403(self):
        self.assert_refused_and_drained(
            _handler("/api/sessions", peer=TAILNET_PEER, method="GET"), 403)

    def test_get_remote_peer_on_v1_403(self):
        self.assert_refused_and_drained(
            _handler("/v1/models", peer=TAILNET_PEER, method="GET"), 403)

    def test_get_unknown_route_404(self):
        # An /api/ path, so a built web app's index-page fallback cannot answer it.
        self.assert_refused_and_drained(
            _handler("/api/no-such-route", method="GET"), 404)

    def test_delete_remote_peer_403(self):
        self.assert_refused_and_drained(
            _handler("/api/sessions/s-1", peer=TAILNET_PEER, method="DELETE"), 403)

    def test_delete_unknown_route_404(self):
        self.assert_refused_and_drained(
            _handler("/api/no-such-route", method="DELETE"), 404)

    def test_an_oversized_body_is_left_alone(self):
        h = _handler("/api/no-such-route", length=http_refusal._DISCARD_CAP + 1)
        h.do_POST()
        self.assertEqual(_status(h), 404)
        self.assertEqual(h.rfile.tell(), 0)

    def test_a_client_that_went_away_or_held_its_body_back_is_absorbed(self):
        for exc in (TimeoutError("timed out"), ConnectionResetError(104, "reset")):
            with self.subTest(exc=type(exc).__name__):
                h = _handler("/api/no-such-route")
                h.rfile = _reader_raising(exc)
                h.connection = MagicMock()
                h.do_POST()
                self.assertEqual(_status(h), 404)
                h.connection.settimeout.assert_called_once_with(http_refusal._DISCARD_WAIT_S)

    def test_an_unexpected_read_error_is_not_swallowed(self):
        h = _handler("/api/no-such-route")
        h.rfile = _reader_raising(OSError(9, "Bad file descriptor"))
        with self.assertRaises(OSError):
            h.do_POST()


if __name__ == "__main__":
    unittest.main()
