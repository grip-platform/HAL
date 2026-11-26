"""Goodhart-proof regression for relay_response() BrokenPipe on header write.

Mutation sensitivity: removing the try/except around end_headers() causes both
tests to fail — the BrokenPipeError propagates instead of being swallowed.
"""
import sys
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

# The checkout this file is in, never a fixed path: a hardcoded ~/.hal
# here made a lone run of this file grade the operator's live tree rather than
# the code under change (tests-must-be-able-to-fail.md, shape 2).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.hal.gateway import UpstreamResponse, relay_response  # noqa: E402


def _make_resp(body: bytes = b"data") -> UpstreamResponse:
    reader = BytesIO(body)
    return UpstreamResponse(status=200, headers=[("Content-Type", "text/plain")], _reader=reader)


class _HandlerBase:
    """Minimal handler stub — just the attributes relay_response touches."""
    def send_response(self, code: int) -> None: ...
    def send_header(self, name: str, value: str) -> None: ...
    def end_headers(self) -> None: ...
    wfile: object


class TestRelayResponseBrokenPipeOnHeaders(unittest.TestCase):

    def test_broken_pipe_on_end_headers_does_not_propagate(self):
        """BrokenPipeError from end_headers() must be caught, not raised."""
        handler = MagicMock(spec=_HandlerBase)
        handler.end_headers.side_effect = BrokenPipeError("client gone")

        resp = _make_resp()
        close_mock = MagicMock()
        resp.close = close_mock

        # Must not raise — the fix swallows and returns
        try:
            relay_response(handler, resp)
        except BrokenPipeError:
            self.fail("relay_response() propagated BrokenPipeError from end_headers()")

        close_mock.assert_called_once()

    def test_connection_reset_on_end_headers_does_not_propagate(self):
        """ConnectionResetError from end_headers() must be caught, not raised."""
        handler = MagicMock(spec=_HandlerBase)
        handler.end_headers.side_effect = ConnectionResetError("reset by peer")

        resp = _make_resp()
        close_mock = MagicMock()
        resp.close = close_mock

        try:
            relay_response(handler, resp)
        except ConnectionResetError:
            self.fail("relay_response() propagated ConnectionResetError from end_headers()")

        close_mock.assert_called_once()

    def test_successful_relay_still_writes_body(self):
        """Mutation guard: normal path must still stream the body."""
        chunks_written = []

        handler = MagicMock()  # no spec — wfile needs dynamic attribute access
        handler.wfile.write.side_effect = lambda c: chunks_written.append(c)

        resp = _make_resp(b"hello world")
        relay_response(handler, resp)

        self.assertEqual(b"".join(chunks_written), b"hello world")


if __name__ == "__main__":
    unittest.main()
