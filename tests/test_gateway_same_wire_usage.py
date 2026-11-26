"""Same-wire chat replies carry their token counts onto the witness row.

The same-wire rail forwards a request verbatim to a non-Anthropic provider and
relays the reply. It used to witness BEFORE relaying, while the body was still
unread, so none of the 29,298 rows witnessed in the ten days to 2026-10-03
carried a token count -- among them 2,654 DeepSeek requests, which are billed in
cash. A spend report reading those rows saw nothing to add up.

Each property below fails under a named mutation:

  - counts reach the row, for a JSON reply and for a stream
        -> CountsReachTheRowTests (witness before the relay again: red)
  - the client still receives the reply byte for byte, in order, through the
    CANONICAL relay_response (never a hand-rolled copy of it, which would
    drift -- review on HAL#634, 2026-10-07), and a tap whose own reading
    breaks reports no usage rather than ever raising into that relay
        -> TheClientStillGetsTheReplyTests
  - latency still means time to response, not time to read the body
        -> test_latency_is_measured_when_the_upstream_answered
  - a relay that dies mid-stream is still witnessed, keeps the counts that
    arrived before the reset (AGORA on HAL#634), and ends in one in-band
    error rather than a bare disconnect (merged with main's relay_response
    fleet finding f72, 2026-10-07)
        -> WitnessTimingTests
  - Gemini still caches its thought signatures, from an unbounded copy, even
    past the JSON cap and after a reset
        -> GeminiCaptureTests
  - a stream is read as it passes, in bounded memory, whatever its length, and
    its usage is found however the chunks fall (AGORA on HAL#634)
        -> StreamMemoryTests
  - CRLF streams, Moonshot's and the Responses API's spellings, partial chunks,
    and nothing-read-means-absent-never-zero
        -> UsageReaderTests

Listed in the pytest (happi) CI job, which runs a fixed set of files.
"""
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from lib.hal import gateway_openai
from lib.hal import gemini_signature_cache as gsc
from lib.hal.cost_tracker import calculate_cost
from lib.hal.gateway import UpstreamResponse

_CHAT_PATH = "/v1/chat/completions"

#: A DeepSeek reply's usage block, cache-warm, in the provider's own spelling.
_USAGE = {"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500,
          "prompt_tokens_details": {"cached_tokens": 1000}}
_EXPECTED = {"input_tokens": 200, "output_tokens": 300, "cache_read_tokens": 1000}

_JSON_REPLY = json.dumps({
    "id": "chatcmpl-1", "object": "chat.completion", "model": "deepseek-v4-pro",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                 "finish_reason": "stop"}],
    "usage": _USAGE,
}).encode("utf-8")

_SIGNATURE = "c2lnbmF0dXJl"


def _sse(*objects, newline="\n") -> bytes:
    """An SSE body: one ``data:`` line per object, then the [DONE] sentinel."""
    blank = newline * 2
    lines = [f"data: {json.dumps(o)}{blank}" for o in objects]
    return ("".join(lines) + f"data: [DONE]{blank}").encode("utf-8")


def _delta(text, usage=None) -> dict:
    return {"id": "c", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            "usage": usage}


_STREAM_REPLY = _sse(_delta("hel"), _delta("lo"),
                     {"id": "c", "object": "chat.completion.chunk", "choices": [],
                      "usage": _USAGE})


class _Handler:
    """A BaseHTTPRequestHandler stand-in that records what the client received."""

    def __init__(self, wfile=None):
        self.path = _CHAT_PATH
        self.wfile = wfile or io.BytesIO()
        self.status = None

    def send_response(self, code):
        self.status = code

    def send_header(self, *_args):
        pass

    def end_headers(self):
        pass


class _DyingReader:
    """Returns one chunk, then fails the way a reset upstream socket does.

    Carries both ``read`` and ``read1`` -- a real upstream reader has both,
    and ``_TeeReader.read1`` is what the canonical relay actually calls
    (fleet finding f79: a stream reads whatever bytes are available, never
    waiting to fill a block)."""

    def __init__(self, first: bytes):
        self._first = first

    def read(self, _size=-1):
        if self._first is not None:
            first, self._first = self._first, None
            return first
        raise OSError("upstream reset")

    read1 = read


class _TwoPartReader:
    """Delivers a body in exactly two pieces -- a real second iter_chunks()
    read through the relay path, so a reader that only looks at the first
    chunk cannot hide behind a body small enough to arrive in one read.
    Carries both ``read`` and ``read1``, as a real upstream reader does."""

    def __init__(self, first: bytes, second: bytes):
        self._parts = [first, second]

    def read(self, _size=-1):
        return self._parts.pop(0) if self._parts else b""

    read1 = read


class _ClosedPipe(io.BytesIO):
    def write(self, _data):
        raise BrokenPipeError


def _upstream(body: bytes, status=200, reader=None) -> UpstreamResponse:
    kind = "text/event-stream" if body.startswith(b"data:") else "application/json"
    return UpstreamResponse(status=status, headers=[("Content-Type", kind)],
                            _reader=reader or io.BytesIO(body))


def _dying(first: bytes) -> UpstreamResponse:
    return _upstream(b"", reader=_DyingReader(first))


class _HalHome:
    """Point HAL_HOME at a fresh temp dir so the witness writes there."""

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = os.environ.get("HAL_HOME")
        os.environ["HAL_HOME"] = self._tmp.name
        return Path(self._tmp.name)

    def __exit__(self, *exc):
        if self._saved is None:
            os.environ.pop("HAL_HOME", None)
        else:
            os.environ["HAL_HOME"] = self._saved
        self._tmp.cleanup()


def _chat_rows():
    path = Path(os.environ["HAL_HOME"]) / "logs" / "gateway-audit.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if r.get("path") == _CHAT_PATH]


def _relay(upstream, *, model="deepseek-v4-pro", provider="deepseek",
           handler=None, started=None, raw=b'{"model": "deepseek-v4-pro"}'):
    """Drive the real same-wire relay with the upstream reply scripted."""
    handler = handler or _Handler()
    started = gateway_openai.time.monotonic() if started is None else started
    with mock.patch("lib.hal.gateway_egress.forward_same_wire", return_value=upstream):
        gateway_openai._relay_same_wire(handler, raw, provider, model, started)
    return handler


def _read(body: bytes, size: int = 0):
    """The counts the usage tap reads from ``body``, fed whole or in pieces."""
    tap = gateway_openai._UsageTap()
    step = size or max(len(body), 1)
    for start in range(0, len(body), step):
        tap.feed(body[start:start + step])
    return tap.usage()


def _picked(counts):
    return {k: counts[k] for k in _EXPECTED}


def _gemini_json_reply() -> bytes:
    return json.dumps({"choices": [{"message": {"role": "assistant", "tool_calls": [{
        "id": "call_g", "type": "function",
        "function": {"name": "get_time", "arguments": "{}"},
        "extra_content": {"google": {"thought_signature": _SIGNATURE}},
    }]}}], "usage": {"prompt_tokens": 40, "completion_tokens": 9}}).encode("utf-8")


def _relay_gemini(upstream):
    return _relay(upstream, model="gemini-2.5-flash", provider="gemini",
                  raw=b'{"model": "gemini-2.5-flash", "messages": []}')


class CountsReachTheRowTests(unittest.TestCase):

    def test_a_json_reply_puts_its_counts_and_cost_on_the_row(self):
        with _HalHome():
            _relay(_upstream(_JSON_REPLY))
            (row,) = _chat_rows()
        self.assertEqual(_picked(row), _EXPECTED)
        self.assertEqual(row["provider"], "deepseek")
        self.assertEqual(row["cost_usd"], round(calculate_cost(
            "deepseek-v4-pro", input_tokens=200, output_tokens=300,
            cache_read_tokens=1000, cache_creation_tokens=0), 8))
        self.assertGreater(row["cost_usd"], 0)

    def test_a_streamed_reply_puts_its_counts_on_the_row(self):
        with _HalHome():
            _relay(_upstream(_STREAM_REPLY))
            (row,) = _chat_rows()
        self.assertEqual(_picked(row), _EXPECTED)

    def test_a_reply_split_across_two_real_chunks_still_finds_its_usage(self):
        # _STREAM_REPLY is small enough to arrive as one iter_chunks() read in
        # every other test here, which would hide a reader that only looks at
        # the first chunk. A real second read, through the actual relay path.
        split = len(_STREAM_REPLY) // 2
        reader = _TwoPartReader(_STREAM_REPLY[:split], _STREAM_REPLY[split:])
        with _HalHome():
            _relay(_upstream(_STREAM_REPLY, reader=reader))
            (row,) = _chat_rows()
        self.assertEqual(_picked(row), _EXPECTED)

    def test_an_error_reply_is_witnessed_without_counts(self):
        error = b'{"error": {"message": "Insufficient Balance"}}'
        with _HalHome():
            handler = _relay(_upstream(error, status=402))
            (row,) = _chat_rows()
        self.assertEqual(row["status"], 402)
        self.assertNotIn("input_tokens", row)

    def test_a_429_is_witnessed_as_429_but_sent_to_the_client_as_503(self):
        # fleet finding f26 (main, merged 2026-10-07): Codex gives up on a
        # bare 429 even with Retry-After set; a 503 is retried. The witness
        # keeps the status the provider really sent either way.
        error = b'{"error": {"message": "rate limited"}}'
        with _HalHome():
            handler = _relay(_upstream(error, status=429))
            (row,) = _chat_rows()
        self.assertEqual(row["status"], 429)
        self.assertEqual(handler.status, 503)
        self.assertNotIn("cost_usd", row)
        self.assertEqual(handler.wfile.getvalue(), error)

    def test_a_stream_without_usage_omits_the_fields(self):
        # The client did not ask for stream_options.include_usage, so the
        # provider sent no counts. Unknown is recorded as absent, never as free.
        with _HalHome():
            _relay(_upstream(_sse(_delta("hel"), _delta("lo"))))
            (row,) = _chat_rows()
        self.assertEqual(row["status"], 200)
        self.assertNotIn("input_tokens", row)
        self.assertNotIn("cost_usd", row)


class TheClientStillGetsTheReplyTests(unittest.TestCase):

    def test_a_json_reply_reaches_the_client_unchanged(self):
        with _HalHome():
            handler = _relay(_upstream(_JSON_REPLY))
        self.assertEqual(handler.status, 200)
        self.assertEqual(handler.wfile.getvalue(), _JSON_REPLY)

    def test_a_streamed_reply_reaches_the_client_unchanged(self):
        with _HalHome():
            handler = _relay(_upstream(_STREAM_REPLY))
        self.assertEqual(handler.wfile.getvalue(), _STREAM_REPLY)

    def test_a_json_reply_past_the_cap_still_reaches_the_client_but_records_no_counts(self):
        with _HalHome():
            with mock.patch.object(gateway_openai, "_JSON_CAP", 64):
                handler = _relay(_upstream(_JSON_REPLY))
            (row,) = _chat_rows()
        self.assertEqual(handler.wfile.getvalue(), _JSON_REPLY)
        self.assertNotIn("input_tokens", row)

    def test_a_broken_tap_reports_no_usage_but_never_breaks_the_relay(self):
        # The _TeeReader contract: an observer's feed() never raises. If the
        # tap's own internal reading breaks, feed() must catch it itself --
        # relay_response never guards the observer it is handed -- and
        # report no usage from then on, never a wrong one.
        with _HalHome():
            with mock.patch.object(gateway_openai._UsageTap, "_take",
                                  side_effect=RuntimeError("a bug in the reader")):
                handler = _relay(_upstream(_JSON_REPLY))
            (row,) = _chat_rows()
        self.assertEqual(handler.wfile.getvalue(), _JSON_REPLY)
        self.assertNotIn("input_tokens", row)


class WitnessTimingTests(unittest.TestCase):

    def test_latency_is_measured_when_the_upstream_answered(self):
        # The upstream answers 250 ms after the request started. Reading the
        # body afterwards must not stretch the recorded latency.
        clock = SimpleNamespace(monotonic=lambda: 100.25)
        with _HalHome():
            with mock.patch.object(gateway_openai, "time", clock):
                _relay(_upstream(_JSON_REPLY), started=100.0)
            (row,) = _chat_rows()
        self.assertEqual(row["latency_ms"], 250)

    def test_a_relay_that_dies_mid_stream_is_still_witnessed(self):
        # relay_response never lets an upstream reset escape as a bare
        # exception (fleet finding f72): the OSError from _DyingReader is
        # caught inside it, not raised, and the witness in our own finally
        # still runs normally.
        with _HalHome():
            _relay(_dying(b'data: {"choices": []}\n\n'))
            (row,) = _chat_rows()
        self.assertEqual(row["status"], 200)
        self.assertNotIn("input_tokens", row)

    def test_a_dying_sse_reply_ends_with_an_in_band_error(self):
        # Content-Type decides the dialect, exactly as relay_response reads it
        # (AGORA on HAL#634): an SSE reply that dies mid-body gets one in-band
        # error event, not a bare disconnect -- in the CHAT-COMPLETIONS dialect
        # specifically (relay_response defaults to Anthropic's otherwise,
        # which also says "event: error" -- "param" is the OpenAI-only key).
        first = b'data: {"choices": []}\n\n'
        with _HalHome():
            handler = _relay(_upstream(first, reader=_DyingReader(first)))
        body = handler.wfile.getvalue()
        self.assertIn(b"event: error", body)
        self.assertIn(b'"param"', body)

    def test_counts_that_arrived_before_a_reset_are_kept(self):
        before_done = _STREAM_REPLY.split(b"data: [DONE]")[0]
        with _HalHome():
            _relay(_dying(before_done))
            (row,) = _chat_rows()
        self.assertEqual(_picked(row), _EXPECTED)

    def test_a_client_that_leaves_mid_stream_is_still_witnessed(self):
        with _HalHome():
            _relay(_upstream(_STREAM_REPLY), handler=_Handler(wfile=_ClosedPipe()))
            rows = _chat_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], 200)


class GeminiCaptureTests(unittest.TestCase):

    def setUp(self):
        gsc._cache.clear()
        self.addCleanup(gsc._cache.clear)

    def test_gemini_signatures_are_still_captured(self):
        with _HalHome():
            _relay_gemini(_upstream(_gemini_json_reply()))
            (row,) = _chat_rows()
        self.assertEqual(gsc._cache.get("call_g"), _SIGNATURE)
        self.assertEqual((row["input_tokens"], row["output_tokens"]), (40, 9))

    def test_a_broken_whole_body_capture_reports_no_signature(self):
        # Same guard as _UsageTap, on the Gemini side: feed() never raises,
        # and once broken it reports nothing rather than a signature read
        # from a partial copy.
        whole = gateway_openai._WholeBodyCapture()
        whole._data = SimpleNamespace(extend=mock.Mock(side_effect=RuntimeError("boom")))
        whole.feed(b"some bytes")
        self.assertEqual(whole.data(), b"")

    def test_signatures_are_captured_from_a_body_past_the_json_cap(self):
        # Gemini's copy is unbounded, as it was before this change; only the
        # usage reader stops holding a JSON reply at the cap.
        with _HalHome():
            with mock.patch.object(gateway_openai, "_JSON_CAP", 64):
                _relay_gemini(_upstream(_gemini_json_reply()))
            (row,) = _chat_rows()
        self.assertEqual(gsc._cache.get("call_g"), _SIGNATURE)
        self.assertNotIn("input_tokens", row)

    def test_signatures_that_arrived_before_a_reset_are_captured(self):
        delta = {"choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [{
            "index": 0, "id": "call_s", "type": "function",
            "function": {"name": "get_time", "arguments": ""},
            "extra_content": {"google": {"thought_signature": _SIGNATURE}}}]}}]}
        with _HalHome():
            _relay_gemini(_dying(f"data: {json.dumps(delta)}\n\n".encode("utf-8")))
        self.assertEqual(gsc._cache.get("call_s"), _SIGNATURE)


class StreamMemoryTests(unittest.TestCase):

    def test_a_stream_fed_in_small_pieces_reads_the_same_counts(self):
        for size in (1, 7, 64):
            with self.subTest(size=size):
                self.assertEqual(_picked(_read(_STREAM_REPLY, size)), _EXPECTED)

    def test_a_long_stream_keeps_its_trailing_usage_in_bounded_memory(self):
        # Far past the JSON cap, which never applies to a stream: the tap holds
        # one unfinished line at a time, never the stream.
        body = _sse(*[_delta("x" * 40) for _ in range(3000)], {"choices": [], "usage": _USAGE})
        tap, most = gateway_openai._UsageTap(), 0
        with mock.patch.object(gateway_openai, "_JSON_CAP", 1024):
            for start in range(0, len(body), 4096):
                tap.feed(body[start:start + 4096])
                most = max(most, len(tap._pending))
        self.assertGreater(len(body), 100 * 1024)
        self.assertLess(most, 1024)
        self.assertEqual(_picked(tap.usage()), _EXPECTED)

    def test_an_overlong_line_is_dropped_without_losing_the_next_count(self):
        tap = gateway_openai._UsageTap()
        with mock.patch.object(gateway_openai, "_LINE_CAP", 64):
            tap.feed(b"data: " + b"y" * 500)
            self.assertLessEqual(len(tap._pending), 64)
            tap.feed(b"\n\n" + _sse({"choices": [], "usage": _USAGE}))
        self.assertEqual(_picked(tap.usage()), _EXPECTED)


class UsageReaderTests(unittest.TestCase):

    def test_a_crlf_stream_is_read_in_full(self):
        # A blank-line block parser sees a CRLF stream as ONE block and reads
        # only its first chunk, which carries no usage.
        body = _sse(_delta("hel"), {"choices": [], "usage": _USAGE}, newline="\r\n")
        self.assertEqual(_picked(_read(body)), _EXPECTED)

    def test_usage_on_the_first_choice_is_read(self):
        # Moonshot's spelling.
        body = _sse({"choices": [{"index": 0, "delta": {},
                                  "usage": {"prompt_tokens": 7, "completion_tokens": 3}}]})
        counts = _read(body)
        self.assertEqual((counts["input_tokens"], counts["output_tokens"]), (7, 3))

    def test_the_responses_completed_event_is_read(self):
        body = _sse({"type": "response.created", "response": {"usage": None}},
                    {"type": "response.completed", "response": {"usage": {
                        "input_tokens": 50, "output_tokens": 20,
                        "input_tokens_details": {"cached_tokens": 30}}}})
        counts = _read(body)
        self.assertEqual((counts["input_tokens"], counts["cache_read_tokens"],
                          counts["output_tokens"]), (20, 30, 20))

    def test_a_partial_chunk_keeps_the_earlier_input_count(self):
        body = _sse({"choices": [], "usage": {"prompt_tokens": 90}},
                    {"choices": [], "usage": {"completion_tokens": 12}})
        counts = _read(body)
        self.assertEqual((counts["input_tokens"], counts["output_tokens"]), (90, 12))

    def test_a_later_count_replaces_an_earlier_one(self):
        # Cumulative reporting: the last count is the total.
        body = _sse({"choices": [], "usage": {"prompt_tokens": 90, "completion_tokens": 1}},
                    {"choices": [], "usage": {"prompt_tokens": 90, "completion_tokens": 12}})
        self.assertEqual(_read(body)["output_tokens"], 12)

    def test_no_usage_reads_as_none(self):
        for body in (b"", b"   ", b"data: [DONE]\n\n", _sse(_delta("hi")),
                     b'{"choices": []}', b"not a reply", b"\xff\xfe\x00garbage",
                     b'{"choices": [], "usage": {}}',
                     b'{"usage": {"prompt_tokens": 0, "completion_tokens": 0}}',
                     b'{"usage": "1200"}', b'{"truncated": '):
            with self.subTest(body=body):
                self.assertIsNone(_read(body))


if __name__ == "__main__":
    unittest.main()
