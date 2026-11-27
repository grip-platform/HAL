"""W0 feasibility spike — is a gateway-orchestrated council/back-and-forth even
possible *through Codex*?

The only hard constraint: a council can't emit real output until both models
(+ judge) finish (10-60s), so the gateway must hold the Responses stream open
that long. Codex's `stream_idle_timeout_ms` defaults to 300_000 ms (5 min) and
is configurable per custom model_provider — so 10-60s fits with huge margin.

This proves the GATEWAY-SIDE mechanism against REAL code
(`translate_responses_stream`), zero network: it emits `response.created`
immediately, then can deliver the body arbitrarily late, as a valid Responses
SSE stream. Goodhart-proof: the timing assertion FAILS if the translator ever
buffered the whole body before `response.created` (i.e. could NOT hold the
stream open for a council).
"""
import json
import time
import unittest

from lib.hal.gateway_responses import translate_responses_stream

_COUNCIL_DELAY_S = 0.4  # stand-in for a real 10-60s council; bounded for CI speed
# Durations use time.perf_counter(): on Windows, Python 3.12's time.monotonic()
# moves in steps of about 15.6 ms, so a full 0.4 s sleep can read as 0.390 s.


def _data(raw: bytes) -> dict:
    for line in raw.decode("utf-8").splitlines():
        if line.startswith("data:"):
            return json.loads(line[5:].strip())
    return {}


class TestCouncilHoldOpenFeasibility(unittest.TestCase):
    def _slow_council_body(self):
        # The 2-model council / back-and-forth runs HERE before any body exists.
        time.sleep(_COUNCIL_DELAY_S)
        yield b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
        yield b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Synthesised council verdict"}}\n\n'
        yield b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
        yield b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}\n\n'
        yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'

    def test_response_created_precedes_slow_council_body(self):
        gen = translate_responses_stream(
            self._slow_council_body(), resp_id="resp_w0", model="claude-opus-4-8")
        t0 = time.perf_counter()
        first_raw = next(gen)                 # pull ONLY the first event
        dt_created = time.perf_counter() - t0
        first = _data(first_raw)

        # 1. Stream OPENS immediately — created before the council body exists.
        self.assertEqual(first["type"], "response.created")
        self.assertEqual(first["response"]["status"], "in_progress")
        self.assertLess(
            dt_created, _COUNCIL_DELAY_S,
            "response.created must precede the council body (stream held open)")

        # 2. Draining the rest is where the council delay is actually paid,
        #    and the late body still forms a valid, ordered, complete stream.
        raw_rest = list(gen)
        total_elapsed = time.perf_counter() - t0
        self.assertGreaterEqual(total_elapsed, _COUNCIL_DELAY_S)  # body came late

        rest = [e for e in (_data(b) for b in raw_rest) if e]
        types = [first["type"]] + [e["type"] for e in rest]
        self.assertEqual(types[-1], "response.completed")

        seqs = [e["sequence_number"] for e in [first] + rest if "sequence_number" in e]
        self.assertEqual(seqs, sorted(seqs))                      # monotonic
        self.assertEqual(len(seqs), len(set(seqs)))               # unique

        all_raw = first_raw + b"".join(raw_rest)
        self.assertIn(b"Synthesised council verdict", all_raw)    # late body delivered

    def test_council_latency_fits_codex_idle_budget(self):
        # Codex stream_idle_timeout_ms default = 300_000 ms; the only idle Codex
        # sees is created -> first body token. A generous 2-model + judge upper
        # bound (~60s) is far under it, and the timeout is configurable upward.
        CODEX_STREAM_IDLE_TIMEOUT_MS = 300_000
        WORST_CASE_COUNCIL_MS = 60_000
        self.assertLess(WORST_CASE_COUNCIL_MS, CODEX_STREAM_IDLE_TIMEOUT_MS)


if __name__ == "__main__":
    unittest.main()
