"""Cross-wire usage accounting — tokens and cost on the witness row.

Before this, a cross-wire witness row proved a request was SERVED and said
nothing about what it cost, so a Claude Code session on a non-Anthropic model
could not be priced at all. These tests pin the properties that make the row
trustworthy, and each one fails under a named mutation:

  - normalised counts lose the chat/Responses spelling difference
        -> ShapeTests
  - ``input_tokens`` is FRESH input, cached portion subtracted
        -> CacheAccountingTests
           WitnessRowUsageTests.test_cost_prices_fresh_input_not_the_raw_total
  - a row with nothing parsed OMITS the fields rather than reporting zero
        -> WitnessRowUsageTests.test_row_omits_usage_fields_when_nothing_parsed
  - the counts survive the trip from provider body to witness, on BOTH the
    buffered-JSON and the buffered-SSE (streaming client) replies
        -> BufferedReplyCarriesUsageTests

The subtraction is the load-bearing one and it is not cosmetic. A provider's
``prompt_tokens`` INCLUDES its ``cached_tokens``, while ``calculate_cost``
prices cached reads separately and ADDS them, so passing the raw total bills
the cached portion twice — once at the full input rate. The anchor test uses a
row measured live on 2026-09-10 through a private gateway, where that error is
worth 10.67x on a single turn.
"""

import io
import json
import unittest

from lib.hal.cost_tracker import calculate_cost
from lib.hal.gateway import UpstreamResponse
from lib.hal.gateway_crosswire import normalise_provider_usage
from lib.hal.gateway_egress import _buffered_anthropic_reply
from lib.hal.gateway_witness import build_chat_record

#: One real cache-warm turn from the 2026-09-10 acceptance run
#: (claude -p, Read tool, openai/gpt-6-astra, status 200). Kept verbatim so the
#: anchor is a measurement rather than an invented example.
_MEASURED_WARM_TURN = {"fresh": 114, "cached": 176856, "output": 97}


def _chat_usage(prompt, completion, cached=0):
    """A provider usage block in /v1/chat/completions spelling."""
    return {"usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                      "prompt_tokens_details": {"cached_tokens": cached}}}


def _responses_usage(inp, out, cached=0, reasoning=0):
    """A provider usage block in /v1/responses spelling."""
    return {"usage": {"input_tokens": inp, "output_tokens": out,
                      "input_tokens_details": {"cached_tokens": cached},
                      "output_tokens_details": {"reasoning_tokens": reasoning}}}


class SpellingPrecedenceTests(unittest.TestCase):
    """Which spelling wins is decided by PRESENCE, never by truthiness.

    ``a or b`` reads a legitimate zero as absent and falls through to the
    other spelling. A council speaker caught it: a body carrying both
    ``prompt_tokens: 0`` and ``input_tokens: 500`` was billed 500.
    """

    def test_a_present_zero_beats_the_other_spelling(self):
        got = normalise_provider_usage({"usage": {
            "prompt_tokens": 0, "input_tokens": 500,
            "completion_tokens": 0, "output_tokens": 42}})
        self.assertEqual(got["input_tokens"], 0, "prompt_tokens was present and said zero")
        self.assertEqual(got["output_tokens"], 0, "completion_tokens was present and said zero")

    def test_a_present_zero_detail_beats_the_other_detail_block(self):
        got = normalise_provider_usage({"usage": {
            "prompt_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 0},
            "input_tokens_details": {"cached_tokens": 90}}})
        self.assertEqual(got["cache_read_tokens"], 0)
        self.assertEqual(got["input_tokens"], 100, "nothing was cached, so nothing is subtracted")

    def test_absent_spelling_still_falls_through(self):
        """The fallback must still work when the first spelling is truly absent."""
        got = normalise_provider_usage({"usage": {"input_tokens": 500, "output_tokens": 42}})
        self.assertEqual(got["input_tokens"], 500)
        self.assertEqual(got["output_tokens"], 42)


class ShapeTests(unittest.TestCase):
    """One helper for both wires — the two shapes differ only in spelling."""

    def test_chat_shape_maps_prompt_and_completion(self):
        got = normalise_provider_usage(_chat_usage(120, 45))
        self.assertEqual(got["input_tokens"], 120)
        self.assertEqual(got["output_tokens"], 45)

    def test_responses_shape_maps_input_and_output(self):
        got = normalise_provider_usage(_responses_usage(120, 45))
        self.assertEqual(got["input_tokens"], 120)
        self.assertEqual(got["output_tokens"], 45)

    def test_responses_shape_carries_reasoning_tokens(self):
        got = normalise_provider_usage(_responses_usage(10, 90, reasoning=76))
        self.assertEqual(got["reasoning_tokens"], 76)


class CacheAccountingTests(unittest.TestCase):
    """``prompt_tokens`` includes the cached portion; the normalised input
    must not, or the cached tokens are billed twice."""

    def test_cached_tokens_are_subtracted_from_input(self):
        got = normalise_provider_usage(_chat_usage(5737, 12, cached=5734))
        self.assertEqual(got["input_tokens"], 3, "input must be FRESH tokens only")
        self.assertEqual(got["cache_read_tokens"], 5734)

    def test_cold_turn_reports_all_input_as_fresh(self):
        got = normalise_provider_usage(_chat_usage(5737, 12, cached=0))
        self.assertEqual(got["input_tokens"], 5737)
        self.assertEqual(got["cache_read_tokens"], 0)

    def test_cached_exceeding_total_clamps_at_zero(self):
        """A provider whose details exceed its total must not go negative."""
        got = normalise_provider_usage(_chat_usage(100, 5, cached=999))
        self.assertEqual(got["input_tokens"], 0)


class MalformedUsageTests(unittest.TestCase):
    """Absent usage is reported as absent, never guessed."""

    def test_missing_usage_returns_none_not_a_dict_of_zeros(self):
        """Inverted from asserting zeros, which encoded the defect.

        An all-zero dict is TRUTHY, so returning one made every downstream
        ``if usage:`` fire and stamped ``cost_usd: 0.0`` on an unmeasured row.
        A council speaker caught it; the original test had pinned the wrong
        contract, so its expectation is corrected rather than the test deleted.
        """
        for body in ({}, {"usage": None}, {"usage": "nope"}, {"id": "x"}):
            self.assertIsNone(normalise_provider_usage(body),
                              f"{body!r} carries no usage; None is the honest answer")

    def test_a_parsed_body_without_usage_never_reports_a_cost(self):
        """The end-to-end shape of the same defect: valid JSON, no usage."""
        row = build_chat_record(
            provider="openai", model="gpt-6-astra", status=200, served=True,
            failed_over=True, latency_ms=1, path="/v1/messages",
            usage=normalise_provider_usage({"id": "r1", "choices": []}))
        self.assertNotIn("cost_usd", row, "unmeasured must not report as free")
        self.assertNotIn("input_tokens", row)

    def test_bool_is_not_a_token_count(self):
        """``isinstance(True, int)`` is True in Python — the guard matters."""
        got = normalise_provider_usage({"usage": {"prompt_tokens": True,
                                                  "completion_tokens": 7}})
        self.assertEqual(got["input_tokens"], 0)
        self.assertEqual(got["output_tokens"], 7)


class WitnessRowUsageTests(unittest.TestCase):
    """The row is the only place a cross-wire session's cost is visible."""

    def _row(self, usage, model="gpt-6-astra"):
        return build_chat_record(
            provider="openai", model=model, status=200, served=True,
            failed_over=True, latency_ms=1200, path="/v1/messages", usage=usage)

    def test_row_carries_tokens_and_cost_when_usage_parsed(self):
        row = self._row(normalise_provider_usage(_chat_usage(120, 45)))
        self.assertEqual(row["input_tokens"], 120)
        self.assertEqual(row["output_tokens"], 45)
        self.assertGreater(row["cost_usd"], 0.0)

    def test_row_omits_usage_fields_when_nothing_parsed(self):
        """A verbatim relay parses no body. Absent must not read as free."""
        row = self._row(None)
        for field in ("input_tokens", "output_tokens", "cost_usd"):
            self.assertNotIn(field, row, "no measurement must not look like zero cost")

    def test_cost_prices_fresh_input_not_the_raw_total(self):
        """The anchor. Measured live 2026-09-10 — the error is worth 10.67x.

        Charging the provider's raw ``prompt_tokens`` as input double-bills the
        cached portion, because ``calculate_cost`` already adds cached reads at
        their own rate.
        """
        m = _MEASURED_WARM_TURN
        row = self._row(normalise_provider_usage(
            _chat_usage(m["fresh"] + m["cached"], m["output"], cached=m["cached"])))
        raw_total_cost = calculate_cost(
            "gpt-6-astra", input_tokens=m["fresh"] + m["cached"],
            output_tokens=m["output"], cache_read_tokens=m["cached"])
        self.assertAlmostEqual(row["cost_usd"], 0.182846, places=5)
        self.assertGreater(raw_total_cost / row["cost_usd"], 10.0,
                           "the defect this guards is ~10x, so the gap must be large")

    def test_unknown_model_still_produces_a_row(self):
        """Pricing is fail-open: an unpriced model must not lose the tokens."""
        row = self._row(normalise_provider_usage(_chat_usage(10, 5)),
                        model="some/model-nobody-priced")
        self.assertEqual(row["input_tokens"], 10)
        self.assertIn("cost_usd", row)


class BufferedReplyCarriesUsageTests(unittest.TestCase):
    """The witness runs BEFORE the relay over a single-pass reader, so the
    counts must ride on the response object rather than be re-read."""

    @staticmethod
    def _provider_response(body: bytes) -> UpstreamResponse:
        return UpstreamResponse(
            status=200,
            headers=[("Content-Type", "application/json")],
            _reader=io.BytesIO(body),
        )

    @staticmethod
    def _translate(data, *, model=""):
        return {"id": "msg_1", "type": "message", "role": "assistant",
                "model": model, "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn"}

    def _reply(self, *, want_stream):
        body = json.dumps({"id": "r1", **_chat_usage(300, 20, cached=250)}).encode()
        return _buffered_anthropic_reply(
            self._provider_response(body), translate=self._translate,
            model="gpt-6-astra", want_stream=want_stream)

    def test_buffered_json_reply_carries_usage_forward(self):
        out = self._reply(want_stream=False)
        self.assertIsNotNone(out.usage)
        self.assertEqual(out.usage["input_tokens"], 50)   # 300 total - 250 cached
        self.assertEqual(out.usage["cache_read_tokens"], 250)

    def test_buffered_sse_burst_carries_usage_forward(self):
        """The streaming client's path. Buffered, so usage is equally in hand —
        this is the half of gap 4 that a streamed request used to lose."""
        out = self._reply(want_stream=True)
        self.assertIsNotNone(out.usage)
        self.assertEqual(out.usage["input_tokens"], 50)
        self.assertEqual(out.usage["output_tokens"], 20)

    def test_reply_body_is_still_readable_after_usage_is_attached(self):
        """Attaching usage must not consume the body the client is owed."""
        out = self._reply(want_stream=False)
        self.assertIn(b"msg_1", out.read_all())

    def test_non_json_provider_body_yields_no_usage(self):
        out = _buffered_anthropic_reply(
            self._provider_response(b"<html>502</html>"), translate=self._translate,
            model="gpt-6-astra", want_stream=False)
        self.assertEqual(out.status, 502)
        self.assertIsNone(out.usage, "a body we could not parse has no counts to report")


class TrueStreamUsageTests(unittest.TestCase):
    """The chat wire's token-incremental path, which reported 0 in / 0 out.

    Two separate defects had to both be fixed for a number to appear: the
    provider was never ASKED for usage on a stream (no ``stream_options``), and
    the translator only ever read ``completion_tokens``. Fixing either alone
    still reports zero input, which is why both are pinned here.
    """

    @staticmethod
    def _events(chunks):
        from lib.hal.gateway_stream_crosswire import translate_chat_stream
        raw = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks)
        return b"".join(translate_chat_stream(iter([raw]), model="gpt-4o"))

    @staticmethod
    def _final_usage_chunk():
        """What OpenAI sends last when include_usage is on: no choices, usage."""
        return {"id": "c1", "choices": [],
                "usage": {"prompt_tokens": 900, "completion_tokens": 40,
                          "prompt_tokens_details": {"cached_tokens": 850}}}

    @staticmethod
    def _capture_request(*, want_stream):
        """Drive cross_wire_forward with the network stubbed; return the body."""
        from unittest import mock
        seen = {}

        def fake_forward(provider, path, body, **kw):
            seen.update(json.loads(body))
            return UpstreamResponse(status=200,
                                    headers=[("Content-Type", "application/json")],
                                    _reader=io.BytesIO(b'{"choices":[]}'))

        with mock.patch("lib.hal.gateway_egress.forward_same_wire", fake_forward):
            from lib.hal.gateway_egress import cross_wire_forward
            cross_wire_forward(
                {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
                provider="openai", want_stream=want_stream)
        return seen

    @staticmethod
    def _message_delta_usage(raw: bytes) -> dict:
        for line in raw.decode().splitlines():
            if not line.startswith("data: "):
                continue
            payload = json.loads(line[6:])
            if payload.get("type") == "message_delta":
                return payload.get("usage") or {}
        return {}

    def test_request_asks_the_provider_for_streamed_usage(self):
        """D1. Without stream_options the provider sends no usage at all."""
        self.assertEqual(self._capture_request(want_stream=True).get("stream_options"),
                         {"include_usage": True})

    def test_no_stream_options_on_a_non_streaming_request(self):
        self.assertNotIn("stream_options", self._capture_request(want_stream=False))

    def test_streamed_message_delta_reports_fresh_input_not_zero(self):
        """D2. The defect the sibling lane measured: input_tokens stuck at 0."""
        delta = self._message_delta_usage(self._events([
            {"id": "c1", "choices": [{"delta": {"content": "hi"}, "finish_reason": None}]},
            {"id": "c1", "choices": [{"delta": {}, "finish_reason": "stop"}]},
            self._final_usage_chunk(),
        ]))
        self.assertEqual(delta.get("input_tokens"), 50, "900 total - 850 cached = 50 fresh")
        self.assertEqual(delta.get("output_tokens"), 40)
        self.assertEqual(delta.get("cache_read_tokens"), 850)

    def test_streamed_usage_omitted_when_provider_sent_none(self):
        """Absent must not read as a free turn, here as everywhere else."""
        delta = self._message_delta_usage(self._events([
            {"id": "c1", "choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]},
        ]))
        self.assertNotIn("input_tokens", delta)
        self.assertEqual(delta.get("output_tokens"), 0)


class CouncilRaisedRobustnessTests(unittest.TestCase):
    """Concerns raised by the AGORA council, kept as anchors.

    The judge dismissed both on citation grounds — the line numbers pointed at
    docstrings. The line numbers were wrong and the substance was not, which is
    why each is pinned here rather than closed on the judge's word.
    """

    def test_partial_usage_chunk_cannot_wipe_an_earlier_input_count(self):
        """A later chunk reporting only output must not zero the input."""
        delta = TrueStreamUsageTests._message_delta_usage(TrueStreamUsageTests._events([
            {"id": "c1", "choices": [{"delta": {"content": "hi"}, "finish_reason": None}]},
            TrueStreamUsageTests._final_usage_chunk(),          # carries the real input
            {"id": "c1", "choices": [],                          # a later, poorer chunk
             "usage": {"completion_tokens": 40}},
        ]))
        self.assertEqual(delta.get("input_tokens"), 50,
                         "the chunk that carried the number must win, in any order")
        self.assertEqual(delta.get("cache_read_tokens"), 850)

    def test_non_dict_message_does_not_raise(self):
        """A provider sending "message": "<string>" must not crash the wire."""
        from lib.hal.gateway_crosswire import openai_to_anthropic_response
        out = openai_to_anthropic_response(
            {"id": "c1", "choices": [{"message": "not-a-dict", "finish_reason": "stop"}]},
            model="gpt-4o")
        self.assertEqual(out["content"], [])
        self.assertEqual(out["stop_reason"], "end_turn")


class NoDoubleCountedInputTests(unittest.TestCase):
    """Input is reported in exactly ONE frame, and which frame differs by path.

    The buffered burst holds the whole message before writing anything, so it
    reports input in ``message_start`` — Anthropic's own placement. The
    translated stream cannot: the provider sends usage last, long after
    ``message_start`` has gone out, so it reports input on the closing
    ``message_delta`` instead. Emitting it in both would let a client that sums
    usage across frames bill the input twice, which is what happened when the
    frame builder was first taught to pass input through.
    """

    @staticmethod
    def _frames(raw: bytes) -> dict:
        out = {}
        for line in raw.decode().splitlines():
            if not line.startswith("data: "):
                continue
            payload = json.loads(line[6:])
            kind = payload.get("type")
            if kind in ("message_start", "message_delta"):
                out[kind] = (payload.get("usage")
                             or (payload.get("message") or {}).get("usage") or {})
        return out

    def test_buffered_burst_reports_input_once_in_message_start(self):
        from lib.hal.gateway_crosswire import anthropic_message_to_sse_chunks
        frames = self._frames(b"".join(anthropic_message_to_sse_chunks({
            "id": "m1", "role": "assistant", "model": "gpt-6-astra",
            "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 11, "output_tokens": 5}})))
        self.assertEqual(frames["message_start"].get("input_tokens"), 11)
        self.assertNotIn("input_tokens", frames["message_delta"],
                         "input already reported in message_start; repeating it double-counts")
        self.assertEqual(frames["message_delta"].get("output_tokens"), 5)

    def test_translated_stream_reports_input_once_in_message_delta(self):
        frames = self._frames(TrueStreamUsageTests._events([
            {"id": "c1", "choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]},
            TrueStreamUsageTests._final_usage_chunk(),
        ]))
        self.assertFalse(frames["message_start"].get("input_tokens"),
                         "the provider had not sent usage yet when this frame was written")
        self.assertEqual(frames["message_delta"].get("input_tokens"), 50)


if __name__ == "__main__":
    unittest.main()
