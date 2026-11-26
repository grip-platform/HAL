"""Tests for cross-wire SSE synthesis (Phase 0 wave 4b).

A COMPLETE Anthropic Messages response is re-emitted as a valid Anthropic SSE
event sequence so an Anthropic-wire client (Claude Code, which always sets
``stream:true``) parses a cross-wired OpenAI response natively. Goodhart-proof:
event ORDER + names are asserted against a hardcoded expected sequence, and the
payload values (full text, tool input, stop_reason, the input/output token
split across message_start vs message_delta) are asserted — so a mutation that
drops a block, reorders events, hardcodes a stop_reason, or swaps the token
counts fails here.
"""
import json
import unittest

from lib.hal.gateway_crosswire import anthropic_message_to_sse_chunks


def _parse(chunks):
    """[(event_name, data_dict)] from raw SSE frames; asserts frame shape."""
    out = []
    for chunk in chunks:
        text = chunk.decode("utf-8")
        assert text.endswith("\n\n"), f"frame not terminated by blank line: {text!r}"
        lines = text.rstrip("\n").split("\n")
        assert lines[0].startswith("event: "), lines
        assert lines[1].startswith("data: "), lines
        out.append((lines[0][len("event: "):], json.loads(lines[1][len("data: "):])))
    return out


class TestSSEEventSequence(unittest.TestCase):

    def _msg(self):
        return {
            "id": "msg_1", "type": "message", "role": "assistant", "model": "gpt-4o",
            "content": [
                {"type": "text", "text": "Hello there"},
                {"type": "tool_use", "id": "call_1", "name": "get_weather",
                 "input": {"city": "SF"}},
            ],
            "stop_reason": "tool_use", "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

    def test_order_is_canonical(self):
        events = [name for name, _ in _parse(anthropic_message_to_sse_chunks(self._msg()))]
        self.assertEqual(events[0], "message_start")
        self.assertEqual(events[-1], "message_stop")
        self.assertEqual(events[-2], "message_delta")
        # two blocks => two start/delta/stop triples between start and message_delta
        self.assertEqual(
            events[1:-2],
            ["content_block_start", "content_block_delta", "content_block_stop"] * 2,
        )

    def test_message_start_has_empty_content_and_zero_output(self):
        parsed = _parse(anthropic_message_to_sse_chunks(self._msg()))
        name, data = parsed[0]
        self.assertEqual(name, "message_start")
        self.assertEqual(data["message"]["content"], [])
        self.assertEqual(data["message"]["id"], "msg_1")
        # input_tokens preserved, output_tokens reset to 0 (final count in message_delta)
        self.assertEqual(data["message"]["usage"], {"input_tokens": 10, "output_tokens": 0})

    def test_text_block_carries_full_text(self):
        parsed = _parse(anthropic_message_to_sse_chunks(self._msg()))
        # index 1 = first content_block_start, 2 = its delta
        self.assertEqual(parsed[1][1]["content_block"], {"type": "text", "text": ""})
        self.assertEqual(parsed[2][1]["delta"], {"type": "text_delta", "text": "Hello there"})

    def test_tool_use_block_emits_input_json_delta(self):
        parsed = _parse(anthropic_message_to_sse_chunks(self._msg()))
        start = parsed[4][1]["content_block"]
        self.assertEqual(start, {"type": "tool_use", "id": "call_1",
                                 "name": "get_weather", "input": {}})
        delta = parsed[5][1]["delta"]
        self.assertEqual(delta["type"], "input_json_delta")
        self.assertEqual(json.loads(delta["partial_json"]), {"city": "SF"})

    def test_message_delta_carries_stop_reason_and_output_tokens(self):
        parsed = _parse(anthropic_message_to_sse_chunks(self._msg()))
        name, data = parsed[-2]
        self.assertEqual(name, "message_delta")
        self.assertEqual(data["delta"]["stop_reason"], "tool_use")
        self.assertEqual(data["usage"], {"output_tokens": 5})

    def test_block_indices_increment(self):
        parsed = _parse(anthropic_message_to_sse_chunks(self._msg()))
        self.assertEqual(parsed[1][1]["index"], 0)
        self.assertEqual(parsed[4][1]["index"], 1)


class TestSSEEdgeCases(unittest.TestCase):

    def test_empty_content_still_well_formed(self):
        msg = {"id": "m", "type": "message", "role": "assistant", "model": "x",
               "content": [], "stop_reason": "end_turn", "stop_sequence": None,
               "usage": {"input_tokens": 3, "output_tokens": 0}}
        events = [name for name, _ in _parse(anthropic_message_to_sse_chunks(msg))]
        self.assertEqual(events, ["message_start", "message_delta", "message_stop"])

    def test_unknown_block_type_skipped(self):
        msg = {"id": "m", "type": "message", "role": "assistant", "model": "x",
               "content": [{"type": "thinking", "thinking": "x"},
                           {"type": "text", "text": "ok"}],
               "stop_reason": "end_turn", "stop_sequence": None,
               "usage": {"input_tokens": 1, "output_tokens": 1}}
        events = [name for name, _ in _parse(anthropic_message_to_sse_chunks(msg))]
        # only the text block produces a triple; the unknown block is dropped
        self.assertEqual(events, ["message_start", "content_block_start",
                                  "content_block_delta", "content_block_stop",
                                  "message_delta", "message_stop"])

    def test_every_frame_is_valid_sse(self):
        msg = {"id": "m", "type": "message", "role": "assistant", "model": "x",
               "content": [{"type": "text", "text": "hi"}],
               "stop_reason": "end_turn", "stop_sequence": None,
               "usage": {"input_tokens": 1, "output_tokens": 1}}
        for chunk in anthropic_message_to_sse_chunks(msg):
            self.assertTrue(chunk.startswith(b"event: "))
            self.assertIn(b"\ndata: ", chunk)
            self.assertTrue(chunk.endswith(b"\n\n"))


class TestMessageStartCanonical(unittest.TestCase):
    """message_start carries ONLY canonical envelope keys; stop_reason null."""

    def _start(self, msg):
        return _parse(anthropic_message_to_sse_chunks(msg))[0][1]["message"]

    def test_stop_reason_null_in_start_final_in_delta(self):
        msg = {"id": "m", "type": "message", "role": "assistant", "model": "x",
               "content": [{"type": "text", "text": "hi"}],
               "stop_reason": "end_turn", "stop_sequence": None,
               "usage": {"input_tokens": 1, "output_tokens": 4}}
        parsed = _parse(anthropic_message_to_sse_chunks(msg))
        self.assertIsNone(parsed[0][1]["message"]["stop_reason"])  # null at start
        self.assertEqual(parsed[-2][1]["delta"]["stop_reason"], "end_turn")  # final in delta

    def test_unknown_top_level_keys_dropped_from_start(self):
        # A non-canonical (and here non-JSON-serialisable) field on the source
        # must NOT leak into message_start — rebuilt from known keys only.
        msg = {"id": "m", "type": "message", "role": "assistant", "model": "x",
               "content": [], "stop_reason": "end_turn", "stop_sequence": None,
               "usage": {"input_tokens": 1, "output_tokens": 0},
               "_internal": object()}  # would crash json.dumps if passed through
        start = self._start(msg)
        self.assertEqual(set(start.keys()),
                         {"id", "type", "role", "model", "content", "stop_reason",
                          "stop_sequence", "usage"})
        self.assertNotIn("_internal", start)

    def test_all_chunks_decode_as_utf8(self):
        msg = {"id": "m", "type": "message", "role": "assistant", "model": "x",
               "content": [{"type": "text", "text": "héllo – 世界"}],
               "stop_reason": "end_turn", "stop_sequence": None,
               "usage": {"input_tokens": 1, "output_tokens": 1}}
        for chunk in anthropic_message_to_sse_chunks(msg):
            chunk.decode("utf-8")  # raises if any frame is not valid UTF-8


if __name__ == "__main__":
    unittest.main()
