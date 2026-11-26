"""Tests for cross-wire STREAMING translation (OpenAI Chat SSE -> Anthropic SSE).

Goodhart-proof: the emitted Anthropic SSE is parsed back into (event, data)
pairs and asserted on ORDER + reassembled content (text concatenated across
deltas, tool-call arguments concatenated across input_json_delta fragments,
stop_reason mapping). A mutation that drops a delta, mis-orders block
start/stop, forgets to close a block, or hardcodes a stop_reason fails here.
"""
import json
import unittest

from lib.hal.gateway_stream_crosswire import translate_chat_stream


def _oai_chunk(delta=None, finish=None, *, cid="chatcmpl-x", model="gpt-4o"):
    obj = {"id": cid, "object": "chat.completion.chunk", "model": model,
           "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    return f"data: {json.dumps(obj)}\n\n".encode("utf-8")


_DONE = b"data: [DONE]\n\n"


def _events(byte_chunks):
    """Translate then parse to [(event_name, data_dict)]."""
    out = []
    raw = b"".join(translate_chat_stream(byte_chunks))
    for frame in raw.split(b"\n\n"):
        if not frame.strip():
            continue
        text = frame.decode("utf-8")
        lines = text.split("\n")
        name = lines[0][len("event: "):]
        data = json.loads(lines[1][len("data: "):])
        out.append((name, data))
    return out


class TestTextStream(unittest.TestCase):

    def _stream(self):
        return [
            _oai_chunk({"role": "assistant", "content": "Hello"}),
            _oai_chunk({"content": " world"}),
            _oai_chunk({}, "stop"),
            _DONE,
        ]

    def test_event_order(self):
        names = [n for n, _ in _events(self._stream())]
        self.assertEqual(names, [
            "message_start", "content_block_start", "content_block_delta",
            "content_block_delta", "content_block_stop", "message_delta", "message_stop"])

    def test_text_reassembles(self):
        deltas = [d["delta"]["text"] for n, d in _events(self._stream())
                  if n == "content_block_delta"]
        self.assertEqual("".join(deltas), "Hello world")

    def test_message_start_carries_id_and_model(self):
        ev = _events(self._stream())
        msg = ev[0][1]["message"]
        self.assertEqual(msg["id"], "chatcmpl-x")
        self.assertEqual(msg["model"], "gpt-4o")
        self.assertEqual(msg["content"], [])
        self.assertIsNone(msg["stop_reason"])

    def test_stop_reason_in_message_delta(self):
        ev = dict((n, d) for n, d in _events(self._stream()))
        self.assertEqual(ev["message_delta"]["delta"]["stop_reason"], "end_turn")


class TestToolCallStream(unittest.TestCase):

    def _stream(self):
        return [
            _oai_chunk({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                        "function": {"name": "get_weather", "arguments": ""}}]}),
            _oai_chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{\"ci"}}]}),
            _oai_chunk({"tool_calls": [{"index": 0, "function": {"arguments": "ty\":\"SF\"}"}}]}),
            _oai_chunk({}, "tool_calls"),
            _DONE,
        ]

    def test_event_order(self):
        names = [n for n, _ in _events(self._stream())]
        self.assertEqual(names, [
            "message_start", "content_block_start", "content_block_delta",
            "content_block_delta", "content_block_stop", "message_delta", "message_stop"])

    def test_tool_block_start_carries_id_and_name(self):
        ev = _events(self._stream())
        start = next(d for n, d in ev if n == "content_block_start")
        self.assertEqual(start["content_block"],
                         {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {}})

    def test_arguments_reassemble_to_valid_json(self):
        frags = [d["delta"]["partial_json"] for n, d in _events(self._stream())
                 if n == "content_block_delta"]
        self.assertEqual(json.loads("".join(frags)), {"city": "SF"})

    def test_stop_reason_tool_use(self):
        ev = dict((n, d) for n, d in _events(self._stream()))
        self.assertEqual(ev["message_delta"]["delta"]["stop_reason"], "tool_use")


class TestTextThenTool(unittest.TestCase):

    def test_two_blocks_indexed_in_order(self):
        stream = [
            _oai_chunk({"content": "Let me check."}),
            _oai_chunk({"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                        "function": {"name": "lookup", "arguments": "{}"}}]}),
            _oai_chunk({}, "tool_calls"),
            _DONE,
        ]
        ev = _events(stream)
        starts = [d for n, d in ev if n == "content_block_start"]
        stops = [d for n, d in ev if n == "content_block_stop"]
        self.assertEqual(starts[0]["index"], 0)
        self.assertEqual(starts[0]["content_block"]["type"], "text")
        self.assertEqual(starts[1]["index"], 1)
        self.assertEqual(starts[1]["content_block"]["type"], "tool_use")
        # the text block (0) is closed BEFORE the tool block (1) opens
        names = [n for n, _ in ev]
        self.assertLess(names.index("content_block_stop"),
                        names.index("content_block_start", names.index("content_block_start") + 1))
        self.assertEqual({s["index"] for s in stops}, {0, 1})


class TestStreamEdgeCases(unittest.TestCase):

    def test_empty_stream_is_well_formed_envelope(self):
        names = [n for n, _ in _events([_DONE])]
        self.assertEqual(names, ["message_start", "message_delta", "message_stop"])

    def test_finish_reason_length_maps_max_tokens(self):
        stream = [_oai_chunk({"content": "x"}), _oai_chunk({}, "length"), _DONE]
        ev = dict((n, d) for n, d in _events(stream))
        self.assertEqual(ev["message_delta"]["delta"]["stop_reason"], "max_tokens")

    def test_usage_output_tokens_surfaced_when_present(self):
        # final usage chunk (OpenAI stream_options.include_usage)
        usage_chunk = json.dumps({
            "id": "c", "object": "chat.completion.chunk", "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 11}})
        stream = [_oai_chunk({"content": "hi"}),
                  f"data: {usage_chunk}\n\n".encode("utf-8"), _DONE]
        ev = dict((n, d) for n, d in _events(stream))
        self.assertEqual(ev["message_delta"]["usage"]["output_tokens"], 11)

    def test_usage_retained_across_a_later_no_usage_chunk(self):
        # usage arrives, THEN a finish chunk with NO usage -> the value is
        # retained (we overwrite ONLY when a chunk carries completion_tokens),
        # so the final message_delta reflects the last *seen* usage, not 0.
        usage_chunk = json.dumps({
            "id": "c", "object": "chat.completion.chunk", "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 7}})
        stream = [f"data: {usage_chunk}\n\n".encode("utf-8"),
                  _oai_chunk({}, "stop"),  # later chunk carries no usage
                  _DONE]
        ev = dict((n, d) for n, d in _events(stream))
        self.assertEqual(ev["message_delta"]["usage"]["output_tokens"], 7)

    def test_every_frame_valid_utf8_sse(self):
        stream = [_oai_chunk({"content": "héllo 世界"}), _oai_chunk({}, "stop"), _DONE]
        for frame in b"".join(translate_chat_stream(stream)).split(b"\n\n"):
            if frame.strip():
                self.assertTrue(frame.startswith(b"event: "))
                frame.decode("utf-8")


if __name__ == "__main__":
    unittest.main()
