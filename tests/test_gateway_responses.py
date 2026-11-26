"""HAL Gateway P6 — OpenAI Responses API translation tests. Goodhart-proof.

Each test fails under the obvious mutation:
  - string input not wrapped as user message       -> test_request_string_input
  - instructions not hoisted to system             -> test_request_instructions
  - function_call item not translated              -> test_request_function_call_item
  - function_call_output item not translated       -> test_request_function_call_output
  - FLAT tool parameters not wired as input_schema -> test_request_flat_tools
  - max_output_tokens not mapped to max_tokens     -> test_request_max_output_tokens
  - text block not surfaced as output_text item    -> test_response_text
  - tool_use not surfaced as function_call item    -> test_response_tool_use
  - usage fields mis-mapped                        -> test_response_usage
  - object/status wrong                            -> test_response_shape
  - delta dropped in streaming                     -> test_stream_text_deltas
  - last event not response.completed              -> test_stream_last_event
  - usage not in completed event                   -> test_stream_usage
  - chunk boundaries break events                  -> test_stream_chunk_boundaries
  - tool call stream events not emitted            -> test_stream_tool_call
  - Bearer not translated to x-api-key            -> test_auth_reuse
  - malformed body crashes                         -> test_failopen_malformed
  - upstream not forwarded on bad body             -> test_failopen_no_forward
  - route absent from server.py                    -> test_wiring_server
  - route absent from gateway.py                   -> test_wiring_gateway
"""
import io
import json
import os
import unittest

from lib.hal.gateway_responses import (
    RESPONSES_PATH,
    anthropic_to_responses,
    anthropic_upstream_headers,
    handle_responses,
    responses_to_anthropic_request,
    translate_responses_stream,
)

_HERE = os.path.dirname(__file__)
_SERVER_SRC = os.path.join(_HERE, "..", "lib", "hal", "server.py")
_GATEWAY_SRC = os.path.join(_HERE, "..", "lib", "hal", "gateway.py")


# ---------------------------------------------------------------------------
# Shared fakes (mirrors test_gateway_openai.py pattern)
# ---------------------------------------------------------------------------

class FakeUpstream:
    """Stand-in for gateway.UpstreamResponse, which always carries ``headers``."""

    def __init__(self, status, body=b"", chunks=None, headers=()):
        self.status = status
        self.headers = list(headers)
        self._body = body
        self._chunks = chunks
        self.closed = False

    def read_all(self):
        return self._body

    def iter_chunks(self, size=65536):
        for c in (self._chunks or [self._body]):
            if c:
                yield c

    def close(self):
        self.closed = True


class FakeProxy:
    def __init__(self, result):
        self._result = result
        self.calls = []

    def forward(self, path, body, client_headers, *, method="POST", max_retries=None):
        self.calls.append({"path": path, "body": body,
                           "headers": list(client_headers), "method": method})
        return self._result


class FakeHandler:
    def __init__(self, command="POST", headers=None, body=b""):
        self.command = command
        self.headers = headers if headers is not None else {}
        self.rfile = io.BytesIO(body)
        self.wfile = io.BytesIO()
        self.status = None
        self.sent_headers = []

    def send_response(self, code):
        self.status = code

    def send_header(self, k, v):
        self.sent_headers.append((k, v))

    def end_headers(self):
        pass

    def body_json(self):
        return json.loads(self.wfile.getvalue())

    def client_text(self):
        parts = [str(self.status)]
        parts += [f"{k}: {v}" for k, v in self.sent_headers]
        parts.append(self.wfile.getvalue().decode("utf-8", "replace"))
        return "\n".join(parts)


def _handler(payload, key="sk-ant-test"):
    raw = json.dumps(payload).encode()
    return FakeHandler(headers={
        "Content-Length": str(len(raw)),
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }, body=raw)


def _parse_responses_events(raw: bytes):
    """Parse Responses SSE bytes into list of event dicts."""
    events = []
    for block in raw.decode("utf-8").split("\n\n"):
        lines = [l for l in block.strip().splitlines() if l]
        data_line = next((l[5:].strip() for l in lines if l.startswith("data:")), None)
        if data_line:
            events.append(json.loads(data_line))
    return events


# ---------------------------------------------------------------------------
# Realistic Anthropic SSE fixture
# ---------------------------------------------------------------------------

_ANTHROPIC_SSE = (
    'event: message_start\n'
    'data: {"type":"message_start","message":{"id":"msg_1","model":"claude-opus-4-7",'
    '"usage":{"input_tokens":10}}}\n\n'
    'event: content_block_start\n'
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
    'event: content_block_delta\n'
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n'
    'event: content_block_delta\n'
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}\n\n'
    'event: content_block_stop\n'
    'data: {"type":"content_block_stop","index":0}\n\n'
    'event: message_delta\n'
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
    '"usage":{"output_tokens":5}}\n\n'
    'event: message_stop\n'
    'data: {"type":"message_stop"}\n\n'
).encode("utf-8")

_ANTHROPIC_TOOL_SSE = (
    'event: message_start\n'
    'data: {"type":"message_start","message":{"id":"msg_2","model":"claude-opus-4-7",'
    '"usage":{"input_tokens":8}}}\n\n'
    'event: content_block_start\n'
    'data: {"type":"content_block_start","index":0,"content_block":'
    '{"type":"tool_use","id":"call_abc","name":"get_weather"}}\n\n'
    'event: content_block_delta\n'
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta",'
    '"partial_json":"{\\"city\\""}}\n\n'
    'event: content_block_delta\n'
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta",'
    '"partial_json":":\\"Cape Town\\"}"}}\n\n'
    'event: content_block_stop\n'
    'data: {"type":"content_block_stop","index":0}\n\n'
    'event: message_delta\n'
    'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},'
    '"usage":{"output_tokens":3}}\n\n'
    'event: message_stop\n'
    'data: {"type":"message_stop"}\n\n'
).encode("utf-8")


# ---------------------------------------------------------------------------
# Request translation
# ---------------------------------------------------------------------------

class TestRequestTranslation(unittest.TestCase):

    def test_request_string_input(self):
        req = responses_to_anthropic_request({"model": "m", "input": "hello"})
        self.assertEqual(req["messages"], [{"role": "user", "content": "hello"}])

    def test_request_array_message_item(self):
        req = responses_to_anthropic_request({"model": "m", "input": [
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "hi"}]},
        ]})
        self.assertEqual(req["messages"][0]["role"], "user")
        self.assertEqual(req["messages"][0]["content"][0]["text"], "hi")

    def test_request_instructions(self):
        req = responses_to_anthropic_request({
            "model": "m", "input": "hi", "instructions": "Be terse."})
        self.assertEqual(req["system"], "Be terse.")

    def test_request_no_system_when_no_instructions(self):
        req = responses_to_anthropic_request({"model": "m", "input": "hi"})
        self.assertNotIn("system", req)

    def test_request_null_instructions_never_injects_literal_none(self):
        # Forward-anchor (Goodhart guard): a payload with "instructions": null
        # must NOT produce a system field — and must NEVER inject the literal
        # string "None" into system text. The `if instructions` guard in
        # _join_system is load-bearing: a mutant that drops it (unconditional
        # str(instructions)) yields system == "None" and fails this test.
        req = responses_to_anthropic_request({
            "model": "m", "input": "hi", "instructions": None})
        self.assertNotIn("system", req)
        # belt-and-braces: even when a system-role message coexists with null
        # instructions, the merged system text must not contain "None".
        req2 = responses_to_anthropic_request({
            "model": "m", "instructions": None, "input": [
                {"type": "message", "role": "developer", "content": "Real rule."},
                {"type": "message", "role": "user", "content": "q"},
            ]})
        self.assertEqual(req2["system"], "Real rule.")
        self.assertNotIn("None", req2["system"])

    def test_content_to_text_accepts_responses_block_types(self):
        # Defensive: _content_to_text handles raw Responses block types directly
        # (text/input_text/output_text), not only the post-translation "text".
        # Closes the AGORA concern that input_text would be silently dropped.
        from lib.hal.gateway_responses import _content_to_text
        self.assertEqual(
            _content_to_text([{"type": "input_text", "text": "a"},
                              {"type": "output_text", "text": "b"},
                              {"type": "text", "text": "c"}]),
            "a\nb\nc")
        self.assertEqual(_content_to_text("plain"), "plain")

    def test_request_developer_role_routed_to_system(self):
        # Codex sends its system prompt as a "developer" role message. Anthropic
        # /v1/messages rejects any role but user/assistant, so it MUST hoist to
        # the top-level system field — never remain in messages. (The live bug.)
        req = responses_to_anthropic_request({"model": "m", "input": [
            {"type": "message", "role": "developer",
             "content": [{"type": "input_text", "text": "You are concise."}]},
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "hi"}]},
        ]})
        self.assertEqual(req["system"], "You are concise.")
        self.assertEqual([m["role"] for m in req["messages"]], ["user"])

    def test_request_system_role_routed_to_system(self):
        req = responses_to_anthropic_request({"model": "m", "input": [
            {"type": "message", "role": "system", "content": "Rules here."},
            {"type": "message", "role": "user", "content": "go"},
        ]})
        self.assertEqual(req["system"], "Rules here.")
        self.assertEqual([m["role"] for m in req["messages"]], ["user"])

    def test_request_instructions_and_developer_merged(self):
        req = responses_to_anthropic_request({
            "model": "m", "instructions": "Top.", "input": [
                {"type": "message", "role": "developer", "content": "Dev."},
                {"type": "message", "role": "user", "content": "q"},
            ]})
        self.assertEqual(req["system"], "Top.\n\nDev.")
        self.assertEqual([m["role"] for m in req["messages"]], ["user"])

    def test_request_only_user_assistant_roles_reach_messages(self):
        # Goodhart guard: whatever roles arrive, ONLY user/assistant may appear
        # in the Anthropic messages array. A mutant that passes role through
        # (the bug) leaks "developer" here and fails.
        req = responses_to_anthropic_request({"model": "m", "input": [
            {"type": "message", "role": "developer", "content": "sys"},
            {"type": "message", "role": "user", "content": "u"},
            {"type": "message", "role": "assistant", "content": "a"},
            {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c", "output": "o"},
        ]})
        roles = {m["role"] for m in req["messages"]}
        self.assertTrue(roles <= {"user", "assistant"}, f"leaked roles: {roles}")
        # developer content still preserved in system, not dropped
        self.assertEqual(req["system"], "sys")

    def test_request_function_call_item(self):
        req = responses_to_anthropic_request({"model": "m", "input": [
            {"type": "function_call", "call_id": "c1", "name": "search",
             "arguments": '{"q":"x"}'},
        ]})
        block = req["messages"][0]["content"][0]
        self.assertEqual(block["type"], "tool_use")
        self.assertEqual(block["id"], "c1")
        self.assertEqual(block["name"], "search")
        self.assertEqual(block["input"], {"q": "x"})

    def test_request_function_call_output_item(self):
        req = responses_to_anthropic_request({"model": "m", "input": [
            {"type": "function_call_output", "call_id": "c1", "output": "42"},
        ]})
        block = req["messages"][0]["content"][0]
        self.assertEqual(block["type"], "tool_result")
        self.assertEqual(block["tool_use_id"], "c1")
        self.assertEqual(block["content"], "42")

    def test_request_flat_tools(self):
        # Responses tools are FLAT (no nested "function" key — unlike Chat Completions)
        req = responses_to_anthropic_request({"model": "m", "input": "x", "tools": [
            {"type": "function", "name": "f", "description": "d",
             "parameters": {"type": "object", "properties": {"a": {"type": "string"}}}},
        ]})
        tool = req["tools"][0]
        self.assertEqual(tool["name"], "f")
        self.assertEqual(tool["input_schema"]["properties"]["a"]["type"], "string")
        # Must NOT have a nested "function" key — flat shape
        self.assertNotIn("function", tool)

    def test_request_max_output_tokens(self):
        req = responses_to_anthropic_request({"model": "m", "input": "x",
                                               "max_output_tokens": 512})
        self.assertEqual(req["max_tokens"], 512)

    def test_request_default_max_tokens(self):
        req = responses_to_anthropic_request({"model": "m", "input": "x"})
        self.assertEqual(req["max_tokens"], 4096)

    def test_request_passthrough_fields(self):
        req = responses_to_anthropic_request({"model": "m", "input": "x",
                                               "temperature": 0.5, "top_p": 0.9,
                                               "stream": True})
        self.assertEqual(req["temperature"], 0.5)
        self.assertEqual(req["top_p"], 0.9)
        self.assertTrue(req["stream"])


# ---------------------------------------------------------------------------
# Non-streaming response translation
# ---------------------------------------------------------------------------

class TestResponseTranslation(unittest.TestCase):

    def test_response_text(self):
        resp = anthropic_to_responses(
            {"model": "claude-opus-4-7",
             "content": [{"type": "text", "text": "Cape Town is sunny."}],
             "usage": {"input_tokens": 5, "output_tokens": 3}},
            resp_id="resp_1", model="m")
        item = resp["output"][0]
        self.assertEqual(item["type"], "message")
        self.assertEqual(item["content"][0]["type"], "output_text")
        self.assertEqual(item["content"][0]["text"], "Cape Town is sunny.")
        self.assertEqual(item["content"][0]["annotations"], [])

    def test_response_tool_use(self):
        resp = anthropic_to_responses(
            {"model": "m",
             "content": [{"type": "tool_use", "id": "tu_1", "name": "search",
                          "input": {"q": "test"}}],
             "usage": {"input_tokens": 4, "output_tokens": 2}},
            resp_id="resp_2", model="m")
        item = resp["output"][0]
        self.assertEqual(item["type"], "function_call")
        self.assertEqual(item["call_id"], "tu_1")
        self.assertEqual(item["name"], "search")
        self.assertEqual(json.loads(item["arguments"]), {"q": "test"})
        self.assertEqual(item["status"], "completed")

    def test_response_usage(self):
        resp = anthropic_to_responses(
            {"model": "m", "content": [{"type": "text", "text": "x"}],
             "usage": {"input_tokens": 12, "output_tokens": 7}},
            resp_id="r", model="m")
        self.assertEqual(resp["usage"]["input_tokens"], 12)
        self.assertEqual(resp["usage"]["output_tokens"], 7)
        self.assertEqual(resp["usage"]["total_tokens"], 19)

    def test_response_shape(self):
        resp = anthropic_to_responses(
            {"model": "claude-opus-4-7", "content": [{"type": "text", "text": "x"}],
             "usage": {}},
            resp_id="resp_x", model="fallback")
        self.assertEqual(resp["object"], "response")
        self.assertEqual(resp["status"], "completed")
        self.assertEqual(resp["model"], "claude-opus-4-7")


# ---------------------------------------------------------------------------
# Streaming translation
# ---------------------------------------------------------------------------

class TestStreamTranslation(unittest.TestCase):

    def _events(self, sse_bytes, chunks=None):
        src = chunks if chunks is not None else [sse_bytes]
        raw = b"".join(translate_responses_stream(
            src, resp_id="resp_s", model="claude-opus-4-7"))
        return _parse_responses_events(raw)

    def test_stream_first_event_is_response_created(self):
        events = self._events(_ANTHROPIC_SSE)
        self.assertEqual(events[0]["type"], "response.created")
        self.assertEqual(events[0]["response"]["status"], "in_progress")

    def test_stream_text_deltas(self):
        events = self._events(_ANTHROPIC_SSE)
        deltas = [e["delta"] for e in events
                  if e.get("type") == "response.output_text.delta"]
        self.assertEqual("".join(deltas), "Hello world",
                         "dropping a delta must fail this assertion")

    def test_stream_last_event_is_response_completed(self):
        events = self._events(_ANTHROPIC_SSE)
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(events[-1]["response"]["status"], "completed")

    def test_stream_completed_carries_full_text(self):
        events = self._events(_ANTHROPIC_SSE)
        completed = events[-1]["response"]
        item = completed["output"][0]
        self.assertEqual(item["type"], "message")
        self.assertEqual(item["content"][0]["text"], "Hello world")

    def test_stream_usage(self):
        events = self._events(_ANTHROPIC_SSE)
        usage = events[-1]["response"]["usage"]
        self.assertIsNotNone(usage, "usage must be present in response.completed")
        self.assertEqual(usage["output_tokens"], 5)
        self.assertEqual(usage["total_tokens"], 15)

    def test_stream_sequence_numbers_monotonic(self):
        events = self._events(_ANTHROPIC_SSE)
        seqs = [e["sequence_number"] for e in events if "sequence_number" in e]
        self.assertEqual(seqs, list(range(len(seqs))),
                         "sequence_number must be monotonically increasing from 0")

    def test_stream_chunk_boundaries(self):
        raw = _ANTHROPIC_SSE
        fragments = [raw[i:i + 7] for i in range(0, len(raw), 7)]
        events = self._events(None, chunks=fragments)
        deltas = [e["delta"] for e in events
                  if e.get("type") == "response.output_text.delta"]
        self.assertEqual("".join(deltas), "Hello world",
                         "events split across 7-byte chunks must survive")
        self.assertEqual(events[-1]["type"], "response.completed")

    def test_stream_tool_call_output_item_added(self):
        events = self._events(_ANTHROPIC_TOOL_SSE)
        added = [e for e in events if e.get("type") == "response.output_item.added"]
        self.assertTrue(any(e["item"]["type"] == "function_call" for e in added),
                        "function_call output_item.added must be emitted for tool_use block")

    def test_stream_tool_call_args_delta(self):
        events = self._events(_ANTHROPIC_TOOL_SSE)
        arg_deltas = [e for e in events
                      if e.get("type") == "response.function_call_arguments.delta"]
        full = "".join(e["delta"] for e in arg_deltas)
        self.assertIn("Cape Town", full)

    def test_stream_tool_call_in_completed_output(self):
        events = self._events(_ANTHROPIC_TOOL_SSE)
        completed_output = events[-1]["response"]["output"]
        fc = next((i for i in completed_output if i.get("type") == "function_call"), None)
        self.assertIsNotNone(fc, "function_call must appear in response.completed output")
        self.assertEqual(fc["name"], "get_weather")


# ---------------------------------------------------------------------------
# Auth reuse
# ---------------------------------------------------------------------------

class TestAuthReuse(unittest.TestCase):

    def test_auth_reuse(self):
        # anthropic_upstream_headers is imported from gateway_openai — verify it
        # still works as expected (reuse-check, not duplication).
        out = dict(anthropic_upstream_headers([
            ("Authorization", "Bearer sk-ant-x"),
        ]))
        self.assertEqual(out["x-api-key"], "sk-ant-x")


# ---------------------------------------------------------------------------
# Handler end-to-end
# ---------------------------------------------------------------------------

class TestHandlerEndToEnd(unittest.TestCase):

    def _ok_upstream(self):
        return FakeUpstream(200, json.dumps({
            "id": "msg_y", "model": "claude-opus-4-7",
            "content": [{"type": "text", "text": "sunny"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }).encode())

    def test_nonstream_roundtrip(self):
        proxy = FakeProxy(self._ok_upstream())
        h = _handler({"model": "claude-opus-4-7", "input": "weather?", "stream": False})
        handle_responses(h, proxy)
        self.assertEqual(h.status, 200)
        body = h.body_json()
        self.assertEqual(body["object"], "response")
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["output"][0]["content"][0]["text"], "sunny")

    def test_upstream_receives_translated_request(self):
        proxy = FakeProxy(self._ok_upstream())
        h = _handler({"model": "claude-opus-4-7", "input": "hi",
                       "instructions": "Be terse.", "stream": False})
        handle_responses(h, proxy)
        call = proxy.calls[0]
        self.assertEqual(call["path"], "/v1/messages")
        sent = json.loads(call["body"])
        self.assertEqual(sent["system"], "Be terse.")
        self.assertEqual(sent["messages"][0]["role"], "user")

    def test_streaming_roundtrip(self):
        proxy = FakeProxy(FakeUpstream(200, chunks=[_ANTHROPIC_SSE]))
        h = _handler({"model": "m", "input": "hi", "stream": True})
        handle_responses(h, proxy)
        self.assertEqual(h.status, 200)
        self.assertIn(("Content-Type", "text/event-stream"), h.sent_headers)
        events = _parse_responses_events(h.wfile.getvalue())
        self.assertEqual(events[0]["type"], "response.created")
        self.assertEqual(events[-1]["type"], "response.completed")

    def test_upstream_socket_closed_nonstream(self):
        ok = self._ok_upstream()
        handle_responses(_handler({"model": "m", "input": "x", "stream": False}),
                         FakeProxy(ok))
        self.assertTrue(ok.closed)

    def test_upstream_socket_closed_stream(self):
        streamed = FakeUpstream(200, chunks=[_ANTHROPIC_SSE])
        handle_responses(_handler({"model": "m", "input": "x", "stream": True}),
                         FakeProxy(streamed))
        self.assertTrue(streamed.closed)

    def test_upstream_socket_closed_error(self):
        err = FakeUpstream(429, json.dumps({"error": {"message": "slow down"}}).encode())
        handle_responses(_handler({"model": "m", "input": "x"}), FakeProxy(err))
        self.assertTrue(err.closed)

    def test_failopen_malformed_body(self):
        proxy = FakeProxy(self._ok_upstream())
        h = FakeHandler(headers={"Content-Length": "7"}, body=b"NOTJSON")
        handle_responses(h, proxy)
        self.assertEqual(h.status, 400)
        self.assertIn("error", h.body_json())
        self.assertEqual(len(proxy.calls), 0,
                         "must not forward body that cannot be translated")

    def test_failopen_no_forward_on_bad_body(self):
        # Alias of above — asserts the upstream is NOT called (Goodhart mutation anchor).
        proxy = FakeProxy(self._ok_upstream())
        FakeHandler(headers={"Content-Length": "3"}, body=b"BAD")
        h2 = FakeHandler(headers={"Content-Length": "3"}, body=b"BAD")
        handle_responses(h2, proxy)
        self.assertEqual(proxy.calls, [])

    def test_upstream_error_relayed(self):
        # Inverted 2026-10-05 (InfiniteMachine v2, Step 5a): this asserted the
        # 429 reached Codex as a 429, which ends its turn even with Retry-After
        # set (fleet finding f26). It goes out as a 503, Retry-After kept.
        proxy = FakeProxy(FakeUpstream(
            429, json.dumps({"error": {"message": "rate limit"}}).encode(),
            headers=[("Retry-After", "4"), ("retry-after-ms", "4000")]))
        h = _handler({"model": "m", "input": "x"})
        handle_responses(h, proxy)
        self.assertEqual(h.status, 503)
        self.assertIn(("Retry-After", "4"), h.sent_headers)
        self.assertIn(("retry-after-ms", "4000"), h.sent_headers)
        self.assertEqual(h.body_json()["error"]["message"], "rate limit")


# ---------------------------------------------------------------------------
# Wiring checks
# ---------------------------------------------------------------------------

class TestWiring(unittest.TestCase):

    def test_responses_path_constant(self):
        self.assertEqual(RESPONSES_PATH, "/v1/responses")

    def test_wiring_server(self):
        src = open(_SERVER_SRC, encoding="utf-8").read()
        self.assertIn("handle_responses", src)
        self.assertIn("/v1/responses", src)

    def test_wiring_gateway(self):
        src = open(_GATEWAY_SRC, encoding="utf-8").read()
        self.assertIn("/v1/responses", src)
        self.assertIn("handle_responses", src)


if __name__ == "__main__":
    unittest.main()
