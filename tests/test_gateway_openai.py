"""HAL Gateway P2 — OpenAI-compat translation tests. Goodhart-proof.

Each test fails under the obvious mutation:

  - wrong stop_reason->finish_reason map   -> test_finish_reason_*
  - system role not hoisted                -> test_request_hoists_system
  - max_tokens default dropped             -> test_request_defaults_max_tokens
  - tool / tool_choice mistranslated       -> test_request_translates_tools*
  - usage fields mis-mapped                -> test_response_maps_usage
  - tool_use not surfaced as tool_calls    -> test_response_tool_use_*
  - a text_delta dropped in streaming      -> test_stream_*
  - SSE event split across chunks lost     -> test_stream_handles_chunk_boundaries
  - Bearer not translated to x-api-key     -> test_auth_*
  - key leaked to client                   -> test_key_never_in_client_output
  - malformed body crashes                 -> test_failopen_*
  - route unwired                          -> test_route_registered_*

Translators are pure functions; the handler is exercised with a fake proxy +
fake handler (DIP) so there is zero network.
"""
import io
import json
import os
import unittest

from lib.hal.gateway_openai import (
    CHAT_COMPLETIONS_PATH,
    anthropic_to_openai_response,
    anthropic_upstream_headers,
    guarded_sse_events,
    handle_chat_completions,
    openai_to_anthropic_request,
    translate_sse_stream,
    upstream_error_text,
)

_HERE = os.path.dirname(__file__)
_SERVER_SRC = os.path.join(_HERE, "..", "lib", "hal", "server.py")
_GATEWAY_SRC = os.path.join(_HERE, "..", "lib", "hal", "gateway.py")


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
    """Records the forward() call and returns a scripted upstream response."""

    def __init__(self, result):
        self._result = result
        self.calls = []

    def forward(self, path, body, client_headers, *, method="POST", max_retries=None):
        self.calls.append({"path": path, "body": body,
                           "headers": list(client_headers), "method": method})
        return self._result


class FakeHandler:
    """Captures everything written to the client."""

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


def _parse_openai_chunks(raw: bytes):
    """Split collected SSE bytes into (chunk dicts, done_seen)."""
    done = False
    chunks = []
    for block in raw.decode("utf-8").split("\n\n"):
        line = block.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            done = True
        elif payload:
            chunks.append(json.loads(payload))
    return chunks, done


class TestRequestTranslation(unittest.TestCase):

    def test_request_hoists_system(self):
        req = openai_to_anthropic_request({
            "model": "claude-opus-4-7",
            "messages": [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "hi"},
            ],
        })
        self.assertEqual(req["system"], "You are terse.")
        self.assertEqual([m["role"] for m in req["messages"]], ["user"])
        self.assertEqual(req["messages"][0]["content"], "hi")

    def test_request_defaults_max_tokens(self):
        req = openai_to_anthropic_request({"model": "m", "messages": []})
        self.assertEqual(req["max_tokens"], 4096)
        req2 = openai_to_anthropic_request({"model": "m", "messages": [], "max_tokens": 256})
        self.assertEqual(req2["max_tokens"], 256)

    def test_request_passes_sampling_and_stop_and_stream(self):
        req = openai_to_anthropic_request({
            "model": "m", "messages": [{"role": "user", "content": "x"}],
            "temperature": 0.3, "top_p": 0.9, "stop": "STOP", "stream": True,
        })
        self.assertEqual(req["temperature"], 0.3)
        self.assertEqual(req["top_p"], 0.9)
        self.assertEqual(req["stop_sequences"], ["STOP"])
        self.assertTrue(req["stream"])

    def test_request_maps_tool_role_to_tool_result(self):
        req = openai_to_anthropic_request({"model": "m", "messages": [
            {"role": "tool", "tool_call_id": "call_9", "content": "42"},
        ]})
        block = req["messages"][0]["content"][0]
        self.assertEqual(req["messages"][0]["role"], "user")
        self.assertEqual(block["type"], "tool_result")
        self.assertEqual(block["tool_use_id"], "call_9")
        self.assertEqual(block["content"], "42")

    def test_request_maps_assistant_tool_calls_to_tool_use(self):
        req = openai_to_anthropic_request({"model": "m", "messages": [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": '{"city":"Cape Town"}'}},
            ]},
        ]})
        block = req["messages"][0]["content"][0]
        self.assertEqual(block["type"], "tool_use")
        self.assertEqual(block["id"], "call_1")
        self.assertEqual(block["name"], "get_weather")
        self.assertEqual(block["input"], {"city": "Cape Town"})

    def test_request_translates_tools_and_choice(self):
        req = openai_to_anthropic_request({
            "model": "m", "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "function", "function": {
                "name": "f", "description": "d",
                "parameters": {"type": "object", "properties": {"a": {"type": "string"}}}}}],
            "tool_choice": "required",
        })
        self.assertEqual(req["tools"][0]["name"], "f")
        self.assertEqual(req["tools"][0]["input_schema"]["properties"]["a"]["type"], "string")
        self.assertEqual(req["tool_choice"], {"type": "any"})

    def test_request_translates_image_part(self):
        req = openai_to_anthropic_request({"model": "m", "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}},
            ]},
        ]})
        blocks = req["messages"][0]["content"]
        self.assertEqual(blocks[0], {"type": "text", "text": "what is this"})
        self.assertEqual(blocks[1]["type"], "image")
        self.assertEqual(blocks[1]["source"],
                         {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"})


class TestResponseTranslation(unittest.TestCase):

    def test_finish_reason_mapping(self):
        cases = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length",
                 "tool_use": "tool_calls", "refusal": "content_filter", "weird": "stop"}
        for stop_reason, expected in cases.items():
            resp = anthropic_to_openai_response(
                {"content": [{"type": "text", "text": "x"}], "stop_reason": stop_reason,
                 "usage": {"input_tokens": 1, "output_tokens": 1}},
                model="m", completion_id="cid", created=1)
            self.assertEqual(resp["choices"][0]["finish_reason"], expected,
                             f"{stop_reason} should map to {expected}")

    def test_response_maps_usage(self):
        resp = anthropic_to_openai_response(
            {"content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
             "usage": {"input_tokens": 12, "output_tokens": 7}},
            model="m", completion_id="cid", created=99)
        self.assertEqual(resp["usage"], {"prompt_tokens": 12, "completion_tokens": 7,
                                         "total_tokens": 19})
        self.assertEqual(resp["object"], "chat.completion")
        self.assertEqual(resp["choices"][0]["message"]["content"], "hi")

    def test_response_tool_use_becomes_tool_calls(self):
        resp = anthropic_to_openai_response(
            {"content": [
                {"type": "text", "text": "calling"},
                {"type": "tool_use", "id": "tu_1", "name": "search", "input": {"q": "x"}},
             ], "stop_reason": "tool_use", "usage": {}},
            model="m", completion_id="cid", created=1)
        msg = resp["choices"][0]["message"]
        self.assertEqual(msg["tool_calls"][0]["id"], "tu_1")
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "search")
        self.assertEqual(json.loads(msg["tool_calls"][0]["function"]["arguments"]), {"q": "x"})
        self.assertEqual(resp["choices"][0]["finish_reason"], "tool_calls")


class TestStreamTranslation(unittest.TestCase):

    _ANTHROPIC_SSE = (
        'event: message_start\n'
        'data: {"type":"message_start","message":{"id":"msg_1","model":"claude-opus-4-7","usage":{"input_tokens":10}}}\n\n'
        'event: content_block_start\n'
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
        'event: content_block_delta\n'
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n'
        'event: content_block_delta\n'
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}\n\n'
        'event: content_block_stop\n'
        'data: {"type":"content_block_stop","index":0}\n\n'
        'event: message_delta\n'
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}\n\n'
        'event: message_stop\n'
        'data: {"type":"message_stop"}\n\n'
    ).encode("utf-8")

    def test_stream_translates_text_and_finish(self):
        out = b"".join(translate_sse_stream(
            [self._ANTHROPIC_SSE], model="m", completion_id="cid", created=1))
        chunks, done = _parse_openai_chunks(out)
        self.assertTrue(done, "must emit [DONE]")
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant"})
        content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        self.assertEqual(content, "Hello world")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        for c in chunks:
            self.assertEqual(c["object"], "chat.completion.chunk")

    def test_stream_handles_chunk_boundaries(self):
        # Split the byte stream into tiny 7-byte fragments — events span chunks.
        raw = self._ANTHROPIC_SSE
        fragments = [raw[i:i + 7] for i in range(0, len(raw), 7)]
        out = b"".join(translate_sse_stream(
            fragments, model="m", completion_id="cid", created=1))
        chunks, done = _parse_openai_chunks(out)
        self.assertTrue(done)
        content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        self.assertEqual(content, "Hello world", "events split across chunks must survive")

    def test_stream_unknown_stop_reason_defaults_stop(self):
        # Ends with message_stop, as every whole Anthropic stream does; a stream
        # that stops short is an interruption now (see test_gateway_error_source).
        sse = (b'event: message_delta\n'
               b'data: {"type":"message_delta","delta":{"stop_reason":"banana"}}\n\n'
               b'event: message_stop\n'
               b'data: {"type":"message_stop"}\n\n')
        out = b"".join(translate_sse_stream([sse], model="m", completion_id="c", created=1))
        chunks, _ = _parse_openai_chunks(out)
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_guarded_sse_events_yields_the_shape_upstream_error_text_reads(self):
        # A mid-stream read failure becomes one {"type": "error"} event whose
        # "error" dict carries the "type" and "message" keys upstream_error_text
        # reads, so translate_sse_stream never raises on it (Step 5a; AGORA C3).
        def chunks():
            yield (b'event: message_start\n'
                   b'data: {"type":"message_start","message":{"id":"m","model":"x","usage":{"input_tokens":1}}}\n\n')
            raise OSError("mid-stream")

        events = list(guarded_sse_events(chunks()))
        self.assertTrue(events)
        self.assertEqual(events[-1]["type"], "error")
        error = events[-1].get("error")
        self.assertIsInstance(error, dict)
        self.assertEqual(error.get("type"), "api_error")
        # lost_upstream_message names the exception's type, never its text.
        self.assertIn("OSError", error.get("message", ""))

        # The shape is exactly what upstream_error_text consumes, without raising.
        message, error_type = upstream_error_text(events[-1])
        self.assertEqual(error_type, "api_error")
        self.assertIn("OSError", message)


class TestAuthTranslation(unittest.TestCase):

    def test_bearer_becomes_x_api_key(self):
        out = dict(anthropic_upstream_headers([
            ("Authorization", "Bearer sk-ant-secret123"),
            ("Content-Type", "application/json"),
        ]))
        self.assertEqual(out["x-api-key"], "sk-ant-secret123")
        self.assertEqual(out["anthropic-version"], "2023-06-01")
        self.assertNotIn("authorization", {k.lower() for k in out})

    def test_oauth_subscription_token_stays_a_bearer(self):
        """An sk-ant-oat token must NOT be rewritten into x-api-key.

        Anthropic's OAuth scheme authenticates via Authorization: Bearer; the two schemes
        are not interchangeable, so converting an OAuth token produces a request Anthropic
        rejects. gateway.py:251 has always guarded this on the native /v1/messages rail.
        This rail did not, so a subscription token could not ride /v1/chat/completions or
        /v1/responses at all — and every non-claude harness speaks this rail, which meant
        those harnesses could only ever reach Anthropic through an API key. Found by a
        converge-fleet mapping pool reachability across 24 harnesses, 2026-08-31.
        """
        out = dict(anthropic_upstream_headers([
            ("Authorization", "Bearer sk-ant-oat01-subscription-token"),
        ]))
        self.assertEqual(out["authorization"], "Bearer sk-ant-oat01-subscription-token")
        self.assertNotIn("x-api-key", {k.lower() for k in out})
        self.assertEqual(out["anthropic-version"], "2023-06-01")

    def test_oauth_guard_matches_the_native_rail(self):
        """The two rails must not disagree about the same token.

        Asserts behavioural parity rather than duplicating the predicate, so a change to
        one rail's rule that is not made to the other shows up here.
        """
        # The native rail's helper is private, and reaching for it is deliberate: the
        # invariant under test is that two rails agree about one token, which cannot be
        # expressed through public API alone. If it is ever renamed, this test failing is
        # the correct outcome — it is exactly the drift it exists to catch.
        from lib.hal.gateway import _normalise_anthropic_auth as native

        for token in ("sk-ant-oat01-abc", "sk-ant-api03-abc"):
            hdrs = [("Authorization", f"Bearer {token}")]
            mine = {k.lower() for k, _ in anthropic_upstream_headers(hdrs)}
            theirs = {k.lower() for k, _ in native(hdrs)}
            self.assertEqual(
                "x-api-key" in mine, "x-api-key" in theirs,
                f"rails disagree on {token[:14]}...: openai-compat={mine}, native={theirs}",
            )

    def test_oauth_beta_header_survives(self):
        out = dict(anthropic_upstream_headers([
            ("Authorization", "Bearer sk-ant-oat01-x"),
            ("anthropic-beta", "oauth-2025-04-20"),
        ]))
        self.assertEqual(out["anthropic-beta"], "oauth-2025-04-20")

    def test_explicit_x_api_key_wins(self):
        out = dict(anthropic_upstream_headers([
            ("x-api-key", "sk-explicit"),
            ("Authorization", "Bearer sk-bearer"),
        ]))
        self.assertEqual(out["x-api-key"], "sk-explicit")

    def test_anthropic_beta_preserved(self):
        out = dict(anthropic_upstream_headers([
            ("Authorization", "Bearer k"),
            ("anthropic-beta", "context-1m-2025-08-07"),
        ]))
        self.assertEqual(out["anthropic-beta"], "context-1m-2025-08-07")


class TestHandlerEndToEnd(unittest.TestCase):

    def _anthropic_ok(self):
        return FakeUpstream(200, json.dumps({
            "id": "msg_x", "model": "claude-opus-4-7",
            "content": [{"type": "text", "text": "Cape Town is sunny."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 8, "output_tokens": 5},
        }).encode())

    def _handler(self, payload, key="sk-ant-secret999"):
        return FakeHandler(headers={
            "Content-Length": str(len(json.dumps(payload))),
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }, body=json.dumps(payload).encode())

    def test_nonstream_roundtrip_and_translated_request(self):
        proxy = FakeProxy(self._anthropic_ok())
        handler = self._handler({
            "model": "claude-opus-4-7", "stream": False,
            "messages": [{"role": "system", "content": "be terse"},
                         {"role": "user", "content": "weather?"}],
        })
        handle_chat_completions(handler, proxy)
        # client got an OpenAI ChatCompletion
        self.assertEqual(handler.status, 200)
        body = handler.body_json()
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"]["content"], "Cape Town is sunny.")
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        # upstream got a correctly translated Anthropic request, at /v1/messages
        call = proxy.calls[0]
        self.assertEqual(call["path"], "/v1/messages")
        sent = json.loads(call["body"])
        self.assertEqual(sent["system"], "be terse")
        self.assertEqual([m["role"] for m in sent["messages"]], ["user"])
        # Bearer was translated to x-api-key on the upstream call
        self.assertEqual(dict(call["headers"])["x-api-key"], "sk-ant-secret999")

    def test_key_never_in_client_output(self):
        proxy = FakeProxy(self._anthropic_ok())
        handler = self._handler({"model": "m", "messages": [{"role": "user", "content": "x"}]},
                                key="sk-ant-LEAK-7777")
        handle_chat_completions(handler, proxy)
        # relayed upstream (must reach Anthropic) ...
        self.assertEqual(dict(proxy.calls[0]["headers"])["x-api-key"], "sk-ant-LEAK-7777")
        # ... but never visible to the client
        self.assertNotIn("sk-ant-LEAK-7777", handler.client_text())

    def test_streaming_roundtrip(self):
        sse = TestStreamTranslation._ANTHROPIC_SSE
        proxy = FakeProxy(FakeUpstream(200, chunks=[sse]))
        handler = self._handler({"model": "m", "stream": True,
                                 "messages": [{"role": "user", "content": "hi"}]})
        handle_chat_completions(handler, proxy)
        self.assertEqual(handler.status, 200)
        self.assertIn(("Content-Type", "text/event-stream"), handler.sent_headers)
        chunks, done = _parse_openai_chunks(handler.wfile.getvalue())
        self.assertTrue(done)
        content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        self.assertEqual(content, "Hello world")

    def test_failopen_malformed_body(self):
        proxy = FakeProxy(self._anthropic_ok())
        handler = FakeHandler(headers={"Content-Length": "7"}, body=b"NOTJSON")
        handle_chat_completions(handler, proxy)
        self.assertEqual(handler.status, 400)
        self.assertIn("error", handler.body_json())
        self.assertEqual(len(proxy.calls), 0, "must not forward an untranslatable body")

    def test_upstream_error_wrapped_as_openai(self):
        # Inverted 2026-10-05 (InfiniteMachine v2, Step 5a): this asserted the
        # 429 went out as a 429, which ends a Codex turn outright (fleet finding
        # f26). The chat route now sends it as a 503 with Retry-After kept.
        proxy = FakeProxy(FakeUpstream(429, json.dumps({
            "type": "error", "error": {"type": "rate_limit", "message": "slow down"}}).encode(),
            headers=[("retry-after", "7")]))
        handler = self._handler({"model": "m", "messages": [{"role": "user", "content": "x"}]})
        handle_chat_completions(handler, proxy)
        self.assertEqual(handler.status, 503)
        self.assertIn(("retry-after", "7"), handler.sent_headers)
        self.assertEqual(handler.body_json()["error"]["message"], "slow down")

    def test_upstream_socket_closed_on_every_path(self):
        # Refutes the AGORA C4 leak concern: every relay path closes upstream.
        ok = self._anthropic_ok()
        handle_chat_completions(
            self._handler({"model": "m", "messages": [{"role": "user", "content": "x"}]}),
            FakeProxy(ok))
        self.assertTrue(ok.closed, "non-stream relay must close upstream")

        streamed = FakeUpstream(200, chunks=[TestStreamTranslation._ANTHROPIC_SSE])
        handle_chat_completions(
            self._handler({"model": "m", "stream": True,
                           "messages": [{"role": "user", "content": "x"}]}),
            FakeProxy(streamed))
        self.assertTrue(streamed.closed, "stream relay must close upstream")

        err = FakeUpstream(429, json.dumps({"error": {"message": "x"}}).encode())
        handle_chat_completions(
            self._handler({"model": "m", "messages": [{"role": "user", "content": "x"}]}),
            FakeProxy(err))
        self.assertTrue(err.closed, "error relay must close upstream")


class TestWiring(unittest.TestCase):

    def test_chat_completions_path_constant(self):
        self.assertEqual(CHAT_COMPLETIONS_PATH, "/v1/chat/completions")

    def test_route_registered_in_server(self):
        src = open(_SERVER_SRC, encoding="utf-8").read()
        self.assertIn("handle_chat_completions", src)
        self.assertIn("/v1/chat/completions", src)

    def test_route_registered_in_standalone_gateway(self):
        src = open(_GATEWAY_SRC, encoding="utf-8").read()
        self.assertIn("/v1/chat/completions", src)


if __name__ == "__main__":
    unittest.main()
