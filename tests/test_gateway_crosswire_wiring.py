"""Tests for cross-wire wiring (Phase 0 wave 4b).

Two surfaces, both hermetic (opener + key_resolver DIP-injected; the routing
gate patched to a spy so zero network):

  * ``cross_wire_forward`` — Anthropic request -> OpenAI provider -> Anthropic
    response (buffered JSON, or synthesised SSE when the client streamed); a
    provider error becomes an Anthropic error envelope, never a 200.
  * ``gateway._maybe_cross_wire`` — the decision gate. The load-bearing #3274
    guard is two-sided: flag-OFF NEVER cross-wires (and never parses the body),
    and flag-ON STILL never cross-wires claude-*/unknown/GET/non-/v1/messages —
    so the verbatim relay is reached unchanged in every one of those cases.
"""
import io
import json
import os
import unittest
from contextlib import contextmanager
from unittest import mock

from lib.hal.gateway import UpstreamResponse, _maybe_cross_wire, handle_proxy_request
from lib.hal.gateway_egress import cross_wire_forward

_FLAG = "HAL_GATEWAY_MULTIPROVIDER"


@contextmanager
def _flag(value):
    prev = os.environ.get(_FLAG)
    if value is None:
        os.environ.pop(_FLAG, None)
    else:
        os.environ[_FLAG] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(_FLAG, None)
        else:
            os.environ[_FLAG] = prev


class _FakeResp:
    def __init__(self, status=200, headers=None, body=b"{}"):
        self.status = status
        self.headers = headers or [("Content-Type", "application/json")]
        self._body = body
        self._read = False

    def read(self, size=-1):
        if self._read:
            return b""
        self._read = True
        return self._body

    def close(self):
        pass


_OAI_OK = json.dumps({
    "id": "chatcmpl-1", "object": "chat.completion", "model": "gpt-4o",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "Hi from OpenAI"}}],
    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
}).encode("utf-8")


class TestCrossWireForward(unittest.TestCase):

    def _capture_opener(self, resp):
        captured = {}

        def opener(method, url, headers, body):
            captured.update(method=method, url=url, headers=headers, body=body)
            return resp

        return opener, captured

    def test_non_stream_returns_anthropic_json(self):
        opener, captured = self._capture_opener(_FakeResp(body=_OAI_OK))
        resp = cross_wire_forward(
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            provider="openai", want_stream=False,
            opener=opener, key_resolver=lambda s: "sk-x",
        )
        self.assertEqual(resp.status, 200)
        hdrs = dict(resp.headers)
        self.assertEqual(hdrs["Content-Type"], "application/json")
        msg = json.loads(resp.read_all())
        self.assertEqual(msg["type"], "message")
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["content"], [{"type": "text", "text": "Hi from OpenAI"}])
        self.assertEqual(msg["stop_reason"], "end_turn")
        self.assertEqual(msg["usage"], {"input_tokens": 7, "output_tokens": 3})

    def test_forwards_to_chat_completions_with_provider_key_and_stream_flag(self):
        opener, captured = self._capture_opener(_FakeResp(body=_OAI_OK))
        cross_wire_forward(
            {"model": "gpt-4o", "system": "be terse",
             "messages": [{"role": "user", "content": "hi"}], "stream": True},
            provider="openai", want_stream=True,
            opener=opener, key_resolver=lambda s: "sk-PROVIDER",
        )
        self.assertEqual(captured["url"], "https://api.openai.com/v1/chat/completions")
        self.assertEqual(dict(captured["headers"])["Authorization"], "Bearer sk-PROVIDER")
        sent = json.loads(captured["body"])
        self.assertIs(sent["stream"], True)  # provider streamed iff the client streamed
        self.assertEqual(sent["messages"][0], {"role": "system", "content": "be terse"})

    def test_non_stream_client_calls_provider_non_streamed(self):
        opener, captured = self._capture_opener(_FakeResp(body=_OAI_OK))
        cross_wire_forward(
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            provider="openai", want_stream=False,
            opener=opener, key_resolver=lambda s: "sk-x",
        )
        self.assertIs(json.loads(captured["body"])["stream"], False)

    def test_stream_returns_sse(self):
        opener, _ = self._capture_opener(_FakeResp(body=_OAI_OK))
        resp = cross_wire_forward(
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            provider="openai", want_stream=True,
            opener=opener, key_resolver=lambda s: "sk-x",
        )
        self.assertEqual(dict(resp.headers)["Content-Type"], "text/event-stream")
        body = resp.read_all()
        self.assertTrue(body.startswith(b"event: message_start"))
        self.assertIn(b"event: message_stop", body)
        self.assertIn(b"Hi from OpenAI", body)

    def test_provider_error_becomes_anthropic_error_not_200(self):
        err_body = json.dumps({"error": {"message": "rate limited"}}).encode("utf-8")
        opener, _ = self._capture_opener(_FakeResp(status=429, body=err_body))
        resp = cross_wire_forward(
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            provider="openai", want_stream=True,  # error is JSON even when streaming
            opener=opener, key_resolver=lambda s: "sk-x",
        )
        self.assertEqual(resp.status, 429)
        self.assertEqual(dict(resp.headers)["Content-Type"], "application/json")
        env = json.loads(resp.read_all())
        self.assertEqual(env["type"], "error")
        self.assertIn("rate limited", env["error"]["message"])

    def test_empty_key_raises_fail_loud(self):
        opener, captured = self._capture_opener(_FakeResp(body=_OAI_OK))
        with self.assertRaises(RuntimeError):
            cross_wire_forward(
                {"model": "gpt-4o", "messages": []},
                provider="openai", want_stream=False,
                opener=opener, key_resolver=lambda s: "",
            )
        self.assertEqual(captured, {})  # never reached the opener


class TestMaybeCrossWireGate(unittest.TestCase):
    """The decision gate; the spy proves whether cross_wire_forward is invoked."""

    def _body(self, **kw):
        return json.dumps(kw).encode("utf-8")

    @contextmanager
    def _spy(self):
        sentinel = object()
        calls = []

        def fake(payload, *, provider, want_stream, **_):
            calls.append({"provider": provider, "want_stream": want_stream,
                          "model": payload.get("model")})
            return sentinel
        with mock.patch("lib.hal.gateway_egress.cross_wire_forward", fake):
            yield sentinel, calls

    def test_flag_off_never_cross_wires_even_gpt(self):
        with _flag("0"), self._spy() as (_sentinel, calls):
            self.assertIsNone(
                _maybe_cross_wire("/v1/messages", "POST", self._body(model="gpt-5.5")))
            self.assertEqual(calls, [], "flag-off must not invoke cross_wire_forward")

    def test_flag_on_claude_stays_verbatim(self):
        with _flag("1"), self._spy() as (_sentinel, calls):
            for m in ["claude-opus-4-8", "claude-sonnet-4-6", "claude3opus"]:
                self.assertIsNone(
                    _maybe_cross_wire("/v1/messages", "POST", self._body(model=m)), m)
            self.assertEqual(calls, [], "#3274: claude-* must never cross-wire")

    def test_flag_on_unknown_model_stays_verbatim(self):
        with _flag("1"), self._spy() as (_sentinel, calls):
            self.assertIsNone(
                _maybe_cross_wire("/v1/messages", "POST", self._body(model="made-up-9000")))
            self.assertEqual(calls, [])

    def test_get_never_cross_wires(self):
        with _flag("1"), self._spy() as (_sentinel, calls):
            self.assertIsNone(_maybe_cross_wire("/v1/models", "GET", None))
            self.assertEqual(calls, [])

    def test_non_messages_path_never_cross_wires(self):
        with _flag("1"), self._spy() as (_sentinel, calls):
            self.assertIsNone(
                _maybe_cross_wire("/v1/responses", "POST", self._body(model="gpt-5.5")))
            self.assertEqual(calls, [])

    def test_unparseable_body_falls_through(self):
        with _flag("1"), self._spy() as (_sentinel, calls):
            self.assertIsNone(_maybe_cross_wire("/v1/messages", "POST", b"not json{"))
            self.assertEqual(calls, [])

    def test_flag_on_gpt_cross_wires_to_openai(self):
        with _flag("1"), self._spy() as (sentinel, calls):
            result = _maybe_cross_wire(
                "/v1/messages", "POST", self._body(model="gpt-5.5", stream=True))
            self.assertIs(result, sentinel)
            self.assertEqual(calls, [{"provider": "openai", "want_stream": True,
                                      "model": "gpt-5.5"}])

    def test_want_stream_false_propagates(self):
        with _flag("1"), self._spy() as (_sentinel, calls):
            _maybe_cross_wire("/v1/messages", "POST", self._body(model="gemini-3.1-pro"))
            self.assertEqual(calls[0]["provider"], "gemini")
            self.assertIs(calls[0]["want_stream"], False)

    def test_query_string_on_path_still_matches(self):
        with _flag("1"), self._spy() as (sentinel, calls):
            result = _maybe_cross_wire(
                "/v1/messages?beta=true", "POST", self._body(model="gpt-5.5"))
            self.assertIs(result, sentinel)
            self.assertEqual(calls[0]["provider"], "openai")

    # --- Anthropic same-wire (Kimi plan W2): a provider with its own Anthropic
    # endpoint is served verbatim there; cross-wire is the fallback for the rest.

    @contextmanager
    def _same_wire_spy(self):
        sentinel = object()
        calls = []

        def fake(provider, gateway_path, raw_body, client_headers, **_):
            calls.append({"provider": provider, "path": gateway_path,
                          "body": raw_body, "headers": list(client_headers)})
            return sentinel
        with mock.patch("lib.hal.gateway_egress.anthropic_same_wire_forward", fake):
            yield sentinel, calls

    def test_flag_on_kimi_takes_anthropic_same_wire_not_cross_wire(self):
        body = self._body(model="kimi-k3[1m]", stream=True)
        hdrs = [("anthropic-beta", "context-1m-2025-08-07")]
        with _flag("1"), self._spy() as (_cw, cw_calls), self._same_wire_spy() as (sw, sw_calls):
            result = _maybe_cross_wire("/v1/messages?beta=true", "POST", body, hdrs)
            self.assertIs(result, sw)
            self.assertEqual(cw_calls, [], "kimi must NOT go through the translator")
            self.assertEqual(sw_calls, [{"provider": "moonshot", "path": "/v1/messages?beta=true",
                                         "body": body, "headers": hdrs}])

    def test_flag_on_gpt_and_gemini_still_cross_wire_never_same_wire(self):
        with _flag("1"), self._spy() as (cw, cw_calls), self._same_wire_spy() as (_sw, sw_calls):
            for m in ("gpt-5.5", "gemini-3.1-pro"):
                self.assertIs(_maybe_cross_wire("/v1/messages", "POST", self._body(model=m)), cw)
            self.assertEqual([c["provider"] for c in cw_calls], ["openai", "gemini"])
            self.assertEqual(sw_calls, [])

    def test_flag_on_claude_stays_verbatim_with_same_wire_present(self):
        with _flag("1"), self._spy() as (_cw, cw_calls), self._same_wire_spy() as (_sw, sw_calls):
            self.assertIsNone(
                _maybe_cross_wire("/v1/messages", "POST", self._body(model="claude-opus-5[1m]")))
            self.assertEqual((cw_calls, sw_calls), ([], []))

    def test_flag_off_kimi_stays_verbatim(self):
        with _flag("0"), self._spy() as (_cw, cw_calls), self._same_wire_spy() as (_sw, sw_calls):
            self.assertIsNone(_maybe_cross_wire("/v1/messages", "POST", self._body(model="kimi-k3")))
            self.assertEqual((cw_calls, sw_calls), ([], []))


class _FakeHandler:
    """Minimal BaseHTTPRequestHandler stand-in capturing the client-facing write."""

    def __init__(self, command="POST", body=b"", headers=None):
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


class TestHandleProxyRequestCrossWire(unittest.TestCase):
    """The wiring in handle_proxy_request: relay unified, exceptions -> 502,
    flag-off uses the verbatim proxy (the #3274 path) untouched."""

    def _handler(self, payload):
        body = json.dumps(payload).encode("utf-8")
        return _FakeHandler(command="POST", body=body,
                            headers={"Content-Length": str(len(body))})

    def test_cross_wire_exception_maps_to_502_no_leak(self):
        # An empty-key RuntimeError (or any translator failure) must become a
        # clean 502, not an unhandled crash — and never leak a key.
        def boom(*a, **k):
            raise RuntimeError("no API key for provider 'openai' (sk-secret-LEAK)")
        handler = self._handler({"model": "gpt-5.5", "messages": []})
        with _flag("1"), mock.patch("lib.hal.gateway_egress.cross_wire_forward", boom):
            handle_proxy_request(handler, mock.Mock(), "/v1/messages")
        self.assertEqual(handler.status, 502)
        out = handler.wfile.getvalue().decode("utf-8")
        self.assertIn("error", out)
        self.assertNotIn("sk-secret-LEAK", out)  # redacted

    def test_cross_wire_success_is_relayed(self):
        synthetic = UpstreamResponse(
            status=200, headers=[("Content-Type", "application/json")],
            _reader=io.BytesIO(b'{"type":"message"}'))
        handler = self._handler({"model": "gpt-5.5", "messages": []})
        proxy = mock.Mock()
        with _flag("1"), mock.patch("lib.hal.gateway_egress.cross_wire_forward",
                                    lambda *a, **k: synthetic):
            handle_proxy_request(handler, proxy, "/v1/messages")
        self.assertEqual(handler.status, 200)
        self.assertIn(b'"type":"message"', handler.wfile.getvalue())
        proxy.forward.assert_not_called()  # cross-wire bypasses the verbatim relay

    def test_flag_off_uses_verbatim_proxy_untouched(self):
        # #3274: flag off -> cross_wire_forward never invoked; proxy.forward IS.
        handler = self._handler({"model": "gpt-5.5", "messages": []})
        proxy = mock.Mock()
        proxy.forward.return_value = UpstreamResponse(
            200, [("Content-Type", "application/json")], io.BytesIO(b'{"ok":1}'))
        cw = mock.Mock()
        with _flag("0"), mock.patch("lib.hal.gateway_egress.cross_wire_forward", cw):
            handle_proxy_request(handler, proxy, "/v1/messages")
        cw.assert_not_called()
        proxy.forward.assert_called_once()
        self.assertEqual(handler.status, 200)

    def test_same_wire_success_is_relayed_with_client_headers_and_witnessed(self):
        # The handler's headers must reach the forwarder (declared input reaches
        # the actor), the verbatim proxy is bypassed, and the witness rail names
        # the served provider. failed_over is False: this asserted True until
        # 2026-10-05, which demanded the defect -- serving the model the caller
        # asked for is routing, and this rail never knows of a failover.
        synthetic = UpstreamResponse(
            status=200, headers=[("Content-Type", "application/json")],
            _reader=io.BytesIO(b'{"type":"message","model":"kimi-k3"}'))
        seen = {}

        def fake(provider, gateway_path, raw_body, client_headers, **_):
            seen.update(provider=provider, path=gateway_path, headers=dict(client_headers))
            return synthetic
        body = json.dumps({"model": "kimi-k3", "messages": []}).encode("utf-8")
        handler = _FakeHandler(command="POST", body=body, headers={
            "Content-Length": str(len(body)), "anthropic-beta": "context-1m-2025-08-07"})
        proxy = mock.Mock()
        witness = mock.Mock(return_value={})
        with _flag("1"), \
                mock.patch("lib.hal.gateway_egress.anthropic_same_wire_forward", fake), \
                mock.patch("lib.hal.gateway_witness.witness_chat", witness):
            handle_proxy_request(handler, proxy, "/v1/messages")
        self.assertEqual(handler.status, 200)
        self.assertIn(b'"model":"kimi-k3"', handler.wfile.getvalue())
        proxy.forward.assert_not_called()
        self.assertEqual(seen["provider"], "moonshot")
        self.assertEqual(seen["headers"]["anthropic-beta"], "context-1m-2025-08-07")
        witness.assert_called_once()
        kw = witness.call_args.kwargs
        self.assertEqual((kw["provider"], kw["model"], kw["failed_over"], kw["status"]),
                         ("moonshot", "kimi-k3", False, 200))

    def test_same_wire_exception_maps_to_502_no_leak(self):
        def boom(*a, **k):
            raise RuntimeError("no API key for provider 'moonshot' (sk-moon-LEAK)")
        handler = self._handler({"model": "kimi-k3", "messages": []})
        with _flag("1"), mock.patch("lib.hal.gateway_egress.anthropic_same_wire_forward", boom):
            handle_proxy_request(handler, mock.Mock(), "/v1/messages")
        self.assertEqual(handler.status, 502)
        out = handler.wfile.getvalue().decode("utf-8")
        self.assertIn("error", out)
        self.assertNotIn("sk-moon-LEAK", out)


class TestCrossWireForwardErrorPaths(unittest.TestCase):

    def _opener(self, resp):
        def op(method, url, headers, body):
            return resp
        return op

    def test_non_json_2xx_returns_502_with_upstream_status(self):
        resp = _FakeResp(status=200, body=b"<html>not json</html>")
        out = cross_wire_forward(
            {"model": "gpt-4o", "messages": []}, provider="openai", want_stream=False,
            opener=self._opener(resp), key_resolver=lambda s: "k")
        self.assertEqual(out.status, 502)
        env = json.loads(out.read_all())
        self.assertEqual(env["type"], "error")
        self.assertIn("200", env["error"]["message"])  # real upstream status kept

    def _oai_length(self):
        return json.dumps({
            "id": "c", "model": "gpt-4o",
            "choices": [{"index": 0, "finish_reason": "length",
                         "message": {"role": "assistant", "content": "truncated"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 9},
        }).encode("utf-8")

    def test_finish_reason_length_maps_to_max_tokens(self):
        # The judge's falsifier: a non-"stop" finish_reason must map correctly.
        out = cross_wire_forward(
            {"model": "gpt-4o", "messages": []}, provider="openai", want_stream=False,
            opener=self._opener(_FakeResp(body=self._oai_length())), key_resolver=lambda s: "k")
        msg = json.loads(out.read_all())
        self.assertEqual(msg["stop_reason"], "max_tokens")
        self.assertEqual(msg["usage"]["output_tokens"], 9)

    def test_finish_reason_length_sse_keeps_usage_in_delta(self):
        out = cross_wire_forward(
            {"model": "gpt-4o", "messages": [], "stream": True},
            provider="openai", want_stream=True,
            opener=self._opener(_FakeResp(body=self._oai_length())), key_resolver=lambda s: "k")
        body = out.read_all()
        self.assertIn(b"max_tokens", body)
        self.assertIn(b'"output_tokens": 9', body)  # message_delta usage never omitted


class TestCrossWireForwardStreaming(unittest.TestCase):
    """want_stream + a provider that actually streams (text/event-stream) ->
    true token-incremental translation; a provider that ignores stream -> the
    buffered-SSE fallback (covered by TestCrossWireForward.test_stream_returns_sse)."""

    def _sse_opener(self, sse_body):
        resp = _FakeResp(status=200, headers=[("Content-Type", "text/event-stream")], body=sse_body)

        def op(method, url, headers, body):
            return resp
        return op

    def test_true_streaming_translates_provider_sse(self):
        oai_sse = (
            b'data: {"id":"c","model":"gpt-4o","choices":[{"index":0,'
            b'"delta":{"content":"Hi"},"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            b'data: [DONE]\n\n')
        resp = cross_wire_forward(
            {"model": "gpt-4o", "messages": [], "stream": True},
            provider="openai", want_stream=True,
            opener=self._sse_opener(oai_sse), key_resolver=lambda s: "k")
        self.assertEqual(dict(resp.headers)["Content-Type"], "text/event-stream")
        body = resp.read_all()
        self.assertTrue(body.startswith(b"event: message_start"))
        self.assertIn(b'"text": "Hi"', body)        # translated text delta
        self.assertIn(b"event: message_stop", body)

    def test_generator_reader_is_lazy_and_chunks(self):
        from lib.hal.gateway_egress import _GeneratorReader
        r = _GeneratorReader(iter([b"abc", b"def", b"ghi"]))
        self.assertEqual(r.read(2), b"ab")
        self.assertEqual(r.read(2), b"cd")
        self.assertEqual(r.read(100), b"efghi")   # drains the rest
        self.assertEqual(r.read(1), b"")          # exhausted

    def test_generator_reader_closes_source(self):
        from lib.hal.gateway_egress import _GeneratorReader
        closed = {"v": False}

        class _Src:
            def close(self):
                closed["v"] = True
        r = _GeneratorReader(iter([b"x"]), _Src())
        r.close()
        self.assertTrue(closed["v"])


if __name__ == "__main__":
    unittest.main()
