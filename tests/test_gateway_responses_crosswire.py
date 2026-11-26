"""Regression: /v1/responses cross-wire (HAL_GATEWAY_MULTIPROVIDER) must TRANSLATE.

Before the fix, handle_responses forwarded the raw *Responses* body verbatim to a
provider's non-existent /responses endpoint and relayed it unchanged — deepseek/
gemini (Chat-Completions-only) returned empty, so codex 0.136 (Responses-wire-only)
could not use any non-Anthropic provider through the gateway.

The fix pivots through Anthropic shape: Responses -> Anthropic (responses_to_
anthropic_request) -> provider chat -> Anthropic (cross_wire_forward) -> Responses
(_relay_nonstream/_relay_stream). These tests are Goodhart-resistant — each fails
under the obvious regression:

  - reverts to forwarding the raw Responses body  -> test_crosswire_translates_request
  - drops the Responses-shape output translation   -> test_crosswire_output_is_responses_shaped
  - loses the client's stream flag                 -> test_crosswire_propagates_stream_flag
  - accidentally cross-wires a claude model (#3274) -> test_claude_model_takes_anthropic_path
"""
import io
import json
import unittest
from unittest import mock

from lib.hal.gateway_responses import handle_responses


class _FakeUpstream:
    def __init__(self, status, body=b""):
        self.status = status
        self._body = body
        self.headers = []
        self.closed = False

    def read_all(self):
        return self._body

    def iter_chunks(self, size=65536):
        if self._body:
            yield self._body

    def close(self):
        self.closed = True


class _FakeProxy:
    def __init__(self, result):
        self._result = result
        self.calls = []

    def forward(self, path, body, client_headers, *, method="POST", max_retries=None):
        self.calls.append({"path": path, "body": body, "method": method})
        return self._result


class _FakeHandler:
    def __init__(self, payload):
        raw = json.dumps(payload).encode()
        self.command = "POST"
        self.path = "/v1/responses"
        self.headers = {
            "Content-Length": str(len(raw)),
            "Authorization": "Bearer sk-ant-test",
            "Content-Type": "application/json",
        }
        self.rfile = io.BytesIO(raw)
        self.wfile = io.BytesIO()
        self.status = None
        self.sent_headers = []

    def send_response(self, code):
        self.status = code

    def send_header(self, k, v):
        self.sent_headers.append((k, v))

    def end_headers(self):
        pass

    def out(self):
        return self.wfile.getvalue().decode("utf-8", "replace")


# A minimal but valid Anthropic Messages response — what cross_wire_forward
# returns after translating the provider's chat completion back to Anthropic shape.
_ANTHROPIC_JSON = json.dumps({
    "id": "msg_x", "type": "message", "role": "assistant", "model": "deepseek-chat",
    "content": [{"type": "text", "text": "PONG"}],
    "stop_reason": "end_turn", "usage": {"input_tokens": 5, "output_tokens": 1},
}).encode("utf-8")


class CrossWireResponsesTest(unittest.TestCase):
    def test_crosswire_translates_request(self):
        """A same-wire model routes through cross_wire_forward with a TRANSLATED
        Anthropic request (messages[], model preserved) — NOT the raw Responses body."""
        captured = {}

        def fake_cwf(req, *, provider, want_stream, **kw):
            captured["req"] = req
            captured["provider"] = provider
            return _FakeUpstream(200, _ANTHROPIC_JSON)

        h = _FakeHandler({"model": "deepseek-chat", "input": "ping", "stream": False})
        with mock.patch("lib.hal.gateway_egress.same_wire_provider", return_value="deepseek"), \
             mock.patch("lib.hal.gateway_egress.cross_wire_forward", side_effect=fake_cwf):
            handle_responses(h, _FakeProxy(None))

        # Routed to the right provider …
        self.assertEqual(captured["provider"], "deepseek")
        # … with a TRANSLATED request: Anthropic shape has `messages`, NOT the raw
        # Responses `input` field (the exact bug — raw body was forwarded before).
        self.assertIn("messages", captured["req"])
        self.assertNotIn("input", captured["req"])
        self.assertEqual(captured["req"].get("model"), "deepseek-chat")

    def test_crosswire_output_is_responses_shaped(self):
        """The provider's (Anthropic-shaped) answer is translated BACK to Responses
        shape before reaching the client — proving the relay translator ran."""
        h = _FakeHandler({"model": "deepseek-chat", "input": "ping", "stream": False})
        with mock.patch("lib.hal.gateway_egress.same_wire_provider", return_value="deepseek"), \
             mock.patch("lib.hal.gateway_egress.cross_wire_forward",
                        return_value=_FakeUpstream(200, _ANTHROPIC_JSON)):
            handle_responses(h, _FakeProxy(None))
        body = json.loads(h.out())
        self.assertEqual(body["object"], "response")          # Responses shape, not Anthropic
        self.assertEqual(body["output"][0]["content"][0]["type"], "output_text")
        self.assertEqual(body["output"][0]["content"][0]["text"], "PONG")

    def test_crosswire_propagates_stream_flag(self):
        """The client's stream flag reaches cross_wire_forward (so the provider is
        called streamed iff the client streamed)."""
        seen = {}

        def fake_cwf(req, *, provider, want_stream, **kw):
            seen["want_stream"] = want_stream
            return _FakeUpstream(200, _ANTHROPIC_JSON)

        h = _FakeHandler({"model": "deepseek-chat", "input": "ping", "stream": True})
        with mock.patch("lib.hal.gateway_egress.same_wire_provider", return_value="deepseek"), \
             mock.patch("lib.hal.gateway_egress.cross_wire_forward", side_effect=fake_cwf):
            handle_responses(h, _FakeProxy(None))
        self.assertTrue(seen["want_stream"])

    def test_claude_model_takes_anthropic_path(self):
        """A claude model (same_wire_provider -> None) must NOT cross-wire; it goes
        verbatim to Anthropic /v1/messages (the #3274 path must stay untouched)."""
        def boom(*a, **k):  # cross_wire_forward must never be called for claude
            raise AssertionError("claude model was cross-wired — #3274 path broken")

        proxy = _FakeProxy(_FakeUpstream(200, _ANTHROPIC_JSON))
        h = _FakeHandler({"model": "claude-opus-4-8", "input": "ping", "stream": False})
        with mock.patch("lib.hal.gateway_egress.same_wire_provider", return_value=None), \
             mock.patch("lib.hal.gateway_egress.cross_wire_forward", side_effect=boom):
            handle_responses(h, proxy)
        self.assertEqual(len(proxy.calls), 1)
        self.assertEqual(proxy.calls[0]["path"], "/v1/messages")

    def test_openai_responses_forwarded_verbatim(self):
        """OpenAI speaks the Responses API natively → the ORIGINAL Responses body is
        forwarded VERBATIM to /v1/responses (no Anthropic pivot). Anchors BOTH gateway
        bugs this fixed: the gpt-5.5->claude-opus-4-8 resolve_model alias hijack, and
        the max_tokens the Responses->Anthropic pivot injected (gpt-5.x rejects it)."""
        captured = {}

        def fake_fsw(provider, gateway_path, raw_body, **kw):
            captured["provider"] = provider
            captured["path"] = gateway_path
            captured["body"] = raw_body
            return _FakeUpstream(200, b'{"object":"response","output":[]}')

        def boom_cwf(*a, **k):  # openai must NOT be pivoted through cross_wire_forward
            raise AssertionError("openai was pivoted — verbatim Responses path broken")

        h = _FakeHandler({"model": "gpt-5.5", "input": "ping"})
        with mock.patch("lib.hal.gateway_egress.same_wire_provider", return_value="openai"), \
             mock.patch("lib.hal.gateway_egress.forward_same_wire", side_effect=fake_fsw), \
             mock.patch("lib.hal.gateway_egress.cross_wire_forward", side_effect=boom_cwf):
            handle_responses(h, _FakeProxy(None))

        self.assertEqual(captured["provider"], "openai")
        self.assertEqual(captured["path"], "/v1/responses")
        sent = json.loads(captured["body"])
        self.assertEqual(sent["model"], "gpt-5.5")     # ORIGINAL id, not the claude alias
        self.assertNotIn("claude", json.dumps(sent))
        self.assertIn("input", sent)                    # raw Responses field, untranslated
        self.assertNotIn("max_tokens", sent)            # pivot's max_tokens never injected


if __name__ == "__main__":
    unittest.main()
