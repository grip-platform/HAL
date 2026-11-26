"""``POST /v1/messages/count_tokens`` for an Anthropic-compatible provider.

Moonshot's Anthropic surface has no count_tokens (probed 2026-09-07:
``/anthropic/v1/messages/count_tokens`` -> HTTP 404 ``url.not_found``), so
the gateway answers from the vendor's free tokenizer instead:
``POST /v1/tokenizers/estimate-token-count`` over the request translated to
the OpenAI shape (``data.total_tokens`` 172 with tools vs 102 without on the
probe, so tools ARE counted). Hermetic: opener + key_resolver injected, the
gate spied, zero network.
"""
import json
import os
import unittest
from contextlib import contextmanager
from unittest import mock

from lib.hal.gateway import _maybe_cross_wire
from lib.hal.gateway_egress import anthropic_count_tokens

_FLAG = "HAL_GATEWAY_MULTIPROVIDER"
_COUNT_PATH = "/v1/messages/count_tokens"


@contextmanager
def _flag(value):
    prev = os.environ.get(_FLAG)
    os.environ[_FLAG] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(_FLAG, None)
        else:
            os.environ[_FLAG] = prev


class _FakeResp:
    def __init__(self, status=200, body=b"{}"):
        self.status = status
        self.headers = [("Content-Type", "application/json")]
        self._body = body
        self._read = False

    def read(self, size=-1):
        if self._read:
            return b""
        self._read = True
        return self._body

    def close(self):
        pass


_PAYLOAD = {
    "model": "kimi-k3[1m]",
    "system": [{"type": "text", "text": "You are terse.",
                "cache_control": {"type": "ephemeral"}}],
    "messages": [{"role": "user", "content": [{"type": "text", "text": "Reply OK"}]}],
    "tools": [{"name": "get_weather", "description": "Weather lookup",
               "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}],
    "thinking": {"type": "enabled", "budget_tokens": 1024},
}
_VENDOR_OK = json.dumps({"code": 0, "data": {"total_tokens": 172},
                         "scode": "0x0", "status": True}).encode()


def _count(payload=_PAYLOAD, provider="moonshot", resp=None, key="sk-moon-PROVIDER"):
    calls = []

    def opener(method, url, headers, body):
        calls.append({"method": method, "url": url, "headers": dict(headers),
                      "body": json.loads(body)})
        return resp or _FakeResp(body=_VENDOR_OK)
    out = anthropic_count_tokens(payload, provider=provider, opener=opener,
                                 key_resolver=lambda _s: key)
    return out, calls


class TestCountTokensGate(unittest.TestCase):
    """The path is exact today, so without this a Kimi count went to Anthropic."""

    def _body(self, model):
        return json.dumps({"model": model, "messages": [{"role": "user", "content": "hi"}]}).encode()

    @contextmanager
    def _spies(self):
        sentinel = object()
        seen = {"count": [], "same_wire": [], "cross_wire": []}

        def count(payload, *, provider, **_):
            seen["count"].append((payload.get("model"), provider))
            return sentinel
        with _flag("1"), \
                mock.patch("lib.hal.gateway_egress.anthropic_count_tokens", count), \
                mock.patch("lib.hal.gateway_egress.anthropic_same_wire_forward",
                           lambda *a, **k: seen["same_wire"].append(a)), \
                mock.patch("lib.hal.gateway_egress.cross_wire_forward",
                           lambda *a, **k: seen["cross_wire"].append(a)):
            yield sentinel, seen

    def test_kimi_count_tokens_is_answered_locally_never_same_wired(self):
        with self._spies() as (sentinel, seen):
            result = _maybe_cross_wire(_COUNT_PATH, "POST", self._body("kimi-k3[1m]"))
        self.assertIs(result, sentinel)
        self.assertEqual(seen["count"], [("kimi-k3[1m]", "moonshot")])
        self.assertEqual((seen["same_wire"], seen["cross_wire"]), ([], []))

    def test_query_string_on_count_path_still_matches(self):
        with self._spies() as (sentinel, _seen):
            self.assertIs(_maybe_cross_wire(_COUNT_PATH + "?beta=true", "POST",
                                            self._body("kimi-k3")), sentinel)

    def test_gpt_and_claude_count_tokens_stay_verbatim(self):
        # No cross-wire count exists; the pre-existing verbatim relay is kept.
        with self._spies() as (_sentinel, seen):
            for model in ("gpt-5.5", "claude-opus-5[1m]", "made-up-9000"):
                self.assertIsNone(_maybe_cross_wire(_COUNT_PATH, "POST", self._body(model)), model)
        self.assertEqual(seen, {"count": [], "same_wire": [], "cross_wire": []})

    def test_messages_path_is_not_the_count_path(self):
        with self._spies() as (_sentinel, seen):
            _maybe_cross_wire("/v1/messages", "POST", self._body("kimi-k3"))
        self.assertEqual(seen["count"], [])
        self.assertEqual(len(seen["same_wire"]), 1)


class TestCountTokensAnswer(unittest.TestCase):

    def test_calls_the_vendor_tokenizer_with_the_translated_request(self):
        out, calls = _count()
        self.assertEqual(calls[0]["url"], "https://api.moonshot.ai/v1/tokenizers/estimate-token-count")
        self.assertEqual(calls[0]["headers"]["Authorization"], "Bearer sk-moon-PROVIDER")
        sent = calls[0]["body"]
        self.assertEqual(set(sent), {"model", "messages", "tools"})
        self.assertEqual(sent["model"], "kimi-k3")  # [1m] stripped: 404 otherwise
        self.assertEqual(sent["messages"][0], {"role": "system", "content": "You are terse."})
        self.assertEqual(sent["messages"][1]["role"], "user")
        self.assertEqual(sent["tools"][0]["function"]["name"], "get_weather")
        self.assertEqual(out.status, 200)
        self.assertEqual(json.loads(out.read_all()), {"input_tokens": 172})
        self.assertEqual(dict(out.headers)["Content-Type"], "application/json")

    def test_without_tools_the_tools_key_is_omitted(self):
        payload = {k: v for k, v in _PAYLOAD.items() if k != "tools"}
        _out, calls = _count(payload=payload)
        self.assertEqual(set(calls[0]["body"]), {"model", "messages"})

    def test_vendor_error_becomes_an_anthropic_error_envelope_with_its_status(self):
        err = json.dumps({"error": {"message": "rate limited", "type": "rate_limit"}}).encode()
        out, _calls = _count(resp=_FakeResp(status=429, body=err))
        self.assertEqual(out.status, 429)
        env = json.loads(out.read_all())
        self.assertEqual(env["type"], "error")
        self.assertIn("rate limited", env["error"]["message"])

    def test_unexpected_2xx_shape_is_a_502_never_a_fake_count(self):
        out, _calls = _count(resp=_FakeResp(body=b'{"foo": 1}'))
        self.assertEqual(out.status, 502)
        self.assertEqual(json.loads(out.read_all())["type"], "error")

    def test_empty_key_fails_loud_before_the_opener(self):
        with self.assertRaises(RuntimeError):
            _count(key="")

    def test_provider_without_a_tokenizer_endpoint_is_501_not_misrouted(self):
        calls = []
        out = anthropic_count_tokens(
            _PAYLOAD, provider="zhipu",
            opener=lambda *a: calls.append(a), key_resolver=lambda _s: "k")
        self.assertEqual(out.status, 501)
        self.assertEqual(json.loads(out.read_all())["type"], "error")
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
