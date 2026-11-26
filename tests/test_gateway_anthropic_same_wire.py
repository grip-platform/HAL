"""Anthropic same-wire egress (Kimi plan, W2 -- HAL side).

A provider that serves the Anthropic Messages wire natively (Moonshot at
``https://api.moonshot.ai/anthropic``) receives an Anthropic-wire client's
body VERBATIM -- thinking, tool_use and cache fields intact -- instead of
going through the cross-wire translator. Hermetic: ``opener`` and
``key_resolver`` are injected, the routing flag is pinned, zero network and
zero keychain access.
"""
import json
import os
import unittest
from contextlib import contextmanager

from lib.hal.gateway_egress import (
    ANTHROPIC_COMPAT_ENDPOINTS,
    PROVIDER_ENDPOINTS,
    AnthropicCompatProxy,
    anthropic_compat_provider,
    anthropic_same_wire_forward,
)

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


def _capture_opener(*responses):
    """An opener that records every call and answers ``responses`` in order."""
    calls = []
    queue = list(responses)

    def opener(method, url, headers, body):
        calls.append({"method": method, "url": url, "headers": list(headers), "body": body})
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return opener, calls


_CLIENT_HEADERS = [
    ("Host", "127.0.0.1:4010"),
    ("Content-Type", "application/json"),
    ("Content-Length", "123"),
    ("Connection", "keep-alive"),
    ("Accept-Encoding", "gzip"),
    ("x-api-key", "sk-ant-api03-CLIENT-KEY"),
    ("Authorization", "Bearer sk-ant-oat01-OPERATOR-OAUTH"),
    ("anthropic-version", "2023-06-01"),
    ("anthropic-beta", "context-1m-2025-08-07,interleaved-thinking-2025-05-14"),
    ("User-Agent", "claude-cli/2.1.261 (external, cli)"),
]
_BODY = json.dumps({"model": "kimi-k3", "max_tokens": 8,
                    "messages": [{"role": "user", "content": "Reply OK"}]}).encode()


def _forward(body=_BODY, path="/v1/messages", headers=_CLIENT_HEADERS,
             key="sk-moon-PROVIDER", responses=(_FakeResp(),)):
    opener, calls = _capture_opener(*responses)
    services = []

    def resolver(service):
        services.append(service)
        return key

    resp = anthropic_same_wire_forward(
        "moonshot", path, body, headers,
        opener=opener, key_resolver=resolver, sleep=lambda _s: None)
    return resp, calls, services


class TestAnthropicCompatTable(unittest.TestCase):
    """The endpoint table: vendor path shape + the cloud invariant."""

    def test_moonshot_row_is_the_vendor_anthropic_base(self):
        base, service = ANTHROPIC_COMPAT_ENDPOINTS["moonshot"]
        self.assertEqual(base, "https://api.moonshot.ai/anthropic")
        self.assertEqual(service, "grip-moonshot")

    def test_every_row_is_https_with_a_grip_keychain_service(self):
        for provider, (base, service) in ANTHROPIC_COMPAT_ENDPOINTS.items():
            self.assertTrue(base.startswith("https://"), provider)
            self.assertTrue(service.startswith("grip-"), provider)

    def test_every_row_is_also_an_openai_same_wire_provider(self):
        # same_wire_provider is the routing gate; a provider listed here but
        # absent from PROVIDER_ENDPOINTS would raise "unroutable" before the
        # Anthropic branch is ever reached.
        for provider in ANTHROPIC_COMPAT_ENDPOINTS:
            self.assertIn(provider, PROVIDER_ENDPOINTS)

    def test_base_does_not_end_with_a_version_segment(self):
        # The gateway path (/v1/messages) is appended AS-IS, so a base that
        # already carried /v1 would produce /v1/v1/messages.
        for provider, (base, _service) in ANTHROPIC_COMPAT_ENDPOINTS.items():
            self.assertFalse(base.rstrip("/").endswith("/v1"), provider)


class TestAnthropicCompatProvider(unittest.TestCase):
    """The predicate narrows same_wire_provider to the Anthropic-compatible set."""

    def test_kimi_family_resolves_to_moonshot(self):
        with _flag("1"):
            for model in ("kimi-k3", "kimi-k3[1m]", "moonshot/kimi-k3", "kimi-k2.7-code"):
                self.assertEqual(anthropic_compat_provider(model), "moonshot", model)

    def test_openai_family_provider_has_no_anthropic_endpoint(self):
        with _flag("1"):
            self.assertIsNone(anthropic_compat_provider("gpt-5.5"))
            self.assertIsNone(anthropic_compat_provider("gemini-3.1-pro"))

    def test_claude_and_unknown_stay_none(self):
        with _flag("1"):
            self.assertIsNone(anthropic_compat_provider("claude-opus-5[1m]"))
            self.assertIsNone(anthropic_compat_provider("made-up-9000"))
            self.assertIsNone(anthropic_compat_provider(""))

    def test_flag_off_is_none_even_for_kimi(self):
        with _flag("0"):
            self.assertIsNone(anthropic_compat_provider("kimi-k3"))


class TestAnthropicCompatProxyHeaders(unittest.TestCase):
    """Custody: the provider's key goes upstream, the operator's never does."""

    def test_url_is_base_plus_gateway_path_verbatim(self):
        _resp, calls, _s = _forward()
        self.assertEqual(calls[0]["url"], "https://api.moonshot.ai/anthropic/v1/messages")
        self.assertEqual(calls[0]["method"], "POST")

    def test_query_string_is_preserved(self):
        _resp, calls, _s = _forward(path="/v1/messages?beta=true")
        self.assertTrue(calls[0]["url"].endswith("/anthropic/v1/messages?beta=true"))

    def test_client_credentials_dropped_provider_bearer_added(self):
        _resp, calls, services = _forward()
        hdrs = calls[0]["headers"]
        names = [n.lower() for n, _v in hdrs]
        self.assertEqual(names.count("authorization"), 1)
        self.assertNotIn("x-api-key", names)
        self.assertEqual(dict(hdrs)["Authorization"], "Bearer sk-moon-PROVIDER")
        joined = " ".join(v for _n, v in hdrs)
        self.assertNotIn("sk-ant-oat01-OPERATOR-OAUTH", joined)
        self.assertNotIn("sk-ant-api03-CLIENT-KEY", joined)
        self.assertEqual(services, ["grip-moonshot"])

    def test_anthropic_version_and_beta_retained_verbatim(self):
        _resp, calls, _s = _forward()
        hdrs = dict(calls[0]["headers"])
        self.assertEqual(hdrs["anthropic-version"], "2023-06-01")
        self.assertEqual(hdrs["anthropic-beta"],
                         "context-1m-2025-08-07,interleaved-thinking-2025-05-14")
        self.assertEqual(hdrs["Content-Type"], "application/json")

    def test_hop_by_hop_and_host_stripped(self):
        _resp, calls, _s = _forward()
        names = {n.lower() for n, _v in calls[0]["headers"]}
        for gone in ("host", "content-length", "connection", "accept-encoding"):
            self.assertNotIn(gone, names)

    def test_client_user_agent_kept_default_added_only_when_absent(self):
        _resp, calls, _s = _forward()
        self.assertEqual(dict(calls[0]["headers"])["User-Agent"],
                         "claude-cli/2.1.261 (external, cli)")
        bare = [(n, v) for n, v in _CLIENT_HEADERS if n.lower() != "user-agent"]
        _resp, calls, _s = _forward(headers=bare)
        self.assertIn("hal-gateway", dict(calls[0]["headers"])["User-Agent"])

    def test_empty_key_fails_loud_before_the_opener(self):
        opener, calls = _capture_opener(_FakeResp())
        with self.assertRaises(RuntimeError):
            anthropic_same_wire_forward(
                "moonshot", "/v1/messages", _BODY, _CLIENT_HEADERS,
                opener=opener, key_resolver=lambda _s: "")
        self.assertEqual(calls, [])

    def test_unlisted_provider_is_refused(self):
        with self.assertRaises((ValueError, KeyError)):
            AnthropicCompatProxy("openai", key_resolver=lambda _s: "k")


class TestAnthropicSameWireBody(unittest.TestCase):
    """The body is the client's, except the model id is spelled as the vendor
    knows it. Probed 2026-09-07: kimi-k3[1m] is HTTP 404 at Moonshot."""

    def _sent(self, body_obj, **kw):
        _resp, calls, _s = _forward(body=json.dumps(body_obj).encode(), **kw)
        return json.loads(calls[0]["body"])

    def test_1m_suffix_is_stripped_on_the_wire(self):
        sent = self._sent({"model": "kimi-k3[1m]", "max_tokens": 8, "messages": []})
        self.assertEqual(sent["model"], "kimi-k3")

    def test_provider_prefix_and_suffix_both_stripped(self):
        sent = self._sent({"model": "moonshot/kimi-k3[1m]", "max_tokens": 8, "messages": []})
        self.assertEqual(sent["model"], "kimi-k3")

    def test_unprefixed_body_is_forwarded_byte_identical(self):
        _resp, calls, _s = _forward(body=_BODY)
        self.assertIs(calls[0]["body"], _BODY)

    def test_thinking_tools_cache_and_output_config_pass_through(self):
        original = {
            "model": "kimi-k3[1m]", "max_tokens": 64, "stream": True,
            "system": [{"type": "text", "text": "be terse",
                        "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 32},
            "tools": [{"name": "t", "input_schema": {"type": "object"}}],
            "output_config": {"effort": "high"},
            "metadata": {"user_id": "session-1"},
        }
        sent = self._sent(original)
        expected = dict(original, model="kimi-k3")
        self.assertEqual(sent, expected)

    def test_retryable_429_resends_the_identical_bytes(self):
        body = json.dumps({"model": "kimi-k3[1m]", "max_tokens": 8, "messages": []}).encode()
        resp, calls, _s = _forward(body=body, responses=(_FakeResp(status=429), _FakeResp()))
        self.assertEqual(resp.status, 200)
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0]["body"], calls[1]["body"])

    def test_400_is_relayed_not_retried(self):
        resp, calls, _s = _forward(responses=(_FakeResp(status=400, body=b'{"type":"error"}'),))
        self.assertEqual(resp.status, 400)
        self.assertEqual(len(calls), 1)
        self.assertEqual(resp.read_all(), b'{"type":"error"}')

    def test_event_stream_response_is_relayed_verbatim(self):
        sse = b"event: message_start\ndata: {}\n\nevent: message_stop\ndata: {}\n\n"
        resp, _calls, _s = _forward(responses=(
            _FakeResp(headers=[("Content-Type", "text/event-stream")], body=sse),))
        self.assertEqual(dict(resp.headers)["Content-Type"], "text/event-stream")
        self.assertEqual(resp.read_all(), sse)


if __name__ == "__main__":
    unittest.main()
