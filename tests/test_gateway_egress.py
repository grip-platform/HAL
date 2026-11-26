"""Tests for HAL gateway same-wire egress (Phase 0 wave 3).

Hermetic: opener + key_resolver are injected, so no network / no keychain. The
load-bearing guard is two-sided: with the flag ON, claude-* and unknown models
must STILL fall through (None) so the #3274 verbatim path is untouched; only
known non-Anthropic providers same-wire route.
"""
import json
import os
import unittest
from contextlib import contextmanager
from unittest import mock

import lib.hal.gateway_egress as ge
from lib.hal.gateway_egress import (
    PROVIDER_ENDPOINTS,
    _default_key_resolver,
    _pin_reasoning_effort_none_with_tools_for_gpt5,
    _fix_cloud_body,
    _provider_model_rewriter,
    _fix_gemini_request,
    _fix_gpt5_request,
    _model_requires_max_completion_tokens,
    _remap_max_tokens_to_completion,
    build_egress_url,
    forward_same_wire,
    multiprovider_enabled,
    same_wire_provider,
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


class TestMultiproviderFlagDefault(unittest.TestCase):

    def test_enabled_by_default(self):
        # Sovereign default: absent HAL_GATEWAY_MULTIPROVIDER => multi-provider ON.
        with _flag(None):
            self.assertTrue(multiprovider_enabled())

    def test_enabled_when_truthy(self):
        for v in ("1", "true", "YES", "on"):
            with _flag(v):
                self.assertTrue(multiprovider_enabled(), v)

    def test_flag_off_never_routes_even_gpt(self):
        with _flag("0"):
            self.assertIsNone(same_wire_provider("gpt-5.5"))


class TestDefaultKeyResolver(unittest.TestCase):
    """Linux/CI sovereignty: env fallbacks resolve provider keys with NO keychain.

    Mutation-killing: drop the GRIP_-prefixed fallback and
    test_grip_prefixed_env_resolves_on_linux fails — that fallback is the exact
    thing that lets a Linux VPS gateway see the keys (GRIP_DEEPSEEK_API_KEY) that
    secrets.env / providers.yaml / hal-server already use.
    """

    @contextmanager
    def _no_keychain(self):
        # Simulate a non-macOS host: the `security` binary is absent.
        with mock.patch.object(ge.subprocess, "run",
                               side_effect=FileNotFoundError("security")):
            yield

    @contextmanager
    def _env(self, **kv):
        prev = {k: os.environ.get(k) for k in kv}
        for k, v in kv.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        try:
            yield
        finally:
            for k, v in prev.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

    def test_grip_prefixed_env_resolves_on_linux(self):
        # THE sovereign path: canonical GRIP_<PROVIDER>_API_KEY, no keychain.
        with self._no_keychain(), self._env(DEEPSEEK_API_KEY=None,
                                             GRIP_DEEPSEEK_API_KEY="sk-grip-ds"):
            self.assertEqual(_default_key_resolver("hal-deepseek"), "sk-grip-ds")

    def test_bare_env_still_resolves(self):
        # Legacy un-prefixed name preserved (backward compat).
        with self._no_keychain(), self._env(DEEPSEEK_API_KEY="sk-bare",
                                             GRIP_DEEPSEEK_API_KEY=None):
            self.assertEqual(_default_key_resolver("hal-deepseek"), "sk-bare")

    def test_the_public_slot_name_resolves_like_the_private_one(self):
        # The public build renames grip- slots to hal- (HAL#859): same env key.
        with self._no_keychain(), self._env(DEEPSEEK_API_KEY="sk-bare",
                                             GRIP_DEEPSEEK_API_KEY=None):
            self.assertEqual(_default_key_resolver("hal-deepseek"), "sk-bare")

    def test_bare_env_takes_precedence_over_grip(self):
        with self._no_keychain(), self._env(DEEPSEEK_API_KEY="sk-bare",
                                             GRIP_DEEPSEEK_API_KEY="sk-grip"):
            self.assertEqual(_default_key_resolver("hal-deepseek"), "sk-bare")

    def test_no_key_anywhere_returns_empty_not_none(self):
        with self._no_keychain(), self._env(DEEPSEEK_API_KEY=None,
                                             GRIP_DEEPSEEK_API_KEY=None):
            self.assertEqual(_default_key_resolver("hal-deepseek"), "")


class TestSameWireGuard(unittest.TestCase):
    """Flag ON: route non-anthropic; NEVER claude/unknown (#3274 untouched)."""

    def test_gpt_routes_to_openai(self):
        with _flag("1"):
            self.assertEqual(same_wire_provider("gpt-5.5"), "openai")
            self.assertEqual(same_wire_provider("gpt-5-codex"), "openai")

    def test_gemini_and_grok_route(self):
        with _flag("1"):
            self.assertEqual(same_wire_provider("gemini-3.1-pro-preview"), "gemini")
            self.assertEqual(same_wire_provider("grok-4"), "xai")

    def test_claude_never_same_wired_flag_on(self):
        with _flag("1"):
            for m in ["claude-opus-4-8", "claude-sonnet-4-6", "claude-3-5-sonnet"]:
                self.assertIsNone(same_wire_provider(m), f"{m} must stay verbatim (#3274)")

    def test_unknown_falls_through_flag_on(self):
        with _flag("1"):
            self.assertIsNone(same_wire_provider("totally-made-up-9000"))


class TestEgressUrl(unittest.TestCase):

    def test_openai_responses_no_double_v1(self):
        self.assertEqual(build_egress_url("openai", "/v1/responses"),
                         "https://api.openai.com/v1/responses")

    def test_openai_chat(self):
        self.assertEqual(build_egress_url("openai", "/v1/chat/completions"),
                         "https://api.openai.com/v1/chat/completions")

    def test_gemini_keeps_its_prefix(self):
        self.assertEqual(
            build_egress_url("gemini", "/v1/chat/completions"),
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")


class TestGpt5MaxCompletionTokens(unittest.TestCase):
    """gpt-5.x rejects `max_tokens` (HTTP 400) and requires
    `max_completion_tokens`. The egress remaps it on the cloud forward path.
    Model-aware: gpt-4* / o-series / non-gpt keep `max_tokens` verbatim."""

    @staticmethod
    def _remap(body):
        return _fix_cloud_body(body, _remap_max_tokens_to_completion)

    def _capture_opener(self):
        captured = {}

        def opener(method, url, headers, body):
            captured.update(body=body)
            return _FakeResp(status=200)

        return opener, captured

    def test_model_predicate_matches_gpt5_family(self):
        for m in ("gpt-5", "gpt-5.5", "gpt-5-codex", "gpt-5.5-codex",
                  "openai/gpt-5.5", "GPT-5.5"):
            self.assertTrue(_model_requires_max_completion_tokens(m), m)

    def test_model_predicate_rejects_non_gpt5(self):
        for m in ("gpt-4o", "gpt-4.1", "o3", "claude-opus-4-8",
                  "gemini-3.1-pro-preview", ""):
            self.assertFalse(_model_requires_max_completion_tokens(m), m)

    def test_remap_renames_for_gpt5_preserving_other_params(self):
        out = json.loads(self._remap(b'{"model":"gpt-5.5","max_tokens":128,"temperature":0.7}'))
        self.assertNotIn("max_tokens", out)
        self.assertEqual(out["max_completion_tokens"], 128)
        self.assertEqual(out["temperature"], 0.7)  # unrelated params untouched

    def test_non_gpt5_body_byte_identical(self):
        body = b'{"model":"gpt-4o","max_tokens":128}'
        self.assertEqual(self._remap(body), body)  # fail-open, verbatim

    def test_explicit_completion_tokens_not_clobbered(self):
        out = json.loads(self._remap(b'{"model":"gpt-5.5","max_tokens":50,"max_completion_tokens":200}'))
        self.assertNotIn("max_tokens", out)
        self.assertEqual(out["max_completion_tokens"], 200)  # explicit value wins

    def test_unparseable_or_empty_body_unchanged(self):
        self.assertEqual(self._remap(b"not json"), b"not json")
        self.assertIsNone(self._remap(None))

    def test_forward_same_wire_remaps_gpt5_body(self):
        opener, captured = self._capture_opener()
        forward_same_wire(
            "openai", "/v1/chat/completions",
            b'{"model":"gpt-5.5","max_tokens":64}',
            opener=opener, key_resolver=lambda svc: "sk-test",
        )
        sent = json.loads(captured["body"])
        self.assertNotIn("max_tokens", sent)  # mutation-proof: fails if remap removed
        self.assertEqual(sent["max_completion_tokens"], 64)

    def test_forward_same_wire_gpt4_body_unchanged(self):
        opener, captured = self._capture_opener()
        forward_same_wire(
            "openai", "/v1/chat/completions",
            b'{"model":"gpt-4o","max_tokens":64}',
            opener=opener, key_resolver=lambda svc: "sk-test",
        )
        self.assertEqual(captured["body"], b'{"model":"gpt-4o","max_tokens":64}')  # verbatim


class TestForwardSameWire(unittest.TestCase):

    def _capture_opener(self):
        captured = {}

        def opener(method, url, headers, body):
            captured.update(method=method, url=url, headers=headers, body=body)
            return _FakeResp(status=200)

        return opener, captured

    def test_forwards_verbatim_with_provider_bearer_key(self):
        opener, captured = self._capture_opener()
        resp = forward_same_wire(
            "openai", "/v1/responses", b'{"model":"gpt-5.5"}',
            opener=opener, key_resolver=lambda svc: "sk-test-OPENAI",
        )
        self.assertEqual(resp.status, 200)
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["url"], "https://api.openai.com/v1/responses")
        self.assertEqual(captured["body"], b'{"model":"gpt-5.5"}')  # verbatim
        hdrs = dict(captured["headers"])
        self.assertEqual(hdrs["Authorization"], "Bearer sk-test-OPENAI")  # provider key, Bearer
        self.assertNotIn("x-api-key", {k.lower() for k in hdrs})  # NOT Anthropic auth

    def test_egress_ua_is_browser_like_never_urllib(self):
        # Cloudflare bans the Python-urllib signature (groq 2026-06-11: default
        # UA -> 403 error 1010, browser UA -> 200). Two-sided anchor: the UA is
        # present and browser-like, and can never regress to the urllib default.
        opener, captured = self._capture_opener()
        forward_same_wire(
            "groq", "/v1/chat/completions", b"{}",
            opener=opener, key_resolver=lambda svc: "k",
        )
        hdrs = {k.lower(): v for k, v in captured["headers"]}
        self.assertIn("user-agent", hdrs)
        self.assertTrue(hdrs["user-agent"].startswith("Mozilla/5.0"))
        self.assertNotIn("python-urllib", hdrs["user-agent"].lower())

    def test_key_resolver_reads_correct_service(self):
        # Exactly one service: OpenAI's own. The second funded account
        # (hal-openai-failover) was a paid rung, removed with the spend guard
        # (Step 5h), so its service is never read. This failed on a dirty run
        # before the removal: a cooldown another test wrote moved the failover
        # account first.
        opener, _ = self._capture_opener()
        seen = []

        def kr(svc):
            seen.append(svc)
            return "k"

        forward_same_wire("openai", "/v1/responses", b"{}", opener=opener, key_resolver=kr)
        self.assertEqual(seen, ["hal-openai"])

    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError):
            forward_same_wire("nope", "/v1/responses", b"{}", opener=self._capture_opener()[0])

    def test_empty_key_raises_not_silent_bearer(self):
        # An empty resolved key must fail loud, never send `Bearer ` and wrap a
        # 200-shaped response (AGORA #405 blocker).
        opener, captured = self._capture_opener()
        with self.assertRaises(RuntimeError):
            forward_same_wire("openai", "/v1/responses", b"{}",
                              opener=opener, key_resolver=lambda svc: "")
        self.assertEqual(captured, {})  # never reached the opener

    def test_every_endpoint_has_service(self):
        for provider, (base, svc) in PROVIDER_ENDPOINTS.items():
            self.assertTrue(base.startswith("https://"), provider)
            # The public build renames five slots to the hal- prefix (HAL#859).
            self.assertTrue(svc.startswith(("grip-", "hal-")), provider)


class TestLocalProviderEgress(unittest.TestCase):
    """Local (on-device) providers route KEYLESS — no keychain, no Authorization."""

    def test_bitnet_same_wires_to_local_flag_on(self):
        with _flag("1"):
            self.assertEqual(same_wire_provider("bitnet-b1.58-2b4t"), "bitnet")

    def test_bitnet_never_routes_flag_off(self):
        with _flag("0"):
            self.assertIsNone(same_wire_provider("bitnet-b1.58-2b4t"))

    def test_local_egress_url_is_localhost(self):
        self.assertEqual(
            build_egress_url("bitnet", "/v1/chat/completions"),
            "http://127.0.0.1:8114/v1/chat/completions")

    def test_local_forward_is_keyless_and_unauthenticated(self):
        # Load-bearing security property: a local provider NEVER reads the keychain
        # and NEVER sends an Authorization header. The resolver EXPLODES if called,
        # so a regression that routed local through the cloud key path fails here
        # (Goodhart-proof — the test can only pass if the keyless path is taken).
        captured = {}

        def opener(method, url, headers, body):
            captured.update(method=method, url=url, headers=headers, body=body)
            return _FakeResp(status=200)

        def exploding_resolver(svc):
            raise AssertionError(f"local provider must not read keychain (svc={svc!r})")

        resp = forward_same_wire(
            "bitnet", "/v1/chat/completions", b'{"model":"bitnet-b1.58-2b4t"}',
            opener=opener, key_resolver=exploding_resolver)
        self.assertEqual(resp.status, 200)
        self.assertEqual(captured["url"], "http://127.0.0.1:8114/v1/chat/completions")
        self.assertEqual(captured["body"], b'{"model":"bitnet-b1.58-2b4t"}')  # verbatim
        hdrs = {k.lower() for k in dict(captured["headers"])}
        self.assertNotIn("authorization", hdrs)  # keyless: no Bearer
        self.assertNotIn("x-api-key", hdrs)

    def test_local_strips_tools_param(self):
        # A local llama.cpp endpoint 500s on `tools`; the gateway strips tool
        # params for LOCAL providers so a tool-less model still answers. Two-sided:
        # the forwarded body has NO tools, but other fields survive.
        captured = {}

        def opener(method, url, headers, body):
            captured["body"] = body
            return _FakeResp(status=200)

        forward_same_wire(
            "bitnet", "/v1/chat/completions",
            b'{"model":"bitnet-b1.58-2b4t","messages":[{"role":"user","content":"hi"}],'
            b'"tools":[{"type":"function"}],"tool_choice":"auto"}',
            opener=opener, key_resolver=lambda s: "unused")
        self.assertNotIn(b'"tools"', captured["body"])
        self.assertNotIn(b'"tool_choice"', captured["body"])
        self.assertIn(b"bitnet-b1.58-2b4t", captured["body"])  # non-tool fields kept
        self.assertIn(b"messages", captured["body"])

    def test_cloud_provider_keeps_tools_with_pinned_effort(self):
        # Cloud providers must NOT be stripped — they handle tools. But a
        # gpt-5.x request with tools must carry the pinned reasoning_effort
        # 'none' (default medium 400s upstream — 2026-08-25 regression).
        captured = {}

        def opener(method, url, headers, body):
            captured["body"] = body
            return _FakeResp(status=200)

        forward_same_wire("openai", "/v1/chat/completions",
                          b'{"model":"gpt-5.5","tools":[{"type":"function"}]}',
                          opener=opener, key_resolver=lambda s: "sk-x")
        sent = json.loads(captured["body"])
        self.assertEqual(sent["tools"], [{"type": "function"}])  # tools intact
        self.assertEqual(sent["reasoning_effort"], "none")


class TestGpt5ReasoningEffortPin(unittest.TestCase):
    """gpt-5.x rejects tools + reasoning_effort on chat/completions (HTTP 400),
    and its DEFAULT effort (medium) trips the same 400 — so with tools present
    the gateway must PIN reasoning_effort to 'none' (OpenAI's own prescription),
    never merely drop it. Tools are load-bearing and must survive."""

    _TOOL = [{"type": "function", "function": {"name": "t"}}]

    def test_pins_none_when_tools_present_with_explicit_effort(self):
        obj = {"model": "gpt-5.5", "tools": self._TOOL, "reasoning_effort": "low"}
        _pin_reasoning_effort_none_with_tools_for_gpt5(obj)
        self.assertEqual(obj["reasoning_effort"], "none")
        self.assertIn("tools", obj)  # tools MUST survive

    def test_pins_none_when_tools_present_without_explicit_effort(self):
        # REGRESSION ANCHOR (2026-08-25): gpt-5.x defaults reasoning_effort to
        # medium, so tools-only requests 400 upstream. Absent + tools => pin.
        obj = {"model": "gpt-5.5", "tools": self._TOOL}
        _pin_reasoning_effort_none_with_tools_for_gpt5(obj)
        self.assertEqual(obj["reasoning_effort"], "none")
        self.assertIn("tools", obj)

    def test_keeps_reasoning_effort_when_no_tools(self):
        obj = {"model": "gpt-5.5", "reasoning_effort": "low"}
        _pin_reasoning_effort_none_with_tools_for_gpt5(obj)
        self.assertEqual(obj["reasoning_effort"], "low")

    def test_noop_for_non_gpt5_model(self):
        obj = {"model": "gpt-4o", "tools": self._TOOL, "reasoning_effort": "low"}
        _pin_reasoning_effort_none_with_tools_for_gpt5(obj)
        self.assertEqual(obj["reasoning_effort"], "low")  # KEPT — not gpt-5.x

    def test_empty_tools_list_keeps_reasoning_effort(self):
        # An empty tools list is not "with tools" — reasoning_effort must survive.
        obj = {"model": "gpt-5.5", "tools": [], "reasoning_effort": "low"}
        _pin_reasoning_effort_none_with_tools_for_gpt5(obj)
        self.assertEqual(obj["reasoning_effort"], "low")

    def test_fix_gpt5_request_applies_both_transforms(self):
        # Composite: max_tokens renamed AND reasoning_effort pinned to 'none'
        # (gpt-5.x defaults to medium, which 400s with tools), tools kept.
        obj = {"model": "gpt-5.5", "tools": self._TOOL,
               "reasoning_effort": "low", "max_tokens": 50}
        _fix_gpt5_request(obj)
        self.assertNotIn("max_tokens", obj)
        self.assertEqual(obj["max_completion_tokens"], 50)
        self.assertEqual(obj["reasoning_effort"], "none")
        self.assertIn("tools", obj)


class TestGeminiThoughtSignatureWorkaround(unittest.TestCase):
    """Gemini rejects replayed OpenAI tool history without thought_signature.

    The gateway must not forward prior assistant tool_calls/tool messages to
    Google's OpenAI-compatible shim as tool history, because the shim maps them
    to unsigned native functionCall parts. It may keep the current tools schema.
    """

    def test_rewrites_prior_tool_history_to_text(self):
        obj = {
            "model": "gemini-3.1-pro-preview",
            "messages": [
                {"role": "user", "content": "check"},
                {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{\"command\":\"date\"}"},
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": "Fri"},
            ],
            "tools": [{"type": "function", "function": {"name": "bash"}}],
        }

        _fix_gemini_request(obj)

        self.assertEqual(obj["messages"][1]["role"], "assistant")
        self.assertNotIn("tool_calls", obj["messages"][1])
        self.assertIn("[tool call bash id=call_1]", obj["messages"][1]["content"])
        self.assertEqual(obj["messages"][2]["role"], "user")
        self.assertIn("[tool result tool id=call_1]", obj["messages"][2]["content"])
        self.assertIn("tools", obj)  # current-turn tool declarations survive

    def test_gemini_forward_applies_rewrite(self):
        captured = {}

        def opener(method, url, headers, body):
            captured["body"] = body
            return _FakeResp(status=200)

        body = json.dumps({
            "model": "gemini-3.1-pro-preview",
            "messages": [
                {"role": "assistant", "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "bash", "arguments": "{}"},
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            ],
            "tools": [{"type": "function", "function": {"name": "bash"}}],
        }).encode("utf-8")

        forward_same_wire("gemini", "/v1/chat/completions", body,
                          opener=opener, key_resolver=lambda s: "sk-gemini")
        sent = json.loads(captured["body"])
        self.assertNotIn("tool_calls", sent["messages"][0])
        self.assertEqual(sent["messages"][1]["role"], "user")
        self.assertIn("tools", sent)

    def test_signed_gemini_tool_history_survives(self):
        obj = {
            "model": "gemini-3.1-pro-preview",
            "messages": [{"role": "assistant", "tool_calls": [{
                "id": "call_1",
                "function": {"name": "bash", "arguments": "{}"},
                "extra_content": {"google": {"thought_signature": "sig-1"}},
            }]}],
        }

        _fix_gemini_request(obj)

        self.assertIn("tool_calls", obj["messages"][0])
        self.assertEqual(
            obj["messages"][0]["tool_calls"][0]["extra_content"]["google"]["thought_signature"],
            "sig-1",
        )

    def test_non_gemini_cloud_provider_keeps_tool_history(self):
        captured = {}

        def opener(method, url, headers, body):
            captured["body"] = body
            return _FakeResp(status=200)

        body = b'{"model":"gpt-4o","messages":[{"role":"assistant","tool_calls":[{"id":"c"}]}]}'
        forward_same_wire("openai", "/v1/chat/completions", body,
                          opener=opener, key_resolver=lambda s: "sk-openai")
        self.assertEqual(captured["body"], body)


class TestBracketSuffixStrip(unittest.TestCase):
    """Claude Code addresses its 1M tier as ``<model>[1m]`` (``kimi-k3[1m]``
    in the vendor's own Claude Code recipe). Anthropic parses that suffix; no
    other provider does -- Moonshot 404s on it -- so the model rewriter strips
    ONE trailing ``[...]`` on non-Anthropic egress and leaves Anthropic-bound
    ids byte-identical. Added 2026-09-07 (W0 of the Kimi plan)."""

    def _rewrite(self, provider, model):
        body = json.dumps({"model": model, "messages": []}).encode()
        out = _fix_cloud_body(body, _provider_model_rewriter(provider))
        return json.loads(out)["model"], out is body

    def test_strips_1m_suffix_for_moonshot(self):
        self.assertEqual(self._rewrite("moonshot", "kimi-k3[1m]")[0], "kimi-k3")

    def test_strips_provider_prefix_and_suffix_together(self):
        self.assertEqual(self._rewrite("moonshot", "moonshot/kimi-k3[1m]")[0], "kimi-k3")

    def test_anthropic_egress_keeps_the_suffix_byte_identical(self):
        model, same = self._rewrite("anthropic", "claude-opus-5[1m]")
        self.assertEqual(model, "claude-opus-5[1m]")
        self.assertTrue(same)

    def test_no_suffix_is_a_byte_identical_noop(self):
        model, same = self._rewrite("moonshot", "kimi-k3")
        self.assertEqual(model, "kimi-k3")
        self.assertTrue(same)

    def test_only_one_trailing_suffix_and_never_an_empty_id(self):
        # a bracket mid-id is not a capability suffix, and a bare "[1m]" must
        # never collapse to an empty model id.
        self.assertEqual(self._rewrite("moonshot", "kimi[x]-k3")[0], "kimi[x]-k3")
        self.assertEqual(self._rewrite("moonshot", "[1m]")[0], "[1m]")


if __name__ == "__main__":
    unittest.main()


class TestReasoningEffortPinIsGpt5Only(unittest.TestCase):
    """HAL#548 follow-up, step 1. The ``reasoning_effort: "none"`` pin runs on
    its own gpt-5-only predicate, not on the ``max_completion_tokens`` set that
    model_quirks shares with the Kimi lineage (HAL#547). Moonshot's platform
    API accepts only ``low`` / ``high`` / ``max`` there (anything else is HTTP
    400), and on the Kimi Code host ``none`` is the thinking-off switch that
    silently routes K3 to K2.6 -- so the pin must never reach a Kimi body.

    Proved able to fail (tests-must-be-able-to-fail.md mandate 3): with the
    gateway's max_completion_tokens predicate switched to model_quirks and the
    pin still keyed on it, ``test_kimi_with_tools_keeps_its_reasoning_effort``
    went red (``'none' != 'high'`` for kimi-k3); the split turned it green."""

    _TOOL = [{"type": "function", "function": {"name": "t"}}]

    def test_gpt5_only_predicate(self):
        from lib.hal.gateway_egress import _model_is_gpt5
        for m in ("gpt-5", "gpt-5.5", "openai/gpt-5-codex", "GPT-5.5"):
            self.assertTrue(_model_is_gpt5(m), m)
        for m in ("kimi-k3", "moonshot/kimi-k3", "kimi-k2.7-code", "gpt-4o", "o3",
                  "deepseek-chat", "claude-opus-5", ""):
            self.assertFalse(_model_is_gpt5(m), m)

    def test_kimi_with_tools_keeps_its_reasoning_effort(self):
        for m in ("kimi-k3", "moonshot/kimi-k3", "kimi-k3[1m]"):
            obj = {"model": m, "tools": self._TOOL, "reasoning_effort": "high"}
            _pin_reasoning_effort_none_with_tools_for_gpt5(obj)
            self.assertEqual(obj["reasoning_effort"], "high", m)
            absent = {"model": m, "tools": self._TOOL}
            _pin_reasoning_effort_none_with_tools_for_gpt5(absent)
            # Absent stays absent: the vendor default (max) applies, never "none".
            self.assertNotIn("reasoning_effort", absent, m)

    def test_gpt5_pin_survives_the_split(self):
        obj = {"model": "openai/gpt-5.5", "tools": self._TOOL, "reasoning_effort": "low"}
        _pin_reasoning_effort_none_with_tools_for_gpt5(obj)
        self.assertEqual(obj["reasoning_effort"], "none")


class TestMaxCompletionTokensSingleHome(unittest.TestCase):
    """HAL#548 follow-up, step 2. The gateway's ``max_completion_tokens``
    predicate takes its model set from ``model_quirks`` (one home, HAL#547), so
    the Kimi lineage — where Moonshot deprecates ``max_tokens`` — gets the
    rename too, while the gpt-5-only reasoning_effort pin (step 1) leaves a
    Kimi body's effort alone. Red at the previous commit: the gateway's private
    tuple was ("gpt-5",), so every kimi case here failed."""

    _TOOL = [{"type": "function", "function": {"name": "t"}}]

    def test_gateway_tuple_is_the_quirks_tuple(self):
        from lib.hal import gateway_egress as ge, model_quirks as mq
        self.assertIs(ge._MAX_COMPLETION_TOKENS_MODELS, mq.MAX_COMPLETION_TOKENS_MODELS)

    def test_predicate_matches_the_kimi_lineage(self):
        for m in ("kimi-k3", "moonshot/kimi-k3", "kimi-k3[1m]", "moonshot/kimi-k2.6",
                  "kimi-k2.7-code-highspeed", "MOONSHOT/KIMI-K2.7-CODE"):
            self.assertTrue(_model_requires_max_completion_tokens(m), m)

    def test_predicate_agrees_with_quirks_on_a_matrix(self):
        from lib.hal import model_quirks as mq
        for m in ("gpt-5.5", "openai/gpt-5-codex", "kimi-k3", "moonshot/kimi-k2.6",
                  "gpt-4o", "o3", "deepseek-chat", "claude-opus-5", "moonshot-v1-8k", ""):
            self.assertEqual(_model_requires_max_completion_tokens(m),
                             mq.uses_max_completion_tokens(m), m)

    def test_composite_on_a_kimi_body_renames_and_keeps_effort(self):
        obj = {"model": "moonshot/kimi-k3", "tools": self._TOOL,
               "reasoning_effort": "high", "max_tokens": 50}
        _fix_gpt5_request(obj)
        self.assertNotIn("max_tokens", obj)
        self.assertEqual(obj["max_completion_tokens"], 50)
        self.assertEqual(obj["reasoning_effort"], "high")
        self.assertIn("tools", obj)

    def test_forward_same_wire_kimi_body(self):
        captured = {}

        def opener(method, url, headers, body):
            captured["body"] = body
            return _FakeResp(status=200)

        forward_same_wire(
            "moonshot", "/v1/chat/completions",
            b'{"model":"moonshot/kimi-k3","max_tokens":64,"tools":[{"type":"function"}],'
            b'"reasoning_effort":"high","temperature":0.2}',
            opener=opener, key_resolver=lambda s: "sk-x")
        sent = json.loads(captured["body"])
        self.assertNotIn("max_tokens", sent)
        self.assertEqual(sent["max_completion_tokens"], 64)
        self.assertEqual(sent["reasoning_effort"], "high")   # never pinned to "none"
        self.assertNotIn("temperature", sent)                 # locked param still dropped
        self.assertEqual(sent["model"], "kimi-k3")            # provider head stripped
