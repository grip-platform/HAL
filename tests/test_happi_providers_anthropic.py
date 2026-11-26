"""Golden tests for the Anthropic provider adapter (α-3).

Exercises token resolution, envelope → request-body translation, SSE event
parsing, error paths (HTTP/network/provider-level), token redaction, and
runtime dispatch of `cmd: chat` envelopes. All HTTP traffic is mocked —
no network dependency, no real Anthropic calls.
"""
from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from lib.hal.happi.envelope import HappiEnvelope
from lib.hal.happi.providers import anthropic as provider
from lib.hal.happi.runtime import run


class _FakeSSEResponse:
    """Minimal SSE-shaped response matching what urlopen returns."""

    def __init__(self, lines: list[str]):
        self._lines = [line.encode("utf-8") for line in lines]
        self.closed = False

    def __iter__(self):
        return iter(self._lines)

    def close(self):
        self.closed = True


def _envelope(**overrides) -> HappiEnvelope:
    """Build a chat envelope with sensible α-3 defaults for tests."""
    payload = {
        "v": "happi/1.0",
        "id": "req-test",
        "cmd": "chat",
        "args": ["Hello, HAPPI."],
        "flags": {"provider": "claude-opus-4-7", "max_tokens": 100},
        # API-key-shape so these defaults exercise the x-api-key dispatch path.
        # OAuth-shape (sk-ant-oat*) routes through CCH (cli-formatted.js:108743
        # / TF>>>>) and is covered in the CCH-dispatch tests below.
        "auth": {"scheme": "apikey", "token": "sk-ant-api03-fake-test-token"},
    }
    payload.update(overrides)
    return HappiEnvelope.parse(payload)


def _collect(events_iter) -> list[dict]:
    return [json.loads(line) for line in events_iter]


# --- Token redaction --------------------------------------------------------

def test_redact_strips_anthropic_oauth_tokens():
    text = "error: bad token sk-ant-oat01-abc-def-123 in header"
    assert "sk-ant-oat01" not in provider._redact(text)
    assert "[REDACTED]" in provider._redact(text)


def test_redact_strips_bearer_tokens():
    text = "Authorization: Bearer abcd.1234_efgh failed"
    assert "Bearer abcd" not in provider._redact(text)


# --- Request body building --------------------------------------------------

def test_build_request_body_uses_args_as_user_message():
    env = _envelope(args=["Tell", "me", "a", "joke"])
    body = provider._build_request_body(env)
    assert body["messages"] == [{"role": "user", "content": "Tell me a joke"}]
    assert body["stream"] is True


def test_build_request_body_uses_top_level_content_field():
    """Regression: envelope.content forwarded to Anthropic messages array.

    Before the fix, a top-level 'content' key was silently dropped by
    HappiEnvelope.parse (no such field), leaving message="" and causing
    Anthropic to reject with 'user messages must have non-empty content'.
    """
    env = HappiEnvelope.parse({
        "v": "happi/1.0",
        "id": "req-content-test",
        "cmd": "chat",
        "content": "Generate a morning brief.",
        "auth": {"scheme": "apikey", "token": "sk-ant-fake"},
    })
    assert env.content == "Generate a morning brief."
    body = provider._build_request_body(env)
    assert body["messages"] == [{"role": "user", "content": "Generate a morning brief."}]


def test_build_request_body_content_takes_priority_over_args():
    """envelope.content wins over args when both are present."""
    env = HappiEnvelope.parse({
        "v": "happi/1.0",
        "id": "req-priority",
        "cmd": "chat",
        "content": "From content field.",
        "args": ["From args."],
        "auth": {"scheme": "apikey", "token": "sk-ant-fake"},
    })
    body = provider._build_request_body(env)
    assert body["messages"][0]["content"] == "From content field."


def test_build_request_body_strips_anthropic_prefix():
    env = _envelope(flags={"provider": "anthropic/claude-sonnet-4-6"})
    body = provider._build_request_body(env)
    assert body["model"] == "claude-sonnet-4-6"


def test_build_request_body_respects_system_prompt():
    env = _envelope(flags={"provider": "claude-opus-4-7", "system": "Be terse."})
    body = provider._build_request_body(env)
    assert body["system"] == "Be terse."


# --- Sampling flags → request body (x-api-key path) ------------------------

def test_build_request_body_threads_temperature_zero():
    # AGORA's determinism case: temperature 0.0 must reach the body.
    env = _envelope(flags={"provider": "claude-opus-4-7", "temperature": 0.0})
    body = provider._build_request_body(env)
    assert body["temperature"] == 0.0


def test_build_request_body_threads_top_p():
    env = _envelope(flags={"provider": "claude-opus-4-7", "top_p": 0.9})
    body = provider._build_request_body(env)
    assert body["top_p"] == 0.9


def test_build_request_body_omits_seed_even_when_set():
    # The Anthropic Messages API has no seed field; passing it would 400, so
    # the adapter must drop it (allow_seed=False) even if the caller sets it.
    env = _envelope(flags={"provider": "claude-opus-4-7", "seed": 42})
    body = provider._build_request_body(env)
    assert "seed" not in body


def test_build_request_body_omits_sampling_by_default():
    # Backwards-compat: no sampling keys present unless the caller sets them.
    body = provider._build_request_body(_envelope())
    assert "temperature" not in body
    assert "top_p" not in body
    assert "seed" not in body


# --- CCH path forward-anchor: sampling is NEVER threaded -------------------

def test_cch_call_args_never_carry_sampling_flags():
    """The CCH (subscription) path must drop sampling flags even when set.

    The CCH signer mimics a Claude Code request with adaptive thinking, and the
    Anthropic Messages API rejects a non-default temperature when thinking is
    active — so threading temperature there would 400 on the common Opus/Sonnet
    models. _build_cch_call_args therefore must NOT carry temperature/top_p/seed
    onto build_kwargs. Forward-anchor for that deliberate exclusion: this fails
    if a future edit naively threads sampling onto the CCH path.
    """
    env = _envelope(flags={
        "provider": "claude-opus-4-7",
        "temperature": 0.0, "top_p": 0.9, "seed": 7, "compact": True,
    })
    _msg, _model, build_kwargs = provider._build_cch_call_args(env)
    assert "temperature" not in build_kwargs
    assert "top_p" not in build_kwargs
    assert "seed" not in build_kwargs
    # compact (a real CCH kwarg) still threads — proves we drop sampling
    # specifically, not all extra flags.
    assert build_kwargs.get("compact") is True


# --- Token resolution -------------------------------------------------------

def test_resolve_token_apikey_literal():
    env = _envelope(auth={"scheme": "apikey", "token": "sk-ant-literal"})
    assert provider._resolve_token(env) == "sk-ant-literal"


def test_resolve_token_subscription_env(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-from-env")
    env = _envelope(auth={"scheme": "subscription"})
    assert provider._resolve_token(env) == "sk-ant-oat01-from-env"


def test_resolve_token_apikey_env_prefix(monkeypatch):
    monkeypatch.setenv("TEST_HAPPI_TOKEN", "sk-ant-from-env-var")
    env = _envelope(auth={"scheme": "apikey", "token": "env:TEST_HAPPI_TOKEN"})
    assert provider._resolve_token(env) == "sk-ant-from-env-var"


# --- Missing credentials path ----------------------------------------------

def test_stream_anthropic_without_token_emits_missing_credentials(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setattr(provider, "_fetch_keychain_token", lambda *a, **kw: None)
    env = _envelope(auth={"scheme": "subscription"})
    events = _collect(provider.stream_anthropic(env))
    assert [e["type"] for e in events] == ["error", "completed"]
    assert events[0]["code"] == "missing_credentials"


# --- Successful streaming ---------------------------------------------------

def _successful_sse_lines() -> list[str]:
    return [
        "event: message_start\n",
        'data: {"type":"message_start","message":{"usage":{"input_tokens":12,"output_tokens":0}}}\n',
        "\n",
        "event: content_block_start\n",
        'data: {"type":"content_block_start"}\n',
        "\n",
        "event: content_block_delta\n",
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}\n',
        "\n",
        "event: content_block_delta\n",
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":", world."}}\n',
        "\n",
        "event: message_delta\n",
        'data: {"type":"message_delta","usage":{"input_tokens":12,"output_tokens":4}}\n',
        "\n",
        "event: message_stop\n",
        'data: {"type":"message_stop"}\n',
        "\n",
    ]


def test_stream_anthropic_successful_stream_emits_deltas_and_completed():
    env = _envelope()
    fake_response = _FakeSSEResponse(_successful_sse_lines())
    with patch.object(provider, "_open_stream", return_value=fake_response):
        events = _collect(provider.stream_anthropic(env))
    types = [e["type"] for e in events]
    assert types == ["delta", "delta", "completed"]
    assert events[0]["text"] == "Hello"
    assert events[1]["text"] == ", world."
    assert events[2]["usage"] == {"in": 12, "out": 4}
    assert fake_response.closed


# --- Provider-level error ---------------------------------------------------

def test_stream_anthropic_error_event_emits_error_and_completed():
    env = _envelope()
    error_lines = [
        "event: content_block_delta\n",
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"partial"}}\n',
        "\n",
        "event: error\n",
        'data: {"type":"error","error":{"type":"overloaded_error","message":"upstream backpressure"}}\n',
        "\n",
    ]
    fake_response = _FakeSSEResponse(error_lines)
    with patch.object(provider, "_open_stream", return_value=fake_response):
        events = _collect(provider.stream_anthropic(env))
    types = [e["type"] for e in events]
    assert types == ["delta", "error", "completed"]
    assert events[1]["code"] == "provider_error:overloaded_error"
    assert "backpressure" in events[1]["message"]


# --- HTTP / network error paths --------------------------------------------

def test_stream_anthropic_http_401_emits_provider_http_error():
    env = _envelope()
    http_err = urllib.error.HTTPError(
        provider.ANTHROPIC_ENDPOINT, 401, "Unauthorized",
        hdrs=None, fp=io.BytesIO(b'{"error":"bad token"}'),
    )
    with patch.object(provider, "_open_stream", side_effect=http_err):
        events = _collect(provider.stream_anthropic(env))
    assert [e["type"] for e in events] == ["error", "completed"]
    assert events[0]["code"] == "provider_http_401"


def test_stream_anthropic_network_error_emits_provider_network():
    env = _envelope()
    url_err = urllib.error.URLError("connection refused")
    with patch.object(provider, "_open_stream", side_effect=url_err):
        events = _collect(provider.stream_anthropic(env))
    assert [e["type"] for e in events] == ["error", "completed"]
    assert events[0]["code"] == "provider_network"


# --- Runtime dispatch integration -------------------------------------------

def test_runtime_chat_envelope_routes_to_anthropic(monkeypatch):
    # API-key-shape token: stays on the x-api-key path. CCH dispatch (OAuth)
    # is exercised in the dedicated CCH tests below.
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-api03-runtime-test")
    payload = {
        "v": "happi/1.0",
        "id": "req-runtime-chat",
        "cmd": "chat",
        "args": ["hi"],
        "flags": {"provider": "claude-opus-4-7"},
    }
    fake_response = _FakeSSEResponse(_successful_sse_lines())
    stdin = io.StringIO(json.dumps(payload))
    stdout = io.StringIO()
    with patch.object(provider, "_open_stream", return_value=fake_response):
        rc = run(stream_in=stdin, stream_out=stdout)
    assert rc == 0
    events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    types = [e["type"] for e in events]
    assert types == ["started", "delta", "delta", "completed"]


def test_runtime_chat_with_unsupported_provider_errors():
    # α-3.5 added openai/groq/mistral/deepseek/openrouter/moonshot/ollama;
    # cohere is genuinely not yet in any adapter family → unsupported.
    payload = {
        "v": "happi/1.0",
        "id": "req-unsup-provider",
        "cmd": "chat",
        "args": ["hi"],
        "flags": {"provider": "cohere/command-r-plus"},
    }
    stdin = io.StringIO(json.dumps(payload))
    stdout = io.StringIO()
    rc = run(stream_in=stdin, stream_out=stdout)
    assert rc == 0
    events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    types = [e["type"] for e in events]
    assert types == ["started", "error", "completed"]
    assert events[1]["code"] == "unsupported_provider"


# --- P5: v1.1 acceptance + v1.0 regression --------------------------------

def test_anthropic_accepts_v11_envelope(monkeypatch):
    """A v1.1 envelope dispatches successfully through the anthropic adapter.

    P5 acceptance: providers must accept v1.1 envelopes without rejection at
    the envelope layer (P1) and without behavioural divergence in the
    streaming pipeline. Mocked SSE — no real provider call fires.
    """
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-api03-v11-test")
    payload = {
        "v": "happi/1.1",
        "id": "req-v11-anthropic",
        "cmd": "chat",
        "args": ["hi"],
        "flags": {"provider": "claude-opus-4-7"},
    }
    fake_response = _FakeSSEResponse(_successful_sse_lines())
    stdin = io.StringIO(json.dumps(payload))
    stdout = io.StringIO()
    with patch.object(provider, "_open_stream", return_value=fake_response):
        rc = run(stream_in=stdin, stream_out=stdout)
    assert rc == 0
    events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    types = [e["type"] for e in events]
    assert types == ["started", "delta", "delta", "completed"]


def test_anthropic_v10_envelope_unchanged(monkeypatch):
    """Regression guard: v1.0 envelopes still dispatch identically post-P5.

    H428 evidence_for: byte-equivalent v1.0 dispatch after provider docstring
    update + v1.1 acceptance. If this test diverges from the pre-P5 baseline
    test_runtime_chat_envelope_routes_to_anthropic, the docstring touch
    accidentally altered runtime behaviour.
    """
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-api03-v10-regress")
    payload = {
        "v": "happi/1.0",
        "id": "req-v10-regress",
        "cmd": "chat",
        "args": ["hi"],
        "flags": {"provider": "claude-opus-4-7"},
    }
    fake_response = _FakeSSEResponse(_successful_sse_lines())
    stdin = io.StringIO(json.dumps(payload))
    stdout = io.StringIO()
    with patch.object(provider, "_open_stream", return_value=fake_response):
        rc = run(stream_in=stdin, stream_out=stdout)
    assert rc == 0
    events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    types = [e["type"] for e in events]
    assert types == ["started", "delta", "delta", "completed"]
    assert events[1]["text"] == "Hello"
    assert events[2]["text"] == ", world."


# --- CCH dispatch for OAuth subscription tokens ----------------------------
#
# Anthropic OAuth tokens (sk-ant-oat*) MUST route through the CCH signer
# (full envelope: Authorization Bearer, User-Agent claude-cli, x-app cli,
# anthropic-beta filtered, signed body). The x-api-key path 401s on them.
# Canonical reference: cli-formatted.js:108743-108748 / TF>>>>.

class _FakeChatResponse:
    """Minimal ChatResponse shape: just .text and .usage with prompt/completion."""

    def __init__(self, text: str, prompt_tokens: int, completion_tokens: int):
        self.text = text

        class _Usage:
            pass
        self.usage = _Usage()
        self.usage.prompt_tokens = prompt_tokens
        self.usage.completion_tokens = completion_tokens


def test_is_oauth_token_classifies_prefix():
    """Catches: prefix detection regressed (would silently route OAuth to x-api-key)."""
    assert provider._is_oauth_token("sk-ant-oat01-some-real-shape") is True
    assert provider._is_oauth_token("sk-ant-oat") is True
    assert provider._is_oauth_token("sk-ant-api03-raw-key") is False
    assert provider._is_oauth_token("sk-ant-fake") is False
    assert provider._is_oauth_token("") is False


def test_stream_anthropic_oauth_token_routes_via_cch_not_open_stream(monkeypatch):
    """Catches: OAuth token reaching the x-api-key path (the original 401 bug).

    Mutation guard: if `_is_oauth_token` is bypassed or `_stream_via_cch` is
    removed, `_open_stream` would be called instead and the assertion fires.
    """
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-test-route")
    env = HappiEnvelope.parse({
        "v": "happi/1.0",
        "id": "req-oauth-route",
        "cmd": "chat",
        "args": ["hi"],
        "flags": {"provider": "claude-opus-4-7", "max_tokens": 50},
    })
    fake_resp = _FakeChatResponse("hello from CCH", prompt_tokens=3, completion_tokens=7)

    with patch.object(provider, "_open_stream") as open_stream_mock, \
         patch("lib.hal.utils.cch.CCHSigner") as cch_mock:
        cch_mock.return_value.call.return_value = fake_resp
        events = _collect(provider.stream_anthropic(env))

    assert open_stream_mock.call_count == 0, "OAuth token must NOT hit x-api-key path"
    assert cch_mock.return_value.call.call_count == 1
    call_kwargs = cch_mock.return_value.call.call_args.kwargs
    assert call_kwargs["oauth_token"] == "sk-ant-oat01-test-route"
    assert call_kwargs["model"] == "claude-opus-4-7"
    assert call_kwargs["max_tokens"] == 50
    types = [e["type"] for e in events]
    assert types == ["delta", "completed"]
    assert events[0]["text"] == "hello from CCH"
    assert events[1]["usage"] == {"in": 3, "out": 7}


def test_stream_anthropic_apikey_token_keeps_x_api_key_path(monkeypatch):
    """Catches: x-api-key path regression — non-OAuth tokens must still use SSE.

    Goodhart-anchor: if dispatch were unconditionally CCH (mutation), this
    fails because CCHSigner would be called.
    """
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-api03-raw-key")
    env = HappiEnvelope.parse({
        "v": "happi/1.0",
        "id": "req-apikey-route",
        "cmd": "chat",
        "args": ["hi"],
        "flags": {"provider": "claude-opus-4-7"},
    })
    fake_sse = _FakeSSEResponse(_successful_sse_lines())

    with patch.object(provider, "_open_stream", return_value=fake_sse) as open_stream_mock, \
         patch("lib.hal.utils.cch.CCHSigner") as cch_mock:
        events = _collect(provider.stream_anthropic(env))

    assert open_stream_mock.call_count == 1, "API key MUST use x-api-key SSE path"
    assert cch_mock.call_count == 0, "API key must NOT touch CCH"
    types = [e["type"] for e in events]
    assert types == ["delta", "delta", "completed"]


def test_stream_anthropic_cch_error_emits_error_and_completed(monkeypatch):
    """Catches: CCH error path missing — silent failure on CCH 4xx/5xx."""
    from lib.hal.utils.cch import CCHError
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-error-test")
    env = HappiEnvelope.parse({
        "v": "happi/1.0",
        "id": "req-cch-error",
        "cmd": "chat",
        "args": ["hi"],
        "flags": {"provider": "claude-opus-4-7"},
    })
    with patch("lib.hal.utils.cch.CCHSigner") as cch_mock:
        cch_mock.return_value.call.side_effect = CCHError("upstream bad gateway")
        events = _collect(provider.stream_anthropic(env))

    types = [e["type"] for e in events]
    assert types == ["error", "completed"]
    assert events[0]["code"] == "provider_cch_error"
    # Token never appears in the surfaced error message (redaction guard).
    assert "sk-ant-oat01" not in events[0]["message"]


def test_stream_anthropic_oauth_path_passes_system_prompt(monkeypatch):
    """Catches: system prompt dropped on the CCH path (envelope contract regression)."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-sys-test")
    env = HappiEnvelope.parse({
        "v": "happi/1.0",
        "id": "req-cch-sys",
        "cmd": "chat",
        "args": ["hello"],
        "flags": {
            "provider": "claude-opus-4-7",
            "system": "You are a friendly bot.",
        },
    })
    with patch("lib.hal.utils.cch.CCHSigner") as cch_mock:
        cch_mock.return_value.call.return_value = _FakeChatResponse("hi", 1, 1)
        list(provider.stream_anthropic(env))

    call_kwargs = cch_mock.return_value.call.call_args.kwargs
    assert call_kwargs["system"] == "You are a friendly bot."
    assert call_kwargs["prompt"] == "hello"
