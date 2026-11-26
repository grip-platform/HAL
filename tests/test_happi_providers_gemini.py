"""Golden tests for the Gemini provider adapter (α-3.6).

Exercises token resolution (apikey scheme + GEMINI_API_KEY env var +
Keychain fallback), model resolution including `models/` prefix strip,
envelope → Gemini body translation (contents+parts shape), SSE parsing
without [DONE] sentinel, text extraction from multi-part content,
usageMetadata handling, the four error paths (HTTP / network / upstream /
missing credentials), token redaction for Google-style AIzaSy... keys,
and runtime dispatch of `cmd: chat` envelopes with `gemini/*` provider
specs. All HTTP traffic is mocked — no network dependency.
"""
from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from lib.hal.happi.envelope import HappiEnvelope
from lib.hal.happi.providers import gemini as provider
from lib.hal.happi.runtime import run


class _FakeSSEResponse:
    """Minimal Gemini-style SSE response (no [DONE] sentinel; stream ends at EOF)."""

    def __init__(self, lines: list[str]):
        self._lines = [line.encode("utf-8") for line in lines]
        self.closed = False

    def __iter__(self):
        return iter(self._lines)

    def close(self):
        self.closed = True


def _envelope(**overrides) -> HappiEnvelope:
    payload = {
        "v": "happi/1.0",
        "id": "req-test",
        "cmd": "chat",
        "args": ["Hello, HAPPI."],
        "flags": {"provider": "gemini/gemini-2.0-flash", "max_tokens": 256},
        "auth": {"scheme": "apikey", "token": "AIzaSyFakeTestKeyValue0123456789abc"},
    }
    payload.update(overrides)
    return HappiEnvelope.parse(payload)


def _collect(events_iter) -> list[dict]:
    return [json.loads(line) for line in events_iter]


def _successful_sse_lines() -> list[str]:
    return [
        'data: {"candidates":[{"content":{"parts":[{"text":"Hel"}],"role":"model"},"index":0}],"modelVersion":"gemini-2.0-flash"}',
        "",
        'data: {"candidates":[{"content":{"parts":[{"text":"lo"}],"role":"model"},"index":0}],"modelVersion":"gemini-2.0-flash"}',
        "",
        'data: {"candidates":[{"content":{"parts":[{"text":"!"}],"role":"model"},"finishReason":"STOP","index":0}],'
        '"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":3,"totalTokenCount":7},'
        '"modelVersion":"gemini-2.0-flash"}',
        "",
    ]


# --- Token redaction --------------------------------------------------------

def test_redact_strips_google_api_keys():
    text = "config: AIzaSyRealLookingGoogleKeyValue1234567 uploaded"
    out = provider._redact(text)
    assert "AIzaSy" not in out
    assert "[REDACTED]" in out


def test_redact_strips_bearer_tokens():
    text = "Authorization: Bearer abc.xyz_123 failed"
    out = provider._redact(text)
    assert "Bearer abc" not in out


# --- Model resolution ------------------------------------------------------

def test_resolve_model_uses_default_for_bare_family():
    env = _envelope(flags={"provider": "gemini"})
    assert provider._resolve_model(env) == "gemini-2.0-flash"


def test_resolve_model_uses_provided_model():
    env = _envelope(flags={"provider": "gemini/gemini-1.5-pro"})
    assert provider._resolve_model(env) == "gemini-1.5-pro"


def test_resolve_model_strips_models_prefix():
    env = _envelope(flags={"provider": "gemini/models/gemini-1.5-flash"})
    assert provider._resolve_model(env) == "gemini-1.5-flash"


# --- Token resolution ------------------------------------------------------

def test_resolve_token_apikey_literal():
    env = _envelope(auth={"scheme": "apikey", "token": "AIzaSy-literal-test-key"})
    assert provider._resolve_token(env) == "AIzaSy-literal-test-key"


def test_resolve_token_subscription_reads_env_var(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-from-env-test")
    env = _envelope(auth={"scheme": "subscription"})
    assert provider._resolve_token(env) == "AIzaSy-from-env-test"


def test_resolve_token_falls_back_to_keychain(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    env = _envelope(auth={"scheme": "subscription"})
    with patch.object(provider, "_fetch_keychain_token", return_value="AIzaSy-kc"):
        assert provider._resolve_token(env) == "AIzaSy-kc"


def test_resolve_token_returns_none_when_all_sources_fail(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    env = _envelope(auth={"scheme": "subscription"})
    with patch.object(provider, "_fetch_keychain_token", return_value=None):
        assert provider._resolve_token(env) is None


# --- Request body ----------------------------------------------------------

def test_build_request_body_uses_args_when_no_message_flag():
    env = _envelope(args=["Tell", "me", "a", "joke"])
    body = provider._build_request_body(env)
    assert body["contents"] == [
        {"role": "user", "parts": [{"text": "Tell me a joke"}]}
    ]
    assert body["generationConfig"]["maxOutputTokens"] == 256


def test_build_request_body_prefers_message_flag_over_args():
    env = _envelope(flags={"provider": "gemini/gemini-2.0-flash", "message": "Custom text"})
    body = provider._build_request_body(env)
    assert body["contents"][0]["parts"][0]["text"] == "Custom text"


def test_build_request_body_includes_system_instruction():
    env = _envelope(flags={"provider": "gemini", "system": "Be concise."})
    body = provider._build_request_body(env)
    assert body["systemInstruction"] == {"parts": [{"text": "Be concise."}]}


# --- Sampling flags → generationConfig -------------------------------------

def test_build_request_body_threads_temperature_zero():
    # AGORA's determinism case: temperature 0.0 must reach generationConfig.
    env = _envelope(flags={"provider": "gemini", "temperature": 0.0})
    gen = provider._build_request_body(env)["generationConfig"]
    assert gen["temperature"] == 0.0


def test_build_request_body_maps_top_p_to_camelcase():
    env = _envelope(flags={"provider": "gemini", "top_p": 0.9})
    gen = provider._build_request_body(env)["generationConfig"]
    assert gen["topP"] == 0.9
    assert "top_p" not in gen  # Gemini wire name is camelCase


def test_build_request_body_threads_seed():
    env = _envelope(flags={"provider": "gemini", "seed": 42})
    gen = provider._build_request_body(env)["generationConfig"]
    assert gen["seed"] == 42


def test_build_request_body_omits_sampling_by_default():
    # Backwards-compat: with no sampling flags, generationConfig carries only
    # maxOutputTokens (byte-identical to pre-feature behaviour).
    gen = provider._build_request_body(_envelope())["generationConfig"]
    assert gen == {"maxOutputTokens": 256}


# --- Text + usage extraction -----------------------------------------------

def test_extract_text_concatenates_multi_part_candidate():
    data = {"candidates": [
        {"content": {"parts": [{"text": "foo"}, {"text": "bar"}, {"text": "baz"}]}}
    ]}
    assert provider._extract_text_from_candidate(data) == "foobarbaz"


def test_extract_text_returns_none_when_no_candidates():
    assert provider._extract_text_from_candidate({"candidates": []}) is None
    assert provider._extract_text_from_candidate({}) is None


def test_extract_usage_carries_forward_when_meta_absent():
    prev = {"in": 5, "out": 10}
    assert provider._extract_usage({}, prev) == prev


# --- Successful streaming --------------------------------------------------

def test_successful_stream_emits_deltas_and_usage():
    env = _envelope()
    fake = _FakeSSEResponse(_successful_sse_lines())
    with patch.object(provider, "_open_stream", return_value=fake):
        events = _collect(provider.stream_gemini(env))
    types = [e["type"] for e in events]
    assert types == ["delta", "delta", "delta", "completed"]
    assert [e["text"] for e in events[:3]] == ["Hel", "lo", "!"]
    assert events[-1]["usage"] == {"in": 4, "out": 3}
    assert fake.closed is True


def test_stream_handles_upstream_error_event():
    env = _envelope()
    err_lines = [
        'data: {"error":{"message":"bad key","status":"UNAUTHENTICATED","code":401}}',
        "",
    ]
    fake = _FakeSSEResponse(err_lines)
    with patch.object(provider, "_open_stream", return_value=fake):
        events = _collect(provider.stream_gemini(env))
    assert [e["type"] for e in events] == ["error", "completed"]
    assert events[0]["code"] == "provider_error:UNAUTHENTICATED"


def test_http_403_emits_http_failure():
    env = _envelope()
    http_err = urllib.error.HTTPError(
        url="https://generativelanguage.googleapis.com/...",
        code=403, msg="Forbidden", hdrs={}, fp=io.BytesIO(b'{"error":"quota"}'),
    )
    with patch.object(provider, "_open_stream", side_effect=http_err):
        events = _collect(provider.stream_gemini(env))
    assert [e["type"] for e in events] == ["error", "completed"]
    assert events[0]["code"] == "provider_http_403"


def test_network_error_emits_network_failure():
    env = _envelope()
    url_err = urllib.error.URLError("DNS lookup failed")
    with patch.object(provider, "_open_stream", side_effect=url_err):
        events = _collect(provider.stream_gemini(env))
    assert [e["type"] for e in events] == ["error", "completed"]
    assert events[0]["code"] == "provider_network"


def test_missing_credentials_emits_auth_failure(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    env = _envelope(auth={"scheme": "subscription"})
    with patch.object(provider, "_fetch_keychain_token", return_value=None):
        events = _collect(provider.stream_gemini(env))
    assert [e["type"] for e in events] == ["error", "completed"]
    assert events[0]["code"] == "missing_credentials"


# --- Runtime dispatch integration ------------------------------------------

def test_runtime_routes_gemini_spec_to_gemini_adapter(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-runtime-test-value")
    payload = {
        "v": "happi/1.0",
        "id": "req-runtime-gemini",
        "cmd": "chat",
        "args": ["hi"],
        "flags": {"provider": "gemini/gemini-2.0-flash"},
    }
    fake = _FakeSSEResponse(_successful_sse_lines())
    stdin = io.StringIO(json.dumps(payload))
    stdout = io.StringIO()
    with patch.object(provider, "_open_stream", return_value=fake):
        rc = run(stream_in=stdin, stream_out=stdout)
    assert rc == 0
    events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    assert [e["type"] for e in events] == ["started", "delta", "delta", "delta", "completed"]


def test_runtime_routes_bare_gemini_to_default_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-bare-family-test")
    payload = {
        "v": "happi/1.0",
        "id": "req-runtime-gemini-bare",
        "cmd": "chat",
        "args": ["hi"],
        "flags": {"provider": "gemini"},
    }
    fake = _FakeSSEResponse(_successful_sse_lines())
    stdin = io.StringIO(json.dumps(payload))
    stdout = io.StringIO()
    with patch.object(provider, "_open_stream", return_value=fake):
        rc = run(stream_in=stdin, stream_out=stdout)
    assert rc == 0
    events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    types = [e["type"] for e in events]
    assert types[0] == "started" and types[-1] == "completed"


# --- P5: v1.1 acceptance + v1.0 regression --------------------------------

def test_gemini_accepts_v11_envelope(monkeypatch):
    """A v1.1 envelope dispatches successfully through the gemini adapter."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-v11-test-key")
    payload = {
        "v": "happi/1.1",
        "id": "req-v11-gemini",
        "cmd": "chat",
        "args": ["hi"],
        "flags": {"provider": "gemini/gemini-2.0-flash"},
    }
    fake = _FakeSSEResponse(_successful_sse_lines())
    stdin = io.StringIO(json.dumps(payload))
    stdout = io.StringIO()
    with patch.object(provider, "_open_stream", return_value=fake):
        rc = run(stream_in=stdin, stream_out=stdout)
    assert rc == 0
    events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    types = [e["type"] for e in events]
    assert types == ["started", "delta", "delta", "delta", "completed"]


def test_gemini_v10_envelope_unchanged(monkeypatch):
    """Regression guard: v1.0 envelopes dispatch byte-equivalently post-P5."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-v10-regress")
    payload = {
        "v": "happi/1.0",
        "id": "req-v10-gemini-regress",
        "cmd": "chat",
        "args": ["hi"],
        "flags": {"provider": "gemini/gemini-2.0-flash"},
    }
    fake = _FakeSSEResponse(_successful_sse_lines())
    stdin = io.StringIO(json.dumps(payload))
    stdout = io.StringIO()
    with patch.object(provider, "_open_stream", return_value=fake):
        rc = run(stream_in=stdin, stream_out=stdout)
    assert rc == 0
    events = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    types = [e["type"] for e in events]
    assert types == ["started", "delta", "delta", "delta", "completed"]
    assert [e["text"] for e in events if e["type"] == "delta"] == ["Hel", "lo", "!"]
