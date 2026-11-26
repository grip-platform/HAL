"""Tests for the HAPPI-backed voice Responder adapter (γ-1).

Covers the factory's behaviour end-to-end by patching each provider's
``_open_stream`` with hand-curated SSE fixtures so the adapter exercises
real runtime dispatch (no provider network traffic). Verifies:

  - delta events concatenate into the Responder's return string
  - all three adapter families (anthropic, openai_compat, gemini) route
    cleanly
  - history is flattened into a ChatML-style prompt that survives
    round-trip through the envelope
  - system prompt propagates to both flags.system and the flattened
    message body
  - error events raise VoiceError (for chat_loop to bubble)
  - the returned callable matches the Responder type signature
  - voice's existing primitives are untouched (baseline 115 tests still
    pass — verified by running the voice suite alongside this file)
"""
from __future__ import annotations

import io
import json
import urllib.error
from typing import Iterable
from unittest.mock import patch

import pytest

from lib.hal.happi.providers import anthropic as anthropic_provider
from lib.hal.happi.providers import gemini as gemini_provider
from lib.hal.happi.providers import openai_compat as openai_provider
from lib.hal.voice import ChatTurn, ListenResult, SpeakResult, Provenance, VoiceError
from lib.hal.voice.happi_responder import (
    DEFAULT_MAX_TOKENS,
    _build_envelope,
    _format_history,
    happi_responder,
)


# ---------------- Fixture helpers ----------------


class _FakeSSEResponse:
    def __init__(self, lines: Iterable[str]):
        self._lines = [line.encode("utf-8") for line in lines]
        self.closed = False

    def __iter__(self):
        return iter(self._lines)

    def close(self):
        self.closed = True


def _anthropic_sse(texts: list[str]) -> list[str]:
    """Emit anthropic-style SSE frames for `texts` deltas."""
    lines = [
        "event: message_start",
        'data: {"type":"message_start","message":{"id":"m","model":"claude-opus-4-7"}}',
        "",
    ]
    for text in texts:
        lines.append("event: content_block_delta")
        lines.append(
            'data: {"type":"content_block_delta","index":0,'
            f'"delta":{{"type":"text_delta","text":{json.dumps(text)}}}}}'
        )
        lines.append("")
    lines.extend([
        "event: message_delta",
        'data: {"type":"message_delta","usage":{"input_tokens":3,"output_tokens":4}}',
        "",
        "event: message_stop",
        'data: {"type":"message_stop"}',
        "",
    ])
    return lines


def _openai_sse(texts: list[str]) -> list[str]:
    lines = []
    for text in texts:
        lines.append(
            f'data: {{"id":"c","choices":[{{"delta":{{"content":{json.dumps(text)}}},'
            '"index":0,"finish_reason":null}]}'
        )
        lines.append("")
    lines.extend([
        'data: {"id":"c","choices":[{"delta":{},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":3,"completion_tokens":4}}',
        "",
        "data: [DONE]",
        "",
    ])
    return lines


def _gemini_sse(texts: list[str]) -> list[str]:
    lines = []
    for i, text in enumerate(texts):
        extra = ""
        if i == len(texts) - 1:
            extra = (
                ',"usageMetadata":{"promptTokenCount":3,'
                '"candidatesTokenCount":4}'
            )
        lines.append(
            f'data: {{"candidates":[{{"content":{{"parts":[{{"text":{json.dumps(text)}}}],'
            f'"role":"model"}}}}]{extra}}}'
        )
        lines.append("")
    return lines


def _fake_turn(user_text: str, response: str) -> ChatTurn:
    """Minimal ChatTurn suitable for history-formatting tests."""
    listen = ListenResult(
        ok=True,
        text=user_text,
        audio_path=None,
        language="en-US",
        confidence=0.95,
        duration_s=1.0,
        provider="mock",
        provenance=Provenance.voice_human,
        tier=1,
        audit_id="a",
    )
    speak = SpeakResult(
        ok=True,
        audio_path=None,
        duration_s=1.0,
        provider="mock",
        voice="mock",
        profile="default",
        emotion="neutral",
        directive_fingerprint="fp",
        tier=1,
        audit_id="b",
    )
    return ChatTurn(
        turn_index=0,
        listen=listen,
        response_text=response,
        speak=speak,
        total_latency_s=3.0,
        latency_breakdown={"stt": 1.0, "llm": 1.0, "tts": 1.0},
    )


# ---------------- _format_history ----------------


def test_format_history_with_no_turns_just_returns_user_prompt():
    out = _format_history("hello", [], system=None)
    assert out == "User: hello"


def test_format_history_prepends_system_prompt_when_provided():
    out = _format_history("hi", [], system="Be concise.")
    assert out == "System: Be concise.\nUser: hi"


def test_format_history_interleaves_user_and_assistant_turns():
    history = [
        _fake_turn("first", "one"),
        _fake_turn("second", "two"),
    ]
    out = _format_history("third", history, system=None)
    assert out == (
        "User: first\n"
        "Assistant: one\n"
        "User: second\n"
        "Assistant: two\n"
        "User: third"
    )


def test_format_history_combines_system_and_history_and_current_turn():
    history = [_fake_turn("q1", "a1")]
    out = _format_history("q2", history, system="You are HAL.")
    assert out.splitlines() == [
        "System: You are HAL.",
        "User: q1",
        "Assistant: a1",
        "User: q2",
    ]


# ---------------- _build_envelope ----------------


def test_build_envelope_sets_provider_and_max_tokens():
    env = _build_envelope(
        user_text="hi",
        history=[],
        provider="groq/llama-3.1-70b-versatile",
        system=None,
        max_tokens=256,
    )
    assert env.cmd == "chat"
    assert env.flags["provider"] == "groq/llama-3.1-70b-versatile"
    assert env.flags["max_tokens"] == 256
    assert env.flags["message"] == "User: hi"
    assert "system" not in env.flags


def test_build_envelope_includes_system_flag_when_provided():
    env = _build_envelope(
        user_text="hi", history=[], provider="",
        system="Be concise.", max_tokens=100,
    )
    assert env.flags["system"] == "Be concise."
    # Also belt-and-braces in the prompt:
    assert "System: Be concise." in env.flags["message"]


def test_build_envelope_generates_unique_ids():
    e1 = _build_envelope(
        "hi", [], provider="", system=None, max_tokens=100,
    )
    e2 = _build_envelope(
        "hi", [], provider="", system=None, max_tokens=100,
    )
    assert e1.id != e2.id
    assert e1.id.startswith("voice-")


# ---------------- happi_responder: provider routing ----------------


def test_responder_routes_to_anthropic_default_provider():
    respond = happi_responder()
    fake = _FakeSSEResponse(_anthropic_sse(["Hel", "lo", "!"]))
    with patch.object(anthropic_provider, "_open_stream", return_value=fake):
        # monkeypatch token resolution to avoid keychain probes
        with patch.object(anthropic_provider, "_resolve_token", return_value="fake"):
            out = respond("hi there", [])
    assert out == "Hello!"


def test_responder_routes_to_explicit_gemini_provider():
    respond = happi_responder(provider="gemini/gemini-2.0-flash")
    fake = _FakeSSEResponse(_gemini_sse(["The ", "weather ", "is fine."]))
    with patch.object(gemini_provider, "_open_stream", return_value=fake):
        with patch.object(gemini_provider, "_resolve_token", return_value="fake"):
            out = respond("how's the weather?", [])
    assert out == "The weather is fine."


def test_responder_routes_to_openai_compat_groq():
    respond = happi_responder(provider="groq/llama-3.1-70b-versatile")
    fake = _FakeSSEResponse(_openai_sse(["Fast ", "local ", "inference."]))
    with patch.object(openai_provider, "_open_stream", return_value=fake):
        with patch.object(openai_provider, "_resolve_token", return_value="fake"):
            out = respond("groq pitch?", [])
    assert out == "Fast local inference."


# ---------------- happi_responder: error handling ----------------


def test_responder_raises_voice_error_on_http_failure():
    respond = happi_responder(provider="gemini")
    http_err = urllib.error.HTTPError(
        url="u", code=403, msg="Forbidden", hdrs={}, fp=io.BytesIO(b'{}'),
    )
    with patch.object(gemini_provider, "_open_stream", side_effect=http_err):
        with patch.object(gemini_provider, "_resolve_token", return_value="fake"):
            with pytest.raises(VoiceError) as excinfo:
                respond("hi", [])
    assert "provider_http_403" in str(excinfo.value)


def test_responder_raises_voice_error_on_missing_credentials():
    respond = happi_responder(provider="openai/gpt-4o-mini")
    # Token resolution returns None -> adapter emits missing_credentials.
    with patch.object(openai_provider, "_resolve_token", return_value=None):
        with pytest.raises(VoiceError) as excinfo:
            respond("hi", [])
    assert "missing_credentials" in str(excinfo.value)


def test_responder_raises_voice_error_on_provider_upstream_error():
    respond = happi_responder(provider="gemini")
    # Simulate a Gemini-style upstream error event in the SSE stream.
    err_sse = [
        'data: {"error":{"message":"quota exceeded","status":"RESOURCE_EXHAUSTED"}}',
        "",
    ]
    fake = _FakeSSEResponse(err_sse)
    with patch.object(gemini_provider, "_open_stream", return_value=fake):
        with patch.object(gemini_provider, "_resolve_token", return_value="fake"):
            with pytest.raises(VoiceError) as excinfo:
                respond("hi", [])
    assert "RESOURCE_EXHAUSTED" in str(excinfo.value) or "provider_error" in str(excinfo.value)


# ---------------- happi_responder: history carry-through ----------------


def test_responder_includes_history_in_envelope_message(monkeypatch):
    """The flattened prompt must appear in the envelope body the adapter sees."""
    captured = {}

    def _capture_body(body, token, timeout):
        captured["body"] = body
        # Return an empty-looking Anthropic stream.
        return _FakeSSEResponse(_anthropic_sse(["ok"]))

    respond = happi_responder()
    history = [_fake_turn("earlier question", "earlier answer")]
    with patch.object(anthropic_provider, "_open_stream", side_effect=_capture_body):
        with patch.object(anthropic_provider, "_resolve_token", return_value="fake"):
            respond("current question", history)

    body_json = captured["body"]
    user_content = body_json["messages"][0]["content"]
    assert "User: earlier question" in user_content
    assert "Assistant: earlier answer" in user_content
    assert "User: current question" in user_content


def test_responder_passes_system_prompt_to_anthropic_body():
    captured = {}

    def _capture(body, token, timeout):
        captured["body"] = body
        return _FakeSSEResponse(_anthropic_sse(["ok"]))

    respond = happi_responder(system="Be HAL.")
    with patch.object(anthropic_provider, "_open_stream", side_effect=_capture):
        with patch.object(anthropic_provider, "_resolve_token", return_value="fake"):
            respond("hi", [])
    # Anthropic top-level system field:
    assert captured["body"].get("system") == "Be HAL."


# ---------------- happi_responder: contract ----------------


def test_default_max_tokens_constant_is_4096():
    assert DEFAULT_MAX_TOKENS == 4096


def test_returned_callable_accepts_responder_signature():
    """The factory's output must match Callable[[str, list[ChatTurn]], str]."""
    respond = happi_responder()
    # Type-check: no TypeError when called with the right positional args.
    fake = _FakeSSEResponse(_anthropic_sse(["x"]))
    with patch.object(anthropic_provider, "_open_stream", return_value=fake):
        with patch.object(anthropic_provider, "_resolve_token", return_value="fake"):
            out = respond("u", [])
    assert isinstance(out, str)


def test_responder_concatenates_many_deltas():
    """A response split across many SSE chunks concatenates cleanly."""
    respond = happi_responder()
    deltas = list("word by word voice response here.")
    fake = _FakeSSEResponse(_anthropic_sse(deltas))
    with patch.object(anthropic_provider, "_open_stream", return_value=fake):
        with patch.object(anthropic_provider, "_resolve_token", return_value="fake"):
            out = respond("hi", [])
    assert out == "word by word voice response here."
