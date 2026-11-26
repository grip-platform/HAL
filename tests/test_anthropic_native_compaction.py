"""Hermetic unit tests for server-side compaction in the native Anthropic provider.

Covers GRIP overnight RSI deliverable H781/H782: opt-in
``context_management`` on :meth:`AnthropicProvider.chat` adds the
``anthropic-beta: compact-2026-01-12`` header AND the ``context_management``
body field, default ``None`` is byte-identical prior behaviour, and the full
response content blocks (incl. the ``compaction`` block) round-trip on
``ChatResponse.raw_content``.

All HTTP traffic is mocked — no network, no real Anthropic call. Each test is
Goodhart-resistant (Rule 14): it asserts the EXACT request shape / returned
values, and is written to fail under an obvious mutation of the production code.
"""
from __future__ import annotations

import io
import json
from unittest.mock import patch

import pytest

from lib.hal.providers import anthropic_native as native
from lib.hal.providers.anthropic_native import (
    _COMPACT_BETA,
    AnthropicError,
    AnthropicProvider,
)

_COMPACT_EDIT = {"edits": [{"type": "compact_20260112"}]}


def _provider() -> AnthropicProvider:
    # api_key set literally → no keychain lookup, forced API-key transport.
    return AnthropicProvider(api_key="sk-ant-api03-test", prefer="api_key")


def _capture_request():
    """Patch _do_request to record (method, url, body, betas) and return a stub.

    The stub response carries a text block + a compaction block so round-trip
    preservation can be asserted in the same call.
    """
    captured: dict = {}

    def _fake(self, method, url, body, *, betas=None):  # noqa: ANN001
        captured["method"] = method
        captured["url"] = url
        captured["body"] = body
        captured["betas"] = betas
        return {
            "model": body["model"],
            "stop_reason": "end_turn",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "compaction", "compacted_tokens": 4096},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }

    return captured, _fake


# --- W1: opt-in adds body field + beta header -------------------------------

def test_context_management_adds_body_field_and_beta_header():
    """Opted in → body carries context_management AND betas carries the beta.

    Mutation guard: if either ``body["context_management"] = ...`` or
    ``betas.append(_COMPACT_BETA)`` is removed from chat(), one assertion fires.
    """
    captured, fake = _capture_request()
    with patch.object(native.AnthropicProvider, "_do_request", fake):
        _provider().chat("hi", context_management=_COMPACT_EDIT)

    assert captured["body"]["context_management"] == _COMPACT_EDIT
    assert captured["betas"] == [_COMPACT_BETA]
    assert _COMPACT_BETA == "compact-2026-01-12"  # exact canonical beta name


def test_no_context_management_omits_field_and_beta():
    """Back-compat: not opted in → no body field, no betas (byte-identical).

    Goodhart-anchor: if the opt-in were applied unconditionally (mutation),
    ``context_management`` would appear in the body and this fails.
    """
    captured, fake = _capture_request()
    with patch.object(native.AnthropicProvider, "_do_request", fake):
        _provider().chat("hi")

    assert "context_management" not in captured["body"]
    assert captured["betas"] == []
    # The rest of the body is the unchanged classic shape.
    assert captured["body"] == {
        "model": "claude-opus-4-7",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16384,
    }


def test_beta_header_lands_on_the_wire():
    """End of the chain: betas → ``anthropic-beta`` HTTP header in _do_request.

    Patches urlopen (the real transport boundary) and asserts the request
    object carries the header. Mutation guard: if the
    ``headers['anthropic-beta'] = ','.join(betas)`` line is removed, the header
    is absent and this fails.
    """
    captured_headers: dict = {}

    class _Resp:
        headers: dict = {}

        def read(self):
            return json.dumps({
                "model": "claude-opus-4-7",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, timeout=120):  # noqa: ANN001
        # urllib lowercases header keys on the Request object.
        captured_headers.update(req.headers)
        return _Resp()

    with patch.object(native.urllib.request, "urlopen", _fake_urlopen):
        _provider().chat("hi", context_management=_COMPACT_EDIT)

    # urllib capitalises header names: "anthropic-beta" → "Anthropic-beta".
    assert captured_headers.get("Anthropic-beta") == _COMPACT_BETA


def test_no_beta_header_when_not_opted_in():
    """Back-compat at the wire: no anthropic-beta header when context_management is None."""
    captured_headers: dict = {}

    class _Resp:
        headers: dict = {}

        def read(self):
            return json.dumps({
                "model": "claude-opus-4-7",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, timeout=120):  # noqa: ANN001
        captured_headers.update(req.headers)
        return _Resp()

    with patch.object(native.urllib.request, "urlopen", _fake_urlopen):
        _provider().chat("hi")

    assert "Anthropic-beta" not in captured_headers


# --- W2: full content blocks round-trip on raw_content ----------------------

def test_raw_content_preserves_all_blocks_including_compaction():
    """raw_content holds EVERY block verbatim and in order (incl. compaction).

    Mutation guard: if ``raw_content=content_blocks`` is dropped from
    _parse_response (or set to text only), the compaction block is missing and
    this fails. ``text`` still extracts only the text block (unchanged).
    """
    captured, fake = _capture_request()
    with patch.object(native.AnthropicProvider, "_do_request", fake):
        resp = _provider().chat("hi", context_management=_COMPACT_EDIT)

    # Full ordered blocks preserved — text AND compaction.
    assert resp.raw_content == [
        {"type": "text", "text": "hello"},
        {"type": "compaction", "compacted_tokens": 4096},
    ]
    # text field is the text-only concatenation (compaction block excluded).
    assert resp.text == "hello"
    # The compaction block is present in raw_content but NOT in text.
    types = [b["type"] for b in resp.raw_content]
    assert "compaction" in types
    assert "compaction" not in resp.text


def test_raw_content_roundtrips_into_next_request():
    """The caller can append raw_content to history and send it back intact.

    Demonstrates the contract the brief mandates: the FULL content (not just
    text) is what feeds the next request, keeping the compaction chain alive.
    """
    captured, fake = _capture_request()
    with patch.object(native.AnthropicProvider, "_do_request", fake):
        first = _provider().chat("hi", context_management=_COMPACT_EDIT)

    # Build the next request's message history from the full blocks.
    history = [
        native.ChatMessage(role="user", content="hi"),
        native.ChatMessage(role="assistant", content=first.raw_content),
    ]
    captured2, fake2 = _capture_request()
    with patch.object(native.AnthropicProvider, "_do_request", fake2):
        _provider().chat("", messages=history, context_management=_COMPACT_EDIT)

    sent_assistant = captured2["body"]["messages"][1]
    assert sent_assistant["role"] == "assistant"
    # The compaction block survived the round-trip into the next request body.
    assert {"type": "compaction", "compacted_tokens": 4096} in sent_assistant["content"]


# --- Misuse guard: subscription transport rejects the opt-in ----------------

def test_context_management_on_subscription_transport_raises():
    """The CCH subscription path filters beta headers → loud refusal, not silent drop.

    Mutation guard: if the AnthropicError guard is removed, the call would fall
    through to subscription.chat (which silently can't honour compaction) and
    this stops raising.
    """
    class _FakeSub:
        def health_check(self):
            return True

        def chat(self, *a, **kw):
            raise AssertionError("subscription.chat must NOT be reached")

    prov = AnthropicProvider(subscription=_FakeSub(), prefer="subscription")
    with pytest.raises(AnthropicError, match="context_management requires the API-key"):
        prov.chat("hi", context_management=_COMPACT_EDIT)


def test_subscription_without_opt_in_unaffected():
    """Back-compat: subscription path still works normally with no opt-in."""
    sentinel = object()

    class _FakeSub:
        def health_check(self):
            return True

        def chat(self, prompt, **kw):  # noqa: ANN001
            assert "context_management" not in kw
            return sentinel

    prov = AnthropicProvider(subscription=_FakeSub(), prefer="subscription")
    assert prov.chat("hi") is sentinel
