"""Regression: every chat provider adapter must honour ``envelope.content``.

Origin: CodeTonight-SA/GRIP#2697 — "AGORA Council Review FAILED". Root cause:
GRIP's council dispatch (``lib/hal_council_dispatch.py::_build_envelope``)
carries the review prompt + PR diff solely in the HAPPI envelope's
``content`` field. The Anthropic adapter resolved ``content`` first, but the
OpenAI-compat and Gemini adapters only looked at ``flags.message`` /
``args`` — so every non-Anthropic Speaker received an EMPTY user message,
free-associated ("Please provide the code you would like me to review"),
and AGORA's validity gate correctly rejected the whole review as filler.

The fix centralises message resolution in
``HappiEnvelope.user_message()`` (content > flags.message > args) and routes
all three adapters through it.

Goodhart protection (PARAMOUNT — rules/efficiency-rules.md Rule 14):

  * The tests assert the *actual prompt text* lands in the provider request
    body, not that a function was merely called. A stub that returned a
    fixed string, or an adapter that silently dropped ``content``, fails
    these assertions.
  * ``test_*_empty_envelope_yields_empty_message`` LOCKS the fail-loud
    property: a genuinely contentless envelope must still resolve to ``""``
    so AGORA's empty-Speaker discriminator keeps failing loud rather than
    hallucinating. The fix must NOT paper over emptiness — if it did, this
    test would fail.
  * ``test_agora_validity_gate_still_fails_loud_on_empty_diff`` is the
    end-to-end anchor: it proves the anti-hallucination gate the original
    bug report praised ("a clear failure beats a hallucinated review")
    still triggers when there is genuinely nothing to review.

All HTTP traffic is mocked — no network dependency.
"""
from __future__ import annotations

import json
from unittest.mock import patch

from lib.hal.happi.envelope import HappiEnvelope
from lib.hal.happi.providers import anthropic as anthropic_provider
from lib.hal.happi.providers import gemini as gemini_provider
from lib.hal.happi.providers import openai_compat as openai_provider


# A realistic AGORA Speaker prompt: instruction text plus an embedded unified
# diff. This is the exact envelope SHAPE GRIP's _build_envelope produces —
# prompt only in `content`, nothing in flags.message or args.
_AGORA_PROMPT = (
    "You are Speaker C in a 3-Speaker AGORA council. Cite file:line.\n\n"
    "PR title: feat: legaltech monitor ingest\n\n"
    "<<<PR_DIFF_START>>>\n"
    "diff --git a/lib/legaltech_monitor_ingest.py "
    "b/lib/legaltech_monitor_ingest.py\n"
    "+++ b/lib/legaltech_monitor_ingest.py\n"
    "@@ -1,3 +1,9 @@\n"
    "+def ingest(source: str) -> int:\n"
    "+    return _persist(_normalise(source))\n"
    "<<<PR_DIFF_END>>>\n"
)

_DIFF_FINGERPRINT = "legaltech_monitor_ingest.py"


def _content_only_envelope(provider: str) -> HappiEnvelope:
    """Envelope exactly as GRIP's AGORA dispatch builds it: prompt in content."""
    return HappiEnvelope.parse(
        {
            "v": "happi/1.0",
            "id": "agora-technical-deadbeef",
            "cmd": "chat",
            "content": _AGORA_PROMPT,
            "flags": {"provider": provider, "max_tokens": 2048,
                      "system": "You are a code reviewer."},
            "auth": {"scheme": "apikey", "token": "sk-faketokenvalue0123456789abcd"},
        }
    )


# ---------------------------------------------------------------------------
# HappiEnvelope.user_message — canonical precedence (content > message > args)
# ---------------------------------------------------------------------------

def test_user_message_prefers_content():
    env = HappiEnvelope.parse(
        {
            "v": "happi/1.0", "id": "r1", "cmd": "chat",
            "content": "REAL PROMPT",
            "flags": {"message": "stale flag message"},
            "args": ["stale", "args"],
        }
    )
    assert env.user_message() == "REAL PROMPT"


def test_user_message_falls_back_to_flag_message_when_no_content():
    env = HappiEnvelope.parse(
        {
            "v": "happi/1.0", "id": "r1", "cmd": "chat",
            "flags": {"message": "flag message"},
            "args": ["args", "fallback"],
        }
    )
    assert env.user_message() == "flag message"


def test_user_message_falls_back_to_args_when_no_content_or_flag():
    env = HappiEnvelope.parse(
        {"v": "happi/1.0", "id": "r1", "cmd": "chat", "args": ["hello", "world"]}
    )
    assert env.user_message() == "hello world"


def test_user_message_empty_when_envelope_carries_no_prompt():
    """Fail-loud anchor: nothing in content/message/args resolves to ''."""
    env = HappiEnvelope.parse({"v": "happi/1.0", "id": "r1", "cmd": "chat"})
    assert env.user_message() == ""


# ---------------------------------------------------------------------------
# Adapter body construction — the actual bug. The PROMPT must reach the wire.
# ---------------------------------------------------------------------------

def test_anthropic_body_embeds_content_prompt():
    env = _content_only_envelope("claude-opus-4-7")
    body = anthropic_provider._build_request_body(env)
    sent = body["messages"][0]["content"]
    assert sent == _AGORA_PROMPT
    assert _DIFF_FINGERPRINT in sent  # the diff actually reached the model


def test_openai_compat_body_embeds_content_prompt():
    """The core regression: this adapter previously dropped envelope.content."""
    env = _content_only_envelope("groq/llama-3.3-70b-versatile")
    body = openai_provider._build_request_body("llama-3.3-70b-versatile", env)
    user_msgs = [m for m in body["messages"] if m["role"] == "user"]
    assert len(user_msgs) == 1
    sent = user_msgs[0]["content"]
    assert sent == _AGORA_PROMPT
    assert _DIFF_FINGERPRINT in sent  # before the fix this was "" — no diff


def test_gemini_body_embeds_content_prompt():
    """The core regression: this adapter previously dropped envelope.content."""
    env = _content_only_envelope("gemini/gemini-2.5-flash")
    body = gemini_provider._build_request_body(env)
    sent = body["contents"][0]["parts"][0]["text"]
    assert sent == _AGORA_PROMPT
    assert _DIFF_FINGERPRINT in sent  # before the fix this was "" — no diff


# ---------------------------------------------------------------------------
# Fail-loud preservation: an empty envelope must still produce an empty
# message in every adapter (so AGORA's discriminator keeps failing loud).
# This is the Goodhart guard — the fix must not "succeed" on emptiness.
# ---------------------------------------------------------------------------

def _empty_envelope(provider: str) -> HappiEnvelope:
    return HappiEnvelope.parse(
        {"v": "happi/1.0", "id": "empty-1", "cmd": "chat",
         "flags": {"provider": provider}}
    )


def test_anthropic_empty_envelope_yields_empty_message():
    body = anthropic_provider._build_request_body(_empty_envelope("claude-opus-4-7"))
    assert body["messages"][0]["content"] == ""


def test_openai_compat_empty_envelope_yields_empty_message():
    body = openai_provider._build_request_body(
        "llama-3.3-70b-versatile", _empty_envelope("groq/llama-3.3-70b-versatile")
    )
    user_msgs = [m for m in body["messages"] if m["role"] == "user"]
    assert user_msgs[0]["content"] == ""


def test_gemini_empty_envelope_yields_empty_message():
    body = gemini_provider._build_request_body(_empty_envelope("gemini/gemini-2.5-flash"))
    assert body["contents"][0]["parts"][0]["text"] == ""


# ---------------------------------------------------------------------------
# End-to-end: a content-only envelope produces a NON-empty completion through
# the OpenAI-compat adapter (the formerly-broken path). Proves the prompt
# reaches a (mocked) provider and a real answer streams back.
# ---------------------------------------------------------------------------

class _FakeSSEResponse:
    def __init__(self, lines: list[str]):
        self._lines = [line.encode("utf-8") for line in lines]
        self.closed = False

    def __iter__(self):
        return iter(self._lines)

    def close(self):
        self.closed = True


def test_openai_compat_content_only_streams_real_completion():
    env = _content_only_envelope("groq/llama-3.3-70b-versatile")
    captured: dict[str, object] = {}

    def fake_open(family, body, token, timeout):
        captured["body"] = body
        return _FakeSSEResponse(
            [
                'data: {"choices":[{"delta":{"content":"### Concrete concerns"}}]}',
                "",
                'data: {"choices":[{"delta":{"content":" lib/x.py:1 ok"}}],'
                '"usage":{"prompt_tokens":120,"completion_tokens":5}}',
                "",
                "data: [DONE]",
                "",
            ]
        )

    with patch.object(openai_provider, "_open_stream", side_effect=fake_open):
        events = [json.loads(l) for l in openai_provider.stream_openai_compat(env)]

    # The prompt (with the diff) actually went on the wire:
    body = captured["body"]
    user_msg = next(m for m in body["messages"] if m["role"] == "user")
    assert _DIFF_FINGERPRINT in user_msg["content"]

    # A real completion streamed back (not filler):
    deltas = "".join(e.get("text", "") for e in events if e.get("type") == "delta")
    assert "Concrete concerns" in deltas
    assert any(e.get("type") == "completed" for e in events)


# ---------------------------------------------------------------------------
# AGORA validity-gate anchor: prove the anti-hallucination gate the original
# bug report praised still fails loud when the diff is genuinely empty. The
# fix repairs the dispatch; it must NOT weaken this gate.
# ---------------------------------------------------------------------------

def test_agora_validity_gate_still_fails_loud_on_empty_diff():
    import importlib.util
    import sys
    from pathlib import Path

    claude_lib = Path.home() / ".claude" / "lib"
    spec = importlib.util.spec_from_file_location(
        "agora_pr_review", claude_lib / "agora_pr_review.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover - env guard
        import pytest

        pytest.skip("GRIP agora_pr_review.py not importable from this env")
    if str(claude_lib.parent) not in sys.path:
        sys.path.insert(0, str(claude_lib.parent))
    agora = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(agora)
    except Exception:  # pragma: no cover - env guard
        import pytest

        pytest.skip("agora_pr_review.py deps unavailable in this env")

    # Empty diff → review MUST be invalid (fail loud, never render content).
    is_valid, reasons = agora._validate_review(
        plain="some plain text", quality="some quality text",
        technical="some technical text", diff_excerpt="",
    )
    assert is_valid is False
    assert any("diff was empty" in r.lower() for r in reasons)

    # Filler Speaker output → invalid even if a diff is present.
    diff = (
        "diff --git a/lib/x.py b/lib/x.py\n+++ b/lib/x.py\n"
        "@@ -1 +1 @@\n+x = 1\n"
    )
    is_valid2, reasons2 = agora._validate_review(
        plain="Please provide the code you would like me to review.",
        quality="I understand. I will review the code and provide feedback.",
        technical="Please provide the code you would like me to review.",
        diff_excerpt=diff,
    )
    assert is_valid2 is False
    assert reasons2  # filler markers caught — gate still fails loud
