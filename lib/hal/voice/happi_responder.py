"""HAPPI-backed Responder adapter for voice.chat_loop (γ-1).

Voice's ``chat_loop(responder=...)`` takes a ``Responder`` callable
matching ``(user_text: str, history: list[ChatTurn]) -> str``. Before
this adapter, operators who wanted voice to speak LLM responses had to
write their own per-provider client (urllib → Anthropic, urllib →
OpenAI, etc.). This module provides a single factory that binds any
HAPPI provider family to that Responder interface.

Under the hood the adapter constructs a HAPPI ``chat`` envelope, runs
it through ``runtime._dispatch`` (the same dispatch path used by stdin,
the β-1 HTTP transport, the β-3 Unix socket, and the γ-2 queue runner),
concatenates every ``delta.text`` event, and returns the assembled
string. Errors surface as ``VoiceError`` so chat_loop can bubble them
via its existing exception path.

Why this is γ-1, not α-4: the original plan note framed γ-1 as "delete
voice's duplicate socket IPC". Investigation showed voice has NO socket
IPC — it's a pure in-process pipeline with a callable seam. The right
γ-1 shape therefore bridges voice's callable seam into the HAPPI
runtime. Operators stop re-inventing LLM clients; voice inherits the
full provider roster (anthropic, gemini, openai, groq, mistral,
deepseek, openrouter, moonshot, ollama) through one import.

Usage::

    from lib.hal.voice.happi_responder import happi_responder
    from lib.hal.voice import chat_loop, ChatPolicy

    respond = happi_responder(provider="gemini", system="Be concise.")
    transcript = chat_loop(
        profile="default",
        policy=ChatPolicy(max_turns=5),
        responder=respond,
    )

Stdlib-only — the adapter never touches urllib itself; the HAPPI
runtime owns all HTTP traffic.
"""
from __future__ import annotations

import json
import secrets
from typing import Callable

from lib.hal.happi.envelope import HappiEnvelope
from lib.hal.happi.runtime import _dispatch
from lib.hal.voice.types import ChatTurn, VoiceError


DEFAULT_MAX_TOKENS = 4096


def _format_history(
    user_text: str,
    history: list[ChatTurn],
    system: str | None,
) -> str:
    """Render voice history + current user turn into a ChatML-style prompt.

    HAPPI α-3's chat envelope takes a single ``message`` string, not a
    structured messages array. The adapter flattens history into a
    role-labelled prompt that every provider interprets sensibly:

        System: <system prompt>
        User: <turn 0 user text>
        Assistant: <turn 0 response>
        User: <turn 1 user text>
        Assistant: <turn 1 response>
        ...
        User: <current user text>

    Note: the system prompt is ALSO passed via ``flags.system`` when
    supported by the adapter (Anthropic sends it as a top-level
    ``system`` field; OpenAI/Gemini prepend it as a system message).
    We include it in the flattened prompt too as a belt-and-braces
    fallback for providers that ignore the flag.
    """
    lines: list[str] = []
    if system:
        lines.append(f"System: {system}")
    for turn in history:
        lines.append(f"User: {turn.listen.text}")
        lines.append(f"Assistant: {turn.response_text}")
    lines.append(f"User: {user_text}")
    return "\n".join(lines)


def _build_envelope(
    user_text: str,
    history: list[ChatTurn],
    provider: str,
    system: str | None,
    max_tokens: int,
) -> HappiEnvelope:
    flags: dict[str, object] = {
        "provider": provider,
        "message": _format_history(user_text, history, system),
        "max_tokens": max_tokens,
    }
    if system:
        flags["system"] = system
    return HappiEnvelope.parse({
        "v": "happi/1.0",
        "id": f"voice-{secrets.token_hex(4)}",
        "cmd": "chat",
        "args": [],
        "flags": flags,
    })


def _collect_response_text(envelope: HappiEnvelope) -> str:
    """Dispatch an envelope, concatenate delta.text events, raise on error.

    The HAPPI event stream is the authoritative source of truth — we do
    not re-implement any provider-specific behaviour here. If the stream
    yields an ``error`` event, that surfaces as ``VoiceError`` to match
    the Responder contract (chat_loop catches Exception at the call
    site).
    """
    text_parts: list[str] = []
    for line in _dispatch(envelope):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            # Runtime should not emit non-JSON but if it ever does we
            # treat it as corrupted and bail rather than silently
            # swallowing. voice.chat_loop's caller can retry.
            raise VoiceError(f"HAPPI runtime emitted non-JSON line: {line!r}")

        etype = event.get("type")
        if etype == "delta":
            text_parts.append(str(event.get("text", "")))
        elif etype == "error":
            raise VoiceError(
                f"HAPPI {event.get('code', 'unknown')}: "
                f"{event.get('message', 'provider error')}"
            )

    return "".join(text_parts)


def happi_responder(
    *,
    provider: str = "",
    system: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Callable[[str, list[ChatTurn]], str]:
    """Return a voice Responder that routes LLM calls through HAPPI.

    Args:
        provider: HAPPI provider spec — empty string means the runtime's
            default (anthropic). Accepts ``"gemini"``, ``"gemini/gemini-2.0-flash"``,
            ``"groq/llama-3.1-70b-versatile"``, ``"openrouter/anthropic/claude-sonnet-4-5"``,
            etc. Any provider family supported by
            :func:`lib.hal.happi.runtime._dispatch_chat` works.
        system: Optional system prompt. Passed through ``flags.system``
            (where supported) AND prepended to the flattened prompt as
            a belt-and-braces fallback.
        max_tokens: Cap on response length. Forwarded as ``flags.max_tokens``
            to the provider adapter.

    Returns:
        A callable matching :data:`lib.hal.voice.Responder` suitable
        for ``voice.chat_loop(responder=...)``.
    """
    def _responder(user_text: str, history: list[ChatTurn]) -> str:
        envelope = _build_envelope(
            user_text=user_text,
            history=history,
            provider=provider,
            system=system,
            max_tokens=max_tokens,
        )
        return _collect_response_text(envelope)

    return _responder


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "happi_responder",
]
