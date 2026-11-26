"""Gemini thought_signature round-trip cache (HAL gateway).

Gemini 3.x is a THINKING-ONLY model: it cannot run with thinking disabled
(``thinking_budget=0`` / ``reasoning_effort=none`` -> HTTP 400 "This model only
works in thinking mode"). With function calling it returns, on each tool call, a
``thought_signature`` at::

    choices[i].message.tool_calls[j].extra_content.google.thought_signature

Google REQUIRES that signature be echoed back in the conversation history on the
next turn, or it 400s ("Function call is missing a thought_signature in
functionCall parts. ... Please use /v1/responses instead"). OpenAI-compat clients
(OpenCode via ``@ai-sdk/openai-compatible``) DROP the unknown ``extra_content``
field between turns, so the next request's assistant ``tool_calls`` lack the
signature and the request fails.

Fix (proven live against :4010 — turn 2 WITH the re-injected signature = HTTP 200,
WITHOUT = HTTP 400): HAL captures ``{tool_call_id -> thought_signature}`` from
Gemini responses and re-injects it into assistant ``tool_calls`` that lack one on
later requests, matched by ``tool_call.id`` (which clients DO preserve across
turns). Bounded LRU; Gemini-ONLY; pure dict<->dict + byte helpers, no network.

Monotone-toward-safe: capture and re-inject are best-effort and never raise into
the relay path; a miss degrades to the pre-fix behaviour (the same 400 the user
already saw), never worse, and NON-Gemini traffic is byte-identical (verbatim).
"""
from __future__ import annotations

import json
from collections import OrderedDict
from typing import Optional

# Bound the cache so a long-lived gateway cannot grow it without limit. Tool
# calls are short-lived within a conversation; 512 distinct in-flight tool_call
# ids is far above any real agentic session's working set.
_MAX_ENTRIES = 512
_cache: "OrderedDict[str, str]" = OrderedDict()


def _is_gemini(model) -> bool:
    """Whether ``model`` names a Gemini variant (provider-prefix + case tolerant).

    The round-trip fix is Gemini-specific; every other provider is untouched."""
    return "gemini" in str(model or "").split("/")[-1].lower()


def _sig_of(tool_call) -> Optional[str]:
    """The thought_signature carried by a tool_call, or None.

    Reads ``tool_call.extra_content.google.thought_signature`` defensively (any
    non-dict / missing layer -> None)."""
    if not isinstance(tool_call, dict):
        return None
    ec = tool_call.get("extra_content")
    g = ec.get("google") if isinstance(ec, dict) else None
    sig = g.get("thought_signature") if isinstance(g, dict) else None
    return sig if isinstance(sig, str) and sig else None


def remember(tool_call_id: str, signature: str) -> None:
    """Cache one ``{id -> signature}`` (LRU; bounded). No-op on empty inputs."""
    if not tool_call_id or not signature:
        return
    _cache[tool_call_id] = signature
    _cache.move_to_end(tool_call_id)
    while len(_cache) > _MAX_ENTRIES:
        _cache.popitem(last=False)


# --- capture (response side): {id -> signature} from JSON or SSE -------------

def _remember_tool_calls(tool_calls) -> int:
    """Remember every (id, signature) pair in a tool_calls list. Returns count."""
    n = 0
    for tc in tool_calls or []:
        sig = _sig_of(tc)
        tid = tc.get("id") if isinstance(tc, dict) else None
        if sig and tid:
            remember(tid, sig)
            n += 1
    return n


def _remember_from_message_obj(obj: dict) -> int:
    """Capture from a NON-stream OpenAI ChatCompletion response object."""
    n = 0
    for choice in obj.get("choices") or []:
        msg = choice.get("message") if isinstance(choice, dict) else None
        n += _remember_tool_calls(msg.get("tool_calls") if isinstance(msg, dict) else None)
    return n


def _iter_sse_data_objs(text: str):
    """Yield each parsed ``data:`` JSON object from an SSE body (skips ``[DONE]``
    and unparseable lines). Scans lines directly — SSE block grouping is
    irrelevant to extracting the ``data:`` payloads."""
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            yield json.loads(payload)
        except (ValueError, TypeError):
            continue


def _merge_choice_deltas(by_index: dict, choice) -> None:
    """Merge one streamed choice's tool_call deltas into ``by_index`` (id +
    signature accrete per tool_call ``index``)."""
    delta = choice.get("delta") if isinstance(choice, dict) else None
    for tc in (delta.get("tool_calls") if isinstance(delta, dict) else None) or []:
        if not isinstance(tc, dict):
            continue
        slot = by_index.setdefault(tc.get("index", 0), {})
        if tc.get("id"):
            slot["id"] = tc["id"]
        sig = _sig_of(tc)
        if sig:
            slot["sig"] = sig


def _merge_tool_call_deltas(by_index: dict, ev) -> None:
    """Merge one SSE event's choices into ``by_index``."""
    for choice in (ev.get("choices") if isinstance(ev, dict) else None) or []:
        _merge_choice_deltas(by_index, choice)


def _remember_from_sse(text: str) -> int:
    """Capture from a streamed SSE body. Tool-call deltas stream incrementally and
    the ``id`` and the ``extra_content`` may arrive in different deltas at the same
    ``index`` — so merge by index across all ``data:`` events, then remember any
    index that resolved both an id and a signature."""
    by_index: dict = {}
    for ev in _iter_sse_data_objs(text):
        _merge_tool_call_deltas(by_index, ev)
    n = 0
    for slot in by_index.values():
        if slot.get("id") and slot.get("sig"):
            remember(slot["id"], slot["sig"])
            n += 1
    return n


def capture_from_bytes(raw) -> int:
    """Capture ``{id -> signature}`` from a Gemini response body (JSON or SSE).

    Returns the number of signatures cached. Best-effort: returns 0 on any parse
    failure (the relay already delivered the bytes to the client; capture is a
    side-effect that must never break a served response)."""
    if not raw:
        return 0
    text = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
    if text.lstrip().startswith("{"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "choices" in obj:
                return _remember_from_message_obj(obj)
        except (ValueError, TypeError):
            pass  # fall through to SSE parsing
    return _remember_from_sse(text)


# --- re-inject (request side): cached signature -> assistant tool_calls -------

def _reinject_tool_calls(tool_calls) -> None:
    """Inject the cached signature into each tool_call that lacks one (by id).
    Never overwrites an existing signature; no-op when the id is not cached."""
    for tc in tool_calls or []:
        if not isinstance(tc, dict) or _sig_of(tc) is not None:
            continue
        sig = _cache.get(tc.get("id", ""))
        if not sig:
            continue
        ec = tc.get("extra_content")
        if not isinstance(ec, dict):
            ec = {}
            tc["extra_content"] = ec
        g = ec.get("google")
        if not isinstance(g, dict):
            g = {}
            ec["google"] = g
        g["thought_signature"] = sig
        _cache.move_to_end(tc["id"])


def reinject_request(req: dict) -> dict:
    """Re-inject cached signatures into a Gemini request's assistant tool_calls.

    Mutates and returns ``req``. No-op for non-Gemini models, for tool_calls that
    already carry a signature (never overwritten), and when the id is not cached."""
    if not isinstance(req, dict) or not _is_gemini(req.get("model", "")):
        return req
    for msg in req.get("messages") or []:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            _reinject_tool_calls(msg.get("tool_calls"))
    return req


def reinject_raw(raw):
    """Parse a raw request body, re-inject Gemini signatures, re-serialise.

    NON-Gemini (or unparseable) bodies are returned VERBATIM (byte-identical) so
    only Gemini traffic is ever rewritten."""
    try:
        req = json.loads(raw)
    except (ValueError, TypeError):
        return raw
    if not (isinstance(req, dict) and _is_gemini(req.get("model", ""))):
        return raw
    reinject_request(req)
    return json.dumps(req).encode("utf-8")
