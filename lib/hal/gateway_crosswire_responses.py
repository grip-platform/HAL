"""Anthropic Messages <-> OpenAI Responses translation (pure, no I/O).

The chat cross-wire (``gateway_crosswire``) turns an Anthropic Messages request
into an OpenAI Chat Completions request. OpenAI's gpt-6 family (Astra) refuses
function tools on that endpoint: a chat/completions request for
``gpt-6-astra`` that carries ``tools`` answers HTTP 400 "Function tools with
reasoning_effort are not supported for gpt-6-astra in /v1/chat/completions. To
use function tools, use /v1/responses ..." (measured live 2026-09-10, and the
vendor's model page says "tool calling requires the Responses API"). Claude
Code always sends its tool definitions, so on that wire a Claude Code session
on Astra fails on its first message.

This module is the Responses-wire twin: the same Anthropic request becomes an
OpenAI Responses request, and a Responses object becomes an Anthropic message.
It reuses the chat translator's block parsing (``_system_text``,
``_translate_message``, ``_translate_tools``, ``_translate_tool_choice``) and
re-shapes the result into Responses items, so the two wires cannot drift on how
an Anthropic block is read. Which model family takes this wire is decided ONCE,
in ``model_quirks.tools_require_responses_api`` -- not here.

First version is buffered: the request is sent with ``stream: false`` and a
streaming client receives the complete message re-emitted as one Anthropic SSE
burst (``gateway_crosswire.anthropic_message_to_sse_chunks``), the same fallback
the chat wire already uses for providers that ignore ``stream``. Token-
incremental Responses streaming (``response.output_text.delta`` events) is a
named follow-up, not a hidden gap.
"""
from __future__ import annotations

import json
from typing import List, Optional

from lib.hal.gateway_crosswire import (
    _arguments_are_corrupt,
    _safe_json_obj,
    _system_text,
    _translate_message,
    _translate_tool_choice,
    _translate_tools,
)
from lib.hal.model_quirks import reasoning_effort_levels

__all__ = ["anthropic_to_responses_request", "responses_to_anthropic_response"]


# --- request: Anthropic Messages -> OpenAI Responses -------------------------

def anthropic_to_responses_request(payload: dict) -> dict:
    """Anthropic Messages request -> OpenAI Responses request (non-streaming).

    ``system`` becomes ``instructions``; each message becomes one or more
    ``input`` items (user/assistant messages, ``function_call`` for a
    ``tool_use`` block, ``function_call_output`` for a ``tool_result`` block);
    ``tools`` become flat function tools (``name``/``parameters`` at top level,
    unlike the chat wire's nested ``function`` object); ``max_tokens`` becomes
    ``max_output_tokens``. ``output_config.effort`` is forwarded as
    ``reasoning.effort`` only when the model documents that level (Astra refuses
    ``none``: an undocumented level is omitted so the vendor default applies).
    ``store`` is always false -- the gateway keeps custody of the transcript,
    OpenAI retains nothing -- and ``stream`` is always false (buffered v1).
    """
    items: List[dict] = []
    for msg in payload.get("messages", []):
        for chat_msg in _translate_message(msg):
            items.extend(_chat_message_to_items(chat_msg))

    out: dict = {"model": payload.get("model", ""), "input": items,
                 "stream": False, "store": False}
    system = _system_text(payload.get("system"))
    if system:
        out["instructions"] = system
    if payload.get("max_tokens") is not None:
        out["max_output_tokens"] = int(payload["max_tokens"])
    # The chat wire drops the sampling fields a model refuses
    # (_fix_gpt6_request_params). This wire is reached by an EARLY RETURN that
    # happens before that runs, so it has to make the same drop itself, from
    # the same single source of truth -- otherwise gpt-6 gets a guaranteed 400.
    from lib.hal.model_quirks import locked_sampling_params
    locked = locked_sampling_params(payload.get("model", ""))
    for key in ("temperature", "top_p"):
        if payload.get(key) is not None and key not in locked:
            out[key] = payload[key]
    tools = [_flatten_tool(t) for t in _translate_tools(payload.get("tools"))]
    if tools:
        out["tools"] = tools
        choice = _translate_tool_choice(payload.get("tool_choice"))
        if choice is not None:
            out["tool_choice"] = _flatten_tool_choice(choice)
    effort = _documented_effort(payload)
    if effort:
        out["reasoning"] = {"effort": effort}
    return out


def _chat_message_to_items(chat_msg: dict) -> List[dict]:
    """One OpenAI chat message (the chat translator's output) -> Responses items."""
    role = chat_msg.get("role")
    if role == "tool":
        return [{"type": "function_call_output",
                 "call_id": chat_msg.get("tool_call_id", ""),
                 "output": chat_msg.get("content") or ""}]
    if role == "assistant":
        return _assistant_items(chat_msg)
    content = chat_msg.get("content")
    if isinstance(content, str):
        return [{"role": "user", "content": [{"type": "input_text", "text": content}]}]
    parts = [_user_part(p) for p in (content or []) if isinstance(p, dict)]
    parts = [p for p in parts if p is not None]
    return [{"role": "user", "content": parts}] if parts else []


def _assistant_items(chat_msg: dict) -> List[dict]:
    items: List[dict] = []
    text = chat_msg.get("content")
    if text:
        items.append({"role": "assistant",
                      "content": [{"type": "output_text", "text": text}]})
    for call in chat_msg.get("tool_calls") or []:
        # Guard the CALL once and skip it outright. An earlier version tested
        # isinstance(call, dict) and then called call.get("id") on the next
        # line, which defeated the guard it had just applied.
        if not isinstance(call, dict):
            continue
        fn = call.get("function")
        if not isinstance(fn, dict):
            fn = {}
        items.append({"type": "function_call", "call_id": call.get("id", ""),
                      "name": fn.get("name", ""), "arguments": fn.get("arguments", "{}")})
    return items


def _user_part(part: dict) -> Optional[dict]:
    """A chat content part -> a Responses input part (``input_text`` / ``input_image``)."""
    if part.get("type") == "text":
        return {"type": "input_text", "text": part.get("text", "")}
    if part.get("type") == "image_url":
        url = (part.get("image_url") or {}).get("url", "")
        return {"type": "input_image", "image_url": url} if url else None
    return None


def _flatten_tool(tool: dict) -> dict:
    """Chat-wire ``{"type":"function","function":{...}}`` -> Responses flat tool."""
    fn = tool.get("function", {})
    return {"type": "function", "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}}}


def _flatten_tool_choice(choice):
    """``"auto"`` / ``"required"`` pass through; a named tool loses its nesting."""
    if isinstance(choice, dict):
        name = (choice.get("function") or {}).get("name", "")
        return {"type": "function", "name": name}
    return choice


def _documented_effort(payload: dict) -> str:
    """The Anthropic ``output_config.effort`` level, when the model documents it."""
    config = payload.get("output_config")
    level = str((config or {}).get("effort", "") or "").strip().lower() if isinstance(config, dict) else ""
    return level if level and level in reasoning_effort_levels(payload.get("model", "")) else ""


# --- response: OpenAI Responses -> Anthropic Messages ------------------------

def responses_to_anthropic_response(data: dict, *, model: str = "") -> dict:
    """OpenAI Responses object -> Anthropic Messages response.

    ``message`` output items contribute their ``output_text`` parts to one
    ``text`` block; each ``function_call`` becomes a ``tool_use`` block whose
    ``id`` is the ``call_id`` the client must echo back; ``reasoning`` items are
    dropped (no Anthropic shape carries them). A ``function_call`` whose
    ``arguments`` string was truncated is DROPPED rather than passed on with an
    empty ``input``. ``stop_reason`` reports ``max_tokens`` whenever the turn
    was cut off -- ahead of ``tool_use`` -- then ``tool_use`` when a complete
    call was made, else ``end_turn``.
    """
    texts: List[str] = []
    calls: List[dict] = []
    truncated = False
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            texts.extend(p.get("text", "") for p in (item.get("content") or [])
                         if isinstance(p, dict) and p.get("type") == "output_text")
        elif item.get("type") == "function_call":
            if _arguments_are_corrupt(item.get("arguments")):
                # The argument string was cut off mid-token. Emitting the block
                # anyway would hand the client `Bash{}` where the model wrote
                # `Bash{command: "rm -r ..."}` -- a valid-looking call it would
                # execute with arguments nobody chose. Drop it and report the
                # truncation instead (see `_stop_reason`).
                truncated = True
                continue
            calls.append({"type": "tool_use",
                          "id": item.get("call_id") or item.get("id", ""),
                          "name": item.get("name", ""),
                          "input": _safe_json_obj(item.get("arguments"))})
    content: List[dict] = []
    if any(texts):
        content.append({"type": "text", "text": "".join(texts)})
    content.extend(calls)
    usage = data.get("usage") or {}
    return {
        "id": data.get("id", ""),
        "type": "message",
        "role": "assistant",
        "model": data.get("model") or model,
        "content": content,
        "stop_reason": _stop_reason(data, bool(calls), truncated=truncated),
        "stop_sequence": None,
        "usage": {"input_tokens": usage.get("input_tokens", 0) or 0,
                  "output_tokens": usage.get("output_tokens", 0) or 0},
    }


def _stop_reason(data: dict, made_calls: bool, *, truncated: bool = False) -> str:
    """Anthropic ``stop_reason``, with truncation checked BEFORE tool use.

    The order is the whole point. A response cut off on ``max_output_tokens``
    can still carry a ``function_call``, so asking ``made_calls`` first reports
    ``tool_use`` on a turn the model never finished -- telling the client the
    turn is complete and these tools should now run. ``max_tokens`` is the
    honest answer there, and it is the one that makes a client retry rather
    than act.

    ``truncated`` covers the case the provider does not flag: an argument
    string that ends mid-token, whose block the translator has already dropped.
    """
    details = data.get("incomplete_details") or {}
    if truncated or (data.get("status") == "incomplete"
                     and details.get("reason") == "max_output_tokens"):
        return "max_tokens"
    return "tool_use" if made_calls else "end_turn"
