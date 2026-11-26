"""HAL cross-wire translation — Anthropic Messages -> OpenAI Chat (Phase 0 wave 4).

The hard 20%: when an Anthropic-wire CLIENT (e.g. Claude Code via /v1/messages)
targets an OpenAI-family model, its request must be translated to OpenAI Chat
Completions (and the OpenAI response translated back — subsequent wave). This is
the INVERSE of ``gateway_openai`` (which translates an OpenAI client -> Anthropic
provider).

NEVER used for ``claude-*`` (those stay on the verbatim relay, #3274). The caller
gates on ``model_route`` + ``HAL_GATEWAY_MULTIPROVIDER`` before invoking this. Pure
+ tested; the response translator, wiring, and streaming are subsequent waves.
"""
from __future__ import annotations

import json
from typing import List, Optional, Tuple


def anthropic_to_openai_request(payload: dict) -> dict:
    """Anthropic Messages request -> OpenAI Chat Completions request.

    Inverse of ``gateway_openai.openai_to_anthropic_request``: the top-level
    ``system`` field becomes a system message; ``tool_use`` / ``tool_result``
    blocks become OpenAI ``tool_calls`` / ``tool`` messages; ``tools``
    ``input_schema`` becomes function ``parameters``. The model is passed through
    unchanged (the caller routed by it — it is already a real OpenAI model id).
    """
    messages: List[dict] = []
    system = _system_text(payload.get("system"))
    if system:
        messages.append({"role": "system", "content": system})
    for msg in payload.get("messages", []):
        messages.extend(_translate_message(msg))

    out: dict = {"model": payload.get("model", ""), "messages": messages}
    if payload.get("max_tokens") is not None:
        out["max_tokens"] = int(payload["max_tokens"])
    for key in ("temperature", "top_p", "stream"):
        if payload.get(key) is not None:
            out[key] = payload[key]
    stop = payload.get("stop_sequences")
    if stop:
        out["stop"] = list(stop)
    tools = _translate_tools(payload.get("tools"))
    if tools:
        out["tools"] = tools
        choice = _translate_tool_choice(payload.get("tool_choice"))
        if choice is not None:
            out["tool_choice"] = choice
    return out


def _system_text(system) -> str:
    """Anthropic ``system`` (str OR list of text blocks) -> one string."""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "\n\n".join(
            b.get("text", "") for b in system
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _translate_message(msg) -> List[dict]:
    """One Anthropic message -> one or more OpenAI messages.

    A user message carrying ``tool_result`` blocks expands into separate OpenAI
    ``role="tool"`` messages (OpenAI represents tool results that way).
    """
    if not isinstance(msg, dict):
        return []
    role = msg.get("role")
    content = msg.get("content")
    if isinstance(content, str):
        return [{"role": role, "content": content}]
    if not isinstance(content, list):
        return []
    return _assistant_message(content) if role == "assistant" else _user_message(content)


def _assistant_message(blocks: list) -> List[dict]:
    """Anthropic assistant text + ``tool_use`` blocks -> OpenAI assistant msg."""
    text_parts: List[str] = []
    tool_calls: List[dict] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input") or {}),
                },
            })
    out: dict = {"role": "assistant", "content": "".join(text_parts) or None}
    if tool_calls:
        out["tool_calls"] = tool_calls
    return [out]


def _user_message(blocks: list) -> List[dict]:
    """Anthropic user content blocks -> OpenAI messages.

    ``tool_result`` -> ``role="tool"`` messages; ``text``/``image`` -> a user
    message (string when it is a single text part, else a parts array).
    """
    out: List[dict] = []
    parts: List[dict] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "tool_result":
            out.append({
                "role": "tool",
                "tool_call_id": block.get("tool_use_id", ""),
                "content": _tool_result_text(block.get("content")),
            })
        elif btype == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif btype == "image":
            image = _image_to_openai(block)
            if image:
                parts.append(image)
    if parts:
        if len(parts) == 1 and parts[0]["type"] == "text":
            out.append({"role": "user", "content": parts[0]["text"]})
        else:
            out.append({"role": "user", "content": parts})
    return out


def _tool_result_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _image_to_openai(block: dict) -> Optional[dict]:
    """Anthropic ``image`` block -> OpenAI ``image_url`` part."""
    source = block.get("source") or {}
    if source.get("type") == "base64":
        url = f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
        return {"type": "image_url", "image_url": {"url": url}}
    if source.get("type") == "url":
        return {"type": "image_url", "image_url": {"url": source.get("url", "")}}
    return None


def _translate_tools(tools) -> list:
    """Anthropic tools (``input_schema``) -> OpenAI function tools (``parameters``)."""
    if not tools:
        return []
    out: list = []
    for tool in tools:
        if not isinstance(tool, dict) or not tool.get("name"):
            continue
        out.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
            },
        })
    return out


def _translate_tool_choice(choice):
    """Anthropic tool_choice -> OpenAI tool_choice (None = OpenAI default)."""
    if not isinstance(choice, dict):
        return None
    ctype = choice.get("type")
    if ctype == "any":
        return "required"
    if ctype == "auto":
        return "auto"
    if ctype == "tool" and choice.get("name"):
        return {"type": "function", "function": {"name": choice["name"]}}
    return None


# --- response: OpenAI ChatCompletion -> Anthropic Messages -------------------

# OpenAI finish_reason -> Anthropic stop_reason (inverse of gateway_openai's map).
_OPENAI_FINISH_TO_ANTHROPIC = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}


def openai_to_anthropic_response(data: dict, *, model: str = "") -> dict:
    """OpenAI ChatCompletion response -> Anthropic Messages response.

    Inverse of ``gateway_openai.anthropic_to_openai_response``: the assistant
    message's text becomes a ``text`` block, its ``tool_calls`` become
    ``tool_use`` blocks (arguments JSON-parsed to ``input``), ``finish_reason``
    maps to ``stop_reason``, and token usage is renamed.

    A tool call whose ``arguments`` string was truncated is dropped rather than
    emitted with an empty ``input``, and forces ``stop_reason: max_tokens``.
    """
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message", {}) if isinstance(choice, dict) else {}

    content, truncated = _chat_content_blocks(message)
    finish = choice.get("finish_reason") if isinstance(choice, dict) else None
    usage = data.get("usage", {})
    return {
        "id": data.get("id", ""),
        "type": "message",
        "role": "assistant",
        "model": data.get("model") or model,
        "content": content,
        "stop_reason": _chat_stop_reason(truncated, finish),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def _chat_content_blocks(message: dict) -> Tuple[List[dict], bool]:
    """Assistant message -> (Anthropic content blocks, truncated?).

    A tool call whose ``arguments`` string was cut off mid-token is DROPPED
    rather than emitted: ``_safe_json_obj`` would render it as ``{}``, a
    valid-looking call the client executes with arguments nobody chose.
    """
    if not isinstance(message, dict):
        # A provider that sends "message": "<string>" would otherwise raise
        # AttributeError on .get before the truncation path is even reached.
        return [], False
    content: List[dict] = []
    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})
    truncated = False
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function", {})
        if _arguments_are_corrupt(fn.get("arguments")):
            truncated = True
            continue
        content.append({
            "type": "tool_use",
            "id": call.get("id", ""),
            "name": fn.get("name", ""),
            "input": _safe_json_obj(fn.get("arguments")),
        })
    return content, truncated


def _chat_stop_reason(truncated: bool, finish: Optional[str]) -> str:
    """Anthropic ``stop_reason`` for the chat wire, truncation checked FIRST.

    A dropped truncated call outranks the provider's own ``finish_reason``:
    ``tool_calls`` would promise blocks we just dropped, telling the client the
    turn is complete and those tools should now run. ``max_tokens`` is the
    answer that makes it retry rather than act.

    Deliberately the SAME precedence as the Responses wire's
    ``gateway_crosswire_responses._stop_reason``, which checks truncation
    before ``made_calls`` for the same reason. The two wires agree on purpose;
    a reader who finds them disagreeing has found a bug in one of them.
    """
    if truncated:
        return "max_tokens"
    return _OPENAI_FINISH_TO_ANTHROPIC.get(finish or "", "end_turn")


# --- usage, normalised across both provider wires ---------------------------

def normalise_provider_usage(data: dict) -> Optional[dict]:
    """Provider usage (chat OR Responses shape) -> one normalised token dict.

    ONE helper for both wires on purpose. The two shapes differ only in
    spelling -- chat says ``prompt_tokens`` / ``completion_tokens``, Responses
    says ``input_tokens`` / ``output_tokens`` -- and every translator needs the
    same three corrections, so doing it per-translator is how the cache
    correction below gets fixed twice and stays wrong a third time.

    ``input_tokens`` here is FRESH input: the provider's prompt/input count
    INCLUDES cached tokens, while :func:`cost_tracker.calculate_cost` prices
    cached reads separately and ADDS them. Passing the raw total therefore
    bills the cached portion at the full input rate on top of the discounted
    rate -- on a measured cache-warm turn (5737 prompt, 5734 cached) that is a
    ~9.7x over-count. Subtracting is what makes the two agree. Clamped at zero
    so a provider whose details ever exceed its total cannot go negative.

    Returns ``None`` -- not a dict of zeros -- when the body carries no usable
    usage block. An all-zero dict is TRUTHY in Python, so returning one made
    every downstream ``if usage:`` fire and stamped ``cost_usd: 0.0`` onto a
    row nobody had measured: "free" where the truth is "unknown", which is the
    precise failure the omit-rather-than-zero-fill rule exists to prevent.
    ``None`` is the only value that survives that check honestly.

    """
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return None

    def _n(value) -> int:
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    def _pick(*keys: str) -> int:
        """First key that is PRESENT wins -- never the first that is truthy.

        ``a or b`` reads a legitimate zero as "absent" and silently falls
        through to the other spelling, so a body carrying both
        ``prompt_tokens: 0`` and ``input_tokens: 500`` would be billed 500.
        Presence is the question being asked, so presence is what is tested.
        """
        for key in keys:
            if key in usage:
                return _n(usage.get(key))
        return 0

    def _detail(inner: str, *blocks: str) -> int:
        """Same rule one level down, across the two detail-block spellings."""
        for name in blocks:
            block = usage.get(name)
            if isinstance(block, dict) and inner in block:
                return _n(block.get(inner))
        return 0

    total_in = _pick("prompt_tokens", "input_tokens")
    total_out = _pick("completion_tokens", "output_tokens")
    cached = _detail("cached_tokens", "prompt_tokens_details", "input_tokens_details")
    written = _detail("cache_creation_tokens",
                      "prompt_tokens_details", "input_tokens_details")
    reasoning = _detail("reasoning_tokens",
                        "completion_tokens_details", "output_tokens_details")
    return {
        "input_tokens": max(0, total_in - cached),   # FRESH input only
        "output_tokens": total_out,
        "cache_read_tokens": cached,
        "cache_write_tokens": written,
        "reasoning_tokens": reasoning,
    }


def _arguments_are_corrupt(raw) -> bool:
    """True iff tool-call arguments were SENT but cannot be read as an object.

    :func:`_safe_json_obj` collapses two very different situations to the same
    ``{}``: a real zero-argument call (``arguments`` absent, empty, or ``"{}"``)
    and a call whose argument string was cut off mid-token
    (``'{"command": "rm -r'``). Only this predicate separates them, and the
    difference decides whether a ``tool_use`` block may be emitted at all --
    passing a truncated call on as ``{}`` hands the client a syntactically
    valid instruction to run the tool with fabricated-empty arguments.
    """
    if isinstance(raw, dict) or not raw:
        return False                       # absent/empty -> a real zero-arg call
    if not isinstance(raw, str):
        return True
    try:
        return not isinstance(json.loads(raw), dict)
    except (json.JSONDecodeError, TypeError):
        return True


def _safe_json_obj(raw) -> dict:
    """Parse tool-call arguments JSON into a dict; {} on failure.

    Callers that must not pass a truncated call to a client should ask
    :func:`_arguments_are_corrupt` first -- this function cannot tell an empty
    call from a broken one.
    """
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


# --- response: Anthropic Messages (complete) -> Anthropic SSE ----------------

def anthropic_message_to_sse_chunks(msg: dict) -> List[bytes]:
    """A COMPLETE Anthropic Messages response -> a valid Anthropic SSE sequence.

    Buffered translation: we already hold the whole message (the provider was
    called non-streamed), so we emit ONE ``content_block_delta`` per block, not
    token-incremental deltas. The event ORDER and shapes match Anthropic's
    streaming contract (``message_start`` -> per-block start/delta/stop ->
    ``message_delta`` -> ``message_stop``) so an Anthropic-wire client (Claude
    Code) parses it natively. True token-incremental streaming is a later wave;
    this is what makes cross-wire end-to-end usable now.

    A ``text`` block emits a ``text_delta``; a ``tool_use`` block emits an
    ``input_json_delta`` carrying the whole input as one ``partial_json`` string
    (the client concatenates partials, so a single complete chunk is valid).
    """
    content = msg.get("content") or []
    usage = msg.get("usage") or {}
    events: List[bytes] = [_message_start_event(msg, usage)]
    for index, block in enumerate(content):
        if isinstance(block, dict):
            events.extend(_block_events(index, block))
    # Output only. This burst already reported input_tokens in message_start
    # (the whole message is in hand before the first frame is written), so
    # repeating it here would let a client that sums usage across frames count
    # the input twice. The streamed translator passes input here BECAUSE its
    # message_start could not carry it -- the difference is which frame already
    # told the truth, and only the caller knows that.
    events.append(_message_delta_event(msg, {"output_tokens": usage.get("output_tokens", 0)}))
    events.append(_sse_event("message_stop", {"type": "message_stop"}))
    return events


def _message_start_event(msg: dict, usage: dict) -> bytes:
    """The ``message_start`` frame — ONLY canonical envelope keys, rebuilt
    explicitly (never a passthrough of ``msg``) so an unexpected / non-JSON-
    serialisable source field can never leak into or crash the stream. ``content``
    is empty (it streams via the block events) and ``stop_reason`` + output_tokens
    are null/0 here — their final values arrive in ``message_delta`` (Anthropic's
    streaming contract)."""
    start_message = {
        "id": msg.get("id", ""),
        "type": "message",
        "role": msg.get("role", "assistant"),
        "model": msg.get("model", ""),
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": 0},
    }
    return _sse_event("message_start", {"type": "message_start", "message": start_message})


#: Usage keys a ``message_delta`` frame may carry, beyond the always-present
#: ``output_tokens``. An explicit allowlist, not a passthrough: the frame
#: builders rebuild every envelope field by name so an unexpected or
#: non-JSON-serialisable source field can never leak into or crash the stream.
_DELTA_USAGE_KEYS = ("input_tokens", "cache_read_tokens",
                     "cache_write_tokens", "reasoning_tokens")


def _message_delta_event(msg: dict, usage: dict) -> bytes:
    """The ``message_delta`` frame — final stop_reason/stop_sequence + usage.

    ``output_tokens`` is always present (Anthropic delivers the final count
    here rather than in ``message_start``). The input and cache counts ride
    here too when the caller has them: on a translated OpenAI stream the
    provider sends usage in its LAST chunk, by which point ``message_start``
    has long gone out, so this is the earliest honest place to report input.
    Hardcoding output_tokens alone -- as this did -- meant a streamed
    cross-wire turn could only ever report zero input, however correctly the
    caller had counted it.
    """
    counts = {"output_tokens": int(usage.get("output_tokens", 0) or 0)}
    for key in _DELTA_USAGE_KEYS:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            counts[key] = value
    return _sse_event("message_delta", {
        "type": "message_delta",
        "delta": {
            "stop_reason": msg.get("stop_reason"),
            "stop_sequence": msg.get("stop_sequence"),
        },
        "usage": counts,
    })


def _block_events(index: int, block: dict) -> List[bytes]:
    """The start/delta/stop SSE triple for one content block."""
    btype = block.get("type")
    if btype == "text":
        start = {"type": "text", "text": ""}
        delta = {"type": "text_delta", "text": block.get("text", "")}
    elif btype == "tool_use":
        start = {"type": "tool_use", "id": block.get("id", ""),
                 "name": block.get("name", ""), "input": {}}
        delta = {"type": "input_json_delta",
                 "partial_json": json.dumps(block.get("input") or {})}
    else:
        return []
    return [
        _sse_event("content_block_start",
                   {"type": "content_block_start", "index": index, "content_block": start}),
        _sse_event("content_block_delta",
                   {"type": "content_block_delta", "index": index, "delta": delta}),
        _sse_event("content_block_stop",
                   {"type": "content_block_stop", "index": index}),
    ]


def _sse_event(event_name: str, data_obj: dict) -> bytes:
    """One Anthropic SSE event frame: ``event: <name>\\ndata: <json>\\n\\n``."""
    return (f"event: {event_name}\n"
            f"data: {json.dumps(data_obj)}\n\n").encode("utf-8")
