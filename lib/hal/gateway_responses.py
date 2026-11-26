"""HAL Gateway — OpenAI Responses API ``/v1/responses`` endpoint (P6).

Lets OpenAI Codex CLI (``wire_api="responses"``) point at the gateway and
reach Anthropic through it. Translates Responses <-> Anthropic Messages,
both streaming (SSE) and non-streaming. Mirrors P2 (gateway_openai.py) in
structure; reuses its auth + SSE parsing helpers directly.

Design: pure dict<->dict translators, DIP proxy injection, fail-open.
"""
from __future__ import annotations

import json
import secrets
from http.server import BaseHTTPRequestHandler
from typing import Iterable, Iterator, List, Optional, Tuple

from lib.hal.gateway import (
    GatewayProxy,
    failure_reply,
    openai_wire_response,
    openai_wire_status,
    read_request_body,
    redact,
    relay_response,
    retry_headers,
)
from lib.hal.gateway_crosswire import anthropic_message_to_sse_chunks
from lib.hal.gateway_openai import (
    CUT_SHORT_MESSAGE,
    anthropic_upstream_headers,
    guarded_sse_events,
    upstream_error_text,
    send_openai_spend_refusal,
)
from lib.hal.gateway_translate import (
    resolve_model,
    translate_content_blocks,
)
from lib.hal.spend_guard import SpendRefused

RESPONSES_PATH = "/v1/responses"

# Providers whose OpenAI-compatible endpoint speaks the Responses API natively,
# so a Responses-wire request routes to them VERBATIM (no Anthropic pivot). Only
# OpenAI today; gemini/groq/xai/deepseek expose chat-completions only and still
# pivot. Keep in sync with gateway_egress.PROVIDER_ENDPOINTS.
_RESPONSES_NATIVE_PROVIDERS = frozenset({"openai"})
_MESSAGES_PATH = "/v1/messages"
_DEFAULT_MAX_TOKENS = 4096


# ---------------------------------------------------------------------------
# Request: OpenAI Responses -> Anthropic Messages
# ---------------------------------------------------------------------------

# Anthropic /v1/messages accepts ONLY these roles in the messages array; any
# system/developer content must live in the top-level `system` field. OpenAI's
# Responses API sends the system prompt as a "developer" (or "system") role
# message item — passing it through verbatim makes Anthropic 400 with
# "Unexpected role 'developer'. Allowed roles are 'user' or 'assistant'".
_SYSTEM_ROLES = frozenset({"system", "developer"})
_ASSISTANT_ROLE = "assistant"


def responses_to_anthropic_request(payload: dict) -> dict:
    """Translate an OpenAI Responses request body into Anthropic Messages."""
    system_text, messages = _translate_input(payload.get("input", []))
    out: dict = {
        "model": resolve_model(payload.get("model", "")),
        "max_tokens": int(payload.get("max_output_tokens") or _DEFAULT_MAX_TOKENS),
        "messages": messages,
    }
    system = _join_system(payload.get("instructions"), system_text)
    if system:
        out["system"] = system
    if payload.get("temperature") is not None:
        out["temperature"] = payload["temperature"]
    if payload.get("top_p") is not None:
        out["top_p"] = payload["top_p"]
    if payload.get("stream"):
        out["stream"] = True
    tools = _translate_tools(payload.get("tools"))
    if tools:
        out["tools"] = tools
    return out


def _join_system(instructions, system_text: str) -> str:
    """Merge top-level ``instructions`` with system/developer-role message text,
    in that order. Either part may be empty; returns "" when both are."""
    parts = [p for p in (str(instructions) if instructions else "", system_text) if p]
    return "\n\n".join(parts)


def _translate_input(inp):
    """Responses ``input`` -> ``(system_text, messages)``.

    System/developer-role message items are routed OUT of the messages array
    into ``system_text`` (Anthropic messages allow only user/assistant). Every
    message that stays is role-coerced so ONLY ``user``/``assistant`` survive —
    the invariant whose violation 400'd Codex's first live Responses call.
    """
    if isinstance(inp, str):
        return "", [{"role": "user", "content": inp}]
    if not isinstance(inp, list):
        return "", []
    system_chunks: List[str] = []
    messages: list = []
    for item in inp:
        translated = _translate_input_item(item)
        if translated is None:
            continue
        if translated.get("role") in _SYSTEM_ROLES:
            text = _content_to_text(translated.get("content"))
            if text:
                system_chunks.append(text)
        else:
            messages.append(_coerce_role(translated))
    return "\n\n".join(system_chunks), messages


def _coerce_role(message: dict) -> dict:
    """Force a message's role to one Anthropic accepts (user/assistant).

    ``function_call`` -> assistant and ``function_call_output`` -> user already
    comply; any other non-assistant role collapses to ``user`` so the content
    is preserved and Anthropic never sees an unexpected role.
    """
    if message.get("role") != _ASSISTANT_ROLE:
        return {**message, "role": "user"}
    return message


def _content_to_text(content) -> str:
    """Flatten message content to a plain string.

    Accepts a bare string, or a list of text-bearing blocks. ``text`` is the
    type produced by _translate_message_item (it already converts Responses
    ``input_text``/``output_text`` to ``text``); the input_text/output_text
    types are ALSO accepted here so the helper is correct even when called on
    a raw Responses content block, not only post-translation.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") in ("text", "input_text", "output_text")
        )
    return ""


def _translate_input_item(item) -> Optional[dict]:
    if not isinstance(item, dict):
        return None
    itype = item.get("type")
    if itype == "message":
        return _translate_message_item(item)
    if itype == "function_call":
        return _translate_function_call_item(item)
    if itype == "function_call_output":
        return _translate_function_call_output_item(item)
    return None


def _translate_message_item(item: dict) -> dict:
    role = item.get("role", "user")
    content = item.get("content", "")
    if isinstance(content, str):
        return {"role": role, "content": content}
    # content is a list of input_text / output_text / input_image parts.
    # translate_content_blocks converts image blocks to Anthropic `image`
    # (Bug 1) and input_text/output_text to `text`; unknown blocks are dropped.
    return {"role": role, "content": translate_content_blocks(content)}


def _translate_function_call_item(item: dict) -> dict:
    block = {
        "type": "tool_use",
        "id": item.get("call_id", ""),
        "name": item.get("name", ""),
        "input": _safe_json_obj(item.get("arguments")),
    }
    return {"role": "assistant", "content": [block]}


def _translate_function_call_output_item(item: dict) -> dict:
    # ``output`` may be a plain string OR a content array. When it is an array
    # it can carry an ``input_image`` block (Codex nests a pasted screenshot in
    # a tool result) — translate_content_blocks converts that to an Anthropic
    # ``image`` block so the resulting tool_result.content has no input_image
    # tag, which Anthropic would otherwise 400 on (Bug 1, the nested case).
    output = item.get("output", "")
    content = output if isinstance(output, str) else translate_content_blocks(output)
    block = {
        "type": "tool_result",
        "tool_use_id": item.get("call_id", ""),
        "content": content,
    }
    return {"role": "user", "content": [block]}


def _translate_tools(tools) -> list:
    """Responses FLAT tools -> Anthropic tools (no nested 'function' key)."""
    if not tools:
        return []
    return [t for t in (_translate_tool(t) for t in tools) if t is not None]


def _translate_tool(tool) -> Optional[dict]:
    if not isinstance(tool, dict) or tool.get("type") != "function":
        return None
    return {
        "name": tool.get("name", ""),
        "description": tool.get("description", ""),
        "input_schema": tool.get("parameters") or {"type": "object", "properties": {}},
    }


def _safe_json_obj(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


# ---------------------------------------------------------------------------
# Response: Anthropic Messages -> OpenAI Responses (non-streaming)
# ---------------------------------------------------------------------------

def anthropic_to_responses(data: dict, *, resp_id: str, model: str) -> dict:
    """Translate a non-streaming Anthropic response into Responses shape."""
    output = []
    item_id = "msg_" + secrets.token_hex(8)
    texts: List[str] = []
    for block in data.get("content", []):
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            texts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            output.append({
                "type": "function_call",
                "id": block.get("id", ""),
                "call_id": block.get("id", ""),
                "name": block.get("name", ""),
                "arguments": json.dumps(block.get("input", {})),
                "status": "completed",
            })
    if texts:
        output.insert(0, {
            "type": "message",
            "id": item_id,
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "".join(texts), "annotations": []}],
        })
    usage = data.get("usage", {})
    inp = usage.get("input_tokens", 0)
    out = usage.get("output_tokens", 0)
    return {
        "id": resp_id,
        "object": "response",
        "status": "completed",
        "model": data.get("model") or model,
        "output": output,
        "usage": {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out},
    }


# ---------------------------------------------------------------------------
# Streaming: Anthropic SSE -> Responses SSE
# ---------------------------------------------------------------------------

def translate_responses_stream(
    byte_chunks: Iterable[bytes], *, resp_id: str, model: str
) -> Iterator[bytes]:
    """Translate Anthropic SSE bytes into Responses SSE event bytes.

    ``response.completed`` goes out once Anthropic's ``message_stop`` arrives. A
    stream that fails instead -- Anthropic's own ``error`` event, a lost
    connection, or an end before ``message_stop`` -- ends with one in-band
    ``error`` event (:func:`responses_stream_error`) and no
    ``response.completed``, so it never passes for a finished answer (fleet
    finding f72); Codex 0.156.1 then shows one "Reconnecting..." notice and asks
    again (fleet finding f35; wire-matrix S7).
    """
    seq = _SeqGen()
    state = _StreamState(resp_id=resp_id, model=model)
    yield _sse({"type": "response.created", "sequence_number": seq(), "response": {
        "id": resp_id, "object": "response", "status": "in_progress",
        "model": model, "output": [], "usage": None,
    }})
    for event in guarded_sse_events(byte_chunks):
        etype = event.get("type")
        if etype == "error":
            yield responses_stream_error(upstream_error_text(event)[0], seq())
            return
        if etype == "message_stop":
            yield _sse({"type": "response.completed", "sequence_number": seq(), "response": {
                "id": resp_id, "object": "response", "status": "completed",
                "model": state.model_out or model, "output": state.completed_output(),
                "usage": state.usage,
            }})
            return
        yield from _handle_event(event, seq, state)
    yield responses_stream_error(CUT_SHORT_MESSAGE, seq())


def responses_stream_error(message: str, sequence_number: Optional[int] = None) -> bytes:
    """One in-band ``error`` event in the Responses stream dialect.

    Never ``response.failed``: Codex 0.156.1 ends the turn outright on some of
    its codes (``server_is_overloaded`` became "Selected model is at capacity"
    in the fleet's Codex lane), while an ``error`` event followed by a stream
    with no ``response.completed`` is answered with one reconnect (wire-matrix
    S7). ``sequence_number`` is left out where it is not known (a verbatim relay).
    """
    event = {"type": "error", "code": "server_error", "message": redact(message),
             "param": None}
    if sequence_number is not None:
        event["sequence_number"] = sequence_number
    return _sse(event)


class _SeqGen:
    def __init__(self):
        self._n = 0

    def __call__(self) -> int:
        n = self._n
        self._n += 1
        return n


class _StreamState:
    """Accumulates streaming state to build the final response.completed."""

    def __init__(self, *, resp_id: str, model: str):
        self.resp_id = resp_id
        self.model_out: str = model
        self.usage: Optional[dict] = None
        # current block
        self._block_type: Optional[str] = None  # "text" or "tool_use"
        self._block_index: int = 0
        self._item_id: str = ""
        self._item_name: str = ""
        self._text_buf: str = ""
        self._args_buf: str = ""
        # completed items (text items keyed by index, function_call items)
        self._text_items: dict = {}   # index -> {"item_id", "text"}
        self._tool_items: List[dict] = []

    def completed_output(self) -> list:
        out = []
        for idx in sorted(self._text_items):
            t = self._text_items[idx]
            out.append({
                "type": "message", "id": t["item_id"], "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": t["text"], "annotations": []}],
            })
        out.extend(self._tool_items)
        return out

    def on_text_block_start(self, index: int, item_id: str) -> None:
        self._block_type = "text"
        self._block_index = index
        self._item_id = item_id
        self._text_buf = ""

    def on_text_delta(self, text: str) -> None:
        self._text_buf += text

    def on_text_block_stop(self) -> None:
        self._text_items[self._block_index] = {
            "item_id": self._item_id, "text": self._text_buf}
        self._block_type = None

    def on_tool_block_start(self, index: int, item_id: str, name: str) -> None:
        self._block_type = "tool_use"
        self._block_index = index
        self._item_id = item_id
        self._item_name = name
        self._args_buf = ""

    def on_args_delta(self, partial: str) -> None:
        self._args_buf += partial

    def on_tool_block_stop(self) -> None:
        self._tool_items.append({
            "type": "function_call",
            "id": self._item_id, "call_id": self._item_id,
            "name": self._item_name, "arguments": self._args_buf,
            "status": "completed",
        })
        self._block_type = None


# Event-type -> handler name: explicit dispatch (Goodhart-anchored).
_EVENT_HANDLERS = {
    "message_start": "_on_message_start",
    "content_block_start": "_on_content_block_start",
    "content_block_delta": "_on_content_block_delta",
    "content_block_stop": "_on_content_block_stop",
    "message_delta": "_on_message_delta",
}


def _handle_event(
    event: dict, seq: _SeqGen, state: _StreamState
) -> Iterator[bytes]:
    etype = event.get("type", "")
    handler_name = _EVENT_HANDLERS.get(etype)
    if handler_name is None:
        return
    yield from globals()[handler_name](event, seq, state)


def _on_message_start(event: dict, seq: _SeqGen, state: _StreamState) -> Iterator[bytes]:
    msg = event.get("message", {})
    state.model_out = msg.get("model", state.model_out)
    inp = (msg.get("usage") or {}).get("input_tokens")
    if inp is not None:
        state.usage = {"input_tokens": inp, "output_tokens": 0, "total_tokens": inp}
    return iter([])


def _on_content_block_start(event: dict, seq: _SeqGen, state: _StreamState) -> Iterator[bytes]:
    index = event.get("index", 0)
    cb = event.get("content_block", {})
    btype = cb.get("type")
    item_id = cb.get("id") or ("item_" + secrets.token_hex(6))
    if btype == "text":
        state.on_text_block_start(index, item_id)
        yield _sse({"type": "response.output_item.added", "sequence_number": seq(),
                    "output_index": index,
                    "item": {"type": "message", "id": item_id, "role": "assistant",
                             "status": "in_progress", "content": []}})
        yield _sse({"type": "response.content_part.added", "sequence_number": seq(),
                    "item_id": item_id, "output_index": index, "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []}})
    elif btype == "tool_use":
        name = cb.get("name", "")
        state.on_tool_block_start(index, item_id, name)
        yield _sse({"type": "response.output_item.added", "sequence_number": seq(),
                    "output_index": index,
                    "item": {"type": "function_call", "id": item_id, "call_id": item_id,
                             "name": name, "status": "in_progress"}})


def _on_content_block_delta(event: dict, seq: _SeqGen, state: _StreamState) -> Iterator[bytes]:
    delta = event.get("delta", {})
    index = event.get("index", 0)
    dtype = delta.get("type")
    if dtype == "text_delta":
        text = delta.get("text", "")
        state.on_text_delta(text)
        yield _sse({"type": "response.output_text.delta", "sequence_number": seq(),
                    "item_id": state._item_id, "output_index": index, "content_index": 0,
                    "delta": text})
    elif dtype == "input_json_delta":
        partial = delta.get("partial_json", "")
        state.on_args_delta(partial)
        yield _sse({"type": "response.function_call_arguments.delta", "sequence_number": seq(),
                    "item_id": state._item_id, "output_index": index, "delta": partial})


def _on_content_block_stop(event: dict, seq: _SeqGen, state: _StreamState) -> Iterator[bytes]:
    index = event.get("index", 0)
    if state._block_type == "text":
        full_text = state._text_buf
        item_id = state._item_id
        state.on_text_block_stop()
        yield _sse({"type": "response.output_text.done", "sequence_number": seq(),
                    "item_id": item_id, "output_index": index, "content_index": 0,
                    "text": full_text})
        yield _sse({"type": "response.content_part.done", "sequence_number": seq(),
                    "item_id": item_id, "output_index": index, "content_index": 0,
                    "part": {"type": "output_text", "text": full_text, "annotations": []}})
        yield _sse({"type": "response.output_item.done", "sequence_number": seq(),
                    "output_index": index,
                    "item": {"type": "message", "id": item_id, "role": "assistant",
                             "status": "completed",
                             "content": [{"type": "output_text", "text": full_text,
                                          "annotations": []}]}})
    elif state._block_type == "tool_use":
        item_id = state._item_id
        full_args = state._args_buf
        name = state._item_name
        state.on_tool_block_stop()
        yield _sse({"type": "response.function_call_arguments.done", "sequence_number": seq(),
                    "item_id": item_id, "output_index": index, "arguments": full_args})
        yield _sse({"type": "response.output_item.done", "sequence_number": seq(),
                    "output_index": index,
                    "item": {"type": "function_call", "id": item_id, "call_id": item_id,
                             "name": name, "arguments": full_args, "status": "completed"}})


def _on_message_delta(event: dict, seq: _SeqGen, state: _StreamState) -> Iterator[bytes]:
    out_tokens = (event.get("usage") or {}).get("output_tokens")
    if out_tokens is not None:
        inp = (state.usage or {}).get("input_tokens", 0)
        state.usage = {"input_tokens": inp, "output_tokens": out_tokens,
                       "total_tokens": inp + out_tokens}
    return iter([])


def _sse(obj: dict) -> bytes:
    event_type = obj.get("type", "")
    return (
        f"event: {event_type}\n"
        f"data: {json.dumps(obj)}\n\n"
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# HTTP plumbing helpers
# ---------------------------------------------------------------------------

def _resp_id() -> str:
    return "resp_" + secrets.token_hex(12)


def _send_json(handler, code: int, obj: dict,
               headers: Iterable[Tuple[str, str]] = ()) -> None:
    payload = json.dumps(obj).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    for name, value in headers:
        handler.send_header(name, value)
    handler.send_header("Connection", "close")
    handler.end_headers()
    try:
        handler.wfile.write(payload)
    except (BrokenPipeError, ConnectionResetError):
        pass


def _send_error(handler, code: int, message: str,
                headers: Iterable[Tuple[str, str]] = ()) -> None:
    _send_json(handler, code, {"error": {"message": message, "type": "gateway_error"}},
               headers)


def _relay_upstream_error(handler, resp) -> None:
    """Wrap an upstream error as a Responses-shaped error, keeping its
    ``Retry-After`` (fleet finding f73). A 429 goes out as a 503: one 429 ends a
    Codex turn, a 503 is retried (:func:`lib.hal.gateway.openai_wire_status`,
    fleet finding f26)."""
    message = "upstream error"
    status = openai_wire_status(resp.status)
    try:
        data = json.loads(resp.read_all())
        message = (data.get("error") or {}).get("message") or message
    except Exception:
        pass
    finally:
        resp.close()
    _send_error(handler, status, redact(message), retry_headers(resp.headers))


def _relay_stream(handler, resp, *, resp_id: str, model: str) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "close")
    handler.end_headers()
    stream = translate_responses_stream(resp.iter_chunks(), resp_id=resp_id, model=model)
    try:
        for chunk in stream:
            try:
                handler.wfile.write(chunk)
                handler.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break
    finally:
        resp.close()


def _relay_nonstream(handler, resp, *, resp_id: str, model: str) -> None:
    try:
        data = json.loads(resp.read_all())
        out = anthropic_to_responses(data, resp_id=resp_id, model=model)
    except Exception as exc:  # nothing sent yet: a lost upstream is a 503
        _send_error(handler, *failure_reply(exc, other_label="upstream parse error"))
        return
    finally:
        resp.close()
    _send_json(handler, 200, out)


# ---------------------------------------------------------------------------
# Handler (wired by server.py + make_gateway_handler in gateway.py)
# ---------------------------------------------------------------------------

def handle_responses(handler: BaseHTTPRequestHandler, proxy: GatewayProxy) -> None:
    """End-to-end: translate Responses request -> Anthropic -> translate back.
    Fail-open: any error becomes a clean Responses-shaped error envelope. A
    transport fault talking upstream is a 503, which Codex retries, never the
    400 it gives up on (fleet finding f48). A request the spend guard refuses
    is answered without reaching any upstream.

    Exception (the fast 80%): a Responses-NATIVE provider (OpenAI) reached under
    multiprovider routing is forwarded the ORIGINAL Responses body VERBATIM — it
    speaks /v1/responses itself, so the Anthropic pivot would only inject bugs
    (the resolve_model alias; max_tokens vs max_completion_tokens for gpt-5.x)."""
    raw = read_request_body(handler)
    rid = _resp_id()
    try:
        payload, raw = _canonical_payload(raw)
        resp = _forward_responses(handler, proxy, payload, raw)
    except SpendRefused as exc:
        from lib.hal.gateway_cascade import serve_cascade
        if serve_cascade(handler, payload, "/v1/responses", exc):
            return
        if exc.decision.status == 429:
            _send_spent_notice(handler, exc.decision, payload)
        else:
            send_openai_spend_refusal(handler, exc.decision)
        return
    except Exception as exc:  # fail-open: never crash the request
        _send_error(handler, *failure_reply(
            exc, other_status=400, other_label="gateway translation error"))
        return
    if resp is None:
        return  # a Responses-native provider's answer, relayed as it came
    model = payload.get("model", "")
    if resp.status >= 400:
        _relay_upstream_error(handler, resp)
    elif payload.get("stream"):
        _relay_stream(handler, resp, resp_id=rid, model=model)
    else:
        _relay_nonstream(handler, resp, resp_id=rid, model=model)


def _canonical_payload(raw: bytes):
    """The request parsed, with a /model-picker alias resolved: ``(payload, raw)``.

    THE THIRD INGRESS RAIL. An alias picked from Claude Code's /model picker
    reaches here too: ``declared_model_entries`` appends the alias rows and
    ``_inject_models_list`` mirrors every one into the Codex-compat ``models``
    array, and Codex posts to /v1/responses. Resolve to the id that routes
    BEFORE ``same_wire_provider`` reads it, and re-serialise, because the
    Responses-native branch forwards ``raw`` VERBATIM. Without this
    ``same_wire_provider`` returns None for every alias and the request falls
    through to the verbatim Anthropic relay -- an OpenAI-intended request served
    by the wrong vendor, with no error to say so. Identity for every non-alias
    request, which stays byte for byte as it was. Guarded by ``is_alias`` so a
    body naming NO model is left alone (#575).
    """
    payload = json.loads(raw or b"{}")
    from lib.hal.gateway_picker_alias import canonical_for, is_alias
    if is_alias(payload.get("model", "")):
        payload["model"] = canonical_for(payload["model"])
        raw = json.dumps(payload).encode("utf-8")
    return payload, raw


def _forward_responses(handler, proxy, payload: dict, raw: bytes):
    """Send the request where its model routes; the answer still to translate,
    or None once a Responses-native provider's answer has been relayed here."""
    model = payload.get("model", "")
    from lib.hal.gateway_egress import (
        cross_wire_forward, forward_same_wire, same_wire_provider)
    provider = same_wire_provider(model)
    if provider in _RESPONSES_NATIVE_PROVIDERS:
        # OpenAI speaks the Responses API natively → forward the ORIGINAL body
        # verbatim (model id + every gpt-5.x param preserved). No Anthropic
        # pivot, so neither the resolve_model alias nor the max_tokens ↔
        # max_completion_tokens translation can corrupt the request. Reuses the
        # same forward_same_wire + verbatim relay_response the chat path uses;
        # a 429 goes out as a 503 (fleet finding f26).
        answer = openai_wire_response(forward_same_wire(provider, RESPONSES_PATH, raw))
        relay_response(handler, answer, stream_error=responses_stream_error)
        return None
    # Responses -> Anthropic Messages is the pivot shape for the remaining
    # paths (a chat-only cross-wire provider, or the verbatim-Anthropic relay).
    anthropic_req = responses_to_anthropic_request(payload)
    if provider is not None:
        # A chat-only provider (gemini/groq/…) reached via the Responses wire:
        # it speaks Chat Completions, not Responses, so we MUST pivot. Restore
        # the ORIGINAL model id — responses_to_anthropic_request ran resolve_model,
        # which aliases e.g. gpt-5.5 -> claude-opus-4-8 (correct only for the
        # verbatim-Anthropic fallback below, never when forwarding to a real
        # provider, else the aliased claude-* id 400s at the provider).
        return cross_wire_forward(dict(anthropic_req, model=model), provider=provider,
                                  want_stream=bool(payload.get("stream")))
    body = json.dumps(anthropic_req).encode("utf-8")
    up_headers = anthropic_upstream_headers(list(handler.headers.items()))
    return proxy.forward(_MESSAGES_PATH, body, up_headers, method="POST")


def _send_spent_notice(handler, decision, request: dict) -> None:
    """Step 5g: Codex gets one final reset message; chat retains native 429.

    Reuse the Responses translators and the final-message shape witnessed in
    tools/wire-matrix/fixtures/codex/S0. This is a local notice, with zero usage.
    """
    model = request.get("model", "")
    message = {"id": "msg_spend_refused", "type": "message", "role": "assistant",
               "model": model, "content": [{"type": "text", "text": redact(decision.reason)}],
               "stop_reason": "end_turn", "stop_sequence": None,
               "usage": {"input_tokens": 0, "output_tokens": 0}}
    rid = _resp_id()
    if not request.get("stream"):
        _send_json(handler, 200, anthropic_to_responses(message, resp_id=rid, model=model))
        return
    chunks = anthropic_message_to_sse_chunks(message)
    body = b"".join(translate_responses_stream(chunks, resp_id=rid, model=model))
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Connection", "close")
    handler.end_headers()
    try:
        handler.wfile.write(body)
    except (BrokenPipeError, ConnectionResetError):
        pass
