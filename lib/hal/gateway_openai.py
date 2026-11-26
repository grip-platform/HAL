"""HAL Gateway — OpenAI-compatible ``/v1/chat/completions`` endpoint (P2).

Lets any OpenAI-wire coding harness (Aider, Cline, OpenAI Codex CLI, Goose, ...)
point ``OPENAI_API_BASE`` at the gateway and reach Anthropic through it. The
endpoint translates an OpenAI ChatCompletion request -> Anthropic Messages
request, forwards it via the P1 :class:`~lib.hal.gateway.GatewayProxy` (verbatim
HTTP + byte-identical retry), then translates the Anthropic response back into
OpenAI ChatCompletion shape — both streaming (SSE) and non-streaming. Closes
plan phase P2 of the HAL Gateway roadmap.

Design (syscall-doctrine + llm-augmentation monotone-toward-safe, mirrors P1):

* **Pure dict<->dict translators.** No coupling to HAL ``ChatMessage`` /
  ``ChatResponse`` types — the gateway speaks raw wire-format JSON. Every
  translator is a pure function, fully testable without a socket.
* **Auth translation, never leaked.** An OpenAI client sends
  ``Authorization: Bearer <key>``; Anthropic needs ``x-api-key`` +
  ``anthropic-version``. The key is translated onto the upstream call and never
  logged or echoed (reuses P1's ``redact``).
* **Fail-open.** A translation/parse error returns a clean OpenAI-shaped error
  envelope (the harness sees a normal error); the gateway never crashes a
  request and never becomes a new outage point.
* **Explicit dispatch tables.** ``stop_reason`` -> ``finish_reason`` and usage
  field mappings are explicit dicts so a wrong mapping fails a Goodhart test
  rather than silently mis-reporting (the H-GW-2 mutation check).

Scope: P2 is the OpenAI-compat translation only. Registry failover / cost /
audit layering is P3; Fern SDK (P4) and folded kernel (P5) are gated later.
"""
from __future__ import annotations

import codecs
import json
import secrets
import time
from http.server import BaseHTTPRequestHandler
from typing import Iterable, Iterator, List, Optional, Tuple

from lib.hal.gateway import (
    GatewayProxy,
    failure_reply,
    lost_upstream_message,
    openai_wire_response,
    openai_wire_status,
    read_request_body,
    redact,
    relay_response,
    retry_headers,
)
from lib.hal.gateway_translate import (
    resolve_model,
    translate_content_blocks,
    translate_image_block,
)
from lib.hal.spend_guard import SpendRefused

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
_MESSAGES_PATH = "/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 4096

# Anthropic stop_reason -> OpenAI finish_reason. Explicit so a wrong entry fails
# a test (Goodhart anchor) instead of silently mis-reporting completion state.
_FINISH_REASON = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}
_DEFAULT_FINISH_REASON = "stop"


# --- request translation: OpenAI ChatCompletion -> Anthropic Messages --------

def openai_to_anthropic_request(
    payload: dict, *, default_max_tokens: int = _DEFAULT_MAX_TOKENS
) -> dict:
    """Translate an OpenAI ChatCompletion request body into Anthropic Messages.

    System messages are hoisted to the top-level ``system`` field (Anthropic
    has no system role); ``max_tokens`` is defaulted (Anthropic requires it);
    sampling params, stop sequences, streaming, and tools are mapped across.
    """
    system, convo = _split_system(payload.get("messages", []))
    out: dict = {
        "model": resolve_model(payload.get("model", "")),
        "messages": [m for m in map(_translate_message, convo) if m is not None],
        "max_tokens": int(payload.get("max_tokens") or default_max_tokens),
    }
    if system:
        out["system"] = system
    if payload.get("temperature") is not None:
        out["temperature"] = payload["temperature"]
    if payload.get("top_p") is not None:
        out["top_p"] = payload["top_p"]
    stop = payload.get("stop")
    if stop:
        out["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    if payload.get("stream"):
        out["stream"] = True
    tools = _translate_tools(payload.get("tools"))
    if tools:
        out["tools"] = tools
        choice = _translate_tool_choice(payload.get("tool_choice"))
        if choice is not None:
            out["tool_choice"] = choice
    return out


def _split_system(messages: list) -> Tuple[str, list]:
    """Return (joined system text, non-system messages in order)."""
    sys_parts: List[str] = []
    convo: list = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            sys_parts.append(_text_of(msg.get("content")))
        else:
            convo.append(msg)
    return "\n\n".join(p for p in sys_parts if p), convo


def _translate_message(msg) -> Optional[dict]:
    """Translate one OpenAI message into an Anthropic message (or None)."""
    if not isinstance(msg, dict):
        return None
    role = msg.get("role")
    if role == "tool":
        # A tool message's content may be a plain string OR a content array
        # carrying an image (Codex nests a pasted screenshot here as
        # image_url/input_image). translate_content_blocks converts those to
        # Anthropic `image` blocks so the tool_result.content has no
        # image_url/input_image tag for Anthropic to 400 on (Bug 1).
        return {"role": "user", "content": [{
            "type": "tool_result",
            "tool_use_id": msg.get("tool_call_id", ""),
            "content": translate_content_blocks(msg.get("content")),
        }]}
    if role == "assistant" and msg.get("tool_calls"):
        return {"role": "assistant", "content": _assistant_tool_use(msg)}
    anthropic_role = "assistant" if role == "assistant" else "user"
    return {"role": anthropic_role, "content": _translate_content(msg.get("content"))}


def _assistant_tool_use(msg: dict) -> list:
    """Assistant message with tool_calls -> Anthropic text + tool_use blocks."""
    blocks: list = []
    text = _text_of(msg.get("content"))
    if text:
        blocks.append({"type": "text", "text": text})
    for call in msg.get("tool_calls") or []:
        fn = call.get("function", {}) if isinstance(call, dict) else {}
        blocks.append({
            "type": "tool_use",
            "id": call.get("id", ""),
            "name": fn.get("name", ""),
            "input": _safe_json_obj(fn.get("arguments")),
        })
    return blocks


def _translate_content(content):
    """OpenAI message content (str or part-list) -> Anthropic content."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return [b for b in map(_translate_part, content) if b is not None]


def _translate_part(part) -> Optional[dict]:
    """Translate one OpenAI content part into an Anthropic content block.

    ``text`` -> Anthropic ``text``; ``image_url``/``input_image`` ->
    Anthropic ``image`` (base64 data-URI or url source) via the shared
    :func:`translate_image_block`. Unknown parts return ``None`` (dropped).
    """
    if not isinstance(part, dict):
        return None
    ptype = part.get("type")
    if ptype == "text":
        return {"type": "text", "text": part.get("text", "")}
    if ptype in ("image_url", "input_image"):
        return translate_image_block(part)
    return None


def _translate_tools(tools) -> list:
    """OpenAI tools (function schema) -> Anthropic tools."""
    if not tools:
        return []
    return [t for t in map(_translate_tool, tools) if t is not None]


def _translate_tool(tool) -> Optional[dict]:
    if not isinstance(tool, dict) or tool.get("type") != "function":
        return None
    fn = tool.get("function", {})
    return {
        "name": fn.get("name", ""),
        "description": fn.get("description", ""),
        "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
    }


def _translate_tool_choice(choice) -> Optional[dict]:
    """OpenAI tool_choice -> Anthropic tool_choice (None = Anthropic default)."""
    if choice in (None, "auto"):
        return None
    if choice == "required":
        return {"type": "any"}
    if isinstance(choice, dict) and choice.get("type") == "function":
        return {"type": "tool", "name": (choice.get("function") or {}).get("name", "")}
    return None


def _text_of(content) -> str:
    """Best-effort plain text of OpenAI message content (str or part-list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


def _safe_json_obj(raw) -> dict:
    """Parse a tool-call arguments JSON string into a dict; {} on failure."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


# --- response translation: Anthropic Messages -> OpenAI ChatCompletion -------

def anthropic_to_openai_response(
    data: dict, *, model: str, completion_id: str, created: int
) -> dict:
    """Translate a non-streaming Anthropic response into OpenAI ChatCompletion."""
    text, tool_calls = _split_content(data.get("content", []))
    message: dict = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    usage = data.get("usage", {})
    inp = usage.get("input_tokens", 0)
    out = usage.get("output_tokens", 0)
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": data.get("model") or model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": _finish_reason(data.get("stop_reason")),
        }],
        "usage": {
            "prompt_tokens": inp,
            "completion_tokens": out,
            "total_tokens": inp + out,
        },
    }


def _split_content(blocks) -> Tuple[str, list]:
    """Anthropic content blocks -> (joined text, OpenAI tool_calls list)."""
    texts: List[str] = []
    tool_calls: list = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            texts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })
    return "".join(texts), tool_calls


def _finish_reason(stop_reason: Optional[str]) -> str:
    return _FINISH_REASON.get(stop_reason or "", _DEFAULT_FINISH_REASON)


# --- streaming translation: Anthropic SSE -> OpenAI chat.completion.chunk ----

def translate_sse_stream(
    byte_chunks: Iterable[bytes], *, model: str, completion_id: str, created: int
) -> Iterator[bytes]:
    """Translate an Anthropic SSE byte stream into OpenAI chunk SSE bytes.

    Emits an opening role chunk, text and function-call argument deltas, a final
    chunk carrying the mapped ``finish_reason`` once ``message_stop`` arrives,
    then the ``[DONE]`` sentinel.

    A stream that fails instead -- Anthropic's own ``error`` event, a lost
    connection, or an end before ``message_stop`` -- ends with one in-band
    error (:func:`chat_stream_error`) and no finish chunk or ``[DONE]``, so the
    client sees an interruption, never a short answer passed off as a whole one
    (fleet finding f72).
    """
    yield _sse_chunk(completion_id, created, model, {"role": "assistant"}, None)
    finish: Optional[str] = None
    tool_indices: dict[int, int] = {}
    for event in guarded_sse_events(byte_chunks):
        etype = event.get("type")
        if etype == "error":
            yield chat_stream_error(*upstream_error_text(event))
            return
        if etype == "message_stop":
            yield _sse_chunk(completion_id, created, model, {},
                             finish or _DEFAULT_FINISH_REASON)
            yield b"data: [DONE]\n\n"
            return
        if etype == "content_block_start":
            block = event.get("content_block", {})
            if block.get("type") == "tool_use":
                index = tool_indices.setdefault(event.get("index", 0), len(tool_indices))
                arguments = json.dumps(block["input"]) if block.get("input") else ""
                yield _sse_chunk(completion_id, created, model, {"tool_calls": [{
                    "index": index, "id": block.get("id", ""), "type": "function",
                    "function": {"name": block.get("name", ""), "arguments": arguments},
                }]}, None)
        elif etype == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                yield _sse_chunk(completion_id, created, model,
                                 {"content": delta.get("text", "")}, None)
            elif delta.get("type") == "input_json_delta" and event.get("index") in tool_indices:
                yield _sse_chunk(completion_id, created, model, {"tool_calls": [{
                    "index": tool_indices[event["index"]],
                    "function": {"arguments": delta.get("partial_json", "")},
                }]}, None)
        elif etype == "message_delta":
            stop = (event.get("delta") or {}).get("stop_reason")
            if stop:
                finish = _finish_reason(stop)
    yield chat_stream_error(CUT_SHORT_MESSAGE)


#: What a client is told when an upstream stream ended before its last event.
#: Worded so the harnesses read it as retryable ("retry your request").
CUT_SHORT_MESSAGE = ("HAL gateway: the upstream stream ended before the answer "
                     "was complete; retry your request")


def guarded_sse_events(byte_chunks: Iterable[bytes]) -> Iterator[dict]:
    """``_iter_sse_events``, except that a failure to read the upstream ends the
    stream with one ``{"type": "error"}`` event saying so, rather than raising
    into an answer that has already started (fleet finding f72).

    The synthetic error event always carries an ``error`` dict with ``type`` and
    ``message`` keys -- exactly the shape :func:`upstream_error_text` reads -- so
    a caller may hand it straight to ``upstream_error_text`` without a KeyError.
    Events relayed from the upstream pass through as parsed and are not
    guaranteed to carry that shape; ``upstream_error_text`` fills their defaults.
    """
    try:
        yield from _iter_sse_events(byte_chunks)
    except Exception as exc:  # the upstream failed mid-stream
        yield {"type": "error",
               "error": {"type": "api_error", "message": lost_upstream_message(exc)}}


def upstream_error_text(event: dict) -> Tuple[str, str]:
    """(message, error type) of an Anthropic ``error`` event, defaults filled in."""
    error = event.get("error") if isinstance(event.get("error"), dict) else {}
    message = str(error.get("message") or "the upstream reported an error mid-stream")
    return message, str(error.get("type") or "api_error")


def chat_stream_error(message: str, error_type: str = "server_error") -> bytes:
    """One in-band error in the chat-completions stream dialect: a ``data`` line
    holding an ``error`` object. The openai SDK (Hermes) raises it and OpenCode
    retries on it (wire-matrix S7)."""
    body = json.dumps({"error": {"message": redact(message), "type": error_type,
                                 "code": None, "param": None}})
    return f"event: error\ndata: {body}\n\n".encode("utf-8")


def _iter_sse_events(byte_chunks: Iterable[bytes]) -> Iterator[dict]:
    """Parse arbitrary upstream byte chunks into a stream of SSE data objects.

    Decoded incrementally: a chunk may end inside a multi-byte character (the
    relay passes bytes on as they arrive), and decoding each chunk alone would
    turn that character into U+FFFD."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buffer = ""
    for chunk in byte_chunks:
        buffer += decoder.decode(chunk) if isinstance(chunk, bytes) else chunk
        events, buffer = _drain_events(buffer)
        yield from events
    tail, _ = _drain_events(buffer + decoder.decode(b"", final=True) + "\n\n")
    yield from tail


def _drain_events(buffer: str) -> Tuple[List[dict], str]:
    """Split a buffer into complete SSE blocks; return (parsed, remainder)."""
    parts = buffer.split("\n\n")
    remainder = parts.pop() if parts else ""
    parsed = [e for e in map(_parse_sse_block, parts) if e is not None]
    return parsed, remainder


def _parse_sse_block(block: str) -> Optional[dict]:
    """Extract and JSON-parse the ``data:`` payload of one SSE block."""
    for line in block.split("\n"):
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload and payload != "[DONE]":
                try:
                    return json.loads(payload)
                except json.JSONDecodeError:
                    return None
    return None


def _sse_chunk(
    completion_id: str, created: int, model: str, delta: dict, finish: Optional[str]
) -> bytes:
    """Build one OpenAI ``chat.completion.chunk`` SSE event as bytes."""
    obj = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return b"data: " + json.dumps(obj).encode("utf-8") + b"\n\n"


# --- auth + HTTP plumbing ----------------------------------------------------

def anthropic_upstream_headers(
    client_headers: Iterable[Tuple[str, str]], *, api_version: str = _ANTHROPIC_VERSION
) -> List[Tuple[str, str]]:
    """Build Anthropic upstream headers from OpenAI-client headers.

    Translates ``Authorization: Bearer <key>`` -> ``x-api-key`` (Anthropic's
    scheme) and adds ``anthropic-version``. An explicit ``x-api-key`` wins.
    The key only ever travels upstream — it is never logged or returned.

    EXCEPT for an OAuth subscription token, which must stay a Bearer. Anthropic's
    OAuth scheme (``sk-ant-oat...``) authenticates via ``Authorization: Bearer``;
    rewriting it into ``x-api-key`` produces a request Anthropic rejects, because
    the two schemes are not interchangeable. ``gateway.py:251`` has always guarded
    this on the native ``/v1/messages`` rail; this rail did not, so a subscription
    token could not ride ``/v1/chat/completions`` or ``/v1/responses`` at all —
    and every non-claude harness speaks this rail. The practical effect was that
    those harnesses could only reach Anthropic through an API key, never through
    the operator's subscription. Found 2026-08-31 by a converge-fleet mapping pool
    reachability across all 24 GRIP harnesses; the guard here is a direct port of
    the one in gateway.py so the two rails cannot disagree about the same token.
    """
    headers = {name.lower(): value for name, value in client_headers}
    key = headers.get("x-api-key") or _bearer_token(headers.get("authorization", ""))
    out: List[Tuple[str, str]] = [
        ("content-type", "application/json"),
        ("anthropic-version", headers.get("anthropic-version", api_version)),
    ]
    if key.startswith("sk-ant-oat"):
        # Preserve the caller's OAuth Bearer verbatim; do NOT emit x-api-key.
        out.append(("authorization", f"Bearer {key}"))
        if headers.get("anthropic-beta"):
            out.append(("anthropic-beta", headers["anthropic-beta"]))
        return out
    if key:
        out.append(("x-api-key", key))
    if headers.get("anthropic-beta"):
        out.append(("anthropic-beta", headers["anthropic-beta"]))
    return out


def _bearer_token(auth: str) -> str:
    return auth[7:].strip() if auth.lower().startswith("bearer ") else ""


def _completion_id() -> str:
    return "chatcmpl-" + secrets.token_hex(12)


def _now() -> int:
    return int(time.time())


# --- request handler (wired by server.py + the standalone gateway) -----------

def handle_chat_completions(handler: BaseHTTPRequestHandler, proxy: GatewayProxy) -> None:
    """End-to-end: translate OpenAI request -> Anthropic -> forward -> translate
    back. Fail-open: any translation/transport error becomes a clean
    OpenAI-shaped error envelope, never an unhandled crash.

    WITNESS+SIGN (the moat): every chat request — on BOTH the cross-wire and the
    verbatim path — emits ONE signed audit row capturing the SERVED provider +
    resolved model id (:func:`gateway_witness.witness_chat`). This is the rail
    the failover-layer audit missed entirely (cross-wire) or mislabelled as
    ``/v1/messages`` (verbatim). The witness is decoupled from ``proxy.forward``
    and fully fail-open — it can never break a request.
    """
    started = time.monotonic()
    raw = read_request_body(handler)
    try:
        payload = json.loads(raw or b"{}")
    except Exception as exc:  # noqa: BLE001 — fail-open on a malformed body
        _send_openai_error(handler, 400, f"gateway parse error: {redact(str(exc))}")
        return
    # A model picked from Claude Code's /model picker can arrive here too: the
    # alias rows sit in the same /v1/models body an OpenAI-wire client reads.
    # Resolve to the id that routes, and re-serialise, because `raw` is
    # forwarded VERBATIM on the same-wire relay. Identity for every non-alias
    # request, which stays byte for byte as it was.
    from lib.hal.gateway_picker_alias import canonical_for
    _canonical = canonical_for(payload.get("model", ""))
    if _canonical != payload.get("model"):
        payload["model"] = _canonical
        raw = json.dumps(payload).encode("utf-8")
    model = payload.get("model", "")
    # C' council route (ADR 0012): 'council' proxies to the VPS sovereign council
    # with fail-open. Early return before the normal path — the 40+ existing
    # families never enter this branch.
    if model in _COUNCIL_MODEL_IDS:
        _relay_council(handler, payload, model, started)
        return
    _dispatch_openai_chat(handler, proxy, raw, payload, model, started)


def _dispatch_openai_chat(handler, proxy, raw, payload: dict, model: str, started: float) -> None:
    """The normal chat path: same-wire re-point OR translate OpenAI->Anthropic->
    forward, then witness (the moat: ONE signed audit row with the SERVED provider
    + resolved model) and relay. Fail-open: any translation/transport error becomes
    a clean OpenAI-shaped error, never an unhandled crash: a transport fault is a
    503, which OpenCode and Hermes retry, never the 400 they give up on (fleet
    finding f68)."""
    completion_id = _completion_id()
    created = _now()
    try:
        from lib.hal.gateway_egress import same_wire_provider
        _sw_provider = same_wire_provider(model)
        if _sw_provider is not None:
            _relay_same_wire(handler, raw, _sw_provider, model, started)
            return
        anthropic_req = openai_to_anthropic_request(payload)
        stream = bool(payload.get("stream"))
        body = json.dumps(anthropic_req).encode("utf-8")
        up_headers = anthropic_upstream_headers(list(handler.headers.items()))
        resp = proxy.forward(_MESSAGES_PATH, body, up_headers, method="POST")
    except SpendRefused as exc:
        from lib.hal.gateway_cascade import serve_cascade
        if serve_cascade(handler, payload, "/v1/chat/completions", exc):
            return
        send_openai_spend_refusal(handler, exc.decision)
        return
    except Exception as exc:  # fail-open: never crash the request
        _send_openai_error(handler, *failure_reply(
            exc, other_status=400, other_label="gateway translation error"))
        return
    _witness_chat(_ANTHROPIC_PROVIDER, resolve_model(model), resp.status, started=started)
    if resp.status >= 400:
        _relay_upstream_error(handler, resp)
    elif stream:
        _relay_stream(handler, resp, model=model, completion_id=completion_id, created=created)
    else:
        _relay_nonstream(handler, resp, model=model, completion_id=completion_id, created=created)


# The served-provider label for the verbatim Anthropic chat path.
_ANTHROPIC_PROVIDER = "anthropic"

# --- C' council route (ADR 0012): opencode -> :4010 -> VPS sovereign council ---
_COUNCIL_MODEL_IDS = ("council", "sovereign-council")


def _relay_council(handler, payload: dict, model: str, started: float) -> None:
    """C' route: proxy the 'council' model DOWNSTREAM to the VPS sovereign-council
    endpoint over Tailscale, passing the sha256/IDR seal headers through. FAIL-OPEN
    to a single Pool-B model if the VPS is unreachable/slow, so the opencode default
    never dies on a VPS outage (ADR 0012 — both councils' #1 objection). Every OTHER
    model id is untouched; this is an early return before the normal routing, so the
    40+ existing families never enter this path.

    The VPS endpoint is NON-STREAMING (a synchronous fan-out + judge returns one
    chat.completion). The OpenAI-compat client (OpenCode's @ai-sdk/openai-compatible)
    streams by default, so a stream:true request is answered by SYNTHESISING a
    text/event-stream from that single JSON completion — never by leaking the VPS's
    batch JSON into a stream parser (fleet finding: "OpenAI Chat stream ended without
    finish_reason"). A non-streaming request relays the JSON verbatim. A VPS 4xx/5xx
    is a real reply and uses the OpenAI status mapping (never fail-open, which would
    mask a legit error with a fallback model's answer); only a transport failure
    (unreachable / timeout) fail-opens."""
    import os
    import urllib.request
    import urllib.error
    vps_url = os.environ.get(
        "GRIP_COUNCIL_VPS_URL", "http://100.64.0.10:8791/v1/chat/completions")
    timeout = float(os.environ.get("GRIP_COUNCIL_TIMEOUT_S", "45"))
    stream = bool(payload.get("stream"))
    # The VPS only ever serves one non-streaming completion; ask for it explicitly so
    # its reply is exactly one JSON object to wrap (stream) or relay (non-stream).
    # stream_options is valid only beside stream:true, so it stays with the client.
    batch = {k: v for k, v in payload.items() if k != "stream_options"}
    data = json.dumps(dict(batch, stream=False)).encode("utf-8")
    try:
        req = urllib.request.Request(
            vps_url, data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as up:
            body = up.read()
            status = up.status
            headers = list(up.headers.items())
            sha = up.headers.get("X-Grip-Council-Sha", "")
            idr = up.headers.get("X-Grip-Idr-Leaf", "")
    except urllib.error.HTTPError as up:
        # A real VPS reply at 4xx/5xx (bad model id, empty prompt, council failure):
        # Relay its body with the OpenAI status mapping, without falling back.
        body = _read_error_body(up)
        status = up.code
        headers = list(up.headers.items())
        sha = up.headers.get("X-Grip-Council-Sha", "")
        idr = up.headers.get("X-Grip-Idr-Leaf", "")
    except Exception as exc:  # noqa: BLE001 — VPS down/slow -> FAIL OPEN, never a dead default
        _relay_council_failopen(handler, payload, model, started, str(exc))
        return
    if stream and status == 200:
        options = payload.get("stream_options")
        _relay_council_stream(handler, body, model, sha, idr,
                              include_usage=isinstance(options, dict) and bool(options.get("include_usage")))
    else:
        _relay_council_json(handler, body, status, model, sha, idr, headers)
    _witness_chat("grip-vps-council", model, status, started=started)


def _read_error_body(error) -> bytes:
    """The body of a VPS error reply. The status is already known, so a body that
    times out or resets while being read still reaches the client as that status."""
    try:
        return error.read()
    except Exception:  # noqa: BLE001 — the reply was real; only its body was lost
        return json.dumps({"error": {"message": "council (VPS) error body could not be read"}}).encode()


def _relay_council_json(handler, body: bytes, status: int, model: str,
                        sha: str, idr: str, headers: Iterable[Tuple[str, str]] = ()) -> None:
    """Non-streaming council reply (or any VPS error): relay the VPS's JSON verbatim,
    with the OpenAI status mapping, retry headers and seal headers."""
    handler.send_response(openai_wire_status(status))
    for name, value in retry_headers(headers):
        handler.send_header(name, value)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    if sha:
        handler.send_header("X-Grip-Council-Sha", sha)
    if idr:
        handler.send_header("X-Grip-Idr-Leaf", idr)
    handler.send_header("Connection", "close")
    handler.end_headers()
    try:
        handler.wfile.write(body)
    except (BrokenPipeError, ConnectionResetError):
        pass


def _council_text(completion) -> "str | None":
    """The text of a VPS completion's first choice, or None when it has no text, so
    a malformed or empty reply is reported instead of arriving as an empty answer
    that looks finished."""
    if not isinstance(completion, dict):
        return None
    choices = completion.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    return content if isinstance(content, str) and content else None


def _council_finish(completion: dict) -> str:
    """The VPS's own finish reason, so a client can tell a truncated answer from a whole one."""
    finish = completion["choices"][0].get("finish_reason")
    return finish if isinstance(finish, str) and finish else "stop"


def _relay_council_stream(handler, body: bytes, model: str, sha: str, idr: str,
                          *, include_usage: bool = False) -> None:
    """Streaming council reply: synthesize a text/event-stream from the VPS's single
    JSON completion so a streaming OpenAI-compat client receives a finish_reason and a
    [DONE] instead of a non-SSE JSON body it cannot parse."""
    try:
        completion = json.loads(body)
    except Exception:
        completion = None
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "close")
    if sha:
        handler.send_header("X-Grip-Council-Sha", sha)
    if idr:
        handler.send_header("X-Grip-Idr-Leaf", idr)
    handler.end_headers()
    content = _council_text(completion)
    try:
        if content is not None:
            completion_id = _completion_id()
            created = _now()
            handler.wfile.write(_sse_chunk(completion_id, created, model,
                                           {"role": "assistant"}, None))
            handler.wfile.write(_sse_chunk(completion_id, created, model,
                                           {"content": content}, None))
            handler.wfile.write(_sse_chunk(completion_id, created, model, {}, _council_finish(completion)))
            if include_usage and isinstance(completion.get("usage"), dict):
                usage_chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model, "choices": [],
                    "usage": completion["usage"],
                }
                handler.wfile.write(("data: " + json.dumps(usage_chunk) + "\n\n").encode())
        else:
            message = "council (VPS) returned an unreadable reply"
            if isinstance(completion, dict) and isinstance(completion.get("error"), dict):
                message = str(completion["error"].get("message") or message)
            handler.wfile.write(chat_stream_error(message))
        handler.wfile.write(b"data: [DONE]\n\n")
        handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        pass


def _relay_council_failopen(handler, payload: dict, model: str, started: float, err: str) -> None:
    """VPS council unreachable -> serve ONE fast Pool-B model via the existing
    same-wire path so the default DEGRADES instead of dying (ADR 0012 fail-open)."""
    import os
    # gemini-4-hydrogen is the fail-open model (V>> 2026-11-26): this
    # function's own docstring above asks for a FAST model. Through this
    # gateway, Hydrogen answered a short prompt in a median 2.56s and a
    # council-sized one in 3.95s, against Carbon's 6.0s and 11.32s and Neon's
    # 14.64s and 10.60s (docs/measurements/2026-11-26-gateway-fail-open-
    # latency.md). It is the single exception to model-dogfood decision 6,
    # pinned by tests/test_gemini_catalogue_parity.py.
    fallback = os.environ.get("GRIP_COUNCIL_FAILOPEN_MODEL", "gemini-4-hydrogen")
    try:
        from lib.hal.gateway_egress import same_wire_provider
        prov = same_wire_provider(fallback)
        if prov is not None:
            degraded_raw = json.dumps(dict(payload, model=fallback)).encode("utf-8")
            # Do NOT witness a hardcoded 200 here: _relay_same_wire records the
            # audit row for the fallback provider with its REAL upstream status
            # (see line ~546), so a premature 200 would claim success before the
            # relay has run and could still fail downstream. The same-wire path is
            # the single honest audit boundary for the degraded serve.
            _relay_same_wire(handler, degraded_raw, prov, fallback, started)
            return
    except SpendRefused as exc:
        send_openai_spend_refusal(handler, exc.decision)
        return
    except Exception:  # noqa: BLE001
        pass
    _send_openai_error(handler, 502,
                       f"council (VPS) unavailable and fallback '{fallback}' unroutable: {redact(err)}")


def _relay_same_wire(handler, raw, provider: str, model: str, started: float) -> None:
    """Cross/same-wire path: forward verbatim to the SERVED non-Anthropic
    provider, relay the reply, and witness it. This path never touches
    ``proxy.forward``, so this is the only place its served provider gets
    recorded (the moat hole).

    The witness keeps the status the provider really sent; the client is
    sent a 503 for a 429 instead (:func:`openai_wire_response`, fleet finding
    f26), on a response object kept separate from the one witnessed. A
    successful reply is witnessed AFTER the relay (:func:`_relay_same_wire_2xx`)
    because its token counts arrive at the end of the body -- witnessing
    first, as this path used to, left every row here without counts or cost,
    among them 2,654 DeepSeek requests in the ten days to 2026-10-03. An
    error reply carries no counts and is still witnessed first.

    Phase 0 same-wire re-pointing (HAL_GATEWAY_MULTIPROVIDER); claude-*, unknown,
    and flag-off all return None upstream, so the #3274 verbatim path is never
    entered here.
    """
    from lib.hal.gateway_egress import forward_same_wire
    from lib.hal import gemini_signature_cache as gsc
    is_gemini = gsc._is_gemini(model)
    # Gemini thought_signature round-trip (request side): re-inject the cached
    # signatures the OpenAI-compat client dropped into assistant tool_calls, so a
    # multi-turn agentic loop does not 400. Verbatim for every other provider.
    body = gsc.reinject_raw(raw) if is_gemini else raw
    sw_resp = forward_same_wire(provider, handler.path.split("?", 1)[0], body)
    served_model = resolve_model(model)
    client_resp = openai_wire_response(sw_resp)
    if not 200 <= sw_resp.status < 300:
        _witness_chat(provider, served_model, sw_resp.status, started=started)
        relay_response(handler, client_resp, stream_error=chat_stream_error)
        return
    _relay_same_wire_2xx(handler, client_resp, provider, served_model,
                        sw_resp.status, started, is_gemini)


def _relay_same_wire_2xx(handler, client_resp, provider: str, served_model: str,
                         status, started: float, is_gemini: bool) -> None:
    """The upstream's successful reply: relay it through the CANONICAL
    :func:`lib.hal.gateway.relay_response`, tapped by a
    :class:`lib.hal.gateway._TeeReader` so its usage (and, for Gemini, an
    unbounded whole-body copy for thought signatures) is read as it passes
    -- never a hand-rolled copy of the relay, which would drift from a later
    fix (framing, fleet findings f72/f79) made only to the canonical one.
    Witness AFTER the relay with whatever the tap read: its state lives
    outside the tee's own observer contract, so what arrived before an
    upstream reset still counts, and the row is written either way."""
    from dataclasses import replace
    from lib.hal.gateway import _TeeReader
    tap = _UsageTap()
    whole = _WholeBodyCapture() if is_gemini else None
    observer = _SplitFeed(tap, whole) if whole is not None else tap
    tapped = replace(client_resp, _reader=_TeeReader(client_resp._reader, observer))
    latency_ms = round((time.monotonic() - started) * 1000)
    try:
        relay_response(handler, tapped, stream_error=chat_stream_error)
    finally:
        _witness_chat(provider, served_model, status, started=started,
                      usage=tap.usage(), latency_ms=latency_ms)
        if whole is not None:
            _capture_signatures(whole.data())


class _WholeBodyCapture:
    """Keeps every chunk shown to it, for Gemini's unbounded thought-signature
    capture. The ``_TeeReader`` contract: ``feed`` never raises; on any error
    it stops watching, the same way :class:`_UsageTap` does, rather than hand
    a later turn a signature read from a corrupted copy."""

    def __init__(self) -> None:
        self._data = bytearray()
        self._broken = False

    def feed(self, chunk: bytes) -> None:
        try:
            if not self._broken:
                self._data.extend(chunk)
        except Exception:  # noqa: BLE001 - observation must never break the relay
            self._broken = True

    def data(self) -> bytes:
        return b"" if self._broken else bytes(self._data)


class _SplitFeed:
    """One ``_TeeReader`` observer that feeds both the usage tap and, for
    Gemini, the whole-body signature copy. Each half already guards itself,
    so nothing here can raise either."""

    def __init__(self, tap: "_UsageTap", whole: "_WholeBodyCapture") -> None:
        self._tap = tap
        self._whole = whole

    def feed(self, chunk: bytes) -> None:
        self._tap.feed(chunk)
        self._whole.feed(chunk)


def _capture_signatures(data: bytes) -> None:
    """Cache this turn's Gemini thought signatures (never raises)."""
    try:
        from lib.hal import gemini_signature_cache as gsc
        gsc.capture_from_bytes(data)
    except Exception:
        pass  # capture is a side-effect; never break a served response


#: A JSON reply is held up to this many bytes so its usage can be read when it
#: ends; past it the row records no counts rather than wrong ones. A stream is
#: never held whole: it is read line by line as it passes.
_JSON_CAP = 32 * 1024 * 1024
#: The longest unfinished stream line kept while waiting for its newline. A
#: usage line is a few hundred bytes, so a longer one is content and dropping
#: it loses no count.
_LINE_CAP = 1024 * 1024


class _UsageTap:
    """Reads token counts from an OpenAI-wire reply as it passes through.

    A JSON reply carries one top-level ``usage``. A stream carries it in a late
    chunk, and only when the client asked with ``stream_options.include_usage``:
    this path forwards requests verbatim, so it does not add the option the way
    the cross-wire path does. Moonshot puts it on ``choices[0].usage`` and the
    Responses API inside ``response.usage``.

    A stream is read line by line as chunks arrive, so it costs one unfinished
    line of memory rather than a copy of itself, however long it runs, and
    counts that arrived before an upstream reset are kept. Lines rather than
    blank-line blocks, so a CRLF stream is read in full. Usage blocks merge key
    by key, later winning, so a chunk carrying only the output count keeps an
    earlier input count; the result is normalised once, so the cached-token
    subtraction sees one consistent pair. The first non-blank byte decides
    which kind of reply it is: ``{`` opens a JSON reply, anything else a stream.
    """

    def __init__(self) -> None:
        self._json: Optional[bool] = None
        self._pending = bytearray()
        self._overflow = False
        self._merged: dict = {}
        self._broken = False

    def feed(self, chunk: bytes) -> None:
        """Take the next chunk of the reply -- the ``_TeeReader`` contract:
        never raises. On any error, stop watching: reading a later chunk
        against now-inconsistent state could report a WRONG count, where a
        missing one is honest (``infinite_cost.UsageObserver`` does the same)."""
        try:
            self._take(chunk)
        except Exception:  # noqa: BLE001 - observation must never break the relay
            self._broken = True

    def _take(self, chunk: bytes) -> None:
        if self._broken or self._overflow:
            return
        self._pending.extend(chunk)
        if self._json is None:
            head = self._pending.lstrip()[:1]
            if not head:
                return
            self._json = head == b"{"
        if self._json:
            self._hold_json()
        else:
            self._read_lines()

    def usage(self) -> Optional[dict]:
        """The counts read so far, or None -- never a dict of zeros, and
        never once broken -- when nothing usable arrived, so the row omits
        the fields instead of claiming the call was free or a wrong one.
        Never raises: it runs in the relay's ``finally``."""
        if self._broken:
            return None
        try:
            from lib.hal.gateway_crosswire import normalise_provider_usage
            merged = self._json_usage() if self._json else self._stream_usage()
            counts = normalise_provider_usage({"usage": merged}) if merged else None
            return counts if counts and any(counts.values()) else None
        except Exception:
            return None

    def _hold_json(self) -> None:
        if len(self._pending) > _JSON_CAP:
            self._pending, self._overflow = bytearray(), True

    def _read_lines(self) -> None:
        *lines, rest = self._pending.split(b"\n")
        self._pending = bytearray(rest if len(rest) <= _LINE_CAP else b"")
        for line in lines:
            self._absorb(line)

    def _absorb(self, line: bytes) -> None:
        text = line.strip()
        if not text.startswith(b"data:") or b'"usage"' not in text:
            return
        try:
            self._merged.update(_usage_block(json.loads(text[5:])) or {})
        except ValueError:
            pass  # a line that is not JSON carries no count

    def _json_usage(self) -> dict:
        if self._overflow:
            return {}
        try:
            return _usage_block(json.loads(bytes(self._pending))) or {}
        except ValueError:
            return {}

    def _stream_usage(self) -> dict:
        if self._pending:
            self._absorb(bytes(self._pending))  # a last line with no newline
        return self._merged


def _usage_block(obj) -> Optional[dict]:
    """The usage dict in one reply object, wherever this provider puts it."""
    if not isinstance(obj, dict):
        return None
    choices = obj.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else None
    for holder in (obj, obj.get("response"), first):
        usage = holder.get("usage") if isinstance(holder, dict) else None
        if isinstance(usage, dict):
            return usage
    return None


def _witness_chat(provider: str, model: str, status, *, started: float,
                  usage: Optional[dict] = None,
                  latency_ms: Optional[int] = None) -> None:
    """Witness + sign one chat-rail request (fail-open; never raises).

    Lazy import breaks the gateway <- gateway_witness module cycle and keeps the
    witness entirely outside the forward path. Any failure is swallowed — the
    audit/sign value-add is monotone-toward-safe and can never subtract a
    served request."""
    try:
        from lib.hal.gateway_witness import witness_chat
        witness_chat(provider=provider, model=model, status=status,
                     served=status is not None, started=started,
                     path=CHAT_COMPLETIONS_PATH, usage=usage,
                     latency_ms=latency_ms)
    except Exception:
        pass


def _relay_nonstream(handler, resp, *, model, completion_id, created) -> None:
    try:
        data = json.loads(resp.read_all())
        openai_resp = anthropic_to_openai_response(
            data, model=model, completion_id=completion_id, created=created)
    except Exception as exc:  # nothing sent yet: a lost upstream is a 503
        _send_openai_error(handler, *failure_reply(exc, other_label="upstream parse error"))
        return
    finally:
        resp.close()  # release the upstream socket
    _send_json(handler, 200, openai_resp)


def _relay_stream(handler, resp, *, model, completion_id, created) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "close")
    handler.end_headers()
    stream = translate_sse_stream(
        resp.iter_chunks(), model=model, completion_id=completion_id, created=created)
    try:
        for chunk in stream:
            try:
                handler.wfile.write(chunk)
                handler.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break
    finally:
        resp.close()  # release the upstream socket even on client disconnect


def _relay_upstream_error(handler, resp) -> None:
    """Wrap an Anthropic error response as an OpenAI-shaped error, keeping its
    ``Retry-After`` (fleet finding f73), at the same status, except that a 429
    goes out as a 503 (:func:`lib.hal.gateway.openai_wire_status`, finding f26)."""
    message = "upstream error"
    status = openai_wire_status(resp.status)
    try:
        data = json.loads(resp.read_all())
        message = (data.get("error") or {}).get("message") or message
    except Exception:
        pass
    finally:
        resp.close()  # release the upstream socket
    _send_openai_error(handler, status, redact(message), retry_headers(resp.headers))


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


def _send_openai_error(handler, code: int, message: str,
                       headers: Iterable[Tuple[str, str]] = ()) -> None:
    """Emit an OpenAI-shaped error envelope (message pre-redacted), plus any
    ``headers`` (a ``Retry-After``, say)."""
    _send_json(handler, code, {"error": {
        "message": message, "type": "gateway_error", "code": None, "param": None}},
        headers)


def send_openai_spend_refusal(handler, decision) -> None:
    """The spend guard's refusal as an OpenAI-shaped error, on both OpenAI-wire rails.

    Nothing went upstream. The status is the guard's: 403 for a paid key, 429
    with ``Retry-After`` for a spent subscription.
    """
    payload = json.dumps({"error": {
        "message": redact(decision.reason), "type": "spend_guard",
        "code": "spend_refused", "param": None}}).encode("utf-8")
    handler.send_response(decision.status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    if decision.retry_after:
        handler.send_header("Retry-After", str(decision.retry_after))
    handler.send_header("Connection", "close")
    handler.end_headers()
    try:
        handler.wfile.write(payload)
    except (BrokenPipeError, ConnectionResetError):
        pass
