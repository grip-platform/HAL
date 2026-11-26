"""Real Anthropic ``chat_fn`` factory for the HAL tool loop (W3).

W1 built the schema bridge; W2 built the dispatch driver; W3 provides a
concrete ``chat_fn`` that talks to the real Messages API so callers can
plug ``run_tool_loop`` into production paths without duplicating HTTP
wiring or Messages API shape knowledge.

The factory returns a closure matching the contract
:func:`~lib.hal.tool_loop.run_tool_loop` expects::

    chat_fn(*, messages: list[dict], tools: list[dict]) -> dict

where the returned dict is the raw Anthropic response (``content``,
``stop_reason``, ``usage``, …) — exactly what
:func:`~lib.hal.tool_adapter.parse_tool_use_blocks` was designed to read.

stdlib-only: uses :mod:`urllib.request` so HAL does not gain a
third-party SDK dependency for what is ultimately a single HTTPS POST.

Usage (the end-to-end loop that W1+W2+W3 enable):

    from lib.hal.tool_adapter import to_anthropic_tools
    from lib.hal.tool_dispatch import ToolDispatcher
    from lib.hal.tool_loop import run_tool_loop
    from lib.hal.tool_runtime import make_anthropic_chat_fn

    chat_fn = make_anthropic_chat_fn(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-opus-4-7",
        max_tokens=4096,
    )

    result = run_tool_loop(
        messages=[{"role": "user", "content": "read /tmp/x and summarise"}],
        chat_fn=chat_fn,
        tools=to_anthropic_tools(),
        dispatcher=ToolDispatcher(),
    )
    print(result.final_response["content"][-1].get("text", ""))

The factory deliberately does NOT wrap :class:`CCHSubscription` — CCH
callers already own the full request lifecycle (header signing, body
construction, context_management shaping) and want ``tools`` plumbed
through :class:`~lib.hal.utils.cch.CCHSigner` instead. Adding CCH
support here would duplicate that plumbing; W3-bis / W4 extends the
signer to accept ``tools`` directly.
"""
import json
import urllib.error
import urllib.request
from typing import Callable, Optional

from lib.hal.spend_guard import require_openai_destination
from lib.hal.model_quirks import (
    locked_sampling_params,
    refused_reasoning_efforts,
    tools_require_responses_api,
    uses_max_completion_tokens,
)

_DEFAULT_API_URL = "https://api.anthropic.com/v1/messages"
_DEFAULT_API_VERSION = "2023-06-01"
_DEFAULT_TIMEOUT_S = 120


class AnthropicChatError(RuntimeError):
    """Raised when a chat completion API returns a non-2xx response.

    Name kept for backwards compatibility — this class covers BOTH the
    Anthropic Messages API and OpenAI-compat providers (Groq, OpenAI,
    Gemini-OpenAI, Ollama). The ``provider`` attribute disambiguates so
    callers can render "Groq API returned 400" instead of the misleading
    "Anthropic API returned 400" that misled issue #200's triage.

    The HTTP status code, response body, and provider label are exposed
    so callers can render a useful error message without a second request.
    """

    def __init__(
        self, status: int, body: str, message: str = "",
        provider: Optional[str] = None,
    ):
        self.status = status
        self.body = body
        self.provider = provider or "Anthropic"
        super().__init__(
            message or f"{self.provider} API returned {status}: {body[:200]}"
        )


# Kept as an alias in case callers catch AnthropicChatError by name today;
# new code should reach for ChatAPIError for clarity.
ChatAPIError = AnthropicChatError
# PR #205 introduced ChatTransportError as a more provider-neutral name;
# expose both so existing imports don't break.
ChatTransportError = AnthropicChatError


def _provider_label_from_url(url: str) -> str:
    """Extract a human-readable provider label from a chat endpoint URL.

    Used only for error-message diagnostics — the logic stays behaviour-
    neutral on the happy path.
    """
    if "groq.com" in url:
        return "Groq"
    if "googleapis.com" in url:
        return "Gemini"
    if "openai.com" in url:
        return "OpenAI"
    if "localhost" in url or "127.0.0.1" in url:
        return "Ollama"
    return "Provider"


def _anthropic_tools_to_openai(tools: list[dict]) -> list[dict]:
    """Convert Anthropic-shaped tool specs to OpenAI-compat shape.

    Anthropic shape (input):  ``{name, description, input_schema}``
    OpenAI shape   (output):  ``{"type": "function", "function":
                                 {name, description, parameters}}``

    Issue #200 root cause: :func:`make_openai_compat_chat_fn` received
    tools that :func:`lib.hal.tool_adapter.to_anthropic_tools` had shaped
    for Anthropic, then posted them verbatim to Groq/OpenAI/Gemini — which
    rejected every request with ``'tools.0.type' : property 'type' is
    missing``. This helper is the translator that was missing.

    No mutation of inputs; returns a fresh list. If ``tools`` is already
    in OpenAI shape (has a ``"type"`` key at the top level), it is
    passed through unchanged — makes the conversion idempotent for
    callers that have already done the shape work.
    """
    out: list[dict] = []
    for spec in tools:
        if spec.get("type") == "function" and "function" in spec:
            # Already OpenAI-shaped — leave alone
            out.append(spec)
            continue
        out.append({
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec.get("description", ""),
                "parameters": spec.get("input_schema", {}),
            },
        })
    return out


def _post_messages(
    *,
    url: str,
    api_key: str,
    api_version: str,
    body: dict,
    timeout_s: float,
    extra_headers: Optional[dict] = None,
) -> dict:
    """POST a body to the Messages API and return the parsed JSON dict.

    Split out so tests can patch at a single seam
    (``lib.hal.tool_runtime._post_messages``) without caring about
    urllib shape, header ordering, or JSON encoding details.
    """
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": api_version,
    }
    if extra_headers:
        headers.update(extra_headers)

    request = urllib.request.Request(
        url=url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
        return json.loads(raw)
    except urllib.error.HTTPError as err:
        try:
            error_body = err.read().decode("utf-8")
        except Exception:  # pragma: no cover — defensive against stream exhaustion
            error_body = ""
        raise AnthropicChatError(err.code, error_body) from err


def make_anthropic_chat_fn(
    *,
    api_key: str,
    model: str,
    max_tokens: int = 2048,
    system: Optional[str] = None,
    api_url: str = _DEFAULT_API_URL,
    api_version: str = _DEFAULT_API_VERSION,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    extra_headers: Optional[dict] = None,
) -> Callable[..., dict]:
    """Build a ``chat_fn`` that POSTs to Anthropic's Messages API.

    The returned closure matches :func:`run_tool_loop`'s contract: it is
    invoked as ``chat_fn(messages=[...], tools=[...])`` and returns the
    raw response dict.

    ``api_key``, ``model``, ``max_tokens``, ``system`` are baked into
    the closure — one factory call per "session". Switching model or
    system prompt mid-loop means building a new ``chat_fn``.

    ``tools`` is only added to the request body when non-empty — passing
    ``tools=[]`` is treated the same as "no tools", matching the
    short-circuit path in :func:`run_tool_loop`. This keeps request
    bodies minimal for tool-free turns and avoids the "tools must have
    at least one entry" rejection some API versions surface.

    ``extra_headers`` is useful for beta flags (e.g.
    ``{"anthropic-beta": "tools-2024-04-04"}``) without needing to
    fork the factory — pass through verbatim.
    """

    def chat_fn(*, messages: list[dict], tools: list[dict]) -> dict:
        body: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            body["system"] = system
        if tools:
            body["tools"] = tools

        return _post_messages(
            url=api_url,
            api_key=api_key,
            api_version=api_version,
            body=body,
            timeout_s=timeout_s,
            extra_headers=extra_headers,
        )

    return chat_fn


def extract_final_text(response: dict) -> str:
    """Return the concatenated ``text`` blocks from a Messages API response.

    Convenience for callers who just want the model's final prose after
    :func:`run_tool_loop` returns. Skips ``tool_use``, ``thinking``, and
    any unknown block types — defensive against future content-block
    additions the rest of HAL hasn't been taught yet.

    Returns an empty string if ``response`` is not a dict, has no
    ``content`` list, or contains no text blocks. Callers distinguishing
    "empty response" from "no text in response" should inspect the raw
    ``content`` list themselves.
    """
    if not isinstance(response, dict):
        return ""
    content = response.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = block.get("text", "")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


_FINISH_REASON_MAP = {
    "tool_calls": "tool_use",
    "stop": "end_turn",
    "length": "max_tokens",
}


def _parse_tool_arguments(raw_args) -> tuple[dict, Optional[str]]:
    """Parse an OpenAI ``function.arguments`` value into (dict, error_str).

    ``raw_args`` can be a dict (rare, provider-dependent), a JSON string
    (OpenAI canonical), empty, or malformed. Returns ``({}, err)`` on
    parse failure so the caller can surface the error without crashing.
    """
    if isinstance(raw_args, dict):
        return raw_args, None
    if isinstance(raw_args, str) and raw_args:
        try:
            parsed = json.loads(raw_args)
            return (parsed if isinstance(parsed, dict) else {}), None
        except json.JSONDecodeError as exc:
            return {}, f"{exc.__class__.__name__}: {exc}"
    return {}, None


def _recover_llama_textcall(name: str) -> tuple[str, str]:
    """Recover from LLaMA text-mode function call where name contains embedded JSON args.

    Groq's LLaMA 3.3 70B occasionally slips into XML/text format and emits the
    function name with embedded JSON arguments, e.g. ``Bash{"command": "ls"}``.
    When ``arguments`` is empty/missing AND ``name`` contains ``{``, this helper
    splits the name at the first brace and validates the suffix as JSON.

    Returns ``(original_name, "")`` unchanged when the pattern does not match,
    so callers can unconditionally unpack the two-tuple with zero behaviour change
    for well-formed providers.
    """
    if not isinstance(name, str) or "{" not in name:
        return name, ""
    brace = name.index("{")
    candidate_name = name[:brace].strip()
    candidate_args = name[brace:].strip()
    if not candidate_name or not candidate_args.endswith("}"):
        return name, ""
    try:
        json.loads(candidate_args)
    except json.JSONDecodeError:
        return name, ""
    return candidate_name, candidate_args


def _tool_call_to_block(call: dict) -> Optional[dict]:
    """Convert one OpenAI ``tool_calls[i]`` entry to an Anthropic ``tool_use`` block.

    Non-dict entries are dropped (return ``None``) so a single malformed
    entry does not poison the whole turn. Malformed JSON arguments are
    preserved as an empty ``input`` plus a ``_parse_error`` marker.

    LLaMA text-mode recovery: when ``arguments`` is empty/missing AND ``name``
    contains ``{``, :func:`_recover_llama_textcall` is attempted to split the
    embedded JSON out of the name field. Zero behaviour change for well-behaved
    providers where ``arguments`` is non-empty.
    """
    if not isinstance(call, dict):
        return None
    func = call.get("function") if isinstance(call.get("function"), dict) else {}
    raw_name = func.get("name", "")
    raw_args = func.get("arguments", "")
    # Only attempt recovery when arguments is absent/empty and name looks corrupt.
    if not raw_args and isinstance(raw_name, str) and "{" in raw_name:
        raw_name, raw_args = _recover_llama_textcall(raw_name)
    tool_input, parse_error = _parse_tool_arguments(raw_args)
    block = {
        "type": "tool_use",
        "id": call.get("id", ""),
        "name": raw_name,
        "input": tool_input,
    }
    if parse_error:
        block["_parse_error"] = parse_error
    return block


def _openai_message_to_content(message: dict) -> list[dict]:
    """Build the Anthropic ``content`` list from an OpenAI ``choices[0].message``."""
    blocks: list[dict] = []
    text = message.get("content")
    if isinstance(text, str) and text:
        blocks.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        block = _tool_call_to_block(call)
        if block is not None:
            blocks.append(block)
    return blocks


class NormalizerError(ValueError):
    """Raised when an OpenAI-compat response cannot be safely normalised.

    Silent degradation to ``{"content": [], "stop_reason": "end_turn"}`` on
    malformed input was the original W1/W2 posture. HAL Council review
    (2026-04-17, PR #173) flagged this as a tool-loop footgun: silent
    degradation masks real failures (rate limits, provider errors,
    truncated payloads) as legitimate end-of-turn, which drives
    ``run_tool_loop`` into infinite retries or silent context drops.

    The strict posture propagates the real failure to the caller so the
    loop halts cleanly instead of looping on a synthesised empty turn.

    Base class widened from ``RuntimeError`` to ``ValueError`` (PR #173
    council re-review, W2a): structurally invalid response shapes are
    input-validation failures in the Python sense, not runtime faults in
    our code path, so ``ValueError`` matches Python convention and lets
    callers that already catch ``ValueError`` at loop boundaries pick up
    the new error type without bespoke handling.
    """


# Module-level sentinel for the "model said nothing" case (W2c). Returning
# a shared instance (rather than a fresh ``{}`` literal each time) makes
# the empty-message branch visibly identifiable in a debugger watch — if
# you see ``_EMPTY_MESSAGE`` on an inspected frame, you know the model
# replied with finish_reason but no content, not that the extractor
# returned a newly-allocated empty dict.
_EMPTY_MESSAGE: dict = {}


def _validate_response_envelope(openai_response) -> None:
    """Raise :class:`NormalizerError` on malformed outer response envelope.

    Rejects non-dict payloads and surfaced-in-body provider errors
    (``{"error": {...}}`` without ``choices``) that some providers and
    proxies return with HTTP 200 under load.
    """
    if not isinstance(openai_response, dict):
        raise NormalizerError(
            f"expected dict response, got {type(openai_response).__name__}"
        )
    err = openai_response.get("error")
    if err is None or openai_response.get("choices"):
        return
    msg = err.get("message") or str(err) if isinstance(err, dict) else str(err)
    raise NormalizerError(f"provider error: {msg}")


def _extract_first_choice(openai_response: dict) -> dict:
    """Return the first choice dict or raise :class:`NormalizerError`.

    Rejects missing or empty ``choices`` arrays and non-dict first entries.
    """
    choices = openai_response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise NormalizerError("response has no 'choices' array")
    first = choices[0]
    if not isinstance(first, dict):
        raise NormalizerError(
            f"choices[0] is {type(first).__name__}, expected dict"
        )
    return first


def _extract_message_dict(first: dict) -> dict:
    """Return the message dict, accepting legitimate empty responses.

    A model that chose to say nothing (missing ``message``) is valid IF
    ``finish_reason`` is present — that's a legitimate end-of-turn. Missing
    both is structurally invalid.

    Returns the module-level :data:`_EMPTY_MESSAGE` sentinel for the
    "model said nothing" path (W2c) instead of a fresh ``{}`` literal so
    debuggers can distinguish a deliberate silent-turn from an incidental
    empty dict constructed elsewhere in the pipeline.
    """
    message_raw = first.get("message")
    if isinstance(message_raw, dict):
        return message_raw
    if first.get("finish_reason") is None:
        raise NormalizerError(
            "choices[0] has neither 'message' dict nor 'finish_reason'"
        )
    return _EMPTY_MESSAGE


def _normalize_openai_to_anthropic(openai_response: dict) -> dict:
    """Convert an OpenAI-compat chat-completions response to Anthropic shape.

    The HAL tool loop (:func:`run_tool_loop`) consumes Anthropic-shape
    ``content`` blocks (``text`` and ``tool_use``) plus a ``stop_reason``.
    This function is the single translation seam between the two shapes so
    the rest of HAL stays Anthropic-canonical.

    Strict by design (post-HAL-Council fix, PR #173): structurally invalid
    input raises :class:`NormalizerError` via the three validation helpers
    rather than silently degrading. A legitimately empty response (model
    chose to say nothing with ``finish_reason`` set) is NOT an error.

    Validation is delegated to :func:`_validate_response_envelope`,
    :func:`_extract_first_choice`, and :func:`_extract_message_dict` so
    each failure mode is a testable, named condition.

    Notes for callers (PR #173 council re-review, W2b)
    ---------------------------------------------------
    :class:`NormalizerError` subclasses :class:`ValueError` and is raised
    on structurally invalid upstream responses. The expected catch point
    is the tool-loop boundary so one bad turn halts the loop cleanly
    instead of being papered over as an empty end-turn. Minimal circuit-
    breaker sketch::

        try:
            response = chat_fn(messages=msgs, tools=tools)
        except NormalizerError as err:
            log.error("provider returned invalid shape: %s", err)
            raise  # or break; don't retry on structural errors
    """
    _validate_response_envelope(openai_response)
    first = _extract_first_choice(openai_response)
    message = _extract_message_dict(first)
    usage_raw = openai_response.get("usage") or {}
    return {
        "id": openai_response.get("id", ""),
        "model": openai_response.get("model", ""),
        "content": _openai_message_to_content(message),
        "stop_reason": _FINISH_REASON_MAP.get(
            first.get("finish_reason") or "stop", "end_turn",
        ),
        "usage": {
            "input_tokens": usage_raw.get("prompt_tokens", 0),
            "output_tokens": usage_raw.get("completion_tokens", 0),
        },
    }


def _derive_provider_label(url: str) -> str:
    """Best-effort provider label from the base URL for error messages.

    Used when an OpenAI-compat request fails so the surfaced error reads
    ``Groq API returned 400`` instead of the historical (misleading)
    ``Anthropic API returned 400``. Fully string-driven — no network
    lookups — so label derivation can't itself fail and produces a sane
    default (``"Provider"``) for unknown hosts.
    """
    lowered = url.lower()
    if "groq.com" in lowered:
        return "Groq"
    if "openai.com" in lowered:
        return "OpenAI"
    if "googleapis.com" in lowered or "generativelanguage" in lowered:
        return "Gemini"
    if "localhost" in lowered or "127.0.0.1" in lowered:
        return "Ollama"
    if "anthropic.com" in lowered:
        return "Anthropic"
    return "Provider"


def _post_openai_chat(
    *,
    url: str,
    api_key: str,
    body: dict,
    timeout_s: float,
    extra_headers: Optional[dict] = None,
) -> dict:
    """POST to an OpenAI-compat ``/chat/completions`` endpoint.

    Separate seam from :func:`_post_messages` so tests can patch exactly
    the OpenAI path without touching Anthropic's, and so the two share no
    header shape (Bearer vs. x-api-key + anthropic-version).

    On non-2xx the raised :class:`AnthropicChatError` carries the derived
    provider label (see :func:`_derive_provider_label`) so error surfaces
    name the real provider rather than the misleading "Anthropic" historical
    default — Issue #200 context.
    """
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "HAL/1.0",
    }
    require_openai_destination(url)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)

    request = urllib.request.Request(
        url=url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
        return json.loads(raw)
    except urllib.error.HTTPError as err:
        try:
            error_body = err.read().decode("utf-8")
        except Exception:  # pragma: no cover — defensive
            error_body = ""
        raise AnthropicChatError(
            err.code, error_body,
            provider=_provider_label_from_url(url),
        ) from err


def _assistant_blocks_to_openai(content: list[dict]) -> dict:
    """Convert Anthropic assistant content blocks to an OpenAI assistant message.

    Single pass over ``content`` — O(m) where m is blocks per turn (bounded,
    typically 1-3). Text blocks are joined into ``content``; tool_use blocks
    become the ``tool_calls`` array. ``content`` is ``None`` when there are
    no text blocks, matching the OpenAI spec for pure-tool-call turns.
    """
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            t = block.get("text", "")
            if t:
                text_parts.append(t)
        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })
    msg: dict = {"role": "assistant", "content": " ".join(text_parts) or None}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _tool_result_blocks_to_openai(content: list[dict]) -> list[dict]:
    """Expand Anthropic tool_result blocks to individual OpenAI tool messages.

    OpenAI uses ``role: "tool"`` with ``tool_call_id`` for each result.
    Any plain-text blocks in the same user turn are preserved as a trailing
    user message. Single pass — O(m).
    """
    out: list[dict] = []
    text_parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_result":
            out.append({
                "role": "tool",
                "content": str(block.get("content", "")),
                "tool_call_id": block.get("tool_use_id", ""),
            })
        elif block.get("type") == "text":
            t = block.get("text", "")
            if t:
                text_parts.append(t)
    if text_parts:
        out.append({"role": "user", "content": " ".join(text_parts)})
    return out


def _convert_messages_for_openai(messages: list[dict]) -> list[dict]:
    """Convert Anthropic-format message history to OpenAI-compat format.

    ``run_tool_loop`` accumulates messages in Anthropic shape. On the second
    and subsequent requests to an OpenAI-compat endpoint (Groq, Ollama,
    Gemini, OpenAI) the intermediate turns must be re-shaped:

    * assistant + ``tool_use`` blocks → ``tool_calls`` array, ``content: null``
    * user + ``tool_result`` blocks → one ``role: "tool"`` message per result

    Plain string messages pass through unchanged. Delegates per-role shaping
    to :func:`_assistant_blocks_to_openai` and
    :func:`_tool_result_blocks_to_openai` to stay under the complexity limit.
    """
    out: list[dict] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")
        if isinstance(content, str) or content is None:
            out.append({"role": role, "content": content or ""})
        elif isinstance(content, list):
            if role == "assistant":
                out.append(_assistant_blocks_to_openai(content))
            elif role == "user":
                out.extend(_tool_result_blocks_to_openai(content))
            else:
                out.append(msg)
        else:
            out.append(msg)
    return out


def make_openai_compat_chat_fn(
    *,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    extra_headers: Optional[dict] = None,
) -> Callable[..., dict]:
    """Build a ``chat_fn`` backed by an OpenAI-compat ``/chat/completions``.

    Matches the :func:`run_tool_loop` contract by normalising the OpenAI
    response to Anthropic shape via :func:`_normalize_openai_to_anthropic`
    before returning — callers can treat Ollama, Groq, Gemini-OpenAI-compat,
    and OpenAI identically to CCH / :func:`make_anthropic_chat_fn`.

    ``base_url`` should NOT end with ``/chat/completions`` — we append that
    here so the factory accepts the same URL shape the
    :class:`OpenAICompatProvider` takes (e.g. ``http://localhost:11434/v1``).

    Tools are forwarded under ``tools`` with ``tool_choice="auto"`` only when
    a non-empty list is supplied — matching the
    :func:`make_anthropic_chat_fn` contract so the two are drop-in
    replacements from the loop's perspective.

    Notes for callers (PR #173 council re-review, W2b)
    ---------------------------------------------------
    The returned closure raises :class:`NormalizerError` (subclass of
    :class:`ValueError`) when the provider returns a structurally invalid
    response — use it to halt the tool loop cleanly rather than retrying
    on synthesised empty turns. Transport-level HTTP failures surface as
    :class:`AnthropicChatError` with the upstream status + body.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    responses_url = base_url.rstrip("/") + "/responses"

    def chat_fn(*, messages: list[dict], tools: list[dict]) -> dict:
        if tools_require_responses_api(model):
            # gpt-6 family (Astra): function tools on /chat/completions answer
            # HTTP 400 ("tool calling requires the Responses API" — vendor
            # model page; measured live 2026-09-10). The HAL chat/REPL always
            # attaches its tool definitions, so this wire is chosen per model
            # family, exactly as gateway_egress early-returns to
            # cross_wire_responses_forward — one decision, one quirks
            # predicate, no drift between the gateway and the direct path.
            # The Anthropic-in / Anthropic-out chat_fn contract is preserved
            # by the Responses-wire twin translators.
            from lib.hal.gateway_crosswire_responses import (
                anthropic_to_responses_request,
                responses_to_anthropic_response,
            )
            payload: dict = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                # Forwarded verbatim: anthropic_to_responses_request itself
                # omits the sampling fields the model has locked (gpt-6:
                # temperature/top_p/logprobs), from the same quirks source.
                "temperature": temperature,
            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = {"type": "auto"}
            raw = _post_openai_chat(
                url=responses_url,
                api_key=api_key,
                body=anthropic_to_responses_request(payload),
                timeout_s=timeout_s,
                extra_headers=extra_headers,
            )
            return responses_to_anthropic_response(raw, model=model)
        body: dict = {
            "model": model,
            # Convert Anthropic-shaped multi-turn messages to OpenAI format.
            # After a tool call, run_tool_loop appends an assistant turn whose
            # content is a list of tool_use blocks (Anthropic shape) and a user
            # turn whose content is tool_result blocks. OpenAI-compat endpoints
            # (Groq, Ollama, Gemini, OpenAI) require content:null + tool_calls
            # on the assistant turn, and role:tool messages for results.
            "messages": _convert_messages_for_openai(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            # Issue #200: upstream callers hand us tools in Anthropic shape
            # (from to_anthropic_tools). OpenAI-compat endpoints require the
            # {"type": "function", "function": {...}} wrapper. Translate here
            # so the chat_fn contract stays "pass tools in Anthropic shape"
            # regardless of which provider is wired underneath.
            body["tools"] = _anthropic_tools_to_openai(tools)
            body["tool_choice"] = "auto"
        _fix_openai_compat_body(body)
        raw = _post_openai_chat(
            url=url,
            api_key=api_key,
            body=body,
            timeout_s=timeout_s,
            extra_headers=extra_headers,
        )
        return _normalize_openai_to_anthropic(raw)

    return chat_fn


def _fix_openai_compat_body(body: dict) -> dict:
    """Apply model-aware OpenAI-compat request fixes in place.

    Every rule is read from ``lib.hal.model_quirks`` -- the single home the
    gateway egress and the HAPPI kernel adapter also consume -- so this
    direct-council subprocess path can never drift from them again (it used
    to carry its own ``startswith("gpt-5")``, which silently missed gpt-6):

    * ``max_tokens`` -> ``max_completion_tokens`` where the model rejects
      (OpenAI gpt-5.x / gpt-6) or deprecates (Moonshot) the legacy field;
    * locked sampling fields dropped (gpt-6: temperature / top_p / logprobs;
      the Kimi lineage: its fixed set);
    * a refused ``reasoning_effort`` level dropped (gpt-6 refuses ``none``).
    """
    model = str(body.get("model") or "")
    if uses_max_completion_tokens(model) and "max_tokens" in body:
        legacy = body.pop("max_tokens")
        body.setdefault("max_completion_tokens", legacy)
    for locked in locked_sampling_params(model):
        body.pop(locked, None)
    effort = str(body.get("reasoning_effort", "") or "").strip().lower()
    if effort and effort in refused_reasoning_efforts(model):
        body.pop("reasoning_effort", None)
    return body


__all__ = [
    "AnthropicChatError",
    "ChatAPIError",
    "ChatTransportError",
    "NormalizerError",
    "_convert_messages_for_openai",
    "_fix_openai_compat_body",
    "extract_final_text",
    "make_anthropic_chat_fn",
    "make_openai_compat_chat_fn",
]
