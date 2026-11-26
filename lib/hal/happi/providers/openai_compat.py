"""OpenAI-compat provider adapter for HAPPI/1.1 (v1.0 envelopes accepted).

Protocol-version-agnostic at the wire layer: emits via shared
`lib.hal.happi.events` encoder; envelope `v` (client's declared version)
flows through unchanged.

Covers 16 providers behind one code path:
    cloud:  openai, groq, mistral, deepseek, openrouter, moonshot,
            qwen, zhipu, xai, soruxgpt, vulavula
    local:  ollama, lemonade, vllm, lmstudio, mlx_vlm

Each uses POST /v1/chat/completions with SSE streaming. Wire differences
(endpoint URL, auth header, keychain service, env var) are captured in
PROVIDER_CONFIGS; everything else is shared.

PROVIDER_CONFIGS mirrors `config/providers.yaml`. A sync test
(`test_happi_router_covers_yaml_registry`) prevents the two from
drifting — adding a YAML provider without a matching config entry
fails CI. Origin: HAL#302 — status registry diverged from HAPPI router.

Provider selection via `envelope.flags.provider` (two formats accepted):
    "groq"                                    -> family=groq, default model
    "groq/llama-3.1-70b-versatile"            -> family=groq, model=...
    "openrouter/anthropic/claude-sonnet-4-5"  -> family=openrouter, model=...

Auth resolution: envelope.auth.token (apikey scheme) OR env var OR Keychain.
Ollama local bind uses auth="none" -- no Authorization header sent.

Tool loop (W1b, 2026-09-07) -- three additive chat flags, no executor:
    flags.tools        OpenAI tool schemas, sent verbatim as `tools`
    flags.tool_choice  "auto" | "none" | "required" | {type:function,...};
                       "required" is refused before dispatch on the Kimi K2
                       lineage (model_quirks.supports_required_tool_choice)
    flags.messages     prior turns, echoed VERBATIM between flags.system and
                       the current user turn. The vendor rule this serves
                       (platform.kimi.ai/docs/guide/use-thinking-models.md):
                       "pass the complete assistant message returned by the
                       API back to `messages` as-is (including
                       `reasoning_content`)". The adapter never adds, strips
                       or reorders a key in that history -- a caller talking
                       to a different vendor must apply that vendor's rule.
The runtime executes nothing: the model's tool request is emitted as a
`tool_call` event, the caller runs the tool and sends the result back as a
`role: "tool"` message in flags.messages on the next envelope; this turn's
tool messages are echoed as `tool_result` events so the NDJSON stream alone
records the whole loop.

Error discipline and event mapping mirror anthropic.py. Stdlib-only.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from typing import Any, Iterator, Optional

from lib.hal.spend_guard import SpendRefused, require_openai_destination
from lib.hal.happi.envelope import HappiEnvelope
from lib.hal.happi.events import (
    emit_completed,
    emit_delta,
    emit_error,
    emit_thinking,
    emit_tool_call,
    emit_tool_result,
)
from lib.hal.happi.sampling import resolve_sampling
from lib.hal.model_quirks import (
    locked_sampling_params,
    reasoning_control,
    supports_required_tool_choice,
    uses_max_completion_tokens,
    refused_reasoning_efforts,
)


PROVIDER_CONFIGS: dict[str, dict[str, Any]] = {
    "openai": {
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "keychain": "hal-openai",
        "env": "OPENAI_API_KEY",
        "auth": "bearer",
        "default_model": "gpt-4o-mini",
    },
    "groq": {
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "keychain": "grip-groq",
        "env": "GROQ_API_KEY",
        "auth": "bearer",
        "default_model": "llama-3.1-70b-versatile",
    },
    "mistral": {
        "endpoint": "https://api.mistral.ai/v1/chat/completions",
        "keychain": "grip-mistral",
        "env": "MISTRAL_API_KEY",
        "auth": "bearer",
        "default_model": "mistral-small-latest",
    },
    "deepseek": {
        "endpoint": "https://api.deepseek.com/v1/chat/completions",
        "keychain": "hal-deepseek",
        "env": "DEEPSEEK_API_KEY",
        "auth": "bearer",
        "default_model": "deepseek-chat",
    },
    "openrouter": {
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "keychain": "grip-openrouter",
        "env": "OPENROUTER_API_KEY",
        "auth": "bearer",
        "default_model": "anthropic/claude-sonnet-4-5",
    },
    "moonshot": {
        "endpoint": "https://api.moonshot.ai/v1/chat/completions",
        "keychain": "grip-moonshot",
        "env": "MOONSHOT_API_KEY",
        "auth": "bearer",
        # moonshot-v1-auto retired 2026-08-31 (HTTP 404); kimi-k2.6 is the
        # cheapest live Kimi id (platform.kimi.ai/docs/models.md).
        "default_model": "kimi-k2.6",
    },
    "kimi-coding": {
        "endpoint": "https://api.kimi.com/coding/v1/chat/completions",
        "keychain": "grip-kimi-coding",
        "env": "KIMI_CODING_API_KEY",
        "auth": "bearer",
        # Kimi Code membership (GRIP Pool D, seal e49c00c214c500be). k3-256k is
        # the same K3 inside 256k at about half the quota of k3 (vendor table).
        # UNARMED until the credential exists; whether model_quirks' locked
        # params cover the bare k3 / kimi-for-coding ids on this host is
        # unverified (HAL#550 row C).
        "default_model": "k3-256k",
    },
    "ollama": {
        "endpoint": "http://localhost:11434/v1/chat/completions",
        "keychain": None,
        "env": None,
        "auth": "none",
        "default_model": "llama3.2",
    },
    "qwen": {
        "endpoint": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        "keychain": "hal-dashscope",
        "env": "DASHSCOPE_API_KEY",
        "auth": "bearer",
        "default_model": "qwen3.5-flash",
    },
    "qwen-token-plan": {
        # Alibaba Model Studio Token Plan Personal Edition (subscription).
        # The installer OBSCURES the API key (o1_ prefix); the DECODED key is
        # what keychain:grip-qwen-token-plan holds. Same OpenAI-compatible wire
        # as the qwen family, but a different host and a subscription auth.
        "endpoint": "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1/chat/completions",
        "keychain": "grip-qwen-token-plan",
        "env": "QWEN_TOKEN_PLAN_API_KEY",
        "auth": "bearer",
        "default_model": "qwen3.8-max",
    },
    "zhipu": {
        "endpoint": "https://api.z.ai/api/coding/paas/v4/chat/completions",
        "keychain": "grip-zhipu",
        "env": "ZHIPU_API_KEY",
        "auth": "bearer",
        # glm-5.3, not glm-5.1 (GRIP#9431, 2026-11-24): Zhipu's GLM Coding Plan has
        # answered every `glm-5.1` request with GLM-5.3 since at least 2026-11-24
        # (two probes, each beside a glm-5.3 control, both HTTP 200 with the reply's
        # `model` field reading glm-5.3). Naming the model that really answers.
        "default_model": "glm-5.3",
    },
    "xai": {
        "endpoint": "https://api.x.ai/v1/chat/completions",
        "keychain": "hal-xai",
        "env": "XAI_API_KEY",
        "auth": "bearer",
        "default_model": "grok-4.6",   # grok-3 unserved since the 2026-09-15 probe
    },
    "soruxgpt": {
        "endpoint": "https://service.soruxgpt.com/api/api/chat/completions",
        "keychain": "grip-soruxgpt",
        "env": "SORUXGPT_API_KEY",
        "auth": "bearer",
        "default_model": "gpt-4o-mini",
    },
    "vulavula": {
        "endpoint": "https://api.lelapa.ai/v1/chat/completions",
        "keychain": "grip-vulavula",
        "env": "VULAVULA_API_KEY",
        "auth": "bearer",
        "default_model": "translate",
    },
    "lemonade": {
        "endpoint": "http://localhost:7860/v1/chat/completions",
        "keychain": None,
        "env": None,
        "auth": "none",
        "default_model": "Llama-3.1-8B-Instruct",
    },
    "vllm": {
        "endpoint": "http://localhost:8000/v1/chat/completions",
        "keychain": None,
        "env": None,
        "auth": "none",
        "default_model": "meta-llama/Meta-Llama-3-8B-Instruct",
    },
    "lmstudio": {
        "endpoint": "http://localhost:1234/v1/chat/completions",
        "keychain": None,
        "env": None,
        "auth": "none",
        "default_model": "local-model",
    },
    "mlx_vlm": {
        "endpoint": "http://localhost:8080/v1/chat/completions",
        "keychain": None,
        "env": None,
        "auth": "none",
        "default_model": "mlx-community/gemma-3-4b-it-4bit",
    },
    "bitnet": {
        "endpoint": "http://127.0.0.1:8114/v1/chat/completions",
        "keychain": None,
        "env": None,
        "auth": "none",
        "default_model": "bitnet-b1.58-2b4t",
    },
}

SUPPORTED_PROVIDERS = frozenset(PROVIDER_CONFIGS.keys())
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT_S = 120

REDACTION_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9_-]{20,}|gsk_[A-Za-z0-9_-]{20,}|Bearer [A-Za-z0-9._-]+)"
)


def _redact(text: str) -> str:
    return REDACTION_PATTERN.sub("[REDACTED]", text)


def _fetch_keychain_token(service: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-w", "-s", service],
            capture_output=True, text=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _resolve_apikey_source(source: str) -> Optional[str]:
    if not source:
        return None
    if source.startswith("env:"):
        return os.environ.get(source[4:])
    if source.startswith("keychain:"):
        return _fetch_keychain_token(source[len("keychain:"):])
    return source


def _parse_provider_spec(spec: str) -> tuple[str, str]:
    """Split 'family/model' into (family, model). Returns ('', '') if empty."""
    if not spec:
        return "", ""
    if "/" not in spec:
        return spec.lower(), ""
    family, model = spec.split("/", 1)
    return family.lower(), model


def _resolve_model(family: str, provided_model: str) -> str:
    if provided_model:
        return provided_model
    return PROVIDER_CONFIGS[family]["default_model"]


def _resolve_token(family: str, envelope: HappiEnvelope) -> Optional[str]:
    cfg = PROVIDER_CONFIGS[family]
    if cfg["auth"] == "none":
        return ""  # Ollama: no auth required, empty sentinel
    scheme = (envelope.auth.get("scheme") or "subscription").lower()
    if scheme == "apikey":
        return _resolve_apikey_source(str(envelope.auth.get("token", "")))
    if cfg["env"]:
        token = os.environ.get(cfg["env"])
        if token:
            return token
    if cfg["keychain"]:
        return _fetch_keychain_token(cfg["keychain"])
    return None


def _build_messages(envelope: HappiEnvelope) -> list[dict[str, Any]]:
    """``[system?] + flags.messages (verbatim) + [current user turn]``.

    Canonical prompt resolution (content > flags.message > args). PARAMOUNT:
    this adapter previously ignored envelope.content, so an envelope carrying
    its prompt solely in `content` (GRIP council / AGORA dispatch) reached
    every OpenAI-compat provider with an EMPTY user message -- the model then
    free-associated and AGORA's validity gate failed the whole review
    (CodeTonight-SA/GRIP#2697). HappiEnvelope.user_message is the single
    source of truth; do not re-implement the precedence here.

    ``flags.messages`` is copied shallowly and otherwise untouched, so an
    assistant message's ``reasoning_content`` / ``tool_calls`` and a tool
    message's ``tool_call_id`` ride through exactly as the caller built them.
    The current user turn is appended when it is non-empty, or when there is
    no history at all (the pre-W1b body for a prompt-less envelope carried an
    empty user message; that byte shape is kept).
    """
    messages: list[dict[str, Any]] = []
    system_prompt = envelope.flags.get("system")
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt)})
    history = envelope.flags.get("messages") or []
    messages.extend(dict(m) for m in history)
    user_message = envelope.user_message()
    if user_message or not history:
        messages.append({"role": "user", "content": str(user_message)})
    return messages


def _apply_tool_flags(body: dict[str, Any], flags: dict[str, Any]) -> None:
    """``flags.tools`` / ``flags.tool_choice`` -> wire fields. Absent flags add
    nothing, so every pre-W1b body is byte-identical. An EMPTY ``tools`` list is
    also not sent: OpenAI rejects ``tools: []`` with HTTP 400 ("Invalid 'tools':
    empty array. Expected an array with minimum length 1"), so the truthiness
    check is the intended behaviour, not a verbatim pass-through (HAL#551
    review, concern B1)."""
    tools = flags.get("tools")
    if tools:
        body["tools"] = list(tools)
    if flags.get("tool_choice") is not None:
        body["tool_choice"] = flags["tool_choice"]


def _build_request_body(model: str, envelope: HappiEnvelope) -> dict[str, Any]:
    max_tokens = int(envelope.flags.get("max_tokens") or DEFAULT_MAX_TOKENS)
    body: dict[str, Any] = {
        "model": model,
        "messages": _build_messages(envelope),
        "max_tokens": max_tokens,
        "stream": True,
    }
    # Optional sampling controls (omitted unless the caller sets them, so the
    # body is byte-identical to prior behaviour by default). The OpenAI
    # chat-completions wire names (temperature/top_p/seed) match the canonical
    # keys, so the resolved dict merges directly. Unknown-field-tolerant compat
    # providers ignore any they don't support.
    body.update(resolve_sampling(envelope.flags, allow_seed=True))
    _apply_tool_flags(body, envelope.flags)
    _fix_request_body(body, envelope.flags)
    return body


def _is_message_list(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(m, dict) and isinstance(m.get("role"), str) for m in value
    )


def _bad_tool_calls(history: list[dict[str, Any]]) -> Optional[str]:
    """Why a history message's ``tool_calls`` cannot be echoed, else ``None``.

    ``_echo_tool_results`` and ``_this_turn_tool_messages`` read
    ``tool_calls[].id`` and ``tool_calls[].function.name``; a value that is not
    a list of dicts (with a dict ``function`` when present) used to reach them
    and raise an uncaught AttributeError instead of the documented
    ``invalid_messages`` refusal (HAL#551 review, concern B2)."""
    for i, msg in enumerate(history):
        calls = msg.get("tool_calls")
        if calls is None:
            continue
        if not isinstance(calls, list) or not all(isinstance(c, dict) for c in calls):
            return f"messages[{i}].tool_calls must be a list of tool-call objects"
        for j, call in enumerate(calls):
            if "function" in call and not isinstance(call["function"], dict):
                return f"messages[{i}].tool_calls[{j}].function must be an object"
    return None


def _tool_flag_error(model: str, flags: dict[str, Any]) -> Optional[tuple[str, str]]:
    """``(code, plain-English message)`` when the tool flags cannot be sent
    as given, else ``None``. Runs before any network call so the caller gets
    the refusal instead of a round trip that can only fail."""
    tools = flags.get("tools")
    if tools is not None and not (
        isinstance(tools, list) and all(isinstance(t, dict) for t in tools)
    ):
        return "invalid_tools", (
            "flags.tools must be a list of OpenAI tool schema objects, each "
            '{"type": "function", "function": {"name": ..., "parameters": {...}}}; '
            f"got {type(tools).__name__}"
        )
    history = flags.get("messages")
    if history is not None and not _is_message_list(history):
        return "invalid_messages", (
            "flags.messages must be a list of chat messages, each an object with "
            "a string 'role' (system / user / assistant / tool), echoed to the "
            "provider as-is"
        )
    if flags.get("tool_choice") == "required" and not supports_required_tool_choice(model):
        return "invalid_tool_choice", (
            f"tool_choice 'required' cannot be sent to {model}: Moonshot documents "
            "that kimi-k2.6 and kimi-k2.7-code return an error when it is passed "
            "(platform.kimi.ai/docs/api/models-overview.md; live 2026-09-07 the "
            "API answered HTTP 400 \"tool_choice 'required' is incompatible with "
            "thinking enabled\"). Use 'auto' or 'none', or switch to kimi-k3, "
            "which supports 'required'."
        )
    if history is not None:
        why = _bad_tool_calls(history)
        if why is not None:
            return "invalid_messages", (
                f"{why}: each entry is the assistant's tool_calls[] element as the "
                'provider returned it ({"id": ..., "type": "function", "function": '
                '{"name": ..., "arguments": ...}}), echoed verbatim'
            )
    return None


def _this_turn_tool_messages(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The trailing run of ``role: "tool"`` messages -- the results the caller
    is handing back for the ``tool_call`` events the previous turn emitted.
    Anything before a later assistant message was already reported by an
    earlier envelope's stream and is not repeated."""
    tail: list[dict[str, Any]] = []
    for msg in reversed(history):
        if msg.get("role") != "tool":
            break
        tail.append(msg)
    tail.reverse()
    return tail


def _echo_tool_results(envelope: HappiEnvelope) -> Iterator[str]:
    """One ``tool_result`` event per tool message in this turn's history, so
    the NDJSON stream alone records call -> result -> answer. ``name`` is
    recovered from the assistant's ``tool_calls`` when the tool message omits
    it (OpenAI proper does not require it; Moonshot's guide sends it)."""
    history = envelope.flags.get("messages") or []
    names = {
        str(call.get("id") or ""): str((call.get("function") or {}).get("name") or "")
        for msg in history for call in (msg.get("tool_calls") or [])
    }
    for msg in _this_turn_tool_messages(history):
        call_id = str(msg.get("tool_call_id") or "")
        name = str(msg.get("name") or names.get(call_id, ""))
        yield emit_tool_result(envelope.id, name, msg.get("content"), call_id)


def _fix_request_body(
    body: dict[str, Any], flags: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Apply model-aware OpenAI-compatible request fixes in place.

    Runs AFTER ``resolve_sampling`` so it sees the caller's knobs and can
    remove the ones a model refuses. Every rule is read from
    ``lib.hal.model_quirks`` (single home; the gateway and the direct adapter
    consume the same table), never decided here:

    * ``max_tokens`` -> ``max_completion_tokens`` where the model rejects
      (OpenAI gpt-5.x, HTTP 400) or deprecates (Moonshot) the legacy field;
    * the Kimi lineage's locked sampling fields are dropped -- Moonshot
      answers HTTP 400 "invalid temperature: only 1 is allowed for this
      model", which is exactly what AGORA's ``temperature: 0.0`` hit
      through this adapter until 2026-09-07;
    * the reasoning knob: kimi-k3 gets a top-level ``reasoning_effort``
      (``flags.effort`` if given, else the vendor default, held constant so a
      session never invalidates its prefix cache); kimi-k2.6 gets a
      ``thinking`` block only when ``flags.thinking`` asks for one; on
      kimi-k2.7-code thinking cannot be turned off, so nothing is sent.

    Any other provider's body is byte-identical to the pre-W1 shape, even
    when the envelope carries ``effort`` / ``thinking``.
    """
    flags = flags or {}
    model = str(body.get("model") or "")
    if uses_max_completion_tokens(model) and "max_tokens" in body:
        legacy = body.pop("max_tokens")
        body.setdefault("max_completion_tokens", legacy)
    for locked in locked_sampling_params(model):
        body.pop(locked, None)
    _apply_reasoning_control(body, flags, reasoning_control(model))
    return body


def _apply_reasoning_control(
    body: dict[str, Any], flags: dict[str, Any], control: dict[str, Any]
) -> None:
    """Write the model's reasoning knob into ``body`` per ``control``.

    ``reasoning_effort`` is passed through as the caller spelt it (lowercased);
    an unknown level is NOT second-guessed here -- the provider returns its own
    clear 400 (the ``happi/sampling.py`` policy). A forced-on ``thinking`` is
    never sent, whatever the caller asked.
    """
    kind = control.get("kind")
    if kind == "reasoning_effort":
        effort = str(flags.get("effort") or "").strip().lower()
        value = effort or control["default"]
        # A model with no forced default (gpt-6) and no caller level gets no
        # field at all -- never a JSON null; a level the model refuses (gpt-6
        # 400s on ``none``) is dropped so the vendor default applies.
        if value and value not in refused_reasoning_efforts(body.get("model") or ""):
            body["reasoning_effort"] = value
    elif kind == "thinking" and not control.get("forced"):
        block = _thinking_block(flags.get("thinking"))
        if block is not None:
            body["thinking"] = block


def _thinking_block(flag: Any) -> Optional[dict[str, str]]:
    """``flags.thinking`` -> Moonshot's ``{"type": "enabled"|"disabled"}``.

    Accepts a bool, the literal strings, or a dict already in wire shape;
    anything else (including absence) means "do not send" -- the vendor
    default applies, byte-identical to the pre-W1 body.
    """
    if isinstance(flag, bool):
        return {"type": "enabled" if flag else "disabled"}
    if isinstance(flag, str) and flag.strip().lower() in ("enabled", "disabled"):
        return {"type": flag.strip().lower()}
    if isinstance(flag, dict) and flag.get("type") in ("enabled", "disabled"):
        return {"type": str(flag["type"])}
    return None


def _open_stream(family: str, body: dict[str, Any], token: str, timeout: int) -> Any:
    cfg = PROVIDER_CONFIGS[family]
    require_openai_destination(cfg["endpoint"])
    headers = {
        "content-type": "application/json",
        "accept": "text/event-stream",
        # Without this, urllib sends its default "Python-urllib/3.x" UA, which
        # some provider CDNs bot-flag. Defensive hardening mirroring
        # lib/hal/providers/openai_compat.py ("HAL/1.0") and GRIP's
        # sovereign_council UA lesson (Groq CDN 403s flagged requests).
        "user-agent": "HAL/1.0",
    }
    if cfg["auth"] == "bearer" and token:
        headers["authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        cfg["endpoint"],
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers=headers,
    )
    return urllib.request.urlopen(req, timeout=timeout)


def _iter_sse_data(response: Any) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON objects from an OpenAI-style SSE stream.

    Format: `data: {...}\\n\\n` per event; `data: [DONE]` terminates. No
    named `event:` lines (unlike Anthropic's SSE dialect).
    """
    for raw in response:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            return
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


def _yield_auth_failure(envelope: HappiEnvelope, family: str) -> Iterator[str]:
    yield emit_error(
        envelope.id,
        f"no token available for provider={family!r}",
        code="missing_credentials",
    )
    yield emit_completed(envelope.id, usage={})


def _yield_http_failure(envelope: HappiEnvelope, exc: urllib.error.HTTPError) -> Iterator[str]:
    detail = ""
    if hasattr(exc, "read"):
        try:
            detail = _redact(exc.read().decode("utf-8", errors="replace"))
        except Exception:
            detail = ""
    yield emit_error(
        envelope.id,
        f"HTTP {exc.code}: {detail[:500]}",
        code=f"provider_http_{exc.code}",
    )
    yield emit_completed(envelope.id, usage={})


def _yield_network_failure(envelope: HappiEnvelope, exc: urllib.error.URLError) -> Iterator[str]:
    yield emit_error(
        envelope.id,
        f"network error: {_redact(str(exc.reason))}",
        code="provider_network",
    )
    yield emit_completed(envelope.id, usage={})


def _yield_invalid_request(envelope: HappiEnvelope, code: str, message: str) -> Iterator[str]:
    yield emit_error(envelope.id, message, code=code)
    yield emit_completed(envelope.id, usage={})


def _yield_unsupported(envelope: HappiEnvelope, family: str) -> Iterator[str]:
    yield emit_error(
        envelope.id,
        f"provider {family!r} not in OpenAI-compat family "
        f"(supported: {sorted(SUPPORTED_PROVIDERS)})",
        code="unsupported_provider",
    )
    yield emit_completed(envelope.id, usage={})


def _first_choice(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices") or []
    return (choices[0] or {}) if choices else {}


def _extract_usage(data: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Token usage from a chunk: OpenAI puts ``usage`` at the top level of the
    final chunk; Moonshot puts it inside the finishing choice
    (``choices[0].usage``, seen live on kimi-k3 / kimi-k2.6, 2026-09-07).
    Top level wins when both are present; a chunk with neither leaves
    ``current`` untouched."""
    block = data.get("usage") or _first_choice(data).get("usage")
    if not block:
        return current
    return {
        "in": block.get("prompt_tokens", current.get("in", 0)),
        "out": block.get("completion_tokens", current.get("out", 0)),
    }


def _merge_tool_call_fragment(calls: dict[int, dict[str, str]], fragment: dict[str, Any]) -> None:
    """Fold one ``delta.tool_calls[]`` fragment into ``calls`` by ``index``.

    The vendor's streaming contract (platform.kimi.ai/docs/guide/
    use-kimi-api-to-complete-tool-calls.md): "we will specify the
    `tool_call.id` and `tool_call.function.name` in the initial data chunk,
    and only `tool_call.function.arguments` will be output in subsequent
    chunks"; with several calls "we will use an additional field called
    `index` ... so that you can correctly concatenate the
    `tool_call.function.arguments`". Same shape as OpenAI's own stream.
    """
    call = calls.setdefault(
        int(fragment.get("index") or 0),
        {"id": "", "type": "function", "name": "", "arguments": ""},
    )
    function = fragment.get("function") or {}
    if fragment.get("id"):
        call["id"] = str(fragment["id"])
    if fragment.get("type"):
        call["type"] = str(fragment["type"])
    if function.get("name"):
        call["name"] = str(function["name"])
    call["arguments"] += str(function.get("arguments") or "")


def _decode_arguments(raw: str) -> Any:
    """``function.arguments`` (a JSON string on the wire) -> the value for
    ``tool_call.input``. Empty means no arguments. A string that does not
    parse (e.g. cut off at the token budget) is returned as-is rather than
    dropped, so the caller sees exactly what the model produced."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


class _StreamState:
    """One chat-completions stream so far: the assistant message being
    assembled (the vendor's own recipe -- concatenate ``content`` and
    ``reasoning_content``, expand ``tool_calls`` by index), the provider's
    ``finish_reason`` and its usage block."""

    def __init__(self) -> None:
        self.content: list[str] = []
        self.reasoning: list[str] = []
        self.calls: dict[int, dict[str, str]] = {}
        self.finish: Optional[str] = None
        self.usage: dict[str, Any] = {}

    def ordered_calls(self) -> list[dict[str, str]]:
        return [self.calls[i] for i in sorted(self.calls)]

    def message(self) -> dict[str, Any]:
        """The assistant message exactly as the vendor asks to have it back:
        ``reasoning_content`` only when the model reasoned, ``tool_calls``
        only when it asked for a tool, ``arguments`` as the raw string."""
        msg: dict[str, Any] = {"role": "assistant", "content": "".join(self.content)}
        if self.reasoning:
            msg["reasoning_content"] = "".join(self.reasoning)
        if self.calls:
            msg["tool_calls"] = [
                {"id": c["id"], "type": c["type"],
                 "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in self.ordered_calls()
            ]
        return msg

    def needs_echo(self) -> bool:
        return bool(self.reasoning or self.calls)


def _translate_chunk(envelope_id: str, data: dict[str, Any], state: _StreamState) -> Iterator[str]:
    """Events for one SSE chunk, folding it into ``state``. Order on the wire
    (vendor guide): ``reasoning_content`` first, then ``content``, then
    ``tool_calls``; ``finish_reason`` only on the final chunk."""
    choice = _first_choice(data)
    delta = choice.get("delta") or {}
    reasoning = delta.get("reasoning_content")
    if reasoning:
        state.reasoning.append(reasoning)
        yield emit_thinking(envelope_id, reasoning)
    text = delta.get("content")
    if text:
        state.content.append(text)
        yield emit_delta(envelope_id, text)
    for fragment in delta.get("tool_calls") or []:
        _merge_tool_call_fragment(state.calls, fragment)
    if choice.get("finish_reason"):
        state.finish = str(choice["finish_reason"])
    state.usage = _extract_usage(data, state.usage)


def _terminate(envelope_id: str, state: _StreamState) -> Iterator[str]:
    """``tool_call`` events (only when the model finished asking:
    ``finish_reason == "tool_calls"``), then ``completed``. A plain text turn
    keeps the pre-W1b ``completed`` shape; a turn that reasoned or asked for
    a tool adds ``stop_reason`` + ``message`` -- the material the caller must
    pass back verbatim on the next envelope."""
    if state.finish == "tool_calls":
        for call in state.ordered_calls():
            yield emit_tool_call(
                envelope_id, call["name"], _decode_arguments(call["arguments"]), call["id"]
            )
    extra: dict[str, Any] = {}
    if state.needs_echo():
        if state.finish:
            extra["stop_reason"] = state.finish
        extra["message"] = state.message()
    yield emit_completed(envelope_id, usage=state.usage, **extra)


def _translate_and_terminate(envelope: HappiEnvelope, response: Any) -> Iterator[str]:
    state = _StreamState()
    for data in _iter_sse_data(response):
        err = data.get("error")
        if err:
            yield emit_error(
                envelope.id,
                _redact(str(err.get("message", "unknown provider error"))),
                code=f"provider_error:{err.get('type', 'unknown')}",
            )
            break
        yield from _translate_chunk(envelope.id, data, state)
    yield from _terminate(envelope.id, state)


def stream_openai_compat(envelope: HappiEnvelope) -> Iterator[str]:
    """Main entry: translate one envelope into a stream of HAPPI NDJSON events."""
    family, provided_model = _parse_provider_spec(
        str(envelope.flags.get("provider") or "")
    )
    if family not in SUPPORTED_PROVIDERS:
        yield from _yield_unsupported(envelope, family)
        return

    token = _resolve_token(family, envelope)
    if token is None:
        yield from _yield_auth_failure(envelope, family)
        return

    model = _resolve_model(family, provided_model)
    problem = _tool_flag_error(model, envelope.flags)
    if problem:
        yield from _yield_invalid_request(envelope, *problem)
        return
    body = _build_request_body(model, envelope)
    timeout = int(envelope.flags.get("timeout_s") or DEFAULT_TIMEOUT_S)
    yield from _echo_tool_results(envelope)

    try:
        response = _open_stream(family, body, token, timeout)
    except SpendRefused as exc:
        yield emit_error(envelope.id, exc.decision.reason, code="spend_refused")
        return
    except urllib.error.HTTPError as exc:
        yield from _yield_http_failure(envelope, exc)
        return
    except urllib.error.URLError as exc:
        yield from _yield_network_failure(envelope, exc)
        return

    try:
        yield from _translate_and_terminate(envelope, response)
    finally:
        try:
            response.close()
        except Exception:
            pass
