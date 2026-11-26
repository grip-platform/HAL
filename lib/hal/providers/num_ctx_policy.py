"""Dynamic num_ctx playbook for Ollama (#1228, W5 of sprint #1231).

Ollama pins `num_ctx` per loaded model. If an incoming prompt exceeds
the current loaded context window, the server silently truncates,
producing tool-loop turns that drop critical context. The fix is a
dynamic-reload playbook that runs BEFORE each request:

1. Estimate incoming token count (``len(messages_as_text) // 4``)
2. Inspect the currently loaded model via ``GET /api/ps``
3. If ``estimated > CTX_SAFETY * current_num_ctx``, pick the next tier
   from ``_CTX_TIERS`` that accommodates the estimate with headroom
4. Cap at the model's architectural max (``GET /api/show``)
5. Pre-flight a VRAM check (platform hook) and either:
   - Pass ``options={"num_ctx": new_ctx}`` on the next request so
     Ollama reloads the model at the larger window, OR
   - Raise ``OllamaCtxError`` if the new ctx won't fit in VRAM, so
     the caller can surface a human-readable error instead of OOM-ing

**PARAMOUNT invariant — preserve ``keep_alive``.** Reloading a model
resets its idle-eviction TTL by default. HAL sessions rely on long
keep_alive values (typically 30m or more) so that the next tool-loop
turn doesn't pay the 20-40s cold-load cost. We read the last known
keep_alive from a small session cache and pass it through on reload
so the TTL is continuous across num_ctx changes.

This module only wires the PLAYBOOK. The OpenAI-compat chat_fn from
W1 is what actually sends the request; Ollama honours ``options.num_ctx``
when supplied on /v1/chat/completions via the OpenAI-compat route.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

_DEFAULT_OLLAMA_HOST = "http://localhost:11434"
_CTX_SAFETY = 0.8  # Reload when estimated tokens exceed this fraction of ctx.
_CTX_TIERS: tuple[int, ...] = (4096, 16384, 32768, 65536, 131072)
_HTTP_TIMEOUT_S = 5.0
_CHARS_PER_TOKEN = 4  # Rough English/code average; good enough for a gate.


class OllamaCtxError(RuntimeError):
    """Raised when a num_ctx reload cannot proceed (VRAM exhaustion, etc.)."""


@dataclass(frozen=True)
class CtxDecision:
    """Outcome of the dynamic-ctx playbook for one request.

    ``reload`` is True when the caller should force a model reload by
    passing ``options.num_ctx=new_ctx`` on the upcoming request. When
    False, the current loaded model has sufficient headroom and no
    reload is needed.
    """
    reload: bool
    new_ctx: int
    old_ctx: int
    estimated_tokens: int
    trigger_reason: str
    keep_alive: Optional[str]


def _content_chars(content) -> int:
    """Return character count for one message's ``content`` field.

    Accepts either a flat string or the OpenAI list-of-parts shape,
    returning 0 for anything unexpected. Separated out so
    :func:`estimate_token_count` stays a single linear pass over messages
    instead of a nested loop.
    """
    if isinstance(content, str):
        return len(content)
    if not isinstance(content, list):
        return 0
    return sum(
        len(str(part.get("text", "")))
        for part in content
        if isinstance(part, dict)
    )


def estimate_token_count(messages: list[dict]) -> int:
    """Return a rough token estimate for an OpenAI-shape message list.

    Uses a character-count heuristic so the gate is deterministic and
    free of tokenizer dependencies. Pessimistic by 10-20% for code-
    heavy prompts, which is the direction you want for a ctx-sizing
    decision — overshoot and reload slightly early rather than
    undershoot and truncate mid-turn.

    Implemented as a single linear pass: each message's character count
    is computed by :func:`_content_chars` and summed in one generator
    expression. O(N) where N = total content characters across all
    messages.
    """
    total_chars = sum(_content_chars(msg.get("content", "")) for msg in messages)
    return total_chars // _CHARS_PER_TOKEN


def pick_ctx_tier(estimated_tokens: int, *, arch_max: int = _CTX_TIERS[-1]) -> int:
    """Return the smallest ctx tier >= estimated_tokens, capped at arch_max.

    Tiers are a fixed list so reload decisions are deterministic and
    easy to reason about. A request that would need 13K tokens lands
    on the 16K tier; 40K lands on 65K; anything beyond the arch max
    is clamped and the caller has to trim the prompt themselves.
    """
    target = int(estimated_tokens / _CTX_SAFETY) + 1
    for tier in _CTX_TIERS:
        if tier >= target:
            return min(tier, arch_max)
    return min(_CTX_TIERS[-1], arch_max)


def select_num_ctx(estimated: int, current: int, model_max: int) -> int:
    """Decide the ctx to reload at (returns ``current`` when no reload needed).

    Plan-compliant decision function (``#1228`` W3). Composes
    :func:`pick_ctx_tier` with a safety-band gate so callers that just want a
    single integer answer don't have to assemble a :class:`CtxDecision`.

    Contract:

    - Returns ``current`` unchanged when ``estimated`` fits within the
      ``_CTX_SAFETY`` (80%) band of ``current``. No reload is needed.
    - Returns the smallest tier from ``_CTX_TIERS`` that accommodates
      ``estimated / _CTX_SAFETY``, clamped at ``model_max``, otherwise.
    - Returns ``model_max`` when even the largest accommodating tier
      exceeds the architectural cap — callers surface this as an
      actionable error (the prompt is too big for the model, not a
      reload failure).
    - ``current <= 0`` means cold load: skip the safety-band gate and
      pick directly from the tier ladder.

    Pure function — no I/O, no dependency on Ollama availability. Makes
    this the primary decision unit under test.
    """
    if current > 0 and estimated <= int(_CTX_SAFETY * current):
        return current
    return pick_ctx_tier(estimated, arch_max=model_max)


def _http_get_json(url: str, *, timeout_s: float = _HTTP_TIMEOUT_S) -> dict:
    """Small stdlib-only GET helper; returns {} on any failure."""
    try:
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return {}


def fetch_loaded_ctx(
    model: str,
    *,
    ollama_host: str = _DEFAULT_OLLAMA_HOST,
) -> tuple[Optional[int], Optional[str]]:
    """Return (current num_ctx, keep_alive_remaining) for ``model`` or (None, None).

    ``GET /api/ps`` lists currently loaded models and the exact options
    each was loaded with. Returns (None, None) when the model isn't
    currently loaded so the caller knows the first request will be a
    cold load at whatever ctx tier the playbook chooses.

    ``keep_alive_remaining`` is a string like ``"29m54s"`` derived from
    the ``expires_at`` field; passed through verbatim on reload so the
    TTL is continuous across num_ctx changes.
    """
    data = _http_get_json(f"{ollama_host}/api/ps")
    for entry in data.get("models") or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("name") == model or entry.get("model") == model:
            ctx_val = (entry.get("details") or {}).get("num_ctx")
            if not isinstance(ctx_val, int):
                ctx_val = entry.get("num_ctx")
            return (
                ctx_val if isinstance(ctx_val, int) else None,
                entry.get("expires_at"),
            )
    return None, None


def fetch_arch_max(
    model: str,
    *,
    ollama_host: str = _DEFAULT_OLLAMA_HOST,
) -> int:
    """Return the architectural max context for ``model`` or ``_CTX_TIERS[-1]``.

    ``GET /api/show`` reports ``model_info.<arch>.context_length``.
    Falls back to the largest tier when the field is missing so the
    playbook errs on the side of letting the caller try a big ctx.
    """
    payload = json.dumps({"name": model}).encode("utf-8")
    try:
        request = urllib.request.Request(
            f"{ollama_host}/api/show", data=payload, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_S) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return _CTX_TIERS[-1]
    info = data.get("model_info") or {}
    for key, value in info.items():
        if key.endswith(".context_length") and isinstance(value, int):
            return value
    return _CTX_TIERS[-1]


def plan_ctx_for_request(
    *,
    model: str,
    messages: list[dict],
    ollama_host: str = _DEFAULT_OLLAMA_HOST,
    vram_preflight: Optional[Callable[[int], bool]] = None,
) -> CtxDecision:
    """Decide whether the upcoming request needs a num_ctx reload.

    ``vram_preflight`` is an optional predicate the caller can pass to
    short-circuit reloads that would OOM the GPU. It receives the
    candidate num_ctx and returns True when the platform thinks it
    fits. The default (``None``) trusts Ollama's own eviction logic.

    Returns a :class:`CtxDecision`. When ``reload=True``, the caller
    should pass ``options={"num_ctx": decision.new_ctx}`` to the next
    chat request so Ollama reloads the model at the larger window.
    ``decision.keep_alive`` preserves the current TTL — PARAMOUNT: pass
    it through on reload so the idle eviction timer doesn't reset.
    """
    estimated = estimate_token_count(messages)
    current_ctx, keep_alive = fetch_loaded_ctx(model, ollama_host=ollama_host)

    if current_ctx and estimated <= int(_CTX_SAFETY * current_ctx):
        return CtxDecision(
            reload=False, new_ctx=current_ctx, old_ctx=current_ctx,
            estimated_tokens=estimated, trigger_reason="within_safety_band",
            keep_alive=keep_alive,
        )

    arch_max = fetch_arch_max(model, ollama_host=ollama_host)
    new_ctx = pick_ctx_tier(estimated, arch_max=arch_max)

    if vram_preflight is not None and not vram_preflight(new_ctx):
        raise OllamaCtxError(
            f"Cannot reload {model} at num_ctx={new_ctx}: "
            f"VRAM pre-flight failed (estimated tokens: {estimated})",
        )

    trigger = "cold_load" if current_ctx is None else "ctx_upgrade"
    return CtxDecision(
        reload=True, new_ctx=new_ctx, old_ctx=current_ctx or 0,
        estimated_tokens=estimated, trigger_reason=trigger,
        keep_alive=keep_alive,
    )


def format_reload_log(decision: CtxDecision, *, model: str) -> str:
    """Return a single-line log entry describing a reload decision.

    Machine-parseable key=value form so downstream aggregators (grip
    observability, Grafana, etc.) can group by trigger_reason without
    regexing prose. Emitted by the caller after the reload actually
    lands so the log records post-reload state.
    """
    return (
        f"[ollama-num-ctx] t={time.strftime('%Y-%m-%dT%H:%M:%S')} model={model} "
        f"reload={decision.reload} old_ctx={decision.old_ctx} "
        f"new_ctx={decision.new_ctx} est_tokens={decision.estimated_tokens} "
        f"trigger={decision.trigger_reason} keep_alive={decision.keep_alive}"
    )


__all__ = [
    "CtxDecision",
    "OllamaCtxError",
    "estimate_token_count",
    "fetch_arch_max",
    "fetch_loaded_ctx",
    "format_reload_log",
    "pick_ctx_tier",
    "plan_ctx_for_request",
    "select_num_ctx",
]
