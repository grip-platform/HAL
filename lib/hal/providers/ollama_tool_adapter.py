"""Ollama tool-loop adapter with dynamic num_ctx escalation (#1228, W4).

The adapter wraps :func:`~lib.hal.tool_runtime.make_openai_compat_chat_fn`
behind a pre-call check that inspects the loaded Ollama model's current
``num_ctx`` and, if the incoming prompt would exceed the safety band,
forces Ollama to reload the model at a larger context window BEFORE
issuing the real request.

**PARAMOUNT invariant — preserve ``keep_alive``.** Ollama's default
reload resets idle-eviction to 5 minutes. HAL sessions pin models with
long keep_alive values (typically 30m or more) to avoid the 20-40s
cold-load cost between tool-loop turns. This adapter reads the current
``expires_at`` timestamp from ``GET /api/ps`` and passes it back to
Ollama verbatim on reload so the TTL is continuous across num_ctx
changes. Resetting keep_alive to Ollama's 5-minute default is a hard
fail — the adapter never does this.

The adapter is stdlib-only to keep HAL free of third-party HTTP
dependencies. Policy decisions live in :mod:`num_ctx_policy`; this
module owns only the HTTP reload dance and the macOS VRAM guard.

Usage::

    from lib.hal.providers.ollama_tool_adapter import OllamaToolAdapter
    adapter = OllamaToolAdapter(
        base_url="http://localhost:11434/v1",
        model="gemma4:e4b",
        max_tokens=4096,
    )
    response = adapter.chat_fn(messages=msgs, tools=tools)
"""
from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from typing import Callable, Optional

from lib.hal.providers.num_ctx_policy import (
    OllamaCtxError,
    _CTX_TIERS,
    estimate_token_count,
    fetch_arch_max,
    fetch_loaded_ctx,
    select_num_ctx,
)

_DEFAULT_OLLAMA_HOST = "http://localhost:11434"
_HTTP_TIMEOUT_S = 60.0  # Reload can take 20-40s (model load time).
_BYTES_PER_MB = 1024 * 1024
_PAGE_SIZE_BYTES = 4096  # macOS vm_stat default page size.
# KV-cache cost per extra token. Calibrated against ``ollama ps`` on
# ``gemma4:e4b`` (GQA — reduced K,V head dims). A 12K-token enlargement
# adds ~300 MB which matches 25 KB × 12288 ≈ 300 MB. See PR #174 audit.
_VRAM_COST_PER_TOKEN_KB = 25


def _free_vram_mb() -> Optional[int]:
    """Return a rough free-memory estimate in MiB, or ``None`` on any error.

    On macOS with Apple Silicon, VRAM is unified system memory — ``vm_stat``
    reports ``free`` + ``inactive`` pages that the kernel can hand back
    to a new allocation without swap. Returning ``None`` when ``vm_stat``
    is missing or unparseable means the VRAM guard fails open (allows the
    reload) rather than closed (blocks valid reloads on non-macOS hosts).

    Not cached — a reload decision is rare enough that a 10ms subprocess
    call is noise.
    """
    try:
        result = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=2.0,
        )
        if result.returncode != 0:
            return None
        free_pages = 0
        for line in result.stdout.splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key_lower = key.strip().lower()
            if key_lower in ("pages free", "pages inactive"):
                free_pages += int(value.strip().rstrip(".") or 0)
        return (free_pages * _PAGE_SIZE_BYTES) // _BYTES_PER_MB
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


def _estimate_reload_cost_mb(old_ctx: int, new_ctx: int) -> int:
    """Return a rough MB estimate for a ctx enlargement.

    Transformer KV-cache scales roughly linearly with ctx size. We budget
    ``_VRAM_COST_PER_TOKEN_KB`` (~25 KB) per extra token — calibrated
    against live ``ollama ps`` measurements on ``gemma4:e4b`` (GQA
    architecture reduces K,V head dims substantially). An earlier draft
    used 500 KB/token which false-positive-blocked every realistic reload
    under transient memory pressure; the calibrated value leaves headroom
    for quantisation variance without gating valid reloads.
    """
    if new_ctx <= old_ctx:
        return 0
    return ((new_ctx - old_ctx) * _VRAM_COST_PER_TOKEN_KB) // 1024  # KB/1024 = MB


def _default_vram_preflight(old_ctx: int, new_ctx: int) -> bool:
    """Return True if the reload probably fits in free memory.

    Fails open on non-macOS / unparseable vm_stat: returning True means
    "trust Ollama's own eviction logic" rather than preemptively blocking.
    Fails closed only when vm_stat gave a concrete free-memory number AND
    that number is less than the estimated reload cost.
    """
    free_mb = _free_vram_mb()
    if free_mb is None:
        return True
    cost_mb = _estimate_reload_cost_mb(old_ctx, new_ctx)
    return free_mb >= cost_mb


def _post_json(url: str, body: dict, timeout_s: float = _HTTP_TIMEOUT_S) -> dict:
    """POST JSON to Ollama, returning the parsed response or ``{}`` on error.

    Ollama's reload endpoint (``/api/generate`` with empty prompt) returns
    a streaming JSON response when called normally, but a single JSON
    object when ``stream=false`` (or when the prompt is empty — the
    server short-circuits). Failure modes degrade to empty-dict so the
    caller decides whether to retry or surface the error.
    """
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url=url, data=payload, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return {}


def reload_with_num_ctx(
    *,
    model: str,
    new_ctx: int,
    keep_alive: Optional[str],
    ollama_host: str = _DEFAULT_OLLAMA_HOST,
) -> bool:
    """Force Ollama to reload ``model`` at ``new_ctx`` preserving ``keep_alive``.

    Posts to ``/api/generate`` with an empty prompt and ``options.num_ctx``.
    Ollama notices that options differ from the currently-loaded model and
    performs a full reload at the new ctx. The ``keep_alive`` parameter
    (either a preserved ISO-8601 ``expires_at`` string or ``None`` to let
    Ollama apply its default) passes through verbatim.

    PARAMOUNT: if ``keep_alive`` is ``None`` the caller accepts Ollama's
    5-minute default. Callers that need the long-pin behaviour MUST pass
    the pre-reload ``expires_at``. The adapter's ``chat_fn`` does this;
    direct callers of this function must do so explicitly.

    Returns ``True`` on HTTP 200 with a non-empty response body.
    """
    body: dict = {
        "model": model,
        "prompt": "",
        "stream": False,
        "options": {"num_ctx": new_ctx},
    }
    if keep_alive is not None:
        body["keep_alive"] = keep_alive
    response = _post_json(f"{ollama_host}/api/generate", body)
    return bool(response)


class OllamaToolAdapter:
    """Tool-loop chat_fn that escalates num_ctx before each request.

    Composition:

    1. At construction, delegate to ``make_openai_compat_chat_fn`` for the
       actual ``/v1/chat/completions`` request shape + Anthropic-shape
       normalisation.
    2. On each ``chat_fn`` call:
       a. Estimate incoming tokens via :func:`estimate_token_count`.
       b. Read currently-loaded ``num_ctx`` and ``expires_at`` from
          ``GET /api/ps``.
       c. Run :func:`select_num_ctx` with the architectural cap from
          ``POST /api/show``.
       d. If no escalation needed → call the inner chat_fn unchanged.
       e. Otherwise VRAM-preflight the reload; if OK,
          :func:`reload_with_num_ctx` passing through the preserved
          ``expires_at``; then call the inner chat_fn.

    This split keeps the decision layer unit-testable (pure) and the
    adapter integration-testable (all I/O lives here).
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "ollama",
        max_tokens: int = 4096,
        ollama_host: str = _DEFAULT_OLLAMA_HOST,
        vram_preflight: Optional[Callable[[int, int], bool]] = None,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._ollama_host = ollama_host
        self._vram_preflight = vram_preflight or _default_vram_preflight
        # Build inner chat_fn once at construction.
        from lib.hal.tool_runtime import make_openai_compat_chat_fn
        self._inner = make_openai_compat_chat_fn(
            base_url=base_url,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
        )

    def _plan_and_maybe_reload(self, messages: list[dict]) -> Optional[int]:
        """Escalation gate. Returns the new ctx when a reload was performed."""
        estimated = estimate_token_count(messages)
        current, keep_alive = fetch_loaded_ctx(
            self._model, ollama_host=self._ollama_host,
        )
        current_val = current or 0
        arch_max = fetch_arch_max(self._model, ollama_host=self._ollama_host)
        new_ctx = select_num_ctx(estimated, current_val, arch_max)
        if new_ctx == current_val:
            return None
        if not self._vram_preflight(current_val, new_ctx):
            raise OllamaCtxError(
                f"Cannot reload {self._model} at num_ctx={new_ctx}: "
                f"VRAM pre-flight failed (free memory insufficient for "
                f"{new_ctx - current_val} token enlargement)"
            )
        reload_with_num_ctx(
            model=self._model, new_ctx=new_ctx,
            keep_alive=keep_alive, ollama_host=self._ollama_host,
        )
        return new_ctx

    def chat_fn(self, *, messages: list[dict], tools: list[dict]) -> dict:
        """The ``run_tool_loop``-compatible closure surface."""
        self._plan_and_maybe_reload(messages)
        return self._inner(messages=messages, tools=tools)


def make_ollama_tool_chat_fn(
    *,
    base_url: str,
    model: str,
    api_key: str = "ollama",
    max_tokens: int = 4096,
    ollama_host: str = _DEFAULT_OLLAMA_HOST,
) -> Callable[..., dict]:
    """Build a ``chat_fn`` bound to a fresh :class:`OllamaToolAdapter`.

    Convenience for callers (including the REPL dispatch) that want a bare
    callable matching the :func:`run_tool_loop` contract rather than an
    object to hold on to. The adapter instance lives inside the closure,
    which is fine — there's no per-call state worth sharing across
    sessions, and garbage collection handles cleanup when the closure
    goes out of scope.
    """
    adapter = OllamaToolAdapter(
        base_url=base_url, model=model, api_key=api_key,
        max_tokens=max_tokens, ollama_host=ollama_host,
    )
    return adapter.chat_fn


__all__ = [
    "OllamaCtxError",
    "OllamaToolAdapter",
    "make_ollama_tool_chat_fn",
    "reload_with_num_ctx",
]
