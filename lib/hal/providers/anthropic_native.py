"""Anthropic native Messages API provider.

Direct access to Anthropic's Messages API for features not available
via OpenAI-compatible format: extended thinking, prompt caching,
tool use with native schema, and streaming.

Uses stdlib urllib — no anthropic SDK dependency.

Two economic paths coexist:

* ``api_key=`` → per-token Messages API billing (classic ``sk-ant-...`` key)
* ``subscription=`` → flat-fee CC subscription via CCH header signing
  (:class:`~lib.hal.subscription.CCHSubscription`)

Both can be configured simultaneously. The ``prefer`` argument chooses per
call: ``"auto"`` uses the subscription first and falls back to the API key
when the subscription is unhealthy; ``"api_key"`` and ``"subscription"``
force a specific path. Every API-key call asks the spend guard
(``lib/hal/spend_guard.py``) first, so the fallback and the forced path alike
raise :class:`AnthropicSpendRefused` (an :class:`AnthropicError`) instead of
billing a key the operator has not opted in, or one over today's daily cap.
Every API-key Messages request carries the prompt-cache markers, and every
response is priced and added to the day (``lib/hal/anthropic_key_route.py``).

Usage:
    # API key only
    provider = AnthropicProvider(api_key="sk-ant-...")

    # Subscription only
    from lib.hal.subscription import CCHSubscription
    provider = AnthropicProvider(subscription=CCHSubscription.from_env())

    # Both — provider prefers subscription, falls back to API key
    provider = AnthropicProvider(
        api_key="sk-ant-...",
        subscription=CCHSubscription.from_env(),
    )
    response = provider.chat("Explain SOLID", model="claude-sonnet-4-6")
"""
import json
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from typing import Optional

from lib.hal import spend_guard
from lib.hal.prompt_cache import apply_cache_control
from lib.hal.subscription import SubscriptionAuth
from lib.hal.types import ChatMessage, ChatResponse, ModelInfo, TokenUsage

_KEYCHAIN_CMD = ["security", "find-generic-password", "-w", "-s"]
#: The keychain item that holds HAL's own Anthropic API key.
API_KEY_SERVICE = "hal-anthropic"
_API_BASE = "https://api.anthropic.com/v1"
_DEFAULT_MODEL = "claude-opus-4-7"
_API_VERSION = "2023-06-01"
# Server-side context compaction beta (verified-canonical, CC bundle SDK docs).
# Opting into context_management auto-appends this to the anthropic-beta header.
_COMPACT_BETA = "compact-2026-01-12"
_MAX_RETRIES = 3
_INITIAL_BACKOFF_S = 1.0
_BACKOFF_MULTIPLIER = 2.0
_KEY_PATTERN = re.compile(r"(sk-ant-|sk-|Bearer )\S+")


def _fetch_keychain_key(service: str) -> Optional[str]:
    try:
        result = subprocess.run(
            [*_KEYCHAIN_CMD, service],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


class AnthropicError(Exception):
    """Non-retryable Anthropic API error."""


class AnthropicRateLimitError(AnthropicError):
    """Rate limit hit after exhausting retries."""


class AnthropicSpendRefused(AnthropicError):
    """The spend guard refused the API key before anything was sent: not opted in,
    no room under today's cap, or a model whose spend could not be counted. Safe
    to fall back from, because no request left HAL."""


class AnthropicProvider:
    """Native Anthropic Messages API provider with dual auth support."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _DEFAULT_MODEL,
        *,
        subscription: Optional[SubscriptionAuth] = None,
        prefer: str = "auto",
    ):
        self._api_key = api_key
        self._key_resolved = api_key is not None
        self._model = model
        self._subscription = subscription
        if prefer not in ("auto", "api_key", "subscription"):
            raise ValueError(
                f"prefer must be 'auto', 'api_key', or 'subscription', got {prefer!r}",
            )
        self._prefer = prefer
        self.name = "anthropic"

    @property
    def api_key(self) -> Optional[str]:
        if not self._key_resolved:
            self._api_key = _fetch_keychain_key(API_KEY_SERVICE)
            self._key_resolved = True
        return self._api_key

    @property
    def subscription(self) -> Optional[SubscriptionAuth]:
        return self._subscription

    def _select_transport(self, prefer: Optional[str]) -> str:
        """Pick 'subscription' or 'api_key' for the next call.

        Force values are honoured even if the chosen transport is unhealthy —
        the caller gets a loud error rather than silent fallback.
        """
        effective = prefer or self._prefer
        if effective == "subscription":
            return "subscription"
        if effective == "api_key":
            return "api_key"
        # auto: subscription if healthy, else api_key
        if self._subscription is not None and self._subscription.health_check():
            return "subscription"
        return "api_key"

    def chat(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        messages: Optional[list[ChatMessage]] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 16384,
        prefer: Optional[str] = None,
        context_management: Optional[dict] = None,
        tools: Optional[list[dict]] = None,
    ) -> ChatResponse:
        """Send a message via Anthropic — subscription or Messages API.

        ``tools`` (Anthropic tool schemas) is sent only when non-empty, on either
        transport; the full response, ``tool_use`` blocks included, is on
        ``ChatResponse.raw``.

        ``temperature`` is optional and omitted from the request body when
        ``None``. Opus 4.7 rejects the ``temperature`` field entirely
        (adaptive thinking sets it internally); pre-4.7 models fall back
        to Anthropic's server-side default (1.0). Pass an explicit float
        only if you need deterministic sampling on a model that still
        accepts it.

        ``context_management`` is opt-in server-side context compaction. When
        set (e.g. ``{"edits": [{"type": "compact_20260112"}]}``) the field is
        added to the request body AND the ``anthropic-beta: compact-2026-01-12``
        header is sent — letting Anthropic summarise older turns server-side to
        stretch effective capacity within the 200K window. Left ``None`` the
        request body and headers are byte-identical to prior behaviour. The
        response's full content blocks (incl. the ``compaction`` block) are
        preserved on ``ChatResponse.raw_content`` so the caller can round-trip
        them on the next request — passing only ``text`` back breaks the chain.
        Supported on Opus 4.7 / Opus 4.6 / Sonnet 4.6 only.
        """
        model = model or self._model
        transport = self._select_transport(prefer)

        if transport == "subscription":
            if self._subscription is None:
                raise AnthropicError("No subscription configured but prefer='subscription'")
            if context_management is not None:
                # The CCH subscription transport filters anthropic-beta headers,
                # so server-side compaction cannot be honoured there. Fail loudly
                # rather than silently drop the caller's opt-in.
                raise AnthropicError(
                    "context_management requires the API-key Messages path; "
                    "it is unsupported on the CCH subscription transport. "
                    "Use prefer='api_key' to opt into server-side compaction.",
                )
            return self._subscription.chat(
                prompt,
                model=model,
                messages=messages,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                **({"tools": tools} if tools else {}),
            )

        if messages:
            msg_list = [{"role": m.role, "content": m.content} for m in messages]
        else:
            msg_list = [{"role": "user", "content": prompt}]

        body: dict = {
            "model": model,
            "messages": msg_list,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if system:
            body["system"] = system
        if tools:
            body["tools"] = tools
        betas: list[str] = []
        if context_management is not None:
            body["context_management"] = context_management
            betas.append(_COMPACT_BETA)

        start_ms = time.monotonic() * 1000
        data = self._request("POST", "/messages", body, betas=betas)
        latency_ms = time.monotonic() * 1000 - start_ms

        return _parse_response(data, "anthropic", latency_ms)

    def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(id="claude-fable-5", provider="anthropic",
                      display_name="Fable 5", context_window=1_000_000),
            ModelInfo(id="claude-opus-4-8", provider="anthropic",
                      display_name="Opus 4.8", context_window=1_000_000),
            ModelInfo(id="claude-opus-4-7-20260416", provider="anthropic",
                      display_name="Opus 4.7", context_window=1_000_000),
            ModelInfo(id="claude-sonnet-4-6-20250514", provider="anthropic",
                      display_name="Sonnet 4.6", context_window=1_000_000),
            ModelInfo(id="claude-haiku-4-5-20251001", provider="anthropic",
                      display_name="Haiku 4.5", context_window=200_000),
        ]

    def health_check(self) -> bool:
        """Ready when a source HAL may use is: the subscription, or an API key
        the spend guard allows today, under its daily cap. A paid key never
        counts as capacity. The guard is asked before the key is read, so a key
        that may not be used is never read."""
        if self._subscription is not None and self._subscription.health_check():
            return True
        return (spend_guard.decide_within_cap(spend_guard.API_KEY).allowed
                and self.api_key is not None)

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[dict] = None,
        *,
        betas: Optional[list[str]] = None,
    ) -> dict:
        """The API-key path, and the only one.

        The spend guard is asked first -- opted in, under today's daily cap, and a
        model whose spend can be counted -- so neither ``prefer="auto"`` nor
        ``prefer="api_key"`` bills a key the operator has not allowed. A refusal
        sends nothing and raises :class:`AnthropicSpendRefused`. A Messages body
        gets the prompt-cache markers, and the response is priced and added to
        today's spend (``lib/hal/anthropic_key_route.py``).
        """
        from lib.hal import anthropic_key_route as route  # lazy: the route imports this module
        model = str((body or {}).get("model") or self._model)
        verdict = route.admit(model)
        if not verdict.allowed:
            raise AnthropicSpendRefused(verdict.reason)
        if body is not None and path == "/messages":
            body = _with_cache_markers(body)
        data = self._send(method, f"{_API_BASE}{path}", body, betas=betas)
        route.settle(model, data.get("usage") if isinstance(data, dict) else None)
        return data

    def _send(self, method: str, url: str, body: Optional[dict], *,
              betas: Optional[list[str]] = None) -> dict:
        """One request, retried on a 429 or a connection failure."""
        backoff = _INITIAL_BACKOFF_S

        for attempt in range(_MAX_RETRIES + 1):
            try:
                return self._do_request(method, url, body, betas=betas)
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < _MAX_RETRIES:
                    time.sleep(backoff)
                    backoff *= _BACKOFF_MULTIPLIER
                    continue
                if e.code == 429:
                    raise AnthropicRateLimitError("Anthropic rate limited") from e
                error_body = _read_error_body(e)
                raise AnthropicError(f"Anthropic HTTP {e.code}: {error_body}") from e
            except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
                if attempt < _MAX_RETRIES:
                    time.sleep(backoff)
                    backoff *= _BACKOFF_MULTIPLIER
                    continue
                raise AnthropicError(f"Anthropic connection failed: {e}") from e

        raise AnthropicError(f"Exhausted {_MAX_RETRIES} retries")

    def _do_request(
        self,
        method: str,
        url: str,
        body: Optional[dict],
        *,
        betas: Optional[list[str]] = None,
    ) -> dict:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key or "",
            "anthropic-version": _API_VERSION,
            "User-Agent": "HAL/1.0",
        }
        if betas:
            # Comma-appendable per the Anthropic beta contract.
            headers["anthropic-beta"] = ",".join(betas)
        req_data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=req_data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
            data["_headers"] = dict(resp.headers)
            return data

    def __repr__(self) -> str:
        return f"AnthropicProvider(model={self._model!r})"


def _with_cache_markers(body: dict) -> dict:
    """``body`` with the prompt-cache markers, the caller's own objects untouched.

    ``apply_cache_control`` marks the last system block and the last tool in
    place, and a tool loop hands the same tools list to every turn, including
    turns that go on CCH, so the blocks it marks are copies.
    """
    marked = dict(body)
    for key in ("system", "tools"):
        if isinstance(marked.get(key), list):
            marked[key] = [dict(item) if isinstance(item, dict) else item for item in marked[key]]
    return apply_cache_control(marked)


def _parse_response(data: dict, provider: str, latency_ms: float) -> ChatResponse:
    content_blocks = data.get("content", [])
    text = "".join(
        block.get("text", "") for block in content_blocks
        if block.get("type") == "text"
    )

    usage_raw = data.get("usage", {})
    inp = usage_raw.get("input_tokens", 0)
    out = usage_raw.get("output_tokens", 0)

    return ChatResponse(
        text=text,
        model=data.get("model", ""),
        provider=provider,
        finish_reason=data.get("stop_reason", "end_turn"),
        usage=TokenUsage(
            prompt_tokens=inp, completion_tokens=out, total_tokens=inp + out,
            cache_read_tokens=usage_raw.get("cache_read_input_tokens", 0),
            cache_creation_tokens=usage_raw.get("cache_creation_input_tokens", 0),
        ),
        latency_ms=latency_ms,
        raw=data,
        # H781: preserve EVERY block (text, compaction, tool_use, …) verbatim
        # and in order so callers can append the full content to history and
        # round-trip the compaction chain. ``text`` alone drops non-text blocks.
        raw_content=content_blocks,
    )


def _read_error_body(e: urllib.error.HTTPError) -> str:
    try:
        raw = e.read().decode("utf-8", errors="replace")[:500]
        return _KEY_PATTERN.sub("[REDACTED]", raw)
    except Exception:
        return ""
