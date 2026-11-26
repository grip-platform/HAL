"""Universal OpenAI-compatible provider adapter.

One adapter for ALL providers that implement /v1/chat/completions:
OpenAI, Groq, DeepSeek, Mistral, Together, Fireworks, OpenRouter,
Ollama, vLLM, llama.cpp, LM Studio, LocalAI, NVIDIA NIM, etc.

Only base_url, api_key, and model change per provider.
Uses stdlib urllib — no external dependencies.

Usage:
    provider = OpenAICompatProvider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key="gsk_...",
    )
    response = provider.chat("Explain SOLID", model="llama-3.3-70b-versatile")
"""
import json
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from typing import Optional

from lib.hal.spend_guard import require_openai_destination
from lib.hal.model_quirks import model_requires_temperature_one
from lib.hal.types import (
    ChatMessage,
    ChatResponse,
    ModelInfo,
    TokenUsage,
)

_KEYCHAIN_CMD = ["security", "find-generic-password", "-w", "-s"]
_DEFAULT_TIMEOUT_S = 120
_MAX_RETRIES = 3
_INITIAL_BACKOFF_S = 1.0
_BACKOFF_MULTIPLIER = 2.0
_KEY_PATTERN = re.compile(r"(sk-|gsk_|key-|Bearer |AIza)\S+")


def _fetch_keychain_key(service: str) -> Optional[str]:
    """Fetch API key from macOS Keychain. Returns None on failure."""
    try:
        result = subprocess.run(
            [*_KEYCHAIN_CMD, service],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _resolve_api_key(source: str) -> Optional[str]:
    """Resolve API key from source spec: 'env:VAR', 'keychain:service', or literal.

    For 'keychain:service', fall back to the environment when the keychain is
    unavailable (Linux/CI/a tailnet Linux peer): the canonical
    GRIP_<PROVIDER>_API_KEY derived from the service name (also the bare
    <PROVIDER>_API_KEY). This lets ONE committed 'keychain:grip-<provider>'
    config resolve on macOS (keychain) AND Linux (env) — no per-host
    providers.yaml divergence. Mirrors gateway_egress._default_key_resolver.
    """
    if not source:
        return None
    if source.startswith("env:"):
        return os.environ.get(source[4:])
    if source.startswith("keychain:"):
        service = source[9:]
        key = _fetch_keychain_key(service)
        if key:
            return key
        # The public build renames five slots from the grip- prefix to hal-; both
        # prefixes name the same variable, so a stranger's env keys still resolve.
        name = service.replace("grip-", "").removeprefix("hal-")
        base = name.replace("-", "_").upper() + "_API_KEY"
        return os.environ.get(base) or os.environ.get("GRIP_" + base)
    return source


def _scrub_keys(text: str) -> str:
    """Remove API key patterns from error text to prevent leakage."""
    return _KEY_PATTERN.sub("[REDACTED]", text)


class OpenAICompatError(Exception):
    """Non-retryable provider error."""


class RateLimitError(OpenAICompatError):
    """Rate limit hit after exhausting retries.

    Carries structured error metadata so callers (notably the Limit-Sentinel)
    can distinguish a transient rate-limit (retryable / backoff) from a billing
    ``insufficient_quota`` exhaustion (never retryable — reject the provider).
    All metadata kwargs are optional, so ``RateLimitError("msg")`` still works.
    """

    def __init__(self, message, *, status_code=None, error_type=None,
                 error_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.error_code = error_code
        self.body = body


class OpenAICompatProvider:
    """Universal adapter for any OpenAI-compatible API endpoint."""

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: Optional[str] = None,
        api_key_source: str = "",
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        max_retries: int = _MAX_RETRIES,
        extra_headers: Optional[dict[str, str]] = None,
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._api_key_source = api_key_source
        self._key_resolved = api_key is not None
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._extra_headers = extra_headers or {}

    @property
    def api_key(self) -> Optional[str]:
        if not self._key_resolved and self._api_key_source:
            self._api_key = _resolve_api_key(self._api_key_source)
            self._key_resolved = True
        return self._api_key

    def __repr__(self) -> str:
        return f"OpenAICompatProvider(name={self.name!r}, base_url={self.base_url!r})"

    def chat(
        self,
        prompt: str,
        *,
        model: str,
        messages: Optional[list[ChatMessage]] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 16384,
        json_mode: bool = False,
        tools: Optional[list[dict]] = None,
    ) -> ChatResponse:
        """Send a chat completion request. Returns normalised ChatResponse."""
        msg_dicts = []
        if system:
            msg_dicts.append({"role": "system", "content": system})
        if messages:
            msg_dicts.extend(_message_to_dict(m) for m in messages)
        else:
            msg_dicts.append({"role": "user", "content": prompt})

        body: dict = {
            "model": model,
            "messages": msg_dicts,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if model_requires_temperature_one(model):
            # Moonshot Kimi lineage 400s on temperature != 1 — omit so the
            # provider applies its own default (lib/hal/model_quirks.py).
            body.pop("temperature", None)
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        start_ms = time.monotonic() * 1000
        data = self._request("POST", "/chat/completions", body)
        latency_ms = time.monotonic() * 1000 - start_ms

        return _parse_chat_response(data, self.name, latency_ms)

    def list_models(self) -> list[ModelInfo]:
        """Fetch available models from the provider."""
        try:
            data = self._request("GET", "/models")
        except OpenAICompatError:
            return []

        models = data.get("data", [])
        return [
            ModelInfo(
                id=m.get("id", ""),
                provider=self.name,
                display_name=m.get("id", ""),
            )
            for m in models
            if isinstance(m, dict) and m.get("id")
        ]

    def health_check(self) -> bool:
        """Check if the provider is reachable."""
        try:
            self._request("GET", "/models")
            return True
        except (OpenAICompatError, urllib.error.URLError, TimeoutError):
            return False

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        """HTTP request with retry and exponential backoff."""
        url = f"{self.base_url}{path}"
        backoff = _INITIAL_BACKOFF_S

        for attempt in range(self._max_retries + 1):
            try:
                return self._do_request(method, url, body)
            except urllib.error.HTTPError as e:
                # Raises on a terminal error; returns True to back off before
                # retrying, or False to retry immediately (max_tokens fix — a
                # parameter correction, not a rate signal, so no backoff).
                if not self._handle_http_error(e, attempt, body):
                    continue
            except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
                if attempt >= self._max_retries:
                    raise OpenAICompatError(
                        f"{self.name}: connection failed — {e}"
                    ) from e
            # Back off then loop. Reached ONLY for retries that should wait: a
            # retryable 429 (handler returned True) or a connection error under
            # the cap. The max_tokens-400 path returned False and `continue`d
            # above, bypassing this sleep — an immediate, backoff-free retry
            # (proved by test_max_tokens_400_retries_without_backoff_sleep).
            time.sleep(backoff)
            backoff *= _BACKOFF_MULTIPLIER

        raise OpenAICompatError(f"{self.name}: exhausted {self._max_retries} retries")

    def _handle_http_error(self, e: urllib.error.HTTPError, attempt: int,
                           body: Optional[dict]) -> bool:
        """Classify an HTTPError. Raise on a terminal error. Return True if the
        caller should back off before retrying (429 rate signal), or False to
        retry immediately WITHOUT backoff (max_tokens graceful degradation is a
        parameter fix, not a rate signal). Mutates ``body`` for that retry."""
        can_retry = attempt < self._max_retries
        if e.code == 429:
            self._handle_429(e, can_retry)
            return True
        error_body = _read_error_body(e)
        # Graceful degradation on a max_tokens-cap 400: some providers
        # (deepseek 8192, moonshot-8k, GLM-4, small groq/qwen) reject a
        # max_tokens larger than the model's output cap with a 400 instead of
        # clamping. Halve max_tokens and retry immediately so a high default
        # (16384) degrades gracefully. High-cap providers accept 16384.
        _mt = (body or {}).get("max_tokens")
        if (e.code == 400 and "max_tokens" in error_body.lower()
                and isinstance(_mt, int) and _mt > 1024 and can_retry):
            body["max_tokens"] = max(1024, _mt // 2)
            return False
        raise OpenAICompatError(f"{self.name}: HTTP {e.code} — {error_body}") from e

    def _handle_429(self, e: urllib.error.HTTPError, can_retry: bool) -> None:
        """A 429: short-circuit a billing ``insufficient_quota`` (never
        retryable), otherwise retry until exhausted. Returns to signal a retry;
        raises with structured metadata (``error_code``) so the Limit-Sentinel
        classifies a billing cap as rejected, not a transient rate-limit."""
        error_body = _read_error_body(e)
        code, etype = _parse_error_code(error_body)
        if code == "insufficient_quota":
            raise RateLimitError(
                f"{self.name}: insufficient_quota — billing quota exhausted "
                f"(not retryable; check plan/billing)",
                status_code=429, error_type=etype, error_code=code, body=error_body,
            ) from e
        if not can_retry:
            raise RateLimitError(
                f"{self.name}: rate limited after {self._max_retries} retries (429)",
                status_code=429, error_type=etype, error_code=code, body=error_body,
            ) from e
        # retryable & not exhausted: return so _request backs off and retries

    def _do_request(self, method: str, url: str, body: Optional[dict]) -> dict:
        """Single HTTP request. Captures response headers for sentinel."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "HAL/1.0",
            **self._extra_headers,
        }
        require_openai_destination(url)
        key = self.api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"

        req_data = json.dumps(body).encode("utf-8") if body else None
        req = urllib.request.Request(
            url, data=req_data, headers=headers, method=method,
        )
        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            # Capture response headers for limit sentinel
            data["_headers"] = dict(resp.headers)
            return data


def _read_error_body(e: urllib.error.HTTPError) -> str:
    """Read and scrub error body to prevent API key leakage."""
    try:
        raw = e.read().decode("utf-8", errors="replace")[:500]
        return _scrub_keys(raw)
    except Exception:
        return ""


def _parse_error_code(body: str) -> "tuple[Optional[str], Optional[str]]":
    """Extract (code, type) from an OpenAI-shaped error body.

    Returns (None, None) when unparseable. Falls back to a substring probe for
    ``insufficient_quota`` so a truncated/scrubbed body still classifies the
    one case callers act on differently (billing exhaustion vs transient 429).
    """
    try:
        err = json.loads(body).get("error", {})
        if isinstance(err, dict):
            return err.get("code"), err.get("type")
    except Exception:
        pass
    low = (body or "").lower()
    return ("insufficient_quota" if "insufficient_quota" in low else None), None


def _message_to_dict(msg: ChatMessage) -> dict:
    """Convert ChatMessage to OpenAI-format dict."""
    d: dict = {"role": msg.role, "content": msg.content}
    if msg.name:
        d["name"] = msg.name
    if msg.tool_calls:
        d["tool_calls"] = list(msg.tool_calls)
    if msg.tool_call_id:
        d["tool_call_id"] = msg.tool_call_id
    return d


def _parse_chat_response(
    data: dict, provider: str, latency_ms: float,
) -> ChatResponse:
    """Parse OpenAI-format response into normalised ChatResponse."""
    choices = data.get("choices", [])
    if not choices:
        return ChatResponse(
            text="", model=data.get("model", ""), provider=provider,
            finish_reason="error", latency_ms=latency_ms, raw=data,
        )

    choice = choices[0]
    message = choice.get("message", {})
    usage_raw = data.get("usage", {})

    usage = TokenUsage(
        prompt_tokens=usage_raw.get("prompt_tokens", 0),
        completion_tokens=usage_raw.get("completion_tokens", 0),
        total_tokens=usage_raw.get("total_tokens", 0),
        cache_read_tokens=usage_raw.get("cache_read_input_tokens", 0),
        cache_creation_tokens=usage_raw.get("cache_creation_input_tokens", 0),
    )

    return ChatResponse(
        text=message.get("content", "") or "",
        model=data.get("model", ""),
        provider=provider,
        finish_reason=choice.get("finish_reason", "stop") or "stop",
        usage=usage,
        cost_usd=usage_raw.get("cost"),  # OpenRouter includes this
        latency_ms=latency_ms,
        raw=data,
    )
