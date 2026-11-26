"""Groq API provider client for GRIP Multi-Provider MoE.

Thin wrapper around Groq REST API for Llama 3.3 70B and DeepSeek R1.
API key fetched from macOS Keychain (grip-groq), cached after first retrieval.

Usage:
    from lib.providers.groq import GroqProvider

    provider = GroqProvider()
    text = provider.generate("Explain SOLID principles", temperature=0.3)

    # Use a reasoning-capable model for analysis tasks
    text = provider.generate("Analyse this proof", model="qwen/qwen3.8-27b")
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional

_KEYCHAIN_SERVICE = "grip-groq"
_API_BASE = "https://api.groq.com/openai/v1"
_DEFAULT_MODEL = "openai/gpt-oss-120b"
# GPT-OSS 120B replaces qwen3-32b (GroqCloud decommission 2026-07-17).
_GPT_OSS_120B = "openai/gpt-oss-120b"
# llama-4-scout and llama-3.3-70b-versatile RETIRED at Groq (2026-08-30).
_QWEN38_27B = "qwen/qwen3.8-27b"

_MAX_RETRIES = 3
_INITIAL_BACKOFF_S = 1.0
_BACKOFF_MULTIPLIER = 2.0


@dataclass
class GroqResult:
    """Response from a Groq generate call."""
    text: str = ""
    model: str = ""
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)


class GroqProviderError(Exception):
    """Raised when Groq API returns a non-retryable error."""


class GroqRateLimitError(GroqProviderError):
    """Raised when rate limit is hit after exhausting retries."""


class GroqProvider:
    """KISS Groq API client with keychain auth and rate-limit backoff."""

    def __init__(self, api_key: Optional[str] = None, model: str = _DEFAULT_MODEL):
        self._api_key = api_key
        self._model = model

    @property
    def api_key(self) -> str:
        if self._api_key is None:
            self._api_key = _fetch_keychain_key()
            if self._api_key is None:
                raise GroqProviderError(
                    f"No API key found in Keychain service '{_KEYCHAIN_SERVICE}' "
                    "and none provided explicitly"
                )
        return self._api_key

    @property
    def model(self) -> str:
        return self._model

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system: Optional[str] = None,
    ) -> GroqResult:
        """Generate text from a prompt via Groq API.

        Args:
            prompt: User message content.
            model: Override model. Use 'qwen/qwen3.8-27b' for reasoning.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system: Optional system message.
        """
        model = model or self._model
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body = self._call_api("chat/completions", payload)
        return _parse_response(body, model)

    def generate_reasoning(
        self,
        prompt: str,
        *,
        max_tokens: int = 4096,
        system: Optional[str] = None,
    ) -> GroqResult:
        """Generate with the GPT-OSS 120B reasoning model."""
        return self.generate(
            prompt,
            model=_GPT_OSS_120B,
            temperature=0.6,
            max_tokens=max_tokens,
            system=system,
        )

    def generate_qwen38(
        self,
        prompt: str,
        *,
        max_tokens: int = 4096,
        system: Optional[str] = None,
    ) -> GroqResult:
        """Generate with Qwen 3.8 27B (current fast reasoning MoE)."""
        return self.generate(
            prompt,
            model=_QWEN38_27B,
            temperature=0.7,
            max_tokens=max_tokens,
            system=system,
        )

    # llama-4-scout retired at Groq (2026-08-30) — kept as a name so callers
    # fail loudly at grep-time rather than silently on an old default.
    def generate_llama4(self, *args, **kwargs) -> GroqResult:
        raise NotImplementedError(
            "groq llama-4-scout retired upstream 2026-08-30; use generate_qwen38()"
        )

    def _call_api(self, endpoint: str, payload: dict) -> dict:
        """Make an API call with exponential backoff on rate limits."""
        url = f"{_API_BASE}/{endpoint}"
        data = json.dumps(payload).encode("utf-8")

        backoff = _INITIAL_BACKOFF_S
        last_error = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                        "User-Agent": "GRIP/6.0",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return json.loads(resp.read().decode("utf-8"))

            except urllib.error.HTTPError as e:
                last_error = e
                if e.code == 429 and attempt < _MAX_RETRIES:
                    time.sleep(backoff)
                    backoff *= _BACKOFF_MULTIPLIER
                    continue
                if e.code == 429:
                    raise GroqRateLimitError(
                        f"Rate limited after {_MAX_RETRIES + 1} attempts"
                    ) from e
                body = ""
                try:
                    body = e.read().decode("utf-8")
                except Exception:
                    pass
                raise GroqProviderError(
                    f"Groq API error {e.code}: {e.reason} {body}"
                ) from e

            except urllib.error.URLError as e:
                raise GroqProviderError(f"Network error: {e.reason}") from e

        raise GroqProviderError(f"Unexpected retry exhaustion: {last_error}")


def _fetch_keychain_key() -> Optional[str]:
    """Fetch Groq API key from macOS Keychain."""
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", _KEYCHAIN_SERVICE, "-w"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _parse_response(body: dict, model: str) -> GroqResult:
    """Extract text and metadata from Groq API response."""
    choices = body.get("choices", [])
    choice = choices[0] if choices else {}
    message = choice.get("message", {})
    usage = body.get("usage", {})
    return GroqResult(
        text=message.get("content", ""),
        model=model,
        finish_reason=choice.get("finish_reason", ""),
        usage={
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    )
