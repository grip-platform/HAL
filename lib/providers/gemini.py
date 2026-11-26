"""Gemini API provider client for GRIP Multi-Provider MoE.

Thin wrapper around Google Generative AI REST API.
API key fetched from macOS Keychain (grip-gemini), cached after first retrieval.
Default model: gemini-3.5-flash-lite (fast, cheap).

Usage:
    from lib.providers.gemini import GeminiProvider

    provider = GeminiProvider()
    text = provider.generate("Explain SOLID principles", temperature=0.3)
    tool_result = provider.generate_with_tools("What time is it?", tools=[...])
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Optional

# Keychain service name — matches lib/gemini_judge.py
_KEYCHAIN_SERVICE = "grip-gemini"
# v1beta for every model. The list call on 2026-10-18 served 63 ids there against
# 24 on v1, and thinking_level and the other 3.x-era fields are documented on
# v1beta. The branch this replaces sent only "preview" ids and the gemini-3
# prefix there, so gemini-4-argon would have gone to v1: the generation-prefix
# trap in GRIP's skills/dogfood-domain.
_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
# Repointed from gemini-2.5-flash on 2026-11-24 (GRIP#9389): Google's
# deprecations page says "effective December 1, 2026, API requests to
# gemini-2.5-flash will automatically route to gemini-3.5-flash-lite" and
# bill at its rates, so this names the model that actually answers.
_DEFAULT_MODEL = "gemini-3.5-flash-lite"
_REASONING_MODEL = "gemini-3.1-pro-preview"

# Rate limit backoff settings
_MAX_RETRIES = 3
_INITIAL_BACKOFF_S = 1.0
_BACKOFF_MULTIPLIER = 2.0


@dataclass
class GenerateResult:
    """Response from a Gemini generate call."""
    text: str = ""
    model: str = ""
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)


@dataclass
class ToolCall:
    """A single tool/function call returned by the model."""
    name: str = ""
    arguments: dict = field(default_factory=dict)


@dataclass
class ToolCallResult:
    """Response from a Gemini generate_with_tools call."""
    tool_calls: list[ToolCall] = field(default_factory=list)
    text: str = ""
    model: str = ""


class GeminiProviderError(Exception):
    """Raised when Gemini API returns a non-retryable error."""


class GeminiRateLimitError(GeminiProviderError):
    """Raised when rate limit is hit after exhausting retries."""


class GeminiProvider:
    """KISS Gemini API client with keychain auth and rate-limit backoff."""

    def __init__(self, api_key: Optional[str] = None, model: str = _DEFAULT_MODEL):
        self._api_key = api_key
        self._model = model

    @property
    def api_key(self) -> str:
        """Lazy-load and cache the API key."""
        if self._api_key is None:
            self._api_key = _fetch_keychain_key()
            if self._api_key is None:
                raise GeminiProviderError(
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
        max_tokens: int = 1024,
    ) -> GenerateResult:
        """Generate text from a prompt.

        Returns GenerateResult with the response text and metadata.
        """
        model = model or self._model
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        body = self._call_api(model, "generateContent", payload)
        return _parse_generate_response(body, model)

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> ToolCallResult:
        """Generate with function calling / tool use.

        tools: list of Gemini function declarations, e.g.:
            [{"name": "get_weather", "description": "...",
              "parameters": {"type": "object", "properties": {...}}}]

        Returns ToolCallResult with parsed tool calls.
        """
        model = model or self._model
        gemini_tools = [{"functionDeclarations": tools}]
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "tools": gemini_tools,
            "generationConfig": {"temperature": temperature},
        }
        body = self._call_api(model, "generateContent", payload)
        return _parse_tool_call_response(body, model)

    def _call_api(
        self,
        model: str,
        method: str,
        payload: dict,
    ) -> dict:
        """Make an API call with exponential backoff on rate limits."""
        url = f"{_API_BASE}/models/{model}:{method}?key={self.api_key}"
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
                    raise GeminiRateLimitError(
                        f"Rate limited after {_MAX_RETRIES + 1} attempts"
                    ) from e
                raise GeminiProviderError(
                    f"Gemini API error {e.code}: {e.reason}"
                ) from e

            except urllib.error.URLError as e:
                raise GeminiProviderError(
                    f"Network error: {e.reason}"
                ) from e

        raise GeminiProviderError(f"Unexpected retry exhaustion: {last_error}")


def _fetch_keychain_key() -> Optional[str]:
    """Fetch Gemini API key from macOS Keychain (matches gemini_judge.py pattern)."""
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


def _parse_generate_response(body: dict, model: str) -> GenerateResult:
    """Extract text and metadata from Gemini API response."""
    candidates = body.get("candidates", [])
    candidate = candidates[0] if candidates else {}
    parts = candidate.get("content", {}).get("parts", [])
    text = parts[0].get("text", "") if parts else ""
    usage = body.get("usageMetadata", {})
    return GenerateResult(
        text=text,
        model=model,
        finish_reason=candidate.get("finishReason", ""),
        usage={
            "prompt_tokens": usage.get("promptTokenCount", 0),
            # Billable output includes thinking: Google counts thoughts apart
            # from the answer ("totalTokenCount ... prompt + thoughts + response
            # candidates") and prices output "including thinking tokens".
            "completion_tokens": (usage.get("candidatesTokenCount", 0)
                                  + usage.get("thoughtsTokenCount", 0)),
            "total_tokens": usage.get("totalTokenCount", 0),
        },
    )


def _parse_tool_call_response(body: dict, model: str) -> ToolCallResult:
    """Extract tool calls from Gemini API response."""
    candidates = body.get("candidates", [])
    candidate = candidates[0] if candidates else {}
    parts = candidate.get("content", {}).get("parts", [])

    tool_calls = []
    text_parts = []

    for part in parts:
        if "functionCall" in part:
            fc = part["functionCall"]
            tool_calls.append(ToolCall(
                name=fc.get("name", ""),
                arguments=fc.get("args", {}),
            ))
        elif "text" in part:
            text_parts.append(part["text"])

    return ToolCallResult(
        tool_calls=tool_calls,
        text="\n".join(text_parts),
        model=model,
    )
