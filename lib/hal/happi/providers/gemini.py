"""Gemini provider adapter for HAPPI/1.1 (v1.0 envelopes accepted).

Protocol-version-agnostic at the wire layer: emits via shared
`lib.hal.happi.events` encoder; envelope `v` (client's declared version)
flows through unchanged.

Calls Google's Generative Language API streaming endpoint:

    POST https://generativelanguage.googleapis.com/v1beta/models/
         <model>:streamGenerateContent?alt=sse

Auth: `x-goog-api-key` header. Token source order:
    - envelope.auth.token when scheme="apikey" (env:/keychain: prefixes honoured)
    - GEMINI_API_KEY environment variable
    - `grip-gemini` Keychain service

Body shape diverges from OpenAI/Anthropic conventions:

    {
        "contents": [{"role": "user", "parts": [{"text": "..."}]}],
        "systemInstruction": {"parts": [{"text": "..."}]},
        "generationConfig": {"maxOutputTokens": <int>}
    }

SSE format is `data: {...}\\n\\n` (no named events, no [DONE] sentinel —
the stream terminates by the connection closing). Text lives under
candidates[0].content.parts[*].text; usage is reported as usageMetadata
in the final chunk.

Stdlib-only. No external deps.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from typing import Any, Iterator, Optional

from lib.hal.happi.envelope import HappiEnvelope
from lib.hal.happi.events import emit_completed, emit_delta, emit_error
from lib.hal.happi.sampling import resolve_sampling


GEMINI_ENDPOINT_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:streamGenerateContent?alt=sse"
)
DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT_S = 120
KEYCHAIN_SERVICE = "grip-gemini"
API_KEY_ENV = "GEMINI_API_KEY"

# Canonical sampling key -> Gemini generationConfig wire name. Gemini uses
# camelCase (topP) where the canonical resolver uses snake_case (top_p). This
# single mapping is the one place to extend when resolve_sampling() grows a new
# key, so the "one contract" the resolver documents is honoured uniformly here
# (rather than scattered across per-key conditionals).
_GEMINI_SAMPLING_WIRE_NAMES = {
    "temperature": "temperature",
    "top_p": "topP",
    "seed": "seed",
}

REDACTION_PATTERN = re.compile(
    r"(AIzaSy[A-Za-z0-9_-]+|sk-[A-Za-z0-9_-]{20,}|Bearer [A-Za-z0-9._-]+)"
)


def _redact(text: str) -> str:
    return REDACTION_PATTERN.sub("[REDACTED]", text)


def _fetch_keychain_token(service: str = KEYCHAIN_SERVICE) -> Optional[str]:
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


def _resolve_token(envelope: HappiEnvelope) -> Optional[str]:
    scheme = (envelope.auth.get("scheme") or "subscription").lower()
    if scheme == "apikey":
        return _resolve_apikey_source(str(envelope.auth.get("token", "")))
    return os.environ.get(API_KEY_ENV) or _fetch_keychain_token()


def _resolve_model(envelope: HappiEnvelope) -> str:
    spec = str(envelope.flags.get("provider") or "")
    if "/" not in spec:
        return DEFAULT_MODEL
    _, model = spec.split("/", 1)
    if model.startswith("models/"):
        model = model[len("models/"):]
    return model or DEFAULT_MODEL


def _build_request_body(envelope: HappiEnvelope) -> dict[str, Any]:
    max_tokens = int(envelope.flags.get("max_tokens") or DEFAULT_MAX_TOKENS)
    # Canonical resolution (content > flags.message > args). PARAMOUNT: this
    # adapter previously ignored envelope.content, so a content-only prompt
    # (GRIP council / AGORA dispatch) reached Gemini with an EMPTY message and
    # the model free-associated, failing AGORA's validity gate
    # (CodeTonight-SA/GRIP#2697). HappiEnvelope.user_message is the single
    # source of truth; do not re-implement the precedence here.
    user_message = envelope.user_message()
    generation_config: dict[str, Any] = {"maxOutputTokens": max_tokens}
    # Optional sampling controls (omitted unless the caller sets them, so the
    # body is byte-identical to prior behaviour by default). AGORA sets
    # `temperature: 0.0` here for deterministic judge verdicts. Remap canonical
    # snake_case keys to Gemini's camelCase wire names via the single mapping.
    sampling = resolve_sampling(envelope.flags, allow_seed=True)
    for canonical, wire_name in _GEMINI_SAMPLING_WIRE_NAMES.items():
        if canonical in sampling:
            generation_config[wire_name] = sampling[canonical]
    body: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": str(user_message)}]}],
        "generationConfig": generation_config,
    }
    system_prompt = envelope.flags.get("system")
    if system_prompt:
        body["systemInstruction"] = {"parts": [{"text": str(system_prompt)}]}
    return body


def _open_stream(model: str, body: dict[str, Any], token: str, timeout: int) -> Any:
    url = GEMINI_ENDPOINT_TEMPLATE.format(model=model)
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "content-type": "application/json",
            "x-goog-api-key": token,
            "accept": "text/event-stream",
        },
    )
    return urllib.request.urlopen(req, timeout=timeout)


def _iter_sse_data(response: Any) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON objects from a Gemini-style SSE stream.

    Format: `data: {...}\\n\\n` per event; no [DONE] sentinel — the stream
    terminates by the connection closing. Event names are absent.
    """
    for raw in response:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload:
            continue
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


def _yield_auth_failure(envelope: HappiEnvelope) -> Iterator[str]:
    yield emit_error(
        envelope.id,
        f"no token available for gemini "
        f"(checked env {API_KEY_ENV} and Keychain {KEYCHAIN_SERVICE!r})",
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


def _extract_text_from_candidate(data: dict[str, Any]) -> Optional[str]:
    """Pull concatenated text from candidates[0].content.parts[*].text."""
    candidates = data.get("candidates") or []
    if not candidates:
        return None
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    pieces = [str(p.get("text")) for p in parts if p.get("text")]
    return "".join(pieces) if pieces else None


def _extract_usage(data: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Token usage from a chunk's usageMetadata; a chunk without it changes nothing.

    ``out`` is the BILLABLE output. Google prices output "including thinking
    tokens", and its API reference counts thoughts apart from the answer:
    "totalTokenCount ... (prompt + thoughts + response candidates)". Reading
    candidatesTokenCount alone under-counted every thinking model, the AGORA
    judge included. ``thinking`` reports the thoughts share when there is one.

    Gemini's counts are cumulative per response, so each part carries forward
    on its own and ``out`` is rebuilt from the parts; adding this chunk's thoughts
    to a carried-forward ``out`` would count earlier thoughts twice. A cumulative
    count never falls, so a chunk that omits a field or sends 0 keeps the larger
    count already seen.
    """
    meta = data.get("usageMetadata")
    if not meta:
        return current
    prior_thinking = current.get("thinking", 0)
    thinking = _cumulative(meta, "thoughtsTokenCount", prior_thinking)
    answer = _cumulative(meta, "candidatesTokenCount",
                         current.get("out", 0) - prior_thinking)
    usage = {"in": _cumulative(meta, "promptTokenCount", current.get("in", 0)),
             "out": answer + thinking}
    if thinking:
        usage["thinking"] = thinking
    return usage


def _cumulative(meta: dict[str, Any], key: str, prior: int) -> int:
    """The larger of the count already seen and this chunk's report of it."""
    return max(prior, meta.get(key) or 0)


def _translate_and_terminate(envelope: HappiEnvelope, response: Any) -> Iterator[str]:
    usage: dict[str, Any] = {}
    for data in _iter_sse_data(response):
        err = data.get("error")
        if err:
            status = err.get("status") or err.get("code") or "unknown"
            yield emit_error(
                envelope.id,
                _redact(str(err.get("message", "unknown provider error"))),
                code=f"provider_error:{status}",
            )
            break
        text = _extract_text_from_candidate(data)
        if text:
            yield emit_delta(envelope.id, text)
        usage = _extract_usage(data, usage)
    yield emit_completed(envelope.id, usage=usage)


def stream_gemini(envelope: HappiEnvelope) -> Iterator[str]:
    """Main entry: translate one envelope into a stream of HAPPI NDJSON events."""
    token = _resolve_token(envelope)
    if not token:
        yield from _yield_auth_failure(envelope)
        return

    model = _resolve_model(envelope)
    body = _build_request_body(envelope)
    timeout = int(envelope.flags.get("timeout_s") or DEFAULT_TIMEOUT_S)

    try:
        response = _open_stream(model, body, token, timeout)
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
