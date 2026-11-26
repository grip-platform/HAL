"""Anthropic provider adapter for HAPPI/1.1 (v1.0 envelopes accepted).

Calls https://api.anthropic.com/v1/messages with stream=true and translates
Anthropic's SSE event stream into HAPPI NDJSON events. Stdlib-only
(urllib + json + os + subprocess for keychain lookup); no external deps.

Protocol-version-agnostic at the wire layer: emits via shared
`lib.hal.happi.events` encoder, so envelope `v` (client's declared version)
flows through unchanged while emitted events stamp the runtime's resolved
PROTOCOL_VERSION.

Auth resolution (by `envelope.auth.scheme`):
    "subscription" or unset → CLAUDE_CODE_OAUTH_TOKEN env var,
                              falling back to Keychain `grip-claude-oauth`.
    "apikey"               → envelope.auth.token, with `env:` / `keychain:`
                              prefixes honoured (same pattern as openai_compat).

Error discipline:
    - Missing token → emit_error(code="missing_credentials") + completed
    - Network / HTTP error → emit_error(code="provider_http_<status>") + completed
    - Anthropic error event → emit_error(code="provider_error:<type>") + completed
    - Token patterns scrubbed from error messages via REDACTION_PATTERN

Event mapping:
    content_block_delta (text)   → delta(text=...)
    message_delta / message_stop → completed(usage={in, out})
    error                        → error + completed
    everything else              → silently consumed (ping, start, stop)
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


ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT_S = 120
ANTHROPIC_VERSION = "2023-06-01"
KEYCHAIN_SERVICE = "grip-claude-oauth"
SUBSCRIPTION_ENV = "CLAUDE_CODE_OAUTH_TOKEN"

# OAuth subscription tokens minted by `claude setup-token` (and the in-app
# login flow) are prefixed `sk-ant-oat` — Anthropic OAuth Access Token. They
# CANNOT be sent on the `x-api-key` header; Anthropic rejects them with
# `authentication_error: invalid x-api-key`. The subscription path requires
# the full CCH envelope (Authorization Bearer, User-Agent claude-cli/<v>,
# x-app cli, anthropic-beta with per-model filter, AND a signed body) — the
# point of CCH is to use an OAuth token to make subscription requests while
# pretending to be a Claude Code API call. The canonical reference is
# @anthropic-ai/claude-code 2.1.112 `cli-formatted.js:108743-108748` mirrored
# in CodeTonight-SA/TF>>>>. Two-header bolt-ons (Bearer + oauth-beta only)
# still 401 on the User-Agent check; only the full CCH signer's envelope is
# accepted upstream. Raw API keys (`sk-ant-api03-*`) take the existing
# x-api-key path.
OAUTH_TOKEN_PREFIX = "sk-ant-oat"

REDACTION_PATTERN = re.compile(
    r"(sk-ant-[a-z0-9_-]+|sk-[A-Za-z0-9_-]{20,}|Bearer [A-Za-z0-9._-]+)"
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
    return os.environ.get(SUBSCRIPTION_ENV) or _fetch_keychain_token()


def _build_request_body(envelope: HappiEnvelope) -> dict[str, Any]:
    model = str(envelope.flags.get("provider") or DEFAULT_MODEL)
    if model.startswith("anthropic/"):
        model = model[len("anthropic/"):]

    max_tokens = int(envelope.flags.get("max_tokens") or DEFAULT_MAX_TOKENS)
    # Canonical resolution (content > flags.message > args). See
    # HappiEnvelope.user_message — every adapter MUST go through it so the
    # envelope contract is honoured identically across providers and the
    # content-only prompt drift (CodeTonight-SA/GRIP#2697) cannot recur.
    message = envelope.user_message()

    body: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "stream": True,
        "messages": [{"role": "user", "content": message}],
    }
    # Optional sampling controls (omitted unless the caller sets them, so the
    # body is byte-identical to prior behaviour by default). `temperature` and
    # `top_p` are the Anthropic Messages API wire names, so the resolved dict
    # merges directly. `seed` is excluded (allow_seed=False) — the Messages API
    # has no seed field and rejects it. Applies to this x-api-key path only;
    # the CCH subscription path cannot honour temperature (see
    # _build_cch_call_args).
    body.update(resolve_sampling(envelope.flags, allow_seed=False))
    system_prompt = envelope.flags.get("system")
    if system_prompt:
        body["system"] = str(system_prompt)
    return body


def _open_stream(body: dict[str, Any], token: str, timeout: int) -> Any:
    req = urllib.request.Request(
        ANTHROPIC_ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "content-type": "application/json",
            "x-api-key": token,
            "anthropic-version": ANTHROPIC_VERSION,
            "accept": "text/event-stream",
        },
    )
    return urllib.request.urlopen(req, timeout=timeout)


def _iter_sse_events(response: Any) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (event_name, data_obj) tuples from an SSE stream response."""
    event_name = ""
    data_lines: list[str] = []
    for raw in response:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data_lines:
                try:
                    yield event_name, json.loads("\n".join(data_lines))
                except json.JSONDecodeError:
                    pass
            event_name = ""
            data_lines = []
            continue
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())


def _yield_auth_failure(envelope: HappiEnvelope) -> Iterator[str]:
    scheme = envelope.auth.get("scheme", "subscription")
    yield emit_error(
        envelope.id,
        f"no token available for scheme={scheme!r}",
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


def _translate_and_terminate(envelope: HappiEnvelope, response: Any) -> Iterator[str]:
    """Translate a live SSE response into HAPPI events + terminal completed."""
    usage: dict[str, Any] = {}
    for event_name, data in _iter_sse_events(response):
        if event_name == "content_block_delta":
            text = (data.get("delta") or {}).get("text")
            if text:
                yield emit_delta(envelope.id, text)
        elif event_name in ("message_delta", "message_stop"):
            usage_block = data.get("usage") or {}
            if usage_block:
                usage = {
                    "in": usage_block.get("input_tokens", usage.get("in", 0)),
                    "out": usage_block.get("output_tokens", usage.get("out", 0)),
                }
        elif event_name == "error":
            err_block = data.get("error") or {}
            yield emit_error(
                envelope.id,
                _redact(str(err_block.get("message", "unknown provider error"))),
                code=f"provider_error:{err_block.get('type', 'unknown')}",
            )
            break
    yield emit_completed(envelope.id, usage=usage)


def _is_oauth_token(token: str) -> bool:
    """True iff `token` is an Anthropic OAuth subscription access token.

    The prefix is the only structural signal (cli-formatted.js:108743 dispatches
    by the credential object's key — `accessToken` vs `apiKey` — but on the
    wire the OAuth Access Token format is stable). Cheap and safe: a false
    negative (real OAuth treated as API key) is the *current* bug, so the
    detection is strictly additive — the historic x-api-key path is preserved
    for anything that doesn't match.
    """
    return bool(token) and token.startswith(OAUTH_TOKEN_PREFIX)


def _flag_is_true(value: Any) -> bool:
    """Coerce a HAPPI flag to bool, default-off for absent/false/garbage.

    Accepts a real JSON boolean (``"compact": true``) or the common string forms
    a CLI / relay might pass. Anything else — absent, ``False``, ``"false"``, a
    stray value — is False, preserving the opt-in / default-off contract.
    """
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "on")


def _build_cch_call_args(envelope: HappiEnvelope) -> tuple[str, str, dict[str, Any]]:
    """Pull the CCH-call arguments out of the envelope: (message, model, build_kwargs)."""
    model = str(envelope.flags.get("provider") or DEFAULT_MODEL)
    if model.startswith("anthropic/"):
        model = model[len("anthropic/"):]
    max_tokens = int(envelope.flags.get("max_tokens") or DEFAULT_MAX_TOKENS)
    build_kwargs: dict[str, Any] = {"max_tokens": max_tokens}
    system_prompt = envelope.flags.get("system")
    if system_prompt:
        build_kwargs["system"] = str(system_prompt)
    # Opt-in compact summarisation: thread it only when the envelope explicitly
    # requests it. Absent (the default) -> build_kwargs has no ``compact`` ->
    # CCHSigner defaults compact=False -> byte-identical request. Whether it then
    # actually fires is still model-capability-gated downstream by
    # _effective_compact (server_compaction + context_management).
    if _flag_is_true(envelope.flags.get("compact")):
        build_kwargs["compact"] = True
    # NOTE: sampling flags (temperature/top_p/seed) are deliberately NOT
    # threaded onto the CCH subscription path. The CCH signer mimics a Claude
    # Code request, which sets `thinking: {"type": "adaptive"}` for thinking-
    # capable models (Opus/Sonnet), and the Anthropic Messages API rejects a
    # non-default temperature when thinking is active. So temperature+CCH would
    # 400 on exactly the common models. Deterministic-temperature requests
    # therefore use the direct x-api-key path (_build_request_body) or a
    # provider whose API has no such constraint (Gemini — which is what AGORA's
    # judge actually uses). Honouring it here would be a latent 400, not parity.
    return envelope.user_message(), model, build_kwargs


def _emit_cch_response(envelope: HappiEnvelope, response: Any) -> Iterator[str]:
    """Translate a ChatResponse into happi delta + completed events."""
    if response.text:
        yield emit_delta(envelope.id, response.text)
    yield emit_completed(
        envelope.id,
        usage={
            "in": response.usage.prompt_tokens,
            "out": response.usage.completion_tokens,
        },
    )


def _hal_chooses_credential(envelope: HappiEnvelope) -> bool:
    """True when the envelope leaves the credential to HAL (scheme subscription or
    unset), read the same way as :func:`_resolve_token`."""
    return (envelope.auth.get("scheme") or "subscription").lower() != "apikey"


def _api_key_route_events(envelope: HappiEnvelope) -> Optional[list[str]]:
    """The envelope's events on HAL's API key, or None to sign it with CCH.

    ``lib/hal/anthropic_key_route.py`` decides, and ships closed. As on the CCH
    path the answer is not streamed: one ``delta``, then ``completed``. The
    ``compact`` flag is not sent on this route, and neither are the sampling
    flags, which the CCH path does not send either. An error from Anthropic
    becomes an error event and is never retried on CCH.
    """
    from lib.hal import anthropic_key_route
    from lib.hal.providers.anthropic_native import AnthropicError

    message, model, build_kwargs = _build_cch_call_args(envelope)
    try:
        response = anthropic_key_route.try_api_key(
            message, model=model, system=build_kwargs.get("system"),
            max_tokens=build_kwargs["max_tokens"],
        )
    except AnthropicError as exc:
        return [emit_error(envelope.id, f"API key error: {_redact(str(exc))[:500]}",
                           code="provider_api_key_error"),
                emit_completed(envelope.id, usage={})]
    return None if response is None else list(_emit_cch_response(envelope, response))


def _stream_via_cch(envelope: HappiEnvelope, token: str) -> Iterator[str]:
    """Dispatch an OAuth-subscription request through the CCH signer.

    CCH is non-streaming by construction (cch.py hardcodes `stream: False`
    in :meth:`CCHSigner.build_body` — the signing contract Anthropic accepts
    mimics CC's non-streaming requests). To preserve the happi event contract
    for upstream consumers we synthesise an equivalent event sequence: one
    `delta` carrying the full text, then one `completed` with the usage tally.
    """
    # Deferred import keeps the x-api-key path's import surface untouched.
    from lib.hal.utils.cch import CCHError, CCHSigner

    message, model, build_kwargs = _build_cch_call_args(envelope)
    try:
        response = CCHSigner().call(
            prompt=message, oauth_token=token, model=model, **build_kwargs,
        )
    except CCHError as exc:
        yield emit_error(
            envelope.id,
            f"CCH error: {_redact(str(exc))[:500]}",
            code="provider_cch_error",
        )
        yield emit_completed(envelope.id, usage={})
        return
    except urllib.error.HTTPError as exc:
        yield from _yield_http_failure(envelope, exc)
        return
    except urllib.error.URLError as exc:
        yield from _yield_network_failure(envelope, exc)
        return
    yield from _emit_cch_response(envelope, response)


def stream_anthropic(envelope: HappiEnvelope) -> Iterator[str]:
    """Main entry: translate one envelope into a stream of HAPPI NDJSON events.

    Auth dispatch (ports cli-formatted.js:108743-108748 / TF>>>>): OAuth
    subscription tokens (`sk-ant-oat*`) route through CCH; raw API keys
    (`sk-ant-api03-*` and friends) keep the existing x-api-key + SSE path.
    """
    token = _resolve_token(envelope)
    if not token:
        yield from _yield_auth_failure(envelope)
        return

    # OAuth subscription path — Anthropic 401s every other shape, so the only
    # valid dispatch is through the CCH signer (the doctrine that earns the
    # subscription path: use the OAuth token to pretend to be a CC API call).
    # When HAL chose that credential itself, its API-key route is asked first.
    if _is_oauth_token(token):
        routed = _api_key_route_events(envelope) if _hal_chooses_credential(envelope) else None
        if routed is not None:
            yield from routed
            return
        yield from _stream_via_cch(envelope, token)
        return

    body = _build_request_body(envelope)
    timeout = int(envelope.flags.get("timeout_s") or DEFAULT_TIMEOUT_S)

    try:
        response = _open_stream(body, token, timeout)
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
