"""HAPPI ``infer`` cmd -- the GRIP LLM syscall through the RAILLM ladder.

This is "happi.md as the shim" for GRIP's own inference. Where the existing
``chat`` cmd routes to ONE provider by ``flags.provider`` (no failover), the
``infer`` cmd dispatches through ``RAILLMSentinel.dispatch_with_history``
over the W2 ``build_grip_ladder()`` (CCH -> Kimi/Qwen -> cheap -> local),
resolving subscription rungs via the W1 ``register_subscriptions`` bridge.
History is carried + context-window-compressed across rungs by RAILLM, so
GRIP is no longer bound to one provider's context window.

Audit is FREE: the runtime's ``_stream_with_audit`` wraps every
``_dispatch_core`` handler and emits the deterministic ``idr`` terminator
when ``flags.audit is True``. This handler therefore yields only
``delta``/``completed``/``error`` (no ``started`` -- runtime emits it; no
``idr`` -- runtime appends it). DRY: the audit chain is reused, not
re-implemented (H-RAILLM-3).

``make_grip_sentinel`` is the dependency-injection seam: tests monkeypatch
it to return a sentinel wired with fakes so ``infer`` is exercised through
the REAL runtime without network or keychain access.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Iterator

from lib.hal.happi.envelope import HappiEnvelope
from lib.hal.happi.events import emit_completed, emit_delta, emit_error
from lib.hal.raillm import RAILLMExhausted, RAILLMSentinel
from lib.hal.raillm_ladder import GRIP_LADDER_ADDRESSES, build_grip_ladder
from lib.hal.raillm_subscription_bridge import register_subscriptions

# ---------------------------------------------------------------------------
# Multimodal capability guard (fail-CLOSED — unknown model => reject)
#
# Conservative allowlist: only models with confirmed audio/image support.
# The ladder passes PROVIDER-QUALIFIED addresses ("anthropic/claude-opus-4-7",
# "openai/gpt-6-astra") while callers may also pass bare ids
# ("gpt-4o-2024-05-13"), so matching is two-tier:
#
#  * BARE entries (no "/") match the MODEL-NAME part of an address — the
#    segment after the provider slash — by prefix, so dated variants
#    resolve ("openai/gpt-4o" and "gpt-4o-2024-05-13" both pass).
#  * QUALIFIED entries (with "/") match the full lowercased address
#    EXACTLY, never by prefix — an unlisted suffixed id ("…-preview-xyz")
#    keeps failing closed.
#
# Anything NOT matched is assumed TEXT-ONLY and will raise on non-text
# content_parts. Unknown capability NEVER silently passes.
# ---------------------------------------------------------------------------
_MULTIMODAL_BARE_PREFIXES: tuple[str, ...] = (
    "gemini",          # Google Gemini (native audio/image/video)
    "claude-3",        # Anthropic Claude 3 family (image capable)
    "claude-opus-4",   # Anthropic Claude 4 Opus (image capable)
    "claude-sonnet-4", # Anthropic Claude 4 Sonnet (image capable)
    "gpt-4o",          # OpenAI GPT-4o (audio + image)
    "gpt-4-turbo",     # OpenAI GPT-4 Turbo (image capable)
)

_MULTIMODAL_QUALIFIED: frozenset[str] = frozenset({
    # Moonshot's live lineup: GET https://api.moonshot.ai/v1/models reports
    # supports_image_in: true and supports_video_in: true for all four
    # (2026-09-07). Full addresses, not a blanket "moonshot/", so a retired
    # or unlisted id keeps failing closed.
    "moonshot/kimi-k3",
    "moonshot/kimi-k2.7-code",
    "moonshot/kimi-k2.7-code-highspeed",
    "moonshot/kimi-k2.6",
    # GPT-6 Astra: vision verified live 2026-09-12 through the HAL gateway
    # on :4010 — four-band colour image at 1-in-24 blind odds, with a
    # no-image control that correctly refused to guess.
    "openai/gpt-6-astra",
})

# Combined view — callers and the regression anchor reference the allowlist
# as one surface; a bare prefix and its qualified exact form coexist.
_MULTIMODAL_PREFIXES: tuple[str, ...] = (
    _MULTIMODAL_BARE_PREFIXES + tuple(sorted(_MULTIMODAL_QUALIFIED))
)


def _has_non_text_parts(content_parts: list[dict]) -> bool:
    """Return True if any part is audio or image (i.e. not plain text)."""
    return any(p.get("type") in ("audio", "image", "image_url")
               for p in content_parts)


def _matches_multimodal_allowlist(addr_lower: str) -> bool:
    """Qualified entries match the full address exactly; bare entries match
    the model-name part by prefix so dated variants resolve."""
    if "/" in addr_lower and addr_lower in _MULTIMODAL_QUALIFIED:
        return True
    model_part = addr_lower.rsplit("/", 1)[-1]
    return any(model_part.startswith(p) for p in _MULTIMODAL_BARE_PREFIXES)


def _assert_multimodal_capable(model_address: str,
                                content_parts: list[dict]) -> None:
    """Raise ValueError if the model cannot handle non-text parts.

    Fail-CLOSED: an unrecognised model address is treated as text-only.
    The text-only path is never touched by this function.
    """
    if not _has_non_text_parts(content_parts):
        return  # text-only parts — no modality constraint
    addr_lower = model_address.lower()
    if not _matches_multimodal_allowlist(addr_lower):
        raise ValueError(
            f"infer: model '{model_address}' is not known to support audio/image "
            "content_parts. Add it to _MULTIMODAL_PREFIXES in cmds_infer.py "
            "once its multimodal capability is confirmed. "
            "Failing closed — no silent ok:true for unknown capability."
        )
from lib.hal.registry import parse_model_address
from lib.hal.types import ChatMessage


def make_grip_sentinel() -> RAILLMSentinel:
    """Build a RAILLMSentinel wired with the GRIP ladder + subscriptions.

    Singleton registry (providers.yaml) augmented with auto-detected CCH /
    Kimi / Qwen subscriptions (W1 bridge), cascading over the W2 ladder.
    Monkeypatch this in tests to inject fakes.
    """
    from lib.hal.registry import get_registry

    registry = get_registry()
    register_subscriptions(registry)
    return RAILLMSentinel(registry=registry, equivalence_map=build_grip_ladder())


def _extract_prompt(envelope: HappiEnvelope) -> str:
    """Prompt is args[0] (positional) or flags.prompt (named)."""
    if envelope.args and isinstance(envelope.args[0], str):
        return envelope.args[0]
    prompt = (envelope.flags or {}).get("prompt")
    return prompt if isinstance(prompt, str) else ""


def _extract_content_parts(envelope: HappiEnvelope) -> list[dict] | None:
    """Multimodal ContentPart list from ``flags.content_parts`` (OpenAI dicts:
    text + image_url + input_audio), else None. Absent -> the text-only string
    path, byte-identical to pre-multimodal behaviour."""
    parts = (envelope.flags or {}).get("content_parts")
    return parts if isinstance(parts, list) and parts else None


def _budget_kwargs(max_tokens: int | None) -> dict:
    """``max_tokens`` as a dispatch keyword, or nothing. Omitted rather than
    passed as None, which would override the provider's own default."""
    return {} if max_tokens is None else {"max_tokens": max_tokens}


def _pin_kwargs(pinned: bool) -> dict:
    """``no_failover`` as a dispatch keyword for a pinned turn, or nothing, so
    an unpinned dispatch receives exactly the arguments it always did."""
    return {"no_failover": True} if pinned else {}


def _dispatch_multimodal(sentinel: RAILLMSentinel, model_address: str,
                         prompt: str, content_parts: list[dict],
                         max_tokens: int | None = None, no_failover: bool = False):
    """Dispatch a multimodal turn via the inherited ``LimitSentinel.dispatch``
    messages path -- the SAME carrier the chat pipeline uses for images. The
    string-thread RAILLM cascade (``dispatch_with_history``) flattens to a
    string and CANNOT carry parts, so multimodal rides ``ChatMessage``
    list-content, which the provider layer passes to the request body as-is
    (see :class:`~lib.hal.types.ChatMessage`). Single-provider limit-aware
    failover; ``no_failover`` (a pinned turn) tries the requested provider
    alone. ``prompt`` is already the trailing text part inside
    ``content_parts`` (built by ``build_multimodal_content``); passing it
    positionally is harmless -- OpenAI-compat providers prefer ``messages``
    when present."""
    messages = [ChatMessage(role="user", content=content_parts)]
    return sentinel.dispatch(model_address, prompt or "", messages=messages,
                             **_budget_kwargs(max_tokens), **_pin_kwargs(no_failover))


# HAPPI's name for each side, and the names a provider may use for it: OpenAI's
# first, Anthropic's second.
_COUNT_NAMES = {"in": ("prompt_tokens", "input_tokens"),
                "out": ("completion_tokens", "output_tokens")}


def _token_counts(response) -> dict[str, int]:
    """The provider's token counts under HAPPI's names; a side it did not report is left out.

    Reads ``response.usage`` as an object (``TokenUsage``) or as any mapping, under
    either naming. HAL's providers all normalise to OpenAI's names before this
    point: ``TokenUsage`` everywhere it is built, and the dicts that
    ``lib/providers/groq.py`` and ``lib/providers/gemini.py`` build (gemini maps
    ``promptTokenCount`` to ``prompt_tokens`` and counts thinking as completion).

    A side is kept only as a positive whole number. ``TokenUsage`` defaults every
    field to 0 and both dict builders write ``usage.get(..., 0)``, so a zero cannot
    be told apart from silence on any current path; leaving it out changes no price.
    Without these counts the completed event carried none, and GRIP recorded zero
    tokens for every grip_infer call (1,027 of 1,027 in the ten days to 2026-10-02).
    """
    usage = getattr(response, "usage", None)
    counts = {side: _first_count(usage, names) for side, names in _COUNT_NAMES.items()}
    return {side: value for side, value in counts.items() if value}


def _first_count(usage, names: tuple[str, ...]) -> int:
    """The first of ``names`` on ``usage`` holding a positive whole number, else 0."""
    for name in names:
        value = usage.get(name) if isinstance(usage, Mapping) else getattr(usage, name, None)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isfinite(value) and value > 0 and value == int(value):
            return int(value)
    return 0


def _emit_success(envelope: HappiEnvelope, model_address: str,
                  response) -> Iterator[str]:
    """Stream the success path: delta + completed(usage with failover).

    RAILLM's ``dispatch_with_history`` (unlike ``sentinel.dispatch``) does
    NOT set ``response.fallback_info`` -- failover here is derived from the
    provider that actually answered vs the one requested. A sibling-model
    rung is recorded by ``_with_served_rung`` and refused under pin by
    ``_pin_violation``; unpinned, it is reported as before.
    """
    yield emit_delta(envelope.id, response.text)
    usage = {
        "provider": response.provider,
        "model": response.model,
        "requested": model_address,
        **_token_counts(response),
    }
    requested_provider, _ = parse_model_address(model_address)
    if response.provider and response.provider != requested_provider:
        usage["failover"] = {
            "requested": requested_provider,
            "served_by": response.provider,
        }
    # HAL#648: carry the provider's finish_reason so the fold can surface an
    # empty answer + "length"/"max_tokens" as answer_budget_exhausted instead
    # of a silent ok=true empty success. Absent (older provider object) ->
    # None, byte-identical to before.
    yield emit_completed(
        envelope.id, usage=usage,
        finish_reason=getattr(response, "finish_reason", None),
    )


def _with_served_rung(response, sentinel: RAILLMSentinel, model_address: str):
    """Record on a text turn's response which rung actually served it.

    ``dispatch_with_history`` can hand a failed address to a same-provider
    tier peer, and ``response.model`` is the API's echo, which can be a dated
    snapshot id, so neither tells a sibling model from the requested one. The
    thread's last assistant message names the exact rung the ladder called.
    When that rung is not the requested one it becomes ``fallback_info``, the
    field ``LimitSentinel.dispatch`` already sets for multimodal turns. A rung
    that cannot be known leaves the response unchanged.
    """
    from dataclasses import is_dataclass, replace
    from lib.hal.types import FallbackInfo

    messages = sentinel.thread.messages if hasattr(sentinel, "thread") else []
    last = messages[-1] if messages else None
    if last is None or last.role != "assistant" or not last.model_hint:
        return response
    provider, model = parse_model_address(model_address)
    if (last.provider_hint, last.model_hint) == (provider, model) or not is_dataclass(response):
        return response
    return replace(response, fallback_info=FallbackInfo(
        requested_provider=provider or "", requested_model=model or "",
        actual_provider=last.provider_hint or "", actual_model=last.model_hint,
    ))


def _resolve_response(model_address: str, prompt: str,
                      content_parts: list[dict] | None,
                      max_tokens: int | None = None, *, pinned: bool = False):
    """Run the dispatch; return ``(response, None)`` on success or
    ``(None, (message, code))`` on failure. Kept free of the event stream so
    ``dispatch_infer`` stays a thin coordinator (KISS). Multimodal turns
    (``content_parts`` present) ride the messages path; text turns ride the
    string-thread RAILLM cascade. ``max_tokens`` (optional) raises the answer
    budget for a turn that would otherwise exhaust the default on reasoning.
    ``pinned`` tries the requested rung alone on either path (HAL#673)."""
    try:
        if content_parts is not None:
            _assert_multimodal_capable(model_address, content_parts)
        sentinel = make_grip_sentinel()
    except Exception as exc:  # noqa: BLE001 -- any setup failure is runtime_error
        return None, (f"infer: dispatch failed -- {type(exc).__name__}: {exc}",
                      "runtime_error")
    try:
        return _dispatch(sentinel, model_address, prompt, content_parts,
                         max_tokens, pinned), None
    except Exception as exc:  # noqa: BLE001 -- classified by _dispatch_failure
        return None, _dispatch_failure(exc, model_address, pinned)


def _dispatch(sentinel: RAILLMSentinel, model_address: str, prompt: str,
              content_parts: list[dict] | None, max_tokens: int | None,
              pinned: bool):
    """One dispatch on the turn's path; a pinned turn tries its own rung only."""
    if content_parts is not None:
        return _dispatch_multimodal(sentinel, model_address, prompt, content_parts,
                                    max_tokens=max_tokens, no_failover=pinned)
    response = sentinel.dispatch_with_history(
        model_address, prompt, **_budget_kwargs(max_tokens), **_pin_kwargs(pinned))
    return _with_served_rung(response, sentinel, model_address)


def _dispatch_failure(exc: Exception, model_address: str,
                      pinned: bool) -> tuple[str, str]:
    """``(message, code)`` for a dispatch that raised.

    Pinned, any failure means the pinned rung could not serve, and no other
    rung was tried, so the code is ``provider_pin_violation``: GRIP's client
    sends that code on to HAL's gateway, which may still serve the pinned model.
    """
    if pinned:
        return (f"infer: pinned '{model_address}' could not serve this request, "
                f"and no other model or provider was tried -- "
                f"{type(exc).__name__}: {exc}. Pass pin=false to allow failover.",
                "provider_pin_violation")
    if isinstance(exc, RAILLMExhausted):
        return f"infer: every ladder rung exhausted -- {exc}", "runtime_error"
    return (f"infer: dispatch failed -- {type(exc).__name__}: {exc}",
            "runtime_error")


def _pin_violation(envelope: HappiEnvelope, model_address: str,
                   response) -> str | None:
    """Plain-language error iff the caller PINNED the model (``flags.pin``
    true) but the answer reports another provider, or (HAL#649) another model
    of the same provider -- else ``None``.

    Why this gate exists: an EXOGENOUS council needs the provider it asked for
    ({gemini + openai} is only diverse if openai genuinely lands on openai),
    so a pinned request must return the pinned model OR fail loudly -- never
    another one mislabelled as the answer. Since HAL#673 a pinned turn tries
    its own rung alone, so the cascade can no longer hand it to another rung
    (that failure is ``_dispatch_failure``'s). This check stays as the
    backstop for an answer that reports a provider or model other than the
    one asked for.

    Opt-in by construction: ``pin`` absent / false -> ``None`` -> byte-identical
    failover (the GRIP default ladder, which WANTS cross-provider failover, is
    untouched). Only a caller that explicitly sets ``flags.pin`` opts into the
    strict no-failover guarantee.
    """
    if not (envelope.flags or {}).get("pin"):
        return None
    requested_provider, requested_model = parse_model_address(model_address)
    served = getattr(response, "provider", None)
    if requested_provider and served and served != requested_provider:
        return (
            f"infer: pinned provider '{requested_provider}' was asked, but the "
            f"answer reports '{served}'. Refusing to report '{served}' as the "
            f"result of a pinned '{model_address}' call (an exogenous council "
            f"needs the provider it asked for, or a clear failure -- not a "
            f"different provider). Pass pin=false to allow failover."
        )
    # HAL#649: same provider, different model. Only the rung the ladder called
    # counts (fallback_info, set by _with_served_rung), never the model id the
    # API echoes back.
    info = getattr(response, "fallback_info", None)
    actual = getattr(info, "actual_model", "") or ""
    if requested_model and actual and actual != requested_model:
        return (
            f"infer: pinned model '{model_address}' was asked, but the answer "
            f"came from '{info.actual_provider}/{actual}', another model. "
            f"Refusing to report it as the result of a pinned call. Pass "
            f"pin=false to allow failover."
        )
    return None


def answer_budget(value: object) -> int | None:
    """The ``max_tokens`` answer budget a caller asked for, or None.

    HAL#648: a reasoning model can spend the provider's default budget on its
    reasoning and never start the answer, so a caller may raise it. Only a
    positive int is a budget. A bool, a float, a string, zero or a negative
    resolves to None and the provider default applies, never a fabricated
    budget. This is the one rule for both ``/api/infer`` and a raw envelope's
    ``flags.max_tokens``.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _refusal(envelope: HappiEnvelope, prompt: str,
             content_parts: list[dict] | None,
             model_address: str) -> tuple[str, str] | None:
    """``(message, code)`` refusing the turn before dispatch, else None.

    No prompt and no parts is a parse_error. HAL#649: a pinned request means
    "this model or nothing", and a bare id with no provider prefix (e.g.
    "deepseek-v4-pro") is not a catalogue address. The ladder would resolve it
    to whatever rung it can reach and report ok=true under the wrong name, so
    it is refused as unknown_model.
    """
    if content_parts is None and not prompt:
        return ("infer: requires a prompt (args[0] or flags.prompt) "
                "or flags.content_parts", "parse_error")
    if (envelope.flags or {}).get("pin") and not parse_model_address(model_address)[0]:
        return (f"infer: unknown model '{model_address}' — a pinned request must "
                "name a provider-qualified catalogue address (e.g. "
                "'deepseek/deepseek-v4-pro'); a bare id has no provider to pin, "
                "so HAL would silently substitute another rung and report it as "
                "the requested one. Pass the qualified address, or pin=false to "
                "allow failover.", "unknown_model")
    return None


def dispatch_infer(envelope: HappiEnvelope) -> Iterator[str]:
    """Dispatch one inference through the RAILLM InfiniteMachine ladder.

    Coordinator: refuse -> dispatch -> pin-check -> stream. ``flags.model`` is
    a RAILLM model address (default: ladder primary); ``flags.audit=true``
    makes the runtime append the deterministic idr terminator.
    ``flags.pin=true`` means "this model or nothing": a bare id is refused, the
    pinned rung alone is tried, so the prompt never reaches another provider
    (HAL#673), and its failure is ``provider_pin_violation``, never a
    mislabelled success. ``flags.content_parts`` makes a multimodal turn on the
    messages path. ``flags.max_tokens`` (HAL#648) raises the answer budget, by
    :func:`answer_budget`'s rule.
    """
    prompt = _extract_prompt(envelope)
    content_parts = _extract_content_parts(envelope)
    flags = envelope.flags or {}
    model_address = str(flags.get("model") or GRIP_LADDER_ADDRESSES[0])
    refusal = _refusal(envelope, prompt, content_parts, model_address)
    if refusal is not None:
        yield emit_error(envelope.id, refusal[0], code=refusal[1])
        return
    response, error = _resolve_response(
        model_address, prompt, content_parts,
        max_tokens=answer_budget(flags.get("max_tokens")),
        pinned=bool(flags.get("pin")))
    if error is not None:
        yield emit_error(envelope.id, error[0], code=error[1])
        return
    pin_error = _pin_violation(envelope, model_address, response)
    if pin_error is not None:
        yield emit_error(envelope.id, pin_error, code="provider_pin_violation")
        return
    yield from _emit_success(envelope, model_address, response)


# Extension dispatch table -- merged by runtime._dispatch_core via the same
# .get() lookup pattern as V11_HANDLERS (no if/elif chain). Kept OUT of
# cmds_v11.py so that module stays byte-parity with the canonical polyglot
# happi.md runtime (which has no `infer` cmd).
INFER_HANDLERS = {"infer": dispatch_infer}
INFER_COMMANDS = frozenset(INFER_HANDLERS)
