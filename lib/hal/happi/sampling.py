"""Optional sampling-parameter resolution for HAPPI provider adapters.

Shared by the gemini / openai_compat / anthropic adapters so the
flag-extraction + type-coercion + presence contract is identical across
providers — one contract, three wire mappings (DRY; the three call sites
justify the abstraction per YSH).

Contract: read optional ``temperature``, ``top_p``, ``seed`` from
``envelope.flags``. A flag that is ABSENT (or explicitly ``null``) is OMITTED
from the result, so each adapter's request body is byte-identical to its
pre-flag behaviour when the caller passes none of them — backwards-compatible
by construction. This is the property AGORA relies on to set
``temperature: 0.0`` for deterministic judge verdicts without perturbing any
other call site (CodeTonight-SA/GRIP AGORA non-determinism, PR #3318 sibling).

The presence test is ``is not None`` — NOT truthiness — because the single most
important value here is ``temperature: 0.0`` (full determinism), and ``0.0`` is
falsy. A ``if temperature:`` guard would silently drop the one value the whole
feature exists to carry.

Canonical (returned) keys are snake_case: ``temperature`` (float), ``top_p``
(float), ``seed`` (int). Adapters whose wire name differs (Gemini's camelCase
``topP``) remap on the way into their request body.

``seed`` is gated by ``allow_seed`` because the Anthropic Messages API does not
accept a ``seed`` field — passing it would surface as a 400. Gemini
(``generationConfig.seed``) and the OpenAI chat-completions family both accept
it, so they call with ``allow_seed=True``.
"""
from __future__ import annotations

from typing import Any, Callable, Optional


def _coerce(value: Any, caster: Callable[[Any], Any]) -> Optional[Any]:
    """Cast ``value`` via ``caster``; return None if absent or not coercible.

    Fail-safe by design: a malformed optional knob (``temperature: "hot"``,
    a list, a dict) is treated as ABSENT — the same well-defined path as not
    setting the flag at all, which falls back to the provider default. A stray
    or fat-fingered tuning flag must never crash a whole inference with an
    uncaught ``ValueError`` / ``TypeError`` mid-stream. This is input
    validation of an optional knob, not the swallowing of an internal bug, so
    fail-safe (omit) is correct rather than fail-loud.

    Note: numeric strings still coerce (``float("0")`` -> 0.0, ``int("7")``
    -> 7) and ``int(3.7)`` truncates to 3 without error — only genuinely
    non-numeric input is omitted.
    """
    if value is None:
        return None
    try:
        return caster(value)
    except (ValueError, TypeError):
        return None


def _coerce_seed(value: Any) -> Optional[int]:
    """Coerce a seed to int, rejecting non-integer-valued input.

    Unlike a bare ``int()``, a fractional float (``3.7``) or fractional string
    (``"3.7"``) is treated as MALFORMED (omitted), not silently truncated to
    ``3``. A seed is conceptually an exact integer used for reproducibility, so
    a fractional seed is a caller error — quietly rounding it would hand the
    provider a different seed than asked for (silent wrong value). Booleans are
    rejected too: ``True`` is an ``int`` subclass, but a boolean seed is
    nonsensical and would otherwise resolve to ``1``.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        return _coerce(value, int)  # "7" -> 7; "3.7"/"x" -> None (not int-parseable)
    return None


def resolve_sampling(
    flags: dict[str, Any], *, allow_seed: bool = True
) -> dict[str, Any]:
    """Extract present-only sampling params from ``flags``.

    Returns a dict carrying only the keys the caller actually set (and could
    coerce), using the canonical snake_case names ``temperature`` / ``top_p``
    / ``seed``. Absent OR malformed keys are omitted (provider default
    preserved — see :func:`_coerce`). ``seed`` is included only when
    ``allow_seed`` is True.

    Range is deliberately NOT validated here: valid ranges differ per provider
    (Gemini 0-2, Anthropic 0-1, ...), so clamping would couple this shared
    helper to per-provider knowledge AND silently rewrite an explicit caller
    value. A numeric-but-out-of-range value is passed through so the provider
    returns its own clear 400 rather than HAL second-guessing the caller.
    """
    out: dict[str, Any] = {}

    temperature = _coerce(flags.get("temperature"), float)
    if temperature is not None:
        out["temperature"] = temperature

    top_p = _coerce(flags.get("top_p"), float)
    if top_p is not None:
        out["top_p"] = top_p

    if allow_seed:
        seed = _coerce_seed(flags.get("seed"))
        if seed is not None:
            out["seed"] = seed

    return out
