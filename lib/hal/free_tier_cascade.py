"""HAL Free-Tier Cascade (Phase γ W5, H-FREECAS).

Ordered fallback list for ``quality_cost=simple`` traffic. Tries free
providers first; cascades down to cheap only when every free tier
returns a rate-limit or transient error. Pure decision primitive —
transport-layer wiring calls ``next_provider`` after each failure.

Council gate note: the preplan requires a broly MESH deliberation on
the rate-limit tolerance claim BEFORE live rollout. Local LLMs can't
meaningfully judge real Gemini/DeepSeek/Groq rate limits without
traffic data, so the council pass is deferred to a follow-up PR that
runs against real usage telemetry. This PR ships the cascade primitive
+ tests; activation requires a rate-limit council pass.

Public API:
    CascadeTier, DEFAULT_CASCADE, RateLimitError, next_provider
"""
from __future__ import annotations

from dataclasses import dataclass


class RateLimitError(RuntimeError):
    """Raised by transport when a tier returns 429/quota exhausted."""


@dataclass(frozen=True)
class CascadeTier:
    provider: str
    model: str
    cost_cls: str  # "free" | "cheap" | "mid" | "premium"


# Ordered: free tiers first, cheap fallback, mid ultimate fallback.
# Operator customises via HAL_FREE_CASCADE env var (see next_provider).
DEFAULT_CASCADE: tuple[CascadeTier, ...] = (
    CascadeTier("gemini", "gemini-3.5-flash-lite", "free"),
    CascadeTier("deepseek", "deepseek-chat", "free"),
    CascadeTier("groq", "llama-3.3-70b-versatile", "free"),
    CascadeTier("groq", "llama-3.1-8b-instant", "cheap"),
    CascadeTier("openai", "gpt-4o-mini", "mid"),
)


def next_provider(
    current: CascadeTier | None,
    *,
    cascade: tuple[CascadeTier, ...] = DEFAULT_CASCADE,
) -> CascadeTier | None:
    """Return the next tier in ``cascade`` after ``current``, or None at end.

    Called by transport after a ``RateLimitError``. Starting call
    passes ``current=None`` to get the first tier. Callers terminate
    the loop when this returns None.
    """
    if not cascade:
        return None
    if current is None:
        return cascade[0]
    try:
        idx = cascade.index(current)
    except ValueError:
        return None  # Unknown current — terminate to avoid surprise
    if idx + 1 >= len(cascade):
        return None
    return cascade[idx + 1]


def free_only(cascade: tuple[CascadeTier, ...] = DEFAULT_CASCADE) -> tuple[CascadeTier, ...]:
    """Return only the free-tier slice of the cascade (for budget-lock mode)."""
    return tuple(t for t in cascade if t.cost_cls == "free")


__all__ = [
    "CascadeTier",
    "DEFAULT_CASCADE",
    "RateLimitError",
    "free_only",
    "next_provider",
]
