"""HAL RAG-aware deliberation prompt composer (Phase β W4, H-RAG_DELIB).

Pure-function utility for building council phase prompts with retrieval
context surgically placed in Phase 0 only (gut-feel / initial take),
never in synthesis or debate. This is the architectural invariant of
H-RAG_DELIB: if kono memories leak into the adversarial phases they
bias the council toward the retrieved frame instead of independent
reasoning from it.

Hypothesis H-RAG_DELIB: constraining RAG injection to Phase 0 improves
the council's factuality + agreement_score delta by >=10% on the
50-question eval (registered separately once the eval harness ships).

Public API:
    Phase (enum-like str constants): GUT_FEEL, SYNTHESIS, DEBATE, VERDICT
    RagContext — recall payload + config knobs
    compose_phase_prompt(base, phase, *, rag) -> str
    injection_spec() -> dict[str, bool] — authoritative map of
        "does this phase receive RAG?" for test assertion.

HALCouncil.deliberate in a follow-up PR will call compose_phase_prompt
on each round; this PR ships the composer + per-phase invariants +
tests so the wiring becomes a surgical 3-line change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


# ---------------------------------------------------------------------------
# Phase names — string constants so consumers needn't import an Enum just to
# pass a string through a dict.
# ---------------------------------------------------------------------------
class Phase:
    GUT_FEEL = "gut_feel"
    SYNTHESIS = "synthesis"
    DEBATE = "debate"
    VERDICT = "verdict"


_PHASES: tuple[str, ...] = (Phase.GUT_FEEL, Phase.SYNTHESIS, Phase.DEBATE, Phase.VERDICT)


def injection_spec() -> dict[str, bool]:
    """Authoritative per-phase RAG injection rule.

    Only ``Phase.GUT_FEEL`` receives retrieved memory. All other phases
    reason unaided so the RAG frame doesn't contaminate the adversarial
    dynamics of synthesis + debate.
    """
    return {
        Phase.GUT_FEEL: True,
        Phase.SYNTHESIS: False,
        Phase.DEBATE: False,
        Phase.VERDICT: False,
    }


@dataclass
class RagContext:
    """RAG context passed to each phase (injected only where allowed)."""

    recalled: list[str] = field(default_factory=list)
    header: str = "# Relevant memory (initial take only)"
    footer: str = (
        "Use the recall above only to inform your FIRST impression. "
        "Subsequent phases reason without it."
    )
    max_entries: int = 5  # trim to prevent prompt bloat


def _truncate(entries: Iterable[str], limit: int) -> list[str]:
    """Keep the first ``limit`` entries; drop the rest with a marker row."""
    picked = list(entries)
    if len(picked) <= limit:
        return picked
    return picked[:limit] + [f"(+{len(picked) - limit} more recall entries truncated)"]


def _format_recall_block(rag: RagContext) -> str:
    """Render the recall block; empty string when there is nothing to render."""
    picked = _truncate(rag.recalled, rag.max_entries)
    if not picked:
        return ""
    body = "\n".join(f"- {item}" for item in picked)
    return f"{rag.header}\n\n{body}\n\n{rag.footer}\n"


def compose_phase_prompt(
    base: str,
    phase: str,
    *,
    rag: RagContext | None = None,
) -> str:
    """Return the per-phase prompt, injecting RAG context iff the phase allows.

    ``phase`` must be one of the ``Phase.*`` constants. Anything else is
    treated as a new, unknown phase and defaults to NO injection (fail
    closed) — the safer default for a contamination gate.
    """
    allowed = injection_spec().get(phase, False)
    if not allowed or rag is None:
        return base
    block = _format_recall_block(rag)
    if not block:
        return base
    return f"{block}\n{base}"


def all_phases() -> tuple[str, ...]:
    """Ordered phase tuple for iteration in tests and callers."""
    return _PHASES


__all__ = [
    "Phase",
    "RagContext",
    "all_phases",
    "compose_phase_prompt",
    "injection_spec",
]
