"""HAL Quorum Shortcut (Phase γ W2, H-QUORUM).

After Phase 0 (gut-feel) of a council round, if members already
strongly agree we can skip synthesis + debate and emit a verdict from
the Phase 0 signals alone. This module ships the pure decision primitive
so ``HALCouncil.deliberate_parallel`` can call into it without owning
the threshold logic itself.

Hypothesis H-QUORUM: ~30% of council runs qualify for shortcut, saving
75% of their compute cost.

Public API:
    Vote — single member's phase-0 signal + confidence
    QuorumDecision — (shortcut, agreement_score, reason)
    agreement_score(votes) -> float in [0, 1]
    should_shortcut(votes, *, threshold=0.85, min_votes=3)
    shortcut_metadata(decision) -> dict[str, str]  (emitted to round log)
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


DEFAULT_THRESHOLD = 0.85
DEFAULT_MIN_VOTES = 3


@dataclass(frozen=True)
class Vote:
    member: str
    signal: str           # e.g. "TRUE" / "FALSE" / "INCONCLUSIVE"
    confidence: float = 1.0


@dataclass(frozen=True)
class QuorumDecision:
    shortcut: bool
    agreement_score: float
    reason: str


def agreement_score(votes: list[Vote]) -> float:
    """Return the fraction of votes matching the modal signal.

    Ties resolve deterministically (Counter.most_common returns
    insertion order on tie) but the score is still the tied count.
    Empty vote list returns 0.0.
    """
    if not votes:
        return 0.0
    tally = Counter(v.signal for v in votes)
    top_count = tally.most_common(1)[0][1]
    return top_count / len(votes)


def should_shortcut(
    votes: list[Vote],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_votes: int = DEFAULT_MIN_VOTES,
) -> QuorumDecision:
    """Decide whether to shortcut based on phase-0 votes.

    Requires BOTH a minimum vote count AND agreement at or above the
    threshold. Fails closed: insufficient votes → no shortcut.
    """
    if len(votes) < min_votes:
        return QuorumDecision(
            shortcut=False,
            agreement_score=agreement_score(votes),
            reason=f"fewer than {min_votes} votes ({len(votes)} received)",
        )
    score = agreement_score(votes)
    if score >= threshold:
        return QuorumDecision(
            shortcut=True,
            agreement_score=score,
            reason=f"agreement {score:.2f} ≥ threshold {threshold:.2f}",
        )
    return QuorumDecision(
        shortcut=False,
        agreement_score=score,
        reason=f"agreement {score:.2f} below threshold {threshold:.2f}",
    )


def shortcut_metadata(decision: QuorumDecision) -> dict[str, str]:
    """Round-log payload for every council run (shortcut yes or no)."""
    return {
        "SHORTCUT": "quorum" if decision.shortcut else "none",
        "agreement_score": f"{decision.agreement_score:.3f}",
        "reason": decision.reason,
    }


__all__ = [
    "DEFAULT_MIN_VOTES",
    "DEFAULT_THRESHOLD",
    "QuorumDecision",
    "Vote",
    "agreement_score",
    "should_shortcut",
    "shortcut_metadata",
]
