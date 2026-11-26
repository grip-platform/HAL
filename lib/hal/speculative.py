"""HAL Speculative Decoding (Phase γ W4, H-SPEC).

A local draft model generates N tokens cheaply; a subscription verify
model validates in one pass. Accept-rate >= 0.7 → commit, else fall
back to direct verify. This module ships the pure decision primitive
(acceptance ratio + commit / fallback verdict); the transport-layer
orchestration wraps a real draft+verify dispatch in a follow-up PR.

Hypothesis H-SPEC: 1K-token responses drop from ~30s (direct subscription
decode) to ~4s (local draft + single-shot verify) at zero marginal
cost — threshold 0.7 acceptance.

Public API:
    AcceptanceResult — (accept_count, total, ratio, committed)
    accept_ratio(draft_tokens, verify_tokens) -> float
    decide_commit(draft, verify, *, threshold=0.7) -> AcceptanceResult
"""
from __future__ import annotations

from dataclasses import dataclass


DEFAULT_THRESHOLD = 0.7


@dataclass(frozen=True)
class AcceptanceResult:
    accept_count: int
    total: int
    ratio: float
    committed: bool


def accept_ratio(draft: list[str], verify: list[str]) -> float:
    """Return fraction of draft tokens accepted by verifier (prefix match).

    Empty draft or verify → 0.0. Uses longest common prefix semantics
    — a divergence anywhere in the stream stops the accepted count,
    matching how real speculative decoders resume from the first mismatch.
    """
    if not draft or not verify:
        return 0.0
    accepted = 0
    for d, v in zip(draft, verify):
        if d != v:
            break
        accepted += 1
    return accepted / len(draft)


def decide_commit(
    draft: list[str],
    verify: list[str],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> AcceptanceResult:
    """Compute acceptance and decide whether to commit the speculative path.

    Committed iff ratio >= threshold AND draft is non-empty. Empty drafts
    can never commit — that's the direct-verify fallback.
    """
    if not draft:
        return AcceptanceResult(accept_count=0, total=0, ratio=0.0, committed=False)
    ratio = accept_ratio(draft, verify)
    accept_count = 0
    for d, v in zip(draft, verify):
        if d != v:
            break
        accept_count += 1
    return AcceptanceResult(
        accept_count=accept_count,
        total=len(draft),
        ratio=ratio,
        committed=ratio >= threshold,
    )


__all__ = [
    "AcceptanceResult",
    "DEFAULT_THRESHOLD",
    "accept_ratio",
    "decide_commit",
]
