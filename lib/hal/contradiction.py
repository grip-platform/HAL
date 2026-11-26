"""HAL Contradiction Detector (Phase β W3).

For each candidate memory entry, pairs it against its top-N semantic
neighbours and asks a ``judge_fn`` whether they contradict. The judge
is injectable — in tests it's a deterministic mock; in production it
can wrap ``HALCouncil.deliberate_parallel`` over 3 local LLMs per the
preplan's H-CONTRA hypothesis. The module itself is pure.

Hypothesis **H-CONTRA**: tagging contradictions reduces stale-claim
bleed-through in subsequent responses by ≥15% on a synthetic eval
(registered separately once the measurement harness lands).

Public API:
    - ContradictionResult — outcome record with flag + supporting refs
    - detect_contradiction(entry, neighbours, *, judge_fn) -> result
    - scan(entries, *, neighbours_fn, judge_fn) -> dict[filename, result]
    - SimpleHeuristicJudge — zero-dependency fallback judge
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from lib.hal.memory_engine import MemoryEntry


JudgeFn = Callable[["MemoryEntry", "MemoryEntry"], "JudgeVerdict"]
NeighboursFn = Callable[[MemoryEntry, list[MemoryEntry]], list[MemoryEntry]]


@dataclass(frozen=True)
class JudgeVerdict:
    """One pairwise verdict from a contradiction judge."""

    contradicts: bool
    confidence: float  # 0..1; higher = judge more certain
    rationale: str = ""


@dataclass
class ContradictionResult:
    """Aggregated outcome for a single entry."""

    entry: MemoryEntry
    contradicting: list[MemoryEntry] = field(default_factory=list)
    max_confidence: float = 0.0

    @property
    def is_contradicted(self) -> bool:
        return bool(self.contradicting)


def detect_contradiction(
    entry: MemoryEntry,
    neighbours: list[MemoryEntry],
    *,
    judge_fn: JudgeFn,
) -> ContradictionResult:
    """Ask ``judge_fn`` whether ``entry`` contradicts any of ``neighbours``.

    Returns a result with a populated ``contradicting`` list (in neighbour
    order) whenever at least one pairwise verdict comes back positive.
    """
    result = ContradictionResult(entry=entry)
    for neighbour in neighbours:
        if neighbour.filename == entry.filename:
            continue
        verdict = judge_fn(entry, neighbour)
        if verdict.contradicts:
            result.contradicting.append(neighbour)
            if verdict.confidence > result.max_confidence:
                result.max_confidence = verdict.confidence
    return result


def scan(
    entries: list[MemoryEntry],
    *,
    neighbours_fn: NeighboursFn,
    judge_fn: JudgeFn,
) -> dict[str, ContradictionResult]:
    """Run contradiction detection across a corpus.

    Returns a ``{filename: ContradictionResult}`` map for every entry.
    Entries with no contradictions still appear in the map so callers
    can distinguish "not yet scanned" from "scanned, clean".
    """
    results: dict[str, ContradictionResult] = {}
    for entry in entries:
        neighbours = neighbours_fn(entry, entries)
        results[entry.filename] = detect_contradiction(
            entry, neighbours, judge_fn=judge_fn,
        )
    return results


# ---------------------------------------------------------------------------
# SimpleHeuristicJudge — zero-dep fallback
# ---------------------------------------------------------------------------
_NEGATION_PAIRS: tuple[tuple[str, str], ...] = (
    (" is ", " is not "),
    (" is ", " isn't "),
    (" are ", " are not "),
    (" are ", " aren't "),
    (" works ", " doesn't work "),
    (" works ", " does not work "),
    (" succeeds ", " fails "),
    (" passes ", " fails "),
    (" enabled ", " disabled "),
    (" true ", " false "),
    (" yes ", " no "),
)
_POLARITY_WORDS = {
    "confirmed": "refuted",
    "refuted": "confirmed",
    "proven": "disproven",
    "increase": "decrease",
    "decrease": "increase",
    "faster": "slower",
    "slower": "faster",
    "better": "worse",
    "worse": "better",
}


def _word_set(text: str) -> set[str]:
    """Case-folded word tokens for Jaccard overlap."""
    return {w for w in re.findall(r"[A-Za-z]{3,}", text.lower())}


class SimpleHeuristicJudge:
    """Pattern-based judge with no external model calls.

    Flags contradictions when two entries share significant vocabulary
    AND one of them toggles a negation or polarity marker that's absent
    in the other. Crude, but deterministic — exactly what H-CONTRA's
    regression harness needs as a baseline.
    """

    def __init__(self, *, min_overlap: float = 0.3, confidence: float = 0.55):
        self.min_overlap = min_overlap
        self.confidence = confidence

    def __call__(self, a: MemoryEntry, b: MemoryEntry) -> JudgeVerdict:
        overlap = self._jaccard(a, b)
        if overlap < self.min_overlap:
            return JudgeVerdict(False, 0.0, "insufficient overlap")
        reason = self._polarity_flip(a, b) or self._negation_flip(a, b)
        if reason:
            return JudgeVerdict(True, self.confidence, reason)
        return JudgeVerdict(False, 0.0, "no polarity / negation flip found")

    def _jaccard(self, a: MemoryEntry, b: MemoryEntry) -> float:
        wa = _word_set(a.content + " " + a.name + " " + a.description)
        wb = _word_set(b.content + " " + b.name + " " + b.description)
        if not wa or not wb:
            return 0.0
        inter = len(wa & wb)
        union = len(wa | wb)
        return inter / union if union else 0.0

    def _negation_flip(self, a: MemoryEntry, b: MemoryEntry) -> str:
        ta, tb = a.content.lower(), b.content.lower()
        for pos, neg in _NEGATION_PAIRS:
            if pos in ta and neg in tb:
                return f"negation flip: {pos.strip()!r} vs {neg.strip()!r}"
            if pos in tb and neg in ta:
                return f"negation flip: {pos.strip()!r} vs {neg.strip()!r}"
        return ""

    def _polarity_flip(self, a: MemoryEntry, b: MemoryEntry) -> str:
        ta, tb = a.content.lower(), b.content.lower()
        for word, anti in _POLARITY_WORDS.items():
            if word in ta and anti in tb:
                return f"polarity flip: {word!r} vs {anti!r}"
        return ""


__all__ = [
    "ContradictionResult",
    "JudgeVerdict",
    "SimpleHeuristicJudge",
    "detect_contradiction",
    "scan",
]
