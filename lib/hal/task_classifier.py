"""HAL Task Classifier (Phase γ W1, H-CLASSIFY).

Pre-dispatch classification of an incoming prompt into one of four
classes — ``draft``, ``reason``, ``verify``, ``vision`` — so the cost
router can pick the cheapest tier that satisfies the task.

DI-clean: ``classify_fn`` is injected so production wraps a Groq free-
tier Llama-3.3 call (per preplan) while tests and offline paths use
``HeuristicClassifier`` (pure-stdlib keyword + length heuristics). Also
logs mis-classifications as JSONL for nightly distillation.

Hypothesis H-CLASSIFY: mis-route rate ≤5% after 200 classifications,
payback ≥4000× per mis-route avoided.

Public API:
    TaskClass — {DRAFT, REASON, VERIFY, VISION}
    Classification — (task_class, confidence, rationale)
    ClassifyFn — callable protocol
    HeuristicClassifier — pure-stdlib fallback
    DISPATCH_TIER — authoritative class → cost-tier map
    dispatch_tier(classification) -> tier
    log_miss(path, expected, actual, prompt) — JSONL miss logger
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Callable


class TaskClass:
    DRAFT = "draft"
    REASON = "reason"
    VERIFY = "verify"
    VISION = "vision"


ALL_CLASSES: tuple[str, ...] = (
    TaskClass.DRAFT, TaskClass.REASON, TaskClass.VERIFY, TaskClass.VISION,
)


DISPATCH_TIER: dict[str, str] = {
    TaskClass.DRAFT: "free",
    TaskClass.REASON: "cheap",
    TaskClass.VERIFY: "premium",
    TaskClass.VISION: "mid",
}


@dataclass(frozen=True)
class Classification:
    task_class: str
    confidence: float  # 0..1
    rationale: str = ""


ClassifyFn = Callable[[str], Classification]


_VERIFY_KEYWORDS = (
    "prove", "proof", "verify", "formal", "correctness",
    "security review", "audit", "falsify", "hypothesis",
)
_REASON_KEYWORDS = (
    "why", "explain", "architecture", "design", "refactor",
    "strategy", "trade-off", "tradeoff", "analyse", "analyze",
)
_VISION_KEYWORDS = (
    "image", "screenshot", ".png", ".jpg", ".jpeg", ".webp",
    "vision", "visual", "diagram", "photo",
)


def _count_hits(text: str, terms: tuple[str, ...]) -> int:
    return sum(1 for t in terms if t in text)


class HeuristicClassifier:
    """Zero-dependency classifier over keyword hits + prompt length."""

    def __init__(self, *, short_len: int = 60, confidence: float = 0.6):
        self.short_len = short_len
        self.confidence = confidence

    def __call__(self, prompt: str) -> Classification:
        if not prompt:
            return Classification(TaskClass.DRAFT, 0.0, "empty prompt")
        lower = prompt.lower()
        if _count_hits(lower, _VISION_KEYWORDS) >= 1:
            return Classification(TaskClass.VISION, self.confidence, "vision keyword")
        if _count_hits(lower, _VERIFY_KEYWORDS) >= 1:
            return Classification(TaskClass.VERIFY, self.confidence, "verify keyword")
        if _count_hits(lower, _REASON_KEYWORDS) >= 1:
            return Classification(TaskClass.REASON, self.confidence, "reason keyword")
        if len(prompt) <= self.short_len:
            return Classification(TaskClass.DRAFT, self.confidence, "short prompt")
        return Classification(TaskClass.REASON, self.confidence * 0.7, "default: long prompt")


def dispatch_tier(classification: Classification) -> str:
    """Return the cost tier for a classification; DRAFT is the safe fallback."""
    return DISPATCH_TIER.get(classification.task_class, "cheap")


def log_miss(
    path: str,
    *,
    expected: str,
    actual: str,
    prompt: str,
    clock: Callable[[], float] = time.time,
) -> None:
    """Append one JSONL miss record. Creates parent dir if needed."""
    if not path:
        return
    full = os.path.expanduser(path)
    parent = os.path.dirname(full)
    if parent:
        os.makedirs(parent, exist_ok=True)
    record = {
        "timestamp": clock(),
        "expected": expected,
        "actual": actual,
        "prompt": prompt[:200],  # truncate to keep log bounded
    }
    with open(full, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def miss_rate(records: list[dict]) -> float:
    """Given JSONL records, return miss rate (1.0 = everything mis-classified)."""
    if not records:
        return 0.0
    return sum(1 for r in records if r.get("expected") != r.get("actual")) / len(records)


__all__ = [
    "ALL_CLASSES",
    "Classification",
    "ClassifyFn",
    "DISPATCH_TIER",
    "HeuristicClassifier",
    "TaskClass",
    "dispatch_tier",
    "log_miss",
    "miss_rate",
]
