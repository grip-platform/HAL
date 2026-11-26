"""HAL hypothesis engine — falsifiable-claim registry, HAL_HOME-rooted.

Lifecycle:

1. Register: ``register(claim, metric, prediction, deadline) -> Hypothesis``.
   Auto-assigns an ``H{N:03d}`` code. Persists to
   ``$HAL_HOME/learning/hypotheses.jsonl``.
2. Falsify: ``falsify(code, evidence) -> Hypothesis``. Marks status
   ``refuted`` with the supplied evidence string.
3. Status: ``status() -> dict``. Aggregate counts by status. No side effects.
4. Verify-all: ``verify_all() -> list[Hypothesis]``. Returns due items
   (deadline passed, status still pending). Caller populates the actual
   measurement and calls :func:`falsify` or :func:`confirm`.

Intentionally simpler than GRIP's engine:

* No ``file_lock`` dependency — HAL uses per-call ``open`` with ``O_APPEND``
  which is atomic on POSIX for lines < 4KB.
* No effect-size / baseline numeric fields — HAL tracks
  predictions as strings; quantitative fitting is out-of-scope.
* No externally_validated flag — removed until a concrete use-case appears.

The JSONL format is compatible with GRIP's on disk (field subset),
so a one-way migration from a GRIP hypothesis file is possible if
the operator wants to seed a new HAL install.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

__all__ = [
    "Hypothesis",
    "HypothesisNotFound",
    "HYPOTHESIS_FILE",
    "default_hypothesis_file",
    "register",
    "verify_all",
    "falsify",
    "confirm",
    "status",
    "_reset_for_tests",
]


STATUS_PENDING = "pending"
STATUS_CONFIRMED = "confirmed"
STATUS_REFUTED = "refuted"
STATUS_INCONCLUSIVE = "inconclusive"

_VALID_STATUSES = frozenset({
    STATUS_PENDING, STATUS_CONFIRMED, STATUS_REFUTED, STATUS_INCONCLUSIVE,
})


class HypothesisNotFound(KeyError):
    """Raised when a code is not present in the registry."""


@dataclass
class Hypothesis:
    """One falsifiable claim, registered and later verified."""
    code: str = ""
    claim: str = ""
    metric: str = ""
    prediction: str = ""
    deadline: str = ""
    registered_at: str = ""
    status: str = STATUS_PENDING
    evidence: str = ""
    verified_at: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "Hypothesis":
        fields = cls.__dataclass_fields__  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in fields})


def _hal_home() -> Path:
    return Path(os.environ.get("HAL_HOME", str(Path.home() / ".hal")))


def default_hypothesis_file() -> Path:
    return _hal_home() / "learning" / "hypotheses.jsonl"


HYPOTHESIS_FILE = default_hypothesis_file()


def _resolve(path: Optional[Path]) -> Path:
    return path if path is not None else default_hypothesis_file()


def _today() -> str:
    return date.today().isoformat()


def _load_all(path: Optional[Path] = None) -> List[Hypothesis]:
    target = _resolve(path)
    if not target.exists():
        return []
    out: List[Hypothesis] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            out.append(Hypothesis.from_dict(json.loads(stripped)))
        except (json.JSONDecodeError, TypeError):
            continue
    return out


def _next_code(path: Optional[Path] = None) -> str:
    existing = _load_all(path)
    max_n = 0
    for h in existing:
        if len(h.code) >= 2 and h.code[0] == "H":
            try:
                max_n = max(max_n, int(h.code[1:]))
            except ValueError:
                continue
    return f"H{max_n + 1:03d}"


def _append(h: Hypothesis, path: Optional[Path] = None) -> None:
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(h.to_dict(), separators=(",", ":")) + "\n")


def _rewrite(all_h: List[Hypothesis], path: Optional[Path] = None) -> None:
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(h.to_dict(), separators=(",", ":")) for h in all_h
    ]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def register(
    claim: str,
    metric: str,
    prediction: str,
    deadline: str,
    *,
    path: Optional[Path] = None,
) -> Hypothesis:
    """Register a new hypothesis with auto-assigned ``H{N:03d}`` code."""
    if not claim.strip():
        raise ValueError("hypothesis claim cannot be empty")
    h = Hypothesis(
        code=_next_code(path),
        claim=claim.strip(),
        metric=metric.strip(),
        prediction=prediction.strip(),
        deadline=deadline.strip(),
        registered_at=_today(),
        status=STATUS_PENDING,
    )
    _append(h, path)
    return h


def _update_status(
    code: str,
    new_status: str,
    evidence: str,
    path: Optional[Path],
) -> Hypothesis:
    if new_status not in _VALID_STATUSES:
        raise ValueError(f"invalid status {new_status!r}")
    records = _load_all(path)
    for idx, h in enumerate(records):
        if h.code == code:
            records[idx] = Hypothesis(
                code=h.code, claim=h.claim, metric=h.metric,
                prediction=h.prediction, deadline=h.deadline,
                registered_at=h.registered_at, status=new_status,
                evidence=evidence, verified_at=_today(),
            )
            _rewrite(records, path)
            return records[idx]
    raise HypothesisNotFound(f"hypothesis {code!r} not in registry")


def falsify(
    code: str,
    evidence: str,
    *,
    path: Optional[Path] = None,
) -> Hypothesis:
    """Mark hypothesis ``code`` as REFUTED with ``evidence``."""
    return _update_status(code, STATUS_REFUTED, evidence, path)


def confirm(
    code: str,
    evidence: str,
    *,
    path: Optional[Path] = None,
) -> Hypothesis:
    """Mark hypothesis ``code`` as CONFIRMED with ``evidence``."""
    return _update_status(code, STATUS_CONFIRMED, evidence, path)


def verify_all(
    *,
    today: Optional[str] = None,
    path: Optional[Path] = None,
) -> List[Hypothesis]:
    """Return every pending hypothesis whose deadline has arrived.

    ``today`` defaults to the current date; tests override it. Only
    pending entries are considered (confirmed/refuted are historical)."""
    cutoff = today or _today()
    out: List[Hypothesis] = []
    for h in _load_all(path):
        if h.status != STATUS_PENDING:
            continue
        if h.deadline and h.deadline <= cutoff:
            out.append(h)
    return out


def status(*, path: Optional[Path] = None) -> Dict[str, int]:
    """Return a count per status value. Missing statuses are reported as 0."""
    counts = {s: 0 for s in _VALID_STATUSES}
    for h in _load_all(path):
        counts[h.status] = counts.get(h.status, 0) + 1
    counts["total"] = sum(counts[s] for s in _VALID_STATUSES)
    return counts


def _reset_for_tests(path: Optional[Path] = None) -> None:
    """Remove the hypothesis file (test isolation)."""
    target = _resolve(path)
    if target.exists():
        target.unlink()
