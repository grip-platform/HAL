"""DFI-FIB Kernel — InnerspeakerGo recursive improvement loop.

One self-improvement loop that composes:
- Direct Execution Bias: one call, no ceremony.
- YSH: three documented uses before abstraction (converge, ultrado, slack-update).
- SOLID Open-Closed: ConvergeFn Protocol closed; concretes open for extension.
- Fibonacci memoization: LRU on fingerprint + fixed-point registry + depth cap F(7)=13.

Usage::

    from lib.improve_kernel import InnerspeakerGo, ImproveResult, fingerprint

    def my_converge(state: str, depth: int) -> ImproveResult[str]:
        improved = do_one_improvement(state)
        return ImproveResult(
            state=improved,
            metrics={"quality": score(improved)},
            depth=depth,
            fingerprint=fingerprint(improved),
        )

    kernel = InnerspeakerGo(
        converge=my_converge,
        threshold={"quality": ("ge", 0.9)},
    )
    result = kernel("initial state")
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Protocol, TypeVar

T = TypeVar("T")

# Default paths (overridable per-instance for testing)
_CLAUDE_DIR = Path.home() / ".claude"
_DEFAULT_FIXED_POINTS = _CLAUDE_DIR / "data" / "converge-fixed-points.jsonl"
_DEFAULT_TELEMETRY = _CLAUDE_DIR / "logs" / "improve-kernel.jsonl"


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

@dataclass
class ImproveResult(Generic[T]):
    """Result of one converge step."""
    state: T
    metrics: dict[str, float]
    depth: int
    fingerprint: str

    def passes(self, threshold: dict[str, tuple[str, float]]) -> bool:
        """Return True when all threshold conditions are satisfied."""
        for key, (op, val) in threshold.items():
            m = self.metrics.get(key, 0.0)
            if op == "ge" and m < val:
                return False
            if op == "le" and m > val:
                return False
        return True


class ConvergeFn(Protocol[T]):
    """Protocol for one improvement step. Must be pure relative to the kernel."""
    def __call__(self, state: T, depth: int) -> ImproveResult[T]: ...


# ---------------------------------------------------------------------------
# Fingerprint (deterministic, cross-session stable)
# ---------------------------------------------------------------------------

def fingerprint(state) -> str:
    """SHA-256 (first 16 hex chars) of normalised state content.

    Deterministic across Python restarts (hashlib.sha256 is stable).
    Works on str, bytes, or any JSON-serialisable object.
    """
    if isinstance(state, bytes):
        raw = state
    elif isinstance(state, str):
        raw = state.encode()
    else:
        raw = json.dumps(state, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

class InnerspeakerGo(Generic[T]):
    """Recursive improvement loop with LRU cache + fixed-point registry.

    CC budget: __call__ = 7 branches (well within DEFAULT threshold of 15).
    """

    DEFAULT_MAX_DEPTH = 13  # F(7) — matches RSI wave cap

    def __init__(
        self,
        converge: ConvergeFn[T],
        threshold: dict[str, tuple[str, float]],
        max_depth: int = DEFAULT_MAX_DEPTH,
        fixed_points_path: Path | None = None,
        telemetry_path: Path | None = None,
    ) -> None:
        self.converge = converge
        self.threshold = threshold
        self.max_depth = max_depth
        self._fixed_points_path = fixed_points_path or _DEFAULT_FIXED_POINTS
        self._telemetry_path = telemetry_path or _DEFAULT_TELEMETRY
        self._cache: dict[str, ImproveResult[T]] = {}
        self._fixed_points: set[str] = self._load_fixed_points()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def __call__(self, state: T, depth: int = 0) -> ImproveResult[T]:
        """Improve state until threshold passes, fixed point, or depth cap."""
        fp = fingerprint(state)
        result = self._try_short_circuit(fp, state, depth)
        if result is not None:
            return result
        return self._improve(fp, state, depth)

    # ------------------------------------------------------------------
    # Internal steps (each < 10 lines, CC ≤ 3 each)
    # ------------------------------------------------------------------

    def _try_short_circuit(
        self, fp: str, state: T, depth: int
    ) -> ImproveResult[T] | None:
        """Return cached result or fixed-point stub; None means proceed."""
        cached = self._cache.get(fp)
        if cached and cached.passes(self.threshold):
            return cached
        if fp in self._fixed_points:
            return cached or ImproveResult(state, {}, depth, fp)
        if depth >= self.max_depth:
            return cached or ImproveResult(state, {}, depth, fp)
        return None

    def _improve(self, fp: str, state: T, depth: int) -> ImproveResult[T]:
        """Run one converge step, cache, detect fixed point, recurse or return."""
        improved = self.converge(state, depth=depth + 1)
        self._cache[fp] = improved
        self._emit_telemetry(fp, improved)

        if improved.fingerprint == fp:
            self._register_fixed_point(fp)
            return improved
        if improved.passes(self.threshold):
            return improved
        return self(improved.state, depth=depth + 1)

    # ------------------------------------------------------------------
    # Fixed-point persistence
    # ------------------------------------------------------------------

    def _load_fixed_points(self) -> set[str]:
        if not self._fixed_points_path.exists():
            return set()
        fps: set[str] = set()
        for line in self._fixed_points_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                fps.add(json.loads(line)["fingerprint"])
            except (json.JSONDecodeError, KeyError):
                continue
        return fps

    def _register_fixed_point(self, fp: str) -> None:
        self._fixed_points.add(fp)
        self._persist_fixed_point(fp)

    def _persist_fixed_point(self, fp: str) -> None:
        self._fixed_points_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"fingerprint": fp, "ts": _now()}
        with self._fixed_points_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def _emit_telemetry(self, input_fp: str, result: ImproveResult) -> None:
        self._telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        outcome = (
            "fixed_point" if result.fingerprint == input_fp
            else "threshold_pass" if result.passes(self.threshold)
            else "recursing"
        )
        record = {
            "ts": _now(),
            "input_fingerprint": input_fp,
            "output_fingerprint": result.fingerprint,
            "depth": result.depth,
            "outcome": outcome,
            "metrics": result.metrics,
        }
        with self._telemetry_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
