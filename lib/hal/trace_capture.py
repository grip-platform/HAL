"""Execution trace capture for HAL's meta-agent loop.

Captures step-by-step reasoning traces from task agent runs so the
meta-agent can analyse WHY something failed, not just WHETHER it failed.
AutoAgent insight: traces outperform scores alone for targeted improvement.
"""
import time
from dataclasses import dataclass, field


@dataclass
class TraceStep:
    """One step in an agent's execution trace."""

    step: int
    action: str
    result: str
    success: bool
    duration_ms: float = 0.0


@dataclass
class ExecutionTrace:
    """Full execution trace for a single task agent run."""

    steps: list[TraceStep] = field(default_factory=list)
    failure_point: int | None = None   # step index where execution diverged
    total_duration_ms: float = 0.0
    tokens_used: int = 0
    score: float = 0.0                 # 0.0 – 1.0 composite score

    def record(self, action: str, result: str, success: bool, duration_ms: float = 0.0) -> None:
        """Append a step to the trace. Marks failure_point on first failure."""
        step = TraceStep(
            step=len(self.steps),
            action=action,
            result=result,
            success=success,
            duration_ms=duration_ms,
        )
        self.steps.append(step)
        if not success and self.failure_point is None:
            self.failure_point = step.step

    def summary(self) -> str:
        """One-line summary for meta-agent consumption."""
        total = len(self.steps)
        passed = sum(1 for s in self.steps if s.success)
        if self.failure_point is not None:
            return (
                f"{passed}/{total} steps OK, failed at step {self.failure_point}: "
                f"{self.steps[self.failure_point].action}"
            )
        return f"{passed}/{total} steps OK"

    @property
    def succeeded(self) -> bool:
        return self.failure_point is None and bool(self.steps)


class TraceCapture:
    """Context manager that wraps a task agent function and captures its trace.

    Usage:
        with TraceCapture() as tc:
            result = my_task_fn()
        trace = tc.trace
    """

    def __init__(self) -> None:
        self.trace = ExecutionTrace()
        self._start: float = 0.0

    def __enter__(self) -> "TraceCapture":
        self._start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.trace.total_duration_ms = (time.monotonic() - self._start) * 1000
        if exc_type is not None:
            self.trace.record(
                action="exception",
                result=str(exc_val),
                success=False,
            )
        return False  # never suppress exceptions
