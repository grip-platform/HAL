"""HAL Progress Tracker — track multi-step task completion.

Tracks progress of multi-step operations like workflows, sprints,
and batch processing. Provides percentage completion, ETA, and
status reporting.

Usage:
    from lib.hal.progress_tracker import ProgressTracker
    tracker = ProgressTracker("build", total=10)
    tracker.advance("Compiled module 1")
    print(tracker.status)  # "build: 10% (1/10)"
"""
import time
from dataclasses import dataclass, field


@dataclass
class ProgressStatus:
    name: str
    completed: int
    total: int
    current_step: str = ""
    started_at: float = 0.0

    @property
    def pct(self) -> float:
        return (self.completed / self.total * 100) if self.total else 0

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.started_at if self.started_at else 0

    @property
    def eta_s(self) -> float:
        if self.completed == 0:
            return 0
        rate = self.elapsed_s / self.completed
        return rate * (self.total - self.completed)

    @property
    def summary(self) -> str:
        return f"{self.name}: {self.pct:.0f}% ({self.completed}/{self.total})"


class ProgressTracker:
    """Tracks multi-step progress."""

    def __init__(self, name: str, total: int):
        self._name = name
        self._total = total
        self._completed = 0
        self._current = ""
        self._started = time.time()
        self._steps: list[str] = []

    def advance(self, step_description: str = "") -> None:
        self._completed = min(self._completed + 1, self._total)
        self._current = step_description
        if step_description:
            self._steps.append(step_description)

    def set_progress(self, completed: int) -> None:
        self._completed = min(completed, self._total)

    @property
    def done(self) -> bool:
        return self._completed >= self._total

    @property
    def status(self) -> ProgressStatus:
        return ProgressStatus(
            self._name, self._completed, self._total,
            self._current, self._started,
        )

    @property
    def steps_completed(self) -> list[str]:
        return list(self._steps)
