"""HAL Context Warnings — graduated alerts at 80/85/95% context usage.

Prevents context overflow by monitoring token accumulation and emitting
graduated warnings with suggested actions. Never auto-actions — always
recommends, letting the user decide.

Three severity levels:
- INFO (80%):  "Context getting full. Consider saving key context."
- WARNING (85%): "Context nearly full. Compact or start fresh session."
- CRITICAL (95%): "Context overflow imminent. Save state NOW."

Usage:
    from lib.hal.context_warnings import ContextMonitor

    monitor = ContextMonitor(max_tokens=200000)
    monitor.add_tokens(1500)
    warning = monitor.check()
    if warning:
        print(warning.message)
"""
import threading
from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    """Warning severity levels."""
    NONE = "none"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


_SEVERITY_RANK = {Severity.NONE: 0, Severity.INFO: 1, Severity.WARNING: 2, Severity.CRITICAL: 3}


@dataclass(frozen=True)
class ContextWarning:
    """A graduated context warning."""
    severity: Severity
    usage_pct: float
    tokens_used: int
    tokens_max: int
    message: str
    suggested_actions: tuple[str, ...]


# Threshold definitions: (percentage, severity, message, actions)
_THRESHOLDS: list[tuple[float, Severity, str, tuple[str, ...]]] = [
    (0.95, Severity.CRITICAL,
     "Context overflow imminent ({pct:.0%} used, {remaining:,} tokens remaining). Save state NOW.",
     ("Save session state immediately",
      "Compact conversation (remove older turns)",
      "Start a fresh session with context transfer")),
    (0.85, Severity.WARNING,
     "Context nearly full ({pct:.0%} used, {remaining:,} tokens remaining).",
     ("Consider compacting conversation",
      "Save important context to memory",
      "Prepare for session handoff")),
    (0.80, Severity.INFO,
     "Context usage at {pct:.0%} ({remaining:,} tokens remaining).",
     ("Save key findings to memory before they scroll out",
      "Consider whether remaining work fits in this session")),
]


class ContextMonitor:
    """Monitors context token usage and emits graduated warnings."""

    def __init__(self, max_tokens: int = 200_000):
        self._max_tokens = max_tokens
        self._tokens_used = 0
        self._last_severity = Severity.NONE
        self._lock = threading.Lock()

    @property
    def tokens_used(self) -> int:
        with self._lock:
            return self._tokens_used

    @property
    def tokens_remaining(self) -> int:
        with self._lock:
            return max(0, self._max_tokens - self._tokens_used)

    @property
    def usage_pct(self) -> float:
        with self._lock:
            return self._tokens_used / self._max_tokens if self._max_tokens else 1.0

    def add_tokens(self, count: int) -> None:
        """Record additional token usage."""
        with self._lock:
            self._tokens_used += count

    def set_tokens(self, count: int) -> None:
        """Set absolute token count (for external tracking)."""
        with self._lock:
            self._tokens_used = count

    def check(self, force: bool = False) -> ContextWarning | None:
        """Check thresholds and return a warning if a threshold is active.

        Args:
            force: If False (default), only returns when a NEW threshold is
                   crossed (escalation only). If True, returns the current
                   threshold regardless of previous warnings (for periodic scans).
        """
        with self._lock:
            pct = self._tokens_used / self._max_tokens if self._max_tokens else 1.0
            remaining = max(0, self._max_tokens - self._tokens_used)

            for threshold_pct, severity, msg_template, actions in _THRESHOLDS:
                is_new = _SEVERITY_RANK[severity] > _SEVERITY_RANK[self._last_severity]
                if pct >= threshold_pct and (force or is_new):
                    if not force:
                        self._last_severity = severity
                    return ContextWarning(
                        severity=severity, usage_pct=pct,
                        tokens_used=self._tokens_used, tokens_max=self._max_tokens,
                        message=msg_template.format(pct=pct, remaining=remaining),
                        suggested_actions=actions,
                    )
            return None

    def reset(self) -> None:
        """Reset for a new session."""
        with self._lock:
            self._tokens_used = 0
            self._last_severity = Severity.NONE


def estimate_turn_tokens(user_msg: str, assistant_msg: str) -> int:
    """Estimate tokens for a conversation turn (both messages).

    Uses word_count / 0.75 heuristic — accurate within 10-20% for English.
    """
    total_words = len(user_msg.split()) + len(assistant_msg.split())
    return int(total_words / 0.75)
