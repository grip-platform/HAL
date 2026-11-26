"""HAL Retry Policy — configurable retry strategies.

Defines retry policies (exponential, linear, fixed) as composable
objects. Used by smart_retry and other modules.

Usage:
    from lib.hal.retry_policy import ExponentialBackoff
    policy = ExponentialBackoff(base_s=1.0, max_s=30.0)
    delay = policy.delay(attempt=3)  # 8.0 seconds
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0


class ExponentialBackoff:
    """Exponential backoff: delay = base * 2^attempt."""

    def __init__(self, base_s: float = 1.0, max_s: float = 30.0):
        self._base = base_s
        self._max = max_s

    def delay(self, attempt: int) -> float:
        return min(self._base * (2 ** attempt), self._max)


class LinearBackoff:
    """Linear backoff: delay = base * attempt."""

    def __init__(self, base_s: float = 1.0, max_s: float = 30.0):
        self._base = base_s
        self._max = max_s

    def delay(self, attempt: int) -> float:
        return min(self._base * (attempt + 1), self._max)


class FixedDelay:
    """Fixed delay between retries."""

    def __init__(self, delay_s: float = 1.0):
        self._delay = delay_s

    def delay(self, attempt: int) -> float:
        return self._delay
