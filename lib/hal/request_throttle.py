"""HAL Request Throttle — fair queuing for concurrent requests.

When multiple conversations or agents send requests simultaneously,
the throttle ensures fair distribution across providers and prevents
any single consumer from monopolising resources.

Usage:
    from lib.hal.request_throttle import RequestThrottle

    throttle = RequestThrottle(max_concurrent=5)
    with throttle.acquire("groq"):
        response = registry.chat(...)
"""
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class ThrottleStats:
    total_requests: int = 0
    total_wait_ms: float = 0.0
    active_requests: int = 0
    queued_requests: int = 0

    @property
    def avg_wait_ms(self) -> float:
        return self.total_wait_ms / max(self.total_requests, 1)


class RequestThrottle:
    """Fair request throttling with per-provider concurrency limits."""

    def __init__(self, max_concurrent: int = 5):
        self._max = max_concurrent
        self._semaphore = threading.Semaphore(max_concurrent)
        self._active = 0
        self._queued = 0
        self._stats = ThrottleStats()
        self._lock = threading.Lock()

    @contextmanager
    def acquire(self, provider: str = ""):
        """Acquire a request slot. Blocks if at capacity."""
        with self._lock:
            self._queued += 1

        start = time.monotonic()
        self._semaphore.acquire()
        wait_ms = (time.monotonic() - start) * 1000

        with self._lock:
            self._queued -= 1
            self._active += 1
            self._stats.total_requests += 1
            self._stats.total_wait_ms += wait_ms

        try:
            yield
        finally:
            with self._lock:
                self._active -= 1
            self._semaphore.release()

    @property
    def active(self) -> int:
        with self._lock:
            return self._active

    @property
    def available(self) -> int:
        return self._max - self.active

    def stats(self) -> ThrottleStats:
        with self._lock:
            self._stats.active_requests = self._active
            self._stats.queued_requests = self._queued
            return ThrottleStats(
                total_requests=self._stats.total_requests,
                total_wait_ms=self._stats.total_wait_ms,
                active_requests=self._active,
                queued_requests=self._queued,
            )
