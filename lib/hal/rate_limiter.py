"""HAL Rate Limiter — token-aware rate limiting per provider.

Complements the Limit Sentinel with proactive rate limiting. Instead of
detecting rate limits after they happen, this module enforces limits
before sending requests.

Supports both RPM (requests per minute) and TPM (tokens per minute) limits.

Usage:
    from lib.hal.rate_limiter import RateLimiter

    limiter = RateLimiter()
    limiter.configure("groq", rpm=30, tpm=6000)
    if limiter.allow("groq", estimated_tokens=500):
        response = registry.chat(...)
        limiter.record("groq", tokens_used=response.usage.prompt_tokens)
"""
import threading
import time
from dataclasses import dataclass


@dataclass
class RateLimitInfo:
    """Structured rate limit state from provider headers (CC SDKRateLimitInfo pattern)."""
    utilization: float = 0.0      # 0.0–1.0, derived from x-ratelimit-* headers
    reset_time_ms: int = 0        # epoch ms when limits reset
    overage: bool = False         # True if currently over limit
    provider: str = ""

    @classmethod
    def from_headers(cls, headers: dict, provider: str) -> "RateLimitInfo":
        """Parse rate limit info from response headers."""
        remaining = int(headers.get("x-ratelimit-remaining-requests", -1))
        limit = int(headers.get("x-ratelimit-limit-requests", -1))
        reset_ms = int(headers.get("x-ratelimit-reset-requests", 0) or 0)
        util = 0.0
        if limit > 0 and remaining >= 0:
            util = 1.0 - (remaining / limit)
        return cls(utilization=util, reset_time_ms=reset_ms,
                   overage=remaining == 0, provider=provider)


@dataclass(frozen=True)
class ProviderLimits:
    """Rate limits for a single provider."""
    rpm: int  # Requests per minute (0 = unlimited)
    tpm: int  # Tokens per minute (0 = unlimited)


@dataclass
class _Window:
    """Sliding window tracker for a single provider."""
    requests: list[float]  # timestamps
    tokens: list[tuple[float, int]]  # (timestamp, token_count)

    def prune(self, cutoff: float) -> None:
        """Remove entries older than cutoff."""
        self.requests = [t for t in self.requests if t > cutoff]
        self.tokens = [(t, n) for t, n in self.tokens if t > cutoff]

    def request_count(self) -> int:
        return len(self.requests)

    def token_count(self) -> int:
        return sum(n for _, n in self.tokens)


# Default limits by provider (from public documentation)
_DEFAULT_LIMITS: dict[str, ProviderLimits] = {
    "groq": ProviderLimits(rpm=30, tpm=6000),
    "gemini": ProviderLimits(rpm=15, tpm=1_000_000),
    "openai": ProviderLimits(rpm=500, tpm=30_000),
    "anthropic": ProviderLimits(rpm=50, tpm=40_000),
    "deepseek": ProviderLimits(rpm=60, tpm=0),
    "ollama": ProviderLimits(rpm=0, tpm=0),  # Local — no limits
}


class RateLimiter:
    """Proactive rate limiting per provider. Thread-safe."""

    def __init__(self):
        self._limits: dict[str, ProviderLimits] = dict(_DEFAULT_LIMITS)
        self._windows: dict[str, _Window] = {}
        self._lock = threading.Lock()

    def configure(self, provider: str, rpm: int = 0, tpm: int = 0) -> None:
        """Set rate limits for a provider."""
        with self._lock:
            self._limits[provider] = ProviderLimits(rpm=rpm, tpm=tpm)

    def allow(self, provider: str, estimated_tokens: int = 0) -> bool:
        """Check if a request is allowed under current limits."""
        limits = self._limits.get(provider)
        if not limits:
            return True  # Unknown provider — allow

        with self._lock:
            window = self._get_window(provider)
            now = time.time()
            window.prune(now - 60)

            if limits.rpm > 0 and window.request_count() >= limits.rpm:
                return False

            if limits.tpm > 0 and (window.token_count() + estimated_tokens) > limits.tpm:
                return False

            return True

    def record(self, provider: str, tokens_used: int = 0) -> None:
        """Record a completed request."""
        with self._lock:
            window = self._get_window(provider)
            now = time.time()
            window.requests.append(now)
            if tokens_used > 0:
                window.tokens.append((now, tokens_used))

    def wait_time(self, provider: str) -> float:
        """Seconds until the next request would be allowed. 0 if allowed now."""
        limits = self._limits.get(provider)
        if not limits or (limits.rpm == 0 and limits.tpm == 0):
            return 0.0

        with self._lock:
            window = self._get_window(provider)
            now = time.time()
            window.prune(now - 60)

            if limits.rpm > 0 and window.request_count() >= limits.rpm:
                oldest = min(window.requests)
                return max(0.0, (oldest + 60) - now)

            return 0.0

    def status(self, provider: str) -> dict:
        """Current rate limit status for a provider."""
        limits = self._limits.get(provider, ProviderLimits(0, 0))
        with self._lock:
            window = self._get_window(provider)
            window.prune(time.time() - 60)
            return {
                "provider": provider,
                "rpm_limit": limits.rpm,
                "rpm_used": window.request_count(),
                "tpm_limit": limits.tpm,
                "tpm_used": window.token_count(),
            }

    def _get_window(self, provider: str) -> _Window:
        """Get or create window for provider. Must be called with lock held."""
        if provider not in self._windows:
            self._windows[provider] = _Window(requests=[], tokens=[])
        return self._windows[provider]


class EWMALatencyTracker:
    """Exponentially weighted moving average latency tracker per provider.

    α=0.3 (recent observations weighted more than history).
    Used to reorder providers within a tier: faster providers are promoted.
    Thread-safety: not required (single-threaded HAL process).
    """

    _ALPHA = 0.3

    def __init__(self) -> None:
        self._ewma: dict[str, float] = {}

    def record(self, provider: str, latency_ms: float) -> None:
        """Update EWMA for a provider with a new observation."""
        if provider not in self._ewma:
            self._ewma[provider] = latency_ms
        else:
            self._ewma[provider] = (
                self._ALPHA * latency_ms + (1 - self._ALPHA) * self._ewma[provider]
            )

    def get(self, provider: str) -> float | None:
        """Return current EWMA latency for a provider, or None if unseen."""
        return self._ewma.get(provider)

    def fastest_in(self, providers: list[str]) -> str | None:
        """Return the provider with the lowest EWMA latency. None if no data."""
        known = {p: self._ewma[p] for p in providers if p in self._ewma}
        if not known:
            return None
        return min(known, key=known.__getitem__)

    def sorted_by_latency(self, providers: list[str]) -> list[str]:
        """Return providers sorted fastest-first. Untracked providers go last."""
        known = [(p, self._ewma[p]) for p in providers if p in self._ewma]
        unknown = [p for p in providers if p not in self._ewma]
        return [p for p, _ in sorted(known, key=lambda x: x[1])] + unknown
