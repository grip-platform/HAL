"""HAL Provider Analytics — track provider performance over time.

Records latency, error rate, and cost per provider per session. Enables
data-driven provider selection and surfacing trends to the user.

Usage:
    from lib.hal.provider_analytics import ProviderAnalytics

    analytics = ProviderAnalytics()
    analytics.record_call("groq", latency_ms=120, tokens=500, success=True)
    analytics.record_call("groq", latency_ms=0, tokens=0, success=False)

    stats = analytics.get_stats("groq")
    print(f"Groq: {stats.avg_latency_ms}ms avg, {stats.error_rate:.0%} errors")
"""
import json
import os
import time
from dataclasses import dataclass

from lib.hal.home import HAL_HOME as _HAL_HOME

_ANALYTICS_PATH = os.path.join(_HAL_HOME, "state", "hal", "provider-analytics.json")


@dataclass
class ProviderStats:
    """Aggregated statistics for a single provider."""
    provider: str
    total_calls: int = 0
    successful_calls: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    last_call: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.successful_calls, 1)

    @property
    def error_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return 1.0 - (self.successful_calls / self.total_calls)

    @property
    def avg_tokens_per_call(self) -> int:
        return self.total_tokens // max(self.successful_calls, 1)

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "error_rate": round(self.error_rate, 3),
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "avg_tokens_per_call": self.avg_tokens_per_call,
            "last_call": self.last_call,
        }


class ProviderAnalytics:
    """Tracks and aggregates provider performance metrics."""

    def __init__(self):
        self._stats: dict[str, ProviderStats] = {}
        self._load()

    def record_call(
        self, provider: str, *,
        latency_ms: float = 0.0, tokens: int = 0,
        cost_usd: float = 0.0, success: bool = True,
    ) -> None:
        """Record a single provider call."""
        if provider not in self._stats:
            self._stats[provider] = ProviderStats(provider=provider)

        stats = self._stats[provider]
        stats.total_calls += 1
        stats.last_call = time.time()
        if success:
            stats.successful_calls += 1
            stats.total_latency_ms += latency_ms
            stats.total_tokens += tokens
            stats.total_cost_usd += cost_usd

    def get_stats(self, provider: str) -> ProviderStats:
        """Get stats for a specific provider."""
        return self._stats.get(provider, ProviderStats(provider=provider))

    def all_stats(self) -> list[ProviderStats]:
        """Get stats for all tracked providers, sorted by call count."""
        return sorted(self._stats.values(), key=lambda s: s.total_calls, reverse=True)

    def ranking(self, metric: str = "latency") -> list[str]:
        """Rank providers by metric. Lower is better for latency/cost."""
        providers = [s for s in self._stats.values() if s.successful_calls > 0]
        key_map = {
            "latency": lambda s: s.avg_latency_ms,
            "cost": lambda s: s.total_cost_usd / max(s.successful_calls, 1),
            "reliability": lambda s: -s.successful_calls / max(s.total_calls, 1),
            "throughput": lambda s: -s.avg_tokens_per_call,
        }
        sort_key = key_map.get(metric, key_map["latency"])
        return [s.provider for s in sorted(providers, key=sort_key)]

    def save(self) -> None:
        """Persist analytics to disk."""
        os.makedirs(os.path.dirname(_ANALYTICS_PATH), exist_ok=True)
        data = {name: stats.to_dict() for name, stats in self._stats.items()}
        with open(_ANALYTICS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load persisted analytics."""
        if not os.path.exists(_ANALYTICS_PATH):
            return
        try:
            with open(_ANALYTICS_PATH, encoding="utf-8") as f:
                data = json.load(f)
            for name, d in data.items():
                self._stats[name] = ProviderStats(
                    provider=d.get("provider", name),
                    total_calls=d.get("total_calls", 0),
                    successful_calls=d.get("successful_calls", 0),
                    total_latency_ms=d.get("avg_latency_ms", 0) * d.get("successful_calls", 0),
                    total_tokens=d.get("total_tokens", 0),
                    total_cost_usd=d.get("total_cost_usd", 0),
                    last_call=d.get("last_call", 0),
                )
        except (json.JSONDecodeError, OSError):
            pass
