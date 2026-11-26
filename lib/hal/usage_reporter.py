"""HAL Usage Reporter — generate usage reports for billing and tracking.

Aggregates cost, token, and query data across sessions into structured
reports. Supports daily, weekly, and monthly rollups.

Usage:
    from lib.hal.usage_reporter import UsageReporter
    reporter = UsageReporter()
    report = reporter.daily_report()
"""
import time
from dataclasses import dataclass, field


@dataclass
class UsageReport:
    """Aggregated usage report."""
    period: str
    total_queries: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    by_provider: dict[str, float] = field(default_factory=dict)
    by_model: dict[str, int] = field(default_factory=dict)
    generated_at: float = field(default_factory=time.time)

    @property
    def avg_cost_per_query(self) -> float:
        return self.total_cost_usd / max(self.total_queries, 1)

    @property
    def avg_tokens_per_query(self) -> int:
        return self.total_tokens // max(self.total_queries, 1)

    def to_dict(self) -> dict:
        return {
            "period": self.period,
            "total_queries": self.total_queries,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "avg_cost_per_query": round(self.avg_cost_per_query, 4),
            "avg_tokens_per_query": self.avg_tokens_per_query,
            "by_provider": {k: round(v, 4) for k, v in self.by_provider.items()},
            "by_model": self.by_model,
        }


class UsageReporter:
    """Generates usage reports from tracked data."""

    def __init__(self):
        self._queries: list[dict] = []

    def record(self, provider: str, model: str, tokens: int, cost_usd: float) -> None:
        self._queries.append({
            "provider": provider, "model": model,
            "tokens": tokens, "cost_usd": cost_usd,
            "timestamp": time.time(),
        })

    def report(self, period: str = "session") -> UsageReport:
        r = UsageReport(period=period)
        for q in self._queries:
            r.total_queries += 1
            r.total_tokens += q["tokens"]
            r.total_cost_usd += q["cost_usd"]
            prov = q["provider"]
            r.by_provider[prov] = r.by_provider.get(prov, 0) + q["cost_usd"]
            model = q["model"]
            r.by_model[model] = r.by_model.get(model, 0) + 1
        return r

    def reset(self) -> None:
        self._queries.clear()
