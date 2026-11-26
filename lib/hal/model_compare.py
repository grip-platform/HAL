"""HAL Model Comparison — side-by-side evaluation of model responses.

Sends the same query to multiple models and compares responses on
quality, cost, and latency. Useful for provider selection, A/B testing,
and council-style deliberation.

Usage:
    from lib.hal.model_compare import ModelComparator

    comparator = ModelComparator(registry)
    result = comparator.compare(
        query="Explain recursion",
        models=["groq/llama-3.3-70b", "gemini/gemini-2.5-flash"],
    )
    print(result.summary)
    print(f"Winner: {result.winner}")
"""
import time
from dataclasses import dataclass, field

from lib.hal.quality_scorer import QualityScore, QualityScorer


@dataclass
class ModelResult:
    """Response from a single model in a comparison."""
    model: str
    response: str
    quality: QualityScore
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    error: str = ""

    @property
    def success(self) -> bool:
        return not self.error

    def to_dict(self) -> dict:
        return {
            "model": self.model, "response": self.response[:500],
            "quality": self.quality.to_dict(), "latency_ms": round(self.latency_ms),
            "cost_usd": round(self.cost_usd, 4), "error": self.error,
        }


@dataclass
class ComparisonResult:
    """Complete comparison across multiple models."""
    query: str
    results: list[ModelResult] = field(default_factory=list)
    total_cost_usd: float = 0.0

    @property
    def winner(self) -> str:
        """Model with highest quality score."""
        successful = [r for r in self.results if r.success]
        if not successful:
            return ""
        return max(successful, key=lambda r: r.quality.overall).model

    @property
    def fastest(self) -> str:
        """Model with lowest latency."""
        successful = [r for r in self.results if r.success]
        if not successful:
            return ""
        return min(successful, key=lambda r: r.latency_ms).model

    @property
    def cheapest(self) -> str:
        """Model with lowest cost."""
        successful = [r for r in self.results if r.success and r.cost_usd > 0]
        if not successful:
            return ""
        return min(successful, key=lambda r: r.cost_usd).model

    @property
    def summary(self) -> str:
        lines = [f"Comparison: {len(self.results)} models, ${self.total_cost_usd:.4f} total"]
        for r in sorted(self.results, key=lambda x: -x.quality.overall):
            status = f"{r.quality.overall}/100 ({r.quality.grade})" if r.success else f"ERROR: {r.error[:50]}"
            lines.append(f"  {r.model}: {status} | {r.latency_ms:.0f}ms | ${r.cost_usd:.4f}")
        if self.winner:
            lines.append(f"  Winner: {self.winner}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "query": self.query[:200], "winner": self.winner,
            "fastest": self.fastest, "cheapest": self.cheapest,
            "total_cost_usd": self.total_cost_usd,
            "results": [r.to_dict() for r in self.results],
        }


class ModelComparator:
    """Compares multiple models on the same query."""

    def __init__(self, registry=None):
        self._registry = registry
        self._scorer = QualityScorer()

    def compare(self, query: str, models: list[str]) -> ComparisonResult:
        """Send query to all models and compare results."""
        results: list[ModelResult] = []
        total_cost = 0.0

        for model in models:
            result = self._query_model(query, model)
            results.append(result)
            total_cost += result.cost_usd

        return ComparisonResult(query=query, results=results, total_cost_usd=total_cost)

    def _query_model(self, query: str, model: str) -> ModelResult:
        """Query a single model and score the response."""
        if not self._registry:
            return ModelResult(model=model, response="", quality=QualityScore(0, 0, 0, 0), error="No registry")

        start = time.monotonic()
        try:
            response = self._registry.chat(model, query, max_tokens=1000)
            latency = (time.monotonic() - start) * 1000
            quality = self._scorer.score(query, response.text)
            return ModelResult(
                model=model, response=response.text, quality=quality,
                latency_ms=latency, cost_usd=response.cost_usd or 0.0,
            )
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return ModelResult(
                model=model, response="", quality=QualityScore(0, 0, 0, 0),
                latency_ms=latency, error=str(e)[:200],
            )
