"""HAL Provider Benchmark — systematic provider evaluation.

Runs standardised test prompts against multiple providers and measures
quality, latency, and cost. Results are persisted for trend analysis.

Usage:
    from lib.hal.benchmark import Benchmark

    bench = Benchmark(registry)
    result = bench.run(models=["groq/llama", "gemini/flash"])
    print(result.summary)
    print(result.leaderboard)
"""
import json
import os
import time
from dataclasses import dataclass, field

from lib.hal.home import HAL_HOME as _HAL_HOME
from lib.hal.quality_scorer import QualityScorer

_RESULTS_DIR = os.path.join(_HAL_HOME, "state", "hal", "benchmarks")

# Standard benchmark prompts covering different capabilities
_BENCHMARK_PROMPTS: list[tuple[str, str]] = [
    ("code", "Write a Python function that checks if a string is a palindrome. Include edge cases."),
    ("explain", "Explain how a hash map works, including collision handling strategies."),
    ("debug", "This Python code has a bug: `def fib(n): return fib(n-1) + fib(n-2)`. Find and fix it."),
    ("reason", "A farmer has 17 sheep. All but 9 run away. How many sheep does he have left?"),
    ("creative", "Write a haiku about a programmer debugging at 3am."),
]


@dataclass
class ModelScore:
    """Benchmark score for a single model."""
    model: str
    avg_quality: float = 0.0
    avg_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    prompts_completed: int = 0
    errors: int = 0

    @property
    def composite(self) -> float:
        """Composite score: quality * 0.6 + speed * 0.3 + cost * 0.1."""
        speed_score = max(0, 100 - self.avg_latency_ms / 50)  # 0-100
        cost_score = max(0, 100 - self.total_cost_usd * 1000)  # 0-100
        return self.avg_quality * 0.6 + speed_score * 0.3 + cost_score * 0.1

    def to_dict(self) -> dict:
        return {
            "model": self.model, "avg_quality": round(self.avg_quality, 1),
            "avg_latency_ms": round(self.avg_latency_ms), "total_cost_usd": round(self.total_cost_usd, 4),
            "prompts_completed": self.prompts_completed, "errors": self.errors,
            "composite": round(self.composite, 1),
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark result across all models."""
    scores: list[ModelScore] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    @property
    def leaderboard(self) -> list[str]:
        """Models ranked by composite score."""
        return [s.model for s in sorted(self.scores, key=lambda s: s.composite, reverse=True)]

    @property
    def summary(self) -> str:
        lines = [f"Benchmark: {len(self.scores)} models, {len(_BENCHMARK_PROMPTS)} prompts each"]
        for s in sorted(self.scores, key=lambda x: x.composite, reverse=True):
            lines.append(f"  {s.model}: Q={s.avg_quality:.0f} L={s.avg_latency_ms:.0f}ms C=${s.total_cost_usd:.4f} ({s.composite:.0f} composite)")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "leaderboard": self.leaderboard,
            "scores": [s.to_dict() for s in self.scores],
        }


class Benchmark:
    """Runs standardised benchmarks against providers."""

    def __init__(self, registry=None):
        self._registry = registry
        self._scorer = QualityScorer()

    def run(self, models: list[str]) -> BenchmarkResult:
        """Run all benchmark prompts against all models."""
        scores: list[ModelScore] = []
        for model in models:
            scores.append(self._benchmark_model(model))

        result = BenchmarkResult(scores=scores)
        self._save(result)
        return result

    def _benchmark_model(self, model: str) -> ModelScore:
        """Run all prompts against a single model."""
        score = ModelScore(model=model)
        qualities: list[float] = []
        latencies: list[float] = []

        for _category, prompt in _BENCHMARK_PROMPTS:
            start = time.monotonic()
            try:
                if not self._registry:
                    score.errors += 1
                    continue
                response = self._registry.chat(model, prompt, max_tokens=500)
                latency = (time.monotonic() - start) * 1000
                q = self._scorer.score(prompt, response.text)
                qualities.append(q.overall)
                latencies.append(latency)
                score.total_cost_usd += response.cost_usd or 0
                score.prompts_completed += 1
            except Exception:
                score.errors += 1

        if qualities:
            score.avg_quality = sum(qualities) / len(qualities)
        if latencies:
            score.avg_latency_ms = sum(latencies) / len(latencies)

        return score

    def _save(self, result: BenchmarkResult) -> None:
        """Persist benchmark result."""
        os.makedirs(_RESULTS_DIR, exist_ok=True)
        path = os.path.join(_RESULTS_DIR, f"bench-{int(result.timestamp)}.json")
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
