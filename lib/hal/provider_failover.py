"""HAL Provider Failover — automatic failover chain configuration.

Defines explicit failover chains: if provider A fails, try B, then C.
More predictable than smart_retry's rotation — the user controls the
exact fallback order.

Usage:
    from lib.hal.provider_failover import FailoverChain
    chain = FailoverChain(["groq/llama", "gemini/flash", "ollama/mistral"])
    model = chain.next_available(failed={"groq/llama"})
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FailoverResult:
    model: str
    position: int  # 0 = primary, 1+ = fallback
    is_fallback: bool


class FailoverChain:
    """Ordered failover chain for provider selection."""

    def __init__(self, models: list[str]):
        self._models = list(models)
        self._failed: set[str] = set()

    @property
    def primary(self) -> str:
        return self._models[0] if self._models else ""

    def next_available(self, failed: set[str] | None = None) -> FailoverResult:
        """Get the next available model, skipping failed ones."""
        skip = failed or self._failed
        for i, model in enumerate(self._models):
            if model not in skip:
                return FailoverResult(model, i, i > 0)
        return FailoverResult("", -1, True)

    def mark_failed(self, model: str) -> None:
        self._failed.add(model)

    def mark_recovered(self, model: str) -> None:
        self._failed.discard(model)

    def reset(self) -> None:
        self._failed.clear()

    @property
    def chain(self) -> list[str]:
        return list(self._models)

    @property
    def healthy(self) -> list[str]:
        return [m for m in self._models if m not in self._failed]
