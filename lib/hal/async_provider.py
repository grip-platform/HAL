"""HAL Async Provider Adapter — non-blocking LLM calls.

Wraps synchronous provider calls in asyncio for concurrent execution.
Enables parallel council queries, background warmup, and non-blocking
chat in server contexts.

Usage:
    from lib.hal.async_provider import AsyncAdapter

    adapter = AsyncAdapter(registry)
    response = await adapter.chat("groq/llama-3.3-70b", "Hello")

    # Parallel queries to multiple providers
    responses = await adapter.parallel_chat(
        [("groq/llama", "Q1"), ("gemini/flash", "Q2")],
    )
"""
import asyncio
import time
from dataclasses import dataclass

from lib.hal.types import ChatResponse


@dataclass(frozen=True)
class AsyncChatResult:
    """Result from an async chat call."""
    model: str
    response: ChatResponse | None
    success: bool
    error: str = ""
    latency_ms: float = 0.0


class AsyncAdapter:
    """Wraps synchronous registry calls for async execution."""

    def __init__(self, registry):
        self._registry = registry

    async def chat(self, model: str, message: str, max_tokens: int = 16384) -> AsyncChatResult:
        """Send a chat message asynchronously."""
        loop = asyncio.get_event_loop()
        start = time.monotonic()
        try:
            response = await loop.run_in_executor(
                None, lambda: self._registry.chat(model, message, max_tokens=max_tokens),
            )
            latency = (time.monotonic() - start) * 1000
            return AsyncChatResult(model=model, response=response, success=True, latency_ms=latency)
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return AsyncChatResult(model=model, response=None, success=False, error=str(e)[:200], latency_ms=latency)

    async def parallel_chat(self, queries: list[tuple[str, str]], max_tokens: int = 16384) -> list[AsyncChatResult]:
        """Send multiple queries in parallel. Returns results in same order as queries."""
        tasks = [self.chat(model, message, max_tokens) for model, message in queries]
        return await asyncio.gather(*tasks)

    async def race(self, models: list[str], message: str, max_tokens: int = 16384) -> AsyncChatResult:
        """Race multiple models — return the first successful response.

        Runs all in parallel via gather, then picks the first success.
        O(n) single scan of results.
        """
        if not models:
            return AsyncChatResult(model="", response=None, success=False, error="No models")

        results = await self.parallel_chat([(m, message) for m in models], max_tokens)
        # Single scan: return first success, or last failure
        for result in results:
            if result.success:
                return result
        return results[-1]

    async def health_check(self, models: list[str]) -> dict[str, bool]:
        """Check health of multiple providers in parallel."""
        async def _check(model: str) -> tuple[str, bool]:
            result = await self.chat(model, "ping", max_tokens=10)
            return model, result.success

        checks = await asyncio.gather(*[_check(m) for m in models])
        return dict(checks)
