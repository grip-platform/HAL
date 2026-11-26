"""HAL Smart Retry — intelligent retry with provider rotation.

When a provider call fails, smart retry doesn't just retry the same
provider — it rotates to the next available provider. Combines
exponential backoff with provider failover.

Usage:
    from lib.hal.smart_retry import SmartRetry

    retry = SmartRetry(registry)
    response = retry.call("groq/llama", "Hello", max_retries=3)
    # If groq fails, automatically tries gemini, then ollama, etc.
"""
import time
from dataclasses import dataclass, field


@dataclass
class RetryAttempt:
    """Record of a single retry attempt."""
    model: str
    success: bool
    latency_ms: float = 0.0
    error: str = ""


@dataclass
class RetryResult:
    """Complete result of a retry sequence."""
    success: bool
    response_text: str = ""
    final_model: str = ""
    attempts: list[RetryAttempt] = field(default_factory=list)
    total_latency_ms: float = 0.0

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def rotated(self) -> bool:
        """True if the final model differs from the initial model."""
        if len(self.attempts) < 2:
            return False
        return self.attempts[0].model != self.attempts[-1].model

    @property
    def summary(self) -> str:
        status = "ok" if self.success else "failed"
        rotation = f" (rotated to {self.final_model})" if self.rotated else ""
        return f"Retry: {status} after {self.attempt_count} attempts, {self.total_latency_ms:.0f}ms{rotation}"


class SmartRetry:
    """Retry with exponential backoff and provider rotation."""

    def __init__(self, registry=None):
        self._registry = registry

    def call(
        self, model: str, message: str, *,
        max_retries: int = 3, base_delay_s: float = 1.0, max_tokens: int = 1000,
    ) -> RetryResult:
        """Call with smart retry. Rotates providers on failure."""
        fallback_models = self._get_fallbacks(model)
        all_models = [model] + [m for m in fallback_models if m != model]

        attempts: list[RetryAttempt] = []
        total_start = time.monotonic()

        for attempt_num in range(max_retries):
            target = all_models[attempt_num % len(all_models)] if all_models else model
            start = time.monotonic()

            try:
                response = self._registry.chat(target, message, max_tokens=max_tokens)
                latency = (time.monotonic() - start) * 1000
                attempts.append(RetryAttempt(model=target, success=True, latency_ms=latency))
                return RetryResult(
                    success=True, response_text=response.text, final_model=target,
                    attempts=attempts, total_latency_ms=(time.monotonic() - total_start) * 1000,
                )
            except Exception as e:
                latency = (time.monotonic() - start) * 1000
                attempts.append(RetryAttempt(model=target, success=False, latency_ms=latency, error=str(e)[:200]))

                # Exponential backoff before next attempt (skip on last attempt)
                if attempt_num < max_retries - 1:
                    delay = base_delay_s * (2 ** attempt_num)
                    time.sleep(min(delay, 10.0))

        return RetryResult(
            success=False, final_model=attempts[-1].model if attempts else model,
            attempts=attempts, total_latency_ms=(time.monotonic() - total_start) * 1000,
        )

    def _get_fallbacks(self, primary: str) -> list[str]:
        """Get fallback models from registry, excluding the primary."""
        if not self._registry:
            return []
        models: list[str] = []
        for p in self._registry.list_providers():
            if p.models:
                addr = f"{p.name}/{p.models[0]}"
                if addr != primary:
                    models.append(addr)
        return models
