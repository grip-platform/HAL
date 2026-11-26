"""RAILLM — Redundant Array of Independent LLMs.

RAID-inspired failover for LLM providers with stateful conversation history
carried across provider switches. When a primary provider hits its limit,
the RAILLMSentinel transparently routes the full conversation history to an
equivalence-class peer, cascading through tiers (frontier -> fast -> free_tier
-> local) until a provider accepts the request.

The user never sees "You have hit your limit." The conversation continues on
the next available LLM without information loss, as long as the compression
strategy keeps the thread under the target provider's context window.

Architecture::

    LimitSentinel (base, lib/hal/sentinel.py — unchanged)
        `-- RAILLMSentinel (this module, composition only)
              |-- ConversationThread (stateful message history)
              |-- ModelEquivalenceMap (tier registry)
              `-- tier cascade chain builder

Hypothesis H047: Creating RAILLMSentinel as a subclass enables history-
carrying failover without modifying the existing LimitSentinel base class.

Falsification: revert `dispatch_with_history`, wire providers directly, and
observe information loss in a 10-message synthetic conversation test.
"""
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from lib.hal.raillm_policy import ModelTier, ModelEquivalenceClass
from lib.hal.sentinel import LimitSentinel
from lib.hal.types import ChatResponse

# OpenTelemetry instrumentation — opt-in via HAL_OTEL_ENABLED=1
_OTEL_ENABLED = os.environ.get("HAL_OTEL_ENABLED", "").strip() in ("1", "true")
_tracer = None
if _OTEL_ENABLED:
    try:
        from opentelemetry import trace
        _tracer = trace.get_tracer("hal.raillm", "0.1.0")
    except ImportError:
        _tracer = None


# ---------------------------------------------------------------------------
# Token estimation (approximate, 4 chars per token heuristic)
# ---------------------------------------------------------------------------


def _approx_tokens(text: str) -> int:
    """Approximate token count. ~4 characters per token on average.

    This is a heuristic, not a tokenizer. It is good enough for compression
    decisions at the thread level where the cost of under-estimating by 10%
    is a rejected dispatch that the cascade chain will pick up anyway.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Message + ConversationThread
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """A single message in a conversation thread.

    `provider_hint` and `model_hint` record which provider/model authored
    the assistant response. They survive across provider switches, so a
    post-hoc analysis can reconstruct the full failover chain that produced
    a given conversation.
    """

    role: str  # "user" | "assistant" | "system" | "tool"
    content: str
    provider_hint: Optional[str] = None
    model_hint: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConversationThread:
    """Stateful message history for RAILLM dispatch.

    Holds a list of `Message` objects plus an optional system prompt.
    Supports token estimation and budget-aware compression so the thread
    can be resized to fit a target provider's context window.
    """

    messages: list[Message] = field(default_factory=list)
    system_prompt: str = ""

    def append(self, msg: Message) -> None:
        """Append a message to the thread in place."""
        self.messages.append(msg)

    def token_estimate(self) -> int:
        """Approximate token count of the full thread (system + messages)."""
        total = _approx_tokens(self.system_prompt)
        for m in self.messages:
            total += _approx_tokens(m.content)
        return total

    def _fit_to_budget(
        self, budget_tokens: int
    ) -> tuple[list[Message], int]:
        """Pure: ``(kept_messages, dropped_count)`` for ``budget_tokens``
        WITHOUT mutating ``self``. System messages are always kept; oldest
        non-system messages drop first; the most recent non-system message
        is never dropped (even if it alone exceeds the budget).

        Single source of truth for the drop-oldest fitting algorithm —
        shared by the mutating ``compress_if_needed`` and the non-mutating
        ``compressed_prompt`` (DRY; HAL#329).
        """
        if self.token_estimate() <= budget_tokens:
            return list(self.messages), 0

        system_msgs = [m for m in self.messages if m.role == "system"]
        non_system = [m for m in self.messages if m.role != "system"]
        if not non_system:
            return list(self.messages), 0

        dropped = 0
        sys_tok = _approx_tokens(self.system_prompt)
        # Always keep the last non-system message, even if it exceeds alone.
        while len(non_system) > 1:
            est = sys_tok + sum(
                _approx_tokens(m.content) for m in system_msgs + non_system
            )
            if est <= budget_tokens:
                break
            non_system.pop(0)
            dropped += 1

        return system_msgs + non_system, dropped

    def compress_if_needed(self, budget_tokens: int) -> int:
        """Drop oldest non-system messages until the thread fits the budget.

        MUTATES ``self`` in place (backward-compatible — direct callers and
        the unit tests rely on this). Returns the number of messages dropped
        (0 if none).

        For the failover WIRE path prefer ``compressed_prompt``: it does NOT
        mutate the shared thread, so a failed small-window rung cannot
        destroy history before the larger fallback rung is tried (HAL#329).
        """
        kept, dropped = self._fit_to_budget(budget_tokens)
        if dropped:
            self.messages = kept
        return dropped

    def compressed_prompt(self, budget_tokens: int) -> tuple[str, int]:
        """NON-mutating: ``(prompt_string, dropped_count)`` for a COPY of
        the thread fitted to ``budget_tokens``. ``self.messages`` is
        untouched, so this is safe to call per-rung in a failover cascade —
        the shared ConversationThread stays the canonical full history.

        The HAL#329 fix: the previous wire path called the *mutating*
        ``compress_if_needed`` for every attempted rung, so a failing
        small-window primary destructively truncated the shared thread
        before the larger fallback rung was tried.
        """
        kept, dropped = self._fit_to_budget(budget_tokens)
        return self._render(kept), dropped

    def _render(self, messages: list[Message]) -> str:
        """Pure: flatten ``messages`` + system prompt to the XML-tagged
        prompt string. One definition of the wire format — used by
        ``to_prompt`` (full thread) and ``compressed_prompt`` (fitted copy).
        """
        parts: list[str] = []
        if self.system_prompt:
            parts.append(f"<system>\n{self.system_prompt}\n</system>")
        for msg in messages:
            parts.append(f"<{msg.role}>\n{msg.content}\n</{msg.role}>")
        return "\n\n".join(parts)

    def to_prompt(self) -> str:
        """Flatten the FULL thread to a single prompt string with XML role
        tags. Used when the target provider only accepts a single `prompt`
        parameter instead of a messages array. Backward-compatible with HAL
        providers not yet upgraded to the OpenAI-style messages interface.
        """
        return self._render(self.messages)

    def to_messages(self) -> list[dict]:
        """Flatten the thread to the OpenAI-style messages array.

        Used when the target provider accepts a native `messages` parameter.
        The system prompt is emitted as a single leading system message
        when present.
        """
        out: list[dict] = []
        if self.system_prompt:
            out.append({"role": "system", "content": self.system_prompt})
        for msg in self.messages:
            out.append({"role": msg.role, "content": msg.content})
        return out


# ---------------------------------------------------------------------------
# Model equivalence tiers
# ---------------------------------------------------------------------------


# Default equivalence map. Callers may pass a custom map to
# RAILLMSentinel.__init__ to override or extend these defaults.
# Model addresses use HAL's provider/model convention from
# config/providers.yaml.
DEFAULT_EQUIVALENCE_MAP: dict[ModelTier, ModelEquivalenceClass] = {
    ModelTier.FRONTIER: ModelEquivalenceClass(
        tier=ModelTier.FRONTIER,
        members=[
            # Carbon replaced gemini-3.1-pro-preview here on 2026-11-26 as a seat
            # move (V>>), not a retirement: 3.1 Pro is still served and priced.
            "gemini/gemini-4-carbon",
            "openai/gpt-4o",
            "deepseek/deepseek-reasoner",
        ],
        context_window=200_000,
    ),
    ModelTier.FAST: ModelEquivalenceClass(
        tier=ModelTier.FAST,
        members=[
            "groq/llama-3.3-70b-versatile",
            "openai/gpt-4o-mini",
            "qwen/qwen3.5-flash",
            "deepseek/deepseek-chat",
        ],
        context_window=128_000,
    ),
    ModelTier.FREE_TIER: ModelEquivalenceClass(
        tier=ModelTier.FREE_TIER,
        members=[
            "groq/llama-3.3-70b-versatile",
            "gemini/gemini-3.5-flash-lite",
            "openrouter/google/gemma-4-31b-it",
        ],
        context_window=128_000,
    ),
    ModelTier.LOCAL: ModelEquivalenceClass(
        tier=ModelTier.LOCAL,
        members=[
            "ollama/gemma4:e4b",
            "lemonade/auto",
            "mlx_vlm/auto",
        ],
        context_window=32_000,
    ),
}


def classify_model(
    model_address: str,
    equivalence_map: Optional[dict[ModelTier, ModelEquivalenceClass]] = None,
) -> ModelTier:
    """Look up which tier a model belongs to.

    Models not in any tier default to FAST (safe middle ground). Callers
    can supply their own `equivalence_map` to customise tier assignments
    for test fixtures or custom deployments.
    """
    emap = equivalence_map or DEFAULT_EQUIVALENCE_MAP
    for tier, eq in emap.items():
        if model_address in eq.members:
            return tier
    return ModelTier.FAST


# ---------------------------------------------------------------------------
# RAILLMSentinel
# ---------------------------------------------------------------------------


class RAILLMExhausted(RuntimeError):
    """All providers in the cascade chain have been tried and none accepted
    the request. The exception message contains the per-provider failure
    reasons for debugging."""


class RAILLMSentinel(LimitSentinel):
    """Stateful failover sentinel — the R in RAILLM.

    Extends `LimitSentinel` with:

    - ConversationThread tracking — history carries across provider switches
    - Model equivalence-class cascade — tier-aware fallback
    - Context-window-aware compression at each switch boundary

    The base class already handles limit detection, exponential backoff,
    and persistent state. This subclass adds the history layer without
    modifying the base class at all (pure composition + one method add).
    """

    def __init__(
        self,
        registry: "ProviderRegistry",
        thread: Optional[ConversationThread] = None,
        equivalence_map: Optional[dict[ModelTier, ModelEquivalenceClass]] = None,
        proactive_threshold: float = 0.85,
        warning_threshold: float = 0.70,
    ) -> None:
        super().__init__(
            registry=registry,
            proactive_threshold=proactive_threshold,
            warning_threshold=warning_threshold,
        )
        self._thread = thread or ConversationThread()
        self._equivalence_map = equivalence_map or DEFAULT_EQUIVALENCE_MAP

    @property
    def thread(self) -> ConversationThread:
        """The stateful conversation thread. Callers may inspect it."""
        return self._thread

    def dispatch_with_history(
        self,
        model_address: str,
        user_message: str,
        tier_fallback: Optional[list[str]] = None,
        no_failover: bool = False,
        **kwargs,
    ) -> ChatResponse:
        """Route a user message with limit-aware, history-preserving failover.

        ``no_failover=True`` makes the chain the primary address alone, so a
        pinned request never reaches a rung the caller did not name (HAL#673);
        its failure raises ``RAILLMExhausted`` as an exhausted chain does.

        Protocol:

        1. Append `user_message` to the conversation thread immediately.
           (Even if dispatch fails, the operator can inspect `self.thread`
           to see what was attempted.)
        2. Build a tier-aware cascade chain: primary address, primary tier
           peers, explicit fallback, then cascade through FRONTIER -> FAST
           -> FREE_TIER -> LOCAL.
        3. For each attempted provider:

           a. Skip if the base class reports it near its limit.
           b. Compress the thread to the target's context window budget.
           c. Call `provider.chat(prompt=thread.to_prompt(), ...)`.
           d. On success: append assistant response to thread, record
              response headers via `_post_response`, return.
           e. On error: record via `_on_error`, continue to next attempt.

        4. If all providers exhausted, raise `RAILLMExhausted` with the
           full failure chain.
        """
        # Local imports to avoid circular imports at module load.
        from lib.hal.providers.openai_compat import OpenAICompatError
        from lib.hal.registry import parse_model_address

        self._thread.append(Message(role="user", content=user_message))

        primary_tier = classify_model(model_address, self._equivalence_map)
        chain = ([model_address] if no_failover
                 else self._build_tier_chain(model_address, primary_tier, tier_fallback))
        errors: list[str] = []

        # OTel: parent span for the full dispatch
        _span = _tracer.start_span(
            "raillm.dispatch_with_history",
            attributes={
                "raillm.primary_model": model_address,
                "raillm.primary_tier": primary_tier.value,
                "raillm.chain_length": len(chain),
                "raillm.thread_tokens": self._thread.token_estimate(),
            },
        ) if _tracer else None

        for attempt_address in chain:
            attempt_provider_name, attempt_model_name = parse_model_address(
                attempt_address
            )
            provider = self._registry.get(attempt_provider_name)
            if not provider:
                errors.append(f"{attempt_address}: provider not registered")
                continue

            if self._is_near_limit(attempt_provider_name):
                errors.append(f"{attempt_address}: near limit, skipped")
                continue

            attempt_tier = classify_model(attempt_address, self._equivalence_map)
            budget = self._equivalence_map[attempt_tier].context_window
            # HAL#329: NON-mutating per-rung wire copy. A failed small-window
            # rung must NOT destroy the shared thread before the larger
            # fallback rung is tried -- the thread is the canonical history;
            # only the serving rung's reply is appended to it (below).
            prompt, dropped = self._thread.compressed_prompt(budget)
            if dropped:
                errors.append(
                    f"{attempt_address}: wire copy compressed, dropped "
                    f"{dropped} oldest message(s) (shared thread intact)"
                )

            # OTel: child span per cascade attempt
            _attempt_span = _tracer.start_span(
                "raillm.attempt",
                attributes={
                    "raillm.attempt_model": attempt_address,
                    "raillm.attempt_tier": attempt_tier.value,
                    "raillm.messages_dropped": dropped,
                },
            ) if _tracer else None

            try:
                response = provider.chat(
                    prompt=prompt, model=attempt_model_name, **kwargs
                )
                self._post_response(attempt_provider_name, response)
                self._thread.append(
                    Message(
                        role="assistant",
                        content=response.text,
                        provider_hint=attempt_provider_name,
                        model_hint=attempt_model_name,
                    )
                )
                if _attempt_span:
                    _attempt_span.set_attribute("raillm.success", True)
                    _attempt_span.end()
                if _span:
                    _span.set_attribute("raillm.served_by", attempt_address)
                    _span.end()
                return response
            except OpenAICompatError as e:
                if _attempt_span:
                    _attempt_span.set_attribute("raillm.error", str(e)[:200])
                    _attempt_span.end()
                self._on_error(attempt_provider_name, e)
                errors.append(f"{attempt_address}: {e}")
                continue
            except (ConnectionError, TimeoutError, OSError) as e:
                if _attempt_span:
                    _attempt_span.set_attribute("raillm.error", str(e)[:200])
                    _attempt_span.end()
                errors.append(f"{attempt_address}: {type(e).__name__}: {e}")
                continue

        if _span:
            _span.set_attribute("raillm.exhausted", True)
            _span.end()
        raise RAILLMExhausted(
            "All RAILLM providers exhausted: " + "; ".join(errors)
        )

    def _build_tier_chain(
        self,
        primary_address: str,
        primary_tier: ModelTier,
        explicit_fallback: Optional[list[str]],
    ) -> list[str]:
        """Build the ordered cascade chain for `dispatch_with_history`.

        Order:

        1. Primary address first.
        2. Remaining members of the primary tier.
        3. Explicit fallback addresses (operator-supplied).
        4. Cascade through FRONTIER -> FAST -> FREE_TIER -> LOCAL,
           skipping the primary tier (already emitted).

        Duplicates are removed while preserving first-occurrence order.
        """
        emap = self._equivalence_map
        # Straight-line priority list + dict.fromkeys dedup: O(n) with no
        # nested loops and no linear "not in list" probes. The order:
        # primary address first, primary-tier peers, explicit fallback,
        # then cascade through remaining tiers (skipping primary_tier).
        candidates: list[str] = [primary_address]
        candidates.extend(emap[primary_tier].members)
        if explicit_fallback:
            candidates.extend(explicit_fallback)
        if primary_tier is not ModelTier.FRONTIER:
            candidates.extend(emap[ModelTier.FRONTIER].members)
        if primary_tier is not ModelTier.FAST:
            candidates.extend(emap[ModelTier.FAST].members)
        if primary_tier is not ModelTier.FREE_TIER:
            candidates.extend(emap[ModelTier.FREE_TIER].members)
        if primary_tier is not ModelTier.LOCAL:
            candidates.extend(emap[ModelTier.LOCAL].members)
        return list(dict.fromkeys(candidates))
