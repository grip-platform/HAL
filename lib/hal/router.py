"""HAL Router — routing strategies for multi-provider LLM dispatch.

Architecture: adapter-pattern dispatch (NOT an if/elif chain). The Router holds
a dispatch table mapping every `RoutingStrategy` -> handler method — including
an explicit `RoutingStrategy.COUNCIL` -> `_route_council` entry that is a
present-tense boundary (it raises rather than executes; see below). Per-backend
quirks (Gemini safety blocks, DeepSeek token counting) live behind the
`OpenAICompatProvider` adapter in `lib/hal/providers/`. There are zero
backend-specific `if`/`elif` arms in this file — adding a new PROVIDER does NOT
require editing router.py. (The one explicit per-STRATEGY branch, the COUNCIL
boundary, is a strategy decision, not a backend arm.)

Linguistic-tell discipline (issue #305): if a future docstring describes
this file as "handling Gemini safety blocks, handling DeepSeek tokens, ...",
that is a tell for drift to a conditional chain — refactor back to the
dispatch table. The right verbs are "routes to" / "delegates to".

Three strategies the Router EXECUTES:
- Fixed: User selects model explicitly (provider/model addressing)
- Fallback: Ordered chain, try next on failure/rate-limit
- Quality-Cost: Triage simple->cheap, complex->premium

RoutingStrategy.COUNCIL is a fourth enum member but is NOT executed here — a
council returns a multi-vote deliberation, not a single ChatResponse. Its
handler (`_route_council`) is a present-tense boundary that names the canonical
council homes (council_agent.py / moe.py / the :4010 gateway 'council' model id,
ADR 0012). See `_route_council` for the full rationale.

Sync invariant: see tests/test_router.py — every provider in
config/providers.yaml must be registry-reachable; the test fails CI if a
new provider is added without a routable entry. Extends the H245 pattern
from HAPPI runtime to the router/registry layer.

Usage:
    from lib.hal.router import Router

    router = Router(registry)
    response = router.route("Explain SOLID", strategy="fallback")
"""
from dataclasses import replace
from typing import NoReturn, Optional

from lib.hal.providers.openai_compat import OpenAICompatError
from lib.hal.cost_router import _classify_tier, _resolve_pricing
from lib.hal.types import ChatResponse, ProviderType, RoutingStrategy

# Actions/intents where data should not leave Western jurisdictions.
# DeepSeek (China) and Qwen/Dashscope (China) are excluded for these topics.
_SENSITIVE_KEYWORDS: frozenset[str] = frozenset({
    "audit", "legal", "compliance", "gdpr", "pii", "personally identifiable",
    "privacy", "contract", "lawsuit", "litigation", "regulatory", "sox",
    "hipaa", "confidential",
})

_JURISDICTION_RESTRICTED: frozenset[str] = frozenset({"deepseek", "qwen"})


def _is_sensitive(message: str) -> bool:
    """Return True if the message contains jurisdiction-sensitive content."""
    lower = message.lower()
    return any(kw in lower for kw in _SENSITIVE_KEYWORDS)


def _local_provider_names(registry: "ProviderRegistry") -> frozenset[str]:
    """Names of providers configured as locally hosted (ProviderType.LOCAL).

    The jurisdiction guard exists because cloud APIs send data off the user's
    machine to a third party in another jurisdiction. A locally-hosted model
    (Ollama/vLLM/LM Studio/MLX on localhost) never sends data anywhere — the
    weights' origin jurisdiction (e.g. Qwen/DeepSeek built in China) is then
    irrelevant because nothing leaves the firm's own infrastructure. The
    authoritative signal is the provider's declared `type` in
    config/providers.yaml, parsed into ProviderType.LOCAL — not the model name.
    """
    configs = getattr(registry, "_configs", None)
    if not configs:
        return frozenset()
    return frozenset(
        name for name, cfg in configs.items()
        if getattr(cfg, "type", None) == ProviderType.LOCAL
    )


def _filter_sensitive_providers(
    chain: list[str], message: str, local_providers: frozenset[str] = frozenset(),
) -> list[str]:
    """Remove jurisdiction-restricted providers if the message is sensitive.

    A provider in `local_providers` is NEVER removed regardless of the model
    or weight origin: locally-hosted inference keeps data on the user's own
    infrastructure, so the jurisdiction guard does not apply to it. Only
    remote/cloud jurisdiction-restricted providers are dropped — the original
    guard is preserved unchanged for them.
    """
    if not _is_sensitive(message):
        return chain
    kept: list[str] = []
    for entry in chain:
        provider_name = entry.split("/", 1)[0]
        if provider_name in local_providers:
            kept.append(entry)
            continue
        if not any(restricted in entry for restricted in _JURISDICTION_RESTRICTED):
            kept.append(entry)
    return kept


# Single source of truth for the COUNCIL boundary explanation. Referenced by
# BOTH ``Router._route_council``'s raised ``NotImplementedError`` AND its
# ``__doc__`` (assigned just below the method), so the docstring and the error
# string can never drift apart (AGORA DEFER, PR #500 concern 3). Deliberately
# carries no deferral/scrub phrasing — it names where council actually lives.
_COUNCIL_BOUNDARY_MESSAGE = (
    "RoutingStrategy.COUNCIL is not a Router.route concern by design: a council "
    "returns a multi-vote deliberation (CouncilDeliberation / CouncilResult), not "
    "a single ChatResponse, and needs a LIST of member models that this "
    "single-model strategy path does not carry. Council routing is owned by two "
    "canonical layers — the agent/engine layer "
    "(lib/hal/council_agent.py::CouncilRunner.deliberate and the 3-phase "
    "lib/hal/moe.py::HALCouncil), and the wire layer (the :4010 gateway "
    "'council' / 'sovereign-council' model id, "
    "lib/hal/gateway_openai.py::_relay_council, GRIP ADR 0012 / PR #491). "
    "Selecting RoutingStrategy.COUNCIL here is a configuration error (e.g. "
    "routing.default: council in providers.yaml) — use fixed / fallback / "
    "quality_cost instead. This is a present-tense boundary that names where "
    "council lives, a deliberate design decision rather than a deferral."
)


class Router:
    """Routes queries across providers using configurable strategies."""

    def __init__(self, registry: "ProviderRegistry") -> None:
        self._registry = registry

    def route(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        strategy: Optional[RoutingStrategy] = None,
        **kwargs,
    ) -> ChatResponse:
        """Route a query using the specified or default strategy."""
        routing = self._registry.routing
        strat = strategy or routing.default_strategy

        dispatch = {
            RoutingStrategy.FIXED: self._route_fixed,
            RoutingStrategy.FALLBACK: self._route_fallback,
            RoutingStrategy.QUALITY_COST: self._route_quality_cost,
            RoutingStrategy.COUNCIL: self._route_council,
        }
        handler = dispatch.get(strat)
        if handler is None:
            # Present-tense boundary, not a deferral: a strategy with no entry in
            # the dispatch table above is a wiring error (a new enum member added
            # without a handler), not "work coming later".
            raise NotImplementedError(
                f"Routing strategy '{strat}' has no registered handler in "
                f"Router.route's dispatch table — wire it in before use."
            )
        return handler(prompt, model=model, **kwargs)

    def _route_fixed(self, prompt: str, *, model: Optional[str] = None, **kwargs) -> ChatResponse:
        """Fixed routing — direct dispatch to specified model."""
        address = model or self._registry.routing.default_model
        if not address:
            raise RuntimeError("No model specified and no default configured")
        return replace(self._registry.chat(address, prompt, **kwargs), reason="fixed")

    def _route_fallback(self, prompt: str, *, model: Optional[str] = None, **kwargs) -> ChatResponse:
        """Fallback routing — try chain in order, skip failures."""
        chain = self._registry.routing.fallback_chain
        if not chain:
            return self._route_fixed(prompt, model=model, **kwargs)

        chain = _filter_sensitive_providers(
            chain, prompt, _local_provider_names(self._registry)
        )
        errors = []
        for i, address in enumerate(chain):
            try:
                # i == 0 is the chain's primary; any later index is a real fallback.
                return replace(
                    self._registry.chat(address, prompt, **kwargs),
                    reason="fallback" if i > 0 else "fixed",
                )
            except (OpenAICompatError, KeyError) as e:
                errors.append(f"{address}: {e}")
                continue

        raise RuntimeError(
            f"All providers in fallback chain failed: {'; '.join(errors)}"
        )

    def _route_quality_cost(
        self, prompt: str, *, model: Optional[str] = None, **kwargs,
    ) -> ChatResponse:
        """Quality-cost routing — triage by estimated complexity.

        On failure, tries the tier-scoped fallback chain before escalating
        to global fallback, preserving cost intent.
        """
        tiers = self._registry.routing.quality_cost_tiers
        if not tiers:
            return self._route_fixed(prompt, model=model, **kwargs)

        tier = _classify_complexity(prompt)
        address = tiers.get(tier, tiers.get("standard", ""))
        if not address:
            return self._route_fixed(prompt, model=model, **kwargs)

        try:
            return replace(
                self._registry.chat(address, prompt, **kwargs),
                reason=f"quality-cost:{tier}",
            )
        except OpenAICompatError:
            return self._route_tier_chain(prompt, address, **kwargs)

    def _route_tier_chain(
        self, prompt: str, failed_address: str, **kwargs,
    ) -> ChatResponse:
        """Try tier-scoped fallbacks, then global standard tier."""
        cost_tier = _detect_cost_tier(failed_address)
        chain = _filter_sensitive_providers(
            self._registry.routing.tier_chains.get(cost_tier, []),
            prompt,
            _local_provider_names(self._registry),
        )
        errors = [failed_address]

        for address in chain:
            if address == failed_address:
                continue
            try:
                return replace(
                    self._registry.chat(address, prompt, **kwargs),
                    reason="fallback",
                )
            except (OpenAICompatError, KeyError) as e:
                errors.append(f"{address}: {e}")

        # Global standard-tier fallback as last resort
        standard = self._registry.routing.quality_cost_tiers.get("standard", "")
        if standard and standard not in errors:
            return replace(
                self._registry.chat(standard, prompt, **kwargs),
                reason="fallback",
            )
        raise RuntimeError(f"All tier-chain providers failed: {'; '.join(errors)}")

    def _route_council(self, prompt: str, *, model: Optional[str] = None, **kwargs) -> NoReturn:
        # No literal docstring on purpose: ``__doc__`` is assigned from the shared
        # ``_COUNCIL_BOUNDARY_MESSAGE`` constant just below, so this boundary's
        # prose and its raised message are the SAME string and can never drift
        # apart (AGORA DEFER, PR #500 concern 3). The method unconditionally
        # raises — hence ``-> NoReturn`` (PR #500 concern 1); a council is
        # dispatched externally (council_agent.py / moe.py / the :4010 gateway),
        # never by the Router.
        raise NotImplementedError(_COUNCIL_BOUNDARY_MESSAGE)

    _route_council.__doc__ = _COUNCIL_BOUNDARY_MESSAGE


def _detect_cost_tier(model_address: str) -> str:
    """Detect the cost tier for a model address. Returns 'mid' as safe default."""
    model_name = model_address.split("/", 1)[1] if "/" in model_address else model_address
    pricing = _resolve_pricing(model_name)
    return _classify_tier(pricing[0]) if pricing else "mid"


def _classify_complexity(prompt: str) -> str:
    """Simple heuristic complexity classification.

    Returns: "simple", "standard", or "complex"
    """
    length = len(prompt)
    word_count = len(prompt.split())

    complexity_signals = sum([
        word_count > 200,
        length > 2000,
        "explain" in prompt.lower() and "step" in prompt.lower(),
        "analyse" in prompt.lower() or "analyze" in prompt.lower(),
        "compare" in prompt.lower() and "contrast" in prompt.lower(),
        prompt.count("```") >= 2,
        "refactor" in prompt.lower(),
        "architecture" in prompt.lower(),
    ])

    if complexity_signals >= 3:
        return "complex"
    if complexity_signals >= 1 or word_count > 50:
        return "standard"
    return "simple"
