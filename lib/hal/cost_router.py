"""HAL Cost-Aware Router — pick the cheapest provider that meets quality needs.

When the user doesn't specify a provider, HAL selects the optimal one
based on the query's complexity and the provider's cost/quality profile.

Strategy:
- Simple queries (< 50 words, no code) -> cheapest provider (Groq free tier)
- Medium queries -> mid-tier (Gemini Flash, DeepSeek)
- Complex queries (code review, architecture, multi-step) -> premium (Claude, GPT-4o)
- Privacy-sensitive -> local only (Ollama)

Usage:
    from lib.hal.cost_router import CostRouter

    router = CostRouter(registry)
    model = router.select(query="Fix this null pointer", privacy=False)
    # Returns "groq/llama-3.3-70b-versatile" for simple queries
"""
import re
from dataclasses import dataclass

from lib.hal.cost_tracker import _PRICING
from lib.hal.warm_router import OllamaLivenessProbe, WarmModelRouter


@dataclass(frozen=True)
class RouteDecision:
    """Result of cost-aware routing decision."""
    model: str
    tier: str  # "free", "cheap", "mid", "premium"
    reason: str
    estimated_cost_usd: float


# Complexity signals
_CODE_INDICATORS = re.compile(r"```|def |class |function |import |const |var |let ")
_COMPLEX_INDICATORS = re.compile(
    r"\b(?:architect|design|review|refactor|migrate|optimise|security|audit|compare)\b",
    re.IGNORECASE,
)

# Vision indicators — triggers VLM pre-check before tier logic.
_VLM_INDICATORS = re.compile(
    r"\b(?:image|photo|screenshot|ocr|detect|segment|video|clip|recording|audio)\b|"
    r"data:(?:image|video|audio)/|"
    r"\.(?:png|jpg|jpeg|webp|gif|bmp|tiff|mp4|mov|avi|mkv|webm|mp3|wav|ogg|flac|m4a|aac)(?![.\w])",
    re.IGNORECASE,
)


class CostRouter:
    """Selects optimal provider based on query complexity and cost."""

    def __init__(self, registry=None):
        self._registry = registry

    def select(self, query: str, privacy: bool = False) -> RouteDecision:
        """Select the best model for this query.

        Args:
            query: The user's query text
            privacy: If True, only consider local providers
        """
        # VLM pre-check is independent of the tier pool — runs before availability guard
        vlm_result = self._try_vlm_route(query, local_only=privacy)
        if vlm_result:
            return vlm_result

        complexity = _assess_complexity(query)
        available = self._get_available_models(local_only=privacy)

        if not available:
            return RouteDecision("", "none", "No providers available", 0.0)

        # Warm-model pre-check: prefer a local Ollama model already in VRAM
        warm_result = _try_warm_route(self._registry, complexity)
        if warm_result:
            return warm_result

        # Partition by cost tier
        tiers = _partition_by_tier(available)

        # Select based on complexity
        tier_preference = {
            "simple": ("free", "cheap", "mid", "premium"),
            "medium": ("cheap", "mid", "free", "premium"),
            "complex": ("mid", "premium", "cheap", "free"),
        }

        for tier_name in tier_preference.get(complexity, ("mid",)):
            if tiers.get(tier_name):
                model = tiers[tier_name][0]
                cost = _estimate_query_cost(model, query)
                return RouteDecision(
                    model=model, tier=tier_name,
                    reason=f"{complexity} query -> {tier_name} tier",
                    estimated_cost_usd=cost,
                )

        # Fallback: first available
        model = available[0]
        return RouteDecision(model=model, tier="unknown", reason="fallback", estimated_cost_usd=0.0)

    def _try_vlm_route(self, query: str, local_only: bool = False) -> "RouteDecision | None":
        """Return a VLM RouteDecision for visual queries, or None to fall through.

        When local_only=True (privacy mode), only selects VLM providers with type=local.
        """
        if not _VLM_INDICATORS.search(query):
            return None
        vlm = _get_vlm_model(self._registry, local_only=local_only)
        if not vlm:
            return None
        return RouteDecision(
            model=vlm, tier="free",
            reason="vision query -> local mlx_vlm (free)",
            estimated_cost_usd=0.0,
        )

    def _get_available_models(self, local_only: bool = False) -> list[str]:
        """Get available model addresses from registry.

        Excludes mlx_vlm: it is multimodal-only and surfaces exclusively via
        _try_vlm_route to prevent text queries from routing to a vision model.
        """
        if not self._registry:
            return []
        models: list[str] = []
        for p in self._registry.list_providers():
            if p.name == "mlx_vlm":
                continue
            if local_only and p.type.value != "local":
                continue
            if p.models:
                models.append(f"{p.name}/{p.models[0]}")
        return models


def _try_warm_route(registry, complexity: str) -> "RouteDecision | None":
    """Return a RouteDecision for a warm local Ollama model, or None to fall through."""
    routing_config = getattr(registry, "routing_config", None) if registry else None
    if not routing_config:
        return None
    tier_chains = getattr(routing_config, "tier_chains", {})
    if not tier_chains:
        return None
    # Map complexity to a cost tier for the warm-model check
    tier = {"simple": "free", "medium": "cheap"}.get(complexity)
    if not tier:
        return None
    router = WarmModelRouter(OllamaLivenessProbe(), {}, tier_chains)
    entry = router.route(tier)
    if not entry:
        return None
    return RouteDecision(
        model=entry,
        tier=tier,
        reason=f"warm local model ({complexity} query -> {tier} tier)",
        estimated_cost_usd=0.0,
    )


def _get_vlm_model(registry, local_only: bool = False) -> str:
    """Return 'mlx_vlm/<model>' if mlx_vlm is registered, healthy, and has models.

    When local_only=True, skips any VLM provider that is not type=local (privacy guard).
    """
    if not registry:
        return ""
    for p in registry.list_providers():
        if p.name != "mlx_vlm" or not p.healthy or not p.models:
            continue
        if local_only and p.type.value != "local":
            continue
        return f"mlx_vlm/{p.models[0]}"
    return ""


def _assess_complexity(query: str) -> str:
    """Assess query complexity: simple, medium, or complex.

    Checks complex signals first (keyword-driven), then code indicators,
    then falls back to word count heuristic.
    """
    word_count = len(query.split())

    # Complex signals take priority regardless of length
    if _COMPLEX_INDICATORS.search(query) or word_count > 200:
        return "complex"

    if _CODE_INDICATORS.search(query) or word_count > 50:
        return "medium"

    return "simple"


def _resolve_pricing(model_name: str) -> tuple[float, float, float, float] | None:
    """Resolve pricing for a model name. O(1) direct lookup + O(k) prefix scan."""
    pricing = _PRICING.get(model_name)
    if pricing:
        return pricing
    # Prefix match using pre-sorted keys (longest first for greedy match)
    for prefix in _PRICING_PREFIXES:
        if model_name.startswith(prefix):
            return _PRICING[prefix]
    return None


# Pre-sorted pricing keys for prefix matching — built once at import time
_PRICING_PREFIXES = sorted(_PRICING.keys(), key=len, reverse=True)


def _classify_tier(input_cost: float) -> str:
    """Classify a model into a cost tier by input price per million tokens."""
    if input_cost == 0:
        return "free"
    if input_cost < 1.0:
        return "cheap"
    if input_cost < 5.0:
        return "mid"
    return "premium"


def _partition_by_tier(models: list[str]) -> dict[str, list[str]]:
    """Partition models into cost tiers. O(n * k) where k = len(pricing keys)."""
    tiers: dict[str, list[str]] = {"free": [], "cheap": [], "mid": [], "premium": []}

    for model_addr in models:
        model_name = model_addr.split("/", 1)[1] if "/" in model_addr else model_addr
        pricing = _resolve_pricing(model_name)
        tier = _classify_tier(pricing[0]) if pricing else "mid"
        tiers[tier].append(model_addr)

    return tiers


def _estimate_query_cost(model_addr: str, query: str) -> float:
    """Rough cost estimate for a query."""
    model_name = model_addr.split("/", 1)[1] if "/" in model_addr else model_addr
    pricing = _PRICING.get(model_name)
    if not pricing:
        return 0.0

    input_tokens = len(query.split()) / 0.75
    output_tokens = 500  # estimate
    return (input_tokens / 1_000_000) * pricing[0] + (output_tokens / 1_000_000) * pricing[1]
