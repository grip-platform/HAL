"""HAL core types — data classes for the universal LLM abstraction.

All types are frozen dataclasses for immutability and hashability.
No external dependencies — stdlib only.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

# Multimodal content part — one element in a message content array.
# Format mirrors OpenAI content-part schema:
#   {"type": "text", "text": "..."}
#   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
#   {"type": "audio_url", "audio_url": {"url": "...", "format": "wav"}}
ContentPart = dict


class ProviderType(str, Enum):
    """Provider deployment category."""
    CLOUD = "cloud"
    ENTERPRISE = "enterprise"
    LOCAL = "local"
    GATEWAY = "gateway"


class RoutingStrategy(str, Enum):
    """How to select a provider/model for a query.

    FIXED / FALLBACK / QUALITY_COST are executed by ``lib/hal/router.py::Router``.
    COUNCIL is a first-class strategy but is dispatched EXTERNALLY (not by the
    Router): a council returns a multi-vote deliberation, not a single
    ChatResponse. It is owned by ``lib/hal/council_agent.py::CouncilRunner`` /
    ``lib/hal/moe.py::HALCouncil`` (agent/engine) and the ``:4010`` gateway
    ``council`` model id (``gateway_openai.py::_relay_council``, ADR 0012).
    ``Router._route_council`` is a present-tense boundary that names those homes.

    ``sentinel`` is NOT a routing strategy: the Limit-Sentinel
    (``lib/hal/sentinel.py``) is an always-on failover layer wrapping every
    dispatch, not a selectable peer of these members.
    """
    FIXED = "fixed"
    FALLBACK = "fallback"
    QUALITY_COST = "quality_cost"
    COUNCIL = "council"  # dispatched externally — see class docstring


class AuthMethod(str, Enum):
    """How to authenticate with a provider."""
    API_KEY = "api_key"
    KEYCHAIN = "keychain"
    AWS_IAM = "aws_iam"
    GCP_TOKEN = "gcp_token"
    AZURE_AD = "azure_ad"
    NONE = "none"


@dataclass(frozen=True)
class ChatMessage:
    """A single message in a conversation.

    content may be a plain string (text-only) or a list of ContentPart dicts
    for multimodal messages (text + image_url + audio_url).  The list form is
    passed through as-is to OpenAI-compatible endpoints — no special handling
    required in the provider layer.
    """
    role: str
    content: Union[str, list["ContentPart"]]
    name: Optional[str] = None
    tool_calls: Optional[tuple] = None
    tool_call_id: Optional[str] = None


@dataclass(frozen=True)
class TokenUsage:
    """Token counts from an LLM response."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


@dataclass(frozen=True)
class FallbackInfo:
    """Structured signal that LimitSentinel swapped providers mid-dispatch.

    Promoted from the magic ``response.raw["_hal_fallback"]`` dict key
    (AGORA Council #311 concern) to a typed field so callers can read it
    without fishing through a vendor-specific raw dict, and so the contract
    survives provider-adapter changes that alter ``response.raw``.

    The legacy ``response.raw["_hal_fallback"]`` mirror is still emitted by
    sentinel.py for backward-compat callers (best-effort — skipped when
    raw is non-mutable like ``MappingProxyType``); new code should prefer
    ``response.fallback_info``.
    """
    requested_provider: str
    requested_model: str
    actual_provider: str
    actual_model: str
    errors_along_way: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChatResponse:
    """Normalised response from any LLM provider."""
    text: str
    model: str
    provider: str
    finish_reason: str = "stop"
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost_usd: Optional[float] = None
    latency_ms: Optional[float] = None
    raw: Optional[dict] = None
    # H347: typed fallback signal — None when the requested provider answered.
    fallback_info: Optional[FallbackInfo] = None
    # Routing "why": a short human string naming the strategy that produced this
    # response (e.g. "fixed", "fallback", "quality-cost:mid"). Set by the Router's
    # solo strategies (fixed / fallback / quality_cost); None for direct
    # ``registry.chat`` callers that bypass routing. COUNCIL raises rather than
    # returning a ChatResponse — its reason is stamped by the server call site.
    reason: Optional[str] = None
    # H781: full ordered content blocks from the provider response, preserved
    # verbatim (text, compaction, tool_use, …). Server-side compaction
    # (anthropic-beta compact-2026-01-12) emits a ``compaction`` block that the
    # caller MUST append to message history and round-trip on the next request;
    # ``text`` alone drops it. None for providers that don't return blocks.
    raw_content: Optional[list] = None

    def sanitised_for_audit(self) -> "ChatResponse":
        """Return a copy safe to serialise to KONO embeddings / audit logs.

        Strips the legacy ``_hal_fallback`` key from ``raw`` so internal
        state doesn't leak into operator-readable artefacts (AGORA Council
        #311 concern #4). The typed ``fallback_info`` field is preserved
        because it's the canonical contract — auditors should see it.
        """
        if not isinstance(self.raw, dict) or "_hal_fallback" not in self.raw:
            return self
        clean_raw = {k: v for k, v in self.raw.items() if k != "_hal_fallback"}
        from dataclasses import replace
        return replace(self, raw=clean_raw)


@dataclass(frozen=True)
class CostSnapshot:
    """Accumulated cost for a session or operation."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    total_usd: float = 0.0
    model: str = ""


@dataclass(frozen=True)
class ModelInfo:
    """Metadata about an available model."""
    id: str
    provider: str
    display_name: str = ""
    context_window: int = 0
    max_output_tokens: int = 0
    supports_tools: bool = False
    supports_json_mode: bool = False
    supports_streaming: bool = True
    supports_vision: bool = False
    cost_per_mtok_input: float = 0.0
    cost_per_mtok_output: float = 0.0


@dataclass(frozen=True)
class ProviderInfo:
    """Runtime info about a registered provider."""
    name: str
    type: ProviderType
    base_url: str
    healthy: bool = True
    models: tuple[str, ...] = ()
    latency_ms: Optional[float] = None


@dataclass
class ProviderConfig:
    """Configuration for a single provider from providers.yaml."""
    name: str
    type: ProviderType
    base_url: str
    auth_method: AuthMethod = AuthMethod.API_KEY
    api_key_source: str = ""
    models: list[str] = field(default_factory=list)
    auto_discover_models: bool = False
    auto_detect: bool = False
    default: bool = False
    enabled: bool = True
    extra_headers: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 120.0
    max_retries: int = 3


@dataclass
class RoutingConfig:
    """Routing strategy configuration."""
    default_strategy: RoutingStrategy = RoutingStrategy.FIXED
    default_model: str = ""
    fallback_chain: list[str] = field(default_factory=list)
    quality_cost_tiers: dict[str, str] = field(default_factory=dict)
    tier_chains: dict[str, list[str]] = field(default_factory=dict)

    @property
    def primary(self) -> str:
        """Return the effective primary model.

        Reads fallback_chain[0] when the chain is non-empty, otherwise
        falls back to default_model.  Callers should prefer this property
        over default_model directly; default_model is effectively deprecated
        in favour of the chain's first position.
        """
        return self.fallback_chain[0] if self.fallback_chain else self.default_model


@dataclass(frozen=True)
class CouncilVote:
    """A single model's vote in a council decision."""
    model: str
    provider: str
    response: ChatResponse
    weight: float = 1.0


@dataclass(frozen=True)
class CouncilRound:
    """One phase of a parallel council deliberation."""
    round_number: int           # 0 = gut-feel, 1+ = debate rounds
    phase: str                  # "gut_feel" | "debate"
    votes: tuple["CouncilVote", ...] = ()
    synthesis: str = ""         # populated after Phase 1
    duration_ms: float = 0.0


@dataclass(frozen=True)
class TriangulatedFinding:
    """A finding classified by cross-model agreement level.

    Classification tiers (Jeffrey modes-of-reasoning absorb):
    - Kernel:     3+ models agree — high confidence
    - Supported:  2 models agree — moderate confidence
    - Hypothesis: 1 model only — worth investigating
    - Disputed:   models disagree — needs resolution
    """
    text: str
    classification: str  # "kernel" | "supported" | "hypothesis" | "disputed"
    agreement_count: int
    models: tuple[str, ...] = ()


@dataclass(frozen=True)
class CouncilResult:
    """Result of a multi-model council query."""
    merged_text: str
    votes: tuple[CouncilVote, ...] = ()
    agreement_score: float = 0.0
    total_cost_usd: float = 0.0
    rounds: tuple["CouncilRound", ...] = ()
    triangulated_findings: tuple["TriangulatedFinding", ...] = ()
