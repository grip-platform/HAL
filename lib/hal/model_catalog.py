"""HAL Model Catalog — metadata registry for available LLM models.

Tracks model capabilities, context windows, pricing, and recommended
use cases. Used by the cost router and model comparator to make
informed decisions.

Usage:
    from lib.hal.model_catalog import ModelCatalog

    catalog = ModelCatalog()
    info = catalog.get("claude-opus-4-7")
    models = catalog.find(min_context=100000, max_cost=5.0)
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for a single LLM model."""
    name: str
    provider: str
    context_window: int
    input_cost_per_mtok: float
    output_cost_per_mtok: float
    strengths: tuple[str, ...] = ()
    max_output_tokens: int = 4096

    @property
    def is_free(self) -> bool:
        return self.input_cost_per_mtok == 0 and self.output_cost_per_mtok == 0


_CATALOG: dict[str, ModelInfo] = {
    # Anthropic
    # Opus 4.7 (April 2026, 1M context GA) — current flagship; 4.6 retained for historical lookup
    "claude-opus-4-7": ModelInfo("claude-opus-4-7", "anthropic", 1_000_000, 5.0, 25.0,
        ("reasoning", "code", "analysis", "writing", "agentic"), 32_000),
    "claude-opus-4-6": ModelInfo("claude-opus-4-6", "anthropic", 200_000, 5.0, 25.0,
        ("reasoning", "code", "analysis", "writing"), 32_000),
    "claude-sonnet-4-6": ModelInfo("claude-sonnet-4-6", "anthropic", 200_000, 3.0, 15.0,
        ("code", "speed", "balance"), 16_000),
    "claude-haiku-4-5": ModelInfo("claude-haiku-4-5", "anthropic", 200_000, 1.0, 5.0,
        ("speed", "cost"), 8_000),
    # OpenAI
    "gpt-4o": ModelInfo("gpt-4o", "openai", 128_000, 2.5, 10.0,
        ("multimodal", "reasoning"), 16_384),
    "gpt-4o-mini": ModelInfo("gpt-4o-mini", "openai", 128_000, 0.15, 0.6,
        ("speed", "cost"), 16_384),
    "o3-mini": ModelInfo("o3-mini", "openai", 128_000, 1.1, 4.4,
        ("reasoning", "code"), 65_536),
    # Groq (free tier)
    "llama-3.3-70b-versatile": ModelInfo("llama-3.3-70b-versatile", "groq", 128_000, 0.59, 0.79,
        ("speed", "cost", "open-source"), 8_000),
    "llama-4-scout-17b-16e-instruct": ModelInfo("llama-4-scout-17b-16e-instruct", "groq", 128_000, 0.11, 0.34,
        ("speed", "cost", "open-source"), 8_000),
    "openai/gpt-oss-120b": ModelInfo("openai/gpt-oss-120b", "groq", 131_072, 0.15, 0.60,
        ("reasoning", "code", "open-source"), 8_000),
    # Google. Repointed from gemini-2.5-flash on 2026-11-24 (GRIP#9389): Google
    # auto-routes that id to gemini-3.5-flash-lite from 2026-12-01. Price
    # corrected to the vendor's published rate in the same edit (the old row's
    # 0.15/0.6 was already a retired rate per lib/hal/cost_tracker.py:108).
    "gemini-3.5-flash-lite": ModelInfo("gemini-3.5-flash-lite", "google", 1_048_576, 0.30, 2.50,
        ("long-context", "speed", "cost"), 65_536),
    # Prices from ai.google.dev/gemini-api/docs/pricing, read 2026-10-18 (the $1.25 /
    # $10 this row carried was 2.5 Pro's, the drift cost_tracker.py fixed on
    # 2026-10-01); windows from the vendor's model pages the same day.
    "gemini-3.1-pro-preview": ModelInfo("gemini-3.1-pro-preview", "google", 1_048_576, 2.0, 12.0,
        ("reasoning", "code", "analysis", "long-context"), 65_536),
    # Gemini 4 Argon: introductory rate through 2026-11-30, then $4 / $20
    # (lib/hal/cost_tracker._PRICE_SCHEDULE carries the date).
    "gemini-4-argon": ModelInfo("gemini-4-argon", "google", 1_048_576, 2.0, 10.0,
        ("reasoning", "code", "agentic", "analysis", "long-context"), 65_536),
    # The flash generation after 3.5 (3.5-flash deprecated 2026-10-08); prices through
    # 2026-12-31, doubling on 2027-01-01 per lib/hal/cost_tracker._PRICE_SCHEDULE.
    "gemini-3.6-flash": ModelInfo("gemini-3.6-flash", "google", 1_048_576, 0.75, 3.75,
        ("speed", "cost", "code", "agentic", "multimodal"), 65_536),
    "gemini-3.8-flash": ModelInfo("gemini-3.8-flash", "google", 1_048_576, 0.75, 3.75,
        ("speed", "cost", "code", "agentic", "multimodal"), 65_536),
    # Gemini 4 Carbon: introductory rate through 2026-11-30, then $4 / $20
    # (lib/hal/cost_tracker._PRICE_SCHEDULE carries the date). GA 2026-11-10.
    "gemini-4-carbon": ModelInfo("gemini-4-carbon", "google", 1_048_576, 2.0, 10.0,
        ("reasoning", "code", "cybersecurity", "agentic", "long-context"), 65_536),
    # Gemini 4 Hydrogen: flat rate, no dated change. GA 2026-11-12.
    "gemini-4-hydrogen": ModelInfo("gemini-4-hydrogen", "google", 1_048_576, 1.50, 7.50,
        ("reasoning", "code", "agentic", "long-context", "value"), 65_536),
    # Gemini 4 Neon: flat rate, no dated change. GA 2026-11-18.
    "gemini-4-neon": ModelInfo("gemini-4-neon", "google", 1_048_576, 3.0, 15.0,
        ("reasoning", "code", "agentic", "science", "long-context"), 65_536),
    # Google Gemma 4 (open weights, Apache 2.0 — released 2 Apr 2026)
    "google/gemma-4-31b-it": ModelInfo("google/gemma-4-31b-it", "openrouter", 256_000, 0.14, 0.40,
        ("reasoning", "code", "agentic", "multimodal", "open-source"), 131_072),
    "google/gemma-4-26b-a4b-it": ModelInfo("google/gemma-4-26b-a4b-it", "openrouter", 256_000, 0.10, 0.30,
        ("reasoning", "speed", "open-source"), 131_072),
    # DeepSeek (V3.2)
    "deepseek-chat": ModelInfo("deepseek-chat", "deepseek", 128_000, 0.28, 0.42,
        ("cost", "code", "reasoning"), 8_192),
    "deepseek-reasoner": ModelInfo("deepseek-reasoner", "deepseek", 128_000, 0.28, 0.42,
        ("reasoning", "code"), 8_192),
    # Qwen (Alibaba DashScope)
    "qwen3.5-flash": ModelInfo("qwen3.5-flash", "qwen", 256_000, 0.065, 0.260,
        ("speed", "cost", "reasoning"), 8_192),
    "qwen3.5-plus": ModelInfo("qwen3.5-plus", "qwen", 256_000, 0.30, 1.20,
        ("reasoning", "code", "multilingual"), 8_192),
    "qwen3.6-plus-preview": ModelInfo("qwen3.6-plus-preview", "qwen", 256_000, 0.30, 1.20,
        ("reasoning", "code"), 8_192),
    # Moonshot (Kimi) -- live GET /v1/models 2026-09-07; kimi-k2.5 and the
    # moonshot-v1 family were retired 2026-08-31. Prices per platform.kimi.ai
    # docs/pricing/chat-k26.md, chat-k27-code.md, chat-k3.md. Max output is the
    # context ceiling minus the prompt (docs/guide/troubleshooting.md: k2.x
    # "256*1024 - prompt_tokens", k3 "1024*1024 - prompt_tokens").
    "kimi-k2.6": ModelInfo("kimi-k2.6", "moonshot", 262_144, 0.95, 4.00,
        ("agentic", "reasoning", "multimodal", "long-context"), 262_144),
    "kimi-k2.7-code": ModelInfo("kimi-k2.7-code", "moonshot", 262_144, 0.95, 4.00,
        ("code", "agentic", "reasoning", "multimodal"), 262_144),
    "kimi-k2.7-code-highspeed": ModelInfo("kimi-k2.7-code-highspeed", "moonshot", 262_144, 1.90, 8.00,
        ("code", "agentic", "speed", "multimodal"), 262_144),
    "kimi-k3": ModelInfo("kimi-k3", "moonshot", 1_048_576, 3.00, 15.00,
        ("long-context", "agentic", "reasoning", "multimodal"), 1_048_576),
    # Zhipu (GLM)
    "glm-5": ModelInfo("glm-5", "zhipu", 128_000, 0.65, 1.95,
        ("reasoning", "multilingual", "open-source"), 8_192),
    "glm-4.7": ModelInfo("glm-4.7", "zhipu", 128_000, 0.50, 1.50,
        ("code", "reasoning", "open-source"), 8_192),
    "glm-4.7-flash": ModelInfo("glm-4.7-flash", "zhipu", 128_000, 0.0, 0.0,
        ("speed", "cost", "open-source"), 8_192),
    # Mistral
    "mistral-small-latest": ModelInfo("mistral-small-latest", "mistral", 262_000, 0.15, 0.60,
        ("long-context", "open-source", "code"), 32_768),
    # Meta Llama 4 (via OpenRouter)
    "meta-llama/llama-4-maverick": ModelInfo("meta-llama/llama-4-maverick", "openrouter", 1_000_000, 0.15, 0.60,
        ("long-context", "multimodal", "open-source"), 131_072),
    # Local Ollama models (free, no egress)
    "gemma4:e4b": ModelInfo("gemma4:e4b", "ollama", 128_000, 0.0, 0.0,
        ("speed", "cost", "open-source", "local"), 4_096),
    # Gemma 4 variants (Google, Apache 2.0) — via Ollama
    # 26B-A4B: MoE (128 experts, 9 active per token = 4B effective params)
    # Near-31B quality at 4B inference speed, 256K context, native multimodal
    "gemma4:26b": ModelInfo("gemma4:26b", "ollama", 256_000, 0.0, 0.0,
        ("quality", "speed", "multimodal", "open-source", "local", "apache2"), 8_192),
    "gemma4:27b": ModelInfo("gemma4:27b", "ollama", 128_000, 0.0, 0.0,
        ("quality", "open-source", "local", "apache2"), 8_192),
    "gemma4:12b": ModelInfo("gemma4:12b", "ollama", 128_000, 0.0, 0.0,
        ("speed", "cost", "open-source", "local", "apache2"), 8_192),
    # DeepSeek V3.2 — near-frontier at $0.28/M input
    "deepseek-v3.2": ModelInfo("deepseek-v3.2", "deepseek", 128_000, 0.00028, 0.00112,
        ("cost", "quality", "coding"), 8_192),
    # Trinity Large Thinking (Arcee) was here until 2026-11-24, when Arcee
    # stopped serving it (HTTP 404).
}


class ModelCatalog:
    """Registry of LLM model metadata."""

    def get(self, name: str) -> ModelInfo | None:
        return _CATALOG.get(name)

    def find(self, min_context: int = 0, max_cost: float = 0, provider: str = "") -> list[ModelInfo]:
        """Find models matching criteria."""
        results = list(_CATALOG.values())
        if min_context:
            results = [m for m in results if m.context_window >= min_context]
        if max_cost > 0:
            results = [m for m in results if m.input_cost_per_mtok <= max_cost]
        if provider:
            results = [m for m in results if m.provider == provider]
        return sorted(results, key=lambda m: m.input_cost_per_mtok)

    def list_all(self) -> list[ModelInfo]:
        return sorted(_CATALOG.values(), key=lambda m: m.name)

    def providers(self) -> list[str]:
        return sorted({m.provider for m in _CATALOG.values()})
