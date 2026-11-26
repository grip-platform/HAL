"""HAL — Harness Abstraction Layer.

The middle layer between GRIP and any LLM runtime. Supports any model,
any provider, any combination — local or cloud, subscription or API key.

Usage:
    from lib.hal import get_registry

    registry = get_registry()
    response = registry.chat("groq/llama-3.3-70b", "Explain SOLID principles")

    providers = registry.list_providers()
    models = registry.list_models("groq")
"""
from lib.hal.code_gen import CodeGenerator, CodeSpec, GeneratedCode
from lib.hal.registry import ProviderRegistry, get_registry
from lib.hal.self_build import BuildConfig, BuildResult
from lib.hal.types import (
    ChatMessage,
    ChatResponse,
    CostSnapshot,
    ModelInfo,
    ProviderConfig,
    ProviderInfo,
    ProviderType,
    RoutingStrategy,
)

__all__ = [
    "BuildConfig",
    "BuildResult",
    "ChatMessage",
    "ChatResponse",
    "CodeGenerator",
    "CodeSpec",
    "CostSnapshot",
    "GeneratedCode",
    "ModelInfo",
    "ProviderConfig",
    "ProviderInfo",
    "ProviderRegistry",
    "ProviderType",
    "RoutingStrategy",
    "get_registry",
]
