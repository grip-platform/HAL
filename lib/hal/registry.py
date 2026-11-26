"""HAL Provider Registry — config-driven provider management.

Loads providers from config/providers.yaml, manages health state,
and provides model addressing via "provider/model-name" convention.
Model listings are enriched from lib/hal/model_catalog on id match
(context window, output cap, per-MTok costs) — see list_models().

Usage:
    registry = get_registry()
    registry.load_config()         # Load from providers.yaml
    registry.discover_local()      # Probe local servers

    provider = registry.get("groq")
    response = provider.chat("Hello", model="llama-3.3-70b-versatile")

    # Model addressing
    response = registry.chat("groq/llama-3.3-70b", "Hello")
"""
import dataclasses
import json
import os
import threading
import time
from typing import Optional

from lib.hal.providers.openai_compat import OpenAICompatProvider
from lib.hal.types import (
    AuthMethod,
    ModelInfo,
    ProviderConfig,
    ProviderInfo,
    ProviderType,
    RoutingConfig,
    RoutingStrategy,
)

from lib.hal.home import HAL_HOME as _HAL_HOME
_CONFIG_PATH = os.path.join(_HAL_HOME, "config", "providers.yaml")

# Singleton
_registry: Optional["ProviderRegistry"] = None
_registry_lock = threading.Lock()


def _enrich_from_catalog(model: ModelInfo) -> ModelInfo:
    """Fill zero-valued listing metadata from the static model catalogue.

    Lazy import: model_catalog is a standalone table, but importing at module
    scope would couple registry import-time to it for every consumer.
    """
    from lib.hal.model_catalog import ModelCatalog
    entry = ModelCatalog().get(model.id)
    if entry is None:
        return model
    updates: dict = {}
    if not model.context_window:
        updates["context_window"] = entry.context_window
    if not model.max_output_tokens:
        updates["max_output_tokens"] = entry.max_output_tokens
    if not model.cost_per_mtok_input:
        updates["cost_per_mtok_input"] = entry.input_cost_per_mtok
    if not model.cost_per_mtok_output:
        updates["cost_per_mtok_output"] = entry.output_cost_per_mtok
    # types.ModelInfo is a frozen dataclass — build the enriched copy.
    return dataclasses.replace(model, **updates) if updates else model


def get_registry() -> "ProviderRegistry":
    """Get or create the singleton provider registry."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ProviderRegistry()
    return _registry


def parse_model_address(address: str) -> tuple[str, str]:
    """Parse 'provider/model-name' into (provider, model).

    Examples:
        'groq/llama-3.3-70b' -> ('groq', 'llama-3.3-70b')
        'ollama/llama3:70b'  -> ('ollama', 'llama3:70b')
        'openrouter/deepseek/deepseek-r1' -> ('openrouter', 'deepseek/deepseek-r1')
        'llama-3.3-70b'      -> ('', 'llama-3.3-70b')  # No provider prefix
    """
    if "/" not in address:
        return "", address

    first_slash = address.index("/")
    provider = address[:first_slash]

    # Known provider prefixes — model name is everything after first slash.
    # Sync invariant: this set MUST cover every provider name in
    # config/providers.yaml. The test test_router_yaml_providers_all_routable
    # in tests/test_router.py fails if a yaml provider is missing here.
    # Issue #305: extending the dispatch surface without editing this set is
    # silent drift — every yaml addition must come with a known_providers entry.
    known_providers = {
        # Cloud providers (in providers.yaml)
        "anthropic", "openai", "groq", "gemini", "deepseek", "mistral",
        "openrouter", "qwen", "moonshot", "zhipu", "soruxgpt", "vulavula",
        "xai",
        # Alibaba's Token Plan, a flat-fee subscription with its own endpoint
        # (HAL#637): qwen-token-plan/qwen4-max reaches that plan, while
        # qwen/qwen4-max is DashScope pay-per-token (HAL#813).
        "qwen-token-plan",
        # Subscription-bridge provider (NOT in providers.yaml -- it enters the
        # registry through raillm_subscription_bridge.register_subscriptions,
        # exactly like anthropic/CCH). kimi-coding is the Kimi Code membership
        # (GRIP Pool D; lib/hal/subscription.py KimiCodingSubscription). This
        # entry is what lets parse_model_address("kimi-coding/k3-256k") hand
        # the bridged provider the bare vendor id instead of sending the whole
        # address to the default provider. Its providers.yaml block lands with
        # the HAPPI PROVIDER_CONFIGS row -- the sync tests bind the two.
        "kimi-coding",
        # Local providers (in providers.yaml)
        "lemonade", "ollama", "vllm", "lmstudio", "mlx_vlm", "bitnet",
        # Reserved future providers (not currently in providers.yaml)
        "together", "fireworks", "llamacpp", "localai", "nim",
        "bedrock", "vertex", "azure",
    }
    if provider in known_providers:
        return provider, address[first_slash + 1:]

    # Unknown prefix — treat entire string as model (could be openrouter path)
    return "", address


class ProviderRegistry:
    """Manages provider instances, health, and model resolution.

    7 public methods (ISP-compliant):
    register, get, chat, list_providers, list_models, check_health, load_config
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self._providers: dict[str, OpenAICompatProvider] = {}
        self._configs: dict[str, ProviderConfig] = {}
        self._health: dict[str, bool] = {}
        self._last_health_check: dict[str, float] = {}
        self._routing = RoutingConfig()
        self._health_ttl_s = 300.0
        self._lock = threading.Lock()
        if config_path is not None:
            self._load_config(config_path)
        elif os.path.exists(_CONFIG_PATH):
            self._load_config(_CONFIG_PATH)

    def register(self, source) -> None:
        """Register a provider from ProviderConfig or OpenAICompatProvider.

        Thread-safe: uses internal lock to prevent concurrent mutation.
        """
        if isinstance(source, ProviderConfig):
            if not source.enabled:
                return
            provider = OpenAICompatProvider(
                name=source.name,
                base_url=source.base_url,
                api_key_source=source.api_key_source,
                timeout_s=source.timeout_s,
                max_retries=source.max_retries,
                extra_headers=source.extra_headers,
            )
            with self._lock:
                self._providers[source.name] = provider
                self._configs[source.name] = source
                self._health[source.name] = True
        else:
            with self._lock:
                self._providers[source.name] = source
                self._health[source.name] = True

    def get(self, name: str) -> Optional[OpenAICompatProvider]:
        """Get a registered provider by name."""
        return self._providers.get(name)

    def chat(self, model_address: str, prompt: str, **kwargs) -> "ChatResponse":
        """Route a chat request using 'provider/model-name' addressing."""
        provider_name, model_name = parse_model_address(model_address)

        if provider_name:
            provider = self._providers.get(provider_name)
            if not provider:
                raise KeyError(f"Provider '{provider_name}' not registered")
            return provider.chat(prompt, model=model_name, **kwargs)

        default = self._find_default_provider()
        if not default:
            raise RuntimeError("No providers registered")
        return default.chat(prompt, model=model_name, **kwargs)

    def list_providers(self) -> list[ProviderInfo]:
        """List all registered providers with health status."""
        return [
            ProviderInfo(
                name=name,
                type=self._configs[name].type if name in self._configs else ProviderType.CLOUD,
                base_url=provider.base_url,
                healthy=self._health.get(name, True),
                models=tuple(self._configs[name].models) if name in self._configs else (),
            )
            for name, provider in self._providers.items()
        ]

    def list_models(self, provider_name: Optional[str] = None) -> list[ModelInfo]:
        """List models from a specific provider or all providers.

        Provider /models endpoints return only ids, so raw listings carry
        context_window=0 and costs 0.0 even for fully-catalogued models —
        three surfaces, one model, no cross-check (issue #509 item 5).
        Entries are enriched from lib/hal/model_catalog on id match;
        provider-supplied non-zero values are never overwritten.
        """
        if provider_name:
            provider = self._providers.get(provider_name)
            models = provider.list_models() if provider else []
        else:
            models = [
                model
                for provider in self._providers.values()
                for model in provider.list_models()
            ]
        return [_enrich_from_catalog(model) for model in models]

    def check_health(self, name: Optional[str] = None) -> dict[str, bool]:
        """Check health for one provider (by name) or all providers.

        Returns dict of {provider_name: healthy}. Uses 5-minute TTL cache.
        """
        targets = [name] if name else list(self._providers)
        result = {}
        now = time.monotonic()

        for pname in targets:
            provider = self._providers.get(pname)
            if not provider:
                result[pname] = False
                continue

            last = self._last_health_check.get(pname, 0)
            if now - last < self._health_ttl_s:
                result[pname] = self._health.get(pname, False)
                continue

            healthy = provider.health_check()
            self._health[pname] = healthy
            self._last_health_check[pname] = now
            result[pname] = healthy

        return result

    def _load_config(self, path: str) -> None:
        """Load provider configuration from YAML or JSON file."""
        if not os.path.exists(path):
            return
        configs, routing = _load_yaml_config(path)
        for config in configs:
            self.register(config)
        if routing:
            self._routing = routing
        # _backfill_auto_models() removed from init path — it queries live
        # provider endpoints synchronously which blocks startup for up to
        # timeout_s (120-180s) per provider. discover_local_providers() in
        # cli.py already populates model lists for running local servers.

    def _backfill_auto_models(self) -> None:
        """Fetch live model lists for providers with auto_discover_models=True.

        Runs once after config load. Providers configured with `models: auto`
        start with an empty model list — this queries each provider's /v1/models
        endpoint and updates the stored config so VLM routing can match on them.
        """
        for name, config in list(self._configs.items()):
            if not config.auto_discover_models or config.models:
                continue
            provider = self._providers.get(name)
            if not provider:
                continue
            try:
                live_models = provider.list_models()
                if live_models:
                    model_ids = [m.id for m in live_models]
                    with self._lock:
                        config.models = model_ids
            except Exception:
                pass

    @property
    def routing(self) -> RoutingConfig:
        """Current routing configuration."""
        return self._routing

    def _find_default_provider(self) -> Optional[OpenAICompatProvider]:
        """Find the default provider.

        Resolution order:
        1. First entry in fallback_chain (declared primary model).
        2. Provider flagged default: true in config.
        3. First registered provider.
        """
        chain = self._routing.fallback_chain
        if chain:
            provider_name, _ = parse_model_address(chain[0])
            if provider_name and provider_name in self._providers:
                return self._providers[provider_name]
        for name, config in self._configs.items():
            if config.default and name in self._providers:
                return self._providers[name]
        if self._providers:
            return next(iter(self._providers.values()))
        return None


def _load_yaml_config(path: str) -> tuple[list[ProviderConfig], Optional[RoutingConfig]]:
    """Parse providers.yaml into ProviderConfig list + RoutingConfig."""
    if path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = _parse_simple_yaml(path)

    configs = [_parse_provider_entry(p) for p in data.get("providers", [])]
    routing = _parse_routing_entry(data.get("routing", {}))
    return configs, routing


def _parse_provider_entry(p: dict) -> ProviderConfig:
    """Parse a single provider dict into ProviderConfig."""
    models = p.get("models", [])
    auto_discover = models == "auto"
    if auto_discover:
        models = []

    return ProviderConfig(
        name=p["name"],
        type=ProviderType(p.get("type", "cloud")),
        base_url=p.get("base_url", ""),
        auth_method=AuthMethod(p.get("auth", p.get("auth_method", "api_key"))),
        api_key_source=p.get("api_key_source", p.get("api_key_env", "")),
        models=models if isinstance(models, list) else [],
        auto_discover_models=auto_discover,
        auto_detect=p.get("detect", False),
        default=p.get("default", False),
        enabled=p.get("enabled", True),
        extra_headers=p.get("extra_headers", {}),
        timeout_s=float(p.get("timeout_s", 120)),
        max_retries=int(p.get("max_retries", 3)),
    )


_TIER_NAMES = ("free", "cheap", "mid", "premium")


def _parse_routing_entry(r: dict) -> Optional[RoutingConfig]:
    """Parse routing config dict into RoutingConfig.

    The custom YAML parser flattens nested dicts, so tier_chains sub-keys
    (free/cheap/mid/premium) land directly in r alongside quality_cost sub-keys.
    """
    if not r:
        return None
    tier_chains = {t: r[t] for t in _TIER_NAMES if isinstance(r.get(t), list)}
    return RoutingConfig(
        default_strategy=RoutingStrategy(r.get("default", "fixed")),
        default_model=r.get("default_model", ""),
        fallback_chain=r.get("fallback_chain", []),
        quality_cost_tiers=r.get("quality_cost", {}),
        tier_chains=tier_chains,
    )


def _parse_simple_yaml(path: str) -> dict:
    """Minimal YAML subset parser for providers.yaml format.

    Supports: top-level keys, list items with key: value pairs.
    For complex configs, use JSON format instead.
    """
    with open(path, "r") as f:
        lines = f.readlines()

    state = _YamlParseState()
    for line in lines:
        stripped = line.rstrip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        _process_yaml_line(state, stripped, indent)

    state.flush()
    return state.result


class _YamlParseState:
    """Mutable state for the YAML line processor."""

    def __init__(self) -> None:
        self.result: dict = {}
        self.current_key = ""
        self.current_list: list = []
        self.current_item: dict = {}
        self.current_dict: dict = {}
        self.in_list = False
        self.in_dict = False

    def flush(self) -> None:
        if not self.current_key:
            return
        if self.in_list:
            if self.current_item:
                self.current_list.append(self.current_item)
            self.result[self.current_key] = self.current_list
        elif self.in_dict:
            self.result[self.current_key] = self.current_dict


def _process_yaml_line(state: _YamlParseState, stripped: str, indent: int) -> None:
    """Process a single YAML line, updating parse state."""
    if indent == 0 and ":" in stripped:
        state.flush()
        key, _, val = stripped.partition(":")
        state.current_key = key.strip()
        val = val.strip()
        if val:
            state.result[state.current_key] = _parse_yaml_value(val)
            state.current_key = ""
        else:
            state.current_list = []
            state.current_item = {}
            state.current_dict = {}
            state.in_list = False
            state.in_dict = False
        return

    if stripped.lstrip().startswith("- "):
        state.in_list = True
        state.in_dict = False
        if state.current_item:
            state.current_list.append(state.current_item)
            state.current_item = {}
        rest = stripped.lstrip()[2:].strip()
        if ":" in rest:
            k, _, v = rest.partition(":")
            state.current_item[k.strip()] = _parse_yaml_value(v.strip())
        elif rest:
            state.current_list.append(_parse_yaml_value(rest))
        return

    if ":" in stripped:
        k, _, v = stripped.strip().partition(":")
        v = v.strip()
        if state.in_list:
            state.current_item[k.strip()] = _parse_yaml_value(v)
        else:
            state.in_dict = True
            state.current_dict[k.strip()] = _parse_yaml_value(v)


def _parse_yaml_value(val: str):
    """Parse a YAML scalar value."""
    if not val:
        return ""
    if val in ("true", "True", "yes"):
        return True
    if val in ("false", "False", "no"):
        return False
    if val == "auto":
        return "auto"
    if val.startswith("[") and val.endswith("]"):
        items = val[1:-1].split(",")
        return [item.strip().strip("'\"") for item in items if item.strip()]
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val.strip("'\"")
