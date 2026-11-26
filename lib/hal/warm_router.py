"""Warm-model-first routing: probe Ollama /api/ps before firing the API chain."""
import json
import logging
import urllib.request
from dataclasses import dataclass

_log = logging.getLogger(__name__)
_PROBE_TIMEOUT_S = 0.2          # 200ms — fast-fail if Ollama is down
_PS_ENDPOINT = "http://localhost:11434/api/ps"


@dataclass
class WarmModel:
    name: str
    size_mb: float = 0.0


class OllamaLivenessProbe:
    """Query Ollama's /api/ps to discover models currently loaded in VRAM."""

    def warm_models(self) -> list[WarmModel]:
        """Return list of models currently in VRAM. Empty list if Ollama is down."""
        try:
            req = urllib.request.Request(_PS_ENDPOINT, method="GET")
            with urllib.request.urlopen(req, timeout=_PROBE_TIMEOUT_S) as resp:
                data = json.loads(resp.read())
            return [
                WarmModel(
                    name=m["name"],
                    size_mb=m.get("size", 0) / (1024 * 1024),
                )
                for m in data.get("models", [])
            ]
        except Exception as exc:
            _log.debug("Ollama liveness probe failed: %s", exc)
            return []

    def is_warm(self, model_name: str) -> bool:
        """Check if a specific model is currently in VRAM."""
        warm = {m.name for m in self.warm_models()}
        base = model_name.split(":")[0]
        return model_name in warm or any(w.split(":")[0] == base for w in warm)


class WarmModelRouter:
    """Pre-flight check: return a warm local model route if one matches the tier.

    Called before the API fallback chain. Returns None if no warm local match,
    allowing the caller to fall through to the normal chain.
    """

    _LOCAL_PREFERRED_TIERS = frozenset({"free", "cheap"})

    def __init__(
        self,
        probe: OllamaLivenessProbe,
        catalog_models: dict,
        tier_chains: dict[str, list[str]],
    ):
        self._probe = probe
        self._catalog = catalog_models
        self._tier_chains = tier_chains

    def route(self, tier: str) -> str | None:
        """Return 'ollama/<model>' if a warm local model matches the tier, else None."""
        if tier not in self._LOCAL_PREFERRED_TIERS:
            return None

        warm_names = {m.name for m in self._probe.warm_models()}
        if not warm_names:
            return None

        chain = self._tier_chains.get(tier, [])
        for entry in chain:
            if not entry.startswith("ollama/"):
                continue
            model_name = entry[len("ollama/"):]
            base = model_name.split(":")[0]
            if model_name in warm_names or any(w.split(":")[0] == base for w in warm_names):
                _log.debug("WarmModelRouter: routing to warm local model %s", entry)
                return entry

        return None
