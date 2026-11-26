"""HAL Memory Manager — VRAM-resident model lifecycle manager.

Keeps local Ollama models warm in VRAM by pinging them before the
5-minute Ollama eviction timeout. Used by the daemon for continuous
keepalive, and as a one-shot CLI for boot scripts.

Usage:
    # As a module (keepalive loop):
    from lib.hal.memory_manager import MemoryManager
    mm = MemoryManager(registry=registry)
    mm.start_keepalive()

    # As a CLI (boot script):
    python3 -m lib.hal.memory_manager --warm-all
"""
import json
import logging
import threading
import urllib.error
import urllib.request
from typing import Optional

from lib.hal.warm_router import OllamaLivenessProbe
from lib.hal.warmup import _get_ollama_models, _ping_model

_log = logging.getLogger(__name__)
_OLLAMA_GENERATE = "http://localhost:11434/api/generate"
_DEFAULT_INTERVAL_S = 240  # just under Ollama's 5-min VRAM eviction default


class MemoryManager:
    """VRAM-resident model lifecycle manager.

    Combines a liveness probe with a keepalive loop. The keepalive thread
    pings cold models every interval_s seconds to prevent VRAM eviction.

    Two mechanisms run in the daemon (intentionally overlapping):
    - Scheduler fires warmup_models every 300s (operator-visible, configurable)
    - Keepalive thread fires every 240s (low-level safety net, always on)
    """

    def __init__(
        self,
        registry=None,
        models: Optional[list[str]] = None,
        interval_s: int = _DEFAULT_INTERVAL_S,
        probe: Optional[OllamaLivenessProbe] = None,
    ):
        """
        registry   — ProviderRegistry or None (boot-script mode: raw HTTP ping)
        models     — explicit list of ollama model names; None = auto-discover
        interval_s — keepalive ping interval in seconds (default 240)
        probe      — OllamaLivenessProbe (injectable for tests)
        """
        self._registry = registry
        self._explicit_models = models
        self._interval_s = interval_s
        self._probe = probe or OllamaLivenessProbe()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def warm_all(self) -> dict[str, bool]:
        """Synchronously warm every configured model. Returns {model_name: success}."""
        results: dict[str, bool] = {}
        for model in self._models():
            results[model] = self._ping(model)
        return results

    def start_keepalive(self) -> None:
        """Start the background keepalive thread. Idempotent — safe to call twice."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._keepalive_loop,
            daemon=True,
            name="hal-memory-keepalive",
        )
        self._thread.start()
        _log.info("MemoryManager keepalive started (interval=%ds)", self._interval_s)

    def stop(self) -> None:
        """Signal the keepalive thread to stop. Blocks until it exits (or times out)."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_s + 1)
        _log.info("MemoryManager keepalive stopped")

    def status(self) -> dict[str, bool]:
        """Return current VRAM state for all managed models."""
        warm_set = {m.name for m in self._probe.warm_models()}
        return {model: (model in warm_set) for model in self._models()}

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _models(self) -> list[str]:
        """Return explicit list or auto-discover via Ollama /api/tags."""
        if self._explicit_models is not None:
            return list(self._explicit_models)
        return _get_ollama_models()

    def _ping(self, model_name: str) -> bool:
        """Ping one model. Uses registry.chat if available, else raw HTTP."""
        if self._registry is not None:
            return _ping_model(self._registry, f"ollama/{model_name}") is not None
        return self._nudge(model_name)

    def _nudge(self, model_name: str) -> bool:
        """Raw HTTP POST to /api/generate (no registry). Returns True on success."""
        payload = json.dumps({
            "model": model_name,
            "prompt": "hi",
            "stream": False,
            "options": {"num_predict": 1},
        }).encode()
        try:
            req = urllib.request.Request(
                _OLLAMA_GENERATE,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status == 200
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def _ping_cold_models(self, warm_set: set[str]) -> None:
        """Ping any model not currently warm. O(n) — hash lookup per model."""
        for model in self._models():
            if model not in warm_set:
                _log.debug("Keepalive: pinging cold model %s", model)
                self._ping(model)

    def _keepalive_loop(self) -> None:
        """Wait interval_s, probe VRAM, ping cold models. Exits on stop()."""
        while not self._stop_event.wait(timeout=self._interval_s):
            warm_set = {m.name for m in self._probe.warm_models()}
            self._ping_cold_models(warm_set)


if __name__ == "__main__":
    import sys
    if "--warm-all" in sys.argv:
        mm = MemoryManager()
        results = mm.warm_all()
        if not results:
            print("  No Ollama models found (is ollama running?)")
            sys.exit(0)
        any_failed = False
        for model, ok in results.items():
            status = "warm" if ok else "FAILED"
            print(f"  {model}: {status}")
            if not ok:
                any_failed = True
        sys.exit(1 if any_failed else 0)
    else:
        print("Usage: python3 -m lib.hal.memory_manager --warm-all")
        sys.exit(1)
