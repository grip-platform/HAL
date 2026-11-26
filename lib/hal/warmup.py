"""HAL Model Warmup — pre-load local models at startup.

Ollama models take 20-30s on cold start but respond in <100ms when warm.
Pings local models in a background thread so the first real query is fast.

Usage:
    from lib.hal.warmup import warmup_background

    thread = warmup_background(registry)  # Non-blocking, returns immediately
"""
import json
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

DIM = "\033[2m"
GREEN = "\033[32m"
RESET = "\033[0m"


def warmup_background(
    registry: "ProviderRegistry", verbose: bool = True,
) -> threading.Thread:
    """Warm up all local models in a background thread.

    Returns the thread handle (daemon — dies with the main process).
    """
    t = threading.Thread(
        target=_warmup_all, args=(registry, verbose),
        daemon=True, name="hal-warmup",
    )
    t.start()
    return t


def warmup_sync(
    registry: "ProviderRegistry", verbose: bool = True,
) -> dict[str, float]:
    """Warm up all local models synchronously. Returns {address: latency_ms}."""
    return _warmup_all(registry, verbose)


def _warmup_all(registry, verbose: bool) -> dict[str, float]:
    """Ping each local model with a 1-token query. Returns {address: latency_ms}."""
    models = _discover_local_models()
    if not models:
        return {}

    if verbose:
        print(
            f"{DIM}  Warming up {len(models)} local model(s)...{RESET}",
            flush=True,
        )

    results: dict[str, float] = {}
    for provider, model in models:
        address = f"{provider}/{model}"
        latency = _ping_model(registry, address)
        if latency is not None:
            results[address] = latency
            if verbose:
                print(
                    f"{DIM}  {GREEN}✓{RESET}{DIM} {address} warm "
                    f"({latency:.0f}ms){RESET}",
                    flush=True,
                )

    return results


def _discover_local_models() -> list[tuple[str, str]]:
    """Discover local models from known servers. Returns [(provider, model)]."""
    models: list[tuple[str, str]] = []
    for m in _get_ollama_models():
        models.append(("ollama", m))
    return models


def _get_ollama_models() -> list[str]:
    """Get model names from Ollama's /api/tags endpoint."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
        return [m["name"] for m in data.get("models", [])]
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return []


def _ping_model(registry, model_address: str) -> Optional[float]:
    """Send a 1-token query to load the model into RAM. Returns latency_ms."""
    try:
        start = time.monotonic()
        registry.chat(model_address, "hi", max_tokens=1)
        return (time.monotonic() - start) * 1000
    except Exception:
        return None
