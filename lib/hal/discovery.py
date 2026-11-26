"""HAL Local Auto-Discovery — probe known ports for local LLM servers.

Checks Ollama, vLLM, llama.cpp, LM Studio, and mlx-vlm on their default ports.
Registers anything that responds with the provider registry.

Port 8080 is shared between llamacpp and mlx-vlm (both use it as their default).
Discovery disambiguates by inspecting the /v1/models response: if any model ID
matches _VLM_MODEL_RE the server is registered as mlx_vlm, otherwise as llamacpp.

Usage:
    from lib.hal.discovery import discover_local_providers

    discovered = discover_local_providers(registry)
"""
import concurrent.futures
import json
import re
import urllib.error
import urllib.request

from lib.hal.types import AuthMethod, ProviderConfig, ProviderType

# (name, port, health_path, model_list_path)
_LOCAL_SERVERS = [
    ("ollama",   11434, "/api/version", "/api/tags"),
    ("vllm",     8000,  "/v1/models",   "/v1/models"),
    ("lmstudio", 1234,  "/v1/models",   "/v1/models"),
    ("llamacpp", 8080,  "/v1/models",   "/v1/models"),  # discriminated below
]

# Model-name patterns that identify an mlx-vlm server on port 8080.
# llamacpp serves text-only models (e.g. "llama-3.1-8b-instruct") which never
# match these patterns.
_VLM_MODEL_RE = re.compile(
    r"qwen\d*[-.]vl|phi[-.]?\d*[-.]?vision|gemma[-.]?4|gemma.*vision|granite.*vision|"
    r"llama.*vision|falcon[-.]?(?:ocr|perception)|sam[-.]?\d|rf[-.]?detr|idefics|llava|minicpm[-.]v",
    re.IGNORECASE,
)

_PROBE_TIMEOUT_S = 0.5


def _probe_server(
    server_entry: tuple, timeout_s: float,
) -> tuple[str, str, str, list[str]] | None:
    """Probe one server entry. Returns (effective_name, base_url, model_path, models) or None."""
    name, port, health_path, model_path = server_entry
    base_url = f"http://localhost:{port}"
    if not _probe(base_url, health_path, timeout_s):
        return None
    if name == "llamacpp":
        # Port 8080 shared — inspect models once to identify server type.
        models = _fetch_models(base_url, model_path, name)
        effective_name = "mlx_vlm" if _is_vlm_server(models) else "llamacpp"
        return effective_name, base_url, model_path, models
    return name, base_url, model_path, []


def discover_local_providers(
    registry: "ProviderRegistry",
    timeout_s: float = _PROBE_TIMEOUT_S,
) -> list[str]:
    """Probe local ports in parallel and register discovered servers.

    Returns list of discovered provider names. Probes run concurrently so
    the worst-case wait is one timeout (0.5s) instead of N × timeout (2s).
    Port 8080: discriminates mlx_vlm from llamacpp by model name content.
    """
    discovered = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(_LOCAL_SERVERS), thread_name_prefix="hal-discover"
    ) as executor:
        futures = {
            executor.submit(_probe_server, entry, timeout_s): entry
            for entry in _LOCAL_SERVERS
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is None:
                continue
            effective_name, base_url, model_path, models = result
            if registry.get(effective_name) is not None:
                continue
            config = _build_local_config(
                effective_name, base_url, model_path,
                models=models if models else None,
            )
            registry.register(config)
            discovered.append(effective_name)

    return discovered


def _is_vlm_server(models: list[str]) -> bool:
    """Return True if any model ID matches known mlx-vlm model name patterns."""
    return any(_VLM_MODEL_RE.search(m) for m in models)


def _probe(base_url: str, path: str, timeout_s: float) -> bool:
    """Check if a local server is running at the given URL."""
    try:
        req = urllib.request.Request(f"{base_url}{path}", method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _build_local_config(
    name: str,
    base_url: str,
    model_path: str,
    models: list[str] | None = None,
) -> ProviderConfig:
    """Build a ProviderConfig for a discovered local server.

    Pass pre-fetched ``models`` to avoid a redundant network call when the
    caller has already fetched the model list (e.g. for port-8080 discrimination).
    """
    oai_base = f"{base_url}/v1" if name == "ollama" else base_url
    return ProviderConfig(
        name=name,
        type=ProviderType.LOCAL,
        base_url=oai_base,
        auth_method=AuthMethod.NONE,
        auto_discover_models=True,
        auto_detect=True,
        models=models if models is not None else _fetch_models(base_url, model_path, name),
    )


def _fetch_models(base_url: str, path: str, name: str) -> list[str]:
    """Fetch model list from a local server."""
    try:
        req = urllib.request.Request(f"{base_url}{path}", method="GET")
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return []

    if name == "ollama":
        return [m["name"] for m in data.get("models", []) if m.get("name")]

    return [m["id"] for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
