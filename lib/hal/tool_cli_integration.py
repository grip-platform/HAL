"""CLI integration helper for the HAL tool-use loop (W9 thin wrapper).

Packages the last-mile wiring the ``hal`` REPL needs to route messages
through :func:`~lib.hal.tool_render.send_with_tools_to_cli`:

1. :func:`detect_tool_chat_fn` probes for the right provider based on the
   ``provider/model`` prefix. Bare ``claude-*`` / no prefix keeps the
   existing CCH-first Anthropic path unchanged. ``ollama/*``, ``groq/*``,
   ``gemini/*``, ``openai/*`` dispatch to the OpenAI-compat path via
   :func:`~lib.hal.tool_runtime.make_openai_compat_chat_fn`.

2. :func:`build_builtin_tool_stack` returns the canonical
   ``(tools, dispatcher)`` pair (all 8 built-ins, a default
   :class:`ToolDispatcher`). Kept in one place so the REPL, tests,
   and future web/agent entry points all agree on the default stack.

Provider-prefix routing (W3 of sprint #1231):

    ============== ============================================ =======================
    Prefix         Base URL                                     API key source
    ============== ============================================ =======================
    ``claude-*``   (CCH, or HAL's API-key route when open)      CCH sub; hal-anthropic
                                                                only via the route
    ``ollama/*``   ``http://localhost:11434/v1``                literal ``"ollama"``
    ``groq/*``     ``https://api.groq.com/openai/v1``           ``grip-groq`` keychain
    ``gemini/*``   ``https://generativelanguage.googleapis.com/v1beta/openai``
                                                                ``grip-gemini`` keychain
    ``openai/*``   ``https://api.openai.com/v1``                ``hal-openai`` keychain
    ============== ============================================ =======================

Returned labels (``"cch"``, ``"api_key"``, ``"ollama"``, ``"groq"``,
``"gemini"``, ``"openai"``) tell the caller which path was taken so the
REPL can print ``tool mode enabled (ollama)`` etc.

Fall-through: if no credentials are available for the dispatched path,
:func:`detect_tool_chat_fn` returns ``(None, None)`` and the caller
surfaces a helpful message rather than crashing on the first request.
"""
import socket
from typing import Callable, Optional
from urllib.parse import urlparse

from lib.hal.spend_guard import require_openai_destination
from lib.hal.tool_adapter import to_anthropic_tools
from lib.hal.tool_dispatch import ToolDispatcher

# Provider-prefix routing table. Ordered as a tuple of tuples so dispatch
# is O(n) over a handful of prefixes — readable + trivially extensible.
# Each entry: (prefix, base_url, keychain_service_or_literal).
# A literal string that doesn't look like a keychain service (e.g.
# "ollama" for local Ollama) is returned as the api_key verbatim.
_OPENAI_COMPAT_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("ollama/",      "http://localhost:11434/v1",                                          "literal:ollama"),
    ("groq/",        "https://api.groq.com/openai/v1",                                    "keychain:grip-groq"),
    ("gemini/",      "https://generativelanguage.googleapis.com/v1beta/openai",            "keychain:grip-gemini"),
    ("openai/",      "https://api.openai.com/v1",                                          "keychain:hal-openai"),
    ("mlx_vlm/",     "http://localhost:8080/v1",                                           "literal:mlx_vlm"),
    ("vllm/",        "http://localhost:8000/v1",                                           "literal:vllm"),
    ("lmstudio/",    "http://localhost:1234/v1",                                           "literal:lmstudio"),
    ("lemonade/",    "http://localhost:7860/v1",                                           "literal:lemonade"),
    ("openrouter/",  "https://openrouter.ai/api/v1",                                      "keychain:grip-openrouter"),
    ("qwen/",        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",            "keychain:hal-dashscope"),
    ("qwen-token-plan/", "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1", "keychain:grip-qwen-token-plan"),
    ("deepseek/",    "https://api.deepseek.com/v1",                                       "keychain:hal-deepseek"),
    ("moonshot/",    "https://api.moonshot.ai/v1",                                        "keychain:grip-moonshot"),
    ("kimi-coding/", "https://api.kimi.com/coding/v1",                                   "keychain:grip-kimi-coding"),
    ("zhipu/",       "https://api.z.ai/api/coding/paas/v4",                               "keychain:grip-zhipu"),
    ("mistral/",     "https://api.mistral.ai/v1",                                         "keychain:grip-mistral"),
    ("xai/",         "https://api.x.ai/v1",                                               "keychain:hal-xai"),
)


def _wrap_cch_subscription(
    subscription, *, model: str, max_tokens: int,
) -> Callable[..., dict]:
    """Return a chat_fn backed by ``CCHSubscription.chat`` (or a transport with
    the same ``chat`` signature, such as HAL's ``ApiKeyTransport``).

    The subscription's ``chat`` method returns a :class:`ChatResponse`
    whose ``raw`` attribute holds the full Anthropic dict (including
    ``tool_use`` content blocks). The returned closure extracts
    ``raw`` so it matches :func:`run_tool_loop`'s chat_fn contract.

    Messages are ``list[dict]`` in Anthropic shape on the wire — the
    subscription path wants ``ChatMessage`` objects internally. We
    build the list at call time so the REPL can keep using plain dicts.
    """
    from lib.hal.types import ChatMessage

    def chat_fn(*, messages: list[dict], tools: list[dict]) -> dict:
        chat_messages = [
            ChatMessage(role=m["role"], content=m["content"])
            for m in messages
        ]
        response = subscription.chat(
            model=model,
            messages=chat_messages,
            max_tokens=max_tokens,
            tools=tools or None,
        )
        return response.raw or {}

    return chat_fn


def _resolve_bearer(source_spec: str) -> Optional[str]:
    """Resolve a source spec to a concrete bearer string.

    ``source_spec`` shape (see ``_OPENAI_COMPAT_ROUTES``):

    * ``"literal:<value>"`` — return ``<value>`` verbatim (e.g. local
      Ollama endpoints ignore auth but still expect a bearer).
    * ``"keychain:<service>"`` — read ``<service>`` from the macOS
      Keychain via the same ``_fetch_keychain_key`` helper used for
      Anthropic. Returns ``None`` on miss so the caller short-circuits.
    """
    if source_spec.startswith("literal:"):
        return source_spec[len("literal:"):]
    if source_spec.startswith("keychain:"):
        from lib.hal.providers.anthropic_native import _fetch_keychain_key
        service = source_spec[len("keychain:"):]
        value = _fetch_keychain_key(service)
        return (value or "").strip() or None
    return None


def _match_openai_compat_route(
    model: str,
) -> Optional[tuple[str, str, str]]:
    """Return ``(stripped_model, base_url, bearer_source)`` or ``None``.

    The first element is the bare model name the API expects — e.g.
    ``ollama/gemma4:e4b`` becomes ``gemma4:e4b``.  Bare provider names
    (``mlx_vlm``, ``vllm``, etc. — auto-discover with no ``/model``
    suffix) also match; ``stripped_model`` is ``""`` in that case and
    the local server uses whichever model it currently has loaded.
    """
    for prefix, base_url, bearer_source in _OPENAI_COMPAT_ROUTES:
        if model.startswith(prefix):
            return model[len(prefix):], base_url, bearer_source
        if model == prefix.rstrip("/"):
            return "", base_url, bearer_source
    return None


def _dispatch_openai_compat(
    model: str, *, max_tokens: int,
) -> tuple[Optional[Callable[..., dict]], Optional[str]]:
    """Build a chat_fn for a prefixed provider, returning (fn, label).

    Ollama models route through :class:`OllamaToolAdapter` (#1228 W4) so
    each request first consults the dynamic num_ctx policy and, if the
    incoming prompt would exceed the loaded context window's safety
    band, forces a pre-call reload at a larger num_ctx with keep_alive
    preserved. All other OpenAI-compat providers use the bare factory
    — they do not pin num_ctx the way Ollama does.
    """
    match = _match_openai_compat_route(model)
    if match is None:
        return None, None
    stripped_model, base_url, bearer_source = match
    require_openai_destination(base_url)
    bearer = _resolve_bearer(bearer_source)
    if not bearer:
        return None, None
    label = next(
        prefix.rstrip("/") for prefix, _, _ in _OPENAI_COMPAT_ROUTES
        if model.startswith(prefix) or model == prefix.rstrip("/")
    )
    if label == "ollama":
        from lib.hal.providers.ollama_tool_adapter import make_ollama_tool_chat_fn
        chat_fn = make_ollama_tool_chat_fn(
            base_url=base_url,
            api_key=bearer,
            model=stripped_model,
            max_tokens=max_tokens,
        )
    else:
        from lib.hal.tool_runtime import make_openai_compat_chat_fn
        chat_fn = make_openai_compat_chat_fn(
            base_url=base_url,
            api_key=bearer,
            model=stripped_model,
            max_tokens=max_tokens,
        )
    return chat_fn, label


def _try_cch_subscription(
    model: str, *, max_tokens: int,
) -> tuple[Optional[Callable[..., dict]], Optional[str]]:
    """CCH-first probe; returns (None, None) on any failure.

    ``CCHSubscription.chat`` asks HAL's API-key route on every call, so the label
    names the transport that route picks now: ``"api_key"`` when it is open.
    """
    try:
        from lib.hal import anthropic_key_route
        from lib.hal.subscription import CCHSubscription
        cch = CCHSubscription.from_env()
        if cch is not None and cch.health_check():
            label = "api_key" if anthropic_key_route.select(model).allowed else "cch"
            return _wrap_cch_subscription(
                cch, model=model, max_tokens=max_tokens,
            ), label
    except Exception:
        pass
    return None, None


def _try_anthropic_keychain(
    model: str, *, max_tokens: int,
) -> tuple[Optional[Callable[..., dict]], Optional[str]]:
    """HAL's API key with no CCH token, only when HAL's API-key route allows it.

    The route (``lib/hal/anthropic_key_route.py``) needs the opt-in, a daily cap
    with room left, a priced model and the keychain item, so this key is guarded,
    cached, priced and capped like every other API-key call HAL makes for itself.
    It is read only when a request is sent; when the route is closed it is not
    read at all.
    """
    try:
        from lib.hal import anthropic_key_route
        if anthropic_key_route.select(model).allowed:
            return _wrap_cch_subscription(
                anthropic_key_route.ApiKeyTransport(), model=model, max_tokens=max_tokens,
            ), "api_key"
    except Exception:
        pass
    return None, None


def _dispatch_anthropic(
    model: str, *, max_tokens: int,
) -> tuple[Optional[Callable[..., dict]], Optional[str]]:
    """Default Anthropic dispatch: CCH first (which asks HAL's API-key route on
    every call), then the API key alone when that route allows it."""
    fn, label = _try_cch_subscription(model, max_tokens=max_tokens)
    if fn is not None:
        return fn, label
    return _try_anthropic_keychain(model, max_tokens=max_tokens)


_LOCAL_START_HINTS: dict[str, str] = {
    "ollama":   "ollama serve",
    "mlx_vlm":  "python -m mlx_vlm.server --port 8080",
    "vllm":     "vllm serve <model_name>",
    "lmstudio": "open /Applications/LM\\ Studio.app",
    "lemonade": "lemonade serve",
}

# Providers where we can safely auto-start with no additional arguments.
# vllm requires a model name; lmstudio is a GUI app — both excluded.
_LOCAL_AUTO_START: dict[str, list[str]] = {
    "ollama":   ["ollama", "serve"],
    "mlx_vlm":  ["python3", "-m", "mlx_vlm.server", "--port", "8080"],
    "lemonade": ["lemonade", "serve"],
}


def probe_local_server(base_url: str, timeout: float = 0.5) -> bool:
    """Socket-level probe — True if the local server port is accepting connections."""
    parsed = urlparse(base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 80
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def auto_start_local_server(
    provider_name: str,
    base_url: str,
    *,
    timeout: float = 15.0,
) -> bool:
    """Launch the local server in the background and poll until ready.

    Returns True if the server accepts connections within *timeout* seconds,
    False if the provider has no auto-start command or the process fails to
    come up in time.
    """
    import subprocess
    import time

    cmd = _LOCAL_AUTO_START.get(provider_name)
    if cmd is None:
        return False
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, OSError):
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if probe_local_server(base_url, timeout=0.5):
            return True
        time.sleep(0.5)
    return False


def get_local_provider_info(model: str) -> Optional[tuple[str, str, str]]:
    """If *model* routes to a local provider, return (name, base_url, start_hint).

    Returns None for cloud providers and unknown models.  Used by the REPL to
    pre-check server availability before enabling tool mode.
    """
    match = _match_openai_compat_route(model)
    if match is None:
        return None
    _, base_url, bearer_source = match
    if not bearer_source.startswith("literal:"):
        return None
    provider_name = bearer_source[len("literal:"):]
    hint = _LOCAL_START_HINTS.get(provider_name, f"start the {provider_name} server")
    return provider_name, base_url, hint


def detect_tool_chat_fn(
    model: str,
    *,
    max_tokens: int = 4096,
) -> tuple[Optional[Callable[..., dict]], Optional[str]]:
    """Probe stored bearers and return (chat_fn, auth_label) for ``model``.

    Dispatch is prefix-based (W3):

    * ``ollama/<m>`` -> local Ollama at ``localhost:11434/v1`` with a
      literal bearer value. Label ``"ollama"``.
    * ``groq/<m>`` -> Groq OpenAI-compat endpoint. Bearer from
      ``grip-groq`` Keychain. Label ``"groq"``.
    * ``gemini/<m>`` -> Gemini OpenAI-compat endpoint. Bearer from
      ``grip-gemini`` Keychain. Label ``"gemini"``.
    * ``openai/<m>`` -> OpenAI. Bearer from ``hal-openai`` Keychain.
      Label ``"openai"``.
    * Anything else (bare ``claude-*`` / unprefixed) -> existing
      CCH-first Anthropic path. Behaviour unchanged from pre-W3.

    Env-var lookup is deliberately out of scope — HAL's stored-bearer
    policy is Keychain-only for auto-detection. Operators can still
    wire their own chat_fn via the public factories in
    :mod:`lib.hal.tool_runtime`.
    """
    if _match_openai_compat_route(model) is not None:
        return _dispatch_openai_compat(model, max_tokens=max_tokens)
    return _dispatch_anthropic(model, max_tokens=max_tokens)


def build_builtin_tool_stack() -> tuple[list[dict], ToolDispatcher]:
    """Return ``(tools, dispatcher)`` with all 8 built-in HAL tools.

    Canonical stack for the REPL and any programmatic caller that
    wants the default tool surface without rebuilding it. Kept in
    one place so drift between entry points (CLI, web, agent) stays
    impossible — change the defaults here, every surface updates.
    """
    return to_anthropic_tools(), ToolDispatcher()


__all__ = [
    "auto_start_local_server",
    "build_builtin_tool_stack",
    "detect_tool_chat_fn",
    "get_local_provider_info",
    "probe_local_server",
]
