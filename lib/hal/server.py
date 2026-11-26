"""HAL Unified Server — HTTP API + Web UI + Streaming.

Routes all queries through the registry and conversation engine.
Supports multiple concurrent sessions and JSONL streaming for GRIP-GUI.

Usage:
    hal --serve                     # Start on port 3850
    hal --serve --port 8080         # Custom port
    python3 -m lib.hal.server       # Direct invocation
"""
import json
import os
import re
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Optional

_hal_home = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
sys.path.insert(0, _hal_home)

# Best-effort access to GRIP's lib.active_llm so HAL can publish the active
# model to the GRIP status line. We load the file directly via importlib
# instead of `from lib import active_llm`, because both ~/.hal and ~/.claude
# ship a `lib/__init__.py` and the standard import system resolves to ~/.hal's
# package, shadowing GRIP's module entirely. Direct file load bypasses that.
# If GRIP isn't installed (HAL standalone), the loader returns None and the
# server keeps running; the status line just doesn't update.
def _grip_active_llm():
    """Cached lookup of GRIP's active_llm module. Returns None outside GRIP."""
    cache = getattr(_grip_active_llm, "_cache", "unset")
    if cache != "unset":
        return cache
    module = None
    try:
        import importlib.util
        grip_home = os.environ.get("CLAUDE_DIR", os.path.expanduser("~/.claude"))
        path = os.path.join(grip_home, "lib", "active_llm.py")
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("grip_active_llm", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
    except Exception:
        module = None
    _grip_active_llm._cache = module
    return module


def _register_anthropic_cch(registry) -> bool:
    """Register the native Anthropic provider via CCH subscription if the
    OAuth token is present in the environment. Best-effort: any failure is
    swallowed so HAL startup is never blocked by CCH wiring problems.

    Returns True when the provider was registered, False otherwise.
    """
    try:
        from lib.hal.subscription import CCHSubscription
        from lib.hal.providers.anthropic_native import AnthropicProvider
        sub = CCHSubscription.from_env()
        if sub is None:
            return False
        provider = AnthropicProvider(subscription=sub, prefer="auto")
        # Registry.list_providers() reads provider.base_url for every entry;
        # AnthropicProvider does not expose one (it routes via subscription),
        # so we attach a labelling value here to keep the registry happy.
        if not hasattr(provider, "base_url"):
            provider.base_url = "https://api.anthropic.com"
        registry.register(provider)
        print("[hal] anthropic provider registered via CCH", file=sys.stderr)
        return True
    except Exception as exc:
        print(f"[hal] CCH registration skipped: {exc}", file=sys.stderr)
        return False


def _mark_active(address: str, reason: Optional[str] = None) -> None:
    """Publish the active solo model to the GRIP status line. Best-effort.

    ``reason`` is a short human "why" (e.g. "query", "fallback: a→b") that the
    status line surfaces next to the model name.
    """
    mod = _grip_active_llm()
    if mod is None:
        return
    try:
        mod.set_active(address, reason=reason)
    except (KeyboardInterrupt, GeneratorExit):
        raise  # never eat a genuine interrupt / generator-close signal
    except BaseException as exc:
        # Cosmetic statusline write — must NEVER crash a server request thread.
        # active_llm's catalogue check raises SystemExit (a BaseException) on an
        # uncatalogued provider/model, so `except Exception` was insufficient; we
        # swallow that here. Real control-flow signals (KeyboardInterrupt,
        # GeneratorExit) are re-raised by the clause above, so this can't eat a
        # deliberate shutdown. Log loudly, never re-raise the rest.
        print(f"[hal] statusline mark skipped for {address!r}: {exc}", file=sys.stderr)


def _mark_council(addresses: list, reason: Optional[str] = None) -> None:
    """Publish the active council members to the GRIP status line. Best-effort."""
    mod = _grip_active_llm()
    if mod is None:
        return
    try:
        mod.set_council(addresses, mode="council", reason=reason)
    except Exception:
        pass


def _fallback_reason(response) -> str:
    """Short "why" for a served response — the routing strategy the Router
    stamped (``response.reason``: "fixed", "fallback", "quality-cost:<tier>")
    when present, else a fallback arrow naming the requested → served model
    from ``response.fallback_info`` (set by LimitSentinel on a mid-dispatch
    provider swap). Returns '' for a clean, directly-routed answer.
    """
    reason = getattr(response, "reason", None)
    if reason:
        return reason
    fi = getattr(response, "fallback_info", None)
    if fi is None:
        return ""
    req = f"{fi.requested_provider}/{fi.requested_model}"
    got = f"{fi.actual_provider}/{fi.actual_model}"
    return f"fallback: {req}→{got}"


def _active_provider_default() -> str:
    """Bidirectional read: if the GRIP marker is fresh AND in solo mode,
    return its provider so /api/query without an explicit provider routes
    where the operator's last `/cheap` / `/local` / `/frontier` / `/llm set`
    pointed. Returns empty string if no usable marker — caller falls back
    to the static default (typically "groq")."""
    mod = _grip_active_llm()
    if mod is None:
        return ""
    try:
        state = mod.show_active()
        if not state or not mod._is_fresh(state):
            return ""
        if state.get("mode") != "solo":
            return ""  # council/MoE/fallback — don't bias single-provider routing
        return state.get("provider", "")
    except Exception:
        return ""


from lib.hal.agent_registry import get_agent_registry
from lib.hal.api_throttle import client_ip_from_headers, get_chat_throttle
from lib.hal.background_executor import BackgroundExecutor
from lib.hal.conversation import Conversation
from lib.hal.cost_tracker import CostTracker
from lib.hal.discovery import discover_local_providers
from lib.hal.environment import EnvironmentRegistry
from lib.hal.fleet import FleetManager
from lib.hal.gateway import GatewayProxy, handle_proxy_request
from lib.hal.gateway_failover import FailoverProxy, build_failover_proxy
from lib.hal.gateway_guard import enforce_gateway_access, is_loopback_caller
from lib.hal.gateway_openai import handle_chat_completions
from lib.hal.gateway_responses import handle_responses
from lib.hal.http_refusal import _discard_unread_body
from lib.hal.moe import HALCouncil
from lib.hal.registry import get_registry
from lib.hal.response_streamer import ResponseStreamer
from lib.hal.session_registry import SessionRegistry
from lib.hal.warmup import _get_ollama_models

# Path patterns for parameter routes
_SESSION_ID_RE = re.compile(r"^/api/sessions/([a-f0-9-]+)$")
_AGENT_ID_RE = re.compile(r"^/api/agents/([a-f0-9-]+)$")
_TASK_ID_RE = re.compile(r"^/api/tasks/([a-f0-9-]+)$")
_TASK_EVENTS_RE = re.compile(r"^/api/tasks/([a-f0-9-]+)/events$")
_ENV_ID_RE = re.compile(r"^/api/environments/([a-f0-9-]+)$")

# --- Inference route classes (cost-aware model selection) -------------------
#
# A "route" maps a workload CLASS to a cheap-first model, so latency-tolerant
# bulk calls (e.g. live-meeting classification: ~300-token prompt -> small
# JSON verdict, every 10-60s) do NOT burn the operator's CCH-Anthropic
# subscription quota on opus-4-7. The chosen model becomes grip_infer's
# PRIMARY; the RAILLM cascade (CCH -> Kimi -> cheap -> local) remains the
# fallback BENEATH it, so a cheap-rung failure still degrades to a working
# provider rather than failing hard.
#
# Benchmarked 2026-06-10 (drafts/hal-classify-routing-council-2026-06-10.md in
# GRIP): groq/llama-3.3-70b-versatile scored 10/10 verdict-match vs the
# opus-4-7 baseline on 10 real meeting chunks, at ~2.8s avg (3.4x faster than
# opus's ~9.3s) and on the free tier.
#
# Operators add a route here with no other code change. An explicit ``model``
# in the request body ALWAYS wins over a route (caller override preserved).
_INFER_ROUTES: dict[str, str] = {
    "classify": "groq/llama-3.3-70b-versatile",
}


def resolve_infer_model(body: dict) -> str | None:
    """Resolve the model for an ``/api/infer`` body.

    Precedence: explicit ``model`` > ``route`` lookup > ``None``. ``None``
    means "let grip_infer pick its default primary" — byte-identical to the
    pre-route behaviour, so a body with neither field is unchanged. An
    unknown ``route`` string falls through to ``None`` (a typo never fails
    the call): fail-safe, which is the route mechanism's whole purpose.
    """
    explicit = body.get("model")
    if explicit:
        return str(explicit)
    route = body.get("route")
    if isinstance(route, str):
        return _INFER_ROUTES.get(route)
    return None


def _read_max_tokens(body: dict) -> int | None:
    """The optional ``max_tokens`` answer budget from an ``/api/infer`` body
    (HAL#648), by the one rule in ``lib.hal.happi.cmds_infer.answer_budget``.

    Imported on call, as the handler imports the runtime itself, so this
    module still loads when the happi stack cannot.
    """
    from lib.hal.happi.cmds_infer import answer_budget
    return answer_budget(body.get("max_tokens"))


# --- Remote access policy (default-deny-remote for the /api/* surface) ------
#
# The /v1/* proxy is guarded by enforce_gateway_access; the native /api/* routes
# were NOT — making tool-execution, fleet control, and metered inference
# reachable UNAUTHENTICATED through the public tunnel (hal.grip-web.com). This
# policy closes that: a REMOTE (tunnelled) caller may reach ONLY the public
# chat demo surface without a gateway token; everything else under /api/* and
# every DELETE is default-deny-remote. LOOPBACK callers are unaffected
# (enforce_gateway_access allows a loopback peer carrying no proxy header).
# See rules/untrusted-content-capability-lock.md + lib/hal/gateway_guard.py.
#
# Adding a route here WIDENS the unauthenticated surface — do so deliberately.
_PUBLIC_GET_PATHS = frozenset({"/", "/index.html", "/api/health"})
_PUBLIC_POST_PATHS = frozenset({"/api/chat"})


def _is_public_path(method: str, path: str) -> bool:
    """True iff (method, path) may be served to a REMOTE caller WITHOUT a
    gateway token. Narrows only the remote surface; loopback is always allowed
    by :func:`enforce_gateway_access`. Default-deny: anything not explicitly
    listed is gated.

    Policy:
      * GET  /, /index.html, /api/health, and any non-/api//non-/v1/ path
        (static UI dist assets) -> public
      * POST /api/chat (the public chat demo) -> public
      * every other /api/*, every DELETE, everything else -> gated
    """
    p = path.split("?", 1)[0]
    if method == "GET":
        if p in _PUBLIC_GET_PATHS:
            return True
        return not p.startswith("/api/") and not p.startswith("/v1/")
    if method == "POST":
        return p in _PUBLIC_POST_PATHS
    return False


# Gateway proxy (P1 verbatim Anthropic + P2 OpenAI-compat + P3 failover/audit):
# one per process, built lazily so importing the server never opens a socket or
# touches the network. P3's FailoverProxy duck-types GatewayProxy.forward and is
# backward-compatible — with a single configured upstream it behaves byte-
# identically to a bare GatewayProxy, so the P1/P2 handlers use it unchanged.
_GATEWAY_PROXY = None
_GATEWAY_LOCK = threading.Lock()


def _get_gateway_proxy() -> FailoverProxy:
    """Return the shared gateway proxy (failover-aware), built on first use.

    Thread-safe double-checked locking: the server is multi-threaded
    (ThreadingHTTPServer below), so two concurrent first requests could
    otherwise both observe ``None`` and build the proxy twice. The outer check
    keeps the hot path lock-free once built; the inner check (under the lock)
    closes the race.
    """
    global _GATEWAY_PROXY
    if _GATEWAY_PROXY is None:
        with _GATEWAY_LOCK:
            if _GATEWAY_PROXY is None:
                _GATEWAY_PROXY = build_failover_proxy()
    return _GATEWAY_PROXY


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTPServer that serves each request in its own thread.

    HAL request handlers are I/O-bound (they await upstream LLM/provider HTTP),
    so threads yield real concurrency under the GIL — a single slow upstream
    call no longer blocks every other client. ``daemon_threads = True`` lets the
    process exit cleanly on Ctrl+C without joining in-flight request threads.
    """

    daemon_threads = True
    #: Connections the kernel queues before ``accept()``. socketserver's
    #: default is 5, which a burst of concurrent clients overflows.
    request_queue_size = 128


BOLD = "\033[1m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"


class HALServer:
    """HAL HTTP server with REST API, streaming, and multi-session support."""

    def __init__(
        self,
        port: int = 3850,
        registry=None,
    ):
        self._port = port
        self._registry = registry or get_registry()
        discover_local_providers(self._registry)
        self._sessions = SessionRegistry(registry=self._registry)
        self._conversation = Conversation(registry=self._registry)
        self._council = HALCouncil(self._registry)
        self._streamer = ResponseStreamer()
        # Fleet Command singletons
        self._agent_registry = get_agent_registry()
        self._cost_tracker = CostTracker()
        self._env_registry = EnvironmentRegistry()
        self._fleet = FleetManager(self._agent_registry, self._sessions, self._cost_tracker)
        self._executor = BackgroundExecutor(self._agent_registry, provider_registry=self._registry)

    def start(self) -> None:
        """Start the server (blocking)."""
        from lib.hal.decide.http import with_decide_route
        handler = with_decide_route(_make_handler(
            self._registry, self._conversation, self._council,
            self._sessions, self._streamer,
            self._agent_registry, self._fleet, self._executor, self._env_registry,
        ))
        ThreadingHTTPServer.allow_reuse_address = True
        server = ThreadingHTTPServer(("127.0.0.1", self._port), handler)
        print(f"\n  {BOLD}{RED}HAL{RESET} Server — http://localhost:{self._port}")
        print(f"  {DIM}Ctrl+C to stop{RESET}\n")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print(f"\n  {DIM}Server stopped.{RESET}")
            server.server_close()


# ---------------------------------------------------------------------------
# /api/infer-with-tools helpers (W3 — wires file_sandbox + tool_loop + CCH)
# ---------------------------------------------------------------------------
#
# These live at module scope (NOT inside Handler) so they're individually
# testable + can pass the CC + KISS gates. The handler method
# (``_api_infer_with_tools``) is the thin orchestrator that calls them
# in order. Each helper returns a value or a {"error_code", "error_message"}
# dict so the orchestrator just chains them.
#
# Naming convention: ``_iwt_*`` ("infer-with-tools") — greppable, scoped.


def _iwt_parse_body(body: dict) -> dict:
    """Validate request body, return parsed params or ``{"error": "..."}``
    on missing required fields. Required: ``prompt``, ``sandbox_root``.
    Optional fields fall back to documented defaults.

    Mutation a test catches: drop the empty-prompt 400 → silently
    returns a dispatched parsed dict with ``prompt=""``.
    """
    prompt = body.get("prompt", "")
    sandbox_root = body.get("sandbox_root", "")
    if not prompt:
        return {"error": "No prompt provided"}
    if not sandbox_root:
        return {"error": "No sandbox_root provided"}
    # AGORA Council concern (R0-verified, GRIP PR #3286 forward-anchor):
    # ``body.get("tools") or default`` silently treats EMPTY LIST the same
    # as MISSING — replaces with the full allowlist. That violates caller
    # intent. Distinguish "absent" from "explicitly empty".
    if "tools" not in body:
        requested_tools = ["Read", "Grep", "Glob"]
    else:
        tools_field = body.get("tools")
        if not isinstance(tools_field, list) or not tools_field:
            return {"error": "tools must be a non-empty list (or omitted)"}
        requested_tools = tools_field
    return {
        "prompt": prompt,
        "sandbox_root": sandbox_root,
        "requested_tools": requested_tools,
        "model": body.get("model") or "claude-opus-4-7",
        "max_iterations": int(body.get("max_iterations") or 8),
        "max_tokens": int(body.get("max_tokens") or 16384),
        "system": body.get("system") or None,
    }


def _iwt_failsafe(code: str, msg) -> dict:
    """Build the public ``ok=false`` response shape. All error paths in
    the route funnel through here so the shape stays Goodhart-locked."""
    return {
        "ok": False, "text": "", "tool_calls": [],
        "iterations": 0, "hit_iteration_cap": False,
        "error_code": code, "error_message": str(msg)[:300],
    }


def _iwt_build_chat_fn(sub, model: str, system, max_tokens: int):
    """Return a chat_fn closure that wraps :meth:`CCHSubscription.chat`
    into the signature :func:`lib.hal.tool_loop.run_tool_loop` expects.

    The loop hands us list-of-dict messages; we convert to ChatMessage,
    call the signer, return the Anthropic response.raw dict that
    :func:`lib.hal.tool_adapter.parse_tool_use_blocks` reads.
    """
    from lib.hal.types import ChatMessage

    def chat_fn(*, messages, tools):
        chat_messages = [
            ChatMessage(role=m["role"], content=m["content"])
            for m in messages
        ]
        response = sub.chat(
            "", model=model, messages=chat_messages,
            system=system, tools=tools or None,
            max_tokens=max_tokens,
        )
        return response.raw or {
            "content": [{"type": "text", "text": response.text}],
            "stop_reason": "end_turn",
        }

    return chat_fn


def _iwt_extract_text(final_response) -> str:
    """Concat the text blocks out of an Anthropic response.content array.
    Empty string on malformed input (Goodhart-safe: prefers degrading
    over crashing the route)."""
    if not isinstance(final_response, dict):
        return ""
    content = final_response.get("content", [])
    if not isinstance(content, list):
        return ""
    return "".join(
        b.get("text", "") for b in content
        if isinstance(b, dict) and b.get("type") == "text"
    )


def _iwt_summarise_tool_calls(tool_calls: list) -> list:
    """Project the loop's internal tool_calls log into the public
    contract (4 keys per entry). Drops raw tool_use/tool_result dicts
    so internal shape changes don't leak to clients."""
    return [
        {
            "tool": tc["tool_use"].get("name", ""),
            "input": tc["tool_use"].get("input", {}),
            "is_error": tc["tool_result"].get("is_error", False),
            "iteration": tc.get("iteration", 0),
        }
        for tc in tool_calls
    ]


def _iwt_import_substrate():
    """Import every W1/W2/W3 module the pipeline needs.

    Returns ``(modules_dict, None)`` on success, or
    ``(None, failsafe_response)`` on any ImportError so the orchestrator
    chains cleanly.
    """
    try:
        from lib.hal.file_sandbox import SandboxRootRefused
        from lib.hal.sandboxed_tool_dispatch import (
            SandboxedDispatcher, _ALLOWED_TOOLS,
        )
        from lib.hal.tool_adapter import (
            _BUILTIN_SPECS, to_anthropic_tools,
        )
        from lib.hal.tool_loop import run_tool_loop
        from lib.hal.subscription import CCHSubscription
    except Exception as exc:
        return None, _iwt_failsafe("import_failed", exc)
    return {
        "SandboxRootRefused": SandboxRootRefused,
        "SandboxedDispatcher": SandboxedDispatcher,
        "_ALLOWED_TOOLS": _ALLOWED_TOOLS,
        "_BUILTIN_SPECS": _BUILTIN_SPECS,
        "to_anthropic_tools": to_anthropic_tools,
        "run_tool_loop": run_tool_loop,
        "CCHSubscription": CCHSubscription,
    }, None


def _iwt_build_dispatcher_and_schemas(parsed: dict, mods: dict):
    """Build the SandboxedDispatcher and the Anthropic tool-schema list.

    Returns ``((dispatcher, schemas), None)`` on success, or
    ``(None, failsafe_response)`` on bad sandbox root / empty allowlist
    intersection — both are recoverable client-side via fail-safe.
    """
    try:
        dispatcher = mods["SandboxedDispatcher"](parsed["sandbox_root"])
    except mods["SandboxRootRefused"] as exc:
        return None, _iwt_failsafe("sandbox_root_refused", exc)
    requested = parsed["requested_tools"]
    allowed = [t for t in requested if t in mods["_ALLOWED_TOOLS"]]
    if not allowed:
        return None, _iwt_failsafe(
            "no_allowed_tools",
            f"none of {requested} are in allowlist "
            f"{sorted(mods['_ALLOWED_TOOLS'])}",
        )
    schemas = mods["to_anthropic_tools"](
        tuple(s for s in mods["_BUILTIN_SPECS"] if s.name in allowed)
    )
    return (dispatcher, schemas), None


def _iwt_run(parsed: dict) -> dict:
    """Drive the full /api/infer-with-tools pipeline given a parsed body.

    Each step's failure becomes ``ok=false`` via :func:`_iwt_failsafe`
    so the GRIP adapter degrades cleanly. Returns the response dict the
    route emits as HTTP 200.
    """
    mods, err = _iwt_import_substrate()
    if err is not None:
        return err
    pair, err = _iwt_build_dispatcher_and_schemas(parsed, mods)
    if err is not None:
        return err
    dispatcher, tool_schemas = pair
    sub = mods["CCHSubscription"].from_env()
    if sub is None:
        return _iwt_failsafe(
            "cch_subscription_unavailable",
            "CLAUDE_CODE_OAUTH_TOKEN missing (env + keychain empty)",
        )
    chat_fn = _iwt_build_chat_fn(
        sub, parsed["model"], parsed["system"], parsed["max_tokens"],
    )
    try:
        result = mods["run_tool_loop"](
            messages=[{"role": "user", "content": parsed["prompt"]}],
            chat_fn=chat_fn, tools=tool_schemas,
            dispatcher=dispatcher,
            max_iterations=parsed["max_iterations"],
        )
    except Exception as exc:
        return _iwt_failsafe("tool_loop_raised", exc)
    return {
        "ok": True,
        "text": _iwt_extract_text(result.final_response),
        "tool_calls": _iwt_summarise_tool_calls(result.tool_calls),
        "iterations": result.iterations,
        "hit_iteration_cap": result.hit_iteration_cap,
        "error_code": None, "error_message": None,
    }


def _response_token_cost(response) -> float:
    """Best-effort token count from a provider response, for budget settlement.

    Reads ``usage.prompt_tokens`` + ``usage.completion_tokens`` defensively —
    missing / odd-shaped fields count as 0 — and floors at the 1-unit
    reservation when no usable token data is present. Never raises: an odd
    response shape must not crash throttle settlement (AGORA #384 r3).
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return 1.0
    try:
        prompt = float(getattr(usage, "prompt_tokens", 0) or 0)
        completion = float(getattr(usage, "completion_tokens", 0) or 0)
    except (TypeError, ValueError):
        return 1.0
    total = prompt + completion
    return total if total > 0 else 1.0


def _usage_for_http(usage) -> dict:
    """A result's token counts under the names GRIP's adapter reads.

    HAPPI says ``in`` and ``out``; lib/hal_llm_adapter.py in GRIP reads
    ``input_tokens`` and ``output_tokens``. Before this /api/infer returned no
    usage at all, so GRIP recorded zero tokens for every one of 1,027
    primary-path calls in the ten days to 2026-10-02. A count the provider did
    not report stays absent rather than becoming 0.
    """
    if not isinstance(usage, dict):
        return {}
    names = {"in": "input_tokens", "out": "output_tokens"}
    return {names[key]: value for key, value in usage.items() if key in names}


def _infer_response(r) -> dict:
    """The /api/infer JSON for a grip_infer result.

    ``media_attached`` is an attestation grip_infer computed from the parts
    actually built. On version skew (an older result object without the field)
    it is None, UNKNOWN rather than a fabricated False, so consumers fail
    closed; ``usage`` likewise falls back to empty, never to zero counts.
    """
    return {
        "ok": r.ok, "text": r.text, "provider": r.provider,
        "model": r.model, "idr_ref": r.idr_ref,
        "error_code": r.error_code, "error_message": r.error_message,
        "media_attached": getattr(r, "media_attached", None),
        "usage": _usage_for_http(getattr(r, "usage", None)),
        "finish_reason": getattr(r, "finish_reason", None),
    }


def _make_handler(registry, conversation, council, sessions, streamer,
                  agent_registry=None, fleet=None, executor=None, env_registry=None):
    """Create a request handler class with access to HAL components."""

    class Handler(BaseHTTPRequestHandler):

        def do_GET(self):
            # Gateway access guard (default-deny remote) + P1 /v1/* passthrough.
            if self.path.split("?", 1)[0].startswith("/v1/"):
                if not enforce_gateway_access(self):
                    _discard_unread_body(self)
                    return
                handle_proxy_request(self, _get_gateway_proxy(), self.path)
                return
            # Default-deny-remote for the native surface: a remote (tunnelled)
            # caller reaches only the public chat demo paths (UI assets +
            # /api/health) without a gateway token. Loopback is unaffected.
            if not _is_public_path("GET", self.path) and not enforce_gateway_access(self):
                _discard_unread_body(self)
                return
            routes = {
                "/": self._serve_ui,
                "/index.html": self._serve_ui,
                "/api/status": self._api_status,
                "/api/providers": self._api_providers,
                "/api/health": self._api_health,
                "/api/sessions": self._api_sessions_list,
                "/api/tools": self._api_tools,
                "/api/agents": self._api_agents_list,
                "/api/fleet": self._api_fleet,
                "/api/tasks": self._api_tasks_list,
                "/api/environments": self._api_envs_list,
            }
            handler = routes.get(self.path)
            if handler:
                handler()
                return
            if self._dispatch_get_param_route():
                return
            # Static files from grip-web dist (SPA fallback)
            if self._serve_static():
                return
            self.send_error(404)
            _discard_unread_body(self)

        def _dispatch_get_param_route(self) -> bool:
            """Serve GET path-parameter routes (/api/{sessions,agents,tasks}
            /<id> + tasks/<id>/events). Returns True iff one matched. Keeps
            do_GET within the KISS line budget (extract-dispatch pattern).
            """
            param_routes = (
                (_SESSION_ID_RE, self._api_session_detail),
                (_AGENT_ID_RE, self._api_agent_detail),
                (_TASK_ID_RE, self._api_task_detail),
                (_TASK_EVENTS_RE, self._api_task_events),
            )
            for regex, view in param_routes:
                match = regex.match(self.path)
                if match:
                    view(match.group(1))
                    return True
            return False

        def do_POST(self):
            _p = self.path.split("?", 1)[0]
            # Gateway access guard (default-deny remote /v1/* — open-relay
            # protection) THEN dispatch: P2 OpenAI-compat, P6 Responses, P1
            # verbatim /v1/* catch-all (closes GRIP #3274).
            if _p.startswith("/v1/"):
                if not enforce_gateway_access(self):
                    _discard_unread_body(self)
                    return
                if _p == "/v1/chat/completions":
                    handle_chat_completions(self, _get_gateway_proxy())
                    return
                if _p == "/v1/responses":
                    handle_responses(self, _get_gateway_proxy())
                    return
                handle_proxy_request(self, _get_gateway_proxy(), self.path)
                return
            # Default-deny-remote: only POST /api/chat (the public chat demo) is
            # public. Every other /api/* (infer, query, council, tools/execute,
            # agents, tasks, environments) requires a gateway token remotely.
            if not _is_public_path("POST", self.path) and not enforce_gateway_access(self):
                _discard_unread_body(self)
                return
            routes = {
                "/api/query": self._api_query,
                "/api/infer": self._api_infer,
                "/api/infer-with-tools": self._api_infer_with_tools,
                "/api/chat": self._api_chat,
                "/api/council": self._api_council,
                "/api/conversation": self._api_conversation_stream,
                "/api/sessions": self._api_session_create,
                "/api/tools/execute": self._api_tool_execute,
                "/api/tools/stream": self._api_tools_stream,
                "/api/agents": self._api_agents_create,
                "/api/tasks": self._api_tasks_submit,
                "/api/environments": self._api_envs_create,
            }
            handler = routes.get(self.path)
            if handler:
                handler()
            else:
                self.send_error(404)
                _discard_unread_body(self)

        def do_DELETE(self):
            # No DELETE route is public — always gated (default-deny-remote).
            if not enforce_gateway_access(self):
                _discard_unread_body(self)
                return
            match = _SESSION_ID_RE.match(self.path)
            if match:
                self._api_session_delete(match.group(1))
                return
            match = _AGENT_ID_RE.match(self.path)
            if match:
                self._api_agent_delete(match.group(1))
                return
            match = _ENV_ID_RE.match(self.path)
            if match:
                self._api_env_delete(match.group(1))
                return
            self.send_error(404)
            _discard_unread_body(self)

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors_headers()
            self.end_headers()

        def _api_query(self):
            """Single-turn query via provider. Compatible with web_demo.py.

            When the request body omits ``provider``, consult the GRIP marker
            (``state/session/active-llm.json``) for the operator's currently
            preferred provider. This closes the GRIP↔HAL round-trip:
            - HAL writes the marker after each route (status-line update)
            - HAL reads the marker for default routing (operator preference)
            Falls back to "groq" if no marker / stale marker / non-solo mode.
            """
            body = self._read_body()
            provider = body.get("provider") or _active_provider_default() or "groq"
            prompt = body.get("prompt", "")
            if not prompt:
                self._json({"error": "No prompt provided"}, 400)
                return
            try:
                result = _query_provider(registry, provider, prompt)
                self._json(result)
            except Exception as e:
                self._json({"error": str(e)[:300]}, 500)

        def _api_infer(self):
            """GRIP canonical AI syscall over HTTP — wraps grip_infer (RAILLM).

            Transport B for GRIP #2722: lets the GRIP adapter seam flip from
            its subprocess bridge to a clean HTTP boundary. Returns the SAME
            normalised shape the #2722 bridge produces so the swap is
            byte-compatible. Fail-safe by contract: a grip_infer/import
            failure is HTTP 200 with ok=false (the GRIP adapter falls back
            to its legacy chain on ok=false) — never a 5xx, never fail-hard.

            Multimodal (optional): ``media`` (list of local file paths) or
            pre-built ``content_parts`` (list of OpenAI ContentPart dicts)
            route the turn through grip_infer's messages path so an
            audio/image-capable provider receives the bytes inline. With a
            ``media``/``content_parts`` body the bare-prompt requirement is
            relaxed (the prompt may be empty — the parts carry the turn).
            """
            body = self._read_body()
            prompt = body.get("prompt", "")
            media = body.get("media") if isinstance(body.get("media"), list) else None
            content_parts = (body.get("content_parts")
                             if isinstance(body.get("content_parts"), list) else None)
            if not prompt and not media and not content_parts:
                self._json({"error": "No prompt, media, or content_parts provided"}, 400)
                return
            fail = lambda code, msg: self._json({
                "ok": False, "text": "", "provider": None, "model": None,
                "idr_ref": None, "error_code": code,
                "error_message": str(msg)[:300],
                "media_attached": None})  # no result object -> UNKNOWN, never False
            try:
                from lib.hal.grip_runtime import grip_infer
            except Exception as e:
                fail("grip_runtime_unavailable", e)
                return
            try:
                r = grip_infer(prompt, model=resolve_infer_model(body),
                                audit=bool(body.get("audit")),
                                media=media, content_parts=content_parts,
                                no_failover=bool(body.get("no_failover")),
                                max_tokens=_read_max_tokens(body))
            except Exception as e:
                fail("grip_infer_raised", e)
                return
            self._json(_infer_response(r))

        def _api_infer_with_tools(self):
            """Run an Anthropic Messages tool-loop over a capability-locked,
            sandbox-bound Read/Grep/Glob dispatcher.

            Wires together the W1+W2 building blocks (file_sandbox +
            SandboxedDispatcher + tool_loop + CCHSubscription). Fail-safe
            like ``/api/infer``: any internal failure is HTTP 200 with
            ``ok=false``; only missing-input is HTTP 400. Capability-lock
            (PARAMOUNT, per ``rules/untrusted-content-capability-lock.md``):
            this route filters caller-requested tools through the
            dispatcher's hardcoded frozenset allowlist, AND the dispatcher
            re-checks per tool_use — defence-in-depth.

            Body: ``{prompt, sandbox_root, tools?, model?, max_iterations?,
            system?, max_tokens?}``.
            Response: ``{ok, text, tool_calls[], iterations,
            hit_iteration_cap, error_code, error_message}``.
            See ``_iwt_*`` helpers for the per-step logic.
            """
            body = self._read_body()
            parsed = _iwt_parse_body(body)
            if "error" in parsed:
                self._json({"error": parsed["error"]}, 400)
                return
            response = _iwt_run(parsed)
            self._json(response)

        def _chat_throttle_check(self):
            """Resolve client IP + run the /api/chat throttle. Fail-open: any
            error (odd handler / client_address shape) yields ALLOW so the
            public route never crashes. Returns (throttle|None, allowed, code,
            reason)."""
            try:
                throttle = get_chat_throttle()
                peer = self.client_address[0] if self.client_address else ""
                ip = client_ip_from_headers(self.headers, peer)
                allowed, code, reason = throttle.check(ip)
                return throttle, allowed, code, reason
            except Exception:
                return None, True, 200, ""

        def _api_chat(self):
            """Multi-turn conversation query (public chat demo; throttled)."""
            # Throttle per-IP + daily budget BEFORE any LLM call so the internet
            # cannot drain the operator's prepaid subscription.
            _throttle, _allowed, _code, _reason = self._chat_throttle_check()
            if not _allowed:
                self._json({"error": _reason}, _code)
                _discard_unread_body(self)
                return
            self._serve_chat(_throttle)

        def _serve_chat(self, _throttle) -> None:
            """Run one /api/chat turn, owning the throttle reservation lifecycle.
            check() reserved a unit; this settles it via record_cost on success
            and releases it on every other exit — settle flag and release in ONE
            finally, so it can neither leak nor double-refund if a _json write
            raises (AGORA #384).

            Routes through the per-session SessionRegistry (via _read_chat_request)
            so concurrent callers on the multi-threaded ThreadingHTTPServer (#390)
            don't share one ``conversation`` — history/model/cost bleeding across
            users on the public endpoint. _throttle is None on fail-open only."""
            settled = False
            try:
                req = _read_chat_request(self, sessions)
                if req is None:
                    return
                session, message, model_override = req
                response, _ = _execute_turn(session, message, model_override)
                if _throttle is not None:
                    _throttle.record_cost(_response_token_cost(response))
                settled = True  # real cost recorded; reservation is settled
                # Persist the turn before the cosmetic statusline mark, so a
                # _mark_active failure (e.g. uncatalogued model) can't drop it.
                sessions.record_turn(session.id)
                if response.provider and response.model:
                    _mark_active(
                        f"{response.provider}/{response.model}",
                        reason=_fallback_reason(response),
                    )
                usage = getattr(response, "usage", None)
                self._json({
                    "text": response.text,
                    "model": response.model,
                    "provider": response.provider,
                    "latency_ms": round(response.latency_ms or 0),
                    "cost_usd": response.cost_usd or 0.0,
                    "tokens": {
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                        "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                    },
                    "turn": len(session.conversation.history) // 2,
                    "session_id": session.id,
                    "session_cost": session.conversation.cost.snapshot().total_usd,
                })
            except Exception as e:
                self._json({"error": str(e)[:300]}, 500)
            finally:
                if _throttle is not None and not settled:
                    _throttle.release_reservation()

        def _api_council(self):
            """Multi-LLM council query."""
            body = self._read_body()
            query = body.get("query", "")
            models = body.get("models") or _default_council_models(registry)
            if not query:
                self._json({"error": "No query provided"}, 400)
                return
            try:
                _mark_council(models, reason=f"council ({len(models)} seats)")
                result = council.deliberate(query, models)
                self._json({
                    "merged_text": result.merged_text,
                    "agreement": result.agreement_score,
                    "total_cost_usd": result.total_cost_usd,
                    "votes": [
                        {
                            "model": v.model,
                            "text": v.response.text,
                            "latency_ms": round(v.response.latency_ms or 0),
                        }
                        for v in result.votes
                    ],
                })
            except Exception as e:
                self._json({"error": str(e)[:300]}, 500)

        def _api_conversation_stream(self):
            """Streaming conversation endpoint for GRIP-GUI.

            Accepts: { "message": str, "session_id"?: str, "model"?: str }
            Streams: JSONL events matching GRIP-GUI StreamEvent protocol.
            """
            req = _read_chat_request(self, sessions)
            if req is None:
                return
            session, message, model_override = req
            self._start_stream()

            try:
                self._stream_event({"type": "system", "subtype": "init", "session_id": session.id})
                response, latency_ms = _execute_turn(session, message, model_override)
                _stream_response_chunks(self._stream_event, streamer, response.text)
                _stream_result_event(self._stream_event, session, response, latency_ms)
                sessions.record_turn(session.id)
            except Exception as e:
                self._stream_event({"type": "error", "error": {"message": str(e)[:300]}})

        def _start_stream(self):
            """Begin a chunked JSONL streaming response."""
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self._cors_headers()
            self.end_headers()

        def _api_sessions_list(self):
            """List all sessions, most recently updated first."""
            summaries = sessions.list_sessions()
            self._json({
                "sessions": [s.to_dict() for s in summaries],
                "count": len(summaries),
            })

        def _api_session_detail(self, session_id: str):
            """Get session detail including conversation history."""
            session = sessions.get(session_id)
            if not session:
                self._json({"error": "Session not found"}, 404)
                return
            self._json({
                "id": session.id,
                "title": session.title,
                "model": session.model,
                "message_count": session.message_count,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "history": [
                    {"role": m.role, "content": m.content}
                    for m in session.conversation.history
                ],
                "status": session.conversation.status(),
            })

        def _api_session_create(self):
            """Create a new empty session."""
            body = self._read_body()
            model = body.get("model", "")
            title = body.get("title", "New conversation")
            session_id = sessions.create(model=model, title=title)
            self._json({"session_id": session_id}, 201)

        def _api_session_delete(self, session_id: str):
            """Delete a session."""
            deleted = sessions.delete(session_id)
            if deleted:
                self._json({"deleted": True})
            else:
                self._json({"error": "Session not found"}, 404)

        def _api_status(self):
            """System status: providers, models, skills, health."""
            self._json(_build_status(registry))

        def _api_health(self):
            """System health dashboard.

            /api/health is public (served to remote, unauthenticated callers
            too — see _PUBLIC_GET_PATHS), so the body is narrowed for anyone
            off the loopback boundary: reveal_path=False keeps name/status
            for every check (a monitor keyed on those still works) but
            replaces the hal_home check's absolute-path detail with a
            generic word. See lib/hal/health.py and is_loopback_caller.
            """
            from lib.hal.health import HealthDashboard
            dashboard = HealthDashboard(registry)
            report = dashboard.check(reveal_path=is_loopback_caller(self))
            self._json(report.to_dict())

        def _api_providers(self):
            """List providers with health and models."""
            providers = []
            for p in registry.list_providers():
                models = list(p.models)
                if p.name == "ollama" and not models:
                    models = _get_ollama_models()
                providers.append({
                    "name": p.name,
                    "type": p.type.value,
                    "healthy": p.healthy,
                    "models": models,
                    "base_url": p.base_url,
                })
            self._json({"providers": providers})

        def _api_tools(self):
            """List available native tools with rich spec data.

            Delegates payload construction to :func:`build_tools_payload` so
            the Web and CLI surfaces share the same source of truth
            (``tool_adapter._BUILTIN_SPECS``) and the response shape is
            testable without standing up an HTTP server.
            """
            self._json(build_tools_payload())

        def _api_tool_execute(self):
            """Execute a single tool and return the result."""
            from lib.hal.tool_dispatch import ToolDispatcher
            body = self._read_body()
            tool_name = body.get("tool", "")
            tool_input = body.get("input", {})
            if not tool_name:
                self._json({"error": "No tool specified"}, 400)
                return
            dispatcher = ToolDispatcher()
            result = dispatcher.execute(tool_name, tool_input)
            if result.blocked:
                self._json({
                    "status": "blocked",
                    "reason": result.reason,
                }, 403)
            elif result.error:
                self._json({
                    "status": "error",
                    "error": result.error,
                }, 500)
            else:
                self._json({
                    "status": "ok",
                    "output": result.output,
                })

        def _api_tools_stream(self):
            """Stream a tool-augmented conversation turn as SSE events.

            Accepts: ``{"message": str, "session_id"?: str, "model"?: str}``
            Streams SSE events (``tool_call``, ``final``, ``error``). See
            :func:`lib.hal.tool_sse.iter_tool_events`.
            """
            from lib.hal.tool_cli_integration import detect_tool_chat_fn

            req = _read_chat_request(self, sessions)
            if req is None:
                return
            session, message, model_override = req
            model = model_override or session.conversation.model
            chat_fn, auth = detect_tool_chat_fn(model)
            self._start_sse_stream()
            self._pipe_tool_events(session, message, chat_fn, auth)

        def _pipe_tool_events(self, session, message, chat_fn, auth):
            """Pipe iter_tool_events through the SSE writer (or emit error)."""
            from lib.hal.tool_cli_integration import build_builtin_tool_stack
            from lib.hal.tool_sse import format_sse_event, iter_tool_events

            if chat_fn is None:
                self._write_sse(format_sse_event(
                    "error",
                    {"message": "no auth configured; set up CCH or store a key in the hal-anthropic keychain"},
                ))
                return
            tools, dispatcher = build_builtin_tool_stack()
            self._write_sse(format_sse_event(
                "session", {"session_id": session.id, "auth": auth},
            ))
            try:
                for event in iter_tool_events(
                    session.conversation, message,
                    chat_fn=chat_fn, tools=tools, dispatcher=dispatcher,
                ):
                    self._write_sse(event)
                sessions.record_turn(session.id)
            except Exception as e:
                self._write_sse(format_sse_event(
                    "error", {"message": str(e)[:300]},
                ))

        def _start_sse_stream(self):
            """Begin a text/event-stream SSE response."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")  # disable nginx buffering
            self._cors_headers()
            self.end_headers()

        def _write_sse(self, chunk: str):
            """Write one pre-formatted SSE event string to the client."""
            try:
                self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # Client disconnected mid-stream — nothing to do, drop.
                pass

        def _serve_ui(self):
            """Serve the compiled UI locally, or the public HAL page."""
            content = _WEB_UI.encode()
            if is_loopback_caller(self):
                index = os.path.join(_hal_home, "apps", "grip-web", "dist", "index.html")
                if os.path.isfile(index):
                    with open(index, "rb") as f:
                        content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(content)

        def _serve_static(self) -> bool:
            """Serve static files from grip-web dist directory."""
            dist_dir = os.path.join(_hal_home, "apps", "grip-web", "dist")
            # Sanitise path to prevent traversal
            clean = os.path.normpath(self.path.lstrip("/"))
            if ".." in clean:
                return False
            filepath = os.path.join(dist_dir, clean)
            if not os.path.isfile(filepath):
                # SPA fallback: serve index.html for non-API, non-asset paths
                if not self.path.startswith("/api/") and not "." in os.path.basename(self.path):
                    index = os.path.join(dist_dir, "index.html")
                    if os.path.isfile(index):
                        with open(index, "rb") as f:
                            content = f.read()
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.end_headers()
                        self.wfile.write(content)
                        return True
                return False
            # Serve the file with appropriate content type
            ext = os.path.splitext(filepath)[1].lower()
            content_types = {
                ".html": "text/html", ".css": "text/css", ".js": "application/javascript",
                ".json": "application/json", ".png": "image/png", ".svg": "image/svg+xml",
                ".ico": "image/x-icon", ".woff2": "font/woff2", ".woff": "font/woff",
            }
            ctype = content_types.get(ext, "application/octet-stream")
            with open(filepath, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            if ext in (".js", ".css", ".woff2"):
                self.send_header("Cache-Control", "public, max-age=31536000, immutable")
            self.end_headers()
            self.wfile.write(content)
            return True

        # --- Fleet Command: Agent endpoints ---

        def _api_agents_list(self):
            if not agent_registry:
                return self._json({"error": "Fleet Command not initialised"}, 503)
            agents = [a.summary() for a in agent_registry.list_agents()]
            self._json({"agents": agents, "total": len(agents)})

        def _api_agents_create(self):
            if not agent_registry:
                return self._json({"error": "Fleet Command not initialised"}, 503)
            body = self._read_body()
            name = body.get("name", "")
            if not name:
                return self._json({"error": "name is required"}, 400)
            defn = agent_registry.create(
                name=name,
                model=body.get("model", ""),
                system_prompt=body.get("system_prompt", ""),
                tools=body.get("tools", []),
                skills=body.get("skills", []),
                mcp_servers=body.get("mcp_servers", []),
                max_turns=body.get("max_turns", 100),
                budget_usd=body.get("budget_usd", 5.0),
                routing_strategy=body.get("routing_strategy", "fallback"),
                fallback_chain=body.get("fallback_chain", []),
                council_models=body.get("council_models", []),
                environment_id=body.get("environment_id", ""),
                metadata=body.get("metadata", {}),
            )
            self._json(defn.to_dict(), 201)

        def _api_agent_detail(self, agent_id):
            if not agent_registry:
                return self._json({"error": "Fleet Command not initialised"}, 503)
            defn = agent_registry.get(agent_id)
            if defn is None:
                return self._json({"error": "Agent not found"}, 404)
            versions = agent_registry.store.list_versions(agent_id)
            data = defn.to_dict()
            data["versions"] = versions
            self._json(data)

        def _api_agent_delete(self, agent_id):
            if not agent_registry:
                return self._json({"error": "Fleet Command not initialised"}, 503)
            if agent_registry.delete(agent_id):
                self._json({"deleted": True})
            else:
                self._json({"error": "Agent not found"}, 404)

        # --- Fleet Command: Fleet snapshot ---

        def _api_fleet(self):
            if not fleet:
                return self._json({"error": "Fleet Command not initialised"}, 503)
            snap = fleet.snapshot()
            self._json({
                "timestamp": snap.timestamp,
                "total_agents": snap.total_agents,
                "total_sessions": snap.total_sessions,
                "active_sessions": snap.active_sessions,
                "completed_tasks": snap.completed_tasks,
                "total_cost_usd": snap.total_cost_usd,
                "cost_by_provider": dict(snap.cost_by_provider),
                "cost_by_model": dict(snap.cost_by_model),
                "zero_cost_ratio": snap.zero_cost_ratio,
                "provider_diversity": snap.provider_diversity,
                "council_usage": snap.council_usage,
                "sentinel_interventions": snap.sentinel_interventions,
            })

        # --- Fleet Command: Background tasks ---

        def _api_tasks_list(self):
            if not executor:
                return self._json({"error": "Fleet Command not initialised"}, 503)
            tasks = [t.summary() for t in executor.list_tasks()]
            self._json({"tasks": tasks, "total": len(tasks)})

        def _api_tasks_submit(self):
            if not executor:
                return self._json({"error": "Fleet Command not initialised"}, 503)
            body = self._read_body()
            agent_id = body.get("agent_id", "")
            description = body.get("description", "")
            if not agent_id or not description:
                return self._json({"error": "agent_id and description required"}, 400)
            task_id = executor.submit(agent_id, description)
            self._json({"task_id": task_id, "status": "queued"}, 201)

        def _api_task_detail(self, task_id):
            if not executor:
                return self._json({"error": "Fleet Command not initialised"}, 503)
            task = executor.status(task_id)
            if task is None:
                return self._json({"error": "Task not found"}, 404)
            self._json(task.to_dict())

        def _api_task_events(self, task_id):
            if not executor:
                return self._json({"error": "Fleet Command not initialised"}, 503)
            since = 0
            events = executor.events(task_id, since=since)
            self._json({"events": [e.to_dict() for e in events], "total": len(events)})

        # --- Fleet Command: Environments ---

        def _api_envs_list(self):
            if not env_registry:
                return self._json({"error": "Fleet Command not initialised"}, 503)
            envs = [e.to_dict() for e in env_registry.list_environments()]
            self._json({"environments": envs, "total": len(envs)})

        def _api_envs_create(self):
            if not env_registry:
                return self._json({"error": "Fleet Command not initialised"}, 503)
            body = self._read_body()
            name = body.get("name", "")
            if not name:
                return self._json({"error": "name is required"}, 400)
            safe_keys = {
                "description", "allowed_tools", "blocked_tools", "network_policy",
                "allowed_domains", "max_turns", "budget_usd", "timeout_s",
                "work_dir", "persistent_fs", "allowed_providers", "require_local_only",
            }
            kwargs = {k: v for k, v in body.items() if k in safe_keys}
            env = env_registry.create(name=name, **kwargs)
            self._json(env.to_dict(), 201)

        def _api_env_delete(self, env_id):
            if not env_registry:
                return self._json({"error": "Fleet Command not initialised"}, 503)
            if env_registry.delete(env_id):
                self._json({"deleted": True})
            else:
                self._json({"error": "Environment not found"}, 404)

        # --- Utility methods ---

        def _read_body(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw)

        def _json(self, data, code=200):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def _cors_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _stream_event(self, event: dict):
            """Write a single JSONL event with chunked transfer encoding."""
            line = json.dumps(event) + "\n"
            chunk = f"{len(line.encode()):x}\r\n{line}\r\n"
            self.wfile.write(chunk.encode())
            self.wfile.flush()

        def log_message(self, fmt, *args):
            pass  # Suppress default request logging

    return Handler


def _read_chat_request(handler, sessions):
    """Parse + validate a chat body and resolve its session — the shared preamble
    for the per-session chat handlers (/api/chat, /api/conversation/stream,
    /api/tools/stream). Reads the JSON body, rejects an empty message with a 400,
    then resolve-or-creates the session. Returns ``(session, message,
    model_override)``, or ``None`` after emitting a 400 (empty message) or 500
    (body-read / session-resolve failure) so the caller can simply ``return``.
    ``model_override`` is the raw body model (may be None) passed to
    _execute_turn; _prepare_session is given ``model_override or ""``."""
    try:
        body = handler._read_body()
        message = body.get("message", "")
        if not message:
            handler._json({"error": "No message provided"}, 400)
            return None
        model_override = body.get("model")
        session = _prepare_session(
            sessions, body.get("session_id"), model_override or "", message,
        )
    except (KeyboardInterrupt, GeneratorExit):
        raise  # never eat a genuine interrupt / generator-close signal
    except BaseException as exc:
        # A resolve failure (malformed body, or a session-backend error including
        # a BaseException like a catalogue SystemExit) must surface as a 500 via
        # the per-handler _json contract — not propagate uncaught out of the
        # streaming handlers, whose _read_chat_request call sits outside their own
        # try and would otherwise crash the worker thread. Real control-flow
        # signals are re-raised above. The error write is itself best-effort: if
        # the client already dropped (broken pipe), there's nothing left to send.
        try:
            handler._json({"error": str(exc)[:300]}, 500)
        except Exception as write_exc:
            print(f"[hal] failed to send 500 for chat request: {write_exc}", file=sys.stderr)
        return None
    return session, message, model_override


def _prepare_session(sessions, session_id, model, message):
    """Resolve or create session, apply model override, auto-title."""
    session = sessions.get_or_create(session_id, model=model)
    if model and model != session.model:
        session.conversation.configure(model=model)
        session.model = model
    if session.message_count == 0:
        title = message[:50].strip() + ("..." if len(message) > 50 else "")
        sessions.update_title(session.id, title)
    return session


def _execute_turn(session, message, model_override):
    """Send message through the conversation engine. Returns (response, latency_ms).

    Holds the per-session lock around the turn so two requests sharing one
    session_id can't interleave writes to the same Conversation (history/cost
    corruption) under the multi-threaded server. Distinct sessions hold distinct
    locks, so cross-session concurrency is unaffected."""
    start = time.monotonic()
    with session.lock:
        response = session.conversation.send(message, model=model_override or None)
    latency_ms = (time.monotonic() - start) * 1000
    return response, latency_ms


def _stream_response_chunks(emit, streamer, text):
    """Emit text as chunked text events matching GRIP-GUI StreamEvent protocol."""
    for chunk in streamer.stream_chars(text, chars_per_chunk=80):
        emit({"type": "text", "subtype": "text_delta", "delta": {"type": "text_delta", "text": chunk.text}})


def _stream_result_event(emit, session, response, latency_ms):
    """Emit the final result event with session metrics."""
    snap = session.conversation.cost.snapshot()
    emit({
        "type": "result",
        "cost_usd": snap.total_usd,
        "num_turns": len(session.conversation.history) // 2,
        "session_id": session.id,
        "model": response.model,
        "duration_ms": round(latency_ms),
    })


def _query_provider(registry, provider_name: str, prompt: str) -> dict:
    """Route a single query through the registry to a specific provider."""
    address = _resolve_provider_model(registry, provider_name)
    _mark_active(address, reason="query")
    start = time.monotonic()
    response = registry.chat(address, prompt, max_tokens=512)
    latency = (time.monotonic() - start) * 1000

    is_local = provider_name in ("ollama", "vllm", "lmstudio", "llamacpp")
    return {
        "text": response.text,
        "model": response.model,
        "provider": response.provider,
        "latency_ms": round(latency),
        "cost_usd": response.cost_usd or 0.0,
        "tokens": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        },
        "private": is_local,
    }


def _resolve_provider_model(registry, provider_name: str) -> str:
    """Resolve a provider name to its first available model address. O(n) scan."""
    providers_by_name = {p.name: p for p in registry.list_providers()}
    p = providers_by_name.get(provider_name)
    if p is None:
        raise ValueError(f"Unknown provider: {provider_name}")
    if p.models:
        return f"{p.name}/{p.models[0]}"
    local_models = _get_ollama_models()
    if local_models:
        return f"{p.name}/{local_models[0]}"
    raise ValueError(f"No models available for {provider_name}")


def build_tools_payload() -> dict:
    """Build the ``/api/tools`` response body.

    Binds the Web payload to ``lib/hal/tool_adapter._BUILTIN_SPECS`` so it
    cannot drift from ``hal --tools`` output. Each entry carries enough
    information for a front-end to render a full capability table without
    round-tripping extra calls: name, description, concurrency flag, required
    inputs, full input schema, and a ``registered`` boolean that records
    whether the native ``ToolDispatcher`` currently has a matching handler
    (always True on stock HAL — False if a deployment ships a subset).
    """
    from lib.hal.tool_adapter import _BUILTIN_SPECS
    from lib.hal.tool_dispatch import ToolDispatcher

    registered = set(ToolDispatcher().available_tools())
    tools = []
    for spec in sorted(_BUILTIN_SPECS, key=lambda s: s.name):
        tools.append({
            "name": spec.name,
            "type": "native",
            "description": spec.description,
            "is_concurrent_safe": spec.is_concurrent_safe,
            "required": list(spec.input_schema.get("required", [])),
            "input_schema": spec.input_schema,
            "registered": spec.name in registered,
        })
    return {"tools": tools, "count": len(tools)}


def _default_council_models(registry) -> list[str]:
    """Pick up to 5 models for council from available providers. Single pass."""
    models: list[str] = []
    ollama_models = None
    for p in registry.list_providers():
        if p.models:
            models.append(f"{p.name}/{p.models[0]}")
        elif p.name == "ollama":
            if ollama_models is None:
                ollama_models = _get_ollama_models()
            if ollama_models:
                models.append(f"ollama/{ollama_models[0]}")
    return models[:5]


def _build_status(registry) -> dict:
    """Build status dict for the /api/status endpoint."""
    from lib.hal.agent_runner import AgentRunner
    from lib.hal.memory_engine import MemoryEngine
    from lib.hal.skill_loader import SkillLoader

    loader = SkillLoader()
    skills = loader.discover()
    modes = loader.list_modes()
    agents = AgentRunner().list_agents()
    memory = MemoryEngine()
    entries = memory.list_entries()
    ollama_models = _get_ollama_models()

    providers = {}
    for p in registry.list_providers():
        is_local = p.type.value == "local"
        providers[p.name] = {
            "name": _provider_display_name(p.name, ollama_models),
            "type": p.type.value,
            "speed": "7-12 tok/s" if is_local else "Fast",
            "privacy": "FULLY PRIVATE" if is_local else "Cloud API",
        }

    return {
        "skills": len(skills),
        "modes": len(modes),
        "agents": len(agents),
        "memory_entries": len(entries),
        "local_models": ollama_models,
        "providers": providers,
    }


def _provider_display_name(name: str, ollama_models: list) -> str:
    """Human-readable provider name for the web UI."""
    names = {
        "groq": "Groq (Llama 3.3 70B)",
        "gemini": "Gemini 2.5 Flash",
        "ollama": "Ollama (Local)",
        "vllm": "vLLM (Local)",
        "lmstudio": "LM Studio (Local)",
    }
    return names.get(name, name.title())


_WEB_UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
  <meta name="theme-color" content="#0a0a0a">
  <title>HAL -- Harness Abstraction Layer</title>
  <meta name="description" content="HAL connects AI providers through the same call, routes requests and records which provider answered.">
  <style>
    :root {
      --bg: #0a0a0a; --surface: #14141a; --elevated: #1c1c24;
      --border: #2a2a35; --text: #f5f5f7; --muted: #a3a3ad;
      --accent: #d4a373; --accent-soft: rgba(212,163,115,0.12);
      --ok: #22c55e; --warn: #f59e0b; --user-bubble: #1e3a5f;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body, #app { height: 100%; min-height: 100dvh; }
    body { background: var(--bg); color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", system-ui, sans-serif;
      font-size: 14px; line-height: 1.5; -webkit-font-smoothing: antialiased; }
    #app { display: grid; grid-template-rows: auto 1fr auto; }
    header { padding: 14px 20px; background: var(--surface); border-bottom: 1px solid var(--border);
      display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; }
    .brand { font-size: 16px; font-weight: 600; letter-spacing: -0.01em; }
    .brand .hal { color: var(--accent); }
    .brand .sub { color: var(--muted); font-weight: 400; margin-left: 8px; font-size: 13px; }
    .badge { background: var(--elevated); padding: 4px 10px; border-radius: 12px; border: 1px solid var(--border);
      font-size: 11px; color: var(--muted); font-variant-numeric: tabular-nums; }
    main { overflow-y: auto; padding: 24px 20px 12px; display: flex; flex-direction: column; gap: 14px; }
    .welcome { max-width: 760px; margin: 0 auto; padding: 32px 0 16px; }
    .welcome h1 { font-size: 22px; font-weight: 600; margin-bottom: 8px; }
    .welcome h1 .hal { color: var(--accent); }
    .welcome p { color: var(--muted); font-size: 14px; max-width: 560px; }
    .welcome .examples { margin-top: 20px; display: grid; gap: 8px; }
    .example { background: var(--elevated); border: 1px solid var(--border); border-radius: 8px;
      padding: 10px 14px; cursor: pointer; font-size: 13px; transition: border-color 0.15s; }
    .example:hover { border-color: var(--accent); }
    .msg { max-width: 760px; width: 100%; margin: 0 auto; display: flex; gap: 10px; }
    .msg.user { justify-content: flex-end; }
    .bubble { padding: 10px 14px; border-radius: 12px; max-width: 80%;
      white-space: pre-wrap; word-wrap: break-word; }
    .msg.user .bubble { background: var(--user-bubble); color: var(--text); }
    .msg.assistant .bubble { background: var(--elevated); border: 1px solid var(--border); color: var(--text); }
    .meta { font-size: 11px; color: var(--muted); margin-top: 4px; font-family: ui-monospace, monospace; }
    footer { padding: 12px 20px; background: var(--surface); border-top: 1px solid var(--border); }
    .composer { max-width: 760px; margin: 0 auto; display: grid; grid-template-columns: 1fr auto; gap: 8px; }
    #prompt { width: 100%; padding: 12px 14px; border: 1px solid var(--border); border-radius: 10px;
      background: var(--elevated); color: var(--text); font: inherit; resize: none; min-height: 48px;
      max-height: 200px; }
    #prompt:focus { outline: none; border-color: var(--accent); }
    #send { padding: 0 18px; border: 1px solid var(--accent); border-radius: 10px;
      background: var(--accent); color: #0a0a0a; font-weight: 600; cursor: pointer; }
    #send:disabled { opacity: 0.5; cursor: not-allowed; }
    .credit { text-align: center; font-size: 11px; color: var(--muted); margin-top: 8px; }
    .credit a { color: var(--muted); }
    @media (max-width: 600px) {
      .welcome h1 { font-size: 18px; } main { padding: 16px 14px 8px; }
    }
  </style>
</head>
<body>
<div id="app">
  <header>
    <div class="brand">
      <span class="hal">HAL</span>
      <span class="sub">Harness Abstraction Layer</span>
    </div>
    <div class="badge" id="model-badge" hidden></div>
  </header>
  <main id="transcript">
    <div class="welcome" id="welcome">
      <h1>Any model, one cable.</h1>
      <p>HAL connects AI models through one interface. The same call works for any provider.
        If one is slow or down, HAL routes to the next and records which one answered.</p>
      <p>For high-stakes calls, HAL's council asks models from different companies the same question.
        A model from another company judges their answers.</p>
      <div class="examples">
        <div class="example" data-q="Explain how AI provider routing works in plain English.">Explain provider routing</div>
        <div class="example" data-q="Why might models from different companies review the same question?">Explain council review</div>
      </div>
    </div>
  </main>
  <footer>
    <form id="form" action="javascript:void(0)" onsubmit="return false">
      <div class="composer">
        <textarea id="prompt" placeholder="Send a message" autofocus></textarea>
        <button id="send" type="submit">Send</button>
      </div>
    </form>

  </footer>
</div>
<script>
const transcript = document.getElementById('transcript');
const welcome = document.getElementById('welcome');
const form = document.getElementById('form');
const prompt = document.getElementById('prompt');
const send = document.getElementById('send');
const badge = document.getElementById('model-badge');
function _genSessionId(){
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  // Fallback for older browsers / non-secure contexts
  return 'sess-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 10);
}
let sessionId = _genSessionId();

function escapeHtml(s){return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}

function appendMsg(role, text, meta){
  if (welcome.parentNode) welcome.remove();
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;
  wrap.innerHTML = `<div style="display:flex;flex-direction:column;align-items:${role==='user'?'flex-end':'flex-start'}"><div class="bubble">${escapeHtml(text)}</div>${meta?`<div class="meta">${escapeHtml(meta)}</div>`:''}</div>`;
  transcript.appendChild(wrap);
  transcript.scrollTop = transcript.scrollHeight;
  return wrap;
}

async function ask(msg){
  const userEl = appendMsg('user', msg);
  send.disabled = true;
  const thinking = appendMsg('assistant', '…', 'thinking');
  try {
    const res = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({message: msg, session_id: sessionId})
    });
    const data = await res.json();
    thinking.remove();
    if (data.error) {
      appendMsg('assistant', `Error: ${data.error}`, '');
    } else {
      const meta = `${data.provider||'?'}/${data.model||'?'} · ${data.latency_ms||'?'}ms · turn ${data.turn||'?'}`;
      appendMsg('assistant', data.text || '(no response)', meta);
      if (data.provider && data.model) {
        badge.textContent = `served by: ${data.provider}/${data.model}`;
        badge.hidden = false;
      }
    }
  } catch (e) {
    thinking.remove();
    appendMsg('assistant', `Network error: ${e.message}`, '');
  }
  send.disabled = false;
  prompt.focus();
}

form.addEventListener('submit', e => {
  e.preventDefault();
  const v = prompt.value.trim();
  if (!v) return;
  prompt.value = '';
  ask(v);
});
prompt.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});
document.querySelectorAll('.example').forEach(el => {
  el.addEventListener('click', () => ask(el.dataset.q));
});
</script>
</body>
</html>"""




if __name__ == "__main__":
    from lib.hal.env_boot import load_env
    load_env()  # keychain->env bridge for pm2/launchd; CCH canonical-first
    port = 3850
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv[1:]):
            if arg == "--port" and i + 2 <= len(sys.argv):
                port = int(sys.argv[i + 2])
            elif arg.isdigit():
                port = int(arg)

    from lib.hal.warmup import warmup_background
    reg = get_registry()
    _register_anthropic_cch(reg)
    discover_local_providers(reg)
    warmup_background(reg)
    HALServer(port=port, registry=reg).start()
