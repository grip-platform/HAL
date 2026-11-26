#!/usr/bin/env python3
"""HAL — standalone Knowledge Engine.

Usage:
    hal                          # Interactive chat (warms up local models)
    hal --serve [--port 3850]    # Start HTTP server + web UI
    hal --model groq/llama-3.3   # Chat with specific model
    hal --mode legal             # Chat in domain mode
    hal --status                 # Show providers, models, health
    hal --models                 # List every model with strengths and tier

Commands (in chat): /models, /model, /mode, /cost, /providers, /council, /history, /clear, /quit
"""
import contextlib
import json
import shutil
import signal
import subprocess
import sys
import os
import tempfile
import textwrap
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

try:
    import termios
except ImportError:  # Windows: there is no termios, and nothing for /dojo to restore
    termios = None

_hal_home = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
sys.path.insert(0, _hal_home)

from lib.hal.conversation import Conversation
from lib.hal.model_switcher import detect_model_switch, fuzzy_resolve_model
from lib.hal.cost_heatmap import format_citation_chip, render_heatmap
from lib.hal.discovery import discover_local_providers
from lib.hal.moe import HALCouncil
from lib.hal.registry import get_registry
from lib.hal.response_formatter import ResponseFormatter
from lib.hal.response_streamer import ResponseStreamer
from lib.hal.status_footer import FooterState, StatusFooter
from lib.hal.warmup import warmup_background

_STREAMER = ResponseStreamer()


def _stream_chunk_chars() -> int:
    try:
        return max(1, int(os.environ.get("HAL_STREAM_CHUNK_CHARS", "20")))
    except ValueError:
        return 20


def _stream_delay_ms() -> int:
    try:
        return max(0, int(os.environ.get("HAL_STREAM_DELAY_MS", "20")))
    except ValueError:
        return 20

CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD = "\033[1m"
DIM = "\033[2m"
GRAY = "\033[38;5;240m"
RESET = "\033[0m"

# ── Improvement 2: latency colour thresholds ─────────────────────────
_LATENCY_GREEN_MS = 500
_LATENCY_YELLOW_MS = 2000

# ── Improvement 1: cost display thresholds ───────────────────────────
_COST_DIM_FLOOR = 0.01    # below → hidden
_COST_NORMAL_FLOOR = 0.10  # below → dim
_COST_YELLOW_FLOOR = 1.00  # below → normal weight; above → yellow

# ── Improvement 11: known commands for did-you-mean ──────────────────
_KNOWN_COMMANDS = frozenset({
    "/model", "/mode", "/cost", "/providers", "/models",
    "/council", "/history", "/clear", "/tools", "/help",
    "/quit", "/exit", "/voice", "/settings", "/dojo", "/?",
})


def _format_latency(latency_ms: float) -> str:
    """Colour-coded latency: green fast, yellow medium, red slow."""
    if latency_ms < _LATENCY_GREEN_MS:
        return f"{GREEN}{latency_ms:.0f}ms{RESET}"
    if latency_ms < _LATENCY_YELLOW_MS:
        return f"{YELLOW}{latency_ms:.0f}ms{RESET}"
    return f"{RED}{latency_ms:.0f}ms{RESET}"


def _format_cost(total_usd: float, turn_usd: float = 0.0) -> str:
    """Progressive cost disclosure: hidden below threshold, coloured above."""
    if total_usd < _COST_DIM_FLOOR:
        return ""
    delta = f"+${turn_usd:.4f} · " if turn_usd > 0 else ""
    amount = f"{delta}${total_usd:.4f} session"
    if total_usd < _COST_NORMAL_FLOOR:
        return f"{DIM}  ·  {amount}{RESET}"
    if total_usd < _COST_YELLOW_FLOOR:
        return f"  ·  {amount}"
    return f"{YELLOW}  ·  {amount}{RESET}"


def _provider_badge(provider: str, conv) -> str:
    """Return '[LOCAL]' dim tag for on-device providers, else empty."""
    try:
        for p in conv._registry.list_providers():
            if p.name == provider and p.type.value == "local":
                return f" {DIM}[LOCAL]{RESET}"
    except Exception:
        pass
    return ""


def _did_you_mean(command: str) -> str:
    """Return the closest known command, or empty string."""
    import difflib
    close = difflib.get_close_matches(command, _KNOWN_COMMANDS, n=1, cutoff=0.6)
    return close[0] if close else ""


_HELP_TEXT = """\
  HAL — Slash Commands
  ─────────────────────────────────────────────────────────────────
  Conversation
    /model <name>        switch model  e.g. groq/llama-3.3, ollama/mistral
    /mode  <name>        set domain mode  e.g. legal, code, research
    /history             show recent turns (role + first 100 chars)
    /clear               wipe history and start fresh

  Providers & Models
    /models [filter]     full model catalogue with tiers, strengths, cost
    /providers           registered providers and live health status
    /cost                session cost breakdown by provider and total tokens

  Tools  (requires CCH auth: CLAUDE_CODE_OAUTH_TOKEN or keychain grip-claude-oauth)
    /tools               show current tool mode status
    /tools on            enable built-in file / shell tools
    /tools off           disable tool mode

  Council  (queries all configured models in parallel)
    /council <question>  send question to every model, merge the answers

  Voice
    /voice               enable spoken output (Jamie Premium · macos_say)
    /voice off           disable spoken output

  Settings
    /settings            open model browser TUI (↑↓ providers, ←→ models, Enter)

  Dojo
    /dojo                open the GRIP dojo — three runnable lessons

  Session
    /quit | /exit        end session and print final cost summary

  Command-line flags
    hal --status         provider/model health report
    hal --models         browse full model catalogue
    hal voice <text>     speak via TTS ladder: Piper → Google Cloud → Gemini
    hal --serve          start HTTP + web UI server (default :3850)
  ─────────────────────────────────────────────────────────────────"""


def _print_help() -> None:
    """Print the HAL command reference."""
    print(f"\n{DIM}{_HELP_TEXT}{RESET}\n")


def _bar(char: str = "─") -> str:
    """Return a separator bar scaled to terminal width (max 72)."""
    w = min(shutil.get_terminal_size((80, 24)).columns - 4, 72)
    return f"  {GRAY}{char * w}{RESET}"


def _bar_with_turn(turn: int = 0, char: str = "╌") -> str:
    """Turn-labelled separator: ╌╌╌ 3 ╌╌╌ at terminal width."""
    w = min(shutil.get_terminal_size((80, 24)).columns - 4, 72)
    if not turn:
        return f"  {GRAY}{char * w}{RESET}"
    label = f" {turn} "
    remaining = max(4, w - len(label))
    left = remaining // 2
    right = remaining - left
    return f"  {GRAY}{char * left}{RESET}{DIM}{turn}{RESET}{GRAY}{char * right}{RESET}"


def _clear_processing_indicator() -> None:
    """Erase the ◈ thinking glyph printed before an API call."""
    sys.stdout.write("   \r")
    sys.stdout.flush()


def _boot_providers(args: dict) -> tuple:
    """Init registry, probe local servers with TTY feedback, start warmup thread."""
    registry = get_registry()
    if sys.stdout.isatty():
        print(f"{DIM}  Discovering local providers...{RESET}", end="\r", flush=True)
    discovered = discover_local_providers(registry)
    if sys.stdout.isatty():
        print(" " * 38, end="\r", flush=True)
    warmup_background(registry, verbose=not args["status"])
    return registry, discovered


def main() -> None:
    from lib.hal.env_boot import load_env
    load_env()  # keychain->env bridge for non-login shells; CCH canonical-first
    args = _parse_args()

    if args["daemon"]:
        _handle_daemon(args["daemon"], args["port"])
        return

    if args["help"]:
        _show_help()
        return

    if args["subscribe"]:
        from lib.hal.subscribe_wizard import DEFAULT_PROVIDER, run_wizard
        provider = DEFAULT_PROVIDER if args["subscribe"] == "*" else args["subscribe"]
        sys.exit(run_wizard(provider))

    if args["happi_api"]:
        from lib.hal.happi.runtime import main as happi_main
        sys.exit(happi_main())

    registry, discovered = _boot_providers(args)

    if args["status"]:
        _show_status(registry, discovered)
        return

    if args["tools"]:
        _show_tools()
        return

    if args["models"]:
        from lib.hal.models_command import handle_models_command
        filter_term = "" if args["models"] == "*" else args["models"]
        print(handle_models_command(registry, filter_term))
        return

    if args["serve"]:
        _start_server(registry, args["port"])
        return

    _run_chat(registry, discovered, args["model"], args["mode"], args["initial_message"])


def _print_banner(n_providers: int, model: str, discovered: list, mode: str) -> None:
    """Render the HAL startup banner — data only, no decorative taglines."""
    local_tag = f"  {DIM}local: {', '.join(discovered)}{RESET}" if discovered else ""
    mode_tag = f"  {DIM}mode: {mode}{RESET}" if mode else ""
    bar = _bar()
    print()
    print(f"  {BOLD}{RED}HAL{RESET}  {DIM}Knowledge Engine{RESET}")
    print(bar)
    print(f"  {DIM}{n_providers} providers  ·  {model}{RESET}{local_tag}{mode_tag}")
    print(f"  {GRAY}/model  /mode  /tools  /council  /models  /cost  /help  /quit{RESET}")
    print(bar)
    print()


def _render_thinking_line(rail: str, raw_line: str, width: int) -> None:
    """Wrap one thinking line and print each segment behind the ╎ rail."""
    for line in textwrap.wrap(raw_line, width=width - 2):
        print(f"{GRAY}{rail}{line}{RESET}")


def _render_thinking(text: str, width: int) -> None:
    """Render extended-thinking text dimmed above the answer.

    Haiku 4.5 and other thinking-enabled models populate
    ``ChatResponse.thinking`` when the beta header is set. The CLI
    surfaces that text as a dimmed pre-block so the operator can see
    the model's reasoning alongside the answer. Empty or whitespace-only
    thinking is a no-op — the answer block renders unchanged.
    """
    if not text or not text.strip():
        return
    print(f"\n  {GRAY}[thinking]{RESET}")
    rail = f"{GRAY}╎{RESET} "  # dashed vertical — distinct from │ main response
    for raw_line in text.splitlines():
        if not raw_line.strip():
            print()
            continue
        _render_thinking_line(rail, raw_line, width)
    print(f"  {GRAY}{'─' * max(1, width - 2)}{RESET}")


def _print_response(response, conv, turn: int = 0) -> None:
    """Render an assistant response with Swiss grid structure.

    Turn number is embedded in the top separator bar. A │ left-rail
    marks HAL's voice throughout the wrapped text. No bottom bar —
    the top bar of the next turn serves as the natural boundary.
    """
    w = min(shutil.get_terminal_size((80, 24)).columns - 4, 72)
    print(f"\n{_bar_with_turn(turn)}")
    _render_thinking(getattr(response, "thinking", ""), w)
    plain = ResponseFormatter().for_plain(response.text)
    # Swiss left-rail: │ identifies HAL's voice; ╎ used for thinking (above)
    lines = textwrap.wrap(plain, width=w - 2) or [plain]
    print()
    for line in lines:
        # Write the ANSI │ prefix atomically so chunk-splitting in
        # render_to_stdout never cuts through an escape sequence.
        sys.stdout.write(f"{GRAY}│{RESET} ")
        sys.stdout.flush()
        _STREAMER.render_to_stdout(
            line, chars_per_chunk=_stream_chunk_chars(), delay_ms=_stream_delay_ms(),
        )
        sys.stdout.write("\n")
        sys.stdout.flush()
    print()
    _maybe_print_fallback_warning(response)
    _print_response_footer(response, conv)
    _maybe_print_cache_marker(conv, response)
    _maybe_print_citation_chip(conv)
    _maybe_print_cost_heatmap(conv)


def _print_response_footer(response, conv) -> None:
    """Render the per-turn footer with coloured latency, progressive cost,
    LOCAL badge, token count, mode tag, and context % (improvements 1-6)."""
    lat = _format_latency(response.latency_ms or 0)
    badge = _provider_badge(response.provider, conv)
    cost = _format_cost(conv.cost.total_usd, response.cost_usd or 0.0)
    tok = response.usage.completion_tokens if response.usage else 0
    tok_str = f"  ·  {DIM}{tok} tok{RESET}" if tok else ""
    mode = getattr(conv, "_mode", "") or ""
    mode_str = f"  ·  {DIM}mode:{mode}{RESET}" if mode else ""
    try:
        ctx_pct = conv.status().get("context_usage_pct") or 0
    except Exception:
        ctx_pct = 0
    ctx_str = f"  {DIM}[{ctx_pct:.0f}% ctx]{RESET}" if ctx_pct > 20 else ""
    print(f"  {DIM}{response.provider}/{response.model}{badge}  ·  "
          f"{lat}{cost}{tok_str}{mode_str}{RESET}")
    if ctx_str:
        print(ctx_str)
    print()


def _maybe_print_cache_marker(conv, response=None) -> None:
    """Emit a dim cache marker; include savings estimate when available."""
    if not getattr(conv, "last_cache_hit", False):
        return
    saved = getattr(response, "cost_usd", None) if response else None
    if saved and saved > 0:
        print(f"  {DIM}[cache · saved ${saved:.4f}]{RESET}")
    else:
        print(f"  {DIM}[cache]{RESET}")


def _maybe_print_fallback_warning(response) -> None:
    """H338+H347: warn user when LimitSentinel silently swapped providers.

    Reads the typed ``response.fallback_info`` field first (canonical, set
    by sentinel.py via dataclasses.replace). Falls back to the legacy
    ``response.raw['_hal_fallback']`` mirror for backward-compat with
    callers that pre-date the typed field.
    """
    info = getattr(response, "fallback_info", None)
    if info is not None:
        requested = f"{info.requested_provider}/{info.requested_model}"
        actual = f"{info.actual_provider}/{info.actual_model}"
        print(f"  {YELLOW}⚠ {requested} unavailable — fell back to {actual}{RESET}")
        return

    # Legacy path — read the magic dict key from response.raw.
    raw = getattr(response, "raw", None) or {}
    fb = raw.get("_hal_fallback") if isinstance(raw, dict) else None
    if not fb:
        return
    requested = f"{fb.get('requested_provider', '?')}/{fb.get('requested_model', '?')}"
    actual = f"{fb.get('actual_provider', '?')}/{fb.get('actual_model', '?')}"
    print(f"  {YELLOW}⚠ {requested} unavailable — fell back to {actual}{RESET}")


def _maybe_print_citation_chip(conv) -> None:
    """Emit the memory chip when KONO entries fed into the last turn."""
    count = int(getattr(conv, "last_recall_count", 0) or 0)
    if count <= 0:
        return
    chip = format_citation_chip(count)
    if chip:
        print(f"  {DIM}{chip}{RESET}")


def _maybe_print_cost_heatmap(conv) -> None:
    """Render per-provider cost bars if ``HAL_COST_HEATMAP`` is enabled.

    Pulls ``cost_by_provider`` from ``conv.status()`` and prints one dim
    bar per provider, omitting providers with zero spend. Skipped when
    the env var is unset or there is no measurable session cost yet.
    """
    if not os.environ.get("HAL_COST_HEATMAP"):
        return
    try:
        status = conv.status()
    except Exception:
        return
    cost_by_provider = status.get("cost_by_provider", {}) or {}
    if not cost_by_provider:
        return
    lines = render_heatmap(cost_by_provider, min_display_cost=1e-9)
    if not lines:
        return
    print(f"  {DIM}cost by provider:{RESET}")
    for line in lines:
        print(f"{DIM}{line}{RESET}")
    print()


def _render_status_footer(footer: StatusFooter, providers_count: int, conv: Conversation, tool_state: dict) -> None:
    """Paint the two-line status footer between turns. No-op when disabled."""
    s = conv.status()
    footer.render(
        FooterState(
            providers=providers_count,
            model=conv.model,
            cost_usd=conv.cost.total_usd,
            tools_enabled=bool(tool_state.get("enabled")),
            context_pct=s.get("context_usage_pct", 0.0),
        )
    )


def _apply_nl_model_switch(user_input: str, registry, conv, tool_state) -> bool:
    """Detect NL model switch ('switch to deepseek') and apply. Returns True if consumed."""
    nl_model = detect_model_switch(user_input, registry)
    if not nl_model:
        return False
    conv.configure(model=nl_model)
    _maybe_rebind_tools(nl_model, tool_state)
    print(f"  {DIM}model → {nl_model}{RESET}")
    return True


def _run_chat(registry, discovered: list, model: str, mode: str,
              initial_message: str = "") -> None:
    """Interactive conversation loop. initial_message seeds the first turn."""
    if not model:
        model = registry.routing.default_model or "groq/llama-3.3-70b-versatile"

    providers = registry.list_providers()
    providers_count = len(providers)
    _print_banner(providers_count, model, discovered, mode)

    conv = Conversation(model=model, mode=mode, registry=registry)
    council = HALCouncil(registry)
    tool_state = _init_tool_state(model)
    pending_input = initial_message
    turn_count = 0

    footer = StatusFooter()
    footer.install_resize_handler()
    try:
        while True:
            _render_status_footer(footer, providers_count, conv, tool_state)
            if pending_input:
                user_input = pending_input
                pending_input = ""
            else:
                try:
                    print()
                    user_input = input(f"  {GREEN}❯{RESET}  ").strip()
                except (EOFError, KeyboardInterrupt):
                    _show_cost(conv)
                    break

            if not user_input:
                continue

            if _apply_nl_model_switch(user_input, registry, conv, tool_state):
                continue

            if user_input.startswith("/"):
                if _handle_command(user_input, conv, registry, council, tool_state):
                    break
                continue

            turn_count += 1
            _dispatch_turn(conv, tool_state, user_input, turn_count)
    finally:
        footer.uninstall_resize_handler()


def _dispatch_turn(
    conv: "Conversation",
    tool_state: dict,
    user_input: str,
    turn: int,
) -> None:
    """Route one turn: show ◈ indicator, call model, render response or error."""
    sys.stdout.write(f"  {DIM}◈{RESET}\r")
    sys.stdout.flush()
    try:
        if tool_state["enabled"] and tool_state["chat_fn"] is not None:
            _run_tool_turn(conv, user_input, tool_state, turn)
        else:
            response = conv.send(user_input)
            _clear_processing_indicator()
            _print_response(response, conv, turn)
            if tool_state.get("voice_enabled") and getattr(response, "text", "").strip():
                _speak_response(response.text)
    except Exception as e:
        _clear_processing_indicator()
        # Swiss structured error: type on first line, message indented below
        print(f"\n  {RED}⊘ {type(e).__name__}{RESET}\n  {DIM}{e}{RESET}\n")


_TOOL_SCHEMA_CATEGORY = "tool_schema_400"


# H339+H351 — tool-friendly recommendations.
#
# Originally a hardcoded dict in this module (H339). AGORA Council PR #311
# concern #3 flagged that as code-shaped knowledge that should be data — so
# H351 migrates the source-of-truth to config/providers.yaml under each
# provider's `tool_unfriendly_models` block. Operators can add entries
# without editing this module.
#
# The hardcoded fallback is kept for two reasons:
#   1. Tests can monkey-patch _TOOL_UNFRIENDLY_RECOMMENDATIONS without
#      mocking yaml file IO.
#   2. The yaml load is best-effort — if config is malformed, the CLI
#      degrades to the hardcoded fallback rather than crashing the REPL.
_TOOL_UNFRIENDLY_RECOMMENDATIONS: dict[str, str] = {
    # Smaller llama on Groq struggles with tool-call JSON output;
    # llama-4-scout (also on Groq) handles tools reliably.
    "groq/llama-3.3-70b-versatile": "groq/llama-4-scout-17b-16e-instruct",
}


def _load_tool_unfriendly_from_yaml() -> dict[str, str]:
    """Read tool_unfriendly_models blocks from config/providers.yaml.

    Returns ``{provider/model: alternative}`` flattened across all providers.
    Best-effort — returns empty dict on any parse error so the CLI keeps
    working with the hardcoded fallback. AGORA Council #312 concern #6
    flagged the previous stdlib YAML parser as brittle, so we use pyyaml
    (already a HAL runtime dep) here.
    """
    try:
        import yaml  # pyyaml — runtime dep, available in HAL's venv
    except ImportError:
        return {}
    config_path = Path(__file__).resolve().parents[2] / "config" / "providers.yaml"
    if not config_path.exists():
        return {}
    try:
        with config_path.open() as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}
    # Flat comprehension — single pass over (provider, model) pairs.
    # Total work O(N) where N = total tool_unfriendly_models entries across
    # all providers, not O(providers × models). The double `for` is yaml
    # path traversal, not nested iteration over a square space.
    return {
        f"{provider.get('name', '')}/{model}": info["alternative"]
        for provider in (data.get("providers") or [])
        if isinstance(provider, dict)
        for model, info in (provider.get("tool_unfriendly_models") or {}).items()
        if isinstance(info, dict) and info.get("alternative")
    }


def _recommended_tool_friendly_alternative(model: str | None) -> str | None:
    """Return a tool-friendly alternative model when one is known.

    Resolution order: providers.yaml (data-driven, H351) >
    _TOOL_UNFRIENDLY_RECOMMENDATIONS (hardcoded fallback, H339).
    Returns None for unknown / unmapped models.
    """
    if not model:
        return None
    yaml_recs = _load_tool_unfriendly_from_yaml()
    if model in yaml_recs:
        return yaml_recs[model]
    return _TOOL_UNFRIENDLY_RECOMMENDATIONS.get(model)


def _speak_response(text: str) -> None:
    """Speak the AI response via macos_say TTS (best-effort — never raises)."""
    try:
        from lib.hal.voice.voice import speak
        speak(text=text, provider="macos_say")
    except Exception as exc:
        print(f"  {DIM}[voice] {exc}{RESET}")


def _init_tool_state(model: Optional[str] = None) -> dict:
    """Build the REPL's tool-loop state dict, auto-detecting tool auth.

    When ``model`` is provided, probe for an auth path immediately so
    tools start enabled if credentials are already configured. The
    circuit breaker auto-disables tools after two consecutive schema-
    rejection 400s (Issue #200). ``/tools on`` resets it.
    """
    from lib.hal.tool_circuit_breaker import CircuitBreaker
    from lib.hal.tool_cli_integration import detect_tool_chat_fn
    chat_fn, auth = None, None
    if model:
        try:
            chat_fn, auth = detect_tool_chat_fn(model)
        except Exception:
            pass
    return {
        "enabled": chat_fn is not None,
        "chat_fn": chat_fn,
        "auth": auth,
        "breaker": CircuitBreaker(threshold=2),
        "voice_enabled": False,
    }


def _run_tool_turn(conv: Conversation, user_input: str, tool_state: dict, turn: int = 0) -> None:
    """Route one turn through the tool loop; stream tool calls to stdout.

    Issue #200: a malformed tool schema (e.g. OpenAI-compat provider
    rejecting Anthropic-shape tools) used to loop forever because the REPL
    had no circuit breaker. Now, two consecutive tool-schema 400s auto-
    disable tools and the REPL falls back to plain-text mode with a clear
    message. The user can re-enable with ``/tools on`` once the issue is
    fixed — that path calls ``breaker.reset()`` so the breaker gives them
    a fresh chance.
    """
    from lib.hal.tool_cli_integration import build_builtin_tool_stack
    from lib.hal.tool_circuit_breaker import is_tool_schema_400
    from lib.hal.tool_render import send_with_tools_to_cli

    tools, dispatcher = build_builtin_tool_stack()
    breaker = tool_state["breaker"]
    result = None
    try:
        result = send_with_tools_to_cli(
            conv, user_input,
            chat_fn=tool_state["chat_fn"],
            tools=tools,
            dispatcher=dispatcher,
        )
    except Exception as exc:
        if is_tool_schema_400(exc):
            if breaker.record_failure(_TOOL_SCHEMA_CATEGORY):
                _auto_disable_tools(tool_state, exc, model=conv.model)
                return
            # Below threshold — re-raise so outer loop shows the error.
            raise
        if _is_transport_failure(exc):
            # Server is definitively unreachable — disable immediately rather
            # than letting every subsequent message fail with the same error.
            _auto_disable_tools_transport(tool_state, exc)
            return
        raise
    else:
        breaker.record_success(_TOOL_SCHEMA_CATEGORY)
    tool_count = len(result.tool_calls) if result else 0
    _clear_processing_indicator()
    _stream_plain_tool_response(conv._last_response_text or "", tool_count, turn=turn)


def _stream_plain_tool_response(text: str, tool_count: int = 0, turn: int = 0) -> None:
    """Stream a plain-text tool-loop response, with turn bar and │ rail."""
    w = min(shutil.get_terminal_size((80, 24)).columns - 4, 72)
    print(f"\n{_bar_with_turn(turn)}")
    plain = ResponseFormatter().for_plain(text)
    lines = textwrap.wrap(plain, width=w - 2) or [plain]
    print()
    for line in lines:
        sys.stdout.write(f"{CYAN}│{RESET} ")
        sys.stdout.flush()
        _STREAMER.render_to_stdout(
            line, chars_per_chunk=_stream_chunk_chars(), delay_ms=_stream_delay_ms(),
        )
        sys.stdout.write("\n")
        sys.stdout.flush()
    print()
    if tool_count:
        print(f"  {DIM}{tool_count} tool{'s' if tool_count != 1 else ''}{RESET}")
    print()


def _auto_disable_tools(tool_state: dict, exc: Exception, model: str | None = None) -> None:
    """Flip the tool state to off and explain why — Issue #200 circuit break.

    Prints a multi-line message so the user understands the REPL just made
    a decision for them: tools off, schema rejection repeated, manual
    ``/tools on`` to retry. The underlying error is included (truncated)
    so they can file the real bug.

    H339: when ``model`` is in the known tool-unfriendly registry, append
    a one-line recommendation pointing to a tool-capable alternative on
    the same provider. Silent for unmapped models so we don't false-suggest.
    """
    tool_state["enabled"] = False
    tool_state["chat_fn"] = None
    tool_state["auth"] = None
    provider = getattr(exc, "provider", "provider")
    body = str(getattr(exc, "body", "") or exc)[:160]
    print(
        f"\n  {YELLOW}⚠  tool mode auto-disabled after 2 consecutive tool-schema rejections.{RESET}\n"
        f"  {DIM}Cause: {provider} rejected the tool schema — {body}{RESET}\n"
        f"  {DIM}Fix the schema issue, then `/tools on` to re-enable.{RESET}\n",
    )
    suggestion = _recommended_tool_friendly_alternative(model)
    if suggestion:
        print(
            f"  {DIM}Tip: {model} struggles with tool-call JSON. "
            f"Try `/model {suggestion}` then `/tools on`.{RESET}\n",
        )


def _auto_disable_tools_transport(tool_state: dict, exc: Exception) -> None:
    """Disable tools immediately on a transport failure (server unreachable).

    Unlike the schema-400 path (two-strike circuit breaker), a ConnectionRefused
    or URLError is definitive — the local server is not running. Disabling after
    the first failure prevents every subsequent message from hitting the same
    dead endpoint until the session ends.
    """
    tool_state["enabled"] = False
    tool_state["chat_fn"] = None
    tool_state["auth"] = None
    reason = getattr(exc, "reason", None)
    err_str = str(reason or exc)[:120]
    print(
        f"\n  {YELLOW}⚠  tool mode auto-disabled: provider unreachable.{RESET}\n"
        f"  {DIM}Cause: {err_str}{RESET}\n"
        f"  {DIM}Start the local server or switch models, then `/tools on` to re-enable.{RESET}\n",
    )


def _is_transport_failure(exc: Exception) -> bool:
    """Return True when exc represents a network-level connection failure.

    Covers urllib.error.URLError (which wraps ConnectionRefusedError on ECONNREFUSED)
    and bare ConnectionRefusedError for callers that don't go through urllib.
    A transport failure means the provider is definitively unreachable — distinct
    from a transient timeout, though we treat both the same here since a retry
    without fixing the root cause (server not running) will fail identically.
    """
    if isinstance(exc, (ConnectionRefusedError, urllib.error.URLError)):
        return True
    reason = getattr(exc, "reason", None)
    if isinstance(reason, (ConnectionRefusedError, OSError)):
        return True
    return False


def _ensure_local_server(provider_name: str, base_url: str, start_hint: str) -> bool:
    """Auto-start a local server if possible; print status. Returns True if ready."""
    from lib.hal.tool_cli_integration import auto_start_local_server
    print(f"  {DIM}Starting {provider_name} server...{RESET} ", end="", flush=True)
    if auto_start_local_server(provider_name, base_url):
        print(f"{GREEN}ready{RESET}")
        return True
    print(f"{YELLOW}timed out{RESET}")
    print(f"  {DIM}Try manually: {start_hint}{RESET}")
    return False


def _maybe_rebind_tools(new_model: str, tool_state: Optional[dict]) -> None:
    """Rebind tool_state's chat_fn after a /model switch.

    The chat_fn returned by detect_tool_chat_fn captures the provider URL at
    creation time (e.g. mlx_vlm -> localhost:8080). Leaving it stale after a
    model switch causes every subsequent message to target the OLD provider's
    endpoint — a ConnectionRefused that persists until the session ends.

    When tools are enabled and the new model has a valid auth path, the fn is
    rebound silently. When no auth path is found, tools are disabled and the
    user is told why.
    """
    if tool_state is None or not tool_state["enabled"]:
        return
    from lib.hal.tool_cli_integration import detect_tool_chat_fn, get_local_provider_info, probe_local_server
    local = get_local_provider_info(new_model)
    if local is not None:
        provider_name, base_url, start_hint = local
        if not probe_local_server(base_url) and not _ensure_local_server(provider_name, base_url, start_hint):
            tool_state["enabled"] = False
            tool_state["chat_fn"] = None
            tool_state["auth"] = None
            return
    chat_fn, auth = detect_tool_chat_fn(new_model)
    tool_state["chat_fn"] = chat_fn
    tool_state["auth"] = auth
    if chat_fn is None:
        tool_state["enabled"] = False
        print(f"  {YELLOW}tool mode disabled — no auth path for {new_model}{RESET}")
    else:
        print(f"  {DIM}tool mode rebound ({auth}){RESET}")


def _cmd_info(command: str, arg: str, conv: Conversation, registry) -> None:
    """Handle read-only info commands: /cost /providers /models /history."""
    if command == "/cost":
        s = conv.status()
        print(f"{DIM}Turns: {s['turns']} | Cost: ${s['total_cost_usd']:.6f} | "
              f"Tokens: {s['total_tokens']}{RESET}")
        for p, c in s["cost_by_provider"].items():
            print(f"  {DIM}{p}: ${c:.6f}{RESET}")
    elif command == "/providers":
        for p in registry.list_providers():
            status = f"{GREEN}ok{RESET}" if p.healthy else f"{RED}down{RESET}"
            models = ", ".join(p.models[:3])
            print(f"  {p.name} ({p.type.value}) [{status}] {DIM}{models}{RESET}")
    elif command == "/models":
        from lib.hal.models_command import handle_models_command
        print(handle_models_command(registry, arg))
    elif command == "/history":
        for msg in conv.history:
            clr = GREEN if msg.role == "user" else CYAN
            print(f"  {clr}{msg.role}>{RESET} {msg.content[:100]}")


_INFO_COMMANDS = frozenset({"/cost", "/providers", "/models", "/history"})


def _handle_command(
    cmd: str,
    conv: Conversation,
    registry,
    council: HALCouncil,
    tool_state: Optional[dict] = None,
) -> bool:
    """Handle slash command. Returns True to exit."""
    parts = cmd.split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if command in ("/quit", "/exit"):
        _show_cost(conv)
        return True
    elif command == "/model":
        if arg:
            resolved = fuzzy_resolve_model(arg, registry) or arg
            conv.configure(model=resolved)
            suffix = f"  {DIM}(matched from {arg!r}){RESET}" if resolved != arg else ""
            print(f"  {DIM}model → {resolved}{RESET}{suffix}")
            _maybe_rebind_tools(resolved, tool_state)
        else:
            print(f"  {DIM}model → {conv.model}{RESET}")
            print(f"  {DIM}Use: /model <name>  or  /settings to browse{RESET}")
    elif command == "/mode":
        if arg:
            if "/" in arg:
                # provider/model string — same as /model
                conv.configure(model=arg)
                print(f"  {DIM}model → {arg}{RESET}")
                _maybe_rebind_tools(arg, tool_state)
            else:
                conv.configure(mode=arg)
                print(f"  {DIM}mode → {arg}{RESET}")
        else:
            print(f"  {DIM}Use: /mode <name> or /mode provider/model{RESET}")
    elif command in _INFO_COMMANDS:
        _cmd_info(command, arg, conv, registry)
    elif command == "/council":
        _cmd_council(council, registry, arg)
    elif command == "/clear":
        conv.configure(clear=True)
        print(f"  {DIM}cleared{RESET}")
    elif command == "/tools":
        if tool_state is None:
            print(f"{YELLOW}Tool mode not available (state missing){RESET}")
        else:
            _handle_tools_command(arg, conv, tool_state)
    elif command == "/voice":
        if tool_state is None:
            print(f"{YELLOW}Voice mode not available (state missing){RESET}")
        else:
            _handle_voice_command(arg, tool_state)
    elif command == "/settings":
        from lib.hal.settings_tui import open_settings
        selected = open_settings(registry, conv)
        if selected:
            conv.configure(model=selected)
            _maybe_rebind_tools(selected, tool_state)
            print(f"  {GREEN}✓{RESET} {DIM}model → {selected}{RESET}")
    elif command == "/dojo":
        _cmd_dojo()
    elif command in ("/help", "/?"):
        _print_help()
    else:
        hint = _did_you_mean(command)
        suffix = f"  Did you mean {hint}?" if hint else "  (try /help)"
        print(f"{YELLOW}Unknown: {command}{RESET}{suffix}")
    return False


# The child pins the name `lib` to GRIP's folder before importing. GRIP's lib is
# a namespace package (no __init__.py), and a regular `lib` package anywhere on
# sys.path, HAL's included, wins that name over a namespace one whatever the
# order, so putting GRIP first on PYTHONPATH is not enough on its own.
_DOJO_LAUNCH = (
    "import os, sys, types, pathlib; "
    "lib = types.ModuleType('lib'); "
    "lib.__path__ = [os.path.join(os.environ['CLAUDE_DIR'], 'lib')]; "
    "sys.modules['lib'] = lib; "
    "from lib.dojo_shell import open_dojo; "
    "_c = open_dojo(); "
    "pathlib.Path(os.environ['GRIP_DOJO_RESULT']).write_text(_c, encoding='utf-8') "
    "if isinstance(_c, str) and _c.strip() else None"
)
_GRIP_PYTHONS = (Path("venv") / "bin" / "python", Path("venv") / "Scripts" / "python.exe")
# A child ended by Ctrl-C: killed by SIGINT (-2), or exited 130 as shells report it.
_INTERRUPTED = (-signal.SIGINT, 128 + signal.SIGINT)


def _cmd_dojo() -> None:
    """Open the GRIP dojo shell (three runnable lessons) as a child process.

    The shell lives in the GRIP tree (``$CLAUDE_DIR/lib/dojo_shell.py``). GRIP
    and HAL both name their top-level package ``lib``, so it cannot be imported
    into this process: HAL's ``lib`` already owns the name, and ``lib.dojo_shell``
    would be looked up in HAL's folder, raising ModuleNotFoundError every time.
    The child runs with the GRIP tree as its working directory and PYTHONPATH,
    under GRIP's own interpreter when the tree has one, and it inherits this
    terminal, so curses works as normal. There is no timeout, because the dojo is
    interactive; Ctrl-C ends it and returns to the REPL, and the terminal settings
    are put back afterwards whatever happened to the child. A missing GRIP tree,
    or a shell that fails, is a one-line message; the REPL never crashes and its
    sys.path is left alone.

    The chosen ``try_it`` command crosses back to the PARENT via a result file
    (``GRIP_DOJO_RESULT``), not by the child printing: the child may return before
    the tty is back in cooked mode, so anything it prints can be garbled or
    dropped. The parent reads the file and prints the command only AFTER its own
    terminal restore, so the "run this …" line always lands in a cooked terminal.
    """
    claude_dir = Path(os.environ.get("CLAUDE_DIR") or Path.home() / ".claude").expanduser()
    if not (claude_dir / "lib" / "dojo_shell.py").is_file():
        print(f"  {YELLOW}dojo unavailable: no GRIP dojo under {claude_dir}{RESET}")
        return
    python = _grip_python(claude_dir)
    result_path = ""
    try:
        try:
            fd, result_path = tempfile.mkstemp(suffix=".txt")
        except OSError as exc:
            print(f"  {YELLOW}dojo unavailable: {exc}{RESET}")
            return
        os.close(fd)
        env = _dojo_env(claude_dir, python)
        env["GRIP_DOJO_RESULT"] = result_path
        try:
            done = _run_keeping_terminal([python, "-c", _DOJO_LAUNCH], cwd=claude_dir, env=env)
        except OSError as exc:
            print(f"  {YELLOW}dojo unavailable: {exc}{RESET}")
            return
        except KeyboardInterrupt:
            print(f"  {YELLOW}dojo interrupted{RESET}")
            return
        if done.returncode in _INTERRUPTED:
            print(f"  {YELLOW}dojo interrupted{RESET}")
            return
        if done.returncode:
            print(f"  {YELLOW}dojo exited with code {done.returncode}{RESET}")
            return
        _surface_dojo_choice(result_path)
    finally:
        if result_path:
            with contextlib.suppress(OSError):
                Path(result_path).unlink(missing_ok=True)


def _surface_dojo_choice(result_path: str) -> None:
    """Print the chosen command, after the terminal is restored and cooked.

    Reads size-capped and decode-safe, so a buggy child can never crash the
    REPL or spike its memory. Validates the payload before printing: it must be
    a single non-empty line of printable characters (no control or format bytes
    that would hijack the prompt, no newline/CR, no surrounding blanks).
    Anything else is dropped as a dead instruction rather than printed.
    """
    chosen = _read_dojo_choice(result_path)
    if chosen:
        print(f"  {GREEN}✓{RESET} {DIM}run this in your GRIP session:{RESET} "
              f"{BOLD}{chosen}{RESET}")


def _read_dojo_choice(result_path: str) -> str:
    """Read + sanitise the child's chosen command, or '' when unsafe/absent."""
    try:
        stat = os.stat(result_path)
        if stat.st_size > 4096:
            return ""
        data = Path(result_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    chosen = data.strip()
    if not chosen or "\n" in chosen or "\r" in chosen:
        return ""
    # Reject any control (Cc) or format (Cf) character — C0 (ESC/NUL), C1 CSI,
    # DEL, and bidi/zero-width (U+202E, U+200B) all fall in those two Unicode
    # categories, and any of them could inject into the restored prompt. This
    # matches the docstring's "printable range" claim, which a bare
    # ``ord < 32`` did not.
    if any(unicodedata.category(ch) in ("Cc", "Cf") for ch in chosen):
        return ""
    return chosen


def _run_keeping_terminal(argv: list, **kwargs) -> subprocess.CompletedProcess:
    """subprocess.run, then put the terminal settings back as they were.

    On Ctrl-C, run() kills and reaps the child at once, which can leave curses'
    raw mode behind if the child had not yet restored the terminal itself.
    """
    saved = _terminal_settings()
    try:
        return subprocess.run(argv, check=False, **kwargs)
    finally:
        if saved is not None:
            with contextlib.suppress(Exception):
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, saved)


def _terminal_settings():
    """This terminal's settings, or None where there is no terminal to restore."""
    if termios is None:                  # Windows has no termios
        return None
    try:
        return termios.tcgetattr(sys.stdin.fileno())
    except Exception:  # noqa: BLE001 -- not a terminal (a pipe, a test run): nothing to restore
        return None


def _grip_python(claude_dir: Path) -> str:
    """GRIP's own interpreter when the tree has a runnable one, else this one."""
    for candidate in (claude_dir / p for p in _GRIP_PYTHONS):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return sys.executable


def _dojo_env(claude_dir: Path, python: str) -> dict:
    """This environment with the GRIP tree first on PYTHONPATH, keeping the rest.

    GRIP's own interpreter finds its packages through its own prefix, so a
    PYTHONHOME or VIRTUAL_ENV set for HAL would point it at HAL's instead; both
    are dropped for it, and kept when this interpreter runs the child.
    """
    inherited = os.environ.get("PYTHONPATH", "")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (str(claude_dir), inherited) if p),
           "CLAUDE_DIR": str(claude_dir)}
    if python != sys.executable:
        env.pop("PYTHONHOME", None)
        env.pop("VIRTUAL_ENV", None)
    return env


def _handle_tools_command(arg: str, conv: Conversation, tool_state: dict) -> None:
    """Handle /tools with no arg (status), ``on`` (enable), or ``off`` (disable)."""
    from lib.hal.tool_cli_integration import detect_tool_chat_fn, get_local_provider_info, probe_local_server

    sub = arg.lower().strip()
    if not sub:
        status = f"{GREEN}on{RESET} ({tool_state['auth']})" if tool_state["enabled"] else f"{DIM}off{RESET}"
        print(f"  tool mode: {status}")
        print(f"  {DIM}Use: /tools on | /tools off{RESET}")
        return

    if sub == "off":
        tool_state["enabled"] = False
        tool_state["chat_fn"] = None
        tool_state["auth"] = None
        print(f"{DIM}tool mode disabled{RESET}")
        return

    if sub == "on":
        local = get_local_provider_info(conv.model)
        if local is not None:
            provider_name, base_url, start_hint = local
            if not probe_local_server(base_url) and not _ensure_local_server(provider_name, base_url, start_hint):
                return
        chat_fn, auth = detect_tool_chat_fn(conv.model)
        if chat_fn is None:
            print(
                f"{YELLOW}No auth path found. Configure a CCH subscription "
                f"(CLAUDE_CODE_OAUTH_TOKEN env or keychain `grip-claude-oauth`) "
                f"and try again.{RESET}",
            )
            return
        tool_state["enabled"] = True
        tool_state["chat_fn"] = chat_fn
        tool_state["auth"] = auth
        # Reset the Issue #200 circuit breaker: user explicitly re-enabling
        # tools deserves a fresh chance to succeed, regardless of prior trips.
        breaker = tool_state.get("breaker")
        if breaker is not None:
            breaker.reset()
        print(f"{GREEN}tool mode enabled{RESET} {DIM}({auth}){RESET}")
        return

    print(f"{YELLOW}Usage: /tools [on|off]{RESET}")


def _handle_voice_command(arg: str, tool_state: dict) -> None:
    """Handle /voice with no arg or ``on`` (enable), ``off`` (disable)."""
    sub = arg.lower().strip()
    if sub == "off":
        tool_state["voice_enabled"] = False
        print(f"{DIM}voice output disabled{RESET}")
    elif sub in ("", "on"):
        tool_state["voice_enabled"] = True
        print(f"{GREEN}voice output enabled{RESET} {DIM}(Jamie (Premium) · macos_say){RESET}")
    else:
        status = "on" if tool_state.get("voice_enabled") else "off"
        print(f"  voice: {status}  {DIM}(use /voice on | /voice off){RESET}")


def _council_progress(completed: int, total: int, model: str, phase: str) -> None:
    """Real-time progress bar for council deliberation."""
    pct = completed / total if total else 0
    bar_len = 30
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)
    phase_label = {"gut_feel": "querying", "query": "querying", "debate": "debating"}.get(phase, phase)
    status = f"  {DIM}{phase_label}: [{bar}] {completed}/{total} — {model.split('/')[-1]}{RESET}"
    print(f"\r{status}", end="", flush=True)
    if completed == total:
        print()  # newline after final update


def _cmd_council(council: HALCouncil, registry, query: str) -> None:
    """Query multiple LLMs and merge results with real-time progress."""
    if not query:
        print(f"{DIM}Usage: /council <question>{RESET}")
        return
    models = _get_council_models(registry)
    if len(models) < 2:
        print(f"{YELLOW}Need at least 2 models for council. Available: {models}{RESET}")
        return
    print(f"{DIM}Querying {len(models)} models: {', '.join(models)}...{RESET}")
    result = council.deliberate(query, models, on_progress=_council_progress)
    print(f"\n{BOLD}{result.merged_text}{RESET}")
    print(f"{DIM}Agreement: {result.agreement_score:.0%} | "
          f"Models: {len(result.votes)}/{len(models)} | "
          f"Cost: ${result.total_cost_usd:.6f}{RESET}")
    for vote in result.votes:
        snippet = vote.response.text[:120].replace("\n", " ")
        print(f"  {DIM}{vote.model}: {snippet}...{RESET}")


def _get_council_models(registry) -> list[str]:
    """Pick available models for council. Single pass + lazy Ollama lookup."""
    models: list[str] = []
    ollama_fetched = None
    for p in registry.list_providers():
        if p.models:
            models.append(f"{p.name}/{p.models[0]}")
        elif p.name == "ollama":
            if ollama_fetched is None:
                ollama_fetched = _get_ollama_models_quick()
            if ollama_fetched:
                models.append(f"ollama/{ollama_fetched[0]}")
    return models[:5]


def _get_ollama_models_quick() -> list[str]:
    """Quick Ollama model list from /api/tags."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
        return [m["name"] for m in data.get("models", [])]
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return []


def _show_status(registry, discovered: list) -> None:
    """Non-interactive status display."""
    print(f"\n{BOLD}{RED}HAL{RESET} — Status\n")
    for p in registry.list_providers():
        status = f"{GREEN}healthy{RESET}" if p.healthy else f"{RED}down{RESET}"
        models = ", ".join(p.models[:5]) if p.models else "auto-discover"
        print(f"  {BOLD}{p.name}{RESET} ({p.type.value}) [{status}]")
        print(f"    {DIM}{p.base_url}{RESET}")
        print(f"    {DIM}Models: {models}{RESET}")

    if discovered:
        print(f"\n  {GREEN}Discovered: {', '.join(discovered)}{RESET}")

    ollama_models = _get_ollama_models_quick()
    if ollama_models:
        print(f"  {DIM}Ollama models: {', '.join(ollama_models)}{RESET}")

    chain = registry.routing.fallback_chain
    if chain:
        print(f"\n  {DIM}Fallback chain: {' -> '.join(chain)}{RESET}")
    print()


def _show_tools() -> None:
    """List every tool available to HAL agents via the unified adapter.

    Reads ``lib/hal/tool_adapter._BUILTIN_SPECS`` — the single source of truth
    for tool name, description, required input fields, and whether the tool is
    safe to run in parallel with other calls. Shipped separately from
    ``--status`` because tools and providers are orthogonal capability axes.
    """
    from lib.hal.tool_adapter import _BUILTIN_SPECS

    print(f"\n{BOLD}{RED}HAL{RESET} — Tools\n")
    print(f"  {DIM}{len(_BUILTIN_SPECS)} built-in tools registered via "
          f"ToolDispatcher + ToolProtocol adapter.{RESET}\n")

    # Columns: name, concurrency flag, required params, description
    name_w = max(len(s.name) for s in _BUILTIN_SPECS)
    for spec in sorted(_BUILTIN_SPECS, key=lambda s: (not s.is_concurrent_safe, s.name)):
        safe_tag = (
            f"{GREEN}parallel{RESET}" if spec.is_concurrent_safe
            else f"{YELLOW}serial  {RESET}"
        )
        required = ", ".join(spec.input_schema.get("required", [])) or "-"
        print(f"  {BOLD}{spec.name:<{name_w}}{RESET}  {safe_tag}  "
              f"{DIM}[{required}]{RESET}")
        print(f"    {DIM}{spec.description}{RESET}")
    print()


def _handle_daemon(action: str, port: int) -> None:
    """Handle --daemon start|stop|status."""
    from lib.hal.daemon import HALDaemon

    daemon = HALDaemon(port=port)
    if action == "start":
        print(f"{BOLD}{RED}HAL{RESET} — Starting daemon on port {port}...")
        daemon.start()
    elif action == "stop":
        if daemon.stop():
            print(f"{GREEN}HAL daemon stopped.{RESET}")
        else:
            print(f"{YELLOW}No running daemon found.{RESET}")
    elif action == "status":
        info = HALDaemon.status()
        if info.get("running"):
            print(f"{GREEN}HAL daemon running{RESET} (PID {info['pid']})")
        else:
            print(f"{DIM}HAL daemon not running.{RESET}")
    else:
        print(f"{YELLOW}Unknown daemon action: {action}. Use start|stop|status{RESET}")


def _start_server(registry, port: int) -> None:
    """Start the HAL HTTP server."""
    from lib.hal.server import HALServer
    server = HALServer(port=port, registry=registry)
    server.start()


def _show_cost(conv: Conversation) -> None:
    """Swiss-grid session summary table on exit."""
    s = conv.status()
    total = conv.cost.total_usd
    bar = _bar()
    print(f"\n{bar}")
    print(f"  {DIM}{'turns':<10}{'tokens':>12}{'cost':>14}{RESET}")
    print(f"  {GRAY}{'─' * 36}{RESET}")
    print(f"  {s['turns']:<10}{s['total_tokens']:>12,}  ${total:>9.4f}")
    breakdown = {p: c for p, c in (s.get("cost_by_provider") or {}).items() if c > 0}
    if breakdown:
        print(f"  {GRAY}{'─' * 36}{RESET}")
        for p, c in breakdown.items():
            print(f"  {DIM}{p:<10}{'':>12}  ${c:>9.4f}{RESET}")
    print(f"\n{bar}\n")


def _show_help() -> None:
    """Print usage and exit cleanly — no registry init, no warmup."""
    print(f"{BOLD}{RED}HAL{RESET} — Knowledge Engine")
    print(f"{DIM}Any model. Any provider. Zero limits.{RESET}\n")
    print("Usage:")
    print("  hal                            Interactive chat")
    print("  hal --model <provider/model>   Use a specific model")
    print("  hal --mode  <mode>             Domain mode (legal, code, ...)")
    print("  hal --status                   Show providers and health")
    print("  hal --models [filter]          List models with strengths and tier")
    print("  hal --subscribe [provider]     Activate subscription (anthropic|kimi-coding|qwen)")
    print("  hal --happi-api                Headless HAPPI/1.0 runtime (stdin envelope -> stdout NDJSON)")
    print("  hal --tools                    List built-in tools + concurrency flags")
    print("  hal --serve [--port N]         Start HTTP server (default: 3850)")
    print("  hal --daemon start|stop|status Manage background daemon")
    print("  hal --help                     Show this message")


def _consume_arg(args: dict, argv: list, i: int) -> int:
    """Apply one CLI argument to the args dict and return the next index.

    Extracted from _parse_args so the dispatch table stays readable and
    the parent loop is a 5-line orchestrator.
    """
    cur = argv[i]
    nxt = argv[i + 1] if i + 1 < len(argv) else None
    nxt_is_value = nxt is not None and not nxt.startswith("--")

    if cur in ("--help", "-h"):
        args["help"] = True
        return i + 1
    if cur == "--serve":
        args["serve"] = True
        return i + 1
    if cur == "--status":
        args["status"] = True
        return i + 1
    if cur == "--tools":
        args["tools"] = True
        return i + 1
    if cur == "--subscribe":
        args["subscribe"] = nxt if nxt_is_value else "*"
        return i + 2 if nxt_is_value else i + 1
    if cur == "--happi-api":
        args["happi_api"] = True
        return i + 1
    if cur == "--models":
        args["models"] = nxt if nxt_is_value else "*"
        return i + 2 if nxt_is_value else i + 1
    if cur == "--daemon":
        args["daemon"] = nxt if nxt_is_value else "start"
        return i + 2 if nxt_is_value else i + 1
    # Long flags that always require a value.
    for key, flag in (("model", "--model"), ("mode", "--mode")):
        if cur == flag and nxt_is_value:
            args[key] = nxt
            return i + 2
    if cur == "--port" and nxt_is_value:
        args["port"] = int(nxt)
        return i + 2
    if not cur.startswith("-"):
        args["initial_message"] = cur
        return i + 1
    return i + 1


def _parse_args() -> dict:
    """Parse command-line arguments into a dict."""
    args = {
        "model": "", "mode": "", "serve": False,
        "status": False, "tools": False, "port": 3850,
        "daemon": "", "help": False, "models": "",
        "subscribe": "", "happi_api": False, "initial_message": "",
    }
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        i = _consume_arg(args, argv, i)
    return args


if __name__ == "__main__":
    main()
