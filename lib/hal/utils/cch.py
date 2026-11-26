"""CCH — Claude Code billing-header signer.

Generates the `x-anthropic-billing-header: cc_version=...; cc_entrypoint=cli;
cch=HASH` header that the Claude Code CLI sends to the Anthropic Messages API
to bill requests against a CC subscription instead of per-token API credits.

Reverse-engineering source:
    https://a10k.co/b/reverse-engineering-claude-code-cch.html
    https://github.com/paoloanzn/free-code/pull/9

HAL uses this so a CC subscription can be treated as a first-class HAL
"subscription" provider alongside per-token Anthropic API keys, Z.ai GLM
coding subscriptions, Kimi K2, Qwen, etc. See lib/hal/subscription.py for
the cross-provider SubscriptionAuth abstraction.

Tracking issue: https://github.com/CodeTonight-SA/HAL/issues/21

Context management strategies
------------------------------
Two strategies can be combined inside the ``context_management`` body field:

1. ``clear_thinking_20251015`` (always emitted when thinking is active):
   Prunes old thinking blocks, keeping either all or just the last turn.

2. ``clear_tool_uses_20250919`` (opt-in, reduces billable input tokens):
   Ported from TF>>'s ``apiMicrocompact.ts`` — upstream it is ant-only and
   gated on ``USE_API_CLEAR_TOOL_RESULTS=1`` / ``USE_API_CLEAR_TOOL_USES=1``.
   HAL exposes both modes as explicit Python parameters, not env-var gates:

   * ``clear_tool_results=True``: clears results from read-type tools
     (Bash, Glob, Grep, FileRead, WebFetch, WebSearch).  Uses
     ``clear_tool_inputs`` with those tool names.

   * ``clear_tool_uses=True``: clears tool_use blocks, excluding write-type
     tools (FileEdit, FileWrite, NotebookEdit).  Uses ``exclude_tools`` with
     those names.

   Both accept ``max_input_tokens`` (trigger threshold, default 180 000) and
   ``target_input_tokens`` (how aggressively to prune, default 40 000).

Usage:
    from lib.hal.utils.cch import CCHSigner
    from lib.hal.types import ChatResponse

    signer = CCHSigner()  # cc_version derived from the installed CLI
    response: ChatResponse = signer.call(
        prompt="Explain SOLID principles",
        oauth_token=os.environ["CLAUDE_CODE_OAUTH_TOKEN"],
        model="claude-opus-4-7",
    )
    print(response.text)

    # Opt-in to tool-result pruning for long sessions:
    headers, body = signer.build_request(
        prompt="hello", oauth_token=token, model="claude-opus-4-7",
        clear_tool_results=True,
    )
"""
import functools
import hashlib
import json
import logging
import os
import re
import socket
import subprocess
import time
import tomllib
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

from lib.hal.types import ChatMessage, ChatResponse, TokenUsage

_API_URL = "https://api.anthropic.com/v1/messages"
# Anthropic gates newer models on the client version advertised in the billing
# header, names the floor in the rejection, and the caller then silently fails
# over to another provider. A hardcoded literal therefore stops serving each new
# model generation without announcing it, so the version is DERIVED from the
# installed CLI and the fallback below is only for when that fails (a launchd
# job, a host without the CLI). Each row is Anthropic's floor, quoted:
#   claude-fable-5-1  "version 2.1.251 or newer is required"
#   claude-opus-5-5   "Claude Code 2.1.251 does not support this model; version
#                     2.1.280 or newer is required" (probed through this signer
#                     2026-09-24; 2.1.281 passed the same gate)
#   claude-sonnet-5-5 and claude-haiku-5-5 are NOT gated down to 2.1.251 at
#                     least: probed through this signer on 2026-10-12 with
#                     cc_version pinned to 2.1.251, both served and answered as
#                     themselves while Opus 5.5 was refused in the same run, so
#                     neither has a row.
#   claude-fable-5-2  "Claude Code 2.1.280 does not support this model; version
#                     2.1.290 or newer is required" (probed through this signer
#                     2026-11-11 with cc_version pinned to 2.1.280 and to
#                     2.1.251, while Opus 5.5 and Fable 5.1 kept their recorded
#                     floors in the same runs; 2.1.319 passed the same gate)
# A newly gated model is one row here: the fallback is computed as the highest
# floor, so it cannot be left behind the way the 2.1.251 literal was.
_MODEL_MIN_CC_VERSION = {
    "claude-fable-5-1": "2.1.251",
    "claude-opus-5-5": "2.1.280",
    "claude-fable-5-2": "2.1.290",
}
_FALLBACK_CC_VERSION = max(
    _MODEL_MIN_CC_VERSION.values(), key=lambda v: tuple(int(p) for p in v.split(".")))
_CC_VERSION_RE = re.compile(r"(\d+\.\d+\.\d+)")
_CC_VERSION_TIMEOUT_S = 5

_LOG = logging.getLogger(__name__)
_DEFAULT_SALT = "59cf53e54c78"
_DEFAULT_CCH_SEED = 0x6E52736AC806831E
_CCH_MASK = 0xFFFFF
_REQUEST_TIMEOUT_S = 120
_ANTHROPIC_API_VERSION = "2023-06-01"


def _version_probe_env() -> dict:
    """Environment for the ``claude --version`` probe, minus credential vars.

    Defence in depth, NOT a defeat of a PATH attacker: anyone able to place a
    ``claude`` earlier on PATH already runs code as this user. This only keeps
    the credentials out of a probe that has no use for them.
    """
    drop = {"CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_OAUTH_TOKEN",
            "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"}
    return {k: v for k, v in os.environ.items() if k not in drop}


@functools.lru_cache(maxsize=1)
def detect_cc_version(timeout_s: float = _CC_VERSION_TIMEOUT_S) -> str:
    """Return the Claude Code version to advertise in the billing header.

    Resolution order: ``HAL_CC_VERSION`` (explicit pin, and the test seam) ->
    ``claude --version`` on PATH -> :data:`_FALLBACK_CC_VERSION`.

    Never raises. A missing binary, a non-zero exit, a timeout or unparseable
    output all fall through to the fallback, because a signing call must not
    fail on version detection. Cached: call ``detect_cc_version.cache_clear()``
    after changing ``HAL_CC_VERSION`` in-process.
    """
    pinned = os.environ.get("HAL_CC_VERSION", "").strip()
    if pinned:
        return pinned
    try:
        completed = subprocess.run(["claude", "--version"], capture_output=True,
                                   text=True, timeout=timeout_s,
                                   env=_version_probe_env())
    except (OSError, subprocess.SubprocessError) as exc:
        _LOG.warning("cc_version detection failed (%s: %s); using fallback %s",
                     type(exc).__name__, exc, _FALLBACK_CC_VERSION)
        return _FALLBACK_CC_VERSION
    if completed.returncode != 0:
        _LOG.warning("cc_version detection exited %s; using fallback %s",
                     completed.returncode, _FALLBACK_CC_VERSION)
        return _FALLBACK_CC_VERSION
    match = _CC_VERSION_RE.search(completed.stdout or "")
    if match is None:
        _LOG.warning("cc_version detection output carried no version; using fallback %s",
                     _FALLBACK_CC_VERSION)
        return _FALLBACK_CC_VERSION
    return match.group(1)

# TF>> liberation (commit 77ed9bc): cli-internal-2026-02-09 was gated on
# USER_TYPE=ant; the gate was removed so it now defaults on for all callers.
# HAL mirrors the same pattern. Set CLAUDE_CODE_CLI_INTERNAL_BETA="" to disable.
_CLI_INTERNAL_BETA = os.environ.get(
    "CLAUDE_CODE_CLI_INTERNAL_BETA", "cli-internal-2026-02-09"
)

# Added in CC 2.1.113. Reduces token cost of tool definitions + results.
# Sent unconditionally in TF>> betas.ts (no feature gate).
_TOKEN_EFFICIENT_TOOLS_BETA_HEADER = "token-efficient-tools-2026-03-28"

# Feature-gated in TF>> bundle (feature('_$L') — obfuscated flag, ant-only in
# production builds). HAL includes it unconditionally; filter_beta_header strips
# it for models that don't declare afk_mode=true. CCHSigner circuit-breaks it
# per-instance if the API rejects it — mirrors TF>>'s setAfkModeHeaderLatched.
_AFK_MODE_BETA_HEADER = "afk-mode-2026-01-31"

# Built at import time so the env-var override is respected.
# filter(None, ...) drops an empty _CLI_INTERNAL_BETA when the user opts out.
_ANTHROPIC_BETA = ",".join(filter(None, [
    "claude-code-20250219",
    "oauth-2025-04-20",
    "adaptive-thinking-2026-01-28",
    "research-preview-2026-02-01",
    "context-management-2025-06-27",
    _TOKEN_EFFICIENT_TOOLS_BETA_HEADER,
    _AFK_MODE_BETA_HEADER,
    _CLI_INTERNAL_BETA,
]))

# TF>> capEnabled=true: on stop_reason=="max_tokens" without an explicit
# max_tokens override, escalate to 64k (ESCALATED_MAX_TOKENS in
# TF>>'s src/utils/context.ts). HAL's call() implements the same one-shot retry.
_ESCALATED_MAX_TOKENS = 64_000

# Tool names used by ``clear_tool_uses_20250919`` context-management edits.
# Mirrored from TF>>'s ``apiMicrocompact.ts`` TOOLS_CLEARABLE_RESULTS /
# TOOLS_CLEARABLE_USES constants, mapped to HAL/CC canonical tool name strings.
_TOOLS_CLEARABLE_RESULTS: tuple[str, ...] = (
    "bash", "computer", "glob", "grep", "read",
    "web_fetch", "web_search",
)
_TOOLS_CLEARABLE_USES: tuple[str, ...] = (
    "str_replace_editor", "str_replace_based_edit_tool",
    "write", "notebook_edit",
)

# Beta header that gates the ``context_management`` body field upstream.
# Must be present in _ANTHROPIC_BETA above for the field to reach the API.
_CONTEXT_MANAGEMENT_BETA_HEADER = "context-management-2025-06-27"

# Server-side SUMMARISATION edit — opt-in capacity-extender, distinct from the
# clear_* family. Wire shape: betas=[compact-2026-01-12] +
# context_management.edits=[{type:compact_20260112}]. Gated behind the
# ``server_compaction`` capability, which defaults False for every model
# (_CONSERVATIVE_DEFAULTS + configs/models.toml) — so the wiring is inert until a
# model is explicitly marked. The edit and its beta header are gated by the one
# ``_effective_compact`` flag, so they always co-occur (a compact edit without
# its beta 400s; the beta alone is wasted). Shape provenance + live-confirmation
# evidence: PR #369 and the git history.
_COMPACT_BETA_HEADER = "compact-2026-01-12"
_COMPACT_EDIT_TYPE = "compact_20260112"

# Beta header that gates the ``thinking: {"type": "adaptive"}`` body field.
# Haiku 4.5 rejects requests that carry this header even when thinking is
# absent from the body, so the header MUST be stripped for models that
# don't declare adaptive_thinking=True in MODEL_CAPABILITIES.
_ADAPTIVE_THINKING_BETA_HEADER = "adaptive-thinking-2026-01-28"


# Per-model capability flags. Unknown models get a conservative default
# (both False) via MODEL_CAPABILITIES.get(model, _CONSERVATIVE_DEFAULTS)
# so a new model that lands without a registry entry degrades to the
# baseline feature set rather than 400-ing on an unsupported beta header.
#
# The registry is sourced from configs/models.toml (D2 from TF>>'s
# liberation matrix, 2026-04-20). Mirrors TF>>'s declarative model
# registry pattern in src/utils/model/configs.ts. Adding a new model
# means editing the TOML, not the code. Only set a capability to true
# when Anthropic confirms the feature is GA for that model — a
# speculative true surfaces as a 400 at request time.
_HAL_HOME_PATH = Path(os.environ.get("HAL_HOME", os.path.expanduser("~/.hal")))
_MODELS_TOML = _HAL_HOME_PATH / "configs" / "models.toml"


def _load_model_capabilities(path: Path = _MODELS_TOML) -> dict[str, dict[str, bool]]:
    """Load MODEL_CAPABILITIES from a TOML registry at import time.

    Schema: top-level [models."<model-id>"] sections with bool capability
    fields. Raises ImportError on missing or empty file — silent fallback
    to an empty dict would mask known-model regressions by sending every
    model to _CONSERVATIVE_DEFAULTS without warning.
    """
    if not path.is_file():
        raise ImportError(
            f"HAL model registry missing at {path}. "
            f"Set HAL_HOME or restore configs/models.toml."
        )
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    models = data.get("models", {})
    if not isinstance(models, dict) or not models:
        raise ImportError(
            f"HAL model registry at {path} has no [models.*] sections."
        )
    return {
        name: {k: bool(v) for k, v in caps.items() if isinstance(v, bool)}
        for name, caps in models.items()
        if isinstance(caps, dict)
    }


MODEL_CAPABILITIES: dict[str, dict[str, bool]] = _load_model_capabilities()

_CONSERVATIVE_DEFAULTS = {
    "adaptive_thinking": False,
    "context_management": False,
    "token_efficient_tools": False,
    "afk_mode": False,
    "server_compaction": False,
}


# D5 (TF>> liberation matrix, 2026-04-20):
# Mode → context-pruning strategy table. Maps GRIP mode names to the
# kwargs accepted by ``_compute_context_management``. Lives in TOML so
# operators can tune pruning per mode (analysis is read-heavy, code is
# write-heavy, rsi is fire-heavy) without code changes. Missing file
# falls back to ``_DEFAULT_CONTEXT_STRATEGY``, mirroring TF>>'s
# default-config-on-missing-source pattern from src/utils/model/configs.ts.
_MODE_STRATEGIES_TOML = _HAL_HOME_PATH / "configs" / "mode_context_strategies.toml"

_VALID_STRATEGY_KEYS = frozenset({
    "clear_tool_results",
    "clear_tool_uses",
    "max_input_tokens",
    "target_input_tokens",
})

_DEFAULT_CONTEXT_STRATEGY: dict = {
    "clear_tool_results": False,
    "clear_tool_uses": False,
    "max_input_tokens": 180_000,
    "target_input_tokens": 40_000,
}


def _load_mode_strategies(path: Path = _MODE_STRATEGIES_TOML) -> dict[str, dict]:
    """Load the mode → strategy table from TOML at import time.

    Permissive (unlike :func:`_load_model_capabilities`): a missing file
    returns ``{}`` so callers fall through to ``_DEFAULT_CONTEXT_STRATEGY``.
    The model registry is required for correctness (wrong beta headers
    cause 400s), but the strategy table is an optimisation — its absence
    means "use the conservative default", which is always a safe choice.

    Per-mode dicts are filtered to ``_VALID_STRATEGY_KEYS`` so a typo
    (``clear_tool_uss = true``) doesn't silently flow into the API call.
    """
    if not path.is_file():
        return {}
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    modes = data.get("modes", {})
    if not isinstance(modes, dict):
        return {}
    return {
        name: {k: v for k, v in cfg.items() if k in _VALID_STRATEGY_KEYS}
        for name, cfg in modes.items()
        if isinstance(cfg, dict)
    }


_MODE_STRATEGIES: dict[str, dict] = _load_mode_strategies()


def resolve_context_strategy(mode: str | None) -> dict:
    """Return the CCH context-management kwargs for a GRIP mode.

    Lookup order:
        1. ``_MODE_STRATEGIES[mode]`` — explicit table entry for this mode
        2. ``_MODE_STRATEGIES["default"]`` — operator-defined default
        3. ``_DEFAULT_CONTEXT_STRATEGY`` — code-level conservative default

    Returns a fresh dict each call so callers can mutate the result
    (e.g. for one-off overrides) without poisoning the module-level table.
    """
    if mode and mode in _MODE_STRATEGIES:
        base = _MODE_STRATEGIES[mode]
    elif "default" in _MODE_STRATEGIES:
        base = _MODE_STRATEGIES["default"]
    else:
        base = _DEFAULT_CONTEXT_STRATEGY
    return {**_DEFAULT_CONTEXT_STRATEGY, **base}


def supports_capability(model: str, capability: str) -> bool:
    """Return whether ``model`` advertises ``capability`` as supported.

    Unknown models degrade to ``False`` via ``_CONSERVATIVE_DEFAULTS`` —
    the safer failure mode is "feature off" rather than "400 on an
    unsupported beta header mid-turn".
    """
    caps = MODEL_CAPABILITIES.get(model, _CONSERVATIVE_DEFAULTS)
    return caps.get(capability, False)


def filter_beta_header(beta_value: str, *, model: str) -> str:
    """Strip beta tokens that ``model`` does not support from ``beta_value``.

    ``beta_value`` is the comma-joined ``anthropic-beta`` header value.
    Tokens gated by a capability flag (adaptive-thinking, context-
    management) are dropped when the model doesn't declare that flag.
    Tokens that aren't capability-gated pass through unchanged — HAL
    doesn't know which ones care about the model.
    """
    gated = {
        _ADAPTIVE_THINKING_BETA_HEADER: "adaptive_thinking",
        _CONTEXT_MANAGEMENT_BETA_HEADER: "context_management",
        _TOKEN_EFFICIENT_TOOLS_BETA_HEADER: "token_efficient_tools",
        _AFK_MODE_BETA_HEADER: "afk_mode",
        _COMPACT_BETA_HEADER: "server_compaction",
    }
    kept = [
        tok for tok in (t.strip() for t in beta_value.split(","))
        if tok and (tok not in gated or supports_capability(model, gated[tok]))
    ]
    return ",".join(kept)


def _effective_compact(model: str, requested: bool) -> bool:
    """Single source of truth for the compact_20260112 gate.

    Both the body edit (:meth:`CCHSigner.build_body`) and the beta header
    (:meth:`CCHSigner.build_request`) call this, so the compact edit and its
    ``compact-2026-01-12`` beta can never desync — they are emitted on EXACTLY
    the same condition. Requires the caller's opt-in AND the
    ``context_management`` field capability (the field that carries the edit)
    AND the ``server_compaction`` edit capability. A model with only one of the
    two capabilities yields neither the edit nor the beta (never a 400-shaped
    edit-without-beta nor a wasted beta-without-edit).
    """
    return (
        bool(requested)
        and supports_capability(model, "context_management")
        and supports_capability(model, "server_compaction")
    )


def _compute_context_management(
    *,
    has_thinking: bool = True,
    is_redact_thinking_active: bool = False,
    clear_all_thinking: bool = False,
    clear_tool_results: bool = False,
    clear_tool_uses: bool = False,
    compact: bool = False,
    max_input_tokens: int = 180_000,
    target_input_tokens: int = 40_000,
    clearable_results: tuple[str, ...] = _TOOLS_CLEARABLE_RESULTS,
    clearable_uses: tuple[str, ...] = _TOOLS_CLEARABLE_USES,
):
    """Return the ``context_management`` body field, or ``None``.

    Ported from ``paoloanzn/free-code``'s ``getAPIContextManagement`` and
    ``apiMicrocompact.ts``; shape recorded in ``plans/cch-context-management-shape.md``.

    ``clear_thinking_20251015`` is always emitted when thinking is active.

    ``clear_tool_uses_20250919`` (two opt-in modes):
    - ``clear_tool_results=True``: prunes result blocks from read-type tools
      (bash, glob, grep, read, web_fetch, web_search) via ``clear_tool_inputs``.
    - ``clear_tool_uses=True``: prunes all tool_use blocks except write-type
      tools (str_replace_editor, write, notebook_edit) via ``exclude_tools``.

    Both tool-clearing modes fire when ``max_input_tokens`` is reached and
    prune down to ``target_input_tokens``.  Defaults mirror the upstream
    values from TF>>'s ``apiMicrocompact.ts`` (180k trigger, 40k keep).

    ``compact`` (opt-in, default off) appends the ``compact_20260112``
    server-side summarisation edit. Like the ``clear_*`` flags above, this is a
    pure flag->edit transform that does no capability lookup of its own — it
    trusts its boolean arguments. The compact gate (``server_compaction`` AND
    ``context_management``) lives in :func:`_effective_compact`, which
    :meth:`CCHSigner.build_body` consults before passing ``compact=`` here, so
    the edit and its beta header are emitted on one identical condition.

    Returns ``None`` when thinking is disabled, redact-thinking is active,
    AND neither a tool-clearing mode nor ``compact`` is requested — the field
    must be omitted entirely when the edits list would be empty (the API treats
    ``{"edits": []}`` as a missing capability, not a no-op).
    """
    edits: list[dict] = []

    if has_thinking and not is_redact_thinking_active:
        if clear_all_thinking:
            edits.append({
                "type": "clear_thinking_20251015",
                "keep": {"type": "thinking_turns", "value": 1},
            })
        else:
            edits.append({"type": "clear_thinking_20251015", "keep": "all"})

    if clear_tool_results and clearable_results:
        edits.append({
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "input_tokens", "value": max_input_tokens},
            "clear_tool_inputs": list(clearable_results),
            "clear_at_least": {"type": "input_tokens", "value": target_input_tokens},
        })

    if clear_tool_uses and clearable_uses:
        edits.append({
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "input_tokens", "value": max_input_tokens},
            "exclude_tools": list(clearable_uses),
            "clear_at_least": {"type": "input_tokens", "value": target_input_tokens},
        })

    if compact:
        edits.append({"type": _COMPACT_EDIT_TYPE})

    return {"edits": edits} if edits else None


class CCHError(Exception):
    """Non-retryable CCH request failure."""


def _first_text_part(content) -> str:
    """Return the first ``type="text"`` part from a multimodal content list."""
    if not isinstance(content, list):
        return ""
    text_parts = (
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )
    return next(text_parts, "")


def _message_text(msg) -> str:
    """Return a single-string view of a ChatMessage's content."""
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    return _first_text_part(content)


def extract_first_user_text(messages) -> str:
    """Return the first user message's plain-text content, for CCH fingerprint.

    W14 preserved: :meth:`CCHSigner.compute_suffix` must see the initial user
    prompt so the 3-char suffix stays stable across turns in a multi-turn
    conversation. A rotating fingerprint would break CCH cache coherence
    upstream, so turn 2, 3, ... use the same suffix as turn 1.

    Content-part handling: if the message's ``content`` is a list (multimodal
    message), return the first ``type="text"`` part's text. Non-text-only
    assistant prior turns (e.g. thinking blocks, tool use blocks) are ignored
    because the fingerprint is supposed to reflect USER input, not model
    output. Returns an empty string when no user message is present, so
    ``compute_suffix`` pads with '0' characters safely.
    """
    if not messages:
        return ""
    user_msgs = (m for m in messages if getattr(m, "role", "") == "user")
    first_user = next(user_msgs, None)
    return _message_text(first_user) if first_user is not None else ""


def _serialise_chat_message(msg) -> dict:
    """Serialise a :class:`~lib.hal.types.ChatMessage` to a JSON-safe dict.

    ``content`` is passed through unchanged — already either a string or a
    list of content-part dicts, both of which are valid JSON values. This
    preserves assistant thinking blocks (``{"type": "thinking", "signature":
    "...", "thinking": "..."}``) verbatim so the Anthropic API can verify
    signature continuity on the next turn.
    """
    return {"role": msg.role, "content": msg.content}


def _is_empty_content(content) -> bool:
    """Return True when a message's content has no model-visible text or blocks.

    Empty string, None, empty list, or list whose every text block is
    blank/whitespace and no non-text blocks (image, tool_use, tool_result,
    document) are present. Unknown block types are treated as non-empty
    (fail-open) so a new Anthropic block type does not silently drop a turn.
    """
    if content is None or content == "":
        return True
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        if not content:
            return True
        for part in content:
            if not isinstance(part, dict):
                return False
            ptype = part.get("type")
            if ptype == "text":
                if (part.get("text") or "").strip():
                    return False
            elif ptype in ("image", "tool_use", "tool_result", "document"):
                return False
            else:
                return False  # unknown type — keep (fail-open)
        return True
    return False


def _filter_empty_messages(msg_list: list[dict]) -> list[dict]:
    """Drop messages with empty content; preserve order for remaining ones.

    Silently drops empty user AND assistant turns — the Anthropic API rejects
    empty user messages with a 400, and empty assistant turns break the
    alternating-role invariant in subsequent turns.
    """
    return [m for m in msg_list if not _is_empty_content(m.get("content"))]


class CCHSigner:
    """Signs and optionally sends CC-subscription-billed Messages API calls.

    Mostly stateless — each call builds a fresh body, computes its CCH, and
    submits it. The one piece of per-session state is ``_afk_latched``: if the
    API rejects the AFK-mode beta header the instance circuit-breaks it for the
    remainder of its lifetime, mirroring TF>>'s ``setAfkModeHeaderLatched``.
    """

    def __init__(
        self,
        *,
        cc_version: Optional[str] = None,
        salt: str = _DEFAULT_SALT,
        cch_seed: int = _DEFAULT_CCH_SEED,
    ):
        self.cc_version = (cc_version if cc_version is not None
                           else detect_cc_version())
        self.salt = salt
        self.cch_seed = cch_seed
        self._afk_latched: bool = False

    # ---- Signing -----------------------------------------------------------

    def compute_suffix(self, prompt: str) -> str:
        """Derive the 3-char cc_version suffix from the prompt + salt + version.

        Algorithm from reverse-engineering: pick characters at indices 4, 7, 20
        from the prompt (padding with '0' if too short), concatenate with salt
        and version, SHA-256, take the first 3 hex chars.
        """
        chars = "".join(prompt[i] if i < len(prompt) else "0" for i in (4, 7, 20))
        digest = hashlib.sha256(
            f"{self.salt}{chars}{self.cc_version}".encode(),
        ).hexdigest()
        return digest[:3]

    def build_body(
        self,
        prompt: str = "",
        *,
        model: str,
        messages: Optional[list[ChatMessage]] = None,
        system: Optional[str] = None,
        max_tokens: int = 32000,
        session_id: str = "",
        fingerprint_text: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        clear_tool_results: bool = False,
        clear_tool_uses: bool = False,
        compact: bool = False,
    ) -> tuple[str, str]:
        """Build the JSON request body and return (body_with_placeholder, suffix).

        The returned body contains ``cch=00000`` as a placeholder — the caller
        (or :meth:`sign`) must replace it with the computed hash.

        Two calling forms are supported for backward compatibility:

        * **Single-turn** (``build_body("hi", model=...)``): the first
          positional is the user prompt; a messages list of length 1 is
          synthesised internally.
        * **Multi-turn** (``build_body(messages=[m1, m2, m3], model=...)``):
          the ``messages`` kwarg supplies the full conversation history.
          Each message's ``content`` (string or content-part list) is passed
          through verbatim, preserving assistant ``thinking`` blocks with
          their signatures so the API can verify reasoning continuity.

        ``system`` is an optional user-supplied system prompt that is
        appended as a second ``text`` block AFTER the CCH billing header.
        The billing header block is always present and always first —
        removing it would break the subscription billing path.

        ``fingerprint_text`` overrides the default suffix derivation. When
        not provided, the suffix is computed from the first user message
        (multi-turn) or from ``prompt`` (single-turn). W14 preserved:
        turn 2+ of a conversation uses the same suffix as turn 1 because
        ``extract_first_user_text`` always targets the initial user turn.

        ``tools`` is an optional list of Anthropic tool schemas (from
        :func:`~lib.hal.tool_adapter.to_anthropic_tools`). When non-empty,
        the list is added to the payload as ``"tools": [...]`` so the
        model can emit ``tool_use`` content blocks for the HAL tool loop
        (:func:`~lib.hal.tool_loop.run_tool_loop`). Empty/``None`` is
        treated as "no tools" and the field is omitted — matching the
        behaviour of :func:`~lib.hal.tool_runtime.make_anthropic_chat_fn`.
        """
        if fingerprint_text is not None:
            fp_text = fingerprint_text
        elif messages:
            fp_text = extract_first_user_text(messages)
        else:
            fp_text = prompt
        suffix = self.compute_suffix(fp_text)
        sid = session_id or f"user_poc_session_{uuid.uuid4()}"

        if messages:
            msg_list = _filter_empty_messages(
                [_serialise_chat_message(m) for m in messages]
            )
            if not msg_list:
                raise CCHError(
                    "Cannot build CCH body: all messages have empty content",
                )
        else:
            msg_list = [{"role": "user", "content": prompt}]

        system_blocks = [{
            "type": "text",
            "text": (
                f"x-anthropic-billing-header: cc_version={self.cc_version}.{suffix};"
                " cc_entrypoint=cli; cch=00000;"
            ),
        }]
        if system:
            system_blocks.append({"type": "text", "text": system})

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "stream": False,
            "metadata": {"user_id": sid},
            "system": system_blocks,
            "messages": msg_list,
        }
        # Adaptive thinking is a model-capability-gated field. Haiku 4.5
        # rejects requests that carry it; Opus 4.7 / Sonnet 4.6 require
        # it to unlock the full reasoning budget. MODEL_CAPABILITIES is
        # the single source of truth — unknown models fall through to
        # the conservative default (off).
        if supports_capability(model, "adaptive_thinking"):
            payload["thinking"] = {"type": "adaptive"}
        # NOTE: `research_preview_2026_04` was a body field used during the
        # April-preview window; Opus 4.7 (GA 2026-04-16) rejects it as
        # "Extra inputs are not permitted" — caught by live CCH smoke.
        # The feature graduated out of the preview gate.
        #
        # Context management is model-capability-gated (Haiku 4.5 supports
        # it but models without adaptive_thinking produce a clear_thinking
        # edit with no thinking to clear — harmless but pointless). Gate on
        # context_management flag and pass has_thinking based on whether
        # the adaptive thinking field was emitted above.
        if supports_capability(model, "context_management"):
            context_management = _compute_context_management(
                has_thinking=supports_capability(model, "adaptive_thinking"),
                is_redact_thinking_active=False,
                clear_all_thinking=False,
                clear_tool_results=clear_tool_results,
                clear_tool_uses=clear_tool_uses,
                compact=_effective_compact(model, compact),
            )
            if context_management is not None:
                payload["context_management"] = context_management
        if tools:
            payload["tools"] = tools
        body = json.dumps(payload, separators=(",", ":"))
        return body, suffix

    def compute_cch(self, body_with_placeholder: str) -> str:
        """Compute the 5-hex-digit CCH for a body with ``cch=00000`` placeholder.

        xxhash is an optional dependency; the import is local so callers that
        only use :meth:`build_body` (e.g. in tests) don't need it installed.
        """
        try:
            import xxhash  # type: ignore
        except ImportError as exc:  # pragma: no cover — import guard
            raise CCHError(
                "xxhash package required for CCH signing: pip install xxhash",
            ) from exc
        digest = xxhash.xxh64(
            body_with_placeholder.encode(), seed=self.cch_seed,
        ).intdigest()
        return format(digest & _CCH_MASK, "05x")

    def sign(
        self,
        prompt: str = "",
        *,
        model: str,
        **build_kwargs,
    ) -> tuple[str, str]:
        """Return the signed body (with real cch=HASH) plus the cc_version suffix.

        ``build_kwargs`` is forwarded verbatim to :meth:`build_body` — see
        that method's docstring for the supported keyword arguments
        (``messages``, ``system``, ``max_tokens``, ``session_id``,
        ``fingerprint_text``). This thin passthrough exists so the CCH
        signing hash is computed on the exact body the caller requested.
        """
        body_with_placeholder, suffix = self.build_body(
            prompt, model=model, **build_kwargs,
        )
        cch = self.compute_cch(body_with_placeholder)
        signed = body_with_placeholder.replace("cch=00000", f"cch={cch}")
        return signed, suffix

    # ---- Request assembly + dispatch --------------------------------------

    def build_request(
        self,
        prompt: str = "",
        *,
        oauth_token: str,
        model: str,
        **build_kwargs,
    ) -> tuple[dict, bytes]:
        """Return (headers, body_bytes) ready for any HTTP client.

        ``build_kwargs`` is forwarded to :meth:`build_body` (via
        :meth:`sign`). Useful when integrating CCH into an existing client
        (httpx, aiohttp) rather than going through HAL's :meth:`call`.
        """
        signed_body, suffix = self.sign(
            prompt, model=model, **build_kwargs,
        )
        # Opt-in compact summarisation: append the compact beta ONLY when the
        # caller requested it AND the model advertises both the context_management
        # field and the server_compaction edit. This mirrors the build_body edit
        # gate exactly, so the compact edit and its beta header always co-occur
        # (a compact edit without its beta 400s; the beta without the edit is
        # wasted). filter_beta_header re-validates server_compaction below as a
        # belt-and-suspenders strip for unsupported models.
        want_compact = _effective_compact(model, build_kwargs.get("compact", False))
        beta_tokens = [
            t for t in _ANTHROPIC_BETA.split(",")
            if not (self._afk_latched and t == _AFK_MODE_BETA_HEADER)
        ]
        if want_compact:
            beta_tokens.append(_COMPACT_BETA_HEADER)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {oauth_token}",
            "User-Agent": f"claude-cli/{self.cc_version} (external, cli)",
            "anthropic-version": _ANTHROPIC_API_VERSION,
            # Per-model beta-header filter: drops adaptive-thinking /
            # context-management / compact tokens when MODEL_CAPABILITIES says
            # the model doesn't advertise them. Required for Haiku 4.5 which
            # 400s on adaptive-thinking-2026-01-28.
            # AFK circuit breaker: strip afk-mode token if this instance has
            # latched it off after a prior API rejection (mirrors TF>>'s
            # setAfkModeHeaderLatched pattern).
            "anthropic-beta": filter_beta_header(",".join(beta_tokens), model=model),
            "x-app": "cli",
            "x-cc-version-suffix": suffix,  # debugging aid — ignored by API
        }
        return headers, signed_body.encode()

    def call(
        self,
        prompt: str = "",
        *,
        oauth_token: str,
        model: str = "claude-opus-4-7",
        **build_kwargs,
    ) -> ChatResponse:
        """Send the signed request via urllib and parse the Messages API response.

        ``build_kwargs`` is forwarded to :meth:`build_body` — see that
        method's docstring for the supported keyword arguments (multi-turn
        ``messages``, ``system``, ``max_tokens``, ``session_id``,
        ``fingerprint_text``). Raises :class:`CCHError` on non-retryable
        HTTP failures. HAL's
        :class:`~lib.hal.providers.anthropic_native.AnthropicProvider`
        wraps this with retry + backoff when used via the
        ``subscription=`` path.

        **Cap escalation** (TF>> ``capEnabled=true``): when the API returns
        ``stop_reason="max_tokens"`` and the caller did not set an explicit
        ``max_tokens`` kwarg, one automatic retry fires at
        :data:`_ESCALATED_MAX_TOKENS` (64k). Pass ``max_tokens=N`` explicitly
        to opt out. Mirrors TF>>'s ``src/query.ts`` escalation.
        """
        headers, body_bytes = self.build_request(
            prompt, oauth_token=oauth_token, model=model, **build_kwargs,
        )
        try:
            data, latency_ms = _dispatch_request(headers, body_bytes)
        except CCHError as exc:
            # AFK circuit breaker: if the API rejects the afk-mode beta header,
            # latch it off for this instance's lifetime and retry once without it.
            # Mirrors TF>>'s setAfkModeHeaderLatched / afkModeHeaderLatched check.
            if _AFK_MODE_BETA_HEADER in str(exc) and not self._afk_latched:
                self._afk_latched = True
                headers, body_bytes = self.build_request(
                    prompt, oauth_token=oauth_token, model=model, **build_kwargs,
                )
                data, latency_ms = _dispatch_request(headers, body_bytes)
            else:
                raise
        response = _parse_messages_response(data, latency_ms)

        if response.finish_reason == "max_tokens" and "max_tokens" not in build_kwargs:
            headers2, body2 = self.build_request(
                prompt, oauth_token=oauth_token, model=model,
                max_tokens=_ESCALATED_MAX_TOKENS, **build_kwargs,
            )
            data2, latency2 = _dispatch_request(headers2, body2)
            response = _parse_messages_response(data2, latency_ms + latency2)

        return response

    def generate_session_recap(
        self,
        *,
        oauth_token: str,
        model: str = "claude-sonnet-4-6",
        messages: Optional[list[ChatMessage]] = None,
    ) -> str:
        """Return a one-line recap of the current conversation.

        Mirrors CC's ``generateSessionRecap()`` (TF>> cli-formatted.js L491954):
        appends "Generate a one-line session recap now." as a user turn and calls
        the API with a 100-token cap. Suitable for session titles or checkpoint
        labels. Requires ``afk-mode-2026-01-31`` to be in the accepted beta
        set — the AFK circuit breaker will handle rejection transparently.
        """
        recap_msg = ChatMessage(role="user", content="Generate a one-line session recap now.")
        full_messages = list(messages or []) + [recap_msg]
        return self.call(
            oauth_token=oauth_token,
            model=model,
            messages=full_messages,
            max_tokens=100,
        ).text.strip()


def _dispatch_request(headers: dict, body_bytes: bytes) -> tuple[dict, float]:
    """Execute one CCH HTTP POST; return (parsed_json_body, latency_ms).

    Extracted from :meth:`CCHSigner.call` so the cap-escalation retry can
    reuse the same dispatch path without duplicating error handling.
    """
    start_ms = time.monotonic() * 1000
    req = urllib.request.Request(
        _API_URL, data=body_bytes, method="POST", headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")[:500]
        raise CCHError(f"CCH HTTP {exc.code}: {err_body}") from exc
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        raise CCHError(f"CCH connection failed: {exc}") from exc
    return data, time.monotonic() * 1000 - start_ms


def _parse_messages_response(data: dict, latency_ms: float) -> ChatResponse:
    """Convert an Anthropic Messages API response dict to a ``ChatResponse``."""
    content_blocks = data.get("content", [])
    text = "".join(
        block.get("text", "") for block in content_blocks
        if block.get("type") == "text"
    )
    usage = data.get("usage", {})
    inp = usage.get("input_tokens", 0)
    out = usage.get("output_tokens", 0)
    return ChatResponse(
        text=text,
        model=data.get("model", ""),
        provider="anthropic-cch",
        finish_reason=data.get("stop_reason", "end_turn"),
        usage=TokenUsage(
            prompt_tokens=inp,
            completion_tokens=out,
            total_tokens=inp + out,
            cache_read_tokens=usage.get("cache_read_input_tokens", 0),
            cache_creation_tokens=usage.get("cache_creation_input_tokens", 0),
        ),
        latency_ms=latency_ms,
        raw=data,
    )
