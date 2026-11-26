"""HAL Gateway egress — same-wire multi-provider re-pointing (Phase 0 wave 3).

When a request's model routes (via ``model_route.route_model``) to a NON-Anthropic
provider AND multi-provider routing is enabled, forward the ORIGINAL OpenAI-wire
body VERBATIM to that provider's OpenAI-compatible endpoint with the provider's
own Bearer key — no Anthropic translation. This is the "same-wire" 80%: an
OpenAI-Responses/Chat client (Codex / Antigravity) reaching a real
OpenAI-compatible provider needs no wire translation, only a different upstream +
key.

This module ALSO orchestrates the cross-wire 20% (:func:`cross_wire_forward`): an
Anthropic-wire client (Claude Code via ``/v1/messages``) targeting an OpenAI-family
model. The pure request/response/SSE translators live in ``gateway_crosswire``; this
function ties them to the egress machinery (provider key + endpoint) and returns a
synthetic Anthropic response (buffered JSON, or a synthesised Anthropic SSE when the
client asked to stream). Same flag, same provider gate as same-wire.

This NEVER touches the #3274 verbatim Anthropic path. ``claude-*``
(``route.verbatim``) and unknown models return ``None`` from
:func:`same_wire_provider`, so the caller falls through to the existing
Anthropic-translating handler unchanged. And it is DEFAULT-ON (sovereign,
non-Anthropic-by-default): absent ``HAL_GATEWAY_MULTIPROVIDER`` ⇒ routing ON, so
a known non-Anthropic model (Codex's ``gpt-5*``, ``deepseek-*``) reaches its OWN
provider by default. The kill-switch is EXPLICIT: only a falsey value
(``0/false/no/off``) makes :func:`same_wire_provider` always return ``None`` and
the gateway behave as a verbatim Anthropic relay. ``claude-*``/unknown stay
verbatim regardless of the flag, so DEFAULT-ON cannot break the #3274 path.
Cross-wire translation (Anthropic-client ↔ OpenAI/Gemini provider tool-call
streaming) is wave 4 — not here.

The ONE case that is neither routed nor a fall-through: a provider the router
KNOWS but this gateway has no endpoint for. That raises ``GatewayUpstreamError``
(a clean 502 at every surface) instead of answering ``None``, because ``None``
means "use Anthropic" to every caller — see :func:`unroutable_provider`.

Syscall doctrine: ``opener`` + ``key_resolver`` are DIP-injected so tests run
with zero network and zero keychain access. The provider key is never logged.
"""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
from typing import Callable, List, Optional, Tuple

from lib.hal.spend_guard import require_openai_destination
from lib.hal.model_quirks import (
    MAX_COMPLETION_TOKENS_MODELS as _QUIRKS_MAX_COMPLETION_TOKENS_MODELS,
    uses_max_completion_tokens as _quirks_uses_max_completion_tokens,
    model_requires_temperature_one as _model_requires_temperature_one,
    is_gpt6_family as _quirks_is_gpt6_family,
    locked_sampling_params as _quirks_locked_sampling_params,
    refused_reasoning_efforts as _quirks_refused_reasoning_efforts,
    tools_require_responses_api as _quirks_tools_require_responses_api,
)

from lib.hal.gateway import (
    GatewayProxy, GatewayUpstreamError, UpstreamResponse, build_default_opener,
    retry_headers)
from lib.hal.model_route import ANTHROPIC, route_model, strip_provider_prefix

# provider name -> (OpenAI-compatible base_url, keychain service). Mirrors
# config/providers.yaml. Same-wire applies to OpenAI-compatible endpoints; the
# provider's base_url already carries its own version prefix (e.g. .../v1).
#
# EVERY name in model_route.KNOWN_PROVIDERS except "anthropic" MUST appear here
# or in LOCAL_PROVIDER_ENDPOINTS. A provider the router knows but this table
# cannot reach is the silent-misroute bug: see :func:`unroutable_provider`.
PROVIDER_ENDPOINTS: dict = {
    "openai":   ("https://api.openai.com/v1", "hal-openai"),
    "gemini":   ("https://generativelanguage.googleapis.com/v1beta/openai", "grip-gemini"),
    "xai":      ("https://api.x.ai/v1", "hal-xai"),
    "deepseek": ("https://api.deepseek.com/v1", "hal-deepseek"),
    "groq":     ("https://api.groq.com/openai/v1", "grip-groq"),
    "qwen":     ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "hal-dashscope"),
    # Alibaba Model Studio Token Plan Personal Edition -- a SEPARATE product
    # and host from the pay-per-token `qwen` row above: a flat-fee monthly
    # subscription that pauses at quota rather than billing more. Claude Code
    # W4's first fallback target (config/w4_cascade.json). Already declared in
    # config/providers.yaml and already wired into HAPPI
    # (happi/providers/openai_compat.py) and tool_cli_integration.py; this row
    # joins it to model_route.KNOWN_PROVIDERS so same_wire_provider can reach
    # it too, same pairing as kimi-coding below.
    "qwen-token-plan": ("https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
                        "grip-qwen-token-plan"),
    "moonshot": ("https://api.moonshot.ai/v1", "grip-moonshot"),
    # Kimi Code -- Moonshot's Kimi MEMBERSHIP coding service (GRIP Pool D,
    # sovereign-council seal e49c00c214c500be), a SEPARATE product and bill from
    # the pay-per-token `moonshot` row above: "the Kimi Code API is independent
    # from the API service on this platform". Same credential name as the GRIP
    # side (lib/provider_secret, GRIP#5336). UNARMED until keychain
    # grip-kimi-coding exists (HAL#550). Deliberately NOT in
    # ANTHROPIC_COMPAT_ENDPOINTS: the vendor documents this host's key in
    # ANTHROPIC_API_KEY (x-api-key), while AnthropicCompatProxy sends a Bearer,
    # and no credential exists to probe which the host accepts -- so an
    # Anthropic-wire client cross-wires onto this OpenAI endpoint instead.
    "kimi-coding": ("https://api.kimi.com/coding/v1", "grip-kimi-coding"),
    # Declared in config/providers.yaml since before the gateway existed, but
    # never mirrored here — so `mistral-small-latest` and `mistral/…` routed to
    # provider "mistral", found no endpoint, and fell through to the Anthropic
    # upstream, which rejected the caller's key as an invalid x-api-key
    # (live 2026-08-17). The credential + base_url are real: GET /v1/models with
    # keychain:grip-mistral answered HTTP 200 with 55 ids including
    # mistral-small-latest.
    "mistral":  ("https://api.mistral.ai/v1", "grip-mistral"),
    # Wired 2026-08-30 (direct calls verified 200 with the keychain keys; the
    # endpoints were declared in config/providers.yaml but never mirrored
    # here — same silent-misroute gap mistral had):
    "zhipu":     ("https://api.z.ai/api/coding/paas/v4", "grip-zhipu"),
    "openrouter": ("https://openrouter.ai/api/v1", "grip-openrouter"),
}

# Local (on-device) OpenAI-compatible endpoints — KEYLESS by construction
# (localhost only, no network egress, no credential). Kept SEPARATE from
# PROVIDER_ENDPOINTS so the cloud invariant (every cloud endpoint has an https
# base_url + a grip- keychain service) is preserved. Mirrors the `type: local`
# entries in config/providers.yaml. SSRF-safe: fixed localhost URLs, never
# client-controlled. A model whose family routes here is forwarded to the
# on-device server with NO Authorization header.
LOCAL_PROVIDER_ENDPOINTS: dict = {
    "bitnet": "http://127.0.0.1:8114/v1",  # Microsoft BitNet b1.58-2B-4T via llama-server
    # Local Ollama daemon; keyless. INERT for implicit routing: no
    # _FAMILY_ROUTES entry maps here (qwen* family-routes to dashscope), so
    # same_wire_provider never selects it — only EXPLICIT (provider, model)
    # rungs (gateway_fanout) forward to this endpoint.
    "ollama": "http://127.0.0.1:11434/v1",
}

# provider name -> (Anthropic-compatible base_url, keychain service). A provider
# listed here serves the Anthropic Messages wire NATIVELY, so an Anthropic-wire
# client (Claude Code via ``/v1/messages``) is forwarded VERBATIM -- thinking,
# tool_use and cache fields intact -- instead of through the cross-wire
# translator, which drops ``thinking`` / ``output_config`` / ``cache_control``
# (gateway_crosswire). The upstream URL is ``base + gateway_path`` AS-IS
# (``.../anthropic`` + ``/v1/messages``), NOT :func:`build_egress_url`, which
# strips the leading ``/v1`` and would yield ``/anthropic/messages``; the
# vendor path is ``/anthropic/v1/messages`` (probed 2026-09-07: HTTP 200).
# Every provider here MUST also be in PROVIDER_ENDPOINTS -- the routing gate
# (:func:`same_wire_provider`) raises "unroutable" for a known provider with
# no OpenAI endpoint before the Anthropic branch is consulted.
ANTHROPIC_COMPAT_ENDPOINTS: dict = {
    "moonshot": ("https://api.moonshot.ai/anthropic", "grip-moonshot"),
}

_FLAG = "HAL_GATEWAY_MULTIPROVIDER"

# Cloudflare in front of provider APIs (verified live on groq 2026-06-11:
# default Python-urllib UA -> 403 error 1010, browser-like UA -> 200) bans the
# urllib signature. Same class as the channel_backends.py Discord/Slack fix —
# every egress send carries a browser-like UA with an honest product suffix.
_USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
               "AppleWebKit/537.36 (KHTML, like Gecko) hal-gateway/0.1")

KeyResolver = Callable[[str], str]


def multiprovider_enabled() -> bool:
    """Whether same-wire multi-provider routing is ON. DEFAULT ON — sovereign,
    non-Anthropic-by-default: a known non-Anthropic model routes to its OWN
    provider (OpenAI/DeepSeek/Gemini/local) using HAL's internal key. The
    kill-switch is an EXPLICIT falsey ``HAL_GATEWAY_MULTIPROVIDER`` in
    {0,false,no,off}; absent => ON. ``claude-*``/unknown models stay verbatim
    regardless (``same_wire_provider`` returns None for them — #3274 untouched),
    so flipping the default cannot break the Anthropic path."""
    return os.environ.get(_FLAG, "1").strip().lower() not in ("0", "false", "no", "off")


def _routed_provider(model: str) -> Optional[str]:
    """The non-Anthropic provider ``model`` resolves to, ignoring reachability.

    ``None`` for the three cases that legitimately fall through to the verbatim
    Anthropic path: the kill-switch is off, the model is Anthropic's (verbatim,
    #3274), or the router does not recognise the id at all.
    """
    if not multiprovider_enabled():
        return None
    route = route_model(model)
    if route.verbatim or not route.known:
        return None
    return route.provider


def _has_endpoint(provider: str) -> bool:
    """Whether this gateway has an OpenAI-compatible endpoint for ``provider``."""
    return provider in PROVIDER_ENDPOINTS or provider in LOCAL_PROVIDER_ENDPOINTS


def unroutable_provider(model: str) -> Optional[str]:
    """The provider ``model`` resolves to that this gateway CANNOT reach.

    The exact complement of :func:`same_wire_provider` over the routed set: for
    any model, at most one of the two ever answers a name. This one names the
    configuration gap — a provider in ``model_route.KNOWN_PROVIDERS`` with no
    entry in ``PROVIDER_ENDPOINTS`` / ``LOCAL_PROVIDER_ENDPOINTS``.

    Deliberately ``None`` for an UNKNOWN model id: the router made no decision
    there, so falling through to the Anthropic upstream contradicts nothing and
    is the long-standing behaviour for ids HAL does not recognise.

    This is the SINGLE definition of "unroutable" — :func:`same_wire_provider`
    raises off this exact answer, so the guard and the predicate cannot drift
    apart into two subtly different notions of the same gap.
    """
    provider = _routed_provider(model)
    if provider is None or _has_endpoint(provider):
        return None
    return provider


def _unroutable_message(provider: str, model: str) -> str:
    """The operator-facing error for a known provider with no endpoint."""
    return (
        f"model {model!r} routes to provider {provider!r}, which has no endpoint "
        f"in this gateway. Refusing to fall back to the {ANTHROPIC} upstream — "
        f"that would send your key to api.anthropic.com and fail as 'invalid "
        f"x-api-key', blaming the credential for a gateway configuration gap. "
        f"Fix: add {provider!r} to gateway_egress.PROVIDER_ENDPOINTS (mirroring "
        f"config/providers.yaml), or remove it from model_route.KNOWN_PROVIDERS "
        f"so the id is treated as unknown."
    )


def same_wire_provider(model: str) -> Optional[str]:
    """Provider to same-wire route ``model`` to, or ``None`` to fall through.

    Returns a provider name ONLY when: the flag is enabled AND the model routes
    to a known NON-Anthropic provider we have an OpenAI-compatible endpoint for.
    ``claude-*`` (verbatim), unknown models, and flag-off all return ``None`` —
    the caller then uses the existing Anthropic-translating path (#3274 untouched).

    RAISES ``GatewayUpstreamError`` for the one remaining case: a provider the
    router KNOWS but this gateway cannot reach. Every caller reads ``None`` as
    "use the Anthropic path", so answering ``None`` there silently sends the
    request to the wrong vendor — the defect this guard closes for
    ``/v1/chat/completions``, ``/v1/responses`` and ``/v1/messages`` at once.
    The guard lives HERE, at the single production consumer of ``route_model``,
    precisely so a fourth surface cannot reopen the hole by forgetting to check.

    Every call site already converts ``GatewayUpstreamError`` into a clean 502
    in its own wire's error shape, so this is loud to the client and never a
    crash. Flag-off still returns ``None`` unconditionally: the kill-switch must
    stay total, or it would itself become a new failure mode.
    """
    unreachable = unroutable_provider(model)
    if unreachable is not None:
        raise GatewayUpstreamError(_unroutable_message(unreachable, model))
    return _routed_provider(model)


def anthropic_compat_provider(model: str) -> Optional[str]:
    """Provider whose ANTHROPIC-compatible endpoint should serve ``model``
    same-wire, or ``None`` to fall through.

    Exactly :func:`same_wire_provider`'s answer narrowed to the providers in
    :data:`ANTHROPIC_COMPAT_ENDPOINTS`: ``claude-*`` (verbatim), unknown ids and
    flag-off stay ``None`` precisely as today, and a provider that has only an
    OpenAI endpoint (openai, gemini, ...) is ``None`` here too, so it keeps
    cross-wiring as before. Raises the same ``GatewayUpstreamError`` for a
    known-but-unreachable provider, because it is the same gate.
    """
    provider = same_wire_provider(model)
    if provider in ANTHROPIC_COMPAT_ENDPOINTS:
        return provider
    return None


def _default_key_resolver(keychain_service: str) -> str:
    """Read a provider key. macOS keychain first; then env fallbacks. Never logged.

    Env fallbacks let a non-macOS host (Linux VPS, CI, a tailnet Linux peer)
    resolve provider keys with NO keychain — the sovereign, cross-platform path:
    a Linux gateway routes non-Anthropic providers from env alone. Two names are
    accepted, in order: the bare ``<PROVIDER>_API_KEY`` (legacy) and the
    canonical GRIP-prefixed ``GRIP_<PROVIDER>_API_KEY`` used across the substrate
    (``secrets.env``, ``providers.yaml``, hal-server). Without the GRIP-prefixed
    fallback the gateway could not see the keys every other HAL surface already
    has on Linux — the exact VPS gap this closes.
    """
    try:
        out = subprocess.run(
            ["security", "find-generic-password", "-s", keychain_service, "-w"],
            capture_output=True, text=True, timeout=5, check=False)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    # The public build renames five slots from the grip- prefix to hal-; both
    # prefixes name the same variable, so a stranger's env keys still resolve.
    name = keychain_service.replace("grip-", "").removeprefix("hal-")
    base = name.replace("-", "_").upper() + "_API_KEY"
    return os.environ.get(base) or os.environ.get("GRIP_" + base, "")


def build_egress_url(provider: str, gateway_path: str) -> str:
    """Provider endpoint URL for a gateway path.

    The gateway path is e.g. ``/v1/responses`` or ``/v1/chat/completions``. The
    provider base_url already includes its version prefix, so a leading ``/v1``
    on the gateway path is stripped to avoid ``/v1/v1`` (openai → /v1/responses;
    gemini → /v1beta/openai/chat/completions).
    """
    base_url = (PROVIDER_ENDPOINTS[provider][0] if provider in PROVIDER_ENDPOINTS
                else LOCAL_PROVIDER_ENDPOINTS[provider])
    suffix = gateway_path[3:] if gateway_path.startswith("/v1") else gateway_path
    return base_url.rstrip("/") + suffix


_LOCAL_UNSUPPORTED_PARAMS = (
    "tools", "tool_choice", "parallel_tool_calls", "functions", "function_call",
)


def _fix_cloud_body(raw_body: Optional[bytes],
                    transform: Callable[[dict], dict]) -> Optional[bytes]:
    """Parse an OpenAI-wire request body, apply ``transform``, and re-serialise.

    A transform may mutate the dict IN PLACE or RETURN A NEW dict (copy-on-write
    transforms are honoured; their input is left untouched). Fail-open +
    verbatim-preserving: a missing / non-dict / unparseable body, OR one
    ``transform`` leaves unchanged, is returned UNCHANGED (byte-identical) — a
    no-op transform never perturbs key order or whitespace. ``transform`` MUST
    be a no-op when not applicable."""
    if not raw_body:
        return raw_body
    try:
        obj = json.loads(raw_body)
    except (ValueError, TypeError):
        return raw_body
    if not isinstance(obj, dict):
        return raw_body
    before = obj.copy()
    result = transform(obj)
    if isinstance(result, dict):
        obj = result
    if obj == before:
        return raw_body  # unchanged -> preserve exact bytes
    return json.dumps(obj).encode("utf-8")


def _drop_local_unsupported(obj: dict) -> dict:
    """Drop params a local llama.cpp endpoint rejects (older builds 500 on
    ``tools``). No-op when none present."""
    for k in _LOCAL_UNSUPPORTED_PARAMS:
        obj.pop(k, None)
    return obj


# A harness may address a model with a bracketed capability suffix: Claude
# Code emits ``kimi-k3[1m]`` for its 1M-context tier (the vendor's own Claude
# Code recipe uses that id). Anthropic parses the suffix; no other provider
# does -- Moonshot 404s on it -- so it is stripped on every non-Anthropic
# egress. Exactly ONE trailing ``[...]``: a bracket mid-id is left alone, and
# an id that is nothing but a suffix is never collapsed to "".
_BRACKET_SUFFIX_RE = re.compile(r"\[[^\[\]]*\]$")


def _strip_bracket_suffix(model: str) -> str:
    bare = _BRACKET_SUFFIX_RE.sub("", model, count=1)
    return bare or model


def _provider_model_rewriter(provider: str) -> Callable[[dict], dict]:
    """A ``_fix_cloud_body`` transform that addresses ``model`` as ``provider``
    knows it: the ``<provider>/`` head goes, and for every provider except
    Anthropic a trailing ``[...]`` capability suffix (``[1m]``) goes too.

    HAL addresses models as ``<provider>/<model>``, but the provider itself has
    never heard of the prefix: Groq publishes ``openai/gpt-oss-120b``, so
    forwarding the catalog's ``groq/openai/gpt-oss-120b`` verbatim 404s
    ``model_not_found``. This strips the ONE leading segment that names the
    provider being forwarded to — see :func:`model_route.strip_provider_prefix`
    for why the match is provider-scoped rather than "any known provider".

    No-op (byte-identical body, per ``_fix_cloud_body``) for an unprefixed model,
    a non-string model, and any id whose head names a different provider.
    """
    def _rewrite(obj: dict) -> dict:
        model = obj.get("model")
        if isinstance(model, str):
            upstream = strip_provider_prefix(model, provider)
            if provider.strip().lower() != "anthropic":
                upstream = _strip_bracket_suffix(upstream)
            if upstream != model:
                # Copy-on-write: return a FRESH dict rather than mutating the
                # caller's, so an earlier transform in the pipeline that kept a
                # reference never observes the stripped id (AGORA #535 C2).
                obj = dict(obj)
                obj["model"] = upstream
        return obj
    return _rewrite


# Moonshot's Kimi lineage rejects temperature != 1 with HTTP 400 ("invalid
# temperature: only 1 is allowed for this model") — live-verified for kimi-k2.5
# + kimi-k2.6 (2026-07-26), kimi-k2.7-code (#2209), kimi-k3 (2026-07-26). A
# coding harness (Codex / Aider) routing through the gateway sends its own
# default temperature (typically != 1), which 400s on these models. We omit
# temperature so Moonshot applies its own (1) — model-aware, never a global
# force. Predicate shared with the direct adapter path via
# lib/hal/model_quirks.py (single source of truth; YSH at N=3 surfaces).


def _drop_temperature_for_fixed_temp_model(obj: dict) -> dict:
    """Remove ``temperature`` from an OpenAI request dict iff its model only
    accepts temperature=1. Pure: returns ``obj`` unchanged when not applicable
    (no model match, or no temperature present), mutating in place otherwise."""
    if (isinstance(obj, dict) and "temperature" in obj
            and _model_requires_temperature_one(obj.get("model", ""))):
        obj.pop("temperature", None)
    return obj


def _fix_cloud_body_temperature(raw_body: Optional[bytes]) -> Optional[bytes]:
    """Drop ``temperature`` from a verbatim cloud body when its model only
    accepts temperature=1. Fail-open + verbatim-preserving (see _fix_cloud_body)."""
    return _fix_cloud_body(raw_body, _drop_temperature_for_fixed_temp_model)


# OpenAI's gpt-5.x family (gpt-5, gpt-5.5, gpt-5*-codex) REJECTS the legacy
# `max_tokens` param with HTTP 400 ("Unsupported parameter: 'max_tokens' is not
# supported with this model. Use 'max_completion_tokens' instead."), and
# Moonshot's Kimi lineage deprecates it ("`max_tokens` is deprecated" in favour
# of `max_completion_tokens`). A client routing through the gateway (OpenCode /
# Codex / any OpenAI-wire harness) still sends `max_tokens`, so we rename it for
# the named families — model-aware, never a global rename: gpt-4* / o-series
# keep `max_tokens`. The model set has ONE home, lib/hal/model_quirks.py
# (HAL#547); this name is kept so the gateway's callers and tests read the same
# tuple object, never a private copy that can drift.
_MAX_COMPLETION_TOKENS_MODELS = _QUIRKS_MAX_COMPLETION_TOKENS_MODELS


def _model_requires_max_completion_tokens(model: str) -> bool:
    """Whether ``model`` takes ``max_completion_tokens`` rather than the legacy
    ``max_tokens`` — OpenAI gpt-5.x (rejects it) and the Kimi lineage
    (deprecates it). Delegates to :func:`model_quirks.uses_max_completion_tokens`,
    which tolerates a provider prefix (``openai/``, ``moonshot/``), any case and
    a trailing ``[1m]``-style suffix. ``gpt-4*`` / ``o3`` / non-gpt, non-Kimi
    models return False. NOT the predicate behind the reasoning_effort pin —
    see :func:`_model_is_gpt5`.
    """
    return _quirks_uses_max_completion_tokens(model)


def _remap_max_tokens_to_completion(obj: dict) -> dict:
    """Rename ``max_tokens`` -> ``max_completion_tokens`` in an OpenAI request
    dict iff its model is a gpt-5.x variant. Pure: returns ``obj`` unchanged when
    not applicable (no model match, or no ``max_tokens`` present), mutating in
    place otherwise. If ``max_completion_tokens`` is already set, the legacy
    ``max_tokens`` is dropped rather than overwriting the explicit value."""
    if (isinstance(obj, dict) and "max_tokens" in obj
            and _model_requires_max_completion_tokens(obj.get("model", ""))):
        legacy = obj.pop("max_tokens")
        obj.setdefault("max_completion_tokens", legacy)
    return obj


def _model_is_gpt5(model: str) -> bool:
    """Whether ``model`` is an OpenAI gpt-5.x variant (prefix match; provider
    prefix and case tolerated). The ONLY predicate behind the reasoning_effort
    pin below. It is deliberately NOT :func:`_model_requires_max_completion_tokens`:
    that set is shared with the Kimi lineage (model_quirks, HAL#547), and the
    pin is OpenAI's prescription for gpt-5.x alone — on Moonshot's platform API
    ``reasoning_effort`` accepts only ``low`` / ``high`` / ``max`` (anything
    else is HTTP 400), and on the Kimi Code host ``none`` is the thinking-off
    switch that silently routes K3 to K2.6."""
    name = str(model or "").split("/")[-1].lower()
    return name.startswith("gpt-5")


def _pin_reasoning_effort_none_with_tools_for_gpt5(obj: dict) -> dict:
    """OpenAI gpt-5.x REJECTS (HTTP 400) function `tools` combined with
    `reasoning_effort` on /v1/chat/completions ("... Please use /v1/responses
    instead"), and an ABSENT reasoning_effort is not safe either — gpt-5.x
    DEFAULTS to medium, which trips the same 400 (live-verified 2026-08-25:
    tools-only requests 400 until reasoning_effort is explicitly 'none').
    Agentic clients (Pi/OpenCode/Codex) always send tools; tools are
    load-bearing (the agent loop), reasoning_effort is the droppable knob — so
    PIN it to 'none' (OpenAI's own prescription), ONLY when tools are present
    on a gpt-5.x model (:func:`_model_is_gpt5`, never the shared
    max_completion_tokens set). Pure: mutates in place, returns obj — a no-op
    for non-gpt5 models (Kimi included), tools-absent requests, and empty
    tools lists."""
    if (isinstance(obj, dict)
            and obj.get("tools")
            and _model_is_gpt5(obj.get("model", ""))):
        obj["reasoning_effort"] = "none"
    return obj


def _model_is_gpt6(model: str) -> bool:
    """Whether ``model`` is an OpenAI gpt-6 variant (gpt-6-astra). Delegates to
    :func:`model_quirks.is_gpt6_family`. Deliberately disjoint from
    :func:`_model_is_gpt5`: the gpt-5 reasoning_effort='none' pin is a
    guaranteed HTTP 400 on Astra, which refuses the ``none`` level."""
    return _quirks_is_gpt6_family(model)


def _fix_gpt6_request_params(obj: dict) -> dict:
    """OpenAI GPT-6 (Astra) body fixes, in place, gpt-6 models ONLY: drop the
    sampling fields the model does not accept (temperature / top_p / logprobs /
    top_logprobs — model_quirks.GPT6_LOCKED_SAMPLING_PARAMS, so OpenAI applies
    its own defaults) and drop a ``reasoning_effort`` the model refuses
    (``none``); never pin anything. Tools are left untouched: on gpt-6 they
    must travel on /v1/responses -- an Anthropic-wire client (Claude Code) is
    carried there by :func:`cross_wire_responses_forward`, a Responses-wire
    client by the verbatim passthrough, and a chat/completions client sending
    them gets OpenAI's own clear 400 rather than a silently rewritten
    request. Pure: a no-op for every other model."""
    if not (isinstance(obj, dict) and _model_is_gpt6(obj.get("model", ""))):
        return obj
    model = obj.get("model", "")
    for locked in _quirks_locked_sampling_params(model):
        obj.pop(locked, None)
    effort = str(obj.get("reasoning_effort", "") or "").strip().lower()
    if effort and effort in _quirks_refused_reasoning_efforts(model):
        obj.pop("reasoning_effort", None)
    return obj


def _fix_gpt5_request(obj: dict) -> dict:
    """Apply the OpenAI-egress body fixes in place: the
    max_tokens->max_completion_tokens rename (gpt-5.x, gpt-6 AND the Kimi
    lineage, per model_quirks), the reasoning_effort/tools 400 fix (pin
    reasoning_effort to 'none' when tools are present — gpt-5.x ONLY) and the
    gpt-6 branch (:func:`_fix_gpt6_request_params`: drop locked sampling fields
    and a refused 'none' effort — gpt-6 ONLY). One composite so same-wire and
    cross-wire forwarders stay symmetric and DRY. Pure: a no-op for requests
    no rule matches. The name predates the gpt-6 branch and is kept because
    callers and tests import it."""
    _remap_max_tokens_to_completion(obj)
    _pin_reasoning_effort_none_with_tools_for_gpt5(obj)
    _fix_gpt6_request_params(obj)
    return obj


def _text_of_openai_content(content) -> str:
    """Best-effort text rendering for OpenAI chat content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return "" if content is None else str(content)


def _textify_tool_call_lines(msg: dict, tool_calls: list) -> list[str]:
    """Render an assistant message's content + tool calls as text lines."""
    lines = []
    text = _text_of_openai_content(msg.get("content"))
    if text:
        lines.append(text)
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        name = fn.get("name") or call.get("name") or "tool"
        call_id = call.get("id") or ""
        args = fn.get("arguments") or "{}"
        suffix = f" id={call_id}" if call_id else ""
        lines.append(f"[tool call {name}{suffix}]\n{args}")
    return lines


def _rewrite_assistant_tool_history(msg: dict) -> tuple[dict, bool]:
    """Textify an assistant tool_calls message unless every call is signed.

    If the signature-cache layer re-injected every Gemini signature, preserve
    the native tool-call history unchanged. Only fall back to text when at
    least one call remains unsigned and would trigger Google's 400.
    Returns (message, changed).
    """
    tool_calls = msg.get("tool_calls") or []
    if all(_gemini_tool_call_signature(call) for call in tool_calls if isinstance(call, dict)):
        return msg, False
    clean = {k: v for k, v in msg.items() if k not in ("tool_calls", "function_call")}
    clean["content"] = "\n".join(_textify_tool_call_lines(msg, tool_calls))
    return clean, True


def _rewrite_tool_result(msg: dict) -> dict:
    """Map an OpenAI tool-result message onto a plain-text user message."""
    call_id = msg.get("tool_call_id") or ""
    name = msg.get("name") or "tool"
    suffix = f" id={call_id}" if call_id else ""
    return {
        "role": "user",
        "content": f"[tool result {name}{suffix}]\n{_text_of_openai_content(msg.get('content'))}",
    }


def _stringify_gemini_tool_history(obj: dict) -> dict:
    """Avoid unsigned Gemini functionCall history on Google's OpenAI shim.

    Gemini 3.x requires every prior native ``functionCall`` part to carry the
    provider-issued ``thought_signature``. OpenAI-wire clients only replay plain
    ``tool_calls``/``tool`` messages, so Google's compatibility shim maps them
    to unsigned functionCall parts and rejects the next turn. For Gemini egress
    only, preserve that history as text while keeping top-level ``tools`` intact
    so the current turn can still request new tools.
    """
    messages = obj.get("messages")
    if not isinstance(messages, list):
        return obj
    rewritten = []
    changed = False
    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else None
        if role == "assistant" and msg.get("tool_calls"):
            new_msg, msg_changed = _rewrite_assistant_tool_history(msg)
            rewritten.append(new_msg)
            changed = changed or msg_changed
        elif role == "tool":
            rewritten.append(_rewrite_tool_result(msg))
            changed = True
        else:
            rewritten.append(msg)
    if changed:
        obj["messages"] = rewritten
    return obj


def _fix_gemini_request(obj: dict) -> dict:
    """Apply Gemini-specific OpenAI-compatible egress fixes in place."""
    return _stringify_gemini_tool_history(obj)


def _gemini_tool_call_signature(call: dict) -> str:
    """Return a Gemini thought_signature from an OpenAI-compatible tool_call."""
    extra = call.get("extra_content") if isinstance(call, dict) else None
    google = extra.get("google") if isinstance(extra, dict) else None
    sig = google.get("thought_signature") if isinstance(google, dict) else None
    return sig if isinstance(sig, str) and sig else ""


def _to_upstream_response(resp) -> UpstreamResponse:
    """Normalise an opener's response object into an :class:`UpstreamResponse`.

    Tolerates ``.status`` (3.8) or ``.code`` (HTTPError on 3.9+) and a urllib
    message OR a plain ``(k, v)`` header iterable — the same shapes the rest of
    the gateway accepts.
    """
    raw_headers = resp.headers
    items = raw_headers.items() if hasattr(raw_headers, "items") else raw_headers
    return UpstreamResponse(
        status=int(getattr(resp, "status", None) or getattr(resp, "code", 0) or 0),
        headers=list(items),
        _reader=resp,
    )


def _wire_url(provider, gateway_path, binding):
    if binding is None:
        return build_egress_url(provider, gateway_path)
    from lib.hal.cascade_target import is_valid
    if (binding.provider != provider or not is_valid(binding) or
            gateway_path != "/v1/chat/completions"):
        raise ValueError("invalid bound destination")
    suffix = gateway_path[3:] if gateway_path.startswith("/v1") else gateway_path
    return binding.base_url.rstrip("/") + suffix


def _wire_body(provider, raw_body, is_local):
    if is_local:
        body = _fix_cloud_body(raw_body, _drop_local_unsupported)
    else:
        body = _fix_cloud_body_temperature(raw_body)
        body = _fix_cloud_body(body, _fix_gpt5_request)
        if provider == "gemini":
            body = _fix_cloud_body(body, _fix_gemini_request)
    return _fix_cloud_body(body, _provider_model_rewriter(provider))


def _check_bound_model(body, binding, *, outgoing=False):
    if binding is None:
        return
    try:
        model = json.loads(body)["model"]
    except (ValueError, TypeError, KeyError):
        raise ValueError("invalid bound model") from None
    expected = {binding.model} if outgoing else {binding.model, binding.label}
    if not isinstance(model, str) or model not in expected:
        raise ValueError("invalid bound model")


def _wire_key(provider, binding, resolver):
    selector = binding.credential if binding else PROVIDER_ENDPOINTS[provider][1]
    key = resolver(selector)
    if not key:
        raise RuntimeError(
            f"no API key for same-wire provider {provider!r} (keychain service "
            f"{selector!r}) — set it or unset HAL_GATEWAY_MULTIPROVIDER")
    return key


def forward_same_wire(
    provider: str,
    gateway_path: str,
    raw_body: Optional[bytes],
    *,
    opener: Optional[Callable] = None,
    key_resolver: KeyResolver = _default_key_resolver,
    binding=None,
) -> UpstreamResponse:
    """Send an OpenAI-wire request through its selected provider or binding."""
    is_local = provider in LOCAL_PROVIDER_ENDPOINTS
    if not is_local and provider not in PROVIDER_ENDPOINTS:
        raise ValueError(f"no same-wire endpoint for provider {provider!r}")
    url = _wire_url(provider, gateway_path, binding)
    require_openai_destination(url)
    _check_bound_model(raw_body, binding)
    out_body = _wire_body(provider, raw_body, is_local)
    _check_bound_model(out_body, binding, outgoing=True)
    op = opener or build_default_opener(subscription_bound=binding is not None)
    base_headers: List[Tuple[str, str]] = [
        ("Content-Type", "application/json"),
        ("User-Agent", _USER_AGENT),
    ]
    if is_local:
        return _to_upstream_response(op("POST", url, base_headers, out_body))

    key = _wire_key(provider, binding, key_resolver)
    headers = [("Authorization", f"Bearer {key}")] + base_headers
    return _to_upstream_response(op("POST", url, headers, out_body))


# --- Anthropic same-wire: Anthropic-wire client -> Anthropic-compatible provider

# Client credential headers that must NEVER reach a third-party provider. The
# verbatim GatewayProxy keeps them because its upstream IS Anthropic; here the
# upstream is another vendor, and the operator's Claude OAuth token / API key
# would otherwise be forwarded to it.
_CLIENT_CREDENTIAL_HEADERS = frozenset({"authorization", "x-api-key"})


class AnthropicCompatProxy(GatewayProxy):
    """:class:`GatewayProxy` pointed at a provider's Anthropic-compatible base.

    Inherits ``forward``'s byte-identical retry on 408/429/5xx (Moonshot's
    Tier0 allows 3 requests a minute, so 429s are routine) and its hop-by-hop
    header filter. Overrides only the credential: the client's
    ``authorization`` and ``x-api-key`` are DROPPED before the parent filter
    runs (so nothing is left for its Bearer-to-x-api-key translation to touch)
    and the provider's own Bearer key, from ``key_resolver``, is added. Every
    other filtered client header -- ``anthropic-version``, ``anthropic-beta``,
    ``content-type``, the harness's ``user-agent`` -- passes through, so the
    vendor sees what the harness sent. The key is resolved once per request and
    never logged; an empty key fails loud rather than sending a silent
    ``Bearer `` that 401s (same contract as :func:`forward_same_wire`).
    """

    #: The upstream is another vendor, billed on that vendor's own key for a
    #: model the caller named; the spend guard reads Anthropic's credentials
    #: only (``lib/hal/spend_guard.py``, "What this guard does not cover").
    checks_anthropic_spend = False

    def __init__(self, provider: str, *, opener: Optional[Callable] = None,
                 key_resolver: KeyResolver = _default_key_resolver,
                 max_retries: Optional[int] = None,
                 sleep: Optional[Callable[[float], None]] = None):
        if provider not in ANTHROPIC_COMPAT_ENDPOINTS:
            raise ValueError(f"no Anthropic-compatible endpoint for provider {provider!r}")
        base, service = ANTHROPIC_COMPAT_ENDPOINTS[provider]
        extra: dict = {}
        if max_retries is not None:
            extra["max_retries"] = max_retries
        if sleep is not None:
            extra["sleep"] = sleep
        super().__init__(upstream_base=base, opener=opener, **extra)
        self.provider = provider
        self._service = service
        self._key_resolver = key_resolver

    def upstream_headers(self, client_headers) -> List[Tuple[str, str]]:
        key = self._key_resolver(self._service)
        if not key:
            raise RuntimeError(
                f"no API key for Anthropic-compatible provider {self.provider!r} "
                f"(keychain service {self._service!r}) -- set it or unset {_FLAG}"
            )
        anonymous = [(n, v) for n, v in client_headers
                     if n.lower() not in _CLIENT_CREDENTIAL_HEADERS]
        kept = super().upstream_headers(anonymous)
        if not any(n.lower() == "user-agent" for n, _v in kept):
            # Cloudflare bans urllib's default UA (groq, 2026-06-11); a harness
            # that sent its own keeps it.
            kept.append(("User-Agent", _USER_AGENT))
        return [("Authorization", f"Bearer {key}")] + kept


def anthropic_same_wire_forward(
    provider: str,
    gateway_path: str,
    raw_body: Optional[bytes],
    client_headers,
    *,
    opener: Optional[Callable] = None,
    key_resolver: KeyResolver = _default_key_resolver,
    sleep: Optional[Callable[[float], None]] = None,
) -> UpstreamResponse:
    """Forward an Anthropic-wire body to ``provider``'s Anthropic-compatible
    endpoint -- same wire in, same wire out; the caller relays the response.

    The only body change is the model id, spelled as the vendor knows it: the
    ``<provider>/`` head and a ``[1m]``-style capability suffix are removed by
    :func:`_provider_model_rewriter` (``kimi-k3[1m]`` is HTTP 404 at Moonshot,
    probed 2026-09-07). ``thinking``, ``tools``, ``cache_control``,
    ``output_config`` and everything else travel untouched, and a body that
    needs no rewrite is forwarded byte-identical. The rewrite is computed ONCE
    before the retry loop, so a retry re-sends the identical bytes. The gateway
    path is appended to the base verbatim, query string included.
    """
    out_body = _fix_cloud_body(raw_body, _provider_model_rewriter(provider))
    proxy = AnthropicCompatProxy(
        provider, opener=opener, key_resolver=key_resolver, sleep=sleep)
    return proxy.forward(gateway_path, out_body, client_headers)


# Vendor tokenizer endpoints that stand in for ``POST /v1/messages/count_tokens``
# on an Anthropic-compatible provider whose Anthropic surface has none.
# Moonshot: ``/anthropic/v1/messages/count_tokens`` is HTTP 404 ``url.not_found``
# (probed 2026-09-07); its free tokenizer takes a Chat-shaped body and counts
# tools too (172 vs 102 tokens with/without on the probe). Keyed by provider so
# a future Anthropic-compatible row without a tokenizer answers 501 instead of
# a misrouted count.
_TOKEN_COUNT_ENDPOINTS: dict = {
    "moonshot": "/v1/tokenizers/estimate-token-count",
}


def anthropic_count_tokens(
    payload: dict,
    *,
    provider: str,
    opener: Optional[Callable] = None,
    key_resolver: KeyResolver = _default_key_resolver,
) -> UpstreamResponse:
    """Answer an Anthropic ``count_tokens`` request from ``provider``'s tokenizer.

    The Anthropic body is translated to the OpenAI Chat shape by the existing
    :func:`gateway_crosswire.anthropic_to_openai_request` (``system`` becomes a
    system message, content blocks become text, ``tools`` become functions) and
    only ``model``, ``messages`` and ``tools`` are sent; :func:`forward_same_wire`
    supplies the provider key, the ``[1m]`` strip and the URL. ``thinking`` and
    cache fields are not part of the count: the answer is the vendor's estimate
    over the translated text, not Anthropic's own accounting. A vendor error
    keeps its status inside an Anthropic error envelope; an unrecognisable 2xx
    body is a 502, never a fabricated count; a provider with no tokenizer row
    is a 501.
    """
    path = _TOKEN_COUNT_ENDPOINTS.get(provider)
    if path is None:
        return _synthetic_json_response(501, _anthropic_error_envelope(
            501, f"provider {provider!r} has no token-count endpoint in this gateway".encode()))
    from lib.hal.gateway_crosswire import anthropic_to_openai_request
    translated = anthropic_to_openai_request(payload)
    body = {k: translated[k] for k in ("model", "messages", "tools") if k in translated}
    resp = forward_same_wire(provider, path, json.dumps(body).encode("utf-8"),
                             opener=opener, key_resolver=key_resolver)
    if not (200 <= resp.status < 300):
        return _provider_error_reply(resp)
    raw = resp.read_all()
    resp.close()
    total = _total_tokens(raw)
    if total is None:
        return _synthetic_json_response(502, _anthropic_error_envelope(502, (
            f"provider returned an unrecognised token-count body "
            f"(upstream status {resp.status})").encode()))
    return _synthetic_json_response(200, json.dumps({"input_tokens": total}).encode("utf-8"))


def _total_tokens(raw: bytes) -> Optional[int]:
    """``data.total_tokens`` of a Moonshot estimate body, or ``None``."""
    try:
        total = json.loads(raw)["data"]["total_tokens"]
    except (ValueError, TypeError, KeyError):
        return None
    return total if isinstance(total, int) and not isinstance(total, bool) else None


# --- cross-wire: Anthropic-wire client -> OpenAI-family provider -------------

def _chat_wire_body(payload, want_stream):
    from lib.hal.gateway_crosswire import anthropic_to_openai_request
    request = anthropic_to_openai_request(payload)
    request["stream"] = bool(want_stream)
    if want_stream:
        request["stream_options"] = {"include_usage": True}
    _drop_temperature_for_fixed_temp_model(request)
    _fix_gpt5_request(request)
    return json.dumps(request).encode("utf-8")


def _chat_wire_reply(resp, payload, want_stream):
    if not 200 <= resp.status < 300:
        return _provider_error_reply(resp)
    if want_stream and _is_event_stream(resp.headers):
        from lib.hal.gateway_stream_crosswire import translate_chat_stream
        translated = translate_chat_stream(resp.iter_chunks(), model=payload.get("model", ""))
        return _streaming_sse_response(translated, resp)
    from lib.hal.gateway_crosswire import openai_to_anthropic_response
    return _buffered_anthropic_reply(
        resp, translate=openai_to_anthropic_response,
        model=payload.get("model", ""), want_stream=want_stream)


def cross_wire_forward(
    payload: dict,
    *,
    provider: str,
    want_stream: bool,
    opener: Optional[Callable] = None,
    key_resolver: KeyResolver = _default_key_resolver,
    binding=None,
) -> UpstreamResponse:
    """Translate Anthropic Messages through a bound OpenAI chat endpoint."""
    if _quirks_tools_require_responses_api(payload.get("model", "")):
        return cross_wire_responses_forward(
            payload, provider=provider, want_stream=want_stream,
            opener=opener, key_resolver=key_resolver, binding=binding)
    body = _chat_wire_body(payload, want_stream)
    resp = forward_same_wire(
        provider, "/v1/chat/completions", body,
        opener=opener, key_resolver=key_resolver, binding=binding,
    )

    return _chat_wire_reply(resp, payload, want_stream)


def cross_wire_responses_forward(
    payload: dict,
    *,
    provider: str,
    want_stream: bool,
    opener: Optional[Callable] = None,
    key_resolver: KeyResolver = _default_key_resolver,
    binding=None,
) -> UpstreamResponse:
    """Translate Anthropic Messages through an OpenAI Responses endpoint."""
    from lib.hal.gateway_crosswire_responses import (
        anthropic_to_responses_request, responses_to_anthropic_response)
    body = json.dumps(anthropic_to_responses_request(payload)).encode("utf-8")
    resp = forward_same_wire(
        provider, "/v1/responses", body, opener=opener, key_resolver=key_resolver, binding=binding)
    if not (200 <= resp.status < 300):
        return _provider_error_reply(resp)
    return _buffered_anthropic_reply(
        resp, translate=responses_to_anthropic_response,
        model=payload.get("model", ""), want_stream=want_stream)


def _provider_error_reply(resp: UpstreamResponse) -> UpstreamResponse:
    """Provider error -> Anthropic-shaped error envelope carrying the real
    upstream status (fail-loud) and its ``Retry-After`` (fleet finding f73).
    Errors are JSON even for a stream request."""
    raw = resp.read_all()
    resp.close()
    return _synthetic_json_response(resp.status, _anthropic_error_envelope(resp.status, raw),
                                    extra_headers=retry_headers(resp.headers))


def _buffered_anthropic_reply(
    resp: UpstreamResponse, *, translate: Callable[..., dict],
    model: str, want_stream: bool,
) -> UpstreamResponse:
    """The whole 2xx provider body is in hand: parse it, translate it with
    ``translate(data, model=model)`` into an Anthropic message, and answer as
    JSON -- or, for a streaming client, as one Anthropic SSE burst so the
    streaming client still works. A 2xx body that is not JSON is a
    gateway-level 502 naming the real upstream status so it is not lost."""
    from lib.hal.gateway_crosswire import (
        anthropic_message_to_sse_chunks, normalise_provider_usage,
    )
    raw = resp.read_all()
    resp.close()
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return _synthetic_json_response(502, _anthropic_error_envelope(
            502, f"provider returned non-JSON body (upstream status {resp.status})".encode("utf-8")))
    anthropic_msg = translate(data, model=model)
    # The provider body is parsed HERE and nowhere later: the witness runs
    # before the relay over a single-pass reader, so the counts have to be
    # carried forward rather than re-read. Same call for a streaming client --
    # the burst is buffered, so its usage is equally in hand.
    usage = normalise_provider_usage(data)
    if want_stream:
        return _synthetic_sse_response(
            anthropic_message_to_sse_chunks(anthropic_msg), usage=usage)
    return _synthetic_json_response(
        200, json.dumps(anthropic_msg).encode("utf-8"), usage=usage)


def _synthetic_json_response(status: int, body: bytes,
                             usage: Optional[dict] = None,
                             extra_headers: List[Tuple[str, str]] = ()) -> UpstreamResponse:
    """A buffered JSON UpstreamResponse backed by an in-memory reader.

    ``usage`` rides along so the witness can report tokens without re-reading
    (and thereby draining) the single-pass body; ``extra_headers`` (a
    provider's ``Retry-After``) ride along to the client.
    """
    return UpstreamResponse(
        status=status,
        headers=[("Content-Type", "application/json"), ("Content-Length", str(len(body))),
                 *extra_headers],
        _reader=io.BytesIO(body),
        usage=usage,
    )


def _synthetic_sse_response(chunks: List[bytes],
                            usage: Optional[dict] = None) -> UpstreamResponse:
    """A buffered text/event-stream UpstreamResponse (no Content-Length — the
    relay sends ``Connection: close`` so the client reads the stream to EOF)."""
    return UpstreamResponse(
        status=200,
        headers=[("Content-Type", "text/event-stream"), ("Cache-Control", "no-cache")],
        _reader=io.BytesIO(b"".join(chunks)),
        usage=usage,
    )


def _streaming_sse_response(chunk_iter, source) -> UpstreamResponse:
    """A LAZY text/event-stream UpstreamResponse: ``chunk_iter`` is pulled on
    demand by the relay (true streaming), and ``source`` (the upstream response)
    is closed when the relay finishes."""
    return UpstreamResponse(
        status=200,
        headers=[("Content-Type", "text/event-stream"), ("Cache-Control", "no-cache")],
        _reader=_GeneratorReader(chunk_iter, source),
    )


def _is_event_stream(headers) -> bool:
    """True if the upstream response declares ``text/event-stream``."""
    for name, value in headers:
        if name.lower() == "content-type" and "event-stream" in str(value).lower():
            return True
    return False


class _GeneratorReader:
    """Adapt an ``Iterator[bytes]`` to the ``.read(size)`` / ``.close()`` reader
    contract :class:`UpstreamResponse` expects, pulling from the generator (and
    thus the upstream socket) lazily so translation stays token-incremental."""

    def __init__(self, chunk_iter, source=None):
        self._it = iter(chunk_iter)
        self._buf = bytearray()
        self._exhausted = False
        self._source = source

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            for chunk in self._it:
                self._buf.extend(chunk)
            self._exhausted = True
        else:
            while len(self._buf) < size and not self._exhausted:
                self._pull()
            data = bytes(self._buf[:size])
            del self._buf[:size]
            return data
        data = bytes(self._buf)
        self._buf.clear()
        return data

    def read1(self, size: int = -1) -> bytes:
        """What is buffered, else the next chunk the stream yields; b"" at the end.
        ``read(size)`` waits for ``size`` bytes, so a translated stream reached
        the client in 64 KiB blocks; this hands each event on as it is made
        (fleet finding f79)."""
        while not self._buf and not self._exhausted:
            self._pull()
        count = len(self._buf) if size is None or size < 0 else min(size, len(self._buf))
        data = bytes(self._buf[:count])
        del self._buf[:count]
        return data

    def _pull(self) -> None:
        try:
            self._buf.extend(next(self._it))
        except StopIteration:
            self._exhausted = True

    def close(self) -> None:
        closer = getattr(self._source, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass


def _anthropic_error_envelope(status: int, raw: bytes) -> bytes:
    """Best-effort map a provider error body to an Anthropic error envelope."""
    detail = ""
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            err = obj.get("error")
            if isinstance(err, dict):
                detail = err.get("message") or ""
            detail = detail or obj.get("message") or ""
    except (ValueError, TypeError):
        pass
    if not detail:
        detail = raw[:200].decode("utf-8", "replace") if raw else f"upstream status {status}"
    return json.dumps({
        "type": "error",
        "error": {"type": "upstream_error", "message": detail},
    }).encode("utf-8")
