"""Model-specific request quirks shared across HAL's egress surfaces.

Single source of truth (YSH, N=3) for provider models that LOCK sampling
params or need a request field renamed. Consumers: lib/hal/providers/
openai_compat.py (direct registry/adapter path), lib/hal/gateway_egress.py
(verbatim cloud-body + crosswire fixers) and lib/hal/happi/providers/
openai_compat.py (the HAPPI kernel adapter, `_fix_request_body`). The
subscription path (lib/hal/subscription.py::MoonshotSubscription.chat)
independently forces temperature=1 for kimi-k2* — equivalent wire behaviour,
left untouched.

Moonshot locks temperature on the whole current Kimi lineage: HTTP 400
"invalid temperature: only 1 is allowed for this model" on any other
value. Live-verified: kimi-k2.5 + kimi-k2.6 (2026-07-26), kimi-k2.7-code
(#2209, 2026-06), kimi-k3 (2026-07-26 and again 2026-09-07 through the
HAPPI kernel). moonshot-v1-* were NOT locked but are retired (2026-08-31).
Strategy: OMIT the locked field so Moonshot applies its own default —
model-aware, never a global force.

The vendor's request-shape rules this module encodes, quoted from
platform.kimi.ai/docs/api/models-overview.md and
platform.kimi.ai/docs/guide/use-thinking-models.md (2026-09-07):

* temperature FIXED (kimi-k3 1.0, kimi-k2.7-code 1.0, kimi-k2.6 1.0 thinking /
  0.6 non-thinking) — "Do not pass `temperature` explicitly";
* `top_p` 0.95, `n` 1, `presence_penalty` 0, `frequency_penalty` 0 —
  "Cannot be modified ... passing any other value returns an error";
* "`max_tokens` is deprecated" in favour of `max_completion_tokens`
  (kimi-k3 default 131072, max 1048576);
* kimi-k3 takes a top-level `reasoning_effort` of `low|high|max`, default
  `max` ("Switching levels invalidates prefix-cache hits" — hold it constant
  per session); the K2.x line uses the `thinking` param; on kimi-k2.7-code
  thinking is "Forced on, cannot be turned off" (thinking-off answers 400
  "invalid thinking: only type=enabled is allowed for this model");
  kimi-k2.6 accepts `{"type": "enabled"}` / `{"type": "disabled"}`;
* tool_choice (platform.kimi.ai/docs/api/models-overview.md): "`kimi-k3`
  supports `auto` / `none` / `required`. `kimi-k2.6` and `kimi-k2.7-code` do
  not support `required` and return an error if it is passed."

Every predicate tolerates a provider prefix (`moonshot/`), any case, and a
trailing bracketed capability suffix (`[1m]`) — HAL addresses models as
`<provider>/<id>` and Claude Code appends `[1m]` to 1M-context ids.
"""
from __future__ import annotations

import re

# Prefix match: "kimi-k2" covers k2.5 / k2.6 / k2.7-code / k2.7-code-highspeed.
FIXED_TEMPERATURE_MODELS: tuple[str, ...] = ("kimi-k2", "kimi-k3")

# Sampling fields the Kimi lineage will not let a caller move. Omitting them
# lets Moonshot apply its own fixed values; sending any of them is at best
# redundant and at worst a 400.
LOCKED_SAMPLING_PARAMS: tuple[str, ...] = (
    "temperature", "top_p", "n", "presence_penalty", "frequency_penalty",
)

# Models that reject (OpenAI gpt-5.x: HTTP 400 "Unsupported parameter:
# 'max_tokens'") or deprecate (Moonshot: "`max_tokens` is deprecated") the
# legacy field and take `max_completion_tokens` instead. The gateway's
# lib/hal/gateway_egress.py::_MAX_COMPLETION_TOKENS_MODELS IS this tuple (it
# imports it; tests/test_model_quirks.py pins the identity), and its
# reasoning_effort "none" pin is deliberately keyed on a separate gpt-5-only
# predicate — the pin would be a 400 on the Kimi lineage.
MAX_COMPLETION_TOKENS_MODELS: tuple[str, ...] = ("gpt-5", "gpt-6", "kimi-k2", "kimi-k3")

# OpenAI GPT-6 family (gpt-6-astra, released 2026-09-03). Quoted from
# platform.openai.com/docs/models/gpt-6-astra and docs/guides/latest-model
# (read 2026-09-08): the model "does not support custom temperature or top_p
# values or log probabilities (logprobs)" -- the fields are OMITTED so OpenAI
# applies its own defaults; reasoning effort is one of low | medium | high |
# xhigh | max and it "does not support the none reasoning effort level"; and
# "tool calling requires the Responses API" (tools on /v1/chat/completions
# answer HTTP 400, so the Responses passthrough is the tool route). gpt-6 is
# listed in MAX_COMPLETION_TOKENS_MODELS above because max_completion_tokens
# is accepted by every OpenAI chat model since o1 (the rename can never 400)
# while max_tokens 400s on the gpt-5 lineage Astra succeeds; whether Astra
# itself rejects max_tokens is unverified -- no paid completion was run.
# GPT-6.2 Astra / Sol, launched 2026-11-17 (onboarded 2026-11-24, dogfood-domain
# family 8): both start with the "gpt-6" prefix below, so every predicate in
# this module already covers them with no code change. Checked against
# developers.openai.com/api/docs/models/gpt-6.2-astra (and -sol), read
# 2026-11-24: both refuse the `none` reasoning effort (quoted: "The API
# rejects requests that set reasoning effort to none") and both pages confirm
# the same low/medium/high/xhigh/max set used below. Neither 6.2 page repeats
# the temperature/top_p/logprobs refusal GPT-6 Astra's own page states, so
# GPT6_LOCKED_SAMPLING_PARAMS applying to 6.2 is INHERITED-BY-PREFIX, not
# independently vendor-confirmed for these two ids (no paid call was made to
# check). tests/test_gpt6_2_quirks.py pins the prefix coverage.
GPT6_MODELS: tuple[str, ...] = ("gpt-6",)
GPT6_LOCKED_SAMPLING_PARAMS: tuple[str, ...] = (
    "temperature", "top_p", "logprobs", "top_logprobs",
)
GPT6_REASONING_EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")
GPT6_REFUSED_REASONING_EFFORTS: tuple[str, ...] = ("none",)
TOOLS_REQUIRE_RESPONSES_MODELS: tuple[str, ...] = GPT6_MODELS
# GPT-6.1 Terra and GPT-6 Luna also start with "gpt-6", but they ACCEPT the none
# effort (HAL#830). Quoted from platform.openai.com/docs/models/gpt-6.1-terra.md and
# .../gpt-6-luna.md, read 2026-11-25 and checked word for word with
# `grip-quote-check find`: "`reasoning.effort` supports `none`, `low`, `medium`
# (default), `high`, `xhigh`, and `max`." For these two the prefix's refusal is
# lifted and none joins the levels. Neither id is onboarded yet; this stops the
# prefix from dropping a valid value the day one is. Their other gpt-6 quirks still
# apply by prefix, unchecked for these ids, as GPT-6.2's do above.
GPT6_NONE_ACCEPTING_MODELS: tuple[str, ...] = ("gpt-6.1-terra", "gpt-6-luna")
GPT6_NONE_ACCEPTING_LEVELS: tuple[str, ...] = ("none",) + GPT6_REASONING_EFFORT_LEVELS

# GET https://api.moonshot.ai/v1/models (2026-09-07): kimi-k3
# `think_efforts {low, high, max}`, default `max`.
REASONING_EFFORT_LEVELS: tuple[str, ...] = ("low", "high", "max")
DEFAULT_REASONING_EFFORT = "max"

# Models that reject `tool_choice: "required"` (see the module docstring).
# kimi-k3 is NOT listed: it takes auto/none/required. The function-specific
# form ({"type":"function","function":{"name":..}}) is deliberately not
# encoded here -- the vendor documents it as incompatible with thinking, but
# that is a per-request condition the provider reports itself (HTTP 400
# "tool_choice 'specified' is incompatible with thinking enabled").
REQUIRED_TOOL_CHOICE_UNSUPPORTED_MODELS: tuple[str, ...] = ("kimi-k2",)

_BRACKET_SUFFIX_RE = re.compile(r"\[[^\]]*\]$")


def _bare_model(model: object) -> str:
    """``moonshot/KIMI-K3[1m]`` -> ``kimi-k3``: drop the provider head, a
    trailing ``[...]`` suffix, and case. Pure; ``None``/"" -> ""."""
    name = str(model or "").split("/")[-1].strip().lower()
    return _BRACKET_SUFFIX_RE.sub("", name, count=1)


def _matches(model: object, prefixes: tuple[str, ...]) -> bool:
    name = _bare_model(model)
    return bool(name) and any(name.startswith(p) for p in prefixes)


def model_requires_temperature_one(model: str) -> bool:
    """Whether ``model`` only accepts temperature=1 (Moonshot Kimi lineage)."""
    return _matches(model, FIXED_TEMPERATURE_MODELS)


def is_gpt6_family(model: object) -> bool:
    """Whether ``model`` is an OpenAI gpt-6 variant (``gpt-6-astra``; provider
    prefix, case and a ``[1m]`` suffix tolerated)."""
    return _matches(model, GPT6_MODELS)


def locked_sampling_params(model: str) -> tuple[str, ...]:
    """The request fields to OMIT for ``model``: :data:`LOCKED_SAMPLING_PARAMS`
    for the Kimi lineage, :data:`GPT6_LOCKED_SAMPLING_PARAMS` for the gpt-6
    family, ``()`` for every other model."""
    if model_requires_temperature_one(model):
        return LOCKED_SAMPLING_PARAMS
    if is_gpt6_family(model):
        return GPT6_LOCKED_SAMPLING_PARAMS
    return ()


def _accepts_none_effort(model: object) -> bool:
    """Whether ``model`` is one of the gpt-6 ids that accept the none effort (HAL#830)."""
    return _matches(model, GPT6_NONE_ACCEPTING_MODELS)


def refused_reasoning_efforts(model: object) -> tuple[str, ...]:
    """Reasoning-effort levels ``model`` answers HTTP 400 to: ``("none",)`` for
    the gpt-6 family except GPT-6.1 Terra and GPT-6 Luna, which accept it
    (HAL#830), ``()`` everywhere else. A consumer DROPS a refused level (the
    vendor default then applies) rather than forwarding a guaranteed 400."""
    if is_gpt6_family(model) and not _accepts_none_effort(model):
        return GPT6_REFUSED_REASONING_EFFORTS
    return ()


def reasoning_effort_levels(model: object) -> tuple[str, ...]:
    """The top-level ``reasoning_effort`` values ``model`` documents: the gpt-6
    five, plus none for Terra and Luna (HAL#830), kimi-k3's
    :data:`REASONING_EFFORT_LEVELS`, ``()`` for a model with no such knob."""
    if _accepts_none_effort(model):
        return GPT6_NONE_ACCEPTING_LEVELS
    if is_gpt6_family(model):
        return GPT6_REASONING_EFFORT_LEVELS
    if _bare_model(model).startswith("kimi-k3"):
        return REASONING_EFFORT_LEVELS
    return ()


def tools_require_responses_api(model: object) -> bool:
    """Whether function tools for ``model`` must travel on /v1/responses
    (gpt-6: tools on /v1/chat/completions are refused)."""
    return _matches(model, TOOLS_REQUIRE_RESPONSES_MODELS)


def uses_max_completion_tokens(model: str) -> bool:
    """Whether ``model`` takes ``max_completion_tokens`` rather than the legacy
    ``max_tokens`` (OpenAI gpt-5.x rejects it; Moonshot deprecates it)."""
    return _matches(model, MAX_COMPLETION_TOKENS_MODELS)


def reasoning_control(model: str) -> dict[str, object]:
    """How ``model`` exposes its reasoning knob, if at all.

    Returns ``{"kind", "default", "forced"}``:

    * kimi-k3 -> ``kind="reasoning_effort"``, ``default="max"`` (top-level
      field, one of :data:`REASONING_EFFORT_LEVELS`);
    * kimi-k2.7-code / -highspeed -> ``kind="thinking"``, ``forced=True``:
      thinking cannot be disabled, so a caller's ``thinking`` must never be
      sent (it can only 400);
    * kimi-k2.6 -> ``kind="thinking"``, ``forced=False``: optional, send only
      what the caller asked for;
    * gpt-6 (Astra) -> ``kind="reasoning_effort"``, ``default=None``: an
      optional top-level field sent only when the caller asks for a level the
      model accepts (``none`` is refused except by Terra and Luna -- see
      :func:`refused_reasoning_efforts`); nothing is forced;
    * anything else -> ``kind=None``.
    """
    name = _bare_model(model)
    if name.startswith(GPT6_MODELS):
        return {"kind": "reasoning_effort", "default": None, "forced": False}
    if name.startswith("kimi-k3"):
        return {"kind": "reasoning_effort", "default": DEFAULT_REASONING_EFFORT,
                "forced": False}
    if name.startswith("kimi-k2"):
        forced = name.startswith("kimi-k2.7-code")
        return {"kind": "thinking", "default": None, "forced": forced}
    return {"kind": None, "default": None, "forced": False}


def supports_required_tool_choice(model: object) -> bool:
    """Whether ``model`` accepts ``tool_choice: "required"``.

    ``False`` only for the Kimi K2 lineage (``kimi-k2.6``, ``kimi-k2.7-code``,
    ``kimi-k2.7-code-highspeed``), where the vendor documents an error;
    ``True`` for kimi-k3 and for every model this table knows nothing about,
    so an unknown model's value reaches its provider unjudged.
    """
    return not _matches(model, REQUIRED_TOOL_CHOICE_UNSUPPORTED_MODELS)
