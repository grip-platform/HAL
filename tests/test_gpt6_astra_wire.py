"""GPT-6 Astra request-shape quirks across every OpenAI-wire egress (2026-09-08).

Vendor facts encoded here, read on 2026-09-08 from
platform.openai.com/docs/models/gpt-6-astra, docs/guides/latest-model and
docs/changelog (Sep 3 release entry):

* "does not support custom temperature or top_p values or log probabilities
  (logprobs)" — the fields are OMITTED so OpenAI applies its own defaults;
* reasoning effort is one of low | medium | high | xhigh | max and the model
  "does not support the none reasoning effort level";
* "tool calling requires the Responses API".

Why a distinct branch: every gpt-5 guard in HAL was a ``startswith("gpt-5")``
prefix, so Astra silently missed the max_tokens rename; and the gateway's
gpt-5 prescription — pin ``reasoning_effort`` to ``none`` when tools are
present — would be a guaranteed HTTP 400 on Astra. The gpt-6 branch therefore
renames ``max_tokens``, drops the locked sampling fields, drops a refused
``none`` effort, and never pins ``none``.

Live wire evidence for this machine (2026-09-08): ``GET /v1/models`` with the
hal-openai key returned 125 ids including ``gpt-6-astra`` (no dated snapshot,
no ``-pro``); Codex CLI 0.148.0 on a ChatGPT Business login answered
``{"type":"error","status":400,"error":{"type":"invalid_request_error",
"message":"The 'gpt-6-astra' model requires a newer version of Codex. Please
upgrade to the latest app or CLI and try again."}}`` — so the API path is
served and the Codex (Pool C) path is client-version gated, not account gated.
Each assertion below was observed red against the pre-change code.
"""
from __future__ import annotations

import json

import pytest

from lib.hal import model_quirks as mq
from lib.hal.gateway_egress import (
    _fix_cloud_body,
    _fix_gpt5_request,
    _model_is_gpt5,
    _model_requires_max_completion_tokens,
)
from lib.hal.happi.providers.openai_compat import _fix_request_body
from lib.hal.tool_runtime import _fix_openai_compat_body

ASTRA_SPELLINGS = ["gpt-6-astra", "openai/gpt-6-astra", "GPT-6-ASTRA", "openai/gpt-6-astra[1m]"]
NOT_GPT6 = ["gpt-5.5", "openai/gpt-5.6-sol", "gpt-4o", "kimi-k3", "claude-opus-4-8", "", None]
TOOL = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]


# --- model_quirks: the single home -------------------------------------------

@pytest.mark.parametrize("model", ASTRA_SPELLINGS)
def test_gpt6_family_predicate_tolerates_prefix_case_and_suffix(model):
    assert mq.is_gpt6_family(model) is True


@pytest.mark.parametrize("model", NOT_GPT6)
def test_gpt6_family_predicate_is_false_elsewhere(model):
    assert mq.is_gpt6_family(model) is False


def test_gpt6_locked_params_are_exactly_the_documented_three_plus_top_logprobs():
    assert mq.GPT6_LOCKED_SAMPLING_PARAMS == ("temperature", "top_p", "logprobs", "top_logprobs")


@pytest.mark.parametrize("model", ASTRA_SPELLINGS)
def test_locked_sampling_params_for_gpt6(model):
    assert mq.locked_sampling_params(model) == mq.GPT6_LOCKED_SAMPLING_PARAMS


def test_kimi_locked_set_is_untouched_by_the_gpt6_branch():
    assert mq.locked_sampling_params("kimi-k3") == mq.LOCKED_SAMPLING_PARAMS
    assert mq.locked_sampling_params("gpt-5.5") == ()


@pytest.mark.parametrize("model", ASTRA_SPELLINGS)
def test_gpt6_takes_max_completion_tokens(model):
    assert mq.uses_max_completion_tokens(model) is True
    assert _model_requires_max_completion_tokens(model) is True


def test_gpt6_reasoning_levels_and_refused_none():
    assert mq.reasoning_effort_levels("gpt-6-astra") == ("low", "medium", "high", "xhigh", "max")
    assert mq.refused_reasoning_efforts("gpt-6-astra") == ("none",)
    assert mq.refused_reasoning_efforts("gpt-5.5") == ()
    assert mq.refused_reasoning_efforts("kimi-k3") == ()
    assert mq.reasoning_effort_levels("kimi-k3") == mq.REASONING_EFFORT_LEVELS
    assert mq.reasoning_effort_levels("gpt-4o") == ()


def test_gpt6_reasoning_control_is_an_optional_top_level_effort_with_no_forced_default():
    assert mq.reasoning_control("gpt-6-astra") == {
        "kind": "reasoning_effort", "default": None, "forced": False}


@pytest.mark.parametrize("model", ASTRA_SPELLINGS)
def test_gpt6_tools_require_the_responses_api(model):
    assert mq.tools_require_responses_api(model) is True


@pytest.mark.parametrize("model", NOT_GPT6)
def test_tools_on_chat_completions_are_fine_elsewhere(model):
    assert mq.tools_require_responses_api(model) is False


# --- gateway egress: the composite fixer --------------------------------------

def test_gpt5_pin_predicate_never_matches_gpt6():
    for m in ASTRA_SPELLINGS:
        assert _model_is_gpt5(m) is False, m


def test_gateway_fixer_on_astra_drops_locked_fields_keeps_effort_renames_max_tokens():
    obj = {"model": "gpt-6-astra", "tools": TOOL, "reasoning_effort": "low",
           "max_tokens": 32, "temperature": 0.2, "top_p": 0.9,
           "logprobs": True, "top_logprobs": 3, "messages": []}
    _fix_gpt5_request(obj)
    for gone in ("temperature", "top_p", "logprobs", "top_logprobs", "max_tokens"):
        assert gone not in obj, gone
    assert obj["max_completion_tokens"] == 32
    assert obj["reasoning_effort"] == "low"      # NOT pinned to none
    assert obj["tools"] == TOOL                  # tools untouched


def test_gateway_fixer_drops_a_refused_none_effort_on_astra_only():
    astra = {"model": "openai/gpt-6-astra", "reasoning_effort": "none", "messages": []}
    _fix_gpt5_request(astra)
    assert "reasoning_effort" not in astra
    five = {"model": "gpt-5.5", "tools": TOOL, "reasoning_effort": "low"}
    _fix_gpt5_request(five)
    assert five["reasoning_effort"] == "none"    # the gpt-5 prescription stands


def test_gateway_fixer_leaves_a_gpt5_temperature_alone():
    five = {"model": "gpt-5.5", "temperature": 0.3, "max_tokens": 8}
    _fix_gpt5_request(five)
    assert five["temperature"] == 0.3 and five["max_completion_tokens"] == 8


def test_verbatim_cloud_body_path_applies_the_gpt6_branch():
    raw = json.dumps({"model": "gpt-6-astra", "temperature": 0.5,
                      "reasoning_effort": "none", "max_tokens": 4}).encode()
    out = json.loads(_fix_cloud_body(raw, _fix_gpt5_request))
    assert out == {"model": "gpt-6-astra", "max_completion_tokens": 4}


# --- HAPPI kernel adapter ------------------------------------------------------

def test_happi_adapter_strips_locked_fields_and_renames_for_astra():
    body = {"model": "openai/gpt-6-astra", "temperature": 0.0, "top_p": 1.0,
            "logprobs": True, "max_tokens": 16, "messages": []}
    out = _fix_request_body(body, {})
    assert out == {"model": "openai/gpt-6-astra", "max_completion_tokens": 16, "messages": []}


def test_happi_adapter_passes_a_caller_effort_and_never_writes_null():
    with_effort = _fix_request_body({"model": "gpt-6-astra", "messages": []}, {"effort": "xhigh"})
    assert with_effort["reasoning_effort"] == "xhigh"
    without = _fix_request_body({"model": "gpt-6-astra", "messages": []}, {})
    assert "reasoning_effort" not in without


def test_happi_adapter_drops_a_refused_none_effort_for_astra():
    out = _fix_request_body({"model": "gpt-6-astra", "messages": []}, {"effort": "none"})
    assert "reasoning_effort" not in out


def test_happi_adapter_gpt5_body_is_unchanged_by_the_gpt6_branch():
    out = _fix_request_body({"model": "gpt-5.5", "temperature": 0.0, "max_tokens": 16}, {})
    assert out == {"model": "gpt-5.5", "temperature": 0.0, "max_completion_tokens": 16}


# --- tool_runtime (the direct-council subprocess path) -------------------------

def test_tool_runtime_fixer_reads_the_shared_table_for_astra():
    out = _fix_openai_compat_body({"model": "gpt-6-astra", "max_tokens": 9,
                                   "temperature": 0.1, "top_p": 0.5, "logprobs": False,
                                   "reasoning_effort": "none"})
    assert out == {"model": "gpt-6-astra", "max_completion_tokens": 9}


def test_tool_runtime_fixer_keeps_gpt5_and_kimi_behaviour():
    assert _fix_openai_compat_body({"model": "gpt-5.5", "max_tokens": 9, "temperature": 0.1}) == {
        "model": "gpt-5.5", "max_completion_tokens": 9, "temperature": 0.1}
    assert _fix_openai_compat_body({"model": "gpt-4o", "max_tokens": 9}) == {
        "model": "gpt-4o", "max_tokens": 9}
