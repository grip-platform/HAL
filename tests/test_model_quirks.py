"""lib/hal/model_quirks.py — the single home for Kimi request quirks.

W1 of the Kimi plan (HAPPI core kernel parity). Before this module grew,
the HAPPI kernel 400'd on every Kimi call: ``happi/sampling.py`` pins AGORA
calls to ``temperature: 0.0``, the body carried it, and Moonshot answered
``invalid temperature: only 1 is allowed for this model`` (live, 2026-09-07).

Vendor facts these tests encode (platform.kimi.ai/docs/api/models-overview.md
and guide/use-thinking-models.md, quoted in the module):

* temperature is FIXED; ``top_p`` 0.95, ``n`` 1, ``presence_penalty`` 0,
  ``frequency_penalty`` 0 "Cannot be modified ... passing any other value
  returns an error";
* ``max_tokens`` is deprecated in favour of ``max_completion_tokens``;
* ``kimi-k3`` takes a top-level ``reasoning_effort`` in ``low|high|max``
  (default ``max``); the K2.x line uses the ``thinking`` param, and on
  ``kimi-k2.7-code`` thinking "cannot be turned off".

Every helper must tolerate a ``moonshot/`` prefix and a ``[1m]`` suffix,
because HAL addresses models as ``<provider>/<id>`` and Claude Code appends
``[1m]`` to 1M-context ids.
"""
from __future__ import annotations

import pytest

from lib.hal import model_quirks as mq


# --- FIXED_TEMPERATURE_MODELS stays what PR #514 made it --------------------

def test_fixed_temperature_models_pin():
    # Regression anchor: the temperature lock covers the whole live lineage.
    assert mq.FIXED_TEMPERATURE_MODELS == ("kimi-k2", "kimi-k3")


# --- locked_sampling_params ---------------------------------------------------

def test_locked_sampling_params_constant():
    assert mq.LOCKED_SAMPLING_PARAMS == (
        "temperature", "top_p", "n", "presence_penalty", "frequency_penalty",
    )


@pytest.mark.parametrize("model", [
    "kimi-k3",
    "moonshot/kimi-k3",
    "kimi-k3[1m]",
    "moonshot/kimi-k3[1m]",
    "moonshot/KIMI-K2.7-CODE[1m]",
    "kimi-k2.7-code-highspeed",
    "kimi-k2.6",
])
def test_locked_sampling_params_for_kimi_lineage(model):
    assert mq.locked_sampling_params(model) == mq.LOCKED_SAMPLING_PARAMS


@pytest.mark.parametrize("model", [
    "gpt-5.5", "openai/gpt-4o-mini", "deepseek-chat", "claude-opus-4-8",
    "moonshot-v1-8k", "moonshot/moonshot-v1-auto", "", None,
])
def test_locked_sampling_params_empty_for_everything_else(model):
    assert mq.locked_sampling_params(model) == ()


# --- MAX_COMPLETION_TOKENS_MODELS / uses_max_completion_tokens ---------------

def test_gateway_reads_this_tuple_and_agrees_with_this_predicate():
    # The gateway switched its import here (HAL#548 follow-up). A subset check
    # would now be a tautology (set(x) <= set(x)); identity plus predicate
    # agreement over a matrix can each fail on their own: a re-created tuple
    # breaks the first, a private gateway list breaks the second.
    from lib.hal import gateway_egress
    assert gateway_egress._MAX_COMPLETION_TOKENS_MODELS is mq.MAX_COMPLETION_TOKENS_MODELS
    for model in ("gpt-5.5", "openai/gpt-5-codex", "kimi-k3", "moonshot/kimi-k2.6[1m]",
                  "gpt-4o", "o3", "deepseek-chat", "moonshot-v1-8k", ""):
        assert (gateway_egress._model_requires_max_completion_tokens(model)
                == mq.uses_max_completion_tokens(model)), model


def test_max_completion_tokens_models_names_gpt5_and_kimi():
    assert "gpt-5" in mq.MAX_COMPLETION_TOKENS_MODELS
    assert "kimi-k2" in mq.MAX_COMPLETION_TOKENS_MODELS
    assert "kimi-k3" in mq.MAX_COMPLETION_TOKENS_MODELS


@pytest.mark.parametrize("model", [
    "gpt-5", "gpt-5.5", "gpt-5-codex", "openai/GPT-5.5",
    "kimi-k3", "moonshot/kimi-k3[1m]", "moonshot/kimi-k2.6",
    "kimi-k2.7-code-highspeed[1m]",
])
def test_uses_max_completion_tokens_true(model):
    assert mq.uses_max_completion_tokens(model) is True


@pytest.mark.parametrize("model", [
    "gpt-4o", "gpt-4.1", "o3", "deepseek-chat", "claude-opus-4-8",
    "gemini-3.1-pro-preview", "moonshot-v1-8k", "", None,
])
def test_uses_max_completion_tokens_false(model):
    assert mq.uses_max_completion_tokens(model) is False


# --- reasoning_control -------------------------------------------------------

def test_reasoning_effort_levels_constant():
    # GET /v1/models: kimi-k3 think_efforts {low, high, max}, default max.
    assert mq.REASONING_EFFORT_LEVELS == ("low", "high", "max")
    assert mq.DEFAULT_REASONING_EFFORT == "max"


@pytest.mark.parametrize("model", ["kimi-k3", "moonshot/kimi-k3", "kimi-k3[1m]",
                                   "moonshot/KIMI-K3[1m]"])
def test_reasoning_control_k3_is_reasoning_effort_default_max(model):
    assert mq.reasoning_control(model) == {
        "kind": "reasoning_effort", "default": "max", "forced": False,
    }


@pytest.mark.parametrize("model", ["kimi-k2.7-code", "moonshot/kimi-k2.7-code",
                                   "kimi-k2.7-code-highspeed[1m]"])
def test_reasoning_control_k27_code_thinking_forced_on(model):
    # "Forced on, cannot be turned off": thinking-off 400s with
    # "invalid thinking: only type=enabled is allowed for this model".
    assert mq.reasoning_control(model) == {
        "kind": "thinking", "default": None, "forced": True,
    }


@pytest.mark.parametrize("model", ["kimi-k2.6", "moonshot/kimi-k2.6",
                                   "moonshot/kimi-k2.6[1m]"])
def test_reasoning_control_k26_thinking_optional(model):
    assert mq.reasoning_control(model) == {
        "kind": "thinking", "default": None, "forced": False,
    }


@pytest.mark.parametrize("model", [
    "gpt-5.5", "deepseek-chat", "claude-opus-4-8", "moonshot-v1-8k", "", None,
])
def test_reasoning_control_none_for_non_kimi(model):
    assert mq.reasoning_control(model) == {
        "kind": None, "default": None, "forced": False,
    }


# --- the existing predicate keeps its contract and gains [1m] tolerance ----

@pytest.mark.parametrize("model", ["kimi-k3", "moonshot/kimi-k3[1m]",
                                   "KIMI-K2.6", "kimi-k2.7-code-highspeed"])
def test_model_requires_temperature_one_true(model):
    assert mq.model_requires_temperature_one(model) is True


@pytest.mark.parametrize("model", ["moonshot-v1-8k", "gpt-5.5", "", None])
def test_model_requires_temperature_one_false(model):
    assert mq.model_requires_temperature_one(model) is False


# --- W1b: tool_choice "required" support ------------------------------------
# platform.kimi.ai/docs/api/models-overview.md (2026-09-07): "`kimi-k3`
# supports `auto` / `none` / `required`. `kimi-k2.6` and `kimi-k2.7-code` do
# not support `required` and return an error if it is passed."

@pytest.mark.parametrize("model", [
    "kimi-k2.6", "moonshot/kimi-k2.6", "kimi-k2.7-code", "moonshot/kimi-k2.7-code",
    "kimi-k2.7-code-highspeed", "KIMI-K2.6", "kimi-k2.6[1m]",
])
def test_required_tool_choice_unsupported_on_k2_lineage(model):
    assert mq.supports_required_tool_choice(model) is False


@pytest.mark.parametrize("model", [
    "kimi-k3", "moonshot/kimi-k3", "kimi-k3[1m]",
    "gpt-4o-mini", "deepseek-chat", "openai/gpt-5.2", "", None,
])
def test_required_tool_choice_supported_everywhere_else(model):
    """kimi-k3 and every non-Kimi model: the adapter sends the caller's
    value and lets the provider judge it; only the documented k2 error is
    refused before dispatch."""
    assert mq.supports_required_tool_choice(model) is True

