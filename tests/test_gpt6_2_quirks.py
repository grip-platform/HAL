"""GPT-6.2 Astra / Sol inherit the gpt-6 family's wire quirks by PREFIX ALONE
(2026-11-24) -- the forward anchor that model_quirks.py's GPT6_MODELS comment
promises.

Both ids start with the literal substring "gpt-6" (gpt-6.2-astra, gpt-6.2-sol),
so every predicate in lib/hal/model_quirks.py that gates on
``is_gpt6_family`` already covers them with NO code change -- this file proves
that claim rather than leaving it implicit, mirroring the coverage
test_gpt6_astra_wire.py already runs for gpt-6-astra itself.

Vendor-confirmed for 6.2 specifically (developers.openai.com/api/docs/models/
gpt-6.2-astra and .../gpt-6.2-sol, read 2026-11-24): both "reject requests
that set reasoning effort to none", and both pages list the same
low/medium/high/xhigh/max set this module already encodes. NOT repeated on
either 6.2 page: the temperature/top_p/logprobs refusal GPT-6 Astra's own page
states -- so GPT6_LOCKED_SAMPLING_PARAMS applying here is INHERITED BY PREFIX,
not independently vendor-confirmed for these two ids (no paid call was made).
"""
from __future__ import annotations

import pytest

from lib.hal import model_quirks as mq

GPT_6_2_IDS = ["gpt-6.2-astra", "gpt-6.2-sol"]
GPT_6_2_SPELLINGS = [
    "gpt-6.2-astra", "openai/gpt-6.2-astra", "GPT-6.2-ASTRA", "openai/gpt-6.2-astra[1m]",
    "gpt-6.2-sol", "openai/gpt-6.2-sol", "GPT-6.2-SOL", "openai/gpt-6.2-sol[1m]",
]


@pytest.mark.parametrize("model", GPT_6_2_SPELLINGS)
def test_gpt6_family_predicate_covers_both_6_2_ids_by_prefix(model):
    assert mq.is_gpt6_family(model) is True


@pytest.mark.parametrize("model", GPT_6_2_IDS)
def test_locked_sampling_params_match_the_gpt6_family_set(model):
    assert mq.locked_sampling_params(model) == mq.GPT6_LOCKED_SAMPLING_PARAMS


@pytest.mark.parametrize("model", GPT_6_2_IDS)
def test_refused_reasoning_efforts_is_none_as_the_vendor_page_confirms(model):
    assert mq.refused_reasoning_efforts(model) == ("none",)


@pytest.mark.parametrize("model", GPT_6_2_IDS)
def test_reasoning_effort_levels_match_the_vendor_pages_exactly(model):
    assert mq.reasoning_effort_levels(model) == ("low", "medium", "high", "xhigh", "max")


@pytest.mark.parametrize("model", GPT_6_2_IDS)
def test_tools_require_the_responses_api(model):
    assert mq.tools_require_responses_api(model) is True


@pytest.mark.parametrize("model", GPT_6_2_IDS)
def test_uses_max_completion_tokens(model):
    assert mq.uses_max_completion_tokens(model) is True


@pytest.mark.parametrize("model", GPT_6_2_IDS)
def test_reasoning_control_is_an_optional_top_level_effort_with_no_forced_default(model):
    assert mq.reasoning_control(model) == {
        "kind": "reasoning_effort", "default": None, "forced": False}


@pytest.mark.parametrize("model", GPT_6_2_IDS)
def test_supports_required_tool_choice(model):
    """Not in the Kimi-K2 unsupported list, so an unknown-to-this-table model's
    tool_choice reaches its provider unjudged -- unchanged by this onboarding,
    asserted so a future edit to REQUIRED_TOOL_CHOICE_UNSUPPORTED_MODELS cannot
    silently start refusing it for these ids too."""
    assert mq.supports_required_tool_choice(model) is True
