"""GPT-6.1 Terra and GPT-6 Luna accept reasoning_effort none (HAL#830).

Both ids start with "gpt-6", so the family prefix used to give them the gpt-6
refusal of none, and the gateway would have dropped a value the vendor accepts.
Vendor pages (platform.openai.com/docs/models/gpt-6.1-terra.md and gpt-6-luna.md,
read 2026-11-25, checked word for word): "`reasoning.effort` supports `none`,
`low`, `medium` (default), `high`, `xhigh`, and `max`." The GPT-6.2 pages say
"The API rejects requests that set reasoning effort to `none`", so those ids, and
GPT-6 Astra, must keep the refusal.
"""
from __future__ import annotations

import pytest

from lib.hal import model_quirks as mq

NONE_ACCEPTING = [
    "gpt-6.1-terra", "openai/gpt-6.1-terra", "GPT-6.1-TERRA", "openai/gpt-6.1-terra[1m]",
    "gpt-6-luna", "openai/gpt-6-luna", "GPT-6-LUNA", "openai/gpt-6-luna[1m]",
]
NONE_REFUSING = ["gpt-6-astra", "gpt-6.2-astra", "gpt-6.2-sol", "openai/gpt-6.2-sol[1m]"]


@pytest.mark.parametrize("model", NONE_ACCEPTING)
def test_terra_and_luna_refuse_no_reasoning_effort(model):
    assert mq.refused_reasoning_efforts(model) == ()


@pytest.mark.parametrize("model", NONE_ACCEPTING)
def test_terra_and_luna_list_none_among_their_levels(model):
    levels = mq.reasoning_effort_levels(model)
    assert levels == ("none", "low", "medium", "high", "xhigh", "max")


@pytest.mark.parametrize("model", NONE_ACCEPTING)
def test_terra_and_luna_stay_in_the_gpt6_family_for_every_other_quirk(model):
    assert mq.is_gpt6_family(model) is True
    assert mq.locked_sampling_params(model) == mq.GPT6_LOCKED_SAMPLING_PARAMS


@pytest.mark.parametrize("model", NONE_REFUSING)
def test_astra_and_the_6_2_ids_still_refuse_none(model):
    assert mq.refused_reasoning_efforts(model) == ("none",)
    assert "none" not in mq.reasoning_effort_levels(model)


def test_a_model_outside_the_family_is_unaffected():
    assert mq.refused_reasoning_efforts("gpt-5.5") == ()
    assert mq.reasoning_effort_levels("kimi-k3") == mq.REASONING_EFFORT_LEVELS
