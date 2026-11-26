"""Kimi K3.1 Pro and K3.1 Pro Highspeed in HAL (2026-11-24), the pair of GRIP's onboarding.

Vendor facts (platform.kimi.ai/docs/guide/kimi-k3-1-pro-quickstart.md and the
price table embedded in platform.kimi.ai/docs/pricing/chat, both read
2026-11-24): a 1,048,576-token window; always-on thinking with a top-level
reasoning_effort of low, high or max (default max); max_completion_tokens;
temperature, top_p, n, presence_penalty and frequency_penalty "are fixed; omit
them from requests"; tool_choice "required" is supported. Per million tokens,
K3.1 Pro bills $2.00 input, $10.00 output, $0.20 cached input, $2.00 for a
5-minute cache write and $4.00 for a 1-hour one; Highspeed $4.00, $20.00,
$0.20, $4.00 and $8.00. A probe the same day answered 200 from both, the
reply's model field naming exactly the id asked for.

The quirk tests are forward anchors: model_quirks matches the Kimi K3 rules by
the prefix "kimi-k3", which "kimi-k3.1-pro" also carries, so they pass before
this change. Each was proved able to fail by a mutation that narrows the
prefix, so a later "tidy" of a guard cannot drop K3.1 without a red test.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lib.hal import cost_tracker, model_quirks

REPO = Path(__file__).resolve().parents[1]
NEW = ("kimi-k3.1-pro", "kimi-k3.1-pro-highspeed")
PUBLISHED = {
    # (input, output, cached input, 5-minute cache write), USD per million tokens
    "kimi-k3.1-pro": (2.00, 10.00, 0.20, 2.00),
    "kimi-k3.1-pro-highspeed": (4.00, 20.00, 0.20, 4.00),
}


def _moonshot_models(config: str) -> list[str]:
    text = (REPO / "config" / config).read_text(encoding="utf-8")
    block = text.split("- name: moonshot\n", 1)[1].split("\n  - name:", 1)[0]
    line = next(raw for raw in block.splitlines() if raw.strip().startswith("models:"))
    return [m.strip() for m in line.split("[", 1)[1].split("]", 1)[0].split(",")]


@pytest.mark.parametrize("config", ["providers.yaml", "providers.windows.yaml"])
def test_listed_after_every_existing_kimi_id(config):
    models = _moonshot_models(config)
    assert models[-2:] == list(NEW), models


@pytest.mark.parametrize("model", NEW)
def test_catalogued_with_its_one_million_window(model):
    text = (REPO / "config" / "model_catalog.yaml").read_text(encoding="utf-8")
    assert f"- address: moonshot/{model}\n" in text
    row = text.split(f"- address: moonshot/{model}\n", 1)[1].split("\n  - address:", 1)[0]
    assert "context_window: 1048576" in row


@pytest.mark.parametrize("model", NEW)
def test_priced_by_its_own_published_row(model):
    assert cost_tracker._resolve_pricing(model) == PUBLISHED[model]
    assert cost_tracker._resolve_pricing(model) != cost_tracker._PRICING["kimi-k3"]


@pytest.mark.parametrize("model,write_1h", [("kimi-k3.1-pro", '"4"'),
                                           ("kimi-k3.1-pro-highspeed", '"8"')])
def test_vendor_price_file_carries_both_cache_write_rates(model, write_1h):
    text = (REPO / "config" / "model_prices.yaml").read_text(encoding="utf-8")
    row = text.split(f"  {model}:\n", 1)[1].splitlines()[0]
    assert f"cache_write_1h: {write_1h}" in row, row


@pytest.mark.parametrize("model", NEW)
def test_fixed_sampling_parameters_are_omitted(model):
    assert model_quirks.locked_sampling_params(model) == model_quirks.LOCKED_SAMPLING_PARAMS
    assert "temperature" in model_quirks.locked_sampling_params(model)


@pytest.mark.parametrize("model", NEW)
def test_reasoning_effort_is_top_level_with_max_by_default(model):
    control = model_quirks.reasoning_control(model)
    assert control["kind"] == "reasoning_effort" and control["default"] == "max"
    assert model_quirks.reasoning_effort_levels(model) == ("low", "high", "max")


@pytest.mark.parametrize("model", NEW)
def test_uses_max_completion_tokens(model):
    assert model_quirks.uses_max_completion_tokens(model) is True


@pytest.mark.parametrize("model", NEW)
def test_required_tool_choice_is_allowed(model):
    assert model_quirks.supports_required_tool_choice(model) is True
