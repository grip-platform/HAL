"""Qwen4-Max (flagship) and Qwen3.8-Flash (workhorse) in HAL (2026-11-24),
the pair of GRIP's onboarding. dogfood-domain plan, phase 2 family 4.

Vendor facts, all read 2026-11-24: alibabacloud.com/help/en/model-studio/
qwen4-max ("Context Length" 1000000, "Maximum Output" 131072 non-thinking /
262144 thinking, enable_thinking defaults true); alibabacloud.com/press-room/
alibaba-unveils-qwen4-max (GA "November 16, 2026"; "Input (Implicit Cache Hit)
is priced at $0.075 per 1M tokens"); alibabacloud.com/help/en/model-studio/
billing-for-model-studio (Singapore/International, one tier up to 1M tokens:
qwen4-max $1.5 in / $4.5 out; qwen3.8-flash $0.15 in / $0.47 out -- genuinely
MORE than qwen3.5-flash's $0.10/$0.40, not less). A live probe the same day
answered HTTP 200 for both ids on BOTH DashScope and the Qwen Token Plan, the
reply's `model` field naming exactly the id asked for; a nonsense control id
answered 404 "Model not exist" on both. qwen4-max's default max_tokens budget
can return empty content because enable_thinking's default spends completion
tokens on hidden reasoning first (measured: 7 reasoning_tokens against a
max_tokens of 8); enable_thinking: false removes them.

Unlike GRIP (which has no Token Plan provider), HAL already serves some Qwen
ids from both DashScope (metered, provider `qwen`) and the Token Plan
(subscription, provider `qwen-token-plan`) -- qwen3.8-max already carries both
a `qwen/qwen3.8-max` and a `qwen-token-plan/qwen3.8-max` catalogue row. This
PR mirrors that pattern for both new ids. qwen3.8-flash was already listed in
BOTH providers' `models:` arrays before this PR but had NO catalogue row at
all; this PR fills that pre-existing gap as well as adding qwen4-max.

qwen4-max-preview is deliberately NOT catalogued: its preview window closes
2026-11-25 10:00 UTC+8, a day after this PR. The Token Plan's "auto" is a
router, not a model -- its own page states it "does not support tool_choice"
or a thinking budget, and names no underlying model, so it cannot be priced
or trialled honestly; qwen3.8-flash is the workhorse.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lib.hal import cost_tracker

REPO = Path(__file__).resolve().parents[1]
NEW = ("qwen4-max", "qwen3.8-flash")
PUBLISHED = {
    # (input, output, cache_read, cache_write), USD per million tokens.
    "qwen4-max": (1.50, 4.50, 0.075, 1.50),
    "qwen3.8-flash": (0.15, 0.47, 0.15, 0.15),
}


def _provider_models(config: str, provider: str) -> list[str]:
    text = (REPO / "config" / config).read_text(encoding="utf-8")
    block = text.split(f"- name: {provider}\n", 1)[1].split("\n  - name:", 1)[0]
    line = next(raw for raw in block.splitlines() if raw.strip().startswith("models:"))
    rest = line.split("[", 1)[1]
    # The qwen-token-plan models: array wraps onto a second line.
    if "]" not in rest:
        rest += "\n" + block.split(line, 1)[1].split("]", 1)[0]
    return [m.strip() for m in rest.split("]", 1)[0].replace("\n", " ").split(",")]


def _catalogue_text() -> str:
    return (REPO / "config" / "model_catalog.yaml").read_text(encoding="utf-8")


@pytest.mark.parametrize("model", NEW)
def test_listed_by_the_dashscope_provider(model):
    assert model in _provider_models("providers.yaml", "qwen"), (
        f"{model} is missing from config/providers.yaml's qwen `models:` list")


@pytest.mark.parametrize("model", NEW)
def test_listed_by_the_windows_dashscope_provider(model):
    assert model in _provider_models("providers.windows.yaml", "qwen"), (
        f"{model} is missing from config/providers.windows.yaml's qwen `models:` list")


@pytest.mark.parametrize("model", NEW)
def test_listed_by_the_token_plan_provider(model):
    assert model in _provider_models("providers.yaml", "qwen-token-plan"), (
        f"{model} is missing from config/providers.yaml's qwen-token-plan `models:` list")


def test_qwen_max_qualifier_did_not_move():
    """qwen3.8-max was already first in the qwen-token-plan list; qwen4-max is
    now listed BEFORE it (matching the vendor's own Token Plan page order), but
    this provider has no qualifier-resolution feature (unlike GRIP's
    grip_harness_family), so there is no --qwen-style shortcut to protect --
    only the literal list membership above is a contract."""
    models = _provider_models("providers.yaml", "qwen-token-plan")
    assert models[0] == "qwen4-max", models
    assert "qwen3.8-max" in models


def test_qwen4_max_catalogue_row_declares_vision_modality():
    """AGORA (xAI Grok 4.3, HAL#827): catch a future modality regression on the
    one new row that is genuinely multimodal -- qwen4-max accepts
    text/image/video input per the vendor page; qwen3.8-flash is text-only."""
    row = _catalogue_text().split("- address: qwen/qwen4-max\n", 1)[1].split("\n  - address:", 1)[0]
    assert "modality: text+vision" in row, row


@pytest.mark.parametrize("model", NEW)
def test_catalogued_on_dashscope_with_its_one_million_window(model):
    text = _catalogue_text()
    assert f"- address: qwen/{model}\n" in text
    row = text.split(f"- address: qwen/{model}\n", 1)[1].split("\n  - address:", 1)[0]
    assert "context_window: 1000000" in row


@pytest.mark.parametrize("model", NEW)
def test_catalogued_on_the_token_plan_with_its_subscription_window(model):
    text = _catalogue_text()
    assert f"- address: qwen-token-plan/{model}\n" in text
    row = text.split(f"- address: qwen-token-plan/{model}\n", 1)[1].split("\n  - address:", 1)[0]
    assert "tier: subscription" in row
    assert "context_window: 983616" in row, (
        "the Token Plan window is Alibaba's own Claude Code setup value, not "
        "the raw 1M DashScope window -- see the existing qwen-token-plan/"
        "qwen3.8-max row")


@pytest.mark.parametrize("model", NEW)
def test_priced_by_its_own_published_row(model):
    assert cost_tracker._resolve_pricing(model) == PUBLISHED[model]
    assert cost_tracker._resolve_pricing(model) != cost_tracker._PRICING["qwen3.5-flash"]


def test_qwen3_8_flash_is_not_cheaper_than_qwen3_5_flash():
    """A reviewer could assume 'flash' always means 'the cheapest row'; it does
    not here. Pinned against the PUBLISHED tuple (not just a relative
    comparison): an uncatalogued qwen id falls back to the $3/$15 Sonnet-tier
    default, which is also pricier than qwen3.5-flash, so a relative-only
    check would pass even with no row at all. Mutation: price qwen3.8-flash at
    qwen3.5-flash's row (or delete it) -> RED."""
    flash_35 = cost_tracker._resolve_pricing("qwen3.5-flash")
    flash_38 = cost_tracker._resolve_pricing("qwen3.8-flash")
    assert flash_38 == PUBLISHED["qwen3.8-flash"]
    assert flash_38[0] > flash_35[0] and flash_38[1] > flash_35[1]


def test_model_prices_yaml_carries_both_rows():
    text = (REPO / "config" / "model_prices.yaml").read_text(encoding="utf-8")
    for model, input_price in (("qwen4-max", "1.5"), ("qwen3.8-flash", "0.15")):
        assert f"  {model}:\n" in text, model
        row = text.split(f"  {model}:\n", 1)[1].splitlines()[0]
        assert f'input: "{input_price}"' in row, row
        assert 'effective: "2026-11-24"' in row, row


def test_no_routing_chain_or_council_seat_names_the_new_models():
    """Decision 6 (V>> 2026-11-24): no default, routing chain or council seat
    may name qwen4-max or qwen3.8-flash. HAL's routing.fallback_chain and
    routing.council both still name qwen/qwen3.5-flash and nothing else
    Qwen-shaped; the W4 subscription-redirect cascade
    (lib/hal/cascade_target.py MODELS, config/w4_cascade.json) is untouched --
    it is a routing chain in all but name. Scoped to the qwen entries
    specifically (not a verbatim full-line match) so an unrelated edit to
    another provider's entry on the same line -- e.g. family 6's
    gemini-2.5-flash repoint -- cannot make this test fragile."""
    providers_text = (REPO / "config" / "providers.yaml").read_text(encoding="utf-8")
    routing_block = providers_text.split("\nrouting:\n", 1)[1]
    assert "qwen/qwen3.5-flash" in routing_block, (
        "AGORA's quality seat / HAL's cost-tier chain must still name "
        "qwen/qwen3.5-flash")
    for new_id in ("qwen4-max", "qwen3.8-flash"):
        assert f"qwen/{new_id}" not in routing_block, (
            f"qwen/{new_id} appears in the routing: block -- it must not "
            "name a fallback chain, tier chain or council seat")
    cascade_text = (REPO / "lib" / "hal" / "cascade_target.py").read_text(encoding="utf-8")
    assert "qwen4-max" not in cascade_text and "qwen3.8-flash" not in cascade_text
