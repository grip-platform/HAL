"""GLM flagship repoint: glm-5.1 -> glm-5.3 on HAL's Zhipu bridge (GRIP#9431, 2026-11-24).

Zhipu's GLM Coding Plan answers every `glm-5.1` request with GLM-5.3 (two probes on
2026-11-24, each beside a glm-5.3 control, both HTTP 200 with the reply's `model`
field reading glm-5.3; docs.z.ai/guides/llm/glm-5.1 is still live but its own
"Migrate to GLM-5.3" guide never mentions glm-5.1). V>>'s decision: name the model
that really answers. This moves the two live HAL defaults that drove a `glm-5.1`
request -- `_ZAI_MODEL_MAP["ANTHROPIC_DEFAULT_OPUS_MODEL"]`
(lib/hal/cc_env_bridge.py) and `PROVIDER_CONFIGS["zhipu"]["default_model"]`
(lib/hal/happi/providers/openai_compat.py) -- from glm-5.1 to glm-5.3. Behaviour is
unchanged (Zhipu already answered glm-5.1 requests with GLM-5.3); the label becomes
true. glm-5.1 stays listed in config/providers.yaml's zhipu `models:` and reachable
by name, since Zhipu still serves it.

config/providers.yaml and config/providers.windows.yaml already listed `glm-5.3` in
the zhipu `models:` list (added alongside glm-5.3-flash and glm-5.4, 2026-11-24),
but NEITHER a config/model_catalog.yaml row NOR a lib/hal/cost_tracker.py `_PRICING`
row existed for it -- this PR's own /dogfood-domain audit found that gap. That
mattered more than usual here: GLM-5.3 is now the model BEHIND every glm-5.1
request these two defaults drive, so without its own cost row every one of those
calls would have silently inherited glm-5's row ($0.65/$1.95), the same
dot-separated-version trap tests/test_glm_5_4_onboarding.py documents for
glm-5.3-flash (_VERSION_CONTINUATION only recognises a dash before the continuation
digits, never a dot).

Vendor facts, both read 2026-11-24:
- docs.z.ai/guides/llm/glm-5.3: 1M context, 128K max output, "Disabling reasoning
  is no longer supported" (thinking.type only accepts "enabled"), reasoning_effort
  in {low, high, max} with max recommended for coding, no locked sampling parameter
  documented. Served on both the Model API (OpenAI Chat Completion, OpenAI
  Response, Anthropic Message endpoints) and the GLM Coding Plan subscription.
- docs.z.ai/guides/overview/pricing.md: "GLM-5.3 | $1.4 | $0.26 | Limited-time Free
  | $4.4" (input, cached input, cached input storage, output).
"""
from __future__ import annotations

import unittest.mock
from pathlib import Path

import pytest

from lib.hal import cost_tracker
from lib.hal.cc_env_bridge import _ZAI_MODEL_MAP
from lib.hal.happi.providers import openai_compat as provider

REPO = Path(__file__).resolve().parents[1]


def _zhipu_models(config: str) -> list[str]:
    text = (REPO / "config" / config).read_text(encoding="utf-8")
    block = text.split("- name: zhipu\n", 1)[1].split("\n  - name:", 1)[0]
    line = next(raw for raw in block.splitlines() if raw.strip().startswith("models:"))
    return [m.strip() for m in line.split("[", 1)[1].split("]", 1)[0].split(",")]


def test_catalogued_with_its_one_million_window():
    text = (REPO / "config" / "model_catalog.yaml").read_text(encoding="utf-8")
    assert "- address: zhipu/glm-5.3\n" in text
    row = text.split("- address: zhipu/glm-5.3\n", 1)[1].split("\n  - address:", 1)[0]
    assert "context_window: 1000000" in row


def test_priced_by_its_own_published_row():
    assert cost_tracker._resolve_pricing("glm-5.3") == (1.4, 4.4, 0.26, 0.0)
    assert cost_tracker._resolve_pricing("glm-5.3") != cost_tracker._PRICING["glm-5"], (
        "a dot-separated point release billed at glm-5's row is the "
        "version-continuation trap _VERSION_CONTINUATION does not catch for dots")


def test_would_silently_inherit_glm_5_without_its_own_row():
    """Forward anchor / regression proof: without this row, every glm-5.1 flagship
    launch (which now actually calls glm-5.3) would have silently billed at glm-5's
    row -- a different, older, unrelated model's price."""
    pricing = dict(cost_tracker._PRICING)
    del pricing["glm-5.3"]
    with unittest.mock.patch.object(cost_tracker, "_PRICING", pricing):
        price = cost_tracker._resolve_pricing("glm-5.3")
    assert price == cost_tracker._PRICING["glm-5"]


def test_glm_5_3_flash_would_now_fall_back_to_glm_5_3_not_the_old_glm_5_trap():
    """A side effect of this row worth pinning explicitly: "glm-5.3-flash" also
    starts with "glm-5.3" (and "-flash" is a named-variant suffix _inherits_row
    allows, not a version continuation), so _priced_key's longest-match rule now
    intercepts glm-5.3-flash's fallback at glm-5.3 BEFORE it could ever reach the
    bare "glm-5" row -- a strictly safer fallback than before this row existed.
    tests/test_glm_5_4_onboarding.py's forward anchor pops this row too, to keep
    proving the deeper bare-"glm-5" trap with every protecting row removed."""
    pricing = dict(cost_tracker._PRICING)
    del pricing["glm-5.3-flash"]
    with unittest.mock.patch.object(cost_tracker, "_PRICING", pricing):
        price = cost_tracker._resolve_pricing("glm-5.3-flash")
    assert price == cost_tracker._PRICING["glm-5.3"]


def test_model_prices_yaml_carries_glm_5_3():
    text = (REPO / "config" / "model_prices.yaml").read_text(encoding="utf-8")
    assert "glm-5.3:\n" in text


@pytest.mark.parametrize("config", ["providers.yaml", "providers.windows.yaml"])
def test_glm_5_3_was_already_listed_by_the_zhipu_provider(config):
    """Documents the pre-existing state this PR builds on (both config files already
    listed glm-5.3, just with no catalogue or cost row) so this suite does not
    silently depend on an unstated external fact."""
    assert "glm-5.3" in _zhipu_models(config)


def test_cc_env_bridge_opus_model_is_glm_5_3():
    assert _ZAI_MODEL_MAP["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "glm-5.3"


def test_cc_env_bridge_other_tiers_did_not_move():
    """Only the Opus slot moves (GRIP#9431 scopes the repoint to the bare-flag
    flagship and these two HAL defaults); Sonnet and Haiku keep their own ids."""
    assert _ZAI_MODEL_MAP["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "glm-4.7"
    assert _ZAI_MODEL_MAP["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "glm-4.5-air"


def test_openai_compat_zhipu_default_model_is_glm_5_3():
    assert provider.PROVIDER_CONFIGS["zhipu"]["default_model"] == "glm-5.3"


def test_glm_5_1_no_longer_named_by_either_live_default():
    """Regression anchor against a partial revert: glm-5.1 may still appear in
    providers.yaml's models: list (it is still served, as an alias) but must not
    reappear as either of these two live defaults."""
    assert _ZAI_MODEL_MAP["ANTHROPIC_DEFAULT_OPUS_MODEL"] != "glm-5.1"
    assert provider.PROVIDER_CONFIGS["zhipu"]["default_model"] != "glm-5.1"
