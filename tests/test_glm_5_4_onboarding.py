"""GLM-5.4 and GLM-5.3-Flash in HAL (2026-11-24), the pair of GRIP's onboarding.

Vendor facts (docs.z.ai/guides/llm/glm-5.4.md, docs.z.ai/guides/llm/glm-5.3-flash.md and
docs.z.ai/guides/overview/pricing.md, all read 2026-11-24): both are 1M-context, 128K-max-
output models. GLM-5.4 is "now available to all GLM Coding Plan users" with its metered
Model API "pending completion of the security assessment" -- so it has NO published
per-token price, unlike every other model this repo prices. GLM-5.3-Flash bills $0.15
input / $0.03 cached input / $0.50 output per million tokens, with no distinct cache-write
rate published (the "Cached Input Storage" column reads "Limited-time Free"). GLM-5.3 (the
family both belong to) "always operates with reasoning enabled" -- `thinking.type` only
accepts "enabled" -- and exposes `reasoning_effort` in {low, high, max}, default max;
GLM-5.3-Flash's own page RECOMMENDS (never locks) `temperature: 1` and `top_p: 0.95`,
unlike the Kimi lineage, which rejects any other value.

GLM-5.3-Flash was already listed in config/providers.yaml's zhipu models with no catalogue
or cost row at all; this change fills both. GLM-5.4 is new everywhere.

What these tests hold in place:
- each id is catalogued with its 1M window and listed by the zhipu provider (both configs,
  providers.yaml and providers.windows.yaml);
- GLM-5.3-Flash is priced by its OWN row in cost_tracker.py and model_prices.yaml, never
  glm-5's -- the forward-anchor tests below prove this matters, because "glm-5.3-flash" and
  "glm-5.4" both start with the real "glm-5" key, and the dash-only _VERSION_CONTINUATION
  guard does not recognise a DOT-separated version continuation;
- GLM-5.4 gets an explicit zero cost_tracker.py row for the same reason, and deliberately
  NO model_prices.yaml row at all (that file's lookup is an exact dict `get` with no prefix
  fallback, so an absent row there correctly prices it as unknown, per the file's own
  documented convention);
- model_quirks.py is UNCHANGED: this repo has no zhipu/glm entries there at all, a
  pre-existing gap this change does not expand (see the PR body for the follow-up).

model_quirks.py has no "glm" prefix anywhere, so there is nothing to anchor a quirk test
against -- unlike the Kimi K3.1 Pro onboarding, where the existing "kimi-k3" prefix rules
already covered the new sibling and the tests there only needed to VERIFY that coverage.
"""
from __future__ import annotations

import unittest.mock
from pathlib import Path

import pytest

from lib.hal import cost_tracker, model_quirks

REPO = Path(__file__).resolve().parents[1]
NEW = ("glm-5.4", "glm-5.3-flash")
PUBLISHED = {
    # (input, output, cached input, cache write), USD per million tokens
    "glm-5.3-flash": (0.15, 0.50, 0.03, 0.0),
}


def _zhipu_models(config: str) -> list[str]:
    text = (REPO / "config" / config).read_text(encoding="utf-8")
    block = text.split("- name: zhipu\n", 1)[1].split("\n  - name:", 1)[0]
    line = next(raw for raw in block.splitlines() if raw.strip().startswith("models:"))
    return [m.strip() for m in line.split("[", 1)[1].split("]", 1)[0].split(",")]


@pytest.mark.parametrize("config", ["providers.yaml", "providers.windows.yaml"])
def test_glm_5_4_listed_last(config):
    models = _zhipu_models(config)
    assert models[-1] == "glm-5.4", models
    # Exact length, not just the tail: catches an accidental duplicate entry
    # (e.g. a second "glm-5.3-flash") that a tail-only check would miss (AGORA
    # suggestion, HAL#822).
    assert len(models) == len(set(models)) == 10, models


@pytest.mark.parametrize("config", ["providers.yaml", "providers.windows.yaml"])
def test_glm_5_3_flash_already_listed(config):
    assert "glm-5.3-flash" in _zhipu_models(config)


@pytest.mark.parametrize("model", NEW)
def test_catalogued_with_its_one_million_window(model):
    text = (REPO / "config" / "model_catalog.yaml").read_text(encoding="utf-8")
    assert f"- address: zhipu/{model}\n" in text
    row = text.split(f"- address: zhipu/{model}\n", 1)[1].split("\n  - address:", 1)[0]
    assert "context_window: 1000000" in row


def test_glm_5_3_flash_priced_by_its_own_published_row():
    assert cost_tracker._resolve_pricing("glm-5.3-flash") == PUBLISHED["glm-5.3-flash"]
    assert cost_tracker._resolve_pricing("glm-5.3-flash") != cost_tracker._PRICING["glm-5"], (
        "a dot-separated point release billed at glm-5's row is the "
        "version-continuation trap _VERSION_CONTINUATION does not catch for dots")


def test_glm_5_4_falls_to_the_visible_default_not_glm_5s_row():
    """GLM-5.4 has no published per-token price. cost_tracker.py's own stated policy
    for an unpriced id in a paid family (deepseek/qwen/glm) is the VISIBLE
    _DEFAULT_PRICING estimate, "never a silent zero ... under-counting to zero is
    budget-blindness" -- so GLM-5.4 resolves there, not to an invented (0,0,0,0) row
    and not to glm-5's row (it starts with the "glm-5" key; see the forward anchor
    below for what would happen without the _KNOWN_UNPRICED guard)."""
    assert cost_tracker._resolve_pricing("glm-5.4") == cost_tracker._DEFAULT_PRICING
    assert cost_tracker._resolve_pricing("glm-5.4") != cost_tracker._PRICING["glm-5"]


@pytest.mark.parametrize("model,sibling", [("glm-5.4", "glm-5"), ("glm-5.3-flash", "glm-5")])
def test_would_silently_inherit_its_sibling_without_the_guard(model, sibling):
    """Forward anchor / regression proof for the trap itself: without its protection
    (the _KNOWN_UNPRICED guard for glm-5.4; its own _PRICING row for glm-5.3-flash),
    _resolve_pricing silently returns the SIBLING's tuple -- a different, older,
    unrelated model's price -- because _VERSION_CONTINUATION only recognises a dash
    before the continuation digits, never a dot. If either assertion ever fails
    because that guard has been fixed to recognise a dot, retire this anchor along
    with the fix.

    glm-5.3 (GRIP#9431, 2026-11-24) is popped alongside glm-5.3-flash here: it is
    now a real, longer-match row ("glm-5.3-flash".startswith("glm-5.3") too, and
    "-flash" is a named-variant suffix _inherits_row allows, not a version
    continuation), so left in place it would correctly intercept glm-5.3-flash's
    fallback before it ever reached glm-5 -- see
    tests/test_glm_5_3_default_repoint.py for that safer, REAL one-level fallback.
    Popping it here keeps this test's job the same: prove the bare "glm-5" trap is
    real with every protecting row removed."""
    if model == "glm-5.4":
        with unittest.mock.patch.object(cost_tracker, "_KNOWN_UNPRICED", ()):
            price = cost_tracker._resolve_pricing(model)
    else:
        pricing = dict(cost_tracker._PRICING)
        del pricing[model]
        pricing.pop("glm-5.3", None)  # would otherwise intercept the fallback first
        with unittest.mock.patch.object(cost_tracker, "_PRICING", pricing):
            price = cost_tracker._resolve_pricing(model)
    assert price == cost_tracker._PRICING[sibling]


def test_model_prices_yaml_carries_glm_5_3_flash_but_not_glm_5_4():
    text = (REPO / "config" / "model_prices.yaml").read_text(encoding="utf-8")
    assert "glm-5.3-flash:\n" in text
    assert "glm-5.4:\n" not in text, (
        "GLM-5.4 has no published per-token price; model_prices.yaml's own header "
        "convention is to leave an unpriced model OUT, not guess a row for it")


def test_no_quirks_entries_exist_for_glm_yet():
    """Documents, rather than fixes, a pre-existing gap: model_quirks.py has no glm
    prefix anywhere, so GLM-5.3+'s always-on thinking and recommended (not locked)
    temperature/top_p are not encoded there. Filed as a follow-up, not expanded here."""
    assert model_quirks.locked_sampling_params("glm-5.3-flash") == ()
    assert model_quirks.reasoning_control("glm-5.3-flash") == {
        "kind": None, "default": None, "forced": False}
