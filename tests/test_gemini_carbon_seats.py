"""Gemini 4 Carbon holds the HAL seats Gemini 3.1 Pro held (V>> 2026-11-26).

V>>'s words: "add the new Google LLMs -- to replace 3.1 Pro". Carbon is the one
Google calls "Our strongest reasoning model" (ai.google.dev/gemini-api/docs/models,
read 2026-11-26). This mirrors GRIP's tests/test_gemini_carbon_seats.py. HAL's
AGORA judge moves in its own operator-merged change and is not pinned here.

Both directions are decisions: 3.1 Pro stays catalogued and priced, because the
model is still served.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CARBON = "gemini/gemini-4-carbon"
PRO = "gemini/gemini-3.1-pro-preview"
PROVIDER_FILES = ("config/providers.yaml", "config/providers.windows.yaml")


def _routing(relative: str) -> dict:
    return yaml.safe_load((REPO / relative).read_text(encoding="utf-8"))["routing"]


def test_the_complex_tier_and_the_premium_chain_name_carbon_in_both_provider_files():
    """Mutation: restore 3.1 Pro in quality_cost.complex or tier_chains.premium of
    either file -> RED."""
    for relative in PROVIDER_FILES:
        routing = _routing(relative)
        assert routing["quality_cost"]["complex"] == CARBON, relative
        assert CARBON in routing["tier_chains"]["premium"], relative
        chains = [routing.get("fallback_chain", []), *routing["tier_chains"].values(),
                  list(routing["quality_cost"].values())]
        assert not [chain for chain in chains if PRO in chain], (relative, chains)


def test_the_raillm_frontier_class_holds_carbon():
    """Mutation: restore 3.1 Pro in DEFAULT_EQUIVALENCE_MAP's frontier class -> RED."""
    from lib.hal.raillm import DEFAULT_EQUIVALENCE_MAP, ModelTier
    members = DEFAULT_EQUIVALENCE_MAP[ModelTier.FRONTIER].members
    assert CARBON in members and PRO not in members, members


def test_3_1_pro_stays_catalogued_and_priced():
    """Mutation: drop 3.1 Pro's catalogue row or its price row -> RED."""
    from lib.hal import cost_tracker as ct
    assert f"address: {PRO}" in (REPO / "config" / "model_catalog.yaml").read_text(encoding="utf-8")
    assert ct._pricing_key("gemini-3.1-pro-preview") == "gemini-3.1-pro"
    assert ct._PRICING["gemini-3.1-pro"] == (2.00, 12.00, 0.20, 2.00)
