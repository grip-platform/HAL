"""GPT-6 Astra provider-registration regression anchor (2026-09-08).

OpenAI released GPT-6 Astra on 2026-09-03 (developers.openai.com/api/docs/changelog:
"Sep 3 · Feature · gpt-6-astra · v1/responses v1/chat/completions"). The single
API id is ``gpt-6-astra`` — the model page lists no dated snapshot and no
``-pro`` variant, and a live ``GET /v1/models`` with the hal-openai key on
2026-09-08 returned 125 ids including ``gpt-6-astra`` and no other ``gpt-6*``.

This test anchors the config registration so a future edit that drops the
model fails loudly, and it anchors the DELIBERATE absence of Astra from every
routing chain: at $10/$50 per 1M tokens (2.5x Sol) it is address-only, never a
default, fallback, tier-chain member or council seat.

Falsifier: this test passes while HAL cannot actually route to Astra. Guarded by
asserting the id appears in BOTH the provider routing table (providers.yaml)
AND HAL's real /models consumer (models_command.load_catalogue), and that the
family router sends a bare ``gpt-6-astra`` to the openai provider.
"""
import sys
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MODEL_ID = "gpt-6-astra"
ADDRESS = f"openai/{MODEL_ID}"
CONTEXT_WINDOW = 1_050_000


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _provider_models(data: dict, provider_name: str) -> list[str]:
    for p in data["providers"]:
        if p["name"] == provider_name:
            return p.get("models", [])
    return []


class TestGpt6AstraRegistration(unittest.TestCase):

    def test_native_openai_provider_lists_astra(self) -> None:
        models = _provider_models(_load(REPO_ROOT / "config" / "providers.yaml"), "openai")
        self.assertIn(MODEL_ID, models, "providers.yaml openai provider is missing gpt-6-astra")

    def test_windows_provider_parity(self) -> None:
        models = _provider_models(
            _load(REPO_ROOT / "config" / "providers.windows.yaml"), "openai")
        self.assertIn(MODEL_ID, models, "providers.windows.yaml openai provider is missing gpt-6-astra")

    def test_catalogue_exposes_astra_via_real_consumer(self) -> None:
        from lib.hal.models_command import load_catalogue

        rows = load_catalogue()

        def _get(row, key):
            return getattr(row, key, None) if not isinstance(row, dict) else row.get(key)

        by_addr = {_get(r, "address"): r for r in rows}
        self.assertIn(ADDRESS, by_addr, f"model_catalog.yaml missing {ADDRESS}")
        row = by_addr[ADDRESS]
        self.assertEqual(int(_get(row, "context_window")), CONTEXT_WINDOW)
        self.assertEqual(_get(row, "tier"), "frontier")
        self.assertTrue(_get(row, "paid"), "Astra is metered; paid must be true")

    def test_catalogue_places_astra_after_every_anthropic_address(self) -> None:
        """The catalogue resolver is first-match; an OpenAI frontier row must
        never sit above the Anthropic rows (the Mythos placement caveat)."""
        cat = _load(REPO_ROOT / "config" / "model_catalog.yaml")
        node = cat["models"] if isinstance(cat, dict) and "models" in cat else cat
        addrs = [m["address"] for m in node]
        self.assertIn(ADDRESS, addrs)
        last_anthropic = max(i for i, a in enumerate(addrs) if a.startswith("anthropic/"))
        self.assertGreater(addrs.index(ADDRESS), last_anthropic)
        # And ahead of its own family's previous flagship, so the picker shows
        # the newest OpenAI frontier row first within the family.
        self.assertLess(addrs.index(ADDRESS), addrs.index("openai/gpt-5.6-sol"))

    def test_astra_is_in_no_routing_chain(self) -> None:
        routing = _load(REPO_ROOT / "config" / "providers.yaml")["routing"]
        flat: list[str] = [str(routing.get("default_model", ""))]
        flat += [str(m) for m in routing.get("fallback_chain", [])]
        flat += [str(m) for m in routing.get("council", [])]
        flat += [str(m) for m in routing.get("quality_cost", {}).values()]
        for chain in routing.get("tier_chains", {}).values():
            flat += [str(m) for m in chain]
        offenders = [m for m in flat if MODEL_ID in m]
        self.assertEqual(offenders, [], f"Astra must stay address-only, found in routing: {offenders}")

    def test_family_router_sends_bare_id_to_openai(self) -> None:
        from lib.hal.gateway_egress import same_wire_provider

        self.assertEqual(same_wire_provider(MODEL_ID), "openai")
        self.assertEqual(same_wire_provider(ADDRESS), "openai")


if __name__ == "__main__":
    unittest.main()
