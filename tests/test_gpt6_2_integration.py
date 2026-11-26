"""GPT-6.2 Astra and GPT-6.2 Sol provider-registration regression anchor (2026-11-24).

OpenAI launched GPT-6.2 Astra and GPT-6.2 Sol on 2026-11-17 alongside GPT-6.1
Terra and GPT-6 Luna (developers.openai.com/api/docs/models, read 2026-11-24).
Only the flagship (Astra) and workhorse (Sol) are in scope here (V>>'s decision
1, 2026-11-24 HITL round 1) -- Terra and Luna get no rows.

Both ids are $8/$40 and $2/$10 per 1M tokens respectively (Standard, <=272K
input) -- cheaper than GPT-6 Astra's $10/$50, but still address-only: never a
routing default, fallback, tier-chain member or council seat (decision 6).

Falsifier: this test passes while HAL cannot actually route to either model.
Guarded the same way as test_gpt6_astra_integration.py: the id appears in BOTH
providers.yaml (and its windows mirror) AND the real /models consumer
(models_command.load_catalogue), and the family router sends a bare id to the
openai provider.
"""
import sys
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ASTRA_2 = "gpt-6.2-astra"
SOL_2 = "gpt-6.2-sol"
CONTEXT_WINDOW = 1_050_000


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _provider_models(data: dict, provider_name: str) -> list[str]:
    for p in data["providers"]:
        if p["name"] == provider_name:
            return p.get("models", [])
    return []


class TestGpt62Registration(unittest.TestCase):

    def test_native_openai_provider_lists_both(self) -> None:
        models = _provider_models(_load(REPO_ROOT / "config" / "providers.yaml"), "openai")
        self.assertIn(ASTRA_2, models, "providers.yaml openai provider is missing gpt-6.2-astra")
        self.assertIn(SOL_2, models, "providers.yaml openai provider is missing gpt-6.2-sol")

    def test_windows_provider_parity(self) -> None:
        models = _provider_models(
            _load(REPO_ROOT / "config" / "providers.windows.yaml"), "openai")
        self.assertIn(ASTRA_2, models, "providers.windows.yaml openai provider is missing gpt-6.2-astra")
        self.assertIn(SOL_2, models, "providers.windows.yaml openai provider is missing gpt-6.2-sol")

    def test_catalogue_exposes_both_via_real_consumer(self) -> None:
        from lib.hal.models_command import load_catalogue

        rows = load_catalogue()

        def _get(row, key):
            return getattr(row, key, None) if not isinstance(row, dict) else row.get(key)

        by_addr = {_get(r, "address"): r for r in rows}
        for model_id in (ASTRA_2, SOL_2):
            address = f"openai/{model_id}"
            self.assertIn(address, by_addr, f"model_catalog.yaml missing {address}")
            row = by_addr[address]
            self.assertEqual(int(_get(row, "context_window")), CONTEXT_WINDOW)
            self.assertEqual(_get(row, "tier"), "frontier")
            self.assertTrue(_get(row, "paid"), f"{model_id} is metered; paid must be true")

    def test_catalogue_places_both_after_every_anthropic_address_and_ahead_of_gpt_6_astra(self) -> None:
        """The catalogue resolver is first-match; an OpenAI frontier row must
        never sit above the Anthropic rows (the Mythos placement caveat), and
        the newest OpenAI frontier pair shows ahead of the family's previous
        flagship, mirroring test_gpt6_astra_integration.py's own placement
        check for gpt-6-astra relative to gpt-5.6-sol."""
        cat = _load(REPO_ROOT / "config" / "model_catalog.yaml")
        node = cat["models"] if isinstance(cat, dict) and "models" in cat else cat
        addrs = [m["address"] for m in node]
        last_anthropic = max(i for i, a in enumerate(addrs) if a.startswith("anthropic/"))
        for model_id in (ASTRA_2, SOL_2):
            address = f"openai/{model_id}"
            self.assertIn(address, addrs)
            self.assertGreater(addrs.index(address), last_anthropic)
            self.assertLess(addrs.index(address), addrs.index("openai/gpt-6-astra"))

    def test_both_are_in_no_routing_chain(self) -> None:
        routing = _load(REPO_ROOT / "config" / "providers.yaml")["routing"]
        flat: list[str] = [str(routing.get("default_model", ""))]
        flat += [str(m) for m in routing.get("fallback_chain", [])]
        flat += [str(m) for m in routing.get("council", [])]
        flat += [str(m) for m in routing.get("quality_cost", {}).values()]
        for chain in routing.get("tier_chains", {}).values():
            flat += [str(m) for m in chain]
        for model_id in (ASTRA_2, SOL_2):
            offenders = [m for m in flat if model_id in m]
            self.assertEqual(offenders, [], f"{model_id} must stay address-only, found in routing: {offenders}")

    def test_family_router_sends_bare_ids_to_openai(self) -> None:
        from lib.hal.gateway_egress import same_wire_provider

        for model_id in (ASTRA_2, SOL_2):
            self.assertEqual(same_wire_provider(model_id), "openai")
            self.assertEqual(same_wire_provider(f"openai/{model_id}"), "openai")


if __name__ == "__main__":
    unittest.main()
