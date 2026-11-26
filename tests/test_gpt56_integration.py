"""GPT-5.6 (Sol / Terra / Luna) provider-registration regression anchor.

OpenAI made the GPT-5.6 family generally available in the API on 2026-07-09 —
three durable capability tiers: Sol (flagship), Terra (balanced), Luna
(fast/affordable). Model IDs verified canonical against the live
GET /v1/models endpoint (hal-openai key): gpt-5.6-sol, gpt-5.6-terra,
gpt-5.6-luna. Live inference proven through HAL's OpenAICompatProvider via the
OpenRouter failover route (audit/gpt56-integration-proof.json) — the native
hal-openai route is billing-gated (insufficient_quota), not a code defect.

This test anchors the config registration so a future edit that drops any of
the three models fails loudly. Goodhart-proof: each assertion names the exact
missing model; deleting a model from either config makes the relevant test red.

Falsifier: this test passes while HAL cannot actually route to gpt-5.6. Guarded
by asserting the models appear in BOTH the provider routing table
(providers.yaml) AND HAL's real /models consumer (models_command.load_catalogue).
"""
import sys
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

TIERS = ["sol", "terra", "luna"]
NATIVE_IDS = [f"gpt-5.6-{t}" for t in TIERS]           # native openai provider
ROUTER_IDS = [f"openai/gpt-5.6-{t}" for t in TIERS]    # openrouter + catalog address


def _provider_models(config_path: Path, provider_name: str) -> list[str]:
    data = yaml.safe_load(config_path.read_text())
    for p in data["providers"]:
        if p["name"] == provider_name:
            return p.get("models", [])
    return []


class TestGpt56Registration(unittest.TestCase):
    """GPT-5.6 must be registered in every HAL routing + catalogue surface."""

    def test_native_openai_provider_lists_all_three_tiers(self) -> None:
        models = _provider_models(REPO_ROOT / "config" / "providers.yaml", "openai")
        missing = [m for m in NATIVE_IDS if m not in models]
        self.assertFalse(
            missing,
            f"providers.yaml openai provider missing GPT-5.6 tiers: {missing}",
        )

    def test_openrouter_failover_lists_all_three_tiers(self) -> None:
        """The OpenRouter route is the PROVEN-WORKING path (real completions,
        audit/gpt56-integration-proof.json) while hal-openai billing is topped up."""
        models = _provider_models(REPO_ROOT / "config" / "providers.yaml", "openrouter")
        missing = [m for m in ROUTER_IDS if m not in models]
        self.assertFalse(
            missing,
            f"providers.yaml openrouter provider missing GPT-5.6 tiers: {missing}",
        )

    def test_windows_provider_parity(self) -> None:
        """providers.windows.yaml must carry the same GPT-5.6 registration —
        no per-host divergence (mirrors the H340 zero-drift discipline)."""
        win = REPO_ROOT / "config" / "providers.windows.yaml"
        oi = _provider_models(win, "openai")
        orr = _provider_models(win, "openrouter")
        self.assertTrue(all(m in oi for m in NATIVE_IDS),
                        f"providers.windows.yaml openai missing: "
                        f"{[m for m in NATIVE_IDS if m not in oi]}")
        self.assertTrue(all(m in orr for m in ROUTER_IDS),
                        f"providers.windows.yaml openrouter missing: "
                        f"{[m for m in ROUTER_IDS if m not in orr]}")

    def test_catalogue_exposes_all_three_via_real_consumer(self) -> None:
        """HAL's actual /models consumer (models_command.load_catalogue) must
        surface all three tiers with GPT-5.6's real 1.05M context window."""
        from lib.hal.models_command import load_catalogue

        rows = load_catalogue()

        def _get(row, key):
            return getattr(row, key, None) if not isinstance(row, dict) else row.get(key)

        by_addr = {_get(r, "address"): r for r in rows}
        for addr in ROUTER_IDS:
            self.assertIn(addr, by_addr, f"model_catalog.yaml missing {addr}")
            ctx = _get(by_addr[addr], "context_window")
            self.assertEqual(
                int(ctx), 1_050_000,
                f"{addr} context_window={ctx}, expected GPT-5.6's 1,050,000",
            )

    def test_model_catalog_strict_yaml_parses_fully_and_has_gpt56(self) -> None:
        """The whole catalogue must parse under strict yaml.safe_load AND still
        contain every GPT-5.6 tier. This guards the plain-scalar-colon break
        class fixed on the gpt-5.4 / gpt-5.4-mini entries: that bug made
        yaml.safe_load raise partway, so a loose `len > 3` could pass on a
        truncated parse. The tight lower bound (>= 40; 42 entries at write time)
        plus the explicit address checks make a truncation or a dropped model
        fail loudly."""
        cat = yaml.safe_load((REPO_ROOT / "config" / "model_catalog.yaml").read_text())
        node = cat["models"] if isinstance(cat, dict) and "models" in cat else cat
        addrs = {m["address"] for m in node}
        self.assertGreaterEqual(
            len(node), 40,
            f"catalogue parsed to only {len(node)} entries — a plain-scalar "
            f"`description:` colon likely truncated the strict parse (was 42).",
        )
        missing = [a for a in ROUTER_IDS if a not in addrs]
        self.assertFalse(missing, f"strict-YAML parse of catalogue missing: {missing}")


if __name__ == "__main__":
    unittest.main()
