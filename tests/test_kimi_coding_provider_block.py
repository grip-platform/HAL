"""The ``kimi-coding`` provider blocks for providers.yaml, proven against the REAL loader.

The two blocks below are the exact text destined for ``config/providers.yaml``
(macOS, keychain) and ``config/providers.windows.yaml`` (Windows, env). They
are NOT in those files yet, on purpose: three sync tests bind every
``- name:`` line in providers.yaml to rows this change must not add --

* ``tests/test_happi_providers_openai_compat.py::test_happi_router_covers_yaml_registry_no_drift``
  demands a HAPPI ``PROVIDER_CONFIGS`` row (its only exemptions are
  ``anthropic`` and ``gemini``);
* ``tests/test_provider_config_drift.py::test_every_cloud_yaml_provider_routed_or_exempt``
  demands a ``tool_cli_integration`` route or an ``EXEMPT_FROM_ROUTES`` entry;
* ``tests/test_gateway_unroutable_provider.py`` demands a ``gateway_egress``
  endpoint the moment ``kimi-coding`` enters ``KNOWN_PROVIDERS``.

Those rows belong to the gateway/HAPPI lane, so the blocks ship here as
VERIFIED text -- fed through ``lib.hal.registry._load_yaml_config``, the
hand-rolled parser the registry actually uses -- rather than as prose that says
"this will parse" (``tests-must-be-able-to-fail.md`` mandate 5). When the
follow-up moves them into the live files, the last test below is already the
drift guard: a live ``kimi-coding`` block, if present, must match these fields.

Vendor facts: kimi.com/code/docs/en/kimi-code/models ("OpenAI compatible
https://api.kimi.com/coding/v1 | Anthropic compatible https://api.kimi.com/coding/")
and /kimi-code/membership. Council: seal ``e49c00c214c500be`` (Pool D; the
metered ``moonshot`` block stays Pool B). Hermetic: temp files, no keychain, no
network. UNVERIFIED live -- no Kimi Code credential exists.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from lib.hal.registry import ProviderRegistry, _load_yaml_config

_REPO = Path(__file__).resolve().parents[1]

KIMI_CODING_PROVIDERS_YAML = """\
  - name: kimi-coding
    type: cloud
    # Kimi Code -- Moonshot's Kimi MEMBERSHIP coding service, GRIP Pool D
    # (sovereign-council seal e49c00c214c500be, 2026-09-07). A SEPARATE product
    # from the pay-per-token `moonshot` block: "the Kimi Code API is independent
    # from the API service on this platform" (platform.kimi.ai product-plans).
    # Same credential name as the GRIP side
    # (lib/provider_secret.resolve_provider_key("kimi-coding"), GRIP#5336).
    # Anthropic-compatible base for the SAME key: https://api.kimi.com/coding/
    # -- what a Claude Code launch sets as ANTHROPIC_BASE_URL, with the key in
    # ANTHROPIC_API_KEY (not ANTHROPIC_AUTH_TOKEN). Billed as a membership
    # quota, not per token: "refreshes automatically every 7 days ... a rolling
    # 5-hour rate window ... Requests from the CLI, VS Code, and third-party
    # tools all count toward that quota". UNARMED until keychain
    # grip-kimi-coding exists. A 401 can mean "the requested capability exceeds
    # your plan's entitlements", not a bad key.
    base_url: https://api.kimi.com/coding/v1
    api_key_source: keychain:grip-kimi-coding
    # Exactly the four ids in the vendor's model table. "Entering a model
    # version name like Kimi K3 or K2.7 Code will cause the call to fail."
    models: [k3, k3-256k, kimi-for-coding, kimi-for-coding-highspeed]
"""

KIMI_CODING_PROVIDERS_WINDOWS_YAML = """\
  - name: kimi-coding
    type: cloud
    # Kimi Code membership (GRIP Pool D, seal e49c00c214c500be) -- see the
    # macOS block in providers.yaml for the full note. Windows reads the key
    # from the env var the GRIP resolver tries first for this service
    # (lib/provider_secret.env_candidates("grip-kimi-coding")[0]).
    base_url: https://api.kimi.com/coding/v1
    api_key_source: env:KIMI_CODING_API_KEY
    models: [k3, k3-256k, kimi-for-coding, kimi-for-coding-highspeed]
"""

_MODELS = ["k3", "k3-256k", "kimi-for-coding", "kimi-for-coding-highspeed"]
_BASE = "https://api.kimi.com/coding/v1"


def _load_block(block: str):
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as fh:
        fh.write("providers:\n" + block)
    try:
        configs, _routing = _load_yaml_config(path)
    finally:
        os.unlink(path)
    return configs


class TestKimiCodingProviderBlocks(unittest.TestCase):

    def _assert_fields(self, cfg, api_key_source: str) -> None:
        self.assertEqual(cfg.name, "kimi-coding")
        self.assertEqual(cfg.type.value, "cloud")
        self.assertEqual(cfg.base_url, _BASE)
        self.assertEqual(cfg.api_key_source, api_key_source)
        self.assertEqual(cfg.models, _MODELS)
        self.assertTrue(cfg.enabled)
        self.assertNotIn("moonshot", cfg.base_url, "never the metered host")

    def test_macos_block_parses_to_the_keychain_credential(self):
        (cfg,) = _load_block(KIMI_CODING_PROVIDERS_YAML)
        self._assert_fields(cfg, "keychain:grip-kimi-coding")

    def test_windows_block_parses_to_the_env_credential(self):
        (cfg,) = _load_block(KIMI_CODING_PROVIDERS_WINDOWS_YAML)
        self._assert_fields(cfg, "env:KIMI_CODING_API_KEY")

    def test_blocks_agree_on_everything_but_the_credential_source(self):
        (mac,) = _load_block(KIMI_CODING_PROVIDERS_YAML)
        (win,) = _load_block(KIMI_CODING_PROVIDERS_WINDOWS_YAML)
        self.assertEqual((mac.name, mac.type, mac.base_url, mac.models),
                         (win.name, win.type, win.base_url, win.models))
        self.assertNotEqual(mac.api_key_source, win.api_key_source)

    def test_enabled_false_is_the_registry_level_off_switch(self):
        # The optional-provider precedent (lib/hal/registry.py register():
        # ``if not source.enabled: return``). A disabled block never builds a
        # provider, so the registry has nothing under the name.
        block = KIMI_CODING_PROVIDERS_YAML.replace(
            "    base_url:", "    enabled: false\n    base_url:", 1)
        (cfg,) = _load_block(block)
        self.assertFalse(cfg.enabled)
        # An ISOLATED registry loaded from a file holding only the disabled
        # block -- a bare ProviderRegistry() auto-loads the live providers.yaml,
        # which would make this assert about 19 unrelated providers (and go
        # red the day the follow-up adds the live kimi-coding block).
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w") as fh:
            fh.write("providers:\n" + block)
        try:
            reg = ProviderRegistry(config_path=path)
        finally:
            os.unlink(path)
        self.assertIsNone(reg.get("kimi-coding"))
        self.assertEqual([p.name for p in reg.list_providers()], [])

    def test_live_yaml_block_if_present_matches_this_text(self):
        # Today neither live file carries the block (see module docstring); the
        # moment the follow-up adds it, this asserts it did not drift.
        for rel, source in (("config/providers.yaml", "keychain:grip-kimi-coding"),
                            ("config/providers.windows.yaml", "env:KIMI_CODING_API_KEY")):
            path = _REPO / rel
            if not path.exists():
                continue
            configs, _routing = _load_yaml_config(str(path))
            live = [c for c in configs if c.name == "kimi-coding"]
            with self.subTest(file=rel, present=bool(live)):
                if live:
                    self._assert_fields(live[0], source)


class TestKimiCodingRowsLanded(unittest.TestCase):
    """HAL#550 rows C-F: the blocks above are no longer only verified text.
    Every anchor below was red before the rows landed (KeyError / missing
    route / no live block / no catalogue address)."""

    def test_live_providers_yaml_carries_the_block(self):
        for rel in ("config/providers.yaml", "config/providers.windows.yaml"):
            configs, _routing = _load_yaml_config(str(_REPO / rel))
            live = [c for c in configs if c.name == "kimi-coding"]
            self.assertEqual(len(live), 1, rel)

    def test_happi_provider_config_row(self):
        from lib.hal.happi.providers.openai_compat import PROVIDER_CONFIGS, SUPPORTED_PROVIDERS
        row = PROVIDER_CONFIGS["kimi-coding"]
        self.assertIn("kimi-coding", SUPPORTED_PROVIDERS)
        self.assertEqual(row["endpoint"], "https://api.kimi.com/coding/v1/chat/completions")
        self.assertEqual(row["keychain"], "grip-kimi-coding")
        self.assertEqual(row["env"], "KIMI_CODING_API_KEY")
        self.assertEqual(row["auth"], "bearer")
        self.assertEqual(row["default_model"], "k3-256k")

    def test_tool_cli_route(self):
        from lib.hal.tool_cli_integration import _OPENAI_COMPAT_ROUTES
        self.assertIn(("kimi-coding/", _BASE, "keychain:grip-kimi-coding"), _OPENAI_COMPAT_ROUTES)

    def test_catalogue_carries_exactly_the_four_addresses(self):
        from lib.hal.models_command import load_catalogue
        got = sorted(e.address for e in load_catalogue(str(_REPO / "config" / "model_catalog.yaml"))
                     if e.address.startswith("kimi-coding/"))
        self.assertEqual(got, [f"kimi-coding/{m}" for m in _MODELS])


if __name__ == "__main__":
    unittest.main()
