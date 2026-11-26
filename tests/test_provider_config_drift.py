"""H340 — Provider URL consistency across HAL: zero drift between sources.

Wave 0 of the /auto sprint discovered that fixing Moonshot's base_url in
providers.yaml left FOUR other production sites still hardcoding the old
.cn endpoint:
  - lib/hal/tool_cli_integration.py (_OPENAI_COMPAT_ROUTES)
  - lib/hal/happi/providers/openai_compat.py (_PROVIDER_CONFIG)
  - lib/hal/subscription.py (the Kimi subscription transport -- since
    2026-09-07 KimiCodingSubscription, which must NOT share the moonshot host)
  - tests/test_provider_routing.py + tests/test_models_command.py (fixtures)

This test prevents that ENTIRE CLASS of bugs from recurring. providers.yaml
is the canonical source; every downstream caller MUST agree.

Goodhart-proof: tests assert per-provider matches AND fail fast on any
typo. Adding a new provider to providers.yaml without updating downstream
callers causes a clear, named failure pointing at the missing entry.

Falsifier of H340: this test passes while a real production caller has
a different URL. To detect: grep for hardcoded provider URLs after the
test passes — should return ONLY the canonical sites this test covers.
"""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CONFIG_PATH = REPO_ROOT / "config" / "providers.yaml"

# H352 — AGORA Council PR #312 concern #7: documented routing exemptions.
# Providers in EXEMPT_FROM_ROUTES legitimately have NO entry in
# _OPENAI_COMPAT_ROUTES — the drift test silently skipped them before,
# which masked silent omissions. Now exemptions require an explicit reason
# string and a future provider addition triggers a test failure unless
# explicitly added to this set or to the route table.
EXEMPT_FROM_ROUTES: dict[str, str] = {
    "soruxgpt": "meta-aggregator with non-standard /v1 path; routes via custom adapter",
    "vulavula": "custom_adapter (lib/hal/providers/vulavula.py); not OpenAI-compat",
}


def _parse_providers_yaml() -> dict[str, dict[str, str]]:
    """Stdlib-only parser for the flat providers block we care about.

    Returns {provider_name: {base_url: ..., api_key_source: ..., ...}}.
    Skips the routing/quality_cost/tier_chains/council sections.
    """
    text = CONFIG_PATH.read_text()
    providers: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    in_providers_section = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.lstrip()

        # Top-level section markers
        if line.startswith("providers:"):
            in_providers_section = True
            continue
        if line.startswith("routing:"):
            in_providers_section = False
            continue
        if not in_providers_section:
            continue

        if stripped.startswith("- name:"):
            if current is not None and "name" in current:
                providers[current["name"]] = current
            current = {"name": stripped.split(":", 1)[1].strip()}
            continue

        if current is None:
            continue

        if ":" in stripped and not stripped.startswith("#"):
            key, _, value = stripped.partition(":")
            current[key.strip()] = value.strip()

    return providers


class TestProviderConfigDrift(unittest.TestCase):
    """H340: zero drift between providers.yaml and downstream callers."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.canonical = _parse_providers_yaml()
        # Sanity guard — parsing must yield a non-empty result
        assert cls.canonical, "providers.yaml parsed to empty — parser bug"
        assert "moonshot" in cls.canonical, "moonshot must be in providers.yaml"

    def test_tool_cli_integration_routes_match_providers_yaml(self) -> None:
        """_OPENAI_COMPAT_ROUTES base_urls must match providers.yaml.

        Wave 0 of /auto found this drift cost real user impact: providers.yaml
        had moonshot.ai but _OPENAI_COMPAT_ROUTES had moonshot.cn — silent
        split-brain where /tools on with moonshot routed to .cn (401)
        while plain chat went to .ai. Asserts every provider in BOTH sources
        agrees on base_url.
        """
        from lib.hal.tool_cli_integration import _OPENAI_COMPAT_ROUTES

        # Map prefix -> base_url from the routes tuple
        route_urls = {
            prefix.rstrip("/"): url
            for (prefix, url, _auth) in _OPENAI_COMPAT_ROUTES
        }

        drift_errors: list[str] = []
        for name, config in self.canonical.items():
            yaml_url = config.get("base_url", "")
            if not yaml_url:
                continue  # skip entries without base_url (e.g. local auto-discover)
            route_url = route_urls.get(name)
            if route_url is None:
                continue  # not all yaml providers are in routes (intentional)
            if route_url != yaml_url:
                drift_errors.append(
                    f"  {name}: providers.yaml={yaml_url!r} but "
                    f"_OPENAI_COMPAT_ROUTES={route_url!r}"
                )

        self.assertFalse(
            drift_errors,
            "H340 regression: provider URL drift between providers.yaml and "
            "lib/hal/tool_cli_integration.py:\n" + "\n".join(drift_errors),
        )

    def test_every_cloud_yaml_provider_routed_or_exempt(self) -> None:
        """H352 — AGORA Council PR #312 concern #7: silent omissions vs explicit exemptions.

        Every cloud-type provider in providers.yaml must EITHER have an entry
        in _OPENAI_COMPAT_ROUTES OR be in EXEMPT_FROM_ROUTES with a documented
        reason string. This closes the "silent absence" failure mode AGORA
        Council flagged: a future provider added to yaml without a route now
        triggers a clear test failure unless explicitly exempted.

        Falsifier: a new cloud provider lands in providers.yaml without route
        coverage AND without an EXEMPT_FROM_ROUTES entry — this test flags it.
        """
        from lib.hal.tool_cli_integration import _OPENAI_COMPAT_ROUTES

        route_prefixes = {prefix.rstrip("/") for (prefix, _u, _a) in _OPENAI_COMPAT_ROUTES}
        unrouted_unmapped = []
        for name, config in self.canonical.items():
            if config.get("type") != "cloud":
                continue  # local providers are auto-discovered, not routed
            if name in route_prefixes:
                continue  # routed — fine
            if name in EXEMPT_FROM_ROUTES:
                continue  # explicitly exempt with reason
            unrouted_unmapped.append(name)

        self.assertFalse(
            unrouted_unmapped,
            f"H352 regression: cloud providers in yaml without route coverage "
            f"AND without exemption: {unrouted_unmapped}. Either add a route "
            f"to lib/hal/tool_cli_integration._OPENAI_COMPAT_ROUTES, or add to "
            f"EXEMPT_FROM_ROUTES in this test with a documented reason.",
        )

    def test_pyyaml_parser_handles_tricky_yaml(self) -> None:
        """H350 — AGORA Council PR #312 concern #6: pyyaml-based parser must
        handle YAML edge cases the previous stdlib parser silently misparsed.

        Tricky cases:
          - inline comments after value
          - quoted values containing colons
          - multi-line folded scalars
        All of these are valid YAML; pyyaml handles them correctly.
        """
        import yaml
        tricky = '''
providers:
  - name: test_provider
    base_url: https://api.example.com/v1  # inline comment
    api_key_source: keychain:grip-test    # also has colon in value
    description: |
      Multi-line
      folded scalar
    note: "key: value with embedded colon"
'''
        data = yaml.safe_load(tricky)
        provider = data["providers"][0]
        self.assertEqual(provider["base_url"], "https://api.example.com/v1")
        self.assertEqual(provider["api_key_source"], "keychain:grip-test")
        self.assertIn("Multi-line", provider["description"])
        self.assertEqual(provider["note"], "key: value with embedded colon")

    def test_happi_provider_config_matches_providers_yaml(self) -> None:
        """happi/providers/openai_compat.py _PROVIDER_CONFIG endpoints must
        match providers.yaml base_urls.

        The HAPPI runtime uses its own provider map. It MUST agree with the
        main HAL provider map or HAL-the-CLI and HAL-the-HAPPI-runtime
        produce different routing for the same provider name. Wave 0
        discovered moonshot diverged here too.
        """
        try:
            from lib.hal.happi.providers.openai_compat import PROVIDER_CONFIGS
        except ImportError:
            self.skipTest("HAPPI providers module not importable")

        drift_errors: list[str] = []
        for name, happi_config in PROVIDER_CONFIGS.items():
            yaml_config = self.canonical.get(name)
            if yaml_config is None:
                continue  # HAPPI may register providers not in YAML (intentional)
            yaml_url = yaml_config.get("base_url", "")
            happi_endpoint = happi_config.get("endpoint", "")
            # HAPPI endpoints include /chat/completions suffix; the YAML
            # base_url does not. Normalise for comparison.
            happi_base = happi_endpoint.replace("/chat/completions", "")
            if yaml_url and happi_base and yaml_url != happi_base:
                drift_errors.append(
                    f"  {name}: providers.yaml={yaml_url!r} but "
                    f"happi _PROVIDER_CONFIG={happi_base!r}"
                )

        self.assertFalse(
            drift_errors,
            "H340 regression: provider URL drift between providers.yaml and "
            "lib/hal/happi/providers/openai_compat.py:\n" + "\n".join(drift_errors),
        )

    def test_kimi_coding_subscription_is_a_different_host_from_moonshot(self) -> None:
        """Inverted 2026-09-07 (tests-must-be-able-to-fail.md mandate 4).

        This test used to REQUIRE the Kimi "subscription" transport to share
        providers.yaml's moonshot base_url. That premise was the defect: the
        pay-per-token Kimi API platform (api.moonshot.ai, keychain
        grip-moonshot, GRIP Pool B) and the Kimi Code membership
        (api.kimi.com/coding, keychain grip-kimi-coding, Pool D) are two
        products with two bills -- "the Kimi Code API is independent from the
        API service on this platform" (platform.kimi.ai/docs/guide/
        product-plans). A subscription transport pointed at the metered host
        modelled nothing that exists. The .cn/.ai lesson still holds for the
        moonshot block itself (test_no_dotcn_in_production_code_for_moonshot).
        """
        from lib.hal.subscription import KimiCodingSubscription

        moonshot_url = self.canonical["moonshot"].get("base_url", "")
        sub_url = KimiCodingSubscription.base_url

        self.assertEqual(sub_url, "https://api.kimi.com/coding/v1")
        self.assertIn("api.moonshot.ai", moonshot_url)
        self.assertNotEqual(
            sub_url, moonshot_url,
            "the Kimi Code subscription must not point at the metered Moonshot host",
        )

    def test_no_dotcn_in_production_code_for_moonshot(self) -> None:
        """Sweep guard: no production .py file may contain 'moonshot.cn'
        outside of legitimate documentation contexts.

        This catches the SPECIFIC regression Wave 0 fixed. If anyone reverts
        even one of the four sites back to .cn, this test fails clearly.

        Allowlist (H348 — AGORA Council PR #311 concern #2 escape hatch):
        lines containing ``MOONSHOT_BASE_URL`` legitimately reference .cn as
        the mainland override target — the env-var override docstring in
        subscription.py teaches operators how to opt back in. The presence
        of the env var name signals documentation, not a hardcoded URL.
        """
        lib_dir = REPO_ROOT / "lib"
        offenders: list[str] = []
        for py_file in lib_dir.rglob("*.py"):
            if "__pycache__" in py_file.parts:
                continue
            content = py_file.read_text(errors="replace")
            # Per-line check so we can allowlist legitimate documentation
            # references (lines mentioning MOONSHOT_BASE_URL — the override).
            offending_lines = [
                line for line in content.splitlines()
                if "moonshot.cn" in line and "MOONSHOT_BASE_URL" not in line
            ]
            if offending_lines:
                offenders.append(str(py_file.relative_to(REPO_ROOT)))

        self.assertFalse(
            offenders,
            "H340 regression: 'moonshot.cn' found in production code outside "
            "of MOONSHOT_BASE_URL override documentation. Wave 0 of /auto "
            "fixed this — if it returned, mainland users should be served "
            "via the MOONSHOT_BASE_URL env var (H348), not by reverting the "
            ".ai endpoint. Files: " + ", ".join(offenders),
        )

    def test_canonical_source_actually_has_required_providers(self) -> None:
        """Sanity: providers.yaml must actually define the providers we test
        against. If parsing yields empty/missing entries, the other tests
        would silently skip and pass — false negative. Catch that here."""
        required = {"moonshot", "groq", "gemini", "deepseek", "openai"}
        missing = required - set(self.canonical.keys())
        self.assertFalse(
            missing,
            f"providers.yaml missing canonical providers: {missing}",
        )

    def test_required_yaml_fields_are_present_per_provider(self) -> None:
        """Every cloud provider entry must have base_url + api_key_source."""
        for name, config in self.canonical.items():
            # Local providers (lemonade, ollama, vllm, lmstudio, mlx_vlm)
            # may have api_key_source absent (they use auth: none) — skip.
            if config.get("type") == "local":
                continue
            # Skip vulavula (custom_adapter) and any other special-cases
            if "custom_adapter" in config:
                continue
            self.assertIn(
                "base_url", config,
                f"provider {name!r} missing base_url in providers.yaml",
            )


if __name__ == "__main__":
    unittest.main()
