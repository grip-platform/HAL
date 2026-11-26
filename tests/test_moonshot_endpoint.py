"""Regression test for H333 — Moonshot must use api.moonshot.ai, not api.moonshot.cn.

The .cn endpoint returns 401 for keys issued by the international moonshot.ai
service. Before this regression test was added, providers.yaml pointed at .cn
and every Moonshot invocation failed with `Invalid Authentication`.

Goodhart-proof: the test asserts BOTH the positive (.ai is present) AND the
negative (.cn is absent) so a partial revert (e.g. someone adding .cn back as
a fallback) still trips the regression.

Live API verification is opt-in via MOONSHOT_LIVE=1 to avoid CI failures when
the keychain entry is missing or the network is unavailable.
"""
import os
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "providers.yaml"
WINDOWS_CONFIG_PATH = REPO_ROOT / "config" / "providers.windows.yaml"
# The two declared surfaces (rules/adding-a-new-llm-to-hal.md section 3 keeps
# them in step); every model-id assertion below runs against BOTH.
DECLARED_CONFIGS = (CONFIG_PATH, WINDOWS_CONFIG_PATH)

# Retired by the vendor on 2026-08-31 -- platform.kimi.ai/docs/models.md: "The
# `kimi-k2.5` and `moonshot-v1` series were officially retired on August 31,
# 2026. Calls to these models now return a 404". A declared retired id makes
# its rung fail over silently, which is how eight dead ids survived until
# 2026-09-07.
RETIRED_ID_PREFIXES = ("moonshot-v1", "kimi-k2.5")
LIVE_FRONTIER_ID = "kimi-k3"   # the RAILLM Kimi rung (lib/hal/raillm_ladder.py)

CATALOG_PATH = REPO_ROOT / "config" / "model_catalog.yaml"
# GET https://api.moonshot.ai/v1/models on 2026-09-07: context_length per id,
# supports_image_in true for all four. The catalog must say what the API says.
# The two K3.1 rows were read from the same endpoint on 2026-11-24
# (context_length 1048576, supports_image_in and supports_video_in true).
LIVE_CONTEXT_WINDOWS = {
    "kimi-k3": 1048576,
    "kimi-k2.6": 262144,
    "kimi-k2.7-code": 262144,
    "kimi-k2.7-code-highspeed": 262144,
    "kimi-k3.1-pro": 1048576,
    "kimi-k3.1-pro-highspeed": 1048576,
}


def _load_moonshot_config(path: Path = CONFIG_PATH) -> dict:
    """Stdlib-only YAML parser sufficient for this file's flat shape."""
    text = path.read_text()
    in_moonshot = False
    fields: dict = {}
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("- name: moonshot"):
            in_moonshot = True
            continue
        if in_moonshot:
            if stripped.startswith("- name:"):
                break  # next provider
            if ":" in stripped:
                key, _, value = stripped.partition(":")
                fields[key.strip()] = value.strip()
    return fields


def _declared_models(fields: dict) -> list[str]:
    """``models: [a, b, c]`` -> ``["a", "b", "c"]`` (the flow-sequence form
    both providers files use for this block)."""
    raw = fields.get("models", "").strip()
    return [m.strip() for m in raw.strip("[]").split(",") if m.strip()]


def _catalog_moonshot_entries() -> dict[str, dict]:
    """``{bare_id: {field: value}}`` for every ``moonshot/`` address in the
    catalog (stdlib line scan; each entry's scalar fields sit one per line and
    precede its free-text description, so first-occurrence wins)."""
    entries: dict[str, dict] = {}
    current: dict | None = None
    for line in CATALOG_PATH.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("- address:"):
            address = stripped.partition(":")[2].strip()
            current = None
            if address.startswith("moonshot/"):
                current = entries.setdefault(address.split("/", 1)[1], {})
            continue
        if current is not None and ":" in stripped and not stripped.startswith("-"):
            key, _, value = stripped.partition(":")
            current.setdefault(key.strip(), value.strip())
    return entries


class TestMoonshotEndpointRegression(unittest.TestCase):
    """H333: Moonshot config must point at api.moonshot.ai (international)."""

    def setUp(self) -> None:
        self.config = _load_moonshot_config()
        self.assertTrue(self.config, "moonshot block missing from providers.yaml")

    def test_base_url_uses_ai_endpoint(self) -> None:
        """Positive assertion: base_url must equal the canonical .ai endpoint.

        Tightened per skeptic council recommendation C: substring match would
        accept typos like 'moonshot.aiX.evil.com'. Exact match is the
        Goodhart-resistant shape.
        """
        base_url = self.config.get("base_url", "")
        self.assertEqual(
            base_url,
            "https://api.moonshot.ai/v1",
            f"H333 regression: Moonshot base_url must be exactly the canonical "
            f".ai endpoint (got {base_url!r})",
        )

    def test_base_url_does_not_revert_to_cn(self) -> None:
        """Negative assertion: .cn must NOT be present (Goodhart-proof)."""
        base_url = self.config.get("base_url", "")
        self.assertNotIn(
            ".cn",
            base_url,
            f"H333 regression: Moonshot reverted to .cn endpoint: {base_url!r}",
        )

    def test_api_key_source_is_keychain(self) -> None:
        """Auth must come from macOS keychain entry grip-moonshot."""
        source = self.config.get("api_key_source", "")
        self.assertEqual(source, "keychain:grip-moonshot")

    def test_declared_models_exclude_retired_ids(self) -> None:
        """Inverted 2026-09-07 (tests-must-be-able-to-fail.md mandate 4): this
        test used to REQUIRE ``moonshot-v1-auto``. The vendor retired that whole
        family and ``kimi-k2.5`` on 2026-08-31, so a declared retired id is now
        the defect. ``kimi-k3`` is the ladder's Kimi rung and must stay declared.
        Checked on both providers files so the Windows mirror cannot drift.
        """
        for path in DECLARED_CONFIGS:
            with self.subTest(config=path.name):
                declared = _declared_models(_load_moonshot_config(path))
                self.assertIn(
                    LIVE_FRONTIER_ID, declared,
                    f"{path.name}: {LIVE_FRONTIER_ID} missing from {declared!r}",
                )
                retired = [m for m in declared if m.startswith(RETIRED_ID_PREFIXES)]
                self.assertEqual(
                    retired, [],
                    f"{path.name} declares retired Moonshot ids {retired!r}; "
                    f"they return 404 since 2026-08-31",
                )

    def test_catalog_moonshot_entries_match_declared_and_live(self) -> None:
        """config/model_catalog.yaml must describe exactly the declared Moonshot
        ids, each with the context window and vision flag the live API reports
        (catalog header: "keep this file in sync with providers.yaml"). Red on
        2026-09-07 before the catalog sweep: it still carried the retired
        kimi-k2.5 (text-only, 200000), lacked k2.6 / k2.7-code /
        k2.7-code-highspeed, and said kimi-k3 was text-only although the API
        reports supports_image_in true."""
        declared = set(_declared_models(_load_moonshot_config(CONFIG_PATH)))
        entries = _catalog_moonshot_entries()
        self.assertEqual(
            set(entries), declared,
            f"catalog moonshot/* ids {sorted(entries)} != declared {sorted(declared)}",
        )
        for bare, fields in sorted(entries.items()):
            with self.subTest(model=bare):
                self.assertEqual(fields.get("modality"), "text+vision",
                                 f"{bare}: live API reports supports_image_in true")
                self.assertEqual(int(fields.get("context_window", 0)),
                                 LIVE_CONTEXT_WINDOWS[bare],
                                 f"{bare}: context_window differs from live context_length")
                self.assertEqual(fields.get("paid"), "true", f"{bare}: metered API is paid")

    @unittest.skipUnless(
        os.environ.get("MOONSHOT_LIVE") == "1",
        "Set MOONSHOT_LIVE=1 to run the live API health check",
    )
    def test_live_api_health_check(self) -> None:
        """Opt-in live verification: actual provider must reach 200 OK.

        This is the strongest falsifier — the config can read 'moonshot.ai'
        and the network call can still fail if the key is wrong or the
        endpoint is down. Live test catches both.
        """
        import sys
        sys.path.insert(0, str(REPO_ROOT))
        from lib.hal.providers.openai_compat import OpenAICompatProvider

        provider = OpenAICompatProvider(
            name="moonshot",
            base_url="https://api.moonshot.ai/v1",
            api_key_source="keychain:grip-moonshot",
        )
        self.assertTrue(
            provider.health_check(),
            "Live Moonshot health check failed — endpoint or key issue",
        )

    @unittest.skipUnless(
        os.environ.get("MOONSHOT_LIVE") == "1",
        "Set MOONSHOT_LIVE=1 to run the live declared-vs-served check",
    )
    def test_declared_models_are_all_served_live(self) -> None:
        """Liveness anchor (W0, 2026-09-07): every id declared for provider
        ``moonshot`` in BOTH providers files must appear in the live
        GET /v1/models list. No structural test compared the declared list
        with what the API serves, which is how kimi-k2.5 and seven moonshot-v1
        ids sat declared for a week after the vendor retired them on
        2026-08-31. Proved able to fail by temporarily declaring kimi-k2.5
        (red), then reverting (green) -- see the W0 PR body.

        The key is resolved by the provider from the keychain and never
        printed. An empty live list is a credential or network problem and
        fails loudly instead of letting the subset check pass vacuously.
        """
        import sys
        sys.path.insert(0, str(REPO_ROOT))
        from lib.hal.providers.openai_compat import OpenAICompatProvider

        provider = OpenAICompatProvider(
            name="moonshot",
            base_url="https://api.moonshot.ai/v1",
            api_key_source="keychain:grip-moonshot",
        )
        live = {m.id for m in provider.list_models()}
        self.assertTrue(
            live,
            "GET /v1/models returned no ids -- key or network problem, not a config verdict",
        )
        for path in DECLARED_CONFIGS:
            with self.subTest(config=path.name):
                declared = _declared_models(_load_moonshot_config(path))
                missing = sorted(set(declared) - live)
                self.assertEqual(
                    missing, [],
                    f"{path.name} declares ids the API does not serve: {missing}; "
                    f"live={sorted(live)}",
                )


if __name__ == "__main__":
    unittest.main()
