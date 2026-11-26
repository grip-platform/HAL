"""GPT-6 Astra liveness anchor — the id must still be SERVED, not just declared.

Origin: 2026-09-08. PR #558 registered ``gpt-6-astra`` from the release notes and
a single ``GET /v1/models`` read, and the post-merge probe on the same day
confirmed both wires answer for this account (chat/completions 1.82s, responses
2.16s, HTTP 200 each). Every existing Astra test reads YAML and Python — none of
them asks the provider whether the id is still there, so a vendor retirement
would leave the config green and the model unreachable. That is exactly how
``kimi-k2.5`` and seven ``moonshot-v1`` ids sat declared for a week after their
2026-08-31 retirement (see ``test_moonshot_endpoint.py``), and this file is the
OpenAI-side counterpart to the anchor that closed it.

Opt-in via ``OPENAI_LIVE=1`` so offline and CI runs skip rather than fail. The
credential is resolved by the provider from whatever source this platform's
providers file names — the keychain on posix, an env var on Windows — exactly
as the gateway resolves it, and is never printed.

Scope, stated so a later reader does not over-read a green run: this checks that
the ids are SERVED. It does not check that HAL routes to them correctly through
the live gateway — that is step 4 of ``~/.hal/rules/adding-a-new-llm-to-hal.md``
section 7 and remains unverified for Astra.
"""
from __future__ import annotations

import functools
import os
import sys
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MODEL_ID = "gpt-6-astra"
PROVIDER_NAME = "openai"
LIVE_FLAG = "OPENAI_LIVE"

POSIX_CONFIG = REPO_ROOT / "config" / "providers.yaml"
WINDOWS_CONFIG = REPO_ROOT / "config" / "providers.windows.yaml"

# Both files must declare only served ids, so both are checked. Connection
# details come from the one for THIS platform: the posix file resolves the key
# from the keychain and the windows file from an env var, so hardcoding either
# makes an OPENAI_LIVE=1 run on the other platform fail at key resolution and
# report it as a serving failure.
DECLARED_CONFIGS = (POSIX_CONFIG, WINDOWS_CONFIG)
CONNECTION_CONFIG = WINDOWS_CONFIG if sys.platform == "win32" else POSIX_CONFIG

_LIVE_REQUIRED = f"Set {LIVE_FLAG}=1 to run the live GET /v1/models probe"


def _provider_block(path: Path) -> dict:
    """The openai provider block from one providers file.

    Raises rather than returning a default when the provider is absent: a
    missing block means the file moved or the name drifted, and a silent
    empty result would let every subset check below pass vacuously.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for provider in data["providers"]:
        if provider["name"] == PROVIDER_NAME:
            return provider
    raise AssertionError(
        f"{path.name}: no provider named {PROVIDER_NAME!r} -- config drift, "
        f"not a passing check"
    )


def _declared_models(path: Path) -> list[str]:
    """Ids declared for the openai provider in one providers file."""
    return list(_provider_block(path).get("models", []))


@functools.lru_cache(maxsize=1)
def _live_model_ids() -> frozenset[str]:
    """Ids the account is actually served, read once per process.

    base_url and api_key_source are read from the platform's own providers
    file rather than duplicated here, so this cannot drift from what the
    gateway actually resolves.
    """
    from lib.hal.providers.openai_compat import OpenAICompatProvider

    block = _provider_block(CONNECTION_CONFIG)
    provider = OpenAICompatProvider(
        name=PROVIDER_NAME,
        base_url=block["base_url"],
        api_key_source=block["api_key_source"],
    )
    return frozenset(m.id for m in provider.list_models())


class TestGpt6AstraLiveness(unittest.TestCase):
    """Liveness anchors. Red the day the vendor stops serving a declared id."""

    def _live(self) -> frozenset[str]:
        live = _live_model_ids()
        # An empty list is a credential or network problem, not a verdict that
        # nothing is served. Fail loudly instead of letting a subset check pass
        # vacuously against the empty set.
        self.assertTrue(
            live,
            "GET /v1/models returned no ids -- key or network problem, "
            "not a config verdict",
        )
        return live

    @unittest.skipUnless(os.environ.get(LIVE_FLAG) == "1", _LIVE_REQUIRED)
    def test_astra_is_still_served(self) -> None:
        """The narrow anchor: gpt-6-astra specifically must still be served."""
        live = self._live()
        self.assertIn(
            MODEL_ID, live,
            f"{MODEL_ID} is declared in providers.yaml but the API no longer "
            f"serves it -- the catalogue row's verified-serving claim is stale",
        )

    @unittest.skipUnless(os.environ.get(LIVE_FLAG) == "1", _LIVE_REQUIRED)
    def test_every_declared_openai_id_is_served(self) -> None:
        """The broad anchor: no declared id may outlive its retirement.

        Catches the whole class the Moonshot anchor was written for, not just
        Astra -- a declared-but-retired id fails at call time with a confusing
        404 rather than at config time with a clear message.
        """
        live = self._live()
        for path in DECLARED_CONFIGS:
            with self.subTest(config=path.name):
                declared = _declared_models(path)
                self.assertTrue(
                    declared,
                    f"{path.name} declares no openai models -- parse failure, "
                    f"not a passing subset check",
                )
                missing = sorted(set(declared) - live)
                self.assertEqual(
                    missing, [],
                    f"{path.name} declares ids the API does not serve: {missing}",
                )


if __name__ == "__main__":
    unittest.main()
