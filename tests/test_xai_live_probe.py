"""H354 — xai live-probe regression test (opt-in for CI reproducibility).

AGORA Council PR #312 concern #8: the original PR claimed "live-probed HTTP 200"
for xai (api.x.ai), but the verification was V>>'s local keychain only. CI runs
without those keychain entries; the live claim was single-host, single-time.

This test makes the live-probe reproducible:
  - Default: skips with a clear opt-in instruction (no flaky CI failures).
  - Opt-in via env vars: the test runs against the actual endpoint and
    asserts HTTP 200, validating the route shipped in tool_cli_integration.py.

Required env vars (any of these enable the probe):
  XAI_API_KEY         — direct API key for x.ai
  XAI_LIVE=1          — read the key from hal-xai keychain entry

CI reproducibility: configure XAI_API_KEY as a repo secret (same shape as
GEMINI_API_KEY / GROQ_API_KEY in hal-council-review.yml) to run this test on
every push. When the secret is absent the test skips cleanly — no false
failures.

The arcee half of this file (api.arcee.ai) was removed 2026-11-24, renaming
the file from test_xai_arcee_live_probe.py, when the Arcee provider was
dropped: its only model, trinity-large-thinking, answered HTTP 404, and
Arcee now only resells DeepSeek/Kimi/GLM at those vendors' own prices.
"""
import os
import subprocess
import unittest
import urllib.error
import urllib.request


def _resolve_key(env_var: str, keychain_service: str, live_flag: str) -> str | None:
    """Return the API key from env var or keychain, or None if neither configured.

    Matches the resolution order used by other HAL provider live tests:
    explicit env > keychain (when LIVE flag opts in) > skip.
    """
    direct = os.environ.get(env_var, "").strip()
    if direct:
        return direct
    if os.environ.get(live_flag) != "1":
        return None
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-w", "-s", keychain_service],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def _probe(url: str, key: str, timeout: float = 10.0) -> int:
    """Return the HTTP status code from a GET to url with Bearer auth.

    Catches HTTPError to surface the actual status (401, 403, etc.) so a
    test failure tells the operator which side broke.
    """
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


class TestXaiLiveProbe(unittest.TestCase):
    """H354 — xai (api.x.ai/v1) reachable with a configured key."""

    def setUp(self) -> None:
        self.key = _resolve_key("XAI_API_KEY", "hal-xai", "XAI_LIVE")
        if not self.key:
            self.skipTest(
                "Set XAI_API_KEY env var OR XAI_LIVE=1 (with hal-xai "
                "keychain entry) to run the xai live probe.",
            )

    def test_xai_v1_models_returns_200(self) -> None:
        status = _probe("https://api.x.ai/v1/models", self.key)
        self.assertEqual(
            status, 200,
            f"H354 regression: xai /v1/models returned HTTP {status} — "
            f"the route added in tool_cli_integration.py is unreachable.",
        )


if __name__ == "__main__":
    unittest.main()
