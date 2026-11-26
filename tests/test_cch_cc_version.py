"""Tests for client-version derivation in lib/hal/utils/cch.py.

Origin: 2026-09-14. CCH hardcoded ``cc_version="2.1.109"``. Anthropic gates
newer models on that value, so ``claude-fable-5-1`` was rejected with
"Claude Code 2.1.109 does not support this model; version 2.1.251 or newer is
required" -- and RAILLM then silently served gemini-2.5-flash with ok=true.
Measured: at 2.1.270 the same call returns served_model='claude-fable-5-1'.

Covers: env pin wins; CLI output parsed; every failure path falls back;
CCHSigner() picks up the derived value; an explicit cc_version still wins
(regression -- existing suites pass "9.9.9"); and the fallback may not be
lowered below Anthropic's stated floor.
"""
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.hal.utils.cch import (  # noqa: E402
    CCHSigner, _FALLBACK_CC_VERSION, _MODEL_MIN_CC_VERSION, _version_probe_env,
    detect_cc_version,
)


def _completed(stdout, returncode=0):
    return subprocess.CompletedProcess(["claude", "--version"], returncode, stdout, "")


class DetectCCVersionTest(unittest.TestCase):
    def setUp(self):
        detect_cc_version.cache_clear()
        self.addCleanup(detect_cc_version.cache_clear)

    def test_env_pin_wins_without_invoking_the_cli(self):
        with mock.patch.dict("os.environ", {"HAL_CC_VERSION": "3.0.1"}), \
             mock.patch("subprocess.run") as run:
            self.assertEqual(detect_cc_version(), "3.0.1")
        run.assert_not_called()

    def test_parses_version_from_cli_output(self):
        with mock.patch.dict("os.environ", {"HAL_CC_VERSION": ""}), \
             mock.patch("subprocess.run",
                        return_value=_completed("2.1.270 (Claude Code)\n")):
            self.assertEqual(detect_cc_version(), "2.1.270")

    def test_missing_binary_falls_back(self):
        with mock.patch.dict("os.environ", {"HAL_CC_VERSION": ""}), \
             mock.patch("subprocess.run", side_effect=FileNotFoundError):
            self.assertEqual(detect_cc_version(), _FALLBACK_CC_VERSION)

    def test_timeout_falls_back(self):
        with mock.patch.dict("os.environ", {"HAL_CC_VERSION": ""}), \
             mock.patch("subprocess.run",
                        side_effect=subprocess.TimeoutExpired("claude", 5)):
            self.assertEqual(detect_cc_version(), _FALLBACK_CC_VERSION)

    def test_nonzero_exit_falls_back(self):
        with mock.patch.dict("os.environ", {"HAL_CC_VERSION": ""}), \
             mock.patch("subprocess.run",
                        return_value=_completed("2.1.270", returncode=1)):
            self.assertEqual(detect_cc_version(), _FALLBACK_CC_VERSION)

    def test_unparseable_output_falls_back(self):
        with mock.patch.dict("os.environ", {"HAL_CC_VERSION": ""}), \
             mock.patch("subprocess.run", return_value=_completed("unknown")):
            self.assertEqual(detect_cc_version(), _FALLBACK_CC_VERSION)


class SignerDefaultTest(unittest.TestCase):
    def setUp(self):
        detect_cc_version.cache_clear()
        self.addCleanup(detect_cc_version.cache_clear)

    def test_default_uses_derived_version_not_the_old_literal(self):
        with mock.patch.dict("os.environ", {"HAL_CC_VERSION": "2.1.270"}):
            self.assertEqual(CCHSigner().cc_version, "2.1.270")

    def test_explicit_cc_version_still_wins(self):
        with mock.patch.dict("os.environ", {"HAL_CC_VERSION": "2.1.270"}):
            self.assertEqual(CCHSigner(cc_version="9.9.9").cc_version, "9.9.9")

    def test_fallback_not_below_anthropics_stated_floor(self):
        # (2, 1, 280): the floor Anthropic states for claude-opus-5-5. Probed
        # 2026-09-24 through this signer: advertising 2.1.251 (the old fallback)
        # returned 400 "Claude Code 2.1.251 does not support this model; version
        # 2.1.280 or newer is required", advertising 2.1.281 passed the gate.
        parts = tuple(int(p) for p in _FALLBACK_CC_VERSION.split("."))
        self.assertGreaterEqual(parts, (2, 1, 280))

    def test_each_gated_models_floor_is_the_one_anthropic_quoted(self):
        """Literal rows, so the table cannot drift down unnoticed."""
        self.assertEqual(_MODEL_MIN_CC_VERSION["claude-fable-5-1"], "2.1.251")
        self.assertEqual(_MODEL_MIN_CC_VERSION["claude-opus-5-5"], "2.1.280")
        # 2026-11-11, pinned to 2.1.280: "Claude Code 2.1.280 does not support
        # this model; version 2.1.290 or newer is required".
        self.assertEqual(_MODEL_MIN_CC_VERSION["claude-fable-5-2"], "2.1.290")

    def test_the_fallback_reaches_fable_5_2s_floor(self):
        """The old computed fallback, 2.1.280, was refused for Fable 5.2."""
        parts = tuple(int(p) for p in _FALLBACK_CC_VERSION.split("."))
        self.assertGreaterEqual(parts, (2, 1, 290))

    def test_the_fallback_clears_every_gated_models_floor(self):
        """A newly gated model is one table row; the fallback follows it."""
        fallback = tuple(int(p) for p in _FALLBACK_CC_VERSION.split("."))
        for model, floor in _MODEL_MIN_CC_VERSION.items():
            with self.subTest(model):
                self.assertGreaterEqual(fallback, tuple(int(p) for p in floor.split(".")))

class AgoraConvergenceTest(unittest.TestCase):
    """Concerns raised by the AGORA council on PR #590, converged.

    The two falsifier anchors are written per falsifier-as-forward-anchor:
    current code already passes both, so they exist as regression anchors.
    """

    def setUp(self):
        detect_cc_version.cache_clear()
        self.addCleanup(detect_cc_version.cache_clear)

    def test_explicit_empty_string_is_preserved_not_detected(self):
        """Blocking concern: `or` coerced an explicit "" into detection.

        origin/main assigned cc_version verbatim, so "" survived. `is not
        None` restores that while still detecting on the None default.
        """
        with mock.patch.dict("os.environ", {"HAL_CC_VERSION": "2.1.270"}):
            self.assertEqual(CCHSigner(cc_version="").cc_version, "")

    def test_every_fallback_path_logs_its_reason(self):
        """The bug this PR fixes was a SILENT downgrade; so is a silent fallback."""
        cases = [
            ("missing binary", {"side_effect": FileNotFoundError()}),
            ("timeout", {"side_effect": subprocess.TimeoutExpired("claude", 5)}),
            ("nonzero exit", {"return_value": _completed("2.1.270", returncode=1)}),
            ("unparseable", {"return_value": _completed("unknown")}),
        ]
        for label, kw in cases:
            with self.subTest(label):
                detect_cc_version.cache_clear()
                with mock.patch.dict("os.environ", {"HAL_CC_VERSION": ""}), \
                     mock.patch("subprocess.run", **kw), \
                     self.assertLogs("lib.hal.utils.cch", level="WARNING") as caught:
                    detect_cc_version()
                self.assertTrue(
                    any(_FALLBACK_CC_VERSION in line for line in caught.output),
                    f"{label}: fallback not reported in logs",
                )

    def test_fallback_paths_return_a_version_at_or_above_the_floor(self):
        """Asserting == _FALLBACK_CC_VERSION is a shared-constant tautology.

        Lower the constant and such a test still passes. Compare against the
        literal floor instead, so the failure path itself is anchored.
        """
        with mock.patch.dict("os.environ", {"HAL_CC_VERSION": ""}), \
             mock.patch("subprocess.run", side_effect=FileNotFoundError()):
            got = tuple(int(x) for x in detect_cc_version().split("."))
        self.assertGreaterEqual(got, (2, 1, 280))

    def test_first_version_string_in_the_output_wins(self):
        """Falsifier anchor: output carrying several version-shaped strings."""
        with mock.patch.dict("os.environ", {"HAL_CC_VERSION": ""}), \
             mock.patch("subprocess.run", return_value=_completed(
                 "2.1.270 (Claude Code) built with python 3.14.7\n")):
            self.assertEqual(detect_cc_version(), "2.1.270")

    def test_non_triple_version_still_builds_a_valid_billing_header(self):
        """Falsifier anchor: a cc_version with no dotted triple must not raise.

        build_body never reaches compute_cch, so this holds with xxhash absent
        -- which is CI's condition.
        """
        signer = CCHSigner(cc_version="unknown")
        body, suffix = signer.build_body(prompt="hello", model="claude-opus-5")
        header = json.loads(body)["system"][0]["text"]
        self.assertIn(f"cc_version=unknown.{suffix};", header)
        self.assertIn("cc_entrypoint=cli", header)

    def test_timeout_parameter_is_honoured(self):
        with mock.patch.dict("os.environ", {"HAL_CC_VERSION": ""}), \
             mock.patch("subprocess.run",
                        return_value=_completed("2.1.270")) as run:
            detect_cc_version(timeout_s=0.25)
        self.assertEqual(run.call_args.kwargs["timeout"], 0.25)

    def test_probe_env_withholds_credentials(self):
        with mock.patch.dict("os.environ", {"CLAUDE_CODE_OAUTH_TOKEN": "sekrit",
                                            "ANTHROPIC_API_KEY": "sekrit",
                                            "PATH": "/usr/bin"}):
            env = _version_probe_env()
        self.assertNotIn("CLAUDE_CODE_OAUTH_TOKEN", env)
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertEqual(env.get("PATH"), "/usr/bin")


if __name__ == "__main__":
    unittest.main()
