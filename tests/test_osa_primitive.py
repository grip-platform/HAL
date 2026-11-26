"""Tests for ``lib.hal.osa.osa`` — kernel execution primitive."""
from __future__ import annotations

import subprocess
import unittest
from unittest.mock import MagicMock, patch

from lib.hal.osa.osa import (
    OsaError,
    OsaResult,
    OsaTimeout,
    PlatformUnsupported,
    osa_raw,
)


class PlatformProbing(unittest.TestCase):
    """`osa_raw` must reject non-darwin hosts before spawning anything."""

    @patch("lib.hal.osa.osa.sys")
    def test_non_darwin_raises_platform_unsupported(self, mock_sys) -> None:
        mock_sys.platform = "linux"
        with self.assertRaises(PlatformUnsupported):
            osa_raw('return "hi"')

    @patch("lib.hal.osa.osa.sys")
    @patch("lib.hal.osa.osa.shutil.which", return_value=None)
    def test_missing_osascript_raises_platform_unsupported(
        self, _which, mock_sys
    ) -> None:
        mock_sys.platform = "darwin"
        with self.assertRaises(PlatformUnsupported):
            osa_raw('return "hi"')


class SubprocessIntegration(unittest.TestCase):
    """Shape of the subprocess call + OsaResult contract."""

    def _patch_darwin(self):
        sys_p = patch("lib.hal.osa.osa.sys")
        which_p = patch(
            "lib.hal.osa.osa.shutil.which", return_value="/usr/bin/osascript"
        )
        mock_sys = sys_p.start()
        which_p.start()
        self.addCleanup(sys_p.stop)
        self.addCleanup(which_p.stop)
        mock_sys.platform = "darwin"

    @patch("lib.hal.osa.osa.subprocess.run")
    def test_happy_path_returns_osa_result(self, mock_run) -> None:
        self._patch_darwin()
        mock_run.return_value = MagicMock(
            returncode=0, stdout="hello\n", stderr=""
        )
        result = osa_raw('return "hello"')
        self.assertIsInstance(result, OsaResult)
        self.assertTrue(result.ok)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "hello\n")
        self.assertEqual(result.language, "AppleScript")

    @patch("lib.hal.osa.osa.subprocess.run")
    def test_non_zero_exit_raises_osa_error_by_default(self, mock_run) -> None:
        self._patch_darwin()
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="some error\n"
        )
        with self.assertRaises(OsaError):
            osa_raw("bad script")

    @patch("lib.hal.osa.osa.subprocess.run")
    def test_check_false_returns_failed_result_instead(self, mock_run) -> None:
        self._patch_darwin()
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="boom\n"
        )
        result = osa_raw("bad", check=False)
        self.assertFalse(result.ok)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stderr, "boom\n")

    @patch("lib.hal.osa.osa.subprocess.run")
    def test_timeout_raises_osa_timeout(self, mock_run) -> None:
        self._patch_darwin()
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd="osascript", timeout=1
        )
        with self.assertRaises(OsaTimeout):
            osa_raw("slow", timeout_s=1)

    @patch("lib.hal.osa.osa.subprocess.run")
    def test_subprocess_args_include_language_flag(self, mock_run) -> None:
        self._patch_darwin()
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        osa_raw("return 1", language="JavaScript")
        args, _ = mock_run.call_args
        self.assertEqual(args[0][:3], ["osascript", "-l", "JavaScript"])

    @patch("lib.hal.osa.osa.subprocess.run")
    def test_duration_is_positive(self, mock_run) -> None:
        self._patch_darwin()
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = osa_raw("return 1")
        self.assertGreaterEqual(result.duration_s, 0.0)


class JavaScriptOSA(unittest.TestCase):
    """JavaScript OSA is supported but flagged for α-3 gating."""

    @patch("lib.hal.osa.osa.sys")
    @patch(
        "lib.hal.osa.osa.shutil.which", return_value="/usr/bin/osascript"
    )
    @patch("lib.hal.osa.osa.subprocess.run")
    def test_javascript_language_reflected_in_result(
        self, mock_run, _which, mock_sys
    ) -> None:
        mock_sys.platform = "darwin"
        mock_run.return_value = MagicMock(
            returncode=0, stdout="42\n", stderr=""
        )
        result = osa_raw("42", language="JavaScript")
        self.assertEqual(result.language, "JavaScript")


if __name__ == "__main__":
    unittest.main()
