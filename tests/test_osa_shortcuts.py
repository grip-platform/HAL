"""Tests for ``lib.hal.osa.shortcuts`` — γ-2d CLI-based Shortcut backend.

Covers:

* Argv composition — name as literal, ``--input-path -`` when input_data given
* Shell-injection safety — backtick / ``$(...)`` / pipe in input_data reach
  subprocess as bytes, never expanded
* Allowlist integration — reject unlisted shortcut names
* Rate-limit integration — hit cap raises RateLimitExceeded
* Timeout propagation — slow shortcut raises ShortcutError
* Audit log — one entry per dispatch (success or failure)
* ShortcutUnavailable on non-Darwin platforms (mocked)
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from lib.hal.osa.allowlist import (
    _reset_cache_for_tests as _reset_allowlist,
    allowlist_path,
)
from lib.hal.osa.audit import AuditEntry
from lib.hal.osa.rate_limit import (
    RateLimit,
    RateLimitExceeded,
    _reset_for_tests as _reset_rate,
)
from lib.hal.osa.shortcuts import (
    ShortcutError,
    ShortcutResult,
    ShortcutUnavailable,
    _build_argv,
    _shortcuts_binary_available,
    run_shortcut,
)


def _completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["/usr/bin/shortcuts", "run", "name"],
        returncode=returncode, stdout=stdout, stderr=stderr,
    )


class BuildArgv(unittest.TestCase):
    """argv composition is literal — no shell wrapper, no AppleScript."""

    def test_argv_no_input_data(self) -> None:
        argv = _build_argv("Home: Goodnight", None)
        self.assertEqual(argv, ["/usr/bin/shortcuts", "run", "Home: Goodnight"])

    def test_argv_with_input_data_adds_stdin_flag(self) -> None:
        argv = _build_argv("Echo", "payload")
        self.assertEqual(
            argv,
            ["/usr/bin/shortcuts", "run", "Echo", "--input-path", "-"],
        )

    def test_spaces_in_name_preserved_as_single_token(self) -> None:
        argv = _build_argv("My Favourite Shortcut", None)
        self.assertEqual(argv[2], "My Favourite Shortcut")

    def test_backtick_in_name_is_literal(self) -> None:
        argv = _build_argv("foo`whoami`", None)
        self.assertEqual(argv[2], "foo`whoami`")


class ShellInjectionSafety(unittest.TestCase):
    """H168 falsification anchor — shell meta-chars never reach a shell."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="hal-sc-inj-")
        os.environ["HAL_HOME"] = self._tmp
        os.environ["HAL_OSA_SKIP_IMMUTABLE"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("HAL_HOME", None)
        os.environ.pop("HAL_OSA_SKIP_IMMUTABLE", None)
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    @patch("lib.hal.osa.shortcuts._shortcuts_binary_available", return_value=True)
    @patch("lib.hal.osa.shortcuts.subprocess.run")
    def test_backtick_reaches_subprocess_as_literal_stdin(
        self, mock_run, _bin_available,
    ) -> None:
        mock_run.return_value = _completed(returncode=0, stdout="ok")
        run_shortcut("Echo", input_data="`whoami`")
        kwargs = mock_run.call_args.kwargs
        self.assertEqual(kwargs.get("input"), "`whoami`")
        argv = mock_run.call_args.args[0]
        self.assertEqual(argv[0], "/usr/bin/shortcuts")
        self.assertNotIn("sh", argv)
        self.assertNotIn("bash", argv)

    @patch("lib.hal.osa.shortcuts._shortcuts_binary_available", return_value=True)
    @patch("lib.hal.osa.shortcuts.subprocess.run")
    def test_dollar_paren_in_input_not_expanded(
        self, mock_run, _bin_available,
    ) -> None:
        mock_run.return_value = _completed(returncode=0)
        run_shortcut("Echo", input_data="$(id)")
        self.assertEqual(
            mock_run.call_args.kwargs.get("input"), "$(id)",
        )

    @patch("lib.hal.osa.shortcuts._shortcuts_binary_available", return_value=True)
    @patch("lib.hal.osa.shortcuts.subprocess.run")
    def test_pipe_and_semicolon_reach_as_literals(
        self, mock_run, _bin_available,
    ) -> None:
        mock_run.return_value = _completed(returncode=0)
        run_shortcut("Echo", input_data="alpha | grep beta; rm -rf /")
        self.assertEqual(
            mock_run.call_args.kwargs.get("input"),
            "alpha | grep beta; rm -rf /",
        )


class AllowlistIntegration(unittest.TestCase):
    """``allowlist_kind=`` rejects unlisted shortcut names."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="hal-sc-al-")
        os.environ["HAL_HOME"] = self._tmp
        os.environ["HAL_OSA_SKIP_IMMUTABLE"] = "1"
        self.list_path = allowlist_path("shortcuts")
        self.list_path.parent.mkdir(parents=True, exist_ok=True)
        self.list_path.write_text("GoodShortcut\nAnotherOne\n")
        _reset_allowlist()

    def tearDown(self) -> None:
        os.environ.pop("HAL_HOME", None)
        os.environ.pop("HAL_OSA_SKIP_IMMUTABLE", None)
        _reset_allowlist()
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    @patch("lib.hal.osa.shortcuts._shortcuts_binary_available", return_value=True)
    @patch("lib.hal.osa.shortcuts.subprocess.run")
    def test_allowlisted_name_passes(self, mock_run, _bin) -> None:
        mock_run.return_value = _completed(returncode=0, stdout="ok")
        result = run_shortcut("GoodShortcut", allowlist_kind="shortcuts")
        self.assertTrue(result.ok)

    @patch("lib.hal.osa.shortcuts._shortcuts_binary_available", return_value=True)
    @patch("lib.hal.osa.shortcuts.subprocess.run")
    def test_unlisted_name_raises_and_skips_subprocess(
        self, mock_run, _bin,
    ) -> None:
        with self.assertRaises(ShortcutError):
            run_shortcut("ImpostorShortcut", allowlist_kind="shortcuts")
        mock_run.assert_not_called()


class RateLimitIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="hal-sc-rl-")
        os.environ["HAL_HOME"] = self._tmp
        os.environ["HAL_OSA_SKIP_IMMUTABLE"] = "1"
        _reset_rate()

    def tearDown(self) -> None:
        os.environ.pop("HAL_HOME", None)
        os.environ.pop("HAL_OSA_SKIP_IMMUTABLE", None)
        _reset_rate()
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    @patch("lib.hal.osa.shortcuts._shortcuts_binary_available", return_value=True)
    @patch("lib.hal.osa.shortcuts.subprocess.run")
    def test_rate_limit_rejects_over_cap(self, mock_run, _bin) -> None:
        mock_run.return_value = _completed(returncode=0, stdout="ok")
        limit = RateLimit(window_s=60, max_calls=2)
        run_shortcut("Echo", rate_limit=limit)
        run_shortcut("Echo", rate_limit=limit)
        with self.assertRaises(RateLimitExceeded):
            run_shortcut("Echo", rate_limit=limit)


class TimeoutHandling(unittest.TestCase):
    @patch("lib.hal.osa.shortcuts._shortcuts_binary_available", return_value=True)
    @patch("lib.hal.osa.shortcuts.subprocess.run")
    def test_timeout_raises_shortcut_error(self, mock_run, _bin) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd="/usr/bin/shortcuts", timeout=5,
        )
        os.environ["HAL_OSA_SKIP_IMMUTABLE"] = "1"
        try:
            with self.assertRaises(ShortcutError) as ctx:
                run_shortcut("Slow", timeout_s=5)
            self.assertIn("timed out", str(ctx.exception))
        finally:
            os.environ.pop("HAL_OSA_SKIP_IMMUTABLE", None)


class NonDarwinUnavailable(unittest.TestCase):
    @patch("lib.hal.osa.shortcuts._shortcuts_binary_available", return_value=False)
    def test_raises_unavailable(self, _bin) -> None:
        with self.assertRaises(ShortcutUnavailable):
            run_shortcut("Anything")


class AuditLogging(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="hal-sc-au-")
        os.environ["HAL_HOME"] = self._tmp
        os.environ["HAL_OSA_SKIP_IMMUTABLE"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("HAL_HOME", None)
        os.environ.pop("HAL_OSA_SKIP_IMMUTABLE", None)
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    @patch("lib.hal.osa.shortcuts._shortcuts_binary_available", return_value=True)
    @patch("lib.hal.osa.shortcuts.subprocess.run")
    @patch("lib.hal.osa.shortcuts.write_audit")
    def test_success_writes_one_audit_entry(
        self, mock_write, mock_run, _bin,
    ) -> None:
        mock_run.return_value = _completed(returncode=0, stdout="ok")
        run_shortcut("Echo")
        self.assertEqual(mock_write.call_count, 1)
        entry: AuditEntry = mock_write.call_args[0][0]
        self.assertTrue(entry.ok)
        self.assertEqual(entry.script, "shortcut:Echo")

    @patch("lib.hal.osa.shortcuts._shortcuts_binary_available", return_value=True)
    @patch("lib.hal.osa.shortcuts.subprocess.run")
    @patch("lib.hal.osa.shortcuts.write_audit")
    def test_non_zero_exit_raises_and_audits_failure(
        self, mock_write, mock_run, _bin,
    ) -> None:
        mock_run.return_value = _completed(returncode=2, stderr="no such shortcut")
        with self.assertRaises(ShortcutError):
            run_shortcut("Missing")
        entry: AuditEntry = mock_write.call_args[0][0]
        self.assertFalse(entry.ok)
        self.assertIn("no such shortcut", entry.error or "")


if __name__ == "__main__":
    unittest.main()
