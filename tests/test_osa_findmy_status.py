"""Tests for the ``findmy_status`` osa registry entry.

The entry lets any HAL surface, the phone bridge included, ask GRIP Find My
for its address-free status. Three properties matter, and each test below
fails when one of them breaks:

* it is a read-only tier-1 entry with nothing to inject through;
* it never drives Find My or any other app (no activation, no keystroke),
  because a status read must never steal focus or touch the user's screen;
* it runs ``grip-find-my status --brief`` from the GRIP install, and fails
  with a clear one-line reason, not a stack trace, when GRIP Find My is absent.

The shell half of the body is executed for real under ``/bin/sh`` against a
temporary HOME, so the tests exercise the command the dispatcher would run,
not a description of it. On macOS the whole AppleScript body also runs under
``osascript`` to prove it parses.
"""
from __future__ import annotations

import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from lib.hal.osa.registry import _reset_cache_for_tests, load_registry

NAME = "findmy_status"
PREFIX = 'do shell script "'


def _entry():
    _reset_cache_for_tests()
    return load_registry()[NAME]


def _shell_part(body: str) -> str:
    """Return the /bin/sh source inside ``do shell script "..."``.

    AppleScript string literals escape only the double quote and the
    backslash, so decoding takes the character after each backslash
    literally.
    """
    if not (body.startswith(PREFIX) and body.endswith('"')):
        raise AssertionError(f"body is not a single do shell script call: {body!r}")
    escaped = body[len(PREFIX):-1]
    out, i = [], 0
    while i < len(escaped):
        ch = escaped[i]
        if ch == "\\" and i + 1 < len(escaped):
            out.append(escaped[i + 1])
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _fake_grip_find_my(bin_dir: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    tool = bin_dir / "grip-find-my"
    tool.write_text('#!/bin/sh\necho "fake-grip-find-my $*"\n', encoding="utf-8")
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)


def _run_shell(src: str, home: Path, claude_dir: str | None = None) -> subprocess.CompletedProcess:
    env = {"HOME": str(home), "PATH": "/usr/bin:/bin"}
    if claude_dir is not None:
        env["CLAUDE_DIR"] = claude_dir
    return subprocess.run(["/bin/sh", "-c", src], env=env, capture_output=True, text=True, timeout=10)


class FindMyStatusContract(unittest.TestCase):
    def test_is_tier_1_with_no_params(self) -> None:
        entry = _entry()
        self.assertEqual(entry.tier, 1)
        self.assertEqual(entry.params, ())
        self.assertEqual(entry.language, "AppleScript")

    def test_body_never_drives_an_app(self) -> None:
        body = _entry().body.lower()
        for forbidden in ("activate", "keystroke", "key code", "click", "tell application",
                          "system events", "--address", "open -a"):
            self.assertNotIn(forbidden, body, msg=f"status body must not contain {forbidden!r}")

    def test_body_asks_only_for_the_brief_status(self) -> None:
        src = _shell_part(_entry().body)
        self.assertIn('"$d/bin/grip-find-my" status --brief', src)
        self.assertEqual(src.count("grip-find-my"), 2, "one existence check and one call, nothing else")

    def test_description_states_the_exit_codes_the_caller_sees(self) -> None:
        # Re-review of HAL#815 (2026-11-25): it said "fails with exit 3", but osascript exits 1.
        description = _entry().description
        for phrase in ("exits 3", "AppleScript error 3", "exits 1"):
            self.assertIn(phrase, description)


class _TempHome(unittest.TestCase):
    """A temporary HOME and the decoded shell source, shared by the shell tests."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="osa-findmy-"))
        self.src = _shell_part(_entry().body)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


class FindMyStatusShellBehaviour(_TempHome):
    def test_fails_cleanly_when_grip_is_absent(self) -> None:
        r = _run_shell(self.src, home=self.tmp)
        self.assertEqual(r.returncode, 3)
        self.assertEqual(r.stdout, "")
        self.assertEqual(r.stderr.strip(), "find-my: GRIP Find My is not installed on this Mac")

    def test_runs_the_brief_status_from_the_default_install(self) -> None:
        _fake_grip_find_my(self.tmp / ".claude" / "bin")
        r = _run_shell(self.src, home=self.tmp)
        self.assertEqual(r.returncode, 0, msg=r.stderr)
        self.assertEqual(r.stdout.strip(), "fake-grip-find-my status --brief")

    def test_claude_dir_overrides_the_default_install(self) -> None:
        other = self.tmp / "elsewhere"
        _fake_grip_find_my(other / "bin")
        r = _run_shell(self.src, home=self.tmp, claude_dir=str(other))
        self.assertEqual(r.returncode, 0, msg=r.stderr)
        self.assertEqual(r.stdout.strip(), "fake-grip-find-my status --brief")

    def test_a_non_executable_file_counts_as_absent(self) -> None:
        bin_dir = self.tmp / ".claude" / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "grip-find-my").write_text("#!/bin/sh\necho should-not-run\n", encoding="utf-8")
        r = _run_shell(self.src, home=self.tmp)
        self.assertEqual(r.returncode, 3)
        self.assertNotIn("should-not-run", r.stdout)


class FindMyStatusHardening(_TempHome):
    """HAL#817: malformed CLAUDE_DIR values, and the tool's own failure."""

    def test_a_claude_dir_with_a_space_still_runs_the_tool(self) -> None:
        other = self.tmp / "with space"
        _fake_grip_find_my(other / "bin")
        r = _run_shell(self.src, home=self.tmp, claude_dir=str(other))
        self.assertEqual(r.returncode, 0, msg=r.stderr)
        self.assertEqual(r.stdout.strip(), "fake-grip-find-my status --brief")

    def test_an_empty_claude_dir_falls_back_to_the_default_install(self) -> None:
        _fake_grip_find_my(self.tmp / ".claude" / "bin")
        r = _run_shell(self.src, home=self.tmp, claude_dir="")
        self.assertEqual(r.returncode, 0, msg=r.stderr)
        self.assertEqual(r.stdout.strip(), "fake-grip-find-my status --brief")

    def test_a_claude_dir_value_is_never_run_as_a_command(self) -> None:
        marker = self.tmp / "ran"
        r = _run_shell(self.src, home=self.tmp, claude_dir=f"$(touch {marker})")
        self.assertEqual(r.returncode, 3)
        self.assertFalse(marker.exists(), "the value of CLAUDE_DIR must stay text")

    def test_the_tools_own_exit_code_and_output_pass_through(self) -> None:
        bin_dir = self.tmp / ".claude" / "bin"
        bin_dir.mkdir(parents=True)
        tool = bin_dir / "grip-find-my"
        tool.write_text('#!/bin/sh\necho partial\necho "find-my: boom" >&2\nexit 7\n', encoding="utf-8")
        tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
        r = _run_shell(self.src, home=self.tmp)
        self.assertEqual(r.returncode, 7, "the wrapper must not mask the tool's exit code")
        self.assertEqual(r.stdout.strip(), "partial")
        self.assertEqual(r.stderr.strip(), "find-my: boom")


@unittest.skipUnless(sys.platform == "darwin" and shutil.which("osascript"), "needs macOS osascript")
class FindMyStatusUnderOsascript(unittest.TestCase):
    def test_body_parses_and_reports_the_missing_install(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="osa-findmy-osa-"))
        try:
            env = {"HOME": str(tmp), "PATH": "/usr/bin:/bin"}
            r = subprocess.run(["osascript", "-e", _entry().body], env=env,
                               capture_output=True, text=True, timeout=15)
            # do shell script turns the shell's exit 3 into AppleScript error 3,
            # which osascript reports as "... (3)" and exit 1, as the description says.
            self.assertEqual(r.returncode, 1)
            self.assertIn("GRIP Find My is not installed on this Mac (3)", r.stderr)
            self.assertNotIn("syntax error", r.stderr.lower())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
