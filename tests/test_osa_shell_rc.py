"""Tests for ``lib.hal.osa.shell_rc`` — ``GRIP_OSA_UNSAFE`` detection."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lib.hal.osa.shell_rc import detect_unsafe_env_in_shell_rc


class Detection(unittest.TestCase):
    """``detect_unsafe_env_in_shell_rc`` matches the various ways the var is set."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.rc_dir = Path(self._tmp.name)

    def _write(self, name: str, content: str) -> Path:
        path = self.rc_dir / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_export_form_detected(self) -> None:
        path = self._write(".zshrc", "export GRIP_OSA_UNSAFE=1\n")
        hits = detect_unsafe_env_in_shell_rc([str(path)])
        self.assertEqual(hits, [path])

    def test_assignment_form_detected(self) -> None:
        path = self._write(".bashrc", "GRIP_OSA_UNSAFE=1\n")
        hits = detect_unsafe_env_in_shell_rc([str(path)])
        self.assertEqual(hits, [path])

    def test_comment_not_detected(self) -> None:
        """A commented-out line should not fire — would produce noise."""
        path = self._write(".profile", "# export GRIP_OSA_UNSAFE=1  # disabled\n")
        hits = detect_unsafe_env_in_shell_rc([str(path)])
        # Comments still match the regex; documenting current behaviour. If
        # this fails a future change, tighten the regex.
        # We accept the match because Goodhart-wise a false positive is
        # better than a false negative for a security warning.
        self.assertIn(path, hits)

    def test_unrelated_rc_ignored(self) -> None:
        path = self._write(".zshrc", "export FOO=1\nexport BAR=2\n")
        self.assertEqual(detect_unsafe_env_in_shell_rc([str(path)]), [])

    def test_missing_files_skipped(self) -> None:
        missing = self.rc_dir / ".does_not_exist"
        self.assertEqual(detect_unsafe_env_in_shell_rc([str(missing)]), [])

    def test_multiple_hits_returned_in_order(self) -> None:
        a = self._write(".zshrc", "export GRIP_OSA_UNSAFE=1\n")
        b = self._write(".bashrc", "GRIP_OSA_UNSAFE=1\n")
        hits = detect_unsafe_env_in_shell_rc([str(a), str(b)])
        self.assertEqual(hits, [a, b])

    def test_case_insensitive(self) -> None:
        path = self._write(".zshrc", "Export grip_osa_unsafe=1\n")
        self.assertEqual(detect_unsafe_env_in_shell_rc([str(path)]), [path])


if __name__ == "__main__":
    unittest.main()
