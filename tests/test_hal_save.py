"""Tests for ``lib.hal.save`` — HAL session-save subcommand."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from lib.hal.save import (
    SaveEntry,
    instance_next_up_path,
    main,
    save_session,
    session_context_path,
    session_log_path,
)


class PathRooting(unittest.TestCase):
    """All save paths resolve under HAL_HOME."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="hal-save-p-")
        os.environ["HAL_HOME"] = self._tmp

    def tearDown(self) -> None:
        os.environ.pop("HAL_HOME", None)
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_log_under_hal_home(self) -> None:
        self.assertEqual(
            session_log_path(),
            Path(self._tmp) / "state" / "session-log.jsonl",
        )

    def test_context_under_hal_home(self) -> None:
        self.assertEqual(
            session_context_path(),
            Path(self._tmp) / "state" / "session-context.md",
        )

    def test_next_up_under_hal_home(self) -> None:
        self.assertEqual(
            instance_next_up_path(),
            Path(self._tmp) / "state" / "instance-next-up.txt",
        )


class SaveSessionBehaviour(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="hal-save-b-")
        os.environ["HAL_HOME"] = self._tmp

    def tearDown(self) -> None:
        os.environ.pop("HAL_HOME", None)
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_appends_jsonl_entry(self) -> None:
        save_session("first achievement")
        lines = session_log_path().read_text().strip().splitlines()
        self.assertEqual(len(lines), 1)
        payload = json.loads(lines[0])
        self.assertEqual(payload["achievement"], "first achievement")

    def test_multiple_saves_accumulate(self) -> None:
        save_session("one")
        save_session("two")
        save_session("three")
        lines = session_log_path().read_text().strip().splitlines()
        self.assertEqual(len(lines), 3)

    def test_context_md_contains_latest_entry(self) -> None:
        save_session("latest achievement")
        md = session_context_path().read_text()
        self.assertIn("latest achievement", md)
        self.assertIn("# HAL Session Context", md)

    def test_next_up_written_when_achievement_present(self) -> None:
        result = save_session("something-done")
        self.assertTrue(result.next_up_written)
        self.assertIn("something-done", instance_next_up_path().read_text())

    def test_next_up_skipped_when_achievement_empty(self) -> None:
        result = save_session("")
        self.assertFalse(result.next_up_written)
        self.assertFalse(instance_next_up_path().exists())

    def test_result_exposes_entry_timestamp(self) -> None:
        result = save_session("x")
        self.assertTrue(result.entry.ts.endswith("Z"))


class MainCli(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="hal-save-c-")
        os.environ["HAL_HOME"] = self._tmp

    def tearDown(self) -> None:
        os.environ.pop("HAL_HOME", None)
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_main_with_positional_argv(self) -> None:
        rc = main(["shipped", "phase-b"])
        self.assertEqual(rc, 0)
        payload = json.loads(session_log_path().read_text().strip().splitlines()[0])
        self.assertEqual(payload["achievement"], "shipped phase-b")

    def test_main_empty_args_still_writes_entry(self) -> None:
        rc = main([])
        self.assertEqual(rc, 0)
        self.assertTrue(session_log_path().exists())


if __name__ == "__main__":
    unittest.main()
