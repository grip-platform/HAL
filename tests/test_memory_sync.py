"""Tests for HAL Memory Sync — HAL <-> optional peer harness memory."""
import os
import tempfile
import time
import unittest

from lib.hal.memory_sync import MemorySync, SyncReport


class TestSyncReport(unittest.TestCase):

    def test_default_success(self):
        r = SyncReport()
        self.assertTrue(r.success)
        self.assertEqual(r.copied, 0)

    def test_error_means_not_success(self):
        r = SyncReport(errors=["fail"])
        self.assertFalse(r.success)

    def test_to_dict(self):
        r = SyncReport(copied=2, skipped=1, conflicts=1)
        d = r.to_dict()
        self.assertEqual(d["copied"], 2)
        self.assertTrue(d["success"])


class TestMemorySync(unittest.TestCase):

    def setUp(self):
        self.peer_dir = tempfile.mkdtemp()
        self.hal_dir = tempfile.mkdtemp()
        self.sync = MemorySync(peer_dir=self.peer_dir, hal_dir=self.hal_dir)

    def _write(self, directory, name, content, mtime=None):
        path = os.path.join(directory, name)
        with open(path, "w") as f:
            f.write(content)
        if mtime:
            os.utime(path, (mtime, mtime))

    def test_peer_only_copies_to_hal(self):
        self._write(self.peer_dir, "note.md", "peer content")
        report = self.sync.sync()
        self.assertEqual(report.copied, 1)
        with open(os.path.join(self.hal_dir, "note.md")) as f:
            self.assertEqual(f.read(), "peer content")

    def test_hal_only_copies_to_peer(self):
        self._write(self.hal_dir, "hal_note.md", "hal content")
        report = self.sync.sync()
        self.assertEqual(report.copied, 1)
        with open(os.path.join(self.peer_dir, "hal_note.md")) as f:
            self.assertEqual(f.read(), "hal content")

    def test_identical_files_skipped(self):
        now = time.time()
        self._write(self.peer_dir, "same.md", "content", mtime=now)
        self._write(self.hal_dir, "same.md", "content", mtime=now)
        report = self.sync.sync()
        self.assertEqual(report.skipped, 1)
        self.assertEqual(report.copied, 0)

    def test_newer_peer_wins(self):
        old = time.time() - 100
        new = time.time()
        self._write(self.peer_dir, "conflict.md", "new peer", mtime=new)
        self._write(self.hal_dir, "conflict.md", "old hal", mtime=old)
        report = self.sync.sync()
        self.assertEqual(report.conflicts, 1)
        with open(os.path.join(self.hal_dir, "conflict.md")) as f:
            self.assertEqual(f.read(), "new peer")

    def test_newer_hal_wins(self):
        old = time.time() - 100
        new = time.time()
        self._write(self.peer_dir, "conflict.md", "old peer", mtime=old)
        self._write(self.hal_dir, "conflict.md", "new hal", mtime=new)
        report = self.sync.sync()
        self.assertEqual(report.conflicts, 1)
        with open(os.path.join(self.peer_dir, "conflict.md")) as f:
            self.assertEqual(f.read(), "new hal")

    def test_status_without_modifying(self):
        self._write(self.peer_dir, "only_peer.md", "a")
        self._write(self.hal_dir, "only_hal.md", "b")
        self._write(self.peer_dir, "shared.md", "c")
        self._write(self.hal_dir, "shared.md", "c")
        status = self.sync.status()
        self.assertEqual(status["peer_only"], ["only_peer.md"])
        self.assertEqual(status["hal_only"], ["only_hal.md"])
        self.assertEqual(status["shared"], 1)

    def test_ignores_non_md_files(self):
        self._write(self.peer_dir, "data.json", "{}")
        report = self.sync.sync()
        self.assertEqual(report.copied, 0)

    def test_empty_dirs(self):
        report = self.sync.sync()
        self.assertEqual(report.copied, 0)
        self.assertTrue(report.success)

    def test_no_peer_is_noop(self):
        sync = MemorySync(hal_dir=self.hal_dir)
        self._write(self.hal_dir, "note.md", "content")
        report = sync.sync()
        self.assertEqual(report.copied, 0)
        self.assertTrue(report.success)
        self.assertFalse(sync.enabled)


if __name__ == "__main__":
    unittest.main()
