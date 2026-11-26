"""Tests for HAL diff tracker."""
import os
import sys
import tempfile
import unittest

from lib.hal.diff_tracker import DiffTracker, FileChange, _file_hash


class TestFileHash(unittest.TestCase):

    def test_deterministic(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("content")
            path = f.name
        try:
            self.assertEqual(_file_hash(path), _file_hash(path))
        finally:
            os.unlink(path)

    def test_nonexistent(self):
        self.assertEqual(_file_hash("/nonexistent"), "")


class TestDiffTracker(unittest.TestCase):

    def test_no_changes(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "t.py"), "w") as f:
                f.write("x")
            t = DiffTracker(d)
            t.snapshot("a")
            t.snapshot("b")
            self.assertEqual(len(t.diff("a", "b").changes), 0)

    def test_added(self):
        with tempfile.TemporaryDirectory() as d:
            t = DiffTracker(d)
            t.snapshot("before")
            with open(os.path.join(d, "new.py"), "w") as f:
                f.write("new")
            t.snapshot("after")
            self.assertEqual(len(t.diff("before", "after").added), 1)

    def test_modified(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.py")
            with open(p, "w") as f:
                f.write("v1")
            t = DiffTracker(d)
            t.snapshot("a")
            with open(p, "w") as f:
                f.write("v2")
            t.snapshot("b")
            self.assertEqual(len(t.diff("a", "b").modified), 1)

    def test_deleted(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "g.py")
            with open(p, "w") as f:
                f.write("bye")
            t = DiffTracker(d)
            t.snapshot("a")
            os.unlink(p)
            t.snapshot("b")
            self.assertEqual(len(t.diff("a", "b").deleted), 1)

    def test_summary(self):
        t = DiffTracker("/tmp")
        t._snapshots["a"] = {}
        t._snapshots["b"] = {"n.py": ("h", 100)}
        self.assertIn("1 added", t.diff("a", "b").summary)

    def test_list_snapshots(self):
        t = DiffTracker("/tmp")
        t._snapshots["v1"] = {}
        self.assertEqual(t.list_snapshots(), ["v1"])

    def test_explicit_paths(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "a.py"), "w") as f:
                f.write("a")
            self.assertEqual(DiffTracker(d).snapshot("s", paths=["a.py"]), 1)


if __name__ == "__main__":
    unittest.main()
