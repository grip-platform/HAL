"""Tests for lib/hal/memory_engine — format freeze + backward compatibility."""
import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr

from lib.hal.memory_engine import (
    FORMAT_VERSION,
    LEGACY_FORMAT_VERSION,
    MemoryEngine,
    MemoryEntry,
    _parse_entry,
)


class TestFormatVersionConstant(unittest.TestCase):

    def test_constant_exposed(self):
        self.assertEqual(FORMAT_VERSION, "1.1")

    def test_module_docstring_mentions_frozen_schema(self):
        from lib.hal import memory_engine
        self.assertIn("FROZEN SCHEMA", memory_engine.__doc__)
        self.assertIn("format_version", memory_engine.__doc__)


class TestMemoryEntryDefaults(unittest.TestCase):

    def test_format_version_default(self):
        entry = MemoryEntry(filename="x.md", content="body")
        self.assertEqual(entry.format_version, FORMAT_VERSION)

    def test_to_dict_includes_format_version(self):
        entry = MemoryEntry(filename="x.md", content="body")
        d = entry.to_dict()
        self.assertIn("format_version", d)
        self.assertEqual(d["format_version"], FORMAT_VERSION)


class TestWriteEmitsFormatVersion(unittest.TestCase):

    def test_frontmatter_has_version_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MemoryEngine(memory_dir=tmpdir)
            engine.write_entry("note.md", "Note", "Desc", "project", "Body content here.")
            path = os.path.join(tmpdir, "note.md")
            with open(path, "r", encoding="utf-8") as fh:
                raw = fh.read()
            self.assertIn(f"format_version: {LEGACY_FORMAT_VERSION}", raw)

    def test_round_trip_preserves_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MemoryEngine(memory_dir=tmpdir)
            engine.write_entry("note.md", "Note", "Desc", "project", "Body content here.")
            entry = engine.read_entry("note.md")
            self.assertIsNotNone(entry)
            self.assertEqual(entry.format_version, LEGACY_FORMAT_VERSION)


class TestBackwardCompatibility(unittest.TestCase):

    def test_legacy_frontmatter_defaults_to_current_version(self):
        raw = (
            "---\nname: Legacy\ndescription: Older entry\n"
            "type: project\n---\n\nLegacy body."
        )
        entry = _parse_entry("legacy.md", raw)
        self.assertEqual(entry.format_version, FORMAT_VERSION)
        self.assertEqual(entry.name, "Legacy")

    def test_no_frontmatter_at_all_defaults_to_current_version(self):
        entry = _parse_entry("raw.md", "Just a body, no frontmatter.")
        self.assertEqual(entry.format_version, FORMAT_VERSION)
        self.assertEqual(entry.content, "Just a body, no frontmatter.")


class TestVersionMismatchWarns(unittest.TestCase):

    def test_mismatch_logs_to_stderr_but_parses(self):
        raw = (
            "---\nname: Future\ndescription: From a later version\n"
            "type: project\nformat_version: 2.0\n---\n\nFuture body."
        )
        buf = io.StringIO()
        with redirect_stderr(buf):
            entry = _parse_entry("future.md", raw)
        self.assertEqual(entry.format_version, "2.0")
        self.assertEqual(entry.name, "Future")
        self.assertIn("format_version", buf.getvalue())
        self.assertIn("2.0", buf.getvalue())

    def test_matching_version_silent(self):
        raw = (
            f"---\nname: Current\ndescription: Current\n"
            f"type: project\nformat_version: {FORMAT_VERSION}\n---\n\nBody."
        )
        buf = io.StringIO()
        with redirect_stderr(buf):
            _parse_entry("current.md", raw)
        self.assertEqual(buf.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
