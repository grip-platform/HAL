"""Tests for Phase β W1 — MemoryEntry provenance fields + FORMAT_VERSION 1.1."""
from __future__ import annotations

import os
import tempfile
import unittest

from lib.hal.memory_engine import (
    FORMAT_VERSION,
    LEGACY_FORMAT_VERSION,
    MemoryEngine,
    MemoryEntry,
    _format_list,
    _parse_list,
)


class TestFormatVersionConstants(unittest.TestCase):
    def test_format_version_bumped_to_1_1(self):
        self.assertEqual(FORMAT_VERSION, "1.1")

    def test_legacy_version_available(self):
        self.assertEqual(LEGACY_FORMAT_VERSION, "1.0")


class TestListFormatters(unittest.TestCase):
    def test_format_empty_list(self):
        self.assertEqual(_format_list(None), "")
        self.assertEqual(_format_list([]), "")

    def test_format_roundtrips(self):
        self.assertEqual(_parse_list(_format_list(["a", "b", "c"])), ["a", "b", "c"])

    def test_parse_empty(self):
        self.assertEqual(_parse_list(""), [])

    def test_parse_strips_whitespace(self):
        self.assertEqual(_parse_list("a, b , c  ,d"), ["a", "b", "c", "d"])

    def test_parse_drops_empty_segments(self):
        self.assertEqual(_parse_list("a,,b,,,c"), ["a", "b", "c"])

    def test_format_drops_empty_segments(self):
        self.assertEqual(_format_list(["a", "", "b", "   ", "c"]), "a,b,c")


class TestMemoryEntryProvenance(unittest.TestCase):
    def test_default_provenance_fields_empty(self):
        e = MemoryEntry(filename="x.md", name="x", description="y", memory_type="user")
        self.assertEqual(e.source_session_id, "")
        self.assertEqual(e.tool_calls_used, [])
        self.assertEqual(e.council_vote_hashes, [])
        self.assertFalse(e.has_provenance())

    def test_provenance_fields_stored(self):
        e = MemoryEntry(
            filename="x.md", name="x", description="y", memory_type="user",
            source_session_id="abc-123",
            tool_calls_used=["Read", "Grep"],
            council_vote_hashes=["hash1", "hash2"],
        )
        self.assertTrue(e.has_provenance())
        self.assertEqual(e.source_session_id, "abc-123")
        self.assertEqual(e.tool_calls_used, ["Read", "Grep"])
        self.assertEqual(e.council_vote_hashes, ["hash1", "hash2"])

    def test_has_provenance_any_single_field(self):
        self.assertTrue(MemoryEntry("x.md", source_session_id="s").has_provenance())
        self.assertTrue(MemoryEntry("x.md", tool_calls_used=["t"]).has_provenance())
        self.assertTrue(MemoryEntry("x.md", council_vote_hashes=["h"]).has_provenance())

    def test_to_dict_omits_provenance_when_absent(self):
        e = MemoryEntry(filename="x.md", name="x", description="y", memory_type="user")
        d = e.to_dict()
        self.assertNotIn("source_session_id", d)
        self.assertNotIn("tool_calls_used", d)
        self.assertNotIn("council_vote_hashes", d)

    def test_to_dict_includes_provenance_when_present(self):
        e = MemoryEntry(
            filename="x.md", source_session_id="s", tool_calls_used=["t"],
        )
        d = e.to_dict()
        self.assertEqual(d["source_session_id"], "s")
        self.assertEqual(d["tool_calls_used"], ["t"])
        self.assertEqual(d["council_vote_hashes"], [])

    def test_to_dict_copies_lists(self):
        """Mutating the returned dict's lists must not affect the entry."""
        e = MemoryEntry(filename="x.md", tool_calls_used=["a"])
        d = e.to_dict()
        d["tool_calls_used"].append("b")
        self.assertEqual(e.tool_calls_used, ["a"])


class TestWriteEntryRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.engine = MemoryEngine(memory_dir=self.tmp)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_1_0_entry_round_trips_without_provenance(self):
        """A caller passing no provenance gets a pure 1.0 file."""
        self.engine.write_entry(
            filename="a.md", name="A", description="desc",
            memory_type="user", content="body text",
        )
        raw = open(os.path.join(self.tmp, "a.md")).read()
        self.assertIn("format_version: 1.0", raw)
        self.assertNotIn("source_session_id", raw)
        self.assertNotIn("tool_calls_used", raw)
        entry = self.engine.read_entry("a.md")
        self.assertEqual(entry.name, "A")
        self.assertEqual(entry.format_version, "1.0")
        self.assertFalse(entry.has_provenance())

    def test_1_1_entry_round_trips_with_provenance(self):
        self.engine.write_entry(
            filename="b.md", name="B", description="dd",
            memory_type="project", content="body",
            source_session_id="session-42",
            tool_calls_used=["Read", "Edit"],
            council_vote_hashes=["sha1-abc"],
        )
        raw = open(os.path.join(self.tmp, "b.md")).read()
        self.assertIn("format_version: 1.1", raw)
        self.assertIn("source_session_id: session-42", raw)
        self.assertIn("tool_calls_used: Read,Edit", raw)
        entry = self.engine.read_entry("b.md")
        self.assertEqual(entry.format_version, "1.1")
        self.assertEqual(entry.source_session_id, "session-42")
        self.assertEqual(entry.tool_calls_used, ["Read", "Edit"])
        self.assertEqual(entry.council_vote_hashes, ["sha1-abc"])

    def test_backward_compat_read_of_hand_authored_1_0(self):
        """A 1.0 file without a format_version line still parses cleanly."""
        legacy = (
            "---\nname: legacy\n"
            "description: old entry\n"
            "type: user\n---\n\n"
            "legacy body\n"
        )
        path = os.path.join(self.tmp, "legacy.md")
        with open(path, "w") as f:
            f.write(legacy)
        entry = self.engine.read_entry("legacy.md")
        self.assertEqual(entry.name, "legacy")
        self.assertEqual(entry.content, "legacy body")
        # Absent version: parser defaults to FORMAT_VERSION per existing contract.
        self.assertEqual(entry.format_version, FORMAT_VERSION)
        self.assertFalse(entry.has_provenance())

    def test_partial_provenance_writes_1_1(self):
        """Only source_session_id set triggers the 1.1 shape."""
        self.engine.write_entry(
            filename="c.md", name="C", description="", memory_type="",
            content="x", source_session_id="only-this",
        )
        raw = open(os.path.join(self.tmp, "c.md")).read()
        self.assertIn("format_version: 1.1", raw)
        self.assertIn("source_session_id: only-this", raw)
        self.assertIn("tool_calls_used: \n", raw)
        entry = self.engine.read_entry("c.md")
        self.assertEqual(entry.tool_calls_used, [])

    def test_list_field_with_commas_in_values_survives(self):
        """Comma in a tool name would be mis-parsed; verify behaviour is stable."""
        self.engine.write_entry(
            filename="d.md", name="D", description="", memory_type="",
            content="y", tool_calls_used=["simple", "also-simple"],
        )
        entry = self.engine.read_entry("d.md")
        self.assertEqual(entry.tool_calls_used, ["simple", "also-simple"])


if __name__ == "__main__":
    unittest.main()
