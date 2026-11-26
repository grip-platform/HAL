"""Tests for HAL AutoDream memory consolidation engine."""
import os
import sys
import tempfile
import unittest

from lib.hal.autodream import (
    AutoDream,
    _compact_verbose,
    _content_hash,
    _deduplicate,
    _deduplicate_near,
    _fix_relative_dates,
    _jaccard_shingle_similarity,
    _offset_date,
    _partition,
    _prune_empty,
    _shingles,
    _tokenise,
)
from lib.hal.memory_engine import MemoryEngine, MemoryEntry


def _entry(filename: str, content: str, name: str = "", desc: str = "", mtype: str = "project") -> MemoryEntry:
    return MemoryEntry(filename=filename, name=name, description=desc, memory_type=mtype, content=content)


class TestContentHash(unittest.TestCase):

    def test_identical_content(self):
        self.assertEqual(_content_hash("hello world"), _content_hash("hello world"))

    def test_normalises_whitespace(self):
        self.assertEqual(_content_hash("hello  world"), _content_hash("hello world"))

    def test_case_insensitive(self):
        self.assertEqual(_content_hash("Hello"), _content_hash("hello"))

    def test_different_content(self):
        self.assertNotEqual(_content_hash("hello"), _content_hash("goodbye"))


class TestDeduplicate(unittest.TestCase):

    def test_no_duplicates(self):
        entries = [_entry("a.md", "unique content one"), _entry("b.md", "unique content two")]
        kept, actions = _deduplicate(entries)
        self.assertEqual(len(kept), 2)
        self.assertEqual(len(actions), 0)

    def test_removes_duplicate(self):
        entries = [_entry("a.md", "same content here"), _entry("b.md", "same content here")]
        kept, actions = _deduplicate(entries)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action, "merge")

    def test_keeps_longer_on_hash_match(self):
        """When two entries have identical normalised content, the longer one is kept."""
        entries = [_entry("a.md", "same content here"), _entry("b.md", "Same  Content  Here")]
        kept, actions = _deduplicate(entries)
        # Normalised hashes match (whitespace + case), so dedup fires
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(actions), 1)


class TestPartition(unittest.TestCase):

    def test_keeps_all(self):
        entries = [_entry("a.md", "good content")]
        kept, actions = _partition(entries, lambda e: None)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(actions), 0)

    def test_discards_all(self):
        from lib.hal.autodream import ConsolidationAction
        entries = [_entry("a.md", "bad")]
        kept, actions = _partition(entries, lambda e: ConsolidationAction("prune", e.filename, "test"))
        self.assertEqual(len(kept), 0)
        self.assertEqual(len(actions), 1)


class TestPruneEmpty(unittest.TestCase):

    def test_keeps_valid(self):
        entries = [_entry("a.md", "This is a valid memory entry with enough content.")]
        kept, actions = _prune_empty(entries)
        self.assertEqual(len(kept), 1)

    def test_removes_short(self):
        entries = [_entry("a.md", "tiny")]
        kept, actions = _prune_empty(entries)
        self.assertEqual(len(kept), 0)
        self.assertEqual(actions[0].action, "prune")


class TestFixRelativeDates(unittest.TestCase):

    def test_replaces_today(self):
        entries = [_entry("a.md", "Meeting today at noon")]
        updated, actions = _fix_relative_dates(entries)
        self.assertNotIn("today", updated[0].content.lower())
        self.assertEqual(len(actions), 1)

    def test_no_change_for_absolute_dates(self):
        entries = [_entry("a.md", "Meeting on 2026-03-15")]
        updated, actions = _fix_relative_dates(entries)
        self.assertEqual(len(actions), 0)


class TestCompactVerbose(unittest.TestCase):

    def test_no_change_for_short(self):
        entries = [_entry("a.md", "Short content")]
        updated, actions = _compact_verbose(entries, max_chars=100)
        self.assertEqual(len(actions), 0)

    def test_truncates_long(self):
        long_content = "This is a sentence. " * 200
        entries = [_entry("a.md", long_content)]
        updated, actions = _compact_verbose(entries, max_chars=100)
        self.assertEqual(len(actions), 1)
        self.assertLessEqual(len(updated[0].content), 100)


class TestOffsetDate(unittest.TestCase):

    def test_next_day(self):
        self.assertEqual(_offset_date("2026-04-01", 1), "2026-04-02")

    def test_previous_day(self):
        self.assertEqual(_offset_date("2026-04-02", -1), "2026-04-01")

    def test_month_boundary(self):
        self.assertEqual(_offset_date("2026-03-31", 1), "2026-04-01")


class TestTokenise(unittest.TestCase):

    def test_splits_on_non_word(self):
        self.assertEqual(_tokenise("hello, world!"), ["hello", "world"])

    def test_lowercases(self):
        self.assertEqual(_tokenise("Hello World"), ["hello", "world"])

    def test_empty(self):
        self.assertEqual(_tokenise("   "), [])


class TestShingles(unittest.TestCase):

    def test_exact_k_tokens(self):
        sh = _shingles("one two three four five", k=5)
        self.assertEqual(len(sh), 1)
        self.assertIn(("one", "two", "three", "four", "five"), sh)

    def test_short_text_empty(self):
        self.assertEqual(_shingles("a b", k=5), frozenset())
        self.assertEqual(_shingles("a b", k=3), frozenset())

    def test_sliding_window(self):
        sh = _shingles("alpha beta gamma delta epsilon zeta", k=5)
        self.assertEqual(len(sh), 2)

    def test_default_k_is_three(self):
        sh = _shingles("one two three four")
        self.assertEqual(len(sh), 2)
        self.assertIn(("one", "two", "three"), sh)
        self.assertIn(("two", "three", "four"), sh)


class TestJaccardShingle(unittest.TestCase):

    def test_identical(self):
        text = "the quick brown fox jumps over the lazy dog today"
        self.assertEqual(_jaccard_shingle_similarity(text, text), 1.0)

    def test_disjoint(self):
        a = "the quick brown fox jumps over the lazy dog today"
        b = "entirely different content with unrelated vocabulary items here now"
        self.assertLess(_jaccard_shingle_similarity(a, b), 0.1)

    def test_near_dup_one_extra_sentence(self):
        base = "the quick brown fox jumps over the lazy dog today for testing purposes"
        extended = base + " with one extra sentence"
        # k=3 default — base 13 tokens -> 11 shingles; extended 17 tokens -> 15;
        # shared = 11, union = 15, Jaccard = 0.733. Sits above the 0.6 threshold.
        self.assertGreaterEqual(_jaccard_shingle_similarity(base, extended), 0.7)

    def test_short_text_fallback(self):
        self.assertEqual(_jaccard_shingle_similarity("hello world", "hello world"), 1.0)
        self.assertEqual(_jaccard_shingle_similarity("hello", "goodbye"), 0.0)


class TestDeduplicateNear(unittest.TestCase):

    def test_empty_noop(self):
        kept, actions = _deduplicate_near([])
        self.assertEqual(kept, [])
        self.assertEqual(actions, [])

    def test_single_entry_noop(self):
        entries = [_entry("a.md", "content one here in isolation")]
        kept, actions = _deduplicate_near(entries)
        self.assertEqual(len(kept), 1)
        self.assertEqual(actions, [])

    def test_merges_near_duplicate(self):
        base = "the quick brown fox jumps over the lazy dog today for testing purposes"
        extended = base + " and more"
        # k=3 default: A has 11 shingles, B has 13; shared=11, union=13, Jaccard=0.846.
        entries = [_entry("short.md", base), _entry("long.md", extended)]
        kept, actions = _deduplicate_near(entries)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].filename, "long.md")
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action, "merge")
        self.assertIn("Near-duplicate of long.md", actions[0].reason)

    def test_keeps_distinct_entries(self):
        entries = [
            _entry("a.md", "the quick brown fox jumps over the lazy dog in the morning"),
            _entry("b.md", "entirely different content about unrelated topics with distinct vocabulary tokens"),
        ]
        kept, actions = _deduplicate_near(entries)
        self.assertEqual(len(kept), 2)
        self.assertEqual(actions, [])


class TestAutoDreamIntegration(unittest.TestCase):

    def test_consolidate_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MemoryEngine(memory_dir=tmpdir)
            dream = AutoDream(memory_engine=engine)
            report = dream.consolidate()
            self.assertEqual(report.entries_before, 0)
            self.assertEqual(report.entries_after, 0)

    def test_consolidate_prunes_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MemoryEngine(memory_dir=tmpdir)
            engine.write_entry("good.md", "Good", "Valid entry", "project", "This entry has enough valid content to survive.")
            engine.write_entry("bad.md", "Bad", "Empty", "project", "tiny")
            dream = AutoDream(memory_engine=engine)
            report = dream.consolidate()
            self.assertEqual(report.entries_before, 2)
            self.assertEqual(report.entries_after, 1)
            pruned = [a for a in report.actions if a.action == "prune"]
            self.assertEqual(len(pruned), 1)


if __name__ == "__main__":
    unittest.main()
