"""Tests for the Obsidian vault indexer — parsing, indexing, search, backlinks."""
import os
import tempfile
import unittest

from lib.hal.obsidian_vault import (
    VaultIndexer,
    VaultNote,
    _extract_title,
    _is_hidden,
    _parse_note,
    _score_note,
    _split_frontmatter,
    _WIKILINK_RE,
    _TAG_RE,
)
from pathlib import Path


class TestWikilinkRegex(unittest.TestCase):
    def test_simple_link(self):
        self.assertEqual(_WIKILINK_RE.findall("See [[My Note]]"), ["My Note"])

    def test_aliased_link(self):
        self.assertEqual(_WIKILINK_RE.findall("See [[My Note|alias]]"), ["My Note"])

    def test_multiple_links(self):
        text = "[[A]] and [[B/C]] connected"
        self.assertEqual(_WIKILINK_RE.findall(text), ["A", "B/C"])

    def test_no_links(self):
        self.assertEqual(_WIKILINK_RE.findall("plain text"), [])


class TestTagRegex(unittest.TestCase):
    def test_simple_tag(self):
        self.assertIn("project", _TAG_RE.findall("text #project here"))

    def test_nested_tag(self):
        self.assertIn("project/hal", _TAG_RE.findall("#project/hal done"))

    def test_no_tag_on_numbers(self):
        self.assertEqual(_TAG_RE.findall("#123"), [])


class TestSplitFrontmatter(unittest.TestCase):
    def test_with_frontmatter(self):
        raw = "---\ntitle: Test\ntags: ai\n---\nBody text here"
        fm, body = _split_frontmatter(raw)
        self.assertEqual(fm["title"], "Test")
        self.assertEqual(fm["tags"], "ai")
        self.assertIn("Body text here", body)

    def test_without_frontmatter(self):
        raw = "Just body text"
        fm, body = _split_frontmatter(raw)
        self.assertEqual(fm, {})
        self.assertEqual(body, raw)


class TestExtractTitle(unittest.TestCase):
    def test_h1_heading(self):
        self.assertEqual(_extract_title("note.md", "# My Title\nContent"), "My Title")

    def test_fallback_to_filename(self):
        self.assertEqual(_extract_title("Projects/HAL.md", "No heading here"), "HAL")

    def test_ignores_h2(self):
        self.assertEqual(_extract_title("note.md", "## Not H1\nContent"), "note")


class TestIsHidden(unittest.TestCase):
    def test_obsidian_dir(self):
        self.assertTrue(_is_hidden(".obsidian/config.md"))

    def test_git_dir(self):
        self.assertTrue(_is_hidden(".git/HEAD.md"))

    def test_normal_path(self):
        self.assertFalse(_is_hidden("Projects/HAL.md"))

    def test_deep_hidden(self):
        self.assertTrue(_is_hidden("notes/.trash/old.md"))


class TestScoreNote(unittest.TestCase):
    def test_title_match_weighted(self):
        note = VaultNote("a.md", "Machine Learning", "some body", (), (), {}, 0)
        score = _score_note(note, {"machine", "learning"})
        self.assertGreater(score, 0)

    def test_body_match(self):
        note = VaultNote("a.md", "Title", "machine learning is useful", (), (), {}, 0)
        score = _score_note(note, {"machine"})
        self.assertGreater(score, 0)

    def test_no_match(self):
        note = VaultNote("a.md", "Title", "body text", (), (), {}, 0)
        score = _score_note(note, {"quantum"})
        self.assertEqual(score, 0)

    def test_tag_match(self):
        note = VaultNote("a.md", "Title", "body", ("ai",), (), {}, 0)
        score = _score_note(note, {"ai"})
        self.assertGreater(score, 0)


class TestVaultIndexer(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        # Create test vault structure
        os.makedirs(os.path.join(self._tmpdir, "Projects"))
        os.makedirs(os.path.join(self._tmpdir, ".obsidian"))

        # Note 1: with frontmatter and wikilinks
        with open(os.path.join(self._tmpdir, "Projects", "HAL.md"), "w") as f:
            f.write("---\ntitle: HAL Project\nstatus: active\n---\n"
                    "# HAL\n\nHAL is a provider-agnostic LLM runtime. See [[RAILLM]].\n"
                    "#project #ai\n")

        # Note 2: simple note
        with open(os.path.join(self._tmpdir, "RAILLM.md"), "w") as f:
            f.write("# RAILLM\n\nRedundant Array of Independent LLMs.\n"
                    "Related to [[Projects/HAL]] and [[Fleet Command]].\n")

        # Hidden note (should be skipped)
        with open(os.path.join(self._tmpdir, ".obsidian", "config.md"), "w") as f:
            f.write("Should not be indexed")

    def test_index_finds_notes(self):
        indexer = VaultIndexer(self._tmpdir)
        report = indexer.index(force=True)
        self.assertEqual(report.indexed, 2)
        self.assertEqual(report.errors, 0)

    def test_hidden_notes_skipped(self):
        indexer = VaultIndexer(self._tmpdir)
        report = indexer.index(force=True)
        paths = [n.path for n in report.notes]
        self.assertNotIn(".obsidian/config.md", paths)

    def test_incremental_index(self):
        indexer = VaultIndexer(self._tmpdir)
        report1 = indexer.index(force=True)
        self.assertEqual(report1.indexed, 2)

        report2 = indexer.index()
        self.assertEqual(report2.indexed, 0)
        self.assertEqual(report2.skipped, 2)

    def test_wikilinks_extracted(self):
        indexer = VaultIndexer(self._tmpdir)
        report = indexer.index(force=True)
        hal_note = next(n for n in report.notes if "HAL" in n.path)
        self.assertIn("RAILLM", hal_note.wikilinks)

    def test_tags_extracted(self):
        indexer = VaultIndexer(self._tmpdir)
        report = indexer.index(force=True)
        hal_note = next(n for n in report.notes if "HAL" in n.path)
        self.assertIn("project", hal_note.tags)
        self.assertIn("ai", hal_note.tags)

    def test_frontmatter_parsed(self):
        indexer = VaultIndexer(self._tmpdir)
        report = indexer.index(force=True)
        hal_note = next(n for n in report.notes if "HAL" in n.path)
        self.assertEqual(hal_note.frontmatter.get("status"), "active")

    def test_search(self):
        indexer = VaultIndexer(self._tmpdir)
        indexer.index(force=True)
        results = indexer.search("RAILLM")
        self.assertGreater(len(results), 0)
        titles = [r.title for r in results]
        self.assertIn("RAILLM", titles)

    def test_backlinks(self):
        indexer = VaultIndexer(self._tmpdir)
        indexer.index(force=True)
        back = indexer.backlinks("RAILLM.md")
        self.assertIn("Projects/HAL.md", back)

    def test_read_note(self):
        indexer = VaultIndexer(self._tmpdir)
        content = indexer.read_note("RAILLM.md")
        self.assertIn("Redundant Array", content)

    def test_read_nonexistent(self):
        indexer = VaultIndexer(self._tmpdir)
        self.assertIsNone(indexer.read_note("nonexistent.md"))

    def test_graph(self):
        indexer = VaultIndexer(self._tmpdir)
        indexer.index(force=True)
        g = indexer.graph("Projects/HAL.md")
        self.assertIn("RAILLM", g["forward_links"])
        self.assertEqual(g["note"], "Projects/HAL.md")

    def test_nonexistent_vault(self):
        indexer = VaultIndexer("/nonexistent/vault")
        report = indexer.index()
        self.assertEqual(report.indexed, 0)


if __name__ == "__main__":
    unittest.main()
