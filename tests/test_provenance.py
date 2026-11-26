"""Tests for Phase β W2 /provenance command."""
from __future__ import annotations

import unittest

from lib.hal.memory_engine import MemoryEntry
from lib.hal.provenance import (
    ProvenanceNode,
    build_tree,
    default_recall,
    provenance_text,
    render_tree,
)


def _entry(fn, name="", desc="", content="", session="", tools=None, votes=None):
    return MemoryEntry(
        filename=fn, name=name, description=desc, content=content,
        source_session_id=session,
        tool_calls_used=tools or [],
        council_vote_hashes=votes or [],
    )


class TestDefaultRecall(unittest.TestCase):
    def test_empty_query_returns_empty(self):
        self.assertEqual(default_recall("", [_entry("a.md", content="x")]), [])

    def test_matches_name(self):
        entries = [_entry("a.md", name="Phase alpha notes")]
        self.assertEqual(len(default_recall("alpha", entries)), 1)

    def test_matches_description(self):
        entries = [_entry("a.md", desc="cost heatmap")]
        self.assertEqual(len(default_recall("heatmap", entries)), 1)

    def test_matches_content(self):
        entries = [_entry("a.md", content="H091 is the semantic cache hypothesis")]
        self.assertEqual(len(default_recall("semantic cache", entries)), 1)

    def test_case_insensitive(self):
        entries = [_entry("a.md", name="Phase ALPHA")]
        self.assertEqual(len(default_recall("alpha", entries)), 1)

    def test_no_match_returns_empty(self):
        entries = [_entry("a.md", name="x", content="y")]
        self.assertEqual(default_recall("nonexistent", entries), [])


class TestBuildTree(unittest.TestCase):
    def test_no_matches_returns_empty(self):
        self.assertEqual(build_tree("xyz", [_entry("a.md", name="a")]), [])

    def test_single_match_produces_match_node(self):
        tree = build_tree("alpha", [_entry("a.md", name="alpha")])
        self.assertEqual(len(tree), 1)
        self.assertEqual(tree[0].kind, "match")
        self.assertEqual(tree[0].depth, 0)

    def test_match_with_session_walks_siblings(self):
        """The preplan spec: synthetic 3-entry chain round-trips correctly."""
        entries = [
            _entry("a.md", name="root match", content="alpha claim", session="sess-1"),
            _entry("b.md", name="sibling B", session="sess-1"),
            _entry("c.md", name="sibling C", session="sess-1"),
            _entry("d.md", name="unrelated", session="sess-2"),
        ]
        tree = build_tree("alpha", entries)
        kinds = [n.kind for n in tree]
        # match + 2 siblings (B and C, filtered out self and unrelated session)
        self.assertEqual(kinds.count("match"), 1)
        self.assertEqual(kinds.count("sibling"), 2)
        sibling_files = [n.entry.filename for n in tree if n.kind == "sibling"]
        self.assertEqual(sorted(sibling_files), ["b.md", "c.md"])

    def test_tool_and_vote_nodes_attached(self):
        entries = [_entry(
            "a.md", name="root", content="target",
            tools=["Read", "Grep"],
            votes=["sha-aaa", "sha-bbb"],
        )]
        tree = build_tree("target", entries)
        kinds = [n.kind for n in tree]
        self.assertEqual(kinds.count("tool"), 2)
        self.assertEqual(kinds.count("vote"), 2)
        tool_labels = [n.label for n in tree if n.kind == "tool"]
        self.assertIn("tool: Read", tool_labels)
        self.assertIn("tool: Grep", tool_labels)

    def test_match_without_session_has_no_siblings(self):
        entries = [
            _entry("a.md", name="root match", content="alpha"),
            _entry("b.md", name="also no session"),
        ]
        tree = build_tree("alpha", entries)
        kinds = [n.kind for n in tree]
        self.assertEqual(kinds, ["match"])

    def test_multiple_matches_stable_ordering(self):
        entries = [
            _entry("b.md", name="z match one", content="target"),
            _entry("a.md", name="a match two", content="target"),
        ]
        tree = build_tree("target", entries)
        matches = [n for n in tree if n.kind == "match"]
        # Sorted by filename asc.
        self.assertEqual([m.entry.filename for m in matches], ["a.md", "b.md"])

    def test_custom_recall_fn_injection(self):
        entries = [_entry("a.md", name="always"), _entry("b.md", name="never")]
        tree = build_tree("anything", entries, recall_fn=lambda q, es: [es[0]])
        self.assertEqual(len(tree), 1)
        self.assertEqual(tree[0].entry.filename, "a.md")


class TestRenderTree(unittest.TestCase):
    def test_empty_tree_message(self):
        self.assertIn("no provenance", render_tree([]))

    def test_indentation_scales_with_depth(self):
        nodes = [
            ProvenanceNode(_entry("a.md"), depth=0, kind="match", label="root"),
            ProvenanceNode(_entry("b.md"), depth=1, kind="sibling", label="child"),
        ]
        out = render_tree(nodes)
        lines = out.strip().splitlines()
        self.assertFalse(lines[0].startswith(" "))
        self.assertTrue(lines[1].startswith("  "))

    def test_bullet_per_kind(self):
        nodes = [
            ProvenanceNode(_entry("a.md"), 0, "match", "m"),
            ProvenanceNode(_entry("a.md"), 1, "sibling", "s"),
            ProvenanceNode(_entry("a.md"), 1, "tool", "t"),
            ProvenanceNode(_entry("a.md"), 1, "vote", "v"),
        ]
        out = render_tree(nodes)
        self.assertIn("●", out)
        self.assertIn("↳", out)
        self.assertIn("·", out)
        self.assertIn("✓", out)


class TestProvenanceTextEndToEnd(unittest.TestCase):
    def test_chain_renders_expected_structure(self):
        entries = [
            _entry("a.md", name="Phase α 6/6 shipped", content="6 PRs",
                   session="s1", tools=["Read", "Grep"], votes=["h1"]),
            _entry("b.md", name="Phase α 4/6 superseded", session="s1"),
            _entry("c.md", name="unrelated", session="s2"),
        ]
        text = provenance_text("phase α", entries)
        self.assertIn("a.md", text)
        self.assertIn("session: s1", text)
        self.assertIn("b.md", text)
        self.assertIn("tool: Read", text)
        self.assertIn("vote: h1", text)
        # Unrelated entry (different session) MUST NOT appear as sibling.
        self.assertNotIn("c.md", text)


if __name__ == "__main__":
    unittest.main()
