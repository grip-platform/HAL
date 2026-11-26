"""HAL Provenance — walk the 1.1 MemoryEntry graph for a claim.

Phase β W2. Finds memory entries matching a query and renders a
provenance tree showing which session wrote each entry, which tool
calls fed it, and which council votes attested it.

Pure function today. Kono semantic recall will slot in later via the
``recall_fn`` injection point — current implementation uses case-
insensitive substring matching over name + description + content so the
module is deterministic and testable without the embedding engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from lib.hal.memory_engine import MemoryEntry


RecallFn = Callable[[str, list[MemoryEntry]], list[MemoryEntry]]


@dataclass(frozen=True)
class ProvenanceNode:
    """One row in the rendered provenance tree."""

    entry: MemoryEntry
    depth: int  # 0 = root match; 1 = sibling from same session; 2 = tool/vote leaf
    kind: str   # "match" | "sibling" | "tool" | "vote"
    label: str  # Text to render


def default_recall(query: str, entries: list[MemoryEntry]) -> list[MemoryEntry]:
    """Simple substring recall over name + description + content (case-insensitive)."""
    if not query:
        return []
    q = query.lower()
    hits = []
    for e in entries:
        haystack = f"{e.name}\n{e.description}\n{e.content}".lower()
        if q in haystack:
            hits.append(e)
    return hits


def build_tree(
    query: str,
    entries: list[MemoryEntry],
    *,
    recall_fn: RecallFn | None = None,
) -> list[ProvenanceNode]:
    """Build a flat tree of ``ProvenanceNode`` for ``query`` over ``entries``.

    Layout (BFS, alphabetical inside each level):

        root-match entry
          sibling from same session_id
          tool: Read
          tool: Grep
          vote: sha1-abcd
    """
    recall = recall_fn or default_recall
    matched = recall(query, entries)
    if not matched:
        return []

    # Index all entries by session_id for sibling lookup.
    by_session: dict[str, list[MemoryEntry]] = {}
    for e in entries:
        if e.source_session_id:
            by_session.setdefault(e.source_session_id, []).append(e)

    out: list[ProvenanceNode] = []
    for root in sorted(matched, key=lambda e: e.filename):
        out.append(_match_node(root))
        out.extend(_sibling_nodes(root, by_session))
        out.extend(_tool_nodes(root))
        out.extend(_vote_nodes(root))
    return out


def render_tree(nodes: list[ProvenanceNode]) -> str:
    """Render a flat node list as indented ASCII tree."""
    if not nodes:
        return "no provenance matches\n"
    lines = []
    for node in nodes:
        indent = "  " * node.depth
        bullet = {"match": "●", "sibling": "↳", "tool": "·", "vote": "✓"}[node.kind]
        lines.append(f"{indent}{bullet} {node.label}")
    return "\n".join(lines) + "\n"


def provenance_text(
    query: str,
    entries: list[MemoryEntry],
    *,
    recall_fn: RecallFn | None = None,
) -> str:
    """Convenience: build + render in one call."""
    return render_tree(build_tree(query, entries, recall_fn=recall_fn))


# ---------------------------------------------------------------------------
# Internal helpers — one per node kind keeps CC low.
# ---------------------------------------------------------------------------
def _match_node(entry: MemoryEntry) -> ProvenanceNode:
    label = f"{entry.filename} — {entry.name or '(unnamed)'}"
    if entry.source_session_id:
        label += f" [session: {entry.source_session_id}]"
    return ProvenanceNode(entry=entry, depth=0, kind="match", label=label)


def _sibling_nodes(
    root: MemoryEntry,
    by_session: dict[str, list[MemoryEntry]],
) -> list[ProvenanceNode]:
    if not root.source_session_id:
        return []
    siblings = [
        e for e in by_session.get(root.source_session_id, [])
        if e.filename != root.filename
    ]
    siblings.sort(key=lambda e: e.filename)
    return [
        ProvenanceNode(
            entry=e, depth=1, kind="sibling",
            label=f"{e.filename} — {e.name or '(unnamed)'}",
        )
        for e in siblings
    ]


def _tool_nodes(entry: MemoryEntry) -> list[ProvenanceNode]:
    return [
        ProvenanceNode(entry=entry, depth=1, kind="tool", label=f"tool: {t}")
        for t in entry.tool_calls_used
    ]


def _vote_nodes(entry: MemoryEntry) -> list[ProvenanceNode]:
    return [
        ProvenanceNode(entry=entry, depth=1, kind="vote", label=f"vote: {h}")
        for h in entry.council_vote_hashes
    ]


__all__ = [
    "ProvenanceNode",
    "build_tree",
    "default_recall",
    "provenance_text",
    "render_tree",
]


def _iter_entries(source: Iterable[MemoryEntry]) -> list[MemoryEntry]:
    """Defensive coercion for callers who pass generators."""
    return list(source)
