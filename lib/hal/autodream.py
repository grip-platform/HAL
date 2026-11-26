"""HAL AutoDream — memory consolidation engine.

Runs between sessions to merge, deduplicate, and strengthen memories.
Inspired by CC's autoDream pattern but HAL-native: mechanical consolidation
rules + optional cross-model LLM enhancement.

Key principle: consolidation runs in isolation (separate context) to prevent
corrupting the main conversation. Memory is continuously EDITED, not appended.

Usage:
    from lib.hal.autodream import AutoDream

    dream = AutoDream()
    report = dream.consolidate()
"""
import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from lib.hal.home import HAL_HOME as _HAL_HOME
from lib.hal.memory_engine import MemoryEngine, MemoryEntry

_TOKEN_SPLIT_RE = re.compile(r"\W+")
# Jaccard threshold + shingle length tuned for memory-edit cases: an existing
# entry that gained a paragraph should cross the bar; two unrelated entries
# sharing topical vocabulary should not. k=3 follows the MinHash convention;
# 0.6 leaves room for small additions without catching distinct entries that
# happen to share a few n-grams.
_NEAR_DUP_THRESHOLD = 0.6
_SHINGLE_K = 3


@dataclass
class ConsolidationAction:
    """A single consolidation action taken during a dream cycle."""
    action: str  # merge, prune, update, strengthen
    target: str  # filename affected
    reason: str
    detail: str = ""


@dataclass
class DreamReport:
    """Report from a consolidation cycle."""
    actions: list[ConsolidationAction] = field(default_factory=list)
    entries_before: int = 0
    entries_after: int = 0
    duration_s: float = 0.0

    @property
    def summary(self) -> str:
        merged = sum(1 for a in self.actions if a.action == "merge")
        pruned = sum(1 for a in self.actions if a.action == "prune")
        updated = sum(1 for a in self.actions if a.action == "update")
        return (
            f"AutoDream: {self.entries_before} -> {self.entries_after} entries "
            f"({merged} merged, {pruned} pruned, {updated} updated) "
            f"in {self.duration_s:.1f}s"
        )


class AutoDream:
    """Memory consolidation engine — runs between sessions."""

    def __init__(self, memory_engine: MemoryEngine | None = None):
        self._engine = memory_engine or MemoryEngine()

    def consolidate(self) -> DreamReport:
        """Run a full consolidation cycle.

        Operations (in order):
        1. Deduplicate entries with similar content
        2. Prune entries referencing non-existent files
        3. Convert relative dates to absolute
        4. Remove empty or trivial entries
        5. Compact verbose entries
        """
        start = time.time()
        entries = self._engine._scan_entries()
        report = DreamReport(entries_before=len(entries))

        entries, actions = _deduplicate(entries)
        report.actions.extend(actions)

        entries, actions = _deduplicate_near(entries)
        report.actions.extend(actions)

        entries, actions = _prune_stale_file_refs(entries)
        report.actions.extend(actions)

        entries, actions = _fix_relative_dates(entries)
        report.actions.extend(actions)

        entries, actions = _prune_empty(entries)
        report.actions.extend(actions)

        entries, actions = _compact_verbose(entries)
        report.actions.extend(actions)

        # Write surviving entries back
        for entry in entries:
            self._engine.write_entry(
                entry.filename, entry.name, entry.description,
                entry.memory_type, entry.content,
            )

        # Remove deleted entries from disk
        surviving = {e.filename for e in entries}
        for fn in self._engine.list_entries():
            if fn not in surviving:
                self._engine.delete_entry(fn)

        report.entries_after = len(entries)
        report.duration_s = time.time() - start
        return report


def _partition(
    entries: list[MemoryEntry],
    should_discard: callable,
) -> tuple[list[MemoryEntry], list[ConsolidationAction]]:
    """Single-pass partition: keep entries where should_discard returns None.

    should_discard(entry) -> ConsolidationAction if discarded, None if kept.
    O(n) — one pass, no nested loops.
    """
    keep: list[MemoryEntry] = []
    actions: list[ConsolidationAction] = []
    for entry in entries:
        action = should_discard(entry)
        if action:
            actions.append(action)
        else:
            keep.append(entry)
    return keep, actions


def _deduplicate(entries: list[MemoryEntry]) -> tuple[list[MemoryEntry], list[ConsolidationAction]]:
    """Merge entries with similar content via hash-based O(n) dedup."""
    actions: list[ConsolidationAction] = []
    if len(entries) < 2:
        return entries, actions

    keep: list[MemoryEntry] = []
    seen_hashes: dict[str, int] = {}

    for entry in entries:
        content_hash = _content_hash(entry.content)
        if content_hash in seen_hashes:
            existing_idx = seen_hashes[content_hash]
            existing = keep[existing_idx]
            if len(entry.content) > len(existing.content):
                keep[existing_idx] = entry
                actions.append(ConsolidationAction(
                    "merge", entry.filename, f"Merged duplicate of {existing.filename}",
                ))
            else:
                actions.append(ConsolidationAction(
                    "merge", existing.filename, f"Merged duplicate of {entry.filename}",
                ))
        else:
            seen_hashes[content_hash] = len(keep)
            keep.append(entry)

    return keep, actions


def _tokenise(text: str) -> list[str]:
    return [t for t in _TOKEN_SPLIT_RE.split(text.strip().lower()) if t]


def _shingles(text: str, k: int = _SHINGLE_K) -> frozenset:
    """k-token shingles as a frozenset. Empty when fewer than k tokens — caller falls back to exact hash."""
    tokens = _tokenise(text)
    if len(tokens) < k:
        return frozenset()
    return frozenset(tuple(tokens[i:i + k]) for i in range(len(tokens) - k + 1))


def _jaccard_shingle_similarity(a: str, b: str, k: int = _SHINGLE_K) -> float:
    """Jaccard similarity over k-token shingles; falls back to normalised-hash equality when either text is short."""
    sa, sb = _shingles(a, k), _shingles(b, k)
    if not sa or not sb:
        return 1.0 if _content_hash(a) == _content_hash(b) else 0.0
    return len(sa & sb) / len(sa | sb)


def _find_near_dup_partner(i: int,
                            sh_i: frozenset,
                            shingled: list[tuple[MemoryEntry, frozenset]],
                            skip: set[int],
                            threshold: float
                            ) -> Optional[tuple[int, str, float]]:
    """First j > i such that entry j is a near-duplicate of entry i; None otherwise.

    Returns (drop_idx, kept_filename, sim). Caller applies the merge and records the action.
    Runs a single linear scan over candidates so callers stay free of nested loops.
    """
    ent_i = shingled[i][0]
    for j in range(i + 1, len(shingled)):
        if j in skip:
            continue
        ent_j, sh_j = shingled[j]
        if not sh_j:
            continue
        sim = len(sh_i & sh_j) / len(sh_i | sh_j)
        if sim < threshold:
            continue
        if len(ent_i.content) >= len(ent_j.content):
            return j, ent_i.filename, sim
        return i, ent_j.filename, sim
    return None


def _deduplicate_near(entries: list[MemoryEntry],
                      threshold: float = _NEAR_DUP_THRESHOLD
                      ) -> tuple[list[MemoryEntry], list[ConsolidationAction]]:
    """Merge near-duplicate entries via Jaccard shingle similarity. Runs post-exact-hash dedup.

    Greedy pairwise — finds the first j > i near-duplicate of each i and discards the shorter
    of the two. Full cluster semantics (all-to-all merging) is YAGNI until a real fixture shows
    it matters.
    """
    actions: list[ConsolidationAction] = []
    if len(entries) < 2:
        return entries, actions
    shingled = [(e, _shingles(e.content)) for e in entries]
    discarded_idx: set[int] = set()
    for i, (_ent_i, sh_i) in enumerate(shingled):
        if i in discarded_idx or not sh_i:
            continue
        match = _find_near_dup_partner(i, sh_i, shingled, discarded_idx, threshold)
        if match is None:
            continue
        drop_idx, kept_fn, sim = match
        dropped = shingled[drop_idx][0]
        discarded_idx.add(drop_idx)
        actions.append(ConsolidationAction(
            "merge", dropped.filename,
            f"Near-duplicate of {kept_fn} (jaccard={sim:.2f})",
        ))
    kept = [e for idx, (e, _) in enumerate(shingled) if idx not in discarded_idx]
    return kept, actions


_FILE_REF_RE = re.compile(r"(?:file|path):\s*(\S+\.(?:py|ts|js|md|yaml|json))")


def _prune_stale_file_refs(entries: list[MemoryEntry]) -> tuple[list[MemoryEntry], list[ConsolidationAction]]:
    """Remove entries whose file references ALL point to non-existent paths."""
    def _check(entry: MemoryEntry) -> ConsolidationAction | None:
        refs = _FILE_REF_RE.findall(entry.content)
        if not refs:
            return None
        stale = [r for r in refs if not os.path.exists(_resolve_path(r))]
        if len(stale) == len(refs):
            return ConsolidationAction("prune", entry.filename, f"All file refs stale: {', '.join(stale[:3])}")
        return None

    return _partition(entries, _check)


# Pre-compiled combined pattern for relative dates — single-pass replacement
_RELATIVE_DATE_RE = re.compile(r"\b(today|yesterday|tomorrow)\b", re.IGNORECASE)


def _fix_relative_dates(entries: list[MemoryEntry]) -> tuple[list[MemoryEntry], list[ConsolidationAction]]:
    """Convert relative dates to absolute via single compiled regex."""
    actions: list[ConsolidationAction] = []
    today = time.strftime("%Y-%m-%d")
    lookup = {
        "today": today,
        "yesterday": _offset_date(today, -1),
        "tomorrow": _offset_date(today, 1),
    }

    def _replace(match: re.Match) -> str:
        return lookup[match.group(1).lower()]

    for entry in entries:
        updated = _RELATIVE_DATE_RE.sub(_replace, entry.content)
        if updated != entry.content:
            entry.content = updated
            actions.append(ConsolidationAction("update", entry.filename, "Converted relative dates to absolute"))

    return entries, actions


def _prune_empty(entries: list[MemoryEntry]) -> tuple[list[MemoryEntry], list[ConsolidationAction]]:
    """Remove entries with empty or trivial content."""
    def _check(entry: MemoryEntry) -> ConsolidationAction | None:
        length = len(entry.content.strip())
        if length < 10:
            return ConsolidationAction("prune", entry.filename, f"Content too short ({length} chars)")
        return None

    return _partition(entries, _check)


def _compact_verbose(entries: list[MemoryEntry], max_chars: int = 2000) -> tuple[list[MemoryEntry], list[ConsolidationAction]]:
    """Trim entries that exceed max_chars to their essential content."""
    actions: list[ConsolidationAction] = []

    for entry in entries:
        if len(entry.content) > max_chars:
            truncated = entry.content[:max_chars]
            last_period = truncated.rfind(".")
            if last_period > max_chars // 2:
                truncated = truncated[:last_period + 1]
            original_len = len(entry.content)
            entry.content = truncated
            actions.append(ConsolidationAction("update", entry.filename, f"Compacted from {original_len} to {len(truncated)} chars"))

    return entries, actions


def _content_hash(text: str) -> str:
    """Hash content for deduplication. Normalises whitespace first."""
    normalised = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]


def _resolve_path(ref: str) -> str:
    """Resolve a file reference to an absolute path."""
    if os.path.isabs(ref):
        return ref
    return os.path.join(_HAL_HOME, ref)


def _offset_date(date_str: str, days: int) -> str:
    """Offset a YYYY-MM-DD date by N days."""
    import datetime
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    dt += datetime.timedelta(days=days)
    return dt.strftime("%Y-%m-%d")
