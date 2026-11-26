"""Obsidian Vault Indexer — bridge external vaults into HAL's knowledge graph.

Indexes an Obsidian vault directory into KONO by walking .md files, extracting
frontmatter, body text, wikilinks, and tags. Supports incremental re-indexing
via mtime comparison so only changed files are processed.

Usage:
    from lib.hal.obsidian_vault import VaultIndexer

    indexer = VaultIndexer("/path/to/vault")
    report = indexer.index()
    print(f"Indexed {report.indexed} notes, {report.skipped} unchanged")

    results = indexer.search("machine learning")
    note = indexer.read_note("Projects/HAL.md")
    links = indexer.backlinks("Projects/HAL.md")
"""
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lib.hal.home import hal_home


def _index_state_path() -> Path:
    return hal_home() / "state" / "hal" / "vault-index.json"


# Patterns
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
_TAG_RE = re.compile(r"(?:^|\s)#([a-zA-Z][\w/-]{1,50})")
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass(frozen=True)
class VaultNote:
    """A single indexed note from the vault."""
    path: str           # relative to vault root
    title: str
    body: str           # first 2000 chars of body (after frontmatter)
    tags: tuple[str, ...]
    wikilinks: tuple[str, ...]
    frontmatter: dict
    mtime: float


@dataclass
class IndexReport:
    """Result of a vault indexing run."""
    vault_path: str
    indexed: int = 0
    skipped: int = 0
    errors: int = 0
    duration_s: float = 0.0
    notes: list[VaultNote] = field(default_factory=list)


class VaultIndexer:
    """Indexes an Obsidian vault for HAL's knowledge graph."""

    def __init__(self, vault_path: str):
        self._vault = Path(vault_path)
        self._mtime_cache: dict[str, float] = _load_mtime_cache()
        self._notes: dict[str, VaultNote] = {}  # rel_path -> note (persists across index calls)

    def index(self, force: bool = False) -> IndexReport:
        """Walk the vault and index changed .md files.

        Args:
            force: If True, re-index all files regardless of mtime.
        """
        start = time.time()
        report = IndexReport(vault_path=str(self._vault))

        if not self._vault.is_dir():
            return report

        md_files = sorted(self._vault.rglob("*.md"))
        for md_path in md_files:
            rel = md_path.relative_to(self._vault).as_posix()
            if _is_hidden(rel):
                continue
            mtime = md_path.stat().st_mtime
            if not force and self._mtime_cache.get(rel, 0) >= mtime:
                report.skipped += 1
                continue
            note = _parse_note(md_path, rel, mtime)
            if note:
                self._notes[rel] = note
                report.indexed += 1
                self._mtime_cache[rel] = mtime
            else:
                report.errors += 1

        _save_mtime_cache(self._mtime_cache)
        report.notes = list(self._notes.values())
        report.duration_s = time.time() - start
        return report

    def search(self, query: str, limit: int = 10) -> list[VaultNote]:
        """Keyword search across indexed notes. Returns matching notes sorted by relevance."""
        query_words = set(query.lower().split())
        if not query_words:
            return []

        scored: list[tuple[float, VaultNote]] = []
        report = self.index()  # ensure index is current
        for note in report.notes:
            score = _score_note(note, query_words)
            if score > 0:
                scored.append((score, note))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [note for _, note in scored[:limit]]

    def read_note(self, rel_path: str) -> Optional[str]:
        """Read a note's full content by relative path."""
        full = self._vault / rel_path
        if not full.is_file():
            return None
        return full.read_text(errors="replace", encoding="utf-8")

    def backlinks(self, rel_path: str) -> list[str]:
        """Find notes that link TO the given note via [[wikilinks]]."""
        target = Path(rel_path).stem.lower()
        results: list[str] = []
        report = self.index()
        for note in report.notes:
            link_targets = {Path(lnk).stem.lower() for lnk in note.wikilinks}
            if target in link_targets:
                results.append(note.path)
        return results

    def graph(self, rel_path: str) -> dict:
        """Return forward links + backlinks for a note as a graph fragment."""
        report = self.index()
        note_map = {n.path: n for n in report.notes}
        target = note_map.get(rel_path)
        forward = list(target.wikilinks) if target else []
        back = self.backlinks(rel_path)
        return {"note": rel_path, "forward_links": forward, "backlinks": back}


# --- Parsing helpers ---


def _is_hidden(rel_path: str) -> bool:
    """Skip hidden directories (.obsidian, .git, .trash)."""
    return any(part.startswith(".") for part in Path(rel_path).parts)


def _parse_note(path: Path, rel: str, mtime: float) -> Optional[VaultNote]:
    """Parse a single markdown note."""
    try:
        raw = path.read_text(errors="replace", encoding="utf-8")
    except OSError:
        return None

    frontmatter, body = _split_frontmatter(raw)
    title = _extract_title(rel, body)
    tags = tuple(_TAG_RE.findall(body))
    wikilinks = tuple(_WIKILINK_RE.findall(raw))

    return VaultNote(
        path=rel,
        title=title,
        body=body[:2000],
        tags=tags,
        wikilinks=wikilinks,
        frontmatter=frontmatter,
        mtime=mtime,
    )


def _split_frontmatter(raw: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and return (frontmatter_dict, body)."""
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw

    fm_text = match.group(1)
    body = raw[match.end():]

    # Simple key: value parser (avoids PyYAML dependency)
    fm: dict = {}
    for line in fm_text.split("\n"):
        if ":" in line:
            key, _, val = line.partition(":")
            fm[key.strip()] = val.strip()
    return fm, body


def _extract_title(rel_path: str, body: str) -> str:
    """Extract title from H1 heading or filename."""
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("##"):
            return stripped[2:].strip()
    return Path(rel_path).stem


def _score_note(note: VaultNote, query_words: set[str]) -> float:
    """Score a note against query words. Title matches weighted 3x."""
    title_words = set(note.title.lower().split())
    body_words = set(note.body.lower().split())
    tag_words = {t.lower() for t in note.tags}

    title_hits = len(query_words & title_words) * 3.0
    body_hits = len(query_words & body_words)
    tag_hits = len(query_words & tag_words) * 2.0

    return title_hits + body_hits + tag_hits


# --- Mtime cache persistence ---


def _load_mtime_cache() -> dict[str, float]:
    """Load mtime cache from disk."""
    path = _index_state_path()
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_mtime_cache(cache: dict[str, float]) -> None:
    """Persist mtime cache to disk."""
    path = _index_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
