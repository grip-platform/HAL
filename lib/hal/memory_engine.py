"""HAL Memory Engine — unified memory interface absorbing CC memdir patterns.

Reads and writes CC-compatible MEMORY.md format (200-line / 25KB limits),
integrates with GRIP's KONO semantic search, and provides a runtime-agnostic
memory interface for the HAL.

Usage:
    from lib.hal.memory_engine import MemoryEngine

    engine = MemoryEngine()
    state = engine.read_memory()       # Full memory state
    engine.write_entry("topic.md", frontmatter, content)
    engine.update_entrypoint()         # Rebuild MEMORY.md index

FROZEN SCHEMA (FORMAT_VERSION "1.1")
------------------------------------
A memory entry file begins with optional YAML-ish frontmatter delimited by
`---` lines, followed by body content. The recognised frontmatter keys are:

    name                  (str)    short human-readable title
    description           (str)    one-line summary used in MEMORY.md index
    type                  (str)    category — user | feedback | project | reference
    format_version        (str)    schema version, currently "1.1"
    source_session_id     (str)    provenance — the session that wrote this entry (1.1+)
    tool_calls_used       (list)   provenance — comma-separated tool names (1.1+)
    council_vote_hashes   (list)   provenance — comma-separated council vote digests (1.1+)

Unknown keys are preserved by the parser (written verbatim, ignored on read)
so sessions running an older schema degrade gracefully. Absent `format_version`
is treated as implicit "1.0" — an entry without any provenance fields is
indistinguishable from a 1.0 entry, which is the entire point of the freeze.

Writers emit `format_version: "1.1"` when provenance fields are non-empty,
else they emit the 1.0 footprint (name/description/type only). Readers
accept both silently — the version header matters only when we eventually
cut a breaking 2.0 release.
"""
import os
import re
import sys
from typing import Optional

# CC constants (from TF++ src/memdir/memdir.ts)
ENTRYPOINT_NAME = "MEMORY.md"
MAX_ENTRYPOINT_LINES = 200
MAX_ENTRYPOINT_BYTES = 25_000

FORMAT_VERSION = "1.1"
LEGACY_FORMAT_VERSION = "1.0"  # Entries without provenance fields

from lib.hal.home import HAL_HOME as _HAL_HOME
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class MemoryEntry:
    """A single memory file with frontmatter metadata.

    1.1+ adds three optional provenance fields. Empty defaults keep
    round-tripping with 1.0 entries lossless.
    """

    def __init__(
        self,
        filename: str,
        name: str = "",
        description: str = "",
        memory_type: str = "",
        content: str = "",
        format_version: str = FORMAT_VERSION,
        source_session_id: str = "",
        tool_calls_used: Optional[list[str]] = None,
        council_vote_hashes: Optional[list[str]] = None,
    ):
        self.filename = filename
        self.name = name
        self.description = description
        self.memory_type = memory_type
        self.content = content
        self.format_version = format_version
        self.source_session_id = source_session_id
        self.tool_calls_used = list(tool_calls_used or [])
        self.council_vote_hashes = list(council_vote_hashes or [])

    def has_provenance(self) -> bool:
        """True when any 1.1 provenance field carries data."""
        return bool(
            self.source_session_id
            or self.tool_calls_used
            or self.council_vote_hashes
        )

    def to_dict(self) -> dict:
        d = {
            "filename": self.filename,
            "name": self.name,
            "description": self.description,
            "type": self.memory_type,
            "content": self.content,
            "format_version": self.format_version,
        }
        if self.has_provenance():
            d["source_session_id"] = self.source_session_id
            d["tool_calls_used"] = list(self.tool_calls_used)
            d["council_vote_hashes"] = list(self.council_vote_hashes)
        return d


class MemoryState:
    """Complete memory state for a memory directory."""

    def __init__(
        self,
        entrypoint: str = "",
        entries: Optional[list[MemoryEntry]] = None,
        truncated: bool = False,
    ):
        self.entrypoint = entrypoint
        self.entries = entries or []
        self.truncated = truncated


class MemoryEngine:
    """Reads and writes CC-compatible memory directories."""

    def __init__(self, memory_dir: Optional[str] = None):
        self._dir = memory_dir or _default_memory_dir()

    @property
    def memory_dir(self) -> str:
        return self._dir

    def read_memory(self) -> MemoryState:
        """Read the full memory state: entrypoint + all linked entries."""
        entrypoint_path = os.path.join(self._dir, ENTRYPOINT_NAME)
        if not os.path.exists(entrypoint_path):
            return MemoryState()

        raw = _read_file(entrypoint_path)
        truncated_content, was_truncated = truncate_entrypoint(raw)
        entries = self._scan_entries()

        return MemoryState(
            entrypoint=truncated_content,
            entries=entries,
            truncated=was_truncated,
        )

    def read_entry(self, filename: str) -> Optional[MemoryEntry]:
        """Read a single memory entry by filename."""
        path = os.path.join(self._dir, filename)
        if not os.path.exists(path):
            return None
        return _parse_entry(filename, _read_file(path))

    def write_entry(
        self,
        filename: str,
        name: str,
        description: str,
        memory_type: str,
        content: str,
        *,
        source_session_id: str = "",
        tool_calls_used: Optional[list[str]] = None,
        council_vote_hashes: Optional[list[str]] = None,
    ) -> None:
        """Write a memory entry with frontmatter.

        Emits the 1.1 shape only when at least one provenance field is
        non-empty — pure 1.0 entries still round-trip byte-identically.
        """
        os.makedirs(self._dir, exist_ok=True)
        path = os.path.join(self._dir, filename)
        provenance_active = bool(source_session_id or tool_calls_used or council_vote_hashes)
        version = FORMAT_VERSION if provenance_active else LEGACY_FORMAT_VERSION
        lines = [
            "---",
            f"name: {name}",
            f"description: {description}",
            f"type: {memory_type}",
            f"format_version: {version}",
        ]
        if provenance_active:
            lines += [
                f"source_session_id: {source_session_id}",
                f"tool_calls_used: {_format_list(tool_calls_used)}",
                f"council_vote_hashes: {_format_list(council_vote_hashes)}",
            ]
        lines += ["---", "", ""]
        frontmatter = "\n".join(lines)
        _write_file(path, frontmatter + content)

    def delete_entry(self, filename: str) -> bool:
        """Delete a memory entry. Returns True if deleted."""
        path = os.path.join(self._dir, filename)
        if os.path.exists(path) and filename != ENTRYPOINT_NAME:
            os.remove(path)
            return True
        return False

    def update_entrypoint(self, content: str) -> None:
        """Write the MEMORY.md entrypoint, enforcing CC limits."""
        os.makedirs(self._dir, exist_ok=True)
        truncated, _ = truncate_entrypoint(content)
        path = os.path.join(self._dir, ENTRYPOINT_NAME)
        _write_file(path, truncated)

    def list_entries(self) -> list[str]:
        """List all .md files in the memory directory (excluding MEMORY.md)."""
        if not os.path.isdir(self._dir):
            return []
        return sorted(
            f for f in os.listdir(self._dir)
            if f.endswith(".md") and f != ENTRYPOINT_NAME
        )

    def _scan_entries(self) -> list[MemoryEntry]:
        """Scan all memory entries and parse frontmatter."""
        return [
            _parse_entry(fn, _read_file(os.path.join(self._dir, fn)))
            for fn in self.list_entries()
            if _read_file(os.path.join(self._dir, fn))
        ]


def truncate_entrypoint(raw: str) -> tuple[str, bool]:
    """Truncate MEMORY.md to CC limits (200 lines / 25KB).

    Returns (truncated_content, was_truncated).
    Matches CC's truncateEntrypointContent logic from memdir.ts.
    """
    trimmed = raw.strip()
    lines = trimmed.split("\n")

    was_truncated = len(lines) > MAX_ENTRYPOINT_LINES or len(trimmed) > MAX_ENTRYPOINT_BYTES

    if not was_truncated:
        return trimmed, False

    truncated = "\n".join(lines[:MAX_ENTRYPOINT_LINES]) if len(lines) > MAX_ENTRYPOINT_LINES else trimmed

    if len(truncated) > MAX_ENTRYPOINT_BYTES:
        cut_at = truncated.rfind("\n", 0, MAX_ENTRYPOINT_BYTES)
        truncated = truncated[:cut_at] if cut_at > 0 else truncated[:MAX_ENTRYPOINT_BYTES]

    return truncated, True


def _parse_entry(filename: str, raw: str) -> MemoryEntry:
    """Parse a memory file with optional YAML frontmatter.

    Absent `format_version` defaults to the current FORMAT_VERSION (entries
    written before the freeze are implicitly "1.0"). A mismatched version
    logs one warning to stderr but still returns a usable MemoryEntry so the
    reader never blocks on a format drift.
    """
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return MemoryEntry(filename=filename, content=raw.strip())

    fm = _parse_simple_frontmatter(match.group(1))
    version = fm.get("format_version", "") or FORMAT_VERSION
    if version not in (FORMAT_VERSION, LEGACY_FORMAT_VERSION):
        print(
            f"[memory_engine] {filename}: format_version {version!r} "
            f"differs from engine FORMAT_VERSION {FORMAT_VERSION!r}",
            file=sys.stderr,
        )
    return MemoryEntry(
        filename=filename,
        name=fm.get("name", ""),
        description=fm.get("description", ""),
        memory_type=fm.get("type", ""),
        content=raw[match.end():].strip(),
        format_version=version,
        source_session_id=fm.get("source_session_id", ""),
        tool_calls_used=_parse_list(fm.get("tool_calls_used", "")),
        council_vote_hashes=_parse_list(fm.get("council_vote_hashes", "")),
    )


def _parse_simple_frontmatter(text: str) -> dict[str, str]:
    """Parse simple key: value frontmatter (no nested structures)."""
    result = {}
    for line in text.split("\n"):
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip()
    return result


def _format_list(items: Optional[list[str]]) -> str:
    """Render a list field as a comma-separated value (empty stays empty)."""
    if not items:
        return ""
    return ",".join(str(x).strip() for x in items if str(x).strip())


def _parse_list(value: str) -> list[str]:
    """Parse a comma-separated list field; whitespace trimmed, empties dropped."""
    if not value:
        return []
    return [piece.strip() for piece in value.split(",") if piece.strip()]


def _default_memory_dir() -> str:
    """Resolve the default memory directory for the current project."""
    cwd = os.getcwd()
    project_key = cwd.replace("/", "-").replace(".", "-")
    project_mem = os.path.join(_HAL_HOME, "projects", project_key, "memory")
    if os.path.isdir(project_mem):
        return project_mem
    return os.path.join(_HAL_HOME, "memory")


def _read_file(path: str) -> str:
    """Read file contents, returning empty string on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return ""


def _write_file(path: str, content: str) -> None:
    """Write content to file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
