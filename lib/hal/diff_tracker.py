"""HAL Diff Tracker — track file changes over time.

Records file state snapshots and computes diffs between them. Useful
for tracking what an agent session changed, auditing modifications,
and generating change reports.

Usage:
    from lib.hal.diff_tracker import DiffTracker

    tracker = DiffTracker()
    tracker.snapshot("before")
    # ... agent makes changes ...
    tracker.snapshot("after")
    changes = tracker.diff("before", "after")
"""
import hashlib
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FileChange:
    """A single file change between snapshots."""
    path: str
    change_type: str  # "added", "modified", "deleted"
    size_before: int = 0
    size_after: int = 0


@dataclass
class DiffReport:
    """Changes between two snapshots."""
    from_label: str
    to_label: str
    changes: list[FileChange] = field(default_factory=list)

    @property
    def added(self) -> list[FileChange]:
        return [c for c in self.changes if c.change_type == "added"]

    @property
    def modified(self) -> list[FileChange]:
        return [c for c in self.changes if c.change_type == "modified"]

    @property
    def deleted(self) -> list[FileChange]:
        return [c for c in self.changes if c.change_type == "deleted"]

    @property
    def summary(self) -> str:
        return (
            f"Diff {self.from_label} -> {self.to_label}: "
            f"{len(self.added)} added, {len(self.modified)} modified, "
            f"{len(self.deleted)} deleted"
        )


class DiffTracker:
    """Tracks file state snapshots and computes diffs."""

    def __init__(self, root: str = ""):
        self._root = root or os.getcwd()
        self._snapshots: dict[str, dict[str, tuple[str, int]]] = {}

    def snapshot(self, label: str, paths: list[str] | None = None) -> int:
        """Take a snapshot of file states. Returns file count."""
        if paths is None:
            paths = _discover_files(self._root)

        state: dict[str, tuple[str, int]] = {}
        for path in paths:
            full = os.path.join(self._root, path) if not os.path.isabs(path) else path
            if os.path.isfile(full):
                state[path] = (_file_hash(full), os.path.getsize(full))

        self._snapshots[label] = state
        return len(state)

    def diff(self, from_label: str, to_label: str) -> DiffReport:
        """Compute diff between two snapshots."""
        before = self._snapshots.get(from_label, {})
        after = self._snapshots.get(to_label, {})

        changes: list[FileChange] = []
        all_paths = set(before) | set(after)

        for path in sorted(all_paths):
            in_before = path in before
            in_after = path in after

            if in_before and not in_after:
                changes.append(FileChange(path, "deleted", size_before=before[path][1]))
            elif not in_before and in_after:
                changes.append(FileChange(path, "added", size_after=after[path][1]))
            elif before[path][0] != after[path][0]:
                changes.append(FileChange(path, "modified", before[path][1], after[path][1]))

        return DiffReport(from_label, to_label, changes)

    def list_snapshots(self) -> list[str]:
        """List all snapshot labels."""
        return sorted(self._snapshots.keys())


_SKIP_DIRS = frozenset({"node_modules", "venv", "__pycache__", "dist", "build", ".git", ".venv", "state"})


_TRACK_EXTENSIONS = frozenset({".py", ".md", ".json", ".yaml", ".toml"})


def _is_trackable(filename: str) -> bool:
    """Check if a file should be tracked. O(1) set lookup."""
    return os.path.splitext(filename)[1] in _TRACK_EXTENSIONS and not filename.startswith(".")


def _discover_files(root: str) -> list[str]:
    """Discover trackable files. Uses glob-style walk to avoid nested loops."""
    import glob
    files: list[str] = []
    for ext in _TRACK_EXTENSIONS:
        pattern = os.path.join(root, "**", f"*{ext}")
        matches = glob.glob(pattern, recursive=True)
        files.extend(os.path.relpath(m, root) for m in matches if _should_include(m, root))
    return files


def _should_include(path: str, root: str) -> bool:
    """Check if path should be included (not in skip dirs, not dotfile)."""
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    return not any(p in _SKIP_DIRS or p.startswith(".") for p in parts)


def _file_hash(path: str) -> str:
    """SHA256 hash of file contents. Reads entire file (typical source < 1MB)."""
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        return ""
