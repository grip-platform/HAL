"""HAL Memory Sync — optional bidirectional sync with an external harness.

Syncs MEMORY.md index and individual memory files between HAL's memory
directory (~/.hal/memory/) and an external harness memory directory. The
external directory is opt-in via the HAL_PEER_MEMORY_DIR environment variable
(or by passing peer_dir= explicitly to MemorySync). When no peer is set, sync
is a no-op — HAL is harness-agnostic by default.

Conflict resolution: newer timestamp wins.

Usage:
    # Enable sync with an external harness
    export HAL_PEER_MEMORY_DIR=~/.claude/projects/-Users-me--hal/memory

    from lib.hal.memory_sync import MemorySync

    sync = MemorySync()
    report = sync.sync()
    print(f"Synced: {report.copied} files, {report.skipped} unchanged")
"""
import os
import shutil
from dataclasses import dataclass

from lib.hal.home import HAL_HOME as _HAL_HOME

_HAL_MEMORY = os.path.join(_HAL_HOME, "memory")
_PEER_MEMORY = os.environ.get("HAL_PEER_MEMORY_DIR", "")


@dataclass
class SyncReport:
    """Result of a memory sync operation."""
    copied: int = 0
    skipped: int = 0
    conflicts: int = 0
    errors: list = None

    def __post_init__(self):
        self.errors = self.errors or []

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict:
        return {
            "copied": self.copied,
            "skipped": self.skipped,
            "conflicts": self.conflicts,
            "errors": self.errors,
            "success": self.success,
        }


class MemorySync:
    """Bidirectional memory sync between HAL and an optional peer harness.

    The peer directory is resolved in this order:
      1. peer_dir= constructor argument (explicit override)
      2. HAL_PEER_MEMORY_DIR environment variable
      3. Legacy grip_dir= constructor argument (deprecated alias)

    When no peer directory is configured, sync() is a no-op.
    """

    def __init__(self, peer_dir: str = "", hal_dir: str = "", *, grip_dir: str = ""):
        self._peer = peer_dir or grip_dir or _PEER_MEMORY
        self._hal = hal_dir or _HAL_MEMORY

    @property
    def enabled(self) -> bool:
        return bool(self._peer)

    def sync(self) -> SyncReport:
        """Run bidirectional sync. Newer file wins conflicts."""
        report = SyncReport()
        if not self._peer:
            return report
        os.makedirs(self._peer, exist_ok=True)
        os.makedirs(self._hal, exist_ok=True)

        peer_files = self._list_md_files(self._peer)
        hal_files = self._list_md_files(self._hal)
        all_names = peer_files.keys() | hal_files.keys()

        for name in all_names:
            peer_path = os.path.join(self._peer, name)
            hal_path = os.path.join(self._hal, name)
            peer_exists = name in peer_files
            hal_exists = name in hal_files

            try:
                if peer_exists and not hal_exists:
                    shutil.copy2(peer_path, hal_path)
                    report.copied += 1
                elif hal_exists and not peer_exists:
                    shutil.copy2(hal_path, peer_path)
                    report.copied += 1
                elif peer_exists and hal_exists:
                    peer_mtime = peer_files[name]
                    hal_mtime = hal_files[name]
                    if abs(peer_mtime - hal_mtime) < 1.0:
                        report.skipped += 1
                    elif peer_mtime > hal_mtime:
                        shutil.copy2(peer_path, hal_path)
                        report.copied += 1
                        report.conflicts += 1
                    else:
                        shutil.copy2(hal_path, peer_path)
                        report.copied += 1
                        report.conflicts += 1
            except OSError as exc:
                report.errors.append(f"{name}: {exc}")

        return report

    def status(self) -> dict:
        """Report sync status without modifying files."""
        if not self._peer:
            return {"peer_only": [], "hal_only": [], "shared": 0, "drifted": [], "in_sync": 0}
        peer_files = self._list_md_files(self._peer)
        hal_files = self._list_md_files(self._hal)
        peer_only = peer_files.keys() - hal_files.keys()
        hal_only = hal_files.keys() - peer_files.keys()
        shared = peer_files.keys() & hal_files.keys()
        drifted = [
            n for n in shared
            if abs(peer_files[n] - hal_files[n]) >= 1.0
        ]
        return {
            "peer_only": sorted(peer_only),
            "hal_only": sorted(hal_only),
            "shared": len(shared),
            "drifted": sorted(drifted),
            "in_sync": len(shared) - len(drifted),
        }

    @staticmethod
    def _list_md_files(directory: str) -> dict:
        """List .md files with their modification times."""
        result = {}
        if not os.path.isdir(directory):
            return result
        for f in os.listdir(directory):
            if f.endswith(".md"):
                path = os.path.join(directory, f)
                result[f] = os.path.getmtime(path)
        return result
