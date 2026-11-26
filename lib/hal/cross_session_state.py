"""CrossSessionStateFile — fcntl-guarded shared JSON state.

W3 of the RAILLM ultrado plan (Gen 481). Enables safe concurrent dispatch
from multiple Claude Code sessions sharing a single RAILLM state file
(``state/hal/limits.json``). Wraps a path with shared/exclusive lock
semantics so read-modify-write sequences stay atomic across processes.

Design invariant — truncate-in-place, not atomic-rename
-------------------------------------------------------

The write path uses ``f.truncate()`` + ``f.seek(0)`` + ``json.dump`` while
holding ``fcntl.LOCK_EX`` on the same file descriptor used throughout the
update. We deliberately do NOT use the classic ``tempfile + os.replace``
atomic-rename pattern here.

Reason: ``os.replace`` swaps the underlying inode. A second writer that
opens the path after the rename acquires a lock on a DIFFERENT inode, and
its lock therefore does not serialise against the previous writer's lock.
This re-introduces the race we are trying to prevent. Truncate-in-place
keeps the inode stable for the lifetime of the file, so every writer
locks the same kernel inode and the locks genuinely serialise.

Trade-off: a process killed mid-write can leave a truncated JSON blob on
disk. We handle that in ``read``/``update`` by treating a JSONDecodeError
as "empty state, start over" rather than crashing the dispatcher. For
RAILLM's use case (limit state regenerated on every response header)
this is the correct trade — partial loss of limit bookkeeping is
strictly better than a crashed failover chain.

Platform note — fcntl is POSIX-only. On Windows the wrapper degrades to
non-locking reads and writes, preserving functional parity while
sacrificing the multi-process safety guarantee.

Hypothesis H049 (plan label; engine ID H063):
``concurrent_write_safety_test_pass`` = 1. Concrete verification: spawn N
processes that each append a unique key to the state dict; after all
complete, the merged state must contain all N keys and zero duplicates.

Falsification: any race-condition test that produces a merged state with
fewer keys than writers, or a JSON-corrupted file that a reader cannot
recover from, kills the claim.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Union


# fcntl is POSIX-only. On Windows the locks become no-ops so callers still
# get correct JSON serialisation; they just lose the cross-process safety.
try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    fcntl = None  # type: ignore[assignment]
    _HAS_FCNTL = False


PathLike = Union[str, Path]
StateDict = Dict[str, Any]
Mutator = Callable[[StateDict], StateDict]


# ---------------------------------------------------------------------------
# Low-level lock context manager
# ---------------------------------------------------------------------------


@contextmanager
def _locked(fd: int, *, exclusive: bool) -> Iterator[None]:
    """Acquire a POSIX advisory lock on ``fd``; no-op on non-fcntl platforms.

    ``LOCK_SH`` for readers, ``LOCK_EX`` for writers. The caller owns the
    fd; this helper only wraps the lock/unlock pair so it is
    exception-safe.
    """
    if not _HAS_FCNTL:
        yield
        return
    mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    fcntl.flock(fd, mode)
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# CrossSessionStateFile
# ---------------------------------------------------------------------------


def _default_factory() -> StateDict:
    return {}


@dataclass
class CrossSessionStateFile:
    """JSON state file guarded by fcntl shared/exclusive locks.

    Usage::

        sf = CrossSessionStateFile("/tmp/shared.json")
        data = sf.read()                       # SHARED lock
        sf.write({"k": "v"})                    # EXCLUSIVE lock + truncate
        sf.update(lambda d: {**d, "k": "v2"})  # EXCLUSIVE lock + RMW

    The ``update`` method is the primary atomicity primitive: the lock is
    held across load, mutate, and persist so no other writer can interleave.
    """

    path: PathLike
    default_factory: Callable[[], StateDict] = field(default=_default_factory)

    def __post_init__(self) -> None:
        self._path = Path(self.path)

    # -- read -------------------------------------------------------------

    def read(self) -> StateDict:
        """Return a snapshot of the state. Empty dict if file is missing.

        Acquires a SHARED lock during the read. Multiple readers can
        proceed concurrently; a writer blocks until all readers release.
        """
        if not self._path.exists():
            return self.default_factory()
        with self._path.open("r") as f:
            with _locked(f.fileno(), exclusive=False):
                return self._decode_or_default(f.read())

    # -- write ------------------------------------------------------------

    def write(self, data: StateDict) -> None:
        """Overwrite the state under an exclusive lock.

        Uses in-place truncate+write so the fcntl lock stays valid on the
        SAME inode throughout the write. See module docstring for why
        atomic-rename is rejected here.
        """
        self._ensure_file()
        with self._path.open("r+") as f:
            with _locked(f.fileno(), exclusive=True):
                self._write_under_lock(f, data)

    # -- read-modify-write -----------------------------------------------

    def update(self, mutator: Mutator) -> StateDict:
        """Atomic read-modify-write under an exclusive lock.

        ``mutator`` receives the current dict and returns the new dict.
        The exclusive lock is held for the full round trip, so concurrent
        updates serialise rather than racing. Returns a defensive copy of
        the new dict so post-call mutations on the caller side do not
        bleed into the persisted state.
        """
        self._ensure_file()
        with self._path.open("r+") as f:
            with _locked(f.fileno(), exclusive=True):
                f.seek(0)
                current = self._decode_or_default(f.read())
                new = mutator(current)
                self._write_under_lock(f, new)
                return dict(new)

    # -- internals --------------------------------------------------------

    def _ensure_file(self) -> None:
        """Create parent directory + empty JSON object if missing."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text("{}")

    def _decode_or_default(self, raw: str) -> StateDict:
        """Decode JSON; return default factory on empty or corrupt input."""
        if not raw.strip():
            return self.default_factory()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Corrupt state: return factory so callers keep working. A
            # corrupt file is a bug to investigate, not a crash reason.
            return self.default_factory()

    def _write_under_lock(self, f: Any, data: StateDict) -> None:
        """Truncate+write on an already-locked fd, fsync for durability."""
        f.seek(0)
        f.truncate()
        json.dump(data, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------


def has_fcntl_support() -> bool:
    """Return True when fcntl locking is available (POSIX)."""
    return _HAS_FCNTL


__all__ = [
    "CrossSessionStateFile",
    "has_fcntl_support",
]
