"""file_lock -- one portable advisory lock on an open file.

On POSIX, locks are the same blocking ``fcntl.flock`` calls HAL used before:
exclusive and shared locks stay distinct, and unlock releases either kind.

Windows ``msvcrt.locking`` locks bytes rather than whole files and has no shared
mode. Both shared and exclusive requests therefore take one exclusive byte lock
on Windows. This narrows read/read concurrency there intentionally while keeping
the state safe.

The lock byte lives at ``LOCK_OFFSET``, far beyond the real data in HAL's small
JSONL and state files. A process that reads or writes their real content without
taking this lock therefore cannot collide with the lock byte. ``msvcrt`` starts
at the descriptor's current position, so every lock and unlock saves the current
position, seeks to that fixed byte, then restores the original position.

``msvcrt.LK_LOCK`` gives up after roughly ten seconds. HAL needs the blocking
behaviour of ``fcntl.flock``, which never gives up, so Windows instead retries
``LK_NBLCK`` until the byte becomes available.
"""
from __future__ import annotations

import errno
import os
import time
from contextlib import contextmanager
from typing import IO, Iterator, Union

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None  # type: ignore[assignment]

try:
    import msvcrt
except ImportError:  # POSIX
    msvcrt = None  # type: ignore[assignment]


FileDescriptor = Union[int, IO[object]]

LOCK_OFFSET = 1 << 30  # 1 GiB, outside the real content of HAL state files
_RETRY_SECONDS = 0.01


def _fileno(file_or_fd: FileDescriptor) -> int:
    return file_or_fd if isinstance(file_or_fd, int) else file_or_fd.fileno()


def lock(file_or_fd: FileDescriptor, *, shared: bool = False) -> None:
    """Block until ``file_or_fd`` has an advisory lock."""
    _lock(file_or_fd, None, shared=shared)


def lock_within(file_or_fd: FileDescriptor, seconds: float, *, shared: bool = False) -> bool:
    """Try to lock ``file_or_fd`` until ``seconds`` pass; report whether held."""
    return _lock(file_or_fd, seconds, shared=shared)


def _lock(file_or_fd: FileDescriptor, seconds: float | None, *, shared: bool) -> bool:
    fd = _fileno(file_or_fd)
    deadline = None if seconds is None else time.monotonic() + max(0.0, seconds)
    if fcntl is not None:
        operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        if deadline is None:
            # Preserve POSIX's native blocking flock semantics.
            fcntl.flock(fd, operation)
            return True
        operation |= fcntl.LOCK_NB
        while True:
            try:
                fcntl.flock(fd, operation)
                return True
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    return False
                time.sleep(_RETRY_SECONDS)

    position = os.lseek(fd, 0, os.SEEK_CUR)
    try:
        os.lseek(fd, LOCK_OFFSET, os.SEEK_SET)
        while True:
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return True
            except OSError as exc:
                deadlock_errno = getattr(errno, "EDEADLOCK", errno.EDEADLK)
                if exc.errno not in (errno.EACCES, deadlock_errno):
                    raise
                if deadline is not None and time.monotonic() >= deadline:
                    return False
                time.sleep(_RETRY_SECONDS)
    finally:
        os.lseek(fd, position, os.SEEK_SET)


def unlock(file_or_fd: FileDescriptor) -> None:
    """Release the advisory lock held on ``file_or_fd``."""
    fd = _fileno(file_or_fd)
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return

    position = os.lseek(fd, 0, os.SEEK_CUR)
    try:
        os.lseek(fd, LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    finally:
        os.lseek(fd, position, os.SEEK_SET)


@contextmanager
def file_lock(file_or_fd: FileDescriptor, *, shared: bool = False) -> Iterator[None]:
    """Hold a lock, exclusive by default, and always release it after use."""
    lock(file_or_fd, shared=shared)
    try:
        yield
    finally:
        unlock(file_or_fd)


__all__ = ["LOCK_OFFSET", "file_lock", "lock", "lock_within", "unlock"]
