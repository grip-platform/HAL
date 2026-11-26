"""locked_json -- a JSON state file that changes whole or not at all, one writer at a time.

Two properties, kept apart on purpose:

``atomic``      A reader sees the old file or the new one, never half of one. Every
                write goes to a temporary file beside the target, is flushed to
                disk, and is then renamed over the target with ``os.replace``.
``serialised``  A read-modify-write holds an exclusive advisory lock from before its
                read until after its rename, so two writers -- threads or processes --
                never interleave and lose each other's change.

The lock is taken on a separate file that is never replaced (``<name>.lock``), not on
the data file. ``os.replace`` gives the data file a new inode at every write, so a
lock held on the old inode would not exclude a writer that opened the new one.
(``cross_session_state.py`` keeps its inode by truncating in place instead, which
gives up atomicity: a writer killed mid-write leaves a torn file.) Here both hold.

Readers take no lock: the rename is atomic, so a plain read is always a whole file.

The wait for the lock is bounded. A writer that cannot get it in time raises
``LockTimeout`` and changes nothing. A caller on a request path catches it and goes
on, because a state file must never stall the traffic it describes.

The lock protocol works across processes on both POSIX and Windows.
"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Union

from lib.hal import file_lock

PathLike = Union[str, Path]

DEFAULT_LOCK_TIMEOUT_S = 2.0
_POLL_S = 0.005
_THREAD_LOCKS: dict = {}
_THREAD_LOCKS_GUARD = threading.Lock()


class LockTimeout(OSError):
    """The lock was not free in time; nothing was read or written."""


def read_json(path: PathLike) -> dict:
    """The file's JSON object; ``{}`` when it is missing, unreadable or not an object."""
    try:
        data = json.loads(Path(path).read_bytes())
    except (OSError, ValueError, RecursionError):
        return {}
    return data if isinstance(data, dict) else {}


def update_json(path: PathLike, mutate: Callable[[dict], dict], *,
                lock_timeout: float = DEFAULT_LOCK_TIMEOUT_S, mode: int = 0o600) -> dict:
    """Read the file, apply ``mutate``, and replace the file with the result, under the lock.

    ``mutate`` receives the current object (``{}`` when there is none or it cannot be
    read) and returns the new one. Returns what was written. Raises ``LockTimeout``
    when the lock is not free within ``lock_timeout`` seconds, and ``OSError`` when
    the file cannot be written; in both cases the file is unchanged.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _locked(target, lock_timeout):
        data = mutate(read_json(target))
        write_json(target, data, mode=mode)
        return data


def write_json(path: PathLike, data: dict, *, mode: int = 0o600) -> None:
    """Replace ``path`` with ``data`` atomically: temporary file, fsync, ``os.replace``.

    Callers that read before they write hold the lock (``update_json``); a bare call
    is atomic but not serialised against other writers.
    """
    target = Path(path)
    payload = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")
    tmp = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        _write_all(fd, payload)
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        _remove_quietly(tmp)
        raise
    os.close(fd)
    try:
        _replace(tmp, target)
    except BaseException:
        _remove_quietly(tmp)
        raise


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:  # never spin on a descriptor that accepts nothing
            raise OSError("the file accepted no bytes")
        view = view[written:]


def _remove_quietly(path: Path) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _replace(tmp: Path, target: Path) -> None:
    """Replace atomically, retrying transient Windows sharing violations briefly."""
    deadline = time.monotonic() + 1.0
    while True:
        try:
            os.replace(tmp, target)
            return
        except PermissionError:
            if os.name != "nt" or time.monotonic() >= deadline:
                raise
            time.sleep(_POLL_S)


def _thread_lock(target: Path) -> threading.Lock:
    """One lock per file within this process, paired with the process lock."""
    key = str(target)
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.Lock())


@contextmanager
def _locked(target: Path, timeout: float) -> Iterator[None]:
    """Hold this process's lock for ``target`` and its portable ``.lock`` file."""
    deadline = time.monotonic() + max(0.0, timeout)
    local = _thread_lock(target)
    if not local.acquire(timeout=max(0.0, timeout)):
        raise LockTimeout(f"{target.name} is locked by another thread")
    try:
        with _file_lock(target.with_name(target.name + ".lock"), deadline):
            yield
    finally:
        local.release()


@contextmanager
def _file_lock(lock_path: Path, deadline: float) -> Iterator[None]:
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    held = False
    try:
        if not file_lock.lock_within(fd, max(0.0, deadline - time.monotonic())):
            raise LockTimeout(f"{lock_path.name} is held by another process")
        held = True
        yield
    finally:
        if held:
            file_lock.unlock(fd)
        os.close(fd)


__all__ = ["LockTimeout", "read_json", "update_json", "write_json"]
