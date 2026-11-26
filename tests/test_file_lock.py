"""Tests for the portable blocking file lock."""
from __future__ import annotations

import errno
import multiprocessing
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from lib.hal import file_lock


def _try_lock(path: str, result: multiprocessing.Queue) -> None:
    import fcntl

    with open(path, "a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            result.put(False)
        else:
            result.put(True)
            fcntl.flock(handle, fcntl.LOCK_UN)


def _hold_windows_lock(path: str, ready: multiprocessing.Event,
                       release: multiprocessing.Event) -> None:
    from lib.hal import file_lock as child_file_lock

    with open(path, "a+") as handle, child_file_lock.file_lock(handle):
        ready.set()
        release.wait(timeout=5)


def _hold_portable_lock(path: str, ready: multiprocessing.Event,
                        release: multiprocessing.Event) -> None:
    from lib.hal import file_lock as child_file_lock

    with open(path, "a+") as handle, child_file_lock.file_lock(handle):
        ready.set()
        release.wait(timeout=5)


def _bounded_lock_times_out(tmp_path: Path) -> None:
    path = tmp_path / "bounded-state.jsonl"
    ready = multiprocessing.Event()
    release = multiprocessing.Event()
    process = multiprocessing.Process(
        target=_hold_portable_lock, args=(str(path), ready, release)
    )
    process.start()
    try:
        assert ready.wait(timeout=5)
        with path.open("a+") as handle:
            assert file_lock.lock_within(handle, 0.05) is False
    finally:
        release.set()
        process.join(timeout=5)
    assert process.exitcode == 0


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX lock_within case")
def test_posix_bounded_lock_gives_up(tmp_path: Path):
    _bounded_lock_times_out(tmp_path)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows lock_within case")
def test_windows_bounded_lock_gives_up(tmp_path: Path):
    _bounded_lock_times_out(tmp_path)


@pytest.mark.skipif(sys.platform == "win32", reason="uses POSIX flock directly")
def test_posix_exclusive_lock_excludes_another_process(tmp_path: Path):
    path = tmp_path / "state.jsonl"
    result = multiprocessing.Queue()
    with path.open("a+") as handle:
        file_lock.lock(handle)
        blocked = multiprocessing.Process(target=_try_lock, args=(str(path), result))
        blocked.start()
        blocked.join(timeout=5)
        assert blocked.exitcode == 0
        assert result.get(timeout=1) is False
        file_lock.unlock(handle)

        available = multiprocessing.Process(target=_try_lock, args=(str(path), result))
        available.start()
        available.join(timeout=5)
        assert available.exitcode == 0
        assert result.get(timeout=1) is True


def test_windows_lock_retries_restores_position_and_unlocks_same_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[tuple[int, int, int]] = []
    attempts = 0

    def locking(fd: int, mode: int, count: int) -> None:
        nonlocal attempts
        calls.append((os.lseek(fd, 0, os.SEEK_CUR), mode, count))
        if mode == 1 and attempts < 2:
            attempts += 1
            raise OSError(errno.EACCES, "busy")

    fake_msvcrt = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=locking)
    monkeypatch.setattr(file_lock, "fcntl", None)
    monkeypatch.setattr(file_lock, "msvcrt", fake_msvcrt)
    monkeypatch.setattr(file_lock.time, "sleep", lambda _: None)

    path = tmp_path / "state.jsonl"
    path.write_text("real data", encoding="utf-8")
    with path.open("r+") as handle:
        handle.seek(4)
        with file_lock.file_lock(handle, shared=True):
            assert handle.tell() == 4
        assert handle.tell() == 4

    assert attempts == 2
    assert calls == [
        (file_lock.LOCK_OFFSET, 1, 1),
        (file_lock.LOCK_OFFSET, 1, 1),
        (file_lock.LOCK_OFFSET, 1, 1),
        (file_lock.LOCK_OFFSET, 2, 1),
    ]


def test_windows_lock_propagates_non_contention_error_and_restores_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    attempts = 0

    def locking(fd: int, mode: int, count: int) -> None:
        nonlocal attempts
        attempts += 1
        raise OSError(errno.EBADF, "bad file descriptor")

    fake_msvcrt = SimpleNamespace(LK_NBLCK=1, locking=locking)
    monkeypatch.setattr(file_lock, "fcntl", None)
    monkeypatch.setattr(file_lock, "msvcrt", fake_msvcrt)
    monkeypatch.setattr(
        file_lock.time,
        "sleep",
        lambda _: pytest.fail("non-contention error was retried"),
    )

    path = tmp_path / "state.jsonl"
    path.write_text("real data", encoding="utf-8")
    with path.open("r+") as handle:
        handle.seek(4)
        with pytest.raises(OSError) as exc_info:
            file_lock.lock(handle)
        assert exc_info.value.errno == errno.EBADF
        assert handle.tell() == 4

    assert attempts == 1


@pytest.mark.skipif(sys.platform != "win32", reason="requires real msvcrt locking")
def test_windows_exclusive_lock_excludes_another_process(tmp_path: Path):
    path = tmp_path / "state.jsonl"
    ready = multiprocessing.Event()
    release = multiprocessing.Event()

    process = multiprocessing.Process(
        target=_hold_windows_lock, args=(str(path), ready, release)
    )
    process.start()
    assert ready.wait(timeout=5)
    with path.open("a+") as handle:
        os.lseek(handle.fileno(), file_lock.LOCK_OFFSET, os.SEEK_SET)
        with pytest.raises(OSError):
            file_lock.msvcrt.locking(handle.fileno(), file_lock.msvcrt.LK_NBLCK, 1)
    release.set()
    process.join(timeout=5)
    assert process.exitcode == 0
    with path.open("a+") as handle:
        os.lseek(handle.fileno(), file_lock.LOCK_OFFSET, os.SEEK_SET)
        file_lock.msvcrt.locking(handle.fileno(), file_lock.msvcrt.LK_NBLCK, 1)
        file_lock.msvcrt.locking(handle.fileno(), file_lock.msvcrt.LK_UNLCK, 1)


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("error", [errno.EACCES, getattr(errno, "EDEADLOCK", errno.EDEADLK)])
def test_windows_shared_retry_loop(tmp_path, monkeypatch, bounded, error):
    attempts = []

    def locking(fd, mode, count):
        attempts.append(os.lseek(fd, 0, os.SEEK_CUR))
        if len(attempts) == 1:
            raise OSError(error, "contended")

    monkeypatch.setattr(file_lock, "fcntl", None)
    monkeypatch.setattr(file_lock, "msvcrt", SimpleNamespace(LK_NBLCK=1, locking=locking))
    monkeypatch.setattr(file_lock.time, "sleep", lambda _: None)
    with (tmp_path / "lock").open("w+") as handle:
        handle.seek(3)
        if bounded:
            assert file_lock.lock_within(handle, 1) is True
        else:
            assert file_lock.lock(handle) is None
        assert handle.tell() == 3
    assert attempts == [file_lock.LOCK_OFFSET] * 2
