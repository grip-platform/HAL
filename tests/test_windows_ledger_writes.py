"""Exercise Windows byte translation and replacement failures on every platform."""
import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_thread_creation_and_append_preserve_raw_bytes(tmp_path, monkeypatch):
    thread = importlib.import_module("lib.hal.infinite_thread")
    binary = getattr(os, "O_BINARY", 0x8000)
    modes = {}

    def open_fd(path, flags, mode):
        fd = os.open(path, flags if os.name == "nt" else flags & ~binary, mode)
        modes[fd] = bool(flags & binary)
        return fd

    def write_fd(fd, data):
        raw = bytes(data)
        stored = raw if modes[fd] else raw.replace(b"\n", b"\r\n")
        assert os.write(fd, stored) == len(stored)
        return len(raw)

    # Model the CRT's default text mode without changing the host's os module.
    proxy = SimpleNamespace(**vars(os))
    proxy.open, proxy.write = open_fd, write_fd
    monkeypatch.setattr(thread, "os", proxy)
    monkeypatch.setattr(thread, "_BINARY", binary)
    path = tmp_path / "thread.jsonl"
    first, second = b'{"seq":1}\n', b'{"seq":2}\n'
    thread._atomic_write(path, first)
    assert path.read_bytes() == first
    with thread._flocked(path, create=False) as fd:
        thread._write_all(fd, second)
    assert path.read_bytes() == first + second


@pytest.mark.parametrize("platform,error,failures,succeeds", [
    ("nt", PermissionError, 2, True),
    ("nt", PermissionError, 1000, False),
    ("posix", PermissionError, 1, False),
    ("nt", OSError, 1, False),
])
def test_daily_ledger_replace_retry(tmp_path, monkeypatch, platform, error,
                                  failures, succeeds):
    guard = importlib.import_module("lib.hal.spend_guard")
    path = tmp_path / "daily.json"
    old, new = {"old": {}}, {"new": {}}
    guard._write_days(path, old)
    real_replace = Path.replace
    attempts, sleeps = [], []
    clock = [0.0]

    def replace(source, target):
        attempts.append(source)
        assert guard._read_days(path) == old
        if len(attempts) <= failures:
            raise error("injected replacement failure")
        return real_replace(source, target)

    def sleep(delay):
        sleeps.append(delay)
        clock[0] += delay

    proxy = SimpleNamespace(**vars(os))
    proxy.name = platform
    monkeypatch.setattr(guard, "os", proxy)
    monkeypatch.setattr(guard, "time", SimpleNamespace(
        monotonic=lambda: clock[0], sleep=sleep))
    monkeypatch.setattr(Path, "replace", replace)
    if succeeds:
        guard._write_days(path, new)
        assert guard._read_days(path) == new
        assert len(attempts) == failures + 1
        assert sleeps == [0.005] * failures
        assert not attempts[-1].exists()
    else:
        with pytest.raises(error, match="injected replacement failure"):
            guard._write_days(path, new)
        assert guard._read_days(path) == old
        if platform == "nt" and error is PermissionError:
            assert 1.0 <= clock[0] < 1.006
        else:
            assert len(attempts) == 1
            assert sleeps == []
