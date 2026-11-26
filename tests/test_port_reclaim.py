"""Tests for lib.hal.port_reclaim — the idempotent-launch port primitive.

Mutation-resistant by construction: the load-bearing cases drive a REAL
listener subprocess through REAL ``lsof`` and a REAL signal (so gutting
``reclaim_port`` makes them fail), alongside fast mocked cases that pin the
parsing, self-exclusion, and SIGTERM->SIGKILL escalation contracts.
"""
from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import time

import pytest

from lib.hal import port_reclaim
from lib.hal.port_reclaim import listening_pids, port_in_use, reclaim_port


def _free_port() -> int:
    """A currently-free 127.0.0.1 TCP port (bind to :0, read it back, release)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --- listening_pids: parsing + graceful degradation -----------------------

def test_listening_pids_parses_bare_pids(monkeypatch):
    monkeypatch.setattr(port_reclaim.shutil, "which", lambda _name: "/usr/sbin/lsof")

    class _Completed:
        stdout = "123\n456\n789\n"

    monkeypatch.setattr(port_reclaim.subprocess, "run", lambda *a, **k: _Completed())
    assert listening_pids(4010) == [123, 456, 789]


def test_listening_pids_skips_non_numeric_tokens(monkeypatch):
    monkeypatch.setattr(port_reclaim.shutil, "which", lambda _name: "/usr/sbin/lsof")

    class _Completed:
        stdout = "123\nnot-a-pid\n\n456\n"

    monkeypatch.setattr(port_reclaim.subprocess, "run", lambda *a, **k: _Completed())
    assert listening_pids(4010) == [123, 456]


def test_listening_pids_empty_when_lsof_absent(monkeypatch):
    monkeypatch.setattr(port_reclaim.shutil, "which", lambda _name: None)

    def _must_not_run(*a, **k):  # pragma: no cover - asserts non-invocation
        raise AssertionError("subprocess.run called despite missing lsof")

    monkeypatch.setattr(port_reclaim.subprocess, "run", _must_not_run)
    assert listening_pids(4010) == []


def test_listening_pids_empty_on_lsof_error(monkeypatch):
    monkeypatch.setattr(port_reclaim.shutil, "which", lambda _name: "/usr/sbin/lsof")

    def _raise(*a, **k):
        raise subprocess.TimeoutExpired(cmd="lsof", timeout=5)

    monkeypatch.setattr(port_reclaim.subprocess, "run", _raise)
    assert listening_pids(4010) == []


# --- real free port: port_in_use False, reclaim is a no-op ----------------

def test_port_in_use_false_for_free_port():
    assert port_in_use(_free_port()) is False


def test_reclaim_port_noop_when_free():
    assert reclaim_port(_free_port()) == []


# --- reclaim_port: self-exclusion + signal escalation (mocked) ------------

def test_reclaim_port_never_signals_self(monkeypatch):
    monkeypatch.setattr(port_reclaim, "listening_pids", lambda _p: [os.getpid()])
    calls = []
    monkeypatch.setattr(port_reclaim.os, "kill", lambda pid, sig: calls.append((pid, sig)))
    assert reclaim_port(4010) == []
    assert calls == []  # own PID filtered -> os.kill never invoked


def test_reclaim_port_sigterm_then_sigkill_on_straggler(monkeypatch):
    holder = 999999  # a PID that "keeps listening" through the grace window
    monkeypatch.setattr(port_reclaim, "listening_pids", lambda _p: [holder])
    sent = []
    monkeypatch.setattr(port_reclaim.os, "kill", lambda pid, sig: sent.append((pid, sig)))

    assert reclaim_port(4010, grace_s=0.2) == [holder]
    assert (holder, signal.SIGTERM) in sent
    assert (holder, signal.SIGKILL) in sent
    assert sent.index((holder, signal.SIGTERM)) < sent.index((holder, signal.SIGKILL))


def test_reclaim_port_graceful_exit_no_sigkill(monkeypatch):
    holder = 999998  # present on first scan, gone afterwards -> SIGTERM suffices
    scans = iter([[holder], [], [], []])
    monkeypatch.setattr(port_reclaim, "listening_pids", lambda _p: next(scans, []))
    sent = []
    monkeypatch.setattr(port_reclaim.os, "kill", lambda pid, sig: sent.append((pid, sig)))

    assert reclaim_port(4010, grace_s=1.0) == [holder]
    assert (holder, signal.SIGTERM) in sent
    assert (holder, signal.SIGKILL) not in sent


# --- the Goodhart anchor: real subprocess, real lsof, real kill -----------

_LISTENER_SRC = (
    "import socketserver, http.server, sys;"
    "httpd = socketserver.TCPServer(('127.0.0.1', int(sys.argv[1])),"
    " http.server.BaseHTTPRequestHandler);"
    "httpd.serve_forever()"
)


@pytest.mark.skipif(shutil.which("lsof") is None, reason="lsof required")
def test_reclaim_port_kills_real_listener():
    port = _free_port()
    proc = subprocess.Popen([sys.executable, "-c", _LISTENER_SRC, str(port)])
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not port_in_use(port):
            time.sleep(0.05)
        assert port_in_use(port), "listener subprocess never bound the port"

        killed = reclaim_port(port)
        assert proc.pid in killed

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and port_in_use(port):
            time.sleep(0.05)
        assert not port_in_use(port), "port still held after reclaim"
        assert proc.wait(timeout=5) is not None  # process actually exited
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
