"""Goodhart-proof tests for GET /health + the InfiniteMachine brake (ADR-0002 W0).

Mutation map: the identity tests fail if ok/service change or the route
vanishes; the exploding-proxy fixture fails the suite if /health ever
touches the proxy (an upstream call would explode); the brake tests fail
if presence stops engaging, if a corrupt marker stops engaging
(fail-closed inverted), or if the WhatsApp outbound-stop marker starts
cross-tripping the InfiniteMachine brake (false-trip anchor).
"""
from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from lib.hal import gateway_brake
from lib.hal.gateway import InflightCap, make_gateway_handler


class _ExplodingProxy:
    """Stand-in that fails LOUDLY if /health touches the proxy at all."""

    def __getattr__(self, name):
        raise AssertionError(f"/health must not touch the proxy (accessed .{name})")


@pytest.fixture()
def health_server(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_DIR", str(tmp_path))
    monkeypatch.delenv("HAL_INFINITE_BRAKE_MARKER", raising=False)
    handler = make_gateway_handler(_ExplodingProxy(), InflightCap(64, 5))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_port}", tmp_path
    finally:
        srv.shutdown()


def _get_json(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


# --- /health route ------------------------------------------------------------

def test_health_identity_no_auth_no_upstream(health_server):
    base, _ = health_server
    status, doc = _get_json(base + "/health")
    assert status == 200
    assert doc["ok"] is True
    assert doc["service"] == "hal-gateway"
    assert doc["brake"]["engaged"] is False


def test_health_reports_engaged_brake_with_detail(health_server):
    base, tmp = health_server
    marker = tmp / "state" / "session" / "infinite-stop.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({"by": "test", "reason": "drill"}), encoding="utf-8")
    status, doc = _get_json(base + "/health")
    assert status == 200
    assert doc["brake"]["engaged"] is True
    assert doc["brake"]["detail"]["reason"] == "drill"


def test_non_v1_non_health_paths_stay_404(health_server):
    base, _ = health_server
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(base + "/nope", timeout=5)
    assert exc.value.code == 404


# --- brake semantics ----------------------------------------------------------

def test_brake_absent_marker_released(tmp_path):
    assert gateway_brake.brake_engaged({"CLAUDE_DIR": str(tmp_path)}) is False


def test_brake_present_marker_engaged(tmp_path):
    m = tmp_path / "state" / "session" / "infinite-stop.json"
    m.parent.mkdir(parents=True)
    m.write_text("{}", encoding="utf-8")
    assert gateway_brake.brake_engaged({"CLAUDE_DIR": str(tmp_path)}) is True


def test_brake_corrupt_marker_still_engages_fail_closed(tmp_path):
    m = tmp_path / "state" / "session" / "infinite-stop.json"
    m.parent.mkdir(parents=True)
    m.write_bytes(b"\xff\xfe not json {")
    env = {"CLAUDE_DIR": str(tmp_path)}
    assert gateway_brake.brake_engaged(env) is True
    st = gateway_brake.brake_state(env)
    assert st["engaged"] is True
    assert st["detail"] is None  # cannot explain itself, still brakes


def test_brake_env_override_wins(tmp_path):
    m = tmp_path / "elsewhere.json"
    m.write_text("{}", encoding="utf-8")
    env = {"HAL_INFINITE_BRAKE_MARKER": str(m),
           "CLAUDE_DIR": str(tmp_path / "unused")}
    assert gateway_brake.marker_path(env) == m
    assert gateway_brake.brake_engaged(env) is True


def test_false_trip_outbound_stop_does_not_engage_im_brake(tmp_path):
    # The WhatsApp brake (grip-stop → outbound-stop.json) and the IM brake
    # (infinite-stop.json) are DISTINCT markers by design — engaging one
    # must never trip the other.
    other = tmp_path / "state" / "session" / "outbound-stop.json"
    other.parent.mkdir(parents=True)
    other.write_text(json.dumps({"by": "grip-stop"}), encoding="utf-8")
    assert gateway_brake.brake_engaged({"CLAUDE_DIR": str(tmp_path)}) is False


def test_false_trip_the_wire_egress_brake_and_the_im_brake_are_distinct(tmp_path):
    from lib.hal.wire_proxy import EGRESS_BRAKE
    env = {"CLAUDE_DIR": str(tmp_path)}
    egress = EGRESS_BRAKE.path(env)
    egress.parent.mkdir(parents=True)
    egress.write_text("{}", encoding="utf-8")
    assert EGRESS_BRAKE.engaged(env) is True
    assert gateway_brake.brake_engaged(env) is False
    egress.unlink()
    gateway_brake.marker_path(env).write_text("{}", encoding="utf-8")
    assert EGRESS_BRAKE.engaged(env) is False


def test_presence_is_decided_without_path_exists(tmp_path, monkeypatch):
    """Version-independent anchor for the test below: with Path.exists() made to
    answer False, as 3.14's does for a path it may not inspect, a present marker
    must still engage. Catches a return to Path.exists() on any Python, where the
    permission test below catches it only on 3.14."""
    marker = tmp_path / "infinite-stop.json"
    marker.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(Path, "exists", lambda self, *args, **kwargs: False)
    env = {"HAL_INFINITE_BRAKE_MARKER": str(marker)}
    assert gateway_brake.brake_engaged(env) is True
    assert gateway_brake.brake_state(env)["detail"] == {}


def test_brake_unstattable_marker_still_engages_fail_closed(tmp_path, monkeypatch):
    # Windows chmod cannot make a directory unstattable, so induce the exact
    # cross-platform contract under test: os.stat raises a non-absence error.
    marker = tmp_path / "locked" / "infinite-stop.json"
    env = {"HAL_INFINITE_BRAKE_MARKER": str(marker)}
    real_stat = os.stat

    def _denied(path, *args, **kwargs):
        if os.fspath(path) == os.fspath(marker):
            raise PermissionError(13, "Permission denied")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", _denied)
    assert gateway_brake.brake_engaged(env) is True
    assert gateway_brake.brake_state(env)["engaged"] is True


@pytest.mark.skipif(os.name == "nt",
                    reason="Windows chmod cannot make a directory "
                           "unstattable; covered by the mocked os.stat "
                           "version above")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                    reason="root can inspect any path, so nothing is "
                           "uncheckable")
def test_brake_real_unstattable_marker_still_engages_fail_closed_posix(tmp_path):
    """Mutation: use ``Path.exists`` and treat an inspection error as absent.
    Survives through a real POSIX permission failure, complementing the
    cross-platform mocked syscall test above."""
    locked = tmp_path / "locked"
    locked.mkdir()
    env = {"HAL_INFINITE_BRAKE_MARKER": str(locked / "infinite-stop.json")}
    locked.chmod(0)
    try:
        assert gateway_brake.brake_engaged(env) is True
        assert gateway_brake.brake_state(env)["engaged"] is True
    finally:
        locked.chmod(0o700)
