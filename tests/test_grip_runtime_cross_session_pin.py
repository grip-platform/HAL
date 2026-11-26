"""Goodhart-resistant: cross-session pin resolution for the HAL daemon.

Root cause (2026-05-19): the active-llm marker is CLAUDE_CODE_SESSION_ID
-keyed (active_llm._resolve_cc_pid); the long-lived cross-session
hal-server daemon has no such env, so it never saw a CC session's
`/llm pin` and always fell to the CCH ladder primary. `_active_marker`
now falls back to the most-recent valid per-session marker.

Mutation sensitivity: revert the fallback / the gate / the primary path
and a test below fails.
"""
import json
import os
import time

import lib.hal.grip_runtime as G

KIMI = "moonshot/kimi-k3"  # the ladder's Kimi rung (kimi-k2.5 retired 2026-08-31)


def _marker(addr, mode="solo", ttl=3600, ts=None):
    return {"mode": mode, "address": addr,
            "ts": ts if ts is not None else int(time.time()),
            "ttl_seconds": ttl}


def test_daemon_resolves_most_recent_valid_session_pin(tmp_path, monkeypatch):
    monkeypatch.setenv("GRIP_TMP_DIR", str(tmp_path))
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    monkeypatch.delenv("GRIP_CC_PID", raising=False)
    monkeypatch.setattr(G, "_grip_active_llm", lambda: None)  # daemon: no session marker
    old = tmp_path / "grip-session-active-llm-A.json"
    new = tmp_path / "grip-session-active-llm-B.json"
    old.write_text(json.dumps(_marker("groq/llama-3.3-70b-versatile")))
    new.write_text(json.dumps(_marker(KIMI)))
    os.utime(old, (1, 1))            # force `old` older
    os.utime(new, (time.time(), time.time()))
    rec = G._active_marker()
    assert rec is not None and rec["address"] == KIMI


def test_failsafe_when_no_session_markers(tmp_path, monkeypatch):
    monkeypatch.setenv("GRIP_TMP_DIR", str(tmp_path))
    monkeypatch.setattr(G, "_grip_active_llm", lambda: None)
    assert G._active_marker() is None  # -> fixed-ladder primary (CCH), unchanged


def test_stale_session_marker_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("GRIP_TMP_DIR", str(tmp_path))
    monkeypatch.setattr(G, "_grip_active_llm", lambda: None)
    (tmp_path / "grip-session-active-llm-S.json").write_text(
        json.dumps(_marker(KIMI, ts=1, ttl=1)))  # expired
    assert G._active_marker() is None


def test_astra_pin_counts_only_while_paid_openai_is_opted_in(tmp_path, monkeypatch):
    """The Astra rung bills OpenAI per token, so a pin to it is honoured only
    while GRIP_CODEX_ALLOW_PAID opts that in, the same rule the ladder applies.
    An empty marker directory keeps the cross-session fallback hermetic."""
    class _Mod:
        @staticmethod
        def show_active():
            return _marker("openai/gpt-6-astra")
    monkeypatch.setenv("GRIP_TMP_DIR", str(tmp_path))
    monkeypatch.setattr(G, "_grip_active_llm", lambda: _Mod)
    monkeypatch.setenv("GRIP_CODEX_ALLOW_PAID", "1")
    rec = G._active_marker()
    assert rec is not None and rec["address"] == "openai/gpt-6-astra"
    monkeypatch.delenv("GRIP_CODEX_ALLOW_PAID")
    assert G._active_marker() is None


def test_in_session_primary_path_unchanged(monkeypatch):
    class _Mod:
        @staticmethod
        def show_active():
            return _marker(KIMI)
    monkeypatch.setattr(G, "_grip_active_llm", lambda: _Mod)
    rec = G._active_marker()
    assert rec is not None and rec["address"] == KIMI


def test_in_session_nonsolo_marker_not_overridden_by_cross_session(
        tmp_path, monkeypatch):
    """HAL-Council PR #360 concern: an IN-SESSION deliberate non-solo
    marker (round-robin) must NOT be overridden by a *different*
    session's pin. _resolve_cc_pid() not-None => in-session =>
    show_active() authoritative => no scan. Mutation: drop the
    _resolve_cc_pid in-session gate -> this returns KIMI and fails."""
    monkeypatch.setenv("GRIP_TMP_DIR", str(tmp_path))
    # a valid cross-session pin sits on disk and MUST be ignored
    (tmp_path / "grip-session-active-llm-OTHER.json").write_text(
        json.dumps(_marker(KIMI)))

    class _Mod:
        @staticmethod
        def show_active():
            return _marker(KIMI, mode="round-robin")  # deliberate no-pin

        @staticmethod
        def _resolve_cc_pid():
            return 4242  # in-session: resolves its OWN marker

    monkeypatch.setattr(G, "_grip_active_llm", lambda: _Mod)
    assert G._active_marker() is None  # honour in-session non-pin


def test_daemon_scans_even_when_mod_present_but_unresolvable(
        tmp_path, monkeypatch):
    """The REAL daemon: _grip_active_llm() loads (mod present) but
    _resolve_cc_pid() is None (no session env -> stale GLOBAL). The
    original suite forced mod=None and never exercised this — the
    council found the gap. Fix must still scan here. Mutation: gate the
    scan on 'mod is None' instead of _resolve_cc_pid -> daemon never
    sees the pin and this fails."""
    monkeypatch.setenv("GRIP_TMP_DIR", str(tmp_path))
    (tmp_path / "grip-session-active-llm-A.json").write_text(
        json.dumps(_marker(KIMI)))

    class _Mod:
        @staticmethod
        def show_active():
            return None  # daemon: stale GLOBAL empty

        @staticmethod
        def _resolve_cc_pid():
            return None  # cross-session consumer (the bug axis)

    monkeypatch.setattr(G, "_grip_active_llm", lambda: _Mod)
    rec = G._active_marker()
    assert rec is not None and rec["address"] == KIMI


def test_scan_failsafe_when_glob_raises(tmp_path, monkeypatch):
    """HAL-Council PR #360 round-2 (glob.glob raises). Called DIRECTLY
    (not via _active_marker, whose own catch-all would mask this) so it
    is mutation-sensitive to _scan_session_markers' OWN outer fail-safe:
    a denied/locked TMPDIR -> None (fixed ladder), never propagates."""
    monkeypatch.setenv("GRIP_TMP_DIR", str(tmp_path))
    (tmp_path / "grip-session-active-llm-X.json").write_text(
        json.dumps(_marker(KIMI)))
    import glob as _g

    def _boom(*a, **k):
        raise PermissionError("simulated denied TMPDIR")

    monkeypatch.setattr(_g, "glob", _boom)
    assert G._scan_session_markers() is None


def test_scan_skips_unstattable_sibling_keeps_valid_pin(
        tmp_path, monkeypatch):
    """The REAL gap behind the round-2 instinct: one sibling marker
    vanishes/locks between glob and stat (TOCTOU). Old code sorted with
    key=os.stat -> the whole sort raised -> a *valid* pin in another
    file was dropped. Mutation: revert to key=os.stat in sorted() and
    this returns None and fails. Called directly (scoped to the
    per-file resilience, not the caller's catch-all)."""
    monkeypatch.setenv("GRIP_TMP_DIR", str(tmp_path))
    good = tmp_path / "grip-session-active-llm-GOOD.json"
    bad = tmp_path / "grip-session-active-llm-BAD.json"
    good.write_text(json.dumps(_marker(KIMI)))
    bad.write_text(json.dumps(_marker("groq/llama-3.3-70b-versatile")))
    real_stat = os.stat

    def _flaky_stat(path, *a, **k):
        if str(path).endswith("BAD.json"):
            raise PermissionError("simulated vanished/locked sibling")
        return real_stat(path, *a, **k)

    monkeypatch.setattr(os, "stat", _flaky_stat)
    rec = G._scan_session_markers()
    assert rec is not None and rec["address"] == KIMI
