"""Antigravity as a first-class HAL harness — parity + detection + live peer.

Covers the demo's "three giants" coherence claim: Claude Code, Codex, and
Antigravity are ALL first-class harnesses in HAL. The a2a peer mesh
(``lib/hal/a2a_peer.py``) already delegates to the Antigravity/Gemini CLI; this
asserts ``harness_detect`` agrees that the harness exists (name parity), detects
it via the established ``*_SESSION_ID`` convention, and stays fail-safe when no
harness env is present.

Goodhart-resistant: every test fails under the obvious mutation —
``test_antigravity_in_detection_rules`` fails if the rule is removed,
``test_detect_failsafe_standalone`` fails if detection is hardcoded to a harness.
"""
import os

import pytest

from lib.hal import a2a_peer
from lib.hal.harness_detect import Harness, _DETECTION_RULES, detect_harness

# Every harness env var the rules key on — cleared before each detection test so
# the ambient harness (tests run UNDER Claude Code, so CLAUDE_CODE_SESSION_ID is
# set) cannot leak into a first-match-wins result.
_ALL_HARNESS_ENV = [env for env, _ in _DETECTION_RULES]


def _clear_harness_env(monkeypatch):
    for env in _ALL_HARNESS_ENV:
        monkeypatch.delenv(env, raising=False)


def test_antigravity_is_first_class_harness():
    assert Harness.ANTIGRAVITY.value == "antigravity"


def test_harness_name_parity_with_a2a_peer():
    """Each a2a_peer canonical peer id has a Harness member of the SAME NAME.

    Parity is by enum NAME, not value: a2a_peer.CLAUDE_CODE == "claude_code"
    (underscore) while Harness.CLAUDE_CODE.value == "claude-code" (hyphen), so a
    value comparison would be wrong. The invariant that matters is that the two
    modules agree the harness *exists*.
    """
    peer_names = {"CLAUDE_CODE", "CODEX", "ANTIGRAVITY"}
    # Guard: these names really are defined in a2a_peer (refactor-proof).
    for name in peer_names:
        assert hasattr(a2a_peer, name), f"a2a_peer missing peer constant {name}"
    harness_names = set(Harness.__members__)
    missing = peer_names - harness_names
    assert not missing, f"Harness enum missing first-class peers: {missing}"


def test_antigravity_in_detection_rules():
    """Regression anchor: the ANTIGRAVITY_SESSION_ID rule must be present."""
    assert ("ANTIGRAVITY_SESSION_ID", Harness.ANTIGRAVITY) in _DETECTION_RULES


def test_detect_antigravity_via_session_id(monkeypatch):
    _clear_harness_env(monkeypatch)
    monkeypatch.setenv("ANTIGRAVITY_SESSION_ID", "agy-demo-uuid")
    assert detect_harness() is Harness.ANTIGRAVITY


def test_detect_failsafe_standalone(monkeypatch):
    """No harness env -> STANDALONE (proves detection isn't hardcoded)."""
    _clear_harness_env(monkeypatch)
    assert detect_harness() is Harness.STANDALONE


def test_detect_first_match_wins_does_not_misroute_antigravity(monkeypatch):
    """A real Claude Code env must NOT be mis-detected as Antigravity."""
    _clear_harness_env(monkeypatch)
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "cc-uuid")
    assert detect_harness() is Harness.CLAUDE_CODE


def test_live_antigravity_peer_roundtrip():
    """Env-gated live e2e: the Antigravity/Gemini CLI answers, read-only locked.

    Skipped unless HAL_LIVE_PEER=1 (needs the `gemini` CLI installed + authed).
    This is the demo's "Antigravity participates" proof as a repeatable test —
    the manual run returned rc 0 / "MESH-OK" on 2026-06-02.
    """
    if os.environ.get("HAL_LIVE_PEER") != "1":
        pytest.skip("set HAL_LIVE_PEER=1 to run the live Antigravity peer round-trip")
    reply = a2a_peer.ask_peer("antigravity", "Reply with exactly: MESH-OK", timeout=90.0)
    assert reply.harness == a2a_peer.ANTIGRAVITY
    assert reply.ok, f"peer failed: {reply.output[:200]}"
    assert "MESH-OK" in reply.output
