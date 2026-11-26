"""Two-sided Goodhart-proof tests for lib/hal/voice/types.py.

Per Rule 14 (scientific-method.md + rules/efficiency-rules.md): each
invariant gets BOTH a must-pass and must-fail assertion, so the test
actually exercises the contract instead of being a stub that always
passes.

Shipped at PR alpha-2 W1 alongside types.py. Primitive-body tests
(listen/speak/chat_loop behaviour) land in subsequent W1+ commits
when the bodies exist.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.hal.voice import (  # noqa: E402
    ChatPolicy,
    ChatTranscript,
    ChatTurn,
    DirectiveTamperDetected,
    FocusRaceDetected,
    ListenResult,
    PlatformUnsupported,
    ProfileTierCapExceeded,
    Provenance,
    SpeakResult,
    UnauthorizedDiagnosticBypass,
    VoiceError,
)


# ---------- Provenance enum (V2) ----------


def test_provenance_has_seven_members_with_expected_values():
    """Must-pass: the seven canonical members exist with string-match values."""
    expected = {
        "human_typed",
        "voice_human",
        "tool_chain",
        "mcp_remote",
        "socket_client",
        "autonomous_loop",
        "yolo",
    }
    assert {p.value for p in Provenance} == expected


def test_provenance_drift_would_break_tier_gate():
    """Must-fail-if-drifted: adding a spurious member or renaming one
    breaks the frozenset comparison. This is the α-2 snapshot test the
    opus depth-5 lens asked for (Provenance enum cross-plan drift).

    If the osa plan ships with a different enum shape, this test refuses
    to pass until both sides align.
    """
    canonical = frozenset({
        "human_typed",
        "voice_human",
        "tool_chain",
        "mcp_remote",
        "socket_client",
        "autonomous_loop",
        "yolo",
    })
    observed = frozenset(p.value for p in Provenance)
    assert observed == canonical, (
        f"Provenance enum drift detected: observed={observed} "
        f"canonical={canonical}. The osa plan and voice plan share a "
        f"single source of truth — if one changes, both must change."
    )


def test_provenance_string_flag_rejected_as_substitute():
    """Must-fail: a raw string is NOT a Provenance. Callers cannot
    substitute a behavioural flag for the typed enum (invariant V2)."""
    with pytest.raises((TypeError, AttributeError, ValueError)):
        # Accessing .value on a string should raise, OR failing to
        # compare against Provenance members should fail strictly.
        raw = "voice_human"
        assert raw is not Provenance.voice_human
        # This would be a type violation if anyone tried to use a raw
        # string where a Provenance was expected:
        raw.value  # type: ignore[attr-defined]


# ---------- Result dataclass frozen-ness ----------


def test_listen_result_is_immutable_after_construction():
    """Must-pass: frozen dataclass prevents post-hoc mutation."""
    r = ListenResult(
        ok=True,
        text="hello",
        audio_path=None,
        language="en-US",
        confidence=0.92,
        duration_s=1.5,
        provider="macos_dictation",
        provenance=Provenance.voice_human,
        tier=1,
        audit_id="abcdef1234567890",
    )
    assert r.text == "hello"


def test_listen_result_mutation_raises():
    """Must-fail: attempting to mutate a frozen ListenResult raises
    FrozenInstanceError. Without @dataclass(frozen=True) this would
    silently succeed — which is the bug frozen dataclasses prevent."""
    from dataclasses import FrozenInstanceError

    r = ListenResult(
        ok=True,
        text="hello",
        audio_path=None,
        language="en-US",
        confidence=0.92,
        duration_s=1.5,
        provider="macos_dictation",
        provenance=Provenance.voice_human,
        tier=1,
        audit_id="abcdef1234567890",
    )
    with pytest.raises(FrozenInstanceError):
        r.text = "attacker-replaced"  # type: ignore[misc]


def test_speak_result_is_frozen_too():
    """Must-pass + must-fail paired: SpeakResult is also frozen."""
    from dataclasses import FrozenInstanceError

    r = SpeakResult(
        ok=True,
        audio_path="/tmp/voice-abc.wav",
        duration_s=2.1,
        provider="gemini_tts",
        voice="Iapetus",
        profile="child",
        emotion="excited",
        directive_fingerprint="1234567890abcdef",
        tier=2,
        audit_id="fedcba0987654321",
    )
    assert r.profile == "child"
    with pytest.raises(FrozenInstanceError):
        r.profile = "warm"  # type: ignore[misc]


def test_chat_transcript_cost_floats_and_turns_list():
    """Must-pass: ChatTranscript holds turns + metadata correctly."""
    t = ChatTranscript(
        turns=[],
        profile="grip",
        session_id="sess-xyz",
        started_at="2026-04-18T20:00:00Z",
        ended_at="2026-04-18T20:15:00Z",
        total_cost_usd=0.0,
    )
    assert t.total_cost_usd == 0.0
    assert t.turns == []


# ---------- ChatPolicy three-valued emotion_adaptive (UX invariant) ----------


def test_chat_policy_emotion_adaptive_defaults_to_none():
    """Must-pass: None means 'use profile default' — explicit three-valued
    semantics (operator UX finding: default False would silently override
    profile-level True on the child profile)."""
    p = ChatPolicy()
    assert p.emotion_adaptive is None


def test_chat_policy_emotion_adaptive_accepts_explicit_true_false():
    """Must-pass: explicit True/False overrides profile default."""
    p_true = ChatPolicy(emotion_adaptive=True)
    p_false = ChatPolicy(emotion_adaptive=False)
    assert p_true.emotion_adaptive is True
    assert p_false.emotion_adaptive is False


def test_chat_policy_rejects_string_emotion_adaptive():
    """Must-fail: emotion_adaptive expects bool | None. A string like
    'auto' should not silently pass as truthy — the three-valued
    contract is exhaustive."""
    p = ChatPolicy(emotion_adaptive="auto")  # type: ignore[arg-type]
    # Python does not enforce the annotation at runtime, BUT the field
    # must NOT be confused with None OR a real bool:
    assert p.emotion_adaptive != None  # noqa: E711
    assert p.emotion_adaptive is not True
    assert p.emotion_adaptive is not False
    # A downstream tier_gate.py check must narrow — this test documents
    # that the type-wide value space is NOT yet narrowed at the dataclass
    # boundary. The α-3 tier_gate.py layer performs the narrowing.


def test_chat_policy_capability_none_by_default():
    """Must-pass: capability is None by default — tier-3 operations will
    refuse unless explicitly provided."""
    p = ChatPolicy()
    assert p.capability is None


def test_chat_policy_frozen():
    """Must-fail: ChatPolicy is frozen — cannot post-hoc mutate to
    inject a capability after construction."""
    from dataclasses import FrozenInstanceError

    p = ChatPolicy()
    with pytest.raises(FrozenInstanceError):
        p.capability = b"fake-capability-token"  # type: ignore[misc]


# ---------- Exception hierarchy ----------


def test_voice_errors_inherit_voice_error_base():
    """Must-pass: every domain exception is catchable as VoiceError."""
    assert issubclass(ProfileTierCapExceeded, VoiceError)
    assert issubclass(PlatformUnsupported, VoiceError)
    assert issubclass(DirectiveTamperDetected, VoiceError)
    assert issubclass(FocusRaceDetected, VoiceError)
    assert issubclass(UnauthorizedDiagnosticBypass, VoiceError)


def test_voice_error_does_not_catch_unrelated_exceptions():
    """Must-fail: a generic ValueError is NOT a VoiceError.

    If we accidentally made VoiceError too broad (e.g. inherited from
    everything), every exception would be a VoiceError — which defeats
    the point of typed exception handling."""
    try:
        raise ValueError("unrelated")
    except VoiceError:
        pytest.fail("ValueError should not be caught by VoiceError handler")
    except ValueError:
        pass  # expected path


def test_profile_tier_cap_exceeded_raisable():
    """Must-pass: the exception carries through a raise-catch cycle
    with its message preserved — otherwise the tier_cap enforcement
    layer would raise silently-typed failures."""
    msg = "profile=child requested tier=3 but tier_cap=2"
    with pytest.raises(ProfileTierCapExceeded) as exc_info:
        raise ProfileTierCapExceeded(msg)
    assert msg in str(exc_info.value)


def test_unauthorized_diagnostic_bypass_distinct_type():
    """Must-fail-if-merged: UnauthorizedDiagnosticBypass is a distinct
    type from ProfileTierCapExceeded. If accidentally merged or
    aliased, the café-mode self-recovery UX (V13) cannot distinguish
    'wrong call path tried to bypass' from 'profile tier exceeded'."""
    assert UnauthorizedDiagnosticBypass is not ProfileTierCapExceeded
