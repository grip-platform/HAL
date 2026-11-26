"""HAL Voice — listen / speak / chat_loop / chat primitive bodies.

Per plans/voice-first-harness-triphase.md Rev 2.3 section 4.5 and
plans/voice-tier3-cloud-fallover.md §4 W4 (ladder-aware dispatch).

These four public functions are the operator-facing surface of the
voice harness. Each one is a small dispatcher that looks up a
provider from PROVIDER_REGISTRY, optionally consults the cache,
optionally logs to the audit chain, and delegates OS I/O (microphone
recording, speaker playback) to a caller-injectable callable.

The injectable seams (``recorder``, ``player``, ``responder``) let
unit tests avoid touching the real mic/speaker and let the alpha-3
CLI plug in macOS-specific subprocess wrappers (SwitchAudioSource,
afplay) without the primitives themselves growing platform branches.

At alpha-2 W5 the default recorder raises ``PlatformUnsupported`` —
there is no generic cross-platform microphone capture in the standard
library. Alpha-3 ships the default SwitchAudioSource + SoX-based
recorder for macOS alongside the concrete provider classes.

Contract summary:

- ``listen``: 60-second hard clamp on ``max_duration_s``; tier
  downgrades to 1 when the STT provider reports confidence < 0.85;
  raises ``VoiceError`` when the requested provider isn't in the
  PROVIDER_REGISTRY.
- ``speak``: fingerprint the request, consult the cache, call the
  provider on miss, store iff emotion_match ≥ CACHE_EMOTION_THRESHOLD.
  Writes the audio to tmpfs via ``tmpfs_audio_path`` so the player
  sees an owned, short-lived file. W4 adds a ``ladder=`` kwarg that
  exchanges the single-provider path for a Tier-3 fallover cascade
  classified by ``dispatch.is_retryable``.
- ``chat_loop``: alternates ``listen`` → ``responder`` → ``speak``
  with per-phase latency instrumentation; honours ``policy.max_turns``
  and ``policy.terminator``.
- ``chat``: convenience wrapper with a default ``ChatPolicy``.

Shipped at PR alpha-2 W5. Ladder dispatch shipped at PR alpha-3e W4.
"""
from __future__ import annotations

import os
import platform
import secrets
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .audit import AuditLogger
from .cache import (
    CACHE_EMOTION_THRESHOLD,
    VoiceCache,
    fingerprint,
    tmpfs_audio_path,
)
from .dispatch import is_retryable
from .ladder import get_ladder
from .providers import PROVIDER_REGISTRY
from .usage import UsageCounter
from .types import (
    ChatPolicy,
    ChatTranscript,
    ChatTurn,
    ListenResult,
    PlatformUnsupported,
    Provenance,
    SpeakResult,
    VoiceError,
)


# ---------- injectable I/O seams ----------

Recorder = Callable[[float, Path], None]
"""Record mic audio for up to ``max_duration_s`` seconds into ``out_path``."""

Player = Callable[[Path], float]
"""Play audio at ``path``; return realised playback duration in seconds."""

Responder = Callable[[str, list[ChatTurn]], str]
"""Generate a response given latest user text + conversation history."""


def _default_player(path: Path) -> float:
    from .audio_controller import AudioController
    return AudioController.get().play_file(path)


def _default_recorder(max_duration_s: float, out_path: Path) -> None:
    if platform.system() != "Darwin":
        raise PlatformUnsupported(
            "default recorder: macOS only (ffmpeg avfoundation)"
        )
    from .providers.macos_recorder import record as _ffmpeg_record
    _ffmpeg_record(max_duration_s, out_path)


# ---------- constants ----------

MAX_DURATION_HARD_CLAMP_S = 60.0
"""Hard upper bound on ``max_duration_s`` across every listen caller.

Protects against runaway microphones (crash, focus race, adversarial
caller) filling disk or hogging the STT provider. The plan's V1 invariant.
"""

TIER_CONFIDENCE_FLOOR = 0.85
"""Below this STT confidence the effective tier is forced to 1.

Keeps an uncertain transcription from authorising tier-2+ actions.
"""


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


# ---------- listen ----------


def listen(
    *,
    provider: str = "macos_dictation",
    language: str = "en-US",
    max_duration_s: float = 15.0,
    profile: str = "default",
    tier: int = 1,
    recorder: Recorder = _default_recorder,
    audit_logger: AuditLogger | None = None,
    provenance: Provenance = Provenance.voice_human,
) -> ListenResult:
    """Record and transcribe a single utterance.

    The recorder writes into a tmpfs path that is unlinked after the
    provider reads it, so transcribed audio never lands on persistent
    disk.
    """
    if max_duration_s > MAX_DURATION_HARD_CLAMP_S:
        raise VoiceError(
            f"max_duration_s={max_duration_s} exceeds hard clamp "
            f"{MAX_DURATION_HARD_CLAMP_S}"
        )
    if max_duration_s <= 0:
        raise VoiceError(f"max_duration_s must be positive, got {max_duration_s}")

    provider_class = PROVIDER_REGISTRY["stt"].get(provider)
    if provider_class is None:
        known = sorted(PROVIDER_REGISTRY["stt"].keys())
        raise VoiceError(
            f"STT provider '{provider}' not registered. Known: {known}"
        )
    prov = provider_class()

    with tmpfs_audio_path(b"", suffix=".wav") as audio_path:
        recorder(max_duration_s, audio_path)
        stt_result = prov.transcribe(
            audio_path=audio_path,
            language=language,
            max_duration_s=max_duration_s,
        )

    effective_tier = tier
    if stt_result.confidence < TIER_CONFIDENCE_FLOOR and effective_tier > 1:
        effective_tier = 1

    audit_id = ""
    if audit_logger is not None:
        audit_id = audit_logger.append(
            event="voice.listen",
            payload={
                "provider": provider,
                "language": stt_result.language,
                "confidence": stt_result.confidence,
                "duration_s": stt_result.duration_s,
                "profile": profile,
                "tier": effective_tier,
                "provenance": provenance.value,
            },
        )

    return ListenResult(
        ok=True,
        text=stt_result.text,
        audio_path=None,
        language=stt_result.language,
        confidence=stt_result.confidence,
        duration_s=stt_result.duration_s,
        provider=stt_result.provider_name,
        provenance=provenance,
        tier=effective_tier,
        audit_id=audit_id,
    )


# ---------- speak ----------


def speak(
    *,
    text: str,
    voice: str = "Iapetus",
    emotion: str | None = None,
    provider: str | None = None,
    ladder: str | None = None,
    profile: str = "default",
    tier: int = 1,
    cache: VoiceCache | None = None,
    audit_logger: AuditLogger | None = None,
    player: Player = _default_player,
    provenance: Provenance = Provenance.tool_chain,
    usage_counter: UsageCounter | None = None,
) -> SpeakResult:
    """Synthesise and play text.

    Two dispatch modes, mutually exclusive:

    - ``provider=<name>`` (default ``"macos_say"`` when neither is set):
      single-provider path. Cache ⇒ provider ⇒ tmpfs ⇒ player. Cache
      miss invokes the provider and stores the result only when
      ``emotion_match`` clears the plan's 0.8 threshold.
    - ``ladder=<name>``: Tier-3 fallover path. Iterate the named ladder
      in order, skip unregistered names with a warning, and try the
      next provider on any ``is_retryable`` exception. Non-retryable
      exceptions (caller errors) are re-raised immediately so the
      operator fixes the input rather than burning API calls.

    Passing BOTH ``provider=`` and ``ladder=`` raises ``VoiceError`` —
    the dispatch mode is a single choice, never overlapping.
    """
    if not text:
        raise VoiceError("speak() requires non-empty text")

    if provider is not None and ladder is not None:
        raise VoiceError(
            "speak(): provider= and ladder= are mutually exclusive"
        )

    common = dict(
        text=text,
        voice=voice,
        emotion=emotion,
        profile=profile,
        tier=tier,
        cache=cache,
        audit_logger=audit_logger,
        player=player,
        provenance=provenance,
        usage_counter=usage_counter,
    )

    if ladder is not None:
        return _speak_with_ladder(ladder_name=ladder, **common)

    return _speak_with_provider(
        provider=provider if provider is not None else "macos_say",
        **common,
    )


def _speak_with_provider(
    *,
    text: str,
    voice: str,
    emotion: str | None,
    provider: str,
    profile: str,
    tier: int,
    cache: VoiceCache | None,
    audit_logger: AuditLogger | None,
    player: Player,
    provenance: Provenance,
    usage_counter: UsageCounter | None = None,
) -> SpeakResult:
    """Single-provider synthesise + play path.

    Factored out of ``speak`` so ``_speak_with_ladder`` can loop over
    it without the ladder-dispatch logic leaking into the provider
    invocation semantics.

    Streaming shortcut (macos_say + default player + multi-sentence):
    When all three conditions hold, text is routed through
    ``AudioController.speak()`` which splits at sentence boundaries and
    overlaps synthesis of sentence N+1 with playback of sentence N,
    hiding synthesis latency. Cache and tmpfs are bypassed on this path;
    the single-shot path handles cache-hit replay and all other providers.
    """
    provider_class = PROVIDER_REGISTRY["tts"].get(provider)
    if provider_class is None:
        known = sorted(PROVIDER_REGISTRY["tts"].keys())
        raise VoiceError(
            f"TTS provider '{provider}' not registered. Known: {known}"
        )
    prov = provider_class()

    fp = fingerprint(
        text=text, voice=voice, emotion=emotion, provider=provider
    )

    audio: bytes | None = None
    emotion_match = 1.0
    cache_hit = False
    duration_s = 0.0

    if cache is not None:
        audio = cache.get(fp)
        cache_hit = audio is not None

    if audio is None:
        # Streaming path: macos_say + default player + multi-sentence text.
        # Deferred imports keep the module-level graph clean.
        _streaming = False
        if provider == "macos_say" and player is _default_player:
            from .audio_controller import split_sentences
            _streaming = len(split_sentences(text)) > 1

        if _streaming:
            from .audio_controller import AudioController
            from .providers.macos_say import make_synthesise_fn as _make_say_fn
            duration_s = AudioController.get().speak(text, _make_say_fn(voice=voice))
            if usage_counter is not None:
                usage_counter.record(
                    provider=provider, chars=len(text), profile=profile
                )
        else:
            synth = prov.synthesise(text=text, voice=voice, emotion=emotion)
            audio = synth.audio
            emotion_match = synth.emotion_match
            if cache is not None:
                cache.put(fp, audio, emotion_match=emotion_match)
            if usage_counter is not None:
                # Record chars only on cache-miss so cache hits don't
                # inflate the monthly quota total.
                usage_counter.record(
                    provider=provider, chars=len(text), profile=profile
                )

    if audio is not None:
        with tmpfs_audio_path(audio) as audio_path:
            duration_s = player(audio_path)

    audit_id = ""
    if audit_logger is not None:
        audit_id = audit_logger.append(
            event="voice.speak",
            payload={
                "provider": provider,
                "voice": voice,
                "emotion": emotion,
                "directive_fingerprint": fp,
                "emotion_match": emotion_match,
                "cache_hit": cache_hit,
                "cache_stored": (
                    not cache_hit
                    and cache is not None
                    and emotion_match >= CACHE_EMOTION_THRESHOLD
                ),
                "duration_s": duration_s,
                "profile": profile,
                "tier": tier,
                "provenance": provenance.value,
            },
        )

    return SpeakResult(
        ok=True,
        audio_path=None,
        duration_s=duration_s,
        provider=provider,
        voice=voice,
        profile=profile,
        emotion=emotion,
        directive_fingerprint=fp,
        tier=tier,
        audit_id=audit_id,
    )


def _speak_with_ladder(
    *,
    text: str,
    voice: str,
    emotion: str | None,
    ladder_name: str,
    profile: str,
    tier: int,
    cache: VoiceCache | None,
    audit_logger: AuditLogger | None,
    player: Player,
    provenance: Provenance,
    usage_counter: UsageCounter | None = None,
) -> SpeakResult:
    """Iterate the named TTS ladder, stopping on first success.

    Unregistered provider names are skipped with a warning
    (dispatch may be ahead of registry as providers ship). Retryable
    exceptions advance to the next rung. Non-retryable exceptions
    (caller errors) halt the ladder immediately so the operator sees
    the original message rather than a misleading "all providers
    failed" cascade.
    """
    try:
        ladder = get_ladder(ladder_name)
    except KeyError as exc:
        raise VoiceError(str(exc)) from exc

    tts_names = list(ladder.get("tts") or [])
    if not tts_names:
        raise VoiceError(
            f"ladder {ladder_name!r} has no TTS providers configured"
        )

    last_exc: BaseException | None = None
    for provider_name in tts_names:
        if provider_name not in PROVIDER_REGISTRY["tts"]:
            _log_fallover(
                audit_logger=audit_logger,
                ladder_name=ladder_name,
                provider_name=provider_name,
                reason="unregistered",
                profile=profile,
            )
            continue
        try:
            return _speak_with_provider(
                provider=provider_name,
                text=text,
                voice=voice,
                emotion=emotion,
                profile=profile,
                tier=tier,
                cache=cache,
                audit_logger=audit_logger,
                player=player,
                provenance=provenance,
                usage_counter=usage_counter,
            )
        except BaseException as exc:
            if not is_retryable(exc):
                raise
            last_exc = exc
            _log_fallover(
                audit_logger=audit_logger,
                ladder_name=ladder_name,
                provider_name=provider_name,
                reason=f"{type(exc).__name__}: {exc}",
                profile=profile,
            )

    if last_exc is not None:
        raise last_exc
    raise VoiceError(
        f"ladder {ladder_name!r} has no registered TTS providers; "
        f"names tried: {tts_names}"
    )


def _log_fallover(
    *,
    audit_logger: AuditLogger | None,
    ladder_name: str,
    provider_name: str,
    reason: str,
    profile: str,
) -> None:
    """Emit a ``voice.dispatch.fallover`` audit event + stderr line.

    Stderr suppressed by ``HAL_VOICE_QUIET=1`` for operators running
    scripted pipelines who don't want per-skip noise. Audit trail is
    unconditional — the chain is the canonical dispatch history.
    """
    if audit_logger is not None:
        audit_logger.append(
            event="voice.dispatch.fallover",
            payload={
                "ladder": ladder_name,
                "skipped_provider": provider_name,
                "reason": reason,
                "profile": profile,
            },
        )
    if not os.environ.get("HAL_VOICE_QUIET"):
        print(
            f"[voice.dispatch] ladder={ladder_name} skipping {provider_name!r}: {reason}",
            file=sys.stderr,
        )


# ---------- chat_loop ----------


def chat_loop(
    profile: str,
    policy: ChatPolicy,
    *,
    responder: Responder,
    session_id: str | None = None,
    cache: VoiceCache | None = None,
    audit_logger: AuditLogger | None = None,
    recorder: Recorder = _default_recorder,
    player: Player = _default_player,
) -> ChatTranscript:
    """Run a listen → respond → speak loop.

    Terminates on whichever of these fires first:

    1. ``policy.max_turns`` reached (when non-zero).
    2. ``policy.terminator(turn)`` returns True after a completed turn.
    3. The responder raises or the providers raise — the caller handles.
    """
    session_id = session_id or secrets.token_hex(8)
    started_at = _now_iso()

    listen_provider = policy.listen_provider or "macos_dictation"
    speak_provider = policy.speak_provider or "macos_say"

    turns: list[ChatTurn] = []
    turn_index = 0

    while True:
        if policy.max_turns and turn_index >= policy.max_turns:
            break

        t0 = time.monotonic()
        listen_res = listen(
            provider=listen_provider,
            profile=profile,
            recorder=recorder,
            audit_logger=audit_logger,
        )
        t1 = time.monotonic()

        response_text = responder(listen_res.text, list(turns))
        t2 = time.monotonic()

        speak_res = speak(
            text=response_text,
            provider=speak_provider,
            profile=profile,
            cache=cache,
            audit_logger=audit_logger,
            player=player,
        )
        t3 = time.monotonic()

        turn = ChatTurn(
            turn_index=turn_index,
            listen=listen_res,
            response_text=response_text,
            speak=speak_res,
            total_latency_s=t3 - t0,
            latency_breakdown={
                "stt": t1 - t0,
                "llm": t2 - t1,
                "tts": t3 - t2,
            },
        )
        turns.append(turn)
        turn_index += 1

        if policy.terminator is not None and policy.terminator(turn):
            break

    return ChatTranscript(
        turns=turns,
        profile=profile,
        session_id=session_id,
        started_at=started_at,
        ended_at=_now_iso(),
        total_cost_usd=0.0,
    )


# ---------- chat convenience ----------


def chat(
    profile: str,
    *,
    responder: Responder,
    **kwargs,
) -> ChatTranscript:
    """Run chat_loop with a default ChatPolicy."""
    return chat_loop(profile=profile, policy=ChatPolicy(), responder=responder, **kwargs)


__all__ = [
    "MAX_DURATION_HARD_CLAMP_S",
    "Player",
    "Recorder",
    "Responder",
    "TIER_CONFIDENCE_FLOOR",
    "chat",
    "chat_loop",
    "listen",
    "speak",
]
