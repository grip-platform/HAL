"""HAL Voice — Gemini 2.5 native audio provider (Tier 3 W3).

Per plans/voice-tier3-cloud-fallover.md §4 W3.

Gemini 2.5 Flash Native Audio Preview is the top cloud rung of the
FALLOVER_LADDER — the only provider in Tier 3 that supports emotion
conditioning at synthesis time. Drops to Google Cloud TTS → Piper →
``macos_say`` when RPM-saturated, rate-limited by the server, or the
network is unreachable.

Shape:
- Auth: ``x-goog-api-key`` header. Source order:
    - ``GEMINI_API_KEY`` env var
    - macOS keychain entry ``grip-gemini`` (reused — same entry the
      HAPPI Gemini chat adapter uses, so operators have one key to
      rotate)
    - Constructor ``api_key=`` override (for tests + injection)
- Endpoint: ``generativelanguage.googleapis.com/v1beta/models/<MODEL>:generateContent``
- Default model: ``gemini-2.5-flash-native-audio-preview``
  (preview model id can shift — surface it as a constant operators can
  override via env ``HAL_GEMINI_TTS_MODEL`` when Google renames it)
- Default voice: ``Iapetus`` (already in voices.yaml's Gemini namespace)
- ``emotion_match = 0.9`` on success (above Piper's 0.85, below Cloud
  TTS Neural2's 0.95 because native-audio preview quality is cloud-grade
  but less mature)

Emotion passthrough:
- Allowed emotions default to a 7-value set tuned for Jamie/Iapetus
  register (calm / focused / serious / warm / neutral / excited /
  concerned). Override via ``allowed_emotions=`` constructor kwarg —
  α-3c DirectiveResolver will pass a profile-specific set.
- Off-allowlist emotion → ``VoiceError`` (defence-in-depth; the
  DirectiveResolver at α-3c is the primary gate but silent downgrade
  on fallover would break the "emotion-intended" contract).
- Encoding: the emotion is injected into the text as a bracketed style
  prefix (``[calm] <text>``) rather than a structured API field. The
  preview schema is in flux; prompt-level encoding survives API drift
  and is how Google documents "styled" Gemini Live audio today.

RPM accounting (not char accounting — Gemini free tier caps by
requests-per-minute, not content length):
- State: ``~/.hal/state/voice/gemini_tts_rpm.jsonl`` (one record per
  successful reservation, ``{"ts": ISO8601, "ts_epoch": float}``)
- ``_reserve_rpm()`` does count-in-window + append under a single
  exclusive ``fcntl.flock`` so two concurrent processes cannot double-
  book the last few slots of the minute
- Reservation BEFORE the API call (same trade-off as W2 Google Cloud
  TTS): over-count on transport failure, never under-count on partial
  success

Falsifiable hypothesis H-VO-P3: Gemini 2.5 audio synthesis latency p50
≤ 1500ms for 50-char input. Deadline 2026-05-24.
"""
from __future__ import annotations

import base64
import binascii
import fcntl
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from ..types import VoiceError
from ._errors import AuthFailure, NetworkError, RateLimited
from .base import TtsSynthResult


# ---------- constants ----------


KEYCHAIN_SERVICE = "grip-gemini"
API_KEY_ENV = "GEMINI_API_KEY"
MODEL_ENV = "HAL_GEMINI_TTS_MODEL"
DEFAULT_MODEL = "gemini-2.5-flash-native-audio-preview"
ENDPOINT_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)
DEFAULT_VOICE = "Iapetus"
GEMINI_EMOTION_MATCH = 0.9
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_RPM_LIMIT = 15
DEFAULT_SAMPLE_RATE_HZ = 24000
DEFAULT_RPM_PATH = Path.home() / ".hal" / "state" / "voice" / "gemini_tts_rpm.jsonl"
DEFAULT_ALLOWED_EMOTIONS: frozenset[str] = frozenset({
    "calm", "focused", "serious", "warm",
    "neutral", "excited", "concerned",
})
_RPM_WINDOW_S = 60.0


# ---------- auth resolution ----------


def _fetch_keychain_token(service: str = KEYCHAIN_SERVICE) -> str | None:
    """Read the Gemini API key from the macOS keychain.

    Shares the ``grip-gemini`` service name with
    ``lib/hal/happi/providers/gemini.py`` so operators only rotate one
    keychain entry to switch keys across both chat and audio.
    """
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-w", "-s", service],
            capture_output=True, text=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _resolve_api_key() -> str | None:
    """Env first, keychain fallback."""
    return os.environ.get(API_KEY_ENV) or _fetch_keychain_token()


def _resolve_model() -> str:
    """Pick the model id at call time so preview rename can be picked up
    via env without recompiling."""
    return os.environ.get(MODEL_ENV) or DEFAULT_MODEL


# ---------- RPM accounting ----------


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _requests_in_window(
    rpm_path: Path,
    *,
    now_epoch: float | None = None,
    window_s: float = _RPM_WINDOW_S,
) -> int:
    """Count records whose ``ts_epoch`` falls within the last ``window_s``.

    Pure read under a shared lock — the authoritative check+reserve is
    ``_reserve_rpm`` under an exclusive lock.
    """
    if not rpm_path.exists():
        return 0
    cutoff = (now_epoch if now_epoch is not None else time.time()) - window_s
    total = 0
    with rpm_path.open("r", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        try:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts_epoch = rec.get("ts_epoch")
                if isinstance(ts_epoch, (int, float)) and ts_epoch >= cutoff:
                    total += 1
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return total


def _reserve_rpm(
    rpm_path: Path,
    *,
    limit: int = DEFAULT_RPM_LIMIT,
    now_epoch: float | None = None,
    window_s: float = _RPM_WINDOW_S,
) -> int:
    """Atomically count-in-window + append one record.

    Under a single exclusive flock: re-counts the sliding window, raises
    ``RateLimited`` if the new count would exceed ``limit``, else appends
    a record and returns the new count.

    Reservation is BEFORE the API call — if the call fails we keep the
    record (over-count on failure beats under-count on success because
    a server-side 429 caused by our double-book is costlier than a local
    reservation wasted on a transport error).
    """
    rpm_path.parent.mkdir(parents=True, exist_ok=True)
    now = now_epoch if now_epoch is not None else time.time()
    cutoff = now - window_s
    with rpm_path.open("a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            count = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts_epoch = rec.get("ts_epoch")
                if isinstance(ts_epoch, (int, float)) and ts_epoch >= cutoff:
                    count += 1
            if count + 1 > limit:
                raise RateLimited(
                    f"gemini_tts: local RPM limit reached "
                    f"({count}/{limit} in last {int(window_s)}s)"
                )
            record = {"ts": _now_ts(), "ts_epoch": float(now)}
            line_out = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            f.seek(0, os.SEEK_END)
            f.write(line_out)
            f.flush()
            os.fsync(f.fileno())
            return count + 1
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# ---------- emotion encoding ----------


def _encode_emotion(text: str, emotion: str | None) -> str:
    """Prefix the text with a bracketed style tag when emotion is set.

    Gemini's audio model interprets bracketed style hints at the start
    of a prompt as acting directions. This survives the preview-schema
    churn that a structured ``emotion`` field would be vulnerable to.
    """
    if not emotion:
        return text
    return f"[{emotion}] {text}"


def _validate_emotion(
    emotion: str | None,
    allowed: frozenset[str],
) -> None:
    """Raise ``VoiceError`` for off-allowlist emotions.

    ``None`` is always permitted (means "no emotion requested"). Any
    non-None string must be in ``allowed``. The empty string is treated
    as None (operator-friendly: passing ``emotion=""`` is the same as
    not passing emotion).
    """
    if not emotion:
        return
    if emotion not in allowed:
        raise VoiceError(
            f"gemini_tts: emotion {emotion!r} not in allowed set "
            f"{sorted(allowed)} (defence-in-depth; see α-3c DirectiveResolver)"
        )


# ---------- audio format helpers ----------


def _parse_sample_rate_from_mime(mime: str) -> int:
    """Extract ``rate=<N>`` from a mimeType like ``audio/L16;codec=pcm;rate=24000``.

    Returns ``DEFAULT_SAMPLE_RATE_HZ`` if the rate parameter is missing
    or unparseable.
    """
    for part in mime.split(";"):
        part = part.strip()
        if part.startswith("rate="):
            try:
                return int(part[len("rate="):])
            except ValueError:
                return DEFAULT_SAMPLE_RATE_HZ
    return DEFAULT_SAMPLE_RATE_HZ


def _is_pcm_mime(mime: str) -> bool:
    """True when the mimeType denotes raw LINEAR16 / PCM."""
    m = mime.lower()
    return "l16" in m or "pcm" in m


# Shared WAV helpers (consolidated at ``providers/_wav_utils.py`` —
# YSH threshold crossed at the 4th site). Re-exported with the
# private names so existing tests keep importing them from here.
from ._wav_utils import (  # noqa: E402
    pcm16_duration_s as _pcm16_duration_s,
    wrap_pcm16_as_wav as _wrap_pcm16_as_wav,
)


# ---------- provider ----------


class GeminiTtsProvider:
    """Gemini 2.5 Flash Native Audio Preview synthesiser with emotion
    passthrough.

    Satisfies the ``TtsProvider`` Protocol. Does not inherit.

    Unlike Piper (emotion-neutral) and Google Cloud TTS (no emotion
    channel), this provider honours the ``emotion`` kwarg via prompt-
    level style injection. An off-allowlist emotion raises ``VoiceError``
    before any network call — the ladder then treats this as the only
    emotion-capable rung falling over to an emotion-neutral one, which
    is a conscious UX degradation rather than a silent drop.
    """

    name = "gemini_tts_free"

    def __init__(
        self,
        *,
        rpm_path: Path | None = None,
        rpm_limit: int = DEFAULT_RPM_LIMIT,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        api_key: str | None = None,
        model: str | None = None,
        allowed_emotions: frozenset[str] | None = None,
    ) -> None:
        self._rpm_path = rpm_path or DEFAULT_RPM_PATH
        self._rpm_limit = rpm_limit
        self._timeout_s = timeout_s
        self._api_key_override = api_key
        self._model_override = model
        self._allowed_emotions = allowed_emotions or DEFAULT_ALLOWED_EMOTIONS

    def _resolve_key(self) -> str:
        key = self._api_key_override or _resolve_api_key()
        if not key:
            raise AuthFailure(
                "gemini_tts: missing API key "
                f"(set {API_KEY_ENV} env or keychain entry {KEYCHAIN_SERVICE!r})"
            )
        return key

    def _resolve_model_name(self) -> str:
        return self._model_override or _resolve_model()

    def _build_request(
        self,
        *,
        text_with_emotion: str,
        voice: str,
        api_key: str,
    ) -> urllib.request.Request:
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": text_with_emotion}],
                }
            ],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {"voiceName": voice}
                    }
                },
            },
        }
        url = ENDPOINT_TEMPLATE.format(model=self._resolve_model_name())
        return urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": api_key,
            },
        )

    def _decode_response(self, raw: bytes) -> tuple[bytes, str]:
        """Extract (audio_bytes, mime_type) from a generateContent body.

        Raises ``VoiceError`` on any schema deviation so the caller can
        fall over cleanly rather than replay garbage.
        """
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise VoiceError(
                "gemini_tts: malformed JSON response"
            ) from exc
        try:
            candidates = parsed["candidates"]
            parts = candidates[0]["content"]["parts"]
            inline = None
            for part in parts:
                if isinstance(part, dict) and "inlineData" in part:
                    inline = part["inlineData"]
                    break
            if inline is None:
                raise KeyError("inlineData")
            mime = str(inline.get("mimeType", "audio/L16;rate=24000"))
            data_b64 = inline["data"]
            if not isinstance(data_b64, str):
                raise TypeError("data must be str")
        except (KeyError, IndexError, TypeError) as exc:
            raise VoiceError(
                "gemini_tts: response missing candidates[0].content.parts[*].inlineData"
            ) from exc
        try:
            audio = base64.b64decode(data_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise VoiceError(
                "gemini_tts: inlineData.data is not valid base64"
            ) from exc
        return audio, mime

    def synthesise(
        self,
        *,
        text: str,
        voice: str,
        emotion: str | None,
    ) -> TtsSynthResult:
        _validate_emotion(emotion, self._allowed_emotions)
        resolved_voice = voice or DEFAULT_VOICE
        text_with_emotion = _encode_emotion(text, emotion)

        # Reserve RPM BEFORE the API call — single exclusive lock kills
        # the double-book race.
        _reserve_rpm(self._rpm_path, limit=self._rpm_limit)

        api_key = self._resolve_key()
        req = self._build_request(
            text_with_emotion=text_with_emotion,
            voice=resolved_voice,
            api_key=api_key,
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise AuthFailure(
                    f"gemini_tts: HTTP {exc.code} — check API key"
                ) from exc
            if exc.code == 429:
                raise RateLimited(
                    "gemini_tts: HTTP 429 — rate-limited by provider"
                ) from exc
            raise VoiceError(f"gemini_tts: HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise NetworkError(
                f"gemini_tts: network error — {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise NetworkError(
                f"gemini_tts: timed out after {self._timeout_s}s"
            ) from exc

        audio_bytes, mime = self._decode_response(raw)

        if _is_pcm_mime(mime):
            sample_rate = _parse_sample_rate_from_mime(mime)
            wav = _wrap_pcm16_as_wav(audio_bytes, sample_rate=sample_rate)
            duration_s = _pcm16_duration_s(audio_bytes, sample_rate=sample_rate)
        elif audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
            wav = audio_bytes  # already WAV
            # Best-effort duration from the sample rate in the fmt chunk.
            duration_s = _wav_duration_best_effort(audio_bytes)
        else:
            # Unknown container (mp3 / ogg / etc). Pass through; duration
            # unknown. The cache + player handle non-WAV via afplay which
            # auto-detects format.
            wav = audio_bytes
            duration_s = 0.0

        return TtsSynthResult(
            audio=wav,
            duration_s=duration_s,
            voice=resolved_voice,
            provider_name=self.name,
            emotion_match=GEMINI_EMOTION_MATCH,
        )


def _wav_duration_best_effort(wav_bytes: bytes) -> float:
    """Parse duration from a canonical 44-byte-header WAV, else 0.0.

    Thin wrapper around ``_wav_utils.parse_wav_duration_s`` that swallows
    the ``VoiceError`` the shared helper raises on non-RIFF input. Unlike
    the L16/pcm path where a malformed response should fallover, the
    audio/wav branch MAY receive an unexpected container that we still
    want to pass through with duration 0.0 (afplay handles format
    auto-detection)."""
    from ._wav_utils import parse_wav_duration_s
    try:
        return parse_wav_duration_s(wav_bytes)
    except VoiceError:
        return 0.0


__all__ = [
    "API_KEY_ENV",
    "AuthFailure",
    "DEFAULT_ALLOWED_EMOTIONS",
    "DEFAULT_MODEL",
    "DEFAULT_RPM_LIMIT",
    "DEFAULT_RPM_PATH",
    "DEFAULT_TIMEOUT_S",
    "DEFAULT_VOICE",
    "ENDPOINT_TEMPLATE",
    "GEMINI_EMOTION_MATCH",
    "GeminiTtsProvider",
    "KEYCHAIN_SERVICE",
    "MODEL_ENV",
    "NetworkError",
    "RateLimited",
]
