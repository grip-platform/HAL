"""HAL Voice — Google Cloud TTS free-tier provider (Tier 3 W2).

Per plans/voice-tier3-cloud-fallover.md §4 W2.

Google Cloud TTS Neural2 is the first cloud rung of the FALLOVER_LADDER:
cloud-neural quality ($0 within the 1M chars/month Neural2 free tier),
drops to Piper → macos_say when quota exhausted or network unreachable.

Shape:
- Auth: ``x-goog-api-key`` header. Source order:
    - ``GOOGLE_CLOUD_TTS_API_KEY`` env var
    - macOS keychain entry ``grip-google-cloud-tts``
    - Constructor ``api_key=`` override (for tests + injection)
- Endpoint: ``https://texttospeech.googleapis.com/v1/text:synthesize``
- Default voice: ``en-GB-Neural2-B`` (male, matches Jamie's register)
- Default language code derived from the voice name (``en-GB`` from prefix)
- Audio encoding: ``LINEAR16`` (raw 16-bit PCM). The provider WAV-wraps
  the response so cache/player see the same shape as Piper output.
- ``emotion_match = 0.95`` (cloud-neural tier, above Piper's 0.85)

Quota accounting:
- Persists to ``~/.hal/state/voice/google_cloud_tts_quota.jsonl``
- Per-record shape: ``{"ts": ISO8601, "ym": "YYYY-MM", "chars": int}``
- ``_reserve_chars()`` does check-AND-record under a single exclusive
  portable advisory lock so two concurrent processes cannot double-book the last
  few characters of the monthly allotment.
- Reservations are made BEFORE the API call: on network or auth failure
  the chars stay counted (over-conservative, avoids the commit-after-call
  race where a crash between call and record silently burns quota).

Exception hierarchy (all inherit ``VoiceError`` so the W4 ladder
``is_retryable()`` helper can classify without knowing provider internals):
- ``AuthFailure``: missing key, HTTP 401/403 — no retry at this provider
- ``RateLimited``: HTTP 429 — ladder may retry at next rung
- ``QuotaExceeded``: local counter caught the limit before the call
- ``NetworkError``: URLError / timeout — retryable at next rung

Falsifiable hypothesis H-VO-P2: the 1M Neural2 chars/mo free tier suffices
for V>>'s median month of spoken output. Metric: ``max(chars_per_month) <
700_000`` over a 30-day window. Deadline 2026-06-01.
"""
from __future__ import annotations

import base64
import binascii
import json
import os
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from lib.hal.file_lock import lock, unlock

from ..types import VoiceError
from ._errors import AuthFailure, NetworkError, QuotaExceeded, RateLimited
from .base import TtsSynthResult


# ---------- constants ----------


KEYCHAIN_SERVICE = "grip-google-cloud-tts"
API_KEY_ENV = "GOOGLE_CLOUD_TTS_API_KEY"
API_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"
DEFAULT_VOICE = "en-GB-Neural2-B"
FREE_TIER_MONTHLY_CHARS = 1_000_000
CLOUD_EMOTION_MATCH = 0.95
DEFAULT_TIMEOUT_S = 15.0
DEFAULT_SAMPLE_RATE_HZ = 24000  # Neural2 voices ship at 24kHz
DEFAULT_QUOTA_PATH = Path.home() / ".hal" / "state" / "voice" / "google_cloud_tts_quota.jsonl"


# ---------- auth resolution ----------


def _fetch_keychain_token(service: str = KEYCHAIN_SERVICE) -> str | None:
    """Read an API key from the macOS keychain.

    Mirrors the pattern in ``lib/hal/happi/providers/gemini.py``. Silently
    returns None on every failure path (timeout, missing binary, exit != 0)
    so the caller can escalate to other sources without exception handling.
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
    """Env var first, keychain as fallback."""
    return os.environ.get(API_KEY_ENV) or _fetch_keychain_token()


# ---------- quota accounting ----------


def _now_ym() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _chars_this_month(quota_path: Path) -> int:
    """Sum chars written to quota file within the current UTC month.

    Uses a shared lock so concurrent writers don't corrupt our read, but
    the number returned is a SNAPSHOT — the canonical check+reserve is
    done under an exclusive lock in ``_reserve_chars``.
    """
    if not quota_path.exists():
        return 0
    ym = _now_ym()
    total = 0
    with quota_path.open("r", encoding="utf-8") as f:
        lock(f, shared=True)
        try:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("ym") == ym:
                    total += int(record.get("chars", 0))
        finally:
            unlock(f)
    return total


def _reserve_chars(
    quota_path: Path,
    chars: int,
    *,
    limit: int = FREE_TIER_MONTHLY_CHARS,
) -> int:
    """Atomically check + record a reservation.

    Under a single exclusive flock: reads the current month's total,
    raises ``QuotaExceeded`` if the new total would exceed ``limit``,
    else appends a reservation record and returns the new total.

    Reservation is BEFORE the API call — if the call fails, the chars
    stay counted. Over-counting by failed calls is acceptable; under-
    counting by silent races would burn the operator's real quota.
    """
    quota_path.parent.mkdir(parents=True, exist_ok=True)
    ym = _now_ym()
    with quota_path.open("a+", encoding="utf-8") as f:
        lock(f)
        try:
            f.seek(0)
            total = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("ym") == ym:
                    total += int(rec.get("chars", 0))
            if total + chars > limit:
                raise QuotaExceeded(
                    f"google_cloud_tts: monthly free-tier exceeded "
                    f"({total} + {chars} > {limit})"
                )
            record = {"ts": _now_ts(), "ym": ym, "chars": int(chars)}
            line_out = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            f.seek(0, os.SEEK_END)
            f.write(line_out)
            f.flush()
            os.fsync(f.fileno())
            return total + chars
        finally:
            unlock(f)


# ---------- audio format helpers ----------


def _language_code_from_voice(voice: str) -> str:
    """Extract the BCP-47 language tag prefix from a voice name.

    ``en-GB-Neural2-B`` → ``en-GB``. If the voice doesn't follow the
    standard prefix pattern we fall back to ``en-GB``.
    """
    parts = voice.split("-")
    if len(parts) >= 2 and parts[0].isalpha() and parts[1].isalpha():
        return f"{parts[0]}-{parts[1]}"
    return "en-GB"


# PCM/WAV helpers are re-exported from ``._wav_utils`` so provider-
# local tests that imported the private names continue to resolve.
from ._wav_utils import (  # noqa: E402
    pcm16_duration_s as _pcm16_duration_s_shared,
    wrap_pcm16_as_wav as _wrap_pcm16_as_wav_shared,
)


def _pcm16_duration_s(
    pcm_bytes: bytes,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE_HZ,
    channels: int = 1,
) -> float:
    return _pcm16_duration_s_shared(
        pcm_bytes, sample_rate=sample_rate, channels=channels
    )


def _wrap_pcm16_as_wav(
    pcm_bytes: bytes,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE_HZ,
    channels: int = 1,
) -> bytes:
    return _wrap_pcm16_as_wav_shared(
        pcm_bytes, sample_rate=sample_rate, channels=channels
    )


# ---------- provider ----------


class GoogleCloudTtsProvider:
    """Google Cloud TTS Neural2 free-tier synthesiser.

    Satisfies the ``TtsProvider`` Protocol. Does not inherit.

    ``emotion`` is accepted for Protocol conformance and ignored: Google
    Cloud TTS v1 has no emotion-conditioning channel. The DirectiveResolver
    at α-3c is the primary emotion gate; this provider is intentionally
    emotion-neutral so the ladder's defence in depth doesn't silently
    drop emotion on fallover.
    """

    name = "google_cloud_tts_free"

    def __init__(
        self,
        *,
        quota_path: Path | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        api_key: str | None = None,
        free_tier_monthly_chars: int = FREE_TIER_MONTHLY_CHARS,
    ) -> None:
        self._quota_path = quota_path or DEFAULT_QUOTA_PATH
        self._timeout_s = timeout_s
        self._api_key_override = api_key
        self._monthly_limit = free_tier_monthly_chars

    def _resolve_key(self) -> str:
        key = self._api_key_override or _resolve_api_key()
        if not key:
            raise AuthFailure(
                "google_cloud_tts: missing API key "
                "(set GOOGLE_CLOUD_TTS_API_KEY env or keychain entry "
                f"{KEYCHAIN_SERVICE!r})"
            )
        return key

    def _build_request(self, *, text: str, voice: str, api_key: str) -> urllib.request.Request:
        payload = {
            "input": {"text": text},
            "voice": {
                "languageCode": _language_code_from_voice(voice),
                "name": voice,
            },
            "audioConfig": {"audioEncoding": "LINEAR16"},
        }
        return urllib.request.Request(
            url=API_URL,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": api_key,
            },
        )

    def synthesise(
        self,
        *,
        text: str,
        voice: str,
        emotion: str | None,
    ) -> TtsSynthResult:
        del emotion  # Protocol-compliant; Google Cloud TTS v1 has no channel
        resolved_voice = voice or DEFAULT_VOICE

        # Reserve quota BEFORE the API call — single exclusive lock kills
        # the double-book race between concurrent processes.
        _reserve_chars(
            self._quota_path, len(text), limit=self._monthly_limit
        )

        api_key = self._resolve_key()
        req = self._build_request(text=text, voice=resolved_voice, api_key=api_key)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise AuthFailure(
                    f"google_cloud_tts: HTTP {exc.code} — check API key"
                ) from exc
            if exc.code == 429:
                raise RateLimited(
                    "google_cloud_tts: HTTP 429 — rate-limited by provider"
                ) from exc
            raise VoiceError(
                f"google_cloud_tts: HTTP {exc.code}"
            ) from exc
        except urllib.error.URLError as exc:
            raise NetworkError(
                f"google_cloud_tts: network error — {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise NetworkError(
                f"google_cloud_tts: timed out after {self._timeout_s}s"
            ) from exc

        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise VoiceError(
                "google_cloud_tts: malformed JSON response"
            ) from exc

        audio_b64 = parsed.get("audioContent")
        if not audio_b64 or not isinstance(audio_b64, str):
            raise VoiceError(
                "google_cloud_tts: response missing audioContent"
            )

        try:
            pcm = base64.b64decode(audio_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise VoiceError(
                "google_cloud_tts: audioContent is not valid base64"
            ) from exc

        wav = _wrap_pcm16_as_wav(pcm, sample_rate=DEFAULT_SAMPLE_RATE_HZ)
        duration_s = _pcm16_duration_s(pcm, sample_rate=DEFAULT_SAMPLE_RATE_HZ)

        return TtsSynthResult(
            audio=wav,
            duration_s=duration_s,
            voice=resolved_voice,
            provider_name=self.name,
            emotion_match=CLOUD_EMOTION_MATCH,
        )


__all__ = [
    "API_KEY_ENV",
    "API_URL",
    "AuthFailure",
    "CLOUD_EMOTION_MATCH",
    "DEFAULT_QUOTA_PATH",
    "DEFAULT_SAMPLE_RATE_HZ",
    "DEFAULT_TIMEOUT_S",
    "DEFAULT_VOICE",
    "FREE_TIER_MONTHLY_CHARS",
    "GoogleCloudTtsProvider",
    "KEYCHAIN_SERVICE",
    "NetworkError",
    "QuotaExceeded",
    "RateLimited",
]
