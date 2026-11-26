"""Two-sided Goodhart-proof tests for lib.hal.voice.providers.google_cloud_tts.

Per Rule 14 each invariant has a must-pass branch (correct shape accepted,
correct output produced) AND a must-fail branch (broken precondition
rejected). Without the must-fail branches, a regression that swallowed an
AuthFailure or quietly skipped the quota check would silently pass CI.

Mocking strategy:
- ``urllib.request.urlopen`` is patched with a context manager that yields
  a bytes body and a status code — no real HTTPS call made
- ``subprocess.run`` is patched for keychain fallback tests
- ``quota_path`` is injected into each provider instance (tmp_path fixture)
  so tests never touch the operator's real ``~/.hal/state/voice/``
"""
from __future__ import annotations

import base64
import io
import json
import os
import struct
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.hal.voice.providers import google_cloud_tts as gct  # noqa: E402
from lib.hal.voice.providers.base import TtsSynthResult  # noqa: E402
from lib.hal.voice.providers import SttProvider, TtsProvider  # noqa: E402
from lib.hal.voice.providers.google_cloud_tts import (  # noqa: E402
    API_KEY_ENV,
    API_URL,
    AuthFailure,
    CLOUD_EMOTION_MATCH,
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_VOICE,
    FREE_TIER_MONTHLY_CHARS,
    GoogleCloudTtsProvider,
    KEYCHAIN_SERVICE,
    NetworkError,
    QuotaExceeded,
    RateLimited,
    _chars_this_month,
    _fetch_keychain_token,
    _language_code_from_voice,
    _pcm16_duration_s,
    _reserve_chars,
    _resolve_api_key,
    _wrap_pcm16_as_wav,
)
from lib.hal.voice.types import VoiceError  # noqa: E402


# ---------- helpers ----------


def _make_response(audio_bytes: bytes) -> bytes:
    """Build the JSON body the Google Cloud TTS endpoint returns."""
    return json.dumps(
        {"audioContent": base64.b64encode(audio_bytes).decode("ascii")}
    ).encode("utf-8")


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self._body = body
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _fake_urlopen(body: bytes):
    def _impl(req, **kwargs):
        _ = req, kwargs
        return _FakeResponse(body)

    return _impl


def _raising_urlopen(exc: BaseException):
    def _impl(req, **kwargs):
        _ = req, kwargs
        raise exc

    return _impl


@pytest.fixture
def quota_file(tmp_path: Path) -> Path:
    return tmp_path / "quota.jsonl"


@pytest.fixture
def provider(quota_file: Path) -> GoogleCloudTtsProvider:
    return GoogleCloudTtsProvider(
        quota_path=quota_file,
        api_key="test-key",  # explicit key so tests don't touch env/keychain
    )


# ---------- exception hierarchy ----------


def test_auth_failure_is_voice_error() -> None:
    """Must-pass: W4 ladder catches VoiceError as the retryability root type."""
    assert issubclass(AuthFailure, VoiceError)


def test_rate_limited_is_voice_error() -> None:
    assert issubclass(RateLimited, VoiceError)


def test_quota_exceeded_is_voice_error() -> None:
    assert issubclass(QuotaExceeded, VoiceError)


def test_network_error_is_voice_error() -> None:
    assert issubclass(NetworkError, VoiceError)


def test_subclasses_are_distinct_types() -> None:
    """Must-fail (type-confusion): the subclasses are NOT interchangeable —
    ladder dispatch classifies each differently."""
    assert not issubclass(AuthFailure, RateLimited)
    assert not issubclass(RateLimited, AuthFailure)
    assert not issubclass(QuotaExceeded, NetworkError)


# ---------- Protocol conformance ----------


def test_provider_satisfies_tts_protocol(provider: GoogleCloudTtsProvider) -> None:
    """Must-pass: runtime_checkable Protocol accepts our shape."""
    assert isinstance(provider, TtsProvider)


def test_provider_is_not_an_stt_provider(provider: GoogleCloudTtsProvider) -> None:
    """Must-fail: slot-safety — no transcribe() method, registry rejects."""
    assert not isinstance(provider, SttProvider)


def test_provider_name_is_google_cloud_tts_free(
    provider: GoogleCloudTtsProvider,
) -> None:
    """Must-pass: registry-key discipline — FALLOVER_LADDER looks this up."""
    assert provider.name == "google_cloud_tts_free"


# ---------- API key resolution ----------


def test_missing_key_raises_auth_failure(quota_file: Path) -> None:
    """Must-fail: no env var, keychain returns None → AuthFailure with
    actionable message. Synth never reaches urllib."""
    p = GoogleCloudTtsProvider(quota_path=quota_file)
    env = dict(os.environ)
    env.pop(API_KEY_ENV, None)
    with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(
        gct, "_fetch_keychain_token", return_value=None
    ):
        with pytest.raises(AuthFailure, match="missing API key"):
            p.synthesise(text="hello", voice=DEFAULT_VOICE, emotion=None)


def test_env_var_api_key_used(quota_file: Path) -> None:
    """Must-pass: env var wins when present."""
    p = GoogleCloudTtsProvider(quota_path=quota_file)
    with mock.patch.dict(os.environ, {API_KEY_ENV: "env-key"}):
        assert _resolve_api_key() == "env-key"


def test_keychain_fallback_used_when_env_missing(quota_file: Path) -> None:
    """Must-pass: keychain subprocess succeeds → returned as the key."""
    env = dict(os.environ)
    env.pop(API_KEY_ENV, None)
    fake_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="keychain-key\n", stderr=""
    )
    with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(
        gct.subprocess, "run", return_value=fake_result
    ):
        assert _resolve_api_key() == "keychain-key"


def test_keychain_timeout_returns_none() -> None:
    """Must-fail (graceful): subprocess timeout → keychain helper returns
    None instead of raising. Caller escalates without exception handling."""
    with mock.patch.object(
        gct.subprocess,
        "run",
        side_effect=subprocess.TimeoutExpired(cmd="security", timeout=5),
    ):
        assert _fetch_keychain_token() is None


def test_keychain_missing_binary_returns_none() -> None:
    """Must-fail (graceful): security binary absent → None, not crash."""
    with mock.patch.object(
        gct.subprocess, "run", side_effect=FileNotFoundError()
    ):
        assert _fetch_keychain_token() is None


def test_constructor_api_key_beats_env(quota_file: Path) -> None:
    """Must-pass: constructor override used for deterministic tests + injection."""
    p = GoogleCloudTtsProvider(quota_path=quota_file, api_key="ctor-key")
    with mock.patch.dict(os.environ, {API_KEY_ENV: "env-should-lose"}):
        # Call _resolve_key directly via synthesise's path — but avoid urllib
        # by asserting no AuthFailure raises (would raise if key was empty).
        req = p._build_request(text="x", voice=DEFAULT_VOICE, api_key=p._resolve_key())
    assert req.headers["X-goog-api-key"] == "ctor-key"


# ---------- quota accounting ----------


def test_chars_this_month_is_zero_when_file_absent(quota_file: Path) -> None:
    """Must-pass edge: fresh operator → 0 chars used, no file on disk."""
    assert _chars_this_month(quota_file) == 0


def test_chars_this_month_ignores_other_months(quota_file: Path) -> None:
    """Must-pass: records from a different YYYY-MM are NOT counted toward
    the current month's quota — month rollover must reset the budget."""
    quota_file.parent.mkdir(parents=True, exist_ok=True)
    prior_month_line = json.dumps(
        {"ts": "2026-01-15T00:00:00Z", "ym": "2026-01", "chars": 900_000},
        sort_keys=True,
        separators=(",", ":"),
    )
    quota_file.write_text(prior_month_line + "\n")
    assert _chars_this_month(quota_file) == 0


def test_reserve_chars_increments_total(quota_file: Path) -> None:
    """Must-pass: two reservations sum correctly."""
    assert _reserve_chars(quota_file, 100) == 100
    assert _reserve_chars(quota_file, 250) == 350


def test_reserve_chars_raises_at_boundary(quota_file: Path) -> None:
    """Must-fail: reservation that crosses the limit raises and does NOT
    write a record — quota stays clean on rejection."""
    _reserve_chars(quota_file, 999_900)
    with pytest.raises(QuotaExceeded, match="monthly free-tier exceeded"):
        _reserve_chars(quota_file, 101)
    # The failed reservation must NOT have been recorded.
    assert _chars_this_month(quota_file) == 999_900


def test_reserve_chars_sequential_race_simulation(quota_file: Path) -> None:
    """Must-fail race prevention: two back-to-back reservations with the same
    pre-read view — the exclusive flock serialises them so the second sees
    the first's write. Simulates the race a naive check-then-write would lose."""
    _reserve_chars(quota_file, 500_000, limit=1_000_000)
    _reserve_chars(quota_file, 400_000, limit=1_000_000)
    with pytest.raises(QuotaExceeded):
        _reserve_chars(quota_file, 200_000, limit=1_000_000)


def test_reserve_chars_malformed_line_tolerated(quota_file: Path) -> None:
    """Must-pass: a partially-written line (JSON decode error) is skipped
    rather than breaking the counter. Operator would rather under-count a
    corrupted line than lose the whole quota file."""
    quota_file.parent.mkdir(parents=True, exist_ok=True)
    quota_file.write_text("not-json\n")
    assert _reserve_chars(quota_file, 100) == 100


def test_synthesise_rejects_on_quota_before_api_call(
    provider: GoogleCloudTtsProvider, quota_file: Path
) -> None:
    """Must-fail: quota exhausted → synth raises QuotaExceeded WITHOUT
    hitting urllib. Proven by patching urlopen to raise loudly if called."""
    _reserve_chars(quota_file, FREE_TIER_MONTHLY_CHARS)
    called = []

    def _should_not_be_called(req, **kwargs):
        called.append(req)
        raise AssertionError("urlopen must not be invoked when quota exhausted")

    with mock.patch.object(
        gct.urllib.request, "urlopen", side_effect=_should_not_be_called
    ):
        with pytest.raises(QuotaExceeded):
            provider.synthesise(text="hi", voice=DEFAULT_VOICE, emotion=None)
    assert called == []


# ---------- HTTPS happy path ----------


def test_synthesise_returns_tts_synth_result(
    provider: GoogleCloudTtsProvider,
) -> None:
    """Must-pass: happy path produces a WAV-wrapped TtsSynthResult."""
    pcm = b"\x00\x00" * 4800  # 0.1s at 24000Hz mono 16-bit
    body = _make_response(pcm)
    with mock.patch.object(
        gct.urllib.request, "urlopen", side_effect=_fake_urlopen(body)
    ):
        r = provider.synthesise(text="hello", voice=DEFAULT_VOICE, emotion=None)
    assert isinstance(r, TtsSynthResult)
    assert r.audio.startswith(b"RIFF")
    assert r.audio[8:12] == b"WAVE"
    assert r.voice == DEFAULT_VOICE
    assert r.provider_name == "google_cloud_tts_free"


def test_synthesise_emotion_match_is_cloud_constant(
    provider: GoogleCloudTtsProvider,
) -> None:
    """Must-pass: exact 0.95 — cloud-neural tier. Drift here would break
    ladder ordering against Piper (0.85) and the cache threshold (0.8)."""
    body = _make_response(b"\x00\x00" * 100)
    with mock.patch.object(
        gct.urllib.request, "urlopen", side_effect=_fake_urlopen(body)
    ):
        r = provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion="calm")
    assert r.emotion_match == CLOUD_EMOTION_MATCH == 0.95


def test_synthesise_ignores_emotion_value(
    provider: GoogleCloudTtsProvider, quota_file: Path
) -> None:
    """Must-pass: Google Cloud TTS v1 has no emotion channel. Three calls
    with different emotions yield the same emotion_match."""
    body = _make_response(b"\x00\x00" * 100)
    p = GoogleCloudTtsProvider(
        quota_path=quota_file, api_key="k", free_tier_monthly_chars=10**9
    )
    with mock.patch.object(
        gct.urllib.request, "urlopen", side_effect=_fake_urlopen(body)
    ):
        a = p.synthesise(text="x", voice=DEFAULT_VOICE, emotion="calm")
        b = p.synthesise(text="y", voice=DEFAULT_VOICE, emotion="angry")
        c = p.synthesise(text="z", voice=DEFAULT_VOICE, emotion=None)
    assert a.emotion_match == b.emotion_match == c.emotion_match


def test_synthesise_empty_voice_falls_back_to_default(
    provider: GoogleCloudTtsProvider,
) -> None:
    """Must-pass: empty voice string → DEFAULT_VOICE rather than a 400 from
    the API. Ladder callers don't always supply an explicit voice."""
    body = _make_response(b"\x00\x00" * 100)
    with mock.patch.object(
        gct.urllib.request, "urlopen", side_effect=_fake_urlopen(body)
    ):
        r = provider.synthesise(text="x", voice="", emotion=None)
    assert r.voice == DEFAULT_VOICE


def test_synthesise_records_chars_under_quota(
    provider: GoogleCloudTtsProvider, quota_file: Path
) -> None:
    """Must-pass: successful synth increments the monthly counter."""
    body = _make_response(b"\x00\x00" * 100)
    with mock.patch.object(
        gct.urllib.request, "urlopen", side_effect=_fake_urlopen(body)
    ):
        provider.synthesise(text="12345", voice=DEFAULT_VOICE, emotion=None)
    assert _chars_this_month(quota_file) == 5


# ---------- HTTP error branches ----------


def _http_error(code: int, reason: str = "boom") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url=API_URL, code=code, msg=reason, hdrs=None, fp=io.BytesIO(b"")
    )


def test_http_401_raises_auth_failure(
    provider: GoogleCloudTtsProvider,
) -> None:
    """Must-fail: HTTP 401 surfaces as AuthFailure (not-retryable)."""
    with mock.patch.object(
        gct.urllib.request,
        "urlopen",
        side_effect=_raising_urlopen(_http_error(401, "Unauthorized")),
    ):
        with pytest.raises(AuthFailure, match="HTTP 401"):
            provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)


def test_http_403_raises_auth_failure(
    provider: GoogleCloudTtsProvider,
) -> None:
    """Must-fail: HTTP 403 also → AuthFailure (same actionable path)."""
    with mock.patch.object(
        gct.urllib.request,
        "urlopen",
        side_effect=_raising_urlopen(_http_error(403, "Forbidden")),
    ):
        with pytest.raises(AuthFailure, match="HTTP 403"):
            provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)


def test_http_429_raises_rate_limited(
    provider: GoogleCloudTtsProvider,
) -> None:
    """Must-fail: HTTP 429 → RateLimited, distinct type from AuthFailure
    so ladder dispatch can retry at next rung."""
    with mock.patch.object(
        gct.urllib.request,
        "urlopen",
        side_effect=_raising_urlopen(_http_error(429, "Too Many Requests")),
    ):
        with pytest.raises(RateLimited, match="HTTP 429"):
            provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)


def test_http_500_raises_voice_error_not_subclass(
    provider: GoogleCloudTtsProvider,
) -> None:
    """Must-fail: unmapped 5xx → bare VoiceError (no misclassification as
    AuthFailure / RateLimited)."""
    with mock.patch.object(
        gct.urllib.request,
        "urlopen",
        side_effect=_raising_urlopen(_http_error(500, "Internal Server Error")),
    ):
        with pytest.raises(VoiceError) as excinfo:
            provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)
    assert not isinstance(excinfo.value, (AuthFailure, RateLimited))


def test_url_error_raises_network_error(
    provider: GoogleCloudTtsProvider,
) -> None:
    """Must-fail: DNS / socket-level failure → NetworkError."""
    with mock.patch.object(
        gct.urllib.request,
        "urlopen",
        side_effect=_raising_urlopen(urllib.error.URLError("name resolution failed")),
    ):
        with pytest.raises(NetworkError, match="name resolution failed"):
            provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)


def test_timeout_error_raises_network_error(
    provider: GoogleCloudTtsProvider,
) -> None:
    """Must-fail: urllib timeout surfaces as TimeoutError → NetworkError."""
    with mock.patch.object(
        gct.urllib.request,
        "urlopen",
        side_effect=_raising_urlopen(TimeoutError("timed out")),
    ):
        with pytest.raises(NetworkError, match="timed out"):
            provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)


# ---------- body-parsing error branches ----------


def test_malformed_json_raises_voice_error(
    provider: GoogleCloudTtsProvider,
) -> None:
    """Must-fail: non-JSON body → VoiceError, not a silent empty synth."""
    with mock.patch.object(
        gct.urllib.request,
        "urlopen",
        side_effect=_fake_urlopen(b"not-json"),
    ):
        with pytest.raises(VoiceError, match="malformed JSON"):
            provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)


def test_missing_audio_content_raises_voice_error(
    provider: GoogleCloudTtsProvider,
) -> None:
    """Must-fail: schema drift in the API response → VoiceError, caller
    knows to fallover rather than replay garbage."""
    body = json.dumps({"unexpected": "shape"}).encode("utf-8")
    with mock.patch.object(
        gct.urllib.request, "urlopen", side_effect=_fake_urlopen(body)
    ):
        with pytest.raises(VoiceError, match="missing audioContent"):
            provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)


def test_invalid_base64_raises_voice_error(
    provider: GoogleCloudTtsProvider,
) -> None:
    """Must-fail: audioContent present but not valid base64 → VoiceError."""
    body = json.dumps({"audioContent": "!!!not-base64!!!"}).encode("utf-8")
    with mock.patch.object(
        gct.urllib.request, "urlopen", side_effect=_fake_urlopen(body)
    ):
        with pytest.raises(VoiceError, match="valid base64"):
            provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)


# ---------- audio helpers ----------


def test_wrap_pcm16_produces_canonical_riff_wave() -> None:
    """Must-pass: 44-byte header with RIFF/WAVE markers + fmt chunk."""
    pcm = b"\x00\x00" * 1000
    wav = _wrap_pcm16_as_wav(pcm, sample_rate=24000, channels=1)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert wav[12:16] == b"fmt "
    # Body == 44-byte header + PCM
    assert len(wav) == 44 + len(pcm)
    # Sample-rate roundtrips through offset 24
    assert struct.unpack_from("<I", wav, 24)[0] == 24000


def test_pcm16_duration_s_computed_from_bytes() -> None:
    """Must-pass: 24000 samples × 2 bytes mono = 1 second of PCM."""
    pcm = b"\x00\x00" * 24000
    assert abs(_pcm16_duration_s(pcm, sample_rate=24000) - 1.0) < 1e-9


def test_pcm16_duration_s_scales_with_sample_rate() -> None:
    """Must-pass: same byte count, half sample rate → double duration."""
    pcm = b"\x00\x00" * 12000
    assert abs(_pcm16_duration_s(pcm, sample_rate=12000) - 1.0) < 1e-9
    assert abs(_pcm16_duration_s(pcm, sample_rate=24000) - 0.5) < 1e-9


def test_language_code_from_voice_standard() -> None:
    """Must-pass: en-GB-Neural2-B → en-GB."""
    assert _language_code_from_voice("en-GB-Neural2-B") == "en-GB"


def test_language_code_from_voice_fallback() -> None:
    """Must-pass: malformed voice → en-GB fallback, no crash."""
    assert _language_code_from_voice("garbled") == "en-GB"
    assert _language_code_from_voice("") == "en-GB"


def test_language_code_from_voice_other_locales() -> None:
    """Must-pass: handles locales beyond en-GB (future voice expansion)."""
    assert _language_code_from_voice("af-ZA-Standard-A") == "af-ZA"
    assert _language_code_from_voice("fr-FR-Neural2-D") == "fr-FR"


# ---------- request building ----------


def test_build_request_has_api_key_header(
    provider: GoogleCloudTtsProvider,
) -> None:
    """Must-pass: API key lands in X-Goog-Api-Key header, not in body
    or URL query string."""
    req = provider._build_request(
        text="hi", voice=DEFAULT_VOICE, api_key="header-key"
    )
    assert req.headers["X-goog-api-key"] == "header-key"
    assert "header-key" not in req.full_url
    body = req.data
    assert isinstance(body, bytes)
    assert b"header-key" not in body


def test_build_request_payload_shape(
    provider: GoogleCloudTtsProvider,
) -> None:
    """Must-pass: payload matches the v1 schema exactly (input, voice,
    audioConfig) — schema drift breaks the API silently."""
    req = provider._build_request(
        text="hello", voice=DEFAULT_VOICE, api_key="k"
    )
    assert req.full_url == API_URL
    body = json.loads(req.data.decode("utf-8"))
    assert body == {
        "input": {"text": "hello"},
        "voice": {"languageCode": "en-GB", "name": DEFAULT_VOICE},
        "audioConfig": {"audioEncoding": "LINEAR16"},
    }


# ---------- full round-trip invariant ----------


def test_synthesise_audio_duration_round_trip(
    provider: GoogleCloudTtsProvider,
) -> None:
    """Must-pass: duration_s reflects the PCM payload, not the WAV byte
    count. Protects against a regression where we accidentally reported
    (len(wav_bytes) / rate) — off by 44 bytes."""
    pcm = b"\x00\x00" * 24000  # exactly 1 second at 24kHz mono
    body = _make_response(pcm)
    with mock.patch.object(
        gct.urllib.request, "urlopen", side_effect=_fake_urlopen(body)
    ):
        r = provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)
    assert abs(r.duration_s - 1.0) < 1e-9
    # And the WAV-wrapped audio has exactly PCM + 44-byte header
    assert len(r.audio) == len(pcm) + 44
