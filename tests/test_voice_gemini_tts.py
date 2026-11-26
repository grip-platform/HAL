"""Two-sided Goodhart-proof tests for lib.hal.voice.providers.gemini_tts.

Per Rule 14 each invariant has a must-pass branch (happy path accepted,
correct output produced) AND a must-fail branch (broken precondition
rejected). The emotion passthrough + allowlist + prompt-prefix encoding
is W3's distinguishing surface vs W1/W2, so those paths have extra
two-sided coverage.

Mocking:
- ``urllib.request.urlopen`` replaced with a fake that yields preset
  response bytes (no real HTTPS in CI)
- ``subprocess.run`` replaced for keychain fallback tests
- ``rpm_path`` injected via ``tmp_path`` so tests never touch the
  operator's real state file
- ``now_epoch`` is passed explicitly into RPM helpers so the sliding
  window is deterministic regardless of wall clock
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
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.hal.voice.providers import gemini_tts as gm  # noqa: E402
from lib.hal.voice.providers._errors import (  # noqa: E402
    AuthFailure,
    NetworkError,
    RateLimited,
)
from lib.hal.voice.providers.base import TtsSynthResult  # noqa: E402
from lib.hal.voice.providers import SttProvider, TtsProvider  # noqa: E402
from lib.hal.voice.providers.gemini_tts import (  # noqa: E402
    API_KEY_ENV,
    DEFAULT_ALLOWED_EMOTIONS,
    DEFAULT_MODEL,
    DEFAULT_RPM_LIMIT,
    DEFAULT_VOICE,
    ENDPOINT_TEMPLATE,
    GEMINI_EMOTION_MATCH,
    GeminiTtsProvider,
    KEYCHAIN_SERVICE,
    MODEL_ENV,
    _encode_emotion,
    _fetch_keychain_token,
    _is_pcm_mime,
    _parse_sample_rate_from_mime,
    _pcm16_duration_s,
    _requests_in_window,
    _reserve_rpm,
    _resolve_api_key,
    _resolve_model,
    _validate_emotion,
    _wrap_pcm16_as_wav,
)
from lib.hal.voice.types import VoiceError  # noqa: E402


# ---------- helpers ----------


def _inline_audio_response(audio_bytes: bytes, mime: str = "audio/L16;codec=pcm;rate=24000") -> bytes:
    """Build the JSON body the generateContent endpoint returns."""
    return json.dumps(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": mime,
                                    "data": base64.b64encode(audio_bytes).decode("ascii"),
                                }
                            }
                        ]
                    }
                }
            ]
        }
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


def _http_error(code: int, reason: str = "boom") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://example.com/x", code=code, msg=reason, hdrs=None, fp=io.BytesIO(b"")
    )


@pytest.fixture
def rpm_file(tmp_path: Path) -> Path:
    return tmp_path / "rpm.jsonl"


@pytest.fixture
def provider(rpm_file: Path) -> GeminiTtsProvider:
    return GeminiTtsProvider(
        rpm_path=rpm_file,
        rpm_limit=1000,  # effectively disabled for most tests
        api_key="test-key",
    )


# ---------- Protocol conformance ----------


def test_provider_satisfies_tts_protocol(
    provider: GeminiTtsProvider,
) -> None:
    assert isinstance(provider, TtsProvider)


def test_provider_is_not_an_stt_provider(
    provider: GeminiTtsProvider,
) -> None:
    """Must-fail: slot-safety — no transcribe method, registry rejects."""
    assert not isinstance(provider, SttProvider)


def test_provider_name_is_gemini_tts_free(
    provider: GeminiTtsProvider,
) -> None:
    """Must-pass: FALLOVER_LADDER looks this key up at W4."""
    assert provider.name == "gemini_tts_free"


def test_exception_subclasses_are_shared_with_google_cloud_tts() -> None:
    """Must-pass: W2 and W3 raise the SAME AuthFailure class (extracted
    to providers/_errors.py). Identity matters for isinstance dispatch."""
    from lib.hal.voice.providers import google_cloud_tts as gct
    assert AuthFailure is gct.AuthFailure
    assert RateLimited is gct.RateLimited
    assert NetworkError is gct.NetworkError


# ---------- API key resolution ----------


def test_missing_key_raises_auth_failure(rpm_file: Path) -> None:
    """Must-fail: no env + keychain returns None → AuthFailure before any
    network call."""
    p = GeminiTtsProvider(rpm_path=rpm_file)
    env = dict(os.environ)
    env.pop(API_KEY_ENV, None)
    with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(
        gm, "_fetch_keychain_token", return_value=None
    ):
        with pytest.raises(AuthFailure, match="missing API key"):
            p.synthesise(text="hi", voice=DEFAULT_VOICE, emotion=None)


def test_env_var_wins_over_keychain() -> None:
    """Must-pass: env var used when present."""
    with mock.patch.dict(os.environ, {API_KEY_ENV: "env-key"}):
        assert _resolve_api_key() == "env-key"


def test_keychain_fallback_when_env_absent() -> None:
    """Must-pass: env missing → keychain subprocess consulted."""
    env = dict(os.environ)
    env.pop(API_KEY_ENV, None)
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="kc-key\n", stderr=""
    )
    with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(
        gm.subprocess, "run", return_value=fake
    ):
        assert _resolve_api_key() == "kc-key"


def test_keychain_timeout_returns_none() -> None:
    """Must-fail (graceful): subprocess timeout → None, no crash."""
    with mock.patch.object(
        gm.subprocess,
        "run",
        side_effect=subprocess.TimeoutExpired(cmd="security", timeout=5),
    ):
        assert _fetch_keychain_token() is None


def test_constructor_api_key_beats_env(rpm_file: Path) -> None:
    """Must-pass: explicit ctor api_key wins over env."""
    p = GeminiTtsProvider(rpm_path=rpm_file, api_key="ctor-key")
    with mock.patch.dict(os.environ, {API_KEY_ENV: "env-should-lose"}):
        assert p._resolve_key() == "ctor-key"


def test_keychain_service_is_grip_gemini() -> None:
    """Must-pass: re-use the HAPPI adapter's keychain service so operators
    have one key to rotate across chat + audio."""
    assert KEYCHAIN_SERVICE == "grip-gemini"


# ---------- model resolution ----------


def test_default_model_used_when_env_absent() -> None:
    """Must-pass: when ``HAL_GEMINI_TTS_MODEL`` is unset we fall back to
    ``gemini-2.5-flash-native-audio-preview``."""
    env = dict(os.environ)
    env.pop(MODEL_ENV, None)
    with mock.patch.dict(os.environ, env, clear=True):
        assert _resolve_model() == DEFAULT_MODEL


def test_env_override_picks_up_new_model() -> None:
    """Must-pass: operator can pin a specific model id via env when
    Google renames the preview."""
    with mock.patch.dict(os.environ, {MODEL_ENV: "gemini-next-preview"}):
        assert _resolve_model() == "gemini-next-preview"


def test_constructor_model_override_beats_env(rpm_file: Path) -> None:
    p = GeminiTtsProvider(rpm_path=rpm_file, model="custom-model")
    with mock.patch.dict(os.environ, {MODEL_ENV: "env-model"}):
        assert p._resolve_model_name() == "custom-model"


# ---------- emotion validation + encoding ----------


def test_none_emotion_passes_validation() -> None:
    """Must-pass: None is always allowed (no emotion requested)."""
    _validate_emotion(None, DEFAULT_ALLOWED_EMOTIONS)  # does not raise


def test_empty_string_emotion_treated_as_none() -> None:
    """Must-pass: ``""`` is operator-friendly shorthand for "no emotion"."""
    _validate_emotion("", DEFAULT_ALLOWED_EMOTIONS)


def test_allowed_emotion_passes(rpm_file: Path) -> None:
    """Must-pass: ``calm`` is in the default allowlist."""
    for emotion in DEFAULT_ALLOWED_EMOTIONS:
        _validate_emotion(emotion, DEFAULT_ALLOWED_EMOTIONS)


def test_disallowed_emotion_raises_voice_error() -> None:
    """Must-fail: off-allowlist emotion → VoiceError with actionable msg."""
    with pytest.raises(VoiceError, match="not in allowed set"):
        _validate_emotion("aggressive", DEFAULT_ALLOWED_EMOTIONS)


def test_custom_allowlist_via_constructor(rpm_file: Path) -> None:
    """Must-pass: ctor ``allowed_emotions=`` overrides the default — α-3c
    DirectiveResolver will pass a profile-specific set."""
    custom = frozenset({"happy"})
    p = GeminiTtsProvider(
        rpm_path=rpm_file, api_key="k", allowed_emotions=custom
    )
    # calm is NOT in the custom set → rejected
    with pytest.raises(VoiceError, match="not in allowed set"):
        p.synthesise(text="x", voice=DEFAULT_VOICE, emotion="calm")


def test_encode_emotion_no_prefix_when_none() -> None:
    """Must-pass: None emotion leaves text untouched."""
    assert _encode_emotion("hello", None) == "hello"


def test_encode_emotion_prefix_when_set() -> None:
    """Must-pass: bracketed style tag prepended to text."""
    assert _encode_emotion("hello", "calm") == "[calm] hello"


def test_encode_emotion_empty_string_leaves_text() -> None:
    """Must-pass: empty string → same as None, no bracket injected."""
    assert _encode_emotion("hello", "") == "hello"


# ---------- RPM accounting ----------


def test_requests_in_window_zero_when_file_absent(rpm_file: Path) -> None:
    """Must-pass edge: fresh operator → 0."""
    assert _requests_in_window(rpm_file) == 0


def test_reserve_rpm_increments(rpm_file: Path) -> None:
    """Must-pass: two reservations within the window sum to 2."""
    assert _reserve_rpm(rpm_file, limit=10, now_epoch=1000.0) == 1
    assert _reserve_rpm(rpm_file, limit=10, now_epoch=1001.0) == 2


def test_reserve_rpm_raises_at_boundary(rpm_file: Path) -> None:
    """Must-fail: attempting to reserve the (limit+1)-th slot raises
    RateLimited AND does NOT write a record."""
    for i in range(5):
        _reserve_rpm(rpm_file, limit=5, now_epoch=1000.0 + i)
    with pytest.raises(RateLimited, match="local RPM limit reached"):
        _reserve_rpm(rpm_file, limit=5, now_epoch=1005.0)
    # The failed reservation must NOT have been recorded.
    assert _requests_in_window(rpm_file, now_epoch=1005.0) == 5


def test_reserve_rpm_old_records_fall_out_of_window(rpm_file: Path) -> None:
    """Must-pass: records older than 60s are excluded — limit re-opens
    as the window slides."""
    # Fill five records at epoch 1000
    for i in range(5):
        _reserve_rpm(rpm_file, limit=5, now_epoch=1000.0 + i)
    # Now jump 70 seconds ahead — all 5 prior records have fallen out
    assert _reserve_rpm(rpm_file, limit=5, now_epoch=1070.0) == 1


def test_reserve_rpm_malformed_line_tolerated(rpm_file: Path) -> None:
    """Must-pass: partially-written line is skipped, not a crash."""
    rpm_file.parent.mkdir(parents=True, exist_ok=True)
    rpm_file.write_text("not-json\n")
    assert _reserve_rpm(rpm_file, limit=10, now_epoch=1000.0) == 1


def test_synthesise_rejects_when_rpm_saturated(
    provider: GeminiTtsProvider, rpm_file: Path
) -> None:
    """Must-fail: saturated local RPM → RateLimited raised BEFORE any
    network call is made. Proven by patching urlopen to raise loudly."""
    # Constructor already set rpm_limit=1000; drop a stricter provider
    p = GeminiTtsProvider(rpm_path=rpm_file, rpm_limit=1, api_key="k")
    _reserve_rpm(rpm_file, limit=100)  # eat the one slot

    called = []

    def _must_not(req, **kwargs):
        called.append(req)
        raise AssertionError("urlopen must not be called when RPM saturated")

    with mock.patch.object(
        gm.urllib.request, "urlopen", side_effect=_must_not
    ):
        with pytest.raises(RateLimited):
            p.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)
    assert called == []


# ---------- request building ----------


def test_build_request_api_key_in_header(
    provider: GeminiTtsProvider,
) -> None:
    """Must-pass: key lands in X-Goog-Api-Key header, NOT in URL or body."""
    req = provider._build_request(
        text_with_emotion="hi", voice=DEFAULT_VOICE, api_key="header-key"
    )
    assert req.headers["X-goog-api-key"] == "header-key"
    assert "header-key" not in req.full_url
    assert b"header-key" not in (req.data or b"")


def test_build_request_payload_shape(
    provider: GeminiTtsProvider,
) -> None:
    """Must-pass: payload matches Gemini 2.5 audio schema (contents +
    responseModalities + speechConfig.voiceConfig.prebuiltVoiceConfig)."""
    req = provider._build_request(
        text_with_emotion="[calm] hello",
        voice=DEFAULT_VOICE,
        api_key="k",
    )
    body = json.loads(req.data.decode("utf-8"))
    assert body == {
        "contents": [
            {"role": "user", "parts": [{"text": "[calm] hello"}]}
        ],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": DEFAULT_VOICE}}
            },
        },
    }


def test_build_request_url_uses_resolved_model(
    provider: GeminiTtsProvider,
) -> None:
    """Must-pass: URL is derived from the resolved model so env overrides
    flow through."""
    url = ENDPOINT_TEMPLATE.format(model=DEFAULT_MODEL)
    req = provider._build_request(
        text_with_emotion="x", voice=DEFAULT_VOICE, api_key="k"
    )
    assert req.full_url == url


# ---------- HTTPS happy path ----------


def test_synthesise_returns_tts_synth_result_pcm(
    provider: GeminiTtsProvider,
) -> None:
    """Must-pass: PCM response → WAV-wrapped TtsSynthResult."""
    pcm = b"\x00\x00" * 4800  # 0.2s at 24kHz
    body = _inline_audio_response(pcm, mime="audio/L16;codec=pcm;rate=24000")
    with mock.patch.object(
        gm.urllib.request, "urlopen", side_effect=_fake_urlopen(body)
    ):
        r = provider.synthesise(text="hi", voice=DEFAULT_VOICE, emotion=None)
    assert isinstance(r, TtsSynthResult)
    assert r.audio.startswith(b"RIFF")
    assert r.audio[8:12] == b"WAVE"
    assert r.provider_name == "gemini_tts_free"


def test_synthesise_emotion_match_is_gemini_constant(
    provider: GeminiTtsProvider,
) -> None:
    """Must-pass: exactly 0.9 — between Piper (0.85) and Cloud (0.95)."""
    body = _inline_audio_response(b"\x00\x00" * 100)
    with mock.patch.object(
        gm.urllib.request, "urlopen", side_effect=_fake_urlopen(body)
    ):
        r = provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion="calm")
    assert r.emotion_match == GEMINI_EMOTION_MATCH == 0.9


def test_synthesise_emotion_flows_into_request_text(
    provider: GeminiTtsProvider,
) -> None:
    """Must-pass: requested emotion is encoded as a bracketed prefix in
    the outgoing prompt. Tests the full request pipeline by capturing
    the payload urlopen received."""
    body = _inline_audio_response(b"\x00\x00" * 100)
    captured: dict[str, bytes] = {}

    def _capture(req, **kwargs):
        captured["data"] = req.data or b""
        return _FakeResponse(body)

    with mock.patch.object(
        gm.urllib.request, "urlopen", side_effect=_capture
    ):
        provider.synthesise(text="hello there", voice=DEFAULT_VOICE, emotion="calm")
    outgoing = json.loads(captured["data"].decode("utf-8"))
    sent_text = outgoing["contents"][0]["parts"][0]["text"]
    assert sent_text == "[calm] hello there"


def test_synthesise_none_emotion_leaves_prompt_unmolested(
    provider: GeminiTtsProvider,
) -> None:
    """Must-pass: no emotion → raw text, no bracket prefix."""
    body = _inline_audio_response(b"\x00\x00" * 100)
    captured: dict[str, bytes] = {}

    def _capture(req, **kwargs):
        captured["data"] = req.data or b""
        return _FakeResponse(body)

    with mock.patch.object(
        gm.urllib.request, "urlopen", side_effect=_capture
    ):
        provider.synthesise(text="hello", voice=DEFAULT_VOICE, emotion=None)
    outgoing = json.loads(captured["data"].decode("utf-8"))
    assert outgoing["contents"][0]["parts"][0]["text"] == "hello"


def test_synthesise_empty_voice_falls_back_to_default(
    provider: GeminiTtsProvider,
) -> None:
    """Must-pass: empty voice → DEFAULT_VOICE (Iapetus)."""
    body = _inline_audio_response(b"\x00\x00" * 100)
    with mock.patch.object(
        gm.urllib.request, "urlopen", side_effect=_fake_urlopen(body)
    ):
        r = provider.synthesise(text="x", voice="", emotion=None)
    assert r.voice == DEFAULT_VOICE


def test_synthesise_passes_through_wav_response(
    provider: GeminiTtsProvider,
) -> None:
    """Must-pass: if Gemini returns a WAV container (mime audio/wav), we
    pass through unchanged — no double-wrapping."""
    # Build a valid 44-byte WAV.
    pcm = b"\x00\x00" * 24000
    wav = _wrap_pcm16_as_wav(pcm, sample_rate=24000)
    body = _inline_audio_response(wav, mime="audio/wav")
    with mock.patch.object(
        gm.urllib.request, "urlopen", side_effect=_fake_urlopen(body)
    ):
        r = provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)
    assert r.audio == wav
    assert len(r.audio) == 44 + len(pcm)
    # Duration parsed from header → 1.0s at 24kHz mono
    assert abs(r.duration_s - 1.0) < 1e-6


def test_synthesise_unknown_mime_returns_bytes_with_zero_duration(
    provider: GeminiTtsProvider,
) -> None:
    """Must-pass: unknown container (mp3 etc.) → pass through with
    duration=0.0 (afplay handles detection)."""
    fake_mp3 = b"ID3\x03\x00" + b"\xff\xfb" * 100  # not a real mp3 but non-RIFF
    body = _inline_audio_response(fake_mp3, mime="audio/mpeg")
    with mock.patch.object(
        gm.urllib.request, "urlopen", side_effect=_fake_urlopen(body)
    ):
        r = provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)
    assert r.audio == fake_mp3
    assert r.duration_s == 0.0


# ---------- HTTP error branches ----------


def test_http_401_raises_auth_failure(
    provider: GeminiTtsProvider,
) -> None:
    with mock.patch.object(
        gm.urllib.request,
        "urlopen",
        side_effect=_raising_urlopen(_http_error(401, "Unauthorized")),
    ):
        with pytest.raises(AuthFailure, match="HTTP 401"):
            provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)


def test_http_403_raises_auth_failure(
    provider: GeminiTtsProvider,
) -> None:
    with mock.patch.object(
        gm.urllib.request,
        "urlopen",
        side_effect=_raising_urlopen(_http_error(403, "Forbidden")),
    ):
        with pytest.raises(AuthFailure, match="HTTP 403"):
            provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)


def test_http_429_raises_rate_limited(
    provider: GeminiTtsProvider,
) -> None:
    """Must-fail: server 429 → RateLimited (same type the local RPM check
    raises — ladder treats both identically)."""
    with mock.patch.object(
        gm.urllib.request,
        "urlopen",
        side_effect=_raising_urlopen(_http_error(429, "Too Many Requests")),
    ):
        with pytest.raises(RateLimited, match="HTTP 429"):
            provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)


def test_http_500_raises_bare_voice_error(
    provider: GeminiTtsProvider,
) -> None:
    """Must-fail: unmapped 5xx → bare VoiceError, not mis-classified as
    AuthFailure / RateLimited."""
    with mock.patch.object(
        gm.urllib.request,
        "urlopen",
        side_effect=_raising_urlopen(_http_error(500, "Internal Server Error")),
    ):
        with pytest.raises(VoiceError) as excinfo:
            provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)
    assert not isinstance(excinfo.value, (AuthFailure, RateLimited))


def test_url_error_raises_network_error(
    provider: GeminiTtsProvider,
) -> None:
    with mock.patch.object(
        gm.urllib.request,
        "urlopen",
        side_effect=_raising_urlopen(urllib.error.URLError("dns fail")),
    ):
        with pytest.raises(NetworkError, match="dns fail"):
            provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)


def test_timeout_error_raises_network_error(
    provider: GeminiTtsProvider,
) -> None:
    with mock.patch.object(
        gm.urllib.request,
        "urlopen",
        side_effect=_raising_urlopen(TimeoutError("slow")),
    ):
        with pytest.raises(NetworkError, match="timed out"):
            provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)


# ---------- body-parsing error branches ----------


def test_malformed_json_raises_voice_error(
    provider: GeminiTtsProvider,
) -> None:
    with mock.patch.object(
        gm.urllib.request,
        "urlopen",
        side_effect=_fake_urlopen(b"not-json"),
    ):
        with pytest.raises(VoiceError, match="malformed JSON"):
            provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)


def test_missing_inline_data_raises_voice_error(
    provider: GeminiTtsProvider,
) -> None:
    """Must-fail: candidates present but no inlineData → VoiceError."""
    body = json.dumps(
        {"candidates": [{"content": {"parts": [{"text": "nope"}]}}]}
    ).encode("utf-8")
    with mock.patch.object(
        gm.urllib.request, "urlopen", side_effect=_fake_urlopen(body)
    ):
        with pytest.raises(VoiceError, match="missing"):
            provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)


def test_missing_candidates_raises_voice_error(
    provider: GeminiTtsProvider,
) -> None:
    body = json.dumps({"promptFeedback": "nope"}).encode("utf-8")
    with mock.patch.object(
        gm.urllib.request, "urlopen", side_effect=_fake_urlopen(body)
    ):
        with pytest.raises(VoiceError, match="missing"):
            provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)


def test_invalid_base64_raises_voice_error(
    provider: GeminiTtsProvider,
) -> None:
    body = json.dumps(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "audio/L16;rate=24000",
                                    "data": "!!!not-base64!!!",
                                }
                            }
                        ]
                    }
                }
            ]
        }
    ).encode("utf-8")
    with mock.patch.object(
        gm.urllib.request, "urlopen", side_effect=_fake_urlopen(body)
    ):
        with pytest.raises(VoiceError, match="valid base64"):
            provider.synthesise(text="x", voice=DEFAULT_VOICE, emotion=None)


# ---------- mime helpers ----------


def test_parse_sample_rate_from_mime_standard() -> None:
    assert _parse_sample_rate_from_mime("audio/L16;codec=pcm;rate=24000") == 24000
    assert _parse_sample_rate_from_mime("audio/L16;rate=16000") == 16000


def test_parse_sample_rate_from_mime_default_when_absent() -> None:
    """Must-pass fallback: missing ``rate=`` → DEFAULT_SAMPLE_RATE_HZ."""
    assert _parse_sample_rate_from_mime("audio/L16") == gm.DEFAULT_SAMPLE_RATE_HZ


def test_parse_sample_rate_from_mime_default_when_unparseable() -> None:
    """Must-pass fallback: non-int rate → default (no crash)."""
    assert (
        _parse_sample_rate_from_mime("audio/L16;rate=notanumber")
        == gm.DEFAULT_SAMPLE_RATE_HZ
    )


def test_is_pcm_mime_detects_l16_and_pcm() -> None:
    assert _is_pcm_mime("audio/L16;codec=pcm;rate=24000") is True
    assert _is_pcm_mime("audio/pcm") is True
    assert _is_pcm_mime("audio/wav") is False
    assert _is_pcm_mime("audio/mpeg") is False


# ---------- duration helpers ----------


def test_pcm16_duration_s_from_byte_count() -> None:
    """Must-pass: 24000 samples × 2 bytes at 24kHz = 1.0s."""
    pcm = b"\x00\x00" * 24000
    assert abs(_pcm16_duration_s(pcm, sample_rate=24000) - 1.0) < 1e-9


def test_pcm16_duration_s_scales_with_sample_rate() -> None:
    """Must-pass: same bytes at half rate → double duration."""
    pcm = b"\x00\x00" * 12000
    assert abs(_pcm16_duration_s(pcm, sample_rate=12000) - 1.0) < 1e-9
    assert abs(_pcm16_duration_s(pcm, sample_rate=24000) - 0.5) < 1e-9


def test_wrap_pcm16_produces_canonical_wav_header() -> None:
    pcm = b"\x00\x00" * 1000
    wav = _wrap_pcm16_as_wav(pcm, sample_rate=24000)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert struct.unpack_from("<I", wav, 24)[0] == 24000
    assert len(wav) == 44 + len(pcm)
