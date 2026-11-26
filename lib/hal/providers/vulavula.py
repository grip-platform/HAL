"""Vulavula provider — Lelapa AI African NLP middleware for HAL.

Wraps Vulavula's translation, transcription, and sentiment APIs as a
middleware layer: user text in African language -> translate to English ->
HAL LLM processes -> translate response back. Supports 17 African languages.

Not OpenAI-compatible — custom REST endpoints. Acts as a translation bridge
between African-language users and HAL's LLM providers.

Usage:
    from lib.hal.providers.vulavula import VulavulaProvider

    vulavula = VulavulaProvider(api_key_source="keychain:grip-vulavula")
    english = vulavula.translate("Sawubona umhlaba", source="zul", target="eng")
    sentiment = vulavula.analyse_sentiment("This is great", language="eng")
"""
import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

_BASE_URL = "https://api.lelapa.ai/v1"
_KEYCHAIN_CMD = ["security", "find-generic-password", "-w", "-s"]
_TIMEOUT_S = 30
_MAX_RETRIES = 5
_INITIAL_BACKOFF_S = 5.0
_BACKOFF_MULTIPLIER = 2.0
_RATE_LIMIT_DELAY_S = 1.0

_log = logging.getLogger("hal.vulavula")

# Language codes: Vulavula API expects ISO 639-3 + _Latn script suffix
# (discovered via TNMR integration — bare codes fail)
LANGUAGES: dict[str, str] = {
    "zul": "isiZulu",
    "xho": "isiXhosa",
    "sot": "Sesotho",
    "tsn": "Setswana",
    "nso": "Sepedi",
    "afr": "Afrikaans",
    "eng": "English",
    "ssw": "siSwati",
    "tso": "Xitsonga",
    "ven": "Tshivenda",
    "nbl": "isiNdebele",
    "nde": "Sindebele",
    "sna": "Shona",
    "nya": "Chichewa",
    "swh": "Kiswahili",
    "kin": "Kinyarwanda",
    "lin": "Lingala",
}

# API code format: append _Latn script suffix (from TNMR pattern)
_API_CODE: dict[str, str] = {code: f"{code}_Latn" for code in LANGUAGES}


@dataclass(frozen=True)
class TranslationResult:
    """Result of a Vulavula translation call."""

    source_lang: str
    target_lang: str
    source_text: str
    translated_text: str


@dataclass(frozen=True)
class SentimentResult:
    """Result of a Vulavula sentiment analysis call."""

    text: str
    sentiment: str
    confidence: float
    language: str


@dataclass(frozen=True)
class TranscriptionResult:
    """Result of a Vulavula transcription call."""

    text: str
    language: str
    duration_s: float = 0.0


class VulavulaProvider:
    """Lelapa AI Vulavula — African NLP middleware for HAL."""

    def __init__(self, api_key_source: str = "keychain:grip-vulavula"):
        self._api_key = _resolve_api_key(api_key_source)
        self._cache: dict[tuple[str, str, str], str] = {}

    def translate(self, text: str, source: str, target: str) -> TranslationResult:
        """Translate text between African languages and English."""
        cache_key = (text, source, target)
        if cache_key in self._cache:
            return TranslationResult(
                source_lang=source, target_lang=target,
                source_text=text, translated_text=self._cache[cache_key],
            )

        payload = {
            "input_text": text,
            "source_lang": _to_api_code(source),
            "target_lang": _to_api_code(target),
        }
        data = self._post_with_retry("/translate/process", payload)
        translated = data.get("translation", [{}])
        output = translated[0].get("translated_text", "") if translated else ""

        if output:
            self._cache[cache_key] = output

        return TranslationResult(
            source_lang=source, target_lang=target,
            source_text=text, translated_text=output,
        )

    def analyse_sentiment(self, text: str, language: str = "eng") -> SentimentResult:
        """Analyse sentiment of text."""
        payload = {"encoded_text": text, "language": language}
        data = self._post("/sentiment/process", payload)
        return SentimentResult(
            text=text,
            sentiment=data.get("sentiment", "neutral"),
            confidence=data.get("confidence", 0.0),
            language=language,
        )

    def transcribe(self, audio_path: str, language: str = "zul") -> TranscriptionResult:
        """Transcribe an audio file to text via Vulavula's sync STT endpoint.

        POSTs the audio as a multipart ``file`` to ``/transcribe/sync`` with
        ``lang_code`` as a URL query parameter (the API contract — NOT a form
        field) and reads the transcript from ``transcription_text``.
        """
        if not os.path.exists(audio_path):
            return TranscriptionResult(text="", language=language)

        with open(audio_path, "rb") as f:
            audio_data = f.read()

        data = self._post_multipart(
            "/transcribe/sync",
            file_data=audio_data,
            file_name=os.path.basename(audio_path),
            lang_code=language,
        )
        return TranscriptionResult(
            text=data.get("transcription_text", ""),
            language=data.get("language_code", language),
            duration_s=data.get("audio_length_seconds", 0.0),
        )

    def list_languages(self) -> dict[str, str]:
        """Return supported language codes and names."""
        return dict(LANGUAGES)

    def bridge(self, text: str, source_lang: str, llm_fn: "callable", target_lang: str = "") -> str:
        """Full bridge: translate in -> LLM process -> translate back.

        Args:
            text: User input in source language
            source_lang: ISO 639-3 code of input language
            llm_fn: Callable that takes English text, returns English text
            target_lang: Output language (defaults to source_lang)
        """
        target_lang = target_lang or source_lang
        if source_lang == "eng":
            english_response = llm_fn(text)
        else:
            english_input = self.translate(text, source_lang, "eng").translated_text
            english_response = llm_fn(english_input)

        if target_lang == "eng":
            return english_response
        return self.translate(english_response, "eng", target_lang).translated_text

    def _post_with_retry(self, path: str, payload: dict) -> dict:
        """POST with exponential backoff retry (from TNMR pattern)."""
        backoff = _INITIAL_BACKOFF_S
        for attempt in range(_MAX_RETRIES):
            data = self._post(path, payload)
            if data:
                return data
            if attempt < _MAX_RETRIES - 1:
                _log.warning("Vulavula %s attempt %d failed, retrying in %.0fs", path, attempt + 1, backoff)
                time.sleep(backoff)
                backoff = min(backoff * _BACKOFF_MULTIPLIER, 80.0)
        _log.error("Vulavula %s failed after %d attempts", path, _MAX_RETRIES)
        return {}

    def _post(self, path: str, payload: dict) -> dict:
        """POST JSON to Vulavula API."""
        url = f"{_BASE_URL}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={
                "Content-Type": "application/json",
                "X-CLIENT-TOKEN": self._api_key,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                _log.warning("Vulavula rate limited (429)")
            return {}
        except (urllib.error.URLError, json.JSONDecodeError, OSError):
            return {}

    def _post_multipart(
        self, path: str, file_data: bytes, file_name: str, lang_code: str,
    ) -> dict:
        """POST a multipart ``file`` for transcription.

        Per the Vulavula transcribe contract the audio is the ONLY multipart
        field (``file``); ``lang_code`` (bare ISO 639-3, e.g. "afr") is a URL
        query parameter, not a form field. Logs HTTP errors rather than
        swallowing them silently, then degrades to {} (best-effort).
        """
        boundary = "----HALVulavulaBoundary"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{file_name}"\r\n'
            f"Content-Type: {_audio_mime(file_name)}\r\n\r\n"
        ).encode("utf-8") + file_data + f"\r\n--{boundary}--\r\n".encode("utf-8")

        url = f"{_BASE_URL}{path}?lang_code={lang_code}"
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "X-CLIENT-TOKEN": self._api_key,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            _log.warning("Vulavula transcribe HTTP %s: %s", e.code,
                         e.read().decode("utf-8", "replace")[:200])
            return {}
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
            _log.warning("Vulavula transcribe failed: %s", e)
            return {}


def _audio_mime(file_name: str) -> str:
    """Best-effort audio MIME type from a file extension (for the multipart part)."""
    ext = os.path.splitext(file_name)[1].lower()
    return {
        ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".wav": "audio/wav",
        ".flac": "audio/flac", ".ogg": "audio/ogg",
    }.get(ext, "application/octet-stream")


def _to_api_code(lang: str) -> str:
    """Convert bare ISO 639-3 code to Vulavula API format (with _Latn suffix)."""
    return _API_CODE.get(lang, f"{lang}_Latn")


def _resolve_api_key(source: str) -> str:
    """Resolve API key from keychain or env var."""
    if source.startswith("keychain:"):
        service = source.split(":", 1)[1]
        if not _is_safe_service_name(service):
            _log.warning("Unsafe keychain service name rejected: %s", service[:50])
            return ""
        try:
            result = subprocess.run(
                _KEYCHAIN_CMD + [service],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except (subprocess.TimeoutExpired, OSError):
            return ""
    if source.startswith("env:"):
        return os.environ.get(source.split(":", 1)[1], "")
    return source


def _is_safe_service_name(name: str) -> bool:
    """Validate keychain service name — alphanumeric, hyphens, underscores only."""
    import re
    return bool(re.match(r"^[a-zA-Z0-9_-]+$", name))
