"""HAL Voice — macOS ``say`` local TTS provider.

The always-on final rung of ``FALLOVER_LADDER``. Speech keeps working
when cloud providers are down, Piper is uninstalled, or the operator
is offline — ``say`` is a built-in macOS binary that has shipped in
every release since Mac OS X 10.3.

Shape:
- Shells to ``say --file-format=WAVE --data-format=LEI16 -v <voice>
  -o <out> <text>`` so the output matches the LINEAR16 PCM WAV format
  Piper and the cloud providers produce. Cache + player see uniform
  bytes across every rung of the ladder.
- ``is_available()`` returns True iff the ``say`` binary is on PATH
  (always True on Darwin; exists for uniformity with Piper's missing-
  binary handling).
- Default voice: ``"Jamie (Premium)"`` — the operator's chosen default
  (PR #241). Voice name must match the exact ``say -v`` argument.
- ``emotion_match = 0.80`` — exactly at the cache threshold, so the
  output is cached for fast replay but positioned below Piper (0.85)
  and both cloud providers (0.90 / 0.95). ``say`` has no emotion
  channel, so ``emotion`` kwarg is accepted and ignored.

Registered into ``PROVIDER_REGISTRY["tts"]["macos_say"]`` by
``providers/__init__.py`` so ladder dispatch can invoke it as the
final rung instead of skipping it with an "unregistered" warning.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from ..types import VoiceError
from ._wav_utils import parse_wav_duration_s as _parse_wav_duration_s
from .base import TtsSynthResult


SAY_BIN = "say"
DEFAULT_VOICE = "Jamie (Premium)"
MACOS_SAY_EMOTION_MATCH = 0.80
SUBPROCESS_TIMEOUT_S = 30.0


def is_available() -> bool:
    """True iff the ``say`` binary is on PATH.

    Mirrors Piper's ``is_available()`` shape so the W4 ladder dispatch
    treats both local providers uniformly. The check is cheap (PATH
    lookup, no subprocess), so we re-run it on every ladder resolution
    rather than caching across calls.
    """
    return shutil.which(SAY_BIN) is not None


def synthesise_bytes(
    text: str,
    voice: str = DEFAULT_VOICE,
    timeout_s: float = SUBPROCESS_TIMEOUT_S,
) -> bytes:
    """Synthesise ``text`` to raw WAV bytes via the ``say`` binary.

    Standalone free function — usable directly as ``synthesise_fn`` in
    ``AudioController.speak()`` via ``make_synthesise_fn()``, without
    constructing a ``MacosSayProvider`` instance.  Raises ``VoiceError``
    on all failure modes so the caller can fallover or halt.
    """
    resolved_voice = voice or DEFAULT_VOICE
    if shutil.which(SAY_BIN) is None:
        raise VoiceError(
            "macos_say: `say` binary not found on PATH — this provider "
            "requires macOS"
        )
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        out_path = Path(tf.name)
    try:
        try:
            result = subprocess.run(
                [
                    SAY_BIN,
                    "--file-format=WAVE",
                    "--data-format=LEI16",
                    "-v", resolved_voice,
                    "-o", str(out_path),
                    text,
                ],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise VoiceError(
                f"macos_say: synthesis timed out after {timeout_s}s"
            ) from exc
        if result.returncode != 0:
            raise VoiceError(
                f"macos_say: subprocess exited {result.returncode}: "
                f"{result.stderr.strip()}"
            )
        if not out_path.is_file() or out_path.stat().st_size == 0:
            raise VoiceError("macos_say: produced empty output file")
        return out_path.read_bytes()
    finally:
        out_path.unlink(missing_ok=True)


def make_synthesise_fn(
    voice: str = DEFAULT_VOICE,
    timeout_s: float = SUBPROCESS_TIMEOUT_S,
) -> Callable[[str], bytes]:
    """Return a closure for ``AudioController.speak(synthesise_fn=...)``.

    Usage::

        ctrl = AudioController.get()
        ctrl.speak(text, synthesise_fn=make_synthesise_fn())
    """
    def _synth(text: str) -> bytes:
        return synthesise_bytes(text, voice=voice, timeout_s=timeout_s)
    return _synth


# ``_parse_wav_duration_s`` is re-exported from ``._wav_utils`` so
# tests that imported the private name keep resolving. Canonical
# implementation lives in one place for all four WAV-producing
# providers (see _wav_utils for the contract).


class MacosSayProvider:
    """macOS ``say`` local TTS provider.

    Satisfies the ``TtsProvider`` Protocol. Does not inherit — the
    runtime_checkable Protocol validates shape at registration time.

    ``emotion`` is accepted for Protocol conformance and ignored: the
    ``say`` binary has no emotion-conditioning. Unlike Gemini, this
    provider does NOT inject bracketed style prefixes because the
    synthesis engine would speak them verbatim (``[calm] hello`` would
    become spoken audio of the word "calm" followed by "hello"). The
    DirectiveResolver at α-3c treats ``say`` as emotion-neutral.
    """

    name = "macos_say"

    def __init__(self, *, timeout_s: float = SUBPROCESS_TIMEOUT_S) -> None:
        self._timeout_s = timeout_s

    def synthesise(
        self,
        *,
        text: str,
        voice: str,
        emotion: str | None,
    ) -> TtsSynthResult:
        del emotion  # Protocol-compliant; say has no emotion channel
        resolved_voice = voice or DEFAULT_VOICE
        audio = synthesise_bytes(text, voice=resolved_voice, timeout_s=self._timeout_s)
        duration_s = _parse_wav_duration_s(audio)
        return TtsSynthResult(
            audio=audio,
            duration_s=duration_s,
            voice=resolved_voice,
            provider_name=self.name,
            emotion_match=MACOS_SAY_EMOTION_MATCH,
        )


__all__ = [
    "DEFAULT_VOICE",
    "MACOS_SAY_EMOTION_MATCH",
    "MacosSayProvider",
    "SAY_BIN",
    "SUBPROCESS_TIMEOUT_S",
    "is_available",
    "make_synthesise_fn",
    "synthesise_bytes",
]
