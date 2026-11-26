"""HAL Voice — ``hal voice realtime`` CLI command.

Closes the "provider exists but nothing calls it" gap: makes the GA
OpenAI Realtime backend invokable end-to-end from the command line.

Entry points
------------
    hal voice realtime --text "say hello"   # typed-text turn, no mic
    hal voice realtime                      # continuous mic loop (sounddevice)

Design
------
- PCM→WAV buffering: ``audio_delta`` bytes accumulate in a list; on
  ``turn_done`` they are wrapped into a valid WAV (stdlib ``wave``,
  PCM16/mono/24000) written to a tmpfile, then played via
  ``AudioController.get().play_file(path)``.  No new playback machinery.
- Mic capture behind a DIP seam (``mic_source`` callable) so the loop
  is unit-testable with synthetic PCM.  The default seam lazily imports
  ``sounddevice``; if absent it raises a clean error.
- ``--text`` mode drives a full round-trip without a mic: ``send_text``
  + response trigger + wait for ``turn_done``.  This is the headlessly
  verifiable dogfood path.
- All live OpenAI calls run synchronously with a bounded timeout
  (``REALTIME_TIMEOUT_S``).
"""
from __future__ import annotations

import argparse
import io
import sys
import tempfile
import threading
import wave
from pathlib import Path
from typing import Callable, Iterator, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REALTIME_TIMEOUT_S = 30  # max seconds to wait for turn_done
PCM_RATE = 24_000         # GA endpoint output: PCM16 @ 24 kHz mono
PCM_CHANNELS = 1
PCM_SAMPWIDTH = 2         # 16-bit = 2 bytes per sample

MIC_CHUNK_FRAMES = 4096
MIC_RATE = 24_000


# ---------------------------------------------------------------------------
# WAV helper
# ---------------------------------------------------------------------------

def pcm_chunks_to_wav(chunks: list[bytes]) -> bytes:
    """Wrap raw PCM16 chunks into a valid WAV blob (in-memory).

    Returns an empty-but-valid WAV when *chunks* is empty.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(PCM_CHANNELS)
        wf.setsampwidth(PCM_SAMPWIDTH)
        wf.setframerate(PCM_RATE)
        for chunk in chunks:
            wf.writeframes(chunk)
    return buf.getvalue()


def _validate_wav(wav_bytes: bytes) -> bool:
    """True iff *wav_bytes* is well-formed WAV with the expected GA params."""
    try:
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            return (
                wf.getframerate() == PCM_RATE
                and wf.getnchannels() == PCM_CHANNELS
                and wf.getsampwidth() == PCM_SAMPWIDTH
            )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Event accumulator  (coordinator pattern — CC≤5 each method)
# ---------------------------------------------------------------------------

class _TurnAccumulator:
    """Collect audio_delta + transcript + terminal signal from the backend."""

    def __init__(self) -> None:
        self.pcm_chunks: list[bytes] = []
        self.transcript_parts: list[str] = []
        self.error: Optional[str] = None
        self._done = threading.Event()

    # Dispatch table avoids if/elif chain (CC reduction)
    def _handle_audio(self, ev) -> None:
        if ev.audio:
            self.pcm_chunks.append(ev.audio)

    def _handle_transcript(self, ev) -> None:
        if not ev.text:
            return
        # GA emits incremental .delta events (is_final=False) then a .done with
        # the FULL transcript (is_final=True). Accumulate deltas; use the final
        # ONLY when no deltas arrived — otherwise the deltas already form the
        # transcript and appending the final doubles it.
        if getattr(ev, "is_final", False):
            if not self.transcript_parts:
                self.transcript_parts.append(ev.text)
            return
        self.transcript_parts.append(ev.text)
        print(ev.text, end="", flush=True)

    def _handle_done(self, _ev) -> None:
        self._done.set()

    def _handle_error(self, ev) -> None:
        self.error = ev.text
        self._done.set()

    _DISPATCH: dict = {}  # populated after class body (avoids forward refs)

    def on_event(self, ev) -> None:
        handler = self._DISPATCH.get(ev.kind)
        if handler:
            handler(self, ev)

    def wait(self, timeout_s: float) -> None:
        self._done.wait(timeout=timeout_s)

    @property
    def done(self) -> bool:
        return self._done.is_set()


_TurnAccumulator._DISPATCH = {
    "audio_delta": _TurnAccumulator._handle_audio,
    "assistant_transcript": _TurnAccumulator._handle_transcript,
    "turn_done": _TurnAccumulator._handle_done,
    "error": _TurnAccumulator._handle_error,
}


# ---------------------------------------------------------------------------
# Turn drivers (one concern each)
# ---------------------------------------------------------------------------

def _drive_text_turn(session, text: str, acc: _TurnAccumulator, timeout_s: float) -> None:
    session.send_text(text)
    # commit_audio=False: a text turn has no audio buffer — committing an empty
    # one triggers GA's "buffer too small" error; we only want the response.
    session.commit_turn(commit_audio=False)
    acc.wait(timeout_s)


def _drive_mic_turn(
    session,
    mic_source: Optional[Callable[[], Iterator[bytes]]],
    acc: _TurnAccumulator,
    timeout_s: float,
) -> None:
    src = mic_source() if mic_source else _default_mic_source()
    for chunk in src:
        if acc.done:
            break
        session.send_audio(chunk)
    acc.wait(timeout_s)


# ---------------------------------------------------------------------------
# Default mic seam (lazy sounddevice)
# ---------------------------------------------------------------------------

def _default_mic_source(chunk_frames: int = MIC_CHUNK_FRAMES) -> Iterator[bytes]:
    """Yield PCM16 mono chunks; raises clean RuntimeError if sounddevice absent."""
    try:
        import sounddevice as sd  # type: ignore[import]
    except ImportError:
        raise RuntimeError(
            "mic capture unavailable — pip install sounddevice, or use --text"
        )
    import numpy as np  # type: ignore[import]

    with sd.InputStream(
        samplerate=MIC_RATE,
        channels=PCM_CHANNELS,
        dtype="int16",
        blocksize=chunk_frames,
    ) as stream:
        while True:
            data, _ = stream.read(chunk_frames)
            yield data.astype("int16").tobytes()


# ---------------------------------------------------------------------------
# Playback helper
# ---------------------------------------------------------------------------

def _play_wav(wav_bytes: bytes) -> None:
    """Write WAV to a tmpfile and play via AudioController (non-fatal on error)."""
    try:
        from lib.hal.voice.audio_controller import AudioController
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = Path(tmp.name)
        try:
            AudioController.get().play_file(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
    except Exception as exc:
        print(f"[realtime] playback unavailable: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_realtime_turn(
    *,
    text: Optional[str] = None,
    system: Optional[str] = None,
    voice: str = "alloy",
    mic_source: Optional[Callable[[], Iterator[bytes]]] = None,
    timeout_s: float = REALTIME_TIMEOUT_S,
    open_realtime_fn=None,
    realtime_config_cls=None,
) -> dict:
    """Run one realtime turn; return ``{audio_bytes, wav_valid, transcript, error}``."""
    if open_realtime_fn is None:
        from lib.hal.voice.realtime_session import open_realtime
        open_realtime_fn = open_realtime
    if realtime_config_cls is None:
        from lib.hal.voice.realtime_base import RealtimeConfig
        realtime_config_cls = RealtimeConfig

    config = realtime_config_cls(
        system=system or "You are a helpful assistant. Be concise.",
        voice=voice,
        turn_detection="server_vad",
    )
    acc = _TurnAccumulator()
    session = open_realtime_fn(backend="openai_realtime", config=config)
    session.on_transcript(acc.on_event)
    session.start()
    try:
        if text is not None:
            _drive_text_turn(session, text, acc, timeout_s)
        else:
            _drive_mic_turn(session, mic_source, acc, timeout_s)
    finally:
        print()
        session.close()

    total_audio_bytes = sum(len(c) for c in acc.pcm_chunks)
    wav_bytes = pcm_chunks_to_wav(acc.pcm_chunks)
    wav_valid = _validate_wav(wav_bytes)
    if total_audio_bytes > 0 and wav_valid:
        _play_wav(wav_bytes)
    return {
        "audio_bytes": total_audio_bytes,
        "wav_valid": wav_valid,
        "transcript": "".join(acc.transcript_parts),
        "error": acc.error,
    }


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def _cmd_realtime(args: argparse.Namespace) -> int:
    text: Optional[str] = args.text or None
    if text is None:
        try:
            import sounddevice  # type: ignore[import]  # noqa: F401
        except ImportError:
            print(
                "mic capture unavailable — pip install sounddevice, or use --text",
                file=sys.stderr,
            )
            return 1
        print("Listening … (Ctrl-C to stop)", file=sys.stderr)

    try:
        result = run_realtime_turn(
            text=text,
            system=args.system or None,
            voice=args.voice,
            timeout_s=args.timeout,
        )
    except KeyboardInterrupt:
        print("\n[realtime] interrupted", file=sys.stderr)
        return 0
    except Exception as exc:
        print(f"realtime error: {exc}", file=sys.stderr)
        return 1

    if result["error"]:
        print(f"realtime error: {result['error']}", file=sys.stderr)
        return 1

    print(
        f"[realtime] audio_delta bytes={result['audio_bytes']} "
        f"wav_valid={result['wav_valid']} "
        f"transcript_len={len(result['transcript'])}",
        file=sys.stderr,
    )
    return 0


def build_realtime_subparser(subparsers) -> None:
    """Register ``realtime`` onto an existing subparsers object."""
    p = subparsers.add_parser(
        "realtime",
        help="full-duplex voice via OpenAI Realtime GA endpoint",
    )
    p.add_argument("--text", default="", metavar="TEXT",
                   help="drive a turn from typed text (no mic); empty = mic mode")
    p.add_argument("--voice", default="alloy",
                   help="assistant voice (alloy, echo, shimmer, …; default: alloy)")
    p.add_argument("--system", default="", help="system prompt override")
    p.add_argument("--timeout", type=float, default=float(REALTIME_TIMEOUT_S),
                   help=f"max seconds to wait for turn_done (default: {REALTIME_TIMEOUT_S})")
    p.set_defaults(func=_cmd_realtime)


__all__ = [
    "build_realtime_subparser",
    "pcm_chunks_to_wav",
    "run_realtime_turn",
    "PCM_RATE",
    "PCM_CHANNELS",
    "PCM_SAMPWIDTH",
    "_TurnAccumulator",
]
