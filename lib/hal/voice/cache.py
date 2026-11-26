"""HAL Voice — fingerprint-keyed LRU cache + tmpfs audio-path wrapper.

Per plans/voice-first-harness-triphase.md Rev 2.3 section 4.3.

Two independent concerns live in this module:

1. ``VoiceCache`` — an in-memory LRU over TTS audio bytes keyed by a
   short hex fingerprint of (directive_text, voice, emotion, provider).
   The cache enforces a plan invariant: results whose emotion_match
   score is below ``CACHE_EMOTION_THRESHOLD`` (0.8) are silently NOT
   cached, so low-quality synth doesn't pollute the hit path.

2. ``tmpfs_audio_path`` — a context manager that materialises a bytes
   payload to ``$TMPDIR`` for a single playback invocation and unlinks
   the file when the context exits (including on exception). This is
   how voice.speak shuttles audio to the OS player without leaving
   evidence on disk.

Design choice: the cache stores bytes, not filesystem paths. The
tmpfs wrapper is the single bridge from bytes-in-memory to a file
the macOS ``afplay`` / ``say`` binary can open. Storing paths in the
cache would make eviction racy with in-flight playbacks.

Thread-safety: ``VoiceCache`` acquires a single ``threading.Lock``
around every mutation and read. Voice is a single-session product so
contention is minimal — the lock exists to keep the OrderedDict
invariants under the hypothetical case of two coroutines sharing one
cache instance.

Shipped at PR alpha-2 W3. ``voice.speak`` consumes this module;
``voice.listen`` does not cache (user utterances are unique inputs
we never reconstruct).
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


# ---------- constants ----------

CACHE_EMOTION_THRESHOLD = 0.8
"""Minimum emotion_match score for a result to enter the cache.

Below this floor we generated synth that probably doesn't match the
requested tone, so serving it again from the cache would replay the
low-quality miss. The plan's JIT emotion-adaptive protocol guarantees
the next retry will produce a better match, at which point THAT
result is eligible to cache.
"""

FINGERPRINT_HEX_CHARS = 16

MAX_ENTRIES_DEFAULT = 256


# ---------- fingerprint ----------


def fingerprint(
    *,
    text: str,
    voice: str,
    emotion: str | None,
    provider: str,
) -> str:
    """Compute a stable 16-char hex key over the TTS input tuple.

    Separator ``\\x1f`` (unit separator) is ASCII control so it can't
    appear inside user-supplied text and collide with a constructed
    key. Truncating sha256 to 16 hex chars (64 bits) keeps the key
    short for log readability; collision risk at 256 entries is
    ~2^-49 which is acceptable for a cache that corrects itself on
    the next miss.
    """
    canonical = "\x1f".join([text, voice, emotion or "", provider])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[
        :FINGERPRINT_HEX_CHARS
    ]


# ---------- cache ----------


class VoiceCache:
    """Thread-safe fingerprint-keyed LRU of TTS audio bytes."""

    def __init__(self, max_entries: int = MAX_ENTRIES_DEFAULT) -> None:
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {max_entries}")
        self._data: OrderedDict[str, bytes] = OrderedDict()
        self._max_entries = int(max_entries)
        self._lock = threading.Lock()

    @property
    def max_entries(self) -> int:
        return self._max_entries

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def get(self, key: str) -> bytes | None:
        """Return cached audio and mark it most-recently-used."""
        with self._lock:
            audio = self._data.get(key)
            if audio is not None:
                self._data.move_to_end(key)
            return audio

    def put(
        self,
        key: str,
        audio: bytes,
        *,
        emotion_match: float = 1.0,
    ) -> bool:
        """Cache iff emotion_match >= CACHE_EMOTION_THRESHOLD.

        Returns True when stored, False when the threshold guard
        rejected the entry. Callers that need to distinguish 'not
        worth caching' from 'write failed' can inspect the return.
        """
        if emotion_match < CACHE_EMOTION_THRESHOLD:
            return False
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = audio
            while len(self._data) > self._max_entries:
                self._data.popitem(last=False)
        return True

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


# ---------- tmpfs playback wrapper ----------


@contextmanager
def tmpfs_audio_path(audio: bytes, *, suffix: str = ".wav") -> Iterator[Path]:
    """Yield a tmpfs path holding ``audio``; unlink on context exit.

    Uses ``$TMPDIR`` which on macOS is a per-user tmpfs mount that
    doesn't persist across reboots and has default 0o700 ancestors.
    On Linux, ``tempfile`` falls back to ``/tmp`` (also commonly
    tmpfs). On exception the ``finally`` clause still unlinks, so
    we never leave playback files behind.

    The file descriptor returned by ``mkstemp`` is closed immediately
    — consumers receive a path, not an fd. That means the OS player
    spawned to read the path sees an owned, readable file until we
    unlink, and any child that keeps the path after yield will get
    ``FileNotFoundError`` on read (intentional: defence against
    cross-invocation path replay).
    """
    fd, raw_path = tempfile.mkstemp(
        suffix=suffix,
        prefix="voice-",
        dir=os.environ.get("TMPDIR") or None,
    )
    path = Path(raw_path)
    try:
        try:
            os.write(fd, audio)
        finally:
            os.close(fd)
        yield path
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


__all__ = [
    "CACHE_EMOTION_THRESHOLD",
    "FINGERPRINT_HEX_CHARS",
    "MAX_ENTRIES_DEFAULT",
    "VoiceCache",
    "fingerprint",
    "tmpfs_audio_path",
]
