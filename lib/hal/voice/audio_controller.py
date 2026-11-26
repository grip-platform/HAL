"""HAL Voice — AudioController: cross-process + in-process audio serialiser.

Architecture
------------
Two failure modes cause overlapping audio or stuttering:

1. Cross-process overlap — two ``hal voice say`` invocations both call
   ``afplay`` simultaneously.  Fixed by an ``fcntl.flock`` on
   ``~/.hal/state/voice/audio.lock`` held for the entire playback duration.

2. In-process race — concurrent ``speak()`` calls within one process
   (future daemon / TUI mode).  Fixed by a ``threading.Lock`` protecting the
   active ``Popen`` reference.

Streaming pipeline (sentence-chunked two-thread design)
-------------------------------------------------------
For texts longer than ``CHUNK_THRESHOLD`` characters the controller splits
input at sentence boundaries and runs two daemon threads:

  SynthThread  → synth_queue (text chunks)
               → audio_queue (audio bytes, bounded maxsize=4)

  PlayThread   → audio_queue
               → afplay Popen (one at a time, lock-protected)

While PlayThread plays sentence N, SynthThread is already synthesising
sentence N+1.  Inter-sentence silence is hidden behind synthesis latency.
For ``macos_say`` on M-series hardware synthesis runs at ~3x realtime,
so the audio_queue is never empty.

Interrupt / barge-in
--------------------
``interrupt()`` terminates the current ``afplay`` process via ``SIGTERM``,
drains both queues, and sets an ``_interrupt_event`` that causes any
in-flight synthesis result to be dropped rather than played.
``_interrupt_event`` is cleared after draining so subsequent calls work.
"""
from __future__ import annotations

import fcntl
import platform
import queue
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from .cache import tmpfs_audio_path

# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

_SENT_BREAK = re.compile(
    r'(?<=[.!?])'          # after sentence-ending punctuation
    r'(?!\s*[a-z])'        # not followed by lowercase (avoids "e.g. word")
    r'\s+'                 # whitespace separator
    r'(?=[A-Z"\'])',       # followed by uppercase or quote start
)

CHUNK_THRESHOLD = 80       # chars; shorter texts are sent whole


def split_sentences(text: str) -> list[str]:
    """Split *text* at sentence boundaries for streaming synthesis.

    Falls back to ``[text]`` for short inputs (below ``CHUNK_THRESHOLD``)
    or when no split points are found — ensures the caller always gets at
    least one chunk.
    """
    if len(text) <= CHUNK_THRESHOLD:
        return [text]
    parts = [p.strip() for p in _SENT_BREAK.split(text.strip()) if p.strip()]
    return parts if len(parts) > 1 else [text]


# ---------------------------------------------------------------------------
# Cross-process audio lock
# ---------------------------------------------------------------------------

class _AudioLock:
    """``fcntl.flock`` context manager — blocks until the audio device is free.

    Held for the duration of every ``afplay`` call.  Any concurrent process
    (e.g. a cron-fired ``hal voice say``) waits here rather than mixing audio.
    """

    def __init__(self, lock_path: Path) -> None:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = lock_path
        self._fd: Optional[int] = None

    def acquire(self) -> None:
        self._fd = open(self._path, "w")
        fcntl.flock(self._fd, fcntl.LOCK_EX)

    def release(self) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            self._fd.close()
            self._fd = None

    def __enter__(self) -> "_AudioLock":
        self.acquire()
        return self

    def __exit__(self, *_) -> None:
        self.release()


# ---------------------------------------------------------------------------
# AudioController — singleton
# ---------------------------------------------------------------------------

_HAL_STATE = Path.home() / ".hal" / "state" / "voice"


class AudioController:
    """Singleton audio serialiser.

    Obtain via ``AudioController.get()`` — creates the instance on first call
    and starts the background threads.

    Thread model:
      _synth_thread: pulls (text, synthesise_fn) from _synth_q,
                     pushes bytes onto _audio_q.
      _play_thread:  pulls bytes from _audio_q, plays via afplay Popen.
      Both are daemon threads so they do not block process exit.
    """

    _instance: Optional["AudioController"] = None
    _class_lock: threading.Lock = threading.Lock()

    @classmethod
    def get(cls) -> "AudioController":
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self._audio_lock = _AudioLock(_HAL_STATE / "audio.lock")
        self._proc_lock = threading.Lock()
        self._current_proc: Optional[subprocess.Popen] = None
        self._interrupt_event = threading.Event()

        self._synth_q: queue.Queue = queue.Queue()
        self._audio_q: queue.Queue = queue.Queue(maxsize=4)

        self._done_event = threading.Event()
        self._total_duration: float = 0.0
        self._duration_lock: threading.Lock = threading.Lock()

        self._synth_thread = threading.Thread(
            target=self._synth_loop, daemon=True, name="hal-voice-synth"
        )
        self._play_thread = threading.Thread(
            target=self._play_loop, daemon=True, name="hal-voice-play"
        )
        self._synth_thread.start()
        self._play_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def play_file(self, path: Path) -> float:
        """Play a pre-synthesised audio file, serialised across processes.

        Drop-in replacement for ``_default_player`` — same signature,
        same blocking semantics.  Returns realised playback duration.
        """
        if platform.system() != "Darwin":
            from .types import PlatformUnsupported
            raise PlatformUnsupported(
                f"AudioController.play_file is macOS-only, not {platform.system()}"
            )
        start = time.monotonic()
        with self._audio_lock:
            proc = subprocess.Popen(["afplay", str(path)])
            with self._proc_lock:
                self._current_proc = proc
            proc.wait()
            with self._proc_lock:
                self._current_proc = None
        return time.monotonic() - start

    def speak(
        self,
        text: str,
        synthesise_fn: Callable[[str], bytes],
        *,
        interrupt_current: bool = False,
    ) -> float:
        """Synthesise *text* sentence-by-sentence and play via the pipeline.

        Blocks the calling thread until all audio has been played.
        Returns total playback duration.

        ``synthesise_fn`` must accept a text string and return raw audio bytes
        (any format ``afplay`` understands — WAV/AIFF/MP3/AAC).

        ``interrupt_current=True`` terminates any in-progress audio before
        enqueueing new text (barge-in mode).
        """
        if interrupt_current:
            self.interrupt()

        chunks = split_sentences(text)
        total = len(chunks)

        self._done_event.clear()
        self._interrupt_event.clear()
        with self._duration_lock:
            self._total_duration = 0.0

        for idx, chunk in enumerate(chunks):
            self._synth_q.put((chunk, synthesise_fn, idx == total - 1))

        self._done_event.wait()
        with self._duration_lock:
            return self._total_duration

    def interrupt(self) -> None:
        """Terminate current playback and drain pending synthesis queue."""
        self._interrupt_event.set()
        with self._proc_lock:
            if self._current_proc and self._current_proc.poll() is None:
                self._current_proc.terminate()
        _drain(self._synth_q)
        _drain(self._audio_q)
        self._interrupt_event.clear()

    # ------------------------------------------------------------------
    # Background threads
    # ------------------------------------------------------------------

    def _synth_loop(self) -> None:
        while True:
            chunk, synthesise_fn, is_last = self._synth_q.get()
            if self._interrupt_event.is_set():
                if is_last:
                    self._done_event.set()
                continue
            try:
                audio_bytes = synthesise_fn(chunk)
            except Exception:
                if is_last:
                    self._done_event.set()
                continue
            self._audio_q.put((audio_bytes, is_last))

    def _play_loop(self) -> None:
        while True:
            audio_bytes, is_last = self._audio_q.get()
            if self._interrupt_event.is_set():
                if is_last:
                    self._done_event.set()
                continue
            elapsed = 0.0
            try:
                with tmpfs_audio_path(audio_bytes) as path:
                    with self._audio_lock:
                        proc = subprocess.Popen(["afplay", str(path)])
                        with self._proc_lock:
                            self._current_proc = proc
                        t0 = time.monotonic()
                        proc.wait()
                        elapsed = time.monotonic() - t0
                        with self._proc_lock:
                            self._current_proc = None
            except Exception:
                pass
            finally:
                if is_last:
                    self._done_event.set()
            with self._duration_lock:
                self._total_duration += elapsed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _drain(q: queue.Queue) -> None:
    """Empty a queue without blocking."""
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass
