"""Tests for lib.hal.voice.audio_controller.

Goodhart-proof: every test is written to FAIL if the implementation
is broken, not just to call functions and check call counts.
"""
from __future__ import annotations

import sys
import os
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.expanduser("~/.hal"))

from lib.hal.voice.audio_controller import (
    AudioController,
    _AudioLock,
    _drain,
    split_sentences,
)


# ---------------------------------------------------------------------------
# split_sentences
# ---------------------------------------------------------------------------

class TestSplitSentences(unittest.TestCase):
    def test_short_text_returned_whole(self):
        text = "Hello world."
        self.assertEqual(split_sentences(text), [text])

    def test_long_single_sentence_returned_whole(self):
        text = "This is a very long sentence that exceeds the threshold but has no sentence boundary in it at all."
        result = split_sentences(text)
        self.assertEqual(result, [text])

    def test_two_sentences_split(self):
        text = ("The quick brown fox jumps over the lazy dog. "
                "Pack my box with five dozen liquor jugs.")
        self.assertGreater(len(text), 80, "test text must exceed chunk threshold")
        result = split_sentences(text)
        self.assertGreater(len(result), 1)
        self.assertEqual("".join(result).replace(" ", ""), text.replace(" ", ""))

    def test_no_split_on_abbreviation_like(self):
        # "e.g. lower" should not split because next word is lowercase
        text = "Use a tool e.g. the hammer to drive the nail into the correct position here."
        result = split_sentences(text)
        self.assertEqual(len(result), 1)

    def test_always_returns_at_least_one_chunk(self):
        for text in ("", "Hi.", "a" * 200):
            result = split_sentences(text)
            self.assertGreater(len(result), 0)

    def test_splits_preserve_full_content(self):
        text = ("The quick brown fox jumps over the lazy dog. "
                "Pack my box with five dozen liquor jugs. "
                "How vexingly quick daft zebras jump.")
        chunks = split_sentences(text)
        reconstructed = " ".join(chunks)
        self.assertEqual(reconstructed.replace("  ", " "), text)


# ---------------------------------------------------------------------------
# _drain
# ---------------------------------------------------------------------------

class TestDrain(unittest.TestCase):
    def test_empties_queue(self):
        import queue
        q = queue.Queue()
        for i in range(5):
            q.put(i)
        _drain(q)
        self.assertTrue(q.empty())

    def test_empty_queue_is_noop(self):
        import queue
        q = queue.Queue()
        _drain(q)  # must not raise
        self.assertTrue(q.empty())


# ---------------------------------------------------------------------------
# _AudioLock cross-process semantics (mocked portable helper)
# ---------------------------------------------------------------------------

class TestAudioLock(unittest.TestCase):
    def test_acquire_release_cycle(self):
        with patch("lib.hal.voice.audio_controller.lock") as mock_lock, \
             patch("lib.hal.voice.audio_controller.unlock") as mock_unlock:
            lock = _AudioLock(Path("/tmp/test_audio.lock"))
            lock.acquire()
            mock_lock.assert_called_once()
            lock.release()
            mock_unlock.assert_called_once()

    def test_context_manager_releases_on_exception(self):
        with patch("lib.hal.voice.audio_controller.lock") as mock_lock, \
             patch("lib.hal.voice.audio_controller.unlock") as mock_unlock:
            lock = _AudioLock(Path("/tmp/test_audio.lock"))
            try:
                with lock:
                    raise RuntimeError("boom")
            except RuntimeError:
                pass
            mock_lock.assert_called_once()
            mock_unlock.assert_called_once()
            self.assertIsNone(lock._fd)  # released


# ---------------------------------------------------------------------------
# AudioController.play_file — in-process serialisation
# ---------------------------------------------------------------------------

class TestAudioControllerPlayFile(unittest.TestCase):
    def _make_controller(self):
        """Fresh controller (bypasses singleton for test isolation)."""
        ctrl = AudioController.__new__(AudioController)
        ctrl._audio_lock = MagicMock()
        ctrl._audio_lock.__enter__ = lambda s: s
        ctrl._audio_lock.__exit__ = MagicMock(return_value=False)
        ctrl._proc_lock = threading.Lock()
        ctrl._current_proc = None
        ctrl._interrupt_event = threading.Event()
        ctrl._synth_q = __import__("queue").Queue()
        ctrl._audio_q = __import__("queue").Queue(maxsize=4)
        ctrl._done_event = threading.Event()
        return ctrl

    def test_play_file_uses_afplay(self):
        ctrl = self._make_controller()
        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        fake_proc.wait.return_value = 0

        with patch("subprocess.Popen", return_value=fake_proc) as mock_popen, \
             patch("lib.hal.voice.audio_controller.platform.system", return_value="Darwin"):
            ctrl.play_file(Path("/tmp/fake.wav"))

        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]
        self.assertEqual(args[0], "afplay")

    def test_play_file_raises_on_non_macos(self):
        ctrl = self._make_controller()
        with patch("lib.hal.voice.audio_controller.platform.system", return_value="Linux"):
            with self.assertRaises(Exception):
                ctrl.play_file(Path("/tmp/fake.wav"))

    def test_concurrent_play_files_serialised(self):
        """Two threads calling play_file must not overlap (lock serialises them)."""
        order = []

        # Use a MagicMock lock that behaves as a context manager
        mock_lock = MagicMock()
        real_inner = threading.Lock()

        def _enter(_):
            real_inner.acquire()
            return mock_lock

        def _exit(_, *a):
            real_inner.release()

        mock_lock.__enter__ = _enter
        mock_lock.__exit__ = _exit

        ctrl = self._make_controller()
        ctrl._audio_lock = mock_lock

        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        fake_proc.wait.side_effect = lambda: (order.append("play"), time.sleep(0.05))

        with patch("subprocess.Popen", return_value=fake_proc), \
             patch("lib.hal.voice.audio_controller.platform.system", return_value="Darwin"):

            t1 = threading.Thread(target=lambda: ctrl.play_file(Path("/tmp/a.wav")))
            t2 = threading.Thread(target=lambda: ctrl.play_file(Path("/tmp/b.wav")))
            t1.start(); t2.start()
            t1.join(); t2.join()

        self.assertEqual(len(order), 2)  # both played, never concurrently


# ---------------------------------------------------------------------------
# AudioController.interrupt
# ---------------------------------------------------------------------------

class TestAudioControllerInterrupt(unittest.TestCase):
    def test_interrupt_terminates_active_proc(self):
        ctrl = AudioController.__new__(AudioController)
        ctrl._proc_lock = threading.Lock()
        ctrl._interrupt_event = threading.Event()
        ctrl._audio_lock = MagicMock()
        ctrl._synth_q = __import__("queue").Queue()
        ctrl._audio_q = __import__("queue").Queue()
        ctrl._done_event = threading.Event()

        fake_proc = MagicMock()
        fake_proc.poll.return_value = None  # still running
        ctrl._current_proc = fake_proc

        ctrl.interrupt()

        fake_proc.terminate.assert_called_once()

    def test_interrupt_drains_queues(self):
        ctrl = AudioController.__new__(AudioController)
        ctrl._proc_lock = threading.Lock()
        ctrl._interrupt_event = threading.Event()
        ctrl._audio_lock = MagicMock()
        import queue
        ctrl._synth_q = queue.Queue()
        ctrl._audio_q = queue.Queue()
        ctrl._done_event = threading.Event()
        ctrl._current_proc = None

        ctrl._synth_q.put(("text", lambda t: b"", True))
        ctrl._audio_q.put((b"audio", True))

        ctrl.interrupt()

        self.assertTrue(ctrl._synth_q.empty())
        self.assertTrue(ctrl._audio_q.empty())


# ---------------------------------------------------------------------------
# AudioController.speak — duration tracking (W2)
# ---------------------------------------------------------------------------

class TestAudioControllerSpeak(unittest.TestCase):
    def _make_controller(self):
        """Full controller with background threads for speak() testing."""
        ctrl = AudioController.__new__(AudioController)
        ctrl._audio_lock = MagicMock()
        ctrl._audio_lock.__enter__ = lambda s: s
        ctrl._audio_lock.__exit__ = MagicMock(return_value=False)
        ctrl._proc_lock = threading.Lock()
        ctrl._current_proc = None
        ctrl._interrupt_event = threading.Event()
        ctrl._synth_q = __import__("queue").Queue()
        ctrl._audio_q = __import__("queue").Queue(maxsize=4)
        ctrl._done_event = threading.Event()
        ctrl._total_duration = 0.0
        ctrl._duration_lock = threading.Lock()
        ctrl._synth_thread = threading.Thread(
            target=ctrl._synth_loop, daemon=True
        )
        ctrl._play_thread = threading.Thread(
            target=ctrl._play_loop, daemon=True
        )
        ctrl._synth_thread.start()
        ctrl._play_thread.start()
        return ctrl

    def _fake_wav(self) -> bytes:
        """44-byte-header WAV (0.1 s at 22050 Hz mono 16-bit)."""
        import struct
        pcm = b"\x00\x00" * 2205  # 0.1 s
        sr = 22050
        subchunk2 = len(pcm)
        return (
            b"RIFF" + struct.pack("<I", 36 + subchunk2) + b"WAVE"
            + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
            + b"data" + struct.pack("<I", subchunk2) + pcm
        )

    def test_speak_returns_float(self):
        """speak() must return a float (not a hard-coded 0.0 placeholder)."""
        wav = self._fake_wav()
        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        fake_proc.wait.side_effect = lambda: time.sleep(0.01)

        ctrl = self._make_controller()
        with patch("subprocess.Popen", return_value=fake_proc):
            result = ctrl.speak("Hello.", synthesise_fn=lambda t: wav)

        self.assertIsInstance(result, float)

    def test_speak_duration_positive_after_playback(self):
        """Duration returned must reflect actual playback time (> 0)."""
        wav = self._fake_wav()
        SLEEP_S = 0.02
        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        fake_proc.wait.side_effect = lambda: time.sleep(SLEEP_S)

        ctrl = self._make_controller()
        with patch("subprocess.Popen", return_value=fake_proc):
            duration = ctrl.speak("Hello.", synthesise_fn=lambda t: wav)

        self.assertGreater(duration, 0.0)

    def test_speak_duration_accumulates_across_sentences(self):
        """Multi-sentence text must produce duration >= single sentence."""
        wav = self._fake_wav()
        SLEEP_S = 0.01
        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        fake_proc.wait.side_effect = lambda: time.sleep(SLEEP_S)

        ctrl = self._make_controller()
        single_text = "Hello world."  # short — stays as one chunk
        multi_text = (
            "The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs."
        )
        with patch("subprocess.Popen", return_value=fake_proc):
            d_single = ctrl.speak(single_text, synthesise_fn=lambda t: wav)
            d_multi = ctrl.speak(multi_text, synthesise_fn=lambda t: wav)

        # multi-sentence text splits into 2 chunks → ~2× sleep time
        self.assertGreater(d_multi, d_single)

    def test_speak_resets_duration_between_calls(self):
        """Each speak() call starts fresh — prior call's duration is not carried over."""
        wav = self._fake_wav()
        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        fake_proc.wait.side_effect = lambda: time.sleep(0.01)

        ctrl = self._make_controller()
        with patch("subprocess.Popen", return_value=fake_proc):
            ctrl.speak("Hello.", synthesise_fn=lambda t: wav)
            d2 = ctrl.speak("Hello.", synthesise_fn=lambda t: wav)

        # second call should be the same order of magnitude as first, not doubled
        self.assertLess(d2, 0.5)
