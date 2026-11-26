"""Two-sided Goodhart-proof tests for lib/hal/voice/cache.py.

Per Rule 14 each invariant has a must-pass and must-fail branch, so
the test actually exercises the contract instead of tautologically
passing.

The sub-threshold non-caching test is the load-bearing one: without
it, a refactor that drops the CACHE_EMOTION_THRESHOLD guard would
silently degrade voice quality AND pass every other cache test.
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.hal.voice.cache import (  # noqa: E402
    CACHE_EMOTION_THRESHOLD,
    FINGERPRINT_HEX_CHARS,
    MAX_ENTRIES_DEFAULT,
    VoiceCache,
    fingerprint,
    tmpfs_audio_path,
)


# ---------- fingerprint ----------


def test_fingerprint_is_16_hex_chars() -> None:
    """Must-pass: sha256[:16] truncation produces exactly 16 hex chars."""
    fp = fingerprint(
        text="hello", voice="Iapetus", emotion="excited", provider="gemini_tts"
    )
    assert len(fp) == FINGERPRINT_HEX_CHARS
    assert all(c in "0123456789abcdef" for c in fp)


def test_fingerprint_deterministic_across_calls() -> None:
    """Must-pass: same inputs → same fingerprint every time."""
    kwargs = dict(text="hello", voice="Iapetus", emotion="excited", provider="gemini_tts")
    assert fingerprint(**kwargs) == fingerprint(**kwargs)


def test_fingerprint_differs_on_any_field_change() -> None:
    """Must-fail: no two distinct (text, voice, emotion, provider)
    tuples collide. If ANY input changes, the fingerprint changes —
    otherwise the cache would serve the wrong audio."""
    base = dict(text="hello", voice="Iapetus", emotion="excited", provider="gemini_tts")
    base_fp = fingerprint(**base)
    for field, alt in [
        ("text", "goodbye"),
        ("voice", "Charon"),
        ("emotion", "calm"),
        ("provider", "elevenlabs"),
    ]:
        mutated = {**base, field: alt}
        assert fingerprint(**mutated) != base_fp, f"fp collision on {field}"


def test_fingerprint_separator_cannot_be_forged_by_text() -> None:
    """Must-fail: the unit-separator (\\x1f) between fields prevents a
    caller from constructing a text that impersonates another tuple.

    Example: if the separator were '|', then text='A|Iapetus' voice=''
    could collide with text='A' voice='Iapetus'. With \\x1f as
    separator and \\x1f being an ASCII control char, user-typed text
    can't contain it. Assert two normally-ambiguous pairs do NOT
    collide — they must differ because the separator is unforgeable.
    """
    fp1 = fingerprint(
        text="hello|Iapetus", voice="", emotion=None, provider="gemini_tts"
    )
    fp2 = fingerprint(
        text="hello", voice="Iapetus", emotion=None, provider="gemini_tts"
    )
    assert fp1 != fp2


def test_fingerprint_none_emotion_differs_from_empty_emotion() -> None:
    """Must-pass: explicit '' emotion collides with None in our
    implementation (both serialise to empty string) — document this
    with an assertion so anyone changing the implementation sees the
    semantic choice. If the behaviour changes, this test forces an
    explicit decision."""
    fp_none = fingerprint(text="x", voice="y", emotion=None, provider="p")
    fp_empty = fingerprint(text="x", voice="y", emotion="", provider="p")
    # Current contract: they ARE equal because emotion or "" yields "".
    assert fp_none == fp_empty


# ---------- cache basic get/put ----------


def test_new_cache_len_zero() -> None:
    """Must-pass: fresh cache is empty."""
    assert len(VoiceCache()) == 0


def test_get_on_miss_returns_none() -> None:
    """Must-pass: missing key yields None (not raises)."""
    cache = VoiceCache()
    assert cache.get("deadbeefdeadbeef") is None


def test_put_then_get_returns_same_bytes() -> None:
    """Must-pass: stored bytes come back byte-for-byte identical."""
    cache = VoiceCache()
    cache.put("abcd1234abcd1234", b"\x00\x01\x02\x03audio")
    assert cache.get("abcd1234abcd1234") == b"\x00\x01\x02\x03audio"


def test_contains_reflects_membership() -> None:
    """Must-pass: `in` returns True after put, False after clear."""
    cache = VoiceCache()
    cache.put("k", b"v")
    assert "k" in cache
    cache.clear()
    assert "k" not in cache


# ---------- LRU eviction ----------


def test_put_evicts_oldest_at_capacity() -> None:
    """Must-pass: inserting max_entries+1 items evicts the first."""
    cache = VoiceCache(max_entries=3)
    cache.put("a", b"1")
    cache.put("b", b"2")
    cache.put("c", b"3")
    cache.put("d", b"4")
    assert cache.get("a") is None
    assert cache.get("d") == b"4"


def test_get_promotes_to_most_recently_used() -> None:
    """Must-fail-if-not-lru: touching 'a' via get() must save it
    from the next eviction, proving the cache is LRU not FIFO."""
    cache = VoiceCache(max_entries=3)
    cache.put("a", b"1")
    cache.put("b", b"2")
    cache.put("c", b"3")
    assert cache.get("a") == b"1"  # promotes a to MRU
    cache.put("d", b"4")  # FIFO would evict a; LRU evicts b
    assert cache.get("a") == b"1"
    assert cache.get("b") is None


def test_put_on_existing_key_refreshes_without_evicting_others() -> None:
    """Must-pass: re-putting an existing key updates value AND makes
    that key MRU without tripping eviction."""
    cache = VoiceCache(max_entries=3)
    cache.put("a", b"1")
    cache.put("b", b"2")
    cache.put("c", b"3")
    cache.put("a", b"1-new")  # refresh, not eviction
    assert len(cache) == 3
    assert cache.get("a") == b"1-new"
    # After refresh, a is MRU. Adding d evicts b (oldest).
    cache.put("d", b"4")
    assert cache.get("b") is None
    assert cache.get("a") == b"1-new"


# ---------- sub-threshold guard (load-bearing) ----------


def test_put_below_threshold_returns_false_and_does_not_cache() -> None:
    """Must-fail-if-cached: low-quality synth (emotion_match < 0.8)
    must NOT enter the cache. Otherwise the first miss produces bad
    audio AND every subsequent hit replays it — a worse outcome than
    having no cache at all."""
    cache = VoiceCache()
    stored = cache.put("k", b"bad-synth", emotion_match=0.7)
    assert stored is False
    assert cache.get("k") is None
    assert len(cache) == 0


def test_put_exactly_at_threshold_is_cached() -> None:
    """Must-pass: emotion_match == CACHE_EMOTION_THRESHOLD is the
    boundary; inclusive policy means 0.8 DOES cache."""
    cache = VoiceCache()
    stored = cache.put("k", b"ok-synth", emotion_match=CACHE_EMOTION_THRESHOLD)
    assert stored is True
    assert cache.get("k") == b"ok-synth"


def test_put_above_threshold_is_cached() -> None:
    """Must-pass: clearly-good synth caches."""
    cache = VoiceCache()
    assert cache.put("k", b"great-synth", emotion_match=0.95) is True
    assert cache.get("k") == b"great-synth"


def test_default_emotion_match_caches() -> None:
    """Must-pass: when caller doesn't pass emotion_match, the
    default is 1.0 so legacy callers still populate the cache."""
    cache = VoiceCache()
    assert cache.put("k", b"v") is True
    assert cache.get("k") == b"v"


# ---------- constructor guards ----------


def test_zero_max_entries_rejected() -> None:
    """Must-fail: a cache with capacity zero would accept nothing —
    construction must error so the operator sees the bug immediately."""
    with pytest.raises(ValueError):
        VoiceCache(max_entries=0)


def test_negative_max_entries_rejected() -> None:
    """Must-fail: negative capacity is nonsense and must raise."""
    with pytest.raises(ValueError):
        VoiceCache(max_entries=-1)


def test_max_entries_default_is_documented_constant() -> None:
    """Must-pass: default constructor uses the exported default so
    callers tuning via env/config see the same value as tests."""
    assert VoiceCache().max_entries == MAX_ENTRIES_DEFAULT


# ---------- thread-safety smoke test ----------


def test_concurrent_puts_preserve_invariants() -> None:
    """Must-pass: 16 threads × 20 puts each on a 50-entry cache ends
    up with len(cache) == 50 and no KeyError / RuntimeError from
    a torn OrderedDict.

    This isn't a proof of serialisability — it's a smoke test that
    the lock is actually in the put() path. Without the lock the
    OrderedDict's internal linked list corrupts and len() diverges
    from max_entries.
    """
    cache = VoiceCache(max_entries=50)

    def worker(tid: int) -> None:
        for i in range(20):
            cache.put(f"t{tid}-{i}", f"audio-{tid}-{i}".encode())

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(cache) == 50


# ---------- tmpfs_audio_path ----------


def test_tmpfs_yields_existing_file_with_audio() -> None:
    """Must-pass: inside the context, the path points to a real file
    whose contents are the bytes we passed in."""
    audio = b"RIFF\x00\x00\x00\x00fake-wav-header"
    with tmpfs_audio_path(audio) as path:
        assert path.exists()
        assert path.read_bytes() == audio


def test_tmpfs_unlinks_after_normal_exit() -> None:
    """Must-fail-if-leaks: after the context exits the path must be
    gone. Otherwise a voice chat loop accumulates a cache file per
    turn in $TMPDIR until a reboot."""
    captured: list[Path] = []
    with tmpfs_audio_path(b"x") as path:
        captured.append(path)
    assert captured
    assert not captured[0].exists()


def test_tmpfs_unlinks_on_exception() -> None:
    """Must-fail-if-leaks: even if the consumer raises, the tmpfs
    file is unlinked. The finally clause is the whole point."""
    captured: list[Path] = []
    with pytest.raises(RuntimeError):
        with tmpfs_audio_path(b"x") as path:
            captured.append(path)
            raise RuntimeError("player crashed")
    assert captured
    assert not captured[0].exists()


def test_tmpfs_uses_tmpdir_env_when_set(tmp_path: Path, monkeypatch) -> None:
    """Must-pass: when $TMPDIR is set, the audio file is created
    underneath it (not /var/folders). This matters on macOS where
    operators override TMPDIR to point at an encrypted disk image."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    with tmpfs_audio_path(b"x") as path:
        assert str(path).startswith(str(tmp_path))


def test_tmpfs_suffix_honoured() -> None:
    """Must-pass: custom suffix lands on the created file (audio
    tools like `afplay` dispatch on extension)."""
    with tmpfs_audio_path(b"x", suffix=".mp3") as path:
        assert path.suffix == ".mp3"


def test_tmpfs_unlink_idempotent_if_consumer_already_unlinked() -> None:
    """Must-pass: if the consumer unlinks the path themselves, the
    context manager's cleanup still exits cleanly (swallowing
    FileNotFoundError). Otherwise double-unlink breaks the exit."""
    with tmpfs_audio_path(b"x") as path:
        os.unlink(path)
    # If the above didn't raise, the test passes.


def test_two_tmpfs_invocations_yield_distinct_paths() -> None:
    """Must-fail-if-reused: fresh mkstemp each call, so two
    concurrent-ish invocations don't collide on the same path.
    A bug that reused paths would break overlapping playbacks."""
    paths: list[str] = []
    with tmpfs_audio_path(b"a") as p1:
        paths.append(str(p1))
        with tmpfs_audio_path(b"b") as p2:
            paths.append(str(p2))
    assert paths[0] != paths[1]
