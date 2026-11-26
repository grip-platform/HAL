"""Goodhart-proof tests for audio encoding in the HAL media pipeline.

Asserts CONCRETE values: a synthetic audio file yields a non-empty OpenAI
``input_audio`` content part whose base64 round-trips to the original bytes;
empty / oversize / missing -> None; non-media files yield no parts (the
text-only path is unchanged). Mutation-killing: a stub returning ``[]``, a
``data:`` URI instead of a bare payload, or a dropped format token all fail.
"""
import base64

from lib.hal.media import (
    AUDIO_EXTS,
    _encode_audio,
    build_multimodal_content,
    file_to_content_parts,
)

# Arbitrary bytes — _encode_audio does not parse container structure, only
# size-guards + base64-encodes. The leading RIFF/WAVE is cosmetic.
_AUDIO_BYTES = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x01\x02\x03" * 16


def _write(tmp_path, name, data=_AUDIO_BYTES):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def test_encode_audio_produces_input_audio_part(tmp_path):
    part = _encode_audio(_write(tmp_path, "song.wav"))
    assert part is not None
    assert part["type"] == "input_audio"
    assert part["input_audio"]["format"] == "wav"
    # bare base64 payload (NO "data:" URI) that round-trips to the bytes
    assert not part["input_audio"]["data"].startswith("data:")
    assert base64.b64decode(part["input_audio"]["data"]) == _AUDIO_BYTES


def test_encode_audio_format_token_per_extension(tmp_path):
    for ext, fmt in [(".mp3", "mp3"), (".m4a", "m4a"), (".flac", "flac"),
                     (".ogg", "ogg"), (".aac", "aac")]:
        part = _encode_audio(_write(tmp_path, f"clip{ext}"))
        assert part is not None and part["input_audio"]["format"] == fmt


def test_every_audio_ext_is_encodable(tmp_path):
    # No silent gap: every extension the pipeline calls "audio" must encode.
    for ext in AUDIO_EXTS:
        assert _encode_audio(_write(tmp_path, f"x{ext}")) is not None


def test_file_to_content_parts_routes_audio(tmp_path):
    parts = file_to_content_parts(_write(tmp_path, "track.mp3"))
    assert len(parts) == 1 and parts[0]["type"] == "input_audio"


def test_encode_audio_rejects_empty(tmp_path):
    assert _encode_audio(_write(tmp_path, "empty.wav", b"")) is None


def test_encode_audio_rejects_oversize(tmp_path, monkeypatch):
    # Shrink the inline cap rather than writing a 20 MB fixture.
    monkeypatch.setattr("lib.hal.media.MAX_INLINE_BYTES", 4)
    assert _encode_audio(_write(tmp_path, "big.wav", b"12345")) is None


def test_encode_audio_rejects_missing(tmp_path):
    assert _encode_audio(str(tmp_path / "nope.wav")) is None


def test_build_multimodal_content_audio_then_text(tmp_path):
    parts = build_multimodal_content("describe this track",
                                     [_write(tmp_path, "song.flac")])
    assert parts[0]["type"] == "input_audio"                 # media precedes text
    assert parts[-1] == {"type": "text", "text": "describe this track"}


def test_non_media_file_unchanged(tmp_path):
    # text-only contract: a non-media extension yields no parts.
    assert file_to_content_parts(_write(tmp_path, "notes.txt", b"hello")) == []
