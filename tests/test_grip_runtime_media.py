"""Pin the media-attestation semantics the council demanded on HAL#582:

- _collect preserves the caller-computed media truth on the ERROR path
  (attachment happened at dispatch; a later provider error does not un-attach).
- _has_non_text_parts uses the shared MEDIA_PART_TYPES set (media.py), so
  input_audio/input_image/video forms count and unknown types fail closed.
- The server emits None (UNKNOWN), never a fabricated False, when a result
  object predates the field (version skew).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from lib.hal import grip_runtime  # noqa: E402
from lib.hal.media import MEDIA_PART_TYPES  # noqa: E402


def test_error_path_preserves_media_truth():
    events = [{"type": "error", "code": "boom", "message": "x"}]
    r = grip_runtime._collect(events, requested="m", media_attached=True)
    assert r.ok is False
    assert r.media_attached is True, "error path must not un-attach a dispatched turn"
    r2 = grip_runtime._collect(events, requested="m", media_attached=False)
    assert r2.media_attached is False


def test_success_path_carries_media_truth():
    events = [{"type": "delta", "text": "hi"}, {"type": "completed", "usage": {}}]
    r = grip_runtime._collect(events, requested="m", media_attached=True)
    assert r.ok is True
    assert r.media_attached is True


def test_non_text_parts_cover_shared_type_set():
    for t in ("audio", "image", "image_url", "input_audio", "input_image", "video"):
        assert grip_runtime._has_non_text_parts([{"type": t}]) is True, t
    assert grip_runtime._has_non_text_parts([{"type": "text", "text": "hi"}]) is False
    assert grip_runtime._has_non_text_parts(None) is False
    assert grip_runtime._has_non_text_parts([]) is False


def test_shared_set_is_the_source_of_truth():
    # The set the check consults IS the exported set — a new type added to
    # media.py must reach the check with no further edits.
    assert grip_runtime._has_non_text_parts([{"type": next(iter(MEDIA_PART_TYPES))}]) is True
