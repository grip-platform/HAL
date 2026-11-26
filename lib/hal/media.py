"""HAL Media Pipeline — local file detection, encoding, and multimodal message construction.

Detects local file paths in user messages, reads and base64-encodes them,
and builds OpenAI-compatible ContentPart arrays for multimodal LLM APIs.

Video files are converted to key frames via ffmpeg before encoding —
local vision models (Gemma 4, Llama 3.2 Vision) process images, not raw
video bytes.

Supported formats:
  - Images: png, jpg, jpeg, webp, gif, bmp, tiff
  - Video:  mp4, mov, avi, mkv, webm (extracted to frames)
  - Audio:  mp3, wav, ogg, flac, m4a, aac

Usage:
    from lib.hal.media import detect_media_paths, build_multimodal_content

    paths = detect_media_paths("analyse this video: ~/clip.mp4")
    if paths:
        content_parts = build_multimodal_content(message, paths)
"""
import base64
import mimetypes
import os
import re
import subprocess
import tempfile

# --------------------------------------------------------------------------- #
# Supported media extensions                                                   #
# --------------------------------------------------------------------------- #
IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"})
VIDEO_EXTS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})
AUDIO_EXTS = frozenset({".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"})
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS | AUDIO_EXTS

# OpenAI-compatible ContentPart types that count as MEDIA (non-text) in a
# dispatched turn. Single source of truth for every attachment check —
# grip_runtime and the server both import this set; adding a type here updates
# every attestation site at once.
MEDIA_PART_TYPES = frozenset({
    "audio", "image", "image_url",          # OpenAI/legacy forms
    "input_audio", "input_image",           # OpenAI input-* forms
    "video", "input_video",                 # video parts (server pre-extracts frames)
})

# Max file size for inline base64 (20 MB — Gemini's inline limit).
MAX_INLINE_BYTES = 20 * 1024 * 1024

# Max frames to extract from video (keeps payload manageable).
MAX_VIDEO_FRAMES = 4

# --------------------------------------------------------------------------- #
# Path detection                                                               #
# --------------------------------------------------------------------------- #
_EXT_ALT = "|".join(sorted(e.lstrip(".") for e in MEDIA_EXTS))

_FILE_PATH_RE = re.compile(
    rf"""(                        # capture group
        (?:/[^\x00-\x1f:*?"<>|,]+)   # absolute path segments
        \.(?:{_EXT_ALT})         # media extension
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def detect_media_paths(message: str) -> list[str]:
    """Extract local file paths from *message* that point to existing media files."""
    seen: set[str] = set()
    paths: list[str] = []
    for match in _FILE_PATH_RE.finditer(message):
        raw = match.group(1).strip()
        resolved = os.path.expanduser(raw)
        if resolved in seen:
            continue
        ext = os.path.splitext(resolved)[1].lower()
        if ext in MEDIA_EXTS and os.path.isfile(resolved):
            seen.add(resolved)
            paths.append(resolved)
    return paths


# --------------------------------------------------------------------------- #
# Video frame extraction                                                       #
# --------------------------------------------------------------------------- #

def _get_video_duration(path: str) -> float:
    """Get video duration in seconds via ffprobe. Returns 0.0 on failure."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", "--", path],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip()) if result.returncode == 0 else 0.0
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        return 0.0


def extract_video_frames(path: str, max_frames: int = MAX_VIDEO_FRAMES) -> list[str]:
    """Extract evenly-spaced key frames from a video file using ffmpeg.

    Returns a list of temporary JPEG file paths. Caller should clean up
    the temp directory after encoding.
    """
    duration = _get_video_duration(path)
    if duration <= 0:
        return []

    tmpdir = tempfile.mkdtemp(prefix="hal_frames_")

    # Calculate FPS to get ~max_frames evenly spaced
    fps = max_frames / duration if duration > max_frames else 1.0

    try:
        subprocess.run(
            ["ffmpeg", "-i", path, "-vf", f"fps={fps:.4f}", "-frames:v",
             str(max_frames), "-q:v", "2", "-y",
             os.path.join(tmpdir, "frame_%03d.jpg")],
            capture_output=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []

    frames = sorted(
        os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".jpg")
    )
    return frames[:max_frames]


# --------------------------------------------------------------------------- #
# Encoding                                                                     #
# --------------------------------------------------------------------------- #

def _encode_image(path: str) -> dict | None:
    """Base64-encode an image file into an OpenAI image_url ContentPart."""
    if not os.path.isfile(path):
        return None
    size = os.path.getsize(path)
    if size == 0 or size > MAX_INLINE_BYTES:
        return None
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        return None

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")

    data_uri = f"data:{mime};base64,{encoded}"
    return {"type": "image_url", "image_url": {"url": data_uri}}


# Audio extension -> OpenAI ``input_audio.format`` token. Unlike images
# (a ``data:`` URI), the OpenAI audio content part carries a BARE base64
# payload plus a short format token (no ``audio/`` prefix, no ``data:``
# scheme). The extension stem is the token for every supported codec.
_AUDIO_FORMATS = {
    ".mp3": "mp3", ".wav": "wav", ".ogg": "ogg",
    ".flac": "flac", ".m4a": "m4a", ".aac": "aac",
}


def _encode_audio(path: str) -> dict | None:
    """Base64-encode an audio file into an OpenAI ``input_audio`` ContentPart.

    Mirrors :func:`_encode_image`: rejects a missing, empty, or oversized file
    (``MAX_INLINE_BYTES`` — Gemini's inline cap). The OpenAI-compatible audio
    shape is ``{"type": "input_audio", "input_audio": {"data": <b64>,
    "format": <token>}}``. The provider layer passes list-form ``content``
    through as-is (see :class:`~lib.hal.types.ChatMessage`), so a Gemini
    OpenAI-compat endpoint receives the audio inline — the same rail images
    already ride.
    """
    if not os.path.isfile(path):
        return None
    size = os.path.getsize(path)
    if size == 0 or size > MAX_INLINE_BYTES:
        return None
    fmt = _AUDIO_FORMATS.get(os.path.splitext(path)[1].lower())
    if not fmt:
        return None

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")

    return {"type": "input_audio", "input_audio": {"data": encoded, "format": fmt}}


def file_to_content_parts(path: str) -> list[dict]:
    """Convert a local file to one or more OpenAI-compatible ContentPart dicts.

    - Images: single image_url part with base64 data URI
    - Video: extracted key frames as multiple image_url parts
    - Audio: single input_audio part (base64 payload + format token)
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in IMAGE_EXTS:
        part = _encode_image(path)
        return [part] if part else []

    if ext in AUDIO_EXTS:
        part = _encode_audio(path)
        return [part] if part else []

    if ext in VIDEO_EXTS:
        frames = extract_video_frames(path)
        parts = []
        for frame_path in frames:
            part = _encode_image(frame_path)
            if part:
                parts.append(part)
        # Clean up temp frames
        for frame_path in frames:
            try:
                os.remove(frame_path)
            except OSError:
                pass
        if frames:
            tmpdir = os.path.dirname(frames[0])
            try:
                os.rmdir(tmpdir)
            except OSError:
                pass
        return parts

    return []


# --------------------------------------------------------------------------- #
# Multimodal message construction                                              #
# --------------------------------------------------------------------------- #

def build_multimodal_content(message: str, media_paths: list[str]) -> list[dict]:
    """Build a ``ContentPart`` list from a text message and media file paths.

    Media parts are placed before the text part so the model "sees" the media
    before reading the question. Video files are expanded to key frames.

    Returns an empty list if no media could be loaded.
    """
    parts: list[dict] = []
    for path in media_paths:
        parts.extend(file_to_content_parts(path))

    if not parts:
        return []

    parts.append({"type": "text", "text": message})
    return parts


def media_summary(message: str, media_paths: list[str]) -> str:
    """Create a text-only summary for conversation history.

    Avoids storing base64 data in history (re-sent on every subsequent turn).
    """
    names = [os.path.basename(p) for p in media_paths]
    return f"[Attached: {', '.join(names)}] {message}"
