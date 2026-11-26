"""mlx_vlm adapter — bridges HAL's multimodal inputs to OpenAI-format content parts.

mlx_vlm's server is OpenAI-compatible (/v1/chat/completions), so the actual
HTTP transport goes through OpenAICompatProvider unchanged.  This adapter adds
the *content assembly* layer: converting image paths, raw bytes, video frames,
and audio files into the ContentPart list that the provider sends on the wire.

Usage:
    from lib.hal.providers.mlx_vlm import MlxVlmAdapter

    adapter = MlxVlmAdapter(registry)
    response = adapter.describe_image("/tmp/screenshot.png", "What is shown?")
    response = adapter.chat_with_image(image_bytes, "Describe this")
    response = adapter.process_video("/tmp/clip.mp4", "Summarise", max_frames=4)
"""
import base64
import mimetypes
import os
from typing import Optional

from lib.hal.types import ChatMessage, ChatResponse

# MIME type mapping for common media formats
_IMAGE_MIMES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
}

_AUDIO_MIMES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
}

_VIDEO_EXTS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})

# Default mlx_vlm provider name — must match providers.yaml and registry
PROVIDER_NAME = "mlx_vlm"


class MlxVlmAdapter:
    """Content assembly layer for mlx_vlm vision/audio models.

    Converts file paths, bytes, and URLs into OpenAI-format ContentPart
    lists, then dispatches through the registry's OpenAICompatProvider.
    """

    def __init__(self, registry, provider_name: str = PROVIDER_NAME):
        self._registry = registry
        self._provider_name = provider_name

    def describe_image(
        self,
        image_path: str,
        prompt: str = "Describe this image in detail.",
        *,
        model: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        """Describe an image from a file path.

        Reads the file, base64-encodes it, and sends as a multimodal message.
        """
        data_uri = image_path_to_data_uri(image_path)
        parts = _build_vision_parts(prompt, data_uri)
        return self._chat(parts, model=model, max_tokens=max_tokens)

    def chat_with_image(
        self,
        image_data: bytes,
        prompt: str,
        *,
        mime_type: str = "image/jpeg",
        model: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        """Chat with an image from raw bytes."""
        data_uri = bytes_to_data_uri(image_data, mime_type)
        parts = _build_vision_parts(prompt, data_uri)
        return self._chat(parts, model=model, max_tokens=max_tokens)

    def chat_with_image_url(
        self,
        image_url: str,
        prompt: str,
        *,
        model: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        """Chat with an image from a URL (including data: URIs)."""
        parts = _build_vision_parts(prompt, image_url)
        return self._chat(parts, model=model, max_tokens=max_tokens)

    def process_video(
        self,
        video_path: str,
        prompt: str = "Describe what happens in this video.",
        *,
        max_frames: int = 4,
        model: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> ChatResponse:
        """Extract frames from a video file and send as multi-image query.

        Requires the ``av`` library (PyAV) for frame extraction.  Falls back
        gracefully if av is not installed.
        """
        frames = extract_video_frames(video_path, max_frames=max_frames)
        if not frames:
            raise FileNotFoundError(
                f"Could not extract frames from {video_path} "
                "(is PyAV installed? pip install av)"
            )
        parts: list[dict] = [{"type": "text", "text": prompt}]
        for frame_bytes in frames:
            uri = bytes_to_data_uri(frame_bytes, "image/jpeg")
            parts.append({"type": "image_url", "image_url": {"url": uri}})
        return self._chat(parts, model=model, max_tokens=max_tokens)

    def chat_with_audio(
        self,
        audio_path: str,
        prompt: str = "Transcribe this audio.",
        *,
        model: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> ChatResponse:
        """Send an audio file to a multimodal model that supports audio input."""
        ext = os.path.splitext(audio_path)[1].lower()
        mime = _AUDIO_MIMES.get(ext, "audio/wav")
        with open(audio_path, "rb") as f:
            raw = f.read()
        data_uri = bytes_to_data_uri(raw, mime)
        parts: list[dict] = [
            {"type": "text", "text": prompt},
            {"type": "audio_url", "audio_url": {"url": data_uri}},
        ]
        return self._chat(parts, model=model, max_tokens=max_tokens)

    def _resolve_model(self, model: Optional[str]) -> str:
        """Resolve model to use — explicit override or first registered model."""
        if model:
            return model
        config = self._registry._configs.get(self._provider_name)
        if config and config.models:
            return config.models[0]
        raise RuntimeError(f"No models available for '{self._provider_name}'")

    def _chat(
        self,
        content_parts: list[dict],
        *,
        model: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        """Send a multimodal message through the registry."""
        resolved_model = self._resolve_model(model)
        messages = [ChatMessage(role="user", content=content_parts)]
        return self._registry.chat(
            f"{self._provider_name}/{resolved_model}",
            "",
            messages=messages,
            max_tokens=max_tokens,
        )


# --- Public utility functions (usable without the adapter) ---


def image_path_to_data_uri(path: str) -> str:
    """Convert an image file path to a base64 data URI."""
    ext = os.path.splitext(path)[1].lower()
    mime = _IMAGE_MIMES.get(ext)
    if not mime:
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        raw = f.read()
    return bytes_to_data_uri(raw, mime)


def bytes_to_data_uri(data: bytes, mime_type: str) -> str:
    """Convert raw bytes to a base64 data URI string."""
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def extract_video_frames(
    video_path: str,
    max_frames: int = 4,
) -> list[bytes]:
    """Extract evenly-spaced JPEG frames from a video file.

    Returns a list of JPEG-encoded bytes, one per frame.
    Requires PyAV (``pip install av``).  Returns empty list if unavailable.
    """
    try:
        import av
    except ImportError:
        return []

    if not os.path.isfile(video_path):
        return []

    frames: list[bytes] = []
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        total = stream.frames or 100
        step = max(1, total // max_frames)

        for i, frame in enumerate(container.decode(video=0)):
            if i % step == 0:
                img = frame.to_image()
                import io
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                frames.append(buf.getvalue())
                if len(frames) >= max_frames:
                    break
        container.close()
    except Exception:
        pass
    return frames


def _build_vision_parts(prompt: str, image_url: str) -> list[dict]:
    """Build OpenAI-format content parts for a single-image vision query."""
    return [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
