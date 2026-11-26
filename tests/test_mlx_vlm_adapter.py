"""Tests for the mlx_vlm adapter — content assembly + registry dispatch.

Verifies that the adapter correctly converts file paths, raw bytes, and URLs
into OpenAI-format ContentPart dicts and dispatches them through the registry.
"""
import base64
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from lib.hal.providers.mlx_vlm import (
    MlxVlmAdapter,
    bytes_to_data_uri,
    image_path_to_data_uri,
    _build_vision_parts,
    _IMAGE_MIMES,
    _AUDIO_MIMES,
    PROVIDER_NAME,
)
from lib.hal.types import ChatMessage, ChatResponse, TokenUsage


def _make_registry(models=None):
    """Build a mock registry with mlx_vlm registered."""
    registry = MagicMock()
    config = MagicMock()
    config.models = models if models is not None else ["mlx-community/gemma-4-27b-it-4bit"]
    registry._configs = {"mlx_vlm": config}
    registry.chat.return_value = ChatResponse(
        text="A cat sitting on a mat",
        model="mlx-community/gemma-4-27b-it-4bit",
        provider="mlx_vlm",
        usage=TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
    )
    return registry


class TestBytesToDataUri(unittest.TestCase):
    """Unit tests for the bytes_to_data_uri utility."""

    def test_returns_valid_data_uri(self):
        data = b"\x89PNG\r\n\x1a\n"
        uri = bytes_to_data_uri(data, "image/png")
        self.assertTrue(uri.startswith("data:image/png;base64,"))

    def test_roundtrip_base64(self):
        original = b"hello world"
        uri = bytes_to_data_uri(original, "text/plain")
        b64_part = uri.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        self.assertEqual(decoded, original)

    def test_jpeg_mime_type(self):
        uri = bytes_to_data_uri(b"\xff\xd8\xff", "image/jpeg")
        self.assertIn("image/jpeg", uri)


class TestImagePathToDataUri(unittest.TestCase):
    """Unit tests for image_path_to_data_uri."""

    def test_png_file(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
            path = f.name
        try:
            uri = image_path_to_data_uri(path)
            self.assertTrue(uri.startswith("data:image/png;base64,"))
        finally:
            os.unlink(path)

    def test_jpeg_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff" + b"\x00" * 100)
            path = f.name
        try:
            uri = image_path_to_data_uri(path)
            self.assertTrue(uri.startswith("data:image/jpeg;base64,"))
        finally:
            os.unlink(path)

    def test_webp_file(self):
        with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as f:
            f.write(b"RIFF" + b"\x00" * 100)
            path = f.name
        try:
            uri = image_path_to_data_uri(path)
            self.assertTrue(uri.startswith("data:image/webp;base64,"))
        finally:
            os.unlink(path)

    def test_nonexistent_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            image_path_to_data_uri("/nonexistent/image.png")


class TestBuildVisionParts(unittest.TestCase):
    """Unit tests for _build_vision_parts content assembly."""

    def test_returns_text_and_image_parts(self):
        parts = _build_vision_parts("Describe this", "data:image/png;base64,abc")
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0]["type"], "text")
        self.assertEqual(parts[0]["text"], "Describe this")
        self.assertEqual(parts[1]["type"], "image_url")
        self.assertEqual(parts[1]["image_url"]["url"], "data:image/png;base64,abc")

    def test_preserves_url_format(self):
        url = "https://example.com/image.jpg"
        parts = _build_vision_parts("What is this?", url)
        self.assertEqual(parts[1]["image_url"]["url"], url)


class TestMlxVlmAdapter(unittest.TestCase):
    """Integration tests for the MlxVlmAdapter."""

    def test_describe_image_dispatches_to_registry(self):
        registry = _make_registry()
        adapter = MlxVlmAdapter(registry)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG" + b"\x00" * 50)
            path = f.name
        try:
            result = adapter.describe_image(path, "What is this?")
            self.assertEqual(result.text, "A cat sitting on a mat")
            self.assertEqual(result.provider, "mlx_vlm")
            # Verify registry.chat was called with mlx_vlm address
            call_args = registry.chat.call_args
            self.assertTrue(call_args[0][0].startswith("mlx_vlm/"))
        finally:
            os.unlink(path)

    def test_chat_with_image_bytes(self):
        registry = _make_registry()
        adapter = MlxVlmAdapter(registry)
        result = adapter.chat_with_image(b"\xff\xd8\xff", "Describe")
        self.assertEqual(result.text, "A cat sitting on a mat")
        call_args = registry.chat.call_args
        messages = call_args[1].get("messages") or call_args[0][2] if len(call_args[0]) > 2 else None
        # Verify multimodal message was constructed
        registry.chat.assert_called_once()

    def test_chat_with_image_url(self):
        registry = _make_registry()
        adapter = MlxVlmAdapter(registry)
        result = adapter.chat_with_image_url(
            "data:image/jpeg;base64,/9j",
            "What colour?",
        )
        self.assertEqual(result.provider, "mlx_vlm")

    def test_resolve_model_uses_first_registered(self):
        registry = _make_registry(models=["gemma-4-27b-it-4bit", "qwen2-vl-7b"])
        adapter = MlxVlmAdapter(registry)
        model = adapter._resolve_model(None)
        self.assertEqual(model, "gemma-4-27b-it-4bit")

    def test_resolve_model_explicit_override(self):
        registry = _make_registry()
        adapter = MlxVlmAdapter(registry)
        model = adapter._resolve_model("custom-model")
        self.assertEqual(model, "custom-model")

    def test_resolve_model_no_models_raises(self):
        registry = _make_registry(models=[])
        adapter = MlxVlmAdapter(registry)
        with self.assertRaises(RuntimeError):
            adapter._resolve_model(None)

    def test_chat_with_audio(self):
        registry = _make_registry()
        adapter = MlxVlmAdapter(registry)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF" + b"\x00" * 100)
            path = f.name
        try:
            result = adapter.chat_with_audio(path, "Transcribe this")
            self.assertEqual(result.provider, "mlx_vlm")
            registry.chat.assert_called_once()
        finally:
            os.unlink(path)

    def test_process_video_without_av_raises(self):
        """process_video raises FileNotFoundError when PyAV is unavailable."""
        registry = _make_registry()
        adapter = MlxVlmAdapter(registry)
        with self.assertRaises(FileNotFoundError):
            adapter.process_video("/nonexistent/clip.mp4")


class TestMimeMaps(unittest.TestCase):
    """Verify MIME type mappings are correct."""

    def test_image_mimes_cover_common_formats(self):
        for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"):
            self.assertIn(ext, _IMAGE_MIMES, f"Missing MIME for {ext}")

    def test_audio_mimes_cover_common_formats(self):
        for ext in (".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"):
            self.assertIn(ext, _AUDIO_MIMES, f"Missing MIME for {ext}")


if __name__ == "__main__":
    unittest.main()
