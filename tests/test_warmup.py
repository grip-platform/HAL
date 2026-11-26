"""Tests for HAL warmup module."""
import os
import sys
import threading
import unittest
from unittest.mock import MagicMock, patch

from lib.hal.warmup import (
    _discover_local_models,
    _get_ollama_models,
    _ping_model,
    warmup_background,
)


class TestGetOllamaModels(unittest.TestCase):
    """Test Ollama model discovery."""

    @patch("lib.hal.warmup.urllib.request.urlopen")
    def test_returns_model_names(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b'{"models":[{"name":"llama3.2:3b"},{"name":"mistral:7b"}]}'
        mock_urlopen.return_value = mock_resp

        result = _get_ollama_models()
        self.assertEqual(result, ["llama3.2:3b", "mistral:7b"])

    @patch("lib.hal.warmup.urllib.request.urlopen", side_effect=OSError("refused"))
    def test_returns_empty_on_connection_error(self, _mock):
        result = _get_ollama_models()
        self.assertEqual(result, [])

    @patch("lib.hal.warmup.urllib.request.urlopen")
    def test_returns_empty_on_bad_json(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b"not json"
        mock_urlopen.return_value = mock_resp

        result = _get_ollama_models()
        self.assertEqual(result, [])


class TestDiscoverLocalModels(unittest.TestCase):
    """Test local model discovery aggregation."""

    @patch("lib.hal.warmup._get_ollama_models", return_value=["model-a", "model-b"])
    def test_discovers_ollama_models(self, _mock):
        result = _discover_local_models()
        self.assertEqual(result, [("ollama", "model-a"), ("ollama", "model-b")])

    @patch("lib.hal.warmup._get_ollama_models", return_value=[])
    def test_empty_when_no_ollama(self, _mock):
        result = _discover_local_models()
        self.assertEqual(result, [])


class TestPingModel(unittest.TestCase):
    """Test model ping for warmup."""

    def test_returns_latency_on_success(self):
        registry = MagicMock()
        registry.chat.return_value = MagicMock()
        latency = _ping_model(registry, "ollama/test")
        self.assertIsNotNone(latency)
        self.assertGreaterEqual(latency, 0)
        registry.chat.assert_called_once_with("ollama/test", "hi", max_tokens=1)

    def test_returns_none_on_failure(self):
        registry = MagicMock()
        registry.chat.side_effect = ConnectionError("refused")
        latency = _ping_model(registry, "ollama/test")
        self.assertIsNone(latency)


class TestWarmupBackground(unittest.TestCase):
    """Test background warmup thread."""

    @patch("lib.hal.warmup._discover_local_models", return_value=[])
    def test_returns_daemon_thread(self, _mock):
        registry = MagicMock()
        thread = warmup_background(registry, verbose=False)
        self.assertIsInstance(thread, threading.Thread)
        self.assertTrue(thread.daemon)
        self.assertEqual(thread.name, "hal-warmup")
        thread.join(timeout=2)

    @patch("lib.hal.warmup._discover_local_models", return_value=[("ollama", "llama3:8b")])
    @patch("lib.hal.warmup._ping_model", return_value=42.0)
    def test_verbose_true_emits_messages(self, _mock_ping, _mock_discover):
        """verbose=True (the interactive default) must print warmup output."""
        import io
        registry = MagicMock()
        buf = io.StringIO()
        with patch("builtins.print") as mock_print:
            from lib.hal.warmup import _warmup_all
            _warmup_all(registry, verbose=True)
        # At least the "Warming up" header and one model completion line
        self.assertGreaterEqual(mock_print.call_count, 2)

    @patch("lib.hal.warmup._discover_local_models", return_value=[("ollama", "llama3:8b")])
    @patch("lib.hal.warmup._ping_model", return_value=42.0)
    def test_verbose_false_emits_nothing(self, _mock_ping, _mock_discover):
        """verbose=False must produce no print output."""
        registry = MagicMock()
        with patch("builtins.print") as mock_print:
            from lib.hal.warmup import _warmup_all
            _warmup_all(registry, verbose=False)
        mock_print.assert_not_called()

    def test_cli_verbose_flag_is_inverted_status(self):
        """cli.py must pass verbose=not args['status'] — interactive mode gets True."""
        import ast, os
        cli_path = os.path.join(os.path.dirname(__file__), "..", "lib", "hal", "cli.py")
        with open(cli_path) as f:
            source = f.read()
        # The call must use 'not args["status"]' not 'args["status"]'
        self.assertIn('verbose=not args["status"]', source,
            "cli.py warmup_background must use verbose=not args['status'] "
            "so interactive mode shows warmup messages (regression guard)")
        self.assertNotIn('verbose=args["status"]', source,
            "cli.py has the inverted verbose bug — warmup silent in interactive mode")


if __name__ == "__main__":
    unittest.main()
