"""Tests for HAL unified server."""
import io
import os
import sys
import unittest
from unittest.mock import MagicMock, mock_open, patch

from lib.hal.server import (
    _make_handler,
    _default_council_models,
    _provider_display_name,
    _resolve_provider_model,
)
from lib.hal.types import ProviderInfo, ProviderType


class TestPublicPage(unittest.TestCase):
    def test_root_serves_hal_without_private_branding(self):
        handler_class = _make_handler(*(MagicMock() for _ in range(5)))
        for path in ("/", "/index.html"):
            with self.subTest(path=path):
                request = MagicMock()
                request.makefile.return_value = io.BytesIO(
                    f"GET {path} HTTP/1.0\r\nHost: hal.grip-web.com\r\n\r\n".encode()
                )
                # A local UI build must never replace the public page.
                with patch("lib.hal.server.os.path.isfile", return_value=True):
                    handler_class(request, ("203.0.113.1", 12345), MagicMock())
                response = b"".join(
                    call.args[0] for call in request.sendall.call_args_list
                ).decode()
                headers, body = response.split("\r\n\r\n", 1)
                self.assertIn("200 OK", headers)
                self.assertIn("text/html", headers)
                self.assertIn("Harness Abstraction Layer", body)
                # A name on the protected-terms list is deliberately not a literal in
                # this fixture (scripts/release_contamination_gate.py's protected_term
                # detector covers that across the whole candidate, not just this page).
                for forbidden in (
                    "decision-oriented network notarisation for attorneys",
                    "claude-opus-4-7", "raillm", "github.com/codetonight-sa/hal",
                ):
                    self.assertNotIn(forbidden, body.lower())
                self.assertIn("fetch('/api/chat'", body)
                self.assertIn("JSON.stringify({message: msg, session_id: sessionId})", body)
                self.assertIn("models from different companies the same question", body)
                self.assertIn("A model from another company judges their answers.", body)
                self.assertNotIn("sealed", body)

    def _check_ui(self, peer, *, exists=True, forwarding="", compiled=False):
        handler_class = _make_handler(*(MagicMock() for _ in range(5)))
        for path in ("/", "/index.html"):
            with self.subTest(path=path, peer=peer, forwarding=forwarding):
                request = MagicMock()
                request.makefile.return_value = io.BytesIO(
                    f"GET {path} HTTP/1.0\r\nHost: localhost\r\n{forwarding}\r\n".encode()
                )
                opened = mock_open(read_data=b"<html>Compiled operator UI</html>")
                with patch("lib.hal.server.os.path.isfile", return_value=exists), \
                        patch("builtins.open", opened):
                    handler_class(request, (peer, 12345), MagicMock())
                response = b"".join(
                    call.args[0] for call in request.sendall.call_args_list
                ).decode()
                headers, body = response.split("\r\n\r\n", 1)
                self.assertIn("200 OK", headers)
                self.assertIn("text/html", headers)
                if compiled:
                    self.assertEqual(body, "<html>Compiled operator UI</html>")
                    self.assertTrue(opened.call_args.args[0].endswith(
                        "apps/grip-web/dist/index.html"
                    ))
                    self.assertEqual(opened.call_args.args[1], "rb")
                else:
                    self.assertIn("Harness Abstraction Layer", body)
                    opened.assert_not_called()

    def test_local_caller_gets_compiled_ui(self):
        for peer in ("127.0.0.1", "::1"):
            self._check_ui(peer, compiled=True)

    def test_local_caller_without_build_gets_public_page(self):
        self._check_ui("127.0.0.1", exists=False)

    def test_remote_caller_with_build_gets_public_page(self):
        self._check_ui("203.0.113.1")

    def test_forwarded_loopback_with_build_gets_public_page(self):
        for peer in ("127.0.0.1", "::1"):
            for header in ("Forwarded", "X-Forwarded-For", "True-Client-IP",
                           "X-Real-IP", "CF-Connecting-IP", "CF-Ray"):
                self._check_ui(peer, forwarding=f"{header}: 203.0.113.1\r\n")


class TestResolveProviderModel(unittest.TestCase):
    """Test provider-to-model resolution."""

    def _make_registry(self, providers: list) -> MagicMock:
        registry = MagicMock()
        registry.list_providers.return_value = providers
        return registry

    def test_resolves_cloud_provider_with_models(self):
        providers = [
            ProviderInfo(
                name="groq", type=ProviderType.CLOUD,
                base_url="https://api.groq.com", healthy=True,
                models=("llama-3.3-70b",),
            ),
        ]
        registry = self._make_registry(providers)
        result = _resolve_provider_model(registry, "groq")
        self.assertEqual(result, "groq/llama-3.3-70b")

    def test_raises_for_unknown_provider(self):
        registry = self._make_registry([])
        with self.assertRaises(ValueError):
            _resolve_provider_model(registry, "unknown")

    @patch("lib.hal.server._get_ollama_models", return_value=["mistral:7b"])
    def test_resolves_local_via_ollama_api(self, _mock):
        providers = [
            ProviderInfo(
                name="ollama", type=ProviderType.LOCAL,
                base_url="http://localhost:11434/v1", healthy=True,
                models=(),
            ),
        ]
        registry = self._make_registry(providers)
        result = _resolve_provider_model(registry, "ollama")
        self.assertEqual(result, "ollama/mistral:7b")

    @patch("lib.hal.server._get_ollama_models", return_value=[])
    def test_raises_when_no_local_models(self, _mock):
        providers = [
            ProviderInfo(
                name="ollama", type=ProviderType.LOCAL,
                base_url="http://localhost:11434/v1", healthy=True,
                models=(),
            ),
        ]
        registry = self._make_registry(providers)
        with self.assertRaises(ValueError):
            _resolve_provider_model(registry, "ollama")


class TestDefaultCouncilModels(unittest.TestCase):
    """Test council model selection."""

    @patch("lib.hal.server._get_ollama_models", return_value=["llama3.2:3b"])
    def test_picks_from_all_providers(self, _mock):
        registry = MagicMock()
        registry.list_providers.return_value = [
            ProviderInfo(
                name="groq", type=ProviderType.CLOUD,
                base_url="", healthy=True,
                models=("llama-70b",),
            ),
            ProviderInfo(
                name="ollama", type=ProviderType.LOCAL,
                base_url="", healthy=True,
                models=(),
            ),
        ]
        models = _default_council_models(registry)
        self.assertEqual(models, ["groq/llama-70b", "ollama/llama3.2:3b"])

    def test_caps_at_five(self):
        registry = MagicMock()
        providers = [
            ProviderInfo(
                name=f"p{i}", type=ProviderType.CLOUD,
                base_url="", healthy=True,
                models=(f"model-{i}",),
            )
            for i in range(8)
        ]
        registry.list_providers.return_value = providers
        models = _default_council_models(registry)
        self.assertEqual(len(models), 5)


class TestProviderDisplayName(unittest.TestCase):
    """Test display name mapping."""

    def test_known_providers(self):
        self.assertEqual(_provider_display_name("groq", []), "Groq (Llama 3.3 70B)")
        self.assertEqual(_provider_display_name("ollama", []), "Ollama (Local)")

    def test_unknown_provider_titlecased(self):
        self.assertEqual(_provider_display_name("custom", []), "Custom")


if __name__ == "__main__":
    unittest.main()
