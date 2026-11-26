"""Tests for HAL local provider discovery."""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from lib.hal.discovery import discover_local_providers, _probe, _build_local_config
from lib.hal.types import ProviderType


class TestProbe(unittest.TestCase):

    @patch("lib.hal.discovery.urllib.request.urlopen")
    def test_healthy_server(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 200
        mock_urlopen.return_value = mock_resp

        self.assertTrue(_probe("http://localhost:11434", "/api/version", 2.0))

    @patch("lib.hal.discovery.urllib.request.urlopen", side_effect=OSError("refused"))
    def test_down_server(self, _mock):
        self.assertFalse(_probe("http://localhost:99999", "/health", 1.0))


class TestBuildLocalConfig(unittest.TestCase):

    def test_creates_local_provider(self):
        config = _build_local_config("ollama", "http://localhost:11434", "/api/tags")
        self.assertEqual(config.name, "ollama")
        self.assertEqual(config.type, ProviderType.LOCAL)
        self.assertTrue(config.auto_discover_models)


class TestDiscoverLocalProviders(unittest.TestCase):

    def test_skips_already_registered(self):
        registry = MagicMock()
        registry.get.return_value = MagicMock()  # Already registered
        result = discover_local_providers(registry)
        self.assertEqual(result, [])

    @patch("lib.hal.discovery._probe", return_value=False)
    def test_no_servers_found(self, _mock_probe):
        registry = MagicMock()
        registry.get.return_value = None
        result = discover_local_providers(registry)
        self.assertEqual(result, [])

    @patch("lib.hal.discovery._probe_server")
    def test_parallel_probes_all_four_servers(self, mock_probe_server):
        """All 4 server entries must be submitted concurrently, not sequentially."""
        import time
        from lib.hal.discovery import _LOCAL_SERVERS

        call_times: list[float] = []

        def slow_probe(entry, timeout_s):
            call_times.append(time.monotonic())
            time.sleep(0.05)  # 50ms simulated latency per probe
            return None

        mock_probe_server.side_effect = slow_probe
        registry = MagicMock()
        registry.get.return_value = None

        start = time.monotonic()
        discover_local_providers(registry, timeout_s=0.1)
        elapsed = time.monotonic() - start

        # Sequential would take >= 4 × 50ms = 200ms; parallel << 200ms
        self.assertEqual(mock_probe_server.call_count, len(_LOCAL_SERVERS),
            "Expected one probe per server entry")
        self.assertLess(elapsed, 0.18,
            f"Discovery took {elapsed:.3f}s — expected parallel (<0.18s), "
            "got sequential-like timing (regression: probes not concurrent)")

    def test_registry_init_does_not_call_backfill_synchronously(self):
        """_load_config must NOT call _backfill_auto_models — startup regression guard."""
        import ast, os
        registry_path = os.path.join(
            os.path.dirname(__file__), "..", "lib", "hal", "registry.py"
        )
        with open(registry_path) as f:
            source = f.read()
        tree = ast.parse(source)

        backfill_calls_in_load_config = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_load_config":
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        func = child.func
                        name = (
                            func.attr if isinstance(func, ast.Attribute) else
                            func.id if isinstance(func, ast.Name) else ""
                        )
                        if name == "_backfill_auto_models":
                            backfill_calls_in_load_config.append(child)

        self.assertEqual(backfill_calls_in_load_config, [],
            "_load_config() must not call _backfill_auto_models() synchronously — "
            "it blocks startup with 120s timeouts per provider")


if __name__ == "__main__":
    unittest.main()
