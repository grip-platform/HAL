"""Regression tests for HAL startup speed and warmup feedback visibility.

Two confirmed regressions guarded here:
1. _PROBE_TIMEOUT_S was 2.0 — caused ~2s blank wait on every cold start when
   no local servers run. Fixed: reduced to 0.5s (localhost responds <50ms or not).
2. warmup verbose flag was args["status"] instead of not args["status"]:
   warmup messages only appeared on `hal status`, never on normal startup.
   Fixed in d40c4790.
"""
import io
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

from lib.hal.discovery import _PROBE_TIMEOUT_S, discover_local_providers
from lib.hal.warmup import warmup_background


class TestDiscoveryTimeout(unittest.TestCase):
    """Guard against probe timeout regression."""

    def test_probe_timeout_is_fast(self):
        """_PROBE_TIMEOUT_S must be <= 0.5s.

        Increasing it silently makes every cold start slower. Localhost either
        responds in <50ms or it is not running — 0.5s is generous enough.
        """
        self.assertLessEqual(
            _PROBE_TIMEOUT_S, 0.5,
            f"_PROBE_TIMEOUT_S is {_PROBE_TIMEOUT_S} — must be <= 0.5 to avoid "
            "blank startup wait. See discovery.py.",
        )


class TestDiscoverySpeed(unittest.TestCase):
    """Guard against synchronous startup blocking."""

    @patch("lib.hal.discovery._probe_server", return_value=None)
    def test_completes_under_one_second_when_no_servers(self, _mock):
        """discover_local_providers must return in < 1s when all probes fail.

        With _PROBE_TIMEOUT_S=0.5 and 4 parallel workers, worst case is ~0.5s.
        This test fails if the timeout reverts to 2.0 (would take ~2s).
        """
        registry = MagicMock()
        start = time.monotonic()
        discover_local_providers(registry, timeout_s=_PROBE_TIMEOUT_S)
        elapsed = time.monotonic() - start
        self.assertLess(
            elapsed, 1.0,
            f"Discovery took {elapsed:.2f}s — exceeds 1s budget. "
            "Check _PROBE_TIMEOUT_S in discovery.py.",
        )


class TestWarmupVerboseFlag(unittest.TestCase):
    """Guard against verbose flag regression (was inverted before d40c4790)."""

    def test_warmup_silent_when_no_models(self):
        """With verbose=True but Ollama down, warmup produces no output."""
        buf = io.StringIO()
        registry = MagicMock()
        with patch("lib.hal.warmup._discover_local_models", return_value=[]):
            with patch("sys.stdout", buf):
                t = warmup_background(registry, verbose=True)
                t.join(timeout=1.0)
        self.assertEqual(buf.getvalue(), "")

    def test_warmup_prints_when_models_present(self):
        """verbose=True + models discovered → 'Warming up N local model(s)...' printed."""
        buf = io.StringIO()
        registry = MagicMock()
        with patch("lib.hal.warmup._discover_local_models",
                   return_value=[("ollama", "llama3.2:3b")]):
            with patch("lib.hal.warmup._ping_model", return_value=42.0):
                with patch("sys.stdout", buf):
                    t = warmup_background(registry, verbose=True)
                    t.join(timeout=2.0)
        output = buf.getvalue()
        self.assertIn("Warming up", output)
        self.assertIn("warm", output)

    def test_warmup_silent_when_verbose_false(self):
        """verbose=False must produce no output even when models are present."""
        buf = io.StringIO()
        registry = MagicMock()
        with patch("lib.hal.warmup._discover_local_models",
                   return_value=[("ollama", "llama3.2:3b")]):
            with patch("lib.hal.warmup._ping_model", return_value=42.0):
                with patch("sys.stdout", buf):
                    t = warmup_background(registry, verbose=False)
                    t.join(timeout=2.0)
        self.assertEqual(buf.getvalue(), "")


class TestBootProvidersProgressMessage(unittest.TestCase):
    """Guard: _boot_providers must print discovery progress to TTY, not to pipes."""

    def test_discovery_progress_printed_to_tty(self):
        """When stdout is a TTY, 'Discovering local providers...' must be written."""
        from lib.hal.cli import _boot_providers

        written = []
        registry = MagicMock()

        with patch("lib.hal.cli.get_registry", return_value=registry), \
             patch("lib.hal.cli.discover_local_providers", return_value=[]), \
             patch("lib.hal.cli.warmup_background"), \
             patch("sys.stdout.isatty", return_value=True), \
             patch("sys.stdout.write", side_effect=lambda s: written.append(s)), \
             patch("sys.stdout.flush"):
            _boot_providers({"status": False})

        combined = "".join(written)
        self.assertIn("Discovering", combined,
                      "No 'Discovering local providers...' message on TTY")

    def test_no_discovery_message_when_piped(self):
        """When stdout is not a TTY (piped), no discovery progress line is written."""
        from lib.hal.cli import _boot_providers

        written = []
        registry = MagicMock()

        with patch("lib.hal.cli.get_registry", return_value=registry), \
             patch("lib.hal.cli.discover_local_providers", return_value=[]), \
             patch("lib.hal.cli.warmup_background"), \
             patch("sys.stdout.isatty", return_value=False), \
             patch("sys.stdout.write", side_effect=lambda s: written.append(s)), \
             patch("sys.stdout.flush"):
            _boot_providers({"status": False})

        combined = "".join(written)
        self.assertNotIn("Discovering", combined,
                         "Discovery progress should not appear in piped output")
