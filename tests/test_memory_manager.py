"""Tests for lib/hal/memory_manager.py — MemoryManager."""
import sys
import pathlib
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from lib.hal.memory_manager import MemoryManager
from lib.hal.warm_router import WarmModel


def _probe_returning(names: list[str]):
    """Return a mock OllamaLivenessProbe whose warm_models() returns WarmModel list."""
    probe = MagicMock()
    probe.warm_models.return_value = [WarmModel(name=n) for n in names]
    return probe


class TestWarmAll(unittest.TestCase):

    def test_warm_all_returns_dict(self):
        """_nudge returning True → all models map to True."""
        mm = MemoryManager(models=["m1", "m2"], probe=_probe_returning([]))
        with patch.object(mm, "_nudge", return_value=True):
            result = mm.warm_all()
        self.assertEqual(result, {"m1": True, "m2": True})

    def test_warm_all_false_on_failure(self):
        """_nudge returning False (e.g. network down) → warm_all maps model to False."""
        mm = MemoryManager(models=["bad"], probe=_probe_returning([]))
        with patch.object(mm, "_nudge", return_value=False):
            result = mm.warm_all()
        self.assertFalse(result["bad"])

    def test_warm_all_uses_explicit_list(self):
        """Explicit models list is respected; auto-discovery not called."""
        mm = MemoryManager(models=["gemma", "llama"], probe=_probe_returning([]))
        with patch.object(mm, "_nudge", return_value=True):
            result = mm.warm_all()
        self.assertSetEqual(set(result.keys()), {"gemma", "llama"})


class TestStatus(unittest.TestCase):

    def test_status_returns_warm_models(self):
        """probe.warm_models returns ['m1'] → status is {'m1': True}."""
        probe = _probe_returning(["m1"])
        mm = MemoryManager(models=["m1"], probe=probe)
        status = mm.status()
        self.assertEqual(status, {"m1": True})

    def test_status_empty_when_ollama_down(self):
        """probe.warm_models returns [] (Ollama down) → all models map to False."""
        probe = _probe_returning([])  # empty — nothing warm
        mm = MemoryManager(models=["m1"], probe=probe)
        status = mm.status()
        self.assertEqual(status, {"m1": False})


class TestKeepaliveThread(unittest.TestCase):

    def test_keepalive_thread_starts_and_stops(self):
        """start_keepalive() starts a live thread; stop() kills it."""
        mm = MemoryManager(models=["m1"], interval_s=60, probe=_probe_returning([]))
        mm.start_keepalive()
        self.assertTrue(mm._thread.is_alive())
        mm.stop()
        mm._thread.join(timeout=2)
        self.assertFalse(mm._thread.is_alive())

    def test_start_keepalive_idempotent(self):
        """Calling start_keepalive() twice leaves only one live thread."""
        mm = MemoryManager(models=["m1"], interval_s=60, probe=_probe_returning([]))
        mm.start_keepalive()
        first_thread = mm._thread
        mm.start_keepalive()
        second_thread = mm._thread
        self.assertIs(first_thread, second_thread)
        mm.stop()

    def test_cold_model_triggers_ping(self):
        """probe returns [] (nothing warm) → _nudge called for the cold model."""
        probe = _probe_returning([])
        mm = MemoryManager(models=["m1"], interval_s=0, probe=probe)
        nudge_calls = []

        def fake_nudge(name):
            nudge_calls.append(name)
            return True

        with patch.object(mm, "_nudge", side_effect=fake_nudge):
            # Manually trigger one keepalive cycle without sleeping
            warm_set = {m.name for m in probe.warm_models()}
            mm._ping_cold_models(warm_set)

        self.assertIn("m1", nudge_calls)


if __name__ == "__main__":
    unittest.main()
