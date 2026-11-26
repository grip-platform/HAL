"""Regression tests — `lembed_guard` is wired into `safe_lembed` and `hal_embed`.

H123's Phase-0a guard shipped in `lib/hal/kono/lembed_guard.py` but was not
invoked by the live lembed call sites in `substrate.py`. The 2026-04-18
20:46 SIGSEGV at ``embed_single + 308`` reproduced the exact failure mode
(pathological input) while the guard sat dead-coded on disk.

This test suite asserts that:

  1. ``safe_lembed`` calls the guard BEFORE ``conn.execute`` — a pathological
     input raises :class:`LembedInputTooRisky` and no SQL executes on the
     connection.
  2. ``hal_embed`` calls the guard BEFORE ``conn.execute`` — pathological
     input returns ``None`` (matching the existing Optional[bytes] contract)
     and no SQL executes.
  3. Safe inputs still pass through the guard and reach ``conn.execute``
     (regression against accidentally rejecting everything).

Uses ``unittest.mock.MagicMock`` for the apsw connection so the tests run
without needing the real lembed extension loaded — and crucially without
risking a SIGSEGV in the test runner if the guard ever regresses.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from lib.hal.kono.lembed_guard import LembedInputTooRisky
from lib.hal.kono.substrate import hal_embed, safe_lembed


# Pathological input that reproduces the W3 crash class.
_PATHOLOGICAL = "a" * 1100
# Safe input — short and varied enough that the guard passes.
_SAFE = "the quick brown fox jumps over the lazy dog"


class SafeLembedGuardWiring(unittest.TestCase):
    """`safe_lembed` must reject pathological inputs before touching apsw."""

    def test_pathological_input_raises_before_execute(self) -> None:
        conn = MagicMock()
        with self.assertRaises(LembedInputTooRisky):
            safe_lembed(conn, _PATHOLOGICAL)
        conn.execute.assert_not_called()

    def test_safe_input_reaches_execute(self) -> None:
        conn = MagicMock()
        conn.execute.return_value = iter([(b"\x00\x00\x00\x00",)])
        try:
            safe_lembed(conn, _SAFE)
        except Exception:
            pass  # registration / decode paths are out of scope here
        self.assertTrue(conn.execute.called, "guard should let safe input through")


class HalEmbedGuardWiring(unittest.TestCase):
    """`hal_embed` must return None on pathological inputs before touching apsw."""

    def test_pathological_input_returns_none(self) -> None:
        conn = MagicMock()
        result = hal_embed(conn, _PATHOLOGICAL)
        self.assertIsNone(result)
        conn.execute.assert_not_called()

    def test_pathological_input_uses_guard_not_registration(self) -> None:
        """Guard fires independently of model-registration state."""
        conn = MagicMock()
        fake_cursor = MagicMock()
        fake_cursor.fetchone.return_value = (b"\x00\x00\x00\x00",)
        conn.execute.return_value = fake_cursor
        result = hal_embed(conn, _PATHOLOGICAL)
        self.assertIsNone(result)
        conn.execute.assert_not_called()


class GuardThresholds(unittest.TestCase):
    """The guard's threshold band is preserved end-to-end."""

    def test_boundary_just_under_length_threshold_passes(self) -> None:
        conn = MagicMock()
        conn.execute.return_value = iter([(b"\x00\x00\x00\x00",)])
        try:
            safe_lembed(conn, "a" * 800)  # below _LEN_THRESHOLD=900
        except LembedInputTooRisky:
            self.fail("800-char all-'a' input should pass the length-gated guard")
        except Exception:
            pass  # downstream failures are not what we're testing


if __name__ == "__main__":
    unittest.main()
