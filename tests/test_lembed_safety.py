"""Regression tests for issue #199 — lembed NULL deref segfault.

Scope:
  * `_ensure_lembed_registered` is idempotent (no duplicate INSERT).
  * `_ensure_lembed_registered` raises `LembedNotRegisteredError` when the
    extension is not loaded.
  * `safe_lembed` surfaces the typed exception when lembed SQL fails.
  * `KonoLifecycleMixin.connect` invokes `_ensure_lembed_registered`
    after loading the lembed extension (Goodhart guard — test fails if
    the connect() wiring is removed).
  * Live concurrency test (HAL_TEST_LIVE=1): three subprocesses opening
    the shared embeddings.db all embed without crashing.
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import apsw

from lib.hal.kono.substrate import (
    LembedNotRegisteredError,
    _LEMBED_MODEL_NAME,
    DEFAULT_LEMBED_MODEL,
    _ensure_lembed_registered,
    safe_lembed,
)


# ---------------------------------------------------------------------------
# Fake apsw connection — records every SQL call so we can assert behaviour
# without needing the real lembed extension loaded.
# ---------------------------------------------------------------------------

class _FakeConn:
    """Minimal apsw.Connection stand-in.

    ``on_select`` controls what SELECT queries return. ``calls`` records
    every execute() invocation so tests can assert call shapes. ``raise_on``
    triggers apsw.SQLError for specific SQL substrings (simulating the
    extension-not-loaded case where `temp.lembed_models` doesn't exist).
    """

    def __init__(self, on_select=None, raise_on=None):
        self.calls: list[tuple[str, tuple]] = []
        self._on_select = on_select or (lambda sql, params: [])
        self._raise_on = raise_on or []

    def execute(self, sql: str, params: tuple = ()):  # noqa: D401
        self.calls.append((sql.strip(), params))
        for substr in self._raise_on:
            if substr in sql:
                raise apsw.SQLError(f"simulated error: {substr}")
        return self._on_select(sql, params)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestEnsureLembedRegistered(unittest.TestCase):
    """Contract of `_ensure_lembed_registered(conn)` — PR #208 version.

    The function now returns bool and is health-gated: it calls
    ``is_lembed_healthy()`` internally and silently returns False on any
    SQL error rather than raising. Tests use patch to bypass the health
    check and the model-file existence check.
    """

    def _call(self, conn, healthy=True, model_exists=True):
        with patch("lib.hal.kono.substrate.is_lembed_healthy", return_value=healthy), \
             patch("lib.hal.kono.substrate.MODEL_PATH") as mp:
            mp.exists.return_value = model_exists
            mp.__str__ = lambda self: "/fake/model.gguf"
            return _ensure_lembed_registered(conn)

    def test_returns_false_when_lembed_unhealthy(self):
        """Health check fails — return False without touching conn."""
        conn = _FakeConn()
        result = self._call(conn, healthy=False)
        self.assertFalse(result)
        self.assertEqual(len(conn.calls), 0)

    def test_returns_false_when_model_missing(self):
        """Model file absent — return False without touching conn."""
        conn = _FakeConn()
        result = self._call(conn, model_exists=False)
        self.assertFalse(result)
        self.assertEqual(len(conn.calls), 0)

    def test_registers_when_missing(self):
        """If temp.lembed_models has no matching row, issue the INSERT."""
        conn = _FakeConn(on_select=lambda sql, params: [])  # empty SELECT
        result = self._call(conn)

        self.assertTrue(result)
        self.assertEqual(len(conn.calls), 2)
        sel_sql, sel_params = conn.calls[0]
        self.assertIn("SELECT 1 FROM temp.lembed_models", sel_sql)
        self.assertEqual(sel_params, (DEFAULT_LEMBED_MODEL,))

        ins_sql, ins_params = conn.calls[1]
        self.assertIn("INSERT INTO temp.lembed_models", ins_sql)
        self.assertIn("lembed_model_from_file", ins_sql)
        self.assertEqual(ins_params[0], DEFAULT_LEMBED_MODEL)

    def test_idempotent_when_already_registered(self):
        """If the SELECT returns a row, no INSERT is issued."""
        conn = _FakeConn(on_select=lambda sql, params: [(1,)])
        result = self._call(conn)

        self.assertTrue(result)
        self.assertEqual(len(conn.calls), 1)
        self.assertIn("SELECT", conn.calls[0][0])

    def test_returns_false_on_sql_error(self):
        """SQL error during registration -> returns False (no raise)."""
        conn = _FakeConn(raise_on=["temp.lembed_models"])
        result = self._call(conn)
        self.assertFalse(result)


class TestSafeLembed(unittest.TestCase):
    """Contract of `safe_lembed(conn, text)`."""

    def test_returns_float_list_on_success(self):
        # Simulate lembed returning a 4-float blob
        blob = struct.pack("4f", 0.1, 0.2, 0.3, 0.4)

        def on_select(sql, params):
            if "SELECT 1 FROM temp.lembed_models" in sql:
                return [(1,)]  # already registered
            if "SELECT lembed" in sql:
                return [(blob,)]
            return []

        conn = _FakeConn(on_select=on_select)
        result = safe_lembed(conn, "hello")
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(result[0], 0.1, places=5)
        self.assertAlmostEqual(result[3], 0.4, places=5)

    def test_raises_when_lembed_returns_null(self):
        def on_select(sql, params):
            if "SELECT 1 FROM temp.lembed_models" in sql:
                return [(1,)]
            if "SELECT lembed" in sql:
                return [(None,)]
            return []

        conn = _FakeConn(on_select=on_select)
        with self.assertRaises(LembedNotRegisteredError) as cm:
            safe_lembed(conn, "hello")
        self.assertIn("returned NULL", str(cm.exception))

    def test_raises_when_registration_fails(self):
        conn = _FakeConn(raise_on=["temp.lembed_models"])
        with self.assertRaises(LembedNotRegisteredError):
            safe_lembed(conn, "hello")


class TestConnectCallsRegister(unittest.TestCase):
    """Goodhart guard — if connect() stops calling _ensure_lembed_registered,
    the NULL-deref protection is gone. This test fails in that regression.
    """

    def test_connect_invokes_registration_when_lembed_healthy(self):
        """Mock out apsw + sqlite_vec + sqlite_lembed, verify the wiring."""
        from lib.hal.kono.substrate import MemorySubstrate

        sub = MemorySubstrate()
        sub.conn = None

        fake_conn = MagicMock()
        fake_conn.execute = MagicMock(return_value=[])

        with patch("apsw.Connection", return_value=fake_conn), \
             patch("lib.hal.kono.substrate.is_lembed_healthy", return_value=True), \
             patch("lib.hal.kono.substrate._ensure_lembed_registered") as mock_reg, \
             patch.dict(sys.modules, {
                 "sqlite_vec": MagicMock(loadable_path=lambda: "/fake/vec"),
                 "sqlite_lembed": MagicMock(loadable_path=lambda: "/fake/lembed"),
             }):
            sub.connect()

        mock_reg.assert_called_once_with(fake_conn)

    def test_connect_skips_registration_when_lembed_unhealthy(self):
        """If lembed probe failed, don't try to register — it would error."""
        from lib.hal.kono.substrate import MemorySubstrate

        sub = MemorySubstrate()
        sub.conn = None

        fake_conn = MagicMock()

        with patch("apsw.Connection", return_value=fake_conn), \
             patch("lib.hal.kono.substrate.is_lembed_healthy", return_value=False), \
             patch("lib.hal.kono.substrate._ensure_lembed_registered") as mock_reg, \
             patch.dict(sys.modules, {
                 "sqlite_vec": MagicMock(loadable_path=lambda: "/fake/vec"),
             }):
            sub.connect()

        mock_reg.assert_not_called()


# ---------------------------------------------------------------------------
# Live concurrency regression (gated on HAL_TEST_LIVE=1)
# ---------------------------------------------------------------------------

class TestLembedConcurrency(unittest.TestCase):
    """The original crash shape — 3 processes opening embeddings.db and
    embedding in parallel. Before the fix, the third process would segfault
    with NULL deref in embed_single. After the fix, all three succeed.

    Gated on HAL_TEST_LIVE=1 because it requires the real lembed extension
    and the GGUF model file to be present. CI without a macOS runner skips.
    """

    def setUp(self):
        if os.environ.get("HAL_TEST_LIVE") != "1":
            self.skipTest("HAL_TEST_LIVE=1 required for live lembed test")

        from lib.hal.kono.constants import MODEL_PATH
        if not MODEL_PATH.exists():
            self.skipTest(f"GGUF model not present at {MODEL_PATH}")

    def _spawn_worker(self, db_path: str) -> subprocess.CompletedProcess:
        """Spawn a subprocess that opens the DB, calls safe_lembed, prints OK."""
        script = (
            "import apsw, sqlite_vec, sqlite_lembed; "
            "from lib.hal.kono.substrate import _ensure_lembed_registered, safe_lembed; "
            f"c = apsw.Connection({db_path!r}); "
            "c.enableloadextension(True); "
            "c.loadextension(sqlite_vec.loadable_path()); "
            "c.loadextension(sqlite_lembed.loadable_path()); "
            "_ensure_lembed_registered(c); "
            "v = safe_lembed(c, 'concurrency test'); "
            "assert len(v) > 0; "
            "print('OK')"
        )
        return subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "PYTHONPATH": str(Path.cwd())},
        )

    def test_three_concurrent_processes_do_not_crash(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            # Launch 3 concurrent workers
            procs = [
                subprocess.Popen(
                    [sys.executable, "-c",
                     "import apsw, sqlite_vec, sqlite_lembed; "
                     "from lib.hal.kono.substrate import _ensure_lembed_registered, safe_lembed; "
                     f"c = apsw.Connection({db_path!r}); "
                     "c.enableloadextension(True); "
                     "c.loadextension(sqlite_vec.loadable_path()); "
                     "c.loadextension(sqlite_lembed.loadable_path()); "
                     "_ensure_lembed_registered(c); "
                     "v = safe_lembed(c, 'test'); "
                     "print('OK')"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    env={**os.environ, "PYTHONPATH": str(Path.cwd())},
                )
                for _ in range(3)
            ]
            outs = [p.communicate(timeout=30) for p in procs]

            for i, (out, err) in enumerate(outs):
                self.assertEqual(procs[i].returncode, 0,
                                 f"worker {i} crashed: stderr={err!r}")
                self.assertIn(b"OK", out)
        finally:
            Path(db_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
