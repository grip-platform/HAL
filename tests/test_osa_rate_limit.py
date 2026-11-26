"""Tests for ``lib.hal.osa.rate_limit`` — γ-2b per-script rolling-window cap.

Covers:

* :class:`RateLimit` dataclass validation (positive window / max_calls)
* :func:`check_and_record` contract — under cap passes, at cap raises,
  expired rows don't count, independent scripts are independent
* Concurrent dispatcher serialisation via SQLite BEGIN IMMEDIATE lock
* Dispatch integration — rate-limited script rejection path writes
  ``error="ratelimit_exceeded"`` audit entry and never calls the
  transport; scripts without a ``rate_limit`` field pass through
* Registry parsing — JSON ``rate_limit`` becomes a :class:`RateLimit`
  attached to the :class:`Script`
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from lib.hal.osa.audit import AuditEntry
from lib.hal.osa.auth import (
    _reset_for_tests as _reset_auth,
    mint_token,
)
from lib.hal.osa.consent import (
    _reset_for_tests as _reset_consent,
    set_consent_callback,
)
from lib.hal.osa.dispatch import _enforce_rate_limit, osa
from lib.hal.osa.osa import OsaResult
from lib.hal.osa.provenance import ProvenanceType
from lib.hal.osa.rate_limit import (
    RateLimit,
    RateLimitExceeded,
    check_and_record,
    _reset_for_tests as _reset_rate,
)
from lib.hal.osa.registry import (
    ParamSpec,
    Script,
    load_registry,
)
from lib.hal.osa.transport import LocalTransport


def _fake_result(stdout: str = "ok") -> OsaResult:
    return OsaResult(
        ok=True, returncode=0, stdout=stdout, stderr="",
        duration_s=0.01, language="AppleScript",
    )


class RateLimitDataclass(unittest.TestCase):
    """``RateLimit`` validates its fields at construction."""

    def test_accepts_positive_fields(self) -> None:
        RateLimit(window_s=60, max_calls=3)

    def test_rejects_zero_window(self) -> None:
        with self.assertRaises(ValueError):
            RateLimit(window_s=0, max_calls=3)

    def test_rejects_negative_window(self) -> None:
        with self.assertRaises(ValueError):
            RateLimit(window_s=-1, max_calls=3)

    def test_rejects_zero_max_calls(self) -> None:
        with self.assertRaises(ValueError):
            RateLimit(window_s=60, max_calls=0)


class CheckAndRecord(unittest.TestCase):
    """Direct contract tests for :func:`check_and_record`."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp(prefix="hal-rl-")
        self.db = Path(self._tmpdir) / "rl.db"
        self.limit = RateLimit(window_s=60, max_calls=3)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_first_call_passes(self) -> None:
        check_and_record("scriptA", self.limit, db_path=self.db)

    def test_under_cap_allows_all(self) -> None:
        for _ in range(3):
            check_and_record("scriptA", self.limit, db_path=self.db)

    def test_at_cap_raises(self) -> None:
        for _ in range(3):
            check_and_record("scriptA", self.limit, db_path=self.db)
        with self.assertRaises(RateLimitExceeded):
            check_and_record("scriptA", self.limit, db_path=self.db)

    def test_different_scripts_are_independent(self) -> None:
        for _ in range(3):
            check_and_record("scriptA", self.limit, db_path=self.db)
        check_and_record("scriptB", self.limit, db_path=self.db)

    def test_expired_rows_do_not_count(self) -> None:
        base = 1000.0
        for i in range(3):
            check_and_record(
                "scriptA", self.limit, db_path=self.db, now=base + i,
            )
        check_and_record(
            "scriptA", self.limit, db_path=self.db, now=base + 61.0,
        )

    def test_error_message_names_script_and_window(self) -> None:
        for _ in range(3):
            check_and_record("scriptA", self.limit, db_path=self.db)
        with self.assertRaises(RateLimitExceeded) as ctx:
            check_and_record("scriptA", self.limit, db_path=self.db)
        msg = str(ctx.exception)
        self.assertIn("scriptA", msg)
        self.assertIn("60s", msg)


class Concurrency(unittest.TestCase):
    """BEGIN IMMEDIATE + single-transaction check-and-insert prevents races."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp(prefix="hal-rl-c-")
        self.db = Path(self._tmpdir) / "rl.db"

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_concurrent_dispatches_respect_cap(self) -> None:
        """Ten threads race to dispatch; at cap=5, exactly 5 must succeed."""
        limit = RateLimit(window_s=60, max_calls=5)
        success_count = [0]
        fail_count = [0]
        lock = threading.Lock()

        def attempt() -> None:
            try:
                check_and_record("shared", limit, db_path=self.db)
                with lock:
                    success_count[0] += 1
            except RateLimitExceeded:
                with lock:
                    fail_count[0] += 1

        threads = [threading.Thread(target=attempt) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(success_count[0], 5)
        self.assertEqual(fail_count[0], 5)


class RegistryParsing(unittest.TestCase):
    """``load_registry`` converts JSON ``rate_limit`` into a :class:`RateLimit`."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp(prefix="hal-rl-reg-")

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _write_registry(self, scripts: dict) -> Path:
        path = Path(self._tmpdir) / "registry.json"
        path.write_text(json.dumps({"scripts": scripts}))
        return path

    def test_rate_limit_field_parsed_to_dataclass(self) -> None:
        path = self._write_registry({
            "limited": {
                "tier": 1, "body": "return 1",
                "rate_limit": {"window_s": 30, "max_calls": 5},
            },
        })
        reg = load_registry(path)
        self.assertIsInstance(reg["limited"].rate_limit, RateLimit)
        self.assertEqual(reg["limited"].rate_limit.window_s, 30)
        self.assertEqual(reg["limited"].rate_limit.max_calls, 5)

    def test_missing_rate_limit_is_none(self) -> None:
        path = self._write_registry({
            "unlimited": {"tier": 1, "body": "return 1"},
        })
        reg = load_registry(path)
        self.assertIsNone(reg["unlimited"].rate_limit)


class EnforceRateLimitFunction(unittest.TestCase):
    """Direct unit tests on ``_enforce_rate_limit`` (dispatch-layer helper)."""

    def setUp(self) -> None:
        os.environ["HAL_OSA_SKIP_IMMUTABLE"] = "1"
        self._tmpdir = tempfile.mkdtemp(prefix="hal-rl-e-")
        os.environ["HAL_HOME"] = self._tmpdir
        _reset_rate()

    def tearDown(self) -> None:
        _reset_rate()
        os.environ.pop("HAL_OSA_SKIP_IMMUTABLE", None)
        os.environ.pop("HAL_HOME", None)
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _script(self, rl: RateLimit = None) -> Script:
        return Script(
            name="rl_test", tier=1, body="return 1",
            params=(), timeout_s=5, rate_limit=rl,
        )

    def test_no_rate_limit_field_is_passthrough(self) -> None:
        _enforce_rate_limit(
            self._script(rl=None), {}, "local",
            ProvenanceType.LOCAL, LocalTransport(), audit=False,
        )

    @patch("lib.hal.osa.dispatch.write_audit")
    def test_rate_limit_under_cap_passes(self, mock_write: MagicMock) -> None:
        limit = RateLimit(window_s=60, max_calls=2)
        for _ in range(2):
            _enforce_rate_limit(
                self._script(rl=limit), {}, "local",
                ProvenanceType.LOCAL, LocalTransport(), audit=False,
            )
        mock_write.assert_not_called()

    @patch("lib.hal.osa.dispatch.write_audit")
    def test_rate_limit_at_cap_raises_and_audits(
        self, mock_write: MagicMock
    ) -> None:
        limit = RateLimit(window_s=60, max_calls=1)
        _enforce_rate_limit(
            self._script(rl=limit), {}, "local",
            ProvenanceType.LOCAL, LocalTransport(), audit=True,
        )
        with self.assertRaises(RateLimitExceeded):
            _enforce_rate_limit(
                self._script(rl=limit), {}, "local",
                ProvenanceType.LOCAL, LocalTransport(), audit=True,
            )
        mock_write.assert_called_once()
        entry: AuditEntry = mock_write.call_args[0][0]
        self.assertEqual(entry.error, "ratelimit_exceeded")
        self.assertFalse(entry.ok)


class DispatchRateLimitEndToEnd(unittest.TestCase):
    """Full ``osa()`` dispatch path with a rate-limited script."""

    _TEST_SECRET = bytes(range(32))

    def setUp(self) -> None:
        os.environ["HAL_OSA_SKIP_IMMUTABLE"] = "1"
        os.environ["HAL_OSA_CAPABILITY_SECRET"] = self._TEST_SECRET.hex()
        self._tmpdir = tempfile.mkdtemp(prefix="hal-rl-d-")
        os.environ["HAL_HOME"] = self._tmpdir
        _reset_rate()
        _reset_auth()
        _reset_consent()
        set_consent_callback(lambda n, p: True)

    def tearDown(self) -> None:
        _reset_rate()
        _reset_auth()
        _reset_consent()
        os.environ.pop("HAL_OSA_SKIP_IMMUTABLE", None)
        os.environ.pop("HAL_OSA_CAPABILITY_SECRET", None)
        os.environ.pop("HAL_HOME", None)
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @patch("lib.hal.osa.dispatch.write_audit")
    @patch("lib.hal.osa.dispatch.LocalTransport")
    @patch("lib.hal.osa.dispatch.get_script")
    def test_rate_limit_rejection_skips_transport_execute(
        self, mock_get_script, mock_transport_cls, mock_write
    ) -> None:
        mock_get_script.return_value = Script(
            name="capped", tier=1, body="return 1",
            params=(), timeout_s=5,
            rate_limit=RateLimit(window_s=60, max_calls=1),
        )
        transport = MagicMock()
        transport.is_remote = False
        transport.host = "local"
        transport.execute.return_value = _fake_result()
        mock_transport_cls.return_value = transport

        osa("capped")  # first call succeeds
        with self.assertRaises(RateLimitExceeded):
            osa("capped")  # second hits cap

        self.assertEqual(transport.execute.call_count, 1)
        rejection_entry: AuditEntry = mock_write.call_args_list[-1][0][0]
        self.assertEqual(rejection_entry.error, "ratelimit_exceeded")

    @patch("lib.hal.osa.dispatch.write_audit")
    @patch("lib.hal.osa.dispatch.LocalTransport")
    @patch("lib.hal.osa.dispatch.get_script")
    def test_rate_limit_runs_before_authorise(
        self, mock_get_script, mock_transport_cls, mock_write
    ) -> None:
        """Cap is hit BEFORE the capability-token check, so an invalid token
        still produces a rate-limit rejection (not a MissingSecret).

        This pins the order — swapping it would let a caller with no token
        exhaust the rate-limit quota by probing tier-2 scripts."""
        mock_get_script.return_value = Script(
            name="tier2_capped", tier=2, body="return 1",
            params=(), timeout_s=5,
            rate_limit=RateLimit(window_s=60, max_calls=1),
        )
        transport = MagicMock()
        transport.is_remote = False
        transport.host = "local"
        transport.execute.return_value = _fake_result()
        mock_transport_cls.return_value = transport
        token = mint_token("tier2_capped", secret=self._TEST_SECRET)

        osa("tier2_capped", capability_token=token)
        with self.assertRaises(RateLimitExceeded):
            osa("tier2_capped")  # no token — still hits rate limit first


if __name__ == "__main__":
    unittest.main()
