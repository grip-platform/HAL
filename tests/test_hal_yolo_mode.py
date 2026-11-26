"""Tests for ``lib.hal.gates.yolo_mode`` — HAL-native autonomous-loop marker."""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from lib.hal.gates.yolo_mode import (
    DEFAULT_TTL_SECONDS,
    MAX_TTL_SECONDS,
    activate_yolo,
    deactivate_yolo,
    is_yolo_active,
    yolo_log_path,
    yolo_marker_path,
)


class YoloMarkerRooting(unittest.TestCase):
    """Paths resolve under HAL_HOME (not CLAUDE_DIR)."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="hal-yolo-")
        os.environ["HAL_HOME"] = self._tmp

    def tearDown(self) -> None:
        os.environ.pop("HAL_HOME", None)
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_marker_rooted_under_hal_home(self) -> None:
        self.assertEqual(
            yolo_marker_path(),
            Path(self._tmp) / "state" / "session" / "yolo-mode.json",
        )

    def test_log_rooted_under_hal_home(self) -> None:
        self.assertEqual(
            yolo_log_path(),
            Path(self._tmp) / "logs" / "yolo-mode.log",
        )


class ActivateDeactivate(unittest.TestCase):
    """Round-trip activate → is_active → deactivate."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="hal-yolo-a-")
        os.environ["HAL_HOME"] = self._tmp

    def tearDown(self) -> None:
        os.environ.pop("HAL_HOME", None)
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_activate_writes_marker(self) -> None:
        activate_yolo("/ultrado", ttl_seconds=60, reason="test")
        self.assertTrue(yolo_marker_path().exists())

    def test_activate_payload_includes_context_and_ttl(self) -> None:
        activate_yolo("/rsi-optimal", ttl_seconds=120)
        payload = json.loads(yolo_marker_path().read_text())
        self.assertEqual(payload["context"], "/rsi-optimal")
        self.assertEqual(payload["ttl_seconds"], 120)

    def test_is_active_true_after_activate(self) -> None:
        activate_yolo("/ultrado", ttl_seconds=60)
        active, context = is_yolo_active()
        self.assertTrue(active)
        self.assertEqual(context, "/ultrado")

    def test_deactivate_removes_marker(self) -> None:
        activate_yolo("/ultrado", ttl_seconds=60)
        deactivate_yolo("/ultrado")
        self.assertFalse(yolo_marker_path().exists())
        active, _ = is_yolo_active()
        self.assertFalse(active)

    def test_deactivate_missing_marker_is_idempotent(self) -> None:
        deactivate_yolo("/ultrado")  # no marker exists yet — must not raise


class TtlClampAndExpiry(unittest.TestCase):
    """TTL is clamped; expired markers fail closed."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="hal-yolo-t-")
        os.environ["HAL_HOME"] = self._tmp

    def tearDown(self) -> None:
        os.environ.pop("HAL_HOME", None)
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_ttl_above_max_is_clamped(self) -> None:
        activate_yolo("/ultrado", ttl_seconds=MAX_TTL_SECONDS + 999_999)
        payload = json.loads(yolo_marker_path().read_text())
        self.assertEqual(payload["ttl_seconds"], MAX_TTL_SECONDS)

    def test_ttl_below_60_is_clamped_up(self) -> None:
        activate_yolo("/ultrado", ttl_seconds=1)
        payload = json.loads(yolo_marker_path().read_text())
        self.assertEqual(payload["ttl_seconds"], 60)

    def test_expired_marker_reports_inactive(self) -> None:
        """Force the marker's expires_at_epoch into the past."""
        activate_yolo("/ultrado", ttl_seconds=60)
        marker = yolo_marker_path()
        payload = json.loads(marker.read_text())
        payload["expires_at_epoch"] = time.time() - 1
        marker.write_text(json.dumps(payload))
        active, _ = is_yolo_active()
        self.assertFalse(active)


class FailClosed(unittest.TestCase):
    """Malformed JSON must never return True."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="hal-yolo-f-")
        os.environ["HAL_HOME"] = self._tmp
        self.marker = yolo_marker_path()
        self.marker.parent.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        os.environ.pop("HAL_HOME", None)
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_malformed_json_returns_false(self) -> None:
        self.marker.write_text("not-json{{{")
        active, _ = is_yolo_active()
        self.assertFalse(active)

    def test_missing_expires_field_returns_false(self) -> None:
        self.marker.write_text(json.dumps({"context": "/x"}))
        active, _ = is_yolo_active()
        self.assertFalse(active)


if __name__ == "__main__":
    unittest.main()
