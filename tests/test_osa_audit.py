"""Tests for ``lib.hal.osa.audit`` — append-only JSONL audit writer."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lib.hal.osa.audit import (
    AuditEntry,
    make_entry,
    write_audit,
)


class AuditWriter(unittest.TestCase):
    """``write_audit`` appends JSONL lines to the target file."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.log_path = Path(self._tmp.name) / "osa.log"

    def test_first_write_creates_file(self) -> None:
        entry = make_entry(
            script="x", tier=1, ok=True, duration_s=0.1
        )
        write_audit(entry, log_path=self.log_path)
        self.assertTrue(self.log_path.exists())
        line = self.log_path.read_text().strip()
        payload = json.loads(line)
        self.assertEqual(payload["script"], "x")
        self.assertEqual(payload["tier"], 1)
        self.assertTrue(payload["ok"])

    def test_second_write_appends(self) -> None:
        write_audit(
            make_entry(script="a", tier=1, ok=True, duration_s=0.1),
            log_path=self.log_path,
        )
        write_audit(
            make_entry(script="b", tier=2, ok=False, duration_s=0.2, error="boom"),
            log_path=self.log_path,
        )
        lines = self.log_path.read_text().strip().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["script"], "a")
        self.assertEqual(json.loads(lines[1])["script"], "b")
        self.assertEqual(json.loads(lines[1])["error"], "boom")

    def test_parent_dir_auto_created(self) -> None:
        nested = Path(self._tmp.name) / "nested" / "logs" / "osa.log"
        write_audit(
            make_entry(script="x", tier=1, ok=True, duration_s=0.0),
            log_path=nested,
        )
        self.assertTrue(nested.exists())

    def test_ts_populated_by_make_entry(self) -> None:
        entry = make_entry(script="x", tier=1, ok=True, duration_s=0.0)
        self.assertIsNotNone(entry.ts)
        self.assertTrue(entry.ts.endswith("Z"))

    def test_newline_injection_in_params_rendered_safe(self) -> None:
        """Goodhart guard — json.dumps must escape newlines so log entries stay 1 line each."""
        entry = make_entry(
            script="imessage_send",
            tier=2,
            ok=True,
            duration_s=0.0,
            params={"text": "line1\nline2\nFAKE ENTRY"},
        )
        write_audit(entry, log_path=self.log_path)
        lines = self.log_path.read_text().strip().splitlines()
        self.assertEqual(
            len(lines), 1, "newlines in params must not produce multiple log lines"
        )

    def test_returncode_and_error_optional(self) -> None:
        default_entry = AuditEntry(
            ts="2026-01-01T00:00:00Z",
            script="x",
            tier=1,
            ok=True,
            duration_s=0.0,
        )
        self.assertIsNone(default_entry.returncode)
        self.assertIsNone(default_entry.error)

    def test_provenance_defaults_to_local(self) -> None:
        entry = make_entry(script="x", tier=1, ok=True, duration_s=0.0)
        self.assertEqual(entry.provenance, "local")

    def test_provenance_field_roundtrips_through_json(self) -> None:
        entry = make_entry(
            script="x", tier=2, ok=True, duration_s=0.0, provenance="tool_chain"
        )
        write_audit(entry, log_path=self.log_path)
        payload = json.loads(self.log_path.read_text().strip())
        self.assertEqual(payload["provenance"], "tool_chain")


class AppendOnlyFlag(unittest.TestCase):
    """``_apply_append_only`` — chflags behaviour is opt-out via env."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.log_path = Path(self._tmp.name) / "osa.log"

    def test_chflags_skipped_when_env_set(self) -> None:
        """Goodhart guard — HAL_OSA_SKIP_IMMUTABLE=1 must actually skip."""
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {"HAL_OSA_SKIP_IMMUTABLE": "1"}), \
             patch("lib.hal.osa.audit.subprocess.run") as mock_run:
            write_audit(
                make_entry(script="x", tier=1, ok=True, duration_s=0.0),
                log_path=self.log_path,
            )
        mock_run.assert_not_called()

    def test_chflags_called_once_per_path(self) -> None:
        """First write applies the flag; subsequent writes do not retry."""
        import os
        import sys
        from unittest.mock import patch

        from lib.hal.osa import audit as audit_mod

        audit_mod._IMMUTABLE_APPLIED.pop(str(self.log_path), None)
        with patch.dict(os.environ, {}, clear=False), \
             patch.object(sys, "platform", "darwin"), \
             patch("lib.hal.osa.audit.subprocess.run") as mock_run:
            os.environ.pop("HAL_OSA_SKIP_IMMUTABLE", None)
            for _ in range(3):
                write_audit(
                    make_entry(script="x", tier=1, ok=True, duration_s=0.0),
                    log_path=self.log_path,
                )
        self.assertEqual(mock_run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
