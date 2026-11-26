"""Tests for Phase β W5 live-ingest primitives (H-LIVE_INGEST, opt-in)."""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from lib.hal.live_ingest import (
    IngestionCandidate,
    IngestionDecision,
    IngestionDisabledError,
    LiveIngestConfig,
    append_audit,
    audit_record,
    build_memory_entry,
    should_ingest,
)


def _candidate(path="~/Downloads/notes.md", source="downloads", summary="") -> IngestionCandidate:
    return IngestionCandidate(
        path=path, source=source, detected_at=1_700_000_000.0, summary=summary,
    )


class TestDefaultIsOptIn(unittest.TestCase):
    """Concern #2 invariant: daemon is OFF by default."""

    def test_default_config_disabled(self):
        self.assertFalse(LiveIngestConfig().enabled)

    def test_should_ingest_rejects_when_disabled(self):
        decision = should_ingest(_candidate(), LiveIngestConfig())
        self.assertFalse(decision.should_ingest)
        self.assertIn("disabled", decision.reason)

    def test_build_memory_entry_raises_when_disabled(self):
        with self.assertRaises(IngestionDisabledError):
            build_memory_entry(_candidate(), LiveIngestConfig(), content="x")


class TestFromEnv(unittest.TestCase):
    def test_hal_live_ingest_enabled_by_env(self):
        with mock.patch.dict(os.environ, {"HAL_LIVE_INGEST": "1"}, clear=False):
            self.assertTrue(LiveIngestConfig.from_env().enabled)

    def test_unset_env_is_disabled(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HAL_LIVE_INGEST", None)
            self.assertFalse(LiveIngestConfig.from_env().enabled)

    def test_sources_parsed_from_env(self):
        with mock.patch.dict(os.environ, {
            "HAL_LIVE_INGEST": "1",
            "HAL_LIVE_INGEST_SOURCES": "downloads, clipboard ",
        }):
            cfg = LiveIngestConfig.from_env()
            self.assertEqual(cfg.allow_sources, ("downloads", "clipboard"))


class TestShouldIngestBarriers(unittest.TestCase):
    """Each of the four barriers (enabled, source, ext, glob) is in AND."""

    def _enabled(self, **kw) -> LiveIngestConfig:
        base = dict(enabled=True, allow_sources=(), allow_extensions=(),
                    allow_path_globs=())
        base.update(kw)
        return LiveIngestConfig(**base)

    def test_enabled_with_no_other_gates_allows(self):
        d = should_ingest(_candidate(), self._enabled())
        self.assertTrue(d.should_ingest)

    def test_source_allow_list_blocks_unknown(self):
        cfg = self._enabled(allow_sources=("downloads",))
        d = should_ingest(_candidate(source="clipboard"), cfg)
        self.assertFalse(d.should_ingest)
        self.assertIn("source 'clipboard'", d.reason)

    def test_source_allow_list_permits_known(self):
        cfg = self._enabled(allow_sources=("downloads",))
        d = should_ingest(_candidate(source="downloads"), cfg)
        self.assertTrue(d.should_ingest)

    def test_extension_allow_list_blocks_unknown(self):
        cfg = self._enabled(allow_extensions=(".md",))
        d = should_ingest(_candidate(path="/tmp/x.exe"), cfg)
        self.assertFalse(d.should_ingest)
        self.assertIn("extension '.exe'", d.reason)

    def test_glob_allow_list_blocks_mismatch(self):
        cfg = self._enabled(allow_path_globs=("/tmp/ok/*",))
        d = should_ingest(_candidate(path="/tmp/bad/notes.md"), cfg)
        self.assertFalse(d.should_ingest)
        self.assertIn("glob", d.reason)

    def test_glob_allow_list_permits_match(self):
        cfg = self._enabled(allow_path_globs=("/tmp/ok/*.md",))
        d = should_ingest(_candidate(path="/tmp/ok/notes.md"), cfg)
        self.assertTrue(d.should_ingest)

    def test_all_barriers_passed(self):
        cfg = self._enabled(
            allow_sources=("downloads",),
            allow_extensions=(".md",),
            allow_path_globs=("/tmp/ok/*.md",),
        )
        d = should_ingest(
            _candidate(path="/tmp/ok/x.md", source="downloads"), cfg,
        )
        self.assertTrue(d.should_ingest)

    def test_reason_differs_per_barrier(self):
        """Audit granularity invariant — each rejection reason is distinct."""
        cfg1 = self._enabled(allow_sources=("x",))
        cfg2 = self._enabled(allow_extensions=(".y",))
        cfg3 = self._enabled(allow_path_globs=("/nope/*",))
        r1 = should_ingest(_candidate(), cfg1).reason
        r2 = should_ingest(_candidate(), cfg2).reason
        r3 = should_ingest(_candidate(), cfg3).reason
        self.assertNotEqual(r1, r2)
        self.assertNotEqual(r2, r3)
        self.assertNotEqual(r1, r3)


class TestBuildMemoryEntry(unittest.TestCase):
    def test_produces_1_1_entry_with_provenance(self):
        cfg = LiveIngestConfig(enabled=True, session_id="sess-42")
        entry = build_memory_entry(_candidate(summary="weekly notes"), cfg, content="body")
        self.assertEqual(entry.memory_type, "reference")
        self.assertEqual(entry.source_session_id, "sess-42")
        self.assertEqual(entry.tool_calls_used, ["LiveIngest"])
        self.assertTrue(entry.has_provenance())
        self.assertIn("notes", entry.filename)  # uses path stem
        self.assertEqual(entry.content, "body")

    def test_falls_back_to_daemon_session_id(self):
        cfg = LiveIngestConfig(enabled=True)
        entry = build_memory_entry(_candidate(), cfg, content="x")
        self.assertEqual(entry.source_session_id, "live-ingest-daemon")

    def test_default_name_from_path(self):
        cfg = LiveIngestConfig(enabled=True)
        entry = build_memory_entry(_candidate(), cfg, content="x")
        self.assertIn("notes.md", entry.name)


class TestAuditRecord(unittest.TestCase):
    def test_accept_line_format(self):
        line = audit_record(
            IngestionDecision(True, "allowed"), _candidate(),
        )
        parts = line.split("\t")
        self.assertEqual(len(parts), 5)
        self.assertEqual(parts[2], "ACCEPT")
        self.assertEqual(parts[3], "allowed")

    def test_reject_line_format(self):
        line = audit_record(
            IngestionDecision(False, "daemon disabled"),
            _candidate(),
        )
        self.assertIn("REJECT", line)
        self.assertIn("daemon disabled", line)

    def test_audit_log_append(self):
        with tempfile.TemporaryDirectory() as td:
            log_path = os.path.join(td, "subdir", "audit.log")
            append_audit(log_path, "line-1")
            append_audit(log_path, "line-2")
            with open(log_path) as f:
                contents = f.read()
            self.assertIn("line-1", contents)
            self.assertIn("line-2", contents)

    def test_audit_log_noop_when_path_empty(self):
        append_audit("", "line")  # must not raise


if __name__ == "__main__":
    unittest.main()
