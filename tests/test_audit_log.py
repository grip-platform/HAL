"""Tests for HAL audit log."""
import os
import sys
import unittest

from lib.hal.audit_log import AuditEntry, AuditLog


class TestAuditEntry(unittest.TestCase):

    def test_to_dict(self):
        entry = AuditEntry(1000.0, "llm_call", "user", "test query")
        d = entry.to_dict()
        self.assertEqual(d["operation"], "llm_call")
        self.assertIn("time_iso", d)

    def test_frozen(self):
        entry = AuditEntry(0, "op", "user", "detail")
        with self.assertRaises(AttributeError):
            entry.operation = "changed"

    def test_detail_truncated(self):
        entry = AuditEntry(0, "op", "user", "x" * 1000)
        d = entry.to_dict()
        self.assertLessEqual(len(d["detail"]), 500)


class TestAuditLog(unittest.TestCase):

    def test_record(self):
        log = AuditLog()
        entry = log.record("test_op", actor="user", detail="did something")
        self.assertEqual(entry.operation, "test_op")
        self.assertEqual(entry.actor, "user")

    def test_count(self):
        log = AuditLog()
        log.record("a", actor="user")
        log.record("b", actor="agent")
        log.record("a", actor="system")
        self.assertEqual(log.count(), 3)
        self.assertEqual(log.count("a"), 2)

    def test_query_by_actor(self):
        log = AuditLog()
        log.record("op", actor="user")
        log.record("op", actor="agent")
        log.record("op", actor="user")
        results = log.query(actor="user")
        self.assertEqual(len(results), 2)

    def test_query_by_operation(self):
        log = AuditLog()
        log.record("llm_call", actor="user")
        log.record("file_write", actor="agent")
        results = log.query(operation="llm_call")
        self.assertEqual(len(results), 1)

    def test_query_by_outcome(self):
        log = AuditLog()
        log.record("op", outcome="success")
        log.record("op", outcome="failure")
        log.record("op", outcome="blocked")
        results = log.query(outcome="failure")
        self.assertEqual(len(results), 1)

    def test_query_limit(self):
        log = AuditLog()
        for i in range(20):
            log.record(f"op_{i}")
        results = log.query(limit=5)
        self.assertEqual(len(results), 5)

    def test_metadata(self):
        log = AuditLog()
        entry = log.record("deploy", actor="system", model="groq", tokens=500)
        self.assertEqual(entry.metadata["model"], "groq")
        self.assertEqual(entry.metadata["tokens"], 500)

    def test_default_outcome_success(self):
        log = AuditLog()
        entry = log.record("test")
        self.assertEqual(entry.outcome, "success")


if __name__ == "__main__":
    unittest.main()
