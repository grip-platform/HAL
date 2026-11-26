"""Tests for ``lib.hal.hypothesis.engine`` — HAL-native falsification tracker."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from lib.hal.hypothesis.engine import (
    HypothesisNotFound,
    STATUS_CONFIRMED,
    STATUS_PENDING,
    STATUS_REFUTED,
    confirm,
    default_hypothesis_file,
    falsify,
    register,
    status,
    verify_all,
)


class RegisterAssignsCode(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="hal-h-r-")
        os.environ["HAL_HOME"] = self._tmp
        self.path = default_hypothesis_file()

    def tearDown(self) -> None:
        os.environ.pop("HAL_HOME", None)
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_first_hypothesis_is_H001(self) -> None:
        h = register("First claim", "metric_a", "positive", "2026-06-01", path=self.path)
        self.assertEqual(h.code, "H001")

    def test_codes_are_monotonic(self) -> None:
        register("a", "m", "p", "2026-06-01", path=self.path)
        register("b", "m", "p", "2026-06-01", path=self.path)
        third = register("c", "m", "p", "2026-06-01", path=self.path)
        self.assertEqual(third.code, "H003")

    def test_empty_claim_rejected(self) -> None:
        with self.assertRaises(ValueError):
            register("   ", "m", "p", "2026-06-01", path=self.path)

    def test_registered_at_is_set(self) -> None:
        h = register("claim", "m", "p", "2026-06-01", path=self.path)
        self.assertNotEqual(h.registered_at, "")
        self.assertEqual(h.status, STATUS_PENDING)

    def test_file_is_appended_not_rewritten(self) -> None:
        register("a", "m", "p", "2026-06-01", path=self.path)
        register("b", "m", "p", "2026-06-01", path=self.path)
        lines = self.path.read_text().strip().splitlines()
        self.assertEqual(len(lines), 2)


class StatusAggregates(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="hal-h-s-")
        os.environ["HAL_HOME"] = self._tmp
        self.path = default_hypothesis_file()

    def tearDown(self) -> None:
        os.environ.pop("HAL_HOME", None)
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_empty_registry_has_zero_counts(self) -> None:
        counts = status(path=self.path)
        self.assertEqual(counts["pending"], 0)
        self.assertEqual(counts["total"], 0)

    def test_counts_reflect_registered_state(self) -> None:
        register("a", "m", "p", "2026-06-01", path=self.path)
        register("b", "m", "p", "2026-06-01", path=self.path)
        counts = status(path=self.path)
        self.assertEqual(counts["pending"], 2)
        self.assertEqual(counts["total"], 2)


class FalsifyConfirm(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="hal-h-f-")
        os.environ["HAL_HOME"] = self._tmp
        self.path = default_hypothesis_file()

    def tearDown(self) -> None:
        os.environ.pop("HAL_HOME", None)
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_falsify_marks_refuted_with_evidence(self) -> None:
        h = register("claim", "m", "p", "2026-06-01", path=self.path)
        updated = falsify(h.code, "metric came in negative", path=self.path)
        self.assertEqual(updated.status, STATUS_REFUTED)
        self.assertEqual(updated.evidence, "metric came in negative")

    def test_confirm_marks_confirmed(self) -> None:
        h = register("claim", "m", "p", "2026-06-01", path=self.path)
        updated = confirm(h.code, "matched prediction", path=self.path)
        self.assertEqual(updated.status, STATUS_CONFIRMED)

    def test_falsify_missing_code_raises(self) -> None:
        with self.assertRaises(HypothesisNotFound):
            falsify("H999", "no such", path=self.path)

    def test_falsify_persists_to_disk(self) -> None:
        h = register("claim", "m", "p", "2026-06-01", path=self.path)
        falsify(h.code, "e", path=self.path)
        lines = self.path.read_text().strip().splitlines()
        record = json.loads(lines[0])
        self.assertEqual(record["status"], STATUS_REFUTED)


class VerifyAll(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="hal-h-v-")
        os.environ["HAL_HOME"] = self._tmp
        self.path = default_hypothesis_file()

    def tearDown(self) -> None:
        os.environ.pop("HAL_HOME", None)
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_returns_pending_past_deadline(self) -> None:
        register("old", "m", "p", "2026-01-01", path=self.path)
        register("fresh", "m", "p", "2030-01-01", path=self.path)
        due = verify_all(today="2026-06-01", path=self.path)
        self.assertEqual([d.claim for d in due], ["old"])

    def test_skips_confirmed_and_refuted(self) -> None:
        h = register("old", "m", "p", "2026-01-01", path=self.path)
        confirm(h.code, "done", path=self.path)
        due = verify_all(today="2026-06-01", path=self.path)
        self.assertEqual(due, [])


if __name__ == "__main__":
    unittest.main()
