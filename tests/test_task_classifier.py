"""Tests for Phase γ W1 task classifier (H-CLASSIFY)."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from lib.hal.task_classifier import (
    ALL_CLASSES,
    Classification,
    DISPATCH_TIER,
    HeuristicClassifier,
    TaskClass,
    dispatch_tier,
    log_miss,
    miss_rate,
)


class TestDispatchMap(unittest.TestCase):
    def test_all_classes_have_tier(self):
        for cls in ALL_CLASSES:
            self.assertIn(cls, DISPATCH_TIER)

    def test_dispatch_tier_uses_map(self):
        self.assertEqual(dispatch_tier(Classification(TaskClass.DRAFT, 0.5)), "free")
        self.assertEqual(dispatch_tier(Classification(TaskClass.REASON, 0.5)), "cheap")
        self.assertEqual(dispatch_tier(Classification(TaskClass.VERIFY, 0.5)), "premium")
        self.assertEqual(dispatch_tier(Classification(TaskClass.VISION, 0.5)), "mid")

    def test_unknown_class_defaults_to_cheap(self):
        self.assertEqual(dispatch_tier(Classification("unknown", 0.5)), "cheap")


class TestHeuristicClassifier(unittest.TestCase):
    def setUp(self):
        self.clf = HeuristicClassifier()

    def test_empty_prompt_is_draft(self):
        self.assertEqual(self.clf("").task_class, TaskClass.DRAFT)

    def test_short_prompt_is_draft(self):
        self.assertEqual(self.clf("hi").task_class, TaskClass.DRAFT)

    def test_vision_keyword_routes_to_vision(self):
        self.assertEqual(self.clf("describe this image please").task_class, TaskClass.VISION)

    def test_verify_keyword_routes_to_verify(self):
        self.assertEqual(
            self.clf("prove the correctness of this invariant").task_class,
            TaskClass.VERIFY,
        )

    def test_reason_keyword_routes_to_reason(self):
        self.assertEqual(
            self.clf("explain the trade-offs of this architecture").task_class,
            TaskClass.REASON,
        )

    def test_long_prompt_defaults_to_reason(self):
        self.assertEqual(self.clf("x" * 200).task_class, TaskClass.REASON)

    def test_vision_beats_verify(self):
        result = self.clf("prove this image shows correct output")
        self.assertEqual(result.task_class, TaskClass.VISION)

    def test_confidence_in_valid_range(self):
        result = self.clf("explain everything here")
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


class TestLogMiss(unittest.TestCase):
    def test_writes_jsonl_record(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "subdir", "misses.jsonl")
            log_miss(path, expected=TaskClass.VERIFY, actual=TaskClass.DRAFT,
                     prompt="prove X", clock=lambda: 1_000_000.0)
            with open(path) as f:
                record = json.loads(f.read())
            self.assertEqual(record["expected"], TaskClass.VERIFY)
            self.assertEqual(record["actual"], TaskClass.DRAFT)
            self.assertEqual(record["prompt"], "prove X")
            self.assertEqual(record["timestamp"], 1_000_000.0)

    def test_appends_multiple_records(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "misses.jsonl")
            for i in range(3):
                log_miss(path, expected="a", actual="b", prompt=f"p{i}")
            with open(path) as f:
                self.assertEqual(len(f.readlines()), 3)

    def test_empty_path_is_noop(self):
        log_miss("", expected="a", actual="b", prompt="x")

    def test_long_prompt_truncated(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "misses.jsonl")
            log_miss(path, expected="a", actual="b", prompt="x" * 5000)
            with open(path) as f:
                record = json.loads(f.read())
            self.assertLessEqual(len(record["prompt"]), 200)


class TestMissRate(unittest.TestCase):
    def test_empty_is_zero(self):
        self.assertEqual(miss_rate([]), 0.0)

    def test_all_misses_is_one(self):
        records = [{"expected": "a", "actual": "b"}] * 5
        self.assertEqual(miss_rate(records), 1.0)

    def test_no_misses_is_zero(self):
        records = [{"expected": "a", "actual": "a"}] * 5
        self.assertEqual(miss_rate(records), 0.0)

    def test_mixed_rate(self):
        records = [
            {"expected": "a", "actual": "a"},
            {"expected": "a", "actual": "b"},
            {"expected": "a", "actual": "a"},
            {"expected": "b", "actual": "a"},
        ]
        self.assertAlmostEqual(miss_rate(records), 0.5)


if __name__ == "__main__":
    unittest.main()
