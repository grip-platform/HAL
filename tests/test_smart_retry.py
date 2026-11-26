"""Tests for HAL smart retry."""
import os
import sys
import unittest
from unittest.mock import MagicMock

from lib.hal.smart_retry import RetryAttempt, RetryResult, SmartRetry


class TestRetryAttempt(unittest.TestCase):

    def test_success(self):
        a = RetryAttempt("model", True, 100)
        self.assertTrue(a.success)

    def test_failure(self):
        a = RetryAttempt("model", False, error="timeout")
        self.assertFalse(a.success)


class TestRetryResult(unittest.TestCase):

    def test_success_no_rotation(self):
        r = RetryResult(True, "ok", "m1", [RetryAttempt("m1", True)])
        self.assertTrue(r.success)
        self.assertFalse(r.rotated)

    def test_rotated(self):
        r = RetryResult(True, "ok", "m2", [
            RetryAttempt("m1", False),
            RetryAttempt("m2", True),
        ])
        self.assertTrue(r.rotated)

    def test_summary(self):
        r = RetryResult(True, "ok", "m1", [RetryAttempt("m1", True)], 100)
        self.assertIn("ok", r.summary)

    def test_attempt_count(self):
        r = RetryResult(False, "", "m", [
            RetryAttempt("m1", False),
            RetryAttempt("m2", False),
        ])
        self.assertEqual(r.attempt_count, 2)


class TestSmartRetry(unittest.TestCase):

    def test_success_first_try(self):
        mock_registry = MagicMock()
        mock_registry.chat.return_value = MagicMock(text="hello", cost_usd=0.01)
        mock_registry.list_providers.return_value = []
        retry = SmartRetry(mock_registry)
        result = retry.call("m1", "Hi", max_retries=3, base_delay_s=0)
        self.assertTrue(result.success)
        self.assertEqual(result.attempt_count, 1)
        self.assertEqual(result.response_text, "hello")

    def test_fails_then_succeeds(self):
        mock_registry = MagicMock()
        mock_registry.chat.side_effect = [
            RuntimeError("fail"),
            MagicMock(text="ok", cost_usd=0),
        ]
        mock_registry.list_providers.return_value = []
        retry = SmartRetry(mock_registry)
        result = retry.call("m1", "Hi", max_retries=2, base_delay_s=0)
        self.assertTrue(result.success)
        self.assertEqual(result.attempt_count, 2)

    def test_all_fail(self):
        mock_registry = MagicMock()
        mock_registry.chat.side_effect = RuntimeError("down")
        mock_registry.list_providers.return_value = []
        retry = SmartRetry(mock_registry)
        result = retry.call("m1", "Hi", max_retries=2, base_delay_s=0)
        self.assertFalse(result.success)
        self.assertEqual(result.attempt_count, 2)

    def test_rotates_providers(self):
        mock_prov = MagicMock()
        mock_prov.name = "alt"
        mock_prov.models = ["alt-model"]
        mock_registry = MagicMock()
        mock_registry.chat.side_effect = [
            RuntimeError("fail"),
            MagicMock(text="ok from alt", cost_usd=0),
        ]
        mock_registry.list_providers.return_value = [mock_prov]
        retry = SmartRetry(mock_registry)
        result = retry.call("primary/model", "Hi", max_retries=2, base_delay_s=0)
        self.assertTrue(result.success)
        self.assertTrue(result.rotated)


if __name__ == "__main__":
    unittest.main()
