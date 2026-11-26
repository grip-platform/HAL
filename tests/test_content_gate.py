"""Tests for HAL content-sensitivity routing gate."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.hal.router import _filter_sensitive_providers, _is_sensitive


class TestIsSensitive(unittest.TestCase):

    def test_detects_legal(self):
        self.assertTrue(_is_sensitive("Review this legal contract for compliance issues"))

    def test_detects_gdpr(self):
        self.assertTrue(_is_sensitive("We need to handle GDPR data deletion requests"))

    def test_detects_pii(self):
        self.assertTrue(_is_sensitive("This contains PII from our customer database"))

    def test_detects_audit(self):
        self.assertTrue(_is_sensitive("Run an audit on our SOX compliance posture"))

    def test_clean_message_not_sensitive(self):
        self.assertFalse(_is_sensitive("Write a Python function to sort a list"))

    def test_case_insensitive(self):
        self.assertTrue(_is_sensitive("LEGAL contract review"))


class TestFilterSensitiveProviders(unittest.TestCase):

    _CHAIN = [
        "groq/llama-3.3-70b-versatile",
        "gemini/gemini-2.5-flash",
        "deepseek/deepseek-chat",
        "qwen/qwen3.5-flash",
        "openai/gpt-4o",
    ]

    def test_sensitive_message_removes_deepseek_qwen(self):
        filtered = _filter_sensitive_providers(self._CHAIN, "GDPR compliance audit")
        providers = [e.split("/")[0] for e in filtered]
        self.assertNotIn("deepseek", providers)
        self.assertNotIn("qwen", providers)

    def test_sensitive_message_keeps_other_providers(self):
        filtered = _filter_sensitive_providers(self._CHAIN, "GDPR compliance audit")
        self.assertIn("groq/llama-3.3-70b-versatile", filtered)
        self.assertIn("gemini/gemini-2.5-flash", filtered)
        self.assertIn("openai/gpt-4o", filtered)

    def test_clean_message_keeps_all_providers(self):
        filtered = _filter_sensitive_providers(self._CHAIN, "Sort a list in Python")
        self.assertEqual(filtered, self._CHAIN)


if __name__ == "__main__":
    unittest.main()
