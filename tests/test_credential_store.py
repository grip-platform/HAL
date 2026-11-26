"""Tests for HAL credential store."""
import os
import sys
import unittest

from lib.hal.credential_store import CredentialInfo, CredentialStore, _env_var_name, _mask


class TestEnvVarName(unittest.TestCase):

    def test_groq(self):
        self.assertEqual(_env_var_name("groq"), "GROQ_API_KEY")

    def test_openai(self):
        self.assertEqual(_env_var_name("openai"), "OPENAI_API_KEY")


class TestMask(unittest.TestCase):

    def test_long_key(self):
        result = _mask("gsk_abcdef12345xyz")
        self.assertTrue(result.startswith("gsk_"))
        self.assertIn("***", result)
        self.assertTrue(result.endswith("xyz"))

    def test_short_key(self):
        self.assertEqual(_mask("abc"), "***")


class TestCredentialStore(unittest.TestCase):

    def test_get_from_env(self):
        store = CredentialStore()
        with unittest.mock.patch.dict(os.environ, {"GROQ_API_KEY": "test_key_123"}):
            key = store.get("groq")
            self.assertEqual(key, "test_key_123")

    def test_get_missing(self):
        store = CredentialStore()
        # Use a provider name that definitely doesn't have env var set
        key = store.get("nonexistent_provider_xyz")
        self.assertEqual(key, "")

    def test_has_with_env(self):
        store = CredentialStore()
        with unittest.mock.patch.dict(os.environ, {"GROQ_API_KEY": "key"}):
            self.assertTrue(store.has("groq"))

    def test_has_without(self):
        store = CredentialStore()
        self.assertFalse(store.has("nonexistent_provider_xyz"))

    def test_list_providers(self):
        store = CredentialStore()
        providers = store.list_providers()
        self.assertIsInstance(providers, list)
        # Check that any found credentials are properly structured
        for p in providers:
            self.assertIsInstance(p, CredentialInfo)
            self.assertIn("***", p.masked)


class TestCredentialInfo(unittest.TestCase):

    def test_frozen(self):
        info = CredentialInfo("groq", "env", "gsk_***123")
        with self.assertRaises(AttributeError):
            info.provider = "changed"

    def test_fields(self):
        info = CredentialInfo("openai", "keychain", "sk-***abc")
        self.assertEqual(info.source, "keychain")


if __name__ == "__main__":
    unittest.main()
