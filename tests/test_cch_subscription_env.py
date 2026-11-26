"""Tests for CCHSubscription.from_env env-var precedence + keychain hex decode."""
import os
import unittest
from unittest.mock import patch

from lib.hal.subscription import (
    CCHSubscription,
    _fetch_keychain_secret,
    _fetch_token,
)

_ENV_KEYS = ("CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_OAUTH_TOKEN")


class _FakeResult:
    def __init__(self, out, code=0):
        self.stdout = out
        self.returncode = code


def _security_returns(stdout, returncode=0):
    """Patch subprocess.run in subscription module to return a canned value.

    Patching at subprocess level is immune to test-suite pollution that
    sometimes leaks monkeypatches of higher-level helpers.
    """
    return patch(
        "lib.hal.subscription.subprocess.run",
        return_value=_FakeResult(stdout, returncode),
    )


class _EnvIsolation(unittest.TestCase):
    """setUp/tearDown that bulletproof-isolates the OAuth env vars."""

    def setUp(self):
        self._saved = {k: os.environ[k] for k in _ENV_KEYS if k in os.environ}
        for k in _ENV_KEYS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update(self._saved)


class TestEnvVarPrecedence(_EnvIsolation):
    def test_canonical_name_wins_over_legacy(self):
        os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = "canonical-token"
        os.environ["CLAUDE_OAUTH_TOKEN"] = "legacy-token"
        token = _fetch_token(_ENV_KEYS, "nonexistent-service")
        self.assertEqual(token, "canonical-token")

    def test_legacy_name_used_when_canonical_absent(self):
        os.environ["CLAUDE_OAUTH_TOKEN"] = "legacy-token"
        token = _fetch_token(_ENV_KEYS, "nonexistent-service")
        self.assertEqual(token, "legacy-token")

    def test_empty_env_falls_back_to_keychain(self):
        with _security_returns("kc-token\n"):
            token = _fetch_token(_ENV_KEYS, "grip-claude-oauth")
        self.assertEqual(token, "kc-token")

    def test_from_env_returns_none_when_no_token(self):
        with _security_returns("", returncode=44):
            sub = CCHSubscription.from_env()
        self.assertIsNone(sub)


class TestKeychainHexDecode(_EnvIsolation):
    def test_plain_text_password_returned_as_is(self):
        with _security_returns("sk-ant-oat01-plaintext-example\n"):
            self.assertEqual(_fetch_keychain_secret("any-service"),
                             "sk-ant-oat01-plaintext-example")

    def test_hex_encoded_password_decoded_to_utf8(self):
        original = "sk-ant-oat01-abc"
        hex_encoded = original.encode("utf-8").hex()
        with _security_returns(hex_encoded + "\n"):
            self.assertEqual(_fetch_keychain_secret("any-service"), original)

    def test_odd_length_hex_like_string_returned_raw(self):
        with _security_returns("abcdef1\n"):
            self.assertEqual(_fetch_keychain_secret("any-service"), "abcdef1")

    def test_real_token_shape_hex_roundtrip(self):
        real = "sk-ant-oat01-mock_value_with_underscores-ABC"
        with _security_returns(real + "\n"):
            self.assertEqual(_fetch_keychain_secret("any-service"), real)


if __name__ == "__main__":
    unittest.main()
