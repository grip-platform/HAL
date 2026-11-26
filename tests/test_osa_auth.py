"""Tests for ``lib.hal.osa.auth`` — capability tokens (HMAC, TTL, single-use)."""
from __future__ import annotations

import os
import time
import unittest
from unittest.mock import patch

from lib.hal.osa.auth import (
    CapabilityToken,
    MissingSecret,
    TokenExpired,
    TokenInvalid,
    TokenReused,
    _reset_for_tests,
    default_secret,
    mint_token,
    verify_token,
)


_GOOD_SECRET = bytes(range(32))  # 32 bytes, fixed for reproducibility
_OTHER_SECRET = bytes(range(100, 132))


class DefaultSecret(unittest.TestCase):
    """``default_secret`` enforces the hex-encoded env var contract."""

    def test_missing_env_raises(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HAL_OSA_CAPABILITY_SECRET", None)
            with self.assertRaises(MissingSecret):
                default_secret()

    def test_short_secret_raises(self) -> None:
        with patch.dict(os.environ, {"HAL_OSA_CAPABILITY_SECRET": "abcd" * 4}):
            with self.assertRaises(MissingSecret):
                default_secret()

    def test_invalid_hex_raises(self) -> None:
        with patch.dict(os.environ, {"HAL_OSA_CAPABILITY_SECRET": "not hex at all"}):
            with self.assertRaises(MissingSecret):
                default_secret()

    def test_valid_32_byte_hex_ok(self) -> None:
        with patch.dict(os.environ, {"HAL_OSA_CAPABILITY_SECRET": _GOOD_SECRET.hex()}):
            self.assertEqual(default_secret(), _GOOD_SECRET)


class Minting(unittest.TestCase):
    """``mint_token`` returns a fresh nonce + valid MAC."""

    def setUp(self) -> None:
        _reset_for_tests()

    def test_mint_fields_populated(self) -> None:
        tok = mint_token("notification", secret=_GOOD_SECRET, now=1000.0, ttl_s=60)
        self.assertEqual(tok.script_name, "notification")
        self.assertEqual(tok.issued_at, 1000.0)
        self.assertEqual(tok.expires_at, 1060.0)
        self.assertEqual(len(tok.nonce), 32)  # 16 bytes hex = 32 chars
        self.assertEqual(len(tok.mac), 64)    # sha256 hex = 64 chars

    def test_two_mints_have_different_nonces(self) -> None:
        t1 = mint_token("x", secret=_GOOD_SECRET)
        t2 = mint_token("x", secret=_GOOD_SECRET)
        self.assertNotEqual(t1.nonce, t2.nonce)

    def test_zero_ttl_raises(self) -> None:
        with self.assertRaises(ValueError):
            mint_token("x", secret=_GOOD_SECRET, ttl_s=0)


class Verification(unittest.TestCase):
    """``verify_token`` enforces name, MAC, TTL, single-use in order."""

    def setUp(self) -> None:
        _reset_for_tests()

    def test_happy_path(self) -> None:
        tok = mint_token("notification", secret=_GOOD_SECRET, now=1000.0)
        verify_token(tok, script_name="notification", secret=_GOOD_SECRET, now=1000.5)

    def test_wrong_script_name(self) -> None:
        tok = mint_token("notification", secret=_GOOD_SECRET)
        with self.assertRaises(TokenInvalid):
            verify_token(tok, script_name="shutdown", secret=_GOOD_SECRET)

    def test_wrong_secret_is_timing_safe(self) -> None:
        tok = mint_token("x", secret=_GOOD_SECRET)
        with self.assertRaises(TokenInvalid):
            verify_token(tok, script_name="x", secret=_OTHER_SECRET)

    def test_expired(self) -> None:
        tok = mint_token("x", secret=_GOOD_SECRET, now=1000.0, ttl_s=60)
        with self.assertRaises(TokenExpired):
            verify_token(tok, script_name="x", secret=_GOOD_SECRET, now=1100.0)

    def test_single_use_enforced(self) -> None:
        tok = mint_token("x", secret=_GOOD_SECRET, now=1000.0)
        verify_token(tok, script_name="x", secret=_GOOD_SECRET, now=1000.5)
        with self.assertRaises(TokenReused):
            verify_token(tok, script_name="x", secret=_GOOD_SECRET, now=1000.6)

    def test_tampered_mac_detected(self) -> None:
        """Goodhart guard — forging the MAC must fail, not pass silently."""
        tok = mint_token("x", secret=_GOOD_SECRET)
        tampered = CapabilityToken(
            script_name=tok.script_name,
            issued_at=tok.issued_at,
            expires_at=tok.expires_at,
            nonce=tok.nonce,
            mac="0" * 64,
        )
        with self.assertRaises(TokenInvalid):
            verify_token(tampered, script_name="x", secret=_GOOD_SECRET)

    def test_tampered_expiry_detected(self) -> None:
        """Extending a token's expiry without re-signing must fail."""
        tok = mint_token("x", secret=_GOOD_SECRET, now=1000.0, ttl_s=60)
        forged = CapabilityToken(
            script_name=tok.script_name,
            issued_at=tok.issued_at,
            expires_at=tok.expires_at + 10_000,
            nonce=tok.nonce,
            mac=tok.mac,
        )
        with self.assertRaises(TokenInvalid):
            verify_token(forged, script_name="x", secret=_GOOD_SECRET, now=1050.0)


if __name__ == "__main__":
    unittest.main()
