"""Tests for ``lib.hal.osa.consent`` — tier-3 interactive approval.

Covers the consent callback registry (swap/restore), the default TTY
prompt's fail-closed contract, the sha256 redaction helper, and the
end-to-end dispatch path where tier-3 scripts invoke the callback,
refuse silent bypass under yolo mode, and write audit entries with
``provenance=user_rejected`` + redacted params on rejection.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import unittest
from unittest.mock import MagicMock, patch

from lib.hal.osa.audit import AuditEntry
from lib.hal.osa.auth import (
    _reset_for_tests as _reset_auth,
    mint_token,
)
from lib.hal.osa.consent import (
    CONSENT_TIMEOUT_S,
    ConsentDeclined,
    _reset_for_tests,
    default_consent_callback,
    get_consent_callback,
    redact_params_for_tier3,
    set_consent_callback,
)
from lib.hal.osa.dispatch import _authorise_tier3_consent, osa
from lib.hal.osa.osa import OsaResult
from lib.hal.osa.provenance import ProvenanceType
from lib.hal.osa.registry import ParamSpec, Script
from lib.hal.osa.transport import LocalTransport


def _result(returncode: int = 0, stdout: str = "ok") -> OsaResult:
    return OsaResult(
        ok=(returncode == 0),
        returncode=returncode,
        stdout=stdout,
        stderr="" if returncode == 0 else "boom",
        duration_s=0.01,
        language="AppleScript",
    )


class RedactParamsForTier3(unittest.TestCase):
    """``redact_params_for_tier3`` replaces values with 12-char sha256 prefixes."""

    def test_hashes_are_twelve_char_lowercase_hex(self) -> None:
        redacted = redact_params_for_tier3({"recipient": "+27123456789"})
        self.assertEqual(len(redacted["recipient"]), 12)
        self.assertTrue(all(c in "0123456789abcdef" for c in redacted["recipient"]))

    def test_same_value_hashes_to_same_prefix(self) -> None:
        a = redact_params_for_tier3({"body": "hello"})
        b = redact_params_for_tier3({"body": "hello"})
        self.assertEqual(a, b)

    def test_different_values_produce_different_hashes(self) -> None:
        a = redact_params_for_tier3({"body": "hello"})
        b = redact_params_for_tier3({"body": "world"})
        self.assertNotEqual(a["body"], b["body"])

    def test_keys_are_preserved_unredacted(self) -> None:
        redacted = redact_params_for_tier3({"recipient": "x", "body": "y"})
        self.assertEqual(set(redacted.keys()), {"recipient", "body"})

    def test_empty_params_returns_empty_dict(self) -> None:
        self.assertEqual(redact_params_for_tier3({}), {})

    def test_empty_string_value_still_hashes(self) -> None:
        redacted = redact_params_for_tier3({"x": ""})
        expected = hashlib.sha256(b"").hexdigest()[:12]
        self.assertEqual(redacted["x"], expected)

    def test_non_string_value_coerced_via_str(self) -> None:
        redacted_int = redact_params_for_tier3({"n": 42})
        expected = hashlib.sha256(b"42").hexdigest()[:12]
        self.assertEqual(redacted_int["n"], expected)


class ConsentCallbackRegistry(unittest.TestCase):
    """``set_consent_callback`` / ``get_consent_callback`` manage the module singleton."""

    def tearDown(self) -> None:
        _reset_for_tests()

    def test_default_is_the_osascript_dialog_callback(self) -> None:
        _reset_for_tests()
        self.assertIs(get_consent_callback(), default_consent_callback)

    def test_set_then_get_roundtrips(self) -> None:
        stub = lambda name, params: True  # noqa: E731
        set_consent_callback(stub)
        self.assertIs(get_consent_callback(), stub)

    def test_reset_for_tests_restores_default(self) -> None:
        set_consent_callback(lambda name, params: False)
        _reset_for_tests()
        self.assertIs(get_consent_callback(), default_consent_callback)


class DefaultCallbackFailClosed(unittest.TestCase):
    """``default_consent_callback`` returns False on any non-approval path."""

    @patch("lib.hal.osa.consent.subprocess.run")
    def test_timeout_returns_false(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="osascript", timeout=35)
        self.assertFalse(default_consent_callback("x", {"p": "v"}))

    @patch("lib.hal.osa.consent.subprocess.run")
    def test_nonzero_exit_returns_false(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["osascript"], returncode=1, stdout="", stderr="cancelled"
        )
        self.assertFalse(default_consent_callback("x", {"p": "v"}))

    @patch("lib.hal.osa.consent.subprocess.run")
    def test_stdout_without_approve_returns_false(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["osascript"], returncode=0, stdout="gave up: true", stderr=""
        )
        self.assertFalse(default_consent_callback("x", {"p": "v"}))

    @patch("lib.hal.osa.consent.subprocess.run")
    def test_stdout_with_approve_returns_true(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["osascript"], returncode=0,
            stdout="button returned:Approve, gave up:false", stderr="",
        )
        self.assertTrue(default_consent_callback("x", {"p": "v"}))

    @patch("lib.hal.osa.consent.subprocess.run", side_effect=FileNotFoundError)
    def test_missing_osascript_returns_false(self, mock_run: MagicMock) -> None:
        self.assertFalse(default_consent_callback("x", {"p": "v"}))


class AuthoriseTier3ConsentFunction(unittest.TestCase):
    """``_authorise_tier3_consent`` — direct unit tests of the gate helper."""

    def setUp(self) -> None:
        _reset_for_tests()

    def tearDown(self) -> None:
        _reset_for_tests()

    def _script(self, tier: int = 3) -> Script:
        return Script(
            name="tier3_test_echo",
            tier=tier,
            body="return \"ok\"",
            params=(ParamSpec(name="marker"),),
            timeout_s=5,
        )

    def test_tier1_skips_consent_entirely(self) -> None:
        called = MagicMock()
        set_consent_callback(called)
        _authorise_tier3_consent(
            self._script(tier=1), {"marker": "m"},
            "local", ProvenanceType.LOCAL, LocalTransport(),
        )
        called.assert_not_called()

    def test_tier2_skips_consent_entirely(self) -> None:
        called = MagicMock()
        set_consent_callback(called)
        _authorise_tier3_consent(
            self._script(tier=2), {"marker": "m"},
            "local", ProvenanceType.LOCAL, LocalTransport(),
        )
        called.assert_not_called()

    def test_tier3_approved_does_not_raise(self) -> None:
        set_consent_callback(lambda name, params: True)
        _authorise_tier3_consent(
            self._script(), {"marker": "m"},
            "local", ProvenanceType.LOCAL, LocalTransport(),
        )

    @patch("lib.hal.osa.dispatch.write_audit")
    def test_tier3_rejected_raises_consent_declined(
        self, mock_write: MagicMock
    ) -> None:
        set_consent_callback(lambda name, params: False)
        with self.assertRaises(ConsentDeclined):
            _authorise_tier3_consent(
                self._script(), {"marker": "m"},
                "local", ProvenanceType.LOCAL, LocalTransport(),
            )

    @patch("lib.hal.osa.dispatch.write_audit")
    def test_rejection_audit_entry_has_user_rejected_provenance(
        self, mock_write: MagicMock
    ) -> None:
        set_consent_callback(lambda name, params: False)
        with self.assertRaises(ConsentDeclined):
            _authorise_tier3_consent(
                self._script(), {"marker": "m"},
                "local", ProvenanceType.AUTONOMOUS_LOOP, LocalTransport(),
            )
        mock_write.assert_called_once()
        entry: AuditEntry = mock_write.call_args[0][0]
        self.assertEqual(entry.provenance, "user_rejected")

    @patch("lib.hal.osa.dispatch.write_audit")
    def test_rejection_audit_params_are_redacted(
        self, mock_write: MagicMock
    ) -> None:
        set_consent_callback(lambda name, params: False)
        supplied = {"marker": "secret-value-42"}
        with self.assertRaises(ConsentDeclined):
            _authorise_tier3_consent(
                self._script(), supplied,
                "local", ProvenanceType.LOCAL, LocalTransport(),
            )
        entry: AuditEntry = mock_write.call_args[0][0]
        expected_hash = hashlib.sha256(b"secret-value-42").hexdigest()[:12]
        self.assertEqual(entry.params, {"marker": expected_hash})
        self.assertNotIn("secret-value-42", str(entry.params))

    @patch("lib.hal.osa.dispatch.write_audit")
    def test_callback_exception_is_treated_as_rejection(
        self, mock_write: MagicMock
    ) -> None:
        def crashing(name: str, params: dict) -> bool:
            raise RuntimeError("prompt backend failure")
        set_consent_callback(crashing)
        with self.assertRaises(ConsentDeclined):
            _authorise_tier3_consent(
                self._script(), {"marker": "m"},
                "local", ProvenanceType.LOCAL, LocalTransport(),
            )
        mock_write.assert_called_once()


class DispatchTier3EndToEnd(unittest.TestCase):
    """Full ``osa()`` dispatch — consent + audit redaction wired through."""

    _TEST_SECRET = bytes(range(32))

    def setUp(self) -> None:
        _reset_for_tests()
        _reset_auth()
        os.environ["HAL_OSA_SKIP_IMMUTABLE"] = "1"
        os.environ["HAL_OSA_CAPABILITY_SECRET"] = self._TEST_SECRET.hex()

    def tearDown(self) -> None:
        _reset_for_tests()
        _reset_auth()
        os.environ.pop("HAL_OSA_SKIP_IMMUTABLE", None)
        os.environ.pop("HAL_OSA_CAPABILITY_SECRET", None)

    def _token(self, script_name: str = "tier3_test_echo"):
        return mint_token(script_name, secret=self._TEST_SECRET)

    @patch("lib.hal.osa.dispatch.write_audit")
    @patch("lib.hal.osa.dispatch.LocalTransport")
    @patch("lib.hal.osa.dispatch.get_script")
    def test_tier3_approved_calls_transport_and_writes_redacted_audit(
        self, mock_get_script, mock_transport_cls, mock_write
    ) -> None:
        mock_get_script.return_value = Script(
            name="tier3_test_echo", tier=3, body="return \"ok\"",
            params=(ParamSpec(name="marker"),), timeout_s=5,
        )
        transport = MagicMock()
        transport.is_remote = False
        transport.host = "local"
        transport.execute.return_value = _result(stdout="tier3_ok")
        mock_transport_cls.return_value = transport
        set_consent_callback(lambda name, params: True)

        osa("tier3_test_echo", {"marker": "sensitive"},
            capability_token=self._token())

        transport.execute.assert_called_once()
        entry: AuditEntry = mock_write.call_args[0][0]
        self.assertEqual(entry.tier, 3)
        self.assertTrue(entry.ok)
        expected_hash = hashlib.sha256(b"sensitive").hexdigest()[:12]
        self.assertEqual(entry.params, {"marker": expected_hash})

    @patch("lib.hal.osa.dispatch.write_audit")
    @patch("lib.hal.osa.dispatch.LocalTransport")
    @patch("lib.hal.osa.dispatch.get_script")
    def test_tier3_rejected_raises_and_does_not_execute(
        self, mock_get_script, mock_transport_cls, mock_write
    ) -> None:
        mock_get_script.return_value = Script(
            name="tier3_test_echo", tier=3, body="return \"ok\"",
            params=(ParamSpec(name="marker"),), timeout_s=5,
        )
        transport = MagicMock()
        transport.is_remote = False
        transport.host = "local"
        mock_transport_cls.return_value = transport
        set_consent_callback(lambda name, params: False)

        with self.assertRaises(ConsentDeclined):
            osa("tier3_test_echo", {"marker": "sensitive"},
                capability_token=self._token())

        transport.execute.assert_not_called()
        entry: AuditEntry = mock_write.call_args[0][0]
        self.assertEqual(entry.provenance, "user_rejected")

    @patch("lib.hal.osa.dispatch.write_audit")
    @patch("lib.hal.osa.dispatch.LocalTransport")
    @patch("lib.hal.osa.dispatch.get_script")
    def test_yolo_mode_does_not_bypass_tier3_consent(
        self, mock_get_script, mock_transport_cls, mock_write
    ) -> None:
        """Hardcoded invariant: tier-3 consent fires even under yolo mode.

        The consent callback in this test sees the AUTONOMOUS_LOOP
        provenance and returns False; dispatch must still abort. This
        is the falsification anchor for H147 — if a new code path ever
        bypassed consent under yolo, this test flips red."""
        mock_get_script.return_value = Script(
            name="tier3_test_echo", tier=3, body="return \"ok\"",
            params=(ParamSpec(name="marker"),), timeout_s=5,
        )
        transport = MagicMock()
        transport.is_remote = False
        transport.host = "local"
        mock_transport_cls.return_value = transport
        prompts_fired: list = []

        def callback(name: str, params: dict) -> bool:
            prompts_fired.append(name)
            return False

        set_consent_callback(callback)
        with self.assertRaises(ConsentDeclined):
            osa(
                "tier3_test_echo", {"marker": "auto"},
                provenance=ProvenanceType.AUTONOMOUS_LOOP,
                capability_token=self._token(),
            )
        self.assertEqual(prompts_fired, ["tier3_test_echo"])
        transport.execute.assert_not_called()

    @patch("lib.hal.osa.dispatch.write_audit")
    @patch("lib.hal.osa.dispatch.LocalTransport")
    @patch("lib.hal.osa.dispatch.get_script")
    def test_tier1_dispatch_does_not_invoke_consent_callback(
        self, mock_get_script, mock_transport_cls, mock_write
    ) -> None:
        mock_get_script.return_value = Script(
            name="battery_get", tier=1, body="return 50", params=(),
            timeout_s=5,
        )
        transport = MagicMock()
        transport.is_remote = False
        transport.host = "local"
        transport.execute.return_value = _result(stdout="50")
        mock_transport_cls.return_value = transport
        probe = MagicMock()
        set_consent_callback(probe)

        osa("battery_get")

        probe.assert_not_called()
        transport.execute.assert_called_once()


if __name__ == "__main__":
    unittest.main()
