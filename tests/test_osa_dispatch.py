"""Tests for ``lib.hal.osa.dispatch`` — public ``osa()`` entry point."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from lib.hal.osa.auth import (
    MissingSecret,
    TokenExpired,
    TokenInvalid,
    _reset_for_tests as _reset_auth,
    mint_token,
)
from lib.hal.osa.dispatch import _apply_params, osa
from lib.hal.osa.osa import OsaError, OsaResult
from lib.hal.osa.provenance import ProvenanceType
from lib.hal.osa.registry import (
    ParamSpec,
    ParamValidationError,
    Script,
    ScriptNotRegistered,
)


_TEST_SECRET = bytes(range(32))
_TEST_SECRET_HEX = _TEST_SECRET.hex()


def _fake_result(stdout: str = "", returncode: int = 0) -> OsaResult:
    return OsaResult(
        ok=(returncode == 0),
        returncode=returncode,
        stdout=stdout,
        stderr="" if returncode == 0 else "boom",
        duration_s=0.05,
        language="AppleScript",
    )


class Templating(unittest.TestCase):
    """``_apply_params`` escapes values via applescript_escape."""

    def test_single_placeholder_replaced(self) -> None:
        result = _apply_params("notify {title}", {"title": "Hello"})
        self.assertEqual(result, 'notify "Hello"')

    def test_quote_in_value_escaped(self) -> None:
        result = _apply_params("set x to {v}", {"v": 'he said "hi"'})
        self.assertEqual(result, 'set x to "he said \\"hi\\""')

    def test_unused_placeholder_unchanged(self) -> None:
        result = _apply_params("keep {this}", {"other": "x"})
        self.assertEqual(result, "keep {this}")

    def test_empty_params_passes_body_through(self) -> None:
        self.assertEqual(_apply_params("return 1", {}), "return 1")


class DispatchHappyPath(unittest.TestCase):
    """``osa()`` resolves the registry, validates, interpolates, calls kernel."""

    def setUp(self) -> None:
        self._tmp_log = Path(tempfile.mkstemp(suffix=".log")[1])
        self.addCleanup(lambda: self._tmp_log.unlink(missing_ok=True))

    @patch("lib.hal.osa.dispatch.write_audit")
    @patch("lib.hal.osa.osa.osa_raw")
    @patch("lib.hal.osa.dispatch.get_script")
    def test_dispatch_calls_kernel_with_interpolated_body(
        self, mock_get_script, mock_osa_raw, mock_write
    ) -> None:
        mock_get_script.return_value = Script(
            name="notification",
            tier=1,  # auth gating covered in DispatchAuthorisation
            body="display notification {body} with title {title}",
            params=(
                ParamSpec(name="title", required=True),
                ParamSpec(name="body", required=True),
            ),
            timeout_s=5,
        )
        mock_osa_raw.return_value = _fake_result()
        osa("notification", {"title": "T", "body": "B"})
        args, kwargs = mock_osa_raw.call_args
        self.assertEqual(
            args[0],
            'display notification "B" with title "T"',
        )
        self.assertEqual(kwargs["timeout_s"], 5)

    @patch("lib.hal.osa.dispatch.write_audit")
    @patch("lib.hal.osa.osa.osa_raw")
    @patch("lib.hal.osa.dispatch.get_script")
    def test_audit_entry_written_on_success(
        self, mock_get_script, mock_osa_raw, mock_write
    ) -> None:
        mock_get_script.return_value = Script(
            name="battery_get", tier=1, body="return 42", params=(), timeout_s=5
        )
        mock_osa_raw.return_value = _fake_result(stdout="42\n")
        osa("battery_get")
        self.assertEqual(mock_write.call_count, 1)
        entry = mock_write.call_args[0][0]
        self.assertEqual(entry.script, "battery_get")
        self.assertTrue(entry.ok)

    @patch("lib.hal.osa.dispatch.write_audit")
    @patch("lib.hal.osa.osa.osa_raw")
    @patch("lib.hal.osa.dispatch.get_script")
    def test_audit_entry_written_on_failure_and_then_raises(
        self, mock_get_script, mock_osa_raw, mock_write
    ) -> None:
        mock_get_script.return_value = Script(
            name="x", tier=1, body="return 1", params=()
        )
        mock_osa_raw.side_effect = OsaError("kernel exploded")
        with self.assertRaises(OsaError):
            osa("x")
        self.assertEqual(mock_write.call_count, 1)
        entry = mock_write.call_args[0][0]
        self.assertFalse(entry.ok)
        self.assertIn("kernel exploded", entry.error)

    @patch("lib.hal.osa.dispatch.write_audit")
    @patch("lib.hal.osa.osa.osa_raw")
    @patch("lib.hal.osa.dispatch.get_script")
    def test_audit_false_skips_write(
        self, mock_get_script, mock_osa_raw, mock_write
    ) -> None:
        mock_get_script.return_value = Script(
            name="x", tier=1, body="return 1", params=()
        )
        mock_osa_raw.return_value = _fake_result()
        osa("x", audit=False)
        mock_write.assert_not_called()


class DispatchValidation(unittest.TestCase):
    """Validation failures never reach the kernel."""

    @patch("lib.hal.osa.dispatch.write_audit")
    @patch("lib.hal.osa.osa.osa_raw")
    @patch("lib.hal.osa.dispatch.get_script")
    def test_missing_required_param_does_not_call_kernel(
        self, mock_get_script, mock_osa_raw, mock_write
    ) -> None:
        mock_get_script.return_value = Script(
            name="x", tier=2, body="x",
            params=(ParamSpec(name="need_me", required=True),),
        )
        with self.assertRaises(ParamValidationError):
            osa("x", {})
        mock_osa_raw.assert_not_called()
        mock_write.assert_not_called()

    @patch("lib.hal.osa.dispatch.get_script")
    def test_unknown_script_raises_before_anything_else(
        self, mock_get_script
    ) -> None:
        mock_get_script.side_effect = ScriptNotRegistered("unknown")
        with self.assertRaises(ScriptNotRegistered):
            osa("unknown")


class DispatchTimeout(unittest.TestCase):
    """Per-call ``timeout_s`` overrides the registry default."""

    @patch("lib.hal.osa.dispatch.write_audit")
    @patch("lib.hal.osa.osa.osa_raw")
    @patch("lib.hal.osa.dispatch.get_script")
    def test_timeout_override_wins(
        self, mock_get_script, mock_osa_raw, _mock_write
    ) -> None:
        mock_get_script.return_value = Script(
            name="x", tier=1, body="return 1", params=(), timeout_s=5
        )
        mock_osa_raw.return_value = _fake_result()
        osa("x", timeout_s=30)
        _, kwargs = mock_osa_raw.call_args
        self.assertEqual(kwargs["timeout_s"], 30)


class DispatchAuthorisation(unittest.TestCase):
    """α-3b — tier-2+ scripts require a valid capability token."""

    def setUp(self) -> None:
        _reset_auth()
        # Clear GRIP_OSA_UNSAFE for deterministic enforcement tests.
        self._orig_unsafe = os.environ.pop("GRIP_OSA_UNSAFE", None)
        self.addCleanup(self._restore_unsafe)

    def _restore_unsafe(self) -> None:
        if self._orig_unsafe is not None:
            os.environ["GRIP_OSA_UNSAFE"] = self._orig_unsafe

    @patch("lib.hal.osa.dispatch.write_audit")
    @patch("lib.hal.osa.osa.osa_raw")
    @patch("lib.hal.osa.dispatch.get_script")
    def test_tier1_no_token_needed(
        self, mock_get_script, mock_osa_raw, _mock_write
    ) -> None:
        mock_get_script.return_value = Script(
            name="battery_get", tier=1, body="return 42", params=()
        )
        mock_osa_raw.return_value = _fake_result(stdout="42")
        osa("battery_get")  # no token — must succeed
        mock_osa_raw.assert_called_once()

    @patch("lib.hal.osa.dispatch.write_audit")
    @patch("lib.hal.osa.osa.osa_raw")
    @patch("lib.hal.osa.dispatch.get_script")
    def test_tier2_without_token_raises_before_kernel(
        self, mock_get_script, mock_osa_raw, mock_write
    ) -> None:
        mock_get_script.return_value = Script(
            name="notification", tier=2, body="noop", params=()
        )
        with self.assertRaises(MissingSecret):
            osa("notification")
        mock_osa_raw.assert_not_called()
        mock_write.assert_not_called()

    @patch("lib.hal.osa.dispatch.write_audit")
    @patch("lib.hal.osa.osa.osa_raw")
    @patch("lib.hal.osa.dispatch.get_script")
    def test_tier2_with_valid_token_dispatches(
        self, mock_get_script, mock_osa_raw, _mock_write
    ) -> None:
        mock_get_script.return_value = Script(
            name="notification", tier=2, body="noop", params=()
        )
        mock_osa_raw.return_value = _fake_result()
        with patch.dict(os.environ, {"HAL_OSA_CAPABILITY_SECRET": _TEST_SECRET_HEX}):
            token = mint_token("notification", secret=_TEST_SECRET)
            osa("notification", capability_token=token)
        mock_osa_raw.assert_called_once()

    @patch("lib.hal.osa.dispatch.write_audit")
    @patch("lib.hal.osa.osa.osa_raw")
    @patch("lib.hal.osa.dispatch.get_script")
    def test_tier2_with_token_for_wrong_script_rejected(
        self, mock_get_script, mock_osa_raw, mock_write
    ) -> None:
        mock_get_script.return_value = Script(
            name="shutdown", tier=3, body="noop", params=()
        )
        with patch.dict(os.environ, {"HAL_OSA_CAPABILITY_SECRET": _TEST_SECRET_HEX}):
            token = mint_token("notification", secret=_TEST_SECRET)
            with self.assertRaises(TokenInvalid):
                osa("shutdown", capability_token=token)
        mock_osa_raw.assert_not_called()
        mock_write.assert_not_called()

    @patch("lib.hal.osa.dispatch.write_audit")
    @patch("lib.hal.osa.osa.osa_raw")
    @patch("lib.hal.osa.dispatch.get_script")
    def test_grip_osa_unsafe_bypasses_auth(
        self, mock_get_script, mock_osa_raw, _mock_write
    ) -> None:
        """Goodhart guard — the escape hatch must work so the shell-rc warning
        is meaningful. If GRIP_OSA_UNSAFE=1, no token is required."""
        mock_get_script.return_value = Script(
            name="notification", tier=2, body="noop", params=()
        )
        mock_osa_raw.return_value = _fake_result()
        with patch.dict(os.environ, {"GRIP_OSA_UNSAFE": "1"}):
            osa("notification")  # no token — bypass
        mock_osa_raw.assert_called_once()


class DispatchProvenance(unittest.TestCase):
    """α-3b — provenance tag propagates onto the audit entry."""

    @patch("lib.hal.osa.dispatch.write_audit")
    @patch("lib.hal.osa.osa.osa_raw")
    @patch("lib.hal.osa.dispatch.get_script")
    def test_default_provenance_is_local(
        self, mock_get_script, mock_osa_raw, mock_write
    ) -> None:
        mock_get_script.return_value = Script(
            name="battery_get", tier=1, body="return 42", params=()
        )
        mock_osa_raw.return_value = _fake_result()
        osa("battery_get")
        self.assertEqual(mock_write.call_args[0][0].provenance, "local")

    @patch("lib.hal.osa.dispatch.write_audit")
    @patch("lib.hal.osa.osa.osa_raw")
    @patch("lib.hal.osa.dispatch.get_script")
    def test_explicit_provenance_recorded(
        self, mock_get_script, mock_osa_raw, mock_write
    ) -> None:
        mock_get_script.return_value = Script(
            name="battery_get", tier=1, body="return 42", params=()
        )
        mock_osa_raw.return_value = _fake_result()
        osa("battery_get", provenance=ProvenanceType.TOOL_CHAIN)
        self.assertEqual(mock_write.call_args[0][0].provenance, "tool_chain")

    @patch("lib.hal.osa.dispatch.write_audit")
    @patch("lib.hal.osa.osa.osa_raw")
    @patch("lib.hal.osa.dispatch.get_script")
    def test_provenance_captured_on_failure_path(
        self, mock_get_script, mock_osa_raw, mock_write
    ) -> None:
        mock_get_script.return_value = Script(
            name="x", tier=1, body="return 1", params=()
        )
        mock_osa_raw.side_effect = OsaError("boom")
        with self.assertRaises(OsaError):
            osa("x", provenance=ProvenanceType.AUTONOMOUS_LOOP)
        self.assertEqual(mock_write.call_args[0][0].provenance, "autonomous_loop")


class DispatchJavaScriptEscape(unittest.TestCase):
    """α-3b — JavaScript-language scripts use js_escape for interpolation."""

    @patch("lib.hal.osa.dispatch.write_audit")
    @patch("lib.hal.osa.osa.osa_raw")
    @patch("lib.hal.osa.dispatch.get_script")
    def test_jxa_body_uses_js_escape(
        self, mock_get_script, mock_osa_raw, _mock_write
    ) -> None:
        mock_get_script.return_value = Script(
            name="jxa_greet",
            tier=1,
            body="return {msg};",
            params=(ParamSpec(name="msg", required=True),),
            language="JavaScript",
        )
        mock_osa_raw.return_value = _fake_result()
        osa("jxa_greet", {"msg": "line1\nline2"})
        interpolated = mock_osa_raw.call_args[0][0]
        self.assertEqual(interpolated, 'return "line1\\nline2";')


if __name__ == "__main__":
    unittest.main()
