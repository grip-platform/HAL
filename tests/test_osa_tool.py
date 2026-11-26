"""Tests for the ``osa_run`` tool handler wired into :mod:`lib.hal.tool_dispatch`.

β-3 exposes the osa primitive to LLM tool loops. These tests verify the
tier-based auth gating, provenance tagging, and error-path shape without
reaching the real AppleScript kernel.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from lib.hal.osa.osa import OsaError, OsaResult
from lib.hal.osa.registry import ParamSpec, Script
from lib.hal.tool_adapter import _BUILTIN_SPECS, to_anthropic_tools
from lib.hal.tool_dispatch import _tool_osa_run


def _fake_osa_result(stdout: str = "OK", ok: bool = True) -> OsaResult:
    return OsaResult(
        ok=ok,
        returncode=0 if ok else 1,
        stdout=stdout,
        stderr="" if ok else "err",
        duration_s=0.01,
        language="AppleScript",
    )


class OsaRunRegistryIntegration(unittest.TestCase):
    """The tool appears in the built-in spec list and the Anthropic schema."""

    def test_spec_registered(self) -> None:
        names = [s.name for s in _BUILTIN_SPECS]
        self.assertIn("osa_run", names)

    def test_anthropic_schema_includes_osa_run(self) -> None:
        tools = to_anthropic_tools()
        names = [t["name"] for t in tools]
        self.assertIn("osa_run", names)
        entry = next(t for t in tools if t["name"] == "osa_run")
        self.assertIn("name", entry["input_schema"]["required"])
        self.assertEqual(
            entry["input_schema"]["properties"]["name"]["type"], "string"
        )

    def test_osa_run_is_not_concurrent_safe(self) -> None:
        """Goodhart guard — a concurrent dispatch could double-mint tokens."""
        spec = next(s for s in _BUILTIN_SPECS if s.name == "osa_run")
        self.assertFalse(spec.is_concurrent_safe)


class Tier1DispatchNoAuth(unittest.TestCase):
    """Tier-1 scripts reach the kernel without a capability token."""

    @patch("lib.hal.osa.dispatch.write_audit")
    @patch("lib.hal.osa.osa.osa_raw")
    @patch("lib.hal.osa.dispatch.get_script")
    @patch("lib.hal.osa.get_script")
    def test_tier1_dispatch_returns_stdout(
        self, mock_registry_get, mock_dispatch_get, mock_osa_raw, _mock_write
    ) -> None:
        tier1 = Script(name="battery_get", tier=1, body="return 42", params=())
        mock_registry_get.return_value = tier1
        mock_dispatch_get.return_value = tier1
        mock_osa_raw.return_value = _fake_osa_result(stdout="42\n")
        result = _tool_osa_run({"name": "battery_get"})
        self.assertEqual(result.output, "42\n")
        self.assertEqual(result.error, "")

    def test_missing_name_returns_error(self) -> None:
        result = _tool_osa_run({})
        self.assertIn("name", result.error)

    def test_unknown_script_returns_error(self) -> None:
        result = _tool_osa_run({"name": "this_script_does_not_exist"})
        self.assertIn("not in registry", result.error)


class Tier2RequiresConsent(unittest.TestCase):
    """Tier-2+ dispatch demands ``HAL_OSA_TOOL_CHAIN_AUTOMINT=1``."""

    def setUp(self) -> None:
        self._orig = os.environ.pop("HAL_OSA_TOOL_CHAIN_AUTOMINT", None)
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        if self._orig is not None:
            os.environ["HAL_OSA_TOOL_CHAIN_AUTOMINT"] = self._orig

    @patch("lib.hal.osa.get_script")
    def test_tier2_without_consent_env_returns_instructional_error(
        self, mock_get_script
    ) -> None:
        mock_get_script.return_value = Script(
            name="notification", tier=2, body="display notification {b}",
            params=(ParamSpec(name="b", required=True),),
        )
        result = _tool_osa_run({"name": "notification", "params": {"b": "hi"}})
        self.assertIn("HAL_OSA_TOOL_CHAIN_AUTOMINT", result.error)
        self.assertIn("tier-2", result.error)


class Tier2WithConsent(unittest.TestCase):
    """With consent env set, the handler auto-mints and dispatches."""

    def setUp(self) -> None:
        os.environ["HAL_OSA_TOOL_CHAIN_AUTOMINT"] = "1"
        self.addCleanup(
            lambda: os.environ.pop("HAL_OSA_TOOL_CHAIN_AUTOMINT", None)
        )

    @patch("lib.hal.osa.dispatch.write_audit")
    @patch("lib.hal.osa.osa.osa_raw")
    @patch("lib.hal.osa.dispatch.get_script")
    @patch("lib.hal.osa.get_script")
    @patch("lib.hal.osa.default_secret")
    @patch("lib.hal.osa.mint_token")
    def test_tier2_with_consent_dispatches(
        self, mock_mint, mock_secret, mock_registry_get,
        mock_dispatch_get, mock_osa_raw, _mock_write,
    ) -> None:
        tier2 = Script(name="speak", tier=2, body="say {text}",
                       params=(ParamSpec(name="text", required=True),))
        mock_registry_get.return_value = tier2
        mock_dispatch_get.return_value = tier2
        mock_secret.return_value = bytes(range(32))
        fake_token = MagicMock()
        fake_token.script_name = "speak"
        mock_mint.return_value = fake_token
        mock_osa_raw.return_value = _fake_osa_result(stdout="spoken")
        with patch("lib.hal.osa.dispatch._authorise") as mock_auth:
            result = _tool_osa_run(
                {"name": "speak", "params": {"text": "hello"}}
            )
            mock_auth.assert_called_once()
        self.assertEqual(result.output, "spoken")


class ErrorPathShape(unittest.TestCase):
    """Kernel failures surface as ToolResult(error=...)."""

    @patch("lib.hal.osa.dispatch.write_audit")
    @patch("lib.hal.osa.osa.osa_raw")
    @patch("lib.hal.osa.dispatch.get_script")
    @patch("lib.hal.osa.get_script")
    def test_osa_error_converted_to_tool_error(
        self, mock_registry_get, mock_dispatch_get, mock_osa_raw, _mock_write
    ) -> None:
        tier1 = Script(name="battery_get", tier=1, body="return 42", params=())
        mock_registry_get.return_value = tier1
        mock_dispatch_get.return_value = tier1
        mock_osa_raw.side_effect = OsaError("kernel crashed")
        result = _tool_osa_run({"name": "battery_get"})
        self.assertIn("kernel failure", result.error)
        self.assertIn("kernel crashed", result.error)


if __name__ == "__main__":
    unittest.main()
