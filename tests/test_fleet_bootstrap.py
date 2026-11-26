"""Tests for fleet_bootstrap — credential loading + adapter registration."""
import unittest
from unittest.mock import MagicMock, patch

from lib.hal.fleet_bootstrap import (
    BootstrapReport,
    bootstrap_fleet,
    fleet_status_text,
    _keychain_get,
)


class TestKeychainGet(unittest.TestCase):
    @patch("lib.hal.fleet_bootstrap.subprocess.run")
    def test_returns_token_on_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="secret123\n")
        self.assertEqual(_keychain_get("hal-test"), "secret123")

    @patch("lib.hal.fleet_bootstrap.subprocess.run")
    def test_returns_none_on_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=44, stdout="")
        self.assertIsNone(_keychain_get("hal-test"))

    @patch("lib.hal.fleet_bootstrap.subprocess.run", side_effect=FileNotFoundError)
    def test_returns_none_on_no_security_binary(self, _mock):
        self.assertIsNone(_keychain_get("hal-test"))


class TestBootstrapFleet(unittest.TestCase):
    @patch("lib.hal.fleet_bootstrap._keychain_get", return_value=None)
    def test_no_credentials_skips_all(self, _mock):
        router = MagicMock()
        report = bootstrap_fleet(router)
        self.assertEqual(len(report.skipped), 3)
        self.assertEqual(len(report.active), 0)
        router.register.assert_not_called()

    @patch("lib.hal.fleet_bootstrap._import_adapter")
    @patch("lib.hal.fleet_bootstrap._keychain_get")
    def test_telegram_registered_when_token_present(self, mock_kc, mock_import):
        mock_kc.side_effect = lambda s: "tok123" if s == "hal-telegram" else None
        mock_adapter_cls = MagicMock()
        mock_import.return_value = mock_adapter_cls

        router = MagicMock()
        report = bootstrap_fleet(router)

        self.assertIn("telegram", report.active)
        self.assertEqual(len(report.skipped), 2)
        router.register.assert_called_once()

    @patch("lib.hal.fleet_bootstrap._import_adapter", side_effect=ImportError("no module"))
    @patch("lib.hal.fleet_bootstrap._keychain_get", return_value="token")
    def test_import_error_recorded(self, _kc, _imp):
        router = MagicMock()
        report = bootstrap_fleet(router)
        self.assertGreater(len(report.errors), 0)


class TestFleetStatusText(unittest.TestCase):
    def test_all_active(self):
        r = BootstrapReport(active=["telegram", "slack"], skipped=[], errors={})
        text = fleet_status_text(r)
        self.assertIn("2 active", text)
        self.assertIn("telegram", text)

    def test_mixed(self):
        r = BootstrapReport(active=["telegram"], skipped=["slack", "discord"], errors={})
        text = fleet_status_text(r)
        self.assertIn("1 active", text)
        self.assertIn("2 need credentials", text)

    def test_empty(self):
        r = BootstrapReport()
        text = fleet_status_text(r)
        self.assertIn("0 active", text)


if __name__ == "__main__":
    unittest.main()
