"""Tests for lib.hal.osa.fleet — parallel multi-host dispatch."""
from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

from lib.hal.osa.fleet import (
    FleetResult,
    default_transport_factory,
    osa_fleet,
)
from lib.hal.osa.osa import OsaResult
from lib.hal.osa.transport import LocalTransport, SSHTransport


def _ok(stdout: str = "42") -> OsaResult:
    return OsaResult(
        ok=True, returncode=0,
        stdout=stdout, stderr="",
        duration_s=0.01, language="AppleScript",
    )


class DefaultTransportFactory(unittest.TestCase):
    """default_transport_factory maps host keys to the right Transport."""

    def test_local_yields_local_transport(self) -> None:
        t = default_transport_factory("local")
        self.assertIsInstance(t, LocalTransport)

    def test_ip_yields_ssh_transport(self) -> None:
        t = default_transport_factory("100.64.0.10")
        self.assertIsInstance(t, SSHTransport)
        self.assertEqual(t.host, "100.64.0.10")
        self.assertEqual(t.ip, "100.64.0.10")

    def test_arbitrary_hostname_yields_ssh_transport(self) -> None:
        t = default_transport_factory("toopie")
        self.assertIsInstance(t, SSHTransport)
        self.assertEqual(t.host, "toopie")


class FleetBasic(unittest.TestCase):

    def test_empty_plan_returns_empty(self) -> None:
        result = osa_fleet({})
        self.assertEqual(result, {})

    @patch("lib.hal.osa.fleet.osa")
    def test_single_local_host_succeeds(self, mock_osa) -> None:
        mock_osa.return_value = _ok()
        results = osa_fleet({"local": ("battery_get", {})})
        self.assertIn("local", results)
        self.assertIsInstance(results["local"], OsaResult)
        mock_osa.assert_called_once()

    @patch("lib.hal.osa.fleet.osa")
    def test_local_host_uses_local_transport(self, mock_osa) -> None:
        mock_osa.return_value = _ok()
        osa_fleet({"local": ("battery_get", {})})
        transport = mock_osa.call_args[1]["transport"]
        self.assertIsInstance(transport, LocalTransport)

    @patch("lib.hal.osa.fleet.osa")
    def test_remote_host_uses_ssh_transport(self, mock_osa) -> None:
        mock_osa.return_value = _ok()
        osa_fleet({"100.64.0.10": ("battery_get", {})})
        transport = mock_osa.call_args[1]["transport"]
        self.assertIsInstance(transport, SSHTransport)
        self.assertEqual(transport.host, "100.64.0.10")

    @patch("lib.hal.osa.fleet.osa")
    def test_params_forwarded_correctly(self, mock_osa) -> None:
        mock_osa.return_value = _ok()
        osa_fleet({"local": ("notification", {"title": "T", "body": "B"})})
        args, kwargs = mock_osa.call_args
        self.assertEqual(args[0], "notification")
        self.assertEqual(args[1], {"title": "T", "body": "B"})

    @patch("lib.hal.osa.fleet.osa")
    def test_every_plan_host_in_result(self, mock_osa) -> None:
        mock_osa.return_value = _ok()
        plan = {
            "local": ("battery_get", {}),
            "100.1.1.1": ("volume_get", {}),
            "100.2.2.2": ("frontmost_app_get", {}),
        }
        results = osa_fleet(plan)
        self.assertEqual(set(results.keys()), set(plan.keys()))


class FleetIsolation(unittest.TestCase):
    """One host failing must not cancel other dispatches."""

    @patch("lib.hal.osa.fleet.osa")
    def test_failing_host_captured_as_exception(self, mock_osa) -> None:
        ok_result = _ok()

        def side_effect(name, params, *, transport, **kw):
            if transport.host == "bad-host":
                raise RuntimeError("ssh: connection refused")
            return ok_result

        mock_osa.side_effect = side_effect
        results = osa_fleet({
            "local": ("battery_get", {}),
            "bad-host": ("battery_get", {}),
        })
        self.assertIsInstance(results["local"], OsaResult)
        self.assertIsInstance(results["bad-host"], Exception)
        self.assertIn("connection refused", str(results["bad-host"]))

    @patch("lib.hal.osa.fleet.osa")
    def test_all_failing_returns_all_exceptions(self, mock_osa) -> None:
        mock_osa.side_effect = RuntimeError("all dead")
        results = osa_fleet({
            "100.1.1.1": ("battery_get", {}),
            "100.2.2.2": ("battery_get", {}),
        })
        for host, outcome in results.items():
            with self.subTest(host=host):
                self.assertIsInstance(outcome, Exception)

    @patch("lib.hal.osa.fleet.osa")
    def test_osa_error_captured_not_raised(self, mock_osa) -> None:
        from lib.hal.osa.osa import OsaError
        mock_osa.side_effect = OsaError("script failed")
        results = osa_fleet({"local": ("battery_get", {})})
        self.assertIsInstance(results["local"], OsaError)


class FleetParallelism(unittest.TestCase):
    """Dispatches to independent hosts must run concurrently."""

    @patch("lib.hal.osa.fleet.osa")
    def test_parallel_dispatch_intervals_overlap(self, mock_osa) -> None:
        starts: dict[str, float] = {}
        ends: dict[str, float] = {}
        lock = threading.Lock()
        delay = 0.2  # wide enough that a stalled runner still starts every worker inside it

        def slow_dispatch(name, params, *, transport, **kw):
            with lock:
                starts[transport.host] = time.monotonic()
            time.sleep(delay)
            with lock:
                ends[transport.host] = time.monotonic()
            return _ok()

        mock_osa.side_effect = slow_dispatch
        hosts = {f"host-{i}": ("battery_get", {}) for i in range(4)}

        # Use one worker per host so this test does not rely on the fleet default.
        results = osa_fleet(hosts, max_workers=len(hosts))

        self.assertEqual(len(starts), len(hosts), "every host dispatch should start")
        self.assertEqual(len(ends), len(hosts), "every host dispatch should finish")
        self.assertLess(
            max(starts.values()), min(ends.values()),
            "fleet dispatch intervals should overlap",
        )
        self.assertEqual(len(results), 4)


class FleetKwargs(unittest.TestCase):
    """Keyword arguments are threaded through to each dispatch correctly."""

    @patch("lib.hal.osa.fleet.osa")
    def test_audit_false_forwarded(self, mock_osa) -> None:
        mock_osa.return_value = _ok()
        osa_fleet({"local": ("battery_get", {})}, audit=False)
        self.assertFalse(mock_osa.call_args[1]["audit"])

    @patch("lib.hal.osa.fleet.osa")
    def test_timeout_forwarded(self, mock_osa) -> None:
        mock_osa.return_value = _ok()
        osa_fleet({"local": ("battery_get", {})}, timeout_s=42)
        self.assertEqual(mock_osa.call_args[1]["timeout_s"], 42)

    @patch("lib.hal.osa.fleet.osa")
    def test_max_workers_one_still_completes(self, mock_osa) -> None:
        mock_osa.return_value = _ok()
        results = osa_fleet(
            {"local": ("battery_get", {}), "100.1.1.1": ("volume_get", {})},
            max_workers=1,
        )
        self.assertEqual(len(results), 2)

    @patch("lib.hal.osa.fleet.osa")
    def test_custom_transport_factory_used(self, mock_osa) -> None:
        """transport_factory override replaces SSHTransport for all hosts."""
        mock_osa.return_value = _ok()
        stub_transport = LocalTransport()

        def always_local(host_key: str):
            return stub_transport

        osa_fleet(
            {"remote-host": ("battery_get", {})},
            transport_factory=always_local,
        )
        transport = mock_osa.call_args[1]["transport"]
        self.assertIs(transport, stub_transport)


if __name__ == "__main__":
    unittest.main()
