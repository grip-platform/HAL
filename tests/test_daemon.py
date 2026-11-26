"""Tests for HAL daemon."""
import os
import sys
import unittest
from unittest.mock import patch

from lib.hal.daemon import HALDaemon


class TestDaemonStatus(unittest.TestCase):

    @patch("lib.hal.daemon._PID_FILE", "/tmp/hal-test-daemon.pid")
    def test_status_not_running(self):
        try:
            os.remove("/tmp/hal-test-daemon.pid")
        except FileNotFoundError:
            pass
        status = HALDaemon.status()
        self.assertFalse(status["running"])

    @patch("lib.hal.daemon._PID_FILE", "/tmp/hal-test-daemon2.pid")
    def test_status_stale_pid(self):
        with open("/tmp/hal-test-daemon2.pid", "w") as f:
            f.write("99999999")  # Non-existent PID
        status = HALDaemon.status()
        self.assertFalse(status["running"])
        self.assertEqual(status.get("stale_pid"), 99999999)
        os.remove("/tmp/hal-test-daemon2.pid")


class TestDaemonInit(unittest.TestCase):

    def test_create_daemon(self):
        daemon = HALDaemon(port=3851)
        self.assertEqual(daemon._port, 3851)
        self.assertFalse(daemon._running)


class TestCliDaemonArgs(unittest.TestCase):

    def _parse(self, argv):
        original = sys.argv
        sys.argv = ["hal"] + argv
        try:
            from lib.hal.cli import _parse_args
            return _parse_args()
        finally:
            sys.argv = original

    def test_daemon_start(self):
        args = self._parse(["--daemon", "start"])
        self.assertEqual(args["daemon"], "start")

    def test_daemon_stop(self):
        args = self._parse(["--daemon", "stop"])
        self.assertEqual(args["daemon"], "stop")

    def test_daemon_status(self):
        args = self._parse(["--daemon", "status"])
        self.assertEqual(args["daemon"], "status")

    def test_daemon_default_start(self):
        args = self._parse(["--daemon"])
        self.assertEqual(args["daemon"], "start")

    def test_daemon_with_port(self):
        args = self._parse(["--daemon", "start", "--port", "8080"])
        self.assertEqual(args["daemon"], "start")
        self.assertEqual(args["port"], 8080)


if __name__ == "__main__":
    unittest.main()
