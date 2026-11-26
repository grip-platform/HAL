"""Tests for HAL notification system."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lib.hal.notification import Notification, NotificationHub, notifications_dir


class TestNotification(unittest.TestCase):

    def test_to_dict(self):
        n = Notification("test", "message", "info", 1000.0)
        d = n.to_dict()
        self.assertEqual(d["event"], "test")
        self.assertEqual(d["message"], "message")

    def test_frozen(self):
        n = Notification("e", "m")
        with self.assertRaises(AttributeError):
            n.event = "changed"


class TestNotificationHub(unittest.TestCase):

    def test_notify_without_subscribers(self):
        hub = NotificationHub()
        n = hub.notify("test", "message")
        self.assertEqual(n.event, "test")

    def test_subscriber_called(self):
        hub = NotificationHub()
        received = []
        hub.subscribe("task_done", lambda n: received.append(n))
        hub.notify("task_done", "PR merged")
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].message, "PR merged")

    def test_wildcard_subscriber(self):
        hub = NotificationHub()
        received = []
        hub.subscribe("*", lambda n: received.append(n))
        hub.notify("any_event", "hello")
        self.assertEqual(len(received), 1)

    def test_subscriber_error_ignored(self):
        hub = NotificationHub()
        hub.subscribe("test", lambda n: 1 / 0)  # Will raise
        n = hub.notify("test", "still works")
        self.assertEqual(n.message, "still works")

    def test_recent(self):
        hub = NotificationHub()
        hub.notify("a", "first")
        hub.notify("b", "second")
        hub.notify("a", "third")
        recent = hub.recent(limit=2)
        self.assertEqual(len(recent), 2)

    def test_recent_filtered(self):
        hub = NotificationHub()
        hub.notify("type_a", "one")
        hub.notify("type_b", "two")
        hub.notify("type_a", "three")
        filtered = hub.recent(event="type_a")
        self.assertEqual(len(filtered), 2)
        self.assertTrue(all(n.event == "type_a" for n in filtered))

    def test_clear(self):
        hub = NotificationHub()
        hub.notify("test", "msg")
        count = hub.clear()
        self.assertEqual(count, 1)
        self.assertEqual(len(hub.recent()), 0)

    def test_severity(self):
        hub = NotificationHub()
        n = hub.notify("error", "something broke", severity="error")
        self.assertEqual(n.severity, "error")

    def test_metadata(self):
        hub = NotificationHub()
        n = hub.notify("deploy", "v1.2", pr=33, branch="main")
        self.assertEqual(n.metadata["pr"], 33)

    def test_history_capped(self):
        hub = NotificationHub()
        hub._max_history = 5
        for i in range(10):
            hub.notify("test", f"msg {i}")
        self.assertEqual(len(hub.recent(limit=100)), 5)

    def test_history_is_saved_under_the_hal_home_of_the_moment(self):
        """The directory is resolved when the history is saved, never at import. A
        value fixed at import kept the HAL_HOME the process started with, so a
        test's own HAL_HOME was ignored and the live one written instead."""
        with tempfile.TemporaryDirectory() as home, \
                mock.patch.dict(os.environ, {"HAL_HOME": home}):
            hub = NotificationHub()
            hub.notify("account_switch.stopped", "told once", severity="error")
            hub.save_history()
            saved = json.loads((Path(home) / "state/hal/notifications/history.json").read_text())
        self.assertEqual([(n["event"], n["message"]) for n in saved],
                         [("account_switch.stopped", "told once")])

    def test_without_hal_home_the_directory_is_under_the_home_of_the_moment(self):
        """With no HAL_HOME it is ~/.hal's, read from HOME when asked, not from the
        HOME the module was imported under. Asked only; nothing is written."""
        with tempfile.TemporaryDirectory() as home:
            env = {k: v for k, v in os.environ.items() if k != "HAL_HOME"}
            env["HOME"] = home
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertEqual(notifications_dir(),
                                 os.path.join(home, ".hal", "state", "hal", "notifications"))


if __name__ == "__main__":
    unittest.main()
