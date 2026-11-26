"""Tests for HAL scheduler."""
import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from lib.hal.scheduler import Schedule, Scheduler


class TestSchedule(unittest.TestCase):

    def test_create_schedule(self):
        sched = Schedule(name="test", interval_s=60, command="do_thing")
        self.assertEqual(sched.name, "test")
        self.assertEqual(sched.interval_s, 60)
        self.assertTrue(sched.enabled)

    def test_disabled_schedule(self):
        sched = Schedule(name="off", interval_s=60, command="noop", enabled=False)
        self.assertFalse(sched.enabled)


class TestScheduler(unittest.TestCase):

    @patch("lib.hal.scheduler._SCHEDULES_PATH", "/tmp/hal-test-schedules.json")
    def test_add_and_list(self):
        try:
            os.remove("/tmp/hal-test-schedules.json")
        except FileNotFoundError:
            pass
        scheduler = Scheduler()
        scheduler.add(Schedule(name="warmup", interval_s=300, command="warmup_models"))
        schedules = scheduler.list_schedules()
        self.assertEqual(len(schedules), 1)
        self.assertEqual(schedules[0].name, "warmup")

    @patch("lib.hal.scheduler._SCHEDULES_PATH", "/tmp/hal-test-schedules2.json")
    def test_remove(self):
        try:
            os.remove("/tmp/hal-test-schedules2.json")
        except FileNotFoundError:
            pass
        scheduler = Scheduler()
        scheduler.add(Schedule(name="temp", interval_s=60, command="test"))
        self.assertTrue(scheduler.remove("temp"))
        self.assertEqual(len(scheduler.list_schedules()), 0)

    @patch("lib.hal.scheduler._SCHEDULES_PATH", "/tmp/hal-test-schedules3.json")
    def test_remove_nonexistent(self):
        try:
            os.remove("/tmp/hal-test-schedules3.json")
        except FileNotFoundError:
            pass
        scheduler = Scheduler()
        self.assertFalse(scheduler.remove("nonexistent"))

    @patch("lib.hal.scheduler._SCHEDULES_PATH", "/tmp/hal-test-schedules4.json")
    def test_persistence(self):
        try:
            os.remove("/tmp/hal-test-schedules4.json")
        except FileNotFoundError:
            pass
        scheduler = Scheduler()
        scheduler.add(Schedule(name="persist", interval_s=120, command="save"))

        # Load fresh
        scheduler2 = Scheduler()
        schedules = scheduler2.list_schedules()
        self.assertEqual(len(schedules), 1)
        self.assertEqual(schedules[0].command, "save")

    @patch("lib.hal.scheduler._SCHEDULES_PATH", "/tmp/hal-test-schedules5.json")
    def test_handler_registration(self):
        try:
            os.remove("/tmp/hal-test-schedules5.json")
        except FileNotFoundError:
            pass
        scheduler = Scheduler()
        called = []
        scheduler.register_handler("test_cmd", lambda: called.append(True))
        self.assertIn("test_cmd", scheduler._handlers)


class TestSchedulerExecution(unittest.TestCase):

    @patch("lib.hal.scheduler._SCHEDULES_PATH", "/tmp/hal-test-schedules6.json")
    def test_start_and_stop(self):
        try:
            os.remove("/tmp/hal-test-schedules6.json")
        except FileNotFoundError:
            pass
        scheduler = Scheduler()
        scheduler.add(Schedule(name="noop", interval_s=3600, command="noop"))
        scheduler.start()
        self.assertTrue(scheduler._running)
        scheduler.stop()
        self.assertFalse(scheduler._running)


if __name__ == "__main__":
    unittest.main()
