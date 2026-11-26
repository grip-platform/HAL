"""Tests for HAL event bus."""
import os, sys, unittest
from lib.hal.event_bus import EventBus

class TestEventBus(unittest.TestCase):
    def test_emit_and_receive(self):
        bus = EventBus()
        received = []
        bus.on("test", lambda d: received.append(d))
        bus.emit("test", {"key": "value"})
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["key"], "value")

    def test_no_subscribers(self):
        self.assertEqual(EventBus().emit("nobody_listens"), 0)

    def test_multiple_handlers(self):
        bus = EventBus()
        counts = [0, 0]
        bus.on("evt", lambda d: counts.__setitem__(0, counts[0] + 1))
        bus.on("evt", lambda d: counts.__setitem__(1, counts[1] + 1))
        bus.emit("evt")
        self.assertEqual(counts, [1, 1])

    def test_off(self):
        bus = EventBus()
        handler = lambda d: None
        bus.on("evt", handler)
        self.assertTrue(bus.off("evt", handler))
        self.assertEqual(bus.emit("evt"), 0)

    def test_off_nonexistent(self):
        self.assertFalse(EventBus().off("evt", lambda d: None))

    def test_handler_error_ignored(self):
        bus = EventBus()
        bus.on("evt", lambda d: 1 / 0)
        bus.emit("evt")  # Should not raise

    def test_recent(self):
        bus = EventBus()
        bus.emit("a")
        bus.emit("b")
        self.assertEqual(len(bus.recent()), 2)

    def test_clear(self):
        bus = EventBus()
        bus.on("evt", lambda d: None)
        bus.emit("evt")
        bus.clear()
        self.assertEqual(len(bus.recent()), 0)

if __name__ == "__main__":
    unittest.main()
