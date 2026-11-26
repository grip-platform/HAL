"""Tests for HAL A2A (Agent-to-Agent) message bus."""
import os
import sys
import unittest

from lib.hal.a2a import DEFAULT_QUEUE_DEPTH, A2ABus, A2AMessage


class TestA2AMessage(unittest.TestCase):

    def test_create_message(self):
        msg = A2AMessage(from_agent="a1", to_agent="a2", content="hello")
        self.assertEqual(msg.from_agent, "a1")
        self.assertEqual(msg.to_agent, "a2")
        self.assertGreater(msg.timestamp, 0)

    def test_frozen(self):
        msg = A2AMessage(from_agent="a1", to_agent="a2", content="test")
        with self.assertRaises(AttributeError):
            msg.content = "changed"


class TestA2ABus(unittest.TestCase):

    def test_send_and_receive(self):
        bus = A2ABus()
        bus.send(A2AMessage(from_agent="a1", to_agent="a2", content="hello"))
        msgs = bus.receive("a2")
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].content, "hello")

    def test_receive_empties_queue(self):
        bus = A2ABus()
        bus.send(A2AMessage(from_agent="a1", to_agent="a2", content="once"))
        bus.receive("a2")
        msgs = bus.receive("a2")
        self.assertEqual(len(msgs), 0)

    def test_broadcast(self):
        bus = A2ABus()
        bus.broadcast("a1", "all agents: update")
        msgs_a2 = bus.receive("a2")
        msgs_a3 = bus.receive("a3")
        self.assertEqual(len(msgs_a2), 1)
        self.assertEqual(len(msgs_a3), 1)

    def test_broadcast_excludes_sender(self):
        bus = A2ABus()
        bus.broadcast("a1", "not for me")
        msgs = bus.receive("a1")
        self.assertEqual(len(msgs), 0)

    def test_pending_count(self):
        bus = A2ABus()
        bus.send(A2AMessage(from_agent="a1", to_agent="a2", content="1"))
        bus.send(A2AMessage(from_agent="a1", to_agent="a2", content="2"))
        self.assertEqual(bus.pending_count("a2"), 2)

    def test_clear(self):
        bus = A2ABus()
        bus.send(A2AMessage(from_agent="a1", to_agent="a2", content="test"))
        bus.broadcast("a1", "broadcast")
        bus.clear()
        self.assertEqual(bus.pending_count("a2"), 0)

    def test_no_messages_for_unknown(self):
        bus = A2ABus()
        msgs = bus.receive("nonexistent")
        self.assertEqual(msgs, [])


class TestA2ABusQueueCap(unittest.TestCase):
    """Bounded queue depth — DoS mitigation via drop-oldest eviction."""

    def test_default_max_queue_depth(self):
        bus = A2ABus()
        self.assertEqual(bus.max_queue_depth, DEFAULT_QUEUE_DEPTH)

    def test_custom_max_queue_depth(self):
        bus = A2ABus(max_queue_depth=5)
        self.assertEqual(bus.max_queue_depth, 5)

    def test_rejects_invalid_max_queue_depth(self):
        with self.assertRaises(ValueError):
            A2ABus(max_queue_depth=0)
        with self.assertRaises(ValueError):
            A2ABus(max_queue_depth=-1)

    def test_per_queue_flood_capped_at_max_depth(self):
        """Flooding one queue past the cap must not exceed max_queue_depth."""
        cap = 100
        bus = A2ABus(max_queue_depth=cap)
        flood_count = cap * 5  # 5x the cap
        for i in range(flood_count):
            bus.send(A2AMessage(
                from_agent="attacker", to_agent="victim", content=f"msg-{i}"
            ))
        self.assertEqual(bus.pending_count("victim"), cap)

    def test_drop_oldest_eviction_policy(self):
        """When a queue is full, the newest message wins and the oldest drops."""
        cap = 3
        bus = A2ABus(max_queue_depth=cap)
        for i in range(5):
            bus.send(A2AMessage(
                from_agent="a1", to_agent="victim", content=f"msg-{i}"
            ))
        msgs = bus.receive("victim")
        contents = [m.content for m in msgs]
        # Oldest two (msg-0, msg-1) were dropped; newest three survived
        self.assertEqual(contents, ["msg-2", "msg-3", "msg-4"])

    def test_broadcast_queue_also_capped(self):
        """The broadcast queue is bounded by the same cap as per-agent queues."""
        cap = 50
        bus = A2ABus(max_queue_depth=cap)
        for i in range(cap * 3):
            bus.broadcast(from_agent="spammer", content=f"bcast-{i}")
        # Each receiver sees at most `cap` broadcasts (all from "spammer")
        msgs = bus.receive("observer")
        self.assertEqual(len(msgs), cap)

    def test_flood_memory_residency_bounded(self):
        """Big flood (20k messages) caps memory at max_queue_depth."""
        bus = A2ABus(max_queue_depth=DEFAULT_QUEUE_DEPTH)
        for i in range(20_000):
            bus.send(A2AMessage(
                from_agent="flooder", to_agent="target", content=f"m{i}"
            ))
        self.assertEqual(bus.pending_count("target"), DEFAULT_QUEUE_DEPTH)


if __name__ == "__main__":
    unittest.main()
