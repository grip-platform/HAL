"""Tests for HAL Permission Guard Engine."""
import os
import sys
import unittest

from lib.hal.guard import GuardEngine, GuardDecision, GuardRule, GuardVerdict


class TestGuardVerdict(unittest.TestCase):

    def test_allow_verdict(self):
        v = GuardVerdict(decision=GuardDecision.ALLOW)
        self.assertTrue(v.allowed)
        self.assertFalse(v.denied)
        self.assertFalse(v.needs_approval)

    def test_deny_verdict(self):
        v = GuardVerdict(decision=GuardDecision.DENY, reason="blocked")
        self.assertTrue(v.denied)
        self.assertFalse(v.allowed)
        self.assertEqual(v.reason, "blocked")

    def test_ask_verdict(self):
        v = GuardVerdict(decision=GuardDecision.ASK)
        self.assertTrue(v.needs_approval)
        self.assertFalse(v.allowed)


class TestGuardRule(unittest.TestCase):

    def test_exact_match(self):
        rule = GuardRule(tool="Bash", decision=GuardDecision.ASK)
        self.assertTrue(rule.matches("Bash", {}))
        self.assertFalse(rule.matches("Read", {}))

    def test_wildcard_match(self):
        rule = GuardRule(tool="*", decision=GuardDecision.DENY)
        self.assertTrue(rule.matches("Bash", {}))
        self.assertTrue(rule.matches("Read", {}))

    def test_pattern_match(self):
        rule = GuardRule(tool="Bash", pattern="rm -rf", decision=GuardDecision.DENY)
        self.assertTrue(rule.matches("Bash", {"command": "rm -rf /"}))
        self.assertFalse(rule.matches("Bash", {"command": "ls -la"}))


class TestGuardEngine(unittest.TestCase):

    def test_default_read_allowed(self):
        engine = GuardEngine()
        v = engine.evaluate("Read", {"file_path": "/tmp/test"})
        self.assertTrue(v.allowed)

    def test_default_bash_asks(self):
        engine = GuardEngine()
        v = engine.evaluate("Bash", {"command": "echo hi"})
        self.assertTrue(v.needs_approval)

    def test_unknown_tool_asks(self):
        engine = GuardEngine()
        v = engine.evaluate("CustomTool", {})
        self.assertTrue(v.needs_approval)

    def test_deny_overrides_allow(self):
        engine = GuardEngine(rules=[
            GuardRule(tool="Read", decision=GuardDecision.ALLOW),
            GuardRule(tool="Read", decision=GuardDecision.DENY, reason="locked"),
        ])
        v = engine.evaluate("Read")
        self.assertTrue(v.denied)

    def test_auto_approve_mode(self):
        engine = GuardEngine(auto_approve=True)
        v = engine.evaluate("Bash", {"command": "rm -rf /"})
        self.assertTrue(v.allowed)
        self.assertIn("auto-approve", v.reason)

    def test_standing_approval(self):
        engine = GuardEngine()
        engine.approve("Bash")
        v = engine.evaluate("Bash", {"command": "echo hi"})
        self.assertTrue(v.allowed)
        self.assertIn("Standing approval", v.reason)

    def test_add_rule(self):
        engine = GuardEngine(rules=[])
        engine.add_rule(GuardRule(tool="Bash", decision=GuardDecision.DENY, reason="no shell"))
        v = engine.evaluate("Bash")
        self.assertTrue(v.denied)

    def test_list_rules(self):
        engine = GuardEngine()
        rules = engine.list_rules()
        self.assertGreater(len(rules), 0)
        names = [r.tool for r in rules]
        self.assertIn("Read", names)
        self.assertIn("Bash", names)


if __name__ == "__main__":
    unittest.main()
