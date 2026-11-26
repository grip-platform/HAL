"""Tests for HAL conversation branching."""
import os
import sys
import unittest

from lib.hal.conversation_branch import Branch, BranchManager
from lib.hal.types import ChatMessage


def _msg(role: str, content: str) -> ChatMessage:
    return ChatMessage(role=role, content=content)


class TestBranchManager(unittest.TestCase):

    def test_create_branch(self):
        manager = BranchManager("test-session")
        history = [_msg("user", "Q1"), _msg("assistant", "A1"), _msg("user", "Q2"), _msg("assistant", "A2")]
        branch_id = manager.create_branch(history, branch_point=2, description="Explore alternative")
        self.assertTrue(branch_id.startswith("branch-"))

    def test_branch_has_correct_history(self):
        manager = BranchManager()
        history = [_msg("user", "Q1"), _msg("assistant", "A1"), _msg("user", "Q2")]
        branch_id = manager.create_branch(history, branch_point=2)
        branch = manager.get_branch(branch_id)
        self.assertEqual(len(branch.history), 2)
        self.assertEqual(branch.history[0].content, "Q1")

    def test_branch_at_end(self):
        manager = BranchManager()
        history = [_msg("user", "Q1"), _msg("assistant", "A1")]
        branch_id = manager.create_branch(history)  # branch_point = -1 (end)
        branch = manager.get_branch(branch_id)
        self.assertEqual(len(branch.history), 2)

    def test_add_to_branch(self):
        manager = BranchManager()
        history = [_msg("user", "Q1")]
        branch_id = manager.create_branch(history)
        self.assertTrue(manager.add_to_branch(branch_id, _msg("assistant", "A1")))
        branch = manager.get_branch(branch_id)
        self.assertEqual(len(branch.history), 2)

    def test_add_to_nonexistent(self):
        manager = BranchManager()
        self.assertFalse(manager.add_to_branch("fake-id", _msg("user", "Q")))

    def test_list_branches(self):
        manager = BranchManager()
        history = [_msg("user", "Q")]
        manager.create_branch(history, description="Branch A")
        manager.create_branch(history, description="Branch B")
        branches = manager.list_branches()
        self.assertEqual(len(branches), 2)
        self.assertIn("branch_id", branches[0])

    def test_delete_branch(self):
        manager = BranchManager()
        bid = manager.create_branch([_msg("user", "Q")])
        self.assertTrue(manager.delete_branch(bid))
        self.assertIsNone(manager.get_branch(bid))

    def test_delete_nonexistent(self):
        manager = BranchManager()
        self.assertFalse(manager.delete_branch("fake"))

    def test_get_nonexistent(self):
        manager = BranchManager()
        self.assertIsNone(manager.get_branch("fake"))


class TestBranch(unittest.TestCase):

    def test_to_dict(self):
        branch = Branch("b1", "session", 2, 1000.0, description="test")
        d = branch.to_dict()
        self.assertEqual(d["branch_id"], "b1")
        self.assertEqual(d["branch_point"], 2)
        self.assertEqual(d["turn_count"], 0)


if __name__ == "__main__":
    unittest.main()
