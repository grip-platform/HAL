"""Tests for HAL Approval Gate."""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.hal.approval_gate import (
    ApprovalGate,
    ApprovalLevel,
    ApprovalRequest,
    _AUTO_APPROVE,
    _RISK_MAP,
)


class TestApprovalLevel(unittest.TestCase):
    """Test ApprovalLevel enum."""

    def test_values(self):
        self.assertEqual(ApprovalLevel.FULL_MANUAL.value, "full_manual")
        self.assertEqual(ApprovalLevel.APPROVE_PR.value, "approve_pr")
        self.assertEqual(ApprovalLevel.APPROVE_MERGE.value, "approve_merge")
        self.assertEqual(ApprovalLevel.FULL_AUTO.value, "full_auto")

    def test_from_string(self):
        self.assertEqual(ApprovalLevel("full_auto"), ApprovalLevel.FULL_AUTO)


class TestApprovalRequest(unittest.TestCase):
    """Test ApprovalRequest frozen dataclass."""

    def test_creation(self):
        req = ApprovalRequest(
            request_id="test-1", action="write_module",
            description="Writing foo.py", risk_level="low",
        )
        self.assertEqual(req.action, "write_module")
        self.assertEqual(req.risk_level, "low")
        self.assertFalse(req.auto_approved)

    def test_frozen(self):
        req = ApprovalRequest(
            request_id="test-1", action="write_module",
            description="", risk_level="low",
        )
        with self.assertRaises(AttributeError):
            req.action = "delete_file"


class TestRiskMap(unittest.TestCase):
    """Test risk classification dispatch table."""

    def test_write_module_is_low(self):
        self.assertEqual(_RISK_MAP["write_module"], "low")

    def test_write_test_is_low(self):
        self.assertEqual(_RISK_MAP["write_test"], "low")

    def test_commit_is_low(self):
        self.assertEqual(_RISK_MAP["commit"], "low")

    def test_create_pr_is_medium(self):
        self.assertEqual(_RISK_MAP["create_pr"], "medium")

    def test_edit_module_is_medium(self):
        self.assertEqual(_RISK_MAP["edit_module"], "medium")

    def test_merge_is_high(self):
        self.assertEqual(_RISK_MAP["merge"], "high")

    def test_delete_file_is_critical(self):
        self.assertEqual(_RISK_MAP["delete_file"], "critical")

    def test_edit_config_is_high(self):
        self.assertEqual(_RISK_MAP["edit_config"], "high")


class TestAutoApproveTable(unittest.TestCase):
    """Test the auto-approve dispatch table covers all combinations."""

    def test_full_manual_blocks_everything(self):
        for risk in ("low", "medium", "high", "critical"):
            self.assertFalse(
                _AUTO_APPROVE[(ApprovalLevel.FULL_MANUAL, risk)],
                f"FULL_MANUAL should block {risk}",
            )

    def test_full_auto_blocks_critical(self):
        self.assertFalse(_AUTO_APPROVE[(ApprovalLevel.FULL_AUTO, "critical")])

    def test_full_auto_approves_low(self):
        self.assertTrue(_AUTO_APPROVE[(ApprovalLevel.FULL_AUTO, "low")])

    def test_full_auto_approves_medium(self):
        self.assertTrue(_AUTO_APPROVE[(ApprovalLevel.FULL_AUTO, "medium")])

    def test_full_auto_approves_high(self):
        self.assertTrue(_AUTO_APPROVE[(ApprovalLevel.FULL_AUTO, "high")])

    def test_approve_pr_approves_low_only(self):
        self.assertTrue(_AUTO_APPROVE[(ApprovalLevel.APPROVE_PR, "low")])
        self.assertFalse(_AUTO_APPROVE[(ApprovalLevel.APPROVE_PR, "medium")])
        self.assertFalse(_AUTO_APPROVE[(ApprovalLevel.APPROVE_PR, "high")])

    def test_approve_merge_approves_low_and_medium(self):
        self.assertTrue(_AUTO_APPROVE[(ApprovalLevel.APPROVE_MERGE, "low")])
        self.assertTrue(_AUTO_APPROVE[(ApprovalLevel.APPROVE_MERGE, "medium")])
        self.assertFalse(_AUTO_APPROVE[(ApprovalLevel.APPROVE_MERGE, "high")])

    def test_all_combinations_covered(self):
        levels = list(ApprovalLevel)
        risks = ("low", "medium", "high", "critical")
        for level in levels:
            for risk in risks:
                self.assertIn(
                    (level, risk), _AUTO_APPROVE,
                    f"Missing ({level}, {risk}) in _AUTO_APPROVE",
                )


class TestApprovalGate(unittest.TestCase):
    """Test ApprovalGate operations."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.state_path = os.path.join(self.tmp_dir, "approval-level.json")
        self.pending_path = os.path.join(self.tmp_dir, "approval-pending.json")
        self.gate = ApprovalGate(state_path=self.state_path, pending_path=self.pending_path)

    def tearDown(self):
        for f in (self.state_path, self.pending_path):
            if os.path.exists(f):
                os.unlink(f)
        os.rmdir(self.tmp_dir)

    def test_default_level(self):
        self.assertEqual(self.gate.get_level(), ApprovalLevel.APPROVE_PR)

    def test_set_level(self):
        self.gate.set_level(ApprovalLevel.FULL_AUTO)
        self.assertEqual(self.gate.get_level(), ApprovalLevel.FULL_AUTO)

    def test_set_level_persists(self):
        self.gate.set_level(ApprovalLevel.FULL_MANUAL)
        gate2 = ApprovalGate(state_path=self.state_path, pending_path=self.pending_path)
        self.assertEqual(gate2.get_level(), ApprovalLevel.FULL_MANUAL)

    def test_check_auto_approves_low_at_approve_pr(self):
        self.gate.set_level(ApprovalLevel.APPROVE_PR)
        self.assertTrue(self.gate.check("write_module", "Writing foo.py"))

    def test_check_blocks_medium_at_approve_pr(self):
        self.gate.set_level(ApprovalLevel.APPROVE_PR)
        self.assertFalse(self.gate.check("create_pr", "Creating PR"))

    def test_check_blocks_everything_at_full_manual(self):
        self.gate.set_level(ApprovalLevel.FULL_MANUAL)
        self.assertFalse(self.gate.check("write_module", "Writing foo.py"))
        self.assertFalse(self.gate.check("commit", "Committing"))

    def test_check_blocks_critical_at_full_auto(self):
        self.gate.set_level(ApprovalLevel.FULL_AUTO)
        self.assertFalse(self.gate.check("delete_file", "Deleting foo.py"))

    def test_check_approves_high_at_full_auto(self):
        self.gate.set_level(ApprovalLevel.FULL_AUTO)
        self.assertTrue(self.gate.check("merge", "Merging PR"))

    def test_unknown_action_defaults_to_high(self):
        self.gate.set_level(ApprovalLevel.APPROVE_PR)
        self.assertFalse(self.gate.check("unknown_action", "Something weird"))

    def test_pending_requests_queued(self):
        self.gate.set_level(ApprovalLevel.FULL_MANUAL)
        self.gate.check("write_module", "Writing foo.py")
        self.gate.check("commit", "Committing")
        pending = self.gate.get_pending()
        self.assertEqual(len(pending), 2)

    def test_approve_pending(self):
        self.gate.set_level(ApprovalLevel.FULL_MANUAL)
        self.gate.check("write_module", "Writing foo.py")
        pending = self.gate.get_pending()
        self.assertEqual(len(pending), 1)

        result = self.gate.approve(pending[0].request_id)
        self.assertTrue(result)
        self.assertEqual(len(self.gate.get_pending()), 0)

    def test_approve_nonexistent(self):
        self.assertFalse(self.gate.approve("nonexistent-id"))

    def test_deny_pending(self):
        self.gate.set_level(ApprovalLevel.FULL_MANUAL)
        self.gate.check("merge", "Merging PR")
        pending = self.gate.get_pending()

        result = self.gate.deny(pending[0].request_id, "Not ready")
        self.assertTrue(result)
        self.assertEqual(len(self.gate.get_pending()), 0)

    def test_deny_nonexistent(self):
        self.assertFalse(self.gate.deny("nonexistent-id"))

    def test_classify_risk(self):
        self.assertEqual(self.gate.classify_risk("write_module"), "low")
        self.assertEqual(self.gate.classify_risk("merge"), "high")
        self.assertEqual(self.gate.classify_risk("delete_file"), "critical")
        self.assertEqual(self.gate.classify_risk("unknown"), "high")

    def test_auto_approved_not_queued(self):
        self.gate.set_level(ApprovalLevel.APPROVE_PR)
        self.gate.check("write_module", "Writing foo.py")
        self.assertEqual(len(self.gate.get_pending()), 0)


if __name__ == "__main__":
    unittest.main()
