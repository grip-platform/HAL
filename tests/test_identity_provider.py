"""Tests for HAL identity provider — CC env var fast path (CC 2.1.51)."""
import importlib.util
import os
import sys
import unittest
from unittest.mock import patch

# Load identity_provider directly from the worktree, bypassing sys.modules cache.
# Other tests in the suite may have already loaded lib.hal.identity_provider from
# ~/.hal (which lacks the CC env var fast path). Using spec_from_file_location with
# a unique module name avoids the cached version entirely.
_WORKTREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODULE_PATH = os.path.join(_WORKTREE, "lib", "hal", "identity_provider.py")
_spec = importlib.util.spec_from_file_location("_hal_identity_wt", _MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
get_identity = _mod.get_identity
Identity = _mod.Identity


class TestIdentityFromCCEnvVars(unittest.TestCase):

    @patch.dict(os.environ, {
        "CLAUDE_CODE_USER_EMAIL": "development@codetonight.co.za",
        "CLAUDE_CODE_ACCOUNT_UUID": "acct-uuid-123",
        "CLAUDE_CODE_ORGANIZATION_UUID": "org-uuid-456",
    })
    def test_identity_from_cc_env_vars(self):
        identity = get_identity()
        self.assertEqual(identity.name, "development@codetonight.co.za")
        self.assertEqual(identity.signature, "acct-uuid-123")
        self.assertEqual(identity.role, "org-uuid-456")
        self.assertEqual(identity.mode, "cc_env")

    @patch.dict(os.environ, {}, clear=True)
    def test_identity_falls_back_to_file_when_no_env(self):
        os.environ.pop("CLAUDE_CODE_USER_EMAIL", None)
        identity = get_identity(identity_path="/nonexistent/identity.md")
        self.assertIsInstance(identity, Identity)
        self.assertEqual(identity.mode, "solo")

    @patch.dict(os.environ, {"CLAUDE_CODE_USER_EMAIL": "test@example.com"})
    def test_cc_env_takes_priority_over_file(self):
        identity = get_identity()
        self.assertEqual(identity.name, "test@example.com")
        self.assertEqual(identity.mode, "cc_env")


if __name__ == "__main__":
    unittest.main()
