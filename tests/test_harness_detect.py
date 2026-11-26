"""Tests for HAL harness detection."""
import importlib.util
import os
import sys
import unittest
from unittest.mock import patch

# Load harness_detect directly from the worktree path to avoid sys.modules cache
# from ~/.hal (which other tests may have populated before this file runs).
_WORKTREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "_hal_harness_detect_wt",
    os.path.join(_WORKTREE, "lib", "hal", "harness_detect.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Harness = _mod.Harness
HarnessInfo = _mod.HarnessInfo
detect_harness = _mod.detect_harness
detect_harness_info = _mod.detect_harness_info
is_standalone = _mod.is_standalone


class TestDetectHarness(unittest.TestCase):

    @patch.dict(os.environ, {}, clear=True)
    def test_standalone_when_no_env(self):
        os.environ.pop("CLAUDE_CODE_SESSION_ID", None)
        self.assertEqual(detect_harness(), Harness.STANDALONE)

    @patch.dict(os.environ, {"CLAUDE_CODE_SESSION_ID": "test-session"})
    def test_detects_claude_code(self):
        self.assertEqual(detect_harness(), Harness.CLAUDE_CODE)

    @patch.dict(os.environ, {"GEMINI_CLI_SESSION": "gemini-123"})
    def test_detects_gemini(self):
        result = detect_harness()
        self.assertIn(result, [Harness.GEMINI_CLI, Harness.CLAUDE_CODE])

    @patch.dict(os.environ, {"OPENCODE_SESSION_ID": "oc-123"}, clear=True)
    def test_detects_opencode(self):
        self.assertEqual(detect_harness(), Harness.OPENCODE)

    @patch.dict(os.environ, {"CLAUDE_CODE_SESSION_ID": "cc-1",
                             "OPENCODE_SESSION_ID": "oc-1"}, clear=True)
    def test_claude_code_still_wins_alongside_opencode(self):
        # First-match-wins: the CLAUDE_CODE rule precedes OPENCODE, so adding
        # OpenCode never steals an active Claude Code session.
        self.assertEqual(detect_harness(), Harness.CLAUDE_CODE)


class TestIsStandalone(unittest.TestCase):

    def test_standalone(self):
        with patch.object(_mod, "detect_harness", return_value=Harness.STANDALONE):
            self.assertTrue(is_standalone())

    def test_not_standalone(self):
        with patch.object(_mod, "detect_harness", return_value=Harness.CLAUDE_CODE):
            self.assertFalse(is_standalone())


class TestHarnessEnum(unittest.TestCase):

    def test_values(self):
        self.assertEqual(Harness.STANDALONE.value, "standalone")
        self.assertEqual(Harness.CLAUDE_CODE.value, "claude-code")
        self.assertEqual(Harness.GEMINI_CLI.value, "gemini-cli")
        self.assertEqual(Harness.CODEX.value, "codex")
        self.assertEqual(Harness.CURSOR.value, "cursor")
        self.assertEqual(Harness.OPENCODE.value, "opencode")


class TestHarnessInfo(unittest.TestCase):

    @patch.dict(os.environ, {}, clear=True)
    def test_sparse_paths_empty_by_default(self):
        os.environ.pop("CLAUDE_CODE_SESSION_ID", None)
        os.environ.pop("CLAUDE_WORKTREE_SPARSE_PATHS", None)
        info = detect_harness_info()
        self.assertIsInstance(info, HarnessInfo)
        self.assertEqual(info.sparse_paths, [])

    @patch.dict(os.environ, {"CLAUDE_WORKTREE_SPARSE_PATHS": "src,lib,tests"})
    def test_sparse_paths_parsed_from_env(self):
        os.environ.pop("CLAUDE_CODE_SESSION_ID", None)
        info = detect_harness_info()
        self.assertEqual(info.sparse_paths, ["src", "lib", "tests"])

    @patch.dict(os.environ, {"CLAUDE_WORKTREE_SPARSE_PATHS": "apps/web"})
    def test_sparse_paths_single_entry(self):
        os.environ.pop("CLAUDE_CODE_SESSION_ID", None)
        info = detect_harness_info()
        self.assertEqual(info.sparse_paths, ["apps/web"])


if __name__ == "__main__":
    unittest.main()
