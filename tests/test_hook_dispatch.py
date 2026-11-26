"""Tests for HAL hook dispatch engine."""
import importlib.util
import inspect
import os
import sys
import unittest

# Load hook_dispatch directly from the worktree path to avoid sys.modules cache
# from ~/.hal (which other tests may have populated before this file runs).
_WORKTREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "_hal_hook_dispatch_wt",
    os.path.join(_WORKTREE, "lib", "hal", "hook_dispatch.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
HookDispatcher = _mod.HookDispatcher
HookResult = _mod.HookResult
POST_COMPACT = _mod.POST_COMPACT
ELICITATION = _mod.ELICITATION
ELICITATION_RESULT = _mod.ELICITATION_RESULT


class TestHookDispatcher(unittest.TestCase):
    """Test hook dispatch core functionality."""

    def test_creates_without_error(self):
        dispatcher = HookDispatcher()
        self.assertIsInstance(dispatcher, HookDispatcher)

    def test_dispatch_unknown_event_allows(self):
        dispatcher = HookDispatcher()
        result = dispatcher.dispatch("UnknownEvent", {"tool_name": "Test"})
        self.assertFalse(result.blocked)

    def test_hook_result_defaults(self):
        result = HookResult()
        self.assertFalse(result.blocked)
        self.assertEqual(result.reason, "")


class TestHookDispatchPaths(unittest.TestCase):
    """Test that hook dispatch resolves paths correctly."""

    def test_no_claude_dir_reference(self):
        source = inspect.getsource(_mod)
        self.assertNotIn("_CLAUDE_DIR", source,
                         "Bug: _CLAUDE_DIR still referenced in hook_dispatch.py")

    def test_hal_home_used_for_lib_path(self):
        source = inspect.getsource(_mod)
        self.assertIn("_HAL_HOME", source)


class TestNewHookEventConstants(unittest.TestCase):

    def test_post_compact_hook_registered(self):
        self.assertEqual(POST_COMPACT, "PostCompact")

    def test_elicitation_hook_registered(self):
        self.assertEqual(ELICITATION, "Elicitation")

    def test_elicitation_result_hook_registered(self):
        self.assertEqual(ELICITATION_RESULT, "ElicitationResult")


#: Fake hooks, each a real subprocess so the stdin/stdout protocol is exercised.
_GRIP_DENY = (
    'import json\n'
    'print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", '
    '"permissionDecision": "deny", "permissionDecisionReason": "no stubs"}}))\n'
)
_GRIP_ASK = (
    'import json\n'
    'print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", '
    '"permissionDecision": "ask", "permissionDecisionReason": "needs a human"}}))\n'
)
_GRIP_ALLOW = (
    'import json\n'
    'print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", '
    '"permissionDecision": "allow"}}))\n'
)
_LEGACY_DENY = 'import json\nprint(json.dumps({"decision": "deny", "reason": "legacy"}))\n'
_STOP = 'import json\nprint(json.dumps({"continue": False, "stopReason": "halt"}))\n'
_EXIT2 = 'import sys\nprint("exit-two reason")\nsys.exit(2)\n'
_CRASH = 'raise RuntimeError("boom")\n'
_GARBAGE = 'print("{not json")\n'
_SLEEP = 'import time\ntime.sleep(3)\n'
_SILENT_OK = 'pass\n'


class TestDenyShapes(unittest.TestCase):
    """HAL#621: every Claude Code response shape must be read, and a HARD hook
    that cannot answer fails closed."""

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.hooks_dir = self._tmp.name
        self._timeout = _mod.SUBPROCESS_TIMEOUT_S

    def tearDown(self):
        _mod.SUBPROCESS_TIMEOUT_S = self._timeout
        self._tmp.cleanup()

    def _dispatcher(self, hard=(), soft=()):
        for name, body in (*hard, *soft):
            with open(os.path.join(self.hooks_dir, name), "w") as fh:
                fh.write(body)
        d = HookDispatcher(hooks_dir=self.hooks_dir)
        d._hard_hooks = {name: {"event": "PreToolUse"} for name, _ in hard}
        d._soft_hooks = {name: {"event": "PreToolUse"} for name, _ in soft}
        return d

    def _dispatch(self, d):
        return d.dispatch("PreToolUse", {"tool_name": "Write", "tool_input": {"file_path": "x.py"}})

    def test_grip_permission_decision_deny_blocks(self):
        result = self._dispatch(self._dispatcher(hard=[("grip_deny.py", _GRIP_DENY)]))
        self.assertTrue(result.blocked)
        self.assertEqual(result.reason, "no stubs")
        self.assertEqual(result.hook_name, "grip_deny.py")

    def test_grip_permission_decision_ask_blocks_headless(self):
        result = self._dispatch(self._dispatcher(hard=[("grip_ask.py", _GRIP_ASK)]))
        self.assertTrue(result.blocked)
        self.assertEqual(result.reason, "needs a human")

    def test_grip_permission_decision_allow_passes(self):
        result = self._dispatch(self._dispatcher(hard=[("grip_allow.py", _GRIP_ALLOW)]))
        self.assertFalse(result.blocked)

    def test_legacy_top_level_deny_still_blocks(self):
        result = self._dispatch(self._dispatcher(hard=[("legacy.py", _LEGACY_DENY)]))
        self.assertTrue(result.blocked)
        self.assertEqual(result.reason, "legacy")

    def test_continue_false_blocks(self):
        result = self._dispatch(self._dispatcher(hard=[("stop.py", _STOP)]))
        self.assertTrue(result.blocked)
        self.assertEqual(result.reason, "halt")

    def test_exit_code_two_blocks(self):
        result = self._dispatch(self._dispatcher(hard=[("exit2.py", _EXIT2)]))
        self.assertTrue(result.blocked)
        self.assertEqual(result.reason, "exit-two reason")

    def test_hard_hook_crash_fails_closed(self):
        result = self._dispatch(self._dispatcher(hard=[("crash.py", _CRASH)]))
        self.assertTrue(result.blocked)
        self.assertIn("exit 1", result.reason)

    def test_hard_hook_garbage_output_fails_closed(self):
        result = self._dispatch(self._dispatcher(hard=[("garbage.py", _GARBAGE)]))
        self.assertTrue(result.blocked)
        self.assertIn("unparseable", result.reason)

    def test_hard_hook_timeout_fails_closed(self):
        _mod.SUBPROCESS_TIMEOUT_S = 1
        result = self._dispatch(self._dispatcher(hard=[("slow.py", _SLEEP)]))
        self.assertTrue(result.blocked)
        self.assertIn("timeout", result.reason)

    def test_hard_hook_missing_fails_closed(self):
        d = self._dispatcher()
        d._hard_hooks = {"absent.py": {"event": "PreToolUse"}}
        result = self._dispatch(d)
        self.assertTrue(result.blocked)
        self.assertIn("not found", result.reason)

    def test_hard_hook_silent_exit_zero_allows(self):
        result = self._dispatch(self._dispatcher(hard=[("ok.py", _SILENT_OK)]))
        self.assertFalse(result.blocked)

    def test_soft_hook_never_blocks_even_on_deny_or_crash(self):
        d = self._dispatcher(soft=[("soft_deny.py", _GRIP_DENY), ("soft_crash.py", _CRASH)])
        result = self._dispatch(d)
        self.assertFalse(result.blocked)
        decisions = [r.get("decision") for r in result.results]
        self.assertEqual(decisions, ["deny", "allow"])


#: Fixtures for the AGORA concerns on HAL#628, one per concern.
_ALLOW_THEN_CRASH = 'import json, sys\nprint(json.dumps({"decision": "allow"}))\nsys.exit(1)\n'
_JSON_LIST = 'print("[]")\n'
_JSON_STRING = 'print(\'"deny"\')\n'
_JSON_NULL = 'print("null")\n'
_NON_UTF8 = 'import sys\nsys.stdout.buffer.write(b"\\xff\\xfe not text")\n'
_MODULE_RAISES = 'def run(event):\n    raise RuntimeError("run blew up")\n'
_MODULE_DENIES = (
    'def run(event):\n'
    '    return {"hookSpecificOutput": {"permissionDecision": "deny", '
    '"permissionDecisionReason": "in-proc"}}\n'
)
_MODULE_RETURNS_LIST = 'def run(event):\n    return []\n'
_MIXED_PAYLOAD = (
    'import json\n'
    'print(json.dumps({"decision": "allow", "reason": "top-level ok", '
    '"hookSpecificOutput": {"permissionDecision": "deny", "permissionDecisionReason": "specific no"}}))\n'
)
_COMMENT_RUN_SCRIPT = '# def run(event): looks like a module but is a script\nimport sys\nprint("script says no")\nsys.exit(2)\n'


class TestAgoraConcernsHal628(unittest.TestCase):
    """Each test is one of the four REQUEST_CHANGES concerns, as a falsifier."""

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.hooks_dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _dispatcher(self, hard=(), soft=()):
        for name, body in (*hard, *soft):
            with open(os.path.join(self.hooks_dir, name), "w") as fh:
                fh.write(body)
        d = HookDispatcher(hooks_dir=self.hooks_dir)
        d._hard_hooks = {name: {"event": "PreToolUse"} for name, _ in hard}
        d._soft_hooks = {name: {"event": "PreToolUse"} for name, _ in soft}
        return d

    def _dispatch(self, d):
        return d.dispatch("PreToolUse", {"tool_name": "Write", "tool_input": {"file_path": "x.py"}})

    # Concern 1: an answer followed by a crash is not an answer.
    def test_hard_allow_then_nonzero_exit_fails_closed(self):
        result = self._dispatch(self._dispatcher(hard=[("allow_crash.py", _ALLOW_THEN_CRASH)]))
        self.assertTrue(result.blocked)
        self.assertIn("exit 1", result.reason)

    def test_soft_allow_then_nonzero_exit_stays_open(self):
        result = self._dispatch(self._dispatcher(soft=[("allow_crash.py", _ALLOW_THEN_CRASH)]))
        self.assertFalse(result.blocked)
        self.assertEqual(result.results[0]["decision"], "allow")

    # Concern 2: legal JSON that is not an object, and undecodable bytes, never escape as exceptions.
    def test_hard_non_object_json_fails_closed_without_raising(self):
        for name, body in (("list.py", _JSON_LIST), ("string.py", _JSON_STRING), ("null.py", _JSON_NULL)):
            with self.subTest(hook=name):
                result = self._dispatch(self._dispatcher(hard=[(name, body)]))
                self.assertTrue(result.blocked)
                self.assertIn("non-object", result.reason)

    def test_hard_non_utf8_output_fails_closed_without_raising(self):
        result = self._dispatch(self._dispatcher(hard=[("bytes.py", _NON_UTF8)]))
        self.assertTrue(result.blocked)

    # Concern 3: an in-process run() that raises or returns garbage follows the same policy.
    def test_hard_module_run_raises_fails_closed(self):
        result = self._dispatch(self._dispatcher(hard=[("module_raises.py", _MODULE_RAISES)]))
        self.assertTrue(result.blocked)
        self.assertIn("RuntimeError", result.reason)

    def test_soft_module_run_raises_stays_open(self):
        result = self._dispatch(self._dispatcher(soft=[("module_raises.py", _MODULE_RAISES)]))
        self.assertFalse(result.blocked)

    def test_module_run_grip_shape_deny_blocks(self):
        result = self._dispatch(self._dispatcher(hard=[("module_denies.py", _MODULE_DENIES)]))
        self.assertTrue(result.blocked)
        self.assertEqual(result.reason, "in-proc")

    def test_hard_module_run_non_object_fails_closed(self):
        result = self._dispatch(self._dispatcher(hard=[("module_list.py", _MODULE_RETURNS_LIST)]))
        self.assertTrue(result.blocked)
        self.assertIn("non-object", result.reason)

    # Concern 4: the specific shape carries its own reason, and the run() gate reads the AST.
    def test_mixed_payload_denies_with_the_specific_reason(self):
        result = self._dispatch(self._dispatcher(hard=[("mixed.py", _MIXED_PAYLOAD)]))
        self.assertTrue(result.blocked)
        self.assertEqual(result.reason, "specific no")

    def test_comment_mentioning_run_is_still_a_script(self):
        result = self._dispatch(self._dispatcher(hard=[("commented.py", _COMMENT_RUN_SCRIPT)]))
        self.assertTrue(result.blocked)
        self.assertEqual(result.reason, "script says no")

    def test_defines_run_reads_the_ast_not_the_text(self):
        path = os.path.join(self.hooks_dir, "probe.py")
        with open(path, "w") as fh:
            fh.write(_COMMENT_RUN_SCRIPT)
        self.assertFalse(_mod._defines_run(path))
        with open(path, "w") as fh:
            fh.write("def  run (event):\n    return {}\n")
        self.assertTrue(_mod._defines_run(path))


if __name__ == "__main__":
    unittest.main()
