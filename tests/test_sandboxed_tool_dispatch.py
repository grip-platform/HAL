"""Tests for lib/hal/sandboxed_tool_dispatch.py — the capability-locked,
sandbox-bound dispatcher for ``/api/infer-with-tools``.

Goodhart-resistant per Rule 14: every test fails under at least one
obvious code mutation; each docstring names the mutation it survives.

Test register (W2 capability-lock + integration anchors):

| ID  | Mutation it survives                                            |
|-----|-----------------------------------------------------------------|
| T5  | Drop allowlist check → Bash/Write/etc. reach handlers           |
| T6  | Replace frozenset with mutable set → external mutation possible |
| T7  | Drop sandbox validation in __init__ → bad root succeeds         |
| T8  | Drop path canonicalisation → root attribute carries raw input   |
| T9  | Drop Read handler dispatch → no read possible                   |
| T10 | Drop Grep handler dispatch → no grep possible                   |
| T11 | Drop Glob handler dispatch → no glob possible                   |
| T12 | Each-forbidden-absent (parametrised) → adding any to allowlist  |
| T13 | Skip allowlist check for unknown names → unknown not blocked    |
| T14 | Drop escape check on Read → ../../../etc/passwd leaks           |
| T15 | Drop escape check on Glob → /etc/* leaks                        |
| T16 | Return wrong attribute names → loop's tool_result builder breaks|
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Path portability — import from this worktree's lib/hal/
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._platform import can_create_symlinks

from lib.hal.file_sandbox import SandboxRootRefused  # noqa: E402
from lib.hal.sandboxed_tool_dispatch import (  # noqa: E402
    SandboxedDispatcher,
    SandboxedResult,
    make_sandboxed_dispatcher,
)


class _TmpSandbox(unittest.TestCase):
    """Per-test temp directory rooted under ``/private/var/folders``
    (macOS user tempdir). We use ``realpath`` so the dispatcher's
    canonical root matches what we wrote."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = os.path.realpath(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, rel: str, content: str) -> Path:
        p = Path(self.root) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p


# --- T5/T13: capability allowlist enforcement -----------------------------


class TestCapabilityLock(_TmpSandbox):

    def test_bash_is_blocked(self):
        """T5 — Mutation: drop the allowlist check → Bash reaches a
        handler. Survives by asserting ``blocked=True`` for Bash."""
        d = SandboxedDispatcher(self.root)
        r = d.execute("Bash", {"command": "ls -la /"})
        self.assertTrue(r.blocked)
        self.assertIn("allowlist", r.reason.lower())

    def test_write_is_blocked(self):
        """T5b — Mutation: drop allowlist check on Write → file
        creation reaches handler. Survives by asserting blocked=True."""
        d = SandboxedDispatcher(self.root)
        r = d.execute("Write", {"file_path": "/tmp/x", "content": "y"})
        self.assertTrue(r.blocked)

    def test_edit_is_blocked(self):
        """T5c — Mutation: drop allowlist check on Edit → file
        mutation reaches handler."""
        d = SandboxedDispatcher(self.root)
        r = d.execute("Edit", {"file_path": "/tmp/x", "old_string": "a",
                                "new_string": "b"})
        self.assertTrue(r.blocked)

    def test_each_forbidden_tool_blocked_parametrised(self):
        """T12 — Mutation: ADD any of these names to the allowlist →
        a known-dangerous tool gets through. Parametrised so the test
        DOES NOT depend on a single tool name — every entry must be
        denied independently."""
        d = SandboxedDispatcher(self.root)
        for tool in (
            "Bash", "Write", "Edit", "NotebookEdit",
            "Task", "Agent", "WebFetch", "WebSearch",
            "TodoWrite", "ExitPlanMode", "EnterPlanMode",
            "BashOutput", "KillBash",
        ):
            with self.subTest(tool=tool):
                r = d.execute(tool, {})
                self.assertTrue(
                    r.blocked,
                    f"forbidden tool {tool!r} reached handler (not blocked)",
                )

    def test_unknown_tool_name_blocked(self):
        """T13 — Mutation: skip the allowlist check when the name is
        unknown to the dispatch table → ``execute("Fnord", {})``
        falls through to the unreachable branch which currently also
        denies but for the wrong reason. Survives by asserting the
        capability-lock reason mentions 'allowlist'."""
        d = SandboxedDispatcher(self.root)
        r = d.execute("Fnord", {})
        self.assertTrue(r.blocked)
        self.assertIn("allowlist", r.reason.lower())

    def test_empty_tool_name_blocked(self):
        """T13b — Mutation: treat empty name as a legitimate fall-through
        → any future cascade matches '' silently. Survives by asserting
        empty string is blocked at the allowlist gate."""
        d = SandboxedDispatcher(self.root)
        r = d.execute("", {})
        self.assertTrue(r.blocked)


# --- T6: allowlist is immutable -----------------------------------------


class TestAllowlistImmutable(_TmpSandbox):

    def test_allowlist_is_frozenset(self):
        """T6 — Mutation: change ``_ALLOWED_TOOLS`` to a mutable set →
        an attacker who imports the module can ``.add("Bash")`` at
        runtime. Survives by asserting the public ``allowlist``
        property returns a ``frozenset``."""
        d = SandboxedDispatcher(self.root)
        self.assertIsInstance(d.allowlist, frozenset)

    def test_allowlist_exact_membership(self):
        """T6b — Mutation: add an extra tool to the allowlist set →
        capability surface widens silently. Survives by hardcoding
        the expected exact set."""
        d = SandboxedDispatcher(self.root)
        self.assertEqual(set(d.allowlist), {"Read", "Grep", "Glob"})

    def test_module_constant_is_frozenset(self):
        """T6c — Mutation: use a list literal instead of frozenset →
        ``.add`` available. Survives by importing the module-level
        constant and asserting type."""
        from lib.hal.sandboxed_tool_dispatch import _ALLOWED_TOOLS
        self.assertIsInstance(_ALLOWED_TOOLS, frozenset)
        self.assertEqual(_ALLOWED_TOOLS, frozenset({"Read", "Grep", "Glob"}))


# --- T7/T8: construction validates root ---------------------------------


class TestRootValidation(unittest.TestCase):

    def test_empty_root_refused(self):
        """T7 — Mutation: skip the ``validate_root`` call in __init__ →
        empty/garbage roots produce a working dispatcher. Survives by
        asserting :class:`SandboxRootRefused` on empty root."""
        with self.assertRaises(SandboxRootRefused):
            SandboxedDispatcher("")

    def test_nonexistent_root_refused(self):
        """T7b — Mutation: skip realpath/exists check → silent creation.
        Survives by asserting refusal on /nonexistent/...."""
        with self.assertRaises(SandboxRootRefused):
            SandboxedDispatcher("/nonexistent/path/anywhere/abc123xyz")

    def test_system_root_refused(self):
        """T7c — Mutation: drop the refused-prefix set → ``/etc`` works.
        Survives by asserting refusal on ``/etc``."""
        if not os.path.isdir("/etc"):
            self.skipTest("/etc not present on this host")
        with self.assertRaises(SandboxRootRefused):
            SandboxedDispatcher("/etc")

    def test_root_canonicalised(self):
        """T8 — Mutation: return caller's input as ``sandbox_root`` →
        trailing-slash + symlink variants compare unequal. Survives by
        asserting ``sandbox_root`` matches ``os.path.realpath(input)``."""
        with tempfile.TemporaryDirectory() as tmp:
            real = os.path.realpath(tmp)
            d = SandboxedDispatcher(tmp + "/")
            self.assertEqual(d.sandbox_root, real)


# --- T9/T10/T11: handler dispatch table -----------------------------------


class TestDispatchTable(_TmpSandbox):

    def test_read_dispatches(self):
        """T9 — Mutation: drop the ``if tool_name == "Read"`` branch →
        Read calls fall through to the unreachable branch and block.
        Survives by asserting Read returns ``output`` matching disk."""
        self._write("hello.txt", "world\n")
        d = SandboxedDispatcher(self.root)
        r = d.execute("Read", {"file_path": "hello.txt"})
        self.assertFalse(r.blocked)
        self.assertEqual(r.output, "world\n")

    def test_read_accepts_path_arg_too(self):
        """T9b — Mutation: only honour ``file_path`` → the Anthropic
        tool spec uses ``path`` in some shapes and Read silently fails.
        Survives by asserting Read works with ``path``."""
        self._write("hi.txt", "ok\n")
        d = SandboxedDispatcher(self.root)
        r = d.execute("Read", {"path": "hi.txt"})
        self.assertEqual(r.output, "ok\n")

    def test_grep_dispatches(self):
        """T10 — Mutation: drop the Grep branch → grep silently blocks.
        Survives by asserting hits returned in ``path:line:text`` form."""
        self._write("a.py", "needle\n")
        d = SandboxedDispatcher(self.root)
        r = d.execute("Grep", {"pattern": "needle"})
        self.assertFalse(r.blocked)
        self.assertIn("a.py:1:needle", r.output)

    def test_glob_dispatches(self):
        """T11 — Mutation: drop the Glob branch → glob silently blocks.
        Survives by asserting both files surface in the result."""
        self._write("a.py", "")
        self._write("sub/b.py", "")
        d = SandboxedDispatcher(self.root)
        r = d.execute("Glob", {"pattern": "**/*.py"})
        self.assertFalse(r.blocked)
        self.assertIn("a.py", r.output)
        self.assertIn("sub/b.py", r.output)


# --- T14/T15: sandbox escape refused via dispatcher -----------------------


class TestSandboxEscapeThroughDispatcher(_TmpSandbox):

    def test_read_traversal_blocked(self):
        """T14 — Mutation: drop the ``..`` check in file_sandbox →
        ``../../../etc/passwd`` returns host content. Survives by
        asserting blocked=True with reason mentioning 'escape'."""
        d = SandboxedDispatcher(self.root)
        r = d.execute("Read", {"file_path": "../../../etc/passwd"})
        self.assertTrue(r.blocked)
        self.assertIn("escape", r.reason.lower())

    def test_read_symlink_escape_blocked(self):
        """T14b — Mutation: drop the realpath resolution → a symlink
        pointing at /etc/passwd returns host content. Survives by
        asserting blocked when a symlink leaves the sandbox."""
        if not can_create_symlinks():
            self.skipTest(
                "this host cannot create symlinks without an "
                "elevated/Developer-Mode privilege"
            )
        outside = Path(tempfile.mkdtemp())
        try:
            target = outside / "secret.txt"
            target.write_text("HOST_SECRET\n", encoding="utf-8")
            link = Path(self.root) / "evil.txt"
            link.symlink_to(target)
            d = SandboxedDispatcher(self.root)
            r = d.execute("Read", {"file_path": "evil.txt"})
            self.assertTrue(r.blocked)
            self.assertIn("escape", r.reason.lower())
        finally:
            shutil.rmtree(outside)

    def test_glob_absolute_pattern_blocked(self):
        """T15 — Mutation: drop the absolute-pattern check on Glob →
        ``/etc/*`` lists system contents. Survives by asserting
        blocked=True for absolute glob patterns."""
        d = SandboxedDispatcher(self.root)
        r = d.execute("Glob", {"pattern": "/etc/*"})
        self.assertTrue(r.blocked)


# --- T16: SandboxedResult shape matches ToolResult contract ---------------


class TestResultContract(_TmpSandbox):

    def test_attributes_match_tool_loop_contract(self):
        """T16 — Mutation: rename ``output`` to ``result`` (or any other
        attribute the loop's ``_build_tool_result`` reads) → the
        Anthropic tool_loop driver crashes / returns empty strings.
        Survives by asserting all four attributes exist with the
        correct name and type."""
        self._write("a.txt", "x\n")
        d = SandboxedDispatcher(self.root)
        r = d.execute("Read", {"file_path": "a.txt"})
        # The four attributes the loop's _build_tool_result reads
        self.assertTrue(hasattr(r, "output"))
        self.assertTrue(hasattr(r, "error"))
        self.assertTrue(hasattr(r, "blocked"))
        self.assertTrue(hasattr(r, "reason"))
        self.assertEqual(r.output, "x\n")
        self.assertEqual(r.error, "")
        self.assertFalse(r.blocked)
        self.assertEqual(r.reason, "")

    def test_result_is_sandboxed_result_instance(self):
        """T16b — Mutation: return a plain dict / tuple → callers that
        ``isinstance`` check break. Survives by asserting type."""
        d = SandboxedDispatcher(self.root)
        r = d.execute("Bash", {"command": "echo hi"})  # blocked
        self.assertIsInstance(r, SandboxedResult)

    def test_result_is_frozen(self):
        """T16c — Mutation: remove the ``frozen=True`` from the
        dataclass → a tool_loop bug could mutate a result after
        it lands. Survives by attempting mutation."""
        d = SandboxedDispatcher(self.root)
        r = d.execute("Bash", {"command": "x"})
        with self.assertRaises(Exception):  # FrozenInstanceError
            r.output = "mutated"


# --- Missing-arg surfacing (Read/Grep/Glob each need their arg) -----------


class TestMissingArgs(_TmpSandbox):

    def test_read_missing_path(self):
        """Read with no path → error (not block, not crash). The model
        sees the error message and can retry with a valid arg."""
        d = SandboxedDispatcher(self.root)
        r = d.execute("Read", {})
        self.assertFalse(r.blocked)
        self.assertIn("file_path", r.error)

    def test_grep_missing_pattern(self):
        d = SandboxedDispatcher(self.root)
        r = d.execute("Grep", {})
        self.assertIn("pattern", r.error)

    def test_glob_missing_pattern(self):
        d = SandboxedDispatcher(self.root)
        r = d.execute("Glob", {})
        self.assertIn("pattern", r.error)


# --- make_sandboxed_dispatcher factory ------------------------------------


class TestFactory(_TmpSandbox):

    def test_factory_returns_dispatcher(self):
        """Factory mirrors the type the loop expects."""
        d = make_sandboxed_dispatcher(self.root)
        self.assertIsInstance(d, SandboxedDispatcher)

    def test_factory_validates_root(self):
        """Factory propagates :class:`SandboxRootRefused` so a server
        route building a dispatcher per-request can rely on the
        factory to fail-fast on a bad root."""
        with self.assertRaises(SandboxRootRefused):
            make_sandboxed_dispatcher("")


class TestBinaryFileSurfacesAsError(_TmpSandbox):
    """T17-T19 — R0 forward-anchor tests for AGORA #373 concern 4:
    SandboxBinaryError from the file_sandbox primitive must reach the
    model as a recoverable ``error`` tool_result, NOT as a blocked
    capability-violation (the model needs to retry, not give up)."""

    def test_binary_file_read_returns_error_not_blocked(self):
        """T17 — Mutation: drop the SandboxBinaryError except branch
        → the dispatcher raises uncaught. Survives by asserting the
        binary file produces ``error`` not ``blocked``."""
        p = Path(self.root) / "evil.bin"
        p.write_bytes(b"\x00\x01\x02" + b"abc")
        d = SandboxedDispatcher(self.root)
        r = d.execute("Read", {"file_path": "evil.bin"})
        self.assertFalse(r.blocked)  # NOT capability lock
        self.assertNotEqual(r.error, "")  # IS recoverable error
        self.assertIn("binary", r.error.lower())

    def test_binary_file_does_not_consume_capability_lock(self):
        """T18 — Mutation: re-route SandboxBinaryError to
        ``blocked=True`` → the model treats binary files as a security
        block and gives up. Survives by asserting ``output`` is empty
        AND ``blocked`` is False AND no ``reason`` set."""
        p = Path(self.root) / "evil.bin"
        p.write_bytes(b"\x00binary content")
        d = SandboxedDispatcher(self.root)
        r = d.execute("Read", {"file_path": "evil.bin"})
        self.assertEqual(r.output, "")
        self.assertFalse(r.blocked)
        self.assertEqual(r.reason, "")


class TestTimeoutSurfacesAsError(_TmpSandbox):
    """T20-T21 — R0 forward-anchor tests for AGORA #373 concern 1:
    SandboxTimeoutError from grep/glob must reach the model as a
    recoverable ``error`` (so it can narrow the query), NOT as
    ``blocked`` and NOT as a silent empty result."""

    def test_grep_timeout_returns_error_not_blocked(self):
        """T20 — Mutation: drop the SandboxTimeoutError except branch
        in _exec_grep → dispatcher raises uncaught OR returns empty
        ``(no matches)`` (the original silent-empty bug). Survives by
        asserting the dispatcher returns ``error`` with the timeout
        message."""
        self._write("a.txt", "needle\n")
        d = SandboxedDispatcher(self.root)
        # Patch file_sandbox.grep_pattern to raise timeout
        import unittest.mock as mock
        with mock.patch(
            "lib.hal.file_sandbox.grep_pattern",
            side_effect=__import__(
                "lib.hal.file_sandbox", fromlist=["SandboxTimeoutError"],
            ).SandboxTimeoutError("grep timed out after 30s — narrow"),
        ):
            r = d.execute("Grep", {"pattern": "needle"})
        self.assertFalse(r.blocked)
        self.assertIn("timed out", r.error.lower())
        # CRITICALLY NOT "(no matches)" — that's the bug we're killing
        self.assertNotEqual(r.output, "(no matches)")

    def test_glob_timeout_returns_error_not_blocked(self):
        """T21 — Mutation: drop the SandboxTimeoutError except branch
        in _exec_glob → glob hangs or silently truncates. Survives by
        asserting timeout → error (not blocked, not output)."""
        d = SandboxedDispatcher(self.root)
        import unittest.mock as mock
        with mock.patch(
            "lib.hal.file_sandbox.glob_pattern",
            side_effect=__import__(
                "lib.hal.file_sandbox", fromlist=["SandboxTimeoutError"],
            ).SandboxTimeoutError("glob timed out after 60s — narrow"),
        ):
            r = d.execute("Glob", {"pattern": "**/*.py"})
        self.assertFalse(r.blocked)
        self.assertIn("timed out", r.error.lower())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
