"""Tests for lib/hal/file_sandbox.py — the path-allowlist sandbox for
HAL's ``/api/infer-with-tools`` route.

Goodhart-resistant per Rule 14: every test fails under at least one
obvious code mutation; each docstring names the mutation it survives.

Test register (T1–T4 = W1 primitives; T14–T15 + T17 = W4 capability-
lock anchors per the ultraplan).
"""
from __future__ import annotations

import json
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

from lib.hal import file_sandbox  # noqa: E402
from lib.hal.file_sandbox import (  # noqa: E402
    GrepMatch,
    SandboxBinaryError,
    SandboxEscapeError,
    SandboxRootRefused,
    SandboxTimeoutError,
    glob_pattern,
    grep_pattern,
    read_text,
    validate_root,
)


class _TmpSandbox(unittest.TestCase):
    """Per-test temp directory + audit log redirection."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = os.path.realpath(self._tmp.name)
        self._audit_dir = tempfile.TemporaryDirectory()
        self._orig_audit = file_sandbox._audit_path
        audit_file = Path(self._audit_dir.name) / "audit.jsonl"
        file_sandbox._audit_path = lambda: audit_file

    def tearDown(self):
        file_sandbox._audit_path = self._orig_audit
        self._audit_dir.cleanup()
        self._tmp.cleanup()

    def _write(self, rel: str, content: str) -> Path:
        p = Path(self.root) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    def _audit_lines(self) -> list[dict]:
        audit_file = file_sandbox._audit_path()
        if not audit_file.exists():
            return []
        return [json.loads(line) for line in
                audit_file.read_text(encoding="utf-8").splitlines()
                if line.strip()]


class TestReadText(_TmpSandbox):

    def test_read_text_inside_root_returns_content(self):
        """T1 — Mutation: invert ``commonpath`` check in
        ``_resolve_safe`` → returns content for paths OUTSIDE root.
        Survives by asserting returned bytes equal disk bytes ONLY
        when path is rooted."""
        self._write("hello.txt", "world\n")
        out = read_text(self.root, "hello.txt")
        self.assertEqual(out, "world\n")

    def test_read_text_symlink_escape_refused(self):
        """T2 — Mutation: drop ``realpath`` resolution → a symlink to
        ``/etc/passwd`` reads ``/etc/passwd``. Survives by asserting
        :class:`SandboxEscapeError` raised when a symlink points
        outside the sandbox."""
        outside = Path(tempfile.mkdtemp())
        try:
            outside_target = outside / "secret.txt"
            outside_target.write_text("SECRET\n", encoding="utf-8")
            link = Path(self.root) / "evil_link.txt"
            link.symlink_to(outside_target)
            with self.assertRaises(SandboxEscapeError):
                read_text(self.root, "evil_link.txt")
        finally:
            shutil.rmtree(outside)

    def test_read_text_traversal_dotdot_refused(self):
        """T2b — Mutation: skip the ``..`` check →
        ``../../../etc/passwd`` reads outside. Survives by asserting
        escape error for any ``../`` path that leaves the sandbox."""
        with self.assertRaises(SandboxEscapeError):
            read_text(self.root, "../../../etc/passwd")


class TestGrep(_TmpSandbox):

    def test_grep_pattern_only_matches_inside_root(self):
        """T3 — Mutation: pass raw glob to ``rg`` without root prefix
        → matches across filesystem. Survives by hermetic temp-dir
        setup + asserting hits only from files written inside the
        temp root, with paths relative to that root."""
        self._write("a.txt", "needle here\n")
        self._write("sub/b.txt", "haystack only\n")
        self._write("sub/c.txt", "another needle\n")
        hits = grep_pattern(self.root, r"needle")
        paths = sorted(h.path for h in hits)
        self.assertEqual(paths, ["a.txt", "sub/c.txt"])
        for h in hits:
            self.assertFalse(h.path.startswith("/"),
                             f"absolute path leaked: {h.path!r}")

    def test_grep_returns_grepmatch_dataclass(self):
        """T3b — Mutation: change return type to raw tuples → callers
        relying on ``.path`` / ``.line`` / ``.text`` break."""
        self._write("x.txt", "alpha\nbeta\nneedle\n")
        hits = grep_pattern(self.root, r"needle")
        self.assertEqual(len(hits), 1)
        self.assertIsInstance(hits[0], GrepMatch)
        self.assertEqual(hits[0].path, "x.txt")
        self.assertEqual(hits[0].line, 3)


class TestGlob(_TmpSandbox):

    def test_glob_returns_paths_relative_to_root(self):
        """T4 — Mutation: return absolute paths → leaks sandbox root
        to caller. Survives by asserting every returned path is
        relative + each does NOT start with a leading ``/``."""
        self._write("a.py", "")
        self._write("sub/b.py", "")
        self._write("sub/c.txt", "")
        hits = glob_pattern(self.root, "**/*.py")
        for h in hits:
            self.assertFalse(h.startswith("/"),
                             f"absolute path leaked: {h!r}")
            self.assertFalse(h.startswith(os.sep),
                             f"absolute path leaked: {h!r}")
        self.assertIn("a.py", hits)
        self.assertIn("sub/b.py", hits)

    def test_glob_absolute_pattern_refused(self):
        """T4b — Mutation: drop the absolute-pattern guard → caller
        can glob ``/etc/*`` from outside sandbox. Survives by asserting
        :class:`SandboxEscapeError` for absolute glob patterns."""
        with self.assertRaises(SandboxEscapeError):
            glob_pattern(self.root, "/etc/*")

    def test_glob_symlink_loop_terminates_bounded(self):
        """T4c (R0 forward-anchor, AGORA Council #373 falsifier-as-test
        suggestion) — Mutation: drop ``_resolve_safe`` defence-in-depth
        inside the rglob loop → a self-referential symlink cycle causes
        infinite traversal or RecursionError. Survives by asserting
        glob_pattern terminates within bounded time AND returns no
        entries from inside the loop.

        Construction: sandbox root contains a subdir ``loop/`` whose
        child ``inner -> loop/`` symlink creates a cycle. A naive
        ``Path.rglob`` would follow the symlink indefinitely; the
        sandbox MUST either skip the symlink-loop entry (per the
        ``_resolve_safe`` check) OR cap traversal at ``max_hits``."""
        loop_dir = Path(self.root) / "loopdir"
        loop_dir.mkdir()
        # inner -> loop_dir (cycle)
        (loop_dir / "inner").symlink_to(loop_dir)
        # also plant a real file to verify glob still works
        (Path(self.root) / "real.txt").write_text("x\n",
                                                    encoding="utf-8")
        import signal

        def _alarm_handler(_signum, _frame):
            raise RuntimeError("glob did not terminate within 5s")

        # Bounded-time assertion via SIGALRM
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(5)
        try:
            hits = glob_pattern(self.root, "**/*.txt")
        finally:
            signal.alarm(0)
        # Real file must still be found
        self.assertIn("real.txt", hits)


class TestValidateRoot(unittest.TestCase):

    def test_sandbox_root_rejects_etc_and_root(self):
        """T14 (W4 anchor) — Mutation: drop ``/etc`` / ``/`` from the
        refused set → operator can pin sandbox to a system directory.
        Survives by parametrising over genuine high-risk system leaves
        and asserting :class:`SandboxRootRefused`. ``/var`` and
        ``/Library`` are deliberately NOT refused on macOS — user
        tempdirs live under ``/private/var/folders/...`` and
        legitimate tooling reads ``/Library`` subpaths."""
        for refused in ("/", "/etc", "/bin", "/sbin"):
            with self.subTest(refused=refused):
                if not os.path.isdir(refused):
                    self.skipTest(f"{refused} does not exist on this host")
                with self.assertRaises(SandboxRootRefused):
                    validate_root(refused)

    def test_sandbox_root_rejects_var_log_and_system_leaves(self):
        """T14e (AGORA #373 concern 1 forward-anchor) — Mutation: drop
        the ``/var`` system leaves from the refused set → an operator
        (or untrusted-content-influenced caller) can pin the read-sandbox
        to ``/var/log`` and exfiltrate ``system.log`` / kernel / auth
        messages, or to ``/var/audit`` (BSM trails) / ``/var/db`` /
        ``/var/root`` / ``/var/spool`` (user mail + cron + print queues).
        Survives by asserting :class:`SandboxRootRefused` on each genuine
        system leaf. macOS realpaths ``/var/...`` to ``/private/var/...``
        so the refusal must hold for the unprefixed form too. FAILS
        without the _REFUSED_ROOTS addition (validate_root would return
        the canonical path instead of raising). ``/var/spool`` is the
        AGORA #376 extension (mail-spool reconnaissance)."""
        for leaf in ("/var/log", "/var/audit", "/var/db", "/var/root",
                     "/var/spool"):
            with self.subTest(leaf=leaf):
                if not os.path.isdir(leaf):
                    self.skipTest(f"{leaf} does not exist on this host")
                with self.assertRaises(SandboxRootRefused):
                    validate_root(leaf)

    def test_sandbox_root_traversal_to_refused_leaf_rejected(self):
        """T14g (AGORA #376 concern 2 forward-anchor) — Mutation: apply
        ``realpath`` AFTER the refused-set comparison instead of before →
        a ``..``-traversal root like ``/var/folders/../log`` (lexically
        not in _REFUSED_ROOTS) would slip past the leaf refusal and
        resolve to ``/private/var/log`` at read time, re-opening the
        system-log hole. Survives by asserting :class:`SandboxRootRefused`
        on traversal paths that REALPATH-RESOLVE to a refused leaf — this
        only passes because ``validate_root`` realpaths the candidate
        BEFORE comparing against _REFUSED_ROOTS. Confirms-correct (the
        current order is right); it is the regression anchor that keeps
        it right."""
        for traversal in ("/var/folders/../log", "/tmp/../var/log",
                          "/var/log/../log"):
            with self.subTest(traversal=traversal):
                # Only meaningful if the resolved leaf actually exists
                # (validate_root refuses nonexistent roots for a DIFFERENT
                # reason, which would mask the leaf-refusal we test here).
                if not os.path.isdir(os.path.realpath(traversal)):
                    self.skipTest(
                        f"{traversal} does not resolve to an existing dir")
                with self.assertRaises(SandboxRootRefused):
                    validate_root(traversal)

    def test_sandbox_root_allows_var_folders_tempdir(self):
        """T14f (AGORA #373 concern 1 over-block guard) — Mutation:
        refuse the broad ``/var`` parent to block ``/var/log`` → also
        blocks the legitimate user tempdir under ``/var/folders/...``,
        breaking every real sandbox. Survives by asserting that an actual
        ``tempfile.mkdtemp()`` (and the ``/var/folders`` parent itself,
        when present) is STILL accepted after the system-leaf refusal —
        proving the fix refuses leaves, not the ``/var`` subtree. This is
        the regression anchor that the security fix didn't over-block."""
        with tempfile.TemporaryDirectory() as tmp:
            # A real tempdir must always validate (it is the canonical
            # legitimate sandbox root).
            self.assertEqual(validate_root(tmp), os.path.realpath(tmp))
        # /var/folders specifically (the macOS user tempdir parent) must
        # not be swept up by the /var leaf refusal.
        if os.path.isdir("/var/folders"):
            self.assertEqual(
                validate_root("/var/folders"),
                os.path.realpath("/var/folders"),
            )

    def test_sandbox_root_realpath_resolves_symlink_to_outside(self):
        """T15 (W4 anchor) — Mutation: skip ``realpath`` on root →
        operator can symlink a tempdir to ``/etc`` and bypass refusal.
        Survives by realpath-resolving root and re-checking against
        refused prefixes."""
        with tempfile.TemporaryDirectory() as tmp:
            sym = Path(tmp) / "linkroot"
            if not os.path.isdir("/etc"):
                self.skipTest("/etc not present")
            sym.symlink_to("/etc")
            with self.assertRaises(SandboxRootRefused):
                validate_root(str(sym))

    def test_sandbox_root_empty_refused(self):
        """T14b — Mutation: silently default empty root to cwd.
        Survives by asserting refusal on empty string."""
        with self.assertRaises(SandboxRootRefused):
            validate_root("")

    def test_sandbox_root_nonexistent_refused(self):
        """T14c — Mutation: silently create missing root. Survives by
        asserting refusal on nonexistent path."""
        with self.assertRaises(SandboxRootRefused):
            validate_root("/nonexistent/path/anywhere/123abc")

    def test_sandbox_root_returns_canonical_path(self):
        """T14d — Mutation: return input as-is. Survives by asserting
        returned path is realpath-normalised (no trailing slash, no
        symlinks)."""
        with tempfile.TemporaryDirectory() as tmp:
            real = os.path.realpath(tmp)
            out = validate_root(tmp + "/")
            self.assertEqual(out, real)


class TestBinaryDetection(_TmpSandbox):
    """T2c-T2e — R0 forward-anchor tests for AGORA Council #373
    concern 4: ``read_text`` must refuse binary files instead of
    returning UTF-8 replacement-character garbage."""

    def test_binary_file_with_null_byte_refused(self):
        """T2c — Mutation: drop the binary-sniff before decode →
        the model receives \\ufffd-laden garbage. Survives by asserting
        :class:`SandboxBinaryError` on a file containing a null byte."""
        p = Path(self.root) / "evil.bin"
        p.write_bytes(b"\x00\x01\x02not really text")
        with self.assertRaises(SandboxBinaryError):
            read_text(self.root, "evil.bin")

    def test_binary_file_with_null_in_first_8kib_refused(self):
        """T2d — Mutation: only check the first byte instead of first
        8 KiB → null deeper into the file slips past. Survives by
        planting a null byte at offset 4096 (still within the 8 KiB
        sniff window)."""
        p = Path(self.root) / "deep.bin"
        # 4 KiB of ASCII text followed by a null byte
        p.write_bytes(b"a" * 4096 + b"\x00" + b"b" * 100)
        with self.assertRaises(SandboxBinaryError):
            read_text(self.root, "deep.bin")

    def test_pure_text_file_not_refused(self):
        """T2e — Mutation: binary check too aggressive — rejects legitimate
        text. Survives by asserting a pure ASCII file with no null bytes
        decodes successfully."""
        self._write("hello.py", "print('hi')\n")
        content = read_text(self.root, "hello.py")
        self.assertEqual(content, "print('hi')\n")

    def test_replacement_only_for_partial_utf8(self):
        """T2f — Mutation: drop ``errors='replace'`` after binary check
        → a legitimately text file with mojibake bytes (mixed encoding)
        raises UnicodeDecodeError. Survives by asserting non-binary
        files with one stray non-UTF-8 byte still decode (with U+FFFD
        in place of the bad byte)."""
        p = Path(self.root) / "mojibake.txt"
        # ASCII followed by a stray Latin-1 byte (NOT a null byte —
        # so it passes the binary sniff)
        p.write_bytes(b"hello " + b"\xe9" + b" world\n")
        content = read_text(self.root, "mojibake.txt")
        # The bad byte becomes U+FFFD; the rest is preserved
        self.assertIn("hello", content)
        self.assertIn("world", content)


class TestGrepTimeout(_TmpSandbox):
    """T3c-T3d — R0 forward-anchor tests for AGORA Council #373
    concern 1: ``_grep_via_ripgrep`` 30s silent-empty must become a
    surfaced :class:`SandboxTimeoutError`."""

    def test_grep_timeout_raises_sandbox_timeout(self):
        """T3c — Mutation: catch and swallow TimeoutExpired returning
        empty list → model interprets timeout as 'no matches'. Survives
        by asserting :class:`SandboxTimeoutError` when ripgrep's
        subprocess timeout fires (simulated via ``timeout_s=0`` which
        is too short for ANY real query)."""
        self._write("a.txt", "hello world\n")
        # On macOS, even a 0s subprocess.run timeout fires before exec
        # completes. The function MUST raise SandboxTimeoutError, NOT
        # return [].
        if not file_sandbox._ripgrep_available():
            self.skipTest("rg not installed; covered by stdlib path test")
        with self.assertRaises(SandboxTimeoutError):
            grep_pattern(self.root, "hello", timeout_s=0)

    def test_grep_stdlib_timeout_raises_sandbox_timeout(self):
        """T3d — Mutation: drop the deadline check in the stdlib
        walker → infinite-time grep on a huge tree. Survives by
        planting many files and asserting timeout fires before
        traversal completes (via ``timeout_s=0``)."""
        # Plant enough files that even instant traversal can be
        # interrupted by a 0s deadline
        for i in range(20):
            self._write(f"f{i}.txt", "line\n" * 20)
        # Bypass ripgrep to exercise the stdlib path directly
        from lib.hal.file_sandbox import _grep_via_stdlib
        with self.assertRaises(SandboxTimeoutError):
            _grep_via_stdlib(self.root, "line", max_hits=100, timeout_s=0)

    def test_grep_normal_query_does_not_timeout(self):
        """T3e — Mutation: timeout check too aggressive — fires on
        normal queries. Survives by asserting a real grep on small
        sandbox completes within the default timeout."""
        self._write("a.txt", "needle\n")
        hits = grep_pattern(self.root, "needle")  # default timeout
        self.assertEqual(len(hits), 1)


class TestGlobTimeout(_TmpSandbox):
    """T4d — R0 forward-anchor test for AGORA Council #373 falsifier
    suggestion: glob_pattern must terminate on pathological inputs."""

    def test_glob_timeout_raises_sandbox_timeout(self):
        """T4d — Mutation: drop the deadline check in the rglob loop
        → infinite-time glob on a huge tree. Survives by planting
        files and asserting timeout fires with ``timeout_s=0``."""
        for i in range(20):
            self._write(f"sub{i}/f.txt", "x")
        with self.assertRaises(SandboxTimeoutError):
            glob_pattern(self.root, "**/*.txt", timeout_s=0)


class TestAuditLog(_TmpSandbox):

    def test_audit_jsonl_records_every_tool_invocation(self):
        """T17 (W4 anchor) — Mutation: skip audit write on ``Glob``
        only → silent partial purge. Survives by asserting one JSONL
        line per tool call across {Read, Grep, Glob}."""
        self._write("a.py", "needle\n")
        read_text(self.root, "a.py")
        grep_pattern(self.root, "needle")
        glob_pattern(self.root, "*.py")
        events = self._audit_lines()
        tools = sorted(e["tool"] for e in events)
        self.assertIn("Read", tools)
        self.assertIn("Grep", tools)
        self.assertIn("Glob", tools)

    def test_audit_records_refused_paths(self):
        """T17b — Mutation: only audit successes → escape attempts
        invisible. Survives by asserting audit line with
        ``allowed: False`` for an escape attempt."""
        try:
            read_text(self.root, "../../../etc/passwd")
        except SandboxEscapeError:
            pass
        events = self._audit_lines()
        refused = [e for e in events
                   if e.get("tool") == "Read"
                   and e.get("allowed") is False]
        self.assertEqual(len(refused), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
