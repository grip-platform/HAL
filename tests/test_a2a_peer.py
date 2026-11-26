"""Tests for HAL A2A peer-exec (capability-locked harness delegation).

Goodhart-proof: the capability lock is asserted per harness against a HARDCODED
expected argv AND against the forbidden-token set, so a mutation that drops the
read-only flag or adds a write/yolo mode fails here — the security property is
proven, not merely present.
"""
import subprocess
import unittest

from lib.hal.a2a import A2ABus
from lib.hal.a2a_peer import (
    ANTIGRAVITY,
    CLAUDE_CODE,
    CODEX,
    PeerReply,
    UnknownPeerError,
    ask_peer,
    ask_peer_via_bus,
    build_peer_argv,
    canonical_harness,
    list_peers,
    _FORBIDDEN_TOKENS,
)


class TestCapabilityLockedArgv(unittest.TestCase):
    """The lock lives in the argv — hardcoded expectations, mutation-surviving."""

    def test_claude_argv_exact(self):
        self.assertEqual(
            build_peer_argv(CLAUDE_CODE, "hello"),
            ["claude", "-p", "hello", "--tools", "Read,Grep,Glob"],
        )

    def test_codex_argv_exact(self):
        self.assertEqual(
            build_peer_argv(CODEX, "hello"),
            ["codex", "exec", "--sandbox", "read-only", "hello"],
        )

    def test_gemini_argv_exact(self):
        self.assertEqual(
            build_peer_argv(ANTIGRAVITY, "hello"),
            ["gemini", "-p", "hello", "--approval-mode", "plan"],
        )

    def test_every_peer_carries_its_lock(self):
        # claude: read-only tool allowlist; codex: read-only sandbox; gemini: plan mode
        self.assertIn("Read,Grep,Glob", build_peer_argv(CLAUDE_CODE, "x"))
        self.assertIn("read-only", build_peer_argv(CODEX, "x"))
        self.assertIn("plan", build_peer_argv(ANTIGRAVITY, "x"))

    def test_no_peer_argv_contains_a_forbidden_token(self):
        for harness in list_peers():
            argv = build_peer_argv(harness, "do something")
            leaked = set(argv) & _FORBIDDEN_TOKENS
            self.assertEqual(leaked, set(), f"{harness} argv widened surface: {leaked}")


class TestFailClosed(unittest.TestCase):

    def test_unknown_harness_raises(self):
        with self.assertRaises(UnknownPeerError):
            build_peer_argv("evil-harness", "rm -rf /")

    def test_unknown_harness_via_ask_peer_raises_before_spawn(self):
        spawned = []
        with self.assertRaises(UnknownPeerError):
            ask_peer("nope", "x", runner=lambda a, t: spawned.append(a))
        self.assertEqual(spawned, [], "must fail closed BEFORE any spawn")

    def test_aliases_resolve(self):
        self.assertEqual(canonical_harness("gemini"), ANTIGRAVITY)
        self.assertEqual(canonical_harness("claude"), CLAUDE_CODE)
        self.assertEqual(canonical_harness("CC"), CLAUDE_CODE)


class TestNoShellInjection(unittest.TestCase):

    def test_prompt_is_a_single_argv_element(self):
        evil = '"; rm -rf / #'
        argv = build_peer_argv(CODEX, evil)
        self.assertIn(evil, argv)  # present verbatim as ONE element
        self.assertEqual(argv.count(evil), 1)
        # the metachars never get their own tokens
        self.assertNotIn("rm", argv)


class TestAskPeerDIP(unittest.TestCase):
    """Runner is injected — zero real CLI calls."""

    def _ok_runner(self, output="PEER-OK", code=0):
        captured = {}

        def runner(argv, timeout):
            captured["argv"] = argv
            captured["timeout"] = timeout
            return subprocess.CompletedProcess(argv, code, stdout=output, stderr="")

        return runner, captured

    def test_ask_peer_returns_reply_and_uses_locked_argv(self):
        runner, captured = self._ok_runner()
        reply = ask_peer(CLAUDE_CODE, "summarise X", runner=runner)
        self.assertIsInstance(reply, PeerReply)
        self.assertTrue(reply.ok)
        self.assertEqual(reply.output, "PEER-OK")
        self.assertEqual(reply.harness, CLAUDE_CODE)
        # the runner was handed the capability-locked argv, not a bare one
        self.assertEqual(captured["argv"], ["claude", "-p", "summarise X", "--tools", "Read,Grep,Glob"])

    def test_ask_peer_nonzero_exit_is_not_ok(self):
        runner, _ = self._ok_runner(output="boom", code=2)
        reply = ask_peer(CODEX, "x", runner=runner)
        self.assertFalse(reply.ok)
        self.assertEqual(reply.exit_code, 2)

    def test_ask_peer_timeout_fails_gracefully(self):
        def slow(argv, timeout):
            raise subprocess.TimeoutExpired(argv, timeout)

        reply = ask_peer(ANTIGRAVITY, "x", runner=slow, timeout=5.0)
        self.assertFalse(reply.ok)
        self.assertEqual(reply.exit_code, 124)
        self.assertIn("timed out", reply.output)

    def test_arbitrary_runner_exception_propagates(self):
        # Only TimeoutExpired is handled; any other error MUST propagate, not be
        # swallowed into a misleading PeerReply.
        def boom(argv, timeout):
            raise ValueError("unexpected")

        with self.assertRaises(ValueError):
            ask_peer(CLAUDE_CODE, "x", runner=boom)

    def test_failure_surfaces_stderr_not_empty_stdout(self):
        # Provenance: a non-zero exit with stderr must surface stderr, never be
        # mistaken for an empty-stdout success.
        def runner(argv, timeout):
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="real error")

        reply = ask_peer(CODEX, "x", runner=runner)
        self.assertFalse(reply.ok)
        self.assertEqual(reply.output, "real error")


class TestAskPeerViaBus(unittest.TestCase):

    def test_reply_routed_back_to_caller_queue(self):
        bus = A2ABus()

        def runner(argv, timeout):
            return subprocess.CompletedProcess(argv, 0, stdout="from gemini", stderr="")

        ask_peer_via_bus(bus, from_agent="claude_code", harness="gemini", prompt="hi", runner=runner)
        msgs = bus.receive("claude_code")
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].content, "from gemini")
        self.assertEqual(msgs[0].from_agent, ANTIGRAVITY)


if __name__ == "__main__":
    unittest.main()
