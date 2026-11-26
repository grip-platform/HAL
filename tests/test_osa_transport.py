"""Tests for ``lib.hal.osa.transport`` — local + Tailscale SSH transports.

Mocks :mod:`subprocess` throughout. The SSH path never actually reaches
another host during CI. End-to-end testing against a live tailnet peer
is operator-driven and out of scope.
"""
from __future__ import annotations

import os
import subprocess
import unittest
from unittest.mock import patch

from lib.hal.osa.osa import OsaError, OsaResult, OsaTimeout
from lib.hal.osa.transport import (
    REMOTE_DISPATCH_ENV,
    LocalTransport,
    OpenSSHTransport,
    SSHTransport,
    Transport,
)


def _fake_completed(stdout: str = "", returncode: int = 0, stderr: str = ""):
    """Return a minimal CompletedProcess-shaped stub for subprocess.run mocks."""
    class _Stub:
        pass
    stub = _Stub()
    stub.stdout = stdout
    stub.stderr = stderr
    stub.returncode = returncode
    return stub


class TransportInterface(unittest.TestCase):
    """``Transport`` is the shared abstract surface."""

    def test_local_is_not_remote(self) -> None:
        self.assertFalse(LocalTransport().is_remote)

    def test_local_host_string(self) -> None:
        self.assertEqual(LocalTransport().host, "local")

    def test_ssh_is_remote(self) -> None:
        self.assertTrue(SSHTransport(host_name="toopie").is_remote)

    def test_ssh_host_reports_host_name(self) -> None:
        self.assertEqual(SSHTransport(host_name="toopie").host, "toopie")

    def test_both_implement_Transport(self) -> None:
        self.assertIsInstance(LocalTransport(), Transport)
        self.assertIsInstance(SSHTransport(host_name="toopie"), Transport)


class SSHCommandShape(unittest.TestCase):
    """``SSHTransport._build_cmd`` produces the council-mandated flags."""

    def test_uses_batchmode_and_connect_timeout(self) -> None:
        t = SSHTransport(host_name="toopie", connect_timeout_s=7)
        cmd = t._build_cmd("AppleScript")
        self.assertIn("BatchMode=yes", cmd)
        self.assertIn("ConnectTimeout=7", cmd)

    def test_control_persist_capped_to_10min(self) -> None:
        """Goodhart guard — operator cannot set an unbounded mux lifetime."""
        t = SSHTransport(host_name="toopie", control_persist_s=99999)
        cmd = t._build_cmd("AppleScript")
        joined = " ".join(cmd)
        self.assertIn("ControlPersist=600s", joined)

    def test_control_persist_floor_10s(self) -> None:
        t = SSHTransport(host_name="toopie", control_persist_s=1)
        cmd = t._build_cmd("AppleScript")
        joined = " ".join(cmd)
        self.assertIn("ControlPersist=10s", joined)

    def test_ip_overrides_host_name_for_target(self) -> None:
        t = SSHTransport(host_name="toopie", ip="100.64.0.10")
        cmd = t._build_cmd("AppleScript")
        self.assertIn("100.64.0.10", cmd)
        self.assertNotIn("toopie", cmd)

    def test_language_reaches_osascript_flag(self) -> None:
        cmd = SSHTransport(host_name="toopie")._build_cmd("JavaScript")
        self.assertEqual(cmd[-4:], ["osascript", "-l", "JavaScript", "-"])

    def test_stdin_dash_last(self) -> None:
        cmd = SSHTransport(host_name="toopie")._build_cmd("AppleScript")
        self.assertEqual(cmd[-1], "-")


class LocalTransportExecution(unittest.TestCase):
    """``LocalTransport`` delegates to the existing ``osa_raw`` kernel."""

    @patch("lib.hal.osa.osa.subprocess.run")
    @patch("lib.hal.osa.osa._probe_platform")
    def test_execute_returns_osa_result(self, _mock_probe, mock_run) -> None:
        mock_run.return_value = _fake_completed(stdout="hi", returncode=0)
        result = LocalTransport().execute(
            "return \"hi\"", language="AppleScript", timeout_s=5
        )
        self.assertIsInstance(result, OsaResult)
        self.assertTrue(result.ok)
        self.assertEqual(result.stdout, "hi")


class SSHTransportExecution(unittest.TestCase):
    """``SSHTransport.execute`` pipes body over stdin and returns an OsaResult."""

    @patch("lib.hal.osa.transport.shutil.which", return_value="/usr/bin/ssh")
    @patch("lib.hal.osa.transport.subprocess.run")
    def test_body_piped_via_stdin(self, mock_run, _which) -> None:
        mock_run.return_value = _fake_completed(stdout="42\n", returncode=0)
        result = SSHTransport(host_name="toopie").execute(
            'return 42', language="AppleScript", timeout_s=5,
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.stdout, "42\n")
        _, kwargs = mock_run.call_args
        self.assertEqual(kwargs["input"], "return 42")
        self.assertTrue(kwargs["text"])

    @patch("lib.hal.osa.transport.shutil.which", return_value="/usr/bin/ssh")
    @patch("lib.hal.osa.transport.subprocess.run")
    def test_nonzero_exit_returned_not_raised(self, mock_run, _which) -> None:
        """Goodhart guard — Transport.execute uniformly returns OsaResult.
        Raise-vs-return is dispatch.py's decision."""
        mock_run.return_value = _fake_completed(
            stdout="", returncode=1, stderr="boom",
        )
        result = SSHTransport(host_name="toopie").execute(
            "bad", language="AppleScript", timeout_s=5,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stderr, "boom")

    @patch("lib.hal.osa.transport.shutil.which", return_value=None)
    def test_missing_ssh_binary_raises_osa_error(self, _which) -> None:
        with self.assertRaises(OsaError):
            SSHTransport(host_name="toopie").execute(
                "x", language="AppleScript", timeout_s=5,
            )

    @patch("lib.hal.osa.transport.shutil.which", return_value="/usr/bin/ssh")
    @patch("lib.hal.osa.transport.subprocess.run")
    def test_timeout_raises_osa_timeout(self, mock_run, _which) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ssh"], timeout=5)
        with self.assertRaises(OsaTimeout):
            SSHTransport(host_name="toopie").execute(
                "x", language="AppleScript", timeout_s=5,
            )


class RemoteDispatchConsent(unittest.TestCase):
    """Remote dispatch requires the operator consent env var."""

    def setUp(self) -> None:
        self._orig = os.environ.pop(REMOTE_DISPATCH_ENV, None)
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        if self._orig is not None:
            os.environ[REMOTE_DISPATCH_ENV] = self._orig

    def test_remote_env_var_name_is_documented(self) -> None:
        self.assertEqual(REMOTE_DISPATCH_ENV, "HAL_OSA_REMOTE_DISPATCH")


# ---------------------------------------------------------------------------
# γ-1b: OpenSSHTransport
# ---------------------------------------------------------------------------


class OpenSSHCommandShape(unittest.TestCase):
    """``OpenSSHTransport._build_cmd`` uses explicit IdentityFile + User.

    Each assertion is a load-bearing invariant of the hardened variant:
    swap any assertion for ``assertTrue(True)`` and the security story
    (ambient-key refusal, explicit identity, IdentitiesOnly) silently
    regresses. Goodhart guard: test values are specific, not tautologies.
    """

    def _make(self, **over) -> OpenSSHTransport:
        base = dict(
            host_name="toopie",
            identity_file="/tmp/fake_id_ed25519",
            user="bob",
        )
        base.update(over)
        return OpenSSHTransport(**base)

    def test_argv_carries_identity_file_flag(self) -> None:
        cmd = self._make()._build_cmd("AppleScript")
        self.assertIn("-i", cmd)
        self.assertIn("/tmp/fake_id_ed25519", cmd)
        # Flag must IMMEDIATELY precede the value (no stray flags between)
        idx = cmd.index("-i")
        self.assertEqual(cmd[idx + 1], "/tmp/fake_id_ed25519")

    def test_argv_carries_login_name_flag(self) -> None:
        cmd = self._make(user="Andre Theart")._build_cmd("AppleScript")
        self.assertIn("-l", cmd)
        idx = cmd.index("-l")
        # Usernames with spaces pass through as a single argv element —
        # OpenSSH handles them, no shell quoting required at this layer.
        self.assertEqual(cmd[idx + 1], "Andre Theart")

    def test_argv_forces_identities_only(self) -> None:
        """Ambient agent keys must NOT be tried — only the explicit one."""
        cmd = self._make()._build_cmd("AppleScript")
        joined = " ".join(cmd)
        self.assertIn("IdentitiesOnly=yes", joined)

    def test_control_persist_cap_inherited(self) -> None:
        """The 600s mux cap from the parent must remain in force."""
        cmd = self._make(control_persist_s=99999)._build_cmd("AppleScript")
        self.assertIn("ControlPersist=600s", " ".join(cmd))

    def test_ip_overrides_host_name_for_target(self) -> None:
        cmd = self._make(ip="100.64.0.10")._build_cmd("AppleScript")
        self.assertIn("100.64.0.10", cmd)
        self.assertNotIn("toopie", cmd)

    def test_audit_host_is_host_name_not_ip(self) -> None:
        """Operators grep by logical name — ip must not leak into audit."""
        t = self._make(host_name="lab-mac-3", ip="192.0.2.99")
        self.assertEqual(t.host, "lab-mac-3")

    def test_osascript_stdin_dash_still_last(self) -> None:
        cmd = self._make()._build_cmd("JavaScript")
        self.assertEqual(cmd[-4:], ["osascript", "-l", "JavaScript", "-"])

    def test_is_remote_true(self) -> None:
        self.assertTrue(self._make().is_remote)


class OpenSSHValidation(unittest.TestCase):
    """Missing required fields raise ValueError at construction time.

    Goodhart-proof: a silent default of '' or None for identity_file
    would silently degrade to the parent's ambient-auth semantics —
    exactly the regression this subclass exists to prevent.
    """

    def test_missing_identity_file_rejected(self) -> None:
        with self.assertRaises(ValueError) as cm:
            OpenSSHTransport(host_name="toopie", user="bob")
        self.assertIn("identity_file", str(cm.exception))

    def test_missing_user_rejected(self) -> None:
        with self.assertRaises(ValueError) as cm:
            OpenSSHTransport(host_name="toopie", identity_file="/tmp/k")
        self.assertIn("user", str(cm.exception))

    def test_empty_identity_file_rejected(self) -> None:
        with self.assertRaises(ValueError):
            OpenSSHTransport(host_name="toopie", identity_file="", user="x")

    def test_empty_user_rejected(self) -> None:
        with self.assertRaises(ValueError):
            OpenSSHTransport(host_name="toopie", identity_file="/tmp/k", user="")


class OpenSSHExecution(unittest.TestCase):
    """Execute checks identity_file existence before launching ssh."""

    def test_execute_raises_when_identity_file_missing(self) -> None:
        t = OpenSSHTransport(
            host_name="toopie",
            identity_file="/does/not/exist/hal-osa-key",
            user="bob",
        )
        with self.assertRaises(OsaError) as cm:
            t.execute("return 1", language="AppleScript", timeout_s=5)
        self.assertIn("identity_file not found", str(cm.exception))

    @patch("lib.hal.osa.transport.shutil.which", return_value="/usr/bin/ssh")
    @patch("lib.hal.osa.transport.subprocess.run")
    def test_execute_happy_path_delegates_to_parent(
        self, mock_run, _which, *, tmp_path=None
    ) -> None:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".key") as key_file:
            mock_run.return_value = _fake_completed(stdout="ok", returncode=0)
            t = OpenSSHTransport(
                host_name="toopie",
                identity_file=key_file.name,
                user="bob",
            )
            result = t.execute("return 1", language="AppleScript", timeout_s=5)
            self.assertTrue(result.ok)
            self.assertEqual(result.stdout, "ok")
            # Verify argv contains the -i <identity> pair
            argv = mock_run.call_args.args[0]
            self.assertIn(key_file.name, argv)
            self.assertIn("IdentitiesOnly=yes", " ".join(argv))


class OpenSSHArgvShapeH143(unittest.TestCase):
    """H143 synthetic argv-shape invariants that must match live behaviour.

    These tests pin the exact argv shape the OpenSSHTransport emits so
    regressions in flag ordering, timeout clamping, or target resolution
    surface without needing a paired Mac. The live-smoke harness
    (tests/test_osa_transport_live.py) is the same argv executed end-to-
    end; this class keeps the synthetic contract locked so a green
    offline suite is a reasonable predictor of a green live run.
    """

    def _transport(self, **overrides) -> OpenSSHTransport:
        base = dict(
            host_name="toopie", ip="100.64.0.10",
            identity_file="/tmp/hal-key", user="bob",
            connect_timeout_s=5, control_persist_s=30,
        )
        base.update(overrides)
        return OpenSSHTransport(**base)

    def test_identity_file_follows_minus_i(self) -> None:
        argv = self._transport()._build_cmd("AppleScript")
        idx = argv.index("-i")
        self.assertEqual(argv[idx + 1], "/tmp/hal-key")

    def test_user_follows_minus_l(self) -> None:
        argv = self._transport()._build_cmd("AppleScript")
        idx = argv.index("-l")
        self.assertEqual(argv[idx + 1], "bob")

    def test_connect_timeout_argv_respects_field(self) -> None:
        argv = self._transport(connect_timeout_s=9)._build_cmd("AppleScript")
        self.assertIn("ConnectTimeout=9", " ".join(argv))

    def test_control_persist_below_floor_is_clamped_up(self) -> None:
        argv = self._transport(control_persist_s=1)._build_cmd("AppleScript")
        self.assertIn("ControlPersist=10s", " ".join(argv))

    def test_control_persist_above_ceiling_is_clamped_down(self) -> None:
        argv = self._transport(control_persist_s=100_000)._build_cmd("AppleScript")
        self.assertIn("ControlPersist=600s", " ".join(argv))

    def test_batchmode_and_identities_only_always_set(self) -> None:
        joined = " ".join(self._transport()._build_cmd("AppleScript"))
        self.assertIn("BatchMode=yes", joined)
        self.assertIn("IdentitiesOnly=yes", joined)

    def test_target_is_ip_when_available(self) -> None:
        argv = self._transport(ip="100.1.2.3")._build_cmd("AppleScript")
        self.assertIn("100.1.2.3", argv)

    def test_target_falls_back_to_host_name_when_ip_empty(self) -> None:
        argv = self._transport(ip="")._build_cmd("AppleScript")
        self.assertIn("toopie", argv)

    def test_language_is_passed_via_osascript_minus_l(self) -> None:
        argv = self._transport()._build_cmd("JavaScript")
        self.assertEqual(argv[-4:], ["osascript", "-l", "JavaScript", "-"])

    def test_stdin_marker_is_dash_literal(self) -> None:
        """The trailing ``-`` tells osascript to read the script body
        from stdin. Swapping for a file path would break the pipe
        pattern and expose the body on the command line (shell-history
        leak risk)."""
        argv = self._transport()._build_cmd("AppleScript")
        self.assertEqual(argv[-1], "-")


if __name__ == "__main__":
    unittest.main()
