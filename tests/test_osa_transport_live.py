"""H143 live-smoke harness for :class:`OpenSSHTransport`.

Executes a trivial AppleScript on a paired Mac reached via OpenSSH key
auth (not Tailscale SSH). Skipped unless the operator has set the
environment variables pointing at a reachable peer.

To run::

    export HAL_LIVE_SMOKE_HOST=100.64.0.10     # or a hostname
    export HAL_LIVE_SMOKE_USER=toopie              # remote login user
    export HAL_LIVE_SMOKE_KEY=~/.ssh/grip-remote   # private key path
    python3 -m pytest tests/test_osa_transport_live.py -v

See ``docs/osa-openssh-live-smoke.md`` for the full runbook.

The test is intentionally minimal — one AppleScript return that echoes
a fixed string — so the shape of the failure tells the operator which
layer broke (SSH transport / argv / osascript / remote permissions).

Hypothesis H163: OpenSSHTransport.execute correctly composes the argv
shape and runs end-to-end against a paired Mac. This harness is the
executable expression of that claim; passing it counts as live
evidence.
"""
from __future__ import annotations

import os
import shutil
import unittest
from pathlib import Path

from lib.hal.osa.transport import OpenSSHTransport


_HOST = os.environ.get("HAL_LIVE_SMOKE_HOST", "")
_USER = os.environ.get("HAL_LIVE_SMOKE_USER", "")
_KEY = os.environ.get("HAL_LIVE_SMOKE_KEY", "")


def _preconditions_met() -> tuple[bool, str]:
    """Return (ok, reason) — reason is the skip message when ok=False."""
    if not _HOST:
        return False, "HAL_LIVE_SMOKE_HOST not set"
    if not _USER:
        return False, "HAL_LIVE_SMOKE_USER not set"
    if not _KEY:
        return False, "HAL_LIVE_SMOKE_KEY not set"
    if shutil.which("ssh") is None:
        return False, "ssh binary not found on PATH"
    key_path = Path(_KEY).expanduser()
    if not key_path.exists():
        return False, f"identity file {key_path!r} does not exist"
    return True, ""


class OpenSSHLiveSmoke(unittest.TestCase):
    """Live-fire tests — skip unless HAL_LIVE_SMOKE_* env is populated."""

    @classmethod
    def setUpClass(cls) -> None:
        ok, reason = _preconditions_met()
        if not ok:
            raise unittest.SkipTest(reason)

    def _transport(self, **overrides) -> OpenSSHTransport:
        kwargs = dict(
            host_name=_HOST,
            ip=_HOST,
            identity_file=_KEY,
            user=_USER,
            connect_timeout_s=5,
            control_persist_s=30,
        )
        kwargs.update(overrides)
        return OpenSSHTransport(**kwargs)

    def test_echo_applescript_returns_live_ok(self) -> None:
        """Canonical one-liner: `return "live-ok"` echoes back over stdout.

        This is the executable expression of H163 — a failure here means
        one of (transport argv / SSH auth / remote osascript / remote
        accessibility policy) is broken, and the error message identifies
        which.
        """
        transport = self._transport()
        result = transport.execute(
            body='return "live-ok"',
            language="AppleScript",
            timeout_s=15,
        )
        self.assertTrue(
            result.ok,
            msg=f"rc={result.returncode} stderr={result.stderr!r}",
        )
        self.assertIn("live-ok", result.stdout)

    def test_timeout_propagates_on_slow_script(self) -> None:
        """A deliberately slow script exceeds timeout_s and raises OsaTimeout.

        AppleScript's ``delay 10`` sleeps the remote runtime. We set
        timeout_s=2 so the transport must surface a timeout, not a zero
        exit. This pins the timeout-pipeline across the transport layer."""
        from lib.hal.osa.osa import OsaTimeout

        transport = self._transport()
        with self.assertRaises(OsaTimeout):
            transport.execute(
                body='delay 10\nreturn "too late"',
                language="AppleScript",
                timeout_s=2,
            )


if __name__ == "__main__":
    unittest.main()
