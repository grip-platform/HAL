"""Idempotent TCP-port reclaim — free a port by killing whatever LISTENs on it.

Why this exists
---------------
A singleton local service (the HAL gateway, :mod:`lib.hal.gateway`) is launched
by launchd (``com.hal.gateway``, ``KeepAlive``; older GRIP installers named it
``com.grip.hal-gateway``) and, ad hoc, by hand. A
second launch while the first still holds the port dies with::

    OSError: [Errno 48] Address already in use

``SO_REUSEADDR`` (``server.allow_reuse_address``) does **not** rescue this:
it only lets a new socket reclaim a port left in ``TIME_WAIT``, never one an
*actively listening* socket still owns. So "launch again" crashes instead of
taking over.

This module is the idempotent-launch primitive. Given a port it finds the
process(es) holding it and terminates them, so a repeated launch always
converges to exactly one listener — the self-healing a cron/launchd-managed
singleton wants. It is scoped to ``LISTEN`` sockets on the *exact* TCP port
(via ``lsof``, the same scoping GRIP's ``bin/grip-hal`` uses), so it can never
kill an unrelated process. SIGTERM first (graceful); the platform's hard-kill
signal only for stragglers.

syscall-doctrine: one shim over the ``lsof``/``kill`` boundary — callers pass a
port and never touch ``lsof`` argv or signal numbers themselves.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time

__all__ = ["reclaim_port", "listening_pids", "port_in_use", "HARD_KILL_SIGNAL"]

# lsof on a single port is near-instant; cap it so a wedged lsof can never
# block a service launch indefinitely.
_LSOF_TIMEOUT_S = 5.0

#: The signal used to forcibly end a straggler after the grace window. Windows
#: has no SIGKILL, but os.kill(pid, SIGTERM) already calls TerminateProcess
#: there, so SIGTERM is also that platform's hard-kill escalation target.
HARD_KILL_SIGNAL = getattr(signal, "SIGKILL", signal.SIGTERM)


def listening_pids(port: int) -> list[int]:
    """Return PIDs holding a ``LISTEN`` socket on ``port`` (``[]`` if none).

    Uses ``lsof -iTCP:<port> -sTCP:LISTEN -t -n -P``. ``-t`` emits bare PIDs,
    ``-n``/``-P`` skip DNS/port-name lookups (faster, no surprises). Returns
    ``[]`` when ``lsof`` is absent or errors, so callers degrade to a plain
    bind attempt rather than crashing on a missing tool.
    """
    if shutil.which("lsof") is None:
        return []
    try:
        proc = subprocess.run(
            ["lsof", f"-iTCP:{port}", "-sTCP:LISTEN", "-t", "-n", "-P"],
            capture_output=True,
            text=True,
            timeout=_LSOF_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    pids: list[int] = []
    for token in proc.stdout.split():
        try:
            pids.append(int(token))
        except ValueError:
            continue
    return pids


def port_in_use(port: int) -> bool:
    """True if any process is currently ``LISTEN``ing on ``port``."""
    return bool(listening_pids(port))


def _terminate(pid: int, sig: int) -> None:
    """Send ``sig`` to ``pid``, swallowing already-gone / not-permitted."""
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError):
        # Already exited, or owned by another user we cannot signal — either
        # way there is nothing more this primitive can do about that PID.
        pass


def reclaim_port(port: int, *, grace_s: float = 2.0) -> list[int]:
    """Kill every process ``LISTEN``ing on ``port``; return the PIDs signalled.

    The idempotent-launch primitive. No-op (returns ``[]``) when the port is
    free. Never targets the caller's own PID. SIGTERM is sent first (graceful
    shutdown); any holder still listening after ``grace_s`` is escalated to
    the platform's hard kill (SIGKILL on POSIX; SIGTERM serves as Windows'
    hard kill). The returned list is the set we signalled, for caller logging.
    """
    me = os.getpid()
    targets = [pid for pid in listening_pids(port) if pid != me]
    if not targets:
        return []

    for pid in targets:
        _terminate(pid, signal.SIGTERM)

    # Give graceful shutdown a bounded window, then SIGKILL whatever survives.
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if not [pid for pid in listening_pids(port) if pid != me]:
            return targets
        time.sleep(0.1)

    for pid in [pid for pid in listening_pids(port) if pid != me]:
        _terminate(pid, HARD_KILL_SIGNAL)
    return targets
