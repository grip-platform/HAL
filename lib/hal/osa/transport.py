"""Transport abstraction for osa execution — local subprocess or remote SSH.

Phase γ-1 of the osascript control layer plan. Adds cross-host reach for
:func:`lib.hal.osa.osa` by lifting the subprocess-spawning decision out of
the kernel and into a :class:`Transport` strategy. Existing callers pass
nothing (or a :class:`LocalTransport`) and keep the α-2 behaviour; remote
dispatchers pass :class:`SSHTransport` and the same kernel delegates into
Tailscale SSH.

Two concrete transports are shipped in γ-1:

:class:`LocalTransport`
    Wraps the current subprocess invocation. No network. No auth beyond
    the α-3b capability-token layer.

:class:`SSHTransport`
    Invokes ``ssh <host> osascript -l <language> -`` with the pre-escaped
    body piped over stdin. Uses an OpenSSH ControlMaster multiplexer so
    repeated dispatches on the same host share one TCP session — the
    council invariant caps ``ControlPersist`` to prevent unbounded mux
    lifetime (see ``plans/osascript-control-layer.review-verdict.md``).

Authorisation across hosts
--------------------------

The α-3b capability-token system is *machine-local* by design: the HMAC
secret lives in one host's environment and cannot be verified on a remote
that does not share it. γ-1 does not try to synchronise the secret —
instead, the Tailscale SSH key **is** the cross-host auth anchor:

* Remote dispatches require operator consent via
  ``HAL_OSA_REMOTE_DISPATCH=1``.
* The local capability-token check is skipped when the transport is
  non-local (the remote host receives the already-interpolated script
  body and runs it; no tier check happens over the wire).
* Every remote dispatch is audit-logged with
  ``provenance=mcp_remote`` and a ``host`` field identifying the
  Tailscale target.

Why: baking capability-token sync into γ-1 would require a trusted
cross-host store (keychain sync, secret manager). That is γ-2
territory. Until then, the SSH key acts as the bearer token — only
peers on the operator's tailnet can dispatch.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lib.hal.osa.osa import Language, OsaError, OsaResult, OsaTimeout

__all__ = [
    "Transport",
    "LocalTransport",
    "SSHTransport",
    "OpenSSHTransport",
    "REMOTE_DISPATCH_ENV",
]


REMOTE_DISPATCH_ENV = "HAL_OSA_REMOTE_DISPATCH"
_DEFAULT_CONTROL_PERSIST_S = 300
_DEFAULT_CONNECT_TIMEOUT_S = 5


class Transport(ABC):
    """Strategy interface for executing an already-interpolated osa script body.

    Implementations are responsible for turning ``body`` into an
    :class:`OsaResult` — either by spawning ``osascript`` locally or by
    piping it over SSH to a remote Mac. Kernel-facing concerns like
    escape (handled before ``body`` reaches here) and auth (handled by
    dispatch.py before the transport is chosen) are out of scope.
    """

    @property
    @abstractmethod
    def host(self) -> str:
        """Identifier recorded on the audit entry. ``"local"`` for local."""

    @property
    @abstractmethod
    def is_remote(self) -> bool:
        """True when this transport crosses a machine boundary."""

    @abstractmethod
    def execute(
        self, body: str, *, language: Language, timeout_s: int
    ) -> OsaResult:
        """Run ``body`` and return a structured :class:`OsaResult`.

        Raises:
            :class:`OsaError` — transport-level failure (ssh missing,
                network unreachable, remote non-zero exit).
            :class:`OsaTimeout` — wall-clock budget exceeded.
        """


@dataclass(frozen=True)
class LocalTransport(Transport):
    """Run ``osascript`` in a local subprocess — the α-2 behaviour."""

    @property
    def host(self) -> str:
        return "local"

    @property
    def is_remote(self) -> bool:
        return False

    def execute(
        self, body: str, *, language: Language, timeout_s: int
    ) -> OsaResult:
        # Import here to avoid a circular import with osa.py.
        from lib.hal.osa.osa import osa_raw
        return osa_raw(body, language=language, timeout_s=timeout_s, check=False)


@dataclass(frozen=True)
class SSHTransport(Transport):
    """Dispatch ``body`` to a remote Mac via Tailscale SSH.

    The body is streamed over stdin (``osascript -l <lang> -``) so no
    remote shell escaping is required — whatever the caller already
    escaped locally is the exact source the remote osascript sees.

    A per-host OpenSSH ControlMaster socket keeps the mux lifetime
    bounded by :attr:`control_persist_s` — capped here at 600 s to
    respect the council's SSH mux invariant.
    """

    host_name: str
    ip: Optional[str] = None
    control_persist_s: int = _DEFAULT_CONTROL_PERSIST_S
    connect_timeout_s: int = _DEFAULT_CONNECT_TIMEOUT_S
    ssh_binary: str = "ssh"

    @property
    def host(self) -> str:
        return self.host_name

    @property
    def is_remote(self) -> bool:
        return True

    def _build_cmd(self, language: Language) -> list:
        target = self.ip or self.host_name
        control_s = min(max(self.control_persist_s, 10), 600)  # 10s..10min cap
        return [
            self.ssh_binary,
            "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={self.connect_timeout_s}",
            "-o", "ControlMaster=auto",
            "-o", f"ControlPersist={control_s}s",
            "-o", "ControlPath=~/.ssh/hal-osa-mux-%h",
            target,
            "osascript", "-l", language, "-",
        ]

    def execute(
        self, body: str, *, language: Language, timeout_s: int
    ) -> OsaResult:
        if shutil.which(self.ssh_binary) is None:
            raise OsaError(f"ssh binary not found on PATH: {self.ssh_binary!r}")
        cmd = self._build_cmd(language)
        started = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                input=body,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - started
            raise OsaTimeout(
                f"ssh+osascript to {self.host_name!r} exceeded "
                f"{timeout_s}s (elapsed={elapsed:.2f}s)"
            ) from exc
        elapsed = time.monotonic() - started
        return OsaResult(
            ok=(proc.returncode == 0),
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration_s=elapsed,
            language=language,
        )


@dataclass(frozen=True)
class OpenSSHTransport(SSHTransport):
    """Dispatch over explicit-IdentityFile OpenSSH (non-Tailscale-SSH).

    γ-1b extension of :class:`SSHTransport`. Where the Tailscale-SSH
    parent relies on ambient auth (enrolment in the operator's tailnet),
    this variant pins the identity explicitly:

    * ``-i <identity_file>`` forces the given private key
    * ``-o IdentitiesOnly=yes`` stops SSH from trying ambient agent keys
    * ``-l <user>`` declares the remote login name; required because the
      tailnet mapping is not available here

    Used for:

    - Peers not enrolled in Tailscale SSH (fresh Macs, bastion hops)
    - Lab machines reachable over Tailscale but authenticated with a
      dedicated key such as ``~/.ssh/grip-remote`` rather than Tailscale
      SSH enrolment
    - CI runners where a deploy key is checked into the secrets store

    The parent's ``host_name`` still identifies the audit-log entry, so
    operators grep logs by logical name regardless of which transport
    variant dispatched the script.

    Both new fields are keyword-only so dataclass inheritance does not
    reject them for following the parent's defaulted fields.
    """

    identity_file: str = field(default="", kw_only=True)
    user: str = field(default="", kw_only=True)

    def __post_init__(self) -> None:
        if not self.identity_file:
            raise ValueError(
                "OpenSSHTransport requires identity_file (e.g. ~/.ssh/grip-remote)"
            )
        if not self.user:
            raise ValueError(
                "OpenSSHTransport requires user (remote login name)"
            )

    def _build_cmd(self, language: Language) -> list:
        target = self.ip or self.host_name
        control_s = min(max(self.control_persist_s, 10), 600)
        return [
            self.ssh_binary,
            "-i", self.identity_file,
            "-l", self.user,
            "-o", "BatchMode=yes",
            "-o", "IdentitiesOnly=yes",
            "-o", f"ConnectTimeout={self.connect_timeout_s}",
            "-o", "ControlMaster=auto",
            "-o", f"ControlPersist={control_s}s",
            "-o", "ControlPath=~/.ssh/hal-osa-mux-%h",
            target,
            "osascript", "-l", language, "-",
        ]

    def execute(
        self, body: str, *, language: Language, timeout_s: int
    ) -> OsaResult:
        identity_path = Path(self.identity_file).expanduser()
        if not identity_path.exists():
            raise OsaError(
                f"OpenSSHTransport identity_file not found: {self.identity_file!r}"
            )
        return super().execute(body, language=language, timeout_s=timeout_s)
