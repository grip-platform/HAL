"""HAL A2A Peer-Exec — capability-locked delegation to peer harness CLIs.

Downstream plane of the bidirectional model-mesh hub. Lets one harness
(Claude Code, Codex, Antigravity/Gemini) delegate a prompt to ANOTHER harness's
agent and get the reply back — each peer invoked in its NATIVE non-interactive
mode under a read-only capability lock. The three giants collaborate on their
own native models with zero gateway changes (this is the "fast wow" — it does
NOT require the upstream multi-provider gateway).

Layering: sits ON TOP of ``lib/hal/a2a.py`` (the in-process A2ABus). The bus
moves messages between agents; this module is the EDGE that turns an outbound
A2A message into a real peer-CLI invocation and the peer's stdout back into an
inbound A2A message.

Safety (rules/untrusted-content-capability-lock.md): a peer's prompt is
untrusted-origin content driving a spawn, so every peer is invoked with an
ALLOWLISTED read-only capability surface BY CONSTRUCTION — never a bare or
write-capable spawn. The lock lives in the artefact (asserted by the test
suite), not in caller discipline. An unknown harness id fails closed (raises),
never degrading to a default-permissive spawn. The prompt is always a single
argv element (no ``shell=True``) so there is no shell-injection surface.

Syscall doctrine: the subprocess runner is DIP-injected (``runner=``) so tests
exercise the lock + routing with zero real CLI calls and zero network.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from lib.hal.a2a import A2ABus, A2AMessage

# Canonical peer-harness identifiers.
CLAUDE_CODE = "claude_code"
CODEX = "codex"
ANTIGRAVITY = "antigravity"  # the Gemini-based Antigravity CLI

# Accepted aliases → canonical id (kept tiny + explicit; not a fuzzy matcher).
_ALIASES: Dict[str, str] = {
    "claude": CLAUDE_CODE,
    "cc": CLAUDE_CODE,
    "claude-code": CLAUDE_CODE,
    "gemini": ANTIGRAVITY,
    "antigravity-cli": ANTIGRAVITY,
    "agy": ANTIGRAVITY,
}

# Sentinel marking where the prompt is slotted into each argv template. Per-peer
# ORDER matters: gemini/claude take the prompt as the value of -p (so it must
# immediately follow -p); codex takes it as a trailing positional after the
# sandbox flag. A uniform "lock then prompt" ordering would mis-bind -p.
_PROMPT = object()  # unique-identity sentinel; `is` is robust across refactor/
#                     marshal/alternate runtimes (a string sentinel could be
#                     interned to a distinct object and silently mis-substitute).


@dataclass(frozen=True)
class _PeerSpec:
    """Capability-locked invocation template for one peer harness."""
    argv_template: tuple   # tokens; the _PROMPT sentinel is replaced by the prompt
    lock_tokens: tuple     # tokens that MUST be present — the read-only lock


# Read-only capability lock per harness:
#   claude  → --tools Read,Grep,Glob   (no Write/Edit/Bash)
#   codex   → --sandbox read-only
#   gemini  → --approval-mode plan     (plan = read-only mode per `gemini --help`)
_PEERS: Dict[str, _PeerSpec] = {
    CLAUDE_CODE: _PeerSpec(
        argv_template=("claude", "-p", _PROMPT, "--tools", "Read,Grep,Glob"),
        lock_tokens=("--tools", "Read,Grep,Glob"),
    ),
    CODEX: _PeerSpec(
        argv_template=("codex", "exec", "--sandbox", "read-only", _PROMPT),
        lock_tokens=("--sandbox", "read-only"),
    ),
    ANTIGRAVITY: _PeerSpec(
        argv_template=("gemini", "-p", _PROMPT, "--approval-mode", "plan"),
        lock_tokens=("--approval-mode", "plan"),
    ),
}

# Tokens that would WIDEN the surface past read-only. Never permitted in any
# peer argv; the test suite asserts their absence, so a mutation that re-adds a
# write/yolo/full-access mode fails CI rather than shipping a privileged spawn.
_FORBIDDEN_TOKENS: frozenset = frozenset({
    "yolo", "auto_edit", "danger-full-access", "workspace-write",
    "--dangerously-bypass-approvals-and-sandbox", "--dangerously-skip-permissions",
    "acceptEdits", "bypassPermissions", "--full-auto",
})

Runner = Callable[[List[str], Optional[float]], "subprocess.CompletedProcess"]


class UnknownPeerError(ValueError):
    """Unrecognised harness id — fail-closed, never a default-permissive spawn."""


@dataclass(frozen=True)
class PeerReply:
    """Result of a peer-harness delegation."""
    harness: str
    ok: bool
    output: str
    exit_code: int


def list_peers() -> List[str]:
    """Canonical peer ids this module can delegate to."""
    return sorted(_PEERS)


def canonical_harness(name: str) -> str:
    """Resolve a harness name/alias to its canonical id, or fail closed."""
    key = (name or "").strip().lower()
    if key in _PEERS:
        return key
    if key in _ALIASES:
        return _ALIASES[key]
    raise UnknownPeerError(
        f"unknown peer harness {name!r}; known: {sorted(_PEERS)} "
        f"(aliases: {sorted(_ALIASES)})"
    )


def build_peer_argv(harness: str, prompt: str) -> List[str]:
    """Capability-locked argv for a peer harness.

    Read-only by construction; prompt is a single argv element (no shell).
    Unknown harness raises ``UnknownPeerError`` (fail-closed).
    """
    spec = _PEERS[canonical_harness(harness)]
    return [prompt if tok is _PROMPT else tok for tok in spec.argv_template]


def _default_runner(argv: List[str], timeout: Optional[float]) -> "subprocess.CompletedProcess":
    """Real subprocess boundary (list argv, never shell). DIP-injected in tests."""
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)


def ask_peer(
    harness: str,
    prompt: str,
    *,
    runner: Runner = _default_runner,
    timeout: float = 120.0,
) -> PeerReply:
    """Delegate ``prompt`` to a peer harness (read-only locked) and return its reply."""
    canon = canonical_harness(harness)  # raises (fail-closed) before any spawn
    argv = build_peer_argv(canon, prompt)
    try:
        proc = runner(argv, timeout)
    except subprocess.TimeoutExpired:
        return PeerReply(canon, ok=False, output=f"peer timed out after {timeout}s", exit_code=124)
    ok = proc.returncode == 0
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    # Provenance: a successful reply is stdout; a failure surfaces stderr (so a
    # non-zero exit carrying an error on stderr is never mistaken for an
    # empty-stdout success). `ok` remains the authoritative success signal.
    output = stdout if ok else (stderr or stdout)
    return PeerReply(canon, ok=ok, output=output, exit_code=proc.returncode)


def ask_peer_via_bus(
    bus: A2ABus,
    from_agent: str,
    harness: str,
    prompt: str,
    *,
    runner: Runner = _default_runner,
    timeout: float = 120.0,
) -> PeerReply:
    """``ask_peer`` + route the reply back to ``from_agent`` over the A2ABus."""
    reply = ask_peer(harness, prompt, runner=runner, timeout=timeout)
    bus.send(A2AMessage(
        from_agent=reply.harness,
        to_agent=from_agent,
        content=reply.output,
        metadata=(("ok", str(reply.ok)), ("exit_code", str(reply.exit_code))),
    ))
    return reply
