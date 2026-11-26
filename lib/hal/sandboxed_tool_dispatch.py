"""Capability-locked, sandbox-bound dispatcher for ``/api/infer-with-tools``.

This module is the bridge between HAL's pre-existing tool-loop driver
(:mod:`lib.hal.tool_loop`) and the path-allowlist file primitives
(:mod:`lib.hal.file_sandbox`). It exposes ONE class —
:class:`SandboxedDispatcher` — whose public surface is identical to the
duck-type :func:`lib.hal.tool_loop.run_tool_loop` expects from a
dispatcher (a single ``execute(name, input) -> ToolResult-like`` method).

Capability-lock invariant
-------------------------

Per ``rules/untrusted-content-capability-lock.md`` (GRIP), an action
driven by untrusted-origin content must have its **capability surface**
restricted, not just its sender or prompt-skip handled. This dispatcher
encodes that as two compound mechanical constraints:

1. **Allowlist is a hardcoded :class:`frozenset`**:
   :data:`_ALLOWED_TOOLS` = ``frozenset({"Read", "Grep", "Glob"})``.
   There is no public mutator. Any other tool name (``Bash``, ``Write``,
   ``Edit``, ``Task``, ``Agent``, ``WebFetch``, etc.) — including
   unknown / renamed tools — is REJECTED with ``blocked=True`` BEFORE
   any handler dispatch. **Fail-CLOSED on rename**: a future tool that
   gets renamed in the model's tool spec but not here cannot slip
   through; it simply falls off the allowlist and gets denied.

2. **Every filesystem access flows through** :mod:`lib.hal.file_sandbox`
   primitives bound to the validated ``sandbox_root``. ``realpath`` +
   ``commonpath`` checks refuse symlink escape, ``..`` traversal, and
   absolute paths that aren't a descendant of the root.

The dispatcher quacks like
:class:`lib.hal.tool_dispatch.ToolDispatcher` only in the narrow sense
:func:`lib.hal.tool_loop.run_tool_loop` requires:
``execute(name: str, input: dict) -> ToolResult-like``. The return
object exposes ``output`` / ``error`` / ``blocked`` / ``reason``
attributes so the existing :func:`lib.hal.tool_loop._build_tool_result`
adapter loop reads it without modification.

Falsifier
---------

This module is wrong if any of these can happen:

- A tool name outside :data:`_ALLOWED_TOOLS` reaches a handler.
- A filesystem operation escapes ``sandbox_root`` via symlink, ``..``,
  or an absolute path outside the root.
- Construction succeeds with an empty / non-existent / refused root.

Each failure mode is covered by a Goodhart-resistant test in
``tests/test_sandboxed_tool_dispatch.py`` (every test docstring names
the mutation it survives).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from lib.hal import file_sandbox
from lib.hal.file_sandbox import (
    SandboxBinaryError,
    SandboxEscapeError,
    SandboxTimeoutError,
    validate_root,
)


# Hardcoded capability allowlist — frozenset so it cannot be mutated.
# Mirrors the W1 W4-anchor invariant from PR #3268 (capability-locked
# tool surface for untrusted-content-driven spawns).
_ALLOWED_TOOLS: frozenset[str] = frozenset({"Read", "Grep", "Glob"})


@dataclass(frozen=True)
class SandboxedResult:
    """ToolResult-shaped value returned by :meth:`SandboxedDispatcher.execute`.

    Attribute names match :class:`lib.hal.tool_dispatch.ToolResult` so
    :func:`lib.hal.tool_loop._build_tool_result` reads it unchanged:

    - ``output`` — successful tool output (string).
    - ``error`` — recoverable error message (the model sees it, the
      loop continues).
    - ``blocked`` — true when the capability lock or sandbox boundary
      refused the call. The loop sees an ``is_error`` tool_result with
      ``reason`` as content.
    - ``reason`` — human-readable explanation for the block.

    Frozen because the loop never mutates a tool result after it lands.
    """
    output: str = ""
    error: str = ""
    blocked: bool = False
    reason: str = ""


class SandboxedDispatcher:
    """Tool dispatcher whose capability surface is exactly Read/Grep/Glob,
    each routed through :mod:`lib.hal.file_sandbox` bound to a validated
    sandbox root.

    Construction validates ``sandbox_root`` eagerly via
    :func:`lib.hal.file_sandbox.validate_root` — empty, non-existent,
    or refused roots (``/``, ``/etc``, ``/bin``, ``/sbin``, etc.) raise
    :class:`lib.hal.file_sandbox.SandboxRootRefused` so a misconfigured
    server route cannot accept requests at all.

    The dispatcher is stateless beyond the bound root. Multiple
    concurrent calls to :meth:`execute` are safe (the underlying
    file_sandbox primitives use ``os.path.realpath`` per-call).
    """

    def __init__(self, sandbox_root: str):
        # Validate now — raises SandboxRootRefused on bad root.
        # We store the canonical realpath returned by validate_root,
        # NOT the caller's string. This is what enforces "the sandbox
        # root is canonical" — escape checks elsewhere compare against
        # this value.
        self._root = validate_root(sandbox_root)

    @property
    def sandbox_root(self) -> str:
        """Canonical (realpath-resolved) sandbox root."""
        return self._root

    @property
    def allowlist(self) -> frozenset[str]:
        """Read-only view of the capability allowlist. Exposed for
        tests + introspection; mutating the returned frozenset is
        impossible by construction."""
        return _ALLOWED_TOOLS

    def execute(self, tool_name: str, tool_input: dict) -> SandboxedResult:
        """Dispatch one tool call.

        Step 1 — Capability lock: refuse anything not in
        :data:`_ALLOWED_TOOLS`. This fires BEFORE input parsing so a
        malformed call to a disallowed tool still gets the
        capability-lock rejection (not an unhelpful "missing pattern"
        error).

        Step 2 — Per-tool handler: route through ``_exec_read`` /
        ``_exec_grep`` / ``_exec_glob``. Each handler catches sandbox
        escapes and surfaces them as ``blocked=True`` so the loop sees
        ``is_error=True`` rather than the caller crashing.

        Step 3 — Unreachable fall-through: present as a safety belt
        if :data:`_ALLOWED_TOOLS` ever drifts from the handler table
        without the dispatch wiring being updated. Returns
        ``blocked=True`` rather than silently succeeding.
        """
        if tool_name not in _ALLOWED_TOOLS:
            return SandboxedResult(
                blocked=True,
                reason=(
                    f"tool {tool_name!r} is not in the capability allowlist "
                    f"{sorted(_ALLOWED_TOOLS)}"
                ),
            )

        if tool_name == "Read":
            return self._exec_read(tool_input)
        if tool_name == "Grep":
            return self._exec_grep(tool_input)
        if tool_name == "Glob":
            return self._exec_glob(tool_input)

        # Defence-in-depth: allowlist drifted past the dispatch table.
        # Concrete failure mode: someone adds a new tool name to
        # _ALLOWED_TOOLS without adding the matching handler branch.
        return SandboxedResult(
            blocked=True,
            reason=(
                f"tool {tool_name!r} is in the allowlist but has no handler "
                "(internal inconsistency)"
            ),
        )

    # ----- Per-tool handlers -----

    def _exec_read(self, tool_input: dict) -> SandboxedResult:
        """Read a sandboxed file. Honours ``file_path`` (canonical) and
        ``path`` (Anthropic Messages tool input variant) so the same
        tool schema works whether the model emits ``Read`` with
        ``file_path`` or ``path``.

        Returns ``blocked=True`` on sandbox escape (symlink,
        ``..``-traversal, absolute path outside root). Returns
        ``error`` for missing arg / not-a-file / I/O failure — those
        let the model retry with a corrected path."""
        path = tool_input.get("file_path") or tool_input.get("path") or ""
        if not path:
            return SandboxedResult(
                error="Read requires 'file_path' (or 'path') argument",
            )
        try:
            content = file_sandbox.read_text(self._root, path)
        except SandboxEscapeError as exc:
            return SandboxedResult(
                blocked=True, reason=f"sandbox escape on Read: {exc}",
            )
        except SandboxBinaryError as exc:
            # Recoverable: model can pick Glob/Grep with a glob_filter
            # excluding binaries, or a different path. NOT blocked
            # (capability lock is intact) — just an unreadable file.
            return SandboxedResult(error=str(exc))
        except FileNotFoundError as exc:
            return SandboxedResult(error=str(exc))
        except OSError as exc:
            return SandboxedResult(error=str(exc))
        return SandboxedResult(output=content)

    def _exec_grep(self, tool_input: dict) -> SandboxedResult:
        """Grep a regex pattern under the sandbox root. Accepts
        optional ``glob`` (or ``glob_filter``) to restrict the file set.

        ``pattern`` is required. Output is one line per hit in
        ``path:line:text`` form (relative paths). Empty hit list
        yields ``"(no matches)"`` so the model gets a deterministic
        non-empty response rather than guessing why the grep was
        silent."""
        pattern = tool_input.get("pattern", "")
        if not pattern:
            return SandboxedResult(error="Grep requires 'pattern' argument")
        glob_filter: Optional[str] = (
            tool_input.get("glob") or tool_input.get("glob_filter")
        )
        try:
            hits = file_sandbox.grep_pattern(
                self._root, pattern, glob_filter=glob_filter,
            )
        except SandboxTimeoutError as exc:
            # Recoverable: model can retry with a narrower glob_filter
            # or split the query. NOT blocked.
            return SandboxedResult(error=str(exc))
        if not hits:
            return SandboxedResult(output="(no matches)")
        text = "\n".join(f"{h.path}:{h.line}:{h.text}" for h in hits)
        return SandboxedResult(output=text)

    def _exec_glob(self, tool_input: dict) -> SandboxedResult:
        """Glob a pattern under the sandbox root. Absolute patterns
        are refused (``blocked=True``) — the file_sandbox primitive
        enforces this with the matching ``SandboxEscapeError``.

        Returns sorted-by-rglob-walk paths relative to the sandbox
        root, joined by newlines. Empty result → ``"(no matches)"``."""
        pattern = tool_input.get("pattern", "")
        if not pattern:
            return SandboxedResult(error="Glob requires 'pattern' argument")
        try:
            paths = file_sandbox.glob_pattern(self._root, pattern)
        except SandboxEscapeError as exc:
            return SandboxedResult(
                blocked=True, reason=f"sandbox escape on Glob: {exc}",
            )
        except SandboxTimeoutError as exc:
            return SandboxedResult(error=str(exc))
        if not paths:
            return SandboxedResult(output="(no matches)")
        return SandboxedResult(output="\n".join(paths))


def make_sandboxed_dispatcher(sandbox_root: str) -> SandboxedDispatcher:
    """Construct a :class:`SandboxedDispatcher` for ``sandbox_root``.

    Convenience constructor that mirrors
    :func:`lib.hal.tool_adapter.create_builtin_tools` so the server
    route reads symmetrically::

        dispatcher = make_sandboxed_dispatcher(body["sandbox_root"])
        result = run_tool_loop(messages=..., chat_fn=..., tools=...,
                               dispatcher=dispatcher)

    Raises :class:`lib.hal.file_sandbox.SandboxRootRefused` on bad
    root (propagated from :class:`SandboxedDispatcher.__init__`)."""
    return SandboxedDispatcher(sandbox_root)


__all__ = [
    "SandboxedResult",
    "SandboxedDispatcher",
    "make_sandboxed_dispatcher",
]
