"""HAPPI v1.1 audit / IDR terminator support.

Implements the ``flags.audit=true`` opt-in: when enabled on an input
envelope, the runtime buffers every emitted event line in a per-dispatch
buffer and appends one ``idr`` event after ``completed``/``error`` whose
``idr_ref.sha256`` chains ``sha256(envelope_raw_bytes + each_event_line_utf8)``.

Mirrors the canonical polyglot runtime in
``~/.hal/happi-sources/04-python-runtime.py`` (lines 8-12, 23-29, 33-49,
416-447). The polyglot uses module globals because it dispatches one
envelope per process; the lib runtime can be embedded in long-running
servers (HTTP, socket) and must NOT leak buffer state across concurrent
dispatches. Two isolation layers:

1. ``threading.local`` — concurrent transports run on per-request
   threads; each thread sees its own buffer.
2. ``audit_session(...)`` context manager — explicit scope per dispatch.
   Entering pushes a fresh empty buffer; exiting pops it (and any frames
   that nest beneath it). No global accumulator across dispatches even
   on the same thread.

Public API mirrors the polyglot's three globals + ``emit_idr`` semantics:

    audit_session(model_versions=...)   — scope-bounded enable
    is_audit_enabled() -> bool          — current state
    record_event(line: str)             — emit-side buffer write
    compute_idr_sha256(env_raw: bytes)  — terminator hash, no side-effects
    audit_model_versions() -> list[str] — current scope's model_versions

The hashing rule is documented at ``~/.hal/happi.md`` IDR §,
mirrored here verbatim: ``sha256(envelope_raw + ''.join(buffered_lines))``,
each line UTF-8 encoded, no trailing newline (the line as ``record_event``
saw it).
"""
from __future__ import annotations

import hashlib
import threading
from contextlib import contextmanager
from typing import Iterator


_state = threading.local()


def _stack() -> list[dict]:
    """Return the current thread's audit-frame stack, creating it on first use."""
    if not hasattr(_state, "stack"):
        _state.stack = []
    return _state.stack


def _current_frame() -> dict | None:
    stack = _stack()
    return stack[-1] if stack else None


def is_audit_enabled() -> bool:
    """True if a thread-local audit frame is currently active."""
    return _current_frame() is not None


def record_event(line: str) -> None:
    """Append ``line`` to the active audit buffer if one is in scope.

    No-op when audit is disabled. Called from ``runtime._emit_with_audit``
    (the audit-aware wrapper around event emission). The line is captured
    verbatim — no normalisation, no trailing newline added; this matches
    the polyglot ``_emit`` semantics where ``_AUDIT_BUFFER.append(line)``
    sees the JSON exactly as ``json.dumps(...)`` produced it.
    """
    frame = _current_frame()
    if frame is not None:
        frame["lines"].append(line)


def audit_model_versions() -> list[str]:
    """Return the model_versions list bound to the current audit frame."""
    frame = _current_frame()
    if frame is None:
        return []
    return list(frame["model_versions"])


def audit_buffered_lines() -> list[str]:
    """Return a copy of the lines buffered in the current audit frame.

    Test-only helper. Production code uses ``compute_idr_sha256`` which
    consumes the buffer internally without exposing it.
    """
    frame = _current_frame()
    if frame is None:
        return []
    return list(frame["lines"])


def compute_idr_sha256(envelope_raw: bytes) -> str:
    """Compute sha256(envelope_raw + each_buffered_line_utf8).

    Pure function over the current audit frame. Called once at
    dispatch-tail by ``runtime._dispatch`` to derive the ``idr_ref.sha256``
    field. Returns ``""`` if no audit frame is active (the runtime should
    not be calling this in that case — defensive).
    """
    frame = _current_frame()
    if frame is None:
        return ""
    digest = hashlib.sha256()
    digest.update(envelope_raw)
    for line in frame["lines"]:
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


@contextmanager
def audit_session(
    *, model_versions: list[str] | None = None
) -> Iterator[None]:
    """Push a fresh audit frame for the lifetime of the ``with`` block.

    Pattern (canonical use):

        with audit_session(model_versions=env.flags.get("model_versions")):
            for line in _dispatch_core(envelope):
                record_event(line)
                yield line
            yield emit_idr(envelope.id, sha256=compute_idr_sha256(env_raw),
                           model_versions=audit_model_versions())

    The frame is popped on context exit even if an exception propagates
    through the body, so concurrent dispatches on the same thread cannot
    inherit a leaked buffer.
    """
    frame = {"lines": [], "model_versions": list(model_versions or [])}
    _stack().append(frame)
    try:
        yield
    finally:
        _stack().pop()
