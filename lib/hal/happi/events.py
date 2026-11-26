"""HAPPI NDJSON event encoder — stdout wire events.

Response to one envelope is a stream of NDJSON lines, each one a HAPPI
wire event. Canonical event vocabulary (α-1 subset + v1.1):

    started       — dispatch began
    delta         — streamed content chunk (text, token, etc.)
    thinking      — streamed reasoning chunk (a model thinking before it
                    answers); never part of the answer — additive, provider
                    adapters only, 2026-09-07
    tool_call     — the model asked for a tool (name, input, call_id); the
                    caller executes it — no HAPPI runtime hosts an executor
    tool_result   — a tool's output being handed back (name, output, call_id)
    sub_request   — nested HAPPI envelope spawned (for fractal recursion)
    completed     — dispatch finished successfully; carries usage
    error         — dispatch failed; carries code + message
    idr           — audit terminator (v1.1, opt-in via flags.audit=true)
    context       — memory-chain link (v1.1, content address of a decision body)
    continuity    — pre-registered falsifiable premise (v1.4, content address
                    of a commitment body); NOT a terminator — emitted BEFORE
                    the dispatch whose premise it registers
    reasoning     — a reasoning trace (v1.5, content address of a trace body),
                    carried as a DIAGNOSTIC SPECIMEN and never as evidence of
                    cause; stamped authoritative=false, role="diagnostic"

Every event carries `v`, `id`, `type`, `ts` (monotonic millis). Additional
fields are event-type specific. Encoded as one JSON object per line,
UTF-8, no trailing whitespace.

Protocol version flag (P7 — cutover)
------------------------------------
``PROTOCOL_VERSION`` is the value stamped into every emitted event's ``v``
field. The invariant, from P7 onward, is that it MATCHES THE CANONICAL
POLYGLOT RUNTIME's emission stamp (``happi-sources/04-python-runtime.py``
``HAPPI_VERSION``) — that is H428, not the particular number it happened to
be on the day P7 shipped.

That distinction was not academic. P7 set the default to ``"happi/1.1"`` and
the canonical runtime then moved to 1.2, 1.3 and 1.4 while this constant sat
still, so H428 was quietly refuted and the lib stamped a version three minor
releases behind the runtime it claims to match. The v1.4 continuity work made
the drift visible rather than merely untrue: the lib could emit a ``continuity``
event — an event type that does not exist before v1.4 — stamped ``happi/1.1``,
a record that contradicts itself and which a reader trusting the ``v`` field
would be right to reject.

The environment variable ``HAPPI_LIB_PROTOCOL_VERSION`` is an OPT-BACK control:
setting it to ``"happi/1.0"`` (or the bare ``"1.0"``) restores the legacy v1.0
stamp for callers that depend on the old wire shape during their migration
window. Any other value — including empty, whitespace, or a stale version
string such as ``"happi/1.1"`` — yields the current default. Emission MUST NOT
raise on a misconfigured flag.

This inverts the P2 contract (which defaulted to v1.0 and treated v1.1 as
the override). Callers relying on the v1.0 default must set
``HAPPI_LIB_PROTOCOL_VERSION=happi/1.0`` to preserve legacy emission; that
opt-back is the ONLY pinned value and is deliberately not tied to the moving
default.

The flag is process-scoped at import time; tests that need both modes
either reload this module via ``importlib.reload`` or call
``_resolve_protocol_version()`` directly.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any


# Tracks the canonical polyglot runtime (H428). Bump WITH
# happi-sources/04-python-runtime.py::HAPPI_VERSION, never independently.
PROTOCOL_VERSION_DEFAULT = "happi/1.6"
PROTOCOL_VERSION_V11 = "happi/1.1"
PROTOCOL_VERSION_V10 = "happi/1.0"
_ALLOWED_V10_ALIASES = ("1.0", "happi/1.0")


def _resolve_protocol_version() -> str:
    """Return the runtime's protocol version, honouring the env-var flag.

    Defaults to ``PROTOCOL_VERSION_DEFAULT``, which tracks the canonical
    polyglot runtime's stamp (H428). Set
    ``HAPPI_LIB_PROTOCOL_VERSION="happi/1.0"`` (or the bare ``"1.0"``) to
    opt BACK to v1.0 stamping for legacy callers. Any other value —
    including empty, whitespace, a stale version string, or junk — falls
    through to the default. Emission must never raise on a misconfigured
    flag.
    """
    flag = os.environ.get("HAPPI_LIB_PROTOCOL_VERSION", "").strip()
    if flag in _ALLOWED_V10_ALIASES:
        return PROTOCOL_VERSION_V10
    return PROTOCOL_VERSION_DEFAULT


PROTOCOL_VERSION = _resolve_protocol_version()


def _now_ms() -> int:
    """Monotonic-ish wall-clock milliseconds. Used as event timestamp."""
    return int(time.time() * 1000)


def emit_event(
    envelope_id: str,
    event_type: str,
    *,
    version: str = PROTOCOL_VERSION,
    **payload: Any,
) -> str:
    """Encode one wire event as a single NDJSON line.

    Deterministic field order (v, id, type, ts first) keeps golden tests
    stable across Python versions. Trailing newline is NOT included — caller
    writes it to preserve flush-per-event discipline.
    """
    evt: dict[str, Any] = {
        "v": version,
        "id": envelope_id,
        "type": event_type,
        "ts": _now_ms(),
    }
    # The envelope identity is not payload. ``emit_completed(**extra)`` forwards
    # caller-supplied keys here, and a stray ``type=`` / ``id=`` used to win the
    # ``update`` and rewrite the event's identity silently (HAL#551 review,
    # concern A3). Refuse loudly instead of choosing an order.
    clash = sorted(k for k in payload if k in evt)
    if clash:
        raise ValueError(f"event payload may not carry the envelope keys {clash}")
    evt.update(payload)
    return json.dumps(evt, separators=(",", ":"), ensure_ascii=False)


def emit_started(envelope_id: str) -> str:
    return emit_event(envelope_id, "started")


def emit_delta(envelope_id: str, text: str) -> str:
    return emit_event(envelope_id, "delta", text=text)


def emit_thinking(envelope_id: str, text: str) -> str:
    """One streamed reasoning chunk (Moonshot ``delta.reasoning_content``;
    the same slot an Anthropic thinking block would use). Kept OFF the
    ``delta`` channel on purpose: a consumer that joins ``delta.text`` gets
    the answer alone, exactly as before. Joined ``thinking`` texts are the
    ``reasoning_content`` the next turn must echo -- ``completed.message``
    carries that assembled value so nobody has to re-join events.
    """
    return emit_event(envelope_id, "thinking", text=text)


def emit_tool_call(envelope_id: str, name: str, tool_input: Any, call_id: str) -> str:
    """The model requested a tool. Wire fields follow the event table in
    happi.md (Axiom 4 makes that document the contract):

        | `tool_call` | `name`, `input`, `call_id` | The model asked for a tool; the caller runs it |

    ``tool_input`` is the decoded ``function.arguments`` -- a JSON object in
    the normal case; the adapter passes the raw string through when the
    provider's arguments were not valid JSON (e.g. truncated at the token
    budget), so nothing is lost silently. ``call_id`` is the provider's
    ``tool_calls[].id`` and MUST be echoed as ``tool_call_id`` on the result.

    This library previously emitted ``{tool, args}`` with no ``call_id`` --
    the same divergence class PR #523 closed for ``sub_request``. Zero
    callers existed before W1b, so the rename breaks nobody.
    """
    return emit_event(
        envelope_id, "tool_call", name=name, input=tool_input, call_id=call_id
    )


def emit_tool_result(envelope_id: str, name: str, output: Any, call_id: str) -> str:
    """A tool's output being handed back to the model. Wire fields per the
    happi.md event table:

        | `tool_result` | `name`, `output`, `call_id` | Tool returned |

    The former ``{tool, ok, result}`` shape is gone; an error is reported
    inside ``output`` by the tool's own convention, as on the OpenAI wire.
    """
    return emit_event(
        envelope_id, "tool_result", name=name, output=output, call_id=call_id
    )


def emit_sub_request(envelope_id: str, child: dict[str, Any]) -> str:
    """Nested HAPPI envelope — supports the recursive fractal composition
    property. `child` is a full envelope dict as the caller would ship it.

    The wire field is `envelope`, per the event table in happi.md:

        | `sub_request` | `envelope` | Child HAPPI envelope dispatched |

    Axiom 4 makes that document the contract, so the field follows the table
    rather than this implementation's local parameter name. This library
    previously emitted `child`, which PR #523 named as "a real divergence
    from the spec" and left untouched rather than change silently; closing it
    here brings the library to parity with the polyglot reference runtime.
    """
    return emit_event(envelope_id, "sub_request", envelope=dict(child))


def emit_completed(
    envelope_id: str, *, usage: dict[str, Any] | None = None, **extra: Any
) -> str:
    """Emit completed; mirror polyglot semantics — omit ``usage`` when None.

    The canonical polyglot runtime (`~/.hal/happi.md`) only includes the
    ``usage`` field on the wire when an explicit dict is passed. Stamping
    an empty ``{}`` on every completed creates spurious diffs against the
    polyglot byte-equivalence parity tests (H426). Callers that want a
    structured usage object continue to pass ``usage={...}`` and see no
    behavioural change.

    ``**extra`` rides after ``usage`` (the spec's Event schema allows
    additional properties). The chat adapters use it for ``stop_reason`` and
    ``message`` -- the provider's finish reason and the assembled assistant
    message -- on turns whose output the next turn must echo verbatim.
    """
    payload: dict[str, Any] = {} if usage is None else {"usage": dict(usage)}
    payload.update(extra)
    return emit_event(envelope_id, "completed", **payload)


def emit_error(envelope_id: str, message: str, *, code: str = "unknown") -> str:
    return emit_event(envelope_id, "error", code=code, message=message)


def emit_idr(
    envelope_id: str,
    *,
    sha256: str,
    cid: str | None = None,
    model_versions: list[str] | None = None,
    block_anchor: int | None = None,
) -> str:
    """Emit one v1.1 ``idr`` audit-terminator event.

    The ``v`` stamped on the wire is whatever ``PROTOCOL_VERSION`` resolves
    to at import time (the P2 feature flag).

    THE RULE: an event is stamped from its runtime's own version source, never
    from a literal. The cross-runtime parity test is what checks that both
    runtimes obey it.
    """
    idr_ref: dict[str, Any] = {
        "sha256": sha256,
        "cid": cid,
        "model_versions": list(model_versions or []),
        "block_anchor": block_anchor,
    }
    return emit_event(envelope_id, "idr", idr_ref=idr_ref)


def emit_context(
    envelope_id: str,
    *,
    sha256: str,
    predecessor_context: str | None = None,
    snapshot_ref: str | None = None,
    kind: str = "context-delta",
    model_versions: list[str] | None = None,
) -> str:
    """Emit one v1.1 ``context`` memory-chain event (the twin of ``idr``).

    ``context_ref.sha256`` is the CONTENT ADDRESS of a decision body
    (``context.compute_content_addr``), NOT the idr-style envelope+stream hash.
    The ``v`` stamp follows ``PROTOCOL_VERSION`` like every other event. Per HAL
    #429 emission is body-driven (via ``context.append``), never a
    ``flags.context`` auto-terminator. Build the ref with
    ``context.build_context_ref`` or pass an already-computed ``sha256`` here.
    """
    from lib.hal.happi.context import normalise_kind  # deferred: avoid package-init cycle


    context_ref: dict[str, Any] = {
        "sha256": sha256,
        "predecessor_context": predecessor_context,
        "snapshot_ref": snapshot_ref,
        "kind": normalise_kind(kind),
        "model_versions": list(model_versions or []),
    }
    return emit_event(envelope_id, "context", context_ref=context_ref)


def emit_continuity(
    envelope_id: str,
    *,
    sha256: str,
    kind: str = "continuity-commit",
    predecessor_continuity: str | None = None,
    settles: str | None = None,
    model_versions: list[str] | None = None,
) -> str:
    """Emit one v1.4 ``continuity`` event — a pre-registered falsifiable premise.

    ``continuity_ref.sha256`` is the CONTENT ADDRESS of the commitment body
    (``continuity.build_continuity_ref``), computed by the same single algorithm
    that addresses a context body.

    Unlike ``idr`` and ``context`` this is NOT a terminator: a commitment is
    emitted BEFORE the dispatch whose premise it registers, and that ordering IS
    the guarantee — a reader verifies from the stream alone that the premise and
    its falsifier predate the outcome.

    Fail-closed by delegation: ``kind`` is VALIDATED, not coerced, so an
    unrecognised kind raises ``ContinuityError`` rather than silently becoming a
    ``continuity-commit``. This is the deliberate asymmetry with ``emit_context``
    above, which normalises — see ``continuity`` module docstring. Prefer
    ``build_continuity_ref``, which additionally refuses a verdict that settles
    nothing; pass an already-computed ``sha256`` here.
    """
    from lib.hal.happi.continuity import validate_kind  # deferred: avoid package-init cycle

    continuity_ref: dict[str, Any] = {
        "sha256": sha256,
        "kind": validate_kind(kind),
        "predecessor_continuity": predecessor_continuity,
        "settles": settles,
        "model_versions": list(model_versions or []),
    }
    return emit_event(envelope_id, "continuity", continuity_ref=continuity_ref)


def emit_reasoning(
    envelope_id: str,
    *,
    sha256: str,
    precedes: str | None = None,
    model_versions: list[str] | None = None,
    signature: str | None = None,
) -> str:
    """Emit one v1.5 ``reasoning`` event — a trace as a DIAGNOSTIC SPECIMEN.

    ``reasoning_ref.sha256`` is the CONTENT ADDRESS of the trace body
    (``reasoning.build_reasoning_ref``), computed by the same single algorithm
    that addresses a context or continuity body.

    ``authoritative`` and ``role`` are STAMPED, not parameters. A trace is
    admissible as a specimen to be examined and inadmissible as evidence of
    cause, and the record carries that itself rather than leaving it to prose —
    so there is deliberately no way for a caller to say otherwise through this
    function. The canonical runtime refuses an envelope that tries.
    """
    reasoning_ref: dict[str, Any] = {
        "sha256": sha256,
        "authoritative": False,
        "role": "diagnostic",
        "precedes": precedes,
        "model_versions": list(model_versions or []),
        "signature": signature,
    }
    return emit_event(envelope_id, "reasoning", reasoning_ref=reasoning_ref)
