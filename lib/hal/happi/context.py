"""HAPPI v1.1 context / memory-chain event support (lib side).

Implements the ``context`` event — the structural twin of ``idr``. Where ``idr``
chains a hash over (envelope + event stream), ``context`` carries a CONTENT
ADDRESS of a decision body: ``sha256`` of the canonical JSON of the body
EXCLUDING the volatile ``id``/``ts``/``audit`` fields. Two records with
identical semantic content share ONE address — the dedup/merkle coordinate of a
signed memory-chain (design: GRIP ``drafts/happi-context-event-memory-chain-design.md``).

Byte-identical to the canonical polyglot runtime
(``~/.hal/happi-sources/04-python-runtime.py::_content_addr``) AND to GRIP
``lib/precog/idr.py::content_addr`` for the same body — the cross-runtime
invariant the chain depends on. Unlike ``audit.py``, this needs no per-dispatch
buffer or session: a content address is a pure function of the body.

Per HAL #429, the stdlib reference runtime emits ``context`` ONLY via the
explicit ``context.append`` cmd — a generic dispatch has no designated decision
body, and addressing the whole envelope would fold dispatch flags into the
address. This lib mirrors that: emission is body-driven, never flag-driven.

``context_head`` is the thin chain pointer on the envelope ``ctx`` dict
(``ctx.context_head`` = the latest context node id). Read it with
``context_head(envelope)``. It is a forward-compatible passthrough field — an
older runtime that does not know it carries it through ``to_dict`` unchanged.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from lib.hal.happi.decision import (
    Decision,
    DecisionError,
    from_model_dict,
)

# Volatile fields excluded from a content address — mirrors the polyglot runtime
# and GRIP lib/precog/idr.py::_CONTENT_ADDR_EXCLUDE (cross-runtime invariant).
# Any field NOT listed here IS addressed (fail-safe: a new semantic field is part
# of the identity by default, never silently dropped).
_CONTENT_ADDR_EXCLUDE = ("id", "ts", "audit")

_VALID_KINDS = ("context-delta", "context-snapshot", "context-supersede")

# The body slot that carries the agreed Decision shape (v1.3). Mirrors GRIP
# ``lib/decision_record.py::embed_in_idr_ref`` which writes ``out["decision"]`` —
# so the memory-chain node and the IDR audit record carry the SAME field, the SAME
# shape, at two callsites with no divergence (hypothesis H-ONE-SCHEMA).
_DECISION_SLOT = "decision"


def normalise_kind(kind: str) -> str:
    """Coerce an unrecognised context kind to ``context-delta`` (the safe
    default). Shared by ``build_context_ref`` and ``events.emit_context`` so a
    malformed ``kind`` can never reach the wire (HAL #433)."""
    return kind if kind in _VALID_KINDS else "context-delta"



def compute_content_addr(body: dict[str, Any]) -> str:
    """Content address of a decision body — ``"sha256:<hex>"`` over canonical
    JSON (sorted keys, ``ensure_ascii=False``) EXCLUDING the volatile
    ``id``/``ts``/``audit`` fields. Pure; byte-identical to the polyglot
    ``_content_addr`` and GRIP ``content_addr`` for the same body.
    """
    addressed = {k: v for k, v in body.items() if k not in _CONTENT_ADDR_EXCLUDE}
    canonical = json.dumps(addressed, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_body_decision(body: dict[str, Any]) -> Decision | None:
    """Validate the agreed Decision shape carried in a context body (v1.3).

    The memory-chain node carries its delegated decision in ``body["decision"]`` —
    the SAME slot, the SAME shape that GRIP ``embed_in_idr_ref`` writes on the IDR
    audit record. This runs that slot through the ONE shared validator
    (SMT surface-graft gate → tiered honesty check), returning the parsed
    ``Decision`` on success and raising ``DecisionError`` on a malformed one.

    Returns ``None`` when the body has no ``decision`` slot — a context body need
    not carry a Decision (e.g. a raw memory delta). Only a PRESENT decision is
    validated, so this never breaks an existing decision-less body. The validation
    is a gate BEFORE content-addressing; it does NOT change the address (the
    address is over the whole body, byte-identical with or without this call).
    """
    if not hasattr(body, "get"):
        raise DecisionError("context body must be a JSON object to carry a decision")
    slot = body.get(_DECISION_SLOT)
    if slot is None:
        return None
    if isinstance(slot, Decision):
        # already a Decision instance — re-validate (idempotent, fail-closed)
        from lib.hal.happi.decision import validate as _v
        return _v(slot)
    if not hasattr(slot, "get"):
        raise DecisionError(
            f"context body '{_DECISION_SLOT}' must be a JSON object (the Decision "
            f"shape), got {type(slot).__name__}")
    return from_model_dict(slot)


def build_context_ref(
    body: dict[str, Any],
    *,
    predecessor_context: str | None = None,
    snapshot_ref: str | None = None,
    kind: str = "context-delta",
    model_versions: list[str] | None = None,
    validate_decision: bool = False,
) -> dict[str, Any]:
    """Assemble a ``context_ref`` for ``body``.

    The address is over the body, NOT over these metadata fields: the same
    belief threaded to a different predecessor keeps the predecessor OUT of the
    dedup coordinate (the metadata records chain position; the ``sha256``
    records identity). An unrecognised ``kind`` falls back to ``context-delta``.

    ``validate_decision`` (v1.3, opt-in, default OFF for byte-identical
    back-compat): when ``True``, the body's ``decision`` slot — if present — is run
    through the ONE shared Decision validator (``validate_body_decision``) BEFORE
    addressing. A malformed decision raises ``DecisionError`` and no ref is built.
    The address itself is unchanged either way; the flag adds a gate, not a field.
    """
    if validate_decision:
        validate_body_decision(body)
    return {
        "sha256": compute_content_addr(body),
        "predecessor_context": predecessor_context,
        "snapshot_ref": snapshot_ref,
        "kind": normalise_kind(kind),
        "model_versions": list(model_versions or []),
    }


def context_head(envelope: Any) -> str | None:
    """Resolve the thin chain pointer ``ctx.context_head`` (the latest context
    node id) from a parsed ``HappiEnvelope`` or a raw envelope dict.

    Returns ``None`` when absent. Forward-compatible: an older runtime that does
    not understand ``context_head`` still carries it through ``ctx`` unchanged,
    so this resolver reads it whenever a newer writer set it.
    """
    ctx = getattr(envelope, "ctx", None)
    if ctx is None and hasattr(envelope, "get"):
        ctx = envelope.get("ctx")
    if not hasattr(ctx, "get"):
        return None
    head = ctx.get("context_head")
    return head if isinstance(head, str) and head else None
