"""Provenance taxonomy for osa dispatch.

Every :func:`lib.hal.osa.osa` call records a ``provenance`` label that
describes **where the dispatch originated**. The label is enforced as an
enum (not a free-form string) so downstream policy — MCP allowlists,
capability-token requirements, audit queries — can branch on it without
fuzzy string matching.

The five values map onto the council-mandated taxonomy::

    HUMAN_TYPED      — direct user shell invocation (`hal osa ...`)
    TOOL_CHAIN       — agent tool call inside a Claude/LLM turn
    MCP_REMOTE       — cross-host call over the MCP transport
    AUTONOMOUS_LOOP  — ultrado / rsi / sprint-loop background worker
    YOLO             — yolo-mode marker active; halting gates bypassed

:data:`ProvenanceType.LOCAL` is an additional catch-all used when the
dispatching code has no reason to distinguish origin (e.g. legacy call
sites not yet migrated). It will eventually be removed in β-4 once every
call site sets an explicit value.
"""
from __future__ import annotations

from enum import Enum

__all__ = [
    "ProvenanceType",
    "parse_provenance",
    "UnknownProvenance",
]


class UnknownProvenance(ValueError):
    """Raised when a provenance string does not match any enum value."""


class ProvenanceType(str, Enum):
    """Taxonomy of osa dispatch origins.

    Inherits from ``str`` so that JSON serialisation and audit-log rendering
    produce the plain lowercase string without bespoke encoders.
    """
    HUMAN_TYPED = "human_typed"
    TOOL_CHAIN = "tool_chain"
    MCP_REMOTE = "mcp_remote"
    AUTONOMOUS_LOOP = "autonomous_loop"
    YOLO = "yolo"
    LOCAL = "local"
    USER_REJECTED = "user_rejected"


def parse_provenance(s: str) -> ProvenanceType:
    """Return the :class:`ProvenanceType` matching ``s`` or raise.

    Matching is case-insensitive and accepts either the enum *name*
    (``HUMAN_TYPED``) or its *value* (``human_typed``). Unknown strings
    raise :class:`UnknownProvenance` — never silently downgrade to LOCAL.
    """
    if not s:
        raise UnknownProvenance("empty provenance string")
    needle = s.strip().lower()
    for member in ProvenanceType:
        if needle == member.value or needle == member.name.lower():
            return member
    raise UnknownProvenance(
        f"unknown provenance {s!r}; "
        f"expected one of {[m.value for m in ProvenanceType]}"
    )
