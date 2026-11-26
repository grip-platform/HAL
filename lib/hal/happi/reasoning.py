"""HAPPI v1.5 reasoning-trace support (lib side).

Implements the ``reasoning`` event — the fourth record kind, alongside ``idr``
(what was DECIDED), ``context`` (what was BELIEVED) and ``continuity`` (what was
PREMISED). A reasoning record carries what was THOUGHT: a trace, content-
addressed exactly as the other bodies are.

The governing constraint, and the only reason this event type is safe to add:

    A reasoning trace is admissible as a DIAGNOSTIC SPECIMEN to be examined,
    and inadmissible as EVIDENCE OF CAUSE.

``CONGRUENCE.md`` D1 forbids narrated chain-of-thought as the causal unit, and
``rules/prove-it-claims-canonical.md`` ranks self-reports the least reliable
claim class there is. A trace may APPEAR and may NEVER be believed. So the fence
lives on the wire rather than in prose: every record is stamped
``authoritative=False`` and ``role="diagnostic"``, and neither is caller-
settable — there is no parameter for them here, by design. A field a caller
could set is a field a caller could claim authority in.

``settles`` is likewise absent. Settling is what a ``continuity-verdict`` does
to a pre-registered commitment; a trace settles nothing, and accepting the field
would let a specimen pose as a settlement.

This module is the lib-side twin of the canonical polyglot runtime's
``emit_reasoning`` / ``cmd_reasoning_append``
(``~/.hal/happi-sources/04-python-runtime.py``), added in the same change so the
two runtimes do not diverge the way the lib did before v1.4 — three minor
versions behind with a green suite, because ``build-happi``'s version gate reads
only the ``happi-sources/`` layers and is structurally blind to ``lib/``.

The content address is deliberately NOT reimplemented here — it is imported from
``context``. One algorithm, one definition: the address must stay byte-identical
across the polyglot runtime, this lib, and GRIP ``lib/precog/idr.py``, and a
second copy is a second thing to drift.

Note the asymmetry with ``continuity``: that module needs ``validate_kind``
because a commitment has two legitimate kinds and no safe default. A reasoning
record has exactly ONE thing it may be, so there is no vocabulary to validate —
the role is a constant, which is why this module exposes no kind machinery. That
absence is the design, not an omission.
"""
from __future__ import annotations

from typing import Any

# ONE address algorithm, imported rather than duplicated — see module docstring.
from lib.hal.happi.context import compute_content_addr

# Stamped on every record. Not a parameter anywhere in this module: see the
# module docstring on why neither field is caller-settable.
REASONING_ROLE = "diagnostic"
REASONING_AUTHORITATIVE = False


class ReasoningError(ValueError):
    """A reasoning record was refused before reaching the wire.

    Mirrors ``ContinuityError``/``EnvelopeError``/``DecisionError`` so callers
    can catch one exception family across the lib.
    """


def build_reasoning_ref(
    body: dict[str, Any],
    *,
    precedes: str | None = None,
    model_versions: list[str] | None = None,
    signature: str | None = None,
) -> dict[str, Any]:
    """Assemble a ``reasoning_ref`` for ``body``.

    ``body`` is the trace itself. As with ``context`` and ``continuity``, the
    address is over the BODY and not over the metadata: the same trace linked to
    a different ``precedes`` keeps that link out of the dedup coordinate. The
    metadata records position; the ``sha256`` records identity.

    ``precedes`` is a TEMPORAL-POSITION link and nothing stronger. It says this
    trace was recorded before the record at that address. It does not say the
    trace caused it, and a consumer treating it as causal has re-imported the
    exact error the ``authoritative=False`` stamp exists to prevent.
    """
    return {
        "sha256": compute_content_addr(body),
        "authoritative": REASONING_AUTHORITATIVE,
        "role": REASONING_ROLE,
        "precedes": precedes,
        "model_versions": list(model_versions or []),
        "signature": signature,
    }
