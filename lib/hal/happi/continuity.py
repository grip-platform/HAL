"""HAPPI v1.4 continuity / pre-registered-commitment support (lib side).

Implements the ``continuity`` event — the third record kind, alongside ``idr``
(what was DECIDED) and ``context`` (what was BELIEVED). A continuity record
carries what was PREMISED: a falsifiable claim and the observation that would
refute it, content-addressed the same way a context body is.

Unlike ``idr`` and ``context``, a continuity event is NOT a terminator. A
commitment is emitted BEFORE the dispatch whose premise it registers, because
the whole guarantee is ordering — a reader verifies from the stream alone that
the stated premise and its falsifier predate the outcome, so the reasoning
cannot have been fitted to the result after the fact. That is the anti-HARKing
property (``~/.claude/CONGRUENCE.md``); losing the ordering loses the point.

This module is the lib-side twin of the canonical polyglot runtime's
``emit_continuity`` / ``cmd_continuity_append``
(``~/.hal/happi-sources/04-python-runtime.py``). It brings the lib to v1.4
parity, which it had never reached: before this module the lib had zero
continuity support while the polyglot had carried it since v1.4.

FAIL-CLOSED, deliberately — and this is the one place the lib does NOT mirror
its own ``context.py``. ``context.normalise_kind`` coerces an unrecognised kind
to ``context-delta`` (fails OPEN, HAL #433); the polyglot's continuity handler
rejects an unrecognised kind outright. A malformed continuity record is worse
than none: it is a premise nobody can join back to an outcome, and silently
coercing it to ``continuity-commit`` would manufacture a commitment the caller
never made. So ``validate_kind`` RAISES where ``normalise_kind`` coerces, and
a verdict that settles nothing is refused rather than written through.

The content address is deliberately NOT reimplemented here — it is imported
from ``context``. One algorithm, one definition: the address must stay
byte-identical across the polyglot runtime, this lib, and GRIP
``lib/precog/idr.py::content_addr``, and a second copy is a second thing to
drift. The canonical runtime does the same (one ``_content_addr`` serving both
the context and continuity emitters).
"""
from __future__ import annotations

from typing import Any

# ONE address algorithm, imported rather than duplicated — see module docstring.
from lib.hal.happi.context import compute_content_addr

# Mirrors the polyglot ``_CONTINUITY_KINDS``. A commit pre-registers a premise;
# a verdict settles one. There is no third kind, and an unknown one is refused.
_CONTINUITY_KINDS = ("continuity-commit", "continuity-verdict")

_COMMIT_KIND = "continuity-commit"
_VERDICT_KIND = "continuity-verdict"


class ContinuityError(ValueError):
    """A continuity record was refused before reaching the wire.

    Raised rather than coerced: see the fail-closed note in the module
    docstring. Mirrors ``EnvelopeError``/``DecisionError`` so callers can
    catch one exception family across the lib.
    """


def validate_kind(kind: str) -> str:
    """Return ``kind`` if it is a recognised continuity kind, else RAISE.

    The deliberate counterpart to ``context.normalise_kind``, which coerces.
    Coercion is right for a context delta (the safe default is a real, honest
    kind) and wrong for a continuity record (there is no safe default premise —
    inventing one fabricates a commitment the caller never made).
    """
    if kind not in _CONTINUITY_KINDS:
        raise ContinuityError(
            "continuity kind must be one of "
            + ", ".join(_CONTINUITY_KINDS)
            + f", got {kind!r}"
        )
    return kind


def build_continuity_ref(
    body: dict[str, Any],
    *,
    kind: str = _COMMIT_KIND,
    predecessor_continuity: str | None = None,
    settles: str | None = None,
    model_versions: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble a ``continuity_ref`` for ``body``.

    ``body`` is the commitment itself: for a commit, the premise being
    pre-registered (``{claim, falsifier, deps, severity, ...}``); for a verdict,
    the settlement (``{verdict, axis, evidence, verifier, ...}``).

    As with ``context``, the address is over the BODY, not over the metadata:
    the same premise threaded to a different predecessor keeps the predecessor
    out of the dedup coordinate. The metadata records chain position; the
    ``sha256`` records identity.

    Two fail-closed refusals, matching the canonical runtime exactly:

    * an unrecognised ``kind`` raises (``validate_kind``);
    * a ``continuity-verdict`` with no ``settles`` raises — a settlement that
      settles nothing cannot be joined back to its premise, so it can never be
      checked, and an unverifiable settlement record is strictly worse than no
      record at all.

    Both raise BEFORE any ref is built, so a refused record cannot reach the
    wire through this path.
    """
    validate_kind(kind)
    if kind == _VERDICT_KIND and not settles:
        raise ContinuityError(
            "a continuity-verdict must set 'settles' to the content address of "
            "the commitment it settles"
        )
    return {
        "sha256": compute_content_addr(body),
        "kind": kind,
        "predecessor_continuity": predecessor_continuity,
        "settles": settles,
        "model_versions": list(model_versions or []),
    }
