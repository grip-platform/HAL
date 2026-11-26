"""AGORA verdict reverse-parser + action mapping — fail-closed micro-shim.

The producer (``lib/agora_pr_review.py``) synthesises a 3-status verdict
(``MERGE`` / ``REQUEST_CHANGES`` / ``DEFER_CHANGES``) and POSTS it as a
rendered PR comment. ``review_pr`` parses the judge's *generation-time* output;
**nothing read the verdict back from the posted comment** — which is exactly
what an autonomous convergence-loop agent must do after pushing a fix and
waiting for the re-run AGORA review to land.

This module is that missing seam: a pure, fail-closed reverse-parser that maps
a *posted* AGORA comment body to a loop *action*.

Design (each invariant is a falsifiable contract, asserted by
``tests/test_agora_verdict_action.py``):

  * **Pure** — parses text the caller fetched. No network, no ``gh`` calls,
    no I/O. The caller (loop agent / future runner) does the
    ``gh api .../issues/<n>/comments`` fetch and passes the body in (DIP).
    This keeps the dangerous merge decision OUT of the parser and the parser
    trivially unit-testable with zero mocking.
  * **Fail-closed / monotone-toward-safe** — ambiguity (missing / empty /
    malformed / judge-unavailable / FAILED-header / unknown token) maps to
    ``ACTION_REQUEST_CHANGES`` (keep looping). The parser returns
    ``ACTION_MERGE`` *only* for an explicit, well-formed ``MERGE`` verdict
    blockquote, and ``ACTION_DEFER`` *only* for an explicit ``DEFER`` one.
    It can never *cause* a merge — at most it recommends "exit-loop,
    eligible for the merge gate". The merge gate itself (CI status, security
    checks, HAL-Council, actor, bootstrap) is the caller's explicit
    responsibility, kept out of this pure text-parser by design.
  * **Never raises** — a parser that can raise can fail-open via an
    unhandled exception at the call site. Every path returns a string.
  * **DRY** — the canonical verdict tokens are imported from
    ``agora_pr_review``, never re-literal'd. If the producer renames a token,
    this shim follows.

Canonical doctrine: GRIP ``rules/ci-review-driven-convergence.md`` (the
verdict -> action table) + ``rules/exogenous-anchor-law.md`` (the RSI-loop
admin-merge predicate + bootstrap exception). HAL rule:
``memory/feedback_agora_verdict_action_mapping.md``. HAL procedure:
``skills/agora-converge/SKILL.md``.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Resolve HAL repo root so ``lib.*`` imports work regardless of caller cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# DRY — single source of truth for the verdict strings. Do NOT redefine.
from lib.agora_pr_review import (  # noqa: E402
    VERDICT_MERGE,
    VERDICT_REQUEST_CHANGES,
    VERDICT_DEFER_CHANGES,
)

__all__ = [
    "ACTION_MERGE",
    "ACTION_REQUEST_CHANGES",
    "ACTION_DEFER",
    "VERDICT_MERGE",
    "VERDICT_REQUEST_CHANGES",
    "VERDICT_DEFER_CHANGES",
    "parse_posted_verdict",
    "verdict_to_action",
    "action_from_comment",
]

# ---------------------------------------------------------------------------
# Action constants — what the convergence loop should DO.
# ---------------------------------------------------------------------------
ACTION_MERGE = "merge"                       # exit loop; eligible for the merge gate
ACTION_REQUEST_CHANGES = "request_changes"   # continue the fix loop
ACTION_DEFER = "defer"                        # exit loop; ensure follow-up issue

# The fail-closed verdict — every ambiguous parse resolves here. Mirrors the
# producer's own ``VERDICT_JUDGE_UNAVAILABLE_FALLBACK`` contract.
_FAIL_CLOSED_VERDICT = VERDICT_REQUEST_CHANGES

# The header that opens a VALID posted review. A validity-FAILED review opens
# with ``## AGORA Council Review — FAILED`` and carries NO verdict block, so it
# must never be read as a real verdict.
_FAILED_HEADER_MARKER = "## AGORA Council Review — FAILED"

# NEW regex (do NOT reuse the producer's ``_VERDICT_EXTRACTION_RE`` — that one
# matches the GENERATION form ``### Verdict\n[TOKEN]``). This matches the
# RENDERED blockquote the producer POSTS:
#     > **Verdict: MERGE**
#     > **Verdict: REQUEST CHANGES**   (SPACE form — note: not REQUEST_CHANGES)
#     > **Verdict: DEFER CHANGES**
# Anchored on ``> **Verdict:`` so it cannot pick the token out of prose
# elsewhere in the comment. Accepts BOTH space and underscore between words for
# robustness. The judge-unavailable blockquote
# (``> **Verdict: Judge unavailable — manual review required**``) does NOT
# match — falling through to fail-closed, which is the correct outcome.
_POSTED_VERDICT_RE = re.compile(
    r"^>\s*\*\*Verdict:\s*"
    r"(MERGE|REQUEST[\s_]+CHANGES|DEFER[\s_]+CHANGES)\b",
    re.MULTILINE | re.IGNORECASE,
)

# Map a normalised (underscore, upper) parsed token to the canonical verdict.
_NORMALISED_TO_VERDICT = {
    "MERGE": VERDICT_MERGE,
    "REQUEST_CHANGES": VERDICT_REQUEST_CHANGES,
    "DEFER_CHANGES": VERDICT_DEFER_CHANGES,
}

# Map a canonical verdict to a loop action. Only MERGE / DEFER get a
# non-default action; everything else (incl. the fail-closed verdict) keeps
# looping. This is the monotone-toward-safe table.
_VERDICT_TO_ACTION = {
    VERDICT_MERGE: ACTION_MERGE,
    VERDICT_DEFER_CHANGES: ACTION_DEFER,
    VERDICT_REQUEST_CHANGES: ACTION_REQUEST_CHANGES,
}


def parse_posted_verdict(comment_body: str | None) -> str:
    """Extract the canonical verdict token from a RENDERED AGORA PR comment.

    Reads the ``> **Verdict: ...**`` blockquote (rendered, space-separated
    form) and normalises to the canonical underscore token.

    FAIL-CLOSED — returns ``VERDICT_REQUEST_CHANGES`` when ``comment_body`` is
    empty/None, carries the FAILED header, has no verdict blockquote, reads
    "Judge unavailable ...", or carries an unrecognised token. Never returns
    MERGE/DEFER on ambiguity. NEVER raises.
    """
    if not comment_body or not comment_body.strip():
        return _FAIL_CLOSED_VERDICT

    # A validity-FAILED review has no verdict block — never read it as a
    # verdict. (The regex would also miss it, but this is explicit + cheap.)
    if _FAILED_HEADER_MARKER in comment_body:
        return _FAIL_CLOSED_VERDICT

    match = _POSTED_VERDICT_RE.search(comment_body)
    if match is None:
        # No verdict blockquote (includes the "Judge unavailable" banner,
        # which does not match the token alternation) -> fail-closed.
        return _FAIL_CLOSED_VERDICT

    # Normalise: collapse internal whitespace to a single underscore, upper.
    raw = match.group(1)
    normalised = re.sub(r"[\s_]+", "_", raw.strip()).upper()
    return _NORMALISED_TO_VERDICT.get(normalised, _FAIL_CLOSED_VERDICT)


def verdict_to_action(verdict: str | None) -> str:
    """Map a canonical verdict token to a loop action.

    Unknown / None / fail-closed token -> ``ACTION_REQUEST_CHANGES`` (keep
    looping). Only an explicit ``MERGE`` -> ``ACTION_MERGE`` and an explicit
    ``DEFER_CHANGES`` -> ``ACTION_DEFER``. NEVER raises.
    """
    return _VERDICT_TO_ACTION.get(verdict or "", ACTION_REQUEST_CHANGES)


def action_from_comment(comment_body: str | None) -> str:
    """One-shot ``parse_posted_verdict`` then ``verdict_to_action``.

    The single call a loop agent / future runner makes on a fetched AGORA
    comment body. Fail-closed and never-raising end to end.
    """
    return verdict_to_action(parse_posted_verdict(comment_body))
