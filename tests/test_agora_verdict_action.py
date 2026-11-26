"""Goodhart-resistant tests for ``lib/agora_verdict`` — the verdict reverse-parser
+ action mapping shim.

Distinct from ``tests/test_agora_verdict.py`` (which guards the PRODUCER's
generation-time parse + render). This file guards the CONSUMER seam: reading the
verdict BACK from a *posted* AGORA comment and mapping it to a loop action.

Each test fails under an obvious mutation of the shim:

  * parse each of the 3 verdicts (space + underscore forms),
  * fail-closed on empty / malformed / judge-unavailable / FAILED-header /
    unknown-token -> ``ACTION_REQUEST_CHANGES`` (NEVER auto-merge),
  * anti-fail-open anchor: prose containing "MERGE" but no verdict blockquote
    must NOT map to ``ACTION_MERGE``,
  * the MERGE action is parser-recommendation only (it returns ``ACTION_MERGE``,
    which is NOT an unconditional merge — the merge predicate lives at the call
    site),
  * round-trip coupling to the REAL producer render (``AgoraReview.to_markdown``)
    so a producer render-format change turns this red.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Resolve HAL repo root so ``lib.*`` imports work regardless of pytest cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.environ.setdefault("HAL_HOME", str(_REPO_ROOT))

from lib.agora_verdict import (  # noqa: E402
    ACTION_MERGE,
    ACTION_REQUEST_CHANGES,
    ACTION_DEFER,
    VERDICT_MERGE,
    VERDICT_REQUEST_CHANGES,
    VERDICT_DEFER_CHANGES,
    parse_posted_verdict,
    verdict_to_action,
    action_from_comment,
)
from lib.agora_pr_review import AgoraReview  # noqa: E402


# A realistic rendered comment scaffold around the verdict blockquote, so the
# parser is exercised against prose-bearing input (not just the bare line).
def _posted_comment(verdict_line: str) -> str:
    return (
        "## AGORA Council Review\n\n"
        f"{verdict_line}\n"
        "> *Recommended by LLM-AS-JUDGE (Google Gemini). This verdict is "
        "advisory; required CI checks still gate merge.*\n\n"
        "### TL;DR\n\nDoes a thing.\n\n"
        "### ELI5\n\nLike tidying one drawer.\n\n"
        "---\n\n"
        "### 1. What this PR actually does\n\nPlain words.\n\n"
        "### 3. Technical concerns + suggestions\n\nConcern at lib/foo.py:2.\n"
    )


# ---------------------------------------------------------------------------
# 1-4 — PARSE EACH VERDICT (space + underscore forms)
# ---------------------------------------------------------------------------

def test_merge_verdict_maps_to_merge_action():
    body = _posted_comment("> **Verdict: MERGE**")
    assert parse_posted_verdict(body) == VERDICT_MERGE
    assert action_from_comment(body) == ACTION_MERGE


def test_request_changes_space_form_maps_to_request_changes():
    """The POSTED comment renders the space form 'REQUEST CHANGES'."""
    body = _posted_comment("> **Verdict: REQUEST CHANGES**")
    assert parse_posted_verdict(body) == VERDICT_REQUEST_CHANGES
    assert action_from_comment(body) == ACTION_REQUEST_CHANGES


def test_defer_changes_space_form_maps_to_defer_action():
    body = _posted_comment("> **Verdict: DEFER CHANGES**")
    assert parse_posted_verdict(body) == VERDICT_DEFER_CHANGES
    assert action_from_comment(body) == ACTION_DEFER


def test_request_changes_underscore_form_also_parses():
    """Robustness: the underscore form parses identically to the space form."""
    body = _posted_comment("> **Verdict: REQUEST_CHANGES**")
    assert parse_posted_verdict(body) == VERDICT_REQUEST_CHANGES
    assert action_from_comment(body) == ACTION_REQUEST_CHANGES


def test_defer_changes_underscore_form_also_parses():
    body = _posted_comment("> **Verdict: DEFER_CHANGES**")
    assert parse_posted_verdict(body) == VERDICT_DEFER_CHANGES
    assert action_from_comment(body) == ACTION_DEFER


# ---------------------------------------------------------------------------
# 5-8 — FAIL-CLOSED (every ambiguity -> REQUEST_CHANGES, never MERGE/DEFER)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("body", ["", "   \n\t \n", None])
def test_empty_or_none_fails_closed(body):
    assert parse_posted_verdict(body) == VERDICT_REQUEST_CHANGES
    assert action_from_comment(body) == ACTION_REQUEST_CHANGES


def test_failed_header_fails_closed():
    """A validity-FAILED review (no verdict block) must NOT read as a verdict."""
    body = (
        "## AGORA Council Review — FAILED\n\n"
        "This review was flagged as a likely hallucination.\n"
    )
    assert parse_posted_verdict(body) == VERDICT_REQUEST_CHANGES
    assert action_from_comment(body) == ACTION_REQUEST_CHANGES


def test_judge_unavailable_banner_fails_closed():
    """The judge-unavailable banner is NOT a real verdict -> REQUEST_CHANGES."""
    body = (
        "## AGORA Council Review\n\n"
        "> **Verdict: Judge unavailable — manual review required**\n"
        "> *This verdict is advisory; required CI checks still gate merge.*\n\n"
        "### 1. What this PR actually does\n\nPlain words.\n"
    )
    assert parse_posted_verdict(body) == VERDICT_REQUEST_CHANGES
    assert action_from_comment(body) == ACTION_REQUEST_CHANGES


def test_no_verdict_blockquote_fails_closed():
    body = (
        "## AGORA Council Review\n\n"
        "### 1. What this PR actually does\n\nIt does a thing.\n"
    )
    assert parse_posted_verdict(body) == VERDICT_REQUEST_CHANGES
    assert action_from_comment(body) == ACTION_REQUEST_CHANGES


def test_unknown_verdict_token_fails_closed():
    """A blockquote with a non-canonical token -> fail-closed REQUEST_CHANGES."""
    body = _posted_comment("> **Verdict: APPROVE**")
    assert parse_posted_verdict(body) == VERDICT_REQUEST_CHANGES
    assert action_from_comment(body) == ACTION_REQUEST_CHANGES


# ---------------------------------------------------------------------------
# 9 — ANTI-FAIL-OPEN ANCHOR (prose 'MERGE' without the blockquote anchor)
# ---------------------------------------------------------------------------

def test_merge_in_prose_without_blockquote_does_not_merge():
    """Proves the ``> **Verdict:`` anchor.

    A body that mentions MERGE in free prose but has NO verdict blockquote must
    map to ``ACTION_REQUEST_CHANGES``. A mutation dropping the ``> **Verdict:``
    anchor (e.g. matching a bare token anywhere) would flip this to MERGE and
    fail the test — exactly the fail-open this guards against.
    """
    body = (
        "## AGORA Council Review\n\n"
        "### 3. Technical concerns + suggestions\n\n"
        "Speaker C suggests we could MERGE the two helpers, but recommends "
        "addressing the bound first. The judge would never MERGE this as-is.\n"
    )
    assert "MERGE" in body  # the trap is genuinely present
    assert parse_posted_verdict(body) == VERDICT_REQUEST_CHANGES
    assert action_from_comment(body) == ACTION_REQUEST_CHANGES


def test_verdict_not_picked_from_indented_non_blockquote_line():
    """A non-blockquote 'Verdict: MERGE' (no leading '>') must not match."""
    body = (
        "## AGORA Council Review\n\n"
        "The recommended Verdict: MERGE was discussed but not adopted.\n"
    )
    assert parse_posted_verdict(body) == VERDICT_REQUEST_CHANGES


# ---------------------------------------------------------------------------
# verdict_to_action unit contract
# ---------------------------------------------------------------------------

def test_verdict_to_action_only_merge_maps_to_merge():
    """The monotone-toward-safe table: ONLY MERGE -> ACTION_MERGE."""
    assert verdict_to_action(VERDICT_MERGE) == ACTION_MERGE
    assert verdict_to_action(VERDICT_REQUEST_CHANGES) == ACTION_REQUEST_CHANGES
    assert verdict_to_action(VERDICT_DEFER_CHANGES) == ACTION_DEFER


def test_verdict_to_action_unknown_and_none_keep_looping():
    assert verdict_to_action("APPROVE") == ACTION_REQUEST_CHANGES
    assert verdict_to_action("") == ACTION_REQUEST_CHANGES
    assert verdict_to_action(None) == ACTION_REQUEST_CHANGES


def test_merge_action_is_recommendation_not_unconditional_merge():
    """ACTION_MERGE is parser-level 'exit-loop, eligible' — NOT a merge.

    The shim deliberately stops at returning an action string; it performs no
    gh call, no merge, and exposes no merge function. The dangerous merge
    decision (the 5-condition RSI predicate) lives at the call site. This test
    locks that boundary: a mutation that made the shim itself merge (e.g. added
    a ``merge_pr`` that shells out) would have a callable to assert against; the
    public surface intentionally has none.
    """
    import lib.agora_verdict as shim

    # The MERGE path yields only a string verdict/action — nothing executable.
    assert action_from_comment(_posted_comment("> **Verdict: MERGE**")) == ACTION_MERGE
    assert isinstance(ACTION_MERGE, str)
    # No merge-causing function is exported (monotone-toward-safe boundary).
    public = set(shim.__all__)
    for forbidden in ("merge_pr", "do_merge", "admin_merge", "gh_merge"):
        assert forbidden not in public
        assert not hasattr(shim, forbidden)


# ---------------------------------------------------------------------------
# 10 — ROUND-TRIP: couple to the REAL producer render
# ---------------------------------------------------------------------------

def _valid_review(verdict: str, **kw) -> AgoraReview:
    base = dict(
        pr_number=1,
        repo="CodeTonight-SA/HAL",
        plain_language="It does a thing in plain words. Imagine a tidy drawer.",
        quality="Quality is fine. lib/foo.py:1 looks aligned.",
        technical="Concrete concern at lib/foo.py:2 about a bound.",
        speakers_responded=3,
        speakers_total=3,
        verdict=verdict,
        tldr="Short scannable summary.",
        eli5="Like a tidy cupboard.",
        judge_reasoning='Driven by: > "concern at lib/foo.py:2" (Speaker C).',
        judge_provider="Google Gemini 2.5 Flash (judge — exogenous to speakers)",
        judge_valid=True,
    )
    base.update(kw)
    return AgoraReview(**base)


@pytest.mark.parametrize(
    "verdict,expected_action",
    [
        (VERDICT_MERGE, ACTION_MERGE),
        (VERDICT_REQUEST_CHANGES, ACTION_REQUEST_CHANGES),
        (VERDICT_DEFER_CHANGES, ACTION_DEFER),
    ],
)
def test_roundtrip_real_producer_render_to_action(verdict, expected_action):
    """Render via the REAL producer, parse it BACK, assert the action matches.

    This is the Goodhart-proof coupling: if the producer's
    ``_render_verdict_block`` changes the rendered format, the reverse-parser
    no longer matches and this test goes red — surfacing the drift instead of
    silently mis-reading verdicts.
    """
    # DEFER needs a defer_issue_url for the footer line; irrelevant to parse.
    extra = {"defer_issue_url": "https://github.com/x/y/issues/9"} if verdict == VERDICT_DEFER_CHANGES else {}
    rendered = _valid_review(verdict, **extra).to_markdown()
    assert parse_posted_verdict(rendered) == verdict
    assert action_from_comment(rendered) == expected_action


def test_roundtrip_judge_unavailable_render_fails_closed():
    """A judge-unavailable render (no real verdict) round-trips to REQUEST_CHANGES."""
    review = AgoraReview(
        pr_number=2,
        repo="CodeTonight-SA/HAL",
        plain_language="Plain words here. Imagine a drawer.",
        quality="Quality note about lib/foo.py:1.",
        technical="Concern at lib/foo.py:2.",
        speakers_responded=3,
        speakers_total=3,
        verdict=VERDICT_REQUEST_CHANGES,  # sentinel
        judge_valid=False,
        judge_failure_reason="judge returned empty response",
    )
    rendered = review.to_markdown()
    assert "Judge unavailable" in rendered  # producer really rendered the banner
    assert parse_posted_verdict(rendered) == VERDICT_REQUEST_CHANGES
    assert action_from_comment(rendered) == ACTION_REQUEST_CHANGES
