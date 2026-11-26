"""Goodhart-proof tests for AGORA premise-provenance (HAL's own council code).

Guards the soundness fix for the "an exogenous verdict launders a corrupt
premise" gap: the AGORA judge synthesises a verdict from 3 speakers but never
sees the diff/premise itself, so a wrong or truncated diff was laundered into a
verdict with nothing recording WHAT was graded. The fix renders a mechanical
premise-provenance citation (title / file count / byte size), computed from the
real fetched diff metadata -- NOT the LLM's self-report -- into the verdict.

Each test fails under an obvious mutation of the code it guards:

  P1 -- the RENDERED verdict output carries the premise with the SPECIFIC values
       of a known 2-file / known-byte fixture. Removing the footer render, or
       miscounting the files/bytes, fails.
  P2 -- ``_premise_provenance`` returns the exact citation string, and de-aliases
       the file set (basename aliases do NOT inflate the count).
  P3 -- an empty/absent premise is handled gracefully: no crash, and the footer
       omits the "Premise reviewed" line entirely (backwards-compatible).
  P4 -- the FAILED (invalid-review) banner also carries the premise, so a corrupt
       premise is visible exactly when the review is rejected as a hallucination.
  P5 -- WIRING: the real orchestrator (``review_pr``) actually populates the
       field from the fetched diff, network-free. Fails if the orchestrator
       stops passing ``premise_provenance``.

Goodhart guards: assertions check actual rendered values (the file count string,
the byte count, the title, the absence of a specific line), never call counts.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Resolve HAL repo root so ``lib.*`` imports work regardless of pytest cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.environ.setdefault("HAL_HOME", str(_REPO_ROOT))

from lib.agora_pr_review import (  # noqa: E402
    AgoraReview,
    VERDICT_MERGE,
    _premise_provenance,
    _real_diff_files,
)
import lib.agora_pr_review as agora  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers (mirror the idiom in tests/test_agora_verdict.py)
# ---------------------------------------------------------------------------

def _valid_review(**kw) -> AgoraReview:
    """A publishable review with a valid judge verdict; override via **kw."""
    base = dict(
        pr_number=7,
        repo="CodeTonight-SA/HAL",
        plain_language="It does a thing in plain words. Imagine a tidy drawer.",
        quality="Quality is fine. lib/foo.py:1 looks aligned.",
        technical="Concrete concern at lib/bar.py:2 about a bound.",
        speakers_responded=3,
        speakers_total=3,
        verdict=VERDICT_MERGE,
        tldr="Short scannable summary.",
        eli5="Like a tidy cupboard.",
        judge_reasoning='Driven by: > "concern at lib/bar.py:2" (Speaker C).',
        judge_provider="Google Gemini 3.1 Pro (judge -- exogenous to speakers)",
        judge_valid=True,
    )
    base.update(kw)
    return AgoraReview(**base)


# ---------------------------------------------------------------------------
# P1 -- rendered verdict carries the SPECIFIC premise values (mutation-sensitive)
# ---------------------------------------------------------------------------

def test_rendered_verdict_cites_premise_with_specific_values():
    """A 2-file / known-byte premise appears verbatim in the rendered verdict.

    Fails if the footer premise render is removed, or the count/bytes are wrong.
    """
    diff_files = {"lib/foo.py", "lib/bar.py"}
    provenance = _premise_provenance("Add premise provenance", diff_files, 512)
    review = _valid_review(premise_provenance=provenance)

    md = review.to_markdown()

    # The premise citation line is present with the exact fixture values.
    assert "Premise reviewed:" in md
    assert "2 files" in md               # a 2-file diff -> "2 files" (miscount fails)
    assert "512 diff bytes" in md        # exact byte count (wrong size fails)
    assert "Add premise provenance" in md  # the graded PR title


def test_rendered_premise_is_the_full_expected_line():
    """The whole premise line renders in canonical shape (format regression)."""
    review = _valid_review(
        premise_provenance=_premise_provenance("My PR", {"a/x.py", "a/y.py"}, 128),
    )
    md = review.to_markdown()
    assert '*Premise reviewed: PR "My PR" · 2 files · 128 diff bytes*' in md


# ---------------------------------------------------------------------------
# P2 -- the helper is exact + de-aliases the file set
# ---------------------------------------------------------------------------

def test_premise_provenance_exact_string():
    line = _premise_provenance("Title Here", {"lib/a.py", "lib/b.py"}, 4096)
    assert line == 'PR "Title Here" · 2 files · 4096 diff bytes'


def test_premise_provenance_de_aliases_basename_duplicates():
    """_extract_diff_files injects basename aliases; the count must NOT inflate.

    {plans/foo.md, foo.md, lib/bar.py, bar.py} is really 2 files, not 4.
    """
    aliased = {"plans/foo.md", "foo.md", "lib/bar.py", "bar.py"}
    # Sanity: the de-alias helper this relies on collapses to 2.
    assert len(_real_diff_files(aliased)) == 2
    line = _premise_provenance("Doc + code", aliased, 900)
    assert "2 files" in line
    assert "4 files" not in line


def test_premise_provenance_single_line_title():
    """A mis-framed multi-line title cannot break the markdown line."""
    line = _premise_provenance("bad\ntitle\nwith newlines", {"lib/a.py"}, 10)
    assert "\n" not in line
    assert 'PR "bad title with newlines"' in line


# ---------------------------------------------------------------------------
# P3 -- empty / absent premise handled gracefully (no crash, line omitted)
# ---------------------------------------------------------------------------

def test_absent_premise_renders_not_recorded():
    """Default empty premise -> footer ALWAYS cites the premise line.

    The verdict footer must never silently omit its own premise-provenance: an
    empty premise renders "(not recorded)" — consistent with the FAILED banner
    (test_error_banner_absent_premise_renders_not_recorded). This is MORE
    faithful to the doctrine (a verdict always states its premise, even none).
    """
    review = _valid_review(premise_provenance="")  # explicit backwards-compat
    md = review.to_markdown()  # must not raise
    assert "AGORA Council Review" in md
    assert "Premise reviewed:" in md   # footer ALWAYS cites the premise
    assert "(not recorded)" in md      # empty premise -> graceful placeholder


def test_premise_provenance_empty_inputs_no_crash():
    """Empty title + empty file set + zero bytes -> sane fallback, no crash."""
    line = _premise_provenance("", set(), 0)
    assert line == 'PR "(untitled)" · 0 files · 0 diff bytes'


def test_premise_provenance_negative_bytes_clamped():
    """Defensive: a negative byte size never renders a negative count."""
    line = _premise_provenance("X", {"a.py"}, -5)
    assert "0 diff bytes" in line
    assert "-5" not in line


# ---------------------------------------------------------------------------
# P4 -- the FAILED (invalid) banner carries the premise (visible when it matters)
# ---------------------------------------------------------------------------

def test_error_banner_carries_premise():
    """A rejected (hallucination-guard) review still shows what it graded.

    This is the load-bearing case: a truncated/wrong diff is exactly the invalid
    path, and the premise must be visible so the corrupt premise is auditable.
    """
    review = AgoraReview(
        pr_number=9,
        repo="CodeTonight-SA/HAL",
        plain_language="",
        quality="",
        technical="",
        speakers_responded=0,
        speakers_total=3,
        is_valid=False,
        validity_reasons=["PR diff was empty (gh pr diff failed or returned nothing)"],
        diff_files=["lib/x.py", "lib/y.py"],
        premise_provenance=_premise_provenance("Broken diff PR", {"lib/x.py", "lib/y.py"}, 0),
    )
    md = review.to_markdown()
    assert "AGORA Council Review -- FAILED" in md or "AGORA Council Review — FAILED" in md
    assert "Premise reviewed" in md
    assert "Broken diff PR" in md
    assert "2 files" in md


def test_error_banner_absent_premise_renders_not_recorded():
    """FAILED banner with no premise renders a graceful placeholder, no crash."""
    review = AgoraReview(
        pr_number=9,
        repo="CodeTonight-SA/HAL",
        plain_language="",
        quality="",
        technical="",
        speakers_responded=0,
        speakers_total=3,
        is_valid=False,
        validity_reasons=["PR diff was empty"],
        premise_provenance="",
    )
    md = review.to_markdown()  # must not raise
    assert "Premise reviewed" in md
    assert "(not recorded)" in md


# ---------------------------------------------------------------------------
# P5 -- WIRING: the real orchestrator populates the field, network-free
# ---------------------------------------------------------------------------

class _FakeKernelResult:
    """Minimal stand-in for InnerspeakerGo's result: .state + .metrics."""

    def __init__(self, state):
        self.state = state
        self.metrics = {}


def _fake_kernel_factory(state):
    """Return a kernel class whose __call__ ignores input and yields state."""

    class _FakeKernel:
        def __init__(self, *a, **k):
            pass

        def __call__(self, _seed):
            return _FakeKernelResult(state)

    return _FakeKernel


def test_review_pr_wires_premise_from_fetched_diff():
    """review_pr computes premise-provenance from the fetched diff, no network.

    Fails if the orchestrator stops passing ``premise_provenance`` into the
    AgoraReview it builds. A grounded 2-file diff -> the rendered verdict output
    cites "2 files" and the fetched title.
    """
    title = "Wiring fixture PR"
    # A real-ish 2-file diff excerpt that grounds Speakers B/C (they cite paths).
    diff = (
        "diff --git a/lib/foo.py b/lib/foo.py\n"
        "+++ b/lib/foo.py\n@@\n+x = 1\n"
        "diff --git a/lib/bar.py b/lib/bar.py\n"
        "+++ b/lib/bar.py\n@@\n+y = 2\n"
    )
    speakers = {
        "plain_language": "In plain words this adds two small assignments.",
        "quality": "lib/foo.py looks fine and lib/bar.py is aligned.",
        "technical": "Concern: lib/bar.py bound at line 2; lib/foo.py ok.",
    }

    with patch.object(agora, "_fetch_pr_metadata", return_value=(title, diff)), \
         patch.object(agora, "InnerspeakerGo", _fake_kernel_factory(speakers)):
        review = agora.review_pr(
            pr_number=123,
            repo="CodeTonight-SA/HAL",
            dispatch_judge=False,       # no judge / LLM call
            create_defer_issue=False,
        )

    # The orchestrator recorded the premise from the fetched diff.
    assert review.premise_provenance
    assert "2 files" in review.premise_provenance
    assert title in review.premise_provenance
    # And it renders through into the published verdict output.
    md = review.to_markdown()
    assert "Premise reviewed:" in md
    assert "2 files" in md


# ---------------------------------------------------------------------------
# P6 -- FORWARD ANCHOR: "diff bytes" is the UTF-8 BYTE count, not the char count.
#        The reviewer's "what I would falsify": a non-ASCII diff must report
#        len(diff.encode("utf-8")), NOT len(diff). Mutation-sensitive — reverting
#        the byte-count fix (back to len(diff_excerpt)) makes the two differ.
# ---------------------------------------------------------------------------

def test_premise_bytes_are_utf8_not_char_count():
    """A multi-byte-UTF-8 diff reports its UTF-8 byte size, not char count.

    ``café`` (é = 2 bytes) and an em-dash ``—`` (3 bytes) make the byte length
    strictly greater than the char length. The premise MUST cite the byte count
    (len(diff.encode("utf-8"))); if it were len(diff) the two would collide and
    the label "diff bytes" would undercount.
    """
    title = "UTF-8 byte-count PR"
    # A grounded 2-file diff carrying non-ASCII chars (é = 2 bytes, — = 3 bytes).
    diff = (
        "diff --git a/lib/foo.py b/lib/foo.py\n"
        "+++ b/lib/foo.py\n@@\n+cafe = 'café'  # accented — em-dash\n"
        "diff --git a/lib/bar.py b/lib/bar.py\n"
        "+++ b/lib/bar.py\n@@\n+y = 2\n"
    )
    byte_len = len(diff.encode("utf-8"))
    char_len = len(diff)
    # Guard the fixture: it MUST contain multi-byte chars, else the test is inert.
    assert byte_len > char_len, "fixture must contain multi-byte UTF-8 chars"

    speakers = {
        "plain_language": "In plain words this sets a string and a number.",
        "quality": "lib/foo.py looks fine and lib/bar.py is aligned.",
        "technical": "Concern: lib/bar.py bound at line 2; lib/foo.py ok.",
    }
    with patch.object(agora, "_fetch_pr_metadata", return_value=(title, diff)), \
         patch.object(agora, "InnerspeakerGo", _fake_kernel_factory(speakers)):
        review = agora.review_pr(
            pr_number=124,
            repo="CodeTonight-SA/HAL",
            dispatch_judge=False,
            create_defer_issue=False,
        )

    # The recorded premise cites the UTF-8 BYTE count, never the char count.
    assert f"{byte_len} diff bytes" in review.premise_provenance
    assert f"{char_len} diff bytes" not in review.premise_provenance
    # And it renders that byte count into the published verdict output.
    assert f"{byte_len} diff bytes" in review.to_markdown()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
