"""Goodhart-proof tests for HAL AGORA verdict-gating (ported from GRIP).

Covers the load-bearing contracts of the AGORA LLM-as-judge layer:

  T1 — exogeneity guard: the judge provider id is NOT one of the AGORA speakers
       (rules/exogenous-anchor-law.md — producer != verifier). This is THE
       load-bearing test: it fails if anyone adds Gemini to the speaker roster.
  T2 — each of MERGE / REQUEST_CHANGES / DEFER_CHANGES parses from judge output.
  T3 — fail-closed: malformed / empty / unknown-token judge output -> the
       REQUEST_CHANGES sentinel (loop, never silently merge).
  T4 — _render_verdict_block emits the `> **Verdict: <TOKEN>**` blockquote on a
       valid verdict, and a "Judge unavailable" banner on a fail-closed verdict.
  T5 — DEFER render path includes a follow-up issue line and leaks NO
       "sprint-loop" literal (HAL has no sprint-loop infra — plan decision 3).

Goodhart guards: assertions check actual output values (the parsed verdict
token, the exact rendered blockquote text, the absence of a specific literal),
not call counts. Each test fails under an obvious mutation of the code it
guards.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Resolve HAL repo root so `lib.*` imports work regardless of pytest cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.environ.setdefault("HAL_HOME", str(_REPO_ROOT))

from lib.agora_pr_review import (  # noqa: E402
    AgoraReview,
    JudgeOutput,
    VERDICT_MERGE,
    VERDICT_REQUEST_CHANGES,
    VERDICT_DEFER_CHANGES,
    VERDICT_JUDGE_UNAVAILABLE_FALLBACK,
    _ALL_VERDICTS,
    _parse_judge_output,
    _dispatch_judge,
    _JUDGE_MAX_TOKENS,
    _JUDGE_RUNG_MAX_TOKENS,
    _create_defer_issue,
    _detect_knowledge_work_pr,
    _extract_diff_files,
    _real_diff_files,
    _validate_review,
)
from lib.hal_council_dispatch import (  # noqa: E402
    _AGORA_SPEAKER_ROSTER,
    _JUDGE_PROVIDER,
)


# ---------------------------------------------------------------------------
# T1 — EXOGENEITY (the load-bearing guard)
# ---------------------------------------------------------------------------

def test_judge_is_exogenous_to_speakers():
    """The judge provider id must NOT be one of the AGORA speakers.

    rules/exogenous-anchor-law.md anchor #3: producer != verifier. HAL's main
    council roster includes Gemini as a speaker — this asserts AGORA's slim
    panel keeps the judge family-disjoint. Fails the instant someone adds the
    Gemini judge id (or any judge) into the speaker roster.
    """
    speaker_ids = {s[0] for s in _AGORA_SPEAKER_ROSTER}
    judge_id = _JUDGE_PROVIDER[0]
    assert judge_id not in speaker_ids, (
        f"judge {judge_id!r} is also a speaker — exogeneity violated"
    )


def test_speaker_roster_has_no_google_provider():
    """No AGORA speaker may be a Google/Gemini provider (the judge's family).

    Checks the happi_provider string of every speaker, not just the id — a
    relabelled-id-but-still-gemini speaker would slip past an id-only check.
    """
    for pid, happi_provider, _auth, _desc in _AGORA_SPEAKER_ROSTER:
        low = f"{pid} {happi_provider}".lower()
        assert "gemini" not in low and "google" not in low, (
            f"speaker {pid!r} ({happi_provider!r}) is a Google provider — "
            "the judge is Google; speakers must be non-Google (exogeneity)"
        )


def test_judge_is_a_google_provider():
    """Sanity: the judge IS Gemini (the family the speakers must avoid)."""
    judge_provider = _JUDGE_PROVIDER[1].lower()
    assert "gemini" in judge_provider or "google" in judge_provider


def test_judge_is_frontier_tier_not_flash():
    """Anchor the Flash->Pro upgrade (2026-05-29): the judge model must be a
    Pro-tier Gemini, never Flash. The Flash judge produced false-positive
    REQUEST_CHANGES verdicts on correct PRs (#387/#389/#390) that contradicted
    its own speaker synthesis. Goodhart-proof: fails if the judge is reverted to
    a Flash model."""
    judge_model = _JUDGE_PROVIDER[1].lower()
    # Gemini 4 ids carry no "pro", so the tier is read from the catalogue row
    # (V>> 2026-11-26 moved the judge to gemini-4-carbon).
    import yaml
    catalogue = Path(__file__).resolve().parent.parent / "config" / "model_catalog.yaml"
    rows = yaml.safe_load(catalogue.read_text(encoding="utf-8"))["models"]
    tiers = {row.get("address"): row.get("tier") for row in rows}
    assert tiers.get(_JUDGE_PROVIDER[1]) == "frontier", f"judge must be frontier tier, got {judge_model!r}"
    assert "flash" not in judge_model, f"judge regressed to Flash: {judge_model!r}"


def test_judge_model_is_in_catalog():
    """Concern #1 (PR #392 review): validate the judge model id is a REAL,
    catalogued model — not a typo that would fail-close every review. Mechanical
    config validation beats the reviewer's suggested runtime fallback, because
    the judge dispatch path already fail-closes (the 'Judge unavailable'
    contract handles an unreachable endpoint). Goodhart-proof: fails if the
    judge is ever set to an uncatalogued/typo'd model address."""
    import os
    catalog = os.path.join(
        os.path.dirname(__file__), "..", "config", "model_catalog.yaml")
    with open(catalog, encoding="utf-8") as f:
        catalog_text = f.read()
    judge_model = _JUDGE_PROVIDER[1]  # e.g. "gemini/gemini-4-carbon"
    assert judge_model in catalog_text, (
        f"judge model {judge_model!r} not in config/model_catalog.yaml — an "
        "uncatalogued/typo'd judge would fail-close every review"
    )


def test_exactly_three_speakers():
    """AGORA seats exactly 3 speakers (A plain / B quality / C technical).

    Asserted over the SEATS, not the roster length. The roster became a
    fallback ladder, so a rung count of 3 no longer means a panel of 3 — it
    means no rung is left to rescue a failed seat. The panel size is a property
    of the roles, and that is what this pins.
    """
    from lib.agora_pr_review import _build_role_chains
    from tests._agora_roster import seated_roster

    chains = _build_role_chains(seated_roster())
    seated = [role for role, chain in chains.items() if chain]
    assert len(seated) == 3, f"expected 3 seated roles, got {seated!r}"


# ---------------------------------------------------------------------------
# T2 — VERDICT PARSING (all three tokens)
# ---------------------------------------------------------------------------

def _judge_text(verdict_token: str) -> str:
    """A minimal well-formed judge response with the given verdict token."""
    return (
        f"### Verdict\n{verdict_token}\n\n"
        "### TL;DR\nDoes a thing; verdict rationale in one breath.\n\n"
        "### ELI5\nLike tidying one drawer in a big cupboard.\n\n"
        '### Reasoning\nThe blocker was real: > "this will break" (Speaker C).\n'
    )


@pytest.mark.parametrize(
    "token",
    [VERDICT_MERGE, VERDICT_REQUEST_CHANGES, VERDICT_DEFER_CHANGES],
)
def test_parse_each_verdict_token(token):
    """Each canonical verdict token parses to a valid JudgeOutput."""
    out = _parse_judge_output(_judge_text(token))
    assert out.is_valid is True
    assert out.verdict == token
    assert out.failure_reason == ""
    # Sub-sections are extracted, not dropped.
    assert out.tldr
    assert out.eli5
    assert out.reasoning


def test_parse_normalises_space_separated_verdict():
    """'REQUEST CHANGES' (space form) normalises to 'REQUEST_CHANGES'."""
    text = _judge_text("MERGE").replace("### Verdict\nMERGE", "### Verdict\nREQUEST CHANGES")
    out = _parse_judge_output(text)
    assert out.is_valid is True
    assert out.verdict == VERDICT_REQUEST_CHANGES
    assert out.verdict in _ALL_VERDICTS


# ---------------------------------------------------------------------------
# T3 — FAIL-CLOSED (malformed / empty / unknown -> REQUEST_CHANGES)
# ---------------------------------------------------------------------------

def test_fail_closed_sentinel_is_request_changes():
    """The fail-closed contract: the sentinel verdict is REQUEST_CHANGES.

    Guards the doctrine directly — a mutation flipping the sentinel to MERGE
    (which would silently merge on judge failure) fails here.
    """
    assert VERDICT_JUDGE_UNAVAILABLE_FALLBACK == VERDICT_REQUEST_CHANGES


def test_empty_judge_output_fails_closed():
    out = _parse_judge_output("")
    assert out.is_valid is False
    assert out.verdict == VERDICT_REQUEST_CHANGES
    assert out.failure_reason


def test_whitespace_only_judge_output_fails_closed():
    out = _parse_judge_output("   \n\t  \n")
    assert out.is_valid is False
    assert out.verdict == VERDICT_REQUEST_CHANGES


def test_malformed_judge_output_fails_closed():
    """No Verdict section at all -> fail-closed REQUEST_CHANGES."""
    out = _parse_judge_output("This is prose with no verdict header anywhere.")
    assert out.is_valid is False
    assert out.verdict == VERDICT_REQUEST_CHANGES
    assert "malformed" in out.failure_reason.lower() or "missing" in out.failure_reason.lower()


def test_unknown_verdict_token_fails_closed():
    """A Verdict header with a non-canonical token -> fail-closed.

    The extraction regex only matches the 3 canonical tokens, so an unknown
    token like APPROVE simply doesn't match -> malformed/missing path. Either
    way the contract is: not a real verdict -> REQUEST_CHANGES, is_valid False.
    """
    out = _parse_judge_output("### Verdict\nAPPROVE\n\n### TL;DR\nx\n")
    assert out.is_valid is False
    assert out.verdict == VERDICT_REQUEST_CHANGES


# ---------------------------------------------------------------------------
# T4 — VERDICT BLOCK RENDER
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
        judge_provider="Google Gemini 3.1 Pro (judge — exogenous to speakers)",
        judge_valid=True,
    )
    base.update(kw)
    return AgoraReview(**base)


def test_verdict_block_emits_blockquote():
    """A valid MERGE verdict renders the `> **Verdict: MERGE**` blockquote."""
    review = _valid_review(VERDICT_MERGE)
    md = review.to_markdown()
    assert "> **Verdict: MERGE**" in md


def test_verdict_block_renders_request_changes_with_spaces():
    """Underscore-form token renders space-form for readability."""
    review = _valid_review(VERDICT_REQUEST_CHANGES)
    md = review.to_markdown()
    assert "> **Verdict: REQUEST CHANGES**" in md


def test_verdict_block_judge_unavailable_banner():
    """A fail-closed (judge_valid=False) review renders the unavailable banner.

    Crucially it must NOT render a fabricated `> **Verdict: <TOKEN>**` line —
    fail-closed must never look like a real verdict.
    """
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
    md = review.to_markdown()
    assert "Judge unavailable" in md
    assert "manual review required" in md
    # No fabricated verdict line.
    assert "> **Verdict: REQUEST CHANGES**" not in md
    assert "> **Verdict: MERGE**" not in md


# ---------------------------------------------------------------------------
# T5 — DEFER PATH: follow-up line present, NO sprint-loop leak
# ---------------------------------------------------------------------------

def test_defer_render_includes_followup_issue_line():
    """When DEFER + a defer_issue_url is set, the footer links the follow-up."""
    review = _valid_review(
        VERDICT_DEFER_CHANGES,
        defer_issue_url="https://github.com/CodeTonight-SA/HAL/issues/999",
    )
    md = review.to_markdown()
    assert "> **Verdict: DEFER CHANGES**" in md
    assert "Follow-up issue auto-created" in md
    assert "https://github.com/CodeTonight-SA/HAL/issues/999" in md


def test_defer_render_has_no_sprint_loop_literal():
    """HAL has no sprint-loop infra — the DEFER render must not mention it.

    Plan decision 3. Guards against accidentally copying GRIP's sprint-loop
    wording into HAL's render path.
    """
    review = _valid_review(
        VERDICT_DEFER_CHANGES,
        defer_issue_url="https://github.com/CodeTonight-SA/HAL/issues/999",
    )
    md = review.to_markdown().lower()
    assert "sprint-loop" not in md
    assert "sprint loop" not in md


def test_create_defer_issue_no_sprint_loop_label(monkeypatch):
    """_create_defer_issue must NOT pass a `sprint-loop` label to gh.

    Captures the gh argv via a fake subprocess.run and asserts the label set.
    Goodhart-proof: checks the actual argument list, not a call count. Fails if
    GRIP's `--label sprint-loop` is reintroduced.

    Label parsing widened 2026-09-10 (tests-must-be-able-to-fail shape 4). It
    read only the `--label X` spelling, so when the call hardened to
    `--label=X` — which changes nothing about WHICH labels are sent — it saw an
    empty label list and its `agora-defer` assertion went red. The test's
    subject is the label SET, so pinning the flag spelling made a green test
    that could fail for a reason unrelated to what it asserts. Both spellings
    are read now; the assertions are unchanged.
    """
    import lib.agora_pr_review as mod

    captured_argvs: list[list[str]] = []

    class _FakeProc:
        returncode = 0
        stdout = "https://github.com/CodeTonight-SA/HAL/issues/123\n"
        stderr = ""

    def _fake_run(argv, *a, **kw):
        captured_argvs.append(argv)
        return _FakeProc()

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    review = _valid_review(VERDICT_DEFER_CHANGES)
    judge = JudgeOutput(
        verdict=VERDICT_DEFER_CHANGES,
        tldr="defer this",
        eli5="later",
        reasoning="not blocking",
        raw_text="",
        is_valid=True,
    )
    url = _create_defer_issue(review, judge)

    assert url == "https://github.com/CodeTonight-SA/HAL/issues/123"
    # Find the `gh issue create` argv.
    create_argvs = [
        argv for argv in captured_argvs
        if len(argv) >= 3 and argv[0] == "gh" and argv[1] == "issue" and argv[2] == "create"
    ]
    assert create_argvs, "no `gh issue create` call was made"
    create_argv = create_argvs[0]
    # Collect every --label value, in either spelling.
    labels = [
        create_argv[i + 1] if tok == "--label" else tok.split("=", 1)[1]
        for i, tok in enumerate(create_argv)
        if (tok == "--label" and i + 1 < len(create_argv)) or tok.startswith("--label=")
    ]
    assert "sprint-loop" not in labels, (
        f"sprint-loop label leaked into gh issue create: {labels}"
    )
    assert "agora-defer" in labels


# ---------------------------------------------------------------------------
# Bonus — knowledge-work detection (ported alongside the verdict layer)
# ---------------------------------------------------------------------------

def test_knowledge_work_detection_md_dominant():
    """>50% .md files -> knowledge-work mode True."""
    files = {"docs/a.md", "docs/b.md", "docs/c.md", "lib/x.py"}
    assert _detect_knowledge_work_pr(files) is True


def test_knowledge_work_detection_code_dominant():
    """Code-dominant diff -> knowledge-work mode False."""
    files = {"lib/a.py", "lib/b.py", "lib/c.py", "docs/x.md"}
    assert _detect_knowledge_work_pr(files) is False


def test_knowledge_work_detection_empty_is_false():
    assert _detect_knowledge_work_pr(set()) is False


# ---------------------------------------------------------------------------
# Regression — single-file PR grounding false-positive (origin: PR #387,
# 2026-05-29; HAL #388). A 1-file doc PR where a fluent prose Speaker B omits
# the path was wrongly flagged "did not reference any file from the PR diff".
# Grounding is a COUNCIL property: if any non-filler speaker cites the one
# file, the council reviewed reality and a prose-only sibling is fine.
# ---------------------------------------------------------------------------

_SINGLE_FILE_DIFF = (
    "diff --git a/plans/hal-bidirectional-mesh-hub.md "
    "b/plans/hal-bidirectional-mesh-hub.md\n"
    "new file mode 100644\n"
    "--- /dev/null\n"
    "+++ b/plans/hal-bidirectional-mesh-hub.md\n"
    "@@ -0,0 +1,3 @@\n"
    "+# HAL — Bidirectional Model-Mesh Hub\n"
    "+One-line thesis: pivot on HAPPI.\n"
    "+Phase 0 wires the live council.\n"
)

_MULTI_FILE_DIFF = (
    "diff --git a/lib/foo.py b/lib/foo.py\n"
    "--- a/lib/foo.py\n"
    "+++ b/lib/foo.py\n"
    "@@ -1 +1 @@\n"
    "-x = 1\n"
    "+x = 2\n"
    "diff --git a/lib/bar.py b/lib/bar.py\n"
    "--- a/lib/bar.py\n"
    "+++ b/lib/bar.py\n"
    "@@ -1 +1 @@\n"
    "-y = 1\n"
    "+y = 2\n"
)


def test_real_diff_files_dedupes_basename_alias():
    """_real_diff_files drops the basename aliases _extract_diff_files injects."""
    assert _real_diff_files({"plans/x.md", "x.md"}) == {"plans/x.md"}
    assert _real_diff_files({"a/b.py", "b.py", "README.md"}) == {"a/b.py", "README.md"}
    assert _real_diff_files(set()) == set()


def test_real_diff_files_never_empty_for_nonempty_input():
    """Forward anchor — refutes the AGORA #389-R1 'runtime exception' verdict.

    The judge claimed _grounding_reasons could raise on ``sorted(real_files)[0]``.
    It cannot: _real_diff_files returns a NON-EMPTY set for ANY non-empty input
    (every branch returns a superset of a non-empty set), so the index is always
    safe. This test fails if a future change lets the helper return empty for
    non-empty input."""
    for case in (
        {"a/b.py", "b.py"},
        {"README.md"},
        {"x/y/z.md", "z.md"},
        {"only.txt"},
        {"a/b.py", "README.md", "b.py"},
        {"deep/nested/path.rst", "path.rst"},
    ):
        assert _real_diff_files(case), f"unexpectedly empty for {case}"


def test_single_file_pr_prose_quality_review_is_valid():
    """#387 regression: 1-file PR with a prose Speaker B (no path) + a grounded
    Speaker C is VALID. Fails under the pre-fix per-speaker rule; passes
    council-level. This is the regression anchor — it must fail if the
    single-file carve-out is removed."""
    plain = "This plan explains a hub that lets any tool reach any model."
    quality = (
        "The architecture is sound — the central pivot is well justified and "
        "the demand-gated phasing keeps scope honest. The council policy is the "
        "main latency risk worth watching."
    )
    technical = (
        "plans/hal-bidirectional-mesh-hub.md:17 — `dispatch_role` is introduced "
        "without an antecedent definition; add a cross-reference."
    )
    is_valid, reasons = _validate_review(plain, quality, technical, _SINGLE_FILE_DIFF)
    assert is_valid is True, reasons
    assert reasons == []


def test_single_file_pr_no_speaker_grounded_still_invalid():
    """Goodhart guard: the carve-out must NOT blanket-disable L3. A 1-file PR
    where NO non-filler speaker cites the file is still the garbage-fed
    signature -> INVALID. Fails if the fix degenerates to 'always pass when
    <=1 file'."""
    plain = "A general overview with no specifics."
    quality = "The write-up is internally consistent and the thesis holds."
    technical = "The argument in some-other-doc.md is weak and unsupported."
    is_valid, reasons = _validate_review(plain, quality, technical, _SINGLE_FILE_DIFF)
    assert is_valid is False
    assert any("single PR file" in r for r in reasons), reasons


def test_multi_file_pr_ungrounded_speaker_still_flagged():
    """Multi-file per-speaker defense is unchanged: a speaker citing zero diff
    files is flagged even when a sibling speaker is grounded. Fails if the fix
    accidentally relaxed the multi-file path."""
    plain = "Two small one-line edits."
    quality = "This change reads cleanly and the intent is clear."
    technical = "lib/foo.py:1 — the increment looks intentional; fine."
    is_valid, reasons = _validate_review(plain, quality, technical, _MULTI_FILE_DIFF)
    assert is_valid is False
    assert any("Speaker quality did not reference" in r for r in reasons), reasons


def test_single_file_pr_empty_or_filler_speakers_still_invalid_via_l1_l2():
    """Refutes #389-R2 concern A: the single-file branch returning [] for empty
    `graded` does NOT 'silently accept' a garbage review. L1 (empty) and L2
    (filler) run BEFORE L3 in _validate_review, so empty/filler B/C are already
    flagged. Forward anchor on the L1->L2->L3 ordering invariant."""
    # Both B and C empty -> L1 flags them (L3 graded is empty -> []).
    is_valid, reasons = _validate_review("Plain prose.", "", "", _SINGLE_FILE_DIFF)
    assert is_valid is False
    assert any("empty" in r.lower() for r in reasons), reasons
    # Both B and C filler -> L2 flags them.
    filler = "I would need to see the actual code to review this."
    is_valid2, reasons2 = _validate_review("Plain prose.", filler, filler, _SINGLE_FILE_DIFF)
    assert is_valid2 is False
    assert any("filler" in r.lower() for r in reasons2), reasons2


def test_multi_real_file_pr_with_basename_alias_keeps_per_speaker_check():
    """Refutes #389-R2 concern B: distinct real files survive _real_diff_files
    deduping, so a 2-real-file PR (even one whose diff set carries a basename
    alias) uses the per-speaker multi-file path, never the single-file carve-out.
    A multi-file PR cannot 'become single-file after deduping'."""
    diff = (
        "diff --git a/lib/foo.py b/lib/foo.py\n--- a/lib/foo.py\n+++ b/lib/foo.py\n"
        "@@ -1 +1 @@\n-x = 1\n+x = 2\n"
        "diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n"
        "@@ -1 +1 @@\n-a\n+b\n"
    )
    assert len(_real_diff_files(_extract_diff_files(diff))) == 2
    plain = "Two files touched."
    quality = "This reads well and the intent is obvious."  # cites neither file
    technical = "lib/foo.py:1 looks fine."                   # grounded
    is_valid, reasons = _validate_review(plain, quality, technical, diff)
    assert is_valid is False
    assert any("Speaker quality did not reference" in r for r in reasons), reasons


# ---------------------------------------------------------------------------
# T6 — judge thinking-budget headroom (ELI5-truncation root-cause fix)
# ---------------------------------------------------------------------------

def test_judge_envelope_has_ample_max_tokens():
    """The judge (gemini-4-carbon since 2026-11-26, a THINKING model) MUST get a
    max_tokens budget far above the shared 2048 default, so its variable
    thinking tokens never crowd out the visible Verdict/TL;DR/ELI5/Reasoning.

    Root cause (proven via live probe; ported from GRIP #3324): thinking +
    visible output share the Gemini maxOutputTokens pool; thinking measured at
    822/1232/1296/1438 and spiked higher on the real 3-speaker synthesis,
    truncating the ELI5 ("...bake identical cookies every time. Right", with the
    Reasoning section never generated) or emptying the response
    ("Judge unavailable"). 8192 leaves ~6700 tokens of headroom.

    Mutation that re-introduces the bug: drop the
    `envelope["flags"]["max_tokens"] = _JUDGE_MAX_TOKENS` override in
    _dispatch_judge (judge falls back to the fatal 2048 default), or lower
    _JUDGE_MAX_TOKENS below 8192.
    """
    assert _JUDGE_MAX_TOKENS >= 8192, (
        f"Judge max_tokens floor regressed to {_JUDGE_MAX_TOKENS}; a thinking "
        "model needs >=8192 so thinking never starves the visible output."
    )

    captured: dict = {}

    def _fake_call_happi(envelope, hal_bin, timeout):
        captured["envelope"] = envelope
        return (
            "### Verdict\nMERGE\n\n### TL;DR\nFine.\n\n"
            '### ELI5\nLike a green light.\n\n### Reasoning\n> "ok" (Speaker A).\n'
        )

    with patch("lib.agora_pr_review._call_happi", side_effect=_fake_call_happi):
        _dispatch_judge("PR title", "A out", "B out", "C out")

    flags = captured["envelope"]["flags"]
    expected = _JUDGE_RUNG_MAX_TOKENS.get(_JUDGE_PROVIDER[1], _JUDGE_MAX_TOKENS)
    assert flags["max_tokens"] == expected, (
        f"Judge envelope max_tokens={flags.get('max_tokens')!r}, expected "
        f"{expected}. The judge-only override in _dispatch_judge was "
        "dropped — it falls back to the fatal 2048 default."
    )
    assert flags["max_tokens"] > 2048, (
        "Judge max_tokens must exceed the shared 2048 default that caused the "
        "thinking-budget truncation."
    )

    # Goodhart guard: the bump is JUDGE-ONLY. The shared _build_envelope (what
    # the 3 speakers use, unmodified) must still yield the 2048 default — a
    # refactor that moved the override into _build_envelope (applying 8192 to
    # every speaker) would make this fail.
    from lib.hal_council_dispatch import _build_envelope
    speaker_env = _build_envelope(
        "speaker", "groq/llama-3.1-70b-versatile",
        {"scheme": "apikey", "token": "x"}, "hi",
    )
    assert speaker_env["flags"]["max_tokens"] == 2048, (
        "Speaker (non-judge) envelopes must keep the shared 2048 default; the "
        "8192 bump is judge-only. If this is 8192, the override leaked to all "
        "speakers."
    )


# ---------------------------------------------------------------------------
# Grounding parity port from GRIP (#388 recurrence class): stems + symbols.
# A review may ground on a distinctive module stem or a symbol DECLARED on an
# added diff line, not only a literal file path — while a wrong-diff
# hallucination still fails.
# ---------------------------------------------------------------------------

from lib.agora_pr_review import (  # noqa: E402
    _extract_diff_files,
    _extract_diff_symbols,
    _grounding_tokens,
    _is_grounded,
)

_STEM_DIFF = """\
diff --git a/lib/decision_record.py b/lib/decision_record.py
--- a/lib/decision_record.py
+++ b/lib/decision_record.py
@@ -1,0 +2,2 @@
+class DecisionRecord:
+    pass
diff --git a/lib/merkle.py b/lib/merkle.py
--- a/lib/merkle.py
+++ b/lib/merkle.py
@@ -1,0 +2,2 @@
+def _root_over_sorted_leaves(leaves):
+    return leaves
"""


def test_grounding_tokens_adds_distinctive_stem():
    toks = _grounding_tokens({"lib/decision_record.py", "decision_record.py"})
    assert "decision_record" in toks
    assert "decision_record.py" in toks  # original tokens preserved


def test_grounding_tokens_excludes_short_and_generic():
    assert "io" not in _grounding_tokens({"lib/io.py", "io.py"})
    assert "index" not in _grounding_tokens({"web/index.ts", "index.ts"})


def test_is_grounded_accepts_module_stem():
    files = _extract_diff_files(_STEM_DIFF)
    assert _is_grounded("The decision_record dataclass is clean.", files) is True
    assert _is_grounded("merkle root computed over sorted leaves.", files) is True


def test_is_grounded_accepts_declared_symbol():
    files = _extract_diff_files(_STEM_DIFF)
    assert _is_grounded(
        "_root_over_sorted_leaves dedupes before hashing.", files, _STEM_DIFF
    ) is True
    assert "_root_over_sorted_leaves" in _extract_diff_symbols(_STEM_DIFF)


def test_is_grounded_still_rejects_hallucination():
    # SOUNDNESS ANCHOR: stems/symbols must NOT let a wrong-diff review pass.
    files = _extract_diff_files(_STEM_DIFF)
    assert _is_grounded(
        "config/redis.go relies on default DialTimeout.", files, _STEM_DIFF
    ) is False


def test_extract_diff_symbols_ignores_comments_and_removed_lines():
    # AGORA #436 C1+C4: commented-out declarations on added lines and
    # declarations on REMOVED lines must not become grounding anchors.
    d = (
        "diff --git a/x.py b/x.py\n"
        "--- a/x.py\n"
        "+++ b/x.py\n"
        "@@ -1,2 +1,3 @@\n"
        "+    # def commented_out(self):\n"
        "+def real_declaration():\n"
        "-def removed_declaration():\n"
    )
    syms = _extract_diff_symbols(d)
    assert "real_declaration" in syms
    assert "commented_out" not in syms
    assert "removed_declaration" not in syms


def test_is_grounded_symbol_match_is_case_insensitive():
    # Refutes AGORA #436 C2: both sides are lowercased at comparison time, so
    # a review citing the lower-cased form of a declared symbol still grounds.
    files = _extract_diff_files(_STEM_DIFF)
    assert _is_grounded(
        "the decisionrecord class handles signing.", files, _STEM_DIFF
    ) is True
