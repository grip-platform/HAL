"""HAL AGORA sends only shielded text to outside review providers.

All identities and identifiers in this module are deliberately fake.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from tests._public_tree import require

import lib.agora_pr_review as apr
import lib.pr_diff as pd
import lib.hal_review_redaction as rr


NAME = "Jane Testperson"
TOKEN = "[PERSON-1]"
PHONE = "27000000000@s.whatsapp.net"
EMAIL = "jane.testperson@example.invalid"
FAKE_LIST = {
    "version": 1,
    "allowlist": [],
    "entities": [{
        "canonical": NAME,
        "aliases": [NAME],
        "pseudonym": TOKEN,
        "class": "person",
    }],
}


def _encoded_list() -> str:
    return base64.b64encode(json.dumps(FAKE_LIST).encode()).decode()


@pytest.fixture(autouse=True)
def _fake_denylist(monkeypatch):
    monkeypatch.setenv(rr.DENYLIST_ENV, _encoded_list())
    monkeypatch.setenv(rr.rs.OVERLAY_ENV, "none")
    rr.reset()
    yield
    rr.reset()


def test_name_and_identifier_floor_are_redacted():
    out = rr.redact_outbound(f"{NAME} can be reached at {EMAIL} or {PHONE}")
    assert NAME not in out
    assert EMAIL not in out
    assert PHONE not in out
    assert TOKEN in out


def test_failing_shield_withholds_the_whole_text(monkeypatch):
    rr.redact("warm cache")

    def fail(_self, _text):
        raise RuntimeError("synthetic shield failure")

    monkeypatch.setattr(rr.rs.Shield, "redact", fail)
    secret_text = f"prefix {NAME} suffix"
    assert rr.redact_outbound(secret_text) == rr.WITHHELD
    assert NAME not in rr.redact_outbound(secret_text)


def test_redaction_is_idempotent_so_double_scrubbing_is_safe():
    """The diff is scrubbed inside robust_pr_diff and again as part of the whole prompt.

    A second pass over scrubbed text must change nothing, or the prompt would carry
    a different text from the one the footer counted.
    """
    once = rr.redact(f"{NAME} at {EMAIL} or {PHONE}, already {TOKEN}")
    assert NAME not in once and EMAIL not in once and PHONE not in once
    assert rr.redact(once) == once


def test_unreadable_overlay_withholds_even_with_the_names_secret(monkeypatch, tmp_path):
    """A configured overlay that cannot be read withholds the text.

    It never falls back to the shared list alone: the overlay holds names the shared
    list cannot, so scrubbing without it would send exactly those names out.
    """
    bad = tmp_path / "overlay.json"
    bad.write_text("{not json")
    monkeypatch.setenv(rr.rs.OVERLAY_ENV, str(bad))
    rr.reset()
    assert rr.redact_outbound(f"prefix {NAME} suffix") == rr.WITHHELD
    assert "withheld because redaction failed" in rr.render_footer()


def test_removed_diff_line_reaches_speaker_only_redacted(monkeypatch):
    captured = []

    def envelope(_pid, _provider, _auth, prompt, **_kwargs):
        captured.append(prompt)
        return {}

    monkeypatch.setattr(apr, "_build_envelope", envelope)
    monkeypatch.setattr(apr, "_call_happi_ex", lambda *_a, **_k: ("review", None))
    prompt = "Review this diff:\n@@ -1 +1 @@\n-Owner: " + NAME + "\n+Owner: nobody\n"
    assert apr._dispatch_speaker("technical", prompt, "fake/provider", {}) == "review"
    assert captured and NAME not in captured[0]
    assert TOKEN in captured[0]


def test_judge_prompt_is_redacted_immediately_before_dispatch(monkeypatch):
    captured = []
    monkeypatch.setattr(apr, "_JUDGE_PROVIDER", ("judge", "fake/judge", {}, "Fake"))
    monkeypatch.setattr(
        apr, "_build_envelope",
        lambda _pid, _provider, _auth, prompt, **_kwargs: captured.append(prompt) or {},
    )
    monkeypatch.setattr(apr, "_call_happi", lambda *_a, **_k: "")
    apr._dispatch_judge("title", NAME, "quality", "technical")
    assert captured and NAME not in captured[0]
    assert TOKEN in captured[0]


def test_robust_pr_diff_omits_deleted_body_and_scrubs_before_budget(monkeypatch):
    deleted_secret = "never-send-this-deleted-body"
    header = (
        "diff --git a/old.txt b/old.txt\n"
        "deleted file mode 100644\n--- a/old.txt\n+++ /dev/null\n"
        "@@ -1,2 +0,0 @@\n"
        f"-{deleted_secret}\n-{NAME}\n"
        "diff --git a/live.txt b/live.txt\n--- a/live.txt\n+++ b/live.txt\n"
        "@@ -0,0 +1 @@\n+"
    )
    padding = "x" * 80 + " "
    diff = header + padding + NAME + " tail\n"
    omitted = pd._omit_deleted_bodies(header)
    # The pseudonym is shorter than the name: this cut includes the whole
    # replacement, but would split the original name if budgeting ran first.
    cut = len(omitted + padding) + len(TOKEN)
    monkeypatch.setattr(pd, "_poll_mergeability", lambda *_a, **_k: None)
    monkeypatch.setattr(pd, "_run", lambda *_a, **_k: diff)

    out = pd.robust_pr_diff(
        1, "owner/repo", max_chars=cut,
        scrub=lambda text: text.replace(NAME, TOKEN),
    )

    assert deleted_secret not in out
    assert "[deleted file: 2 lines removed; contents not sent for review]" in out
    assert NAME not in out and "Jane" not in out
    assert TOKEN in out


def test_secret_precedes_checkout(monkeypatch):
    monkeypatch.setattr(rr, "_from_files", lambda: rr.rs.Denylist.empty())
    assert rr.redact_outbound(NAME) == TOKEN
    assert rr.summary()["names_list"] == rr.SOURCE_SECRET


def test_no_names_list_still_runs_identifier_floor(monkeypatch):
    monkeypatch.delenv(rr.DENYLIST_ENV)
    monkeypatch.setattr(rr, "_from_files", lambda: None)
    rr.reset()
    out = rr.redact_outbound(f"Call {PHONE} or email {EMAIL}")
    assert PHONE not in out and EMAIL not in out
    assert rr.summary()["names_list"] == rr.SOURCE_NONE
    assert "names list was not available" in rr.render_footer()


def test_workflow_passes_optional_names_secret():
    workflow = require(".github/workflows/agora-pr-review.yml").read_text()
    expected = "REVIEW_REDACTION_DENYLIST_B64: ${{ secrets.REVIEW_REDACTION_DENYLIST_B64 }}"
    assert expected in workflow


def test_posted_comment_includes_redaction_footer(monkeypatch):
    rr.redact_outbound(NAME)
    sent = []
    monkeypatch.setattr(apr.subprocess, "run", lambda argv, **_kwargs: sent.append(argv))
    review = type("Review", (), {
        "pr_number": 1,
        "repo": "owner/repo",
        "to_markdown": lambda self: "verdict",
    })()
    apr._post_review(review)
    body = sent[0][sent[0].index("--body") + 1]
    assert body.startswith("verdict")
    assert "Outbound text passed the redaction shield" in body


def test_every_speaker_prompt_explains_redacted_values_before_the_diff():
    # Speakers read scrubbed text, so they must be told a placeholder is not the source.
    # Mutation: drop the paragraph from the shared footer, and every role fails;
    # move it below the diff marker, and the ordering check fails.
    for role, template in apr.AGORA_SPEAKER_PROMPTS.items():
        prompt = template.format(pr_title="a title", diff_excerpt="THE DIFF")
        assert "REDACTED VALUES" in prompt, role
        assert prompt.index("REDACTED VALUES") < prompt.index("<<<PR_DIFF_START>>>"), role


def test_the_judge_may_not_block_on_a_placeholder():
    # Mutation: drop the judge's rule 7, and this fails.
    rule = apr._JUDGE_PROMPT_TEMPLATE.split("7. REDACTED VALUES.", 1)
    assert len(rule) == 2
    assert "MUST NOT drive REQUEST_CHANGES" in rule[1].split("Output structure", 1)[0]


def test_the_evidence_appendix_marks_a_placeholder_that_is_literal_in_the_source(tmp_path):
    # A literal "[EMAIL]" passes the shield unchanged while a real address becomes the same
    # "[EMAIL]", so only the real file can tell them apart, and the appendix is read from it.
    # Mutation: drop the mark, or mark every line.
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "notify.py").write_text(
        'def notify():\n    return "[EMAIL]"\n    fallback = "dev@example-corp.io"\n')
    lines = apr._build_evidence_appendix([("lib/notify.py", 2)], tmp_path).splitlines()
    literal = next(line for line in lines if '"[EMAIL]"' in line)
    real = next(line for line in lines if "example-corp" in line)
    assert literal.endswith("placeholder literal in the source")
    assert "placeholder literal" not in real


def test_the_judge_may_block_on_a_placeholder_the_appendix_marks_literal():
    # Mutation: drop the exception from rule 7.
    rule = apr._JUDGE_PROMPT_TEMPLATE.split("7. REDACTED VALUES.", 1)[1].split("Output structure", 1)[0]
    assert "MUST NOT drive REQUEST_CHANGES" in rule
    assert "EVIDENCE APPENDIX" in rule and "placeholder literal in the source" in rule


def test_speakers_are_told_a_placeholder_can_be_literal():
    # Mutation: restore the claim that every placeholder hides a real value.
    for role, template in apr.AGORA_SPEAKER_PROMPTS.items():
        prompt = template.format(pr_title="a title", diff_excerpt="THE DIFF")
        note = prompt.split("REDACTED VALUES.", 1)[1].split("<<<PR_DIFF_START>>>", 1)[0]
        assert "literal" in note and "evidence appendix" in note, role
