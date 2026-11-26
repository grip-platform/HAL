"""DEFER follow-ups must actually file, and a label must never swallow one.

Every DEFER follow-up on HAL had silently failed to file. There were two
causes, both in the label set. The first — no ``agora-defer`` label on the
repo — was fixed by creating it. The second is here: ``_create_defer_issue``
also passed ``--label pr-<n>-followup``, and nothing creates that per-PR
label, so ``gh issue create`` aborted the whole call:

    DEFER auto-issue creation failed (rc=1): could not add label:
    'pr-567-followup' not found

``gh`` resolves label names BEFORE creating the issue and fails the call on any
unknown one, so a per-PR label is not merely unused — it is impossible.

The shape mirrors GRIP, which hit this and removed it (GRIP#4168). Labels are
metadata; the tracked issue is the deliverable, so a label failure now costs
the labels rather than the follow-up.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import lib.agora_pr_review as apr  # noqa: E402
from lib.agora_pr_review import (  # noqa: E402
    VERDICT_DEFER_CHANGES,
    AgoraReview,
    JudgeOutput,
    _create_defer_issue,
)

_ISSUE_URL = "https://github.com/CodeTonight-SA/HAL/issues/123"


def _review() -> AgoraReview:
    return AgoraReview(
        pr_number=567,
        repo="CodeTonight-SA/HAL",
        plain_language="plain",
        quality="quality",
        technical="technical concerns",
        speakers_responded=3,
        speakers_total=3,
        verdict=VERDICT_DEFER_CHANGES,
    )


def _judge() -> JudgeOutput:
    return JudgeOutput(
        verdict=VERDICT_DEFER_CHANGES,
        tldr="defer this",
        eli5="later",
        reasoning="not blocking",
        raw_text="",
        is_valid=True,
    )


def _labels_of(argv) -> list[str]:
    """Every --label value, in either spelling.

    Reads both ``--label X`` and ``--label=X`` so the assertion pins the
    PROPERTY (which labels are passed) rather than the flag spelling — a test
    that pins the spelling turns red on a hardening that changes nothing about
    which labels are sent.
    """
    out = []
    for i, tok in enumerate(argv):
        if tok == "--label" and i + 1 < len(argv):
            out.append(argv[i + 1])
        elif tok.startswith("--label="):
            out.append(tok.split("=", 1)[1])
    return out


def _creates(argvs) -> list[list[str]]:
    return [
        a for a in argvs
        if len(a) >= 3 and a[0] == "gh" and a[1] == "issue" and a[2] == "create"
    ]


class _Proc:
    def __init__(self, returncode=0, stdout=_ISSUE_URL + "\n", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _patch_run(monkeypatch, responder):
    seen: list[list[str]] = []

    def _fake(argv, *a, **kw):
        seen.append(list(argv))
        return responder(argv)

    monkeypatch.setattr(apr.subprocess, "run", _fake)
    return seen


def test_no_per_pr_label_is_passed_to_gh(monkeypatch):
    """THE load-bearing assertion — this is what made every follow-up fail.

    Asserts over the real argv rather than a call count, and by SHAPE rather
    than the one literal, so any future per-PR label fails here too.
    """
    seen = _patch_run(monkeypatch, lambda argv: _Proc())

    url = _create_defer_issue(_review(), _judge())

    assert url == _ISSUE_URL
    labels = _labels_of(_creates(seen)[0])
    assert not [x for x in labels if x.startswith("pr-")], (
        f"a per-PR label reached gh issue create and will abort it: {labels}"
    )
    assert "agora-defer" in labels, f"the routing label was dropped: {labels}"


def test_a_label_failure_costs_the_labels_not_the_follow_up(monkeypatch):
    """A missing label must not lose the tracked issue.

    HAL has now been bitten twice by this class in one night. The issue is the
    deliverable; the labels are metadata. Retry unlabelled and keep it.
    """
    def _responder(argv):
        if "--label" in " ".join(argv):
            return _Proc(1, "", "could not add label: 'nope' not found")
        return _Proc()

    seen = _patch_run(monkeypatch, _responder)

    url = _create_defer_issue(_review(), _judge())

    assert url == _ISSUE_URL, "a label failure swallowed the whole follow-up"
    creates = _creates(seen)
    assert len(creates) == 2, f"expected a labelled try then a bare retry: {len(creates)}"
    assert not _labels_of(creates[1]), "the retry still carried labels"


def test_a_non_label_failure_is_not_retried(monkeypatch):
    """Only a LABEL error justifies the bare retry.

    An auth failure, a bad repo or a timeout fails identically without labels.
    Retrying would waste a call and — worse — overwrite the real cause with a
    second copy of itself under a "label retry" narrative.
    """
    seen = _patch_run(
        monkeypatch,
        lambda argv: _Proc(1, "", "HTTP 401: Bad credentials"),
    )

    url = _create_defer_issue(_review(), _judge())

    assert url == ""
    assert len(_creates(seen)) == 1, "a non-label failure was retried unlabelled"


def test_the_command_hal_will_now_run_is_free_of_unknown_labels(monkeypatch):
    """End-to-end on the ACTUAL argv: every label sent must exist on the repo.

    Asserts the consequence rather than the predicate — the reason the old call
    failed is that it named a label the repo does not have.
    """
    seen = _patch_run(monkeypatch, lambda argv: _Proc())
    _create_defer_issue(_review(), _judge())

    # Labels that exist on CodeTonight-SA/HAL (verified against the live repo).
    existing = {"enhancement", "agora-defer"}
    labels = set(_labels_of(_creates(seen)[0]))
    assert labels <= existing, (
        f"gh issue create names labels the repo does not have: {labels - existing}"
    )


# ---------------------------------------------------------------------------
# Label-error classification — AGORA #571 concern 1
# ---------------------------------------------------------------------------
#
# The council blocked the first form of this PR because `_is_label_error` was
# a bare `"label" in message` substring test. That is the substring-matching
# defect class this repo has hit repeatedly: the token matches inside longer
# words that mean something else. These pin the boundary in both directions,
# because a classifier that only ever says yes is not a classifier.


@pytest.mark.parametrize("message", [
    "could not add label: 'pr-567-followup' not found",   # the real one, verbatim
    "COULD NOT ADD LABEL: 'x' not found",                 # case-insensitive
    "gh issue create failed (rc=1): could not add label: 'agora-defer' not found",
    "invalid label format in request payload",
    "unknown labels supplied",
])
def test_a_label_failure_is_classified_as_one(message):
    assert apr._is_label_error(message) is True, message


@pytest.mark.parametrize("message", [
    "HTTP 401: Bad credentials",
    "could not resolve to a Repository with the name 'x'",
    "dial tcp: i/o timeout",
    "subprocess error: TimeoutExpired",
    # The anti-Goodhart anchors: "label" appears INSIDE another word. A
    # substring test says yes to both of these and is wrong both times.
    "HTTP 500: unlabeled internal error",
    "relabelling the milestone failed",
    "",
])
def test_a_non_label_failure_is_not_classified_as_a_label_error(message):
    assert apr._is_label_error(message) is False, message


def test_both_errors_survive_a_failed_retry(monkeypatch, caplog):
    """AGORA #571 concern 2: the original error is the diagnosis, so keep it.

    When the unlabelled retry also fails, logging only the retry's error
    discards the label name that caused the first failure — the one fact
    needed to fix it.
    """
    import logging as _logging

    _patch_run(
        monkeypatch,
        lambda argv: _Proc(1, "", "could not add label: 'agora-defer' not found")
        if any(t.startswith("--label=") for t in argv)
        else _Proc(1, "", "HTTP 503: service unavailable"),
    )

    with caplog.at_level(_logging.WARNING, logger=apr.logger.name):
        url = _create_defer_issue(_review(), _judge())

    assert url == ""
    # Assert on the FINAL line specifically, not the joined stream. The earlier
    # "retrying unlabelled" line also carries the label error, so a joined
    # assertion passes whether or not the final line kept it — it cannot fail
    # for the reason this test exists, which a mutation confirmed.
    final = [
        r.getMessage() for r in caplog.records
        if "unlabelled too" in r.getMessage()
    ]
    assert final, "no final failure line was logged"
    assert "HTTP 503" in final[-1], f"retry error missing from: {final[-1]}"
    assert "agora-defer" in final[-1], (
        f"original label error discarded from the final line: {final[-1]}"
    )


def test_the_back_link_uses_the_bare_retry_url(monkeypatch):
    """Forward anchor — AGORA #571 Speaker C's "what I would falsify".

    Forces the labelled attempt to fail and the unlabelled retry to succeed,
    then asserts the closed-loop PR comment is emitted exactly once and carries
    the URL the retry produced.

    Speaker C read this path as discarding the retry's URL, leaving the comment
    to link an empty string. It does not — the retry rebinds `issue_url`. The
    test is written anyway, because the branch genuinely had no coverage and an
    untested claim is worth exactly as much whether or not it is currently
    true.
    """
    retry_url = "https://github.com/CodeTonight-SA/HAL/issues/999"

    def _responder(argv):
        if any(t.startswith("--label=") for t in argv):
            return _Proc(1, "", "could not add label: 'agora-defer' not found")
        return _Proc(0, retry_url + "\n", "")

    seen = _patch_run(monkeypatch, _responder)

    url = _create_defer_issue(_review(), _judge())

    assert url == retry_url
    comments = [a for a in seen if a[:3] == ["gh", "pr", "comment"]]
    assert len(comments) == 1, f"expected exactly one back-link, got {len(comments)}"
    body = " ".join(comments[0])
    assert retry_url in body, f"back-link does not carry the retry URL: {body}"
