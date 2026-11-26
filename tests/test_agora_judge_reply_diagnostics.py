"""When AGORA's judge sends a reply no verdict can be read from, the log shows what it sent.

Origin: 2026-10-10. Four of that day's ten HAL verdicts came back "Judge
unavailable -- judge output missing or malformed Verdict section", and the
workflow logs held nothing more: the reply was parsed and dropped. The transport
had not failed (``_call_happi_ex`` logs every transport failure, and none was
logged), so the judge had answered with text the parser could not read. A reply
cut off mid-sentence (a token budget) and one in another shape (a prompt or the
parser) need different fixes, and only the reply itself tells them apart.

Each test names the mutation of ``lib/agora_pr_review.py`` that turns it red. No
test reaches a model: ``_call_happi`` is patched where ``_dispatch_judge`` looks
it up, as tests/test_judge_cited_evidence.py does.
"""
from __future__ import annotations

import logging
from unittest.mock import patch

from lib import agora_pr_review as apr

SPEAKERS = ("A plain-language review.", "A quality review.", "A technical review.")
USABLE = "### Verdict\nMERGE\n\n### TL;DR\nFine.\n\n### ELI5\nFine.\n\n### Reasoning\nFine.\n"
# A verdict on the header line and a reply that stops mid-word: two shapes the
# parser rejects, so the log must show both ends to tell them apart.
UNUSABLE = "### Verdict: **MERGE**\n" + "word " * 400 + "and then the reply stops mid-sent"


def _judge(reply: str, tmp_path):
    with patch("lib.agora_pr_review._call_happi", return_value=reply):
        return apr._dispatch_judge("A synthetic PR", *SPEAKERS, repo_root=tmp_path)


def _unusable_lines(caplog) -> list:
    return [r.getMessage() for r in caplog.records if "Judge reply unusable" in r.getMessage()]


def test_an_unusable_reply_is_logged_with_its_length_and_both_ends(tmp_path, caplog):
    """Mutation: drop the log call, so the reply is parsed and thrown away as before."""
    with caplog.at_level(logging.WARNING, logger=apr.logger.name):
        judged = _judge(UNUSABLE, tmp_path)
    assert not judged.is_valid
    lines = _unusable_lines(caplog)
    assert len(lines) == 1
    assert f"{len(UNUSABLE)} chars" in lines[0]
    assert "### Verdict: **MERGE**" in lines[0] and "stops mid-sent" in lines[0]
    assert judged.failure_reason in lines[0]


def test_the_excerpt_is_bounded(tmp_path, caplog):
    """Mutation: log the whole reply, which can run to thousands of characters."""
    with caplog.at_level(logging.WARNING, logger=apr.logger.name):
        _judge(UNUSABLE, tmp_path)
    assert len(_unusable_lines(caplog)[0]) < 2 * apr._UNUSABLE_REPLY_EXCERPT + 400


def test_a_usable_reply_logs_nothing(tmp_path, caplog):
    """Mutation: log every reply, which buries the failures in noise."""
    with caplog.at_level(logging.WARNING, logger=apr.logger.name):
        judged = _judge(USABLE, tmp_path)
    assert judged.is_valid and judged.verdict == "MERGE"
    assert _unusable_lines(caplog) == []
