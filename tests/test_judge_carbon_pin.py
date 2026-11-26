"""HAL's AGORA judge is Gemini 4 Carbon (V>> 2026-11-26), mirroring GRIP's judge.

V>>'s words: "add the new Google LLMs -- to replace 3.1 Pro". Carbon is the one
Google calls "Our strongest reasoning model" (ai.google.dev/gemini-api/docs/models,
read 2026-11-26). The file name matches tests/test_judge_*.py, the glob
pytest-agora.yml runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lib.agora_pr_review import _JUDGE_MAX_TOKENS, _JUDGE_RUNG_MAX_TOKENS  # noqa: E402
from lib.hal_council_dispatch import _AGORA_SPEAKER_ROSTER, _JUDGE_PROVIDER  # noqa: E402


def test_the_judge_is_gemini_4_carbon():
    """Mutation: restore gemini/gemini-3.1-pro-preview in _JUDGE_PROVIDER -> RED."""
    assert _JUDGE_PROVIDER[1] == "gemini/gemini-4-carbon"


def test_no_speaker_shares_the_judges_family():
    """The judge must stay exogenous to every speaker (rules/exogenous-anchor-law.md).
    Mutation: seat a gemini/ speaker -> RED."""
    assert not [seat for seat in _AGORA_SPEAKER_ROSTER if "gemini" in seat[1].lower()]


def test_the_carbon_judge_gets_googles_32k_budget():
    """Google's Carbon migration guide: "Raise max_output_tokens to at least 32,768".
    Asserted on the envelope _dispatch_judge sends, not only on the table.
    Mutation: drop or lower Carbon's _JUDGE_RUNG_MAX_TOKENS entry, or stop reading
    it in _dispatch_judge -> RED."""
    from unittest.mock import patch
    from lib.agora_pr_review import _dispatch_judge
    assert _JUDGE_RUNG_MAX_TOKENS.get(_JUDGE_PROVIDER[1], _JUDGE_MAX_TOKENS) >= 32768
    captured = {}

    def fake_call_happi(envelope, hal_bin, timeout):
        captured["envelope"] = envelope
        return ("### Verdict\nMERGE\n\n### TL;DR\nFine.\n\n"
                '### ELI5\nLike a green light.\n\n### Reasoning\n> "ok" (Speaker A).\n')

    with patch("lib.agora_pr_review._call_happi", side_effect=fake_call_happi):
        _dispatch_judge("PR title", "A out", "B out", "C out")
    assert captured["envelope"]["flags"]["max_tokens"] >= 32768


def test_the_judge_tells_the_provider_its_timeout():
    """HAL's HAPPI Gemini provider times out at 120 s unless flags.timeout_s says
    otherwise; on 2026-11-26 two of five replayed Carbon judgements came back
    empty at about 121 s although the subprocess allowed 180 s (GRIP#10013).
    Mutation: drop the timeout_s flag from _dispatch_judge -> RED."""
    from unittest.mock import patch
    from lib.agora_pr_review import _dispatch_judge
    captured = {}

    def fake_call_happi(envelope, hal_bin, timeout):
        captured["envelope"], captured["timeout"] = envelope, timeout
        return ("### Verdict\nMERGE\n\n### TL;DR\nFine.\n\n"
                '### ELI5\nLike a green light.\n\n### Reasoning\n> "ok" (Speaker A).\n')

    with patch("lib.agora_pr_review._call_happi", side_effect=fake_call_happi):
        _dispatch_judge("PR title", "A out", "B out", "C out", timeout=180)
    assert 120 < captured["envelope"]["flags"]["timeout_s"] < captured["timeout"]
