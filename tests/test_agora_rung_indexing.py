"""The convergence step must dispatch each role's FIRST choice first.

``InnerspeakerGo`` is 1-based: ``_improve`` calls its converge callback with
``depth + 1`` (lib/improve_kernel.py:151), so the first step arrives at
``_converge_step`` as ``depth=1``. Indexing the fallback chain with the raw
kernel depth therefore skipped ``chain[0]`` for every role — each role's
designated first choice had never been dispatched on any HAL pull request.

The defect was invisible until per-seat logging landed (HAL#567): run
34424276620 shows plain_language and quality answering at rung=1 and technical
at rung=2, with no rung=0 anywhere.

These assert BEHAVIOUR observed through the real kernel rather than restating
the arithmetic, so a future change to the kernel's depth base fails here
instead of silently re-rotating the roster.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import lib.agora_pr_review as apr  # noqa: E402
from lib.improve_kernel import InnerspeakerGo  # noqa: E402

_CHAINS = {
    "plain_language": [
        ("first-pl", "groq/openai/gpt-oss-120b", {"scheme": "apikey"}),
        ("second-pl", "xai/grok-3", {"scheme": "apikey"}),
        ("last-pl", "deepseek/deepseek-v4-pro", {"scheme": "apikey"}),
    ],
    "quality": [
        ("first-q", "xai/grok-3", {"scheme": "apikey"}),
        ("second-q", "deepseek/deepseek-v4-pro", {"scheme": "apikey"}),
        ("last-q", "groq/openai/gpt-oss-120b", {"scheme": "apikey"}),
    ],
    "technical": [
        ("first-t", "deepseek/deepseek-v4-pro", {"scheme": "apikey"}),
        ("second-t", "groq/openai/gpt-oss-120b", {"scheme": "apikey"}),
        ("last-t", "xai/grok-3", {"scheme": "apikey"}),
    ],
}


def _recording_dispatch(seen):
    """Stub speaker that records (role, provider) and returns usable text."""
    def _stub(role, prompt, happi_provider, auth, **kwargs):
        seen.append((role, happi_provider))
        return f"text for {role}"
    return _stub


def test_the_first_dispatch_reaches_each_roles_first_choice(monkeypatch):
    """THE load-bearing assertion: the opening round seats chain[0].

    Driven through the real ``InnerspeakerGo`` — the same wiring ``review_pr``
    builds — so the kernel supplies its own depth and the test cannot agree
    with the code about an arithmetic both got wrong. Before the fix this fails
    with every role seated on its SECOND choice.
    """
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(apr, "_dispatch_speaker", _recording_dispatch(seen))

    kernel = InnerspeakerGo(
        converge=lambda state, depth: apr._converge_step(
            state, depth, pr_title="t", diff_excerpt="d", role_chains=_CHAINS,
        ),
        threshold={},          # first round satisfies it; no retries
        max_depth=11,
    )
    kernel({})

    seated = dict(seen)
    expected = {role: chain[0][1] for role, chain in _CHAINS.items()}
    assert seated == expected, (
        "the opening round must seat each role's first choice; got "
        f"{seated!r}, expected {expected!r}"
    )


def test_a_role_whose_first_choice_fails_falls_through_to_its_second(monkeypatch):
    """The fix must not cost the fallback: rung 0 empty still reaches rung 1.

    Guards the obvious wrong repair — clamping or shifting the whole ladder —
    which would seat chain[0] and then never advance.
    """
    seen: list[tuple[str, str]] = []
    chain = {"quality": _CHAINS["quality"]}

    def _stub(role, prompt, happi_provider, auth, **kwargs):
        seen.append((role, happi_provider))
        # First choice returns nothing; the second answers.
        return "" if happi_provider == chain["quality"][0][1] else "text"

    monkeypatch.setattr(apr, "_dispatch_speaker", _stub)

    apr._converge_step({}, 1, pr_title="t", diff_excerpt="d", role_chains=chain)
    apr._converge_step({}, 2, pr_title="t", diff_excerpt="d", role_chains=chain)

    assert [p for _r, p in seen] == [chain["quality"][0][1], chain["quality"][1][1]], (
        f"expected first-then-second choice, got {seen!r}"
    )


def test_depth_zero_raises_rather_than_seating_the_last_rung(monkeypatch):
    """A 0 must be loud, because silently it would seat the LAST rung.

    ``chain[-1]`` is valid Python and wraps to the end of the ladder, so a
    caller modelling the kernel wrongly would dispatch the final fallback as
    though it were the first choice — the same class of bug, one rotation
    further round. Raising is therefore the correct handling, not clamping:
    clamping would paper over a caller that has the kernel's contract wrong.
    """
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(apr, "_dispatch_speaker", _recording_dispatch(seen))

    with pytest.raises(ValueError, match="1-based"):
        apr._converge_step({}, 0, pr_title="t", diff_excerpt="d", role_chains=_CHAINS)

    assert not seen, f"depth=0 dispatched before raising: {seen!r}"


def test_an_exhausted_chain_is_skipped_not_wrapped(monkeypatch):
    """Past the end of the ladder a role is skipped, never restarted.

    The exhaustion bound moves with the index — off by one against the rung
    rather than the depth — so this pins that the two moved together. With a
    three-rung chain, depth 4 is one past the end.
    """
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(apr, "_dispatch_speaker", _recording_dispatch(seen))

    apr._converge_step({}, 4, pr_title="t", diff_excerpt="d", role_chains=_CHAINS)

    assert not seen, f"a role past the end of its chain was dispatched: {seen!r}"


def test_the_last_rung_is_still_reachable(monkeypatch):
    """The other side of the bound: depth 3 must still seat rung 2.

    Without this, tightening the exhaustion check by one too many would pass
    the test above while quietly deleting every ladder's final fallback.
    """
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(apr, "_dispatch_speaker", _recording_dispatch(seen))

    apr._converge_step({}, 3, pr_title="t", diff_excerpt="d", role_chains=_CHAINS)

    seated = dict(seen)
    expected = {role: chain[2][1] for role, chain in _CHAINS.items()}
    assert seated == expected, f"last rung unreachable; got {seated!r}"


# ---------------------------------------------------------------------------
# Forward anchor — AGORA #570 Speaker C, "what I would falsify if I had time"
# ---------------------------------------------------------------------------
#
# The reviewer proposed sweeping every depth from 0 to max_depth + 2 and
# asserting the rung relationship holds throughout. The code already passes it,
# which is exactly why it is written down: it pins the whole mapping rather
# than the three points the tests above sample, so a future re-rotation cannot
# slip between them.


def test_every_depth_maps_to_exactly_one_rung_or_nothing(monkeypatch):
    """Sweep the full depth range and pin the contract at every point.

    For each depth d over 0..len(chain)+2:
      * d == 0 raises, and dispatches nothing;
      * d >= 1 dispatches chain[d - 1] when d - 1 is in range;
      * otherwise nothing is dispatched — never a wrap to an earlier rung.
    """
    chain_len = len(_CHAINS["plain_language"])

    for depth in range(0, chain_len + 3):
        seen: list[tuple[str, str]] = []
        monkeypatch.setattr(apr, "_dispatch_speaker", _recording_dispatch(seen))

        if depth == 0:
            with pytest.raises(ValueError):
                apr._converge_step(
                    {}, depth, pr_title="t", diff_excerpt="d", role_chains=_CHAINS,
                )
            assert not seen, f"depth {depth} dispatched before raising: {seen!r}"
            continue

        apr._converge_step(
            {}, depth, pr_title="t", diff_excerpt="d", role_chains=_CHAINS,
        )
        rung = depth - 1
        seated = dict(seen)

        if rung < chain_len:
            expected = {role: chain[rung][1] for role, chain in _CHAINS.items()}
            assert seated == expected, (
                f"depth {depth} must seat rung {rung}; got {seated!r}"
            )
        else:
            assert not seated, (
                f"depth {depth} is past the end of the chain but dispatched "
                f"{seated!r} — the ladder wrapped instead of terminating"
            )
