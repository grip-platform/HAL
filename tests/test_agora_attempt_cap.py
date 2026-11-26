"""The AGORA fallback ladder must exhaust, not stop after two attempts.

``InnerspeakerGo._improve`` treats an unchanged state fingerprint as
convergence. For a role that produced NO output an unchanged fingerprint means
the opposite of converged — it means stuck — so two consecutive dead steps read
as a fixed point and the loop stopped. The effect was a hard cap of two
dispatch attempts per role no matter how long its fallback chain was, which is
why a three-rung ladder had never been observed to exhaust.

These tests drive the REAL kernel over the REAL ``_converge_step`` with only
``_dispatch_speaker`` stubbed, because the defect lives in the composition of
the two rather than in either one alone. The kernel is given temporary
persistence paths: its fixed-point registry is otherwise a real file under
``~/.claude`` shared with every past run, which would make these assertions
depend on the developer's machine history.
"""
from __future__ import annotations

import lib.agora_pr_review as apr
from lib.improve_kernel import InnerspeakerGo

_ROLES = ("plain_language", "quality", "technical")


def _ladder(rungs: int) -> dict[str, list[tuple]]:
    """One chain per role, provider id encoding ``<role>-r<index>``.

    Distinct ids per role make the recorded dispatch list say exactly which
    seat was tried at which rung, which is what these tests assert on.
    """
    return {
        role: [(f"{role}-r{i}", f"{role}-r{i}", {"scheme": "apikey"}) for i in range(rungs)]
        for role in _ROLES
    }


def _drive(chains, answers, tmp_path, max_depth=11):
    """Run the real kernel over the real ``_converge_step``.

    ``answers`` maps a provider id to the text that provider returns; any
    provider absent from it returns "" — a dead rung. Returns
    ``(dispatched_provider_ids, result, telemetry_line_count)``.
    """
    tried: list[str] = []

    def fake_dispatch(role, prompt, provider, auth, **kwargs):
        tried.append(provider)
        return answers.get(provider, "")

    fixed_points = tmp_path / "fixed-points.jsonl"
    telemetry = tmp_path / "telemetry.jsonl"
    real_dispatch = apr._dispatch_speaker
    apr._dispatch_speaker = fake_dispatch
    try:
        kernel = InnerspeakerGo(
            converge=lambda state, depth: apr._converge_step(
                state, depth, pr_title="t", diff_excerpt="d", role_chains=chains,
            ),
            threshold=apr.AGORA_THRESHOLD,
            max_depth=max_depth,
            fixed_points_path=fixed_points,
            telemetry_path=telemetry,
        )
        result = kernel({})
    finally:
        apr._dispatch_speaker = real_dispatch

    steps = len(telemetry.read_text(encoding="utf-8").splitlines()) if telemetry.exists() else 0
    return tried, result, steps


def _passing_text(kind: str) -> str:
    """Text long enough and marked enough to clear ``AGORA_THRESHOLD``."""
    body = " ".join(["word"] * 140)
    marker = {
        "plain_language": "Imagine a queue of reviewers.",
        "quality": "Alignment with the stated goal is good.",
        "technical": "Concerns: none blocking.",
    }[kind]
    return f"{marker} {body}"


def test_the_deepest_rung_is_seated_when_every_shallower_rung_is_dead(tmp_path):
    """The ladder must exhaust — this is the assertion the fix exists to make true.

    With a five-rung ladder and every rung dead, the loop previously stopped
    after two dispatch attempts per role, so the rungs below never ran at all.
    Stated as "reaches the last rung once the ones before it fail", which holds
    under both the current chain indexing and the corrected indexing in HAL#570
    — an absolute rung number would not, since that PR is precisely what shifts
    the mapping from convergence depth to chain index.
    """
    chains = _ladder(5)
    tried, _result, _steps = _drive(chains, answers={}, tmp_path=tmp_path)

    for role in _ROLES:
        seats = [p for p in tried if p.startswith(role)]
        assert len(seats) >= 4, (
            f"role {role!r} got only {len(seats)} attempts on a 5-rung ladder: {seats!r}"
        )
    deepest = [p for p in tried if p.endswith("-r4")]
    assert len(deepest) == len(_ROLES), (
        f"the deepest rung was never seated for every role: {sorted(tried)!r}"
    )


def test_a_role_that_answered_is_never_dispatched_again(tmp_path):
    """A seat that is filled must not be re-dispatched as the ladder walks on.

    Exercised with one live role and two dead ones so the loop keeps stepping
    after the live role is satisfied — with all three live the threshold ends
    the run immediately and the carry-forward guard is never reached.
    """
    chains = _ladder(4)
    answers = {"quality-r0": _passing_text("quality"), "quality-r1": _passing_text("quality")}
    tried, _result, _steps = _drive(chains, answers, tmp_path)

    quality_seats = [p for p in tried if p.startswith("quality")]
    assert len(quality_seats) == 1, (
        f"an answered role was dispatched more than once: {quality_seats!r}"
    )
    dead_seats = [p for p in tried if p.startswith("plain_language")]
    assert len(dead_seats) > 1, (
        "the loop stopped before the guard could be exercised; the dead role "
        f"was tried only {len(dead_seats)} time(s): {tried!r}"
    )


def test_an_exhausted_ladder_stops_instead_of_spinning_to_the_depth_cap(tmp_path):
    """Termination: once no role has a rung left, the run must end.

    Measured on the kernel's own telemetry rather than on wall time, with a
    depth cap far above the ladder length so that stopping on exhaustion and
    stopping on the cap are distinguishable. No rung may be dispatched twice.
    """
    chains = _ladder(3)
    tried, _result, steps = _drive(chains, answers={}, tmp_path=tmp_path, max_depth=50)

    assert len(tried) == len(set(tried)), f"a rung was dispatched twice: {sorted(tried)!r}"
    assert steps <= 4, (
        f"the loop ran {steps} converge steps for a 3-rung ladder — it did not "
        "stop when the ladder was exhausted"
    )


def test_a_role_with_rungs_left_keeps_going_after_another_role_runs_out(tmp_path):
    """Forward anchor for AGORA#572's surviving concern: unequal chain lengths.

    The reviewer read the ``if futures:`` guard as meaning that two dead rungs
    in a row write no marker and so re-impose the cap. They do not — a dead
    rung is still a dispatch, and the future is submitted before its result is
    known, so ``futures`` is empty only at genuine exhaustion. The reviewer's
    proposed remedy, writing the marker unconditionally, is the mutation that
    makes an exhausted ladder spin to the depth cap.

    What the concern does point at is a case the other tests do not cover: a
    ladder where one role runs out of rungs while another still has several.
    The step then dispatches for the live role only, and must keep going for it
    rather than stopping because two of the three seats contributed nothing.
    """
    chains = {
        "plain_language": [("short-r0", "short-r0", {"scheme": "apikey"})],
        "quality": [
            (f"long-r{i}", f"long-r{i}", {"scheme": "apikey"}) for i in range(4)
        ],
        "technical": [("tiny-r0", "tiny-r0", {"scheme": "apikey"})],
    }
    tried, _result, steps = _drive(chains, answers={}, tmp_path=tmp_path, max_depth=20)

    long_seats = [p for p in tried if p.startswith("long")]
    assert len(long_seats) >= 3, (
        "the role that still had rungs stopped early once the short chains ran "
        f"out: {long_seats!r} (all attempts: {tried!r})"
    )
    assert len(long_seats) == len(set(long_seats)), f"a rung repeated: {long_seats!r}"
    assert steps <= 6, (
        f"the loop ran {steps} converge steps for a 4-rung longest chain — it "
        "did not stop once every chain was exhausted"
    )


def test_the_step_marker_is_not_counted_as_a_speaker(tmp_path, monkeypatch):
    """Bookkeeping the loop writes into the state must not read as a response.

    ``speakers_responded`` counted every truthy value in the converge state, so
    any key the loop carries for its own use inflates the count and makes a
    one-speaker council report two — the number that then decides whether the
    review is quorate. Graded through ``review_pr`` itself rather than through
    a copy of its counting expression, because a copy cannot disagree with the
    thing it was copied from.
    """
    # review_pr builds round-robin chains over a FLAT roster, so every role can
    # reach every provider — only the role decides who answers here.
    roster = [(f"p{i}", f"p{i}", {"scheme": "apikey"}) for i in range(3)]

    monkeypatch.setattr(apr, "_fetch_pr_metadata", lambda pr, repo: ("t", "--- a/x.py\n+++ b/x.py\n"))
    monkeypatch.setattr(
        apr, "_dispatch_speaker",
        lambda role, prompt, provider, auth, **kwargs: (
            _passing_text("quality") if role == "quality" else ""
        ),
    )
    real_kernel = InnerspeakerGo
    monkeypatch.setattr(
        apr, "InnerspeakerGo",
        lambda **kwargs: real_kernel(
            fixed_points_path=tmp_path / "fixed-points.jsonl",
            telemetry_path=tmp_path / "telemetry.jsonl",
            **kwargs,
        ),
    )

    review = apr.review_pr(
        1, "owner/repo", providers=roster,
        dispatch_judge=False, create_defer_issue=False,
    )

    assert review.quality, "the live role never answered; the run proves nothing"
    assert review.speakers_responded == 1, (
        f"one role answered but the council reported {review.speakers_responded}"
    )
