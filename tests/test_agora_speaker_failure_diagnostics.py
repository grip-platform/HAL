"""Goodhart-proof tests for AGORA per-speaker rung failure diagnostics.

Guards the fix for the "empty speaker reads as flakiness" gap named in HAL#551.

Before this fix, a speaker whose whole provider ladder failed surfaced in the
PR comment as one bare line -- ``Speakers returned empty: quality`` -- followed
by a list of GUESSES ("a provider hit a rate limit", "the diff begins with
binary content"). The real reason existed: ``_call_happi_ex`` returns it, and
the workflow log carried ``HTTP 413: Request too large ... Limit 8000,
Requested 40175`` and ``TimeoutExpired ... timeout=90s``. AGORA threw it away by
calling the lossy ``_call_happi`` wrapper, so a reader of the PR could not tell
a deterministic size failure from a transient blip -- and re-ran instead of
fixing anything (HAL#546, #548 one re-run each; HAL#549 three, never green).

Each test fails under an obvious mutation of the code it guards:

  D1 -- ``_dispatch_speaker`` RECORDS the rung's role, provider and reason when
        the call comes back empty. Reverting to ``_call_happi`` (which discards
        the reason) or dropping the accumulator fails.
  D2 -- the FAILED banner NAMES each dead rung and its reason. Removing the
        section, or printing the role without the provider/reason, fails.
  D3 -- a multi-kilobyte provider error (the real groq 413 body) is TRIMMED,
        while the load-bearing numbers survive. Dropping the trim fails on
        length; trimming to nothing fails on content.
  D4 -- WIRING: the real orchestrator threads the accumulator from dispatch to
        the rendered comment, network-free. Fails if ``review_pr`` stops
        passing it.
  D5 -- a review with no failed rungs renders NO diagnostics section
        (backwards-compatible; existing comments are unchanged).
  D6 -- the accumulator is mutated only while ``_ATTEMPTS_LOCK`` is held.
        Removing the lock fails. Note the deliberate asymmetry here: the
        companion many-threads test is a SMOKE check, not a mutation guard --
        it still passed with the lock deleted, which is exactly why the
        lock-held assertion exists beside it. See its docstring.

Goodhart guards: assertions check rendered substrings and recorded field values
from a fixture whose numbers are taken from the real HAL#549 workflow log --
never call counts, never "a section exists".
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

# Resolve HAL repo root so ``lib.*`` imports work regardless of pytest cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.environ.setdefault("HAL_HOME", str(_REPO_ROOT))

from lib.agora_pr_review import (  # noqa: E402
    AgoraReview,
    SpeakerFailure,
    _dispatch_speaker,
)
import lib.agora_pr_review as agora  # noqa: E402


# The verbatim groq refusal from run 34223979811's predecessor on HAL#549 --
# 2026-09-07T21:26:38Z, quality speaker, second and last rung of its ladder.
_GROQ_413 = (
    'HTTP 413: {"error":{"message":"Request too large for model '
    '`openai/gpt-oss-120b` in organization `org_01kkxhw01sfsmv81qfbjcqezhk` '
    'service tier `on_demand` on tokens per minute (TPM): Limit 8000, '
    'Requested 40175, please reduce your message size and try again. Need '
    'more tokens? Upgrade to Dev Tier today at '
    'https://console.groq.com/settings/billing","type":"tokens",'
    '"code":"rate_limit_exceeded"}}'
)


def _failed_review(**kw) -> AgoraReview:
    """An invalid review carrying the two dead rungs HAL#549 actually hit."""
    base = dict(
        pr_number=549,
        repo="CodeTonight-SA/HAL",
        plain_language="Speaker A said something real about the diff.",
        quality="",
        technical="Speaker C said something real about the diff.",
        speakers_responded=2,
        speakers_total=3,
        is_valid=False,
        validity_reasons=["Speakers returned empty: quality"],
        diff_files=["lib/hal/happi/events.py"],
        speaker_failures=[
            SpeakerFailure("quality", "deepseek/deepseek-v4-pro",
                           "timeout"),
            SpeakerFailure("quality", "groq/openai/gpt-oss-120b", _GROQ_413),
            SpeakerFailure("quality", "deepseek/deepseek-v4-flash",
                           "missing_binary"),
        ],
    )
    base.update(kw)
    return AgoraReview(**base)


# ---------------------------------------------------------------------------
# D1 -- the reason is captured at dispatch, not discarded
# ---------------------------------------------------------------------------

def test_dispatch_speaker_records_the_dead_rung_reason() -> None:
    """An empty speaker records role + provider + the reason the call failed."""
    attempts: list[SpeakerFailure] = []
    with patch.object(agora, "_build_envelope", return_value={"id": "x", "flags": {}}), \
         patch.object(agora, "_call_happi_ex", return_value=("", _GROQ_413)):
        out = _dispatch_speaker(
            "quality", "prompt", "groq/openai/gpt-oss-120b", {"scheme": "apikey"},
            attempts=attempts,
        )

    assert out == ""
    assert len(attempts) == 1
    recorded = attempts[0]
    assert recorded.role == "quality"
    assert recorded.provider == "groq/openai/gpt-oss-120b"
    # The SPECIFIC numbers must survive: they are what distinguishes a
    # deterministic size failure from a transient one.
    assert "Limit 8000" in recorded.reason
    assert "Requested 40175" in recorded.reason


def test_dispatch_speaker_records_nothing_when_the_rung_answers() -> None:
    """A rung that returns text records no failure (no false diagnostics)."""
    attempts: list[SpeakerFailure] = []
    with patch.object(agora, "_build_envelope", return_value={"id": "x", "flags": {}}), \
         patch.object(agora, "_call_happi_ex", return_value=("### A real review", None)):
        out = _dispatch_speaker(
            "quality", "prompt", "deepseek/deepseek-v4-pro", {"scheme": "apikey"},
            attempts=attempts,
        )

    assert out == "### A real review"
    assert attempts == []


def test_dispatch_speaker_records_the_exception_that_killed_the_rung() -> None:
    """A raising rung is recorded too -- an exception is not silence."""
    attempts: list[SpeakerFailure] = []
    with patch.object(agora, "_build_envelope", side_effect=RuntimeError("no auth token")):
        out = _dispatch_speaker(
            "technical", "prompt", "xai/grok-3", {"scheme": "apikey"},
            attempts=attempts,
        )

    assert out == ""
    assert len(attempts) == 1
    assert attempts[0].role == "technical"
    assert attempts[0].provider == "xai/grok-3"
    assert "no auth token" in attempts[0].reason


# ---------------------------------------------------------------------------
# D2 -- the failure comment names the rung and the reason
# ---------------------------------------------------------------------------

def test_failed_banner_names_every_dead_rung_with_its_reason() -> None:
    """The comment says WHICH provider died and WHY -- not just which role."""
    md = _failed_review().to_markdown()

    # Both rungs of the quality ladder are named, in order.
    assert "deepseek/deepseek-v4-pro" in md
    assert "groq/openai/gpt-oss-120b" in md
    # Each rung carries its DISTINCT cause. The negative assertion is the guard:
    # the former conflated string "timeout_or_missing_binary" CONTAINS both
    # "timeout" and "missing_binary", so the positive assertions alone pass even
    # under the very regression this exists to catch. AGORA found that on #600.
    assert "timeout" in md
    assert "missing_binary" in md
    assert "timeout_or_missing_binary" not in md
    assert "Limit 8000" in md
    assert "Requested 40175" in md
    # The role is still attributed, so a reader knows which speaker went dark.
    assert "quality" in md


def test_failed_banner_omits_the_section_when_no_rung_failed() -> None:
    """No recorded failures ⇒ no diagnostics section (backwards-compatible)."""
    md = _failed_review(speaker_failures=[]).to_markdown()

    assert "deepseek/deepseek-v4-pro" not in md
    assert "Dead speaker rungs" not in md
    # The pre-existing banner content is untouched.
    assert "Speakers returned empty: quality" in md
    assert "**Speakers responded**: 2/3" in md


# ---------------------------------------------------------------------------
# D3 -- a kilobyte-scale provider error is trimmed, not dumped
# ---------------------------------------------------------------------------

def test_long_provider_error_is_trimmed_but_keeps_the_numbers() -> None:
    """A 4KB provider body must not swamp the comment, nor lose its evidence."""
    noisy = _GROQ_413 + (" filler" * 800)
    md = _failed_review(
        speaker_failures=[SpeakerFailure("quality", "groq/openai/gpt-oss-120b", noisy)]
    ).to_markdown()

    assert "Limit 8000" in md          # the load-bearing evidence survives
    assert "filler filler filler" not in md   # the padding does not
    assert len(md) < 6000             # the whole comment stays readable


def test_reason_trim_flattens_newlines() -> None:
    """A multi-line reason cannot break the markdown list it renders into."""
    md = _failed_review(
        speaker_failures=[
            SpeakerFailure("quality", "deepseek/deepseek-v4-pro",
                           "line one\nline two\nline three")
        ]
    ).to_markdown()

    assert "line one line two line three" in md


# ---------------------------------------------------------------------------
# D4 -- WIRING: the orchestrator actually threads the accumulator through
# ---------------------------------------------------------------------------

def test_review_pr_threads_recorded_failures_into_the_comment() -> None:
    """Network-free end-to-end: a dead rung reaches the rendered comment."""
    diff = (
        "diff --git a/lib/hal/happi/events.py b/lib/hal/happi/events.py\n"
        "--- a/lib/hal/happi/events.py\n"
        "+++ b/lib/hal/happi/events.py\n"
        "@@ -1,2 +1,3 @@\n"
        "+def emit_event(name):\n"
        "+    return name\n"
    )

    def _dead_rung(role, prompt, happi_provider, auth, *a, attempts=None, **kw):
        if attempts is not None:
            attempts.append(SpeakerFailure(role, happi_provider, _GROQ_413))
        return ""

    from tests._agora_roster import seated_roster

    # The live roster seats no rung on a keyless CI runner, so the dead-rung
    # stub would never be called there (HAL#885). Patching the module's roster,
    # rather than passing providers=, keeps review_pr's DEFAULT-roster branch
    # under this wiring test.
    with patch.object(agora, "_fetch_pr_metadata", return_value=("A big PR", diff)), \
         patch.object(agora, "_AGORA_SPEAKER_ROSTER", seated_roster()), \
         patch.object(agora, "_dispatch_speaker", side_effect=_dead_rung):
        review = agora.review_pr(549, "CodeTonight-SA/HAL", dispatch_judge=False)

    assert review.is_valid is False
    assert review.speaker_failures, "orchestrator dropped the recorded failures"
    md = review.to_markdown()
    assert "Requested 40175" in md


# ---------------------------------------------------------------------------
# D6 -- the concurrency falsifier AGORA's Speaker C asked for on HAL#562
# ---------------------------------------------------------------------------

def test_record_failure_holds_the_lock_while_mutating() -> None:
    """Speaker C's concern, answered deterministically.

    The objection to "``list.append`` is atomic under the GIL" was that it is a
    property of one implementation, not of the language. The code no longer
    relies on it. This asserts the guarantee DIRECTLY -- the accumulator is
    mutated only while the lock is held -- because it is the only form of the
    assertion that reliably fails when the lock is removed.

    The obvious alternative, hammering the accumulator from many threads and
    counting entries, was written first and REJECTED: it still passed with the
    lock deleted AND the append rewritten as a read-modify-write, because
    CPython's switch interval almost never lands inside so short an operation.
    A test that cannot be observed failing is an oracle, not a test.
    """
    seen: list[bool] = []

    class _WatchedLock:
        def __init__(self) -> None:
            self.held = False
            self.acquisitions = 0

        def __enter__(self):
            self.held = True
            self.acquisitions += 1
            return self

        def __exit__(self, *exc):
            self.held = False
            return False

    watched = _WatchedLock()

    class _Spy(list):
        def append(self, item):  # records whether the lock was held at mutation
            seen.append(watched.held)
            super().append(item)

    spy = _Spy()
    with patch.object(agora, "_ATTEMPTS_LOCK", watched):
        agora._record_failure(spy, SpeakerFailure("quality", "p", "r"))
        agora._record_failure(spy, SpeakerFailure("technical", "p", "r"))

    assert seen == [True, True], "accumulator was mutated outside the lock"
    assert watched.acquisitions == 2
    assert watched.held is False, "lock was not released"
    assert len(spy) == 2


def test_concurrent_failures_all_arrive() -> None:
    """Smoke check under real threads: nothing is lost, nothing is corrupted.

    Weaker than the lock assertion above by design -- see its docstring for why
    this shape alone is not sufficient. It is kept because it exercises the real
    call under real contention, which the instrumented test does not.
    """
    import threading

    attempts: list[SpeakerFailure] = []
    threads, per_thread = 32, 25
    barrier = threading.Barrier(threads)

    def hammer(worker: int) -> None:
        barrier.wait()  # maximise real overlap rather than hoping for it
        for i in range(per_thread):
            agora._record_failure(
                attempts, SpeakerFailure(f"role{worker}", "p", f"reason{i}")
            )

    workers = [threading.Thread(target=hammer, args=(w,)) for w in range(threads)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()

    assert len(attempts) == threads * per_thread
    assert all(isinstance(f, SpeakerFailure) for f in attempts)
    assert len({f.role for f in attempts}) == threads


def test_record_failure_ignores_a_missing_accumulator() -> None:
    """No accumulator means no crash (callers that do not want diagnostics)."""
    agora._record_failure(None, SpeakerFailure("quality", "p", "r"))  # must not raise
