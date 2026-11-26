"""The Synthesis Ladder's shared contract suite (Liskov): every rung, same tests.

Each rung shape gets the same task and must (1) return a schema-valid Synthesis
of its own effort, (2) turn a provider failure into decided=False without
raising, (3) never seat a vendor the task excludes, and (4) estimate at least
what it then spends. Rung-specific behaviour (verdict parsing, tallies, the
registry adapter) follows the shared suite.
"""
import pytest

from lib.hal.synthesis import (
    Billing,
    ChatReply,
    DeterministicRung,
    Effort,
    Privacy,
    SeatRung,
    SeatSpec,
    Synthesis,
    SynthesisTask,
    contract_violations,
    parse_verdict,
    registry_chat,
)
from lib.hal.synthesis.rungs import upper_bound_tokens

VERDICTS = ("MERGE", "REQUEST_CHANGES", "DEFER_CHANGES")
TASK = SynthesisTask(question="Should this PR merge?", evidence=("diff: +1 line",),
                     verdicts=VERDICTS)


def _seat(model, vendor, billing=Billing.POSTPAID, role="judge", max_tokens=512):
    return SeatSpec(model=model, vendor=vendor, role=role, billing=billing,
                    system=f"You are the {role}.", max_tokens=max_tokens)


def _worst_case_chat(text="VERDICT: MERGE\nLooks right."):
    """A provider that bills the most tokens its prompt and cap allow."""
    def chat(address, system, prompt, max_tokens):
        return ChatReply(text=text, tokens_in=upper_bound_tokens(system) + upper_bound_tokens(prompt),
                         tokens_out=max_tokens)
    return chat


def _failing_chat(address, system, prompt, max_tokens):
    raise RuntimeError("provider down")


SEAT_RUNGS = {
    "low-local": lambda chat: SeatRung(Effort.LOW, [
        _seat("ollama/qwen3:8b", "alibaba", Billing.LOCAL)], chat),
    "medium-postpaid": lambda chat: SeatRung(Effort.MEDIUM, [
        _seat("gemini/gemini-2.5-flash-lite", "google")], chat),
    "high-council": lambda chat: SeatRung(Effort.HIGH, [
        _seat("gemini/gemini-2.5-flash-lite", "google", role="tree-of-thought"),
        _seat("moonshot/kimi-k2.6", "moonshot", Billing.SUBSCRIPTION, role="verifier"),
        _seat("openai/gpt-5.6-luna", "openai", Billing.PREPAID, role="devil")], chat),
}
ALL_RUNGS = {**SEAT_RUNGS,
             "deterministic": lambda chat: DeterministicRung([lambda t: ("MERGE", "gates clean")])}


@pytest.mark.parametrize("name", sorted(ALL_RUNGS))
def test_a_rung_returns_a_valid_synthesis_of_its_effort(name):
    rung = ALL_RUNGS[name](_worst_case_chat())
    result = rung.synthesise(TASK, None)
    assert contract_violations(result, TASK, rung.effort) == []
    assert result.decided and result.verdict == "MERGE"


@pytest.mark.parametrize("name", sorted(SEAT_RUNGS))
def test_a_provider_failure_is_undecided_not_raised(name):
    result = SEAT_RUNGS[name](_failing_chat).synthesise(TASK, None)
    assert isinstance(result, Synthesis) and not result.decided
    assert all("provider down" in s.note for s in result.seats)


def test_a_failing_check_is_a_note_not_a_crash():
    def broken(task):
        raise ValueError("gate exploded")
    result = DeterministicRung([broken]).synthesise(TASK, None)
    assert not result.decided and "gate exploded" in result.answer


@pytest.mark.parametrize("name", sorted(SEAT_RUNGS))
def test_an_excluded_vendor_never_sits(name):
    rung = SEAT_RUNGS[name](_worst_case_chat())
    every = frozenset({"google", "moonshot", "openai", "alibaba"})
    shut_out = SynthesisTask(question="q", verdicts=VERDICTS, excluded_vendors=every)
    assert not rung.available(shut_out)
    assert rung.synthesise(shut_out, None).seats == ()
    no_google = SynthesisTask(question="q", verdicts=VERDICTS, excluded_vendors=frozenset({"google"}))
    assert "google" not in {s.vendor for s in rung.synthesise(no_google, None).seats}


@pytest.mark.parametrize("name", sorted(SEAT_RUNGS))
def test_the_estimate_bounds_what_is_spent(name):
    # The worst-case provider bills one token per prompt byte and the whole output
    # cap, and the prior is the longest the prompt carries, in four-byte characters.
    rung = SEAT_RUNGS[name](_worst_case_chat())
    longest_prior = Synthesis(effort=Effort.DETERMINISTIC, decided=False, answer="😀" * 5_000)
    spent = rung.synthesise(TASK, longest_prior).cost_usd
    assert spent <= rung.estimate(TASK) + 1e-12


def test_money_is_only_counted_for_paid_seats():
    result = SEAT_RUNGS["high-council"](_worst_case_chat()).synthesise(TASK, None)
    paid = {s.billing: s.cost_usd > 0 for s in result.seats}
    assert paid == {Billing.POSTPAID: True, Billing.SUBSCRIPTION: False, Billing.PREPAID: True}


def test_free_tier_seats_never_sit_and_local_only_tasks_stay_local():
    chat = _worst_case_chat()
    free = SeatRung(Effort.MEDIUM, [_seat("gemini/gemini-2.5-flash", "google", Billing.FREE_TIER)], chat)
    assert not free.available(TASK)
    private = SynthesisTask(question="q", verdicts=VERDICTS, privacy=Privacy.LOCAL_ONLY)
    assert SEAT_RUNGS["low-local"](chat).available(private)
    assert not SEAT_RUNGS["medium-postpaid"](chat).available(private)


def test_a_seat_whose_presence_probe_fails_is_not_ready():
    def probe():
        raise OSError("keychain locked")
    spec = SeatSpec(model="gemini/gemini-2.5-flash-lite", vendor="google", role="judge",
                    billing=Billing.POSTPAID, system="s", ready=probe)
    assert not SeatRung(Effort.MEDIUM, [spec], _worst_case_chat()).available(TASK)


def test_a_council_split_two_to_one_decides_with_partial_agreement():
    replies = iter(["VERDICT: MERGE", "VERDICT: MERGE", "VERDICT: REQUEST CHANGES — a bug"])

    def chat(address, system, prompt, max_tokens):
        return ChatReply(text=next(replies), tokens_in=10, tokens_out=5)
    result = SEAT_RUNGS["high-council"](chat).synthesise(TASK, None)
    assert result.decided and result.verdict == "MERGE"
    assert result.agreement == pytest.approx(2 / 3)
    assert result.dissent == ("openai/gpt-5.6-luna: REQUEST_CHANGES",)


@pytest.mark.parametrize("text, verdict", [
    ("VERDICT: MERGE", "MERGE"),
    ("**Verdict:** request changes, because the test is wrong", "REQUEST_CHANGES"),
    ("> VERDICT: DEFER CHANGES\nreasons", "DEFER_CHANGES"),
    ("VERDICT: BLOCK", ""),
    ("no verdict line at all", ""),
    ("VERDICT: BLOCK\nVERDICT: MERGE", "MERGE"),
])
def test_parse_verdict(text, verdict):
    assert parse_verdict(text, VERDICTS) == verdict


def test_a_short_token_never_shadows_a_longer_one():
    assert parse_verdict("VERDICT: MERGE WITH CHANGES", ("MERGE", "MERGE_WITH_CHANGES")) == "MERGE_WITH_CHANGES"


def test_the_registry_adapter_uses_the_registrys_real_signature():
    # ProviderRegistry.chat is (model_address, prompt, **kwargs). CouncilRunner
    # called it with messages=/model= and every seat raised TypeError, silently.
    calls = []

    class Usage:
        prompt_tokens, completion_tokens = 120, 30

    class Response:
        text, usage = "VERDICT: MERGE", Usage()

    class Registry:
        def chat(self, model_address, prompt, **kwargs):
            calls.append((model_address, prompt, kwargs))
            return Response()

    reply = registry_chat(Registry())("gemini/gemini-2.5-flash-lite", "be brief", "the prompt", 256)
    assert reply == ChatReply(text="VERDICT: MERGE", tokens_in=120, tokens_out=30)
    assert calls == [("gemini/gemini-2.5-flash-lite", "the prompt",
                      {"system": "be brief", "max_tokens": 256, "temperature": 0.0})]
