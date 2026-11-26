"""The System One wire format: what a request may hold, and what counts as an answer.

lib/hal/decide/contract.py checks a request before anything is sent and checks
a reply before anything is believed. Each test names the mutation that turns it
red.
"""
from __future__ import annotations

import pytest

from lib.hal.decide.contract import (MAX_CHOICE_OPTIONS, REASONS, DecisionResult, InvalidDecision,
                                     choice, noul, read_answers, score, validate)

GOOD = {"urgent": noul("Is it urgent?", {"true": "Time-sensitive", "false": "Not"}),
        "team": choice("Which team?", {"billing": "Payments", "technical": None}),
        "mood": score("How cross?", ["Calm", "Cross", "Furious"])}


def _options(n: int) -> dict:
    return {f"option-{i}": None for i in range(n)}


def test_the_builders_produce_the_wire_shapes():
    """Mutation: a builder drops the type or renames a field."""
    assert GOOD["urgent"] == {"type": "noul", "instructions": "Is it urgent?",
                              "criteria": {"true": "Time-sensitive", "false": "Not"}}
    assert GOOD["team"]["criteria"] == {"billing": "Payments", "technical": None}
    assert GOOD["mood"] == {"type": "score", "instructions": "How cross?",
                            "criteria": ["Calm", "Cross", "Furious"]}
    assert "criteria" not in noul("Plain?")


@pytest.mark.parametrize("state", ["text", {"thread": []}, [1, 2]])
def test_a_legal_request_passes(state):
    validate(state, GOOD)


@pytest.mark.parametrize("state, questions", [
    (None, GOOD),
    (42, GOOD),
    ("x", {}),
    ("x", ["not", "a", "map"]),
    ("x", {"": noul("?")}),
    ("x", {"q": {"type": "essay", "instructions": "?"}}),
    ("x", {"q": {"type": "noul"}}),
    ("x", {"q": noul("?", {"maybe": "?"})}),
    ("x", {"q": {"type": "choice", "criteria": _options(2)}}),
    ("x", {"q": choice("?", _options(1))}),
    ("x", {"q": choice("?", _options(MAX_CHOICE_OPTIONS + 1))}),
    ("x", {"q": choice("?", {"": None, "b": None})}),
    ("x", {"q": {"type": "score", "criteria": ["a", "b"]}}),
    ("x", {"q": score("?", ["only one"])}),
    ("x", {"q": score("?", [str(i) for i in range(11)])}),
    ("x", {"q": {"type": "score", "instructions": "?", "criteria": {"a": 1}}}),
], ids=["state-none", "state-int", "no-questions", "questions-list", "empty-id", "unknown-type",
        "noul-bare", "noul-bad-criteria", "choice-no-instructions", "choice-one-option",
        "choice-256-options", "choice-empty-option", "score-no-instructions", "score-one-level",
        "score-11-levels", "score-not-a-list"])
def test_an_illegal_request_is_refused_before_any_send(state, questions):
    """Mutation: drop the matching check in ``validate``."""
    with pytest.raises(InvalidDecision):
        validate(state, questions)


def test_a_noul_may_carry_criteria_alone():
    """The live API accepts a Noul with criteria and no instructions (measured 2026-10-10)."""
    validate("x", {"q": {"type": "noul", "criteria": {"true": "yes", "false": "no"}}})


def test_the_limits_are_inclusive():
    """Mutation: an off-by-one at 255 options or 10 levels."""
    validate("x", {"c": choice("?", _options(MAX_CHOICE_OPTIONS)),
                   "s": score("?", [str(i) for i in range(10)])})


ANSWERS = {"urgent": {"type": "noul", "noul": 0.97},
           "team": {"type": "choice", "choice": "billing", "confidence": 0.8,
                    "probabilities": {"billing": 0.9, "technical": 0.1}},
           "mood": {"type": "score", "score": 1.05, "confidence": 0.92,
                    "legend": {"0": "Calm", "1": "Cross", "2": "Furious"},
                    "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05}}}


def test_a_full_answer_is_read():
    out = read_answers(GOOD, ANSWERS)
    assert out == ANSWERS and out is not ANSWERS


def test_a_checked_answer_cannot_change_after_the_check():
    """Mutation: a shallow copy, which leaves nested probabilities shared with the reply.

    AGORA's concern on HAL#724: the checked-answer guarantee must hold at every depth."""
    reply = {key: {**value, "probabilities": dict(value.get("probabilities", {}))}
             for key, value in ANSWERS.items()}
    out = read_answers(GOOD, reply)
    reply["team"]["probabilities"]["billing"] = 0.0
    assert out["team"]["probabilities"]["billing"] == 0.9


def _with(qid, **change):
    return {**ANSWERS, qid: {**ANSWERS[qid], **change}}


@pytest.mark.parametrize("answers", [
    {key: ANSWERS[key] for key in ("urgent", "team")},
    {**ANSWERS, "extra": {"type": "noul", "noul": 0.5}},
    _with("urgent", type="choice"),
    _with("urgent", noul=1.2),
    _with("urgent", noul=True),
    _with("team", choice="sales"),
    _with("team", probabilities={"billing": 1.0}),
    _with("team", confidence=1.5),
    _with("mood", probabilities={"0": 1.0}),
    _with("mood", score=2.5),
    _with("mood", score=True),
    "not a map",
], ids=["missing-answer", "extra-answer", "wrong-type", "noul-over-one", "noul-bool",
        "choice-not-offered", "choice-partial-probabilities", "confidence-over-one",
        "score-partial-levels", "score-out-of-range", "score-bool", "not-a-map"])
def test_a_reply_that_does_not_answer_is_refused(answers):
    """Mutation: drop the matching check in ``read_answers``."""
    with pytest.raises(InvalidDecision):
        read_answers(GOOD, answers)


def test_an_unavailable_result_carries_only_known_reasons():
    """Mutation: accept any reason string (callers branch on the closed set)."""
    with pytest.raises(ValueError):
        DecisionResult.unavailable("it broke")
    result = DecisionResult.unavailable("no_key", "x" * 500)
    assert result.reason in REASONS and len(result.detail) == 300
    assert result.to_dict()["status"] == "unavailable" and result.to_dict()["drifted"] is False


def test_drift_needs_an_answer_a_pin_and_a_different_version():
    """Mutation: ``drifted`` true without a pin, or on an unavailable result."""
    assert DecisionResult(ok=True, model="jev-1.14.0", pinned="jev-1.13.0").drifted
    assert not DecisionResult(ok=True, model="jev-1.13.0", pinned="jev-1.13.0").drifted
    assert not DecisionResult(ok=True, model="jev-1.14.0").drifted
    assert not DecisionResult(ok=False, model="jev-1.14.0", pinned="jev-1.13.0").drifted
