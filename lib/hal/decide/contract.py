"""Typed decisions in the System One wire format: requests checked, answers read.

A decision model takes a ``state`` (a string, an object or an array) and a map of
typed questions, and returns one typed answer per question: a Noul (the
probability that a yes/no question is yes), a Choice (one of up to 255 options,
with every option's probability and a confidence) or a Score (a
probability-weighted level over 2 to 10 ordered levels, with a confidence). It
never returns text, so it is not a chat model.

TypeSafe's Jev is the first such model HAL reaches. The wire format is
TypeSafe's, and other vendors have copied it, so nothing here is TypeSafe's
alone. This module builds and checks requests and reads answers. It sends
nothing: ``lib/hal/decide/typesafe.py`` does, behind GRIP's redaction shield
and HAL's spend guard.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

KINDS = ("noul", "choice", "score")
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS, MAX_SCORE_LEVELS = 2, 10

#: Why a decision came back without answers. A closed set, so a caller can
#: branch on it and a log line can carry it without carrying a body.
REASONS = frozenset({
    "invalid_request",      # the request breaks the format (checked before any send)
    "unknown_model",        # not in config/decision_models.yaml
    "catalog_unavailable",  # config/decision_models.yaml cannot be used
    "spend_refused",        # the spend guard said no: not opted in, no cap, or the cap reached
    "no_key",               # no usable API key in the keychain or the environment
    "shield_unavailable",   # GRIP's decision shield is missing, so nothing was sent
    "shield_refused",       # the shield could not scrub the body, or a key would have changed
    "over_budget",          # too large for the provider's ceiling or the model's token budget
    "circuit_open",         # recent sends kept failing, so this one was not tried
    "auth_failed",          # the provider rejected the key
    "rate_limited",         # 429 after the retries
    "overloaded",           # 529 after the retries
    "server_error",         # another 5xx after the retries
    "timeout",              # no answer inside the time budget
    "unreachable",          # the connection itself failed
    "bad_response",         # a reply that does not answer the questions asked
    "provider_error",       # the provider raised where it should have answered
})


class InvalidDecision(ValueError):
    """A request or reply the format does not allow. The message names the question, not its text."""


@dataclass(frozen=True)
class ProviderSpec:
    """One decision provider, as config/decision_models.yaml declares it."""

    name: str
    endpoint: str
    key_source: str
    spend_source: str
    billing: str
    usd_per_million_input: Decimal
    usd_per_million_output: Decimal
    max_request_bytes: int


@dataclass(frozen=True)
class DecisionModel:
    """A decision model's id, the provider serving it, and the version it is pinned to."""

    id: str
    provider: str
    pinned: str = ""


@dataclass(frozen=True)
class DecisionResult:
    """What a decision returned. ``ok`` means every question has a checked answer."""

    ok: bool
    reason: str = ""
    detail: str = ""
    answers: dict = field(default_factory=dict)
    model: str = ""
    pinned: str = ""
    usage: dict = field(default_factory=dict)
    cost_usd: str = ""
    latency_ms: int = 0
    attempts: int = 0
    provider: str = ""
    #: True only where a measured calibration is on file for the question asked.
    #: None is yet, so this is False for every answer today: a probability here
    #: is the model's own, not a measured error rate.
    calibrated: bool = False

    @classmethod
    def unavailable(cls, reason: str, detail: str = "", **extra: Any) -> "DecisionResult":
        if reason not in REASONS:
            raise ValueError(f"unknown reason {reason!r}")
        return cls(ok=False, reason=reason, detail=detail[:300], **extra)

    @property
    def drifted(self) -> bool:
        """The answering version is not the pinned one, so a tuned threshold may not hold."""
        return bool(self.ok and self.pinned and self.model != self.pinned)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["status"] = "answered" if self.ok else "unavailable"
        out["drifted"] = self.drifted
        return out


@runtime_checkable
class DecisionProvider(Protocol):
    """The narrowest contract a decision provider meets: answer, or say why not."""

    name: str

    def decide(self, model: DecisionModel, state: Any,
               questions: Mapping[str, dict]) -> DecisionResult:
        ...


# --- building questions ------------------------------------------------------

def noul(instructions: Any, criteria: Optional[Mapping] = None) -> dict:
    """A yes/no question. ``criteria`` may say what a yes (``"true"``) and a no (``"false"``) mean."""
    question = {"type": "noul", "instructions": instructions}
    if criteria is not None:
        question["criteria"] = dict(criteria)
    return question


def choice(instructions: Any, options: Mapping) -> dict:
    """Pick one of ``options``: option -> its description, or None."""
    return {"type": "choice", "instructions": instructions, "criteria": dict(options)}


def score(instructions: Any, levels: list) -> dict:
    """Rate along 2 to 10 ordered ``levels``, lowest first."""
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


# --- checking requests -------------------------------------------------------

def validate(state: Any, questions: Any) -> None:
    """Raise InvalidDecision unless ``(state, questions)`` is a request the format allows."""
    if not isinstance(state, (str, dict, list)):
        raise InvalidDecision("state must be a string, an object or an array")
    if not isinstance(questions, Mapping) or not questions:
        raise InvalidDecision("questions must be a non-empty map of question id to question")
    for qid, question in questions.items():
        _check_question(qid, question)


def _check_question(qid: Any, question: Any) -> None:
    if not isinstance(qid, str) or not qid:
        raise InvalidDecision("every question id must be a non-empty string")
    if not isinstance(question, Mapping) or question.get("type") not in KINDS:
        raise InvalidDecision(f"question {qid!r} must have a type in {KINDS}")
    _CHECKS[question["type"]](qid, question)


def _check_noul(qid: str, question: Mapping) -> None:
    # The live API accepts instructions or criteria (measured 2026-10-10:
    # "Noul question must have criteria or instructions").
    criteria = question.get("criteria")
    if not question.get("instructions") and not criteria:
        raise InvalidDecision(f"noul {qid!r} needs instructions or criteria")
    if criteria is not None and (not isinstance(criteria, Mapping)
                                 or not set(criteria) <= {"true", "false"}):
        raise InvalidDecision(f"noul {qid!r} criteria may only describe 'true' and 'false'")


def _check_choice(qid: str, question: Mapping) -> None:
    options = question.get("criteria")
    if not question.get("instructions"):
        raise InvalidDecision(f"choice {qid!r} needs instructions")
    if not isinstance(options, Mapping) or not 2 <= len(options) <= MAX_CHOICE_OPTIONS:
        raise InvalidDecision(f"choice {qid!r} needs 2 to {MAX_CHOICE_OPTIONS} options")
    if not all(isinstance(option, str) and option for option in options):
        raise InvalidDecision(f"choice {qid!r} options must be non-empty strings")


def _check_score(qid: str, question: Mapping) -> None:
    levels = question.get("criteria")
    if not question.get("instructions"):
        raise InvalidDecision(f"score {qid!r} needs instructions")
    if not isinstance(levels, list) or not MIN_SCORE_LEVELS <= len(levels) <= MAX_SCORE_LEVELS:
        raise InvalidDecision(f"score {qid!r} needs {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels")


_CHECKS = {"noul": _check_noul, "choice": _check_choice, "score": _check_score}


# --- reading answers ---------------------------------------------------------

def read_answers(questions: Mapping[str, dict], answers: Any) -> dict:
    """``answers`` checked against ``questions``: the same ids, each the type asked, each in range.

    Raises InvalidDecision on any mismatch, so a malformed reply is never passed off as a decision.
    """
    if not isinstance(answers, Mapping) or set(answers) != set(questions):
        raise InvalidDecision("the answers do not match the questions asked")
    for qid, question in questions.items():
        _check_answer(qid, question, answers[qid])
    # A deep copy: nothing the provider still holds can change an answer after
    # it was checked, nested probabilities included.
    return copy.deepcopy({qid: answers[qid] for qid in questions})


def _check_answer(qid: str, question: Mapping, answer: Any) -> None:
    kind = question["type"]
    if not isinstance(answer, Mapping) or answer.get("type") != kind:
        raise InvalidDecision(f"answer {qid!r} is not a {kind}")
    _ANSWER_CHECKS[kind](qid, question, answer)


def _probability(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= value <= 1.0


def _check_noul_answer(qid: str, question: Mapping, answer: Mapping) -> None:
    if not _probability(answer.get("noul")):
        raise InvalidDecision(f"answer {qid!r} has no probability between 0 and 1")


def _check_choice_answer(qid: str, question: Mapping, answer: Mapping) -> None:
    options = set(question["criteria"])
    probabilities = answer.get("probabilities")
    if answer.get("choice") not in options or not isinstance(probabilities, Mapping) \
            or set(probabilities) != options:
        raise InvalidDecision(f"answer {qid!r} does not choose among the options asked")
    if not all(_probability(p) for p in probabilities.values()) \
            or not _probability(answer.get("confidence")):
        raise InvalidDecision(f"answer {qid!r} has a probability out of range")


def _check_score_answer(qid: str, question: Mapping, answer: Mapping) -> None:
    levels = {str(index) for index in range(len(question["criteria"]))}
    probabilities, value = answer.get("probabilities"), answer.get("score")
    if not isinstance(probabilities, Mapping) or set(probabilities) != levels \
            or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidDecision(f"answer {qid!r} does not rate along the levels asked")
    if not 0 <= value <= len(levels) - 1 or not _probability(answer.get("confidence")):
        raise InvalidDecision(f"answer {qid!r} has a value out of range")


_ANSWER_CHECKS = {"noul": _check_noul_answer, "choice": _check_choice_answer,
                  "score": _check_score_answer}
