"""CarryOverVerifier — fact-retention check across RAILLM provider switches.

W2 of the RAILLM ultrado plan (Gen 481). Detects information loss when
RAILLMSentinel fails over from one provider to another. The verifier asks a
fact-retention question about content present in the thread BEFORE the switch
and checks whether the assistant response AFTER the switch retains it.

Architecture::

    RAILLMSentinel.dispatch_with_history
        `-- on provider switch, optionally invokes:
              CarryOverVerifier.verify(thread_before, response_after)
                |-- extract_facts(thread_before)        # heuristic
                |-- verifier_fn(thread, response, facts) # broly / custom judge
                `-- return CarryOverReport(confidence, retained, lost)

The verifier function is pluggable: production uses a broly council, tests
inject deterministic mocks. This separation keeps unit tests hermetic while
allowing the plan's >=0.95 recall target to be measured against real LLM
judges in integration tests (W4).

Hypothesis H048 (registered 2026-04-16): detection_recall >= 0.95 across
synthetic loss-scenario suite.

Falsification: extract a scenario where the verifier returns has_loss=False
but a downstream analyst flags genuine loss; that single counter-example
kills the >=0.95 claim below its threshold.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from lib.hal.raillm import ConversationThread, Message
from lib.hal.types import ChatResponse


# ---------------------------------------------------------------------------
# Fact extraction heuristics
# ---------------------------------------------------------------------------


_NUMERIC_RE = re.compile(
    r"(?<![\w.])"
    r"(?:"
    r"\$\d+(?:,\d{3})*(?:\.\d+)?"
    r"|\d+(?:,\d{3})+(?:\.\d+)?"
    r"|\d+\.\d+%?"
    r"|\d+%"
    r"|\d+"
    r")"
    r"(?![\w.])"
)

_PROPER_NOUN_RE = re.compile(r"\b(?:[A-Z][a-z]{2,}\s+){1,4}[A-Z][a-z]{2,}\b")

_QUOTED_RE = re.compile(r'"([^"\n]{3,80})"|\'([^\'\n]{3,80})\'')


def _extract_message_facts(content: str) -> list[str]:
    """Pull candidate facts from a single message body.

    Heuristic: numbers + proper-noun phrases + quoted literals. Deduplicated
    while preserving first-occurrence order so the output is deterministic.
    Does NOT rank by importance; that is the judge's job in verifier_fn.
    """
    facts: list[str] = []
    facts.extend(_NUMERIC_RE.findall(content))
    facts.extend(_PROPER_NOUN_RE.findall(content))
    for match in _QUOTED_RE.findall(content):
        facts.append(match[0] or match[1])
    seen: set[str] = set()
    ordered: list[str] = []
    for f in facts:
        f_norm = f.strip()
        if f_norm and f_norm not in seen:
            seen.add(f_norm)
            ordered.append(f_norm)
    return ordered


def extract_facts(thread: ConversationThread, max_facts: int = 20) -> list[str]:
    """Extract up to max_facts candidate facts from a thread.

    Walks user + assistant messages (system prompts are excluded: they are
    instructions, not retrievable content) and collects candidate facts via
    _extract_message_facts. Truncates to max_facts to bound judge cost.
    """
    collected: list[str] = []
    seen: set[str] = set()
    for msg in thread.messages:
        if msg.role == "system":
            continue
        for fact in _extract_message_facts(msg.content):
            if fact not in seen:
                seen.add(fact)
                collected.append(fact)
                if len(collected) >= max_facts:
                    return collected
    return collected


# ---------------------------------------------------------------------------
# Report + verifier protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CarryOverReport:
    """Outcome of a single post-switch fact-retention check."""

    has_loss: bool
    confidence_score: float
    retained_facts: tuple[str, ...]
    lost_facts: tuple[str, ...]
    verifier_model: str = ""
    notes: str = ""


VerifierFn = Callable[[ConversationThread, ChatResponse, list[str]], dict]


def _default_substring_verifier(
    thread: ConversationThread,
    response: ChatResponse,
    facts: list[str],
) -> dict:
    """Deterministic substring-match verifier when no judge is provided.

    Not as accurate as an LLM judge, but works offline and has zero cost:
    useful as a CI fallback and as a speed-of-light lower bound the LLM
    judge must beat in W4 integration tests.
    """
    text = response.text.lower()
    retained: list[str] = []
    lost: list[str] = []
    for fact in facts:
        if fact.lower() in text:
            retained.append(fact)
        else:
            lost.append(fact)
    total = max(1, len(facts))
    confidence = 1.0 - (len(lost) / total) * 0.5
    return {
        "retained": retained,
        "lost": lost,
        "confidence": confidence,
        "model": "substring-default",
        "notes": "default substring matcher — swap for broly judge in production",
    }


# ---------------------------------------------------------------------------
# CarryOverVerifier
# ---------------------------------------------------------------------------


@dataclass
class CarryOverVerifier:
    """Coordinator for post-switch fact-retention checks.

    Pluggable verifier_fn lets tests inject deterministic judges while
    production wires in a broly council. The default is a substring matcher
    so the module is usable out of the box without provider credentials.
    """

    verifier_fn: VerifierFn = _default_substring_verifier
    max_facts: int = 20

    def verify(
        self,
        thread_before: ConversationThread,
        response_after: ChatResponse,
    ) -> CarryOverReport:
        """Run one fact-retention check after a provider switch."""
        facts = extract_facts(thread_before, max_facts=self.max_facts)
        if not facts:
            return CarryOverReport(
                has_loss=False,
                confidence_score=1.0,
                retained_facts=(),
                lost_facts=(),
                verifier_model="extractor-none",
                notes="no extractable facts in thread",
            )
        result = self.verifier_fn(thread_before, response_after, facts)
        retained = tuple(result.get("retained", []))
        lost = tuple(result.get("lost", []))
        confidence = float(result.get("confidence", 0.0))
        return CarryOverReport(
            has_loss=bool(lost),
            confidence_score=max(0.0, min(1.0, confidence)),
            retained_facts=retained,
            lost_facts=lost,
            verifier_model=str(result.get("model", "")),
            notes=str(result.get("notes", "")),
        )

    def detection_recall(
        self,
        scenarios: list[tuple[ConversationThread, ChatResponse, bool]],
    ) -> float:
        """Compute recall = TP / (TP + FN) over a scenario list.

        Each scenario is (thread, response, expected_has_loss). Recall is
        the hypothesis metric (carryover_detection_recall) and is 1.0 when
        scenarios is empty: documented so callers know how to interpret
        an empty-suite return.
        """
        true_positives = 0
        false_negatives = 0
        for thread, response, expected in scenarios:
            if not expected:
                continue
            report = self.verify(thread, response)
            if report.has_loss:
                true_positives += 1
            else:
                false_negatives += 1
        denom = true_positives + false_negatives
        if denom == 0:
            return 1.0
        return true_positives / denom


# ---------------------------------------------------------------------------
# RAILLMSentinel glue — optional hook
# ---------------------------------------------------------------------------


@dataclass
class ProviderSwitchRecord:
    """One entry in a RAILLM failover audit trail."""

    from_provider: str
    to_provider: str
    report: CarryOverReport
    thread_token_estimate: int = 0
    trigger: str = ""


@dataclass
class CarryOverAuditLog:
    """In-memory audit log of switch+verify events during a session."""

    entries: list[ProviderSwitchRecord] = field(default_factory=list)

    def append(self, record: ProviderSwitchRecord) -> None:
        self.entries.append(record)

    def loss_events(self) -> list[ProviderSwitchRecord]:
        return [e for e in self.entries if e.report.has_loss]

    def observed_rate(self) -> float:
        """Fraction of observed switches flagged as lossy.

        NOT the hypothesis recall: this is an observability metric. Callers
        wanting H048 recall must pair this log with labelled scenarios via
        CarryOverVerifier.detection_recall.
        """
        if not self.entries:
            return 0.0
        return len(self.loss_events()) / len(self.entries)


def record_switch(
    log: CarryOverAuditLog,
    verifier: CarryOverVerifier,
    thread_before: ConversationThread,
    response_after: ChatResponse,
    from_provider: str,
    to_provider: str,
    trigger: str = "rate_limit",
) -> CarryOverReport:
    """Convenience helper: verify and record in one call."""
    snapshot = ConversationThread(
        messages=list(thread_before.messages),
        system_prompt=thread_before.system_prompt,
    )
    report = verifier.verify(snapshot, response_after)
    log.append(
        ProviderSwitchRecord(
            from_provider=from_provider,
            to_provider=to_provider,
            report=report,
            thread_token_estimate=snapshot.token_estimate(),
            trigger=trigger,
        )
    )
    return report


__all__ = [
    "CarryOverAuditLog",
    "CarryOverReport",
    "CarryOverVerifier",
    "ProviderSwitchRecord",
    "VerifierFn",
    "extract_facts",
    "record_switch",
]
