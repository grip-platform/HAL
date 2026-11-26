"""Unit tests for CarryOverVerifier — W2 of RAILLM ultrado plan.

Verifies extract_facts heuristics, verify() protocol, detection_recall
metric computation, and the optional audit-log integration. Tests assert
output VALUES per Goodhart guard rule (Rule 14), not call counts, so a
broken implementation cannot pass silently.
"""
import sys
import unittest
from pathlib import Path

try:
    import pytest
except ImportError:
    raise unittest.SkipTest("pytest not installed")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.hal.carryover_verifier import (
    CarryOverAuditLog,
    CarryOverReport,
    CarryOverVerifier,
    ProviderSwitchRecord,
    _default_substring_verifier,
    extract_facts,
    record_switch,
)
from lib.hal.raillm import ConversationThread, Message
from lib.hal.types import ChatResponse


def _resp(text):
    return ChatResponse(text=text, model="fake/model", provider="fake", raw={})


# ---------------------------------------------------------------------------
# extract_facts — numeric extraction
# ---------------------------------------------------------------------------


def test_extract_numeric_integer_fact():
    t = ConversationThread()
    t.append(Message(role="user", content="The meeting is at 42 Main Street"))
    facts = extract_facts(t)
    assert "42" in facts


def test_extract_numeric_percentage():
    t = ConversationThread()
    t.append(Message(role="user", content="Revenue grew 37% this quarter"))
    facts = extract_facts(t)
    assert "37%" in facts


def test_extract_numeric_currency():
    t = ConversationThread()
    t.append(Message(role="user", content="The invoice total was $1,234.56"))
    facts = extract_facts(t)
    assert "$1,234.56" in facts


def test_extract_numeric_large_comma_separated():
    t = ConversationThread()
    t.append(Message(role="user", content="Population reached 1,000,000 by 2024"))
    facts = extract_facts(t)
    assert "1,000,000" in facts
    assert "2024" in facts


# ---------------------------------------------------------------------------
# extract_facts — proper-noun extraction
# ---------------------------------------------------------------------------


def test_extract_proper_noun_phrase():
    t = ConversationThread()
    t.append(Message(role="user", content="Fabio Nunes lives in Cape Town"))
    facts = extract_facts(t)
    assert "Fabio Nunes" in facts


def test_extract_ignores_single_capital_sentence_starts():
    t = ConversationThread()
    t.append(Message(role="user", content="The quick brown fox jumps"))
    facts = extract_facts(t)
    # No 2+ word proper noun phrase should match
    assert not any(
        token in facts for token in ("The", "The quick", "quick")
    )


# ---------------------------------------------------------------------------
# extract_facts — quoted literals
# ---------------------------------------------------------------------------


def test_extract_double_quoted_literal():
    t = ConversationThread()
    t.append(Message(role="user", content='The password is "hunter2-golden" for now'))
    facts = extract_facts(t)
    assert "hunter2-golden" in facts


def test_extract_single_quoted_literal():
    t = ConversationThread()
    t.append(Message(role="user", content="Name the variable \'alpha-one\' please"))
    facts = extract_facts(t)
    assert "alpha-one" in facts


# ---------------------------------------------------------------------------
# extract_facts — structural invariants
# ---------------------------------------------------------------------------


def test_extract_excludes_system_prompts():
    t = ConversationThread(system_prompt="You are assistant 42")
    t.append(Message(role="user", content="Tell me about 99"))
    facts = extract_facts(t)
    assert "99" in facts
    assert "42" not in facts  # system prompt must not leak


def test_extract_deduplicates_preserving_order():
    t = ConversationThread()
    t.append(Message(role="user", content="First we saw 42, then 42 again, then 99"))
    facts = extract_facts(t)
    # 42 appears once, then 99 after it
    assert facts.count("42") == 1
    assert facts.index("42") < facts.index("99")


def test_extract_respects_max_facts_cap():
    t = ConversationThread()
    big = " ".join(str(n) for n in range(100))
    t.append(Message(role="user", content=big))
    facts = extract_facts(t, max_facts=5)
    assert len(facts) == 5


def test_extract_empty_thread_returns_empty_list():
    t = ConversationThread()
    assert extract_facts(t) == []


# ---------------------------------------------------------------------------
# _default_substring_verifier
# ---------------------------------------------------------------------------


def test_default_verifier_all_facts_retained():
    t = ConversationThread()
    t.append(Message(role="user", content="The value is 42"))
    result = _default_substring_verifier(t, _resp("I noted that 42 value"), ["42"])
    assert result["retained"] == ["42"]
    assert result["lost"] == []
    assert result["confidence"] == 1.0


def test_default_verifier_all_facts_lost():
    t = ConversationThread()
    result = _default_substring_verifier(
        t, _resp("sorry I do not remember"), ["42", "99"]
    )
    assert result["lost"] == ["42", "99"]
    assert result["retained"] == []
    # Loss ratio 1.0 -> confidence = 1 - 0.5 = 0.5
    assert result["confidence"] == pytest.approx(0.5)


def test_default_verifier_partial_loss_confidence_between():
    t = ConversationThread()
    result = _default_substring_verifier(
        t, _resp("I only remember 42"), ["42", "99"]
    )
    assert result["retained"] == ["42"]
    assert result["lost"] == ["99"]
    # 1 lost of 2 -> confidence = 1 - (0.5 * 0.5) = 0.75
    assert result["confidence"] == pytest.approx(0.75)


def test_default_verifier_case_insensitive_match():
    t = ConversationThread()
    result = _default_substring_verifier(
        t, _resp("YES I KNOW fabio nunes"), ["Fabio Nunes"]
    )
    assert result["retained"] == ["Fabio Nunes"]


# ---------------------------------------------------------------------------
# CarryOverVerifier.verify
# ---------------------------------------------------------------------------


def test_verify_returns_no_loss_when_thread_has_no_facts():
    t = ConversationThread()
    t.append(Message(role="user", content="hello there friend"))
    v = CarryOverVerifier()
    report = v.verify(t, _resp("hi"))
    assert report.has_loss is False
    assert report.confidence_score == 1.0
    assert report.verifier_model == "extractor-none"


def test_verify_flags_loss_when_facts_missing_from_response():
    t = ConversationThread()
    t.append(Message(role="user", content="Meet at 42 Main with Fabio Nunes"))
    v = CarryOverVerifier()
    report = v.verify(t, _resp("hello"))
    assert report.has_loss is True
    assert "42" in report.lost_facts
    assert "Fabio Nunes" in report.lost_facts
    assert report.retained_facts == ()


def test_verify_reports_retention_when_response_mentions_facts():
    t = ConversationThread()
    t.append(Message(role="user", content="The value is 42"))
    v = CarryOverVerifier()
    report = v.verify(t, _resp("Got it, 42 confirmed"))
    assert report.has_loss is False
    assert "42" in report.retained_facts


def test_verify_uses_custom_verifier_fn():
    calls = {}

    def custom(thread, response, facts):
        calls["invoked"] = True
        return {
            "retained": ["synthetic-fact"],
            "lost": [],
            "confidence": 0.9,
            "model": "custom-judge",
        }

    v = CarryOverVerifier(verifier_fn=custom)
    t = ConversationThread()
    t.append(Message(role="user", content="value 42"))
    report = v.verify(t, _resp("response"))
    assert calls["invoked"] is True
    assert report.verifier_model == "custom-judge"
    assert report.confidence_score == 0.9


def test_verify_clamps_confidence_to_0_1_range():
    def bad(thread, response, facts):
        return {"retained": [], "lost": facts, "confidence": 99.0, "model": "bad"}

    v = CarryOverVerifier(verifier_fn=bad)
    t = ConversationThread()
    t.append(Message(role="user", content="value 42"))
    report = v.verify(t, _resp("x"))
    assert report.confidence_score == 1.0


def test_verify_clamps_negative_confidence_to_zero():
    def bad(thread, response, facts):
        return {"retained": [], "lost": facts, "confidence": -5.0, "model": "bad"}

    v = CarryOverVerifier(verifier_fn=bad)
    t = ConversationThread()
    t.append(Message(role="user", content="value 42"))
    report = v.verify(t, _resp("x"))
    assert report.confidence_score == 0.0


# ---------------------------------------------------------------------------
# detection_recall — H048 metric
# ---------------------------------------------------------------------------


def test_detection_recall_empty_suite_returns_1():
    v = CarryOverVerifier()
    assert v.detection_recall([]) == 1.0


def test_detection_recall_only_negative_scenarios_returns_1():
    v = CarryOverVerifier()
    t = ConversationThread()
    t.append(Message(role="user", content="hello"))
    # expected=False entries do not affect recall (only TP+FN counted)
    scenarios = [(t, _resp("hi"), False), (t, _resp("hi"), False)]
    assert v.detection_recall(scenarios) == 1.0


def test_detection_recall_perfect_detection_returns_1():
    v = CarryOverVerifier()
    loss = ConversationThread()
    loss.append(Message(role="user", content="The value is 42"))
    scenarios = [(loss, _resp("sorry blank"), True)] * 5
    assert v.detection_recall(scenarios) == 1.0


def test_detection_recall_half_miss_returns_0_5():
    v = CarryOverVerifier()
    loss = ConversationThread()
    loss.append(Message(role="user", content="The value is 42"))
    nothing = ConversationThread()
    nothing.append(Message(role="user", content="hello friend"))
    # Two true-loss scenarios, only one is detectable: the other has no
    # extractable facts so the verifier returns has_loss=False.
    scenarios = [
        (loss, _resp("sorry blank"), True),     # TP
        (nothing, _resp("hello back"), True),   # FN (no facts -> no_loss)
    ]
    assert v.detection_recall(scenarios) == 0.5


def test_detection_recall_meets_h048_target_on_synthetic_suite():
    """H048 gate: detection_recall >= 0.95 on synthetic loss scenarios.

    Builds 20 loss scenarios where the response clearly fails to retain
    facts. The default substring verifier should hit >= 0.95 recall.
    A regression that flips has_loss to False for factful threads fails
    this test, mirroring the H048 falsification criterion.
    """
    v = CarryOverVerifier()
    scenarios = []
    for i in range(20):
        t = ConversationThread()
        t.append(Message(role="user", content=f"Reference point {i + 100} is critical"))
        scenarios.append((t, _resp("I do not recall that"), True))
    recall = v.detection_recall(scenarios)
    assert recall >= 0.95


# ---------------------------------------------------------------------------
# CarryOverAuditLog + record_switch
# ---------------------------------------------------------------------------


def test_audit_log_appends_records():
    log = CarryOverAuditLog()
    v = CarryOverVerifier()
    t = ConversationThread()
    t.append(Message(role="user", content="value 42"))
    record_switch(log, v, t, _resp("blank"), "anthropic", "groq")
    assert len(log.entries) == 1
    assert log.entries[0].from_provider == "anthropic"
    assert log.entries[0].to_provider == "groq"


def test_audit_log_loss_events_filters_correctly():
    log = CarryOverAuditLog()
    v = CarryOverVerifier()
    lossy = ConversationThread()
    lossy.append(Message(role="user", content="value 42"))
    clean = ConversationThread()
    clean.append(Message(role="user", content="value 42"))
    record_switch(log, v, lossy, _resp("blank"), "a", "b")
    record_switch(log, v, clean, _resp("got 42 noted"), "b", "c")
    loss = log.loss_events()
    assert len(loss) == 1
    assert loss[0].from_provider == "a"


def test_audit_log_observed_rate_empty_returns_zero():
    log = CarryOverAuditLog()
    assert log.observed_rate() == 0.0


def test_audit_log_observed_rate_computes_fraction():
    log = CarryOverAuditLog()
    v = CarryOverVerifier()
    lossy = ConversationThread()
    lossy.append(Message(role="user", content="value 42"))
    clean = ConversationThread()
    clean.append(Message(role="user", content="value 42"))
    record_switch(log, v, lossy, _resp("blank"), "a", "b")
    record_switch(log, v, clean, _resp("got 42"), "b", "c")
    record_switch(log, v, clean, _resp("42 seen"), "c", "d")
    assert log.observed_rate() == pytest.approx(1.0 / 3.0)


def test_record_switch_snapshots_thread():
    """Mutating the original thread after record_switch must NOT affect
    the audit log entry — verify the snapshot is a real copy."""
    log = CarryOverAuditLog()
    v = CarryOverVerifier()
    t = ConversationThread()
    t.append(Message(role="user", content="value 42"))
    record_switch(log, v, t, _resp("blank"), "a", "b")
    original_token_count = log.entries[0].thread_token_estimate
    # Mutate the original thread after recording.
    t.append(Message(role="user", content="y" * 1000))
    assert log.entries[0].thread_token_estimate == original_token_count


# ---------------------------------------------------------------------------
# Goodhart check — mutation test mindset
# ---------------------------------------------------------------------------


def test_goodhart_verifier_cannot_trivially_claim_no_loss():
    """If the verifier unconditionally returned has_loss=False, the H048
    recall test would drop to 0. This test hard-asserts has_loss IS True
    for an obvious loss scenario, so a bypass that always returns False
    fails at least one visible test instead of passing silently."""
    v = CarryOverVerifier()
    t = ConversationThread()
    t.append(Message(role="user", content="The password is 42-gold-alpha"))
    report = v.verify(t, _resp("I do not know anything"))
    assert report.has_loss is True, (
        "Verifier must flag clear information loss — always-False bypass caught"
    )
