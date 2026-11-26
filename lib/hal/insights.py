"""HAL Conversation Insights — post-conversation analytics and patterns.

Analyses a completed conversation to extract: effectiveness metrics,
topic patterns, friction points, and improvement suggestions. Designed
to feed into the RSI engine for continuous self-improvement.

Usage:
    from lib.hal.insights import InsightsEngine

    engine = InsightsEngine()
    report = engine.analyse(conversation_history)
    print(report.summary)
    for suggestion in report.suggestions:
        print(f"  - {suggestion}")
"""
import re
import time
from dataclasses import dataclass, field

from lib.hal.cost_tracker import estimate_tokens
from lib.hal.types import ChatMessage


@dataclass(frozen=True)
class FrictionPoint:
    """A detected friction point in the conversation."""
    turn_index: int
    description: str
    severity: str  # "low", "medium", "high"


@dataclass
class InsightsReport:
    """Post-conversation analytics report."""
    turn_count: int = 0
    total_tokens_est: int = 0
    avg_response_length: int = 0
    topic_shifts: int = 0
    questions_asked: int = 0
    code_blocks: int = 0
    friction_points: list[FrictionPoint] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def tokens_per_turn(self) -> int:
        return self.total_tokens_est // max(self.turn_count, 1)

    @property
    def efficiency_score(self) -> float:
        """0.0 to 1.0 — higher means more output per token spent."""
        if self.total_tokens_est == 0:
            return 0.0
        useful_output = self.code_blocks * 200 + self.turn_count * 50
        return min(1.0, useful_output / self.total_tokens_est)

    @property
    def summary(self) -> str:
        return (
            f"Insights: {self.turn_count} turns, ~{self.total_tokens_est:,} tokens, "
            f"{len(self.friction_points)} friction points, "
            f"efficiency {self.efficiency_score:.0%}"
        )

    def to_dict(self) -> dict:
        return {
            "turn_count": self.turn_count,
            "total_tokens_est": self.total_tokens_est,
            "avg_response_length": self.avg_response_length,
            "topic_shifts": self.topic_shifts,
            "questions_asked": self.questions_asked,
            "code_blocks": self.code_blocks,
            "friction_points": [
                {"turn": f.turn_index, "desc": f.description, "severity": f.severity}
                for f in self.friction_points
            ],
            "suggestions": self.suggestions,
            "topics": self.topics,
            "efficiency_score": self.efficiency_score,
        }


class InsightsEngine:
    """Analyses conversations for patterns and improvement opportunities."""

    def analyse(self, history: list[ChatMessage]) -> InsightsReport:
        """Analyse a conversation history and generate insights."""
        start = time.time()
        if not history:
            return InsightsReport()

        user_msgs = [m for m in history if m.role == "user"]
        asst_msgs = [m for m in history if m.role == "assistant"]

        total_tokens = sum(estimate_tokens(m.content) for m in history)
        avg_resp = (
            sum(len(m.content) for m in asst_msgs) // max(len(asst_msgs), 1)
        )
        questions = sum(1 for m in user_msgs if "?" in m.content)
        code_blocks = sum(m.content.count("```") // 2 for m in asst_msgs)
        topics = _extract_topics(user_msgs)
        topic_shifts = max(0, len(topics) - 1)
        friction = _detect_friction(history)
        suggestions = _generate_suggestions(
            len(user_msgs), total_tokens, friction, topic_shifts, avg_resp,
        )

        return InsightsReport(
            turn_count=len(user_msgs),
            total_tokens_est=total_tokens,
            avg_response_length=avg_resp,
            topic_shifts=topic_shifts,
            questions_asked=questions,
            code_blocks=code_blocks,
            friction_points=friction,
            suggestions=suggestions,
            topics=topics,
            duration_s=time.time() - start,
        )


def _extract_topics(user_msgs: list[ChatMessage]) -> list[str]:
    """Extract distinct topic clusters from user messages."""
    topics: list[str] = []
    prev_words: set[str] = set()

    for msg in user_msgs:
        words = set(re.findall(r"\b[a-z]{4,}\b", msg.content.lower()))
        if prev_words and len(words & prev_words) < len(words) * 0.3:
            # Less than 30% word overlap — likely a topic shift
            key_words = sorted(words - prev_words)[:3]
            if key_words:
                topics.append(", ".join(key_words))
        elif not topics and words:
            topics.append(", ".join(sorted(words)[:3]))
        prev_words = words

    return topics


def _detect_friction(history: list[ChatMessage]) -> list[FrictionPoint]:
    """Detect friction points: repeated questions, corrections, frustration."""
    friction: list[FrictionPoint] = []
    user_turn = 0

    for i, msg in enumerate(history):
        if msg.role != "user":
            continue
        user_turn += 1
        content_lower = msg.content.lower()

        # Correction signals
        if any(kw in content_lower for kw in ("no not", "that's wrong", "i said", "try again")):
            friction.append(FrictionPoint(user_turn, "User correction detected", "medium"))

        # Frustration signals
        if any(kw in content_lower for kw in ("stop", "don't", "!!!", "???")):
            friction.append(FrictionPoint(user_turn, "Frustration signal detected", "high"))

        # Repetition: user repeating same request
        if i >= 2 and msg.role == "user":
            prev_user = [m for m in history[:i] if m.role == "user"]
            if prev_user:
                last = prev_user[-1].content.lower()
                overlap = len(set(content_lower.split()) & set(last.split()))
                total = max(len(set(content_lower.split())), 1)
                if overlap / total > 0.7:
                    friction.append(FrictionPoint(user_turn, "Repeated request — previous response may have missed the point", "medium"))

    return friction


def _generate_suggestions(
    turns: int, tokens: int, friction: list[FrictionPoint],
    topic_shifts: int, avg_response: int,
) -> list[str]:
    """Generate actionable improvement suggestions."""
    suggestions: list[str] = []

    if len(friction) > turns * 0.3:
        suggestions.append("High friction rate — consider asking clarifying questions before acting")

    if avg_response > 2000:
        suggestions.append("Responses are verbose — consider more concise output")

    if topic_shifts > 5:
        suggestions.append("Many topic shifts — consider keeping focus or breaking into separate sessions")

    if tokens > 100_000 and turns < 10:
        suggestions.append("High token usage per turn — check for unnecessary context loading")

    high_friction = [f for f in friction if f.severity == "high"]
    if high_friction:
        suggestions.append(f"{len(high_friction)} high-severity friction points — review and adapt approach")

    if not suggestions:
        suggestions.append("Session ran smoothly — no significant friction detected")

    return suggestions
