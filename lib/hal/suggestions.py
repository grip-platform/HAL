"""HAL Smart Suggestions — proactive next-action recommendations.

Analyses the current conversation context and suggests useful next steps.
Designed to be non-intrusive: suggestions appear only when relevant and
are never auto-executed.

Usage:
    from lib.hal.suggestions import SuggestionEngine

    engine = SuggestionEngine()
    suggestions = engine.suggest(history, context)
    for s in suggestions:
        print(f"[{s.category}] {s.text}")
"""
import re
from dataclasses import dataclass

from lib.hal.types import ChatMessage


@dataclass(frozen=True)
class Suggestion:
    """A proactive suggestion for the user."""
    text: str
    category: str  # "action", "question", "warning", "tip"
    confidence: float  # 0.0 to 1.0
    trigger: str  # What triggered this suggestion


class SuggestionEngine:
    """Generates proactive suggestions based on conversation context."""

    def suggest(
        self, history: list[ChatMessage],
        context: dict | None = None,
    ) -> list[Suggestion]:
        """Generate suggestions based on conversation state.

        Args:
            history: Conversation history
            context: Optional context dict with keys like 'model', 'cost', 'turns'
        """
        if not history:
            return [_welcome_suggestion()]

        suggestions: list[Suggestion] = []
        ctx = context or {}

        suggestions.extend(_check_long_conversation(history, ctx))
        suggestions.extend(_check_unanswered_questions(history))
        suggestions.extend(_check_error_patterns(history))
        suggestions.extend(_check_code_without_tests(history))
        suggestions.extend(_check_cost_awareness(ctx))

        # Sort by confidence, return top 3
        suggestions.sort(key=lambda s: s.confidence, reverse=True)
        return suggestions[:3]


def _welcome_suggestion() -> Suggestion:
    """Suggestion for new conversations."""
    return Suggestion(
        text="Try /council for multi-model analysis, or /status to see your providers",
        category="tip",
        confidence=0.5,
        trigger="empty_conversation",
    )


def _check_long_conversation(history: list[ChatMessage], ctx: dict) -> list[Suggestion]:
    """Suggest compaction for long conversations."""
    user_turns = sum(1 for m in history if m.role == "user")
    if user_turns > 20:
        return [Suggestion(
            text=f"This conversation has {user_turns} turns. Consider saving key context to memory.",
            category="warning",
            confidence=0.8,
            trigger="long_conversation",
        )]
    return []


def _check_unanswered_questions(history: list[ChatMessage]) -> list[Suggestion]:
    """Detect if the user asked questions that weren't addressed."""
    if len(history) < 2:
        return []

    last_user = None
    last_asst = None
    for msg in reversed(history):
        if msg.role == "user" and last_user is None:
            last_user = msg
        elif msg.role == "assistant" and last_asst is None:
            last_asst = msg
        if last_user and last_asst:
            break

    if not last_user or not last_asst:
        return []

    questions = re.findall(r"[^.!?]*\?", last_user.content)
    if len(questions) > 2:
        answered = sum(1 for q in questions if _question_addressed(q, last_asst.content))
        if answered < len(questions):
            return [Suggestion(
                text=f"{len(questions) - answered} of {len(questions)} questions may not have been fully addressed",
                category="question",
                confidence=0.6,
                trigger="unanswered_questions",
            )]
    return []


def _check_error_patterns(history: list[ChatMessage]) -> list[Suggestion]:
    """Suggest debugging approach after repeated errors."""
    error_count = sum(
        1 for m in history[-6:]
        if m.role == "assistant" and re.search(r"\b(?:error|failed|exception|traceback)\b", m.content, re.IGNORECASE)
    )
    if error_count >= 2:
        return [Suggestion(
            text="Multiple errors detected. Consider stepping back to understand the root cause before more fixes.",
            category="action",
            confidence=0.7,
            trigger="repeated_errors",
        )]
    return []


def _check_code_without_tests(history: list[ChatMessage]) -> list[Suggestion]:
    """Suggest tests when code was written but no test discussion."""
    code_written = any(
        "```" in m.content and m.role == "assistant"
        for m in history[-10:]
    )
    tests_mentioned = any(
        re.search(r"\btest", m.content, re.IGNORECASE)
        for m in history[-10:]
    )
    if code_written and not tests_mentioned:
        return [Suggestion(
            text="Code was generated but no tests discussed. Consider adding test coverage.",
            category="tip",
            confidence=0.5,
            trigger="code_no_tests",
        )]
    return []


def _check_cost_awareness(ctx: dict) -> list[Suggestion]:
    """Suggest cost-saving measures for expensive sessions."""
    cost = ctx.get("total_cost_usd", 0)
    if cost > 1.0:
        return [Suggestion(
            text=f"Session cost is ${cost:.2f}. Consider switching to a cheaper provider for simple queries.",
            category="warning",
            confidence=0.7,
            trigger="high_cost",
        )]
    return []


def _question_addressed(question: str, response: str) -> bool:
    """Heuristic check if a question was addressed in the response."""
    q_words = set(re.findall(r"\b\w{4,}\b", question.lower()))
    r_words = set(re.findall(r"\b\w{4,}\b", response.lower()))
    if not q_words:
        return True
    return len(q_words & r_words) / len(q_words) > 0.3
