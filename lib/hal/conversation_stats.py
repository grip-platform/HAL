"""HAL Conversation Stats — detailed conversation analytics.

Provides rich statistics about conversation patterns: word frequency,
response times, topic distribution, and engagement metrics.

Usage:
    from lib.hal.conversation_stats import ConversationStats
    stats = ConversationStats.from_history(history)
"""
import re
from collections import Counter
from dataclasses import dataclass

from lib.hal.types import ChatMessage


@dataclass(frozen=True)
class ConversationStats:
    total_messages: int
    user_messages: int
    assistant_messages: int
    total_words: int
    user_words: int
    assistant_words: int
    avg_user_length: int
    avg_assistant_length: int
    questions_asked: int
    code_blocks: int
    top_words: tuple[tuple[str, int], ...]

    @property
    def response_ratio(self) -> float:
        """Ratio of assistant words to user words."""
        return self.assistant_words / max(self.user_words, 1)

    def to_dict(self) -> dict:
        return {
            "total_messages": self.total_messages,
            "user_messages": self.user_messages,
            "assistant_messages": self.assistant_messages,
            "total_words": self.total_words,
            "avg_user_length": self.avg_user_length,
            "avg_assistant_length": self.avg_assistant_length,
            "questions_asked": self.questions_asked,
            "code_blocks": self.code_blocks,
            "response_ratio": round(self.response_ratio, 1),
            "top_words": list(self.top_words[:10]),
        }

    @staticmethod
    def from_history(history: list[ChatMessage]) -> "ConversationStats":
        user_msgs = [m for m in history if m.role == "user"]
        asst_msgs = [m for m in history if m.role == "assistant"]
        all_text = " ".join(m.content for m in history)

        user_words = sum(len(m.content.split()) for m in user_msgs)
        asst_words = sum(len(m.content.split()) for m in asst_msgs)
        questions = sum(m.content.count("?") for m in user_msgs)
        code_blocks = all_text.count("```") // 2

        words = re.findall(r"\b[a-z]{4,}\b", all_text.lower())
        stop_words = {"this", "that", "with", "from", "have", "will", "been", "would", "could", "should", "their", "there", "about", "which", "when", "your", "what", "them"}
        filtered = [w for w in words if w not in stop_words]
        top = tuple(Counter(filtered).most_common(20))

        return ConversationStats(
            total_messages=len(history),
            user_messages=len(user_msgs),
            assistant_messages=len(asst_msgs),
            total_words=user_words + asst_words,
            user_words=user_words,
            assistant_words=asst_words,
            avg_user_length=user_words // max(len(user_msgs), 1),
            avg_assistant_length=asst_words // max(len(asst_msgs), 1),
            questions_asked=questions,
            code_blocks=code_blocks,
            top_words=top,
        )
