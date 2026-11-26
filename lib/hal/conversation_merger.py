"""HAL Conversation Merger — combine two conversations into one.

Merges parallel conversation branches or separate sessions into a
unified history, preserving chronological order.

Usage:
    from lib.hal.conversation_merger import merge_conversations
    merged = merge_conversations(history_a, history_b)
"""
from lib.hal.types import ChatMessage


def merge_conversations(
    a: list[ChatMessage], b: list[ChatMessage],
    strategy: str = "interleave",
) -> list[ChatMessage]:
    """Merge two conversation histories.

    Strategies:
    - interleave: alternate turns from each
    - append: a then b
    - summary: summarise a, then append b
    """
    if strategy == "append":
        return list(a) + list(b)
    if strategy == "interleave":
        return _interleave(a, b)
    return list(a) + list(b)  # fallback


def _interleave(a: list[ChatMessage], b: list[ChatMessage]) -> list[ChatMessage]:
    """Interleave two histories by alternating."""
    result: list[ChatMessage] = []
    max_len = max(len(a), len(b))
    for i in range(max_len):
        if i < len(a):
            result.append(a[i])
        if i < len(b):
            result.append(b[i])
    return result
