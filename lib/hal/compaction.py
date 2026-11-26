"""HAL Smart Compaction — intelligent conversation summarisation.

When the LLM context window fills up (detected by context_warnings),
compaction summarises older conversation turns while preserving key
decisions, code snippets, and action items. This extends HAL's
never-ending sessions beyond any single context window.

Unlike hard truncation (drop oldest turns), smart compaction:
1. Preserves code blocks and file references
2. Keeps decisions and commitments
3. Summarises discussion into dense context
4. Maintains a compaction chain (each summary references its predecessor)

Usage:
    from lib.hal.compaction import SmartCompactor

    compactor = SmartCompactor()
    summary = compactor.compact(history[:20])  # Compact first 20 turns
    # Insert summary as a system message, drop the original 20 turns
"""
import re
from dataclasses import dataclass

from lib.hal.types import ChatMessage


@dataclass(frozen=True)
class CompactionResult:
    """Result of compacting conversation turns."""
    summary: str
    turns_compacted: int
    code_blocks_preserved: int
    decisions_found: int
    original_tokens_est: int
    compacted_tokens_est: int

    @property
    def compression_ratio(self) -> float:
        if self.original_tokens_est == 0:
            return 0.0
        return 1.0 - (self.compacted_tokens_est / self.original_tokens_est)

    @property
    def summary_message(self) -> ChatMessage:
        """Create a system message containing the compacted summary."""
        return ChatMessage(
            role="system",
            content=f"[Compacted context — {self.turns_compacted} turns summarised]\n\n{self.summary}",
        )


class SmartCompactor:
    """Intelligently compacts conversation history."""

    def __init__(self, registry=None):
        self._registry = registry

    def compact(self, history: list[ChatMessage], model: str = "") -> CompactionResult:
        """Compact a slice of conversation history into a dense summary.

        If a registry and model are available, uses LLM summarisation.
        Otherwise, falls back to mechanical extraction.
        """
        if not history:
            return CompactionResult("", 0, 0, 0, 0, 0)

        code_blocks = _extract_code_blocks(history)
        decisions = _extract_decisions(history)
        original_est = _estimate_tokens(history)

        if self._registry and model:
            summary = self._llm_compact(history, code_blocks, decisions, model)
        else:
            summary = _mechanical_compact(history, code_blocks, decisions)

        compacted_est = len(summary.split()) * 4 // 3

        return CompactionResult(
            summary=summary,
            turns_compacted=len(history),
            code_blocks_preserved=len(code_blocks),
            decisions_found=len(decisions),
            original_tokens_est=original_est,
            compacted_tokens_est=compacted_est,
        )

    def _llm_compact(
        self, history: list[ChatMessage],
        code_blocks: list[str], decisions: list[str], model: str,
    ) -> str:
        """Use an LLM to produce an intelligent summary."""
        transcript = "\n".join(f"[{m.role}]: {m.content[:500]}" for m in history)
        code_section = "\n".join(code_blocks[:5]) if code_blocks else "None"
        decision_section = "\n".join(decisions[:10]) if decisions else "None"

        prompt = (
            "Summarise this conversation into a dense context paragraph. "
            "Preserve: key decisions, file paths, function names, action items. "
            "Drop: greetings, acknowledgements, redundant explanations.\n\n"
            f"TRANSCRIPT ({len(history)} turns):\n{transcript[:4000]}\n\n"
            f"CODE BLOCKS:\n{code_section[:1000]}\n\n"
            f"DECISIONS:\n{decision_section[:500]}\n\n"
            "Output ONLY the summary paragraph."
        )

        try:
            response = self._registry.chat(model, prompt, max_tokens=500)
            return response.text
        except Exception:
            return _mechanical_compact(history, code_blocks, decisions)


def _mechanical_compact(
    history: list[ChatMessage],
    code_blocks: list[str],
    decisions: list[str],
) -> str:
    """Rule-based compaction without LLM. Extracts key content mechanically."""
    parts: list[str] = []

    # Topic summary: first user message as context anchor
    user_msgs = [m for m in history if m.role == "user"]
    if user_msgs:
        parts.append(f"Topic: {user_msgs[0].content[:200]}")

    # Decisions
    if decisions:
        parts.append("Decisions: " + "; ".join(decisions[:5]))

    # Code blocks (preserve as-is)
    for block in code_blocks[:3]:
        parts.append(block[:500])

    # File references
    files = _extract_file_refs(history)
    if files:
        parts.append(f"Files referenced: {', '.join(sorted(files)[:10])}")

    # Last assistant message as final state
    asst_msgs = [m for m in history if m.role == "assistant"]
    if asst_msgs:
        parts.append(f"Last response: {asst_msgs[-1].content[:300]}")

    return "\n\n".join(parts)


_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_DECISION_PATTERNS = re.compile(
    r"(?:decided|agreed|will|should|must|chosen?|plan is|approach is|going to)\b",
    re.IGNORECASE,
)
_FILE_REF_RE = re.compile(r"[\w/.-]+\.(?:py|ts|js|md|json|yaml|rs|go|java)")


def _extract_code_blocks(history: list[ChatMessage]) -> list[str]:
    """Extract all fenced code blocks. O(n) — single concatenation then regex."""
    combined = "\n".join(m.content for m in history)
    return _CODE_BLOCK_RE.findall(combined)


def _extract_decisions(history: list[ChatMessage]) -> list[str]:
    """Extract sentences that look like decisions. O(n) — single pass over combined text."""
    combined = " ".join(m.content for m in history)
    sentences = re.split(r"[.!?]\s+", combined)
    return [
        s.strip()[:200] for s in sentences
        if _DECISION_PATTERNS.search(s) and len(s.strip()) > 20
    ]


def _extract_file_refs(history: list[ChatMessage]) -> set[str]:
    """Extract file path references. O(n) — single concatenation then regex."""
    combined = " ".join(m.content for m in history)
    return set(_FILE_REF_RE.findall(combined))


def _estimate_tokens(history: list[ChatMessage]) -> int:
    """Estimate total tokens in a message list."""
    total_words = sum(len(m.content.split()) for m in history)
    return int(total_words / 0.75)
