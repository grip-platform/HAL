"""HAL Summary Generator — intelligent conversation summaries for handoff.

Generates structured summaries from conversation history, optimised for
session handoff, stakeholder reporting, and documentation. Supports
multiple summary formats: executive, technical, and action-item focused.

Usage:
    from lib.hal.summary_generator import SummaryGenerator

    gen = SummaryGenerator()
    summary = gen.generate(history, style="executive")
    print(summary.text)
"""
import re
from dataclasses import dataclass

from lib.hal.types import ChatMessage


@dataclass(frozen=True)
class Summary:
    """A generated conversation summary."""
    text: str
    style: str
    turn_count: int
    key_decisions: tuple[str, ...]
    action_items: tuple[str, ...]
    files_referenced: tuple[str, ...]


class SummaryGenerator:
    """Generates summaries from conversation history."""

    def generate(self, history: list[ChatMessage], style: str = "technical") -> Summary:
        """Generate a summary in the specified style.

        Styles:
        - executive: high-level outcomes, decisions, next steps
        - technical: code changes, architecture decisions, file modifications
        - actions: just the action items and commitments
        """
        if not history:
            return Summary("No conversation to summarise.", style, 0, (), (), ())

        user_msgs = [m for m in history if m.role == "user"]
        asst_msgs = [m for m in history if m.role == "assistant"]
        combined = " ".join(m.content for m in history)

        decisions = _extract_decisions(combined)
        actions = _extract_actions(combined)
        files = _extract_files(combined)

        generators = {
            "executive": _executive_summary,
            "technical": _technical_summary,
            "actions": _actions_summary,
        }
        gen_fn = generators.get(style, _technical_summary)
        text = gen_fn(user_msgs, asst_msgs, decisions, actions, files)

        return Summary(
            text=text, style=style,
            turn_count=len(user_msgs),
            key_decisions=tuple(decisions[:10]),
            action_items=tuple(actions[:10]),
            files_referenced=tuple(sorted(files)[:20]),
        )


_DECISION_RE = re.compile(
    r"(?:decided|agreed|will use|chosen|approach is|plan is|going with)\s+(.{10,100})",
    re.IGNORECASE,
)
_ACTION_RE = re.compile(
    r"(?:TODO|FIXME|need to|should|must|will|action item|next step)[:\s]+(.{10,100})",
    re.IGNORECASE,
)
_FILE_RE = re.compile(r"[\w/.-]+\.(?:py|ts|js|md|json|yaml|rs|go|java|tsx|jsx)")


def _extract_decisions(text: str) -> list[str]:
    """Extract decision statements."""
    return [m.group(1).strip().rstrip(".") for m in _DECISION_RE.finditer(text)]


def _extract_actions(text: str) -> list[str]:
    """Extract action items and TODOs."""
    return [m.group(1).strip().rstrip(".") for m in _ACTION_RE.finditer(text)]


def _extract_files(text: str) -> set[str]:
    """Extract file path references."""
    return set(_FILE_RE.findall(text))


def _executive_summary(
    user_msgs: list, asst_msgs: list,
    decisions: list, actions: list, files: set,
) -> str:
    """High-level summary for stakeholders."""
    parts = [f"Session: {len(user_msgs)} turns"]

    if user_msgs:
        parts.append(f"Topic: {user_msgs[0].content[:150]}")

    if decisions:
        parts.append("Decisions: " + "; ".join(decisions[:5]))

    if actions:
        parts.append("Next steps: " + "; ".join(actions[:5]))

    if files:
        parts.append(f"Files touched: {', '.join(sorted(files)[:10])}")

    return "\n\n".join(parts)


def _technical_summary(
    user_msgs: list, asst_msgs: list,
    decisions: list, actions: list, files: set,
) -> str:
    """Detailed technical summary."""
    parts = [f"Technical Summary ({len(user_msgs)} turns)"]

    if user_msgs:
        parts.append(f"Initial request: {user_msgs[0].content[:200]}")

    if files:
        parts.append(f"Files: {', '.join(sorted(files)[:15])}")

    code_blocks = sum(m.content.count("```") // 2 for m in asst_msgs)
    if code_blocks:
        parts.append(f"Code blocks generated: {code_blocks}")

    if decisions:
        parts.append("Architecture decisions:\n" + "\n".join(f"- {d}" for d in decisions[:5]))

    if actions:
        parts.append("Remaining work:\n" + "\n".join(f"- [ ] {a}" for a in actions[:5]))

    return "\n\n".join(parts)


def _actions_summary(
    user_msgs: list, asst_msgs: list,
    decisions: list, actions: list, files: set,
) -> str:
    """Action items only."""
    if not actions:
        return "No action items identified."
    return "Action Items:\n" + "\n".join(f"- [ ] {a}" for a in actions)
