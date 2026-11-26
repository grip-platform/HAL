"""Context Collapse — intelligent context compression when token budget is exceeded.

When the assembled context (system prompt + conversation history + tool results)
approaches the model's context window, collapse sections by priority to fit.

Collapse strategy (ordered by expendability):
1. Git recent commits (expendable — git log is always available)
2. Git status (expendable — git status is always available)
3. Skill descriptions (keep names, drop descriptions)
4. Memory entries (keep recent, drop old)
5. CLAUDE.md chain (keep project, summarise global)
6. Conversation history (oldest turns first, keep system + last 4 turns)

Usage:
    from lib.hal.context_collapse import collapse_context

    collapsed = collapse_context(system_prompt, messages, budget_tokens=100000)
    print(f"Reduced from {original_tokens} to {collapsed.token_estimate}")
"""
from dataclasses import dataclass


@dataclass
class CollapseResult:
    """Result of a context collapse operation."""
    system_prompt: str
    messages: list[dict]
    token_estimate: int
    sections_collapsed: list[str]
    original_tokens: int


def _estimate_tokens(text: str) -> int:
    """Approximate token count (~4 chars per token)."""
    return max(1, len(text) // 4) if text else 0


def _total_tokens(system: str, messages: list[dict]) -> int:
    """Estimate total tokens across system prompt + messages."""
    total = _estimate_tokens(system)
    for msg in messages:
        total += _estimate_tokens(msg.get("content", ""))
    return total


def collapse_context(
    system_prompt: str,
    messages: list[dict],
    budget_tokens: int = 100_000,
) -> CollapseResult:
    """Collapse context to fit within a token budget.

    Applies collapse strategies in priority order (most expendable first)
    until the total fits within the budget. Returns early if no collapse needed.
    """
    original = _total_tokens(system_prompt, messages)
    if original <= budget_tokens:
        return CollapseResult(
            system_prompt=system_prompt,
            messages=messages,
            token_estimate=original,
            sections_collapsed=[],
            original_tokens=original,
        )

    collapsed_sections: list[str] = []
    current_system = system_prompt
    current_messages = list(messages)

    # Strategy 1: Strip git sections from system prompt
    current_system = _strip_git_sections(current_system, collapsed_sections)
    if _total_tokens(current_system, current_messages) <= budget_tokens:
        return _result(current_system, current_messages, original, collapsed_sections)

    # Strategy 2: Compress skill listings (names only)
    current_system = _compress_skills(current_system, collapsed_sections)
    if _total_tokens(current_system, current_messages) <= budget_tokens:
        return _result(current_system, current_messages, original, collapsed_sections)

    # Strategy 3: Trim conversation history (keep last 4 turns + system)
    current_messages = _trim_history(current_messages, keep_turns=4, collapsed=collapsed_sections)
    if _total_tokens(current_system, current_messages) <= budget_tokens:
        return _result(current_system, current_messages, original, collapsed_sections)

    # Strategy 4: Aggressive history trim (keep last 2 turns)
    current_messages = _trim_history(current_messages, keep_turns=2, collapsed=collapsed_sections)
    if _total_tokens(current_system, current_messages) <= budget_tokens:
        return _result(current_system, current_messages, original, collapsed_sections)

    # Strategy 5: Truncate system prompt to first 2000 chars
    if len(current_system) > 2000:
        current_system = current_system[:2000] + "\n\n[Context collapsed to fit budget]"
        collapsed_sections.append("system_prompt_truncated")

    return _result(current_system, current_messages, original, collapsed_sections)


def _strip_git_sections(system: str, collapsed: list[str]) -> str:
    """Remove git status and recent commits sections."""
    lines = system.split("\n")
    filtered: list[str] = []
    skip = False
    for line in lines:
        lower = line.lower().strip()
        if lower.startswith("git status:") or lower.startswith("recent commits:"):
            skip = True
            collapsed.append(f"git_{lower.split(':')[0].replace(' ', '_')}")
            continue
        if skip and (line.startswith("  ") or line.startswith("\t") or not line.strip()):
            continue
        skip = False
        filtered.append(line)
    return "\n".join(filtered)


def _compress_skills(system: str, collapsed: list[str]) -> str:
    """Replace skill descriptions with names only."""
    lines = system.split("\n")
    in_skills = False
    filtered: list[str] = []
    skill_names: list[str] = []
    for line in lines:
        if "available skills" in line.lower() or "skill:" in line.lower():
            in_skills = True
            filtered.append(line)
            continue
        if in_skills:
            if line.startswith("  - ") or line.startswith("- /"):
                name = line.split(":")[0].strip().lstrip("- /")
                skill_names.append(name)
                continue
            if not line.strip():
                in_skills = False
                if skill_names:
                    filtered.append(f"  Skills: {', '.join(skill_names)}")
                    collapsed.append("skills_compressed")
                    skill_names = []
        filtered.append(line)
    return "\n".join(filtered)


def _trim_history(
    messages: list[dict], keep_turns: int, collapsed: list[str],
) -> list[dict]:
    """Keep system messages + last N user-assistant turn pairs."""
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    # Keep last keep_turns * 2 messages (user + assistant pairs)
    keep_count = keep_turns * 2
    if len(non_system) <= keep_count:
        return messages

    trimmed = non_system[-keep_count:]
    dropped = len(non_system) - keep_count
    collapsed.append(f"history_trimmed_{dropped}_messages")
    return system_msgs + trimmed


def _result(
    system: str, messages: list[dict], original: int, collapsed: list[str],
) -> CollapseResult:
    return CollapseResult(
        system_prompt=system,
        messages=messages,
        token_estimate=_total_tokens(system, messages),
        sections_collapsed=collapsed,
        original_tokens=original,
    )
