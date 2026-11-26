"""GRIP Context Bridge — injects GRIP system state into HAL's context.

Two modes:
  load_grip_context() — Static snapshot of GRIP state (~500 tokens).
  load_grip_recall(query) — Dynamic KONO semantic recall for a specific query.

Static context answers "what is GRIP right now?"
Dynamic recall answers "what does GRIP know about THIS question?"

Fail-safe: returns empty string if GRIP is not installed.
"""
import json
from pathlib import Path

CLAUDE_DIR = Path.home() / ".claude"
_MAX_CONTEXT_CHARS = 2000  # ~500 tokens


def load_grip_recall(query: str) -> str:
    """Query KONO semantic memory for context relevant to query.

    Delegates to kono_bridge.recall_for_query() which handles import,
    timeout, and formatting. Returns formatted memories or "".
    """
    if not query or not CLAUDE_DIR.exists():
        return ""
    try:
        from lib.hal.kono_bridge import recall_for_query
        return recall_for_query(query)
    except Exception:
        return ""


def load_grip_context() -> str:
    """Read GRIP state and produce a compact context section."""
    if not CLAUDE_DIR.exists():
        return ""

    sections: list[str] = []

    # 1. Core identity (from CLAUDE.md first lines)
    sections.append(_read_identity())

    # 2. Current generation and achievements (from session-context.md)
    sections.append(_read_session_state())

    # 3. Genome pulse
    sections.append(_read_genome())

    # 4. Guild stats
    sections.append(_read_guild_stats())

    # 5. Component counts
    sections.append(_read_component_counts())

    result = "\n".join(s for s in sections if s)
    return result[:_MAX_CONTEXT_CHARS] if result else ""


def _read_identity() -> str:
    """Extract GRIP identity from CLAUDE.md."""
    path = CLAUDE_DIR / ".claude" / "CLAUDE.md"
    if not path.exists():
        return "GRIP is a recursive self-improvement engine at ~/.claude with mechanical safety gates."
    try:
        text = path.read_text()[:500]
        # Find the "What GRIP Is" section
        for line in text.split("\n"):
            if "what grip is" in line.lower():
                idx = text.index(line)
                snippet = text[idx:idx + 300].replace("\n", " ").strip()
                return snippet
    except Exception:
        pass
    return "GRIP: Guarded Response Interception Protocol — recursive self-improvement engine with mechanical safety."


def _read_session_state() -> str:
    """Read current gen, recent achievements from session-context.md."""
    path = CLAUDE_DIR / "rules" / "session-context.md"
    if not path.exists():
        return ""
    try:
        text = path.read_text()
        lines = text.split("\n")
        gen_line = next((l for l in lines if "Gen " in l and "Branch:" in l), "")
        achievements = [l.strip("- ").strip() for l in lines if l.strip().startswith("- Gen ")][:3]
        parts = [gen_line.strip()] if gen_line else []
        if achievements:
            parts.append("Recent: " + "; ".join(achievements))
        return " | ".join(parts) if parts else ""
    except Exception:
        return ""


def _read_genome() -> str:
    """Read genome generation, gene count, fitness."""
    path = CLAUDE_DIR / "cache" / "genome.json"
    if not path.exists():
        path = CLAUDE_DIR / "data" / "genome-shared.json"
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text())
        gen = data.get("generation", "?")
        genes = data.get("genes", {})
        gene_count = len(genes)
        avg_fitness = sum(g.get("fitness", 0) for g in genes.values()) / max(gene_count, 1)
        top = sorted(genes.items(), key=lambda x: x[1].get("fitness", 0), reverse=True)[:3]
        top_names = ", ".join(name for name, _ in top)
        return f"Genome: gen {gen}, {gene_count} genes, avg fitness {avg_fitness:.2f}. Top: {top_names}"
    except Exception:
        return ""


def _read_guild_stats() -> str:
    """Read guild graph stats."""
    data_dir = CLAUDE_DIR / "data"
    if not data_dir.exists():
        return ""
    try:
        graphs = sorted(data_dir.glob("guild-graph-*.json"), key=lambda f: f.stat().st_size, reverse=True)
        if not graphs:
            return ""
        data = json.loads(graphs[0].read_text())  # Largest graph = most content
        name = data.get("guildName", "unknown")
        channels = len(data.get("channels", []))
        members = len(data.get("members", []))
        edges = len(data.get("edges", []))
        return f"Guild: {name} — {channels} channels, {members} members, {edges} edges"
    except Exception:
        return ""


def _read_component_counts() -> str:
    """Count GRIP skills, agents, modes."""
    counts: dict[str, int] = {}
    skills_dir = CLAUDE_DIR / "skills"
    if skills_dir.exists():
        counts["skills"] = sum(1 for d in skills_dir.iterdir() if d.is_dir() and (d / "SKILL.md").exists())
    agents_dir = CLAUDE_DIR / "agents"
    if agents_dir.exists():
        counts["agents"] = sum(1 for f in agents_dir.glob("*.md"))
    modes_dir = CLAUDE_DIR / "modes"
    if modes_dir.exists():
        counts["modes"] = sum(1 for f in modes_dir.glob("*.yaml") if f.stem != "template")
    if not counts:
        return ""
    parts = [f"{v} {k}" for k, v in counts.items()]
    return f"Components: {', '.join(parts)}"
