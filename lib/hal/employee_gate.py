"""HAL Employee Gate — CC employee-gated features made native.

Claude Code gates several quality features behind `process.env.USER_TYPE
=== 'ant'` (Anthropic employees only). HAL implements these features
natively so all users get them.

Gated features HAL implements:
- Post-edit verification (verify.py)
- Plan verification before execution
- High-effort reasoning by default
- Tool search (dynamic tool discovery)
- Session persistence across restarts
- Resume interrupted turns
- Coordinator mode (multi-agent)
- Streaming output format

This module tracks which employee-gated features HAL has implemented
natively vs which still need work.

Usage:
    from lib.hal.employee_gate import get_feature_status, get_implementation_score

    status = get_feature_status()
    for feature in status:
        print(f"[{feature['status']}] {feature['name']}")
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class GatedFeature:
    """A CC employee-gated feature and its HAL implementation status."""
    name: str
    cc_gate: str  # The env var or check that gates it in CC
    hal_module: str  # HAL module that implements it, or "" if not yet
    status: str  # "implemented", "partial", "planned", "not_needed"
    notes: str = ""


# Comprehensive list of employee-gated features found in CC analysis
_GATED_FEATURES: list[GatedFeature] = [
    GatedFeature("Post-edit verification", "USER_TYPE=ant", "verify.py", "implemented",
                 "py_compile for Python, tsc for TS, node --check for JS, cargo check for Rust, go vet for Go"),
    GatedFeature("Plan verification", "CLAUDE_CODE_VERIFY_PLAN", "cc_env_bridge.py", "implemented",
                 "Enabled via env var bridge when running inside CC"),
    GatedFeature("Max-effort reasoning", "CLAUDE_CODE_EFFORT_LEVEL=max", "cc_env_bridge.py", "implemented",
                 "Default in HAL via env bridge — ceiling thinking budget every turn"),
    GatedFeature("Tool search", "ENABLE_TOOL_SEARCH", "tool_registry.py", "implemented",
                 "HAL-native dynamic tool discovery"),
    GatedFeature("Session persistence", "ENABLE_SESSION_PERSISTENCE", "session_manager.py", "implemented",
                 "HAL sessions are always persistent (daemon mode)"),
    GatedFeature("Resume interrupted turns", "CLAUDE_CODE_RESUME_INTERRUPTED_TURN", "cc_env_bridge.py", "implemented",
                 "Enabled via env bridge"),
    GatedFeature("Task management", "CLAUDE_CODE_ENABLE_TASKS", "workflow.py", "implemented",
                 "HAL workflow engine replaces CC task system"),
    GatedFeature("Coordinator mode", "CLAUDE_CODE_COORDINATOR_MODE", "agent_runner.py", "implemented",
                 "HAL recursive agent spawning with configurable depth"),
    GatedFeature("No-flicker renderer", "CLAUDE_CODE_NO_FLICKER", "cc_env_bridge.py", "implemented",
                 "Virtualised viewport — enabled via env bridge"),
    GatedFeature("Streaming output", "CLAUDE_CODE_STREAMING_OUTPUT", "server.py", "partial",
                 "SSE endpoint exists, full streaming in progress"),
    GatedFeature("Proactive agent", "CLAUDE_CODE_PROACTIVE", "suggestions.py", "implemented",
                 "HAL smart suggestions engine"),
    GatedFeature("Agent teams", "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS", "mesh.py", "implemented",
                 "HAL inter-instance mesh discovery"),
    GatedFeature("LSP integration", "ENABLE_LSP_TOOL", "code_search.py", "partial",
                 "AST-aware search implemented, full LSP planned"),
    GatedFeature("Embedded search", "EMBEDDED_SEARCH_TOOLS", "conversation_search.py", "implemented",
                 "HAL-native conversation search"),
    GatedFeature("Telemetry disabled", "DISABLE_TELEMETRY", "cc_env_bridge.py", "implemented",
                 "All telemetry mechanically blocked"),
    GatedFeature("Brief mode", "CLAUDE_CODE_BRIEF", "cc_env_bridge.py", "implemented",
                 "Available as optional env var"),
]


def get_feature_status() -> list[dict]:
    """Get implementation status of all employee-gated features."""
    return [
        {
            "name": f.name,
            "cc_gate": f.cc_gate,
            "hal_module": f.hal_module,
            "status": f.status,
            "notes": f.notes,
        }
        for f in _GATED_FEATURES
    ]


def get_implementation_score() -> dict:
    """Calculate overall implementation score."""
    total = len(_GATED_FEATURES)
    implemented = sum(1 for f in _GATED_FEATURES if f.status == "implemented")
    partial = sum(1 for f in _GATED_FEATURES if f.status == "partial")
    planned = sum(1 for f in _GATED_FEATURES if f.status == "planned")

    score = (implemented + partial * 0.5) / total if total else 0
    return {
        "total_features": total,
        "implemented": implemented,
        "partial": partial,
        "planned": planned,
        "score": round(score, 2),
        "score_pct": f"{score:.0%}",
    }


def list_remaining() -> list[GatedFeature]:
    """List features not yet fully implemented."""
    return [f for f in _GATED_FEATURES if f.status not in ("implemented",)]
