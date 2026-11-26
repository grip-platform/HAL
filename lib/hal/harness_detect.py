"""HAL Harness Detection — auto-detect which agentic tool (if any) is running.

HAL works standalone or inside any harness. This module detects the active
harness so HAL can adapt its behaviour (e.g., delegate tool dispatch to
the harness when available, or run tools natively when standalone).

Usage:
    from lib.hal.harness_detect import detect_harness, Harness

    harness = detect_harness()
    if harness == Harness.STANDALONE:
        # HAL runs everything natively
    elif harness == Harness.CLAUDE_CODE:
        # Can optionally delegate to CC for premium features
"""
import os
from dataclasses import dataclass, field
from enum import Enum


class Harness(str, Enum):
    """Known agentic harnesses."""
    STANDALONE = "standalone"
    CLAUDE_CODE = "claude-code"
    GEMINI_CLI = "gemini-cli"
    CODEX = "codex"
    CURSOR = "cursor"
    ANTIGRAVITY = "antigravity"  # Gemini-based Antigravity CLI/IDE (a2a_peer.ANTIGRAVITY peer)
    OPENCODE = "opencode"  # OpenCode CLI (sst/opencode) routes its `hal` provider to gateway :4010
    UNKNOWN = "unknown"


@dataclass
class HarnessInfo:
    """Detected harness metadata surfaced in --status."""
    harness: Harness = Harness.STANDALONE
    sparse_paths: list[str] = field(default_factory=list)  # CC 2.1.76 worktree.sparsePaths


# Detection rules: env var -> harness. Order matters only if two are ever set
# at once (first match wins); in practice exactly one harness env is present.
# ANTIGRAVITY_SESSION_ID follows the established *_SESSION_ID convention shared by
# Claude Code / Codex / Cursor. Detection is fail-safe: an absent/renamed var
# simply falls through to STANDALONE (no false interop claim) — the gateway and
# the a2a peer mesh work regardless of detection, so this is advisory metadata.
_DETECTION_RULES = [
    ("CLAUDE_CODE_SESSION_ID", Harness.CLAUDE_CODE),
    ("GEMINI_CLI_SESSION", Harness.GEMINI_CLI),
    ("CODEX_SESSION_ID", Harness.CODEX),
    ("CURSOR_SESSION_ID", Harness.CURSOR),
    ("ANTIGRAVITY_SESSION_ID", Harness.ANTIGRAVITY),
    ("OPENCODE_SESSION_ID", Harness.OPENCODE),
]


def detect_harness() -> Harness:
    """Detect which agentic harness is active. Returns STANDALONE if none."""
    for env_var, harness in _DETECTION_RULES:
        if os.environ.get(env_var):
            return harness
    return Harness.STANDALONE


def detect_harness_info() -> HarnessInfo:
    """Detect harness and collect extended metadata (sparse paths, etc.)."""
    harness = detect_harness()
    sparse_raw = os.environ.get("CLAUDE_WORKTREE_SPARSE_PATHS", "")
    sparse_paths = [p.strip() for p in sparse_raw.split(",") if p.strip()] if sparse_raw else []
    return HarnessInfo(harness=harness, sparse_paths=sparse_paths)


def is_standalone() -> bool:
    """True when HAL is running without any external harness."""
    return detect_harness() == Harness.STANDALONE
