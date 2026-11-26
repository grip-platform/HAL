"""HAL Home — resolve the Knowledge Engine's home directory.

Priority:
1. HAL_HOME env var (explicit override)
2. ~/.hal (standalone default)

HAL is harness-agnostic — it does not inherit paths from any specific host
harness. When running inside another harness (Claude Code, Cursor, etc.),
set HAL_HOME explicitly if a non-standard location is required.

Usage:
    from lib.hal.home import HAL_HOME
"""
import os

HAL_HOME: str = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))


def is_cc_session() -> bool:
    """Check if running inside an external harness session (e.g. Claude Code)."""
    return bool(os.environ.get("CLAUDE_CODE_SESSION_ID"))
