"""HAL Home — resolve the Knowledge Engine's home directory.

Priority:
1. HAL_HOME env var (explicit override)
2. $HOME/.hal when HOME is set (on Windows, only with a drive or UNC share)
3. ~/.hal through the platform-native user-home lookup

HAL is harness-agnostic — it does not inherit paths from any specific host
harness. When running inside another harness (Claude Code, Cursor, etc.),
set HAL_HOME explicitly if a non-standard location is required.

Usage:
    from lib.hal.home import HAL_HOME   # fixed at first import
    from lib.hal.home import hal_home   # read each time it is called

An empty HAL_HOME counts as unset in both, as in bin/hal's ${HAL_HOME:-...}.
"""
import os
import sys
from pathlib import Path


def _default_hal_home() -> str:
    user_home = os.environ.get("HOME")
    if not user_home or (sys.platform == "win32" and not os.path.splitdrive(user_home)[0]):
        user_home = os.path.expanduser("~")
    return os.path.join(user_home, ".hal")


HAL_HOME: str = os.environ.get("HAL_HOME") or _default_hal_home()


def hal_home() -> Path:
    """HAL's home, read when called: HAL_HOME, else ~/.hal.

    HAL_HOME above is fixed when this module is first imported, so a process or
    a test that sets HAL_HOME later never reaches it. A path that must follow
    the current setting calls this instead. An empty HAL_HOME counts as unset.
    """
    return Path(os.environ.get("HAL_HOME") or _default_hal_home())


def is_cc_session() -> bool:
    """Check if running inside an external harness session (e.g. Claude Code)."""
    return bool(os.environ.get("CLAUDE_CODE_SESSION_ID"))
