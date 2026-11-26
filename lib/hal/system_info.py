"""HAL System Info — version, environment, and runtime information.

Provides a single source of truth for HAL's version, build info,
environment details, and runtime capabilities.

Usage:
    from lib.hal.system_info import get_system_info, HAL_VERSION
    info = get_system_info()
"""
import os
import platform
import sys

from lib.hal.home import HAL_HOME as _HAL_HOME

HAL_VERSION = "1.0.0"
HAL_CODENAME = "Knowledge Engine"


def get_system_info() -> dict:
    """Get comprehensive system information."""
    modules = _count_modules()
    return {
        "hal_version": HAL_VERSION,
        "hal_codename": HAL_CODENAME,
        "hal_home": _HAL_HOME,
        "modules": modules,
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "pid": os.getpid(),
    }


def get_version_string() -> str:
    """Human-readable version string."""
    return f"HAL {HAL_VERSION} ({HAL_CODENAME})"


def _count_modules() -> int:
    """Count HAL Python modules."""
    mod_dir = os.path.join(_HAL_HOME, "lib", "hal")
    if not os.path.isdir(mod_dir):
        return 0
    return sum(1 for f in os.listdir(mod_dir) if f.endswith(".py") and f != "__init__.py")
