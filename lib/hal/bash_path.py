"""Find Bash for HAPPI bootstrap execution and public shell tests."""
import shutil
import sys
from itertools import product
from pathlib import Path


def find_bash(platform=None, which=shutil.which):
    """Return a usable Bash path, or ``None`` when Windows has no Git Bash."""
    platform = platform or {"win32": "windows", "darwin": "macos"}.get(
        sys.platform, "linux"
    )
    if platform != "windows":
        return "bash"

    on_path = which("bash")
    if on_path and "system32" not in on_path.lower():
        return on_path

    git = which("git")
    # Cover Git\cmd\git.exe and Git\mingw64\bin\git.exe layouts.
    roots = list(Path(git).resolve().parents)[:3] if git else []
    candidates = (
        root / sub / "bash.exe"
        for root, sub in product(roots, ("bin", "usr/bin"))
    )
    return next((str(candidate) for candidate in candidates if candidate.is_file()), None)
