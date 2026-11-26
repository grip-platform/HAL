"""Platform capability probes shared by tests."""
import tempfile
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def can_create_symlinks() -> bool:
    """Return whether this process has permission to create a symlink."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        target = root / "target"
        link = root / "link"
        target.write_text("probe", encoding="utf-8")
        try:
            link.symlink_to(target)
        except OSError:
            return False
        return link.is_symlink()


