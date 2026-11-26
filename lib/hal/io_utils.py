"""HAL I/O Utilities — shared file operations.

Extracted to eliminate DRY violation across agent_registry, background_executor,
and environment modules.
"""
import os


def atomic_write(path: str, data: str) -> None:
    """Write atomically via temp file + os.replace."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(data)
    os.replace(tmp, path)


def dataclass_from_dict(cls, data: dict):
    """Create a dataclass instance from a dict, filtering unknown keys."""
    known = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in data.items() if k in known})
