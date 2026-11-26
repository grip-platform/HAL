"""Let kono reach GRIP's Ollama-backed ``embeddings`` package when GRIP is installed.

GRIP's ``embeddings`` package embeds through the local Ollama daemon
(``OllamaEmbeddingBackend``, nomic-embed-text, 768 dimensions).
``MemorySubstrate.store_session_memory`` imports it by name. But HAL's services
start with only HAL's own folders on ``sys.path``, so that import always failed,
and the method returned before storing anything at all (HAL#925). On
2026-11-26 V>> chose Ollama over sqlite-lembed, whose ``lembed()`` fails under
APSW's SQLite 3.53.2.

GRIP's ``lib`` is appended after HAL's own entries, so a HAL module with the
same name still wins. ``lib/hal/decide/shield.py`` reaches GRIP's redaction
shield the same way. The folder stays on the path, because the package imports
GRIP's own modules (``claude_dir``, ``kono_schema``) after it loads. Without
GRIP, nothing changes and the caller keeps its existing fallback.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# The file that shows a GRIP lib folder carries the Ollama backend.
MARKER = Path("embeddings") / "ollama_backend.py"


def grip_lib_dir() -> Path:
    """GRIP's ``lib`` folder: ``$CLAUDE_DIR/lib``, else ``~/.claude/lib``."""
    return (Path(os.environ.get("CLAUDE_DIR", "~/.claude")).expanduser() / "lib").resolve()


def ensure_importable() -> bool:
    """Append GRIP's ``lib`` to ``sys.path`` when it holds the Ollama backend.

    Returns True when ``import embeddings`` can then find GRIP's package, and
    False, changing nothing, when GRIP is not installed.
    """
    directory = grip_lib_dir()
    if not (directory / MARKER).is_file():
        return False
    if str(directory) not in sys.path:
        sys.path.append(str(directory))
    return True
