"""Path constants for HAL's standalone KONO substrate.

Every CLAUDE_DIR reference in the vendored GRIP modules is rewritten to
HAL_DIR and sourced from this module. Keeping all paths in one place makes
the fork explicit and trivial to audit.
"""
from pathlib import Path

HAL_DIR: Path = Path.home() / ".hal"
DB_PATH: Path = HAL_DIR / "data" / "embeddings.db"
MODEL_PATH: Path = HAL_DIR / "models" / "nomic-embed-text-v1.5.Q8_0.gguf"
STATE_DIR: Path = HAL_DIR / "state"
LEMBED_CACHE_PATH: Path = STATE_DIR / ".lembed-healthy"
SESSION_COUNTER_FILE: Path = STATE_DIR / "kono-session-counters.json"
