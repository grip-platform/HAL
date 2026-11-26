"""Path constants for HAL's standalone KONO substrate.

Every CLAUDE_DIR reference in the vendored GRIP modules is rewritten to
HAL_DIR and sourced from this module. Keeping all paths in one place makes
the fork explicit and trivial to audit.
"""
from pathlib import Path

from lib.hal.home import hal_home

HAL_DIR: Path = Path.home() / ".hal"
DB_PATH: Path = HAL_DIR / "data" / "embeddings.db"
MODEL_PATH: Path = HAL_DIR / "models" / "nomic-embed-text-v1.5.Q8_0.gguf"
# State lives under HAL_HOME, as gateway_threads.py and agent_registry.py keep
# theirs, so a test run's HAL_HOME keeps the health marker and the session
# counters out of the live install (HAL#909). The database and the model stay
# under ~/.hal, so a host whose HAL_HOME differs does not lose its embeddings.
# Read once, when this module is first imported: tests/conftest.py sets HAL_HOME
# to the checkout before collection, so a test run resolves it there.
STATE_DIR: Path = hal_home() / "state"
LEMBED_CACHE_PATH: Path = STATE_DIR / ".lembed-healthy"
SESSION_COUNTER_FILE: Path = STATE_DIR / "kono-session-counters.json"
