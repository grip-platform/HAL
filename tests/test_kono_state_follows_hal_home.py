"""kono keeps its state under HAL_HOME, and its database and model under ~/.hal (HAL#909).

With STATE_DIR rooted at ~/.hal, a local test run wrote the lembed health marker
into the live install. Each check imports the constants afresh in a subprocess,
so a copy already loaded in this process cannot hide the result.
"""
from __future__ import annotations  # "Path | None" below; HAL supports Python 3.9

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_PROBE = ("import json; from lib.hal.kono import constants as c; "
          "print(json.dumps({k: str(getattr(c, k)) for k in "
          "('STATE_DIR', 'LEMBED_CACHE_PATH', 'SESSION_COUNTER_FILE', 'DB_PATH', 'MODEL_PATH')}))")


def _constants(home: Path, *, hal_home: Path | None = None) -> dict:
    """The kono paths a fresh interpreter computes. hal_home None leaves HAL_HOME unset."""
    env = {k: v for k, v in os.environ.items() if k != "HAL_HOME"}
    # Windows Python reads the home folder from USERPROFILE and ignores HOME.
    env.update({"HOME": str(home), "USERPROFILE": str(home), "PYTHONPATH": str(REPO)})
    if hal_home is not None:
        env["HAL_HOME"] = str(hal_home)
    run = subprocess.run([sys.executable, "-c", _PROBE], cwd=REPO, env=env,
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr[-2000:]
    lines = run.stdout.strip().splitlines()
    assert lines, f"the probe printed nothing; stderr: {run.stderr[-2000:]}"
    return json.loads(lines[-1])


def _assert_state_under(got: dict, state: Path) -> None:
    assert got["STATE_DIR"] == str(state)
    assert got["LEMBED_CACHE_PATH"] == str(state / ".lembed-healthy")
    assert got["SESSION_COUNTER_FILE"] == str(state / "kono-session-counters.json")


def test_state_follows_hal_home(tmp_path):
    _assert_state_under(_constants(tmp_path / "home", hal_home=tmp_path / "halhome"), tmp_path / "halhome" / "state")


def test_an_unset_hal_home_keeps_state_under_the_home_folder(tmp_path):
    """Ordinary installs leave HAL_HOME unset, and nothing moves for them (HAL#915)."""
    _assert_state_under(_constants(tmp_path / "home"), tmp_path / "home" / ".hal" / "state")


def test_the_database_and_model_stay_under_the_home_folder(tmp_path):
    got = _constants(tmp_path / "home", hal_home=tmp_path / "halhome")
    assert got["DB_PATH"] == str(tmp_path / "home" / ".hal" / "data" / "embeddings.db")
    assert got["MODEL_PATH"] == str(tmp_path / "home" / ".hal" / "models" / "nomic-embed-text-v1.5.Q8_0.gguf")
