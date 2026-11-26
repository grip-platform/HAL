"""lib/hal/utils/cch.py finds its config files when HAL_HOME has none.

tests/conftest.py gives every test an empty HAL_HOME, and cch.py read
configs/models.toml from HAL_HOME when it was first imported. A test that
imported it first, through mock.patch("lib.hal.utils.cch.CCHSigner") or the
Anthropic provider's function-level import, raised ImportError ("HAL model
registry missing"), and mock.patch reported it as a missing attribute. Four
tests in tests/test_happi_providers_anthropic.py failed whenever that file ran
without an earlier module importing cch (measured 2026-11-26). The mode table
had the same dependency and fell back to the defaults without a word.

Each check imports cch afresh in a subprocess, so a copy already loaded in this
process cannot hide the failure.
"""
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_PROBE = ("import json; from lib.hal.utils import cch; "
          "print(json.dumps({'models': sorted(cch.MODEL_CAPABILITIES), "
          "'modes': sorted(cch._MODE_STRATEGIES)}))")


def _fresh_import(hal_home: Path) -> dict:
    """Import cch in a new interpreter with ``hal_home`` as HAL_HOME; return what it loaded."""
    env = {**os.environ, "HAL_HOME": str(hal_home), "PYTHONPATH": str(REPO)}
    run = subprocess.run([sys.executable, "-c", _PROBE], cwd=REPO, env=env,
                         capture_output=True, text=True, timeout=60)
    if run.returncode != 0:
        raise AssertionError(f"importing cch failed:\n{run.stderr[-2000:]}")
    return json.loads(run.stdout.strip().splitlines()[-1])


def _shipped(name: str, table: str) -> list[str]:
    """The entries a config file in this checkout defines, as the loader keeps them."""
    with (REPO / "configs" / name).open("rb") as fh:
        entries = tomllib.load(fh).get(table, {})
    return sorted(key for key, value in entries.items() if isinstance(value, dict))


class TestConfigResolution(unittest.TestCase):
    def test_an_empty_hal_home_still_gets_the_shipped_model_registry(self):
        with tempfile.TemporaryDirectory() as home:
            got = _fresh_import(Path(home))
        expected = _shipped("models.toml", "models")
        self.assertTrue(expected)
        self.assertEqual(got["models"], expected)

    def test_an_empty_hal_home_still_gets_the_shipped_mode_table(self):
        with tempfile.TemporaryDirectory() as home:
            got = _fresh_import(Path(home))
        expected = _shipped("mode_context_strategies.toml", "modes")
        self.assertTrue(expected)
        self.assertEqual(got["modes"], expected)

    def test_a_registry_in_hal_home_still_wins(self):
        with tempfile.TemporaryDirectory() as home:
            configs = Path(home) / "configs"
            configs.mkdir()
            (configs / "models.toml").write_text('[models."example-model"]\nadaptive_thinking = true\n')
            got = _fresh_import(Path(home))
        self.assertEqual(got["models"], ["example-model"])


if __name__ == "__main__":
    unittest.main()
