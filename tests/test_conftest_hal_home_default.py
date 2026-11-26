"""tests/conftest.py defaults HAL_HOME and HAL_ROOT to the checkout, and keeps an explicit value.

Modules that read HAL_HOME once at import keep the value they saw when the
tests were collected. A local run that left HAL_HOME unset resolved them to the
live ~/.hal, so collection-time imports read its configs and later writes
reached its state. Each check imports conftest in a new interpreter, so this
run's own environment cannot hide the result.
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_PROBE = ("import os, sys; sys.path.insert(0, 'tests'); import conftest; "
          "print(os.environ['HAL_HOME']); print(os.environ['HAL_ROOT'])")


def _run(probe: str, **overrides: str) -> list[str]:
    env = {k: v for k, v in os.environ.items() if k not in ("HAL_HOME", "HAL_ROOT")}
    env.update(overrides)
    run = subprocess.run([sys.executable, "-c", probe], cwd=REPO, env=env,
                         capture_output=True, text=True, timeout=60)
    if run.returncode != 0:
        raise AssertionError(run.stderr[-2000:])
    lines = run.stdout.strip().splitlines()
    if not lines:
        raise AssertionError(f"the probe printed nothing; stderr: {run.stderr[-2000:]}")
    return lines


def _after_conftest(**overrides: str) -> list[str]:
    return _run(_PROBE, **overrides)[-2:]


class TestHalHomeDefault(unittest.TestCase):
    def test_an_unset_hal_home_and_root_become_the_checkout(self):
        self.assertEqual(_after_conftest(), [str(REPO), str(REPO)])

    def test_a_module_that_reads_hal_home_at_import_gets_the_checkout(self):
        """The reason for the default (HAL#911): lib/hal/home.py reads HAL_HOME once, at import."""
        probe = ("import sys; sys.path.insert(0, 'tests'); import conftest; "
                 "from lib.hal.home import HAL_HOME; print(HAL_HOME)")
        self.assertEqual(_run(probe)[-1], str(REPO))

    def test_an_explicit_value_is_kept(self):
        self.assertEqual(_after_conftest(HAL_HOME="/example/home", HAL_ROOT="/example/root"),
                         ["/example/home", "/example/root"])


if __name__ == "__main__":
    unittest.main()
