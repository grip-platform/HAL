"""Goodhart-proof tests for lib.hal.env_boot.load_env.

Every assertion checks an exact resulting value or count — a broken
loader (wrong value, no-op, override inverted, swallowed line) fails at
least one test. No call-count-only assertions.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.hal.env_boot import default_env_path, load_env


class TestEnvBoot(unittest.TestCase):
    def setUp(self):
        self._saved = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved)

    def _write(self, body: str) -> str:
        d = tempfile.mkdtemp()
        p = os.path.join(d, ".env")
        Path(p).write_text(body, encoding="utf-8")
        return p

    def test_loads_exact_value(self):
        os.environ.pop("HAL_T_TOKEN", None)
        p = self._write("HAL_T_TOKEN=oauth-canonical-123\n")
        n = load_env(p)
        self.assertEqual(n, 1)
        self.assertEqual(os.environ["HAL_T_TOKEN"], "oauth-canonical-123")

    def test_process_env_wins_by_default(self):
        os.environ["HAL_T_TOKEN"] = "from-process"
        p = self._write("HAL_T_TOKEN=from-file\n")
        n = load_env(p)
        self.assertEqual(n, 0)
        self.assertEqual(os.environ["HAL_T_TOKEN"], "from-process")

    def test_override_true_clobbers(self):
        os.environ["HAL_T_TOKEN"] = "from-process"
        p = self._write("HAL_T_TOKEN=from-file\n")
        n = load_env(p, override=True)
        self.assertEqual(n, 1)
        self.assertEqual(os.environ["HAL_T_TOKEN"], "from-file")

    def test_missing_file_is_noop(self):
        self.assertEqual(load_env("/no/such/dir/.env"), 0)

    def test_directory_path_is_noop(self):
        self.assertEqual(load_env(tempfile.mkdtemp()), 0)

    def test_comments_and_blanks_ignored(self):
        os.environ.pop("HAL_T_A", None)
        p = self._write("# a comment\n\n   \nHAL_T_A=1\n")
        n = load_env(p)
        self.assertEqual(n, 1)
        self.assertEqual(os.environ["HAL_T_A"], "1")

    def test_quoted_value_unwrapped(self):
        os.environ.pop("HAL_T_Q", None)
        p = self._write('HAL_T_Q="spaced value"\n')
        load_env(p)
        self.assertEqual(os.environ["HAL_T_Q"], "spaced value")

    def test_export_prefix_tolerated(self):
        os.environ.pop("HAL_T_E", None)
        p = self._write("export HAL_T_E=ev\n")
        load_env(p)
        self.assertEqual(os.environ["HAL_T_E"], "ev")

    def test_default_path_follows_hal_home(self):
        os.environ["HAL_HOME"] = "/tmp/hal-home-probe"
        self.assertEqual(default_env_path(), Path("/tmp/hal-home-probe/.env"))


if __name__ == "__main__":
    unittest.main()
