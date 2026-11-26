"""Tests for HAL CC environment bridge."""
import os
import sys
import unittest

from lib.hal.cc_env_bridge import (
    _NEVER_AUTO,
    _OPTIONAL,
    _RECOMMENDED,
    get_optional_env,
    get_recommended_env,
)


class TestRecommendedEnv(unittest.TestCase):

    def test_returns_list(self):
        result = get_recommended_env()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_has_required_fields(self):
        result = get_recommended_env()
        for item in result:
            self.assertIn("variable", item)
            self.assertIn("value", item)
            self.assertIn("description", item)

    def test_telemetry_disabled(self):
        self.assertIn("DISABLE_TELEMETRY", _RECOMMENDED)
        self.assertEqual(_RECOMMENDED["DISABLE_TELEMETRY"], "1")

    def test_nonessential_traffic_disabled(self):
        self.assertIn("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", _RECOMMENDED)

    def test_no_flicker_enabled(self):
        self.assertIn("CLAUDE_CODE_NO_FLICKER", _RECOMMENDED)
        self.assertEqual(_RECOMMENDED["CLAUDE_CODE_NO_FLICKER"], "1")


class TestOptionalEnv(unittest.TestCase):

    def test_returns_list(self):
        result = get_optional_env()
        self.assertIsInstance(result, list)

    def test_coordinator_mode_optional(self):
        names = {item["variable"] for item in get_optional_env()}
        self.assertIn("CLAUDE_CODE_COORDINATOR_MODE", names)


class TestNeverAutoEnv(unittest.TestCase):

    def test_undercover_never_auto(self):
        self.assertIn("CLAUDE_CODE_UNDERCOVER", _NEVER_AUTO)

    def test_command_injection_bypass_never_auto(self):
        self.assertIn("CLAUDE_CODE_DISABLE_COMMAND_INJECTION_CHECK", _NEVER_AUTO)

    def test_telemetry_not_in_never_auto(self):
        self.assertNotIn("DISABLE_TELEMETRY", _NEVER_AUTO)


class TestNoOverlap(unittest.TestCase):

    def test_recommended_and_never_auto_disjoint(self):
        overlap = set(_RECOMMENDED.keys()) & _NEVER_AUTO
        self.assertEqual(overlap, set(), f"Overlap: {overlap}")

    def test_optional_and_never_auto_disjoint(self):
        optional_keys = {k for k in _OPTIONAL}
        overlap = optional_keys & _NEVER_AUTO
        self.assertEqual(overlap, set(), f"Overlap: {overlap}")


class TestEnableCcFeaturesDefensiveFilter(unittest.TestCase):
    """enable_cc_features() must refuse _NEVER_AUTO keys even if a future
    diff accidentally lists them in _RECOMMENDED or _OPTIONAL. This is the
    mechanical backstop for the list-level convention."""

    def setUp(self):
        # Snapshot originals so the test is hermetic
        from lib.hal import cc_env_bridge as m
        self._orig_recommended = dict(m._RECOMMENDED)
        self._orig_optional = dict(m._OPTIONAL)
        self._danger_var = "CLAUDE_CODE_DISABLE_COMMAND_INJECTION_CHECK"
        os.environ.pop(self._danger_var, None)

    def tearDown(self):
        from lib.hal import cc_env_bridge as m
        m._RECOMMENDED.clear()
        m._RECOMMENDED.update(self._orig_recommended)
        m._OPTIONAL.clear()
        m._OPTIONAL.update(self._orig_optional)
        os.environ.pop(self._danger_var, None)

    def test_recommended_pollution_is_blocked(self):
        from lib.hal import cc_env_bridge as m
        m._RECOMMENDED[self._danger_var] = "true"
        enabled = m.enable_cc_features()
        self.assertNotIn(self._danger_var, enabled)
        self.assertNotIn(self._danger_var, os.environ)

    def test_optional_pollution_is_blocked(self):
        from lib.hal import cc_env_bridge as m
        m._OPTIONAL[self._danger_var] = ("true", "Security bypass")
        enabled = m.enable_cc_features(include_optional=True)
        self.assertNotIn(self._danger_var, enabled)
        self.assertNotIn(self._danger_var, os.environ)


if __name__ == "__main__":
    unittest.main()
