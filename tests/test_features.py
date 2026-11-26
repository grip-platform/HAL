"""Tests for HAL feature flags."""
import os
import sys
import unittest
from unittest.mock import patch

import lib.hal.features as features


class TestFeatureFlags(unittest.TestCase):

    def setUp(self):
        # Reset module state
        features._cache.clear()
        features._loaded = False

    @patch("lib.hal.features._FEATURES_PATH", "/tmp/hal-test-features.json")
    def test_default_disabled(self):
        try:
            os.remove("/tmp/hal-test-features.json")
        except FileNotFoundError:
            pass
        self.assertFalse(features.is_enabled("nonexistent"))

    @patch("lib.hal.features._FEATURES_PATH", "/tmp/hal-test-features2.json")
    def test_default_value(self):
        try:
            os.remove("/tmp/hal-test-features2.json")
        except FileNotFoundError:
            pass
        self.assertTrue(features.is_enabled("missing", default=True))

    @patch("lib.hal.features._FEATURES_PATH", "/tmp/hal-test-features3.json")
    def test_enable_and_check(self):
        try:
            os.remove("/tmp/hal-test-features3.json")
        except FileNotFoundError:
            pass
        features.enable("sse_streaming")
        self.assertTrue(features.is_enabled("sse_streaming"))

    @patch("lib.hal.features._FEATURES_PATH", "/tmp/hal-test-features4.json")
    def test_disable(self):
        try:
            os.remove("/tmp/hal-test-features4.json")
        except FileNotFoundError:
            pass
        features.enable("temp")
        features.disable("temp")
        self.assertFalse(features.is_enabled("temp"))

    @patch("lib.hal.features._FEATURES_PATH", "/tmp/hal-test-features5.json")
    def test_list_features(self):
        try:
            os.remove("/tmp/hal-test-features5.json")
        except FileNotFoundError:
            pass
        features.enable("a")
        features.enable("b")
        features.disable("c")
        listing = features.list_features()
        self.assertEqual(listing["a"], True)
        self.assertEqual(listing["b"], True)
        self.assertEqual(listing["c"], False)

    @patch("lib.hal.features._FEATURES_PATH", "/tmp/hal-test-features6.json")
    def test_persistence(self):
        try:
            os.remove("/tmp/hal-test-features6.json")
        except FileNotFoundError:
            pass
        features.enable("persistent")
        features._save()

        # Reset and reload
        features._cache.clear()
        features._loaded = False
        self.assertTrue(features.is_enabled("persistent"))


class TestFeatureOverrides(unittest.TestCase):

    def setUp(self):
        features._cache.clear()
        features._overrides.clear()
        features._loaded = False

    def _apply_env(self, value: str | None):
        if value is None:
            os.environ.pop("HAL_FEATURE_OVERRIDES", None)
        else:
            os.environ["HAL_FEATURE_OVERRIDES"] = value

    def tearDown(self):
        os.environ.pop("HAL_FEATURE_OVERRIDES", None)
        features._cache.clear()
        features._overrides.clear()
        features._loaded = False

    @patch("lib.hal.features._FEATURES_PATH", "/tmp/hal-test-overrides.json")
    def test_empty_json_has_no_effect(self):
        self._apply_env("{}")
        self.assertFalse(features.is_enabled("some_flag"))
        self.assertTrue(features.is_enabled("some_flag", default=True))

    @patch("lib.hal.features._FEATURES_PATH", "/tmp/hal-test-overrides.json")
    def test_override_enables_flag(self):
        self._apply_env('{"my_flag": true}')
        self.assertTrue(features.is_enabled("my_flag"))

    @patch("lib.hal.features._FEATURES_PATH", "/tmp/hal-test-overrides.json")
    def test_override_disables_flag(self):
        self._apply_env('{"my_flag": false}')
        self.assertFalse(features.is_enabled("my_flag", default=True))

    @patch("lib.hal.features._FEATURES_PATH", "/tmp/hal-test-overrides.json")
    def test_malformed_json_is_graceful(self):
        self._apply_env("not-json{")
        # Must not raise; flag falls back to default
        self.assertFalse(features.is_enabled("any_flag"))

    @patch("lib.hal.features._FEATURES_PATH", "/tmp/hal-test-overrides.json")
    def test_non_object_json_is_graceful(self):
        self._apply_env("[1, 2, 3]")
        self.assertFalse(features.is_enabled("any_flag"))

    @patch("lib.hal.features._FEATURES_PATH", "/tmp/hal-test-overrides.json")
    def test_override_is_in_memory_only(self):
        self._apply_env('{"temp_flag": true}')
        self.assertTrue(features.is_enabled("temp_flag"))
        # Remove env var and reset state — flag must revert
        self._apply_env(None)
        features._cache.clear()
        features._overrides.clear()
        features._loaded = False
        self.assertFalse(features.is_enabled("temp_flag"))

    @patch("lib.hal.features._FEATURES_PATH", "/tmp/hal-test-overrides.json")
    def test_list_features_includes_overrides(self):
        self._apply_env('{"extra": true}')
        self.assertTrue(features.list_features().get("extra"))

    @patch("lib.hal.features._FEATURES_PATH", "/tmp/hal-test-overrides.json")
    def test_override_wins_over_disk_cache(self):
        # Seed disk cache with False, override with True
        features._cache["cached_flag"] = False
        features._loaded = True
        features._overrides = {"cached_flag": True}
        self.assertTrue(features.is_enabled("cached_flag"))


if __name__ == "__main__":
    unittest.main()
