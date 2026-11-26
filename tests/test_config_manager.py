"""Tests for HAL config manager."""
import json
import os
import sys
import tempfile
import unittest

from lib.hal.config_manager import ConfigManager, _DEFAULTS, _parse_env_value


class TestConfigManager(unittest.TestCase):

    def test_defaults_loaded(self):
        config = ConfigManager(config_path="/nonexistent/path.json")
        self.assertEqual(config.get("server.port"), 3850)
        self.assertFalse(config.get("telemetry.enabled"))

    def test_get_with_default(self):
        config = ConfigManager(config_path="/nonexistent/path.json")
        self.assertEqual(config.get("nonexistent.key", "fallback"), "fallback")

    def test_set(self):
        config = ConfigManager(config_path="/nonexistent/path.json")
        config.set("server.port", 8080)
        self.assertEqual(config.get("server.port"), 8080)

    def test_all(self):
        config = ConfigManager(config_path="/nonexistent/path.json")
        all_cfg = config.all()
        self.assertIn("server.port", all_cfg)
        self.assertGreater(len(all_cfg), 10)

    def test_reset(self):
        config = ConfigManager(config_path="/nonexistent/path.json")
        config.set("server.port", 9999)
        config.reset("server.port")
        self.assertEqual(config.get("server.port"), 3850)

    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            config = ConfigManager(config_path=path)
            config.set("server.port", 4000)
            config.save()

            config2 = ConfigManager(config_path=path)
            self.assertEqual(config2.get("server.port"), 4000)
        finally:
            os.unlink(path)

    def test_file_overrides_defaults(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"budget.max_usd": 10.0}, f)
            path = f.name
        try:
            config = ConfigManager(config_path=path)
            self.assertEqual(config.get("budget.max_usd"), 10.0)
        finally:
            os.unlink(path)


class TestParseEnvValue(unittest.TestCase):

    def test_bool_true(self):
        self.assertTrue(_parse_env_value("true"))
        self.assertTrue(_parse_env_value("1"))
        self.assertTrue(_parse_env_value("yes"))

    def test_bool_false(self):
        self.assertFalse(_parse_env_value("false"))
        self.assertFalse(_parse_env_value("0"))

    def test_int(self):
        self.assertEqual(_parse_env_value("42"), 42)

    def test_float(self):
        self.assertEqual(_parse_env_value("3.14"), 3.14)

    def test_string(self):
        self.assertEqual(_parse_env_value("hello"), "hello")


class TestDefaults(unittest.TestCase):

    def test_has_server_port(self):
        self.assertIn("server.port", _DEFAULTS)

    def test_telemetry_disabled(self):
        self.assertFalse(_DEFAULTS["telemetry.enabled"])

    def test_has_enough_defaults(self):
        self.assertGreater(len(_DEFAULTS), 15)


if __name__ == "__main__":
    unittest.main()
