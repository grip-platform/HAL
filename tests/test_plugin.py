"""Tests for HAL provider plugin system."""
import os
import sys
import tempfile
import unittest

from lib.hal.plugin import PluginInfo, PluginLoader, PluginReport


class TestPluginInfo(unittest.TestCase):

    def test_to_dict(self):
        info = PluginInfo(name="test", path="/tmp/test.py", loaded=True)
        d = info.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertTrue(d["loaded"])


class TestPluginReport(unittest.TestCase):

    def test_loaded_count(self):
        report = PluginReport(plugins=[
            PluginInfo("a", "/a.py", loaded=True),
            PluginInfo("b", "/b.py", loaded=False),
        ])
        self.assertEqual(report.loaded_count, 1)

    def test_summary(self):
        report = PluginReport(plugins=[PluginInfo("a", "/a.py", loaded=True)])
        self.assertIn("1/1", report.summary)


class TestPluginLoader(unittest.TestCase):

    def test_discover_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = PluginLoader(tmpdir)
            self.assertEqual(len(loader.discover().plugins), 0)

    def test_discover_nonexistent_dir(self):
        loader = PluginLoader("/nonexistent/dir/12345")
        self.assertEqual(len(loader.discover().plugins), 0)

    def test_discover_valid_plugin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "my_prov.py"), "w") as f:
                f.write("class ProviderPlugin:\n    name = 'test'\n    version = '1.0'\n")
            loader = PluginLoader(tmpdir)
            report = loader.discover()
            self.assertEqual(len(report.plugins), 1)
            self.assertTrue(report.plugins[0].loaded)

    def test_discover_invalid_plugin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "bad.py"), "w") as f:
                f.write("x = 1\n")
            loader = PluginLoader(tmpdir)
            report = loader.discover()
            self.assertFalse(report.plugins[0].loaded)

    def test_skips_underscore_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "__init__.py"), "w") as f:
                f.write("")
            self.assertEqual(len(PluginLoader(tmpdir).discover().plugins), 0)

    def test_get_and_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "alpha.py"), "w") as f:
                f.write("class ProviderPlugin:\n    pass\n")
            loader = PluginLoader(tmpdir)
            loader.discover()
            self.assertIsNotNone(loader.get_plugin("alpha"))
            self.assertEqual(loader.list_plugins(), ["alpha"])


if __name__ == "__main__":
    unittest.main()
