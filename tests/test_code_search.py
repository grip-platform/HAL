"""Tests for HAL AST-aware code search."""
import os
import sys
import unittest

from lib.hal.code_search import CodeSearch, _parse_safe, _walk_python


class TestFindFunctions(unittest.TestCase):

    def test_find_in_hal(self):
        search = CodeSearch()
        results = search.find_functions("warmup", path="lib/hal/")
        self.assertGreater(len(results), 0)
        self.assertTrue(any("warmup" in r.name.lower() for r in results))

    def test_no_match(self):
        search = CodeSearch()
        self.assertEqual(search.find_functions("zzz_nonexistent_zzz"), [])


class TestFindClasses(unittest.TestCase):

    def test_find_classes(self):
        search = CodeSearch()
        results = search.find_classes("Engine", path="lib/hal/")
        self.assertGreater(len(results), 0)

    def test_snippet_has_class(self):
        search = CodeSearch()
        results = search.find_classes("Conversation", path="lib/hal/")
        self.assertTrue(any("class" in r.snippet for r in results))


class TestFindImports(unittest.TestCase):

    def test_find_json_imports(self):
        search = CodeSearch()
        results = search.find_imports("json", path="lib/hal/")
        self.assertGreater(len(results), 0)

    def test_find_from_imports(self):
        search = CodeSearch()
        results = search.find_imports("hal.home", path="lib/hal/")
        self.assertGreater(len(results), 0)
        self.assertTrue(any("from" in r.snippet for r in results))


class TestSummary(unittest.TestCase):

    def test_summary_counts(self):
        search = CodeSearch()
        s = search.summary(path="lib/hal/")
        self.assertGreater(s["files"], 10)
        self.assertGreater(s["functions"], 10)
        self.assertGreater(s["classes"], 5)
        self.assertGreater(s["total_lines"], 500)


class TestWalkPython(unittest.TestCase):

    def test_finds_modules(self):
        hal_dir = os.path.join(os.path.expanduser("~/.hal"), "lib", "hal")
        self.assertGreater(len(_walk_python(hal_dir)), 10)

    def test_single_file(self):
        path = os.path.join(os.path.expanduser("~/.hal"), "lib", "hal", "home.py")
        self.assertEqual(len(_walk_python(path)), 1)


class TestParseSafe(unittest.TestCase):

    def test_valid(self):
        path = os.path.join(os.path.expanduser("~/.hal"), "lib", "hal", "home.py")
        self.assertIsNotNone(_parse_safe(path))

    def test_nonexistent(self):
        self.assertIsNone(_parse_safe("/nonexistent/file.py"))


if __name__ == "__main__":
    unittest.main()
