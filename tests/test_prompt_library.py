"""Tests for HAL prompt library."""
import os
import sys
import unittest

from lib.hal.prompt_library import PromptLibrary, PromptTemplate, _BUILTIN_TEMPLATES


class TestPromptTemplate(unittest.TestCase):

    def test_frozen(self):
        t = PromptTemplate("test", "desc", "template", ("var",))
        with self.assertRaises(AttributeError):
            t.name = "changed"

    def test_fields(self):
        t = PromptTemplate("test", "desc", "{x} + {y}", ("x", "y"), "math")
        self.assertEqual(t.name, "test")
        self.assertEqual(t.variables, ("x", "y"))
        self.assertEqual(t.category, "math")


class TestBuiltinTemplates(unittest.TestCase):

    def test_code_review_exists(self):
        self.assertIn("code_review", _BUILTIN_TEMPLATES)

    def test_all_templates_have_variables(self):
        for name, template in _BUILTIN_TEMPLATES.items():
            self.assertIsInstance(template.variables, tuple, f"{name} variables not a tuple")

    def test_builtin_count(self):
        self.assertGreaterEqual(len(_BUILTIN_TEMPLATES), 5)


class TestPromptLibrary(unittest.TestCase):

    def test_render_code_review(self):
        lib = PromptLibrary()
        result = lib.render("code_review", language="Python", code="x = 1")
        self.assertIn("Python", result)
        self.assertIn("x = 1", result)

    def test_render_missing_var_raises(self):
        lib = PromptLibrary()
        with self.assertRaises(ValueError):
            lib.render("code_review", language="Python")  # missing 'code'

    def test_render_unknown_template_raises(self):
        lib = PromptLibrary()
        with self.assertRaises(KeyError):
            lib.render("nonexistent_template")

    def test_list_templates(self):
        lib = PromptLibrary()
        templates = lib.list_templates()
        self.assertGreater(len(templates), 0)

    def test_list_by_category(self):
        lib = PromptLibrary()
        dev = lib.list_templates(category="development")
        self.assertGreater(len(dev), 0)
        for t in dev:
            self.assertEqual(t.category, "development")

    def test_add_custom(self):
        lib = PromptLibrary()
        lib.add_custom("greet", "Greeting", "Hello {name}!", "custom")
        result = lib.render("greet", name="World")
        self.assertEqual(result, "Hello World!")

    def test_custom_overrides_builtin(self):
        lib = PromptLibrary()
        lib.add_custom("code_review", "Custom review", "Custom: {language} {code}", "dev")
        result = lib.render("code_review", language="Rust", code="fn main() {}")
        self.assertIn("Custom:", result)

    def test_get_returns_none_for_unknown(self):
        lib = PromptLibrary()
        self.assertIsNone(lib.get("nonexistent"))


if __name__ == "__main__":
    unittest.main()
