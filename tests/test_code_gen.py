"""Tests for HAL Code Generation Engine."""
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.hal.code_gen import (
    CodeGenerator,
    CodeSpec,
    GeneratedCode,
    ValidationIssue,
    ValidationReport,
    _check_ast,
    _check_security,
    _estimate_confidence,
    _extract_python,
    _parse_spec_response,
)


class TestCodeSpec(unittest.TestCase):
    """Test CodeSpec frozen dataclass."""

    def test_creation(self):
        spec = CodeSpec(
            module_name="test_mod",
            description="A test module",
            public_api=("class Foo", "def bar()"),
            depends_on=("types",),
            test_cases=("test_foo_creates",),
        )
        self.assertEqual(spec.module_name, "test_mod")
        self.assertEqual(len(spec.public_api), 2)

    def test_frozen(self):
        spec = CodeSpec(module_name="x", description="y", public_api=(), depends_on=(), test_cases=())
        with self.assertRaises(AttributeError):
            spec.module_name = "z"

    def test_defaults(self):
        spec = CodeSpec(module_name="x", description="y", public_api=(), depends_on=(), test_cases=())
        self.assertEqual(spec.source_proposal, "")


class TestGeneratedCode(unittest.TestCase):
    """Test GeneratedCode frozen dataclass."""

    def test_creation(self):
        spec = CodeSpec(module_name="x", description="y", public_api=(), depends_on=(), test_cases=())
        gen = GeneratedCode(
            module_path="/tmp/x.py", test_path="/tmp/test_x.py",
            module_source="pass", test_source="pass",
            spec=spec, model_used="groq/llama",
        )
        self.assertEqual(gen.model_used, "groq/llama")
        self.assertEqual(gen.confidence, 0.0)

    def test_frozen(self):
        spec = CodeSpec(module_name="x", description="y", public_api=(), depends_on=(), test_cases=())
        gen = GeneratedCode(
            module_path="/tmp/x.py", test_path="/tmp/test_x.py",
            module_source="pass", test_source="pass",
            spec=spec, model_used="test",
        )
        with self.assertRaises(AttributeError):
            gen.module_source = "new"


class TestValidationIssue(unittest.TestCase):
    """Test ValidationIssue dataclass."""

    def test_creation(self):
        issue = ValidationIssue(gate="ast", severity="critical", message="bad syntax")
        self.assertEqual(issue.gate, "ast")
        self.assertEqual(issue.severity, "critical")


class TestExtractPython(unittest.TestCase):
    """Test Python extraction from LLM responses."""

    def test_fenced_python_block(self):
        text = 'Here is the code:\n```python\ndef hello():\n    return "hi"\n```\nDone.'
        result = _extract_python(text)
        self.assertIn("def hello():", result)
        self.assertIn('return "hi"', result)

    def test_fenced_generic_block(self):
        text = '```\nclass Foo:\n    pass\n```'
        result = _extract_python(text)
        self.assertIn("class Foo:", result)

    def test_raw_python(self):
        text = 'import os\n\ndef main():\n    pass'
        result = _extract_python(text)
        self.assertIn("import os", result)

    def test_no_code(self):
        text = "This is just a plain text response with no code."
        result = _extract_python(text)
        self.assertEqual(result, "")

    def test_empty_input(self):
        self.assertEqual(_extract_python(""), "")


class TestCheckAst(unittest.TestCase):
    """Test AST validation gate."""

    def test_valid_python(self):
        issues = []
        result = _check_ast("def foo():\n    return 42\n", issues)
        self.assertTrue(result)
        self.assertEqual(len(issues), 0)

    def test_invalid_syntax(self):
        issues = []
        result = _check_ast("def foo(:\n    pass\n", issues)
        self.assertFalse(result)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].gate, "ast")
        self.assertEqual(issues[0].severity, "critical")

    def test_empty_source(self):
        issues = []
        result = _check_ast("", issues)
        self.assertFalse(result)
        self.assertEqual(issues[0].message, "Empty source")

    def test_whitespace_only(self):
        issues = []
        result = _check_ast("   \n  \n  ", issues)
        self.assertFalse(result)


class TestCheckSecurity(unittest.TestCase):
    """Test security scan gate."""

    def test_clean_code(self):
        issues = []
        result = _check_security("def foo():\n    return 42\n", issues)
        self.assertTrue(result)
        self.assertEqual(len(issues), 0)

    def test_exec_detected(self):
        issues = []
        result = _check_security("exec(user_input)", issues)
        self.assertFalse(result)
        self.assertTrue(any("exec()" in i.message for i in issues))

    def test_eval_detected(self):
        issues = []
        result = _check_security("result = eval(data)", issues)
        self.assertFalse(result)
        self.assertTrue(any("eval()" in i.message for i in issues))

    def test_pickle_detected(self):
        issues = []
        result = _check_security("obj = pickle.loads(data)", issues)
        self.assertFalse(result)

    def test_os_system_detected(self):
        issues = []
        result = _check_security("os.system(cmd)", issues)
        self.assertFalse(result)

    def test_dunder_import_detected(self):
        issues = []
        result = _check_security("mod = __import__(name)", issues)
        self.assertFalse(result)

    def test_multiple_issues(self):
        issues = []
        _check_security("exec(x)\neval(y)\nos.system(z)", issues)
        self.assertGreaterEqual(len(issues), 3)


class TestEstimateConfidence(unittest.TestCase):
    """Test confidence estimation."""

    def test_empty_source(self):
        spec = CodeSpec(module_name="x", description="y", public_api=(), depends_on=(), test_cases=())
        self.assertEqual(_estimate_confidence("", spec), 0.0)

    def test_invalid_syntax(self):
        spec = CodeSpec(module_name="x", description="y", public_api=(), depends_on=(), test_cases=())
        self.assertEqual(_estimate_confidence("def (:broken", spec), 0.1)

    def test_matching_api(self):
        spec = CodeSpec(
            module_name="x", description="y",
            public_api=("class MyClass", "def my_func()"),
            depends_on=(), test_cases=(),
        )
        source = "class MyClass:\n    pass\n\ndef my_func():\n    pass\n"
        confidence = _estimate_confidence(source, spec)
        self.assertGreater(confidence, 0.5)

    def test_no_matching_api(self):
        spec = CodeSpec(
            module_name="x", description="y",
            public_api=("class Expected",),
            depends_on=(), test_cases=(),
        )
        source = "class Unexpected:\n    pass\n"
        confidence = _estimate_confidence(source, spec)
        self.assertLess(confidence, 0.5)

    def test_confidence_capped_at_one(self):
        spec = CodeSpec(
            module_name="x", description="y",
            public_api=("class A", "class B", "class C", "class D", "class E", "class F"),
            depends_on=(), test_cases=(),
        )
        source = "class A: pass\nclass B: pass\nclass C: pass\nclass D: pass\nclass E: pass\nclass F: pass\n"
        confidence = _estimate_confidence(source, spec)
        self.assertLessEqual(confidence, 1.0)


class TestParseSpecResponse(unittest.TestCase):
    """Test LLM spec response parsing."""

    def test_valid_json(self):
        text = '{"module_name": "foo", "description": "A foo", "public_api": ["class Foo"], "depends_on": ["types"], "test_cases": ["test_create"]}'
        spec = _parse_spec_response(text, "fallback")
        self.assertEqual(spec.module_name, "foo")
        self.assertEqual(spec.description, "A foo")
        self.assertEqual(spec.public_api, ("class Foo",))
        self.assertEqual(spec.depends_on, ("types",))

    def test_json_with_surrounding_text(self):
        text = 'Here is the spec:\n{"module_name": "bar", "description": "A bar", "public_api": [], "depends_on": [], "test_cases": []}\nDone!'
        spec = _parse_spec_response(text, "fallback")
        self.assertEqual(spec.module_name, "bar")

    def test_malformed_json_fallback(self):
        text = "This is not JSON at all."
        spec = _parse_spec_response(text, "Create a skill codifying: retry logic")
        self.assertIn("retry", spec.module_name)
        self.assertEqual(spec.source_proposal, "Create a skill codifying: retry logic")

    def test_empty_text_fallback(self):
        spec = _parse_spec_response("", "some action")
        self.assertEqual(spec.description, "some action")

    def test_partial_json(self):
        text = '{"module_name": "partial"'
        spec = _parse_spec_response(text, "fallback action")
        self.assertEqual(spec.description, "fallback action")


class TestCodeGenerator(unittest.TestCase):
    """Test CodeGenerator with mocked registry."""

    def setUp(self):
        self.registry = MagicMock()
        provider = MagicMock()
        provider.name = "groq"
        provider.healthy = True
        self.registry.list_providers.return_value = [provider]
        self.gen = CodeGenerator(self.registry)

    def test_spec_from_proposal(self):
        proposal = MagicMock()
        proposal.action = "Create a skill codifying: retry with backoff"
        proposal.description = "Pattern seen 5x: retry with backoff"

        self.registry.chat.return_value = MagicMock(
            text='{"module_name": "retry_backoff", "description": "Retry with exponential backoff", "public_api": ["class RetryBackoff"], "depends_on": ["types"], "test_cases": ["test_retry_succeeds"]}'
        )

        spec = self.gen.spec_from_proposal(proposal)
        self.assertEqual(spec.module_name, "retry_backoff")
        self.registry.chat.assert_called_once()

    def test_generate(self):
        spec = CodeSpec(
            module_name="test_gen",
            description="Test generation",
            public_api=("class TestGen",),
            depends_on=(),
            test_cases=("test_create",),
        )
        self.registry.chat.side_effect = [
            MagicMock(text='```python\nclass TestGen:\n    """Test."""\n    pass\n```'),
            MagicMock(text='```python\nimport unittest\nclass TestTestGen(unittest.TestCase):\n    def test_create(self):\n        self.assertTrue(True)\n```'),
        ]

        result = self.gen.generate(spec)
        self.assertIn("class TestGen:", result.module_source)
        self.assertIn("unittest", result.test_source)
        self.assertEqual(result.spec, spec)
        self.assertEqual(self.registry.chat.call_count, 2)

    def test_validate_valid_code(self):
        spec = CodeSpec(module_name="x", description="y", public_api=(), depends_on=(), test_cases=())
        generated = GeneratedCode(
            module_path="/tmp/x.py", test_path="/tmp/test_x.py",
            module_source="class Foo:\n    pass\n",
            test_source="import unittest\nclass T(unittest.TestCase):\n    def test_a(self): pass\n",
            spec=spec, model_used="test",
        )
        report = self.gen.validate(generated)
        self.assertTrue(report.ast_valid)
        self.assertTrue(report.security_clean)

    def test_validate_invalid_syntax(self):
        spec = CodeSpec(module_name="x", description="y", public_api=(), depends_on=(), test_cases=())
        generated = GeneratedCode(
            module_path="/tmp/x.py", test_path="/tmp/test_x.py",
            module_source="def broken(:\n    pass\n",
            test_source="pass",
            spec=spec, model_used="test",
        )
        report = self.gen.validate(generated)
        self.assertFalse(report.passed)
        self.assertFalse(report.ast_valid)

    def test_validate_security_issue(self):
        spec = CodeSpec(module_name="x", description="y", public_api=(), depends_on=(), test_cases=())
        generated = GeneratedCode(
            module_path="/tmp/x.py", test_path="/tmp/test_x.py",
            module_source="exec(user_input)\n",
            test_source="pass\n",
            spec=spec, model_used="test",
        )
        report = self.gen.validate(generated)
        self.assertFalse(report.passed)
        self.assertFalse(report.security_clean)

    def test_refine(self):
        spec = CodeSpec(module_name="x", description="y", public_api=(), depends_on=(), test_cases=())
        generated = GeneratedCode(
            module_path="/tmp/x.py", test_path="/tmp/test_x.py",
            module_source="def broken(:\n    pass",
            test_source="pass",
            spec=spec, model_used="groq/llama",
        )
        self.registry.chat.return_value = MagicMock(
            text='```python\ndef fixed():\n    pass\n```'
        )

        fixed = self.gen.refine(generated, "Syntax error line 1")
        self.assertIn("def fixed():", fixed.module_source)

    def test_pick_model_healthy(self):
        model = self.gen._pick_model()
        self.assertEqual(model, "groq/llama-3.3-70b-versatile")

    def test_pick_model_no_providers(self):
        self.registry.list_providers.return_value = []
        model = self.gen._pick_model()
        self.assertEqual(model, "")


class TestValidationReport(unittest.TestCase):
    """Test ValidationReport dataclass."""

    def test_passed_report(self):
        report = ValidationReport(
            passed=True, issues=(), ast_valid=True,
            security_clean=True, compiles=True, tests_pass=True,
        )
        self.assertTrue(report.passed)
        self.assertEqual(len(report.issues), 0)

    def test_failed_report_with_issues(self):
        issues = (
            ValidationIssue("ast", "critical", "bad"),
            ValidationIssue("security", "high", "exec found"),
        )
        report = ValidationReport(
            passed=False, issues=issues, ast_valid=False,
            security_clean=False, compiles=False,
        )
        self.assertFalse(report.passed)
        self.assertEqual(len(report.issues), 2)


if __name__ == "__main__":
    unittest.main()
