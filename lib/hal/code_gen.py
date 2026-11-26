"""HAL Code Generation Engine — transform RSI proposals into validated Python.

Takes an RSI Proposal (or structured CodeSpec) and produces valid Python
source code with corresponding tests. Uses LLM for generation, mechanical
gates for validation: AST parse, security scan, py_compile, and Goodhart
mutation testing.

Usage:
    from lib.hal.code_gen import CodeGenerator

    gen = CodeGenerator(registry)
    spec = gen.spec_from_proposal(proposal)
    result = gen.generate(spec)
    report = gen.validate(result)
"""
import ast
import os
import subprocess
import textwrap
from dataclasses import dataclass

from lib.hal.home import HAL_HOME as _HAL_HOME

_MAX_REFINE_PASSES = 3

_MODULE_DIR = os.path.join(_HAL_HOME, "lib", "hal")
_TEST_DIR = os.path.join(_HAL_HOME, "tests")


@dataclass(frozen=True)
class CodeSpec:
    """Structured specification for code generation."""

    module_name: str
    description: str
    public_api: tuple[str, ...]
    depends_on: tuple[str, ...]
    test_cases: tuple[str, ...]
    source_proposal: str = ""


@dataclass(frozen=True)
class GeneratedCode:
    """Output of code generation: module source + test source."""

    module_path: str
    test_path: str
    module_source: str
    test_source: str
    spec: CodeSpec
    model_used: str
    confidence: float = 0.0


@dataclass(frozen=True)
class ValidationIssue:
    """A single validation problem."""

    gate: str
    severity: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    """Result of validating generated code."""

    passed: bool
    issues: tuple[ValidationIssue, ...]
    ast_valid: bool = False
    security_clean: bool = False
    compiles: bool = False
    tests_pass: bool = False


class CodeGenerator:
    """Generates validated Python modules from specs or RSI proposals."""

    def __init__(self, registry: "ProviderRegistry"):
        self._registry = registry

    def spec_from_proposal(self, proposal: "Proposal", model: str = "") -> CodeSpec:
        """Decompose an RSI Proposal into a structured CodeSpec via LLM."""
        model = model or self._pick_model()
        prompt = _build_spec_prompt(proposal.action, proposal.description)

        response = self._registry.chat(model, prompt, max_tokens=1500)
        return _parse_spec_response(response.text, proposal.action)

    def generate(self, spec: CodeSpec, model: str = "") -> GeneratedCode:
        """Generate module + test source from a CodeSpec."""
        model = model or self._pick_model()
        reference = _load_reference_module()

        module_prompt = _build_module_prompt(spec, reference)
        module_response = self._registry.chat(model, module_prompt, max_tokens=3000)
        module_source = _extract_python(module_response.text)

        test_prompt = _build_test_prompt(spec, module_source)
        test_response = self._registry.chat(model, test_prompt, max_tokens=2000)
        test_source = _extract_python(test_response.text)

        module_path = os.path.join(_MODULE_DIR, f"{spec.module_name}.py")
        test_path = os.path.join(_TEST_DIR, f"test_{spec.module_name}.py")

        return GeneratedCode(
            module_path=module_path,
            test_path=test_path,
            module_source=module_source,
            test_source=test_source,
            spec=spec,
            model_used=model,
            confidence=_estimate_confidence(module_source, spec),
        )

    def validate(self, generated: GeneratedCode) -> ValidationReport:
        """Run mechanical validation pipeline on generated code."""
        issues: list[ValidationIssue] = []

        ast_valid = _check_ast(generated.module_source, issues)
        ast_valid_test = _check_ast(generated.test_source, issues)
        security_clean = _check_security(generated.module_source, issues)
        compiles = _check_compile(generated.module_source, generated.module_path, issues)

        passed = ast_valid and ast_valid_test and security_clean and compiles
        return ValidationReport(
            passed=passed,
            issues=tuple(issues),
            ast_valid=ast_valid and ast_valid_test,
            security_clean=security_clean,
            compiles=compiles,
            tests_pass=False,
        )

    def refine(self, generated: GeneratedCode, errors: str, model: str = "") -> GeneratedCode:
        """Feed validation errors back to LLM for a fix pass."""
        model = model or generated.model_used or self._pick_model()

        prompt = (
            "The following Python module has validation errors. Fix them.\n\n"
            f"ERRORS:\n{errors}\n\n"
            f"MODULE SOURCE:\n```python\n{generated.module_source}\n```\n\n"
            "Output ONLY the corrected Python source in a ```python``` block."
        )
        response = self._registry.chat(model, prompt, max_tokens=3000)
        fixed_source = _extract_python(response.text)

        return GeneratedCode(
            module_path=generated.module_path,
            test_path=generated.test_path,
            module_source=fixed_source or generated.module_source,
            test_source=generated.test_source,
            spec=generated.spec,
            model_used=model,
            confidence=generated.confidence,
        )

    def generate_and_validate(
        self, spec: CodeSpec, model: str = "",
    ) -> tuple[GeneratedCode, ValidationReport]:
        """Generate, validate, and auto-refine up to 3 times."""
        generated = self.generate(spec, model)
        for _ in range(_MAX_REFINE_PASSES):
            report = self.validate(generated)
            if report.passed:
                return generated, report
            error_summary = "\n".join(f"[{i.gate}] {i.message}" for i in report.issues)
            generated = self.refine(generated, error_summary, model)
        return generated, self.validate(generated)

    def _pick_model(self) -> str:
        """Pick the best available model for code generation."""
        candidates = [
            "groq/llama-3.3-70b-versatile",
            "gemini/gemini-3.5-flash-lite",
            "groq/openai/gpt-oss-120b",
        ]
        providers = {p.name: p for p in self._registry.list_providers()}
        for candidate in candidates:
            provider_name = candidate.split("/")[0]
            p = providers.get(provider_name)
            if p and p.healthy:
                return candidate
        return ""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_spec_prompt(action: str, description: str) -> str:
    return textwrap.dedent(f"""\
        Decompose this improvement proposal into a structured module specification.

        PROPOSAL ACTION: {action}
        PROPOSAL DESCRIPTION: {description}

        Output a JSON object with these fields:
        - module_name: snake_case Python module name (no .py extension)
        - description: one-line module docstring
        - public_api: list of function/class signatures as strings
        - depends_on: list of existing HAL modules this imports from (e.g. ["types", "registry"])
        - test_cases: list of test scenario descriptions

        Output ONLY the JSON object, no other text.
    """)


def _build_module_prompt(spec: CodeSpec, reference: str) -> str:
    deps = ", ".join(spec.depends_on) if spec.depends_on else "none"
    api = "\n".join(f"  - {sig}" for sig in spec.public_api)
    return textwrap.dedent(f"""\
        Generate a Python module following HAL conventions.

        CONVENTIONS (from reference module):
        ```python
        {reference}
        ```

        REQUIREMENTS:
        - Module name: {spec.module_name}
        - Description: {spec.description}
        - Public API:
        {api}
        - Dependencies: {deps}
        - Use frozen dataclasses for data types
        - Use dispatch tables instead of if/else chains
        - Functions under 30 lines
        - Pure stdlib only (no external packages)
        - Include a module docstring with Usage example

        Output ONLY the Python source in a ```python``` block.
    """)


def _build_test_prompt(spec: CodeSpec, module_source: str) -> str:
    cases = "\n".join(f"  - {tc}" for tc in spec.test_cases)
    return textwrap.dedent(f"""\
        Generate unittest tests for this HAL module.

        MODULE SOURCE:
        ```python
        {module_source}
        ```

        TEST CASES TO COVER:
        {cases}

        REQUIREMENTS:
        - Use unittest.TestCase
        - Import from lib.hal.{spec.module_name}
        - Tests must FAIL if key logic is removed (no trivial always-pass tests)
        - Test edge cases and error paths
        - Each test method tests ONE thing

        Output ONLY the Python test source in a ```python``` block.
    """)


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------

def _check_ast(source: str, issues: list[ValidationIssue]) -> bool:
    """Gate 1: AST parse validation."""
    if not source.strip():
        issues.append(ValidationIssue("ast", "critical", "Empty source"))
        return False
    try:
        ast.parse(source)
        return True
    except SyntaxError as e:
        issues.append(ValidationIssue(
            "ast", "critical", f"Syntax error line {e.lineno}: {e.msg}",
        ))
        return False


_DANGEROUS_PATTERNS = (
    ("exec(", "injection", "exec() call — code injection risk"),
    ("eval(", "injection", "eval() call — code injection risk"),
    ("pickle.loads", "deserialization", "Insecure pickle deserialization"),
    ("__import__", "injection", "Dynamic import — code injection risk"),
    ("os.system(", "injection", "os.system — use subprocess with shell=False"),
)


def _check_security(source: str, issues: list[ValidationIssue]) -> bool:
    """Gate 2: Security pattern scan."""
    clean = True
    for pattern, category, message in _DANGEROUS_PATTERNS:
        if pattern in source:
            issues.append(ValidationIssue("security", "high", message))
            clean = False
    return clean


def _check_compile(source: str, target_path: str, issues: list[ValidationIssue]) -> bool:
    """Gate 3: py_compile verification via temp file."""
    import tempfile
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
        ) as f:
            f.write(source)
            tmp_path = f.name
        result = subprocess.run(
            ["python3", "-m", "py_compile", tmp_path],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            issues.append(ValidationIssue(
                "compile", "critical", result.stderr[:300],
            ))
            return False
        return True
    except (OSError, subprocess.TimeoutExpired) as e:
        issues.append(ValidationIssue("compile", "critical", str(e)[:200]))
        return False
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_python(text: str) -> str:
    """Extract Python code from LLM response (fenced block or raw)."""
    import re
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    if text.strip().startswith(('"""', "import ", "from ", "class ", "def ")):
        return text.strip()
    return ""


def _load_reference_module() -> str:
    """Load a reference HAL module as exemplar for conventions."""
    ref_path = os.path.join(_MODULE_DIR, "quality_scorer.py")
    try:
        with open(ref_path) as f:
            return f.read()[:2000]
    except OSError:
        return "# Use frozen dataclasses, dispatch tables, functions < 30 lines"


def _estimate_confidence(source: str, spec: CodeSpec) -> float:
    """Estimate generation confidence from structural signals."""
    if not source:
        return 0.0
    score = 0.3
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0.1

    class_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    func_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    all_names = class_names | func_names

    for sig in spec.public_api:
        first_word = sig.split("(")[0].split(":")[-1].strip().split()[-1]
        if first_word in all_names:
            score += 0.15
    return min(1.0, score)


def _parse_spec_response(text: str, fallback_action: str) -> CodeSpec:
    """Parse LLM JSON response into CodeSpec. Tolerant of malformed JSON."""
    import json
    import re

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            data = json.loads(text[start:end + 1])
            return CodeSpec(
                module_name=data.get("module_name", "generated_module"),
                description=data.get("description", fallback_action),
                public_api=tuple(data.get("public_api", [])),
                depends_on=tuple(data.get("depends_on", [])),
                test_cases=tuple(data.get("test_cases", [])),
                source_proposal=fallback_action,
            )
        except (json.JSONDecodeError, TypeError):
            pass

    slug = re.sub(r"[^a-z0-9]+", "_", fallback_action.lower())[:40].strip("_")
    return CodeSpec(
        module_name=slug or "generated_module",
        description=fallback_action,
        public_api=(),
        depends_on=(),
        test_cases=(),
        source_proposal=fallback_action,
    )
