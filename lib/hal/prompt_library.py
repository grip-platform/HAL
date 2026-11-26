"""HAL Prompt Library — reusable prompt templates for common tasks.

Ships with built-in templates and supports user-defined custom templates.
Templates use {variable} substitution.

Usage:
    from lib.hal.prompt_library import PromptLibrary

    library = PromptLibrary()
    prompt = library.render("code_review", code="def foo(): pass", language="Python")
"""
import json
import os
import re
from dataclasses import dataclass

from lib.hal.home import HAL_HOME as _HAL_HOME

_CUSTOM_TEMPLATES_PATH = os.path.join(_HAL_HOME, "config", "prompt-templates.json")
_VAR_RE = re.compile(r"\{(\w+)\}")


@dataclass(frozen=True)
class PromptTemplate:
    """A reusable prompt template."""
    name: str
    description: str
    template: str
    variables: tuple[str, ...]
    category: str = "general"


# Built-in templates that ship with HAL
_BUILTIN_TEMPLATES: dict[str, PromptTemplate] = {
    "code_review": PromptTemplate(
        name="code_review",
        description="Structured code review with actionable feedback",
        template=(
            "Review this {language} code. Focus on:\n"
            "1. Correctness: bugs, edge cases, error handling\n"
            "2. Performance: algorithmic complexity, resource usage\n"
            "3. Readability: naming, structure, documentation\n"
            "4. Security: injection, authentication, data validation\n\n"
            "Code:\n```{language}\n{code}\n```\n\n"
            "Provide specific, actionable feedback with line references."
        ),
        variables=("language", "code"),
        category="development",
    ),
    "explain_code": PromptTemplate(
        name="explain_code",
        description="Explain code at the specified level",
        template=(
            "Explain this {language} code at a {level} level.\n\n"
            "```{language}\n{code}\n```\n\n"
            "Cover: what it does, how it works, why key decisions were made."
        ),
        variables=("language", "code", "level"),
        category="development",
    ),
    "debug": PromptTemplate(
        name="debug",
        description="Structured debugging analysis",
        template=(
            "Debug this issue:\n\n"
            "**Error**: {error}\n"
            "**Code**: ```\n{code}\n```\n"
            "**Expected**: {expected}\n"
            "**Actual**: {actual}\n\n"
            "Analyse: root cause, fix, and prevention."
        ),
        variables=("error", "code", "expected", "actual"),
        category="development",
    ),
    "summarise": PromptTemplate(
        name="summarise",
        description="Summarise text to specified length",
        template=(
            "Summarise the following in {length}:\n\n{text}\n\n"
            "Focus on key points, decisions, and action items."
        ),
        variables=("text", "length"),
        category="writing",
    ),
    "compare": PromptTemplate(
        name="compare",
        description="Structured comparison of options",
        template=(
            "Compare {option_a} vs {option_b} for {context}.\n\n"
            "Structure as a table with: feature, pros, cons, recommendation.\n"
            "End with a clear recommendation and reasoning."
        ),
        variables=("option_a", "option_b", "context"),
        category="analysis",
    ),
    "rca": PromptTemplate(
        name="rca",
        description="Root cause analysis",
        template=(
            "Perform a root cause analysis:\n\n"
            "**Symptom**: {symptom}\n"
            "**Impact**: {impact}\n"
            "**Timeline**: {timeline}\n\n"
            "Use 5-Whys methodology. Output: root cause, contributing factors, "
            "immediate fix, long-term prevention."
        ),
        variables=("symptom", "impact", "timeline"),
        category="analysis",
    ),
    "generate_module": PromptTemplate(
        name="generate_module",
        description="Generate a HAL Python module from specification",
        template=(
            "Generate a Python module following HAL conventions.\n\n"
            "REFERENCE (follow this style):\n```python\n{reference}\n```\n\n"
            "MODULE: {module_name}\nDESCRIPTION: {description}\n"
            "PUBLIC API:\n{public_api}\nDEPENDENCIES: {depends_on}\n\n"
            "RULES: frozen dataclasses, dispatch tables (not if/else), "
            "functions under 30 lines, pure stdlib, module docstring with Usage.\n\n"
            "Output ONLY the Python source in a ```python``` block."
        ),
        variables=("reference", "module_name", "description", "public_api", "depends_on"),
        category="development",
    ),
    "generate_test": PromptTemplate(
        name="generate_test",
        description="Generate unittest tests for a HAL module",
        template=(
            "Generate unittest tests for this HAL module.\n\n"
            "MODULE SOURCE:\n```python\n{module_source}\n```\n\n"
            "TEST CASES:\n{test_cases}\n\n"
            "RULES: unittest.TestCase, import from lib.hal.{module_name}, "
            "tests MUST fail if key logic removed (no trivial always-pass), "
            "test edge cases and error paths, one assertion per test method.\n\n"
            "Output ONLY the Python test source in a ```python``` block."
        ),
        variables=("module_source", "test_cases", "module_name"),
        category="development",
    ),
}


class PromptLibrary:
    """Manages built-in and custom prompt templates."""

    def __init__(self):
        self._custom: dict[str, PromptTemplate] = {}
        self._load_custom()

    def render(self, template_name: str, **variables: str) -> str:
        """Render a template with the given variables."""
        template = self.get(template_name)
        if not template:
            raise KeyError(f"Template not found: {template_name}")

        result = template.template
        for var, value in variables.items():
            result = result.replace(f"{{{var}}}", value)

        missing = _VAR_RE.findall(result)
        if missing:
            raise ValueError(f"Missing variables: {', '.join(missing)}")
        return result

    def get(self, name: str) -> PromptTemplate | None:
        """Get a template by name. Custom templates override built-ins."""
        return self._custom.get(name) or _BUILTIN_TEMPLATES.get(name)

    def list_templates(self, category: str = "") -> list[PromptTemplate]:
        """List all available templates, optionally filtered by category."""
        all_templates = {**_BUILTIN_TEMPLATES, **self._custom}
        templates = list(all_templates.values())
        if category:
            templates = [t for t in templates if t.category == category]
        return sorted(templates, key=lambda t: t.name)

    def add_custom(self, name: str, description: str, template: str, category: str = "custom") -> None:
        """Add a custom template."""
        variables = tuple(_VAR_RE.findall(template))
        self._custom[name] = PromptTemplate(
            name=name, description=description,
            template=template, variables=variables, category=category,
        )
        self._save_custom()

    def _load_custom(self) -> None:
        """Load custom templates from disk."""
        if not os.path.exists(_CUSTOM_TEMPLATES_PATH):
            return
        try:
            with open(_CUSTOM_TEMPLATES_PATH) as f:
                data = json.load(f)
            for name, d in data.items():
                self._custom[name] = PromptTemplate(
                    name=name, description=d.get("description", ""),
                    template=d.get("template", ""),
                    variables=tuple(d.get("variables", [])),
                    category=d.get("category", "custom"),
                )
        except (json.JSONDecodeError, OSError):
            pass

    def _save_custom(self) -> None:
        """Persist custom templates to disk."""
        os.makedirs(os.path.dirname(_CUSTOM_TEMPLATES_PATH), exist_ok=True)
        data = {
            name: {
                "description": t.description, "template": t.template,
                "variables": list(t.variables), "category": t.category,
            }
            for name, t in self._custom.items()
        }
        with open(_CUSTOM_TEMPLATES_PATH, "w") as f:
            json.dump(data, f, indent=2)
