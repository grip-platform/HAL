"""HAL Output Validator — validate LLM responses against expected schemas.

Checks that LLM output matches expected format (JSON, code, markdown,
list, etc.). Used by workflows and automated pipelines to ensure
responses are machine-parseable.

Usage:
    from lib.hal.output_validator import OutputValidator

    validator = OutputValidator()
    result = validator.validate_json(response_text)
    result = validator.validate_code(response_text, language="python")
"""
import ast
import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationResult:
    """Result of output validation."""
    valid: bool
    format_type: str
    errors: tuple[str, ...] = ()
    extracted: str = ""


class OutputValidator:
    """Validates LLM output format."""

    def validate_json(self, text: str) -> ValidationResult:
        """Check if response contains valid JSON."""
        start = text.find("{")
        end = text.rfind("}")
        if start == -1:
            start = text.find("[")
            end = text.rfind("]")
        if start == -1 or end == -1:
            return ValidationResult(False, "json", ("No JSON found",))
        candidate = text[start:end + 1]
        try:
            json.loads(candidate)
            return ValidationResult(True, "json", extracted=candidate)
        except json.JSONDecodeError as e:
            return ValidationResult(False, "json", (str(e)[:100],))

    def validate_code(self, text: str, language: str = "python") -> ValidationResult:
        """Check if response contains valid code."""
        code = _extract_code_block(text)
        if not code:
            return ValidationResult(False, "code", ("No code block found",))
        if language == "python":
            return _validate_python(code)
        return ValidationResult(True, "code", extracted=code)

    def validate_list(self, text: str) -> ValidationResult:
        """Check if response contains a list."""
        items = re.findall(r"^\s*[-*\d]+[.)\s]+(.+)$", text, re.MULTILINE)
        if not items:
            return ValidationResult(False, "list", ("No list items found",))
        return ValidationResult(True, "list", extracted="\n".join(items))

    def validate_markdown(self, text: str) -> ValidationResult:
        """Check if response uses markdown formatting."""
        has_headers = bool(re.search(r"^#+\s", text, re.MULTILINE))
        has_code = "```" in text
        has_formatting = "**" in text or "*" in text
        if has_headers or has_code or has_formatting:
            return ValidationResult(True, "markdown")
        return ValidationResult(False, "markdown", ("No markdown formatting detected",))


def _extract_code_block(text: str) -> str:
    """Extract first fenced code block."""
    match = re.search(r"```\w*\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _validate_python(code: str) -> ValidationResult:
    """Validate Python syntax."""
    try:
        ast.parse(code)
        return ValidationResult(True, "code", extracted=code)
    except SyntaxError as e:
        return ValidationResult(False, "code", (f"Syntax error: {e.msg} (line {e.lineno})",))
