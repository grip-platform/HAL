"""HAL Code Search — AST-aware code search beyond grep.

Searches code by structure (functions, classes, imports) using Python's
ast module. Falls back to regex for non-Python files.

Usage:
    from lib.hal.code_search import CodeSearch

    search = CodeSearch()
    results = search.find_functions("auth", path="lib/hal/")
    results = search.find_classes("Engine")
"""
import ast
import os
import re
from dataclasses import dataclass

from lib.hal.home import HAL_HOME as _HAL_HOME

_SKIP_DIRS = frozenset({"node_modules", "venv", "__pycache__", "dist", "build", ".git", ".venv"})


@dataclass(frozen=True)
class CodeMatch:
    """A single code search result."""
    file_path: str
    name: str
    line: int
    match_type: str  # "function", "class", "import"
    snippet: str = ""


class CodeSearch:
    """AST-aware code search engine."""

    def __init__(self, root: str = ""):
        self._root = root or _HAL_HOME

    def find_functions(self, pattern: str, path: str = "") -> list[CodeMatch]:
        """Find function definitions matching a pattern."""
        search_path = os.path.join(self._root, path) if path else self._root
        return _search_ast(search_path, self._root, pattern, _match_functions)

    def find_classes(self, pattern: str, path: str = "") -> list[CodeMatch]:
        """Find class definitions matching a pattern."""
        search_path = os.path.join(self._root, path) if path else self._root
        return _search_ast(search_path, self._root, pattern, _match_classes)

    def find_imports(self, module: str, path: str = "") -> list[CodeMatch]:
        """Find import statements for a specific module."""
        search_path = os.path.join(self._root, path) if path else self._root
        return _search_ast(search_path, self._root, module, _match_imports)

    def summary(self, path: str = "") -> dict:
        """Get a summary of code structure."""
        search_path = os.path.join(self._root, path) if path else self._root
        files = functions = classes = total_lines = 0

        for py_file in _walk_python(search_path):
            files += 1
            tree = _parse_safe(py_file)
            if tree:
                counts = _count_defs(tree)
                functions += counts[0]
                classes += counts[1]
            total_lines += _count_lines(py_file)

        return {"files": files, "functions": functions, "classes": classes, "total_lines": total_lines}


def _search_ast(search_path: str, root: str, pattern: str, matcher: callable) -> list[CodeMatch]:
    """Generic AST search: walk files, apply matcher to each tree."""
    results: list[CodeMatch] = []
    for py_file in _walk_python(search_path):
        tree = _parse_safe(py_file)
        if tree:
            results.extend(matcher(tree, os.path.relpath(py_file, root), pattern))
    return results


def _match_functions(tree: ast.Module, rel_path: str, pattern: str) -> list[CodeMatch]:
    """Extract matching function defs from an AST."""
    return [
        CodeMatch(rel_path, node.name, node.lineno, "function", f"def {node.name}(...)")
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and re.search(pattern, node.name, re.IGNORECASE)
    ]


def _match_classes(tree: ast.Module, rel_path: str, pattern: str) -> list[CodeMatch]:
    """Extract matching class defs from an AST."""
    return [
        CodeMatch(rel_path, n.name, n.lineno, "class", _class_snippet(n))
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and re.search(pattern, n.name, re.IGNORECASE)
    ]


def _match_imports(tree: ast.Module, rel_path: str, module: str) -> list[CodeMatch]:
    """Extract matching import statements. Flat collection then filter."""
    # Collect all import info into flat tuples first — O(n) single pass
    raw: list[tuple[str, int, str]] = []  # (name, lineno, snippet)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            raw.extend((a.name, node.lineno, f"import {a.name}") for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = ", ".join(a.name for a in node.names[:5])
            raw.append((node.module, node.lineno, f"from {node.module} import {names}"))
    # Filter — O(m) where m = total import names
    return [
        CodeMatch(rel_path, name, lineno, "import", snippet)
        for name, lineno, snippet in raw if module in name
    ]


def _class_snippet(node: ast.ClassDef) -> str:
    """Build a class definition snippet."""
    bases = ", ".join(_base_name(b) for b in node.bases[:3])
    return f"class {node.name}({bases})" if bases else f"class {node.name}"


def _count_defs(tree: ast.Module) -> tuple[int, int]:
    """Count function and class definitions in one pass."""
    funcs = classes = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs += 1
        elif isinstance(node, ast.ClassDef):
            classes += 1
    return funcs, classes


def _count_lines(path: str) -> int:
    """Count lines in a file."""
    try:
        with open(path) as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def _walk_python(root: str) -> list[str]:
    """Walk directory tree collecting .py files."""
    if not os.path.isdir(root):
        return [root] if root.endswith(".py") and os.path.isfile(root) else []
    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        files.extend(os.path.join(dirpath, f) for f in filenames if f.endswith(".py"))
    return files


def _parse_safe(path: str) -> ast.Module | None:
    """Parse a Python file, returning None on failure."""
    try:
        with open(path, "r", errors="replace") as f:
            return ast.parse(f.read(), filename=path)
    except (SyntaxError, OSError):
        return None


def _base_name(node: ast.expr) -> str:
    """Extract base class name from AST node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return "?"
