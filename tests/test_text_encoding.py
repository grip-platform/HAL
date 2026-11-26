"""Keep HAL file I/O independent of the Windows locale's default encoding."""
import ast
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"node_modules", "venv", "dist", "build", "__pycache__", "target", "vendor"}


def python_files():
    for folder in ("lib", "bin"):
        for directory, dirs, files in os.walk(REPO / folder):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            for name in sorted(files):
                path = Path(directory) / name
                if path.suffix == ".py":
                    yield path
                elif not path.suffix:
                    # Executable Python entrypoints in bin/ have no .py suffix.
                    with path.open(encoding="utf-8") as handle:
                        if "python" in handle.readline():
                            yield path


def missing_encoding(source):
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        if name not in {"open", "fdopen", "read_text", "write_text"}:
            continue
        # These APIs open descriptors, media or HTTP responses, not text files.
        if name == "open" and isinstance(func, ast.Attribute) and ast.unparse(func.value) in {
            "os", "av", "wave", "opener", "self.opener",
        }:
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords}
        # Builtin/io.open and os.fdopen: encoding is argument 4; Path.open: argument 3.
        builtin_open = name == "fdopen" or isinstance(func, ast.Name) or (
            isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
            and func.value.id in {"io", "builtins"}
        )
        encoding_index = (3 if builtin_open else 2) if name in {"open", "fdopen"} else (1 if name == "write_text" else 0)
        if "encoding" in keywords or len(node.args) > encoding_index:
            continue
        if name in {"open", "fdopen"}:
            mode_index = 1 if builtin_open else 0
            mode = keywords.get("mode", node.args[mode_index] if len(node.args) > mode_index else ast.Constant("r"))
            if isinstance(mode, ast.Constant) and isinstance(mode.value, str) and "b" in mode.value:
                continue
        yield node


def test_hal_text_files_have_explicit_encoding():
    violations = [f"{path.relative_to(REPO).as_posix()}:{node.lineno}: missing encoding"
                  for path in python_files()
                  for node in missing_encoding(path.read_text(encoding="utf-8"))]
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("source", [
    'open("state")', 'open("state", "w", errors="replace")',
    'path.open()', 'path.read_text()', 'path.write_text("snowman")',
    'io.open("state")', 'os.fdopen(fd, "w")', 'os.fdopen(fd)',
    'os.fdopen(fd, mode="w")', 'os.fdopen(fd, "w", errors="replace")',
])
def test_guard_rejects_implicit_text_encoding(source):
    assert len(list(missing_encoding(source))) == 1


@pytest.mark.parametrize("source", [
    'os.open("state", os.O_RDONLY)', 'wave.open(buf, "rb")',
    'av.open(video)', 'opener.open(request)', 'self.opener.open(request)',
    'open("state", "rb")', 'path.open("wb")',
    'open("state", encoding="utf-8", errors="replace")',
    'path.read_text(encoding="utf-8")', 'path.write_text("x", encoding="utf-8")',
    'open("state", "r", -1, "utf-8")', 'path.open("r", -1, "utf-8")',
    'os.fdopen(fd, "wb")', 'os.fdopen(fd, "w", encoding="utf-8")',
])
def test_guard_accepts_binary_or_explicit_encoding(source):
    assert not list(missing_encoding(source))
