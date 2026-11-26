"""HAL#925: kono's store_session_memory reaches GRIP's Ollama-backed embeddings.

HAL's services start with only HAL's own folders on sys.path, so the import of
GRIP's ``embeddings`` package always failed and the method stored nothing.
These tests use a fake GRIP install, so nothing reaches Ollama or GRIP's own
database.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from lib.hal.kono import grip_embeddings
from lib.hal.kono.counters import SessionCounterStore
from lib.hal.kono.substrate import MemorySubstrate


def _fake_grip(root: Path) -> Path:
    """A GRIP install whose embeddings package records what it is asked to store."""
    pkg = root / "lib" / "embeddings"
    pkg.mkdir(parents=True)
    (pkg / "ollama_backend.py").write_text("")
    (pkg / "__init__.py").write_text(textwrap.dedent('''
        STORED = []


        class EmbeddingEngine:
            pass
    '''))
    (pkg / "storage.py").write_text(textwrap.dedent('''
        from embeddings import STORED


        def store_embedding(engine, text, embed_type, project_path):
            STORED.append(text)
            return 42
    '''))
    return root


def _embeddings_modules() -> list:
    return [m for m in sys.modules if m == "embeddings" or m.startswith("embeddings.")]


@pytest.fixture
def isolated_imports(monkeypatch):
    """A private copy of sys.path, and no embeddings package loaded, for each test."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    saved = {name: sys.modules.pop(name) for name in _embeddings_modules()}
    yield
    for name in _embeddings_modules():
        del sys.modules[name]
    sys.modules.update(saved)


def test_without_grip_nothing_changes(tmp_path, monkeypatch, isolated_imports):
    monkeypatch.setenv("CLAUDE_DIR", str(tmp_path))
    before = list(sys.path)

    assert grip_embeddings.ensure_importable() is False
    assert sys.path == before


def test_with_grip_its_lib_is_appended_once(tmp_path, monkeypatch, isolated_imports):
    root = _fake_grip(tmp_path)
    monkeypatch.setenv("CLAUDE_DIR", str(root))
    lib = str((root / "lib").resolve())

    assert grip_embeddings.ensure_importable() is True
    assert grip_embeddings.ensure_importable() is True
    assert sys.path.count(lib) == 1
    assert sys.path[-1] == lib, "appended last, so a HAL module of the same name still wins"


def test_store_session_memory_stores_through_grips_package(tmp_path, monkeypatch, isolated_imports):
    """Mutation: drop the _ensure_grip_embeddings() call -> the import fails -> None -> RED."""
    root = _fake_grip(tmp_path)
    monkeypatch.setenv("CLAUDE_DIR", str(root))
    substrate = MemorySubstrate.__new__(MemorySubstrate)
    substrate._counters = SessionCounterStore()
    mined = {}
    monkeypatch.setattr(substrate, "_ensure_instance", lambda: None)
    monkeypatch.setattr(substrate, "mine", lambda **kwargs: mined.update(kwargs))
    text = "The north garden needs water on Fridays."

    result = substrate.store_session_memory(text, memory_type="factual", project_path=str(tmp_path))

    assert result == 42
    assert sys.modules["embeddings"].STORED == [text]
    assert mined["embedding_id"] == 42
