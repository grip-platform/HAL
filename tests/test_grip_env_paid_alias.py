"""grip-env holds back OPENAI_API_KEY unless GRIP_CODEX_ALLOW_PAID is exactly "1".

HAL#668 item 3: grip-env copied the per-token OpenAI key into OPENAI_API_KEY for
anything run under grip-run, so a third-party tool could bill per token without
asking either spend guard. The alias is now opt-in, like every other paid OpenAI
path. tests/conftest.py sets GRIP_CODEX_ALLOW_PAID=1 for every test, so each test
here sets or removes it explicitly.
"""
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

_GE_PATH = Path(__file__).resolve().parent.parent / "bin" / "grip-env"
_BODY = "GRIP_OPENAI=fake-openai-value\nGRIP_GROQ=fake-groq-value\n"


def _load():
    # grip-env has no .py extension, so it needs an explicit SourceFileLoader.
    loader = SourceFileLoader("grip_env_paid_alias_mod", str(_GE_PATH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


ge = _load()


def _exports(text):
    return [line for line in text.splitlines() if line.startswith("export ")]


def test_openai_alias_held_back_without_opt_in(monkeypatch, capsys):
    monkeypatch.delenv("GRIP_CODEX_ALLOW_PAID", raising=False)
    count = ge._emit_exports(_BODY)
    out = capsys.readouterr()
    lines = _exports(out.out)
    assert "export GRIP_OPENAI=fake-openai-value" in lines
    assert not any(line.startswith("export OPENAI_API_KEY=") for line in lines)
    assert "export GROQ_API_KEY=fake-groq-value" in lines  # other aliases unaffected
    assert count == len(lines)
    assert "OPENAI_API_KEY not exported" in out.err
    assert "GRIP_CODEX_ALLOW_PAID=1" in out.err
    assert "fake-openai-value" not in out.err


def test_openai_alias_exported_with_opt_in(monkeypatch, capsys):
    monkeypatch.setenv("GRIP_CODEX_ALLOW_PAID", "1")
    count = ge._emit_exports(_BODY)
    out = capsys.readouterr()
    lines = _exports(out.out)
    assert "export OPENAI_API_KEY=fake-openai-value" in lines
    assert count == len(lines)
    assert out.err == ""


_DIRECT = "OPENAI_API_KEY=fake-direct-value\n"


def test_direct_openai_line_held_back_without_opt_in(monkeypatch, capsys):
    """AGORA on HAL#699: a line under the paid name itself, not only the alias."""
    monkeypatch.delenv("GRIP_CODEX_ALLOW_PAID", raising=False)
    count = ge._emit_exports(_DIRECT)
    out = capsys.readouterr()
    assert _exports(out.out) == []
    assert count == 0
    assert "OPENAI_API_KEY not exported" in out.err
    assert "fake-direct-value" not in out.err


def test_direct_openai_line_exported_with_opt_in(monkeypatch, capsys):
    monkeypatch.setenv("GRIP_CODEX_ALLOW_PAID", "1")
    ge._emit_exports(_DIRECT)
    assert _exports(capsys.readouterr().out) == ["export OPENAI_API_KEY=fake-direct-value"]


@pytest.mark.parametrize("value", ["0", "true", "yes", " 1", "1 ", ""])
def test_opt_in_must_be_exactly_one(monkeypatch, capsys, value):
    monkeypatch.setenv("GRIP_CODEX_ALLOW_PAID", value)
    ge._emit_exports(_BODY)
    lines = _exports(capsys.readouterr().out)
    assert not any(line.startswith("export OPENAI_API_KEY=") for line in lines)
