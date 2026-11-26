"""Regression: grip-env must NEVER emit secrets when given arguments.

Origin 2026-06-22: a `grip-env --help` probe (there is no such flag) fell
through to the bare secret-emitting path and leaked every decrypted key into a
session transcript. The arg-guard in main() short-circuits BEFORE any
decryption. These tests are Goodhart-proof — delete the guard and the
args-must-not-emit tests fail (the bare path would run and emit exports).
"""
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path

_GE_PATH = Path(__file__).resolve().parent.parent / "bin" / "grip-env"


def _load():
    # grip-env has no .py extension, so an explicit SourceFileLoader is needed
    # (spec_from_file_location can't infer a loader for an extensionless file).
    loader = SourceFileLoader("grip_env_mod", str(_GE_PATH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


ge = _load()


def _has_export_line(text: str) -> bool:
    """True if any line STARTS with 'export ' — the real secret-emit format
    (`export NAME=value`). Precise: ignores the usage's mid-sentence mention
    of `export KEY=VALUE`."""
    return any(ln.startswith("export ") for ln in text.splitlines())


def test_help_flag_prints_usage_to_stdout_no_secrets(monkeypatch, capsys):
    monkeypatch.setattr(ge.sys, "argv", ["grip-env", "--help"])
    rc = ge.main()
    out = capsys.readouterr()
    assert rc == 0
    assert "Takes NO arguments" in out.out
    # No EMITTED secret line: real emit is `print(f"export {NAME}={value}")` — a
    # line STARTING with "export ". The usage text mentions "export KEY=VALUE"
    # mid-sentence, so a substring check would false-trip; line-start is precise.
    assert not _has_export_line(out.out)


def test_unknown_flag_exits_2_and_emits_no_secrets(monkeypatch, capsys):
    monkeypatch.setattr(ge.sys, "argv", ["grip-env", "--bogus"])
    rc = ge.main()
    out = capsys.readouterr()
    assert rc == 2
    assert not _has_export_line(out.out)
    assert not _has_export_line(out.err)  # usage on stderr emits no secret line
    assert "Takes NO arguments" in out.err


def test_guard_runs_before_decrypt(monkeypatch):
    # With ANY arg, neither passphrase fetch nor decrypt may be reached.
    called = {"pass": False, "decrypt": False}
    monkeypatch.setattr(ge, "_fetch_passphrase",
                        lambda *_a, **_k: called.__setitem__("pass", True) or "x")
    monkeypatch.setattr(ge, "_decrypt",
                        lambda *_a, **_k: called.__setitem__("decrypt", True) or "X=1")
    monkeypatch.setattr(ge.sys, "argv", ["grip-env", "anything"])
    ge.main()
    assert called == {"pass": False, "decrypt": False}


def test_bare_invocation_still_emits(monkeypatch, capsys):
    # The no-arg path is byte-for-byte unchanged: decrypt + emit exports.
    monkeypatch.setattr(ge.sys, "argv", ["grip-env"])
    monkeypatch.setattr(ge, "_env_path", lambda: _GE_PATH)  # any existing file
    monkeypatch.setattr(ge, "_fetch_passphrase", lambda *_a, **_k: "pw")
    monkeypatch.setattr(ge, "_decrypt", lambda *_a, **_k: "FOO=bar\n")
    rc = ge.main()
    out = capsys.readouterr()
    assert rc == 0
    assert "export FOO=bar" in out.out
