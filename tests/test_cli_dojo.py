"""/dojo in the HAL REPL starts GRIP's dojo shell as a child process.

GRIP and HAL both name their top-level package ``lib``, so the first version of
/dojo (HAL#626) imported ``lib.dojo_shell`` into the REPL's own process, where
``lib`` already meant HAL's package; it raised ModuleNotFoundError every time and
printed "dojo unavailable". These tests pin the child-process launch, and one of
them runs a real child against a fake GRIP tree to show it crosses the name clash
that the in-process import cannot. Nothing here opens curses.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from lib.hal import cli


def _grip_tree(root: Path, body: str = "def open_dojo():\n    pass\n") -> Path:
    """A minimal GRIP tree. Like the real one, its lib/ has NO __init__.py: GRIP's
    lib is a namespace package, which is why a regular `lib` elsewhere can win the
    name. (A fake WITH __init__.py hid exactly that from the first version of these
    tests; a real run in a terminal found it.)"""
    (root / "lib").mkdir(parents=True)
    (root / "lib" / "dojo_shell.py").write_text(body)
    return root


def _hal_like_tree(root: Path) -> Path:
    """A tree holding a REGULAR `lib` package, the shape HAL's own lib has."""
    (root / "lib").mkdir(parents=True)
    (root / "lib" / "__init__.py").write_text("")
    return root


def _capture(monkeypatch, returncode: int = 0) -> list:
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=returncode)
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    return calls


def test_dojo_starts_grips_shell_in_a_child_rooted_in_the_grip_tree(tmp_path, monkeypatch):
    tree = _grip_tree(tmp_path / "grip")
    monkeypatch.setenv("CLAUDE_DIR", str(tree))
    calls = _capture(monkeypatch)
    cli._cmd_dojo()
    [(argv, kwargs)] = calls
    assert argv[:2] == [sys.executable, "-c"]
    assert "sys.modules['lib'] = lib" in argv[2]       # the name is pinned to GRIP's folder first
    # Order-sensitive: the capture must precede the write, and the write must be
    # the terminal call. A substring set would pass even if the write ran before
    # open_dojo(), which would NameError at runtime.
    assert argv[2].endswith(
        "from lib.dojo_shell import open_dojo; "
        "_c = open_dojo(); "
        "pathlib.Path(os.environ['GRIP_DOJO_RESULT']).write_text(_c, encoding='utf-8') "
        "if isinstance(_c, str) and _c.strip() else None"
    )
    # The result-file env var is how the command crosses back — pin it.
    assert kwargs["env"]["GRIP_DOJO_RESULT"]
    assert kwargs["cwd"] == tree
    assert kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(tree)
    assert kwargs["env"]["CLAUDE_DIR"] == str(tree)


def test_the_child_keeps_an_inherited_pythonpath_after_the_grip_tree(tmp_path, monkeypatch):
    # AGORA on HAL#629: replacing PYTHONPATH outright dropped whatever the parent had.
    tree = _grip_tree(tmp_path / "grip")
    monkeypatch.setenv("CLAUDE_DIR", str(tree))
    monkeypatch.setenv("PYTHONPATH", "/opt/extra")
    calls = _capture(monkeypatch)
    cli._cmd_dojo()
    assert calls[0][1]["env"]["PYTHONPATH"] == f"{tree}{os.pathsep}/opt/extra"


def _fake_python(tree: Path, mode: int) -> Path:
    path = tree / "venv" / "bin" / "python"
    path.parent.mkdir(parents=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(mode)
    return path


def test_dojo_prefers_grips_own_interpreter(tmp_path, monkeypatch):
    tree = _grip_tree(tmp_path / "grip")
    grip_python = _fake_python(tree, 0o755)
    monkeypatch.setenv("CLAUDE_DIR", str(tree))
    calls = _capture(monkeypatch)
    cli._cmd_dojo()
    assert calls[0][0][0] == str(grip_python)


@pytest.mark.skipif(
    os.name == "nt",
    reason=("Windows chmod()/stat() cannot represent a POSIX executable-bit "
            "difference between the 0o755 and 0o644 extensionless fixtures"),
)
def test_a_grip_python_that_cannot_run_is_passed_over(tmp_path, monkeypatch):
    # AGORA on HAL#629: a file that merely exists was chosen, so a broken venv
    # made /dojo fail although this interpreter could run it.
    tree = _grip_tree(tmp_path / "grip")
    _fake_python(tree, 0o644)
    monkeypatch.setenv("CLAUDE_DIR", str(tree))
    calls = _capture(monkeypatch)
    cli._cmd_dojo()
    assert calls[0][0][0] == sys.executable


def test_ctrl_c_ends_the_dojo_and_returns_to_the_repl(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CLAUDE_DIR", str(_grip_tree(tmp_path / "grip")))

    def interrupted(argv, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(cli.subprocess, "run", interrupted)
    try:
        cli._cmd_dojo()
    except KeyboardInterrupt:
        # Caught here so an escape fails this test instead of stopping pytest.
        pytest.fail("Ctrl-C escaped the dojo into the REPL")
    assert "dojo interrupted" in capsys.readouterr().out


def test_dojo_leaves_the_repls_import_path_alone(tmp_path, monkeypatch):
    tree = _grip_tree(tmp_path / "grip")
    monkeypatch.setenv("CLAUDE_DIR", str(tree))
    _capture(monkeypatch)
    before = list(sys.path)
    cli._cmd_dojo()
    assert sys.path == before


def test_a_missing_grip_tree_is_one_line_and_starts_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CLAUDE_DIR", str(tmp_path / "nowhere"))
    calls = _capture(monkeypatch)
    cli._cmd_dojo()
    assert calls == []
    assert "dojo unavailable" in capsys.readouterr().out


def test_a_shell_that_fails_is_reported_not_raised(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CLAUDE_DIR", str(_grip_tree(tmp_path / "grip")))
    _capture(monkeypatch, returncode=3)
    cli._cmd_dojo()
    assert "dojo exited with code 3" in capsys.readouterr().out


def test_a_launch_that_cannot_start_is_reported_not_raised(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CLAUDE_DIR", str(_grip_tree(tmp_path / "grip")))

    def cannot_start(argv, **kwargs):
        raise FileNotFoundError("no such interpreter")
    monkeypatch.setattr(cli.subprocess, "run", cannot_start)
    cli._cmd_dojo()
    assert "dojo unavailable: no such interpreter" in capsys.readouterr().out


def test_the_child_reaches_grips_lib_where_an_import_in_this_process_cannot(tmp_path, monkeypatch):
    marker = tmp_path / "opened"
    tree = _grip_tree(tmp_path / "grip", f"def open_dojo():\n    open({str(marker)!r}, 'w').write('ok')\n")
    monkeypatch.setenv("CLAUDE_DIR", str(tree))
    # As inside HAL's REPL: a regular `lib` package rides on the inherited PYTHONPATH.
    monkeypatch.setenv("PYTHONPATH", str(_hal_like_tree(tmp_path / "hal")))

    # The bug HAL#626 shipped: with HAL's `lib` loaded, GRIP's module is unreachable.
    monkeypatch.syspath_prepend(str(tree))
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("lib.dojo_shell")

    cli._cmd_dojo()                                  # a real child process, no curses
    assert marker.exists(), "the child could not reach GRIP's lib past a regular `lib` package"
    assert marker.read_text() == "ok"


def test_a_chosen_command_is_surfaced_by_the_parent_after_terminal_restore(tmp_path, monkeypatch, capsys):
    """Facta, non verba: the parent prints the chosen command, after the child
    returns and the terminal is restored. The child writes the command to the
    result file; the parent reads and prints it — never the child printing into
    a possibly-uncooked tty."""
    tree = _grip_tree(tmp_path / "grip", "def open_dojo():\n    return '/converge'\n")
    monkeypatch.setenv("CLAUDE_DIR", str(tree))
    cli._cmd_dojo()
    import re
    out = re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out)
    assert "run this in your GRIP session: /converge" in out


def test_an_empty_command_is_dropped_not_printed(tmp_path, monkeypatch, capsys):
    """A blank/empty return is a dead instruction, not a command — the parent
    drops it rather than printing 'run this: ' with no payload."""
    tree = _grip_tree(tmp_path / "grip", "def open_dojo():\n    return ''\n")
    monkeypatch.setenv("CLAUDE_DIR", str(tree))
    cli._cmd_dojo()
    out = capsys.readouterr().out
    assert "run this in your GRIP session" not in out


def test_a_multiline_command_is_dropped_not_printed(tmp_path, monkeypatch, capsys):
    """An embedded newline would hijack the prompt — dropped, never printed."""
    tree = _grip_tree(tmp_path / "grip", "def open_dojo():\n    return 'a\\nb'\n")
    monkeypatch.setenv("CLAUDE_DIR", str(tree))
    cli._cmd_dojo()
    assert "run this in your GRIP session" not in capsys.readouterr().out


def test_no_choice_prints_nothing(tmp_path, monkeypatch, capsys):
    """open_dojo returning None (quit without choosing) emits nothing at all."""
    tree = _grip_tree(tmp_path / "grip")  # default body: `pass` -> None
    monkeypatch.setenv("CLAUDE_DIR", str(tree))
    cli._cmd_dojo()
    assert "run this in your GRIP session" not in capsys.readouterr().out


def test_a_control_byte_in_the_command_is_dropped(tmp_path, monkeypatch, capsys):
    """An ESC/control byte would inject into the prompt after restore — dropped."""
    tree = _grip_tree(tmp_path / "grip", "def open_dojo():\n    return '/conv\\x1b[31merge'\n")
    monkeypatch.setenv("CLAUDE_DIR", str(tree))
    cli._cmd_dojo()
    assert "run this in your GRIP session" not in capsys.readouterr().out


def test_a_bidi_or_zero_width_char_is_dropped(tmp_path, monkeypatch, capsys):
    """Format characters (U+202E bidi override, U+200B zero-width) are Cc/Cf and
    must be rejected too — a bare ord<32 check lets them through."""
    for i, payload in enumerate(("/conv\u202ege", "/con\u200bverge")):
        tree = _grip_tree(tmp_path / f"grip{i}", f"def open_dojo():\n    return {payload!r}\n")
        monkeypatch.setenv("CLAUDE_DIR", str(tree))
        cli._cmd_dojo()
        assert "run this in your GRIP session" not in capsys.readouterr().out, payload


def test_a_del_char_is_dropped(tmp_path, monkeypatch, capsys):
    """DEL (\\x7f) is Cc and must be rejected."""
    tree = _grip_tree(tmp_path / "grip", "def open_dojo():\n    return '/conv\\x7ferge'\n")
    monkeypatch.setenv("CLAUDE_DIR", str(tree))
    cli._cmd_dojo()
    assert "run this in your GRIP session" not in capsys.readouterr().out


def test_a_full_temp_dir_is_reported_not_raised(tmp_path, monkeypatch, capsys):
    """mkstemp raising OSError (full/unwritable temp dir) is one line, not a
    crash out of _cmd_dojo (the REPL never crashes)."""
    tree = _grip_tree(tmp_path / "grip")
    monkeypatch.setenv("CLAUDE_DIR", str(tree))
    monkeypatch.setattr(cli.tempfile, "mkstemp",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("No space left")))
    cli._cmd_dojo()  # must not raise
    assert "dojo unavailable: No space left" in capsys.readouterr().out


def test_an_oversized_result_is_dropped(tmp_path, monkeypatch, capsys):
    """A buggy child writing a huge payload is dropped, never read into memory."""
    tree = _grip_tree(tmp_path / "grip", "def open_dojo():\n    return 'x' * 10000\n")
    monkeypatch.setenv("CLAUDE_DIR", str(tree))
    cli._cmd_dojo()
    assert "run this in your GRIP session" not in capsys.readouterr().out


# --- AGORA's second review of HAL#629 -----------------------------------------

def _fake_terminal(monkeypatch) -> list:
    """A terminal whose settings can be read and restored; returns what was restored."""
    restored = []
    monkeypatch.setattr(cli, "termios", SimpleNamespace(
        TCSADRAIN=1, tcgetattr=lambda fd: ["raw-free settings"],
        tcsetattr=lambda fd, when, settings: restored.append(settings)))
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(fileno=lambda: 0))
    return restored


@pytest.mark.parametrize("ending", ["returns", "Ctrl-C"])
def test_the_terminal_is_put_back_whatever_happened_to_the_child(tmp_path, monkeypatch, ending):
    # run() kills the child at once on Ctrl-C, which can leave curses' raw mode behind.
    monkeypatch.setenv("CLAUDE_DIR", str(_grip_tree(tmp_path / "grip")))
    restored = _fake_terminal(monkeypatch)

    def child(argv, **kwargs):
        if ending == "Ctrl-C":
            raise KeyboardInterrupt
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(cli.subprocess, "run", child)
    try:
        cli._cmd_dojo()
    except KeyboardInterrupt:
        pytest.fail("Ctrl-C escaped the dojo into the REPL")
    assert restored == [["raw-free settings"]]


def test_without_a_terminal_nothing_is_restored(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_DIR", str(_grip_tree(tmp_path / "grip")))
    monkeypatch.setattr(cli, "termios", None)
    calls = _capture(monkeypatch)
    cli._cmd_dojo()                                  # Windows, or no termios: still launches
    assert len(calls) == 1


@pytest.mark.parametrize("returncode", [-2, 130])
def test_a_child_ended_by_ctrl_c_reads_as_interrupted(tmp_path, monkeypatch, capsys, returncode):
    monkeypatch.setenv("CLAUDE_DIR", str(_grip_tree(tmp_path / "grip")))
    _capture(monkeypatch, returncode=returncode)
    cli._cmd_dojo()
    out = capsys.readouterr().out
    assert "dojo interrupted" in out and "exited with code" not in out


def test_grips_own_python_is_not_handed_hals_pythonhome(tmp_path, monkeypatch):
    tree = _grip_tree(tmp_path / "grip")
    _fake_python(tree, 0o755)
    monkeypatch.setenv("CLAUDE_DIR", str(tree))
    monkeypatch.setenv("PYTHONHOME", "/opt/hal-prefix")
    monkeypatch.setenv("VIRTUAL_ENV", "/opt/hal-venv")
    calls = _capture(monkeypatch)
    cli._cmd_dojo()
    env = calls[0][1]["env"]
    assert "PYTHONHOME" not in env and "VIRTUAL_ENV" not in env


def test_this_interpreter_keeps_its_own_pythonhome(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_DIR", str(_grip_tree(tmp_path / "grip")))    # no GRIP venv
    monkeypatch.setenv("PYTHONHOME", "/opt/hal-prefix")
    calls = _capture(monkeypatch)
    cli._cmd_dojo()
    assert calls[0][1]["env"]["PYTHONHOME"] == "/opt/hal-prefix"
