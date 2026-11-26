"""Tests for hal welcome --install / --uninstall (W4).

All tests mock launchctl and use per-test tmp paths for the plist, so
no real LaunchAgent is ever installed in the test suite.
"""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lib.hal.welcome import (
    LAUNCH_AGENT_LABEL,
    InstallResult,
    _render_program_args,
    _split_agent_args,
    install_launch_agent,
    main,
    render_launch_agent_plist,
    uninstall_launch_agent,
)


# ---------------- _render_program_args ----------------


def test_render_program_args_prepends_hal_bin_and_welcome():
    out = _render_program_args(Path("~/.hal/bin/hal"), [])
    lines = out.splitlines()
    assert lines[0].strip() == "<string>~/.hal/bin/hal</string>"
    assert lines[1].strip() == "<string>welcome</string>"
    assert len(lines) == 2


def test_render_program_args_appends_extra_args_in_order():
    out = _render_program_args(
        Path("/h"), ["--provider", "gemini", "--voice", "Samantha"],
    )
    extras = [line.strip() for line in out.splitlines()[2:]]
    assert extras == [
        "<string>--provider</string>",
        "<string>gemini</string>",
        "<string>--voice</string>",
        "<string>Samantha</string>",
    ]


def test_render_program_args_escapes_xml_special_characters():
    out = _render_program_args(Path("/h"), ["a&b", "c<d", "e>f"])
    assert "a&amp;b" in out
    assert "c&lt;d" in out
    assert "e&gt;f" in out
    extras_region = "\n".join(out.splitlines()[2:])
    assert "a&b" not in extras_region.replace("&amp;", "")


# ---------------- render_launch_agent_plist ----------------


def test_plist_includes_expected_structural_elements():
    plist = render_launch_agent_plist(Path("/h/hal"))
    assert plist.startswith('<?xml version="1.0"')
    assert "<plist version=\"1.0\">" in plist
    assert f"<string>{LAUNCH_AGENT_LABEL}</string>" in plist
    assert "<key>ProgramArguments</key>" in plist
    assert "<key>RunAtLoad</key>" in plist
    assert "<true/>" in plist
    assert "<key>KeepAlive</key>" in plist
    assert "<false/>" in plist


def test_plist_inlines_operator_hal_bin_path():
    plist = render_launch_agent_plist(Path("~/.hal/bin/hal"))
    assert "<string>~/.hal/bin/hal</string>" in plist


def test_plist_inlines_custom_log_and_err_paths(tmp_path):
    log = tmp_path / "custom.log"
    err = tmp_path / "custom.err"
    plist = render_launch_agent_plist(
        Path("/h/hal"), log_path=log, err_path=err,
    )
    assert str(log) in plist
    assert str(err) in plist


def test_plist_bakes_in_extra_welcome_args():
    plist = render_launch_agent_plist(
        Path("/h/hal"),
        extra_args=["--provider", "groq/llama-3.1-70b-versatile"],
    )
    assert "<string>--provider</string>" in plist
    assert "<string>groq/llama-3.1-70b-versatile</string>" in plist


# ---------------- install_launch_agent ----------------


class _MockLaunchctl:
    def __init__(self, rc: int = 0, output: str = ""):
        self.rc = rc
        self.output = output
        self.calls: list[tuple[str, str]] = []

    def __call__(self, action: str, plist_path: Path):
        self.calls.append((action, str(plist_path)))
        return (self.rc, self.output)


def test_install_writes_plist_and_loads_via_launchctl(tmp_path):
    plist = tmp_path / "agent.plist"
    ctl = _MockLaunchctl(rc=0)
    result = install_launch_agent(
        hal_bin=Path("/h/hal"),
        plist_path=plist,
        launch_agents_dir=tmp_path,
        launchctl_runner=ctl,
    )
    assert result.wrote_plist is True
    assert result.loaded is True
    assert plist.exists()
    actions = [c[0] for c in ctl.calls]
    assert actions == ["unload", "load"]


def test_install_refuses_to_overwrite_without_force(tmp_path):
    plist = tmp_path / "agent.plist"
    plist.write_text("<pre-existing/>", encoding="utf-8")
    ctl = _MockLaunchctl()
    result = install_launch_agent(
        hal_bin=Path("/h/hal"),
        plist_path=plist,
        launch_agents_dir=tmp_path,
        launchctl_runner=ctl,
    )
    assert result.wrote_plist is False
    assert result.loaded is False
    assert "--force" in result.message
    assert plist.read_text() == "<pre-existing/>"
    assert ctl.calls == []


def test_install_force_overwrites_existing_plist(tmp_path):
    plist = tmp_path / "agent.plist"
    plist.write_text("<pre-existing/>", encoding="utf-8")
    ctl = _MockLaunchctl(rc=0)
    result = install_launch_agent(
        hal_bin=Path("/h/hal"),
        plist_path=plist,
        launch_agents_dir=tmp_path,
        launchctl_runner=ctl,
        force=True,
    )
    assert result.wrote_plist is True
    assert result.loaded is True
    assert "<pre-existing/>" not in plist.read_text()
    assert "welcome" in plist.read_text()


def test_install_reports_launchctl_failure_but_keeps_plist(tmp_path):
    plist = tmp_path / "agent.plist"
    ctl = _MockLaunchctl(rc=1, output="bootstrap failed: 5: input/output error")
    result = install_launch_agent(
        hal_bin=Path("/h/hal"),
        plist_path=plist,
        launch_agents_dir=tmp_path,
        launchctl_runner=ctl,
    )
    assert result.wrote_plist is True
    assert result.loaded is False
    assert plist.exists()
    assert "launchctl load returned 1" in result.message


def test_install_creates_launch_agents_dir_when_missing(tmp_path):
    launch_agents = tmp_path / "LaunchAgents"
    plist = launch_agents / "agent.plist"
    ctl = _MockLaunchctl(rc=0)
    install_launch_agent(
        hal_bin=Path("/h/hal"),
        plist_path=plist,
        launch_agents_dir=launch_agents,
        launchctl_runner=ctl,
    )
    assert launch_agents.is_dir()
    assert plist.exists()


# ---------------- uninstall_launch_agent ----------------


def test_uninstall_removes_plist_and_unloads(tmp_path):
    plist = tmp_path / "agent.plist"
    plist.write_text("<existing/>", encoding="utf-8")
    ctl = _MockLaunchctl(rc=0)
    result = uninstall_launch_agent(plist_path=plist, launchctl_runner=ctl)
    assert not plist.exists()
    assert result.loaded is True
    assert ctl.calls == [("unload", str(plist))]


def test_uninstall_is_noop_when_plist_missing(tmp_path):
    plist = tmp_path / "agent.plist"
    ctl = _MockLaunchctl()
    result = uninstall_launch_agent(plist_path=plist, launchctl_runner=ctl)
    assert "nothing to uninstall" in result.message
    assert ctl.calls == []


def test_uninstall_still_removes_plist_even_when_launchctl_fails(tmp_path):
    plist = tmp_path / "agent.plist"
    plist.write_text("<existing/>", encoding="utf-8")
    ctl = _MockLaunchctl(rc=1, output="service not loaded")
    uninstall_launch_agent(plist_path=plist, launchctl_runner=ctl)
    assert not plist.exists()


# ---------------- _split_agent_args ----------------


def test_split_empty_string_returns_empty_list():
    assert _split_agent_args("") == []


def test_split_honours_shell_quoting():
    assert _split_agent_args("--provider gemini --voice 'Jamie (Premium)'") == [
        "--provider", "gemini", "--voice", "Jamie (Premium)",
    ]


def test_split_raises_on_unbalanced_quotes():
    with pytest.raises(ValueError):
        _split_agent_args("--voice 'broken")


# ---------------- main() --install / --uninstall CLI ----------------


def test_main_install_and_uninstall_together_is_rejected():
    stream = io.StringIO()
    rc = main(["--install", "--uninstall"], stream_out=stream)
    assert rc == 2
    assert "mutually exclusive" in stream.getvalue()


def test_main_install_invokes_install_launch_agent(monkeypatch):
    captured = {}

    def _spy(*, extra_args=None, force=False, **_):
        captured["extra_args"] = extra_args
        captured["force"] = force
        return InstallResult(
            plist_path=Path("/tmp/fake.plist"),
            wrote_plist=True,
            loaded=True,
            message="installed (fake)",
        )

    monkeypatch.setattr("lib.hal.welcome.install_launch_agent", _spy)
    stream = io.StringIO()
    rc = main(
        ["--install", "--force", "--agent-args", "--provider gemini"],
        stream_out=stream,
    )
    assert rc == 0
    assert captured["force"] is True
    assert captured["extra_args"] == ["--provider", "gemini"]
    assert "installed (fake)" in stream.getvalue()


def test_main_install_returns_1_when_loaded_false(monkeypatch):
    def _spy(**_):
        return InstallResult(
            plist_path=Path("/x"),
            wrote_plist=True,
            loaded=False,
            message="launchctl load returned 1: woe",
        )
    monkeypatch.setattr("lib.hal.welcome.install_launch_agent", _spy)
    stream = io.StringIO()
    rc = main(["--install"], stream_out=stream)
    assert rc == 1
    assert "woe" in stream.getvalue()


def test_main_uninstall_invokes_uninstall_launch_agent(monkeypatch):
    called = {"n": 0}

    def _spy():
        called["n"] += 1
        return InstallResult(
            plist_path=Path("/x"),
            wrote_plist=False,
            loaded=True,
            message="uninstalled (fake)",
        )
    monkeypatch.setattr("lib.hal.welcome.uninstall_launch_agent", _spy)
    stream = io.StringIO()
    rc = main(["--uninstall"], stream_out=stream)
    assert rc == 0
    assert called["n"] == 1
    assert "uninstalled (fake)" in stream.getvalue()


def test_main_install_with_no_agent_args_passes_empty_list(monkeypatch):
    captured = {}

    def _spy(*, extra_args=None, **_):
        captured["extra_args"] = extra_args
        return InstallResult(
            plist_path=Path("/x"), wrote_plist=True, loaded=True,
            message="ok",
        )
    monkeypatch.setattr("lib.hal.welcome.install_launch_agent", _spy)
    main(["--install"], stream_out=io.StringIO())
    assert captured["extra_args"] == []
