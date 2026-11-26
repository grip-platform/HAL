"""Cross-platform regressions for runtime paths and sandbox text handling."""
import json
import ntpath
import os

import pytest

from lib.hal import deploy_render, file_sandbox, home, notification, obsidian_vault


@pytest.mark.parametrize("value", [r"C:\Users\runner", "C:~", r"\\server\share\runner"])
def test_windows_home_with_drive_or_share_wins(monkeypatch, value):
    monkeypatch.delenv("HAL_HOME", raising=False)
    monkeypatch.setenv("HOME", value)
    monkeypatch.setattr(home.sys, "platform", "win32")
    monkeypatch.setattr(home.os.path, "splitdrive", ntpath.splitdrive)
    monkeypatch.setattr(home.os.path, "expanduser", lambda _: pytest.fail("HOME should win"))
    assert str(home.hal_home()) == str(home.Path(os.path.join(value, ".hal")))


@pytest.mark.parametrize("value", ["/c~", "relative", ""])
def test_windows_home_without_drive_uses_native_lookup(monkeypatch, value):
    monkeypatch.delenv("HAL_HOME", raising=False)
    monkeypatch.setenv("HOME", value)
    monkeypatch.setattr(home.sys, "platform", "win32")
    monkeypatch.setattr(home.os.path, "splitdrive", ntpath.splitdrive)
    monkeypatch.setattr(home.os.path, "expanduser", lambda value: r"C:\Users\native" if value == "~" else value)
    assert home.hal_home() == home.Path(os.path.join(r"C:\Users\native", ".hal"))


def test_posix_home_still_wins(monkeypatch):
    # No "/home/" in these literals: the public release builder rewrites home
    # paths, which would change the expected value in the shipped copy.
    monkeypatch.delenv("HAL_HOME", raising=False)
    monkeypatch.setenv("HOME", "/srv/custom-user")
    monkeypatch.setattr(home.sys, "platform", "linux")
    assert home.hal_home() == home.Path(os.path.join("/srv/custom-user", ".hal"))


@pytest.mark.parametrize("text", ["text", "text:34:more"])
def test_ripgrep_first_line_number_with_colon_in_matched_text(tmp_path, text):
    # _parse_rg_line now resolves the hit through _resolve_safe (the Windows
    # grep-escape fix), which calls realpath and pathlib.Path.relative_to --
    # both tied to the REAL host's path flavour, unlike the plain-string
    # os.path.relpath this test used to fake Windows syntax with. A path that
    # only LOOKS like a Windows path on a POSIX host no longer resolves
    # cleanly, so this uses a real, host-native root/file instead (Windows
    # drive-letter parsing itself is exercised for real on the Windows job;
    # test_file_sandbox.py's escaping-symlink tests cover _resolve_safe's
    # containment check). What stays worth pinning here: the non-greedy
    # "path:line:text" split must not stop at a colon inside the TEXT itself.
    target = tmp_path / "a.py"
    target.write_text("", encoding="utf-8")
    line = f"{target}:12:{text}"
    assert file_sandbox._parse_rg_line(line, str(tmp_path)) == file_sandbox.GrepMatch("a.py", 12, text)


@pytest.mark.parametrize("limit,expected", [(100, "é\nsecond\nthird\n"), (5, "é\ns")])
def test_read_normalizes_newlines_and_audits_returned_bytes(tmp_path, monkeypatch, limit, expected):
    (tmp_path / "text.txt").write_bytes("é\r\nsecond\rthird\n".encode("utf-8"))
    records = []
    monkeypatch.setattr(file_sandbox, "_audit", records.append)
    result = file_sandbox.read_text(str(tmp_path), "text.txt", max_bytes=limit)
    assert result == expected
    assert records[-1]["bytes"] == len(result.encode("utf-8"))
    assert records[-1]["truncated"] == (limit == 5)


def test_filesystem_root_is_refused(tmp_path, monkeypatch):
    # Isolate the root guard from the separately maintained system-directory list.
    monkeypatch.setattr(file_sandbox, "_REFUSED_ROOTS", set())
    with pytest.raises(file_sandbox.SandboxRootRefused, match="filesystem root"):
        file_sandbox.validate_root(tmp_path.anchor)


def test_notification_and_obsidian_follow_home_after_import(tmp_path, monkeypatch):
    for name in ("first", "second"):
        current = tmp_path / name
        monkeypatch.setenv("HAL_HOME", str(current))
        assert notification.notifications_dir() == str(current / "state" / "hal" / "notifications")
        assert obsidian_vault._load_mtime_cache() == {}
        cache = {f"{name}.md": 123.0}
        obsidian_vault._save_mtime_cache(cache)
        assert json.loads((current / "state" / "hal" / "vault-index.json").read_text()) == cache
        assert obsidian_vault._load_mtime_cache() == cache


@pytest.mark.parametrize("platform", ["systemd", "launchd"])
def test_posix_units_refuse_windows_absolute_paths(platform):
    with pytest.raises(deploy_render.RenderError, match="must be an absolute path"):
        deploy_render._check_absolute(platform, "hal_home", "C:/hal")
