"""Tests for lib.hal.voice.usage_cli — ``hal voice usage`` subcommand."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.hal.voice.usage import UsageCounter  # noqa: E402
from lib.hal.voice.usage_cli import main, render  # noqa: E402


@pytest.fixture
def usage_file(tmp_path: Path) -> Path:
    return tmp_path / "usage.jsonl"


def _seed(path: Path, provider: str, chars: int) -> None:
    UsageCounter(path=path).record(provider, chars)


def test_render_has_header_and_separator(usage_file: Path) -> None:
    rows = UsageCounter(path=usage_file).summary()
    out = render(rows)
    assert "provider" in out
    assert "chars/mo" in out
    assert "limit" in out
    assert "pct" in out
    assert "-" * 10 in out


def test_render_includes_all_tier3_providers(usage_file: Path) -> None:
    rows = UsageCounter(path=usage_file).summary()
    out = render(rows)
    for p in ("google_cloud_tts_free", "gemini_tts_free", "piper", "macos_say"):
        assert p in out


def test_render_shows_n_a_for_unlimited_providers(usage_file: Path) -> None:
    rows = UsageCounter(path=usage_file).summary()
    out = render(rows)
    piper_line = next(line for line in out.splitlines() if line.startswith("piper"))
    assert "n/a" in piper_line


def test_render_warn_when_over_threshold(usage_file: Path) -> None:
    _seed(usage_file, "google_cloud_tts_free", 850_000)
    rows = UsageCounter(path=usage_file).summary()
    out = render(rows)
    assert "WARN" in out
    assert "WARNING" in out


def test_render_no_warn_when_under_threshold(usage_file: Path) -> None:
    _seed(usage_file, "google_cloud_tts_free", 500_000)
    rows = UsageCounter(path=usage_file).summary()
    out = render(rows)
    assert "WARN" not in out


def test_main_exits_zero_when_all_under_threshold(usage_file: Path) -> None:
    _seed(usage_file, "google_cloud_tts_free", 500_000)
    assert main(["usage", "--path", str(usage_file)]) == 0


def test_main_exits_one_when_any_over_threshold(usage_file: Path) -> None:
    _seed(usage_file, "google_cloud_tts_free", 850_000)
    assert main(["usage", "--path", str(usage_file)]) == 1


def test_main_respects_month_argument(usage_file: Path, capsys) -> None:
    _seed(usage_file, "google_cloud_tts_free", 500_000)
    assert main(["usage", "--path", str(usage_file), "--month", "2020-01"]) == 0


def test_main_writes_output_to_stdout(usage_file: Path, capsys) -> None:
    main(["usage", "--path", str(usage_file)])
    captured = capsys.readouterr()
    assert "provider" in captured.out
    assert captured.err == ""


def test_main_usage_required_subcommand(usage_file: Path) -> None:
    with pytest.raises(SystemExit):
        main([])
