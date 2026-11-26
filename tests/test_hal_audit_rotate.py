"""bin/hal-audit-rotate moves the audit log aside without losing a row.

The safety claim in its docstring is tested against the real sink rather than
asserted: the gateway's ``_default_sink`` opens the log per event, so a rename
between two events leaves the first in the rotated file and starts a new live
file with the second, and a write through a handle opened before the rename
still lands in the rotated file. HAL_HOME is the per-test home from conftest.
"""
from __future__ import annotations

import importlib.util
import os
from datetime import datetime, timezone
from importlib.machinery import SourceFileLoader
from pathlib import Path

from lib.hal.gateway_failover import _default_sink

_TOOL = Path(__file__).resolve().parent.parent / "bin" / "hal-audit-rotate"


def _load():
    loader = SourceFileLoader("hal_audit_rotate", str(_TOOL))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


tool = _load()


def _live() -> Path:
    return Path(os.environ["HAL_HOME"]) / "logs" / "gateway-audit.jsonl"


def _rotated(live: Path) -> list:
    return sorted(live.parent.glob(live.name + ".*.rotated"))


def test_the_rotated_name_carries_the_utc_date():
    live = Path("/x/logs/gateway-audit.jsonl")
    day = datetime(2026, 10, 5, 23, 59, tzinfo=timezone.utc)
    assert tool.rotated_name(live, day) == Path("/x/logs/gateway-audit.jsonl.2026-10-05.rotated")


def test_a_rename_between_two_sink_events_loses_neither(capsys):
    _default_sink({"event": "served", "n": 1})
    live = _live()
    before = live.read_bytes()

    assert tool.main([]) == 0  # default --audit resolves from HAL_HOME, as the sink does

    (rotated,) = _rotated(live)
    assert rotated.read_bytes() == before
    assert not live.exists()
    _default_sink({"event": "served", "n": 2})
    assert live.read_text().count("\n") == 1 and '"n":2' in live.read_text()
    assert rotated.read_bytes() == before
    assert str(rotated) in capsys.readouterr().out


def test_a_handle_opened_before_the_rename_writes_into_the_rotated_file():
    _default_sink({"event": "served", "n": 1})
    live = _live()
    with live.open("a", encoding="utf-8") as held:
        assert tool.main([]) == 0
        held.write('{"event":"served","n":2}\n')
    (rotated,) = _rotated(live)
    assert rotated.read_text().count("\n") == 2
    assert not live.exists()


def test_the_contamination_manifest_moves_with_the_log():
    _default_sink({"event": "served"})
    live = _live()
    manifest = live.with_name(live.name + ".contamination.json")
    manifest.write_text("{}\n")

    assert tool.main([]) == 0

    (rotated,) = _rotated(live)
    assert not manifest.exists()
    assert rotated.with_name(rotated.name + ".contamination.json").read_text() == "{}\n"


def test_an_existing_rotation_is_never_overwritten(tmp_path):
    live = tmp_path / "gateway-audit.jsonl"
    live.write_text("new\n")
    target = tmp_path / "gateway-audit.jsonl.2026-10-05.rotated"
    target.write_text("earlier\n")

    moves, refusal = tool.plan(live, target)

    assert moves == [] and "already exists" in refusal
    assert target.read_text() == "earlier\n" and live.read_text() == "new\n"


def test_an_existing_manifest_target_also_refuses_the_whole_rotation(tmp_path):
    live = tmp_path / "gateway-audit.jsonl"
    live.write_text("new\n")
    (tmp_path / "gateway-audit.jsonl.contamination.json").write_text("{}\n")
    target = tmp_path / "gateway-audit.jsonl.2026-10-05.rotated"
    (tmp_path / (target.name + ".contamination.json")).write_text("earlier\n")

    moves, refusal = tool.plan(live, target)

    assert moves == [] and "contamination.json" in refusal
    assert not target.exists()


def test_a_missing_log_is_refused_not_created(tmp_path, capsys):
    logs = tmp_path / "logs"
    logs.mkdir()
    assert tool.main(["--audit", str(logs / "gateway-audit.jsonl")]) == 1
    assert "nothing to rotate" in capsys.readouterr().err
    assert list(logs.iterdir()) == []
