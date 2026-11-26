"""bin/hal-audit-contamination flags fixture and mislabelled rows, and nothing else.

Every row below is shaped like a real row of the live log on 2026-10-05: the
fixture rows are copies of what one full test-suite run wrote, and the clean
rows are copies of real traffic. The tool must also never modify the log it
reads, and must not overwrite an earlier manifest.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import SimpleNamespace

import pytest

from lib.hal.gateway_failover import FailoverProxy, UpstreamSpec

_TOOL = Path(__file__).resolve().parent.parent / "bin" / "hal-audit-contamination"


def _load():
    loader = SourceFileLoader("hal_audit_contamination", str(_TOOL))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


tool = _load()


def _row(event, upstream, status, served, failed_over, latency_ms, **extra):
    row = {"ts": "2026-10-05T01:00:00+00:00", "event": event, "path": "/v1/messages",
           "upstream": upstream, "status": status, "served": served,
           "failed_over": failed_over, "latency_ms": latency_ms}
    row.update(extra)
    return row


def _witness(provider, model, status, failed_over, latency_ms, path="/v1/messages"):
    return _row("gateway.chat.witness", provider, status, True, failed_over, latency_ms,
                provider=provider, model=model, path=path)


FIXTURE_UPSTREAM = [
    _row("failover", "a", 503, False, False, 0),
    _row("failover", "primary", 429, False, False, 0),
    _row("unreachable", "b", None, False, True, 0),
    _row("served", "secondary", 200, True, True, 0),
]
FIXTURE_WITNESS = [
    _witness("anthropic", "m", 429, False, 0, path="/v1/chat/completions"),
    _witness("openai", "gpt-5.5", 200, True, 0),
]
MISLABELLED = [_witness("gemini", "gemini-3.1-pro-preview", 200, True, 4210)]
CLEAN = [
    _row("served", "anthropic-primary", 200, True, False, 1814),
    _row("served", "anthropic-primary", 429, True, False, 233),
    _row("unreachable", "anthropic-primary", None, False, False, 30012),
    _row("failover", "a", 503, False, False, 7),            # a fake name, but not instant
    _witness("anthropic", "claude-opus-5", 200, False, 2320),
    _witness("moonshot", "kimi-k3", 501, False, 0),          # the gateway's own instant 501
]


@pytest.mark.parametrize("row", FIXTURE_UPSTREAM)
def test_fixture_upstream_rows_are_flagged(row):
    assert tool.classify(row) == "fixture-upstream"


@pytest.mark.parametrize("row", FIXTURE_WITNESS)
def test_instant_witness_rows_are_flagged_as_fixtures(row):
    assert tool.classify(row) == "fixture-witness"


def test_a_real_cross_wire_row_with_failed_over_true_is_flagged_as_mislabelled():
    assert tool.classify(MISLABELLED[0]) == "mislabelled-failed-over"


@pytest.mark.parametrize("row", CLEAN)
def test_real_traffic_is_never_flagged(row):
    assert tool.classify(row) is None


def test_the_mislabel_rule_is_scoped_to_the_messages_rail():
    # Only _witness_messages_rail ever set failed_over on a witness row; a true
    # value anywhere else would mean something new, not this defect.
    row = _witness("openai", "gpt-5.5", 200, True, 900, path="/v1/chat/completions")
    assert tool.classify(row) is None


def _write_log(path: Path, rows) -> list:
    lines = [json.dumps(r, separators=(",", ":")).encode() for r in rows]
    path.write_bytes(b"\n".join(lines) + b"\n")
    return lines


def test_manifest_indexes_the_log_without_touching_it(tmp_path):
    audit = tmp_path / "gateway-audit.jsonl"
    rows = CLEAN[:2] + FIXTURE_UPSTREAM[:1] + CLEAN[2:3] + MISLABELLED
    lines = _write_log(audit, rows)
    with audit.open("ab") as fh:
        fh.write(b"not json\n")
    before = audit.read_bytes()

    assert tool.main(["--audit", str(audit)]) == 0

    assert audit.read_bytes() == before
    manifest = json.loads((tmp_path / "gateway-audit.jsonl.contamination.json").read_text())
    assert manifest["lines_scanned"] == 6
    assert [(e["line"], e["reason"]) for e in manifest["rows"]] == [
        (3, "fixture-upstream"), (5, "mislabelled-failed-over")]
    for entry in manifest["rows"]:
        expected = hashlib.sha256(lines[entry["line"] - 1]).hexdigest()
        assert entry["sha256"] == expected
    assert manifest["counts"] == {"fixture-upstream": {"failover": 1},
                                  "mislabelled-failed-over": {"gateway.chat.witness": 1}}


def test_an_existing_manifest_is_never_overwritten_without_force(tmp_path):
    audit = tmp_path / "gateway-audit.jsonl"
    _write_log(audit, FIXTURE_UPSTREAM)
    out = tmp_path / "earlier.json"
    out.write_text("earlier manifest\n")

    assert tool.main(["--audit", str(audit), "--out", str(out)]) == 1
    assert out.read_text() == "earlier manifest\n"

    assert tool.main(["--audit", str(audit), "--out", str(out), "--force"]) == 0
    assert len(json.loads(out.read_text())["rows"]) == len(FIXTURE_UPSTREAM)


def _upstream(status):
    """A fake same-format upstream that always answers ``status``."""
    answer = SimpleNamespace(status=status, close=lambda: None)
    return SimpleNamespace(forward=lambda *args, **kwargs: answer)


def test_a_genuine_failover_is_never_flagged():
    """The AGORA council's falsifier on HAL#640: a real failover survives the scan.

    FailoverProxy runs through the default sink with the production upstream
    names, so the per-test log holds a ``failover`` row and a ``served`` row
    with ``failed_over`` true, exactly as a real failover would write them.
    Neither may be flagged: the fixture rule needs a fake upstream name, and the
    mislabel rule applies only to witness rows. Their latency is 0 here, as a
    fixture's is, so this also pins that the name condition is load-bearing.
    """
    names = ("anthropic-primary", "anthropic-fallback")
    upstreams = {names[0]: _upstream(429), names[1]: _upstream(200)}
    proxy = FailoverProxy([UpstreamSpec(n) for n in names],
                          proxy_factory=lambda spec: upstreams[spec.name])
    assert proxy.forward("/v1/messages", b"{}", []).status == 200

    log = Path(os.environ["HAL_HOME"]) / "logs" / "gateway-audit.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(r["event"], r["upstream"], r["failed_over"]) for r in rows] == [
        ("failover", names[0], False), ("served", names[1], True)]
    assert tool.scan(log)["rows"] == []
