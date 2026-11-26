"""Unit tests for the four v1.1 cmd handlers (P3).

Goodhart-proof: each test asserts specific event content (type + payload),
not just call-counts. Mirrors canonical Python runtime behaviour from
``~/.hal/happi-sources/04-python-runtime.py``.

Side-effects gated:
- ``hypothesis.register`` tests use a tmp ``HAL_HYPOTHESES_PATH`` so the
  production registry is never touched.
- ``quine.spawn`` tests patch ``subprocess.run`` so no real ``gh`` calls fire.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
from unittest import mock

import pytest

from lib.hal.happi.envelope import HappiEnvelope
from lib.hal.happi.runtime import _dispatch, run


# ---------- helpers --------------------------------------------------------

def _dispatch_envelope(env_dict: dict) -> list[dict]:
    """Parse envelope dict, dispatch, return list of decoded JSON events."""
    envelope = HappiEnvelope.parse(env_dict)
    lines = list(_dispatch(envelope))
    return [json.loads(line) for line in lines]


def _run_via_stdin(env_dict: dict) -> tuple[int, list[dict]]:
    """Round-trip through ``run()`` reading from StringIO. Returns (rc, events)."""
    raw = json.dumps(env_dict)
    stream_in = io.StringIO(raw)
    stream_out = io.StringIO()
    rc = run(stream_in=stream_in, stream_out=stream_out)
    out = stream_out.getvalue().splitlines()
    events = [json.loads(line) for line in out if line.strip()]
    return rc, events


def _types(events: list[dict]) -> list[str]:
    return [e.get("type") for e in events]


def _final(events: list[dict]) -> dict:
    return events[-1]


# ===== idr.emit ============================================================

class TestIdrEmit:
    def test_happy_path_emits_started_idr_completed(self):
        events = _dispatch_envelope({
            "v": "happi/1.1", "id": "r-idr-1", "cmd": "idr.emit",
            "args": ['{"v":"happi/1.1","id":"x","cmd":"version"}',
                     '{"v":"happi/1.1","id":"x","type":"started","ts":0}\n'],
        })
        assert _types(events) == ["started", "idr", "completed"]
        idr_event = events[1]
        assert idr_event["id"] == "r-idr-1"
        assert "sha256" in idr_event["idr_ref"]
        assert len(idr_event["idr_ref"]["sha256"]) == 64
        assert idr_event["idr_ref"]["cid"] is None
        assert idr_event["idr_ref"]["model_versions"] == []
        assert idr_event["idr_ref"]["block_anchor"] is None

    def test_sha256_deterministic_across_runs(self):
        env = {
            "v": "happi/1.1", "id": "r-idr-det", "cmd": "idr.emit",
            "args": ["envelope-bytes", "events-bytes"],
        }
        events_a = _dispatch_envelope(env)
        events_b = _dispatch_envelope(env)
        sha_a = events_a[1]["idr_ref"]["sha256"]
        sha_b = events_b[1]["idr_ref"]["sha256"]
        assert sha_a == sha_b

    def test_empty_events_string_still_hashes(self):
        events = _dispatch_envelope({
            "v": "happi/1.1", "id": "r-idr-empty", "cmd": "idr.emit",
            "args": ["envelope-bytes", ""],
        })
        sha_with_empty = events[1]["idr_ref"]["sha256"]
        assert len(sha_with_empty) == 64
        # Different from envelope-only hash (proves events_str participates)
        events_just_env = _dispatch_envelope({
            "v": "happi/1.1", "id": "r-idr-env-only", "cmd": "idr.emit",
            "args": ["envelope-bytes-different", ""],
        })
        assert events_just_env[1]["idr_ref"]["sha256"] != sha_with_empty

    def test_missing_args_returns_parse_error(self):
        events = _dispatch_envelope({
            "v": "happi/1.1", "id": "r-idr-bad", "cmd": "idr.emit",
            "args": [],
        })
        assert _types(events) == ["started", "error"]
        err = events[1]
        assert err["code"] == "parse_error"
        assert "idr.emit" in err["message"]

    def test_one_arg_returns_parse_error(self):
        events = _dispatch_envelope({
            "v": "happi/1.1", "id": "r-idr-1arg", "cmd": "idr.emit",
            "args": ["envelope-only"],
        })
        assert _types(events) == ["started", "error"]
        assert events[1]["code"] == "parse_error"

    def test_model_versions_propagate_to_idr_ref(self):
        events = _dispatch_envelope({
            "v": "happi/1.1", "id": "r-idr-mv", "cmd": "idr.emit",
            "args": ["env", "evt"],
            "flags": {"model_versions": ["claude-opus-4-7", "gemini-2.5"]},
        })
        idr_event = events[1]
        assert idr_event["idr_ref"]["model_versions"] == [
            "claude-opus-4-7", "gemini-2.5",
        ]

    def test_block_anchor_propagates(self):
        events = _dispatch_envelope({
            "v": "happi/1.1", "id": "r-idr-ba", "cmd": "idr.emit",
            "args": ["env", "evt"],
            "flags": {"block_anchor": 42},
        })
        assert events[1]["idr_ref"]["block_anchor"] == 42


# ===== pr.reference ========================================================

class TestPrReference:
    def test_happy_path_emits_delta_with_pr_and_repo(self):
        events = _dispatch_envelope({
            "v": "happi/1.1", "id": "r-pr-1", "cmd": "pr.reference",
            "flags": {"pr": 123, "repo": "org/repo"},
        })
        assert _types(events) == ["started", "delta", "completed"]
        delta_text = events[1]["text"]
        assert delta_text == "PR #123 (org/repo)"
        assert events[2]["usage"] == {"pr": 123, "repo": "org/repo"}

    def test_with_description_arg(self):
        events = _dispatch_envelope({
            "v": "happi/1.1", "id": "r-pr-desc", "cmd": "pr.reference",
            "args": ["fixes the parser drift bug"],
            "flags": {"pr": 7, "repo": "CodeTonight-SA/HAL"},
        })
        delta_text = events[1]["text"]
        assert delta_text == (
            "PR #7 (CodeTonight-SA/HAL) | fixes the parser drift bug"
        )

    def test_with_status_and_axiom_flags(self):
        events = _dispatch_envelope({
            "v": "happi/1.1", "id": "r-pr-full", "cmd": "pr.reference",
            "args": ["main commit"],
            "flags": {"pr": 42, "repo": "x/y", "status": "merged",
                      "contributes_axiom": "self-bootstrap"},
        })
        delta_text = events[1]["text"]
        assert delta_text == (
            "PR #42 (x/y) | main commit | "
            "status=merged | axiom=self-bootstrap"
        )

    def test_missing_pr_flag_returns_parse_error(self):
        events = _dispatch_envelope({
            "v": "happi/1.1", "id": "r-pr-no-pr", "cmd": "pr.reference",
            "flags": {"repo": "org/repo"},
        })
        assert events[1]["code"] == "parse_error"
        assert "flags.pr (int)" in events[1]["message"]

    def test_pr_as_string_returns_parse_error(self):
        events = _dispatch_envelope({
            "v": "happi/1.1", "id": "r-pr-str", "cmd": "pr.reference",
            "flags": {"pr": "not-an-int", "repo": "org/repo"},
        })
        assert events[1]["code"] == "parse_error"

    def test_missing_repo_flag_returns_parse_error(self):
        events = _dispatch_envelope({
            "v": "happi/1.1", "id": "r-pr-no-repo", "cmd": "pr.reference",
            "flags": {"pr": 1},
        })
        assert events[1]["code"] == "parse_error"
        assert "flags.repo (str)" in events[1]["message"]

    def test_empty_repo_returns_parse_error(self):
        events = _dispatch_envelope({
            "v": "happi/1.1", "id": "r-pr-empty-repo", "cmd": "pr.reference",
            "flags": {"pr": 1, "repo": ""},
        })
        assert events[1]["code"] == "parse_error"


# ===== hypothesis.register =================================================

class TestHypothesisRegister:
    def test_happy_path_writes_ndjson_record(self, tmp_path,
                                             monkeypatch):
        out_path = tmp_path / "hypotheses.jsonl"
        monkeypatch.setenv("HAL_HYPOTHESES_PATH", str(out_path))
        events = _dispatch_envelope({
            "v": "happi/1.1", "id": "r-hyp-1", "cmd": "hypothesis.register",
            "args": ["H100"],
            "flags": {"claim": "test claim", "metric": "test_metric",
                      "prediction": "eq:0", "deadline": "2026-12-31"},
        })
        assert _types(events) == ["started", "delta", "delta", "completed"]
        usage = events[-1]["usage"]
        assert usage["hypothesis_id"] == "H100"
        assert usage["path"] == str(out_path)

        assert out_path.exists()
        with open(out_path) as f:
            record = json.loads(f.read().strip())
        assert record["v"] == "happi/1.1"
        assert record["id"] == "H100"
        assert record["claim"] == "test claim"
        assert record["metric"] == "test_metric"
        assert record["prediction"] == "eq:0"
        assert record["deadline"] == "2026-12-31"
        assert isinstance(record["registered_at_unix"], int)
        assert record["status"] == "pending"

    def test_record_has_all_seven_required_fields(self, tmp_path, monkeypatch):
        out_path = tmp_path / "hypotheses.jsonl"
        monkeypatch.setenv("HAL_HYPOTHESES_PATH", str(out_path))
        _dispatch_envelope({
            "v": "happi/1.1", "id": "r-hyp-7", "cmd": "hypothesis.register",
            "args": ["H7"],
            "flags": {"claim": "c", "metric": "m",
                      "prediction": "p", "deadline": "d"},
        })
        with open(out_path) as f:
            record = json.loads(f.read().strip())
        expected_keys = {
            "v", "id", "claim", "metric",
            "prediction", "deadline", "registered_at_unix", "status",
        }
        assert set(record.keys()) == expected_keys

    def test_missing_id_arg_returns_parse_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HAL_HYPOTHESES_PATH",
                           str(tmp_path / "hypotheses.jsonl"))
        events = _dispatch_envelope({
            "v": "happi/1.1", "id": "r-hyp-noid", "cmd": "hypothesis.register",
            "args": [],
            "flags": {"claim": "c", "metric": "m",
                      "prediction": "p", "deadline": "d"},
        })
        assert events[1]["code"] == "parse_error"
        assert "args[0]" in events[1]["message"]

    def test_missing_claim_returns_parse_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HAL_HYPOTHESES_PATH",
                           str(tmp_path / "hypotheses.jsonl"))
        events = _dispatch_envelope({
            "v": "happi/1.1", "id": "r-hyp-nc", "cmd": "hypothesis.register",
            "args": ["H1"],
            "flags": {"metric": "m", "prediction": "p", "deadline": "d"},
        })
        assert events[1]["code"] == "parse_error"
        assert "claim" in events[1]["message"]

    def test_appends_not_overwrites(self, tmp_path, monkeypatch):
        out_path = tmp_path / "hyp.jsonl"
        monkeypatch.setenv("HAL_HYPOTHESES_PATH", str(out_path))
        for hid in ("H1", "H2", "H3"):
            _dispatch_envelope({
                "v": "happi/1.1", "id": "r-" + hid,
                "cmd": "hypothesis.register",
                "args": [hid],
                "flags": {"claim": "c", "metric": "m",
                          "prediction": "p", "deadline": "d"},
            })
        with open(out_path) as f:
            lines = [line for line in f.read().splitlines() if line]
        assert len(lines) == 3
        ids = [json.loads(line)["id"] for line in lines]
        assert ids == ["H1", "H2", "H3"]

    def test_uses_env_var_override_path(self, tmp_path, monkeypatch):
        custom = tmp_path / "subdir" / "custom-name.jsonl"
        monkeypatch.setenv("HAL_HYPOTHESES_PATH", str(custom))
        events = _dispatch_envelope({
            "v": "happi/1.1", "id": "r-hyp-env", "cmd": "hypothesis.register",
            "args": ["HENV"],
            "flags": {"claim": "c", "metric": "m",
                      "prediction": "p", "deadline": "d"},
        })
        assert custom.exists()
        assert events[-1]["usage"]["path"] == str(custom)


# ===== quine.spawn =========================================================

def _mock_gh_view_success(repo: str, issue_num: int, title: str = "HAPPI seed generation 5",
                          body: str = "An envelope using quine.spawn cmd."):
    """Build a mock subprocess.run return value for `gh issue view`."""
    return mock.Mock(
        returncode=0,
        stdout=json.dumps({"title": title, "body": body}),
        stderr="",
    )


class TestQuineSpawn:
    def test_dry_run_emits_5_deltas_plus_completed(self, monkeypatch):
        monkeypatch.setenv("HAPPI_QUINE_DEPTH_MAX", "16")
        with mock.patch("lib.hal.happi.cmds_v11.subprocess.run") as mock_run:
            mock_run.return_value = _mock_gh_view_success(
                "org/repo", 100,
                title="HAPPI seed generation 3",
                body="quine.spawn envelope here",
            )
            events = _dispatch_envelope({
                "v": "happi/1.1", "id": "r-q-dry", "cmd": "quine.spawn",
                "flags": {"parent_issue": 100, "repo": "org/repo"},
            })
        types = _types(events)
        # started + 5 deltas + completed = 7 events
        assert types == ["started", "delta", "delta", "delta",
                         "delta", "delta", "completed"]
        # usage marker
        assert events[-1]["usage"]["dry_run"] is True
        assert events[-1]["usage"]["next_generation"] == 4
        # No real LIVE call: only the gh issue view ran
        assert mock_run.call_count == 1
        assert mock_run.call_args[0][0][:3] == ["gh", "issue", "view"]

    def test_dry_run_emits_proposed_title_with_bumped_generation(self,
                                                                  monkeypatch):
        with mock.patch("lib.hal.happi.cmds_v11.subprocess.run") as mock_run:
            mock_run.return_value = _mock_gh_view_success(
                "x/y", 1,
                title="Generation 7 quine",
                body="quine.spawn marker",
            )
            events = _dispatch_envelope({
                "v": "happi/1.1", "id": "r-q-bump", "cmd": "quine.spawn",
                "flags": {"parent_issue": 1, "repo": "x/y"},
            })
        proposed_delta = events[2]["text"]
        assert proposed_delta == "proposed_title: Generation 8 quine"

    def test_live_branch_creates_child_issue_via_gh(self, monkeypatch,
                                                     tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        view_response = _mock_gh_view_success(
            "x/y", 50,
            title="HAPPI seed generation 2",
            body="quine.spawn body",
        )
        create_response = mock.Mock(
            returncode=0,
            stdout="https://github.com/x/y/issues/999\n",
            stderr="",
        )
        with mock.patch("lib.hal.happi.cmds_v11.subprocess.run") as mock_run:
            mock_run.side_effect = [view_response, create_response]
            events = _dispatch_envelope({
                "v": "happi/1.1", "id": "r-q-live", "cmd": "quine.spawn",
                "flags": {"parent_issue": 50, "repo": "x/y", "live": True},
            })
        # started + 4 live deltas + completed
        assert _types(events) == ["started", "delta", "delta",
                                   "delta", "delta", "completed"]
        # First LIVE delta carries the spawned child issue number
        assert "spawned child #999" in events[1]["text"]
        # Two subprocess calls: view + create
        assert mock_run.call_count == 2
        view_call_argv = mock_run.call_args_list[0][0][0]
        create_call_argv = mock_run.call_args_list[1][0][0]
        assert view_call_argv[:2] == ["gh", "issue"]
        assert view_call_argv[2] == "view"
        assert create_call_argv[:3] == ["gh", "issue", "create"]
        # Audit completed usage payload
        usage = events[-1]["usage"]
        assert usage["parent"] == 50
        assert usage["child"] == 999
        assert usage["generation"] == 3

    def test_missing_parent_issue_returns_parse_error(self):
        events = _dispatch_envelope({
            "v": "happi/1.1", "id": "r-q-no-parent", "cmd": "quine.spawn",
            "flags": {"repo": "org/repo"},
        })
        assert events[1]["code"] == "parse_error"
        assert "parent_issue" in events[1]["message"]

    def test_parent_issue_as_string_returns_parse_error(self):
        events = _dispatch_envelope({
            "v": "happi/1.1", "id": "r-q-str", "cmd": "quine.spawn",
            "flags": {"parent_issue": "not-an-int", "repo": "org/repo"},
        })
        assert events[1]["code"] == "parse_error"

    def test_missing_repo_returns_parse_error(self):
        events = _dispatch_envelope({
            "v": "happi/1.1", "id": "r-q-no-repo", "cmd": "quine.spawn",
            "flags": {"parent_issue": 1},
        })
        assert events[1]["code"] == "parse_error"
        assert "repo" in events[1]["message"]

    def test_parent_not_a_seed_returns_runtime_error(self):
        with mock.patch("lib.hal.happi.cmds_v11.subprocess.run") as mock_run:
            mock_run.return_value = _mock_gh_view_success(
                "x/y", 1,
                title="Generation 1",
                body="just a regular issue body",  # No quine.spawn
            )
            events = _dispatch_envelope({
                "v": "happi/1.1", "id": "r-q-not-seed", "cmd": "quine.spawn",
                "flags": {"parent_issue": 1, "repo": "x/y"},
            })
        assert events[1]["code"] == "runtime_error"
        assert "not a HAPPI seed" in events[1]["message"]

    def test_depth_limit_exceeded_returns_runtime_error(self):
        with mock.patch("lib.hal.happi.cmds_v11.subprocess.run") as mock_run:
            mock_run.return_value = _mock_gh_view_success(
                "x/y", 1,
                title="HAPPI generation 16",
                body="quine.spawn marker",
            )
            events = _dispatch_envelope({
                "v": "happi/1.1", "id": "r-q-depth", "cmd": "quine.spawn",
                "flags": {"parent_issue": 1, "repo": "x/y",
                          "depth_limit": 16},
            })
        assert events[1]["code"] == "runtime_error"
        assert "depth_limit" in events[1]["message"]

    def test_gh_unavailable_returns_runtime_error(self):
        with mock.patch("lib.hal.happi.cmds_v11.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("gh: command not found")
            events = _dispatch_envelope({
                "v": "happi/1.1", "id": "r-q-no-gh", "cmd": "quine.spawn",
                "flags": {"parent_issue": 1, "repo": "x/y"},
            })
        assert events[1]["code"] == "runtime_error"
        assert "gh CLI unavailable" in events[1]["message"]


# ===== regression: unsupported cmd still returns unsupported_cmd ===========

class TestRegressions:
    def test_unsupported_cmd_still_returns_unsupported_cmd_error(self):
        events = _dispatch_envelope({
            "v": "happi/1.1", "id": "r-bogus", "cmd": "totally-fake-cmd",
        })
        # started + error + completed (per legacy fallthrough behaviour)
        assert _types(events)[:2] == ["started", "error"]
        assert events[1]["code"] == "unsupported_cmd"

    def test_existing_v10_cmds_still_work(self):
        events = _dispatch_envelope({
            "v": "happi/1.0", "id": "r-noop", "cmd": "noop",
        })
        assert _types(events) == ["started", "completed"]

    def test_v11_cmds_dispatch_with_v10_envelope(self):
        # Per spec: v1.0 envelope on v1.1 runtime — cmd dispatch unchanged
        events = _dispatch_envelope({
            "v": "happi/1.0", "id": "r-v10-pr", "cmd": "pr.reference",
            "flags": {"pr": 1, "repo": "x/y"},
        })
        assert _types(events) == ["started", "delta", "completed"]


# ===== run() integration: stdin → stdout round trip ========================

class TestRunIntegration:
    def test_idr_emit_via_run_returns_zero(self):
        rc, events = _run_via_stdin({
            "v": "happi/1.1", "id": "r-run-idr", "cmd": "idr.emit",
            "args": ["env-bytes", "evt-bytes"],
        })
        assert rc == 0
        assert _types(events) == ["started", "idr", "completed"]

    def test_pr_reference_via_run_returns_zero(self):
        rc, events = _run_via_stdin({
            "v": "happi/1.1", "id": "r-run-pr", "cmd": "pr.reference",
            "flags": {"pr": 1, "repo": "x/y"},
        })
        assert rc == 0
        assert _types(events) == ["started", "delta", "completed"]
