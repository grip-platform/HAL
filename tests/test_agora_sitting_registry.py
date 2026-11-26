"""Each AGORA sitting keeps its own fixed-point registry (HAL#895, CodeTonight-SA/GRIP#9966).

Every sitting is the same caller to the kernel, so a registry shared across sittings let
one sitting that seated nobody register ``fingerprint({})`` and stop every later sitting
on the machine at depth 0. These tests drive the real kernel through ``_run_sitting`` with
the kernel's shared paths moved into a temporary folder.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from lib import improve_kernel
from lib.agora_pr_review import AGORA_THRESHOLD, _converge_step, _run_sitting

ROLES = ("plain_language", "quality", "technical")


@pytest.fixture()
def shared_registry(tmp_path, monkeypatch) -> Path:
    """The kernel's shared registry and telemetry, moved into this test's folder."""
    monkeypatch.setattr(improve_kernel, "_DEFAULT_TELEMETRY", tmp_path / "telemetry.jsonl")
    registry = tmp_path / "shared-fixed-points.jsonl"
    monkeypatch.setattr(improve_kernel, "_DEFAULT_FIXED_POINTS", registry)
    return registry


def _converge_for(chains):
    def converge(state, depth):
        return _converge_step(state, depth, pr_title="t", diff_excerpt="d", role_chains=chains)
    return converge


def _poison(registry: Path, converge) -> None:
    """The exact GRIP#9966 line, in the converge closure's namespace when the kernel has one."""
    record = {"fingerprint": improve_kernel.fingerprint({}), "ts": "2026-11-25T21:30:22Z"}
    namer = getattr(improve_kernel, "_default_namespace", None)
    if namer is not None:
        record["namespace"] = namer(converge)
    registry.write_text(json.dumps(record) + "\n")


def _sit(converge, answer) -> tuple[dict, list]:
    calls = []

    def fake_dispatch(role, prompt, happi_provider, auth, *args, **kwargs):
        calls.append((role, happi_provider))
        return answer(role, happi_provider)

    with patch("lib.agora_pr_review._dispatch_speaker", side_effect=fake_dispatch):
        result = _run_sitting(converge, 11)
    return result.state, calls


def test_control_the_poison_does_stop_a_kernel_on_the_shared_registry(shared_registry):
    """Without this control, a renamed namer or a kernel that binds its path at import
    would make the poison inert and the two tests below pass on a sitting that was
    never at risk."""
    converge = _converge_for({role: [(role, f"p/{role}", {"k": 1})] for role in ROLES})
    _poison(shared_registry, converge)
    calls = []

    def fake_dispatch(role, prompt, happi_provider, auth, *args, **kwargs):
        calls.append(role)
        return f"{role} text"

    with patch("lib.agora_pr_review._dispatch_speaker", side_effect=fake_dispatch):
        result = improve_kernel.InnerspeakerGo(
            converge=converge, threshold=AGORA_THRESHOLD, max_depth=11)({})
    assert calls == [] and result.depth == 0, "the planted line must stop a kernel that reads it"


def test_a_poisoned_shared_registry_neither_stops_nor_changes_a_sitting(shared_registry):
    converge = _converge_for({role: [(role, f"p/{role}", {"k": 1})] for role in ROLES})
    _poison(shared_registry, converge)
    before = shared_registry.read_bytes()
    state, calls = _sit(converge, lambda role, prov: f"{role} text")
    assert sorted(role for role, _ in calls) == sorted(ROLES)
    assert all(state[role] for role in ROLES)
    assert shared_registry.read_bytes() == before


def test_a_sitting_that_seats_nobody_writes_nothing_shared(shared_registry):
    converge = _converge_for({role: [] for role in ROLES})
    state, calls = _sit(converge, lambda role, prov: "unused")
    assert calls == [] and not any(state.get(role) for role in ROLES)
    assert not shared_registry.exists()
