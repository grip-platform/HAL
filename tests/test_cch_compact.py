"""Tests for the opt-in compact-2026-01-12 summarisation edit in cch.py.

The compact wiring is inert-by-construction: gated behind the
``server_compaction`` model capability, which is False for every model by
default. These tests pin the load-bearing invariants:

  * default path is byte-identical (no compact edit, no compact beta) — the
    feature cannot change an existing request unless explicitly opted in;
  * the compact edit and its ``compact-2026-01-12`` beta header ALWAYS
    co-occur (monotonicity — a compact edit without its beta 400s upstream);
  * opt-in alone is insufficient: the model must also advertise
    ``server_compaction`` (so the single-sourced, not-yet-live-confirmed shape
    can never reach a live request for an unmarked model).

Goodhart-proofing: assertions check actual payload values + header contents,
so a mutation that makes compact always-on, drops the edit, or decouples the
edit from the beta fails a test.
"""
import json

import pytest

from lib.hal.utils import cch


# ---- _compute_context_management: edit construction ----------------------

def test_compute_default_off_no_compact_edit():
    cm = cch._compute_context_management(has_thinking=True)
    assert cm is not None
    types = [e["type"] for e in cm["edits"]]
    assert "clear_thinking_20251015" in types
    assert cch._COMPACT_EDIT_TYPE not in types  # default off — behaviour unchanged


def test_compute_empty_returns_none():
    # No thinking, no clears, no compact -> field omitted entirely (unchanged).
    assert cch._compute_context_management(has_thinking=False) is None


def test_compute_compact_sole_edit():
    cm = cch._compute_context_management(has_thinking=False, compact=True)
    assert cm == {"edits": [{"type": "compact_20260112"}]}


def test_compute_compact_coexists_and_is_last():
    cm = cch._compute_context_management(has_thinking=True, compact=True)
    types = [e["type"] for e in cm["edits"]]
    assert "clear_thinking_20251015" in types
    assert types[-1] == "compact_20260112"


def test_compact_constants():
    assert cch._COMPACT_EDIT_TYPE == "compact_20260112"
    assert cch._COMPACT_BETA_HEADER == "compact-2026-01-12"


# ---- capability gate: inert by default -----------------------------------

def test_unknown_model_has_no_server_compaction():
    assert cch._CONSERVATIVE_DEFAULTS["server_compaction"] is False
    assert cch.supports_capability("totally-unknown-model", "server_compaction") is False


@pytest.fixture
def capable_model(monkeypatch):
    """A model that advertises both context_management AND server_compaction."""
    mid = "test-compact-capable"
    monkeypatch.setitem(
        cch.MODEL_CAPABILITIES, mid,
        {"context_management": True, "server_compaction": True, "adaptive_thinking": False},
    )
    return mid


def _body_dict(signer, model, **kw):
    body, _ = signer.build_body("hello", model=model, **kw)
    return json.loads(body)


def _beta_header(signer, model, **kw):
    headers, _ = signer.build_request(
        "hello", oauth_token="sk-ant-oat-test", model=model, **kw,
    )
    return headers["anthropic-beta"]


@pytest.fixture
def incapable_model(monkeypatch):
    """Supports context_management but NOT server_compaction.

    Synthetic (not a real registry model) so the incapable-path tests stay
    correct even after a real model is later marked server_compaction=true.
    """
    mid = "test-compact-incapable"
    monkeypatch.setitem(cch.MODEL_CAPABILITIES, mid, {"context_management": True})
    return mid


@pytest.fixture
def compact_only_model(monkeypatch):
    """Misconfig: server_compaction True but context_management False.

    The compact edit rides inside the context_management field, so this
    combination must yield NEITHER edit nor beta (never a 400-shaped split).
    """
    mid = "test-compact-misconfig"
    monkeypatch.setitem(
        cch.MODEL_CAPABILITIES, mid,
        {"server_compaction": True, "context_management": False},
    )
    return mid


# ---- build_body: edit gated by capability + opt-in -----------------------

def test_build_body_compact_fires_with_capability(capable_model):
    body = _body_dict(cch.CCHSigner(), capable_model, compact=True)
    assert {"type": "compact_20260112"} in body["context_management"]["edits"]


def test_build_body_compact_off_by_default(capable_model):
    body = _body_dict(cch.CCHSigner(), capable_model, compact=False)
    edits = body.get("context_management", {}).get("edits", [])
    assert all(e["type"] != "compact_20260112" for e in edits)


def test_build_body_compact_inert_without_capability(incapable_model):
    # context_management=True but server_compaction absent; opt-in alone must
    # NOT produce a compact edit (the model gate blocks it).
    assert not cch.supports_capability(incapable_model, "server_compaction")
    body = _body_dict(cch.CCHSigner(), incapable_model, compact=True)
    edits = body.get("context_management", {}).get("edits", [])
    assert all(e["type"] != "compact_20260112" for e in edits)


# ---- build_request: beta header gated identically to the edit ------------

def test_build_request_compact_beta_present_when_capable(capable_model):
    assert "compact-2026-01-12" in _beta_header(cch.CCHSigner(), capable_model, compact=True).split(",")


def test_build_request_compact_beta_absent_by_default(capable_model):
    assert "compact-2026-01-12" not in _beta_header(cch.CCHSigner(), capable_model, compact=False).split(",")


def test_build_request_compact_beta_stripped_for_incapable_model(incapable_model):
    assert not cch.supports_capability(incapable_model, "server_compaction")
    assert "compact-2026-01-12" not in _beta_header(cch.CCHSigner(), incapable_model, compact=True).split(",")


# ---- THE invariant: edit and beta always co-occur ------------------------

@pytest.mark.parametrize("compact", [True, False])
def test_monotonicity_edit_and_beta_co_occur(capable_model, compact):
    signer = cch.CCHSigner()
    headers, body_bytes = signer.build_request(
        "hello", oauth_token="sk-ant-oat-test", model=capable_model, compact=compact,
    )
    body = json.loads(body_bytes)
    edits = body.get("context_management", {}).get("edits", [])
    edit_present = any(e["type"] == "compact_20260112" for e in edits)
    beta_present = "compact-2026-01-12" in headers["anthropic-beta"].split(",")
    # Biconditional: the compact edit is in the body IFF the beta is in the header.
    assert edit_present == beta_present
    # And both track the caller's opt-in flag for a capable model.
    assert edit_present is compact


# ---- filter_beta_header gating -------------------------------------------

def test_filter_keeps_compact_for_capable(capable_model):
    assert cch.filter_beta_header("compact-2026-01-12", model=capable_model) == "compact-2026-01-12"


def test_filter_strips_compact_for_incapable(incapable_model):
    assert cch.filter_beta_header("compact-2026-01-12", model=incapable_model) == ""


# ---- misconfig edge + the single-source-of-truth gate --------------------

def test_misconfig_server_compaction_without_context_management(compact_only_model):
    # server_compaction=True but context_management=False: the compact edit
    # needs the context_management field to carry it, so BOTH the edit and the
    # beta must be absent -- never edit-without-beta (400) or beta-without-edit.
    # (The AGORA #369 "what I would falsify if I had time" forward anchor.)
    signer = cch.CCHSigner()
    headers, body_bytes = signer.build_request(
        "hello", oauth_token="sk-ant-oat-test", model=compact_only_model, compact=True,
    )
    body = json.loads(body_bytes)
    edits = body.get("context_management", {}).get("edits", [])
    assert all(e["type"] != "compact_20260112" for e in edits)  # no edit
    assert "compact-2026-01-12" not in headers["anthropic-beta"].split(",")  # no beta


def test_effective_compact_single_source(capable_model, incapable_model, compact_only_model):
    # The one gate both build_body and build_request consult.
    assert cch._effective_compact(capable_model, True) is True
    assert cch._effective_compact(capable_model, False) is False      # opt-in required
    assert cch._effective_compact(incapable_model, True) is False     # needs server_compaction
    assert cch._effective_compact(compact_only_model, True) is False  # needs context_management


# ---- FU1/FU2 registry activation (reads real configs/models.toml) -----------
# Asserts the deliberate activations; reads MODEL_CAPABILITIES as loaded from
# HAL_HOME/configs/models.toml (same pattern as test_cch_model_capabilities).

def test_registry_activates_live_confirmed_models():
    # opus-4-7, sonnet-4-6, opus-4-8 each returned 200 for the compact beta+edit
    # on the live CCH endpoint (2026-05-29) -> server_compaction activated.
    for model in ("claude-opus-4-7", "claude-sonnet-4-6", "claude-opus-4-8"):
        assert cch.supports_capability(model, "server_compaction"), model
        assert cch._effective_compact(model, True) is True, model


def test_opus_4_8_registered_full_capability_set():
    # opus-4-8 was absent (conservative defaults -> no betas); now registered
    # with the full live-confirmed set.
    for cap in ("adaptive_thinking", "context_management", "token_efficient_tools",
                "afk_mode", "server_compaction"):
        assert cch.supports_capability("claude-opus-4-8", cap), cap


def test_haiku_compact_stays_conservative_off():
    # Haiku was not compact-probed -> server_compaction omitted -> False.
    assert not cch.supports_capability("claude-haiku-4-5-20251001", "server_compaction")
