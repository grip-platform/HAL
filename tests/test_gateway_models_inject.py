"""Hermetic tests for the /v1/models codex-compat transform.

`_inject_models_list` adds a top-level ``models`` array alongside Anthropic's
``data`` list so Codex 0.136's models_manager (required field ``models``) stops
failing with "missing field `models`". These tests are Goodhart-resistant: each
asserts a concrete value/shape, and each fails under an obvious mutation of the
implementation (no-op, overwrite-data, drop-id-filter, double-inject).

No network, no gateway process — pure-function tests.
"""
import json

from lib.hal.gateway import _inject_models_list


# A faithful slice of Anthropic's real GET /v1/models 200 body (verified live
# 2026-06-05: keys = data/has_more/first_id/last_id; item keys include type/id/
# display_name/created_at/...).
ANTHROPIC_SHAPE = {
    "data": [
        {"type": "model", "id": "claude-opus-4-8", "display_name": "Claude Opus 4.8"},
        {"type": "model", "id": "claude-sonnet-4-6", "display_name": "Claude Sonnet 4.6"},
    ],
    "has_more": False,
    "first_id": "claude-opus-4-8",
    "last_id": "claude-sonnet-4-6",
}


def _roundtrip(obj) -> dict:
    return json.loads(_inject_models_list(json.dumps(obj).encode("utf-8")))


def test_injects_models_alongside_data():
    out = _roundtrip(ANTHROPIC_SHAPE)
    # models injected, one entry per data item, id-only — the exact codex need
    assert out["models"] == [{"id": "claude-opus-4-8"}, {"id": "claude-sonnet-4-6"}]
    # data preserved BYTE-FOR-BYTE (additive, never a rewrite)
    assert out["data"] == ANTHROPIC_SHAPE["data"]
    assert out["has_more"] is False and out["first_id"] == "claude-opus-4-8"


def test_error_envelope_unchanged():
    # Anthropic auth-error envelope has no `data` list -> fail-open, byte-identical.
    env = {"type": "error", "error": {"type": "authentication_error"}, "request_id": "req_x"}
    raw = json.dumps(env).encode("utf-8")
    assert _inject_models_list(raw) == raw


def test_already_has_models_is_idempotent():
    # An upstream that already speaks `models` is never double-injected.
    obj = {"models": [{"id": "x"}], "data": [{"id": "y"}]}
    raw = json.dumps(obj).encode("utf-8")
    assert _inject_models_list(raw) == raw


def test_items_without_string_id_filtered():
    obj = {"data": [
        {"type": "model", "id": "keep-me"},
        {"type": "model"},               # no id -> dropped
        {"type": "model", "id": 123},    # non-string id -> dropped
        "not-a-dict",                    # non-dict -> dropped
    ]}
    assert _roundtrip(obj)["models"] == [{"id": "keep-me"}]


def test_empty_data_list_unchanged():
    # No real models -> do NOT inject an empty `models`; return raw unchanged.
    raw = json.dumps({"data": [], "has_more": False}).encode("utf-8")
    assert _inject_models_list(raw) == raw


def test_non_dict_json_unchanged():
    raw = json.dumps([1, 2, 3]).encode("utf-8")
    assert _inject_models_list(raw) == raw


def test_unparseable_bytes_unchanged():
    raw = b"\x00\x01 not json at all"
    assert _inject_models_list(raw) == raw
