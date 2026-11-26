"""The token counts a provider reports reach GRIP through /api/infer.

HAPPI's completed event carries "in" and "out", but GripInferResult kept only
provider, model and failover from it, and /api/infer returned no usage at all.
GRIP's adapter (lib/hal_llm_adapter.py) therefore recorded zero tokens for every
one of 1,027 primary-path calls in the ten days to 2026-10-02, and its spend
report read $0 for calls it never measured. These tests pin the counts at each
hop, ending at the JSON the real handler sends.

Named test_happi_* so the pytest (happi) CI job runs it.
"""
import json
from types import MappingProxyType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lib.hal import grip_runtime, server
from lib.hal.happi import cmds_infer
from lib.hal.types import TokenUsage


def _events(usage):
    return [{"type": "delta", "id": "e1", "text": "hello"},
            {"type": "completed", "id": "e1", "usage": usage}]


def test_the_result_keeps_the_completed_events_token_counts():
    result = grip_runtime._collect(
        _events({"in": 120, "out": 40, "provider": "groq", "model": "m"}), "m")
    assert result.usage == {"in": 120, "out": 40}
    assert (result.provider, result.model, result.text) == ("groq", "m", "hello")


@pytest.mark.parametrize("usage", [
    {},
    {"in": "12", "out": True},
    {"in": -1, "out": float("nan")},
    {"in": float("inf")},
])
def test_counts_the_provider_never_reported_stay_absent(usage):
    assert grip_runtime._collect(_events(usage), "m").usage == {}


def test_an_error_or_a_missing_completed_event_carries_no_counts():
    error = [{"type": "error", "id": "e1", "code": "x", "message": "y"}]
    assert grip_runtime._collect(error, "m").usage == {}
    assert grip_runtime._collect([{"type": "delta", "id": "e1", "text": "hi"}], "m").usage == {}


def test_the_http_names_are_the_ones_grips_adapter_reads():
    assert server._usage_for_http({"in": 5, "out": 7}) == {"input_tokens": 5, "output_tokens": 7}
    assert server._usage_for_http({}) == {}
    assert server._usage_for_http(None) == {}


def test_api_infer_sends_the_counts(monkeypatch):
    result = grip_runtime.GripInferResult(ok=True, text="hi", provider="groq", model="m",
                                          requested="m", usage={"in": 5, "out": 7})
    monkeypatch.setattr(grip_runtime, "grip_infer", lambda *_a, **_k: result)
    monkeypatch.setattr(server, "resolve_infer_model", lambda _body: None)
    handler = server._make_handler(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
    sent = []
    fake = SimpleNamespace(_read_body=lambda: {"prompt": "hello"},
                           _json=lambda payload, status=200: sent.append((status, payload)))
    handler._api_infer(fake)
    status, payload = sent[0]
    assert status == 200
    assert payload["ok"] is True
    assert payload["usage"] == {"input_tokens": 5, "output_tokens": 7}


def test_an_older_result_without_counts_sends_empty_usage_not_zero():
    old = SimpleNamespace(ok=True, text="hi", provider="groq", model="m", idr_ref=None,
                          error_code=None, error_message=None)
    payload = server._infer_response(old)
    assert payload["usage"] == {}
    assert payload["media_attached"] is None


# --- the hop below: cmds_infer builds the completed event (found live, 2026-10-03) ---
#
# After the fixes above, a real groq call still sent usage {}: grip_infer goes
# through cmds_infer, whose _emit_success built the completed usage from
# provider, model and requested only, dropping the response's TokenUsage.

def _success_events(usage, provider="groq"):
    response = SimpleNamespace(text="OK", provider=provider, model="openai/gpt-oss-120b", usage=usage)
    lines = cmds_infer._emit_success(SimpleNamespace(id="e1"), "groq/openai/gpt-oss-120b", response)
    return [json.loads(line) for line in lines]


def _completed_usage(usage, provider="groq"):
    return next(e for e in _success_events(usage, provider) if e["type"] == "completed")["usage"]


def test_the_completed_event_carries_the_providers_token_counts():
    events = _success_events(TokenUsage(prompt_tokens=78, completion_tokens=16, total_tokens=94))
    completed = next(e for e in events if e["type"] == "completed")
    assert (completed["usage"]["in"], completed["usage"]["out"]) == (78, 16)
    assert completed["usage"]["provider"] == "groq"
    assert grip_runtime._collect(events, "groq/openai/gpt-oss-120b").usage == {"in": 78, "out": 16}


# A side is kept only as a positive whole number: a zero cannot be told apart from
# TokenUsage's default, and a side nobody reported is left out, never invented as 0
# (AGORA on HAL#633).
@pytest.mark.parametrize("usage, expected", [
    (TokenUsage(prompt_tokens=78, completion_tokens=16), {"in": 78, "out": 16}),
    ({"prompt_tokens": 78, "completion_tokens": 16}, {"in": 78, "out": 16}),
    ({"input_tokens": 5, "output_tokens": 7}, {"in": 5, "out": 7}),
    (TokenUsage(prompt_tokens=0, completion_tokens=16), {"out": 16}),
    (SimpleNamespace(completion_tokens=16), {"out": 16}),
    ({"prompt_tokens": 78.0, "completion_tokens": 7.5}, {"in": 78}),
    (SimpleNamespace(prompt_tokens="9", completion_tokens=1), {"out": 1}),
    (SimpleNamespace(prompt_tokens=True, completion_tokens=3), {"out": 3}),
    ({"prompt_tokens": -4, "completion_tokens": float("inf")}, {}),
    ({"prompt_tokens": float("nan")}, {}),
    (TokenUsage(), {}),
    (None, {}),
    (MappingProxyType({"prompt_tokens": 9, "completion_tokens": 2}), {"in": 9, "out": 2}),
])
def test_each_side_is_kept_only_as_a_positive_whole_number(usage, expected):
    completed = _completed_usage(usage)
    assert {side: completed[side] for side in ("in", "out") if side in completed} == expected


def test_failover_and_token_counts_travel_together():
    usage = _completed_usage(TokenUsage(prompt_tokens=78, completion_tokens=16), provider="deepseek")
    assert usage["failover"] == {"requested": "groq", "served_by": "deepseek"}
    assert (usage["in"], usage["out"]) == (78, 16)


@pytest.mark.parametrize("usage, persisted", [
    ({"prompt_tokens": 78, "completion_tokens": 16}, {"input_tokens": 78, "output_tokens": 16}),
    (TokenUsage(prompt_tokens=0, completion_tokens=16), {"output_tokens": 16}),
])
def test_what_grip_receives_from_a_provider_response(usage, persisted):
    result = grip_runtime._collect(_success_events(usage), "groq/openai/gpt-oss-120b")
    assert server._usage_for_http(result.usage) == persisted


def test_the_public_infer_handler_carries_the_counts(monkeypatch):
    # Through dispatch_infer itself, so a second success emitter cannot slip past.
    response = SimpleNamespace(text="OK", provider="groq", model="openai/gpt-oss-120b",
                               usage={"prompt_tokens": 78, "completion_tokens": 16})
    monkeypatch.setattr(cmds_infer, "_resolve_response", lambda *_args, **_kwargs: (response, None))
    envelope = SimpleNamespace(id="e1", args=["hello"], flags={"model": "groq/openai/gpt-oss-120b"})
    events = [json.loads(line) for line in cmds_infer.dispatch_infer(envelope)]
    completed = next(e for e in events if e["type"] == "completed")
    assert (completed["usage"]["in"], completed["usage"]["out"]) == (78, 16)


def test_a_response_with_no_usage_attribute_adds_no_counts():
    response = SimpleNamespace(text="OK", provider="groq", model="openai/gpt-oss-120b")
    lines = cmds_infer._emit_success(SimpleNamespace(id="e1"), "groq/openai/gpt-oss-120b", response)
    completed = next(e for e in map(json.loads, lines) if e["type"] == "completed")
    assert "in" not in completed["usage"]
    assert "out" not in completed["usage"]
