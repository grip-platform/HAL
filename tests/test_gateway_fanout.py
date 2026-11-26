"""Goodhart-proof tests for the hal-fanout ensemble pseudo-model.

The invariants under test (each fails under the obvious weakening mutation):
  * ONLY the exact model id ``hal-fanout`` enters ensemble code — every other
    id takes the existing path unchanged (mutation: hook matches any id → RED).
  * The FIRST FLOOR-PASSING rung wins — not the first responder (mutation:
    floor accepts junk / first-response-wins → RED).
  * A hung rung can never stall the ensemble (mutation: drop the budget /
    make the race sequential → RED, the hung-rung test times out its assert).
  * All-rungs-fail returns a clean Anthropic-shaped 502 — never a hang or a
    bare 200 (mutation: swallow failures → RED).
  * Client shaping reuses the proven crosswire forms: JSON for a buffered
    client, one valid SSE burst for a streaming client.

Hermetic by DIP: tests inject ``rungs`` + ``runner`` (the seam the real
:func:`gateway_fanout._run_rung` implements) — no network, no monkeypatched
internals of the unit under test.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.hal import gateway as gw                              # noqa: E402
from lib.hal import gateway_egress as egress                   # noqa: E402
from lib.hal import gateway_fanout as fanout                   # noqa: E402
from lib.hal.gateway import UpstreamResponse                   # noqa: E402


def _anthropic_msg_bytes(text: str) -> bytes:
    """A minimal valid Anthropic Messages JSON body (what a non-streamed
    ``cross_wire_forward`` rung returns)."""
    return json.dumps({
        "id": "msg_test", "type": "message", "role": "assistant",
        "model": "rung-model",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }).encode("utf-8")


def _resp(status: int, body: bytes) -> UpstreamResponse:
    return UpstreamResponse(
        status=status,
        headers=[("Content-Type", "application/json")],
        _reader=io.BytesIO(body),
    )


def _runner_from(behaviours: dict):
    """Build a DIP runner: behaviours[(provider, model)] is either
    ("ok", status, body), ("sleep_ok", seconds, body), or ("raise", message)."""
    def run(provider, model, payload, per_rung_timeout):
        kind, *rest = behaviours[(provider, model)]
        if kind == "ok":
            status, body = rest
            return _resp(status, body)
        if kind == "sleep_ok":
            seconds, body = rest
            time.sleep(seconds)
            return _resp(200, body)
        raise ConnectionError(rest[0])
    return run


# --- selection: first FLOOR-PASSING wins, hedge + budget semantics ---

def test_first_floor_passing_wins_over_faster_junk():
    """An instantly-returning junk rung (empty text) must LOSE to a slower
    valid rung. Mutation (floor accepts empty text / first-response-wins) →
    the winner text flips → RED."""
    rungs = (("fastjunk", "a"), ("slowgood", "b"))
    runner = _runner_from({
        ("fastjunk", "a"): ("ok", 200, _anthropic_msg_bytes("")),
        ("slowgood", "b"): ("sleep_ok", 0.15, _anthropic_msg_bytes("real answer")),
    })
    resp = fanout.fanout_response(
        {"model": "hal-fanout"}, want_stream=False,
        rungs=rungs, runner=runner, per_rung_timeout=5, total_budget=5)
    assert resp.status == 200
    data = json.loads(resp.read_all())
    assert data["content"][0]["text"] == "real answer"


def test_hung_rung_cannot_stall_the_ensemble():
    """One rung sleeps far beyond the budget while another answers quickly —
    the ensemble must return on the fast rung's timescale. Mutation (remove
    the budget / run rungs sequentially in roster order) → wall-clock blows
    past the assert → RED."""
    rungs = (("hung", "x"), ("quick", "y"))
    runner = _runner_from({
        ("hung", "x"): ("sleep_ok", 30, _anthropic_msg_bytes("too late")),
        ("quick", "y"): ("sleep_ok", 0.05, _anthropic_msg_bytes("fast answer")),
    })
    started = time.monotonic()
    resp = fanout.fanout_response(
        {"model": "hal-fanout"}, want_stream=False,
        rungs=rungs, runner=runner, per_rung_timeout=60, total_budget=5)
    elapsed = time.monotonic() - started
    assert elapsed < 4, f"ensemble stalled behind a hung rung ({elapsed:.1f}s)"
    data = json.loads(resp.read_all())
    assert data["content"][0]["text"] == "fast answer"


def test_all_rungs_failing_returns_clean_502():
    """Error-status rungs + raising rungs (transport down) → a single clean
    Anthropic-shaped 502 naming hal-fanout, never a hang or a bare 200.
    Mutation (return the junk 200 anyway / raise out of the pool) → RED."""
    rungs = (("errs", "a"), ("down", "b"))
    runner = _runner_from({
        ("errs", "a"): ("ok", 429, b'{"type":"error"}'),
        ("down", "b"): ("raise", "connection refused"),
    })
    resp = fanout.fanout_response(
        {"model": "hal-fanout"}, want_stream=False,
        rungs=rungs, runner=runner, per_rung_timeout=5, total_budget=5)
    assert resp.status == 502
    body = resp.read_all()
    data = json.loads(body)
    assert data.get("type") == "error"
    assert b"hal-fanout" in body


def test_empty_roster_returns_clean_502():
    resp = fanout.fanout_response(
        {"model": "hal-fanout"}, want_stream=False,
        rungs=(), runner=lambda *a: None, per_rung_timeout=1, total_budget=1)
    assert resp.status == 502
    assert json.loads(resp.read_all()).get("type") == "error"


# --- client shaping: the two proven crosswire forms ---

def test_streaming_client_gets_one_valid_sse_burst():
    """want_stream=True → the winner is re-emitted as a text/event-stream
    burst (the same fallback shape a stream-ignoring provider takes today)."""
    rungs = (("good", "m"),)
    runner = _runner_from({("good", "m"): ("ok", 200, _anthropic_msg_bytes("hi"))})
    resp = fanout.fanout_response(
        {"model": "hal-fanout", "stream": True}, want_stream=True,
        rungs=rungs, runner=runner, per_rung_timeout=5, total_budget=5)
    assert resp.status == 200
    headers = dict(resp.headers)
    assert headers.get("Content-Type") == "text/event-stream"
    burst = resp.read_all()
    assert b"message_start" in burst and b"message_stop" in burst


def test_non_streaming_client_gets_json():
    rungs = (("good", "m"),)
    runner = _runner_from({("good", "m"): ("ok", 200, _anthropic_msg_bytes("hi"))})
    resp = fanout.fanout_response(
        {"model": "hal-fanout"}, want_stream=False,
        rungs=rungs, runner=runner, per_rung_timeout=5, total_budget=5)
    headers = dict(resp.headers)
    assert headers.get("Content-Type") == "application/json"
    assert json.loads(resp.read_all())["content"][0]["text"] == "hi"


# --- the deterministic floor ---

@pytest.mark.parametrize("status,body,expected", [
    (200, _anthropic_msg_bytes("a real answer"), True),
    (200, _anthropic_msg_bytes(""), False),                      # empty text
    (200, _anthropic_msg_bytes("   "), False),                   # whitespace only
    (429, _anthropic_msg_bytes("throttled"), False),             # non-200
    (200, b"not json at all", False),                            # non-JSON
    (200, b'{"type":"error","error":{"message":"x"}}', False),   # error envelope
    (200, b"", False),                                           # empty body
])
def test_floor_accepts_only_complete_nonempty_answers(status, body, expected):
    assert fanout._passes_floor(status, body) is expected


def test_floor_rejects_absurdly_long_text():
    huge = _anthropic_msg_bytes("x" * (fanout._MAX_TEXT_CHARS + 1))
    assert fanout._passes_floor(200, huge) is False


# --- roster resolution ---

def test_parse_rungs_and_unknown_provider_filtering(monkeypatch):
    """Env spec parses provider/model pairs (model may contain ':' or '/');
    providers absent from BOTH egress endpoint tables are dropped rather than
    crashing the ensemble. Mutation (skip the endpoint filter) → RED."""
    assert fanout._parse_rungs("groq/llama-x, ollama/qwen3:8b,bad-entry") == (
        ("groq", "llama-x"), ("ollama", "qwen3:8b"))
    monkeypatch.setenv(fanout._RUNGS_ENV,
                       "groq/llama-x,nosuchprovider/m,ollama/qwen3:8b")
    assert fanout.resolve_rungs() == (("groq", "llama-x"), ("ollama", "qwen3:8b"))


def test_default_rungs_all_resolve_to_known_endpoints(monkeypatch):
    """Every DEFAULT rung's provider must exist in the egress endpoint tables
    — a roster entry that cannot be forwarded is a configuration lie.
    Mutation (add a rung for an unknown provider) → RED."""
    monkeypatch.delenv(fanout._RUNGS_ENV, raising=False)
    assert fanout.resolve_rungs() == fanout.DEFAULT_RUNGS
    for provider, _model in fanout.DEFAULT_RUNGS:
        assert (provider in egress.PROVIDER_ENDPOINTS
                or provider in egress.LOCAL_PROVIDER_ENDPOINTS)


def test_default_rungs_carry_no_retired_moonshot_id():
    """Moonshot retired moonshot-v1-* and kimi-k2.5 on 2026-08-31
    (platform.kimi.ai/docs/models.md; HTTP 404 since), so the roster's
    Moonshot rung must be a live id. Red before 2026-09-07, when the rung was
    ("moonshot", "moonshot-v1-auto") and failed on every fan-out."""
    moonshot = [m for p, m in fanout.DEFAULT_RUNGS if p == "moonshot"]
    assert moonshot == ["kimi-k2.6"]
    retired = [m for _, m in fanout.DEFAULT_RUNGS
               if m.startswith(("moonshot-v1", "kimi-k2.5"))]
    assert retired == []


def test_ollama_endpoint_is_local_and_inert_for_implicit_routing():
    """The new ollama endpoint is localhost-only AND unreachable via family
    routing: same_wire_provider never yields 'ollama' for the rung model ids
    (qwen* family-routes to dashscope), so the table addition cannot change
    any existing request's route. Mutation (add an ollama family route) → RED."""
    assert egress.LOCAL_PROVIDER_ENDPOINTS["ollama"] == "http://127.0.0.1:11434/v1"
    for model in ("qwen2.5:3b", "qwen3:8b"):
        assert egress.same_wire_provider(model) != "ollama"


# --- the gateway hook: exact-id opt-in, everything else untouched ---

def _hook(body_obj) -> object:
    return gw._maybe_cross_wire("/v1/messages", "POST",
                                json.dumps(body_obj).encode("utf-8"))


def test_gateway_hook_routes_exactly_the_fanout_id(monkeypatch):
    """model == hal-fanout (and the gateway literal == the module constant) →
    fanout_response is called. Mutation (drop the hook / change the literal)
    → RED."""
    marker = _resp(200, b"{}")
    calls = []
    monkeypatch.setattr(fanout, "fanout_response",
                        lambda payload, want_stream: calls.append(
                            (payload["model"], want_stream)) or marker)
    monkeypatch.delenv("HAL_GATEWAY_MULTIPROVIDER", raising=False)
    out = _hook({"model": fanout.FANOUT_MODEL_ID, "stream": True})
    assert out is marker
    assert calls == [(fanout.FANOUT_MODEL_ID, True)]


def test_gateway_hook_never_fires_for_other_models(monkeypatch):
    """claude-* stays verbatim (None) and a known non-Anthropic model takes
    cross_wire_forward — NOT the ensemble. Mutation (hook matches any id /
    'not claude' instead of exact id) → RED."""
    boom = lambda *a, **k: pytest.fail("fanout entered for a non-fanout model")
    monkeypatch.setattr(fanout, "fanout_response", boom)
    monkeypatch.delenv("HAL_GATEWAY_MULTIPROVIDER", raising=False)

    assert _hook({"model": "claude-haiku-4-5-20251001"}) is None  # verbatim

    crosswired = _resp(200, b"{}")
    seen = []
    monkeypatch.setattr(
        egress, "cross_wire_forward",
        lambda payload, *, provider, want_stream: seen.append(provider) or crosswired)
    assert _hook({"model": "gemini-2.5-flash"}) is crosswired
    assert seen == ["gemini"]


def test_gateway_hook_respects_multiprovider_kill_switch(monkeypatch):
    """HAL_GATEWAY_MULTIPROVIDER=0 kills fanout too (checked FIRST, before
    the body is parsed) — hal-fanout then falls to the verbatim relay like
    every other non-claude id. Mutation (hook before the flag gate) → RED."""
    boom = lambda *a, **k: pytest.fail("fanout ran with multiprovider OFF")
    monkeypatch.setattr(fanout, "fanout_response", boom)
    monkeypatch.setenv("HAL_GATEWAY_MULTIPROVIDER", "0")
    assert _hook({"model": fanout.FANOUT_MODEL_ID}) is None
