"""Server-level: /api/chat must SETTLE or RELEASE the throttle reservation on
every path (AGORA #384 — no budget leak on error).

This is the reviewer's "what I would falsify" anchor: force an exception between
the throttle ALLOW and record_cost, then assert the reservation did not leak.
Uses the _make_handler harness (per tests/test_server_infer.py).
"""

import types
from unittest.mock import MagicMock

import lib.hal.api_throttle as api_throttle
from lib.hal.api_throttle import ChatThrottle
from lib.hal.server import _make_handler


def _handler(conversation):
    """Real Handler from the factory, instantiated without a socket, with the
    throttle-relevant request fields stubbed.

    /api/chat routes through the SessionRegistry (per-session), not the shared
    ``conversation`` singleton, so we wire a ``sessions`` mock whose resolved
    session exposes the test's ``conversation`` as ``session.conversation``
    (fresh-session shape: message_count=0, a real str id). This makes the test's
    ``conversation.send`` side_effect/return_value drive the per-session path so
    the reservation assertions exercise the real settle/release logic rather than
    an auto-generated MagicMock."""
    sessions = MagicMock()
    session = MagicMock()
    session.id = "test-session"
    session.model = "m"
    session.message_count = 0
    session.conversation = conversation
    sessions.get_or_create.return_value = session
    H = _make_handler(MagicMock(), conversation, MagicMock(), sessions, MagicMock())
    h = H.__new__(H)
    h._captured = []
    h._json = lambda data, code=200: h._captured.append((data, code))
    h._read_body = lambda: {"message": "hi"}
    h.headers = {"CF-Connecting-IP": "203.0.113.50"}
    h.client_address = ("203.0.113.50", 1234)
    return h


def test_api_chat_releases_reservation_when_send_raises(monkeypatch):
    # The leak the AGORA review flagged: send() raises AFTER check() reserved a
    # unit. With the fix, _api_chat releases it -> spent back to 0 (no leak).
    throttle = ChatThrottle(per_min=10_000, daily_budget=100.0)
    monkeypatch.setattr(api_throttle, "_INSTANCE", throttle)
    conv = MagicMock()
    conv.send.side_effect = RuntimeError("model timeout")

    _handler(conv)._api_chat()

    assert throttle.spent == 0.0, "send error must release the reserved unit (no leak)"


def test_api_chat_releases_reservation_on_empty_message(monkeypatch):
    # An ALLOW followed by a 400 (empty body) must also release — the request
    # never reached the model, so it must not consume budget.
    throttle = ChatThrottle(per_min=10_000, daily_budget=100.0)
    monkeypatch.setattr(api_throttle, "_INSTANCE", throttle)
    h = _handler(MagicMock())
    h._read_body = lambda: {}  # no message -> 400 before any LLM call

    h._api_chat()

    assert throttle.spent == 0.0, "empty-message 400 must release the reserved unit"
    assert h._captured[-1][1] == 400


def test_api_chat_settles_real_cost_on_success(monkeypatch):
    # Success path: reserved 1, settled to the real token cost (5+7=12) -> spent
    # 12, NOT released and NOT stuck at the 1-unit reservation.
    throttle = ChatThrottle(per_min=10_000, daily_budget=1000.0)
    monkeypatch.setattr(api_throttle, "_INSTANCE", throttle)
    conv = MagicMock()
    conv.send.return_value = types.SimpleNamespace(
        text="ok", model="m", provider=None, latency_ms=10, cost_usd=0.0,
        usage=types.SimpleNamespace(prompt_tokens=5, completion_tokens=7),
    )  # provider=None -> skip _mark_active (its catalogue check is unrelated here)
    conv.history = []
    conv.cost.snapshot.return_value = types.SimpleNamespace(total_usd=0.0)

    h = _handler(conv)
    h._api_chat()

    assert throttle.spent == 12.0, "successful call settles to the real token cost"
    assert h._captured[-1][1] == 200


def test_chat_throttle_check_fails_open_on_unexpected_ip_error(monkeypatch):
    # client_ip_from_headers now lets UNEXPECTED errors propagate (narrowed
    # catch); _chat_throttle_check must catch them and fail-open (ALLOW, no
    # crash, no reservation). Proves the narrowing is safely backstopped.
    throttle = ChatThrottle(per_min=10_000, daily_budget=100.0)
    monkeypatch.setattr(api_throttle, "_INSTANCE", throttle)

    class _RaisingHeaders:
        def get(self, *a, **k):
            raise RuntimeError("boom")

    conv = MagicMock()
    conv.send.return_value = types.SimpleNamespace(
        text="ok", model="m", provider=None, latency_ms=1, cost_usd=0.0,
        usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )
    conv.history = []
    conv.cost.snapshot.return_value = types.SimpleNamespace(total_usd=0.0)
    h = _handler(conv)
    h.headers = _RaisingHeaders()  # .get raises -> propagates -> outer guard fail-opens

    h._api_chat()  # must NOT raise

    assert h._captured[-1][1] == 200, "fail-open: request allowed despite IP-resolve error"
    assert throttle.spent == 0.0, "fail-open path makes no reservation (check never ran)"


def test_response_token_cost_defensive():
    # AGORA #384 r3: token extraction must never raise on an odd response shape
    # and must floor at the 1-unit reservation when no usable token data exists.
    from lib.hal.server import _response_token_cost
    ok = types.SimpleNamespace(usage=types.SimpleNamespace(prompt_tokens=5, completion_tokens=7))
    assert _response_token_cost(ok) == 12.0
    assert _response_token_cost(types.SimpleNamespace(usage=None)) == 1.0
    assert _response_token_cost(types.SimpleNamespace()) == 1.0          # no usage attr
    assert _response_token_cost(types.SimpleNamespace(usage=types.SimpleNamespace())) == 1.0  # missing fields
    odd = types.SimpleNamespace(usage=types.SimpleNamespace(prompt_tokens="x", completion_tokens=None))
    assert _response_token_cost(odd) == 1.0                              # odd-shaped -> floor, no raise
    zero = types.SimpleNamespace(usage=types.SimpleNamespace(prompt_tokens=0, completion_tokens=0))
    assert _response_token_cost(zero) == 1.0                             # zero tokens -> floor


def test_api_chat_releases_reservation_even_if_serve_chat_raises(monkeypatch):
    # AGORA #384 r3: if _serve_chat itself raises (e.g. _json fails on a dropped
    # connection in the error path), _api_chat's finally must STILL release the
    # reservation. Old code (release outside a finally) would leak -> fails this.
    import pytest
    throttle = ChatThrottle(per_min=10_000, daily_budget=100.0)
    monkeypatch.setattr(api_throttle, "_INSTANCE", throttle)
    conv = MagicMock()
    conv.send.side_effect = RuntimeError("model down")
    h = _handler(conv)

    def _boom_json(data, code=200):
        raise BrokenPipeError("client gone")
    h._json = _boom_json

    with pytest.raises(BrokenPipeError):
        h._api_chat()
    assert throttle.spent == 0.0, "reservation released even when _serve_chat raises"


def test_api_chat_no_double_refund_when_success_json_raises(monkeypatch):
    # AGORA #384 r4: settle + release are co-located in _serve_chat's finally, so
    # if the SUCCESS-path _json raises AFTER record_cost (settled=True), the
    # reservation is NOT released -> no double-refund. spent stays the real cost.
    import pytest
    throttle = ChatThrottle(per_min=10_000, daily_budget=1000.0)
    monkeypatch.setattr(api_throttle, "_INSTANCE", throttle)
    conv = MagicMock()
    conv.send.return_value = types.SimpleNamespace(
        text="ok", model="m", provider=None, latency_ms=1, cost_usd=0.0,
        usage=types.SimpleNamespace(prompt_tokens=5, completion_tokens=7),
    )
    conv.history = []
    conv.cost.snapshot.return_value = types.SimpleNamespace(total_usd=0.0)
    h = _handler(conv)

    def _boom_json(data, code=200):
        raise BrokenPipeError("client gone")
    h._json = _boom_json

    with pytest.raises(BrokenPipeError):
        h._api_chat()
    assert throttle.spent == 12.0, "success path: cost recorded, NOT double-refunded"
