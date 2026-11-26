"""The Astra rung: present in the GRIP ladder ONLY when per-token OpenAI
billing is opted in (HAL#687 rework of HAL#683/HAL#610).

Origin: a review of the first version of this fix (commit 2edaa85d) measured
that adding "openai" to raillm_ladder.py's _FRONTIER_HEADS, with
openai/gpt-6-astra left as a STATIC member of GRIP_LADDER_ADDRESSES, made
RAILLMSentinel.dispatch_with_history ABORT past a failing CCH rung. The
cause: lib/hal/providers/openai_compat.py calls
lib.hal.spend_guard.require_openai_destination before any socket opens;
since 2026-10-05 ("GPT runs on the ChatGPT subscription, never on credits",
enforced for HAL in HAL#667) that refuses per-token OpenAI billing unless
GRIP_CODEX_ALLOW_PAID=1, raising SpendRefused -- a plain Exception subclass
RAILLMSentinel.dispatch_with_history does not catch (it only catches
OpenAICompatError and (ConnectionError, TimeoutError, OSError)). The
gateway's own failover (lib/hal/gateway_failover.py) treats a paid-key
refusal as terminal by design, so the fix is not to catch SpendRefused in
RAILLM -- it is to keep the cascade from ever reaching the Astra rung at all
when the opt-in is absent, the same shape already used for the Kimi Code
(Pool D) rung.

So: GRIP_LADDER_ADDRESSES never contains openai/gpt-6-astra.
``ladder_addresses()`` splices in ``ASTRA_LADDER_ADDRESS`` right after CCH
(after the Kimi Code rung when that is also armed) only when
``spend_guard.decide_paid(spend_guard.OPENAI_API_KEY).allowed`` is true, read
fresh on every call. Hermetic: no network, no keychain.
"""
from __future__ import annotations

import socket

from lib.hal.providers.openai_compat import OpenAICompatError, OpenAICompatProvider
from lib.hal.raillm import ModelTier, RAILLMSentinel
from lib.hal.raillm_ladder import (
    ASTRA_LADDER_ADDRESS,
    GRIP_LADDER_ADDRESSES,
    KIMI_CODING_LADDER_ADDRESS,
    build_grip_ladder,
    ladder_addresses,
    ladder_cascade_order,
)
from lib.hal.registry import ProviderRegistry
from lib.hal.types import ChatResponse, TokenUsage

_CCH = "anthropic/claude-opus-5"
_OPENAI_PAID_ENV = "GRIP_CODEX_ALLOW_PAID"


class _Fails:
    """A provider double whose chat() always raises, like a dead CCH rung."""

    def __init__(self, name: str):
        self.name = name

    def chat(self, prompt="", *, model, **kwargs):
        raise OpenAICompatError(f"{self.name}: simulated failure")


class _Serves:
    """A provider double that always answers -- the rung that should serve."""

    def __init__(self, name: str, reply: str):
        self.name = name
        self._reply = reply

    def chat(self, prompt="", *, model, **kwargs):
        return ChatResponse(text=self._reply, model=model,
                            provider=self.name, usage=TokenUsage())


# --- unarmed: byte-identical to before the Astra rung existed ---------------

def test_unopted_ladder_has_no_astra_rung_and_is_the_same_object(monkeypatch):
    monkeypatch.delenv(_OPENAI_PAID_ENV, raising=False)
    assert ASTRA_LADDER_ADDRESS not in GRIP_LADDER_ADDRESSES
    assert ladder_addresses() is GRIP_LADDER_ADDRESSES
    assert ASTRA_LADDER_ADDRESS not in ladder_cascade_order()
    ladder = build_grip_ladder()
    assert ASTRA_LADDER_ADDRESS not in ladder[ModelTier.FRONTIER].members


# --- armed: the rung appears right after CCH when opted in ------------------

def test_opted_in_ladder_splices_astra_right_after_cch(monkeypatch):
    monkeypatch.setenv(_OPENAI_PAID_ENV, "1")
    addresses = ladder_addresses()
    assert addresses[0] == _CCH
    assert addresses[1] == ASTRA_LADDER_ADDRESS
    # Exactly one insertion; every other rung keeps its relative order.
    assert tuple(a for a in addresses if a != ASTRA_LADDER_ADDRESS) == GRIP_LADDER_ADDRESSES
    ladder = build_grip_ladder()
    assert ladder[ModelTier.FRONTIER].members[:2] == [_CCH, ASTRA_LADDER_ADDRESS]
    assert ladder_cascade_order()[:2] == [_CCH, ASTRA_LADDER_ADDRESS]


def test_opted_in_and_kimi_coding_armed_orders_kimi_coding_before_astra(monkeypatch):
    monkeypatch.setenv(_OPENAI_PAID_ENV, "1")
    addresses = ladder_addresses({"kimi-coding": object()})
    assert addresses[:3] == (_CCH, KIMI_CODING_LADDER_ADDRESS, ASTRA_LADDER_ADDRESS)


def test_an_unusable_policy_keeps_the_static_ladder_even_when_opted_in(tmp_path, monkeypatch):
    """AGORA's falsifier on HAL#687: a spend-guard policy that is missing or
    malformed refuses every paid source (decide_paid is fail-closed and never
    raises), so even with the opt-in set the ladder neither crashes nor
    splices Astra."""
    from lib.hal import spend_guard
    monkeypatch.setenv(_OPENAI_PAID_ENV, "1")
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not json", encoding="utf-8")
    for policy in (tmp_path / "missing.json", malformed):
        monkeypatch.setenv(spend_guard.CONFIG_ENV, str(policy))
        assert ladder_addresses() is GRIP_LADDER_ADDRESSES


def test_decide_paid_is_read_fresh_not_cached(monkeypatch):
    """Flipping the env var between two calls in the same process changes the
    answer immediately -- the opt-in is read at call time, never cached."""
    monkeypatch.delenv(_OPENAI_PAID_ENV, raising=False)
    assert ASTRA_LADDER_ADDRESS not in ladder_addresses()
    monkeypatch.setenv(_OPENAI_PAID_ENV, "1")
    assert ASTRA_LADDER_ADDRESS in ladder_addresses()
    monkeypatch.delenv(_OPENAI_PAID_ENV, raising=False)
    assert ASTRA_LADDER_ADDRESS not in ladder_addresses()


# --- the regression probe: CCH fails, Astra is unopted, nothing aborts ------

def test_dispatch_falls_through_past_cch_without_an_openai_opt_in(tmp_path, monkeypatch):
    """The exact review probe (HAL#687): CCH fails, a REAL OpenAICompatProvider
    is registered under "openai" and pointed at api.openai.com with a fake
    key, GRIP_CODEX_ALLOW_PAID is unset, and sockets are patched to raise.
    Dispatch must be served by the next rung and must never raise
    SpendRefused -- it must not even reach the openai provider at all, since
    the opt-in is absent and the ladder never splices Astra in.
    """
    monkeypatch.delenv(_OPENAI_PAID_ENV, raising=False)
    monkeypatch.setattr("lib.hal.sentinel._STATE_PATH", str(tmp_path / "limits.json"))

    def _no_sockets(*args, **kwargs):
        raise AssertionError(
            "a real socket was opened -- dispatch should never reach the network")
    monkeypatch.setattr(socket, "create_connection", _no_sockets)

    reg = ProviderRegistry()
    reg.register(_Fails("anthropic"))
    reg.register(OpenAICompatProvider(
        "openai", "https://api.openai.com/v1", api_key="fake-key-not-a-token"))
    reg.register(_Serves("moonshot", "served by the next rung"))

    sentinel = RAILLMSentinel(registry=reg, equivalence_map=build_grip_ladder())
    resp = sentinel.dispatch_with_history(_CCH, "what is 2+2?")

    assert resp.text == "served by the next rung"
