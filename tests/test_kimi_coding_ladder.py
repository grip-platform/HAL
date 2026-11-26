"""The Pool D rung: present in the GRIP ladder ONLY when the subscription resolves.

Council constraint (2026-09-07, seal ``e49c00c214c500be``): "define the Pool D
seam now but keep it entirely unarmed and absent from normal routing until the
dedicated subscription credential exists". So:

* ``GRIP_LADDER_ADDRESSES`` -- the tuple W0 left, the one ``grip_runtime`` /
  ``learned_lattice`` / HAPPI read -- does NOT contain a ``kimi-coding`` rung,
  and ``build_grip_ladder()`` with no argument is byte-identical to before;
* ``build_grip_ladder(subscriptions)`` splices ``kimi-coding/k3-256k`` in right
  after CCH when, and only when, a resolved ``kimi-coding`` subscription is
  among the ``subscriptions`` it is handed -- prepaid quota is tried before the
  metered ``moonshot/kimi-k3`` rung.

The behavioural test drives the UNMODIFIED ``RAILLMSentinel`` through the
registry: ``parse_model_address("kimi-coding/k3-256k")`` must split so the
bridged ``SubscriptionProvider`` receives the bare id ``k3-256k`` -- the id the
vendor knows -- rather than the whole address (which would 404 upstream) or,
worse, fall to the default provider. Hermetic: no network, no keychain.
"""
from __future__ import annotations

import pytest

from lib.hal.learned_lattice import _provider_for
from lib.hal.providers.openai_compat import OpenAICompatError
from lib.hal.raillm import ModelTier, RAILLMSentinel
from lib.hal.raillm_ladder import (
    GRIP_LADDER_ADDRESSES,
    KIMI_CODING_LADDER_ADDRESS,
    build_grip_ladder,
    ladder_addresses,
    ladder_cascade_order,
)
from lib.hal.raillm_subscription_bridge import SubscriptionProvider, register_subscriptions
from lib.hal.registry import ProviderRegistry, parse_model_address
from lib.hal.types import ChatResponse, TokenUsage

_ARMED = {"kimi-coding": object()}
_CCH = "anthropic/claude-opus-5"
_METERED_KIMI = "moonshot/kimi-k3"


class _Sub:
    """SubscriptionAuth double shaped like the real transports (fixed kwargs)."""

    def __init__(self, provider_id, *, reply=None, raise_exc=None):
        self.provider_id = provider_id
        self.name = f"{provider_id}-sub"
        self._reply, self._raise = reply, raise_exc
        self.calls: list[dict] = []

    def chat(self, prompt="", *, model, system=None, temperature=None,
             max_tokens=2048, messages=None, tools=None):
        self.calls.append({"prompt": prompt, "model": model})
        if self._raise is not None:
            raise self._raise
        return ChatResponse(text=self._reply, model=model,
                            provider=self.provider_id, usage=TokenUsage())

    def health_check(self):
        return True


# --- unarmed: byte-identical to W0 ------------------------------------------

def test_rung_address_is_the_256k_variant_and_not_in_the_static_ladder():
    # Vendor (kimi.com/code/docs/en/kimi-code/models): "k3 (1M) consumes about
    # twice as much quota as k3-256k. Within 256k context, it delivers the same
    # results" -- and the FRONTIER class window is 200_000, under 256k anyway.
    assert KIMI_CODING_LADDER_ADDRESS == "kimi-coding/k3-256k"
    assert KIMI_CODING_LADDER_ADDRESS not in GRIP_LADDER_ADDRESSES
    assert not any(a.startswith("kimi-coding/") for a in GRIP_LADDER_ADDRESSES)


@pytest.mark.parametrize("subs", [None, {}, {"anthropic": object()}, {"alibaba": object()}])
def test_unarmed_ladder_is_the_same_object_and_order(subs, monkeypatch):
    # Isolate from the suite-wide _permissive_spend_guard autouse fixture
    # (tests/conftest.py), which sets GRIP_CODEX_ALLOW_PAID=1 so OTHER tests'
    # made-up credentials are not refused -- it would otherwise splice in the
    # opt-in-gated Astra rung and break the "same object" identity this test
    # checks. This file is about the Kimi Code (Pool D) axis only; the Astra
    # axis is tests/test_astra_ladder.py's job.
    monkeypatch.delenv("GRIP_CODEX_ALLOW_PAID", raising=False)
    assert ladder_addresses(subs) is GRIP_LADDER_ADDRESSES
    assert ladder_cascade_order(subs) == list(GRIP_LADDER_ADDRESSES)
    assert build_grip_ladder(subs) == build_grip_ladder()


# --- armed: the rung appears right after CCH ----------------------------------

def test_armed_ladder_places_pool_d_after_cch_and_before_metered_kimi(monkeypatch):
    monkeypatch.delenv("GRIP_CODEX_ALLOW_PAID", raising=False)
    addresses = ladder_addresses(_ARMED)
    assert addresses[0] == _CCH
    assert addresses[1] == KIMI_CODING_LADDER_ADDRESS
    assert addresses.index(KIMI_CODING_LADDER_ADDRESS) < addresses.index(_METERED_KIMI)
    # Exactly one insertion; every other rung keeps its relative order.
    assert tuple(a for a in addresses if a != KIMI_CODING_LADDER_ADDRESS) == GRIP_LADDER_ADDRESSES
    assert ladder_cascade_order(_ARMED) == list(addresses)


def test_armed_tier_map_changes_only_the_frontier_members(monkeypatch):
    monkeypatch.delenv("GRIP_CODEX_ALLOW_PAID", raising=False)
    armed, unarmed = build_grip_ladder(_ARMED), build_grip_ladder()
    assert armed[ModelTier.FRONTIER].members == [
        _CCH, KIMI_CODING_LADDER_ADDRESS,
        "anthropic/claude-fable-5-1", _METERED_KIMI, "alibaba/qwen-max"]
    for tier in (ModelTier.FAST, ModelTier.FREE_TIER, ModelTier.LOCAL):
        assert armed[tier] == unarmed[tier]
    assert armed[ModelTier.FRONTIER].context_window == unarmed[ModelTier.FRONTIER].context_window


def test_stock_raillm_tries_the_pool_d_rung_second_when_armed(monkeypatch):
    monkeypatch.delenv("GRIP_CODEX_ALLOW_PAID", raising=False)
    sentinel = RAILLMSentinel(registry=ProviderRegistry(), equivalence_map=build_grip_ladder(_ARMED))
    chain = sentinel._build_tier_chain(_CCH, ModelTier.FRONTIER, None)
    # Pool D is tried second (right after CCH) and strictly before the
    # metered Kimi rung -- the Fable rung (HAL#610) sits between the two
    # even unopted (Astra does not, unopted; see test_astra_ladder.py).
    assert chain[:2] == [_CCH, KIMI_CODING_LADDER_ADDRESS]
    assert chain.index(KIMI_CODING_LADDER_ADDRESS) < chain.index(_METERED_KIMI)


# --- the address form reaches the bridged provider with the BARE model id ------

def test_parse_model_address_splits_the_kimi_coding_head():
    assert parse_model_address("kimi-coding/k3-256k") == ("kimi-coding", "k3-256k")
    assert parse_model_address("kimi-coding/kimi-for-coding-highspeed") == (
        "kimi-coding", "kimi-for-coding-highspeed")


def test_register_subscriptions_exposes_kimi_coding_in_the_registry():
    reg = ProviderRegistry()
    sub = _Sub("kimi-coding", reply="from kimi code")
    assert register_subscriptions(reg, {"kimi-coding": sub}) == ["kimi-coding"]
    provider = reg.get("kimi-coding")
    assert isinstance(provider, SubscriptionProvider)
    assert provider.chat(prompt="x", model="k3").text == "from kimi code"


def test_raillm_cascades_cch_to_kimi_coding_with_the_bare_model_id(tmp_path, monkeypatch):
    monkeypatch.setattr("lib.hal.sentinel._STATE_PATH", str(tmp_path / "limits.json"))
    reg = ProviderRegistry()
    cch = _Sub("anthropic", raise_exc=OpenAICompatError("cch weekly limit"))
    kimi_code = _Sub("kimi-coding", reply="answer from kimi code")
    register_subscriptions(reg, {"anthropic": cch, "kimi-coding": kimi_code})

    # The armed ladder, trimmed to the two rungs that are registered here so the
    # cascade cannot wander into unregistered metered/cheap/local rungs.
    emap = build_grip_ladder(_ARMED)
    for tier in (ModelTier.FAST, ModelTier.FREE_TIER, ModelTier.LOCAL):
        emap[tier].members.clear()
    del emap[ModelTier.FRONTIER].members[2:]

    sentinel = RAILLMSentinel(registry=reg, equivalence_map=emap)
    resp = sentinel.dispatch_with_history(_CCH, "what is 2+2?")

    assert resp.text == "answer from kimi code"
    assert cch.calls, "CCH was never attempted"
    assert kimi_code.calls[0]["model"] == "k3-256k"   # bare id, not the address
    assert sentinel.thread.messages[-1].provider_hint == "kimi-coding"


# --- the lattice knows the rung's true shape -----------------------------------

def test_lattice_signature_is_a_subscription_at_256k_with_vision():
    sig = _provider_for(KIMI_CODING_LADDER_ADDRESS)
    assert sig.tier == "subscription"          # quota, not per-token
    assert sig.context_window == 262_144       # k3-256k: "256k only"
    assert sig.modality == "text+vision"       # k3-256k: "Image only"
