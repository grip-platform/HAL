"""Goodhart-proof tests for the subscription -> RAILLM bridge (H-RAILLM-1).

These verify BEHAVIOUR -- a real dispatch through the UNMODIFIED
``RAILLMSentinel`` -- not mock call counts. Each test fails if the
corresponding bridge guarantee is removed or broken (mutation mindset).
The cascade test isolates ``lib.hal.sentinel._STATE_PATH`` to a tmp file so
it neither reads stale real limit state nor mutates the real one.
"""
import pytest

from lib.hal.providers.openai_compat import OpenAICompatError
from lib.hal.raillm import (
    ModelEquivalenceClass,
    ModelTier,
    RAILLMSentinel,
)
from lib.hal.raillm_subscription_bridge import (
    SubscriptionProvider,
    register_subscriptions,
)
from lib.hal.registry import ProviderRegistry
from lib.hal.types import ChatResponse, TokenUsage


class FixedKwargSub:
    """SubscriptionAuth double shaped like CCHSubscription: fixed keyword
    parameters, NO ``**kwargs`` (so a foreign kwarg would TypeError)."""

    def __init__(self, provider_id, *, reply=None, raise_exc=None, healthy=True):
        self.provider_id = provider_id
        self.name = f"{provider_id}-sub"
        self._reply = reply
        self._raise = raise_exc
        self._healthy = healthy
        self.calls = []

    def chat(self, prompt="", *, model, system=None, temperature=None,
             max_tokens=2048, messages=None, tools=None):
        self.calls.append({"prompt": prompt, "model": model})
        if self._raise is not None:
            raise self._raise
        return ChatResponse(text=self._reply, model=model,
                            provider=self.provider_id, usage=TokenUsage())

    def health_check(self):
        return self._healthy


def test_adapter_satisfies_registry_and_raillm_contract():
    sub = FixedKwargSub("anthropic", reply="hi from cch")
    p = SubscriptionProvider(sub, name="anthropic")

    assert p.name == "anthropic"
    assert isinstance(p.base_url, str) and p.base_url
    r = p.chat(prompt="q", model="claude-opus-4-7")
    assert r.text == "hi from cch" and r.provider == "anthropic"
    assert p.health_check() is True
    assert p.list_models() == []


def test_unexpected_kwarg_filtered_not_crash():
    """RAILLM forwards json_mode=True; a CCH-shaped .chat would TypeError.
    The adapter must filter alien kwargs so the cascade does not crash."""
    sub = FixedKwargSub("anthropic", reply="ok")
    p = SubscriptionProvider(sub, name="anthropic")

    r = p.chat(prompt="q", model="m", json_mode=True, foo="bar")
    assert r.text == "ok"
    assert "json_mode" not in sub.calls[0]
    assert "foo" not in sub.calls[0]


def test_var_keyword_subscription_receives_all_kwargs():
    """A Kimi-shaped subscription (**kwargs) must receive everything --
    filtering only applies when the target has no VAR_KEYWORD."""

    class KimiShaped:
        provider_id = "moonshot"

        def __init__(self):
            self.seen = None

        def chat(self, prompt="", **kwargs):
            self.seen = kwargs
            return ChatResponse(text="kimi", model=kwargs.get("model", "?"),
                                provider="moonshot", usage=TokenUsage())

        def health_check(self):
            return True

    k = KimiShaped()
    p = SubscriptionProvider(k, name="moonshot")
    p.chat(prompt="q", model="kimi-k2", json_mode=True)
    assert k.seen.get("json_mode") is True
    assert k.seen.get("model") == "kimi-k2"


def test_arbitrary_failure_normalised_to_openaicompat_error():
    """A subscription raising a non-RAILLM-handled error must become
    OpenAICompatError so the InfiniteMachine fails over, not crashes."""
    sub = FixedKwargSub("anthropic", raise_exc=RuntimeError("CCH auth 401"))
    p = SubscriptionProvider(sub, name="anthropic")

    with pytest.raises(OpenAICompatError) as ei:
        p.chat(prompt="q", model="m")
    assert "CCH auth 401" in str(ei.value)
    assert isinstance(ei.value.__cause__, RuntimeError)


def test_handled_errors_propagate_unchanged():
    """Errors RAILLM already catches must NOT be re-wrapped."""
    sub = FixedKwargSub("anthropic", raise_exc=TimeoutError("slow"))
    p = SubscriptionProvider(sub, name="anthropic")

    with pytest.raises(TimeoutError):
        p.chat(prompt="q", model="m")


def test_health_probe_never_raises():
    class Boom:
        def chat(self, prompt="", *, model):  # pragma: no cover
            ...

        def health_check(self):
            raise RuntimeError("keychain locked")

    p = SubscriptionProvider(Boom(), name="anthropic")
    assert p.health_check() is False


def test_register_subscriptions_populates_registry():
    reg = ProviderRegistry()
    subs = {
        "anthropic": FixedKwargSub("anthropic", reply="a"),
        "moonshot": FixedKwargSub("moonshot", reply="m"),
    }
    names = register_subscriptions(reg, subs)

    assert set(names) == {"anthropic", "moonshot"}
    assert isinstance(reg.get("anthropic"), SubscriptionProvider)
    assert reg.get("moonshot").chat(prompt="x", model="k").text == "m"


def test_raillm_cascades_cch_to_kimi_with_history_unmodified_core(
    tmp_path, monkeypatch,
):
    """H-RAILLM-1 load-bearing: the STOCK RAILLMSentinel cascades from a
    failing CCH-Anthropic subscription to a working Kimi subscription with
    conversation history intact, using ONLY registered SubscriptionProviders
    and ZERO changes to raillm.py."""
    monkeypatch.setattr(
        "lib.hal.sentinel._STATE_PATH", str(tmp_path / "limits.json")
    )

    reg = ProviderRegistry()
    cch = FixedKwargSub("anthropic", raise_exc=OpenAICompatError("cch limit"))
    kimi = FixedKwargSub("moonshot", reply="answer from kimi")
    register_subscriptions(reg, {"anthropic": cch, "moonshot": kimi})

    emap = {
        ModelTier.FRONTIER: ModelEquivalenceClass(
            tier=ModelTier.FRONTIER,
            members=["anthropic/claude-opus-4-7", "moonshot/kimi-k2"],
            context_window=200_000,
        ),
        ModelTier.FAST: ModelEquivalenceClass(ModelTier.FAST, [], 128_000),
        ModelTier.FREE_TIER: ModelEquivalenceClass(
            ModelTier.FREE_TIER, [], 128_000
        ),
        ModelTier.LOCAL: ModelEquivalenceClass(ModelTier.LOCAL, [], 32_000),
    }

    sentinel = RAILLMSentinel(registry=reg, equivalence_map=emap)
    resp = sentinel.dispatch_with_history(
        "anthropic/claude-opus-4-7", "what is 2+2?"
    )

    assert resp.text == "answer from kimi"
    assert cch.calls, "CCH-Anthropic was never attempted"
    assert kimi.calls, "Kimi was never reached after CCH failure"

    msgs = sentinel.thread.messages
    assert msgs[0].role == "user" and "2+2" in msgs[0].content
    assert msgs[-1].role == "assistant"
    assert msgs[-1].content == "answer from kimi"
    assert msgs[-1].provider_hint == "moonshot"
