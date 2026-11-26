"""Goodhart-proof tests for the GRIP ladder (H-RAILLM-2).

Asserts the operator's canonical CCH -> Kimi/Qwen -> cheap -> local order
both as data AND as the order the UNMODIFIED RAILLMSentinel will actually
try (behavioural integration -- fails if the live cascade diverges from
the documented ladder).

Every test here clears GRIP_CODEX_ALLOW_PAID explicitly: tests/conftest.py's
autouse _permissive_spend_guard fixture sets it to "1" for the whole suite
(so OTHER tests' made-up credentials are not refused), which would otherwise
silently splice the opt-in-gated Astra rung (lib/hal/raillm_ladder.py,
ASTRA_LADDER_ADDRESS) into this file's "default ladder" assertions. The
opted-in shape is tests/test_astra_ladder.py's job, not this file's.
"""
from lib.hal.raillm import ModelTier, RAILLMSentinel
from lib.hal.raillm_ladder import (
    GRIP_LADDER_ADDRESSES,
    build_grip_ladder,
    ladder_cascade_order,
)
from lib.hal.registry import ProviderRegistry

_EXPECTED = [
    "anthropic/claude-opus-5",
    "anthropic/claude-fable-5-1",
    "moonshot/kimi-k3",
    "alibaba/qwen-max",
    "groq/llama-3.3-70b-versatile",
    "gemini/gemini-3.5-flash-lite",
    "deepseek/deepseek-chat",
    "ollama/llama3.3:70b",
    "mlx_vlm/auto",
]


def test_canonical_addresses_exact_order(monkeypatch):
    monkeypatch.delenv("GRIP_CODEX_ALLOW_PAID", raising=False)
    assert list(GRIP_LADDER_ADDRESSES) == _EXPECTED
    assert GRIP_LADDER_ADDRESSES[0].startswith("anthropic/")  # CCH primary
    assert GRIP_LADDER_ADDRESSES[-1].split("/")[0] in ("ollama", "mlx_vlm")


def test_tiers_partition_subscription_cheap_local(monkeypatch):
    monkeypatch.delenv("GRIP_CODEX_ALLOW_PAID", raising=False)
    ladder = build_grip_ladder()
    fr = [a.split("/")[0] for a in ladder[ModelTier.FRONTIER].members]
    fast = [a.split("/")[0] for a in ladder[ModelTier.FAST].members]
    loc = [a.split("/")[0] for a in ladder[ModelTier.LOCAL].members]
    assert fr == ["anthropic", "anthropic", "moonshot", "alibaba"]  # CCH, Fable, metered Kimi, Qwen
    assert fast == ["groq", "gemini", "deepseek"]          # cheap/free API
    assert loc == ["ollama", "mlx_vlm"]                    # local
    # FREE_TIER duplicates the cheap pool (extra retry before LOCAL)
    assert ladder[ModelTier.FREE_TIER].members == ladder[ModelTier.FAST].members


def test_context_windows_monotonic_non_increasing():
    """Subscription window >= cheap >= local: a thread carried down the
    ladder is compressed to ever-smaller budgets, never expanded."""
    ladder = build_grip_ladder()
    sub_w = ladder[ModelTier.FRONTIER].context_window
    cheap_w = ladder[ModelTier.FAST].context_window
    local_w = ladder[ModelTier.LOCAL].context_window
    assert sub_w >= cheap_w >= local_w
    assert local_w > 0


def test_cascade_order_matches_documented_ladder(monkeypatch):
    monkeypatch.delenv("GRIP_CODEX_ALLOW_PAID", raising=False)
    assert ladder_cascade_order() == _EXPECTED


def test_stock_raillm_build_tier_chain_equals_ladder(monkeypatch):
    """The UNMODIFIED RAILLMSentinel, given build_grip_ladder(), produces
    a cascade chain identical to the documented ladder -- proving the
    operator's order is exactly what RAILLM will try (H-RAILLM-2)."""
    monkeypatch.delenv("GRIP_CODEX_ALLOW_PAID", raising=False)
    sentinel = RAILLMSentinel(
        registry=ProviderRegistry(), equivalence_map=build_grip_ladder()
    )
    chain = sentinel._build_tier_chain(
        "anthropic/claude-opus-5", ModelTier.FRONTIER, None
    )
    assert chain == _EXPECTED
    assert chain[0] == "anthropic/claude-opus-5"   # CCH tried first
    assert chain[-1] == "mlx_vlm/auto"               # local tried last
    # subscriptions strictly precede cheap which strictly precedes local
    assert chain.index("moonshot/kimi-k3") < chain.index("groq/llama-3.3-70b-versatile")
    assert chain.index("deepseek/deepseek-chat") < chain.index("ollama/llama3.3:70b")
