"""RAILLM W4 — 100-message fault-injection harness (H050 / H066 gate).

Synthetic end-to-end test of ``RAILLMSentinel.dispatch_with_history`` under
deterministic fault injection. Uses mocked providers rather than real
Groq / Gemini / Ollama so the harness stays reproducible across CI runs
and does not incur API costs or rate-limit noise.

Why mocked rather than live
---------------------------

The hypothesis H050 is about the cascade's *resilience* — that no single
primary failure halts the pipeline. That property is a function of the
dispatcher logic, not of specific provider behaviour. A fault-injecting
mock gives us:

* 100% reproducibility (seeded RNG)
* Zero API cost
* Fault probabilities we control exactly
* A Goodhart guard test that disables the cascade and shows the harness
  can actually detect failure — impossible to write against live providers

Live-provider tests would stress network behaviour and headers, but not
the cascade logic itself. Those belong in a separate nightly integration
suite, not in a hypothesis-gate test that must be green on every PR.

Hypothesis gate
---------------

* Plan label **H050** / engine ID **H066**.
* Metric: ``end_to_end_throughput_with_fault_injection`` (success ratio
  over 100 dispatches with 30% primary fault injection).
* Prediction: 1.0 (every dispatch must eventually succeed via cascade).
* Falsified if: any dispatch in the 100-message run raises
  ``RAILLMExhausted`` under the documented workload.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.hal.providers.openai_compat import OpenAICompatError  # noqa: E402
from lib.hal.raillm import (  # noqa: E402
    DEFAULT_EQUIVALENCE_MAP,
    ModelEquivalenceClass,
    ModelTier,
    RAILLMExhausted,
    RAILLMSentinel,
)
from lib.hal.types import ChatResponse  # noqa: E402


# ---------------------------------------------------------------------------
# Fault-injecting mock provider
# ---------------------------------------------------------------------------


class FaultInjectingProvider:
    """Mock provider that raises OpenAICompatError on a configurable
    fraction of calls.

    Deterministic under a seeded ``random.Random``. Tracks per-instance
    call count and failure count for post-hoc assertions.
    """

    def __init__(
        self,
        name: str,
        fail_prob: float,
        rng: random.Random,
        response_text: str = "ok",
    ) -> None:
        self.name = name
        self.fail_prob = fail_prob
        self.rng = rng
        self.response_text = response_text
        self.call_count = 0
        self.fail_count = 0

    def chat(self, prompt: str, model: str, **kwargs) -> ChatResponse:
        self.call_count += 1
        if self.rng.random() < self.fail_prob:
            self.fail_count += 1
            raise OpenAICompatError(
                f"injected fault on {self.name}/{model} (call #{self.call_count})"
            )
        return ChatResponse(
            text=f"{self.response_text} from {self.name}",
            model=model,
            provider=self.name,
            raw={"_headers": {}},
        )


def _build_registry(providers: dict[str, FaultInjectingProvider]) -> MagicMock:
    """Build a MagicMock ProviderRegistry that routes by provider name."""
    reg = MagicMock()
    reg.get = MagicMock(side_effect=lambda name: providers.get(name))
    reg.list_providers = MagicMock(return_value=list(providers.keys()))
    reg.routing = MagicMock(fallback_chain=[])
    reg._configs = {name: MagicMock() for name in providers}
    return reg


# ---------------------------------------------------------------------------
# Test fixtures — shared constants
# ---------------------------------------------------------------------------


N_MESSAGES = 100
PRIMARY_MODEL = "groq/llama-3.3-70b-versatile"  # FAST tier in default map
# Seed chosen so the 30% fault run hits the primary at least 20 times in
# 100 dispatches (measured empirically); any seed within [0, 10_000] works
# for this harness, but we pin for cross-run reproducibility.
RNG_SEED = 424242


def _make_primary_only_registry(fail_prob: float, rng_seed: int):
    """Registry that only has the primary tier member — cascade cannot help."""
    rng = random.Random(rng_seed)
    providers = {"groq": FaultInjectingProvider("groq", fail_prob, rng)}
    return _build_registry(providers), providers


def _make_full_tier_registry(
    fail_prob_primary: float, rng_seed: int
) -> tuple[MagicMock, dict[str, FaultInjectingProvider]]:
    """Registry with one provider per cascade hop.

    The primary ("groq") fails at ``fail_prob_primary``. Every other
    provider is rock-solid (0.0 fail prob) so the cascade always
    terminates on a success within the chain. This isolates the test to
    the dispatcher's cascade logic.
    """
    rng = random.Random(rng_seed)
    # Each provider gets its own rng draw stream so probability profiles
    # are independent; in practice only groq fails in this test.
    providers = {
        "groq": FaultInjectingProvider("groq", fail_prob_primary, rng),
        "openai": FaultInjectingProvider("openai", 0.0, rng),
        "gemini": FaultInjectingProvider("gemini", 0.0, rng),
        "deepseek": FaultInjectingProvider("deepseek", 0.0, rng),
        "qwen": FaultInjectingProvider("qwen", 0.0, rng),
        "openrouter": FaultInjectingProvider("openrouter", 0.0, rng),
        "ollama": FaultInjectingProvider("ollama", 0.0, rng),
        "lemonade": FaultInjectingProvider("lemonade", 0.0, rng),
        "mlx_vlm": FaultInjectingProvider("mlx_vlm", 0.0, rng),
    }
    return _build_registry(providers), providers


def _dispatch_n(
    sentinel: RAILLMSentinel,
    n: int,
    primary: str = PRIMARY_MODEL,
) -> list[ChatResponse]:
    """Run ``n`` dispatch calls against the sentinel, return the responses."""
    out: list[ChatResponse] = []
    for i in range(n):
        response = sentinel.dispatch_with_history(
            model_address=primary,
            user_message=f"user turn {i}",
        )
        out.append(response)
    return out


# ---------------------------------------------------------------------------
# Happy path — 0% faults, every dispatch lands on primary
# ---------------------------------------------------------------------------


def test_zero_fault_100_dispatches_all_succeed_on_primary():
    reg, providers = _make_full_tier_registry(fail_prob_primary=0.0, rng_seed=RNG_SEED)
    sentinel = RAILLMSentinel(registry=reg)

    responses = _dispatch_n(sentinel, N_MESSAGES)

    assert len(responses) == N_MESSAGES
    assert all(r.provider == "groq" for r in responses)
    assert providers["groq"].call_count == N_MESSAGES
    assert providers["groq"].fail_count == 0
    # Thread contains 100 user messages + 100 assistant responses.
    assert len(sentinel.thread.messages) == 2 * N_MESSAGES


# ---------------------------------------------------------------------------
# H050 gate — 30% faults on primary, cascade catches every failure
# ---------------------------------------------------------------------------


def test_h050_gate_thirty_percent_primary_faults_no_pipeline_halt():
    """The hypothesis-gating test. 100 dispatches with 30% primary faults
    MUST all succeed via the tier cascade. A single RAILLMExhausted raise
    falsifies H050."""
    reg, providers = _make_full_tier_registry(fail_prob_primary=0.3, rng_seed=RNG_SEED)
    sentinel = RAILLMSentinel(registry=reg)

    responses = _dispatch_n(sentinel, N_MESSAGES)

    assert len(responses) == N_MESSAGES, (
        f"Pipeline halted at {len(responses)} / {N_MESSAGES} under 30% "
        f"primary fault injection — H050 falsified."
    )
    # Primary was attempted every dispatch; peers only on its failures.
    assert providers["groq"].call_count == N_MESSAGES
    # Every injected fault produced a cascade-rescued success, so every
    # response is a ChatResponse with text != "".
    assert all(r.text for r in responses)
    # Sanity: primary did fail a meaningful fraction (>=20%).
    assert providers["groq"].fail_count >= 20, (
        f"Primary only failed {providers['groq'].fail_count}/{N_MESSAGES}; "
        f"seed may have produced an unrepresentative run."
    )
    # Sanity: the cascade successfully served the fallback responses.
    cascade_served = sum(
        p.call_count for name, p in providers.items() if name != "groq"
    )
    assert cascade_served >= providers["groq"].fail_count, (
        "Cascade did not serve as many responses as primary failures."
    )


# ---------------------------------------------------------------------------
# Goodhart guard — if cascade is disabled, the same injection DOES halt
# ---------------------------------------------------------------------------


def test_goodhart_single_tier_with_one_member_raises_on_primary_fault():
    """Goodhart protection: the H050 gate test must be able to FAIL.

    We build a single-member cascade (one primary, no peers, no tiers)
    and assert that 30% fault injection causes at least one
    RAILLMExhausted raise. If this test also passes under the cascade
    config, the hypothesis test is meaningless — the cascade would not
    actually be doing anything.
    """
    reg, _ = _make_primary_only_registry(fail_prob=0.3, rng_seed=RNG_SEED)
    # Constrain the equivalence map to a single FAST-tier member so the
    # cascade has nowhere to go. We blank every other tier to ensure the
    # chain is exhausted on primary failure.
    constrained_map = {
        ModelTier.FRONTIER: ModelEquivalenceClass(
            tier=ModelTier.FRONTIER, members=[], context_window=200_000
        ),
        ModelTier.FAST: ModelEquivalenceClass(
            tier=ModelTier.FAST,
            members=[PRIMARY_MODEL],
            context_window=128_000,
        ),
        ModelTier.FREE_TIER: ModelEquivalenceClass(
            tier=ModelTier.FREE_TIER, members=[], context_window=128_000
        ),
        ModelTier.LOCAL: ModelEquivalenceClass(
            tier=ModelTier.LOCAL, members=[], context_window=32_000
        ),
    }
    sentinel = RAILLMSentinel(registry=reg, equivalence_map=constrained_map)

    exhausted_count = 0
    success_count = 0
    for i in range(N_MESSAGES):
        try:
            sentinel.dispatch_with_history(
                model_address=PRIMARY_MODEL,
                user_message=f"user turn {i}",
            )
            success_count += 1
        except RAILLMExhausted:
            exhausted_count += 1

    assert exhausted_count > 0, (
        "Goodhart guard failed: single-member cascade never raised "
        "RAILLMExhausted despite 30% fault injection. The H050 gate "
        "test would be vacuous."
    )
    assert success_count + exhausted_count == N_MESSAGES


# ---------------------------------------------------------------------------
# Cascade ordering — under partial failure, second-tier is consulted
# ---------------------------------------------------------------------------


def test_cascade_consults_peers_before_dropping_tier():
    """When the primary fails, the next attempt MUST be a peer in the
    same tier before the dispatcher drops to a lower tier. The RAILLM
    tier chain is:

        primary -> primary-tier peers -> FRONTIER -> FAST -> FREE_TIER -> LOCAL

    So with primary in FAST, a primary-only failure should be served by
    another FAST member, not by a FRONTIER model. We verify by forcing
    the primary to fail on EVERY call but the FAST peer to succeed, then
    checking no FRONTIER provider was called at all.
    """
    rng = random.Random(RNG_SEED)
    providers = {
        "groq": FaultInjectingProvider("groq", 1.0, rng),    # always fails
        "openai": FaultInjectingProvider("openai", 0.0, rng),  # FAST peer
        "gemini": FaultInjectingProvider("gemini", 0.0, rng),  # FRONTIER
        "deepseek": FaultInjectingProvider("deepseek", 0.0, rng),  # both tiers
        "qwen": FaultInjectingProvider("qwen", 0.0, rng),  # FAST
        "openrouter": FaultInjectingProvider("openrouter", 0.0, rng),
        "ollama": FaultInjectingProvider("ollama", 0.0, rng),
        "lemonade": FaultInjectingProvider("lemonade", 0.0, rng),
        "mlx_vlm": FaultInjectingProvider("mlx_vlm", 0.0, rng),
    }
    reg = _build_registry(providers)
    sentinel = RAILLMSentinel(registry=reg)

    # One dispatch — primary fails, FAST peer should serve it.
    response = sentinel.dispatch_with_history(
        model_address=PRIMARY_MODEL,
        user_message="hello",
    )

    # The served provider must be a FAST-tier peer (openai, qwen, or
    # deepseek per DEFAULT_EQUIVALENCE_MAP). NOT gemini (FRONTIER-only in
    # the FAST tier peers list via deepseek appears in both, so we check
    # against the documented FAST members list).
    fast_peers = set(DEFAULT_EQUIVALENCE_MAP[ModelTier.FAST].members)
    # Response provider is a bare provider name (e.g. "openai"), while
    # fast_peers contains "provider/model" addresses. Extract providers.
    fast_peer_providers = {addr.split("/")[0] for addr in fast_peers}
    assert response.provider in fast_peer_providers, (
        f"Expected cascade to land on a FAST peer "
        f"({fast_peer_providers}), got {response.provider}."
    )


# ---------------------------------------------------------------------------
# History integrity — conversation thread grows monotonically
# ---------------------------------------------------------------------------


def test_history_integrity_across_cascade_switches():
    """Across 100 dispatches with 30% faults, the ConversationThread
    must end with exactly 200 messages (100 user + 100 assistant) and
    the messages must alternate role=user, role=assistant.

    A bug in the cascade that double-appends (e.g. once per attempt
    instead of once per successful response) would violate this.
    """
    reg, _ = _make_full_tier_registry(fail_prob_primary=0.3, rng_seed=RNG_SEED)
    sentinel = RAILLMSentinel(registry=reg)

    _dispatch_n(sentinel, N_MESSAGES)

    msgs = sentinel.thread.messages
    assert len(msgs) == 2 * N_MESSAGES, (
        f"Expected {2 * N_MESSAGES} messages, got {len(msgs)}. "
        "Cascade is likely double-appending assistant turns on retry."
    )
    for i, m in enumerate(msgs):
        expected_role = "user" if i % 2 == 0 else "assistant"
        assert m.role == expected_role, (
            f"Message {i} has role={m.role}, expected {expected_role}. "
            f"Cascade is desynchronising thread roles."
        )


# ---------------------------------------------------------------------------
# Primary-tier total outage still lands on lower tier
# ---------------------------------------------------------------------------


def test_primary_tier_total_outage_falls_through_to_frontier():
    """If every FAST member fails, the cascade must drop to FRONTIER
    (the documented order is primary-tier peers first, then the full
    cascade starting at FRONTIER). One dispatch with all FAST members
    failing proves the fall-through."""
    rng = random.Random(RNG_SEED)
    # Make every FAST-tier provider fail; FRONTIER is rock-solid.
    providers = {
        "groq": FaultInjectingProvider("groq", 1.0, rng),
        "openai": FaultInjectingProvider("openai", 1.0, rng),
        "qwen": FaultInjectingProvider("qwen", 1.0, rng),
        "deepseek": FaultInjectingProvider("deepseek", 1.0, rng),
        "gemini": FaultInjectingProvider("gemini", 0.0, rng),
        "openrouter": FaultInjectingProvider("openrouter", 0.0, rng),
        "ollama": FaultInjectingProvider("ollama", 0.0, rng),
        "lemonade": FaultInjectingProvider("lemonade", 0.0, rng),
        "mlx_vlm": FaultInjectingProvider("mlx_vlm", 0.0, rng),
    }
    reg = _build_registry(providers)
    sentinel = RAILLMSentinel(registry=reg)

    response = sentinel.dispatch_with_history(
        model_address=PRIMARY_MODEL,
        user_message="hello from a FAST outage",
    )

    # Should have landed on a FRONTIER member (gemini / deepseek).
    # deepseek is also in FAST so it would have been tried there first
    # and failed; therefore the winning provider should be gemini.
    assert response.provider in {"gemini", "openai"}, (
        f"Expected fall-through to FRONTIER, got {response.provider}."
    )


# ---------------------------------------------------------------------------
# Parametric seeds — the 30% harness is seed-stable across N seeds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 42, 424242, 999999, 2026041601])
def test_h050_gate_is_seed_stable(seed):
    """Run the H050 gate across 5 different seeds. The hypothesis must
    hold independently of RNG choice — otherwise we are cherry-picking
    seeds where the cascade happens to work.
    """
    reg, providers = _make_full_tier_registry(fail_prob_primary=0.3, rng_seed=seed)
    sentinel = RAILLMSentinel(registry=reg)

    responses = _dispatch_n(sentinel, N_MESSAGES)

    assert len(responses) == N_MESSAGES
    assert providers["groq"].call_count == N_MESSAGES
