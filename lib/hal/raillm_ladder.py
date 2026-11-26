"""GRIP LLM ladder: the operator's canonical CCH -> Kimi -> cheap -> local order.

V>>'s intent (/ultrado 2026-05-15): GRIP should run through HAL with a fixed
fall-over ladder --

    1. Anthropic via CCH        (Claude Code subscription, already prepaid)
    2. Kimi K3 (Moonshot, metered API key), then Qwen subscription
    3. cheap / zero-cost API inference (groq, gemini, deepseek free tiers)
    4. local LLM                (ollama / mlx -- never leaves the machine)

``raillm.DEFAULT_EQUIVALENCE_MAP`` encodes a *generic* frontier->local
cascade (gemini/gpt-4o/deepseek). This module expresses the *operator's
specific* ladder as a SEPARATE equivalence map so ``raillm.py`` stays
untouched (the W1 composition story holds: zero core changes) and the
order is independently testable.

``RAILLMSentinel(registry, equivalence_map=build_grip_ladder())`` then
cascades exactly CCH -> Kimi -> Qwen -> cheap API -> local, resolving the
subscription rungs through the W1 ``register_subscriptions`` bridge.

Anthropic-CCH deliberately does NOT enter via ``config/providers.yaml``:
the registry only builds ``OpenAICompatProvider`` (/v1/chat/completions +
Bearer), but Anthropic speaks /v1/messages + x-api-key and CCH adds OAuth
signing -- a yaml entry would ship a broken provider. CCH enters as a
``SubscriptionProvider`` (W1) instead. Kimi/Qwen are OpenAI-compatible so
they could also be yaml providers, but routing them through the same
subscription bridge keeps one prepaid-quota code path (DRY). The Kimi rung
itself is METERED, not a subscription: keychain ``grip-moonshot`` is a
pay-per-token key with a cash balance on ``GET /v1/users/me/balance``
(``rules/adding-a-new-llm-to-hal.md`` section 0). It keeps the FRONTIER slot
because it is the 1M-context primary after CCH.
"""
from __future__ import annotations

from itertools import chain
from typing import Mapping, Optional

from lib.hal import spend_guard
from lib.hal.raillm_policy import ModelEquivalenceClass, ModelTier

# Canonical ladder addresses in operator-intended fall-over order. The
# tuple is the single source of truth the tier map is derived from, so a
# test can assert the exact order without re-encoding it (DRY).
#
# The Astra rung (openai/gpt-6-astra, HAL#610) is deliberately NOT a member
# here -- see ASTRA_LADDER_ADDRESS below for why, and ladder_addresses() for
# where it is spliced in.
GRIP_LADDER_ADDRESSES: tuple[str, ...] = (
    # 1. CCH subscription (primary). Was claude-opus-4-7; updated 2026-09-20 to
    #    the current generation, verified serving via `claude -p --model
    #    claude-opus-5`. NOTE: this rung does NOT currently serve through
    #    /api/infer — pinning it returns "pinned provider 'anthropic' did not
    #    serve this request". That is a dispatch gap in the RAILLM path, not a
    #    credential or model problem: the same models answer through the HAL
    #    GATEWAY on :4010.
    "anthropic/claude-opus-5",
    # 1c. Fable, same CCH subscription as the rung above. Currently INERT on
    #     this path — the gateway does not route `claude-*` ids (they fall
    #     through to the verbatim Anthropic relay and 401 on the placeholder
    #     token, by design) and the ladder's own anthropic dispatch is dead.
    #     Fable is reachable ONLY via `claude -p --model claude-fable-5-1`.
    #     Kept because it costs nothing measurable and states the intent for
    #     whoever wires a CCH dispatch path.
    "anthropic/claude-fable-5-1",
    "moonshot/kimi-k3",                   # 2a. Kimi K3, metered (kimi-k2.5 + moonshot-v1* retired 2026-08-31 -> HTTP 404)
    "alibaba/qwen-max",                   # 2b. Qwen subscription
    "groq/llama-3.3-70b-versatile",       # 3a. cheap/free API
    "gemini/gemini-3.5-flash-lite",        # 3b. cheap/free API (was gemini-2.5-flash, GRIP#9389)
    "deepseek/deepseek-chat",             # 3c. cheap/free API
    "ollama/llama3.3:70b",                # 4a. local (no data leaves host)
    "mlx_vlm/auto",                       # 4b. local fallback
)

# The Pool D rung -- Kimi Code, Moonshot's Kimi MEMBERSHIP coding service
# (provider ``kimi-coding``, host api.kimi.com/coding,
# ``lib/hal/subscription.py`` ``KimiCodingSubscription``). Deliberately NOT a
# member of GRIP_LADDER_ADDRESSES: the sovereign council of 2026-09-07 (seal
# ``e49c00c214c500be``, IDR leaf ``precog-1788805766-c58539b1``) ruled the seam
# "entirely unarmed and absent from normal routing until the dedicated
# subscription credential exists". :func:`ladder_addresses` splices it in
# directly after CCH -- prepaid quota is tried before the METERED
# ``moonshot/kimi-k3`` rung -- only when a resolved ``kimi-coding``
# subscription is handed in, the same presence gate
# ``models_command.annotate_availability`` applies to every subscription row.
# ``k3-256k`` rather than ``k3``: the vendor's model table says "k3 (1M)
# consumes about twice as much quota as k3-256k. Within 256k context, it
# delivers the same results" (kimi.com/code/docs/en/kimi-code/models), and the
# FRONTIER class window below is 200_000, inside 256k anyway.
KIMI_CODING_LADDER_ADDRESS = "kimi-coding/k3-256k"
_KIMI_CODING_PROVIDER = "kimi-coding"

# The Astra rung -- openai/gpt-6-astra (HAL#610). Deliberately NOT a member of
# GRIP_LADDER_ADDRESSES, for the same reason Pool D is spliced rather than
# static, but a sharper one: it is billed through OpenAICompatProvider against
# api.openai.com with a per-token API key, and since 2026-10-05 ("GPT runs on
# the ChatGPT subscription, never on credits") that path is refused by HAL's
# spend guard (lib/hal/spend_guard.py, enforced at the transport in
# lib/hal/providers/openai_compat.py since HAL#667) unless the operator has
# opted in with GRIP_CODEX_ALLOW_PAID=1. The refusal (SpendRefused) is NOT a
# transport error RAILLMSentinel.dispatch_with_history catches -- by design,
# the gateway's own failover treats a paid-key refusal as terminal, never a
# retry (lib/hal/gateway_failover.py). A static membership here (as a first
# version of this fix tried, then measured to abort the whole cascade past
# CCH whenever the opt-in is absent) would hand the cascade a rung that
# raises past every later fallback instead of skipping it. Spliced in only
# when the opt-in is live, exactly like Pool D's subscription-presence gate,
# so the unopted-in cascade never reaches it at all.
ASTRA_LADDER_ADDRESS = "openai/gpt-6-astra"

# Address heads that belong in the FRONTIER (prepaid-or-primary) class.
# "openai" classifies the Astra rung on the rare call where it IS spliced in
# (ladder_addresses(), below) -- it is otherwise never a GRIP_LADDER_ADDRESSES
# member, so this entry is a no-op on every unopted-in call.
_FRONTIER_HEADS = ("anthropic", _KIMI_CODING_PROVIDER, "moonshot", "alibaba", "openai")


def _astra_allowed() -> bool:
    """Is the operator opted in to per-token OpenAI billing right now?

    Reads the spend guard fresh on every call (never cached at import time),
    so a test's monkeypatched env var or policy takes effect immediately and
    a live toggle of GRIP_CODEX_ALLOW_PAID is honoured without a restart.

    Fail-closed, never raising: the guard loads its policy before it reads
    the opt-in, and a policy that is missing, unreadable or malformed refuses
    every paid source. So with such a policy the Astra rung stays out even
    when GRIP_CODEX_ALLOW_PAID=1, and the ladder is the static tuple
    (``test_an_unusable_policy_keeps_the_static_ladder_even_when_opted_in``).
    """
    return spend_guard.decide_paid(spend_guard.OPENAI_API_KEY).allowed


def ladder_addresses(
    subscriptions: Optional[Mapping[str, object]] = None,
) -> tuple[str, ...]:
    """The ladder for THIS process.

    With no resolved ``kimi-coding`` subscription and no live
    ``GRIP_CODEX_ALLOW_PAID`` opt-in, this is ``GRIP_LADDER_ADDRESSES`` itself
    -- the same tuple object, so every reader that pins identity
    (``grip_runtime._ladder_addresses``) is untouched. Otherwise a new tuple
    with :data:`KIMI_CODING_LADDER_ADDRESS` and/or :data:`ASTRA_LADDER_ADDRESS`
    spliced in right after CCH, in that order when both apply; every other
    rung keeps its relative order.
    """
    kimi_coding_armed = bool(subscriptions) and _KIMI_CODING_PROVIDER in subscriptions
    astra_armed = _astra_allowed()
    if not kimi_coding_armed and not astra_armed:
        return GRIP_LADDER_ADDRESSES
    cch, rest = GRIP_LADDER_ADDRESSES[0], GRIP_LADDER_ADDRESSES[1:]
    spliced = [cch]
    if kimi_coding_armed:
        spliced.append(KIMI_CODING_LADDER_ADDRESS)
    if astra_armed:
        spliced.append(ASTRA_LADDER_ADDRESS)
    return (*spliced, *rest)

# Context windows per tier (tokens). The FRONTIER tier carries the largest
# window because CCH/Kimi are the high-context primaries; local is smallest.
# ConversationThread.compress_if_needed uses these to keep a carried-over
# thread inside each rung's window (context-window decoupling).
#
# _SUBSCRIPTION_WINDOW stays at 200_000 although moonshot/kimi-k3 accepts
# 1_048_576 (config/model_catalog.yaml). The window is per EQUIVALENCE CLASS,
# not per rung (raillm.py: ``budget = self._equivalence_map[tier].context_window``),
# and the same FRONTIER class carries alibaba/qwen-max, whose lattice signature
# declares 128_000 (learned_lattice.py). Raising the class window to 1M would
# let a thread compressed for Kimi K3 overflow the Qwen rung it falls over to.
# Per-rung windows are a RAILLM change, outside the 2026-09-07 truth sweep;
# the cost of the conservative value is an under-used K3 window, never a 400.
_SUBSCRIPTION_WINDOW = 200_000
_CHEAP_API_WINDOW = 128_000
_LOCAL_WINDOW = 32_000


def build_grip_ladder(
    subscriptions: Optional[Mapping[str, object]] = None,
) -> dict[ModelTier, ModelEquivalenceClass]:
    """Return the operator's CCH -> Kimi/Qwen -> cheap -> local ladder.

    ``subscriptions`` is the resolved-subscription map (``provider_id`` ->
    transport, as ``auto_detect_subscriptions`` returns it). It decides
    whether the Pool D rung joins the FRONTIER class; the Astra rung's
    membership is decided independently, by :func:`ladder_addresses` reading
    the live ``GRIP_CODEX_ALLOW_PAID`` opt-in on every call (see
    :func:`ladder_addresses`). ``build_grip_ladder()`` with no argument and no
    opt-in is byte-identical to the pre-Pool-D, pre-Astra map.

    Maps onto RAILLM's four tiers without modifying RAILLM:
      FRONTIER  = primaries (CCH subscription, [Kimi Code subscription when
                  armed], [Astra when opted in], metered Kimi K3, then Qwen)
      FAST      = cheap/free API inference
      FREE_TIER = same cheap pool (RAILLM cascades FRONTIER->FAST->FREE_TIER
                  ->LOCAL; FREE_TIER duplicates FAST so a FAST-pool rate
                  limit still tries the pool again before dropping to LOCAL)
      LOCAL     = on-device models (data never leaves the host)
    """
    addresses = ladder_addresses(subscriptions)
    subs = [a for a in addresses if a.split("/", 1)[0] in _FRONTIER_HEADS]
    cheap = [a for a in addresses
             if a.split("/", 1)[0] in ("groq", "gemini", "deepseek")]
    local = [a for a in addresses
             if a.split("/", 1)[0] in ("ollama", "mlx_vlm")]
    return {
        ModelTier.FRONTIER: ModelEquivalenceClass(
            tier=ModelTier.FRONTIER, members=subs,
            context_window=_SUBSCRIPTION_WINDOW),
        ModelTier.FAST: ModelEquivalenceClass(
            tier=ModelTier.FAST, members=cheap,
            context_window=_CHEAP_API_WINDOW),
        ModelTier.FREE_TIER: ModelEquivalenceClass(
            tier=ModelTier.FREE_TIER, members=cheap,
            context_window=_CHEAP_API_WINDOW),
        ModelTier.LOCAL: ModelEquivalenceClass(
            tier=ModelTier.LOCAL, members=local,
            context_window=_LOCAL_WINDOW),
    }


def ladder_cascade_order(
    subscriptions: Optional[Mapping[str, object]] = None,
) -> list[str]:
    """Flatten the tier map into the address order RAILLM will actually try.

    Mirrors RAILLMSentinel._build_tier_chain's emission order
    (FRONTIER -> FAST -> FREE_TIER -> LOCAL) so a test can assert the live
    cascade equals the documented ladder. ``subscriptions`` as for
    :func:`build_grip_ladder`.
    """
    ladder = build_grip_ladder(subscriptions)
    tiers = (ModelTier.FRONTIER, ModelTier.FAST,
             ModelTier.FREE_TIER, ModelTier.LOCAL)
    # Flat single pass + O(1) set membership: same FRONTIER->FAST->
    # FREE_TIER->LOCAL emission order and first-wins dedup as before, but
    # O(n) instead of the prior O(n^2) ``addr not in order`` list scan
    # (the gate's correct hash-based-approach hint; semantics unchanged).
    seen: set[str] = set()
    order: list[str] = []
    for addr in chain.from_iterable(ladder[t].members for t in tiers):
        if addr not in seen:
            seen.add(addr)
            order.append(addr)
    return order


def pin_primary(
    ladder: dict[ModelTier, ModelEquivalenceClass],
    address: str,
) -> dict[ModelTier, ModelEquivalenceClass]:
    """Return a NEW tier map with ``address`` moved first in its tier's
    ``members`` (the rest of that tier kept in order as fallback; every
    other tier untouched). This is the P3 delta: "switch subscription/LLM"
    == "make rung X the primary" — identical to switching the LLM, with
    RAILLM's CCH->Kimi->cheap->local cascade (and HAL#329 context-loss-free
    thread carry) inherited unchanged beneath it.

    Invariants (Goodhart-anchored in tests/test_raillm_pin_primary.py):

    - **Pure / non-mutating.** The input ``ladder`` dict and every
      ``ModelEquivalenceClass`` in it are never mutated; a fresh dict and
      a fresh equivalence class (with a fresh ``members`` list) are
      returned for the matched tier. Unmatched tiers are reused by
      reference — sound because they are only read, never written.
    - **Total / fail-safe.** An ``address`` that is in no tier's members
      (unknown / typo / stale marker value) returns the ladder
      UNCHANGED — never raises, never drops a rung. The fallback ladder
      is preserved exactly, so a bad pin can only ever be a no-op, never
      a regression (H-LLMSW falsifier b: non-opted path byte-identical).
    - **Order-preserving.** Only the matched tier's internal order
      changes (``address`` first, remainder stable); the tier-key order
      and all other tiers' members are byte-identical to the input.
    """
    tier = _tier_of(ladder, address)
    if tier is None:
        return ladder  # unknown address -> unchanged (total, fail-safe)
    eqc = ladder[tier]
    pinned = ModelEquivalenceClass(
        tier=eqc.tier,
        members=_move_first(eqc.members, address),
        context_window=eqc.context_window,
    )
    return {**ladder, tier: pinned}


def _tier_of(
    ladder: dict[ModelTier, ModelEquivalenceClass],
    address: str,
) -> ModelTier | None:
    """The tier whose members contain ``address``, or ``None`` (unknown).
    Single pass over the (fixed, tiny) tier set."""
    for tier, eqc in ladder.items():
        if address in eqc.members:
            return tier
    return None


def _move_first(members: list[str], address: str) -> list[str]:
    """A NEW list with ``address`` first, the remaining members in their
    original relative order. Pure — the input list is never mutated."""
    return [address] + [m for m in members if m != address]
