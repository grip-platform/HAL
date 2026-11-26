"""HAL model→provider routing chokepoint (Phase 0 of the mesh hub).

The ONE source of truth for "which upstream serves this model id" — the DRY
chokepoint every gateway entry path calls so the decision is never duplicated or
divergent (correctness gate 3 of plans/hal-bidirectional-mesh-hub.md). Pure and
I/O-free: no network, no file reads, no gateway surgery — just the decision, so
it is exhaustively testable and safe to land BEFORE the gateway is wired to it.

#3274 INVARIANT (load-bearing): every ``claude-*`` model resolves to a Route
with ``verbatim=True`` (provider "anthropic"). That selects the byte-identical
relay path that preserves Anthropic thinking-block signatures (GRIP #3274). This
function is the single place that guarantees it; ``tests/test_model_route.py``
asserts it two-sided — every claude-* is verbatim AND no non-claude is verbatim
— so a future edit can never silently route Claude through a deserialising
translator. An explicit ``anthropic/`` address is verbatim too: the invariant
that matters is that no NON-Anthropic provider is ever verbatim, since
over-routing to the native relay is safe and under-routing is not.

ADDRESS FORM: a model may be written ``<provider>/<model>`` — HAL's canonical
form in ``config/model_catalog.yaml`` and ``docs/provider-routing.md``. An
explicit prefix beats the family table (``groq/openai/gpt-oss-120b`` is Groq's,
not OpenAI's, despite the ``gpt-`` inside it). :func:`strip_provider_prefix`
removes the prefix for the upstream call, since the provider does not know it.

Provider names match ``config/providers.yaml`` (``openai``, ``gemini``, ``xai``,
``deepseek``, ``qwen``, ``groq``) so the gateway can look the provider config up.
"anthropic" is the gateway's native verbatim upstream, not a providers.yaml
cost-provider. Routing is by model FAMILY (prefix), which is version-stable
(``gpt-5.5`` and ``gpt-4o`` both → ``openai``); exact-model lists in
providers.yaml are the registry/cost layer's concern, not gateway family routing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

ANTHROPIC = "anthropic"

# Provider names this chokepoint can emit (aligned with config/providers.yaml +
# the gateway's native "anthropic" verbatim upstream). The meta-test asserts the
# family table only ever yields a name in this set — and derives its inputs FROM
# the family table, so a name added there without being added here fails.
# ("moonshot" was emitted by the table but missing from this set: the old
# meta-test iterated a hand-written model list that happened to contain no
# moonshot/kimi id, so the contract it claimed to enforce could not fail.)
KNOWN_PROVIDERS = frozenset({
    ANTHROPIC, "openai", "gemini", "xai", "deepseek", "qwen", "groq", "mistral",
    "moonshot", "bitnet", "zhipu", "openrouter",
    # Kimi Code membership (Pool D) -- joined together with its gateway_egress
    # endpoint row (tests/test_gateway_unroutable_provider.py binds the two).
    "kimi-coding",
    # Alibaba Token Plan Personal Edition -- Claude Code W4's first fallback
    # target (config/w4_cascade.json anthropic_targets, lib/hal/wire_cascade.py).
    # Already a declared providers.yaml provider and already wired into HAPPI
    # (happi/providers/openai_compat.py) and tool_cli_integration.py; only this
    # gateway lane was missing, joined together with its gateway_egress
    # endpoint row below (same pairing test_gateway_unroutable_provider.py
    # binds for kimi-coding). A bare model id (e.g. "qwen3.8-max") is NOT
    # routed here -- no _FAMILY_ROUTES entry names this provider, so only the
    # explicit "qwen-token-plan/<id>" address claims it, via the
    # explicit-prefix branch below.
    "qwen-token-plan",
})

# Ordered (model-family prefix → provider name). First match wins; the lowercased
# model id is matched. `claude-` is handled BEFORE this table (it is the only
# verbatim route). Families are version-stable, so unreleased ids route too.
_FAMILY_ROUTES: Tuple[Tuple[str, str], ...] = (
    ("gpt-", "openai"),
    ("chatgpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("gemini-", "gemini"),
    ("gemma-", "gemini"),
    ("grok-", "xai"),
    ("deepseek", "deepseek"),
    ("qwen", "qwen"),
    ("llama", "groq"),
    ("mistral", "mistral"),
    ("mixtral", "mistral"),
    ("moonshot", "moonshot"),
    ("kimi", "moonshot"),
    ("bitnet", "bitnet"),  # local on-device: Microsoft BitNet b1.58 via llama-server
    ("glm", "zhipu"),
    ("zhipu", "zhipu"),
)

# Kimi Code — Moonshot's Kimi MEMBERSHIP coding service (provider
# ``kimi-coding``, host api.kimi.com/coding, ``lib/hal/subscription.py``
# ``KimiCodingSubscription``, GRIP Pool D). Its four ids carry no family marker
# the pay-per-token host knows (Moonshot's metered ids are ``kimi-k3`` /
# ``kimi-k2.*``), yet two of them start with ``kimi`` — and so does the
# ``kimi-coding/<id>`` address form — so the family table would send BOTH the
# bare ``kimi-for-coding`` and ``kimi-coding/k3`` to the metered ``moonshot``
# upstream: wrong product, wrong host, and because ``strip_provider_prefix`` is
# provider-scoped the ``kimi-coding/`` prefix would arrive unstripped (measured
# 2026-09-07, before ``_kimi_coding_guard``: all three address forms routed to
# moonshot with the prefix on). ``kimi-coding`` is now in KNOWN_PROVIDERS (it
# landed together with its gateway_egress endpoint, HAL#550 rows A+B), so the
# ADDRESS form ``kimi-coding/<id>`` is claimed by the explicit-prefix branch
# before this guard runs and routes to the membership host. The BARE ids stay
# UNKNOWN here, never moonshot: without the address form they would hit the
# ``kimi`` family row and the pay-per-token host. The provider is still
# unarmed until keychain grip-kimi-coding exists (sovereign council
# 2026-09-07, seal e49c00c214c500be).
KIMI_CODING = "kimi-coding"
_KIMI_CODING_ONLY_IDS = frozenset({
    "k3", "k3-256k", "kimi-for-coding", "kimi-for-coding-highspeed",
})


def _split_provider_head(model: Optional[str]) -> Tuple[str, str]:
    """Split ``model`` into ``(lowercased_head, tail)`` at the first ``/``.

    ``("", model)`` when there is no separator; ``(head, "")`` when the tail is
    empty (a bare ``groq/`` is not an address). Only the HEAD is case-folded —
    the tail keeps its original case (model ids are case-sensitive upstream).
    The single chokepoint for the partition rule so
    :func:`strip_provider_prefix` and :func:`_explicit_provider_route` can never
    drift on the "bare prefix" / "no tail" edge cases (AGORA #534 C2).
    """
    text = model if isinstance(model, str) else ""
    head, sep, tail = text.partition("/")
    if not sep:
        return "", text
    return head.strip().lower(), tail


def strip_provider_prefix(model: Optional[str], provider: str) -> str:
    """``model`` with a leading ``<provider>/`` removed, for the upstream call.

    HAL addresses models as ``<provider>/<model>`` (``config/model_catalog.yaml``,
    ``docs/provider-routing.md`` "Fixed" strategy). The provider itself does not
    know that prefix — Groq publishes ``openai/gpt-oss-120b``, so a request
    carrying ``groq/openai/gpt-oss-120b`` 404s. This removes exactly one leading
    segment, and ONLY when it equals ``provider``.

    Provider-scoped by design, not "strip any known-provider head": a model id
    may legitimately begin with another provider's name (Groq serves
    ``openai/gpt-oss-120b``; ``gateway_fanout`` sends that bare id on a ``groq``
    rung). Matching only the forwarding provider leaves such ids untouched.

    Case-insensitive on the prefix; the remainder keeps its original case. A
    bare ``"groq/"`` (no model part) is not an address and passes through
    unchanged, as does any id without a matching head.
    """
    text = model if isinstance(model, str) else ""
    head, tail = _split_provider_head(text)
    if not tail.strip() or head != provider.strip().lower():
        return text
    return tail


@dataclass(frozen=True)
class Route:
    """A routing decision for one model id."""
    provider: str       # providers.yaml name, "anthropic", or "" if unknown
    verbatim: bool       # True only for Anthropic (the #3274 byte-identical
                         # relay) — both the claude-* family and an explicit
                         # ``anthropic/`` address.
    reason: str

    @property
    def known(self) -> bool:
        """Whether a provider was resolved (False ⇒ caller must error, not guess)."""
        return bool(self.provider)


def _explicit_provider_route(key: str) -> Optional[Route]:
    """The route named by an explicit ``<provider>/`` prefix, or ``None``.

    HAL's canonical address form is ``<provider>/<model>``
    (``config/model_catalog.yaml``; ``docs/provider-routing.md`` "Fixed"
    strategy). An operator who writes ``groq/openai/gpt-oss-120b`` has STATED the
    provider, so that beats the family table, which would otherwise read the
    ``gpt-`` inside it as OpenAI's.

    Only the head segment is consulted, and only when it names a provider this
    chokepoint knows — ``notaprovider/x`` stays unknown rather than being
    invented into a route, and a bare ``groq/`` (no model part) is not an
    address. An ``anthropic/`` head is verbatim: #3274 safety is asymmetric, and
    an explicit Anthropic address is exactly the native relay.
    """
    head, tail = _split_provider_head(key)
    if head not in KNOWN_PROVIDERS or not tail.strip():
        return None
    return Route(provider=head, verbatim=head == ANTHROPIC,
                 reason=f"explicit '{head}/' provider prefix → {head}")


def _kimi_coding_guard(key: str, model: Optional[str]) -> Optional[Route]:
    """UNKNOWN (never moonshot) for a BARE Kimi Code id.

    Runs after the explicit-prefix branch — which now claims the
    ``kimi-coding/<id>`` address form, ``kimi-coding`` being in
    ``KNOWN_PROVIDERS`` — and BEFORE the family table, whose ``kimi`` row would
    otherwise take ``kimi-for-coding`` to the pay-per-token host. The bare id
    is told the mandatory address form. (The address-form branch this function
    used to carry became unreachable when the provider joined KNOWN_PROVIDERS
    and was removed rather than left as dead code.) See ``KIMI_CODING``.
    """
    if key in _KIMI_CODING_ONLY_IDS:
        return Route(provider="", verbatim=False,
                     reason=(f"{model!r} is a Kimi Code subscription id — address it "
                             f"as {KIMI_CODING}/{key}; the bare id would misroute to "
                             f"the pay-per-token moonshot host"))
    return None


def route_model(model: Optional[str]) -> Route:
    """Resolve a model id to its upstream provider.

    An explicit ``<provider>/`` prefix wins outright. Otherwise: ``claude-*`` →
    verbatim Anthropic relay (#3274); other known families → their providers.yaml
    provider; unknown / empty → ``known is False`` (the caller must surface a
    clear error, never silently misroute to the default upstream).

    The upstream call must address the model as the PROVIDER knows it, without
    the prefix — see :func:`strip_provider_prefix`, which ``gateway_egress``
    applies to the outbound body.
    """
    key = (model or "").strip().lower()
    if not key:
        return Route(provider="", verbatim=False, reason="empty model id")
    explicit = _explicit_provider_route(key)
    if explicit is not None:
        return explicit
    if key.startswith("claude"):
        # Match the whole claude family (claude-*, and defensively any "claude…"
        # id such as a hypothetical "claude3") so a Claude model can NEVER miss
        # the verbatim path. #3274 safety is asymmetric: over-routing to verbatim
        # Anthropic is safe; under-routing (a Claude model translated) is not.
        return Route(provider=ANTHROPIC, verbatim=True,
                     reason="claude family → verbatim Anthropic relay (#3274)")
    guarded = _kimi_coding_guard(key, model)
    if guarded is not None:
        return guarded
    for prefix, provider in _FAMILY_ROUTES:
        if key.startswith(prefix):
            return Route(provider=provider, verbatim=False,
                         reason=f"'{prefix}' family → {provider}")
    return Route(provider="", verbatim=False, reason=f"no family match for {model!r}")
