"""Bridge: expose SubscriptionAuth transports as RAILLM-resolvable providers.

``RAILLMSentinel.dispatch_with_history`` (``lib/hal/raillm.py``) resolves
providers via ``registry.get(name)`` and calls
``provider.chat(prompt=, model=, **kwargs) -> ChatResponse``, catching
``OpenAICompatError`` plus ``(ConnectionError, TimeoutError, OSError)`` to
trigger failover. The ``SubscriptionAuth`` transports (``CCHSubscription``,
``KimiCodingSubscription``, ``QwenSubscription`` in ``lib/hal/subscription.py``)
already expose a compatible ``.chat()`` -- but they are never placed in the
``ProviderRegistry`` (``register()`` only builds ``OpenAICompatProvider``
from ``providers.yaml``). RAILLM therefore physically cannot cascade through
a Claude-Code-subscription (CCH) or Kimi Code subscription today.

This module is the missing connective tissue. Pure composition -- zero
changes to ``RAILLMSentinel`` or ``ProviderRegistry`` core (hypothesis
H-RAILLM-1: ``raillm_core_dispatch_lines_changed == 0``).

Two non-obvious correctness guarantees the adapter provides -- without them
the InfiniteMachine aborts instead of cascading:

1. **Keyword filtering.** RAILLM forwards ``**kwargs`` (e.g.
   ``json_mode=True``) that ``CCHSubscription.chat`` (fixed kwargs, no
   ``**kwargs``) rejects with ``TypeError`` -- which RAILLM does NOT catch,
   crashing the cascade. The adapter introspects the wrapped ``.chat``
   signature and forwards only the keywords it accepts.

2. **Failure normalisation.** RAILLM only fails over on
   ``OpenAICompatError`` / ``(ConnectionError, TimeoutError, OSError)``.
   A subscription auth/transport failure raising anything else would abort
   the cascade. The adapter re-raises non-handled exceptions as
   ``OpenAICompatError`` so the next rung is tried.
"""
from __future__ import annotations

import inspect
from typing import Optional

from lib.hal.providers.openai_compat import OpenAICompatError
from lib.hal.subscription import SubscriptionAuth, auto_detect_subscriptions
from lib.hal.types import ChatResponse

# Exceptions RAILLMSentinel.dispatch_with_history already handles as
# "fail over to the next rung". Anything outside this set must be
# normalised to OpenAICompatError or the InfiniteMachine crashes.
_RAILLM_HANDLED = (OpenAICompatError, ConnectionError, TimeoutError, OSError)


class SubscriptionProvider:
    """Adapts a ``SubscriptionAuth`` to the registry/RAILLM provider contract.

    Satisfies exactly the surface ``ProviderRegistry`` + ``RAILLMSentinel``
    require and nothing more (ISP): ``name``, ``base_url``,
    ``chat(prompt, *, model, **kwargs) -> ChatResponse``,
    ``health_check() -> bool``, ``list_models() -> list``.
    """

    def __init__(self, subscription: SubscriptionAuth, name: str) -> None:
        self._sub = subscription
        self.name = name
        # ProviderRegistry.list_providers reads provider.base_url; a
        # subscription has no REST base, so advertise a stable sentinel
        # URL rather than let list_providers raise AttributeError.
        self.base_url = getattr(subscription, "base_url", "") or f"subscription://{name}"

    def _accepted_kwargs(self, kwargs: dict) -> dict:
        """Forward only kwargs the wrapped ``.chat`` actually accepts.

        If the target declares ``**kwargs`` (VAR_KEYWORD) everything is
        passed through; otherwise only its named parameters survive. This
        prevents a ``TypeError`` (uncaught by RAILLM) from a foreign kwarg
        such as ``json_mode``.
        """
        try:
            params = list(inspect.signature(self._sub.chat).parameters.values())
        except (TypeError, ValueError):
            return kwargs
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
            return kwargs
        allowed = {
            p.name
            for p in params
            if p.kind in (inspect.Parameter.KEYWORD_ONLY,
                          inspect.Parameter.POSITIONAL_OR_KEYWORD)
        }
        return {k: v for k, v in kwargs.items() if k in allowed}

    def chat(self, prompt: str = "", *, model: str, **kwargs) -> ChatResponse:
        """Dispatch to the subscription, normalising failures for RAILLM.

        RAILLM invokes this as ``provider.chat(prompt=, model=, **kwargs)``.
        Exceptions it already handles propagate unchanged; any other
        exception becomes ``OpenAICompatError`` so the cascade continues
        instead of crashing.
        """
        try:
            return self._sub.chat(prompt, model=model, **self._accepted_kwargs(kwargs))
        except _RAILLM_HANDLED:
            raise
        except Exception as exc:  # noqa: BLE001 -- normalise so cascade continues
            raise OpenAICompatError(
                f"{self.name}: subscription dispatch failed -- "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def health_check(self) -> bool:
        """Delegate; a health probe must never raise."""
        try:
            return bool(self._sub.health_check())
        except Exception:  # noqa: BLE001 -- probe failure == unhealthy, not crash
            return False

    def list_models(self) -> list:
        """Subscriptions do not enumerate models (ISP-minimal)."""
        return []


def register_subscriptions(
    registry,
    subscriptions: Optional[dict[str, SubscriptionAuth]] = None,
) -> list[str]:
    """Register auto-detected (or supplied) subscriptions into a registry.

    Keyed by ``provider_id`` so RAILLM's
    ``parse_model_address('anthropic/claude-opus-4-7')`` ->
    ``registry.get('anthropic')`` resolves the CCH transport. Returns the
    registered provider names. Idempotent per registry -- re-registering a
    provider_id overwrites the same key.
    """
    subs = subscriptions if subscriptions is not None else auto_detect_subscriptions()
    registered: list[str] = []
    for provider_id, sub in subs.items():
        registry.register(SubscriptionProvider(sub, name=provider_id))
        registered.append(provider_id)
    return registered
