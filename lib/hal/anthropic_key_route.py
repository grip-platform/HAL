"""anthropic_key_route -- HAL's own Anthropic calls on an API key, cached and capped.

Shipped switched off. HAL's own Anthropic calls go through CCH
(``lib/hal/utils/cch.py``), which presents Claude Code's identity from HAL and
spends the subscription windows the operator's Claude Code sessions need. An
Anthropic API key is the plainly allowed route for programmatic calls (capacity
plan of 2026-10-08, route 3d: an API key with caching behind a daily cap).

**The selection rule.** Where HAL would sign one of its own calls with CCH, the
call goes on the API key instead when ALL of these hold, checked on every call:

1. ``anthropic-api-key`` is in ``paid_opt_in`` (``config/spend_guard.json``);
2. today's (UTC) spend on it is under its ``daily_caps_usd`` cap, shipped at 0;
3. ``config/model_prices.yaml`` prices the model, so its spend can be counted;
4. the keychain holds an item named ``hal-anthropic``, asked by name only.

Otherwise the call is signed with CCH exactly as before. Condition 3 is there
because a response the cap cannot count would leave the cap unenforceable.

**The one place an API-key request is sent** is ``AnthropicProvider._request``
(``lib/hal/providers/anthropic_native.py``). It asks :func:`admit` (conditions
1-3) again, applies ``apply_cache_control`` (``lib/hal/prompt_cache.py``) to a copy
of the body's system block and tools, and after the response calls
:func:`settle`. That prices the response from its usage, with input, cache writes,
cache reads and output each at its own rate from ``config/model_prices.yaml``
(through ``lib/hal/infinite_cost.py``), and adds it to the day. A send refused
there leaves nothing and raises ``AnthropicSpendRefused``, on which
:func:`try_api_key` hands the call back to CCH. Condition 4, the keychain item, is
asked only by :func:`select`, before any send.

**Where it applies.** Two seams carry every call HAL signs with CCH for itself,
and both ask :func:`try_api_key` first: ``CCHSubscription.chat``
(``lib/hal/subscription.py``) and Happi's direct ``CCHSigner().call``
(``lib/hal/happi/providers/anthropic.py``, when the envelope leaves the
credential to HAL). ``CCHSubscription.chat`` is reached by hal-server's registered
``anthropic`` provider and its ``/api/infer-with-tools`` tool loop
(``lib/hal/server.py``), by the RAILLM infer ladder's Anthropic rung
(``lib/hal/raillm_subscription_bridge.py``), and by the ``hal`` REPL's tool mode
(``lib/hal/tool_cli_integration.py``), which with no CCH token uses the key only
through :class:`ApiKeyTransport`.

Not touched: the wire (``lib/hal/wire_proxy.py``) and the gateway, which relay each
client's own sign-in rather than choose one for HAL. Neither uses CCH.
"""
from __future__ import annotations

import logging
import subprocess
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from lib.hal import infinite_cost, spend_guard
from lib.hal.providers import anthropic_native
from lib.hal.types import ChatResponse

_log = logging.getLogger("hal.anthropic_key_route")

#: The paid source this route spends, as the spend guard names it.
SOURCE = spend_guard.API_KEY


def _refused(why: str) -> spend_guard.Decision:
    return spend_guard.Decision(False, SOURCE, f"HAL spend guard refused '{SOURCE}': {why}")


def keychain_item_exists() -> bool:
    """Does the macOS keychain hold HAL's ``hal-anthropic`` item? Asked by name only.

    ``security find-generic-password -s NAME`` without ``-w`` reports the item's
    attributes, never its secret, and both of its output streams are discarded.
    False when the item is absent, when ``security`` is missing (any system but
    macOS), or when it does not answer within five seconds.
    """
    name = anthropic_native.API_KEY_SERVICE
    try:
        done = subprocess.run(["security", "find-generic-password", "-s", name],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def priceable(model: str, *, now: Optional[datetime] = None) -> bool:
    """Does ``config/model_prices.yaml`` price ``model`` on today's (UTC) date?"""
    day = spend_guard.today_utc(now)
    return infinite_cost.price_row(infinite_cost.price_table(), model, day) is not None


def admit(model: str, *, now: Optional[datetime] = None) -> spend_guard.Decision:
    """May an API-key request for ``model`` be sent now? Conditions 1-3 above."""
    verdict = spend_guard.decide_within_cap(SOURCE, now=now)
    if verdict.allowed and not priceable(model, now=now):
        return _refused(f"config/model_prices.yaml has no price for {model!r} on "
                        f"{spend_guard.today_utc(now)}, so its spend could not be "
                        "counted against the daily cap")
    return verdict


def select(model: str, *, now: Optional[datetime] = None) -> spend_guard.Decision:
    """The selection rule: :func:`admit`, and the keychain item exists (asked by name)."""
    verdict = admit(model, now=now)
    if verdict.allowed and not keychain_item_exists():
        return _refused(f"the keychain holds no item named "
                        f"{anthropic_native.API_KEY_SERVICE!r}")
    return verdict


def settle(model: str, raw_usage: Any, *, now: Optional[datetime] = None) -> Optional[Decimal]:
    """Price one API-key response from its usage block and add it to today's spend.

    Returns what the response cost in US dollars, or None when it could not be
    priced; today's spend is then marked unknown, which refuses the source until
    midnight UTC. Never raises: the response is already paid for and the caller
    must still get it.
    """
    try:
        usage = infinite_cost.anthropic_usage(raw_usage)
        cost, _version = infinite_cost.list_price(model, usage, spend_guard.today_utc(now))
        if cost is None:
            spend_guard.record_unknown(SOURCE, f"a response for {model!r} could not be priced",
                                       now=now)
            return None
        amount = Decimal(cost)
        spend_guard.record_spend(SOURCE, amount, usage=usage, now=now)
        return amount
    except Exception as exc:  # noqa: BLE001 -- see the docstring
        _log.warning("an API-key response could not be settled (%s)", type(exc).__name__)
        spend_guard.record_unknown(SOURCE, f"settling failed ({type(exc).__name__})", now=now)
        return None


def api_provider() -> anthropic_native.AnthropicProvider:
    """The native provider on the API key alone. The key is read only when a
    request it admits is about to be sent."""
    return anthropic_native.AnthropicProvider(prefer="api_key")


def try_api_key(prompt: str = "", *, model: str, messages: Optional[list] = None,
                system: Optional[str] = None, max_tokens: int = 16384,
                tools: Optional[list] = None) -> Optional[ChatResponse]:
    """Send one of HAL's own calls on the API key, when :func:`select` allows it.

    Returns the response, or None when the route is closed or the send was
    refused before anything left HAL; the caller then signs the call with CCH as
    before. An error from Anthropic itself (a 400 for an unfunded key, a 5xx) is
    raised as ``AnthropicError`` and never retried on CCH, because that request
    may have reached Anthropic.
    """
    if not select(model).allowed:
        return None
    try:
        return api_provider().chat(prompt, model=model, messages=messages, system=system,
                                   max_tokens=max_tokens, tools=tools, prefer="api_key")
    except anthropic_native.AnthropicSpendRefused:  # nothing was sent
        return None


class ApiKeyTransport:
    """HAL's API key shaped like a subscription transport, for a caller with no CCH
    token (the ``hal`` REPL's tool mode). Every call asks :func:`select` again, so
    once today's cap is reached it raises instead of spending. ``chat`` has
    ``CCHSubscription.chat``'s signature; ``temperature`` is accepted for that
    reason and not sent, as CCH does not send it either."""

    provider_id = "anthropic"
    name = "anthropic-api-key"

    def chat(self, prompt: str = "", *, model: str, messages: Optional[list] = None,
             system: Optional[str] = None, temperature: Optional[float] = None,
             max_tokens: int = 16384, tools: Optional[list] = None) -> ChatResponse:
        response = try_api_key(prompt, model=model, messages=messages, system=system,
                               max_tokens=max_tokens, tools=tools)
        if response is None:
            raise anthropic_native.AnthropicError(
                f"HAL's API-key route did not serve {model!r}; "
                "`python -m lib.hal.spend_guard caps` shows the opt-in, the cap and today's spend")
        return response

    def health_check(self) -> bool:
        """False: a paid key never counts as capacity."""
        return False


__all__ = [
    "SOURCE",
    "ApiKeyTransport",
    "admit",
    "api_provider",
    "keychain_item_exists",
    "priceable",
    "select",
    "settle",
    "try_api_key",
]
