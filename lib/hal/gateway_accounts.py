"""Gateway adapter for the wire's account switch, before any client bytes.

Only buffered Anthropic inference turns enter this adapter. Choice, billing
order, organisation checks and the brake belong to account_switch.
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from lib.hal import account_switch, spend_guard
from lib.hal.wire_classify import header_value


def retry_after(headers, *, default):
    """Seconds or HTTP-date, never retry earlier than a valid Retry-After."""
    value = header_value(headers, "retry-after").strip()
    try:
        delay = float(value)
    except ValueError:
        try:
            delay = parsedate_to_datetime(value).timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            return default
    return max(0.0, delay) if math.isfinite(delay) else float("inf")


def _probe(proxy, token):
    response = proxy._opener("GET", proxy.upstream_base + "/v1/models?limit=1", [
        ("Authorization", "Bearer " + token),
        ("anthropic-version", "2023-06-01"),
        ("anthropic-beta", "oauth-2025-04-20"),
    ], None)
    try:
        if response.status == 200:
            return header_value(response.headers, "anthropic-organization-id") or None
    finally:
        response.close()
    return None


def _replacement(proxy, method, path, headers):
    choice = account_switch.choose(method, path, headers,
                                   probe_org=lambda token: _probe(proxy, token))
    if choice is None:
        return None
    label, token, rank = choice
    replaced = [(name, "Bearer " + token if name.lower() == "authorization" else value)
                for name, value in headers]
    refusal = spend_guard.check_anthropic_request(method, path, replaced)
    if refusal is not None:
        raise spend_guard.SpendRefused(refusal)
    from lib.hal.gateway_failover import GatewayTelemetry
    GatewayTelemetry().record({"event": "gateway.account_switched", "to": label[:64],
                               "rank": rank})
    return replaced


def _exhaustion(headers):
    """A known limit is a refusal even when paid overage is opted in."""
    kind, own = spend_guard.anthropic_credential(headers)
    if kind != spend_guard.SUBSCRIPTION:
        return None
    until = account_switch.exhausted_until(own)
    if until is None:
        return None
    # Only observed exhaustion participates. An untested pool account is never
    # invented as capacity or as a reset time. choose has already had its turn.
    resets = [until]
    if account_switch.switching_on():
        for _, token in account_switch._read_pool():
            reset = account_switch.exhausted_until(spend_guard.account_key(token))
            if reset is not None:
                resets.append(reset)
    earliest = min(resets)
    stamp = datetime.fromtimestamp(earliest, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return spend_guard.Decision(False, spend_guard.SUBSCRIPTION,
        f"The subscription limit is spent. Earliest reset: {stamp}.",
        status=429, retry_after=max(1, math.ceil(earliest - time.time())))


def forward(proxy, path, body, client_headers, *, method, max_retries):
    # The translated OpenAI rails use /v1/messages here too. Auxiliary and
    # account-bound routes never observe, choose or replay an account.
    inference = (proxy.checks_anthropic_spend and method.upper() == "POST"
                 and path.split("?", 1)[0] == "/v1/messages")
    buffered = isinstance(body, bytes)
    length = header_value(client_headers, "content-length")
    if length:
        buffered = buffered and int(length) == len(body)
    if not inference or not buffered:
        return proxy._forward(path, body, client_headers, method=method, max_retries=max_retries)
    headers = proxy.upstream_headers(client_headers)
    headers = _replacement(proxy, method, path, headers) or headers
    refusal = _exhaustion(headers)
    if refusal is not None:
        raise spend_guard.SpendRefused(refusal)
    response = proxy._forward(path, body, headers, method=method, max_retries=max_retries)
    if account_switch.observe(response.status, response.headers, headers) is None:
        return response
    try:
        replacement = _replacement(proxy, method, path, headers)
    except spend_guard.SpendRefused:
        response.close()
        raise
    if replacement is not None:
        try:
            # A rejected turn gets exactly one replacement attempt. No retry of
            # an ambiguously accepted replay, and no recursive account walk.
            replay = proxy._forward(path, body, replacement, method=method, max_retries=0)
        except Exception:
            return response
        account_switch.observe(replay.status, replay.headers, replacement)
        if replay.status == 400:
            # Includes cross-account thinking-signature rejection. The saved
            # rejection remains intact; never edit thinking blocks to force it.
            replay.close()
            return response
        response.close()
        response, headers = replay, replacement
    refusal = _exhaustion(headers)
    if response.status == 429 and refusal is not None:
        # The one-replay budget is not evidence of pool exhaustion. A third
        # usable account belongs to the next client request; do not enter the
        # provider cascade or announce a reset while that capacity remains.
        if account_switch.choose(method, path, headers,
                                 probe_org=lambda token: _probe(proxy, token)) is not None:
            return response
        response.close()
        raise spend_guard.SpendRefused(refusal)
    return response
