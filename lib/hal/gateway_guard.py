"""HAL Gateway access guard — default-deny control for the ``/v1/*`` proxy.

The gateway (:mod:`lib.hal.gateway`) reverse-proxies ``/v1/*`` to
``api.anthropic.com``. Inside ``hal-server`` it is loopback-bound, but the VPS
tunnels ``hal-server`` publicly as ``hal.grip-web.com``. On a redeploy that
wires the proxy without this guard, ``/v1/*`` becomes a PUBLIC UNAUTHENTICATED
open relay to Anthropic. This module is the in-repo, deploy-safe control that
denies that by default.

STRIDE summary (threat-modeling skill): **Information disclosure** — prompts +
the upstream credential transit the VPS; **DoS / abuse** — an open relay
invites resource exhaustion and quota theft; **Elevation of privilege** — the
FailoverProxy reads a fallback upstream credential from the environment, so an
unauthenticated remote caller could consume the operator's upstream quota.

Fail-CLOSED vs fail-OPEN — the deliberate inverse of the gateway:
:mod:`lib.hal.gateway` is **fail-OPEN** by design (its signature-preservation
value is that it can only ever *add* success, never subtract it; a malformed
internal state degrades to relaying the request). This guard is the opposite:
it is a **security** control, so on ANY ambiguity — malformed headers, an
exception mid-evaluation, whois error — it DENIES. Availability is the
gateway's invariant; safety is the guard's. They compose: the guard runs first
(deny-by-default), and only an allowed request reaches the fail-open proxy.

Dependency-light: stdlib only (``os`` for the env default, ``hmac`` for the
timing-safe compare). Network calls are made through an injectable function
so tests can fake them. Third-party imports only for the injectable functions.

Falsifiable hypothesis: after the VPS ``hal-server`` redeploys to main with
this guard wired into the ``/v1/*`` route, an unauthenticated remote request to
``hal.grip-web.com/v1/*`` returns 403, while local callers and the standalone
:4010 gateway are unaffected. Falsifier: a remote unauthenticated ``/v1/*``
request succeeds post-deploy, or a local flow breaks.
"""
from __future__ import annotations

import hmac
import os
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple, Union

# Headers an upstream tunnel/proxy (Cloudflare / cloudflared) injects and that
# an external client cannot strip. A loopback-local caller sends none of these.
# Cloudflare always adds cf-ray + cf-connecting-ip for tunnelled traffic; the
# rest are the common proxy-forwarding markers. Stored lowercase for O(1)
# case-insensitive membership.
_PROXY_HEADERS = frozenset({
    "cf-ray",
    "cf-connecting-ip",
    "cf-worker",
    "true-client-ip",
    "x-forwarded-for",
    "x-real-ip",
    "forwarded",
})

# Headers that can be forged by a remote peer to pretend to be local.
_FORWARDING_HEADERS = frozenset({
    "x-forwarded-for",
    "x-real-ip",
    "forwarded",
    "x-cluster-client-ip",
    "x-originating-ip",
    "x-forwarded-host",
    "x-forwarded-server",
    "x-original-forwarded-for",
    "x-proxy-user-ip",
    "via",
    "x-cnection",
})

# The header an authenticated remote caller must present.
_GATEWAY_TOKEN_HEADER = "x-hal-gateway-token"

# Env var name holding the shared gateway secret. Empty/unset => default-deny
# remote. Named by a readable constant; the value is never logged or echoed.
_AUTH_TOKEN_ENV = "HAL_GATEWAY_AUTH_TOKEN"

# Env var name holding the required ACL tag for Tailscale authentication.
_REQUIRED_TS_TAG_ENV = "HAL_GATEWAY_TS_ACL_TAG"

# Env var for mTLS certificate requirement (optional)
_REQUIRE_CLIENT_CERT_ENV = "HAL_GATEWAY_REQUIRE_CLIENT_CERT"

# Minimal, leak-free 403 body. Reveals nothing about why or what exists.
_FORBIDDEN_BODY = (
    b'{"error":{"type":"forbidden","message":"gateway access denied"}}'
)

HeaderPairs = Union[Mapping[str, str], Iterable[Tuple[str, str]]]


def _header_pairs(headers: HeaderPairs) -> Iterable[Tuple[str, str]]:
    """Yield ``(name, value)`` from a mapping OR a plain ``(k, v)`` iterable.

    Mirrors :func:`lib.hal.gateway._header_items` so the guard accepts the
    same header shapes the gateway already handles (``handler.headers`` is a
    mapping with ``.items()``; tests and callers may pass a list of tuples).
    """
    items = getattr(headers, "items", None)
    if callable(items):
        return items()
    return headers


def _lower_value_map(headers: HeaderPairs) -> dict:
    """Build a ``{lowercased-name: value}`` dict for case-insensitive lookup."""
    return {str(name).lower(): value for name, value in _header_pairs(headers)}


def _is_loopback_address(address: str) -> bool:
    """Check if the given address is a loopback address."""
    try:
        # Handle IPv4 addresses
        if ':' not in address.replace('[', '').replace(']', ''):  # IPv4
            ip_parts = address.split('.')
            if len(ip_parts) == 4:
                return ip_parts[0] == '127'
        else:  # IPv6
            # Remove brackets if present
            clean_addr = address.strip('[]')
            return clean_addr == '::1' or clean_addr.lower() == 'localhost'
        
        # For string representations like 'localhost'
        if address.lower() == 'localhost':
            return True
            
        return False
    except:
        # If parsing fails, default to not loopback
        return False


def _get_client_ip(handler) -> Optional[str]:
    """Extract the real client IP from the request handler."""
    # First, try to get the remote address directly from the handler
    if hasattr(handler, 'client_address') and handler.client_address:
        return handler.client_address[0]

    return None


def is_remote_request(headers: HeaderPairs) -> bool:
    """True if the request arrived via a tunnel/proxy (has a proxy header).

    ``headers`` is a mapping or an iterable of ``(name, value)`` pairs. Matching
    is case-insensitive and presence-based: any one proxy header => remote.
    """
    present = {str(name).lower() for name, _ in _header_pairs(headers)}
    return not present.isdisjoint(_PROXY_HEADERS)


def is_loopback_caller(handler) -> bool:
    """True iff this request is genuinely local: a loopback peer address
    carrying NO forwarding/proxy header.

    A loopback ``client_address`` is NOT sufficient on its own: hal-server
    sits behind a Cloudflare Tunnel, whose local ``cloudflared`` agent
    connects to this process over loopback for EVERY request it forwards —
    local or internet-originated alike. The TCP peer is always loopback;
    only the preserved HTTP headers (``cf-ray`` etc.) still distinguish a
    tunnelled remote caller. This mirrors the exact condition
    :func:`gateway_access_allowed` already uses to admit a trusted local
    peer (see its docstring) — a second, independent re-derivation of that
    condition would silently drift from it.

    For endpoints that narrow their response body for remote callers rather
    than gating them outright — e.g. the public ``/api/health`` liveness
    probe, which must never reveal the absolute HAL_HOME path (and the host
    account name it embeds) to anyone off the loopback boundary.

    Fail-CLOSED on ambiguity, matching this module's own invariant (see its
    docstring): a handler with no readable ``.headers`` cannot be confirmed
    free of a forwarding header, so it is treated as NOT genuinely local —
    the safe direction is to redact, never to reveal.
    """
    ip = _get_client_ip(handler)
    if ip is None or not _is_loopback_address(ip):
        return False
    headers = getattr(handler, "headers", None)
    if headers is None:
        return False
    return not is_remote_request(headers)


def _call_tailscale_whois(ip_address: str, whois_func: Callable[[str], Optional[Dict]]) -> Optional[Dict]:
    """Call the tailscale whois function and handle errors gracefully."""
    try:
        return whois_func(ip_address)
    except Exception:
        # Fail closed: if whois errors, times out or returns something unparseable,
        # the answer is 403
        return None


def gateway_access_allowed(
    headers: HeaderPairs,
    *,
    token: Optional[str] = None,
    handler=None,  # Pass the handler to get client IP
    whois_func: Optional[Callable[[str], Optional[Dict]]] = None,  # Injected for testing
    require_client_cert: Optional[bool] = None,  # Override for testing
) -> bool:
    """Decide whether this request may reach the ``/v1/*`` proxy. Fail-CLOSED.

    Peer classification is by the **connecting address**, never by header
    presence (a remote peer sends no forwarding headers, so header presence
    cannot separate local from remote):

      * **Loopback** — ``handler.client_address`` is a loopback address
        (``127.0.0.0/8``, ``::1``, ``localhost``). Loopback is the trusted
        invariant and is allowed. The legacy sub-case is preserved: if a
        loopback peer *also* carries proxy headers and a static token is
        configured, that token must match.
      * **Remote** — everything else: a non-loopback address, or a peer whose
        address cannot be established (no ``handler``, or no
        ``client_address``). A remote peer is admitted **only** when all of
        the following hold, and is refused (403) otherwise:

          1. A required ACL tag is configured (``HAL_GATEWAY_TS_ACL_TAG``).
             With **no tag configured, every remote peer is refused** — the
             fail-open default is closed.
          2. ``whois_func`` is available and returns a ``tailscale whois
             --json`` result whose ``Node.Tags`` contains the required tag
             (exact, case-sensitive match). Any error, ``None``, missing
             ``Node``, missing/non-list ``Node.Tags``, or absent tag => refuse.
             The guessed ``tags`` / ``caps.tags`` / ``Tags`` / ``Capabilities``
             fields are **not** read.
          3. The peer presents no forwarding header (a forged-local attempt is
             refused outright).
          4. Client certificates are **not** required
             (``HAL_GATEWAY_REQUIRE_CLIENT_CERT``). This server terminates TLS
             in front of the handler and cannot see the peer's TLS certificate,
             so when a client certificate is required every remote peer is
             refused rather than pretending a certificate was verified.
          5. If a static token is configured (``HAL_GATEWAY_AUTH_TOKEN``), the
             ``X-HAL-Gateway-Token`` header must match it (constant-time).

    ``token`` defaults to the ``HAL_GATEWAY_AUTH_TOKEN`` env var (``""`` if
    unset). ``whois_func`` is injected by the calling code; when it is ``None``
    no remote peer can be authenticated, so remote peers are refused (fail
    closed). ``require_client_cert`` defaults to the
    ``HAL_GATEWAY_REQUIRE_CLIENT_CERT`` env var.

    Any exception (malformed headers, etc.) is swallowed and returns ``False``
    — the deliberate inverse of the gateway's fail-OPEN availability invariant.
    """
    try:
        if token is None:
            token = os.environ.get(_AUTH_TOKEN_ENV, "")

        # Get the real client IP address.
        client_ip = None
        if handler is not None:
            client_ip = _get_client_ip(handler)

        # Proven loopback: the trusted invariant (unchanged legacy behaviour).
        if client_ip is not None and _is_loopback_address(client_ip):
            if not is_remote_request(headers):
                return True
            if not token:
                return False
            presented = _lower_value_map(headers).get(_GATEWAY_TOKEN_HEADER, "")
            return hmac.compare_digest(str(presented), str(token))

        # Remote peer: non-loopback address, or address not established. Fail
        # closed. The connecting address must be established before anything
        # can be authenticated, so a peer with no address is refused.
        if client_ip is None:
            return False

        # A remote peer must not carry forwarding headers (forged-local attempt).
        present_headers = {str(name).lower() for name, _ in _header_pairs(headers)}
        if not present_headers.isdisjoint(_FORWARDING_HEADERS):
            return False

        # 1. A required tag must be configured; no tag => refuse every remote peer.
        required_tag = os.environ.get(_REQUIRED_TS_TAG_ENV)
        if not required_tag:
            return False

        # 2. The peer must authenticate through Tailscale whois.
        if whois_func is None:
            return False
        whois_result = _call_tailscale_whois(client_ip, whois_func)
        if not isinstance(whois_result, dict):
            return False

        node = whois_result.get("Node")
        if not isinstance(node, dict):
            return False
        tags = node.get("Tags")
        if not isinstance(tags, (list, tuple)):
            return False
        if required_tag not in tags:
            return False

        # 4. Client certificates cannot be verified here => refuse remote.
        if require_client_cert is None:
            require_client_cert = (
                os.environ.get(_REQUIRE_CLIENT_CERT_ENV, "").lower()
                in ("1", "true", "yes")
            )
        if require_client_cert:
            return False

        # 5. Optional static-token second factor.
        if token:
            presented = _lower_value_map(headers).get(_GATEWAY_TOKEN_HEADER, "")
            return hmac.compare_digest(str(presented), str(token))
        return True

    except Exception:
        return False


def _send_forbidden(handler) -> None:
    """Write a clean, leak-free 403 to ``handler``. Never raises."""
    handler.send_response(403)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(_FORBIDDEN_BODY)))
    handler.send_header("Connection", "close")
    handler.end_headers()
    try:
        handler.wfile.write(_FORBIDDEN_BODY)
    except (BrokenPipeError, ConnectionResetError):
        pass  # client hung up — nothing to recover


def enforce_gateway_access(handler) -> bool:
    """Gate a ``BaseHTTPRequestHandler`` request. Returns True iff allowed.

    On allow: returns True and writes nothing (the caller proceeds to relay).
    On deny: sends a minimal 403 (``{"error":{"type":"forbidden",...}}``,
    ``Content-Type: application/json``, ``Connection: close``) and returns
    False. Write errors (BrokenPipe/ConnectionReset) are caught; this never
    raises, so a wiring caller can safely guard with
    ``if not enforce_gateway_access(self): return``.
    """
    # By default, use a function that will cause non-loopback requests to be denied
    # The actual whois function would be injected by the calling code
    if gateway_access_allowed(handler.headers.items(), handler=handler):
        return True
    _send_forbidden(handler)
    return False
