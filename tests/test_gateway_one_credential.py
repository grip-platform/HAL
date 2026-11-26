"""One auth scheme: a failover upstream receives exactly one credential.

InfiniteMachine v2, Step 5b ("HAL uses one auth scheme: ``Authorization``, only").
Before it, a fallback spec's key was injected as ``x-api-key`` and replaced only a
client header of the same name, so a client signed in with a Bearer token reached
the fallback carrying both: two credentials on one request, and the client's
bearer sent to whatever fallback upstream was configured. The guarantee is now
pinned on the spec path -- a ``UpstreamSpec`` that carries its own credential --
which is exactly the shape account failover (Step 5c) will build through the
spend guard. (The env fallback that once built such a spec from
``HAL_GATEWAY_FALLBACK_UPSTREAM`` / ``_FALLBACK_KEY`` was removed by the spend
guard, Step 5h: a paid key chosen by an env var never asks it. Its removal is
pinned in ``tests/test_gateway_failover.py``.)

Everything runs on loopback; every credential is an obvious fake.
"""
from __future__ import annotations

import pytest

from lib.hal.gateway_failover import (FailoverProxy, GatewayTelemetry, UpstreamSpec,
                                      _upstream_headers)
from tests._loopback_http import HttpUpstream, clear_proxies, credentials


@pytest.mark.parametrize("client, client_value", [
    ([("Authorization", "Bearer fake-client-cred")], "fake-client-cred"),  # an OAuth client
    ([("x-api-key", "fake-client-key")], "fake-client-key"),                # an API-key client
])
def test_a_failover_upstream_receives_exactly_one_credential(monkeypatch, client, client_value):
    """Live, through a real failover (the primary 429s, the chain moves on):
    the fallback upstream is reached with its own credential and nothing of
    the client's -- whatever header shape the client presented it in. The
    specs are built directly, as any future account failover must build them
    (through the spend guard), not via the removed env rung."""
    clear_proxies(monkeypatch)
    primary = HttpUpstream([(429, [])] * 3)
    fallback = HttpUpstream([(200, [])])
    specs = [
        UpstreamSpec(name="anthropic-primary", base_url=primary.url),
        UpstreamSpec(name="anthropic-fallback", base_url=fallback.url,
                     header_overrides=(("authorization", "Bearer fake-fallback-key"),)),
    ]
    proxy = FailoverProxy(specs, sleep=lambda seconds: None,
                          telemetry=GatewayTelemetry(sink=lambda event: None))
    try:
        resp = proxy.forward("/v1/messages", b"{}", client + [("content-type",
                                                               "application/json")])
        assert resp.status == 200
        resp.close()
    finally:
        primary.close()
        fallback.close()
    # GatewayProxy sends a Bearer that is not an OAuth token as x-api-key (its
    # "Bearer " prefix dropped), so each upstream sees its one credential that
    # way: the primary the client's, the fallback its own key and nothing of
    # the client's.
    assert credentials(primary.requests[0]) == [("x-api-key", client_value)]
    assert credentials(fallback.requests[0]) == [("x-api-key", "fake-fallback-key")]
    assert client_value not in str(fallback.requests)


@pytest.mark.parametrize("client", [
    [("Authorization", "Bearer fake-client-cred")],
    [("x-api-key", "fake-client-key")],
    [("authorization", "Bearer fake-client-cred"), ("X-Api-Key", "fake-client-key")],
])
def test_a_spec_credential_replaces_the_clients_whole(client):
    sent = _upstream_headers(client + [("anthropic-version", "2023-06-01")],
                             [("authorization", "Bearer fake-pool-token")])
    assert credentials(sent) == [("authorization", "Bearer fake-pool-token")]
    assert ("anthropic-version", "2023-06-01") in sent
    assert _upstream_headers(client, ()) == client  # no spec credential: the client's stays


@pytest.mark.parametrize("overrides", [
    (("x-api-key", "fake-key"),),
    (("X-Api-Key", "fake-key"),),
    (("authorization", "Bearer fake-one"), ("Authorization", "Bearer fake-two")),
])
def test_a_spec_carries_its_credential_as_one_authorization_header(overrides):
    with pytest.raises(ValueError, match="one Authorization header"):
        UpstreamSpec("fallback", header_overrides=overrides)
    assert UpstreamSpec("fallback", header_overrides=(("authorization", "Bearer fake"),))
