"""Forward anchor for an AGORA concern on HAL#732.

The concern: with a refused paid source as the only configured upstream,
``_finish_refusal`` would "fall through to an unhandled exception path". It does
not. With no observed usage limit there is no evidence that subscription capacity
is spent, so the original spend refusal is re-raised and answered exactly as
before the change: no W4 target is tried and the paid source is never sent.

Kept in its own file because the cascade test file carries older complexity debt
that GRIP's edit gate will not let a new test land beside.
"""
import pytest

from lib.hal import account_switch, gateway, gateway_failover as failover, spend_guard
from tests.test_gateway_cascade import CHAT, RESPONSES, dispatch, request
from tests.test_gateway_cascade import fakes  # noqa: F401 -- pytest fixture


@pytest.mark.parametrize('path', [CHAT, RESPONSES])
def test_paid_only_pool_keeps_its_refusal_and_never_cascades(fakes, monkeypatch, path):  # noqa: F811
    # Paid-only means no subscription account to switch to. Without this the
    # account switch (HAL#686) would read this machine's real token store.
    monkeypatch.setattr(account_switch, '_read_pool', lambda: [])
    # conftest installs a permissive policy for relaying tests; this one needs
    # the real rule for Anthropic (no paid source opted in). The fake W4 targets
    # ARE opted in, so a wrongly started cascade would serve from them and fail
    # this test -- without that, "no cascade" and "cascade with nothing allowed"
    # would look the same.
    strict = fakes.config.with_name('strict-spend.json')
    strict.write_text('{"paid_opt_in": ["groq-api-key", "gemini-api-key"]}')
    monkeypatch.setenv(spend_guard.CONFIG_ENV, str(strict))
    sent = []
    paid = gateway.GatewayProxy(opener=lambda *args: sent.append(args) or
                                pytest.fail('paid Anthropic source was sent'))
    proxy = failover.FailoverProxy(
        [failover.UpstreamSpec('paid', header_overrides=(('Authorization', 'Bearer sk-ant-api-fake'),))],
        proxy_factory=lambda spec: paid)
    handler = dispatch(request(path), path, proxy)
    assert handler.status is not None
    assert b'switched' not in handler.wfile.getvalue()
    assert not sent and not fakes.calls
