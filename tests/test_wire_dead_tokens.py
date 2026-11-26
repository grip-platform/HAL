"""Dead credentials over loopback TLS, using fake tokens only."""
import json
import time

import pytest
from lib.hal import account_ledger, account_switch, spend_guard
from tests import test_wire_proxy as h
from tests.test_wire_proxy import (  # noqa: F401 -- fixtures that pytest finds by name here:
    home, pki, make_wire, strict_spend,  # replay_wire and held ask for them in this module
)
from tests.test_wire_replay import replay_wire, A, B, C, OK, LIMIT, reply, request, posts


@pytest.fixture
def dead_wire(replay_wire):
    replay_wire.proxy._ledger = account_ledger.AccountLedger(resolver=account_ledger.AccountResolver(
        pool=lambda: [('A', A), ('B', B), ('C', C)]))
    return replay_wire


def dead_rows(home, at_least=1):
    """The ``wire.dead_token`` telemetry rows, after waiting for ``at_least`` of
    them. A shorter list comes back when the wait runs out, never None, so the
    caller's assertion says what was missing."""
    def rows():
        return [r for r in h._rows(home) if r.get('event') == 'wire.dead_token']
    h._wait_for(lambda: len(rows()) >= at_least)
    return rows()


def test_dead_presented_replayed(dead_wire):
    dead_wire.fake.answers[A] = reply(401, b'dead original')
    raw = request()
    assert h._exchange(dead_wire, raw) == OK
    assert posts(dead_wire) == [raw, raw.replace(A.encode(), B.encode())]
    assert account_ledger.LedgerView().token_health('A', A) == 'dead'
    rows = dead_rows(dead_wire.home)
    assert len(rows) == 1
    assert {k: rows[0][k] for k in ('account', 'switched', 'outcome', 'status')} == {
        'account': 'A', 'switched': True, 'outcome': 'served', 'status': 401}
    assert all(t not in json.dumps(rows) for t in (A, B, C))


def test_a_presented_403_is_never_switched_or_replayed(dead_wire):
    """Inverted 2026-10-08 (pool hardening). This test used to require the 403 to
    be replayed on B. A 403 is a permission answer, which can be an enforcement
    action, and switching account after one is the shape Anthropic's Usage Policy
    names, so it reaches Claude Code exactly as it came, with a replacement
    available. A 403 that names no suspension is not a dead token either: a
    model-permission refusal leaves the account's token alone, and switching on."""
    original = reply(403, b'{"type":"error","error":{"type":"permission_error",'
                          b'"message":"not allowed to use this model"}}')
    dead_wire.fake.answers[A] = original
    raw = request()
    assert h._exchange(dead_wire, raw) == original
    assert posts(dead_wire) == [raw]
    assert account_ledger.LedgerView().token_health('A', A) != 'dead'
    assert account_switch.switching_on() and account_switch.off_reason() is None
    rows = dead_rows(dead_wire.home)
    assert len(rows) == 1
    assert {k: rows[0][k] for k in ('account', 'switched', 'outcome', 'status', 'signal')} == {
        'account': 'A', 'switched': False, 'outcome': 'original', 'status': 403,
        'signal': 'unknown'}


def test_dead_skipped_and_relogin_returns(dead_wire, monkeypatch):
    with dead_wire.proxy._ledger.exchange([('Authorization', 'Bearer ' + B)]) as watch:
        watch.answered(401, [])
    headers = [('Authorization', 'Bearer ' + A)]
    def choose():
        return account_switch.choose('POST', '/v1/messages', headers,
            after_throttle=True, probe_org=lambda t: 'org-' + t)
    assert choose()[0] == 'C'
    monkeypatch.setattr(account_switch, '_read_pool', lambda: [('B', 'fake-account-B-new')])
    assert choose()[0] == 'B'


@pytest.mark.parametrize('boundary', ['empty', 'route', 'overflow', 'started', 'guard'])
def test_original_preserved_at_boundaries(dead_wire, monkeypatch, boundary):
    bad = reply(401, b'own login expired')
    dead_wire.fake.answers[A] = bad
    raw = request('/v1/code/sessions' if boundary == 'route' else '/v1/messages')
    if boundary == 'empty':
        monkeypatch.setattr(account_switch, '_read_pool', lambda: [])
    if boundary == 'overflow':
        original = dead_wire.proxy._send_for_answer
        def overflow(*args):
            head = original(*args)
            ex = args[-1]
            saved = ex.wire_body.value()
            ex.wire_body.overflow = True
            ex.wire_body.value = lambda: saved
            return head
        monkeypatch.setattr(dead_wire.proxy, '_send_for_answer', overflow)
    if boundary == 'guard':
        real = spend_guard.check_anthropic_request
        monkeypatch.setattr(spend_guard, 'check_anthropic_request', lambda m, p, hs:
            spend_guard.Decision(False, spend_guard.OVERAGE, 'test refusal', status=429)
            if account_ledger.credential_of(hs) == B else real(m, p, hs))
    if boundary == 'started':
        interim = b'HTTP/1.1 103 Early Hints\r\n\r\n'
        dead_wire.fake.answers[A] = interim + bad
        tls = h._tunnel(dead_wire.port, h.API, dead_wire.client)
        try:
            tls.sendall(raw)
            reader = h._Reader(tls)
            assert h._read_response(reader) == interim
            assert h._read_response(reader) == bad
        finally:
            tls.close()
    else:
        assert h._exchange(dead_wire, raw) == bad
    assert posts(dead_wire) == [raw]


@pytest.mark.parametrize('status, health', [(401, 'dead'), (403, 'unknown')])
def test_dead_replacement_restores_original(dead_wire, status, health):
    """Either way the original answer is restored. Only the 401 holds B's token
    dead: a 403 that names no suspension says nothing about the token (amended
    2026-10-08, pool hardening; it used to require 'dead' for both)."""
    dead_wire.fake.answers[B] = reply(status, b'replacement secret error')
    assert h._exchange(dead_wire, request()) == LIMIT
    assert account_ledger.LedgerView().token_health('B', B) == health
    rows = dead_rows(dead_wire.home)
    assert len(rows) == 1 and rows[0]['account'] == 'B'
    assert rows[0]['outcome'] == 'original' and not rows[0]['switched']


@pytest.mark.parametrize('status, health', [(401, 'dead'), (403, 'unknown')])
def test_hold_dead_replacement_retries_own(held, monkeypatch, status, health):
    """Back to the turn's own account, which is no switch. Only the 401 holds B's
    token dead (amended 2026-10-08, pool hardening; it used to require 'dead' for
    both statuses)."""
    from tests.test_wire_hold import exchange, SUCCESS, posts as held_posts
    from lib.hal import wire_hold
    # Five posts and their organisation probes on fresh TLS connections need
    # more than the fixture's 0.5 s on a slow runner (HAL#685).
    monkeypatch.setattr(wire_hold, 'BUDGET_S', 5.0)
    held.proxy._ledger = account_ledger.AccountLedger(resolver=account_ledger.AccountResolver(
        pool=lambda: [('A', A), ('B', B)]))
    held.fake.actions = [reply(429, b'throttle')] * wire_hold.THROTTLE_ATTEMPTS + [reply(status, b'dead replacement'), SUCCESS]
    raw = exchange(held)
    assert b'answer' in raw and b'dead replacement' not in raw
    assert raw.count(b'HTTP/1.1') == 1
    assert [h._bearer_of(r) for r in held_posts(held)] == [A] * wire_hold.THROTTLE_ATTEMPTS + [B, A]
    assert account_ledger.LedgerView().token_health('B', B) == health
    rows = dead_rows(held.home)
    assert len(rows) == 1 and rows[0]['outcome'] == 'retry_original'


def test_hold_dead_replacement_then_spent_own_pauses_once(held, monkeypatch):
    """The hold switches only after its 200 head and pings reached the client, so
    when it returns to its own account it is still the same started response. If
    that account is then spent, the client gets one pause inside that one
    response; nothing sends a second head and nothing is replayed again."""
    from tests.test_wire_hold import exchange, posts as held_posts
    from lib.hal import wire_hold
    monkeypatch.setattr(wire_hold, 'BUDGET_S', 5.0)
    held.proxy._ledger = account_ledger.AccountLedger(resolver=account_ledger.AccountResolver(
        pool=lambda: [('A', A), ('B', B)]))
    spent = reply(429, b'{}', b'anthropic-ratelimit-unified-status: rejected\r\n')
    held.fake.actions = ([reply(429, b'throttle')] * wire_hold.THROTTLE_ATTEMPTS
                         + [reply(401, b'dead replacement'), spent])
    raw = exchange(held)
    assert raw.count(b'HTTP/1.1') == 1
    assert raw.count(b'event: error') == 1 and b'dead replacement' not in raw
    assert [h._bearer_of(r) for r in held_posts(held)] == [A] * wire_hold.THROTTLE_ATTEMPTS + [B, A]


def test_hold_dead_presented_and_dead_replacement_relay_original_once(held, monkeypatch):
    """The double-dead case on the hold path that Claude Code's streamed turns
    take: the hold's first attempt replays once, the replacement is dead too,
    so the original 401 is relayed exactly once, with one row per dead token."""
    from tests.test_wire_hold import exchange, posts as held_posts
    from lib.hal import wire_hold
    monkeypatch.setattr(wire_hold, 'BUDGET_S', 5.0)
    held.proxy._ledger = account_ledger.AccountLedger(resolver=account_ledger.AccountResolver(
        pool=lambda: [('A', A), ('B', B)]))
    original = reply(401, b'dead original')
    held.fake.actions = [original, reply(403, b'dead replacement')]
    assert exchange(held) == original
    assert [h._bearer_of(r) for r in held_posts(held)] == [A, B]
    rows = dead_rows(held.home, at_least=2)
    assert sorted((r['account'], r['switched'], r['outcome']) for r in rows) == [
        ('A', False, 'original'), ('B', False, 'original')]


def test_dead_presented_and_dead_replacement_relay_original_once(dead_wire):
    """Both tokens dead: one replacement attempt, never recursive, so the original
    401 is relayed once and both tokens are held dead (one row each). The next
    turn skips both and is served by the third account. B answers 401 here: since
    2026-10-08 a 403 that names no suspension no longer holds a token dead, so a
    403 could not make the 'both dead' this test is about."""
    original = reply(401, b'dead original')
    dead_wire.fake.answers[A] = original
    dead_wire.fake.answers[B] = reply(401, b'dead replacement')
    raw = request()
    assert h._exchange(dead_wire, raw) == original
    assert posts(dead_wire) == [raw, raw.replace(A.encode(), B.encode())]
    view = account_ledger.LedgerView()
    assert view.token_health('A', A) == 'dead' and view.token_health('B', B) == 'dead'
    rows = dead_rows(dead_wire.home, at_least=2)
    assert sorted((r['account'], r['switched'], r['outcome']) for r in rows) == [
        ('A', False, 'original'), ('B', False, 'original')]
    assert h._exchange(dead_wire, raw) == OK
    assert posts(dead_wire)[-1] == raw.replace(A.encode(), C.encode())


from tests.test_wire_hold import held


@pytest.mark.parametrize('own_org', [None, 'shared-org'])
def test_dead_reason_ignores_own_org_but_limit_does_not(dead_wire, own_org):
    headers = [('Authorization', 'Bearer ' + A)]
    def probe(token):
        return own_org if token == A else 'shared-org'
    assert account_switch.choose('POST', '/v1/messages', headers,
        reason='dead_token', probe_org=probe)[0] == 'B'
    assert account_switch.choose('POST', '/v1/messages', headers,
        after_throttle=True, probe_org=probe) is None


def test_dead_replay_without_own_org(dead_wire, monkeypatch):
    dead_wire.fake.answers[A] = reply(401, b'own expired')
    monkeypatch.setattr(dead_wire.proxy, '_probe_org', lambda t: None if t == A else 'shared-org')
    assert h._exchange(dead_wire, request()) == OK


def test_dead_reason_paid_guard(dead_wire, monkeypatch, strict_spend):
    monkeypatch.setattr(account_switch, '_has_subscription', lambda *a: False)
    assert account_switch.choose('POST', '/v1/messages', [('Authorization', 'Bearer ' + A)],
        reason='dead_token', probe_org=lambda _: 'shared-org') is None


@pytest.mark.parametrize('failures', [1, 2])
def test_org_probe_retries_transport_once(dead_wire, monkeypatch, failures):
    take = dead_wire.proxy.pool.take
    attempts = []
    def flaky(**kwargs):
        attempts.append(1)
        if len(attempts) <= failures:
            raise ConnectionResetError('fake transport failure')
        return take(**kwargs)
    monkeypatch.setattr(dead_wire.proxy.pool, 'take', flaky)
    result = dead_wire.proxy._probe_org(B)
    assert bool(result) == (failures == 1)
    assert len(attempts) == 2


@pytest.mark.parametrize('status', [401, 403, 429, 500])
def test_org_probe_does_not_retry_answer(dead_wire, monkeypatch, status):
    from types import SimpleNamespace
    calls, closed = [], []
    class Socket:
        def settimeout(self, value): pass
        def sendall(self, value): pass
        def shutdown(self, value): pass
        def close(self): closed.append(1)
    def take(**kwargs):
        calls.append(1)
        return SimpleNamespace(sock=Socket(), stream=SimpleNamespace(
            read_head=lambda: reply(status, b'').split(b'\r\n\r\n')[0] + b'\r\n\r\n'))
    monkeypatch.setattr(dead_wire.proxy.pool, 'take', take)
    assert dead_wire.proxy._probe_org(B) is None
    assert calls == [1] and closed == [1]


def test_hold_dead_presented_replayed(held):
    """Claude Code streams its turns, so they reach the hold first: a dead own
    token there is replayed on B, and no byte of the 401 reaches the client."""
    from tests.test_wire_hold import exchange, SUCCESS, posts as held_posts
    held.proxy._ledger = account_ledger.AccountLedger(resolver=account_ledger.AccountResolver(
        pool=lambda: [('A', A), ('B', B)]))
    held.fake.actions = [reply(401, b'dead original'), SUCCESS]
    raw = exchange(held)
    assert b'answer' in raw and b'dead original' not in raw
    assert [h._bearer_of(r) for r in held_posts(held)] == [A, B]
    assert account_ledger.LedgerView().token_health('A', A) == 'dead'


def test_hold_presented_403_is_never_switched_or_replayed(held):
    """Inverted 2026-10-08 (pool hardening): on the hold path too, where Claude
    Code's streamed turns go, a 403 on the presented token reaches the client
    exactly as it came and B is never asked. This test used to require the 403
    to be replayed on B."""
    from tests.test_wire_hold import exchange, SUCCESS, posts as held_posts
    held.proxy._ledger = account_ledger.AccountLedger(resolver=account_ledger.AccountResolver(
        pool=lambda: [('A', A), ('B', B)]))
    original = reply(403, b'{"type":"error","error":{"type":"permission_error",'
                          b'"message":"not allowed to use this model"}}')
    held.fake.actions = [original, SUCCESS]
    assert exchange(held) == original
    assert [h._bearer_of(r) for r in held_posts(held)] == [A]
    assert account_ledger.LedgerView().token_health('A', A) != 'dead'
    assert account_switch.switching_on()


def test_hold_dead_presented_without_replacement_relays_401(held, monkeypatch):
    """With no other account the 401 is relayed unchanged, so Claude Code
    learns that its own login needs refreshing (the one documented exception)."""
    from tests.test_wire_hold import exchange, posts as held_posts
    monkeypatch.setattr(account_switch, '_read_pool', lambda: [('A', A)])
    held.proxy._ledger = account_ledger.AccountLedger(resolver=account_ledger.AccountResolver(
        pool=lambda: [('A', A)]))
    held.fake.actions = [reply(401, b'own login expired')]
    raw = exchange(held)
    assert raw.startswith(b'HTTP/1.1 401') and b'own login expired' in raw
    assert [h._bearer_of(r) for r in held_posts(held)] == [A]


# --------------------------------------------------------------------------- #
# The dead-token split (V>>, 2026-10-08): a dead presented token with no
# account account_switch.choose() or the W4 cascade can carry now gets the
# honest exhaustion 429 -- with a real reset -- when another pooled account
# is merely spent and known to come back; otherwise (every other account
# dead too, or no known future reset) the 401 is relayed exactly as before.
# A 403 is never answered this way, on either path.
# --------------------------------------------------------------------------- #
def test_dead_presented_with_only_pooled_account_spent_pauses(dead_wire, monkeypatch):
    """A's token is dead; the only other pool account, B, is merely spent with
    a known future reset -- so the wire pauses with the honest exhaustion 429
    instead of relaying the 401, and auto-continue can resume at that reset."""
    monkeypatch.setattr(account_switch, '_read_pool', lambda: [('B', B)])
    reset = int(time.time()) + 120
    account_switch._EXHAUSTED[spend_guard.account_key(B)] = reset
    dead_wire.fake.answers[A] = reply(401, b'own login expired')
    raw = request()
    out = h._exchange(dead_wire, raw)
    assert out.startswith(b'HTTP/1.1 429')
    assert b'anthropic-ratelimit-unified-representative-claim: five_hour' in out
    assert f'anthropic-ratelimit-unified-reset: {reset}'.encode() in out
    assert b'own login expired' not in out
    assert posts(dead_wire) == [raw]
    rows = dead_rows(dead_wire.home)
    assert len(rows) == 1
    assert {k: rows[0][k] for k in ('account', 'switched', 'outcome', 'status')} == {
        'account': 'A', 'switched': False, 'outcome': 'paused_until_reset', 'status': 401}


def test_dead_presented_with_only_pooled_account_also_dead_relays_401(dead_wire, monkeypatch):
    """A's token is dead; the only other pool account, B, is ALSO dead (a prior
    401), not merely spent -- so there is nothing to pause for and the 401 is
    relayed exactly as before."""
    monkeypatch.setattr(account_switch, '_read_pool', lambda: [('B', B)])
    with dead_wire.proxy._ledger.exchange([('Authorization', 'Bearer ' + B)]) as watch:
        watch.answered(401, [])
    original = reply(401, b'own login expired')
    dead_wire.fake.answers[A] = original
    raw = request()
    assert h._exchange(dead_wire, raw) == original
    assert posts(dead_wire) == [raw]
    rows = dead_rows(dead_wire.home)
    assert len(rows) == 1 and rows[0]['outcome'] == 'original'


def test_dead_presented_403_never_paused_even_with_spent_pool(dead_wire, monkeypatch):
    """A 403 can be a permission or enforcement answer the operator must see,
    so even with B merely spent to a known future reset, the 403 is relayed
    unchanged -- never paused or masked with a reset."""
    monkeypatch.setattr(account_switch, '_read_pool', lambda: [('B', B)])
    account_switch._EXHAUSTED[spend_guard.account_key(B)] = int(time.time()) + 120
    original = reply(403, b'account disabled')
    dead_wire.fake.answers[A] = original
    raw = request()
    assert h._exchange(dead_wire, raw) == original
    assert posts(dead_wire) == [raw]
    rows = dead_rows(dead_wire.home)
    assert len(rows) == 1
    assert rows[0]['outcome'] == 'original' and rows[0]['status'] == 403


def test_hold_dead_presented_with_spent_pool_pauses(held):
    """Same split on the hold path Claude Code's streamed turns take: the only
    other pool account is merely spent to a known future reset, so the hold
    answers the honest exhaustion 429 instead of relaying the dead 401."""
    from tests.test_wire_hold import exchange, posts as held_posts
    reset = int(time.time()) + 120
    account_switch._EXHAUSTED[spend_guard.account_key(B)] = reset
    held.proxy._ledger = account_ledger.AccountLedger(resolver=account_ledger.AccountResolver(
        pool=lambda: [('A', A), ('B', B)]))
    held.fake.actions = [reply(401, b'own login expired')]
    raw = exchange(held)
    assert raw.startswith(b'HTTP/1.1 429')
    assert f'anthropic-ratelimit-unified-reset: {reset}'.encode() in raw
    assert b'own login expired' not in raw
    assert [h._bearer_of(r) for r in held_posts(held)] == [A]
    rows = dead_rows(held.home)
    assert len(rows) == 1 and rows[0]['outcome'] == 'paused_until_reset'


def test_hold_dead_presented_403_never_paused_even_with_spent_pool(held):
    """A 403 on the hold path is never paused either, even with B merely
    spent to a known future reset."""
    from tests.test_wire_hold import exchange, posts as held_posts
    account_switch._EXHAUSTED[spend_guard.account_key(B)] = int(time.time()) + 120
    held.proxy._ledger = account_ledger.AccountLedger(resolver=account_ledger.AccountResolver(
        pool=lambda: [('A', A), ('B', B)]))
    original = reply(403, b'account disabled')
    held.fake.actions = [original]
    raw = exchange(held)
    assert raw == original
    assert [h._bearer_of(r) for r in held_posts(held)] == [A]
    rows = dead_rows(held.home)
    assert len(rows) == 1 and rows[0]['outcome'] == 'original'
