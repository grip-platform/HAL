"""Hold boundaries over real loopback TLS; only obviously fake credentials."""
import json
import socket
import time

import pytest

from lib.hal import account_ledger, account_switch, spend_guard, wire_hold
from tests import test_wire_proxy as h
from tests.test_wire_proxy import home, pki, make_wire
from tests.test_wire_replay import A, B, request, reply

# pytest resolves these fixtures by name in this module; `make_wire` needs `pki`.
FIXTURES = (home, pki, make_wire)


@pytest.fixture(autouse=True)
def _healthy_attempts(monkeypatch):
    from lib.hal.wire_cascade import Attempts
    monkeypatch.setattr(Attempts, '_unhealthy', False, raising=False)


def _turn_of(ex):
    """The W4 registry's key for the turn an exchange carries."""
    return wire_hold.turn_key(ex.headers, json.loads(ex.body.value()))


def _cap(limit=4, taken=0):
    """A real in-flight cap for a proxy double, with ``taken`` of its slots in use."""
    from lib.hal.gateway import InflightCap
    cap = InflightCap(limit, 4)
    assert all(cap.try_enter() for _ in range(taken))
    return cap

EVENTS = (b'event: message_start\ndata: {"type":"message_start","message":{"model":"fake-model","usage":{"input_tokens":1}}}\n\n'
          b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"answer"}}\n\n'
          b'event: message_stop\ndata: {"type":"message_stop"}\n\n')
SUCCESS = (b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n'
           + f'Content-Length: {len(EVENTS)}\r\n\r\n'.encode() + EVENTS)


class Upstream(h.FakeUpstream):
    actions = ()

    def _one(self, tls, reader, number):
        head = reader.until(b'\r\n\r\n')
        if not head:
            return False
        body = reader.exact(h._header_int(head, b'content-length'))
        self.requests.append((number, head + body))
        if head.startswith(b'GET /v1/models'):
            tls.sendall(reply(200, b'{}', b'anthropic-organization-id: fake-org-'
                               + h._bearer_of(head).encode() + b'\r\n'))
            return True
        n = len([r for _, r in self.requests if r.startswith(b'POST')])
        action = self.actions[min(n-1, len(self.actions)-1)]
        if callable(action):
            action(tls)
        elif action is not None:
            tls.sendall(action)
        return False


@pytest.fixture
def held(make_wire, monkeypatch):
    monkeypatch.setattr(h, 'FakeUpstream', Upstream)
    monkeypatch.setattr(Upstream, 'actions', [SUCCESS])
    monkeypatch.setattr(spend_guard, '_OAUTH_PREFIX', 'fake-account-')
    monkeypatch.setattr(account_switch, '_read_pool', lambda: [('A', A), ('B', B)])
    monkeypatch.setattr(account_switch, '_EXHAUSTED', {})
    monkeypatch.setattr(account_switch, '_ORGS', {})
    monkeypatch.setattr(account_switch, '_ORG_RETRY', {})
    for name, value in [('PING_S', .02), ('RETRY_S', .08), ('BUDGET_S', .5), ('DECIDE_S', .04)]:
        monkeypatch.setattr(wire_hold, name, value)
    return make_wire(pool_size=0)


def turn(**kwargs):
    return request(payload=dict(h.TURN, stream=True), **kwargs)


def posts(wire):
    return [raw for _, raw in wire.fake.requests if raw.startswith(b'POST')]


def exchange(wire, raw=None):
    tls = h._tunnel(wire.port, h.API, wire.client)
    tls.settimeout(3)
    try:
        tls.sendall(raw or turn())
        reader = h._Reader(tls)
        head = reader.until(b'\r\n\r\n')
        if b'content-length:' in head.lower():
            return head + reader.exact(h._header_int(head, b'content-length'))
        received = bytearray(head + reader.buf)
        while True:
            part = tls.recv(65536)
            if not part:
                return bytes(received)
            received.extend(part)
    finally:
        tls.close()


def row(wire):
    return h._wait_for(lambda: next((r for r in h._rows(wire.home) if r.get('event') == 'wire.hold'), None))


@pytest.mark.parametrize('status', [429, 500, 502, 503, 529])
def test_rejection_is_held_then_served_without_error_bytes(held, status):
    held.fake.actions = [reply(status, b'{"error":{"message":"temporary"}}'), SUCCESS]
    raw = exchange(held)
    assert raw.startswith(b'HTTP/1.1 200')
    before, after = raw.split(b'event: message_start', 1)
    assert before.count(b'event: ping') >= 2
    assert b'error' not in before and b'temporary' not in raw
    assert b'answer' in after and raw.count(b'event: message_start') == 1
    assert len(posts(held)) == 2 and posts(held)[0] == posts(held)[1]
    record = row(held)
    assert record['outcome'] == 'served' and record['attempts'] == 2
    assert record['signal'] == str(status) and record['seconds'] >= .08
    assert A not in json.dumps(h._rows(held.home))


def test_pings_flow_before_retry_and_retry_after_is_honoured(held, monkeypatch):
    # A budget the slow-runner allowance below cannot exhaust (HAL#685).
    monkeypatch.setattr(wire_hold, 'BUDGET_S', 5.0)
    held.fake.actions = [reply(429, b'{}', b'Retry-After: 1\r\n'), SUCCESS]
    tls = h._tunnel(held.port, h.API, held.client)
    # The head waits on a fresh TLS connection to the fake upstream (no pool),
    # which a shared runner can take longer than 0.25 s to open (HAL#688's
    # first CI run); the pings after it keep the tight timeout.
    tls.settimeout(2.0)
    try:
        tls.sendall(turn())
        reader = h._Reader(tls)
        assert b'200' in reader.until(b'\r\n\r\n')
        tls.settimeout(.25)
        # Pings flow for 0.3 s, well past the hold's own 0.08 s retry step,
        # and still only one request has gone upstream: the hold is waiting
        # on the 1 s Retry-After, not on its default step.
        until = time.monotonic() + .3
        while time.monotonic() < until:
            assert b'event: ping' in reader.until(b'\n\n')
        assert len(posts(held)) == 1
    finally:
        tls.close()


def test_budget_emits_exactly_one_error(held):
    held.fake.actions = [reply(529, b'{}')]
    started = time.monotonic()
    raw = exchange(held)
    assert .45 <= time.monotonic()-started < .9
    assert raw.count(b'event: error') == 1 and b'event: message_start' not in raw
    assert row(held)['outcome'] == 'budget'


@pytest.mark.parametrize('failure', [ConnectionRefusedError, socket.gaierror])
def test_connect_and_dns_failure_retry_before_sending(held, monkeypatch, failure):
    original = held.proxy.pool.take
    calls = []

    def take(**kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise failure('fake upstream unavailable')
        return original(**kwargs)

    monkeypatch.setattr(held.proxy.pool, 'take', take)
    raw = exchange(held)
    assert b'event: ping' in raw and EVENTS in raw and b'event: error' not in raw
    assert len(posts(held)) == 1 and row(held)['attempts'] == 2


def test_delayed_headers_wait_same_request_with_pings(held):
    def delayed(tls):
        time.sleep(.15)
        tls.sendall(SUCCESS)
    held.fake.actions = [delayed]
    raw = exchange(held)
    assert b'event: ping' in raw and EVENTS in raw
    assert len(posts(held)) == 1 and row(held)['signal'] == 'headers_wait'


@pytest.mark.parametrize('status,body', [
    # Neutral bodies: the status rule alone must refuse the hold. A "context too
    # long" message would be refused by the separate context-error check whatever
    # the status rule said, so it cannot show the status rule works.
    (400, b'{"error":{"type":"invalid_request_error","message":"bad request"}}'),
    (401, b'{"error":{"type":"authentication_error","message":"invalid bearer"}}'),
    (413, b'{"error":{"type":"request_too_large","message":"request too large"}}'),
])
def test_request_errors_are_never_held(held, status, body, monkeypatch):
    monkeypatch.setattr(account_switch, "_read_pool", lambda: [])
    response = reply(status, body)
    held.fake.actions = [response]
    assert exchange(held) == response
    assert len(posts(held)) == 1


def test_context_error_even_with_500_is_not_held(held):
    response = reply(500, b'{"error":{"message":"prompt is too long"}}')
    held.fake.actions = [response]
    assert exchange(held) == response
    assert len(posts(held)) == 1


@pytest.mark.parametrize('raw', [request(), request('/v1/code/sessions', dict(h.TURN, stream=True)),
                                  request(payload=dict(h.QUOTA, stream=True))])
def test_nonstream_account_bound_and_aux_are_not_held(held, raw):
    response = reply(529, b'{}')
    held.fake.actions = [response]
    assert exchange(held, raw) == response and len(posts(held)) == 1


def test_off_switch_is_checked_on_next_request(held):
    held.fake.actions = [reply(529, b'{}'), SUCCESS]
    assert b'event: ping' in exchange(held)
    marker = held.home / 'state/hal/wire-hold.off'
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    held.fake.actions = [reply(529, b'{}')]
    assert exchange(held) == reply(529, b'{}')


def test_spent_ledger_answers_without_upstream(held):
    resolver = account_ledger.AccountResolver(pool=lambda: [('A', A), ('B', B)])
    held.proxy._ledger = account_ledger.AccountLedger(resolver=resolver)
    reset = int(time.time()) + 120
    for token in (A, B):
        with held.proxy._ledger.exchange({'authorization': 'Bearer '+token}) as watch:
            watch.answered(429, {'anthropic-ratelimit-unified-status': 'rejected',
                                 'anthropic-ratelimit-unified-reset': str(reset)})
    started = time.monotonic()
    raw = exchange(held)
    assert raw.startswith(b'HTTP/1.1 429') and b'earliest known reset' in raw
    assert not posts(held) and b'event: ping' not in raw
    assert time.monotonic()-started < .2


def test_throttle_switches_after_three_same_account_attempts(held, monkeypatch):
    # Three attempts, two organisation probes and the switched request each
    # open a fresh TLS connection (pool_size=0), which outran the fixture's
    # 0.5 s budget on the Linux runner (run 37582087855). The budget is not
    # what this test checks, and a served turn returns at once.
    monkeypatch.setattr(wire_hold, 'BUDGET_S', 5.0)
    held.fake.actions = [reply(429, b'{}')]*3 + [SUCCESS]
    raw = exchange(held)
    assert EVENTS in raw
    assert [h._bearer_of(r) for r in posts(held)] == [A, A, A, B]


def test_acceptance_uncertain_is_never_replayed(held):
    held.fake.actions = [None, SUCCESS]
    raw = exchange(held)
    assert raw.startswith(b'HTTP/1.1 503')
    assert len(posts(held)) == 1


def test_no_splice_after_content(held):
    cut = SUCCESS.split(b'\r\n\r\n')[0] + b'\r\n\r\n' + EVENTS[:120]
    held.fake.actions = [reply(529, b'{}'), cut, SUCCESS]
    raw = exchange(held)
    assert len(posts(held)) == 2
    assert b'answer' not in raw and raw.count(b'event: message_start') <= 1


def test_chunked_success_is_decoded_after_hold(held):
    chunked = (b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n'
               + f'{len(EVENTS):x}\r\n'.encode() + EVENTS + b'\r\n0\r\n\r\n')
    held.fake.actions = [reply(529, b'{}'), chunked]
    raw = exchange(held)
    assert raw.endswith(EVENTS) and b'Transfer-Encoding' not in raw


def test_budget_also_bounds_wait_for_message_start(held):
    def delay_body(tls):
        tls.sendall(SUCCESS.split(b'\r\n\r\n')[0] + b'\r\n\r\n')
        time.sleep(.7)
    held.fake.actions = [reply(529, b'{}'), delay_body]
    raw = exchange(held)
    assert raw.count(b'event: error') == 1 and row(held)['outcome'] == 'budget'


def test_budget_fallback_is_native_429_once_and_only_for_same_turn(held):
    held.fake.actions = [reply(529, b'{}')]
    assert exchange(held).count(b'event: error') == 1
    count = len(posts(held))
    unrelated = request(payload=dict(h.TURN, messages=[{'role': 'user', 'content': 'other'}]))
    assert exchange(held, unrelated) == reply(529, b'{}')
    held.fake.actions = [SUCCESS]
    native = exchange(held, request())
    assert native.startswith(b'HTTP/1.1 429')
    assert b'anthropic-ratelimit-unified-representative-claim: five_hour' in native
    assert b'api_error' not in native
    assert len(posts(held)) == count + 1
    assert exchange(held, request()) == SUCCESS


def test_fallback_names_earliest_known_reset_and_is_credential_scoped(held):
    payload = dict(h.TURN, stream=True, system=[{'type': 'text', 'text': 'x-anthropic-billing-header: fake-old'},
                                               {'type': 'text', 'text': 'real instructions'}])
    headers = [('authorization', 'Bearer '+A)]
    reset = int(time.time()) + 120
    key = wire_hold.fallback_key(headers, payload)
    held.proxy._hold_fallbacks.put(key, reset)
    fallback = dict(payload, stream=False, max_tokens=32000,
                    system=[{'type': 'text', 'text': 'x-anthropic-billing-header: fake-new'},
                            {'type': 'text', 'text': 'real instructions'}])
    assert exchange(held, request(payload=fallback).replace(A.encode(), B.encode())) == SUCCESS
    native = exchange(held, request(payload=fallback))
    assert f'anthropic-ratelimit-unified-reset: {reset}'.encode() in native
    assert b'earliest known reset' in native
    assert len(posts(held)) == 1


def test_expired_fallback_does_not_intercept_nonstream(held, monkeypatch):
    key = wire_hold.fallback_key([('authorization', 'Bearer '+A)], h.TURN)
    held.proxy._hold_fallbacks.put(key, None)
    entry = held.proxy._hold_fallbacks.pending[key]
    held.proxy._hold_fallbacks.pending[key] = entry._replace(expires=time.monotonic()-1)
    assert exchange(held, request()) == SUCCESS


def test_an_unsafe_fallback_reaches_the_offer_only_as_uncertain(held, monkeypatch):
    """Found unpinned by HAL#748's mutation proof. A pending fallback whose turn may
    already have been accepted (``safe=False``) must reach the W4 offer as
    "uncertain", which refuses it; only a turn known unsent may go as "rejected".
    Otherwise one turn could be answered by a target and by Anthropic."""
    offered = []
    monkeypatch.setattr(wire_hold, 'offer_cascade',
                        lambda proxy, client, ex, evidence: offered.append(evidence) or False)
    headers = [('authorization', 'Bearer ' + A)]
    payload = dict(h.TURN, stream=True)
    fallback = dict(payload, stream=False, max_tokens=32000)
    for safe in (False, True):
        held.proxy._hold_fallbacks.put(wire_hold.fallback_key(headers, payload), None, safe=safe)
        assert b' 429 ' in exchange(held, request(payload=fallback))
    assert offered == ['uncertain', 'rejected']


def test_quota_after_temporary_error_replays_once(held):
    limit = reply(429, b'{}', b'anthropic-ratelimit-unified-status: rejected\r\n')
    held.fake.actions = [reply(529, b'{}'), limit, SUCCESS]
    assert EVENTS in exchange(held)
    assert [h._bearer_of(r) for r in posts(held)] == [A, A, B]


def test_quota_bearing_429_is_not_a_headerless_throttle(held):
    response = reply(429, b'{}', b'anthropic-ratelimit-unified-status: allowed\r\n')
    held.fake.actions = [response]
    assert exchange(held) == response and len(posts(held)) == 1


def test_failed_normal_stream_never_appends_an_http_response(held):
    held.fake.actions = [b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n'
                         b'Transfer-Encoding: chunked\r\n\r\nNOT-HEX\r\n']
    raw = exchange(held)
    assert raw.count(b'HTTP/1.1') == 1 and b'503' not in raw
    assert len(posts(held)) == 1


# --------------------------------------------------------------------------- #
# dead_token_pause / pooled_reset -- the dead-token split (V>>, 2026-10-08):
# a dead presented token with another pooled account merely spent to a known
# future reset gets the honest exhaustion 429 instead of its 401 relayed;
# with no such account (dead too, or no reset known) the 401 stands. Unit
# level, isolated from account_switch.choose/wire_cascade.move's own gating.
# --------------------------------------------------------------------------- #
from types import SimpleNamespace


def _proxy(pool=()):
    return SimpleNamespace(_ledger=account_ledger.AccountLedger(
        resolver=account_ledger.AccountResolver(pool=lambda: pool)))


def test_pooled_reset_future_exhausted_mark_is_found(monkeypatch):
    monkeypatch.setattr(account_switch, '_read_pool', lambda: [('B', B)])
    monkeypatch.setattr(account_switch, '_EXHAUSTED', {})
    reset = int(time.time()) + 300
    account_switch._EXHAUSTED[spend_guard.account_key(B)] = reset
    headers = [('Authorization', 'Bearer ' + A)]
    assert wire_hold.pooled_reset(_proxy(), headers) == reset
    answer = wire_hold.dead_token_pause(_proxy(), headers)
    assert answer is not None and str(reset).encode() in answer
    assert b'anthropic-ratelimit-unified-representative-claim: five_hour' in answer


def test_pooled_reset_past_mark_is_never_invented(monkeypatch):
    monkeypatch.setattr(account_switch, '_read_pool', lambda: [('B', B)])
    monkeypatch.setattr(account_switch, '_EXHAUSTED', {})
    account_switch._EXHAUSTED[spend_guard.account_key(B)] = time.time() - 60
    headers = [('Authorization', 'Bearer ' + A)]
    assert wire_hold.pooled_reset(_proxy(), headers) is None
    assert wire_hold.dead_token_pause(_proxy(), headers) is None


def test_pooled_reset_ledger_cooling_alone_is_found(monkeypatch):
    """No account_switch mark at all -- only the ledger's own cooling (spent),
    the same fallback source :func:`exhaustion` reads."""
    monkeypatch.setattr(account_switch, '_read_pool', lambda: [('B', B)])
    monkeypatch.setattr(account_switch, '_EXHAUSTED', {})
    proxy = _proxy(pool=[('B', B)])
    reset = int(time.time()) + 300
    with proxy._ledger.exchange([('authorization', 'Bearer ' + B)]) as watch:
        watch.answered(429, {'anthropic-ratelimit-unified-status': 'rejected',
                             'anthropic-ratelimit-unified-reset': str(reset)})
    headers = [('Authorization', 'Bearer ' + A)]
    assert wire_hold.pooled_reset(proxy, headers) == reset


def test_pooled_reset_ignores_cooling_once_the_account_is_dead(monkeypatch):
    """A pool account that was rate-limited and has since gone dead needs a
    re-login, not a timer: its stale cooling reset must never be read as a
    promise it will come back on its own."""
    monkeypatch.setattr(account_switch, '_read_pool', lambda: [('B', B)])
    monkeypatch.setattr(account_switch, '_EXHAUSTED', {})
    proxy = _proxy(pool=[('B', B)])
    reset = int(time.time()) + 300
    with proxy._ledger.exchange([('authorization', 'Bearer ' + B)]) as watch:
        watch.answered(429, {'anthropic-ratelimit-unified-status': 'rejected',
                             'anthropic-ratelimit-unified-reset': str(reset)})
    with proxy._ledger.exchange([('authorization', 'Bearer ' + B)]) as watch:
        watch.answered(401, [])
    headers = [('Authorization', 'Bearer ' + A)]
    assert wire_hold.pooled_reset(proxy, headers) is None


def test_pooled_reset_excludes_the_presented_token_itself(monkeypatch):
    """A stale mark on the PRESENTED (dead) token itself is never read back as
    its own resurrection -- only an OTHER pooled account counts."""
    monkeypatch.setattr(account_switch, '_read_pool', lambda: [('A', A)])
    monkeypatch.setattr(account_switch, '_EXHAUSTED', {})
    account_switch._EXHAUSTED[spend_guard.account_key(A)] = int(time.time()) + 300
    headers = [('Authorization', 'Bearer ' + A)]
    assert wire_hold.pooled_reset(_proxy(), headers) is None


def test_pooled_reset_no_pool_at_all_is_none(monkeypatch):
    monkeypatch.setattr(account_switch, '_read_pool', lambda: [])
    headers = [('Authorization', 'Bearer ' + A)]
    assert wire_hold.pooled_reset(_proxy(), headers) is None


def test_a_ledger_entry_whose_token_is_not_an_object_is_skipped(monkeypatch):
    """AGORA on HAL#770: a damaged ledger entry must not cost the client its answer.
    An entry whose ``token`` is a string sits beside B's real cooling; every reader
    of the ledger skips it and still finds B's mark."""
    monkeypatch.setattr(account_switch, '_read_pool', lambda: [('B', B)])
    monkeypatch.setattr(account_switch, '_EXHAUSTED', {})
    monkeypatch.setattr(account_switch, '_ORGS', {})
    proxy = _proxy(pool=[('B', B)])
    reset = int(time.time()) + 300
    with proxy._ledger.exchange([('authorization', 'Bearer ' + B)]) as watch:
        watch.answered(429, {'anthropic-ratelimit-unified-status': 'rejected',
                             'anthropic-ratelimit-unified-reset': str(reset)})
    real = account_ledger.LedgerView.accounts
    monkeypatch.setattr(account_ledger.LedgerView, 'accounts',
                        lambda view: {**real(view), 'damaged': {'token': 'not-an-object'}})
    assert wire_hold.pooled_reset(proxy, [('Authorization', 'Bearer ' + A)]) == reset
    assert wire_hold.carries_spent_mark(proxy, B) is True
    assert wire_hold.exhaustion(proxy, [('Authorization', 'Bearer ' + B)]) == (True, reset)


def test_fallbacks_keep_named_entries_until_they_expire():
    """HAL#748 item 8: a pending terminal answer is a Fallback with named fields."""
    fallbacks = wire_hold.Fallbacks()
    before = time.monotonic()
    fallbacks.put('k', 1792094000.0, safe=True)
    entry = fallbacks.peek('k')
    assert isinstance(entry, wire_hold.Fallback)
    assert (entry.reset, entry.safe) == (1792094000.0, True)
    assert before + wire_hold.FALLBACK_S <= entry.expires <= time.monotonic() + wire_hold.FALLBACK_S
    fallbacks.pending['k'] = entry._replace(expires=time.monotonic() - 1)
    assert fallbacks.peek('k') is None and fallbacks.take('k') is None


@pytest.mark.parametrize('boundary', ['entry', 'loop'])
@pytest.mark.parametrize('served', [False, True])
def test_preheader_spent_hold_offers_one_terminal(monkeypatch, boundary, served):
    from types import SimpleNamespace
    from lib.hal import wire_cascade, wire_proxy
    from tests.test_wire_proxy import _CascadeSocket
    socket = _CascadeSocket()
    client = SimpleNamespace(sock=socket)
    body = wire_proxy._Copy()
    body(json.dumps(dict(h.TURN, stream=True)).encode())
    ex = wire_proxy._Exchange('POST', '/v1/messages',
                              (('x-claude-code-session-id', 'fake-session'),),
                              time.monotonic(), route=wire_hold.RouteClass.INFERENCE,
                              body=body, wire_body=body)
    proxy = SimpleNamespace(_hold_fallbacks=wire_hold.Fallbacks(), _w4_attempts=wire_cascade.Attempts())
    attempts = []
    checks = iter([(False, None), (True, None)] if boundary == 'loop' else [(True, None)])
    monkeypatch.setattr(wire_hold, 'exhaustion', lambda *args: next(checks))

    def fake_move(proxy, client, exchange):
        attempts.append(exchange)
        if served:
            client.sock.sendall(b'TARGET')
        return served

    monkeypatch.setattr(wire_cascade, 'move', fake_move)
    hold = wire_hold.Hold(proxy, client, None, ex, 'fake-key')
    hold.run()
    assert attempts == [ex]
    assert socket.sent == ([b'TARGET'] if served else [wire_hold.exhaustion_response(None)])
    if not served:
        assert (ex.status, ex.completed) == (429, True)


@pytest.mark.parametrize('boundary', ['entry', 'loop'])
def test_ineligible_preheader_hold_never_moves(monkeypatch, boundary):
    from types import SimpleNamespace
    from lib.hal import wire_cascade, wire_proxy
    from tests.test_wire_proxy import _CascadeSocket
    socket = _CascadeSocket()
    body = wire_proxy._Copy()
    turn = dict(h.TURN, stream=True, messages=[
        {'role': 'assistant', 'content': 'earlier'}, *h.TURN['messages']])
    body(json.dumps(turn).encode())
    ex = wire_proxy._Exchange('POST', '/v1/messages',
                              (('x-claude-code-session-id', 'fake-session'),),
                              time.monotonic(), route=wire_hold.RouteClass.INFERENCE,
                              body=body, wire_body=body)
    proxy = SimpleNamespace(_hold_fallbacks=wire_hold.Fallbacks(),
                            _w4_attempts=wire_cascade.Attempts())
    checks = iter([(False, None), (True, None)] if boundary == 'loop' else [(True, None)])
    monkeypatch.setattr(wire_hold, 'exhaustion', lambda *args: next(checks))
    monkeypatch.setattr(wire_cascade, 'move', lambda *args: pytest.fail('ineligible move'))
    wire_hold.Hold(proxy, SimpleNamespace(sock=socket), None, ex, 'fake-key').run()
    assert socket.sent == [wire_hold.exhaustion_response(None)]


@pytest.mark.parametrize('started,evidence', [(1, 'unsent'), (None, 'uncertain')])
def test_started_or_uncertain_hold_never_moves(monkeypatch, started, evidence):
    from types import SimpleNamespace
    from lib.hal import wire_cascade, wire_proxy
    from tests.test_wire_proxy import _CascadeSocket
    socket = _CascadeSocket()
    body = wire_proxy._Copy()
    body(json.dumps(dict(h.TURN, stream=True)).encode())
    ex = wire_proxy._Exchange('POST', '/v1/messages',
                              (('x-claude-code-session-id', 'fake-session'),), time.monotonic(),
                              route=wire_hold.RouteClass.INFERENCE, body=body, wire_body=body)
    proxy = SimpleNamespace(_hold_fallbacks=wire_hold.Fallbacks())
    monkeypatch.setattr(wire_hold, 'exhaustion', lambda *args: (True, None))
    monkeypatch.setattr(wire_cascade, 'move', lambda *args: pytest.fail('unsafe move'))
    hold = wire_hold.Hold(proxy, SimpleNamespace(sock=socket), None, ex, 'fake-key')
    hold.started, hold.send_state, ex.client_started = started, evidence, bool(started)
    hold._loop()
    terminal = [part for part in socket.sent if part != wire_hold.PING]
    assert len(terminal) == 1
    assert terminal[0].startswith(b'event: error' if started else b'HTTP/1.1 429')
    assert proxy._hold_fallbacks.take('fake-key')[2] is (not started and evidence == 'unsent')


def _fresh_hold_exchange():
    from lib.hal import wire_proxy
    body = wire_proxy._Copy()
    body(json.dumps(dict(h.TURN, stream=True)).encode())
    return wire_proxy._Exchange('POST', '/v1/messages',
                                (('x-claude-code-session-id', 'fake-session'),),
                                time.monotonic(), route=wire_hold.RouteClass.INFERENCE,
                                body=body, wire_body=body)


def test_outstanding_anthropic_hold_blocks_w4_until_head(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from lib.hal import wire_cascade, wire_proxy
    ex = _fresh_hold_exchange()
    key = wire_hold.fallback_key(ex.headers, json.loads(ex.body.value()))
    sent = []
    head = b'HTTP/1.1 429 Limit\r\nContent-Length: 0\r\n\r\n'
    class Stream:
        def read_head(self):
            assert not wire_hold.offer_cascade(proxy, None, _fresh_hold_exchange(), 'unsent')
            return head
    sock = SimpleNamespace(settimeout=lambda *_: None, sendall=lambda data: sent.append(data))
    conn = SimpleNamespace(sock=sock, stream=Stream())
    proxy = SimpleNamespace(pool=SimpleNamespace(take=lambda **_: conn),
        _ledger=SimpleNamespace(exchange=lambda *_: nullcontext(
            SimpleNamespace(answered=lambda *_: None))),
        _judge=lambda *_: None, _w4_attempts=wire_cascade.Attempts())
    moves = []
    monkeypatch.setattr(wire_cascade, 'move', lambda *args: moves.append(1) or False)
    hold = wire_hold.Hold(proxy, None, SimpleNamespace(raw=b'POST /v1/messages\r\n\r\n'), ex, key)
    assert wire_proxy._status(hold._answer()) == 429
    assert moves == []
    assert not wire_hold.offer_cascade(proxy, None, _fresh_hold_exchange(), 'unsent')
    assert moves == [1]


def test_closure_during_w4_reservation_survives_release(monkeypatch):
    from types import SimpleNamespace
    from lib.hal import wire_cascade
    proxy = SimpleNamespace(_w4_attempts=wire_cascade.Attempts())
    calls = []
    def move(proxy, client, ex):
        calls.append(1)
        wire_hold.close_turn(proxy, ex)
        return False
    monkeypatch.setattr(wire_cascade, 'move', move)
    assert not wire_hold.offer_cascade(proxy, None, _fresh_hold_exchange(), 'unsent')
    assert not wire_hold.offer_cascade(proxy, None, _fresh_hold_exchange(), 'unsent')
    assert calls == [1]


def test_presend_connect_timeout_releases_turn_for_second_offer(monkeypatch):
    from types import SimpleNamespace
    from lib.hal import wire_cascade
    proxy = SimpleNamespace(_w4_attempts=wire_cascade.Attempts())
    calls = []
    target = SimpleNamespace(label='fake', window=100)
    monkeypatch.setattr(wire_cascade, 'transform', lambda *args: {})
    monkeypatch.setattr(wire_cascade.policy, 'prepare', lambda *args: {})
    monkeypatch.setattr(wire_cascade, '_request', lambda *args: (None, b''))
    monkeypatch.setattr(wire_cascade, '_record', lambda *args: None)
    def connect(*args):
        calls.append(1)
        raise TimeoutError('no request byte sent')
    monkeypatch.setattr(wire_cascade, '_connect', connect)
    def move(proxy, client, ex):
        wire_cascade._attempt(proxy, client, ex, target, 'fake', {}, 'refused')
        return False
    monkeypatch.setattr(wire_cascade, 'move', move)
    for _ in range(2):
        ex = _fresh_hold_exchange()
        assert not wire_hold.offer_cascade(proxy, None, ex, 'unsent')
        assert not ex.w4_uncertain
    assert calls == [1, 1]


def test_actual_hold_headers_and_ping_preclude_move(monkeypatch):
    from types import SimpleNamespace
    from lib.hal import wire_cascade
    from tests.test_wire_proxy import _CascadeSocket
    ex = _fresh_hold_exchange()
    socket = _CascadeSocket()
    proxy = SimpleNamespace(_hold_fallbacks=wire_hold.Fallbacks())
    monkeypatch.setattr(wire_hold, 'exhaustion', lambda *args: (True, None))
    monkeypatch.setattr(wire_cascade, 'move', lambda *args: pytest.fail('ping moved'))
    hold = wire_hold.Hold(proxy, SimpleNamespace(sock=socket), None, ex, 'fake-key')
    hold.begin('headers_wait')
    hold._loop()
    assert len(socket.sent) == 3 and socket.sent[:2] == [wire_hold.HEAD, wire_hold.PING]
    assert socket.sent[-1].startswith(b'event: error\ndata: ')
    assert proxy._hold_fallbacks.take('fake-key')[2] is False


@pytest.mark.parametrize('failure', ['timeout', 'reset'])
def test_hold_send_timeout_never_moves_or_stores_safe_fallback(monkeypatch, failure):
    import socket as sockets
    from types import SimpleNamespace
    from lib.hal import wire_cascade, wire_proxy
    from tests.test_wire_proxy import _CascadeSocket
    ex = _fresh_hold_exchange()
    client, upstream = _CascadeSocket(), []
    def partial_send(data):
        upstream.append(data)
        raise (sockets.timeout('acceptance unknown') if failure == 'timeout'
               else ConnectionResetError('acceptance unknown'))
    sock = SimpleNamespace(settimeout=lambda *_: None, sendall=partial_send,
                           shutdown=lambda *_: None, close=lambda: None)
    from lib.hal.wire_cascade import Attempts
    proxy = SimpleNamespace(pool=SimpleNamespace(take=lambda **_: SimpleNamespace(sock=sock)),
                            _hold_fallbacks=wire_hold.Fallbacks(), _w4_attempts=Attempts())
    checks = iter([(False, None), (False, None)])
    monkeypatch.setattr(wire_hold, 'exhaustion', lambda *args: next(checks))
    monkeypatch.setattr(wire_cascade, 'move', lambda *args: pytest.fail('timeout moved'))
    request = SimpleNamespace(raw=b'POST /v1/messages HTTP/1.1\r\n\r\n')
    key = wire_hold.fallback_key(ex.headers, json.loads(ex.body.value()))
    wire_hold.Hold(proxy, SimpleNamespace(sock=client), request, ex, key).run()
    assert len(upstream) == 1 and client.sent == [wire_proxy._NO_ANSWER]
    assert proxy._hold_fallbacks.take(key) is None
    assert not proxy._w4_attempts.reserve(_turn_of(ex))
    assert not wire_hold.offer_cascade(proxy, None, _fresh_hold_exchange(), 'unsent')


def test_answered_hold_rejection_does_not_close_turn(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from lib.hal import wire_cascade, wire_proxy
    ex = _fresh_hold_exchange()
    key = wire_hold.fallback_key(ex.headers, json.loads(ex.body.value()))
    raw = b'HTTP/1.1 429 Limit\r\nContent-Length: 0\r\n\r\n'
    conn = SimpleNamespace(sock=SimpleNamespace(settimeout=lambda *_: None,
        sendall=lambda *_: None, shutdown=lambda *_: None, close=lambda: None),
        stream=SimpleNamespace(read_head=lambda: raw))
    proxy = SimpleNamespace(pool=SimpleNamespace(take=lambda **_: conn),
        _ledger=SimpleNamespace(exchange=lambda *_: nullcontext(
            SimpleNamespace(answered=lambda *_: None))),
        _judge=lambda *_: None, _w4_attempts=wire_cascade.Attempts())
    monkeypatch.setattr(wire_hold, 'exhaustion', lambda *args: (False, None))
    hold = wire_hold.Hold(proxy, SimpleNamespace(sock=SimpleNamespace()),
                          SimpleNamespace(raw=b'POST /v1/messages\r\n\r\n'), ex, key)
    monkeypatch.setattr(hold, '_handle_answer', lambda head: wire_proxy._status(head) == 429)
    hold.run()
    assert proxy._w4_attempts.reserve(_turn_of(ex))


def test_hold_brackets_its_send_with_the_key_it_was_given(monkeypatch):
    """`prepare()` works the turn's key out once and hands it to the hold. The hold
    brackets its send with that key and never parses the body again for it."""
    from contextlib import nullcontext
    from types import SimpleNamespace
    from lib.hal import wire_cascade, wire_proxy
    ex, given, seen = _fresh_hold_exchange(), 'ab' * 32, []
    attempts = wire_cascade.Attempts()
    raw = b'HTTP/1.1 429 Limit\r\nContent-Length: 0\r\n\r\n'

    def read_head():
        seen.append(dict(attempts._outstanding))
        return raw

    conn = SimpleNamespace(sock=SimpleNamespace(settimeout=lambda *_: None,
        sendall=lambda *_: None, shutdown=lambda *_: None, close=lambda: None),
        stream=SimpleNamespace(read_head=read_head))
    proxy = SimpleNamespace(pool=SimpleNamespace(take=lambda **_: conn),
        _ledger=SimpleNamespace(exchange=lambda *_: nullcontext(
            SimpleNamespace(answered=lambda *_: None))),
        _judge=lambda *_: None, _w4_attempts=attempts)
    monkeypatch.setattr(wire_hold, 'exhaustion', lambda *args: (False, None))
    monkeypatch.setattr(wire_hold, 'exchange_key', lambda ex: pytest.fail('parsed the body again'))
    hold = wire_hold.Hold(proxy, SimpleNamespace(sock=SimpleNamespace()),
                          SimpleNamespace(raw=b'POST /v1/messages\r\n\r\n'), ex, 'fake-key',
                          turn=given)
    monkeypatch.setattr(hold, '_handle_answer', lambda head: wire_proxy._status(head) == 429)
    hold.run()
    assert seen == [{given: 1}] and attempts._outstanding == {}


def test_offer_is_refused_while_the_session_has_a_request_on_its_way(monkeypatch):
    """A streamed request marks its session until its send is settled. An offer for a
    turn of that session passes the session to the reservation, which refuses it."""
    from types import SimpleNamespace
    from lib.hal import wire_cascade
    proxy = SimpleNamespace(_hold_fallbacks=wire_hold.Fallbacks(),
                            _w4_attempts=wire_cascade.Attempts())
    moved = []
    monkeypatch.setattr(wire_cascade, 'move', lambda *args: moved.append(1) or True)
    proxy._w4_attempts.begin('fake-session')
    assert not wire_hold.offer_cascade(proxy, None, _fresh_hold_exchange(), 'unsent')
    proxy._w4_attempts.end('fake-session')
    assert wire_hold.offer_cascade(proxy, None, _fresh_hold_exchange(), 'unsent')
    assert moved == [1]


def test_budget_fallback_cannot_offer_cascade(monkeypatch):
    from types import SimpleNamespace
    from lib.hal import wire_cascade, wire_proxy
    from tests.test_wire_proxy import _CascadeSocket
    payload = json.dumps(dict(h.TURN, stream=False)).encode()
    header = (f'POST /v1/messages HTTP/1.1\r\nContent-Length: {len(payload)}\r\n'
              'x-claude-code-session-id: fake-session\r\n\r\n').encode()
    request_head = wire_proxy._parse_head(header)
    socket = _CascadeSocket(payload)
    client = wire_proxy._Stream(socket)
    fallbacks = wire_hold.Fallbacks()
    key = wire_hold.fallback_key(request_head.headers, json.loads(payload))
    fallbacks.put(key, None, safe=False)
    proxy = SimpleNamespace(_hold_fallbacks=fallbacks, _record=lambda ex: None, _cap=_cap())
    monkeypatch.setattr(wire_hold, 'enabled', lambda: True)
    monkeypatch.setattr(wire_cascade, 'move', lambda *args: pytest.fail('timeout moved'))
    assert wire_hold.consume_fallback(proxy, client, request_head)
    assert len(socket.sent) == 1 and socket.sent[0] == wire_hold.exhaustion_response(None)


@pytest.mark.parametrize('served', [False, True])
def test_early_stream_exhaustion_uses_one_terminal(monkeypatch, served):
    from types import SimpleNamespace
    from lib.hal import wire_cascade, wire_proxy
    from tests.test_wire_proxy import _CascadeSocket
    payload = json.dumps(dict(h.TURN, stream=True)).encode()
    raw = (f'POST /v1/messages HTTP/1.1\r\nContent-Length: {len(payload)}\r\n'
           'x-claude-code-session-id: fake-session\r\n\r\n').encode()
    request = wire_proxy._parse_head(raw)
    socket, rows = _CascadeSocket(payload), []
    proxy = SimpleNamespace(_hold_fallbacks=wire_hold.Fallbacks(), _cap=_cap(),
                            _w4_attempts=wire_cascade.Attempts(), _record=rows.append)
    monkeypatch.setattr(wire_hold, 'enabled', lambda: True)
    monkeypatch.setattr(wire_hold, 'exhaustion', lambda *args: (True, None))
    def fake_move(proxy, client, ex):
        if served:
            client.sock.sendall(b'TARGET')
        return served
    monkeypatch.setattr(wire_cascade, 'move', fake_move)
    assert wire_hold.consume_fallback(proxy, wire_proxy._Stream(socket), request)
    assert socket.sent == ([b'TARGET'] if served else [wire_hold.exhaustion_response(None)])
    assert len(rows) == 1 and (served or rows[0].completed)


@pytest.mark.parametrize('full', [False, True])
def test_a_terminal_offer_runs_inside_an_in_flight_slot_or_not_at_all(monkeypatch, full):
    """Codex on HAL#747: the terminal path offered W4 without a slot of the in-flight
    cap, so a silent target could be handed any number of waiting connections. The
    offer now holds a slot while it runs. With no slot free there is no offer, and the
    client gets the exhaustion answer it is owed."""
    from types import SimpleNamespace
    from lib.hal import wire_cascade, wire_proxy
    from tests.test_wire_proxy import _CascadeSocket
    payload = json.dumps(dict(h.TURN, stream=True)).encode()
    request = wire_proxy._parse_head(
        (f'POST /v1/messages HTTP/1.1\r\nContent-Length: {len(payload)}\r\n'
         'x-claude-code-session-id: fake-session\r\n\r\n').encode())
    socket, during = _CascadeSocket(payload), []
    cap = _cap(limit=1, taken=1 if full else 0)
    proxy = SimpleNamespace(_hold_fallbacks=wire_hold.Fallbacks(), _cap=cap,
                            _w4_attempts=wire_cascade.Attempts(), _record=lambda ex: None)

    def move(proxy, client, ex):
        during.append(cap.inflight)
        client.sock.sendall(b'TARGET')
        return True

    monkeypatch.setattr(wire_hold, 'enabled', lambda: True)
    monkeypatch.setattr(wire_hold, 'exhaustion', lambda *args: (True, None))
    monkeypatch.setattr(wire_cascade, 'move', move)
    assert wire_hold.consume_fallback(proxy, wire_proxy._Stream(socket), request)
    if full:
        assert during == [] and socket.sent == [wire_hold.exhaustion_response(None)]
        assert not proxy._w4_attempts._path(wire_hold.turn_key(request.headers,
                                                              json.loads(payload))).exists()
    else:
        assert during == [1] and socket.sent == [b'TARGET']
    assert cap.inflight == (1 if full else 0)


def test_started_exchange_never_offers_cascade(monkeypatch):
    from types import SimpleNamespace
    from lib.hal import wire_cascade, wire_proxy
    from tests.test_wire_proxy import _CascadeSocket
    body = wire_proxy._Copy()
    body(json.dumps(h.TURN).encode())
    ex = wire_proxy._Exchange('POST', '/v1/messages',
                              (('x-claude-code-session-id', 'fake-session'),),
                              time.monotonic(), route=wire_hold.RouteClass.INFERENCE,
                              body=body, wire_body=body, client_started=True)
    socket = _CascadeSocket()
    monkeypatch.setattr(wire_cascade, 'move', lambda *args: pytest.fail('started move'))
    assert not wire_hold.offer_cascade(None, SimpleNamespace(sock=socket), ex, 'rejected')
    assert not socket.sent


@pytest.mark.parametrize('spent', [False, True])
def test_expired_hold_budget_stays_closed(monkeypatch, spent):
    from types import SimpleNamespace
    from lib.hal import wire_cascade, wire_proxy
    from tests.test_wire_proxy import _CascadeSocket
    body = wire_proxy._Copy()
    body(json.dumps(dict(h.TURN, stream=True)).encode())
    ex = wire_proxy._Exchange('POST', '/v1/messages',
                              (('x-claude-code-session-id', 'fake-session'),),
                              time.monotonic()-10, route=wire_hold.RouteClass.INFERENCE,
                              body=body, wire_body=body)
    socket = _CascadeSocket()
    proxy = SimpleNamespace(_hold_fallbacks=wire_hold.Fallbacks(),
                            _w4_attempts=wire_cascade.Attempts())
    monkeypatch.setattr(wire_hold, 'exhaustion', lambda *args: (spent, None))
    monkeypatch.setattr(wire_cascade, 'move', lambda *args: pytest.fail('expired move'))
    hold = wire_hold.Hold(proxy, SimpleNamespace(sock=socket), None, ex, 'fake-key')
    hold.deadline = time.monotonic()-1
    hold.run()
    assert socket.sent == [wire_hold.exhaustion_response(None)]
    assert (ex.status, ex.completed) == (429, True)
    assert proxy._hold_fallbacks.take('fake-key')[2] is False


def _timed_out_cascade(monkeypatch, attempts):
    from types import SimpleNamespace
    from lib.hal import wire_cascade
    target = SimpleNamespace(label='fake', window=99999)
    monkeypatch.setattr(wire_cascade, 'configured_targets', lambda: [target])
    monkeypatch.setattr(wire_cascade, 'eligible', lambda ex: {})
    monkeypatch.setattr(wire_cascade, 'independent', lambda target: 'fake-key')
    monkeypatch.setattr(wire_cascade, '_sign_decision', lambda *args: True)
    monkeypatch.setattr(wire_cascade, '_record', lambda *args: None)
    monkeypatch.setattr(wire_cascade, '_connect', lambda *args: SimpleNamespace(
        sock=SimpleNamespace(shutdown=lambda *_: None, close=lambda: None)))
    monkeypatch.setattr(wire_cascade, '_request', lambda *args: (None, b''))
    monkeypatch.setattr(wire_cascade, 'transform', lambda *args: {})
    monkeypatch.setattr(wire_cascade.policy, 'prepare', lambda *args: {})
    def timeout(*args):
        attempts.append(1)
        args[2].w4_sending = True
        raise OSError('target acceptance unknown')
    monkeypatch.setattr(wire_cascade, '_exchange', timeout)


def test_w4_timeout_fallback_cannot_offer_same_turn_twice(monkeypatch):
    from types import SimpleNamespace
    from lib.hal import wire_proxy
    from tests.test_wire_proxy import _CascadeSocket
    ex = _fresh_hold_exchange()
    key = wire_hold.fallback_key(ex.headers, json.loads(ex.body.value()))
    from lib.hal.wire_cascade import Attempts
    proxy = SimpleNamespace(_hold_fallbacks=wire_hold.Fallbacks(), _cap=_cap(),
                            _w4_attempts=Attempts(), _record=lambda ex: None)
    checks = iter([(False, None), (True, None)])
    monkeypatch.setattr(wire_hold, 'exhaustion', lambda *args: next(checks, (True, None)))
    attempts = []
    _timed_out_cascade(monkeypatch, attempts)
    wire_hold.Hold(proxy, SimpleNamespace(sock=_CascadeSocket()), None, ex, key).run()
    stored_safe = proxy._hold_fallbacks.pending[key].safe
    payload = json.dumps(dict(h.TURN, stream=False)).encode()
    raw = (f'POST /v1/messages HTTP/1.1\r\nContent-Length: {len(payload)}\r\n'
           'x-claude-code-session-id: fake-session\r\n\r\n').encode()
    socket = _CascadeSocket(payload)
    assert wire_hold.consume_fallback(proxy, wire_proxy._Stream(socket), wire_proxy._parse_head(raw))
    assert attempts == [1] and stored_safe is False
    assert socket.sent == [wire_hold.exhaustion_response(None)]
    for _ in range(2):
        assert not wire_hold.offer_cascade(proxy, SimpleNamespace(sock=_CascadeSocket()),
                                           _fresh_hold_exchange(), 'unsent')
    assert attempts == [1]


def test_uncertain_attempt_survives_consumption_time_and_restart(monkeypatch):
    from types import SimpleNamespace
    from lib.hal import wire_cascade, wire_proxy
    from tests.test_wire_proxy import _CascadeSocket
    clock, sent = [0.0], []
    monkeypatch.setattr(wire_hold.time, 'monotonic', lambda: clock[0])
    proxy = SimpleNamespace(_hold_fallbacks=wire_hold.Fallbacks(),
                            _w4_attempts=wire_cascade.Attempts())
    def move(_proxy, _client, ex):
        sent.append(1)
        ex.w4_uncertain = True
        return False
    monkeypatch.setattr(wire_cascade, 'move', move)
    key = wire_hold.fallback_key(_fresh_hold_exchange().headers,
                                 json.loads(_fresh_hold_exchange().body.value()))
    assert not wire_hold.offer_cascade(proxy, None, _fresh_hold_exchange(), 'unsent')
    assert (proxy._w4_attempts._path(_turn_of(_fresh_hold_exchange()))).is_file()
    for step in range(4):
        if step == 1:
            proxy._hold_fallbacks.put(key, None, safe=False)
            assert proxy._hold_fallbacks.take(key) is not None
        if step == 2:
            clock[0] += 61
        if step == 3:
            proxy._w4_attempts = wire_cascade.Attempts()
        assert not wire_hold.offer_cascade(proxy, None, _fresh_hold_exchange(), 'unsent')
    assert sent == [1]


def test_missing_registry_or_turn_key_never_offers(monkeypatch):
    from types import SimpleNamespace
    from lib.hal import wire_cascade
    monkeypatch.setattr(wire_cascade, 'move', lambda *args: pytest.fail('unsafe offer'))
    proxy = SimpleNamespace(_hold_fallbacks=wire_hold.Fallbacks())
    assert not wire_hold.offer_cascade(proxy, None, _fresh_hold_exchange(), 'unsent')
    proxy._w4_attempts = wire_cascade.Attempts()
    ex = _fresh_hold_exchange()
    ex.body.value = lambda: b'not json'
    monkeypatch.setattr(wire_cascade, 'eligible', lambda ex: {})
    assert not wire_hold.offer_cascade(proxy, None, ex, 'unsent')


def _exchange_for(session, bearer):
    """A fresh exchange for the standard turn, from one session, sent with one credential."""
    from lib.hal import wire_proxy
    body = wire_proxy._Copy()
    body(json.dumps(dict(h.TURN, stream=True)).encode())
    headers = (('x-claude-code-session-id', session), ('authorization', 'Bearer ' + bearer))
    return wire_proxy._Exchange('POST', '/v1/messages', headers, time.monotonic(),
                                route=wire_hold.RouteClass.INFERENCE, body=body, wire_body=body)


def test_turn_key_follows_the_session_not_the_credential():
    """The wire may send a retry with another pooled credential: it is the same turn.
    Two sessions that send the same content are two turns."""
    first, switched, other = (_exchange_for('fake-session', A), _exchange_for('fake-session', B),
                              _exchange_for('other-session', A))
    assert _turn_of(first) == _turn_of(switched) != _turn_of(other)
    payload = json.loads(first.body.value())
    assert (wire_hold.fallback_key(first.headers, payload)
            != wire_hold.fallback_key(switched.headers, payload))
    assert wire_hold.turn_key((('authorization', 'Bearer ' + A),), payload) is None
    # Claude's non-streaming fallback clamps max_tokens: still the same turn.
    assert _turn_of(first) == wire_hold.turn_key(first.headers,
                                                 dict(payload, stream=False, max_tokens=32000))
    # INVERTED for HAL#748 item 7: this used to require that a one-token payload kept
    # the turn's key. That payload is an account probe, so an unanswered probe closed
    # the key of a real turn with the same messages. A probe now has a key of its own.
    assert _turn_of(first) != wire_hold.turn_key(first.headers,
                                                 dict(payload, stream=False, max_tokens=1))


def test_an_unanswered_probe_never_closes_the_turn_it_repeats(tmp_path, monkeypatch):
    """HAL#748 item 7 (Codex, P2): the probe's closure is the probe's alone."""
    from lib.hal import wire_cascade
    monkeypatch.setenv('HAL_HOME', str(tmp_path))
    monkeypatch.setattr(wire_cascade.Attempts, '_unhealthy', False, raising=False)
    exchange = _exchange_for('fake-session', A)
    payload = json.loads(exchange.body.value())
    turn = wire_hold.turn_key(exchange.headers, payload)
    probe = wire_hold.turn_key(exchange.headers, dict(payload, stream=False, max_tokens=1))
    attempts = wire_cascade.Attempts()
    attempts.close(probe)
    assert attempts.closed(probe) and not attempts.closed(turn)
    assert attempts.reserve(turn)
    # The credential-scoped fallback key tells them apart the same way.
    assert (wire_hold.fallback_key(exchange.headers, dict(payload, max_tokens=1))
            != wire_hold.fallback_key(exchange.headers, dict(payload, stream=False, max_tokens=32000)))
    # A real turn's key is exactly what it was before this fix, so closures already on
    # disk keep matching: session id, NUL, then the turn without stream or max_tokens.
    import hashlib
    unchanged = json.dumps({'model': 'claude-opus-5-5',
                            'messages': [{'role': 'user', 'content': 'hello'}]},
                           sort_keys=True, separators=(',', ':'))
    assert turn == hashlib.sha256(('fake-session\0' + unchanged).encode()).hexdigest()


def _closing_proxy(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from lib.hal import wire_cascade
    monkeypatch.setenv('HAL_HOME', str(tmp_path))
    monkeypatch.setattr(wire_cascade.Attempts, '_unhealthy', False, raising=False)
    return SimpleNamespace(_w4_attempts=wire_cascade.Attempts())


def _unanswered_send(proxy, headers, payload):
    """Run one send that got no answer through the wire's own closing path."""
    from lib.hal import wire_proxy
    body = wire_proxy._Copy()
    body(json.dumps(payload).encode())
    ex = wire_proxy._Exchange('POST', '/v1/messages', headers, time.monotonic(),
                              body=body, wire_body=body)
    wire_proxy._close_uncertain_turn(proxy, ex, answer_seen=False)


def test_a_hinted_one_token_turn_keeps_one_key_with_its_fallback(tmp_path, monkeypatch):
    """HAL#765, Codex P1. The gateway hint header names a one-token request a turn, as
    the classifier does, so it is not marked as a probe: its non-streaming fallback
    with a larger max_tokens has the same key, and the closure an unanswered send
    writes still stops that fallback from being offered to W4."""
    proxy = _closing_proxy(tmp_path, monkeypatch)
    hinted = (('x-claude-code-session-id', 'fake-session'), ('authorization', 'Bearer ' + A),
              ('x-claude-code-request-class', 'main'))
    one_token = dict(h.TURN, stream=True, max_tokens=1)
    fallback = dict(h.TURN, stream=False, max_tokens=32000)
    assert wire_hold.turn_key(hinted, one_token) == wire_hold.turn_key(hinted, fallback)
    assert wire_hold.fallback_key(hinted, one_token) == wire_hold.fallback_key(hinted, fallback)
    _unanswered_send(proxy, hinted, one_token)
    assert proxy._w4_attempts.closed(wire_hold.turn_key(hinted, fallback))


def test_an_unanswered_probe_through_the_closing_path_leaves_the_turn_open(tmp_path, monkeypatch):
    """AGORA on HAL#765: the probe's closure, written by the wire's own path."""
    proxy = _closing_proxy(tmp_path, monkeypatch)
    plain = (('x-claude-code-session-id', 'fake-session'), ('authorization', 'Bearer ' + A))
    _unanswered_send(proxy, plain, dict(h.TURN, max_tokens=1))
    assert proxy._w4_attempts.closed(wire_hold.turn_key(plain, dict(h.TURN, max_tokens=1)))
    turn = wire_hold.turn_key(plain, dict(h.TURN, stream=True))
    assert not proxy._w4_attempts.closed(turn) and proxy._w4_attempts.reserve(turn)


def test_uncertain_turn_stays_closed_when_the_retry_carries_another_credential(monkeypatch):
    """An account switch changes the bearer between a turn and its retry. The retry is
    still recognised, and another session's identical turn is still served."""
    from types import SimpleNamespace
    from lib.hal import wire_cascade
    proxy = SimpleNamespace(_hold_fallbacks=wire_hold.Fallbacks(),
                            _w4_attempts=wire_cascade.Attempts())
    moved = []

    def move(_proxy, _client, ex):
        moved.append(dict(ex.headers)['x-claude-code-session-id'])
        ex.w4_uncertain = len(moved) == 1
        return len(moved) > 1

    monkeypatch.setattr(wire_cascade, 'move', move)
    assert not wire_hold.offer_cascade(proxy, None, _exchange_for('fake-session', A), 'unsent')
    assert not wire_hold.offer_cascade(proxy, None, _exchange_for('fake-session', B), 'unsent')
    assert wire_hold.offer_cascade(proxy, None, _exchange_for('other-session', A), 'unsent')
    assert moved == ['fake-session', 'other-session']


def test_registry_blocks_retry_without_fallback_store(monkeypatch):
    from types import SimpleNamespace
    from lib.hal import wire_cascade
    proxy = SimpleNamespace(_w4_attempts=wire_cascade.Attempts())
    sent = []
    def move(_proxy, _client, ex):
        sent.append(1)
        ex.w4_uncertain = True
        return False
    monkeypatch.setattr(wire_cascade, 'move', move)
    assert not wire_hold.offer_cascade(proxy, None, _fresh_hold_exchange(), 'unsent')
    assert not wire_hold.offer_cascade(proxy, None, _fresh_hold_exchange(), 'unsent')
    assert sent == [1]


def test_overlapping_retry_cannot_enter_second_move(monkeypatch):
    from types import SimpleNamespace
    from lib.hal import wire_cascade
    proxy = SimpleNamespace(_hold_fallbacks=wire_hold.Fallbacks(),
                            _w4_attempts=wire_cascade.Attempts())
    entered = []
    def move(*args):
        entered.append(1)
        assert not wire_hold.offer_cascade(proxy, None, _fresh_hold_exchange(), 'unsent')
        return False
    monkeypatch.setattr(wire_cascade, 'move', move)
    assert not wire_hold.offer_cascade(proxy, None, _fresh_hold_exchange(), 'unsent')
    assert entered == [1]


@pytest.mark.parametrize('outcome', ['before_send', 'non_2xx'])
def test_explicit_target_refusal_releases_turn(monkeypatch, outcome):
    from types import SimpleNamespace
    from lib.hal import wire_cascade, wire_proxy
    proxy = SimpleNamespace(_hold_fallbacks=wire_hold.Fallbacks(),
                            _w4_attempts=wire_cascade.Attempts())
    reached = []
    monkeypatch.setattr(wire_cascade, 'independent', lambda target: 'target-key')
    monkeypatch.setattr(wire_cascade, '_sign_decision', lambda *args: True)
    monkeypatch.setattr(wire_cascade, '_record', lambda *args: None)
    monkeypatch.setattr(wire_cascade, 'transform', lambda *args: {})
    monkeypatch.setattr(wire_cascade.policy, 'prepare', lambda *args: {})
    def request(*args):
        reached.append(outcome)
        if outcome == 'before_send':
            raise wire_cascade.policy.Refused('test_refusal')
        return None, b''
    monkeypatch.setattr(wire_cascade, '_request', request)
    sock = SimpleNamespace(shutdown=lambda *_: None, close=lambda: None)
    monkeypatch.setattr(wire_cascade, '_connect', lambda *args: SimpleNamespace(sock=sock))
    head = wire_proxy._parse_head(b'HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n')
    monkeypatch.setattr(wire_cascade, '_exchange', lambda *args: (head, None))
    target = SimpleNamespace(label='target', window=100)
    monkeypatch.setattr(wire_cascade, 'move', lambda p, c, ex: wire_cascade._serve(
        p, c, ex, {}, [target]))
    assert not wire_hold.offer_cascade(proxy, None, _fresh_hold_exchange(), 'unsent')
    assert not wire_hold.offer_cascade(proxy, None, _fresh_hold_exchange(), 'unsent')
    assert reached == [outcome, outcome]


@pytest.mark.parametrize('failure', ['binding', 'relay'])
def test_accepted_target_remains_closed_after_failure(monkeypatch, failure):
    from types import SimpleNamespace
    from lib.hal import wire_cascade, wire_proxy
    proxy = SimpleNamespace(_hold_fallbacks=wire_hold.Fallbacks(),
                            _w4_attempts=wire_cascade.Attempts())
    reached = []
    monkeypatch.setattr(wire_cascade, 'independent', lambda target: 'target-key')
    monkeypatch.setattr(wire_cascade, '_sign_decision', lambda *args: True)
    conn = SimpleNamespace(sock=SimpleNamespace(shutdown=lambda *_: None, close=lambda: None))
    head = wire_proxy._parse_head(b'HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n')
    def attempt(*args):
        reached.append(1)
        return (conn, head), None
    monkeypatch.setattr(wire_cascade, '_attempt', attempt)
    monkeypatch.setattr(wire_cascade, '_bind_first_move', lambda *args: failure != 'binding')
    def broken_relay(client, ex, conn, answer):
        ex.client_started = True
        return False
    monkeypatch.setattr(wire_cascade, '_relay_answer', broken_relay)
    monkeypatch.setattr(wire_cascade, 'move', lambda p, c, ex: wire_cascade._serve(
        p, c, ex, {}, [SimpleNamespace(label='target')]))
    assert not wire_hold.offer_cascade(proxy, None, _fresh_hold_exchange(), 'unsent')
    assert not wire_hold.offer_cascade(proxy, None, _fresh_hold_exchange(), 'unsent')
    assert reached == [1]


def _unsafe_entry_request(route, payload):
    from lib.hal import wire_proxy
    body = json.dumps(payload).encode()
    chunked = route == 'chunked'
    wire = f'{len(body):x}\r\n'.encode() + body + b'\r\n0\r\n\r\n' if chunked else body
    framing = b'Transfer-Encoding: chunked\r\n' if chunked else f'Content-Length: {len(wire)}\r\n'.encode()
    request = wire_proxy._parse_head(b'POST /v1/messages HTTP/1.1\r\n'
        + b'x-claude-code-session-id: fake-session\r\n' + framing + b'\r\n')
    return request, wire


@pytest.mark.parametrize('route', ['streaming', 'chunked', 'holds_off'])
def test_unsafe_fallback_blocks_every_offer_route(monkeypatch, route):
    from types import SimpleNamespace
    from lib.hal import spend_guard, wire_cascade, wire_proxy
    from tests.test_wire_proxy import _CascadeSocket
    payload = dict(h.TURN, stream=route == 'streaming')
    request, wire = _unsafe_entry_request(route, payload)
    fallbacks = wire_hold.Fallbacks()
    key = wire_hold.fallback_key(request.headers, payload)
    fallbacks.put(key, None, safe=False)
    proxy = SimpleNamespace(_hold_fallbacks=fallbacks, _w4_attempts=wire_cascade.Attempts(),
        _record=lambda ex: None, _telemetry=SimpleNamespace(record=lambda row: None),
        _cap=SimpleNamespace(try_enter=lambda: True, leave=lambda: None))
    proxy._offer_spend_refusal = lambda *args: wire_proxy.WireProxy._offer_spend_refusal(proxy, *args)
    monkeypatch.setattr(wire_hold, 'enabled', lambda: route != 'holds_off')
    monkeypatch.setattr(wire_hold, 'exhaustion', lambda *args: (True, None))
    refusal = spend_guard.Decision(False, spend_guard.OVERAGE, 'spent', status=429)
    monkeypatch.setattr(spend_guard, 'check_anthropic_request', lambda *args: refusal)
    monkeypatch.setattr(wire_cascade, 'configured_targets', lambda: [object()])
    monkeypatch.setattr(wire_cascade, 'move', lambda *args: pytest.fail('unsafe offer moved'))
    called = []
    original_offer = wire_hold.offer_cascade
    monkeypatch.setattr(wire_hold, 'offer_cascade',
                        lambda *args: called.append(1) or original_offer(*args))
    client = wire_proxy._Stream(_CascadeSocket(wire))
    if route == 'streaming':
        assert wire_hold.consume_fallback(proxy, client, request)
    else:
        assert wire_proxy.WireProxy._refused(proxy, client, request)
    assert called == [1]
    assert key in fallbacks.pending


@pytest.mark.parametrize('boundary', ['entry', 'loop'])
def test_hold_deadline_crossed_during_exhaustion_blocks_offer(monkeypatch, boundary):
    from types import SimpleNamespace
    from lib.hal import wire_cascade
    from tests.test_wire_proxy import _CascadeSocket
    clock = [0.0]
    monkeypatch.setattr(wire_hold.time, 'monotonic', lambda: clock[0])
    ex = _fresh_hold_exchange()
    ex.started = 0.0
    checks = iter([(False, None), (True, None)] if boundary == 'loop' else [(True, None)])
    def exhaustion(*args):
        result = next(checks)
        if result[0]:
            clock[0] = 2.0
        return result
    monkeypatch.setattr(wire_hold, 'exhaustion', exhaustion)
    monkeypatch.setattr(wire_cascade, 'move', lambda *args: pytest.fail('expired move'))
    proxy = SimpleNamespace(_hold_fallbacks=wire_hold.Fallbacks(),
                            _w4_attempts=wire_cascade.Attempts())
    client = SimpleNamespace(sock=_CascadeSocket())
    hold = wire_hold.Hold(proxy, client, None, ex, 'fake-key')
    hold.deadline = 1.0
    ex.hold_deadline = 1.0
    hold.run()
    assert client.sock.sent == [wire_hold.exhaustion_response(None)]


def test_a_held_send_that_got_no_answer_is_closed_then_unmarked(monkeypatch):
    """Codex on HAL#747: a held send that failed kept its turn's mark for the life of
    the process. The mark must still stand while the closure is written."""
    from types import SimpleNamespace
    from lib.hal import wire_cascade
    from tests.test_wire_proxy import _CascadeSocket
    ex = _fresh_hold_exchange()
    turn, at_closure, real_close = _turn_of(ex), [], wire_cascade.Attempts.close

    def watched_close(self, key):
        at_closure.append(dict(self._outstanding))
        real_close(self, key)

    def lost_send(data):
        raise ConnectionResetError('acceptance unknown')

    monkeypatch.setattr(wire_cascade.Attempts, 'close', watched_close)
    sock = SimpleNamespace(settimeout=lambda *_: None, sendall=lost_send,
                           shutdown=lambda *_: None, close=lambda: None)
    proxy = SimpleNamespace(pool=SimpleNamespace(take=lambda **_: SimpleNamespace(sock=sock)),
                            _hold_fallbacks=wire_hold.Fallbacks(),
                            _w4_attempts=wire_cascade.Attempts())
    monkeypatch.setattr(wire_hold, 'exhaustion', lambda *args: (False, None))
    request = SimpleNamespace(raw=b'POST /v1/messages HTTP/1.1\r\n\r\n')
    wire_hold.Hold(proxy, SimpleNamespace(sock=_CascadeSocket()), request, ex, 'fake-key',
                   turn=turn).run()
    assert at_closure == [{turn: 1}]
    assert proxy._w4_attempts._outstanding == {} and proxy._w4_attempts.closed(turn)


def test_a_decision_abandoned_at_the_deadline_leaves_no_reservation_and_the_honest_429(monkeypatch):
    """AGORA's falsifier on HAL#747: the hold deadline passes after the decision is
    signed and before any connection. No connection is made, the reservation is removed
    because nothing was sent, and the client still gets the exhaustion answer with its
    rate-limit headers."""
    from types import SimpleNamespace
    from lib.hal import wire_cascade
    from tests.test_wire_proxy import _CascadeSocket
    clock, signed, reset = [0.0], [], 2_000_000_000
    monkeypatch.setattr(wire_hold.time, 'monotonic', lambda: clock[0])
    ex = _fresh_hold_exchange()
    ex.started = 0.0

    def sign(proxy, ex, target):
        signed.append(target.label)
        clock[0] = 2.0
        return True

    monkeypatch.setattr(wire_hold, 'exhaustion', lambda *args: (True, reset))
    monkeypatch.setattr(wire_cascade, 'configured_targets',
                        lambda: [SimpleNamespace(label='fake/target', window=100)])
    monkeypatch.setattr(wire_cascade, 'independent', lambda target: 'fake-key')
    monkeypatch.setattr(wire_cascade, '_sign_decision', sign)
    monkeypatch.setattr(wire_cascade, 'transform', lambda *args: {})
    monkeypatch.setattr(wire_cascade.policy, 'prepare', lambda *args: {})
    monkeypatch.setattr(wire_cascade, '_request', lambda *args: (None, b''))
    monkeypatch.setattr(wire_cascade, '_record', lambda *args: None)
    monkeypatch.setattr(wire_cascade, '_connect',
                        lambda *args: pytest.fail('connected after the deadline'))
    proxy = SimpleNamespace(_hold_fallbacks=wire_hold.Fallbacks(),
                            _w4_attempts=wire_cascade.Attempts())
    client = SimpleNamespace(sock=_CascadeSocket())
    hold = wire_hold.Hold(proxy, client, None, ex, 'fake-key')
    hold.deadline = ex.hold_deadline = 1.0
    hold.run()
    assert signed == ['fake/target']
    assert len(client.sock.sent) == 1 and client.sock.sent[0].startswith(b'HTTP/1.1 429 ')
    assert b'anthropic-ratelimit-unified-status: rejected\r\n' in client.sock.sent[0]
    assert f'anthropic-ratelimit-unified-reset: {reset}\r\n'.encode() in client.sock.sent[0]
    assert not proxy._w4_attempts._path(_turn_of(ex)).exists()
    assert not ex.w4_uncertain and (ex.status, ex.completed) == (429, True)


def test_a_target_send_that_fails_after_its_first_bytes_closes_the_turn_to_w4(monkeypatch):
    """AGORA's second falsifier on HAL#747, through the real move path. A target's
    connection fails after request bytes went out. The turn stays reserved, the hold
    stores no safe fallback, and the same turn offered again is refused before any
    connection is made."""
    from contextlib import nullcontext
    from types import SimpleNamespace
    from lib.hal import account_ledger, wire_cascade
    from tests.test_wire_proxy import _CascadeSocket
    sent, connects, spent = [], [], iter([False])

    def lost_after_send(data):
        sent.append(data)
        raise ConnectionResetError('acceptance unknown')

    def connect(*args):
        connects.append(1)
        return SimpleNamespace(stream=SimpleNamespace(_fill=lambda: False), sock=SimpleNamespace(
            sendall=lost_after_send, settimeout=lambda *_: None,
            shutdown=lambda *_: None, close=lambda: None))

    monkeypatch.setattr(wire_hold, 'exhaustion', lambda *args: (next(spent, True), None))
    monkeypatch.setattr(wire_cascade, 'configured_targets', lambda: [
        SimpleNamespace(label='fake/target', provider='fake', window=100)])
    monkeypatch.setattr(wire_cascade, 'independent', lambda target: 'fake-key')
    monkeypatch.setattr(wire_cascade, '_sign_decision', lambda *args: True)
    monkeypatch.setattr(wire_cascade, 'transform', lambda *args: {})
    monkeypatch.setattr(wire_cascade.policy, 'prepare', lambda *args: {})
    monkeypatch.setattr(wire_cascade, '_request', lambda *args: (
        SimpleNamespace(raw=b'POST fake', headers=()), b'body'))
    monkeypatch.setattr(wire_cascade, '_record', lambda *args: None)
    monkeypatch.setattr(wire_cascade, '_connect', connect)
    monkeypatch.setattr(account_ledger, 'AccountLedger', lambda **kwargs: SimpleNamespace(
        exchange=lambda *args: nullcontext(None)))
    proxy = SimpleNamespace(_hold_fallbacks=wire_hold.Fallbacks(),
                            _w4_attempts=wire_cascade.Attempts())
    ex, client = _fresh_hold_exchange(), SimpleNamespace(sock=_CascadeSocket())
    wire_hold.Hold(proxy, client, None, ex, 'fake-key').run()
    assert sent == [b'POST fakebody'] and connects == [1] and ex.w4_uncertain
    assert client.sock.sent == [wire_hold.exhaustion_response(None)]
    assert proxy._hold_fallbacks.peek('fake-key')[2] is False
    assert proxy._w4_attempts._path(_turn_of(ex)).is_file()
    assert not wire_hold.offer_cascade(proxy, None, _fresh_hold_exchange(), 'unsent')
    assert connects == [1]


# ---- HAL#748 item 4: the "answered turn is not remembered" limit, counted -----------

def _with_session(raw):
    return raw.replace(b'Content-Type:', b'x-claude-code-session-id: fake-session\r\nContent-Type:', 1)


SESSION = (('x-claude-code-session-id', 'fake-session'),)


def test_an_offer_of_a_turn_anthropic_answered_is_counted_not_refused(tmp_path, monkeypatch):
    """The limit seen happening leaves one audit row; the offer goes ahead as before."""
    from types import SimpleNamespace
    from lib.hal import wire_cascade
    monkeypatch.setenv('HAL_HOME', str(tmp_path))
    monkeypatch.setattr(wire_cascade.Attempts, '_unhealthy', False, raising=False)
    rows, moved = [], []
    proxy = SimpleNamespace(_hold_fallbacks=wire_hold.Fallbacks(),
                            _w4_attempts=wire_cascade.Attempts(),
                            _telemetry=SimpleNamespace(record=rows.append))
    monkeypatch.setattr(wire_cascade, 'move', lambda _proxy, _client, ex: moved.append(1) or True)
    fresh, answered = _exchange_for('fake-session', A), _exchange_for('other-session', A)
    assert wire_hold.offer_cascade(proxy, None, fresh, 'unsent')
    assert rows == []
    proxy._w4_attempts.end(_turn_of(answered), 200)
    assert wire_hold.offer_cascade(proxy, None, answered, 'unsent')
    assert rows == [{"event": "w4.limit", "limit": "answered_turn_again"}]
    assert moved == [1, 1]


def test_a_relayed_turn_anthropic_answered_is_noted(held):
    """The plain relay passes the answer's status to the record."""
    payload = dict(h.TURN, stream=False)
    assert exchange(held, _with_session(request(payload=payload))) == SUCCESS
    assert held.proxy._w4_attempts.was_answered(wire_hold.turn_key(SESSION, payload))


def test_a_held_turn_anthropic_answered_is_noted(held):
    """A held send passes its answer's status to the record too."""
    payload = dict(h.TURN, stream=True)
    assert exchange(held, _with_session(request(payload=payload))) == SUCCESS
    assert held.proxy._w4_attempts.was_answered(wire_hold.turn_key(SESSION, payload))
