"""W4 contract: fake credentials/transports only, isolated ledgers and audit sinks."""
import ast
import io
import json
import socket
import time
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from lib.hal import gateway, gateway_cascade as cascade, gateway_egress as egress
from lib.hal import gateway_failover as failover, gateway_openai as chat
from lib.hal import account_switch, gateway_responses as responses, gateway_witness, infinite_thread, spend_guard
from lib.hal.account_ledger import AccountLedger, LedgerView
from tests._public_tree import require
from tests.test_infinite_thread import HOST, _Double, _double_verify

CHAT = '/v1/chat/completions'
RESPONSES = '/v1/responses'
FIRST = 'groq/llama-3.3-70b-versatile'
SECOND = 'gemini/gemini-3.5-flash-lite'


@pytest.fixture(autouse=True)
def signed_cascade_rows(_isolated_hal_home, monkeypatch):
    """Use the wire rig's signer double and an empty account pool in every case."""
    monkeypatch.setattr(account_switch, '_read_pool', lambda: [])
    double = _Double(infinite_thread.state_root())
    monkeypatch.setattr(infinite_thread, 'default_signer', lambda _root, _host: double.signer())
    return double


def request(path=CHAT, stream=False):
    out = {'model': 'claude-opus-5', 'stream': stream}
    out['input' if path == RESPONSES else 'messages'] = [{'role': 'user', 'content': 'hello'}]
    return out


def spent(seconds=60):
    return spend_guard.Decision(False, 'anthropic-overage', f'earliest reset in {seconds} seconds', 429, seconds)


def reply(text='answer'):
    return gateway.UpstreamResponse(200, [], io.BytesIO(json.dumps({
        'id': 'msg_fake', 'type': 'message', 'role': 'assistant',
        'content': [{'type': 'text', 'text': text}], 'stop_reason': 'end_turn',
        'usage': {'input_tokens': 1, 'output_tokens': 1}}).encode()))


class Handler:
    def __init__(self, payload):
        raw = json.dumps(payload).encode()
        self.headers = {'Content-Length': str(len(raw))}
        self.rfile, self.wfile = io.BytesIO(raw), io.BytesIO()
        self.status, self.sent_headers = None, {}
    def send_response(self, status):
        assert self.status is None, 'second response committed'
        self.status = status
    def send_header(self, key, value):
        self.sent_headers[key.lower()] = value
    def end_headers(self):
        pass


def dispatch(payload, path, proxy):
    handler = Handler(payload)
    (responses.handle_responses if path == RESPONSES else chat.handle_chat_completions)(handler, proxy)
    return handler


def pool(outcomes, calls):
    def factory(spec):
        def forward(*args, **kwargs):
            calls.append(spec.name)
            outcome = outcomes[int(spec.name)]
            if isinstance(outcome, BaseException):
                raise outcome
            return reply()
        return SimpleNamespace(forward=forward)
    return failover.FailoverProxy([failover.UpstreamSpec(str(i)) for i in range(len(outcomes))],
                                 proxy_factory=factory)


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    account_switch.forget_exhausted()
    account_switch._ORGS.clear()
    account_switch._ORG_RETRY.clear()
    monkeypatch.setattr(cascade.policy, 'load_shield', lambda: lambda body: body)
    config = tmp_path / 'cascade.json'
    config.write_text(json.dumps({'targets': [SECOND, FIRST]}))
    monkeypatch.setenv('HAL_W4_CONFIG', str(config))
    monkeypatch.setenv('HAL_GATEWAY_MULTIPROVIDER', '0')
    monkeypatch.setattr(egress, '_default_key_resolver', lambda _: 'fake-target-token')
    calls, outcomes = [], []
    def opener(method, url, headers, body):
        calls.append(json.loads(body))
        outcome = outcomes.pop(0) if outcomes else 200
        if isinstance(outcome, BaseException):
            raise outcome
        data = {'id': 'reply/unsafe id', 'choices': [{'message': {'role': 'assistant', 'content': 'switched'},
                'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 1, 'completion_tokens': 1}}
        raw = io.BytesIO(json.dumps(data).encode())
        raw.status, raw.headers = outcome, {'Content-Type': 'application/json', 'Retry-After': '90'}
        return raw
    monkeypatch.setattr(egress, 'build_default_opener', lambda **kwargs: opener)
    # No accidental real network, including the original account pool.
    monkeypatch.setattr(failover, 'build_default_opener', lambda: opener)
    yield SimpleNamespace(config=config, calls=calls, outcomes=outcomes,
                          audit=Path(__import__('os').environ['HAL_HOME']) / 'logs/gateway-audit.jsonl')
    account_switch.forget_exhausted()
    account_switch._ORGS.clear()
    account_switch._ORG_RETRY.clear()


def rows(fakes):
    return [r for r in map(json.loads, fakes.audit.read_text().splitlines()) if 'cascade_reason' in r]


def thread_rows(home):
    paths = list((home / 'state/infinite/threads').glob('*.jsonl'))
    return [json.loads(line) for path in paths for line in path.read_text().splitlines()]


def anthropic_answer(status, headers=()):
    raw = io.BytesIO(b'upstream refusal')
    raw.status, raw.headers = status, dict(headers)
    return raw


def subscription_headers(monkeypatch):
    def passthrough(pairs):
        return [(name, value) for name, value in pairs
                if name.lower() not in ('host', 'content-length')]
    monkeypatch.setattr(chat, 'anthropic_upstream_headers', passthrough)
    monkeypatch.setattr(responses, 'anthropic_upstream_headers', passthrough)


@pytest.mark.parametrize('path', [CHAT, RESPONSES])
def test_quota_rejected_across_account_pool_starts_cascade(fakes, monkeypatch, path):
    first, second = 'fake-account-A-token', 'fake-account-B-token'
    monkeypatch.setattr(spend_guard, '_OAUTH_PREFIX', 'fake-account-')
    monkeypatch.setattr(gateway, '_normalise_anthropic_auth', lambda headers: list(headers))
    monkeypatch.setattr(account_switch, '_read_pool', lambda: [('B', second)])
    subscription_headers(monkeypatch)
    calls = []

    def opener(method, url, headers, body):
        token = next(value.removeprefix('Bearer ') for name, value in headers
                     if name.lower() == 'authorization')
        if method == 'GET':
            return anthropic_answer(200, [('anthropic-organization-id', 'org-' + token)])
        calls.append(token)
        return anthropic_answer(429, [
            ('anthropic-ratelimit-unified-representative-claim', 'five_hour'),
            ('anthropic-ratelimit-unified-status', 'rejected'),
            ('anthropic-ratelimit-unified-reset', str(time.time() + (600 if token == first else 60))),
        ])

    proxy = failover.FailoverProxy([failover.UpstreamSpec('anthropic-primary')],
        proxy_factory=lambda _: gateway.GatewayProxy(opener=opener))
    payload = request(path)
    handler = Handler(payload)
    handler.headers['Authorization'] = 'Bearer ' + first
    (responses.handle_responses if path == RESPONSES else chat.handle_chat_completions)(handler, proxy)
    assert calls == [first, second], (handler.status, handler.wfile.getvalue())
    assert handler.status == 200 and b'switched' in handler.wfile.getvalue()
    assert len(fakes.calls) == 1
    assert 1 <= rows(fakes)[-1]['previous_retry_after'] <= 61


@pytest.mark.parametrize('path', [CHAT, RESPONSES])
def test_spent_pool_with_paid_source_starts_cascade(fakes, monkeypatch, path):
    policy = fakes.config.with_name('spend.json')
    policy.write_text('{"paid_opt_in": ["groq-api-key", "gemini-api-key"]}')
    monkeypatch.setenv(spend_guard.CONFIG_ENV, str(policy))
    refused = spend_guard.check_anthropic_request('POST', '/v1/messages',
                                                   [('Authorization', 'Bearer sk-ant-api-fake')])
    assert refused is not None and refused.status == 403
    assert refused.source == spend_guard.API_KEY
    calls = []
    paid = gateway.GatewayProxy(opener=lambda *args: calls.append(args) or
                                pytest.fail('paid Anthropic source was sent'))
    def spent_primary(seconds):
        def forward(*args, **kwargs):
            raise spend_guard.SpendRefused(spent(seconds))
        return SimpleNamespace(forward=forward)
    proxy = failover.FailoverProxy([
        failover.UpstreamSpec('subscription-a'),
        failover.UpstreamSpec('subscription-b'),
        failover.UpstreamSpec('paid', header_overrides=(('Authorization', 'Bearer sk-ant-api-fake'),)),
    ], proxy_factory=lambda spec: {'subscription-a': spent_primary(600),
                                   'subscription-b': spent_primary(60),
                                   'paid': paid}[spec.name])
    handler = dispatch(request(path), path, proxy)
    assert handler.status == 200 and b'switched' in handler.wfile.getvalue()
    assert not calls and len(fakes.calls) == 1
    assert rows(fakes)[-1]['previous_retry_after'] == 60


@pytest.mark.parametrize('path', [CHAT, RESPONSES])
def test_plain_final_429_does_not_start_cascade(fakes, monkeypatch, path):
    monkeypatch.setattr(spend_guard, '_OAUTH_PREFIX', 'fake-account-')
    monkeypatch.setattr(gateway, '_normalise_anthropic_auth', lambda headers: list(headers))
    subscription_headers(monkeypatch)
    calls = []
    def opener(*args):
        calls.append(args)
        return anthropic_answer(429, [('Retry-After', '90')])
    proxy = failover.FailoverProxy([failover.UpstreamSpec('anthropic-primary')],
        proxy_factory=lambda _: gateway.GatewayProxy(opener=opener))
    handler = Handler(request(path))
    handler.headers['Authorization'] = 'Bearer fake-account-A-token'
    (responses.handle_responses if path == RESPONSES else chat.handle_chat_completions)(handler, proxy)
    assert handler.status == 503, (handler.status, handler.wfile.getvalue())
    assert len(calls) == 1 and not fakes.calls


def test_policy_order_and_import_boundary(fakes):
    assert cascade.configured_targets() == [FIRST, SECOND]
    root = Path(__file__).resolve().parents[1]
    for name in ('raillm', 'gateway_failover'):
        tree = ast.parse((root / 'lib/hal' / (name + '.py')).read_text())
        code_seen = False
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                code_seen = True
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                assert not code_seen, name
    ladder = ast.parse((root / 'lib/hal/raillm_ladder.py').read_text())
    assert not any(isinstance(n, ast.ImportFrom) and n.module == 'lib.hal.raillm' for n in ast.walk(ladder))
    from lib.hal.raillm import ModelTier
    from lib.hal.raillm_policy import ModelTier as PolicyTier
    assert ModelTier is PolicyTier


def test_pool_is_fully_spent_before_switch(fakes):
    calls = []
    p = pool([spend_guard.SpendRefused(spent(600)), True], calls)
    h = dispatch(request(), CHAT, p)
    assert calls == ['0', '1'] and not fakes.calls
    assert h.status == 200 and b'answer' in h.wfile.getvalue()
    calls.clear()
    p = pool([spend_guard.SpendRefused(spent(600)), spend_guard.SpendRefused(spent(60))], calls)
    h = dispatch(request(), CHAT, p)
    assert calls == ['0', '1'] and len(fakes.calls) == 1
    assert h.status == 200 and b'switched' in h.wfile.getvalue()
    assert rows(fakes)[-1]['previous_retry_after'] == 60


def test_production_single_account_only_switches_on_known_spent(fakes, monkeypatch):
    monkeypatch.setattr(spend_guard, '_OAUTH_PREFIX', 'fake-account-')
    monkeypatch.setattr(gateway, '_normalise_anthropic_auth', lambda headers: list(headers))
    token = 'fake-account-A-token'
    policy = fakes.config.with_name('spend.json')
    policy.write_text('{"paid_opt_in": ["groq-api-key", "gemini-api-key"]}')
    monkeypatch.setenv(spend_guard.CONFIG_ENV, str(policy))
    p = failover.build_failover_proxy()
    assert [s.name for s in p._specs] == ['anthropic-primary']
    # A headerless throttle is not pool exhaustion. Its Retry-After: 90
    # exceeds the gateway wait budget, so it is sent upstream only once.
    fakes.outcomes.append(429)
    p._proxies[0]._sleep = lambda _: None
    resp = p.forward('/v1/messages', b'{}', [('Authorization', 'Bearer ' + token)])
    assert resp.status == 429 and len(fakes.calls) == 1
    assert spend_guard.spent_until(spend_guard.account_key(token)) is None
    resp.close()
    fakes.calls.clear()
    spend_guard.mark_spent(spend_guard.account_key(token), time.time() + 600)
    with pytest.raises(failover.AccountsSpent) as caught:
        p.forward('/v1/messages', b'{}', [('Authorization', 'Bearer ' + token)])
    assert not fakes.calls
    selected = cascade.try_cascade(request(), CHAT, caught.value)
    assert selected is not None and len(fakes.calls) == 1
    selected[0].close()


def test_non_exhaustion_never_enters(fakes):
    for decision in (spent(), spend_guard.Decision(False, 'paid', 'refused')):
        assert cascade.try_cascade(request(), CHAT, spend_guard.SpendRefused(decision)) is None
    assert not fakes.calls


def test_paid_and_unknown_billing_refused(fakes, monkeypatch, tmp_path):
    policy = tmp_path / 'strict.json'
    policy.write_text('{"paid_opt_in": []}')
    monkeypatch.setenv(spend_guard.CONFIG_ENV, str(policy))
    assert cascade.refusal(FIRST, request(), CHAT) == 'spend_guard'
    assert cascade.try_cascade(request(), CHAT, failover.AccountsSpent(spent())) is None
    assert not fakes.calls and rows(fakes)[-1]['cascade_refusals'][0]['reason'] == 'spend_guard'


def test_budget_is_one_byte_bound_and_window_refuses(fakes):
    payload = request()
    payload['max_tokens'] = 100
    assert cascade.request_budget(payload, CHAT) == len(json.dumps(payload, ensure_ascii=True).encode()) + 4096 + 100
    payload['messages'][0]['content'] = 'x' * 128000
    assert cascade.refusal(FIRST, payload, CHAT) == 'context_unknown_or_small'
    assert cascade.try_cascade(payload, CHAT, failover.AccountsSpent(spent())) is None
    assert not fakes.calls


def test_uncertain_context_refused(fakes):
    payload = request()
    payload['messages'][0]['content'] = [{'type': 'image_url', 'image_url': {'url': 'https://invalid.example/image'}}]
    assert cascade.refusal(FIRST, payload, CHAT) == 'context_unknown_or_small'
    assert cascade.try_cascade(payload, CHAT, failover.AccountsSpent(spent())) is None
    assert not fakes.calls


@pytest.mark.parametrize('path', [CHAT, RESPONSES])
def test_missing_tool_refuses(fakes, path):
    payload = request(path)
    payload['tools'] = [{'type': 'custom', 'name': 'apply_patch'}]
    assert cascade.refusal(FIRST, payload, path) == 'unsupported_tool'
    assert cascade.try_cascade(payload, path, failover.AccountsSpent(spent())) is None
    assert not fakes.calls


def test_additional_tools_refuses_before_translation(fakes):
    payload = request(RESPONSES)
    payload['input'].append({'type': 'additional_tools', 'tools': [{'type': 'function', 'name': 'exec_command'}]})
    assert cascade.refusal(FIRST, payload, RESPONSES) == 'unsupported_tool'
    assert cascade.try_cascade(payload, RESPONSES, failover.AccountsSpent(spent())) is None
    assert not fakes.calls


def test_cooled_account_refuses(fakes):
    account = 'w4:groq:' + spend_guard.account_key('fake-target-token')
    ledger = AccountLedger(resolver=SimpleNamespace(label_for=lambda _: (account, None)))
    with ledger.exchange([]) as watch:
        watch.answered(429, {'Retry-After': '90'})
    assert LedgerView().cooling(account) is not None
    fakes.config.write_text(json.dumps({'targets': [FIRST]}))
    assert cascade.try_cascade(request(), CHAT, failover.AccountsSpent(spent())) is None
    assert not fakes.calls and rows(fakes)[-1]['cascade_refusals'] == [{'provider': 'groq', 'reason': 'cooldown'}]


def test_shield_scrubs_outgoing_text(fakes, monkeypatch):
    """The W4-amendment shield applies to this cascade too (HAL#684 shipped
    unshielded); asserted on the fake target's own request log."""
    monkeypatch.setattr(cascade.policy, 'load_shield',
                        lambda: lambda body: json.loads(json.dumps(body).replace('DenylistedExample', '[PERSON]')))
    payload = request()
    payload['messages'][0]['content'] = 'DenylistedExample hello'
    selected = cascade.try_cascade(payload, CHAT, failover.AccountsSpent(spent()))
    assert selected is not None
    sent = json.dumps(fakes.calls[0])
    assert 'DenylistedExample' not in sent and '[PERSON]' in sent
    selected[0].close()


def test_shield_failure_refuses_before_send(fakes, monkeypatch):
    monkeypatch.setattr(cascade.policy, 'load_shield', lambda: lambda _: 1 / 0)
    selected = cascade.try_cascade(request(), CHAT, failover.AccountsSpent(spent()))
    assert selected is None and not fakes.calls
    assert rows(fakes)[-1]['cascade_refusals'][0]['reason'] == 'shield'


@pytest.mark.parametrize('path', [CHAT, RESPONSES])
def test_ids_replay_and_tool_semantics_on_switch(fakes, path):
    payload = request(path)
    schema = {'type': 'object', 'properties': {'id': {'type': 'string'}, 'reasoning': {'type': 'string'}}}
    fn = {'name': 'lookup', 'description': 'find it', 'parameters': schema}
    payload['tools'] = [dict(fn, type='function')] if path == RESPONSES else [{'type': 'function', 'function': fn}]
    payload.update(previous_response_id=None, reasoning={'effort': 'high'}, store=True)
    if path == RESPONSES:
        payload['input'] += [{'type': 'reasoning', 'encrypted_content': 'opaque'},
            {'type': 'function_call', 'id': 'fc/old', 'call_id': 'bad/id space', 'name': 'lookup', 'arguments': '{"id":"keep"}'},
            {'type': 'function_call_output', 'call_id': 'bad/id space', 'output': 'found'}]
    else:
        payload['messages'] += [{'role': 'assistant', 'reasoning_content': 'opaque', 'content': None,
            'tool_calls': [{'id': 'bad/id space', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{"id":"keep"}'}, 'extra_content': {'signature': 'opaque'}}]},
            {'role': 'tool', 'tool_call_id': 'bad/id space', 'content': 'found'}]
    original = json.dumps(payload, sort_keys=True)
    selected = cascade.try_cascade(payload, path, failover.AccountsSpent(spent()))
    assert selected is not None
    sent = fakes.calls[0]
    assert sent['tools'][0]['function'] == fn
    tool_call = next(m for m in sent['messages'] if m.get('tool_calls'))['tool_calls'][0]
    result = next(m for m in sent['messages'] if m['role'] == 'tool')
    assert tool_call['id'] == result['tool_call_id'] == cascade.normal_id('bad/id space')
    assert tool_call['function']['arguments'] == '{"id": "keep"}'
    assert all(word not in json.dumps(sent) for word in ('previous_response_id', 'encrypted_content', 'opaque', 'reasoning_content'))
    assert json.dumps(payload, sort_keys=True) == original
    assert cascade.clean_request({'id': 'resp_bad/id'})['id'].startswith('resp_')
    selected[0].close()


def test_audit_records_move_and_refusals(fakes):
    fakes.outcomes.append(503)
    selected = cascade.try_cascade(request(), CHAT, failover.AccountsSpent(spent()))
    assert selected is not None
    row = rows(fakes)[-1]
    assert row['provider'] == 'gemini' and row['model'] == 'gemini-3.5-flash-lite'
    assert row['failed_over'] is True and row['served'] is True
    assert row['cascade_reason'] == 'all_configured_accounts_spent'
    assert row['cascade_refusals'] == [{'provider': 'groq', 'reason': 'target_refused', 'status': 503}]
    assert 'fake-target-token' not in fakes.audit.read_text()
    selected[0].close()


@pytest.mark.parametrize('path', [CHAT, RESPONSES])
@pytest.mark.parametrize('stream', [False, True])
def test_unsigned_decision_keeps_exhaustion_answer(fakes, monkeypatch, path, stream):
    monkeypatch.setattr(gateway_witness, 'append_thread_row',
                        lambda *args, **kwargs: SimpleNamespace(written=False))
    handler = dispatch(request(path, stream), path,
                       pool([spend_guard.SpendRefused(spent())], []))
    assert not fakes.calls
    assert rows(fakes)[-1]['cascade_refusals'] == [{'provider': 'groq', 'reason': 'unsigned'}]
    assert b'earliest reset in 60 seconds' in handler.wfile.getvalue()
    assert handler.status == (200 if path == RESPONSES else 429)


@pytest.mark.parametrize('path', [CHAT, RESPONSES])
@pytest.mark.parametrize('stream', [False, True])
def test_decision_row_precedes_target_and_outcome(fakes, monkeypatch, path, stream,
                                                   _isolated_hal_home, signed_cascade_rows):
    observed = []
    original = egress.cross_wire_forward
    def recording_forward(*args, **kwargs):
        observed.extend(row for row in thread_rows(_isolated_hal_home)
                        if row['class'] == 'failover')
        return original(*args, **kwargs)
    monkeypatch.setattr(egress, 'cross_wire_forward', recording_forward)
    handler = Handler(request(path, stream))
    handler.headers['x-claude-code-session-id'] = 'test-session'
    rail = responses.handle_responses if path == RESPONSES else chat.handle_chat_completions
    rail(handler, pool([spend_guard.SpendRefused(spent())], []))
    assert handler.status == 200 and len(fakes.calls) == 1
    assert len(observed) == 1
    assert {key: observed[0][key] for key in ('provider', 'model', 'billing', 'status')} == {
        'provider': 'groq', 'model': 'llama-3.3-70b-versatile',
        'billing': 'unknown', 'status': None}
    assert observed[0]['session_id'] == 'test-session'
    verdict, _ = _double_verify(infinite_thread.signed_bytes(observed[0]),
                                observed[0]['sig'], signed_cascade_rows.pins()[HOST])
    assert verdict == 'verified'
    failovers = [row for row in thread_rows(_isolated_hal_home) if row['class'] == 'failover']
    assert [(row['provider'], row['model']) for row in failovers] == [
        ('groq', 'llama-3.3-70b-versatile')] * 2
    assert [(row['status'], row['billing']) for row in failovers] == [(None, 'unknown'), (200, 'unknown')]


def test_unsigned_first_target_stops_before_second(fakes, monkeypatch):
    monkeypatch.setattr(gateway_witness, 'append_thread_row',
                        lambda *args, **kwargs: SimpleNamespace(written=False))
    assert cascade.try_cascade(request(), CHAT, failover.AccountsSpent(spent())) is None
    assert not fakes.calls
    assert rows(fakes)[-1]['cascade_refusals'] == [{'provider': 'groq', 'reason': 'unsigned'}]


@pytest.mark.parametrize('status', [400, 401, 403, 429, 500, 503])
def test_non_success_declines_then_tries_next(fakes, status):
    fakes.outcomes.append(status)
    selected = cascade.try_cascade(request(), CHAT, failover.AccountsSpent(spent()))
    assert selected is not None and selected[1] == 'gemini-3.5-flash-lite'
    assert len(fakes.calls) == 2
    assert rows(fakes)[-1]['cascade_refusals'][0]['status'] == status
    selected[0].close()


@pytest.mark.parametrize('fault', [ConnectionRefusedError(), socket.gaierror(), urllib.error.URLError(socket.gaierror())])
def test_definite_connect_failure_tries_next(fakes, fault):
    fakes.outcomes.append(fault)
    selected = cascade.try_cascade(request(), CHAT, failover.AccountsSpent(spent()))
    assert selected is not None and selected[1] == 'gemini-3.5-flash-lite'
    assert len(fakes.calls) == 2 and rows(fakes)[-1]['cascade_refusals'][0]['reason'] == 'before_send'
    selected[0].close()


def test_credential_failure_tries_next(fakes, monkeypatch):
    def credential(service):
        if service == egress.PROVIDER_ENDPOINTS['groq'][1]:
            raise RuntimeError('fake credential unavailable')
        return 'fake-target-token'
    monkeypatch.setattr(egress, '_default_key_resolver', credential)
    selected = cascade.try_cascade(request(), CHAT, failover.AccountsSpent(spent()))
    assert selected is not None and selected[1] == 'gemini-3.5-flash-lite' and len(fakes.calls) == 1
    assert rows(fakes)[-1]['cascade_refusals'][0]['reason'] == 'before_send'
    selected[0].close()


@pytest.mark.parametrize('path', [CHAT, RESPONSES])
@pytest.mark.parametrize('kind', ['empty', 'uncertain', 'error_status', 'unexpected'])
def test_last_rung_unchanged(fakes, monkeypatch, path, kind):
    if kind == 'empty':
        monkeypatch.delenv('HAL_W4_CONFIG')
        assert cascade.configured_targets() == []
    elif kind == 'uncertain':
        fakes.outcomes.append(TimeoutError('acceptance unknown'))
    elif kind == 'error_status':
        fakes.outcomes.extend([503, 400])
    else:
        def broken(*args):
            raise RuntimeError('internal cascade fault')
        monkeypatch.setattr(cascade, 'try_cascade', broken)
    p = pool([spend_guard.SpendRefused(spent(600)), spend_guard.SpendRefused(spent(60))], [])
    h = dispatch(request(path), path, p)
    if path == RESPONSES:
        assert h.status == 200 and b'earliest reset in 60 seconds' in h.wfile.getvalue()
        assert json.loads(h.wfile.getvalue())['status'] == 'completed'
    else:
        assert h.status == 429 and h.sent_headers['retry-after'] == '60'
        assert b'earliest reset in 60 seconds' in h.wfile.getvalue()
    if kind == 'uncertain':
        assert len(fakes.calls) == 1 and rows(fakes)[-1]['cascade_refusals'][0]['reason'] == 'acceptance_uncertain'
    if kind == 'unexpected':
        assert rows(fakes)[-1]['cascade_refusals'][0]['reason'] == 'cascade_failed'


@pytest.mark.parametrize('path', [CHAT, RESPONSES])
def test_stream_failure_never_switches_again(fakes, monkeypatch, path):
    from lib.hal.gateway_crosswire import anthropic_message_to_sse_chunks
    class BrokenStream:
        status, headers, usage = 200, [], None
        def iter_chunks(self):
            message = json.loads(reply().read_all())
            chunks = list(anthropic_message_to_sse_chunks(message))
            yield chunks[0]
            raise ConnectionResetError('fake stream broke')
        def close(self):
            pass
    calls = []
    def forward(*args, **kwargs):
        calls.append(kwargs['provider'])
        return BrokenStream()
    monkeypatch.setattr(egress, 'cross_wire_forward', forward)
    h = dispatch(request(path, True), path, pool([spend_guard.SpendRefused(spent())], []))
    assert calls == ['groq'] and h.status == 200
    assert b'error' in h.wfile.getvalue()
    assert b'response.completed' not in h.wfile.getvalue()


def test_measurement_reports_shapes_not_request_data(monkeypatch):
    import importlib.util
    require('tools/wire-matrix/w4_measure.py')
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / 'tools/wire-matrix'))
    spec = importlib.util.spec_from_file_location('w4_measure', root / 'tools/wire-matrix/w4_measure.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = request(RESPONSES)
    payload['input'].append({'type': 'additional_tools', 'tools': [
        {'type': 'namespace', 'name': 'private-name', 'tools': [
            {'type': 'custom', 'name': 'private-tool'}, {'type': 'function', 'name': 'another-tool'}]}]})
    raw = json.dumps(payload).encode()
    report = module.describe(raw)
    assert report['wire_bytes'] == len(raw)
    assert report['text_byte_bound_with_framing_and_output'] == len(raw) + 8192
    assert report['additional_tool_types'] == {'namespace': 1, 'custom': 1, 'function': 1}
    assert report['tools_supported'] is False
    assert report['candidate_windows'][FIRST] == {'window': 128000, 'fits_size_bound': True}
    assert 'private' not in json.dumps(report) and 'hello' not in json.dumps(report)


def test_hidden_history_and_nontext_tool_output_refused(fakes):
    payload = request(RESPONSES)
    payload['previous_response_id'] = 'resp_old'
    assert cascade.refusal(FIRST, payload, RESPONSES) == 'context_unknown_or_small'
    payload.pop('previous_response_id')
    payload['input'].append({'type': 'function_call_output', 'call_id': 'call_old',
                             'output': [{'type': 'input_image', 'image_url': 'https://invalid.example/image'}]})
    assert cascade.refusal(FIRST, payload, RESPONSES) == 'context_unknown_or_small'
    assert cascade.try_cascade(payload, RESPONSES, failover.AccountsSpent(spent())) is None
    assert not fakes.calls


@pytest.mark.parametrize('path', [CHAT, RESPONSES])
@pytest.mark.parametrize('stream', [False, True])
def test_new_reply_tool_ids_are_normalized(fakes, monkeypatch, path, stream):
    from lib.hal.gateway_crosswire import anthropic_message_to_sse_chunks
    message = {'id': 'resp_bad/id', 'type': 'message', 'role': 'assistant', 'model': 'fake',
               'content': [{'type': 'tool_use', 'id': 'bad/call id', 'name': 'lookup', 'input': {'id': 'leave alone'}}],
               'stop_reason': 'tool_use', 'usage': {'input_tokens': 1, 'output_tokens': 1}}
    def forward(*args, **kwargs):
        body = b''.join(anthropic_message_to_sse_chunks(message)) if stream else json.dumps(message).encode()
        return gateway.UpstreamResponse(200, [], io.BytesIO(body))
    monkeypatch.setattr(egress, 'cross_wire_forward', forward)
    h = dispatch(request(path, stream), path, pool([spend_guard.SpendRefused(spent())], []))
    body = h.wfile.getvalue()
    assert h.status == 200 and cascade.normal_id('bad/call id').encode() in body
    assert b'bad/call id' not in body and b'resp_bad/id' not in body
    assert b'leave alone' in body


def test_chat_stream_function_arguments_and_indices():
    events = [
        {'type': 'content_block_start', 'index': 2, 'content_block': {'type': 'tool_use', 'id': 'call_a', 'name': 'first', 'input': {}}},
        {'type': 'content_block_delta', 'index': 2, 'delta': {'type': 'input_json_delta', 'partial_json': '{"x":'}},
        {'type': 'content_block_delta', 'index': 2, 'delta': {'type': 'input_json_delta', 'partial_json': '7}'}},
        {'type': 'content_block_start', 'index': 4, 'content_block': {'type': 'tool_use', 'id': 'call_b', 'name': 'second', 'input': {}}},
        {'type': 'content_block_delta', 'index': 4, 'delta': {'type': 'input_json_delta', 'partial_json': '{"y":8}'}},
        {'type': 'message_delta', 'delta': {'stop_reason': 'tool_use'}}, {'type': 'message_stop'}]
    raw = [('data: ' + json.dumps(e) + '\n\n').encode() for e in events]
    chunks = list(chat.translate_sse_stream(raw, model='fake', completion_id='chatcmpl_fake', created=1))
    parsed = [json.loads(c[6:]) for c in chunks[:-1]]
    calls = [call for c in parsed for call in c['choices'][0]['delta'].get('tool_calls', [])]
    assert [c['index'] for c in calls] == [0, 0, 0, 1, 1]
    assert [c['id'] for c in calls if 'id' in c] == ['call_a', 'call_b']
    assert ''.join(c['function']['arguments'] for c in calls if c['index'] == 0) == '{"x":7}'
    assert ''.join(c['function']['arguments'] for c in calls if c['index'] == 1) == '{"y":8}'
    assert parsed[-1]['choices'][0]['finish_reason'] == 'tool_calls' and chunks[-1] == b'data: [DONE]\n\n'

# The OpenAI W4 product is activated only in this synthetic policy.
def token_plan_entry(model='qwen3.8-max'):
    return {'provider': 'qwen-token-plan', 'product': 'token-plan-personal',
            'model': model, 'base_url': egress.PROVIDER_ENDPOINTS['qwen-token-plan'][0],
            'credential': 'grip-qwen-token-plan', 'auth': 'bearer',
            'billing': 'subscription', 'transport': 'openai-chat'}


def token_plan_config(fakes, entries):
    fakes.config.write_text(json.dumps({'targets': entries}))


@pytest.mark.parametrize('path', [CHAT, RESPONSES])
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('model', ['qwen3.8-max', 'deepseek-v4-pro'])
def test_token_plan_bound_send_on_both_rails(fakes, monkeypatch, path, stream, model,
                                              _isolated_hal_home):
    token_plan_config(fakes, [token_plan_entry(model)])
    strict = fakes.config.with_name('strict.json')
    strict.write_text('{"paid_opt_in": []}')
    monkeypatch.setenv(spend_guard.CONFIG_ENV, str(strict))
    selected = cascade.try_cascade(request(path, stream), path,
                                    failover.AccountsSpent(spent()), session_id='plan-session')
    assert selected is not None and selected[1] == model
    assert len(fakes.calls) == 1
    row = rows(fakes)[-1]
    assert (row['provider'], row['model'], row['billing'], row['product']) == (
        'qwen-token-plan', model, 'subscription', 'token-plan-personal')
    signed = [r for r in thread_rows(_isolated_hal_home) if r['class'] == 'failover']
    assert signed[0]['billing'] == signed[-1]['billing'] == 'subscription'
    assert signed[0]['status'] is None
    assert signed[0]['provider'] == 'qwen-token-plan'
    assert 'fake-target-token' not in fakes.audit.read_text()
    assert 'fake-target-token' not in json.dumps(signed)
    selected[0].close()


@pytest.mark.parametrize('field,value', [
    ('provider', 'qwen'), ('product', 'other-plan'), ('model', 'qwen3.8-flash'),
    ('base_url', egress.PROVIDER_ENDPOINTS['qwen'][0]),
    ('credential', 'hal-dashscope'), ('auth', 'x-api-key'),
    ('billing', 'paid'), ('transport', 'anthropic'),
])
def test_paid_binding_substitution_refuses_before_resolver(fakes, monkeypatch, field, value):
    entry = token_plan_entry()
    entry[field] = value
    token_plan_config(fakes, [entry])
    monkeypatch.setattr(egress, '_default_key_resolver', lambda _: pytest.fail('secret lookup'))
    assert cascade.configured_targets() == []
    assert cascade.try_cascade(request(), CHAT, failover.AccountsSpent(spent())) is None
    assert not fakes.calls


@pytest.mark.parametrize('entry', [None, {}, FIRST, 'qwen/qwen3.8-max',
                                   {'provider': 'qwen-token-plan', 'billing': 'subscription'}])
def test_missing_or_legacy_binding_stays_paid(fakes, monkeypatch, entry):
    token_plan_config(fakes, [entry])
    strict = fakes.config.with_name('strict-legacy.json')
    strict.write_text('{"paid_opt_in": []}')
    monkeypatch.setenv(spend_guard.CONFIG_ENV, str(strict))
    monkeypatch.setattr(egress, '_default_key_resolver', lambda _: pytest.fail('secret lookup'))
    assert cascade.try_cascade(request(), CHAT, failover.AccountsSpent(spent())) is None
    assert not fakes.calls


def test_bound_selector_endpoint_and_only_target_auth(fakes, monkeypatch):
    token_plan_config(fakes, [token_plan_entry()])
    seen = []
    def resolver(selector):
        seen.append(('selector', selector))
        return 'synthetic-plan-key'
    def opener(method, url, headers, body):
        seen.append(('send', url, dict(headers)))
        raw = io.BytesIO(json.dumps({'choices': [{'message': {'role': 'assistant',
                          'content': 'ok'}, 'finish_reason': 'stop'}]}).encode())
        raw.status, raw.headers = 200, {'Content-Type': 'application/json'}
        return raw
    monkeypatch.setattr(egress, '_default_key_resolver', resolver)
    monkeypatch.setattr(egress, 'build_default_opener', lambda **kw: opener)
    chosen = cascade.try_cascade(request(), CHAT, failover.AccountsSpent(spent()))
    assert chosen is not None
    assert seen[0] == ('selector', 'grip-qwen-token-plan')
    assert seen[1][1] == token_plan_entry()['base_url'] + '/chat/completions'
    assert seen[1][2]['Authorization'] == 'Bearer synthetic-plan-key'
    assert 'fake-target-token' not in str(seen)
    chosen[0].close()


@pytest.mark.parametrize('path,field,value', [
    (CHAT, 'response_format', {'type': 'json_schema', 'json_schema': {'name': 'answer'}}),
    (CHAT, 'max_completion_tokens', 100), (RESPONSES, 'text', {'format': {'type': 'json_object'}}),
    (RESPONSES, 'reasoning', {'effort': 'high'}),
])
def test_bound_unsupported_output_constraint_refuses(fakes, monkeypatch, path, field, value):
    token_plan_config(fakes, [token_plan_entry()])
    payload = request(path)
    payload[field] = value
    monkeypatch.setattr(egress, '_default_key_resolver', lambda _: pytest.fail('secret lookup'))
    assert cascade.try_cascade(payload, path, failover.AccountsSpent(spent())) is None
    assert not fakes.calls


def test_bound_quota_reset_cools_both_models_and_rails(fakes, monkeypatch):
    token_plan_config(fakes, [token_plan_entry(), token_plan_entry('deepseek-v4-pro')])
    sends = []
    def opener(method, url, headers, body):
        sends.append(json.loads(body)['model'])
        raw = io.BytesIO(json.dumps({'code': 'Throttling.AllocationQuota',
                         'message': 'The quota will reset at 12-31 23:59:59 UTC'}).encode())
        raw.status, raw.headers = 429, {'Content-Type': 'application/json'}
        return raw
    monkeypatch.setattr(egress, 'build_default_opener', lambda **kw: opener)
    assert cascade.try_cascade(request(), CHAT, failover.AccountsSpent(spent())) is None
    assert sends == ['qwen3.8-max']
    from datetime import datetime, timezone
    account = 'w4:qwen-token-plan:' + spend_guard.account_key('fake-target-token')
    cooling = LedgerView().cooling(account)
    assert cooling.reason == 'spent' and cooling.source == 'unified-reset'
    assert cooling.until == int(datetime(datetime.now(timezone.utc).year, 12, 31, 23, 59, 59,
                                         tzinfo=timezone.utc).timestamp())
    assert cascade.try_cascade(request(RESPONSES), RESPONSES,
                               failover.AccountsSpent(spent())) is None
    assert sends == ['qwen3.8-max']


@pytest.mark.parametrize('path', [CHAT, RESPONSES])
def test_bound_tools_and_supported_output_limit_survive(fakes, path):
    token_plan_config(fakes, [token_plan_entry()])
    payload = request(path)
    payload['max_output_tokens' if path == RESPONSES else 'max_tokens'] = 128
    fn = {'name': 'lookup', 'parameters': {'type': 'object'}}
    payload['tools'] = [dict(fn, type='function')] if path == RESPONSES else [
        {'type': 'function', 'function': fn}]
    if path == CHAT:
        payload['stop'] = 'THE_END'
        payload['messages'] += [
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'old/id', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 'old/id', 'content': 'found'}]
    else:
        payload['input'] += [
            {'type': 'function_call', 'call_id': 'old/id', 'name': 'lookup', 'arguments': '{}'},
            {'type': 'function_call_output', 'call_id': 'old/id', 'output': 'found'}]
    selected = cascade.try_cascade(payload, path, failover.AccountsSpent(spent()))
    assert selected is not None
    sent = fakes.calls[0]
    assert sent['max_tokens'] == 128
    assert sent['tools'][0]['function']['name'] == 'lookup'
    if path == CHAT:
        assert sent['stop'] == ['THE_END']
    call = next(m['tool_calls'][0] for m in sent['messages'] if m.get('tool_calls'))
    result = next(m for m in sent['messages'] if m['role'] == 'tool')
    assert call['id'] == result['tool_call_id'] == cascade.normal_id('old/id')
    selected[0].close()


@pytest.mark.parametrize('path', [CHAT, RESPONSES])
@pytest.mark.parametrize('stream', [False, True])
def test_bound_subscription_ingress_relay(fakes, monkeypatch, path, stream):
    token_plan_config(fakes, [token_plan_entry()])
    strict = fakes.config.with_name('strict-ingress.json')
    strict.write_text('{"paid_opt_in": []}')
    monkeypatch.setenv(spend_guard.CONFIG_ENV, str(strict))
    handler = dispatch(request(path, stream), path,
                       pool([spend_guard.SpendRefused(spent())], []))
    assert handler.status == 200
    assert b'switched' in handler.wfile.getvalue()
    assert len(fakes.calls) == 1


def test_bound_unsigned_decision_stops_before_target(fakes, monkeypatch):
    token_plan_config(fakes, [token_plan_entry()])
    monkeypatch.setattr(gateway_witness, 'append_thread_row',
                        lambda *args, **kwargs: SimpleNamespace(written=False))
    assert cascade.try_cascade(request(), CHAT, failover.AccountsSpent(spent())) is None
    assert not fakes.calls
    assert rows(fakes)[-1]['cascade_refusals'] == [
        {'provider': 'qwen-token-plan', 'reason': 'unsigned'}]


def test_send_revalidates_binding_before_secret_lookup(fakes, monkeypatch):
    from dataclasses import replace
    from lib.hal.cascade_target import validate
    bound = validate(token_plan_entry())
    assert bound is not None
    forged = replace(bound, base_url=egress.PROVIDER_ENDPOINTS['qwen'][0])
    def forbidden(_):
        pytest.fail('secret lookup')
    with pytest.raises(ValueError, match='invalid bound destination'):
        egress.forward_same_wire('qwen-token-plan', CHAT, b'{}',
            opener=lambda *args: pytest.fail('network'), key_resolver=forbidden,
            binding=forged)


@pytest.mark.parametrize('field,value', [
    ('provider', 'qwen'), ('product', 'other-plan'), ('model', 'deepseek-v4-pro'),
    ('base_url', egress.PROVIDER_ENDPOINTS['qwen'][0]),
    ('credential', 'hal-dashscope'), ('auth', 'x-api-key'),
    ('billing', 'paid'), ('transport', 'anthropic'), ('window', 1),
])
def test_send_revalidates_every_binding_field(fakes, monkeypatch, field, value):
    from dataclasses import replace
    from lib.hal.cascade_target import validate
    bound = validate(token_plan_entry())
    assert bound is not None
    forged = replace(bound, **{field: value})
    with pytest.raises(ValueError, match='invalid bound'):
        egress.forward_same_wire('qwen-token-plan', CHAT,
            json.dumps({'model': bound.model}).encode(), binding=forged,
            opener=lambda *args: pytest.fail('network'),
            key_resolver=lambda _: pytest.fail('secret lookup'))


@pytest.mark.parametrize('path,model', [
    (RESPONSES, 'qwen3.8-max'), ('/v1/other', 'qwen3.8-max'),
    (CHAT, 'deepseek-v4-pro'), (CHAT, 'qwen-token-plan/deepseek-v4-pro'),
])
def test_bound_send_refuses_wrong_operation_or_model_before_secret(fakes, path, model):
    from lib.hal.cascade_target import validate
    bound = validate(token_plan_entry())
    assert bound is not None
    with pytest.raises(ValueError, match='invalid bound'):
        egress.forward_same_wire('qwen-token-plan', path,
            json.dumps({'model': model}).encode(), binding=bound,
            opener=lambda *args: pytest.fail('network'),
            key_resolver=lambda _: pytest.fail('secret lookup'))


def test_bound_send_checks_rewritten_model_before_secret(fakes, monkeypatch):
    from lib.hal.cascade_target import validate
    bound = validate(token_plan_entry())
    assert bound is not None
    monkeypatch.setattr(egress, '_wire_body', lambda *args: b'{"model":"other"}')
    with pytest.raises(ValueError, match='invalid bound model'):
        egress.forward_same_wire('qwen-token-plan', CHAT,
            json.dumps({'model': bound.model}).encode(), binding=bound,
            opener=lambda *args: pytest.fail('network'),
            key_resolver=lambda _: pytest.fail('secret lookup'))


def test_bound_send_checks_incoming_model_before_rewrite(fakes, monkeypatch):
    from lib.hal.cascade_target import validate
    bound = validate(token_plan_entry())
    assert bound is not None
    monkeypatch.setattr(egress, '_wire_body',
                        lambda *args: json.dumps({'model': bound.model}).encode())
    with pytest.raises(ValueError, match='invalid bound model'):
        egress.forward_same_wire('qwen-token-plan', CHAT, b'{"model":"other"}',
            binding=bound, opener=lambda *args: pytest.fail('network'),
            key_resolver=lambda _: pytest.fail('secret lookup'))


def test_chat_tool_choice_none_refused_before_secret(fakes, monkeypatch):
    token_plan_config(fakes, [token_plan_entry()])
    payload = request()
    payload['tools'] = [{'type': 'function', 'function': {'name': 'lookup'}}]
    payload['tool_choice'] = 'none'
    monkeypatch.setattr(egress, '_default_key_resolver',
                        lambda _: pytest.fail('secret lookup'))
    assert cascade.try_cascade(payload, CHAT, failover.AccountsSpent(spent())) is None
    assert not fakes.calls


def test_w4_bound_candidate_requests_redirect_refusal(fakes, monkeypatch):
    token_plan_config(fakes, [token_plan_entry()])
    modes = []
    original = egress.build_default_opener
    def capture(**kwargs):
        modes.append(kwargs.get('subscription_bound'))
        return original(**kwargs)
    monkeypatch.setattr(egress, 'build_default_opener', capture)
    chosen = cascade.try_cascade(request(), CHAT, failover.AccountsSpent(spent()))
    assert chosen is not None
    assert modes == [True]
    chosen[0].close()


def test_bound_guard_runs_before_secret_lookup(fakes, monkeypatch):
    from lib.hal.cascade_target import validate
    bound = validate(token_plan_entry())
    assert bound is not None
    class Guarded(Exception):
        pass
    monkeypatch.setattr(egress, 'require_openai_destination',
                        lambda _: (_ for _ in ()).throw(Guarded()))
    with pytest.raises(Guarded):
        egress.forward_same_wire('qwen-token-plan', CHAT,
            json.dumps({'model': bound.model}).encode(), binding=bound,
            opener=lambda *args: pytest.fail('network'),
            key_resolver=lambda _: pytest.fail('secret lookup'))


def test_bound_secret_absent_from_logs_and_ledger(fakes, caplog, _isolated_hal_home):
    token_plan_config(fakes, [token_plan_entry()])
    chosen = cascade.try_cascade(request(), CHAT, failover.AccountsSpent(spent()))
    assert chosen is not None
    chosen[0].close()
    ledger = _isolated_hal_home / 'state/hal/account-ledger.json'
    assert 'fake-target-token' not in caplog.text
    assert ledger.exists()
    assert 'fake-target-token' not in ledger.read_text()
