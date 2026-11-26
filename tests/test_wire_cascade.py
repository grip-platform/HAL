"""W4 contracts, fake credentials and actual loopback target request logs."""
import copy
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from email.utils import format_datetime
from types import SimpleNamespace

import pytest

from lib.hal import cascade_policy as policy, wire_cascade as cascade

REAL_LOADER = policy.load_shield

ENTRY = dict(provider='moonshot', model='kimi-k3', base_url='https://target.invalid/v1',
             credential='fake-target-credential', billing='subscription')


@pytest.fixture(autouse=True)
def _healthy_attempts(monkeypatch):
    monkeypatch.setattr(cascade.Attempts, '_unhealthy', False, raising=False)


def test_attempts_counts_are_per_key_memory_only(tmp_path, monkeypatch):
    monkeypatch.setenv('HAL_HOME', str(tmp_path))
    attempts = cascade.Attempts()
    a, b = 'a' * 64, 'b' * 64
    attempts.end(a)
    attempts.begin(a)
    attempts.begin(a)
    attempts.begin(b)
    attempts.end(a)
    assert not (tmp_path / 'state/hal/w4-attempts').exists()
    assert not attempts.reserve(a) and not attempts.reserve(b)
    attempts.end(a)
    attempts.end(a)
    assert attempts.reserve(a)
    assert not attempts.reserve(b)
    assert not (tmp_path / 'state/hal/w4-attempts' / b).exists()


def test_attempts_closure_survives_release(tmp_path, monkeypatch):
    monkeypatch.setenv('HAL_HOME', str(tmp_path))
    attempts = cascade.Attempts()
    key = 'a' * 64
    assert attempts.reserve(key)
    attempts.close(key)
    attempts.release(key)
    assert not cascade.Attempts().reserve(key)
    assert (tmp_path / 'state/hal/w4-attempts' / (key + '.closed')).exists()


def test_move_explicitly_refuses_closure_while_reserved(monkeypatch):
    from lib.hal import wire_hold, wire_proxy
    payload = {'messages': [{'role': 'user', 'content': 'fake'}]}
    body = wire_proxy._Copy()
    body(json.dumps(payload).encode())
    ex = wire_proxy._Exchange('POST', '/v1/messages',
        (('x-claude-code-session-id', 'fake-session'),), time.monotonic(),
        body=body, wire_body=body)
    attempts = cascade.Attempts()
    proxy = SimpleNamespace(_w4_attempts=attempts)
    key = wire_hold.turn_key(ex.headers, payload)
    assert attempts.reserve(key)
    wire_hold.close_turn(proxy, ex)
    monkeypatch.setattr(wire_hold, 'exhaustion', lambda *args: (True, None))
    monkeypatch.setattr(cascade, 'configured_targets', lambda: [object()])
    monkeypatch.setattr(cascade, 'eligible', lambda ex: payload)
    monkeypatch.setattr(cascade, '_serve', lambda *args: pytest.fail('closed turn reached W4'))
    assert cascade.move(proxy, None, ex) is False


def test_failed_closure_poison_is_process_wide(tmp_path, monkeypatch):
    monkeypatch.setenv('HAL_HOME', str(tmp_path))
    real_open = cascade.os.open
    def fail_closure(path, flags, mode=0o777):
        if str(path).endswith('.closed'):
            raise OSError('disk failure')
        return real_open(path, flags, mode)
    monkeypatch.setattr(cascade.os, 'open', fail_closure)
    attempts = cascade.Attempts()
    attempts.close('a' * 64)
    assert not attempts.reserve('a' * 64)
    assert not attempts.reserve('b' * 64)
    assert not cascade.Attempts().reserve('c' * 64)
    monkeypatch.setattr(cascade.Attempts, '_unhealthy', False)
    assert not cascade.Attempts().reserve('d' * 64)


def test_failed_closure_without_marker_keeps_process_closed(monkeypatch):
    real_open = cascade.os.open
    def fail_both(path, flags, mode=0o777):
        if str(path).endswith(('.closed', 'UNHEALTHY')):
            raise OSError('disk failure')
        return real_open(path, flags, mode)
    monkeypatch.setattr(cascade.os, 'open', fail_both)
    attempts = cascade.Attempts()
    attempts.close('a' * 64)
    assert cascade.Attempts._unhealthy
    assert not attempts.reserve('b' * 64)
    assert not cascade.Attempts().reserve('c' * 64)


def test_begin_during_reservation_has_one_order(tmp_path, monkeypatch):
    """A send that begins while the reservation's file is being created.

    INVERTED after Astra's fourth review. This test used to require that `begin`
    WAITED behind the reservation's file work (the lock was held across it), which made
    every ordinary send wait on a W4 reservation's disk. `begin` now never waits; the
    one order the two calls then have is "the send first", so the reservation is
    withdrawn. The join is bounded, so a `begin` made to wait fails this test in two
    seconds and cannot hang the run."""
    import threading
    monkeypatch.setenv('HAL_HOME', str(tmp_path))
    attempts, real_open, key, began_at_once = cascade.Attempts(), cascade.os.open, 'a' * 64, []

    def open_while_a_send_begins(path, flags, mode=0o777):
        sender = threading.Thread(target=attempts.begin, args=(key,), daemon=True)
        sender.start()
        sender.join(2)
        began_at_once.append(not sender.is_alive())
        return real_open(path, flags, mode)

    monkeypatch.setattr(cascade.os, 'open', open_while_a_send_begins)
    reserved = attempts.reserve(key)
    assert began_at_once == [True]
    assert not reserved and not attempts._path(key).exists()
    attempts.end(key)
    monkeypatch.setattr(cascade.os, 'open', real_open)
    assert attempts.reserve(key)


def test_a_send_that_came_and_went_unanswered_during_reservation_refuses_it(tmp_path, monkeypatch):
    """HAL#748 items 1 and 2. A send that began AND ended while the reservation's file
    was being made is gone from the counts when they are read again. Its closure still
    refuses the reservation, and the refused reservation leaves no file behind."""
    monkeypatch.setenv('HAL_HOME', str(tmp_path))
    attempts, real_open, key = cascade.Attempts(), cascade.os.open, 'a' * 64

    def open_while_a_send_comes_and_goes(path, flags, mode=0o777):
        fd = real_open(path, flags, mode)
        if str(path).endswith(key):
            attempts.begin(key)
            attempts.close(key)  # the send ended with no answer
            attempts.end(key)
        return fd

    monkeypatch.setattr(cascade.os, 'open', open_while_a_send_comes_and_goes)
    assert attempts.reserve(key) is False
    assert attempts._outstanding == {} and attempts.closed(key)
    assert not attempts._path(key).exists()


def test_answered_turns_are_remembered_in_bounded_memory(monkeypatch):
    """HAL#748 item 4: only a 2xx answer notes a turn, and only the newest are kept."""
    monkeypatch.setattr(cascade, 'ANSWERED_MEMORY', 3)
    attempts = cascade.Attempts()
    first, refused, *later = [letter * 64 for letter in 'abcde']
    attempts.begin(first)
    attempts.end(first, 200)
    assert attempts.was_answered(first) and attempts._outstanding == {}
    for status in (None, 199, 300, 429, 529, '200'):
        attempts.end(refused, status)
    assert not attempts.was_answered(refused)
    for key in later:
        attempts.end(key, 201)
    attempts.end(later[0], 299)  # answered again: it becomes the newest
    attempts.end(refused, 204)
    assert not attempts.was_answered(first)
    assert all(attempts.was_answered(key) for key in (later[0], later[2], refused))
    assert not attempts.was_answered(later[1])


def test_attempts_are_atomic_durable_and_validate_keys(tmp_path, monkeypatch):
    monkeypatch.setenv('HAL_HOME', str(tmp_path))
    key = 'a' * 64
    first, second = cascade.Attempts(), cascade.Attempts()
    assert first.reserve(key)
    path = tmp_path / 'state/hal/w4-attempts' / key
    assert path.is_file() and path.stat().st_size == 0
    assert not second.reserve(key)
    first.release(key)
    assert second.reserve(key)
    second.close(key)
    assert not first.reserve(key)
    before = set(path.parent.iterdir())
    for bad in ('a/b', '.', 'A' * 64, 'a' * 63):
        assert not first.reserve(bad)
        first.close(bad)
        first.release(bad)
    assert set(path.parent.iterdir()) == before
    later = tmp_path / 'later-home'
    monkeypatch.setenv('HAL_HOME', str(later))
    assert first.reserve('c' * 64)
    assert (later / 'state/hal/w4-attempts' / ('c' * 64)).is_file()


def test_attempts_unwritable_root_fails_closed(tmp_path, monkeypatch):
    home = tmp_path / 'not-a-directory'
    home.write_text('occupied')
    monkeypatch.setenv('HAL_HOME', str(home))
    assert not cascade.Attempts().reserve('b' * 64)


def test_deadline_after_independent_prevents_second_signature(monkeypatch):
    clock, signed, connected = [0.0], [], []
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    def lookup(target):
        if target.label == 'second':
            clock[0] = 2.0
        return 'key'
    monkeypatch.setattr(cascade, 'independent', lookup)
    monkeypatch.setattr(cascade, '_sign_decision',
                        lambda proxy, ex, target: signed.append(target.label) or True)
    monkeypatch.setattr(cascade, '_attempt',
                        lambda proxy, client, ex, target, key, body, outcome: (connected.append(target.label) or None, None))
    ex = SimpleNamespace(headers=(), hold_deadline=1.0)
    targets = [SimpleNamespace(label='first'), SimpleNamespace(label='second')]
    assert cascade._serve(None, None, ex, {}, targets) is False
    assert signed == connected == ['first']


def test_connect_rechecks_deadline_after_resolution(monkeypatch):
    import socket
    clock = [0.0]
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(socket, 'create_connection',
                        lambda *args, **kwargs: pytest.fail('connected after deadline'))
    proxy = SimpleNamespace(_resolve=lambda *_: (clock.__setitem__(0, 2.0) or ('fake', 443)))
    with pytest.raises(policy.Refused, match='hold_budget_expired'):
        cascade._connect(proxy, cascade.Target(**ENTRY, window=100),
                         SimpleNamespace(hold_deadline=1.0))


def test_connect_timeout_is_bounded_by_remaining_budget(monkeypatch):
    import socket
    clock, timeouts = [0.0], []
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    def connect(address, timeout):
        timeouts.append(timeout)
        raise TimeoutError('fake connect timeout')
    monkeypatch.setattr(socket, 'create_connection', connect)
    proxy = SimpleNamespace(_resolve=lambda *_: (clock.__setitem__(0, .75) or ('fake', 443)))
    with pytest.raises(TimeoutError):
        cascade._connect(proxy, cascade.Target(**ENTRY, window=100),
                         SimpleNamespace(hold_deadline=1.0))
    assert 0 < timeouts[0] <= .25


@pytest.mark.parametrize('connect_took,handshake', [(6.0, 2.0), (9.0, None)])
def test_the_tls_handshake_gets_only_what_the_connection_left_of_the_budget(
        monkeypatch, connect_took, handshake):
    """Codex on HAL#747: the socket kept the timeout its connection was given, so a slow
    handshake could wait that long again and overrun the hold deadline. With the budget
    used up by the connection, no handshake starts and the socket is closed."""
    import socket
    from lib.hal import wire_proxy
    clock, seen = [0.0], []

    class Raw:
        def settimeout(self, value):
            seen.append(('handshake', value))

        def close(self):
            seen.append('closed')

    class Tls:
        def settimeout(self, value):
            seen.append(('idle', value))

    def connect(address, timeout):
        seen.append(('connect', timeout))
        clock[0] += connect_took
        return Raw()

    def wrap(raw, server_hostname):
        assert handshake is not None, 'a handshake began after the budget was gone'
        return Tls()

    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(socket, 'create_connection', connect)
    proxy = SimpleNamespace(_resolve=lambda *_: ('fake', 443), pool=SimpleNamespace(
        dialer=SimpleNamespace(_context=SimpleNamespace(wrap_socket=wrap))))
    target, ex = cascade.Target(**ENTRY, window=100), SimpleNamespace(hold_deadline=8.0)
    if handshake is None:
        with pytest.raises(policy.Refused, match='hold_budget_expired'):
            cascade._connect(proxy, target, ex)
        assert seen == [('connect', 8.0), 'closed']
    else:
        cascade._connect(proxy, target, ex)
        assert seen == [('connect', 8.0), ('handshake', handshake),
                        ('idle', wire_proxy._IDLE_TIMEOUT_S)]


def test_presend_timeout_is_safe_for_next_attempt(monkeypatch):
    target = cascade.Target(**ENTRY, window=100000)
    ex = SimpleNamespace(headers=(), hold_deadline=None, w4_uncertain=False)
    calls = []
    monkeypatch.setattr(cascade, 'transform', lambda *args: {})
    monkeypatch.setattr(policy, 'prepare', lambda *args: {})
    monkeypatch.setattr(cascade, '_request', lambda *args: (None, b''))
    monkeypatch.setattr(cascade, '_record', lambda *args: None)
    def connect(*args):
        calls.append(1)
        raise TimeoutError('before request byte')
    monkeypatch.setattr(cascade, '_connect', connect)
    assert cascade._attempt(None, None, ex, target, 'fake', {}, 'refused') == (None, None)
    assert not ex.w4_uncertain
    assert cascade._attempt(None, None, ex, target, 'fake', {}, 'refused') == (None, None)
    assert calls == [1, 1]


def test_timeout_after_target_send_is_uncertain(monkeypatch):
    from contextlib import nullcontext
    from lib.hal import account_ledger
    target = cascade.Target(**ENTRY, window=100)
    ex = SimpleNamespace(headers=(), hold_deadline=None, w4_uncertain=False)
    sent = []
    def send(data):
        sent.append(data)
        raise TimeoutError('sent then timed out')
    conn = SimpleNamespace(sock=SimpleNamespace(sendall=send,
        shutdown=lambda *_: None, close=lambda: None))
    monkeypatch.setattr(cascade, 'transform', lambda *args: {})
    monkeypatch.setattr(policy, 'prepare', lambda *args: {})
    monkeypatch.setattr(cascade, '_request', lambda *args:
        (SimpleNamespace(raw=b'POST fake', headers=()), b'body'))
    monkeypatch.setattr(cascade, '_connect', lambda *args: conn)
    monkeypatch.setattr(cascade, '_record', lambda *args: None)
    monkeypatch.setattr(account_ledger, 'AccountLedger', lambda **kwargs:
        SimpleNamespace(exchange=lambda *args: nullcontext(None)))
    assert cascade._attempt(None, None, ex, target, 'fake', {}, 'refused') == (None, 'stop')
    assert sent == [b'POST fakebody'] and ex.w4_uncertain


@pytest.mark.parametrize('failure', [ConnectionRefusedError('after the send'),
                                     policy.Refused('raised after the send')])
def test_no_failure_after_sending_began_proves_the_target_refused(monkeypatch, failure):
    """Once request bytes may have left, the class of an exception says nothing about
    whether the target accepted them: the attempt is uncertain and no later target is
    tried. (Astra's fourth review: a ConnectionRefusedError while reading the answer was
    classed as "before send".)"""
    target = cascade.Target(**ENTRY, window=100)
    ex = SimpleNamespace(headers=(), hold_deadline=None, w4_uncertain=False)

    def exchange(proxy, client, exchange_, conn, *rest):
        exchange_.w4_sending = True
        raise failure

    monkeypatch.setattr(cascade, 'transform', lambda *args: {})
    monkeypatch.setattr(policy, 'prepare', lambda *args: {})
    monkeypatch.setattr(cascade, '_request', lambda *args: (None, b''))
    monkeypatch.setattr(cascade, '_connect', lambda *args: SimpleNamespace(
        sock=SimpleNamespace(shutdown=lambda *_: None, close=lambda: None)))
    monkeypatch.setattr(cascade, '_exchange', exchange)
    reasons = []
    monkeypatch.setattr(cascade, '_record', lambda proxy, ex, label, reason, outcome:
                        reasons.append(str(reason)))
    assert cascade._attempt(None, None, ex, target, 'fake', {}, 'refused') == (None, 'stop')
    assert ex.w4_uncertain and reasons == ['acceptance_uncertain']


def test_an_ordinary_send_never_waits_behind_a_reservations_disk_work(tmp_path, monkeypatch):
    """`begin` and `end` sit on the path of every turn. A reservation's file work must
    not hold the lock they take."""
    monkeypatch.setenv('HAL_HOME', str(tmp_path))
    attempts, real_open, lock_was_free = cascade.Attempts(), cascade.os.open, []

    def watched_open(path, flags, mode=0o777):
        free = attempts._lock.acquire(blocking=False)
        lock_was_free.append(free)
        if free:
            attempts._lock.release()
        return real_open(path, flags, mode)

    monkeypatch.setattr(cascade.os, 'open', watched_open)
    assert attempts.reserve('a' * 64)
    assert lock_was_free == [True]


def test_a_failure_noticed_while_a_reservation_is_made_withdraws_it(tmp_path, monkeypatch):
    """A closure write fails in another thread while this reservation's file is being
    created. The reservation must not authorise an offer, and it is removed again."""
    monkeypatch.setenv('HAL_HOME', str(tmp_path))
    attempts, real_open, key = cascade.Attempts(), cascade.os.open, 'a' * 64

    def open_while_a_closure_fails(path, flags, mode=0o777):
        fd = real_open(path, flags, mode)
        cascade.Attempts._unhealthy = True
        return fd

    monkeypatch.setattr(cascade.os, 'open', open_while_a_closure_fails)
    assert not attempts.reserve(key)
    assert not attempts._path(key).exists()


def test_a_session_with_a_request_on_its_way_blocks_a_reservation(tmp_path, monkeypatch):
    """A streamed request's turn key is unknown until its body has been read, so the
    wire marks the SESSION for that window. A reservation names its session too."""
    monkeypatch.setenv('HAL_HOME', str(tmp_path))
    attempts, key = cascade.Attempts(), 'a' * 64
    attempts.begin('fake-session')
    assert not attempts.reserve(key, 'fake-session')
    assert attempts.reserve('b' * 64, 'other-session')
    attempts.end('fake-session')
    assert attempts.reserve(key, 'fake-session')
    assert not (tmp_path / 'state/hal/w4-attempts' / 'fake-session').exists()


def test_deadline_after_signature_prevents_connection(monkeypatch):
    clock, connected = [0.0], []
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(cascade, '_request', lambda *args: (None, b''))
    monkeypatch.setattr(cascade, 'transform', lambda *args: {})
    monkeypatch.setattr(policy, 'prepare', lambda *args: {})
    monkeypatch.setattr(cascade, '_connect', lambda *args: connected.append(1))
    monkeypatch.setattr(cascade, '_record', lambda *args: None)
    ex = SimpleNamespace(headers=(), hold_deadline=1.0, w4_uncertain=False)
    clock[0] = 2.0
    assert cascade._attempt(None, None, ex, SimpleNamespace(label='fake', window=100),
                            'key', {}, 'refused') == (None, None)
    assert connected == []


def test_hold_deadline_is_checked_at_target_send(monkeypatch):
    from contextlib import nullcontext
    from lib.hal import account_ledger
    sent = []
    class Ledger:
        def __init__(self, *args, **kwargs):
            pass
        def exchange(self, headers):
            return nullcontext()
    monkeypatch.setattr(account_ledger, 'AccountLedger', Ledger)
    monkeypatch.setattr(time, 'monotonic', lambda: 2.0)
    ex = SimpleNamespace(headers=(), method='POST', hold_deadline=1.0)
    conn = SimpleNamespace(sock=SimpleNamespace(sendall=sent.append))
    request = (SimpleNamespace(headers=(), raw=b'fake request'), b'body')
    proxy = SimpleNamespace(_response_head=lambda *args, **kwargs: None)
    with pytest.raises(policy.Refused, match='hold_budget_expired'):
        cascade._exchange(proxy, None, ex, conn, SimpleNamespace(provider='fake'),
                          'fake-key', request)
    assert sent == []


def test_no_later_target_is_signed_or_tried_once_the_hold_deadline_passes(monkeypatch):
    """The first target's attempt runs the clock past the hold deadline. The second
    target gets no signed decision and no attempt: a decision row signed after the
    budget is over would record a move that can no longer happen."""
    clock, signed, tried = [0.0], [], []
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(cascade, 'independent', lambda target: 'fake-key')
    monkeypatch.setattr(cascade, '_sign_decision',
                        lambda proxy, ex, target: signed.append(target.label) or True)

    def attempt(proxy, client, ex, target, key, body, outcome):
        tried.append(target.label)
        clock[0] = 2.0
        return None, None

    monkeypatch.setattr(cascade, '_attempt', attempt)
    ex = SimpleNamespace(headers=(), hold_deadline=1.0)
    targets = [SimpleNamespace(label='first'), SimpleNamespace(label='second')]
    assert cascade._serve(None, None, ex, {}, targets) is False
    assert (signed, tried) == (['first'], ['first'])


class _TargetSocket:
    """A target's socket that records every timeout it is given."""

    def __init__(self, recv):
        self.timeouts, self.sent, self._recv = [], [], recv

    def settimeout(self, value):
        self.timeouts.append(value)

    def sendall(self, data):
        self.sent.append(data)

    def recv(self, size):
        return self._recv(self)


def _exchange_with_target(monkeypatch, recv, deadline, now):
    """Set up one `_exchange` with a target socket; returns (socket, conn, ex, run)."""
    from contextlib import nullcontext
    from lib.hal import account_ledger, wire_proxy
    monkeypatch.setattr(time, 'monotonic', now)
    monkeypatch.setattr(account_ledger, 'AccountLedger', lambda **kwargs: SimpleNamespace(
        exchange=lambda *args: nullcontext(SimpleNamespace(answered=lambda *_: None))))
    sock = _TargetSocket(recv)
    conn = SimpleNamespace(sock=sock, stream=wire_proxy._Stream(sock))
    ex = SimpleNamespace(headers=(), method='POST', hold_deadline=deadline, client_started=False)
    request = (SimpleNamespace(headers=(), raw=b'fake request'), b'body')
    proxy = SimpleNamespace(_response_head=wire_proxy.WireProxy._response_head)

    def run():
        return cascade._exchange(proxy, None, ex, conn, SimpleNamespace(provider='fake'),
                                 'fake-key', request)

    return sock, conn, ex, run


def test_a_silent_target_is_waited_on_only_while_the_hold_budget_lasts(monkeypatch):
    """Codex on HAL#747: after the send, the wait for the target's answer used the
    wire's 900-second idle timeout, whatever was left of the hold's budget."""
    import socket
    from lib.hal import wire_proxy
    clock = [100.0]

    def silent(sock):
        clock[0] += sock.timeouts[-1]  # the read waits out exactly the timeout it was given
        raise socket.timeout('the target said nothing')

    sock, conn, ex, run = _exchange_with_target(monkeypatch, silent, 130.0, lambda: clock[0])
    with pytest.raises(TimeoutError):
        run()
    assert sock.sent == [b'fake requestbody'] and ex.w4_sending
    assert sock.timeouts == [30.0, 30.0] and clock[0] == 130.0
    assert conn.stream._fill.__func__ is wire_proxy._Stream._fill


def test_a_target_that_drips_bytes_cannot_stretch_the_wait_past_the_budget(monkeypatch):
    """The time left is worked out before every read, not once: a target that sends one
    byte before each timeout would otherwise never be cut off."""
    clock = [0.0]

    def drip(sock):
        clock[0] += 100.0
        return b'H'

    sock, _conn, _ex, run = _exchange_with_target(monkeypatch, drip, 250.0, lambda: clock[0])
    with pytest.raises(TimeoutError, match='hold budget'):
        run()
    assert sock.timeouts == [250.0, 250.0, 150.0, 50.0]


def test_once_the_targets_head_is_in_the_idle_timeout_applies_again(monkeypatch):
    from lib.hal import wire_proxy
    parts = [b'HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n']
    sock, conn, _ex, run = _exchange_with_target(monkeypatch, lambda sock: parts.pop(0),
                                                 30.0, lambda: 0.0)
    answer, cooldown = run()
    assert wire_proxy._status(answer) == 200 and cooldown is None
    assert sock.timeouts == [30.0, 30.0, wire_proxy._IDLE_TIMEOUT_S]
    assert conn.stream._fill.__func__ is wire_proxy._Stream._fill


def test_a_move_without_a_hold_deadline_keeps_the_ordinary_wait(monkeypatch):
    parts = [b'HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n']
    sock, _conn, _ex, run = _exchange_with_target(monkeypatch, lambda sock: parts.pop(0),
                                                  None, lambda: 0.0)
    assert run()[1] is None
    assert sock.timeouts == []


def test_a_budget_that_runs_out_just_before_the_send_sends_nothing(monkeypatch):
    """The deadline passes between the last check and the send: no byte leaves, and
    `w4_sending` stays unset, so the failure is recorded as safe."""
    ticks = iter([0.5, 2.0])
    sock, _conn, ex, run = _exchange_with_target(
        monkeypatch, lambda sock: pytest.fail('read from the target'), 1.0, lambda: next(ticks))
    with pytest.raises(TimeoutError, match='hold budget'):
        run()
    assert sock.sent == [] and not getattr(ex, 'w4_sending', False)


def test_target_policy(tmp_path, monkeypatch):
    """configured_targets() keeps its exact pre-refactor behaviour: same
    accepted targets, same fail-closed [] -- for every one of these inputs,
    whether or not config_errors() also has something to say about it."""
    path = tmp_path / 'w4.json'
    monkeypatch.setenv('HAL_W4_CONFIG', str(path))
    assert cascade.configured_targets() == [] and cascade.config_errors() == []
    for value, reason in (('{', 'w4_cascade.json: not valid JSON'),
                           ('{}', None),
                           ('{"anthropic_targets": []}', None),
                           ('{"anthropic_targets":false}', 'anthropic_targets: must be a list')):
        path.write_text(value)
        assert cascade.configured_targets() == [], value
        assert cascade.config_errors() == ([reason] if reason else []), value
    path.write_text(json.dumps({'anthropic_targets': [ENTRY]}))
    assert cascade.configured_targets()[0].label == 'moonshot/kimi-k3'
    assert cascade.config_errors() == []
    for change, reason in (
        ({'accept': ['thinking'], 'drop': ['thinking']}, '"thinking" is in both accept and drop'),
        ({'drop': 'x'}, 'drop must be a list'),
        ({'auth': False}, 'auth must be "x-api-key" or "bearer"'),
        ({'system_messages': 1}, 'system_messages must be "native" or "fold"'),
        ({'billing': None}, 'billing must be a non-empty single-line string'),
        ({'provider': 'kimi'}, 'provider "kimi" is not in config/providers.yaml'),
        ({'model': 'absent'}, 'model "absent" is not catalogued for provider "moonshot"'),
        ({'base_url': 'http://target.invalid'}, 'base_url must be https'),
        ({'drop_tool_fields': [False]}, 'drop_tool_fields must be a list of non-empty strings'),
        ({'credential': ''}, 'credential must be a non-empty single-line string'),
    ):
        path.write_text(json.dumps({'anthropic_targets': [dict(ENTRY, **change)]}))
        assert cascade.configured_targets() == [], change
        assert cascade.config_errors() == [f'anthropic_targets[0]: {reason}'], change


def test_target_policy_missing_required_field(tmp_path, monkeypatch):
    """A deleted required key cannot be reached via dict(ENTRY, **change)."""
    path = tmp_path / 'w4.json'
    monkeypatch.setenv('HAL_W4_CONFIG', str(path))
    entry = dict(ENTRY)
    del entry['billing']
    path.write_text(json.dumps({'anthropic_targets': [entry]}))
    assert cascade.configured_targets() == []
    assert cascade.config_errors() == ['anthropic_targets[0]: missing required field "billing"']


def test_target_policy_unknown_field(tmp_path, monkeypatch):
    path = tmp_path / 'w4.json'
    monkeypatch.setenv('HAL_W4_CONFIG', str(path))
    path.write_text(json.dumps({'anthropic_targets': [dict(ENTRY, mystery='x')]}))
    assert cascade.configured_targets() == []
    assert cascade.config_errors() == ['anthropic_targets[0]: unknown field "mystery"']


def test_target_policy_entry_not_an_object(tmp_path, monkeypatch):
    path = tmp_path / 'w4.json'
    monkeypatch.setenv('HAL_W4_CONFIG', str(path))
    path.write_text(json.dumps({'anthropic_targets': ['not-a-dict']}))
    assert cascade.configured_targets() == []
    assert cascade.config_errors() == ['anthropic_targets[0]: must be an object']


@pytest.mark.parametrize('value', ['[1, 2, 3]', '"just a string"', '42', 'null'])
def test_target_policy_top_level_not_an_object(tmp_path, monkeypatch, value):
    path = tmp_path / 'w4.json'
    monkeypatch.setenv('HAL_W4_CONFIG', str(path))
    path.write_text(value)
    assert cascade.configured_targets() == []
    assert cascade.config_errors() == ['w4_cascade.json: must be a JSON object']


@pytest.mark.parametrize('base_url,reason', [
    ('https:///v1', 'base_url must include a hostname'),
    ('https://evil:pw@target.invalid/v1', 'base_url must not contain a username or password'),
    ('https://target.invalid/v1?x=1', 'base_url must not include a query string or fragment'),
    ('https://target.invalid/v1#frag', 'base_url must not include a query string or fragment'),
    ('https://target.invalid:999999/v1', 'base_url is not a valid URL'),
])
def test_target_policy_base_url_shapes(tmp_path, monkeypatch, base_url, reason):
    path = tmp_path / 'w4.json'
    monkeypatch.setenv('HAL_W4_CONFIG', str(path))
    path.write_text(json.dumps({'anthropic_targets': [dict(ENTRY, base_url=base_url)]}))
    assert cascade.configured_targets() == [], base_url
    assert cascade.config_errors() == [f'anthropic_targets[0]: {reason}'], base_url


def test_config_errors_matches_the_specified_examples(tmp_path, monkeypatch):
    """The exact reasons named when GET /w4/status grew config_errors."""
    path = tmp_path / 'w4.json'
    monkeypatch.setenv('HAL_W4_CONFIG', str(path))
    entries = [dict(ENTRY, provider='kimi'),
               dict(ENTRY, base_url='http://target.invalid'),
               dict(ENTRY, accept=['thinking'], drop=['thinking'])]
    path.write_text(json.dumps({'anthropic_targets': entries}))
    assert cascade.configured_targets() == []
    assert cascade.config_errors() == [
        'anthropic_targets[0]: provider "kimi" is not in config/providers.yaml',
        'anthropic_targets[1]: base_url must be https',
        'anthropic_targets[2]: "thinking" is in both accept and drop',
    ]


def test_config_errors_never_leaks_the_credential_value(tmp_path, monkeypatch):
    """Whatever else a reason may name, the credential's actual value --
    even smuggled into base_url -- never appears in one."""
    path = tmp_path / 'w4.json'
    monkeypatch.setenv('HAL_W4_CONFIG', str(path))
    marker = 'SECRET-DO-NOT-LEAK-7f3c9a'
    marked = dict(ENTRY, credential=marker)
    changes = [{}, {'accept': ['thinking'], 'drop': ['thinking']}, {'drop': 'x'}, {'auth': False},
               {'system_messages': 1}, {'billing': None}, {'provider': 'kimi'}, {'model': 'absent'},
               {'base_url': 'http://target.invalid'}, {'drop_tool_fields': [False]},
               {'base_url': f'https://{marker}.invalid/v1'}, {'mystery': marker}]
    for change in changes:
        path.write_text(json.dumps({'anthropic_targets': [dict(marked, **change)]}))
        reasons = cascade.config_errors()
        assert not any(marker in r for r in reasons), (change, reasons)


def test_real_loader_missing_and_import_priority(tmp_path, monkeypatch):
    monkeypatch.setattr(policy, '_shield', None)
    monkeypatch.setattr(policy, '_shield_retry_at', 0.0)
    monkeypatch.setenv('CLAUDE_DIR', str(tmp_path))
    first = sys.path[0]
    assert policy.load_shield() is None
    with pytest.raises(policy.Refused, match='shield'):
        policy.prepare({'messages': []}, 99999)
    lib = tmp_path / 'lib'
    lib.mkdir()
    (lib / 'review_redaction.py').write_text('import sys\nfrom pathlib import Path\np = str(Path(__file__).resolve().parent)\nif p not in sys.path: sys.path.insert(0, p)\ndef shield_payload(body): return body\n')
    monkeypatch.setattr(policy, '_shield_retry_at', 0.0)
    monkeypatch.delitem(sys.modules, 'review_redaction', raising=False)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    assert policy.load_shield()({'text': 'safe'}) == {'text': 'safe'}
    assert sys.path[0] == first
    monkeypatch.delitem(sys.modules, 'review_redaction', raising=False)


def test_hal_lib_reserves_the_names_the_wire_imports_from_grip():
    # load_shield imports GRIP's review_redaction by bare name and refuses when another
    # file answers ('import priority changed'). Any HAL module that puts lib/ first on
    # sys.path would then hand it HAL's file, or HAL's shield to GRIP's adapter, and the
    # wire would refuse every request. Mutation: add either file and this goes red.
    from pathlib import Path
    lib = Path(__file__).resolve().parents[1] / 'lib'
    assert not (lib / 'review_redaction.py').exists()
    assert not (lib / 'redaction_shield.py').exists()


def test_shared_final_shield_and_bound(monkeypatch):
    monkeypatch.setattr(policy, 'load_shield', lambda: lambda body: dict(body, text='x' * 200))
    with pytest.raises(policy.Refused, match='context_unknown_or_small'):
        policy.prepare({'text': 'small', 'max_tokens': 1}, 4200)
    assert policy.prepare({'text': 'small', 'max_tokens': 1}, 5000)['text'] == 'x' * 200
    monkeypatch.setattr(policy, 'load_shield', lambda: lambda body: 1 / 0)
    with pytest.raises(policy.Refused, match='shield'):
        policy.prepare({}, 99999)


@pytest.mark.parametrize('change,reason', [
    ({'context_management': {}}, 'fields'),
    ({'tools': [{'type': 'web_search', 'name': 'search'}]}, 'tools'),
    ({'tools': [{'name': 'find', 'input_schema': {}, 'defer_loading': True}]}, 'tools'),
    ({'messages': [{'role': 'system', 'content': 'instructions'}]}, 'system_messages'),
])
def test_undeclared_refuses(change, reason):
    target = cascade.Target(**ENTRY, window=100000)
    with pytest.raises(policy.Refused, match=reason):
        cascade.transform(target, dict(messages=[{'role': 'user', 'content': 'hi'}], **{k:v for k,v in change.items() if k != 'messages'}) if 'messages' not in change else change)


def test_declared_transform_and_freshness():
    target = cascade.Target(**ENTRY, window=100000, drop=('thinking', 'context_management'),
                            drop_tool_fields=('defer_loading',), system_messages='fold', accept=('output_config',))
    body = {'system': 'start', 'metadata': {'user_id': 'private'}, 'thinking': {}, 'context_management': {},
            'output_config': {}, 'messages': [{'role': 'user', 'content': 'one'},
              {'role': 'system', 'content': 'folded'}, {'role': 'user', 'content': [{'type': 'text', 'text': 'two'}]}],
            'tools': [{'name': 'find', 'input_schema': {}, 'defer_loading': True}]}
    original = copy.deepcopy(body)
    assert cascade.fresh(body)
    out = cascade.transform(target, body)
    assert body == original
    assert not {'metadata', 'thinking', 'context_management'} & out.keys()
    assert out['system'] == [{'type': 'text', 'text': 'start'}, {'type': 'text', 'text': 'folded'}]
    assert out['messages'] == [{'role': 'user', 'content': [{'type': 'text', 'text': 'one'}, {'type': 'text', 'text': 'two'}]}]
    assert out['tools'] == [{'name': 'find', 'input_schema': {}}] and out['output_config'] == {}
    body['messages'].append({'role': 'assistant', 'content': 'answer'})
    assert not cascade.fresh(body)
    body['messages'].pop()
    body['messages'][0]['content'] = [{'type': 'tool_result', 'content': [{'type': 'thinking'}]}]
    assert not cascade.fresh(body)
    body['messages'][1]['content'] = [{'type': 'image'}]
    with pytest.raises(policy.Refused, match='system_messages'):
        cascade.transform(target, body)

from lib.hal import account_switch, gateway_egress, spend_guard, wire_hold, wire_proxy
from lib.hal.account_ledger import LedgerView
from tests import test_wire_proxy as h
from tests.test_wire_proxy import home, pki, make_wire
from tests.test_wire_replay import A, B, LIMIT, Upstream, request, reply
from tests.test_wire_hold import exchange, SUCCESS


@pytest.fixture(params=[False, True], ids=['plain', 'hold'])
def rig(request, double_signer, make_wire, monkeypatch, tmp_path, pki):
    # A move is refused unless its decision row is signed, and GRIP's signer is
    # absent on HAL's CI, so every rig signs with the thread tests' double.
    monkeypatch.setattr(h, 'FakeUpstream', Upstream)
    monkeypatch.setattr(Upstream, 'answers', {A: LIMIT, B: LIMIT})
    monkeypatch.setattr(spend_guard, '_OAUTH_PREFIX', 'fake-account-')
    monkeypatch.setattr(account_switch, '_read_pool', lambda: [('A', A), ('B', B)])
    for name in ('_EXHAUSTED', '_ORGS', '_ORG_RETRY'):
        monkeypatch.setattr(account_switch, name, {})
    monkeypatch.setattr(policy, 'load_shield', lambda: lambda body: json.loads(json.dumps(body).replace('DenylistedExample', '[PERSON]')))
    monkeypatch.setattr(gateway_egress, '_default_key_resolver', lambda _: 'fake-target-token')
    monkeypatch.setattr(wire_hold, 'enabled', lambda: request.param)
    wire = make_wire(pool_size=0)
    target = Upstream(pki.server_context(h.API))
    target.answers = {'': SUCCESS, 'fake-target-token': SUCCESS}
    wire.proxy._resolve = lambda host, port: ('127.0.0.1', target.port if port == target.port else wire.fake.port)
    entry = dict(ENTRY, base_url=f'https://{h.API}:{target.port}/v1')
    config = tmp_path / 'w4.json'
    config.write_text(json.dumps({'anthropic_targets': [entry]}))
    monkeypatch.setenv('HAL_W4_CONFIG', str(config))
    wire.target, wire.config, wire.entry = target, config, entry
    yield wire
    target.close()


def turn(body=None, session='fake-session', path='/v1/messages', extra_headers=b''):
    body = body or {'model': 'original', 'max_tokens': 100, 'stream': True,
                    'messages': [{'role': 'user', 'content': 'hello'}]}
    raw = request(path=path, payload=body)
    marker = extra_headers + f'x-claude-code-session-id: {session}\r\nX-Unchanged:'.encode()
    return raw.replace(b'X-Unchanged:', marker)


def configure(rig, **changes):
    rig.config.write_text(json.dumps({'anthropic_targets': [dict(rig.entry, **changes)]}))


def refusal_row(body, reason):
    target = cascade.Target(**ENTRY, window=100000)
    rows = []
    proxy = SimpleNamespace(_telemetry=SimpleNamespace(record=rows.append))
    ex = SimpleNamespace(headers=(('x-claude-code-session-id', 'fake-session'),))
    if reason == 'fresh_thread':
        cascade._record(proxy, ex, target.label, policy.Refused(reason), 'refused')
    else:
        assert cascade._attempt(proxy, None, ex, target, 'fake-key', body, 'refused') == (None, None)
    assert rows[0]['reason'] == reason
    return rows[0]


def test_record_shape_filters_detail_at_sink():
    detail = {'fields': ['ok_name', 'Bad Name'], 'secret': 'SENTINEL', 'unnamed': 1}
    row = policy.record_shape('session', 'target', 'fields', 'refused', detail=detail)
    # The caller counted one unnamed field; the sink dropped 'Bad Name' and counts it too.
    assert row['detail'] == {'fields': ['ok_name'], 'unnamed': 2}
    assert 'secret' not in json.dumps(row) and 'SENTINEL' not in json.dumps(row)


def test_record_shape_drops_invalid_detail_values():
    row = policy.record_shape('session', 'target', 'fields', 'refused', detail={
        'fields': ['ok_name', 3, 'Bad Name'] + [f'field_{i}' for i in range(15)],
        'types': 'not a list', 'unnamed': '1', 'more': -1,
        'cooldown_until': '123', 'cooldown_source': 'unknown',
    })
    # 16 safe names: 12 kept and 4 counted as more; 2 unsafe ones counted as unnamed.
    assert row['detail'] == {'fields': ['ok_name'] + [f'field_{i}' for i in range(11)],
                             'unnamed': 2, 'more': 4}


def test_record_shape_bounds_every_integer():
    # AGORA's second falsifier on HAL#735: an integer large enough to carry data.
    row = policy.record_shape('session', 'target', 'target_refused', 'refused', detail={
        'cooldown_until': 2 ** 63, 'cooldown_source': 'default',
        'unnamed': 10 ** 9, 'more': 5})
    assert row['detail'] == {'more': 5}


def test_record_shape_keeps_a_valid_cooldown_and_drops_the_rest():
    row = policy.record_shape('session', 'target', 'target_refused', 'refused', detail={
        'cooldown_until': 1793721600, 'cooldown_source': 'stated_reset', 'secret': 'SENTINEL'})
    assert row['detail'] == {'cooldown_until': 1793721600, 'cooldown_source': 'stated_reset'}


def test_record_shape_keeps_no_cooldown_source_without_a_valid_time():
    # AGORA's falsifier on HAL#735: a disallowed key beside a bad cooldown.
    row = policy.record_shape('session', 'target', 'target_refused', 'refused', detail={
        'secret': 'SENTINEL', 'cooldown_until': -5, 'cooldown_source': 'retry_after',
        'fields': ['ok_name']})
    assert row['detail'] == {'fields': ['ok_name']}
    assert 'SENTINEL' not in json.dumps(row)


def test_record_shape_omits_empty_detail():
    base = policy.record_shape('session', 'target', 'fields', 'refused')
    filtered = policy.record_shape('session', 'target', 'fields', 'refused',
                                   detail={'secret': 'SENTINEL', 'cooldown_until': 0})
    assert filtered == base
    assert 'detail' not in filtered


def test_field_refusal_names_safe_key():
    row = refusal_row({'messages': [{'role': 'user', 'content': 'hi'}],
                       'foo_bar': 'SENTINEL-VALUE-83c'}, 'fields')
    assert row['detail'] == {'fields': ['foo_bar'], 'unnamed': 0}
    assert 'SENTINEL-VALUE-83c' not in json.dumps(row)


def test_field_refusal_counts_unsafe_keys_and_caps_names():
    unsafe = ['Uppercase', 'has space', '1leading', 'a' * 41, 'naïve']
    body = {'messages': [{'role': 'user', 'content': 'hi'}],
            **{name: 'SENTINEL-VALUE-83c' for name in unsafe},
            **{f'field_{i:02d}': None for i in range(14)}}
    row = refusal_row(body, 'fields')
    assert row['detail'] == {'fields': [f'field_{i:02d}' for i in range(12)],
                             'unnamed': 5, 'more': 2}
    encoded = json.dumps(row, ensure_ascii=False)
    assert all(name not in encoded for name in unsafe)
    assert 'SENTINEL-VALUE-83c' not in encoded


@pytest.mark.parametrize('tool,expected', [
    ({'type': 'web_search_20250305', 'name': 'secret_tool_name',
      'description': 'SENTINEL-DESCRIPTION-91b', 'input_schema': {}},
     {'types': ['web_search_20250305'], 'fields': [], 'unnamed': 0}),
    ({'name': 'secret_tool_name', 'description': 'SENTINEL-DESCRIPTION-91b',
      'input_schema': {}, 'cache_control': {'secret': 'SENTINEL-DESCRIPTION-91b'}},
     {'types': [], 'fields': ['cache_control'], 'unnamed': 0}),
])
def test_tool_refusal_names_shape_only(tool, expected):
    row = refusal_row({'messages': [{'role': 'user', 'content': 'hi'}], 'tools': [tool]}, 'tools')
    assert row['detail'] == expected
    encoded = json.dumps(row)
    assert 'secret_tool_name' not in encoded
    assert 'SENTINEL-DESCRIPTION-91b' not in encoded


def test_tool_refusal_filters_and_caps_type_and_extra_names():
    unsafe = ['BadType', 'has space', '2leading', 'x' * 41, 'caché']
    tool = {'name': 'secret_tool_name', 'description': 'SENTINEL-DESCRIPTION-91b',
            'input_schema': {}, 'type': 'BadType',
            **{name: 'SENTINEL-DESCRIPTION-91b' for name in unsafe[1:]},
            **{f'extra_{i:02d}': None for i in range(14)}}
    row = refusal_row({'messages': [{'role': 'user', 'content': 'hi'}], 'tools': [tool]}, 'tools')
    assert row['detail'] == {'fields': [f'extra_{i:02d}' for i in range(12)],
                             'types': [], 'unnamed': 5, 'more': 2}
    encoded = json.dumps(row, ensure_ascii=False)
    assert all(name not in encoded for name in unsafe)
    assert 'secret_tool_name' not in encoded
    assert 'SENTINEL-DESCRIPTION-91b' not in encoded


def test_refusal_without_detail_preserves_row_shape():
    row = refusal_row({'messages': [{'role': 'assistant', 'content': 'answer'}]}, 'fresh_thread')
    expected = {'event': 'w4.cascade', 'session_hash': hashlib.sha256(b'fake-session').hexdigest(),
                'target': 'moonshot/kimi-k3', 'reason': 'fresh_thread', 'outcome': 'refused'}
    assert json.dumps(row) == json.dumps(expected)


def test_refused_detail_does_not_change_reason_string():
    exc = policy.Refused('fields', detail={'fields': ['foo_bar'], 'unnamed': 0})
    assert str(exc) == 'fields'


def test_move_scrubs_final_target_request(rig):
    configure(rig, drop=['context_management', 'thinking'], drop_tool_fields=['defer_loading'], system_messages='fold')
    body = {'model': 'original', 'max_tokens': 100, 'stream': True, 'metadata': {'user_id': 'private'},
            'context_management': {}, 'thinking': {}, 'system': 'DenylistedExample system',
            'messages': [{'role': 'user', 'content': 'DenylistedExample user'},
                         {'role': 'system', 'content': 'folded instructions'},
                         {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'fake-call', 'content': 'DenylistedExample result'}]}],
            'tools': [{'name': 'lookup', 'input_schema': {}, 'defer_loading': True}]}
    extra = b'Cookie: session=secret-cookie-value\r\nanthropic-beta: fake-beta-flag\r\n'
    raw = exchange(rig, turn(body, extra_headers=extra))
    assert raw == SUCCESS
    assert len(rig.target.requests) == 1
    received = rig.target.requests[0][1]
    assert b'DenylistedExample' not in received and received.count(b'[PERSON]') == 3
    head, data = received.split(b'\r\n\r\n', 1)
    assert A.encode() not in head and B.encode() not in head and b'x-claude-code' not in head
    # No inbound header is ever copied: the outgoing head is built from scratch.
    assert b'ookie' not in head.lower() and b'anthropic-beta' not in head.lower()
    assert b'secret-cookie-value' not in head and b'fake-beta-flag' not in head
    sent = json.loads(data)
    assert 'metadata' not in sent and 'defer_loading' not in sent['tools'][0]
    assert all(m['role'] != 'system' for m in sent['messages'])
    assert sent['system'][-1]['text'] == 'folded instructions'
    assert sent['model'] == 'kimi-k3' and len(sent['messages']) == 1


@pytest.mark.parametrize('case', ['assistant', 'thinking', 'field', 'server_tool', 'size', 'shield', 'missing_shield', 'billing', 'session', 'empty', 'malformed', 'credential'])
def test_move_refusals_preserve_original(rig, monkeypatch, tmp_path, case):
    body = {'model': 'original', 'max_tokens': 100, 'stream': True, 'messages': [{'role': 'user', 'content': 'hi'}]}
    session = 'fake-session'
    if case == 'assistant': body['messages'].append({'role': 'assistant', 'content': 'answer'})
    if case == 'thinking': body['messages'][0]['content'] = [{'type': 'tool_result', 'content': [{'type': 'redacted_thinking'}]}]
    if case == 'field': body['context_management'] = {}
    if case == 'server_tool': body['tools'] = [{'type': 'web_search_20250305', 'name': 'web_search'}]
    if case == 'size': body['max_tokens'] = 2000000
    if case == 'shield': monkeypatch.setattr(policy, 'load_shield', lambda: lambda _: 1/0)
    if case == 'missing_shield':
        monkeypatch.setattr(policy, 'load_shield', REAL_LOADER)
        monkeypatch.setattr(policy, '_shield', None)
        monkeypatch.setattr(policy, '_shield_retry_at', 0.0)
        monkeypatch.setenv('CLAUDE_DIR', str(tmp_path / 'empty'))
    if case == 'billing':
        configure(rig, billing='unknown')
        monkeypatch.setattr(policy, 'billing_allowed', lambda *a: False)
    if case == 'credential':
        monkeypatch.setattr(gateway_egress, '_default_key_resolver', lambda _: 'fake-target-token\r\nX-Injected: evil')
    if case == 'session': session = ''
    if case == 'empty': rig.config.write_text('{}')
    if case == 'malformed': rig.config.write_text('{')
    assert exchange(rig, turn(body, session)) == LIMIT
    assert not rig.target.requests


SETTINGS_DROP = ['context_management', 'output_config', 'safeguards', 'thinking']


@pytest.mark.parametrize('thinking_on', [True, False], ids=['thinking-on', 'thinking-off'])
def test_a_constraint_in_a_dropped_field_never_reaches_the_target(rig, thinking_on):
    # AGORA's falsifier on HAL#738, end to end over the loopback target. The body
    # has the shape headless Claude Code 2.1.296 was probed to send: one
    # safeguards entry, and the keep-all clear_thinking edit when thinking is on
    # (values synthetic). The edit clears nothing and is dropped; the entry
    # refuses the move either way.
    configure(rig, drop=SETTINGS_DROP)
    body = {'model': 'original', 'max_tokens': 100, 'stream': True,
            'messages': [{'role': 'user', 'content': 'hi'}],
            'output_config': {'effort': 'max'},
            'safeguards': [{'type': 'dangerous_tool_use',
                            'classifier_context': {'nested_key_91c': 'SENTINEL-VALUE-91c'}}]}
    if thinking_on:
        body['thinking'] = {'type': 'adaptive', 'display': 'omitted'}
        body['context_management'] = {'edits': [{'type': 'clear_thinking_20251015', 'keep': 'all'}]}
    assert exchange(rig, turn(body)) == LIMIT
    assert not rig.target.requests
    rows = [r for r in h._rows(rig.home) if r.get('event') == 'w4.cascade']
    assert [(r['reason'], r['outcome'], r.get('detail')) for r in rows] == [
        ('fields', 'refused', {'fields': ['safeguards'], 'unnamed': 0})]
    encoded = json.dumps(rows)
    assert 'SENTINEL-VALUE-91c' not in encoded and 'nested_key_91c' not in encoded
    assert 'dangerous_tool_use' not in encoded and 'clear_thinking' not in encoded


def test_the_same_settings_in_their_empty_forms_move(rig):
    # The control for the test above: with nothing but empty forms, effort and
    # thinking in those fields, the same rig does move, and none of them is sent.
    configure(rig, drop=SETTINGS_DROP)
    body = {'model': 'original', 'max_tokens': 100, 'stream': True,
            'messages': [{'role': 'user', 'content': 'hi'}],
            'thinking': {'type': 'adaptive', 'display': 'omitted'},
            'output_config': {'effort': 'max'}, 'context_management': {'edits': []},
            'safeguards': []}
    assert exchange(rig, turn(body)) == SUCCESS
    assert len(rig.target.requests) == 1
    sent = json.loads(rig.target.requests[0][1].split(b'\r\n\r\n', 1)[1])
    assert set(sent) == {'model', 'max_tokens', 'stream', 'messages'}


@pytest.mark.parametrize('edit,moves', [
    ({'type': 'clear_thinking_20251015', 'keep': 'all'}, True),
    ({'type': 'clear_thinking_20251015', 'keep': {'type': 'thinking_turns', 'value': 1}}, False),
], ids=['keep-all', 'keep-one'])
def test_claude_codes_interactive_request_moves_without_its_keep_all_edit(rig, edit, moves):
    # HAL#676: in a real terminal, Claude Code 2.1.296 sends the keep-all
    # clear_thinking edit on every turn and no safeguards entry. That edit keeps
    # every thinking block, so it is dropped and the turn moves, and the target
    # never sees it. An edit that does clear blocks still refuses the move.
    configure(rig, drop=SETTINGS_DROP)
    body = {'model': 'original', 'max_tokens': 100, 'stream': True,
            'messages': [{'role': 'user', 'content': 'hi'}],
            'thinking': {'type': 'adaptive', 'display': 'omitted'},
            'output_config': {'effort': 'max'}, 'context_management': {'edits': [edit]}}
    assert exchange(rig, turn(body)) == (SUCCESS if moves else LIMIT)
    assert len(rig.target.requests) == (1 if moves else 0)
    if moves:
        sent = json.loads(rig.target.requests[0][1].split(b'\r\n\r\n', 1)[1])
        assert set(sent) == {'model', 'max_tokens', 'stream', 'messages'}


@pytest.mark.parametrize('status', [400, 500])
def test_target_failure_restores_original(rig, status):
    rig.target.answers = {'': reply(status, b'target-private-error')}
    assert exchange(rig, turn()) == LIMIT
    assert len(rig.target.requests) == 1


@pytest.mark.parametrize('rig', [True], indirect=True, ids=['hold'])
def test_a_real_silent_target_is_cut_off_when_the_hold_budget_ends(rig, monkeypatch):
    """HAL#748 item 10: the bounded wait, end to end over real loopback TLS. A target
    that takes the request and then says nothing is waited on only while the hold's
    budget lasts. The client then gets the refusal it would have had with no move,
    and the turn stays reserved as uncertain, so it is never offered again.
    ``exchange`` reads with a 3-second timeout, so a wait the budget did not bound
    (the wire's 900-second idle timeout) fails this test rather than hanging it."""
    monkeypatch.setattr(wire_hold, 'BUDGET_S', 1.5)
    rig.target.answers = {'': b'', 'fake-target-token': b''}  # reads the request, answers nothing
    started = time.monotonic()
    assert exchange(rig, turn()) == LIMIT
    waited = time.monotonic() - started
    assert len(rig.target.requests) == 1 and 1.0 <= waited < 3.0
    body = {'model': 'original', 'max_tokens': 100, 'stream': True,
            'messages': [{'role': 'user', 'content': 'hello'}]}
    key = wire_hold.turn_key((('x-claude-code-session-id', 'fake-session'),), body)
    assert rig.proxy._w4_attempts._path(key).is_file()
    # Sent again, the turn is refused at once (the wire now knows the pool is spent)
    # and never reaches the target a second time.
    assert exchange(rig, turn()).startswith(b'HTTP/1.1 429 ')
    assert len(rig.target.requests) == 1


def test_only_before_send_advances(rig, monkeypatch):
    first = dict(rig.entry, base_url='https://unreachable.invalid/v1')
    rig.config.write_text(json.dumps({'anthropic_targets': [first, rig.entry]}))
    connect = cascade._connect
    def refused(proxy, target, ex):
        if 'unreachable' in target.base_url: raise ConnectionRefusedError()
        return connect(proxy, target, ex)
    monkeypatch.setattr(cascade, '_connect', refused)
    assert exchange(rig, turn()) == SUCCESS
    assert len(rig.target.requests) == 1


def _thread_rows(monkeypatch, rig, written=True):
    """Capture every thread row; each records how many requests the target had seen."""
    from lib.hal import gateway_witness
    from lib.hal.infinite_thread import Appended
    rows = []

    def append(fields, **kw):
        rows.append({'fields': dict(fields), **kw, 'target_seen': len(rig.target.requests)})
        if written is None:
            return None
        return Appended(written=written, thread_id='0' * 16 if written else None,
                        seq=len(rows) if written else None, reason='' if written else 'no signer here')
    monkeypatch.setattr(gateway_witness, 'append_thread_row', append)
    return rows


@pytest.mark.parametrize('written', [None, False], ids=['store-unreachable', 'store-refused'])
def test_no_move_without_a_signed_decision_row(rig, monkeypatch, written):
    """A move sends the conversation to another vendor, so the signed record of
    that decision comes first (ADR-0016 decision 5). When it cannot be written,
    the honest exhaustion answer stands and the target receives nothing."""
    _thread_rows(monkeypatch, rig, written=written)
    assert exchange(rig, turn()) == LIMIT
    assert not rig.target.requests
    # The refusal names its reason in the audit, so the gap is visible.
    refusals = [r for r in h._rows(rig.home) if r.get('event') == 'w4.cascade']
    assert [(r['reason'], r['outcome']) for r in refusals] == [('unsigned', 'refused')]


def test_signed_decision_precedes_the_target_request(rig, monkeypatch):
    rows = _thread_rows(monkeypatch, rig)
    assert exchange(rig, turn()) == SUCCESS
    failover = [r for r in rows if r['row_class'] == 'failover']
    assert [r['target_seen'] for r in failover] == [0, 1]
    decision, outcome = failover
    assert decision['fields'].get('status') is None and outcome['fields']['status'] == 200
    assert decision['fields']['model'] == outcome['fields']['model'] == 'kimi-k3'


def test_both_failover_rows_carry_the_target_billing(rig, monkeypatch):
    """The billing a row signs is its own field, never overwritten by the cost
    helper's default of 'unknown'."""
    rows = _thread_rows(monkeypatch, rig)
    assert exchange(rig, turn()) == SUCCESS
    assert [r['billing'] for r in rows if r['row_class'] == 'failover'] == ['subscription'] * 2


def test_a_moved_exchange_is_witnessed_with_its_target_billing(rig, monkeypatch):
    """The witness row of a moved exchange is billed as its target declares. The
    inbound credential here says nothing about billing ('unknown'), and for a
    real subscription token it would say 'subscription' even when the target
    is metered -- a paid call signed as free."""
    import time
    from lib.hal.wire_classify import RouteClass
    seen = {}
    monkeypatch.setattr(wire_proxy, 'witness_chat', lambda **kw: seen.update(kw))
    ex = wire_proxy._Exchange('POST', '/v1/messages', [('x-claude-code-session-id', 'fake-session')],
                              time.monotonic(), route=RouteClass.INFERENCE)
    ex.w4_target = cascade.Target(**dict(ENTRY, billing='metered'), window=100000)
    rig.proxy._witness(ex)
    assert seen['provider'] == 'moonshot' and seen['billing'] == 'metered'


@pytest.mark.parametrize('path', ['/v1/messages/count_tokens', '/v1/code/sessions'])
def test_other_routes_never_move(rig, path):
    assert exchange(rig, turn(path=path)) == LIMIT
    assert not rig.target.requests


@pytest.mark.parametrize('gate', ['client_started', 'overflow'])
def test_response_boundary(rig, gate):
    from lib.hal.wire_classify import RouteClass
    from types import SimpleNamespace
    account_switch.observe(429, (('anthropic-ratelimit-unified-status', 'rejected'),), (('authorization', 'Bearer '+B),))
    headers = (('authorization', 'Bearer '+A), ('x-claude-code-session-id', 'fake-session'))
    account_switch.observe(429, (('anthropic-ratelimit-unified-status', 'rejected'),), headers)
    body = wire_proxy._Copy()
    body(json.dumps({'messages': [{'role': 'user', 'content': 'hi'}]}).encode())
    ex = wire_proxy._Exchange('POST', '/v1/messages', headers, 0, route=RouteClass.INFERENCE, body=body, wire_body=body)
    if gate == 'client_started': ex.client_started = True
    else:
        ex.wire_body = wire_proxy._Copy()
        ex.wire_body.overflow = True
    class Sink:
        def sendall(self, data): pass
    assert not cascade.move(rig.proxy, SimpleNamespace(sock=Sink()), ex)
    assert not rig.target.requests


def test_sticky_history_restart_and_new_session(rig):
    assert exchange(rig, turn()) == SUCCESS
    count = len(rig.fake.requests)
    account_switch.forget_exhausted()
    rig.fake.answers = {A: SUCCESS, B: SUCCESS}
    body = {'model': 'original', 'stream': True, 'max_tokens': 100,
            'messages': [{'role': 'user', 'content': 'DenylistedExample'},
                         {'role': 'assistant', 'content': [{'type': 'thinking', 'thinking': 'target thought', 'signature': 'target-signature'}]},
                         {'role': 'user', 'content': 'next'}]}
    # Discard all in-memory bindings as a process restart would.
    rig.proxy._w4_sessions = cascade.Sessions()
    assert exchange(rig, turn(body)) == SUCCESS
    assert len(rig.fake.requests) == count and len(rig.target.requests) == 2
    sent = rig.target.requests[-1][1]
    assert b'DenylistedExample' not in sent and b'target-signature' in sent
    assert exchange(rig, turn(session='new-session')) == SUCCESS
    assert len(rig.fake.requests) > count and len(rig.target.requests) == 2


@pytest.mark.parametrize('failure', ['target', 'shield', 'field', 'size', 'config'])
def test_sticky_failure_never_returns_to_anthropic(rig, monkeypatch, failure):
    assert exchange(rig, turn()) == SUCCESS
    count = len(rig.fake.requests)
    body = {'model': 'original', 'stream': True, 'max_tokens': 100, 'messages': [{'role': 'user', 'content': 'next'}]}
    if failure == 'target': rig.target.answers = {'': reply(500, b'target failure')}
    if failure == 'shield': monkeypatch.setattr(policy, 'load_shield', lambda: lambda _: 1/0)
    if failure == 'field': body['undeclared'] = True
    if failure == 'size': body['max_tokens'] = 2000000
    if failure == 'config': rig.config.write_text('{}')
    answer = exchange(rig, turn(body))
    assert answer.startswith(b'HTTP/1.1 429') and b'earliest known reset' in answer
    assert b'target failure' not in answer and len(rig.fake.requests) == count
    assert len(rig.target.requests) == (2 if failure == 'target' else 1)


def status(rig, session='fake-session'):
    import http.client
    conn = http.client.HTTPConnection('127.0.0.1', rig.port, timeout=2)
    try:
        conn.request('GET', '/w4/status?session=' + session)
        answer = conn.getresponse()
        assert answer.status == 200
        return json.loads(answer.read())
    finally:
        conn.close()


def _target_account():
    return 'w4:moonshot:' + spend_guard.account_key('fake-target-token')


def test_alibaba_quota_refusal_cools_next_turn_and_status(rig):
    message = ('Your token-plan 1-month quota has been exhausted. '
               'The quota will reset at 11-03 16:00:00 UTC.')
    request_id = 'SENTINEL-REQUEST-ID-7f3c9a'
    rig.target.answers = {'': reply(429, json.dumps({
        'code': 'Throttling.AllocationQuota', 'message': message,
        'request_id': request_id}).encode())}
    assert exchange(rig, turn(session='quota-first')) == LIMIT
    assert len(rig.target.requests) == 1
    assert LedgerView().cooling(_target_account()) is not None
    assert status(rig, 'quota-second')['fresh_thread_target'] is None
    # The second exhausted turn is answered with an exhaustion 429: the
    # upstream's own (plain rig) or the wire's notice naming the reset (hold rig).
    assert exchange(rig, turn(session='quota-second')).startswith(b'HTTP/1.1 429')
    assert len(rig.target.requests) == 1
    rows = h._rows(rig.home)
    assert any(row.get('reason') == 'target_refused' for row in rows)
    encoded = json.dumps(rows)
    assert message not in encoded and request_id not in encoded


def test_retry_after_cools_w4_account_for_120_seconds(rig):
    before = time.time()
    rig.target.answers = {'': reply(429, b'ordinary throttling', b'Retry-After: 120\r\n')}
    assert exchange(rig, turn()) == LIMIT
    cooling = LedgerView().cooling(_target_account())
    assert cooling is not None
    assert before + 119 <= cooling.until <= time.time() + 121
    assert status(rig)['fresh_thread_target'] is None


def test_ordinary_429_does_not_cool_w4_target(rig):
    rig.target.answers = {'': reply(429, b'{"code":"TooManyRequests"}')}
    assert exchange(rig, turn(session='ordinary-first')) == LIMIT
    assert LedgerView().cooling(_target_account()) is None
    assert status(rig)['fresh_thread_target'] == 'moonshot/kimi-k3'
    assert exchange(rig, turn(session='ordinary-second')).startswith(b'HTTP/1.1 429')
    assert len(rig.target.requests) == 2


@pytest.mark.parametrize('message,now,expected', [
    ('The quota will reset at 11-03 16:00:00 UTC.',
     datetime(2026, 10, 12, tzinfo=timezone.utc), datetime(2026, 11, 3, 16, tzinfo=timezone.utc)),
    ('The quota will reset at 02-01 08:30:00 UTC.',
     datetime(2026, 10, 12, tzinfo=timezone.utc), datetime(2027, 2, 1, 8, 30, tzinfo=timezone.utc)),
    ('no usable reset date', datetime(2026, 10, 12, tzinfo=timezone.utc), None),
])
def test_alibaba_reset_parser(message, now, expected):
    assert cascade._alibaba_reset(message, now.timestamp()) == (
        int(expected.timestamp()) if expected else None)


@pytest.mark.parametrize('value,seconds', [
    ('120', 120),
    (format_datetime(datetime(2026, 10, 12, 0, 2, tzinfo=timezone.utc), usegmt=True), 120),
    ('garbage', None),
])
def test_retry_after_parser(value, seconds):
    now = datetime(2026, 10, 12, tzinfo=timezone.utc).timestamp()
    assert cascade._retry_after_until((('Retry-After', value),), now) == (
        int(now) + seconds if seconds is not None else None)


def test_quota_cooldown_uses_header_before_message_and_finite_default():
    now = datetime(2026, 10, 12, tzinfo=timezone.utc).timestamp()
    body = json.dumps({'code': 'Throttling.AllocationQuota',
                       'message': 'The quota will reset at 11-03 16:00:00 UTC.'}).encode()
    assert cascade._quota_until((('Retry-After', '120'),), body, now) == (now + 120, 'retry_after')
    assert cascade._quota_until((), body, now) == (
        datetime(2026, 11, 3, 16, tzinfo=timezone.utc).timestamp(), 'stated_reset')
    no_date = b'{"code":"Throttling.AllocationQuota","message":"unknown reset"}'
    assert cascade._quota_until((), no_date, now) == (now + cascade.QUOTA_COOLDOWN_DEFAULT_S, 'default')
    assert cascade._quota_until((('Retry-After', 'garbage'),), b'{}', now) == (
        now + cascade.QUOTA_COOLDOWN_DEFAULT_S, 'default')
    assert cascade._quota_until((), b'{"code":"TooManyRequests"}', now) is None


def _attempt_refusal_without_listener(monkeypatch, body, headers=b''):
    head = wire_proxy._parse_head(reply(429, body, headers).split(b'\r\n\r\n', 1)[0] + b'\r\n\r\n')
    class Socket:
        def sendall(self, data):
            pass
        def shutdown(self, how):
            pass
        def close(self):
            pass
    monkeypatch.setattr(policy, 'prepare', lambda body, window: body)
    monkeypatch.setattr(cascade, '_connect', lambda *args: SimpleNamespace(sock=Socket(), stream=None))
    monkeypatch.setattr(wire_proxy, '_peek_body', lambda *args: body)
    rows = []
    proxy = SimpleNamespace(_response_head=lambda *a, **kw: head,
                            _telemetry=SimpleNamespace(record=rows.append))
    ex = SimpleNamespace(method='POST', headers=(('x-claude-code-session-id', 'fake-session'),))
    target = cascade.Target(**ENTRY, window=100000)
    assert cascade._attempt(proxy, None, ex, target, 'fake-target-token',
                            {'messages': [{'role': 'user', 'content': 'hi'}]}, 'refused') == (None, None)
    return rows


def test_attempt_records_quota_cooldown_without_a_listener(monkeypatch):
    message = 'The quota will reset at 11-03 16:00:00 UTC.'
    request_id = 'SENTINEL-REQUEST-ID-7f3c9a'
    body = json.dumps({'code': 'Throttling.AllocationQuota', 'message': message,
                       'request_id': request_id}).encode()
    rows = _attempt_refusal_without_listener(monkeypatch, body)
    cooling = LedgerView().cooling(_target_account())
    assert cooling is not None and cooling.until == cascade._alibaba_reset(message, time.time())
    assert rows[-1]['reason'] == 'target_refused'
    assert message not in json.dumps(rows) and request_id not in json.dumps(rows)


@pytest.mark.parametrize('headers,body,source', [
    (b'Retry-After: 120\r\n', b'ordinary throttling', 'retry_after'),
    (b'', b'{"code":"Throttling.AllocationQuota","message":"The quota will reset at 11-03 16:00:00 UTC."}',
     'stated_reset'),
    (b'', b'{"code":"Throttling.AllocationQuota","message":"The quota will reset at 13-99 00:00:00 UTC."}',
     'default'),
])
def test_attempt_records_quota_source_without_a_listener(monkeypatch, headers, body, source):
    rows = _attempt_refusal_without_listener(monkeypatch, body, headers)
    detail = rows[-1]['detail']
    assert type(detail['cooldown_until']) is int and detail['cooldown_until'] > 0
    assert detail['cooldown_source'] == source


def test_recorded_quota_cooldown_blocks_selection_without_a_listener(monkeypatch):
    body = b'{"code":"Throttling.AllocationQuota","message":"The quota will reset at 11-03 16:00:00 UTC."}'
    _attempt_refusal_without_listener(monkeypatch, body)
    target = cascade.Target(**ENTRY, window=100000)
    monkeypatch.setattr(policy, 'billing_allowed', lambda *a: True)
    monkeypatch.setattr(policy, 'load_shield', lambda: lambda body: body)
    monkeypatch.setattr(gateway_egress, '_default_key_resolver', lambda _: 'fake-target-token')
    monkeypatch.setattr(spend_guard, 'spent_until', lambda _: None)
    monkeypatch.setattr(cascade, 'configured_targets', lambda: [target])
    monkeypatch.setattr(cascade, 'config_errors', lambda: [])
    monkeypatch.setattr(cascade, '_attempt', lambda *a: pytest.fail('cooling target was sent a request'))
    proxy = SimpleNamespace(_ledger=SimpleNamespace(view=lambda: LedgerView()),
                            _w4_sessions=SimpleNamespace(bound=lambda _: None, paused=lambda _: False),
                            _telemetry=SimpleNamespace(record=lambda row: None))
    ex = SimpleNamespace(headers=(('x-claude-code-session-id', 'new-session'),))
    with pytest.raises(policy.Refused, match='cooldown'):
        cascade.independent(target)
    assert cascade.status(proxy, 'new-session')['fresh_thread_target'] is None
    assert not cascade._serve(proxy, None, ex, {}, [target])


@pytest.mark.parametrize('headers,body,expected', [
    (b'Retry-After: 120\r\n', b'ordinary throttling', 'timed'),
    (b'', b'{"code":"TooManyRequests"}', 'none'),
])
def test_attempt_only_cools_signalled_429_without_a_listener(monkeypatch, headers, body, expected):
    before = time.time()
    _attempt_refusal_without_listener(monkeypatch, body, headers)
    cooling = LedgerView().cooling(_target_account())
    if expected == 'timed':
        assert cooling is not None and before + 119 <= cooling.until <= time.time() + 121
    else:
        assert cooling is None


def test_status_contract(rig, monkeypatch):
    label = 'moonshot/kimi-k3'
    assert status(rig) == {'session': 'fake-session', 'paused': False, 'on_target': None,
        'fresh_thread_target': label, 'anthropic_available': True, 'earliest_reset': None,
        'config_errors': []}
    configure(rig, billing='subscription')
    body = {'model': 'original', 'max_tokens': 100, 'stream': True,
            'messages': [{'role': 'assistant', 'content': 'old'}]}
    assert exchange(rig, turn(body)) == LIMIT
    h._wait_for(lambda: status(rig)['paused'])
    state = status(rig)
    assert state['paused'] and not state['anthropic_available'] and state['earliest_reset'] > 0
    assert not status(rig, 'unknown')['paused']
    assert exchange(rig, turn(session='move-session')) == SUCCESS
    h._wait_for(lambda: status(rig, 'move-session')['on_target'])
    state = status(rig, 'move-session')
    assert state['on_target'] == label and not state['paused']
    monkeypatch.setattr(policy, 'load_shield', lambda: None)
    assert status(rig)['fresh_thread_target'] is None
    answer = exchange(rig, turn(session='move-session'))
    assert answer.startswith(b'HTTP/1.1 429')
    h._wait_for(lambda: status(rig, 'move-session')['paused'])
    assert status(rig, 'move-session')['paused']
    account_switch.forget_exhausted()
    # The ledger still knows the stub's spent accounts; remove only test state.
    monkeypatch.setattr(rig.proxy._ledger, 'view', lambda: __import__('types').SimpleNamespace(
        accounts=lambda: {}, token_health=lambda *a: 'unknown'))
    assert status(rig)['anthropic_available']
    monkeypatch.setattr(rig.proxy._ledger, 'view', lambda: __import__('types').SimpleNamespace(
        accounts=lambda: {}, token_health=lambda *a: 'dead'))
    assert not status(rig)['anthropic_available']
    assert A not in json.dumps(state) and B not in json.dumps(state)


def test_status_reports_a_bad_target_configuration(rig):
    """Wiring, not just the pure function: a typo'd provider in the real
    w4_cascade.json surfaces through the live GET /w4/status endpoint."""
    configure(rig, provider='kimi')
    state = status(rig)
    assert state['config_errors'] == ['anthropic_targets[0]: provider "kimi" is not in config/providers.yaml']
    assert state['fresh_thread_target'] is None


def test_before_send_is_only_definite_connect_failures():
    """A timeout, reset or opaque error is ambiguous and must never advance to
    the next target -- only a definite pre-send connect failure may."""
    import socket
    assert not policy.before_send(TimeoutError())
    assert not policy.before_send(ConnectionResetError())
    assert policy.before_send(ConnectionRefusedError())
    assert policy.before_send(socket.gaierror())


def test_credential_must_be_printable_ascii(monkeypatch):
    """A stray CR/LF (or any non-printable byte) in the key refuses -- it can
    never be smuggled into the raw header line ``_request`` builds."""
    monkeypatch.setattr(policy, 'load_shield', lambda: lambda body: body)
    target = cascade.Target(**ENTRY, window=100000)
    monkeypatch.setattr(gateway_egress, '_default_key_resolver', lambda _: 'fake-target-token')
    assert cascade.independent(target) == 'fake-target-token'
    for bad in ('fake\r\nX-Injected: evil', 'fake\nLF', 'fake\x00null', 'fake\x7fDEL', ''):
        monkeypatch.setattr(gateway_egress, '_default_key_resolver', lambda _, b=bad: b)
        with pytest.raises(policy.Refused, match='no_credential'):
            cascade.independent(target)


def test_auth_header_choice_x_api_key_or_bearer(rig):
    """The amendment's `auth` declaration: x-api-key by default, bearer if named."""
    assert exchange(rig, turn()) == SUCCESS
    head = rig.target.requests[0][1].split(b'\r\n\r\n', 1)[0]
    assert b'x-api-key: fake-target-token\r\n' in head and b'Authorization:' not in head
    configure(rig, auth='bearer')
    assert exchange(rig, turn(session='bearer-session')) == SUCCESS
    head2 = rig.target.requests[-1][1].split(b'\r\n\r\n', 1)[0]
    assert b'Authorization: Bearer fake-target-token\r\n' in head2 and b'x-api-key:' not in head2


def test_shared_paused_and_binding_state_across_wire_processes(tmp_path, monkeypatch):
    """Two Sessions objects -- standing in for two wire processes mid provider
    migration, pointed at the same HAL_HOME -- must agree on both fields."""
    monkeypatch.setenv('HAL_HOME', str(tmp_path))
    target = cascade.Target(**ENTRY, window=100000)
    a, b = cascade.Sessions(), cascade.Sessions()
    assert b.bound('fake-session') is None and not b.paused('fake-session')
    a.bind('fake-session', target)
    assert b.bound('fake-session')['target']['provider'] == 'moonshot'
    a.set_paused('fake-session', True)
    assert b.paused('fake-session')
    path = a._paused_path('fake-session')
    before = path.stat().st_mtime_ns
    a.set_paused('fake-session', True)  # unchanged value: no write
    assert path.stat().st_mtime_ns == before
    a.set_paused('fake-session', False)
    assert not b.paused('fake-session')


def test_shield_load_retries_a_failed_load_after_60_seconds(tmp_path, monkeypatch):
    monkeypatch.setattr(policy, '_shield', None)
    monkeypatch.setattr(policy, '_shield_retry_at', 0.0)
    monkeypatch.setenv('CLAUDE_DIR', str(tmp_path / 'missing'))
    assert policy.load_shield(now=1000.0) is None
    assert policy._shield_retry_at == 1000.0 + policy.SHIELD_RETRY_SECONDS
    lib = tmp_path / 'missing' / 'lib'
    lib.mkdir(parents=True)
    (lib / 'review_redaction.py').write_text('def shield_payload(body): return body\n')
    monkeypatch.delitem(sys.modules, 'review_redaction', raising=False)
    # Still inside the retry window: the cached failure stands even though the
    # file now exists.
    assert policy.load_shield(now=1000.0 + policy.SHIELD_RETRY_SECONDS - 1) is None
    # Past the window: retried, and now it succeeds.
    assert policy.load_shield(now=1000.0 + policy.SHIELD_RETRY_SECONDS + 1)({'a': 1}) == {'a': 1}
    # A success is cached forever, even once the file is gone again.
    (lib / 'review_redaction.py').unlink()
    assert policy.load_shield(now=1000.0 + 10 ** 6) is not None
    monkeypatch.delitem(sys.modules, 'review_redaction', raising=False)


from lib.hal import infinite_thread  # noqa: E402
from tests import test_infinite_thread as _thread_tests  # noqa: E402


@pytest.fixture
def double_signer(monkeypatch):
    """GRIP's signer is absent on HAL's CI, where a skip fails the wire job, so the
    row is signed by the thread tests' Ed25519 double. What this test checks is
    what the wire puts in the row and that it signs it, not the signature scheme;
    tests/test_infinite_thread.py covers GRIP's own signer."""
    double = _thread_tests._Double(infinite_thread.state_root())
    monkeypatch.setattr(infinite_thread, "default_signer", lambda _root, _host: double.signer())
    return double


def test_first_move_signs_a_failover_thread_row(double_signer, rig):
    """Part 6: two signed 'failover' rows per first move -- the decision, written
    before anything reaches the target (status null), then the outcome -- naming
    the target and its declared billing, never the prompt or the credential.
    Read back from the real file."""
    marker = 'zzz-prompt-marker-must-never-be-signed'
    body = {'model': 'original', 'max_tokens': 100, 'stream': True,
            'messages': [{'role': 'user', 'content': marker}]}
    assert exchange(rig, turn(body)) == SUCCESS
    paths = list((rig.home / 'state/infinite/threads').glob('*.jsonl'))
    assert len(paths) == 1
    raw = paths[0].read_bytes()
    assert marker.encode() not in raw and b'fake-target-token' not in raw
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    failovers = [r for r in rows if r['class'] == 'failover']
    assert [r.get('status') for r in failovers] == [None, 200]
    pin = double_signer.pins()[_thread_tests.HOST]
    for row in failovers:
        assert row['provider'] == 'moonshot' and row['model'] == 'kimi-k3'
        assert row['session_id'] == 'fake-session' and row['billing'] == 'subscription'
        # The wire signed this row's own content: the double's key verifies it.
        verdict, _ = _thread_tests._double_verify(infinite_thread.signed_bytes(row), row['sig'], pin)
        assert verdict == 'verified'
