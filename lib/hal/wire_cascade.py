"""Claude Code W4: declared Anthropic-format targets.

With no target declared, W4 stays dormant. ``config/w4_cascade.json`` on main
declares two, so the shipped configuration is armed (``docs/w4-provider-cascade.md``).
"""
from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil
from urllib.parse import urlsplit

from lib.hal import cascade_policy as policy


@dataclass(frozen=True)
class Target:
    provider: str
    model: str
    base_url: str
    credential: str
    billing: str
    window: int
    accept: tuple = ()
    drop: tuple = ()
    drop_tool_fields: tuple = ()
    system_messages: str | None = None
    auth: str = 'x-api-key'

    @property
    def label(self):
        return self.provider + '/' + self.model


def configured_targets():
    from lib.hal.registry import _parse_simple_yaml
    from lib.hal.models_command import load_catalogue
    try:
        entries = policy.config().get('anthropic_targets', [])
        if not isinstance(entries, list):
            return []
        root = policy.CONFIG.parent
        providers = {p['name'] for p in _parse_simple_yaml(str(root / 'providers.yaml'))['providers']}
        windows = {m.address: m.context_window for m in load_catalogue(str(root / 'model_catalog.yaml'))}
        targets = []
        for entry in entries:
            target, _reason = _target(entry, providers, windows)
            if target is not None:
                targets.append(target)
        return targets
    except (OSError, ValueError, KeyError, TypeError):
        return []


def config_errors():
    """Plain-language reasons configured_targets() dropped an entry or found
    nothing to read, for GET /w4/status -- never for a decision.

    Reads the same checks through the same code path as configured_targets()
    (policy._config_or_reason() and _target()), never a second, driftable
    copy of them. A missing file, an absent anthropic_targets, or an empty
    one is W4 staying dormant by design -- never a reason.
    """
    from lib.hal.registry import _parse_simple_yaml
    from lib.hal.models_command import load_catalogue
    value, reason = policy._config_or_reason()
    if reason:
        return [reason]
    entries = value.get('anthropic_targets', [])
    if not isinstance(entries, list):
        return ['anthropic_targets: must be a list']
    try:
        root = policy.CONFIG.parent
        providers = {p['name'] for p in _parse_simple_yaml(str(root / 'providers.yaml'))['providers']}
        windows = {m.address: m.context_window for m in load_catalogue(str(root / 'model_catalog.yaml'))}
    except (OSError, ValueError, KeyError, TypeError):
        return []
    reasons = []
    for index, entry in enumerate(entries):
        _entry_target, why = _target(entry, providers, windows)
        if why:
            reasons.append(f'anthropic_targets[{index}]: {why}')
    return reasons


def _shape_reason(entry, required, optional):
    """Field-presence shape: an unknown or missing key, or a dirty required
    value -- the reason, or None once the shape is clean."""
    unknown = sorted(set(entry) - required - optional)
    if unknown:
        return f'unknown field "{unknown[0]}"'
    order = ('provider', 'model', 'base_url', 'credential', 'billing')
    missing = [k for k in order if k not in entry]
    if missing:
        return f'missing required field "{missing[0]}"'
    for k in order:
        if not isinstance(entry[k], str) or not entry[k] or any(ord(c) < 32 for c in entry[k]):
            return f'{k} must be a non-empty single-line string'
    return None


def _base_url_reason(base_url):
    """Everything _target() demands of base_url, as one reason, or None once
    it is clean. Never the URL itself -- only the scheme literal a rule
    demands, which is always the same constant, never entry data."""
    try:
        url = urlsplit(base_url)
        _ = url.port
    except ValueError:
        return 'base_url is not a valid URL'
    if url.scheme != 'https':
        return 'base_url must be https'
    if not url.hostname:
        return 'base_url must include a hostname'
    if url.username or url.password:
        return 'base_url must not contain a username or password'
    if url.query or url.fragment:
        return 'base_url must not include a query string or fragment'
    return None


def _list_fields_reason(entry):
    """accept / drop / drop_tool_fields: each a list of non-empty strings,
    and accept and drop never naming the same field."""
    for name in ('accept', 'drop', 'drop_tool_fields'):
        values = entry.get(name, [])
        if not isinstance(values, list):
            return f'{name} must be a list'
        if any(not isinstance(v, str) or not v for v in values):
            return f'{name} must be a list of non-empty strings'
    overlap = sorted(set(entry.get('accept', [])) & set(entry.get('drop', [])))
    if overlap:
        return f'"{overlap[0]}" is in both accept and drop'
    return None


def _target(entry, providers, windows):
    """Validate one anthropic_targets entry.

    Returns (Target, None) once every check passes, or (None, reason) at the
    first failing check -- the one path both configured_targets() (which
    keeps only the Target) and config_errors() (which keeps only the
    reason) read, so the two can never drift apart. A reason never carries
    a credential value, a token, a key or a full URL -- only field names,
    declared values like a provider or model name, and the fixed literal a
    rule demands.
    """
    required = {'provider', 'model', 'base_url', 'credential', 'billing'}
    optional = {'accept', 'drop', 'drop_tool_fields', 'system_messages', 'auth'}
    if not isinstance(entry, dict):
        return None, 'must be an object'
    reason = _shape_reason(entry, required, optional)
    if reason:
        return None, reason
    if entry['provider'] not in providers:
        return None, f'provider "{entry["provider"]}" is not in config/providers.yaml'
    window = windows.get(entry['provider'] + '/' + entry['model'])
    if type(window) is not int or window <= 0:
        return None, f'model "{entry["model"]}" is not catalogued for provider "{entry["provider"]}"'
    reason = _base_url_reason(entry['base_url'])
    if reason:
        return None, reason
    reason = _list_fields_reason(entry)
    if reason:
        return None, reason
    if 'system_messages' in entry and entry['system_messages'] not in ('native', 'fold'):
        return None, 'system_messages must be "native" or "fold"'
    if entry.get('auth', 'x-api-key') not in ('x-api-key', 'bearer'):
        return None, 'auth must be "x-api-key" or "bearer"'
    return Target(**{k: entry[k] for k in required}, window=window,
                  **{k: tuple(entry.get(k, [])) for k in ('accept', 'drop', 'drop_tool_fields')},
                  system_messages=entry.get('system_messages'), auth=entry.get('auth', 'x-api-key')), None


CORE = frozenset({'model', 'messages', 'max_tokens', 'stream', 'system', 'tools',
                  'tool_choice', 'temperature', 'top_p', 'top_k', 'stop_sequences'})
_ALIBABA_RESET = re.compile(r'The quota will reset at (\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}) UTC')
QUOTA_COOLDOWN_DEFAULT_S = 15 * 60


def _shape_detail(fields=(), types=None):
    """Bounded names only; never put values or unsafe names in an audit row."""
    detail = {}
    unnamed = more = 0
    for label, values in (('fields', fields), ('types', types)):
        if values is None:
            continue
        safe = sorted(value for value in values
                      if isinstance(value, str) and policy.AUDIT_NAME.fullmatch(value))
        unnamed += len(values) - len(safe)
        more += max(0, len(safe) - 12)
        detail[label] = safe[:12]
    detail['unnamed'] = unnamed
    if more:
        detail['more'] = more
    return detail


def fresh(body):
    messages = body.get('messages')
    if not isinstance(messages, list) or not messages:
        return False
    def thinking(value):
        if isinstance(value, dict):
            return value.get('type') in ('thinking', 'redacted_thinking') or any(thinking(v) for v in value.values())
        return isinstance(value, list) and any(thinking(v) for v in value)
    return not any(isinstance(m, dict) and m.get('role') == 'assistant' for m in messages) and not thinking(body)


def _transform_tools(tools, target):
    """Keep every client tool; drop only the fields the target declared."""
    if not isinstance(tools, list):
        raise policy.Refused('tools')
    for tool in tools:
        if not isinstance(tool, dict):
            raise policy.Refused('tools')
        extras = set(tool) - {'name', 'description', 'input_schema', 'type'}
        unexpected = extras - set(target.drop_tool_fields)
        bad_type = 'type' in tool and tool['type'] != 'custom'
        if (not isinstance(tool.get('name'), str) or not tool['name']
                or not isinstance(tool.get('input_schema'), dict) or bad_type or unexpected):
            detail = _shape_detail(unexpected, [tool['type']] if bad_type else []) if bad_type or unexpected else {}
            raise policy.Refused('tools', detail=detail)
        # O(1): extras is a handful of non-standard keys on one tool dict, never proportional to len(tools)
        for field in extras:
            tool.pop(field)


def _fold_system(out, target):
    """Fold or keep system-role messages in place; merge adjacent same-role turns."""
    folded, messages = [], []
    for message in out['messages']:
        if not isinstance(message, dict):
            raise policy.Refused('system_messages')
        if message.get('role') != 'system':
            messages.append(message)
        elif target.system_messages == 'native':
            messages.append(message)
        elif target.system_messages == 'fold':
            blocks = _blocks(message.get('content'))
            if any(not isinstance(b, dict) or b.get('type') != 'text' or not isinstance(b.get('text'), str) for b in blocks):
                raise policy.Refused('system_messages')
            folded.extend(blocks)
        else:
            raise policy.Refused('system_messages')
    if folded:
        out['system'] = _blocks(out.get('system', [])) + folded
        merged = []
        for message in messages:
            if merged and merged[-1].get('role') == message.get('role'):
                merged[-1]['content'] = _blocks(merged[-1].get('content')) + _blocks(message.get('content'))
            else:
                merged.append(message)
        messages = merged
    out['messages'] = messages


#: The context edit Claude Code sends on every turn in a real terminal (2.1.296,
#: thinking on or off). "keep": "all" keeps every thinking block, so it clears
#: nothing and carries no requirement a target could miss.
KEEP_ALL_THINKING = {'edits': [{'type': 'clear_thinking_20251015', 'keep': 'all'}]}


def _droppable_field(field, value):
    """Whether a declared drop may remove this field without losing a constraint.

    context_management, safeguards and output_config can carry a requirement
    the target would never see, so only their empty forms may vanish, plus an
    effort-only output_config (an accepted downgrade) and a context edit that
    keeps every thinking block (``KEEP_ALL_THINKING``, which clears nothing).
    Every other declared field, thinking and diagnostics included, is dropped
    whatever it holds.
    """
    # Exact shapes of plain containers, so type() rather than isinstance(): a
    # subclass is not an approved empty form, and a wire body comes from
    # json.loads, which makes plain ones. Refusing is the safe direction.
    if field == 'context_management':
        return type(value) is dict and value in ({}, {'edits': []}, KEEP_ALL_THINKING)
    if field == 'safeguards':
        return type(value) is list and value == []
    if field == 'output_config':
        # {} or effort only; any other key is a constraint.
        return type(value) is dict and set(value) <= {'effort'}
    return True


def transform(target, body):
    """Declared compatibility only; never discard a tool or inspect content types.

    Shape, not content: a dropped settings field is checked for its form
    alone, so a declared drop never removes a constraint. A field that
    carries one refuses the move, naming the field only (see
    ``_droppable_field``).
    """
    import copy
    if not isinstance(body, dict) or not isinstance(body.get('messages'), list):
        raise policy.Refused('fields')
    unexpected = set(body) - CORE - {'metadata'} - set(target.accept) - set(target.drop)
    if unexpected:
        raise policy.Refused('fields', detail=_shape_detail(unexpected))
    # A set: a name the config lists twice is still reported once.
    blocked = {field for field in target.drop
               if field in body and not _droppable_field(field, body[field])}
    if blocked:
        raise policy.Refused('fields', detail=_shape_detail(blocked))
    out = copy.deepcopy(body)
    _transform_tools(out.get('tools', []), target)
    _fold_system(out, target)
    for field in target.drop:
        out.pop(field, None)
    out.pop('metadata', None)
    out['model'] = target.model
    return out


def _blocks(content):
    if isinstance(content, str):
        return [{'type': 'text', 'text': content}]
    if isinstance(content, list):
        return content
    raise policy.Refused('system_messages')


def independent(target):
    """Return the pinned credential, or a reason; status uses the same gates."""
    from lib.hal import gateway_egress, spend_guard
    from lib.hal.account_ledger import LedgerView
    if not policy.billing_allowed(target.provider, target.billing):
        raise policy.Refused('spend_guard')
    if policy.load_shield() is None:
        raise policy.Refused('shield')
    try:
        key = gateway_egress._default_key_resolver(target.credential)
        # Printable ASCII only: a stray CR or LF in the key can never inject a
        # header line into the raw request built in ``_request``.
        if not key or any(not (0x20 <= ord(c) <= 0x7e) for c in key):
            raise policy.Refused('no_credential')
        account = 'w4:' + target.provider + ':' + spend_guard.account_key(key)
        if LedgerView().cooling(account) is not None or spend_guard.spent_until(account):
            raise policy.Refused('cooldown')
        return key
    except policy.Refused:
        raise
    except Exception:
        raise policy.Refused('no_credential') from None


def eligible(ex):
    from lib.hal.gateway_witness import session_id_from_headers
    from lib.hal.wire_classify import RouteClass, request_json
    if ex.route is not RouteClass.INFERENCE or ex.method != 'POST' or ex.path != '/v1/messages':
        raise policy.Refused('route')
    if ex.client_started:
        raise policy.Refused('client_started')
    if ex.body is None or ex.body.overflow or ex.wire_body is None or ex.wire_body.overflow:
        raise policy.Refused('overflow')
    if not session_id_from_headers(ex.headers):
        raise policy.Refused('session')
    body = request_json(ex.headers, ex.body.value())
    if not body or not fresh(body):
        raise policy.Refused('fresh_thread')
    return body


def _record(proxy, ex, target, reason, outcome):
    from lib.hal.gateway_witness import session_id_from_headers
    detail = reason.detail if isinstance(reason, policy.Refused) else None
    proxy._telemetry.record(policy.record_shape(session_id_from_headers(ex.headers), target,
                                                str(reason), outcome, detail=detail))


def _connect(proxy, target, ex):
    import socket
    from lib.hal import wire_proxy as w
    url = urlsplit(target.base_url)
    address = proxy._resolve(url.hostname, url.port or 443)
    raw = socket.create_connection(address, timeout=_connect_step(ex))
    try:
        # The handshake gets what the connection left of the budget, not the same
        # timeout over again.
        raw.settimeout(_connect_step(ex))
        tls = proxy.pool.dialer._context.wrap_socket(raw, server_hostname=url.hostname)
    except BaseException:
        raw.close()
        raise
    tls.settimeout(w._IDLE_TIMEOUT_S)
    return w._Upstream(tls)


def _connect_step(ex):
    """Seconds one step of connecting may take: ten, or what is left of the hold budget."""
    import time
    deadline = getattr(ex, 'hold_deadline', None)
    left = deadline - time.monotonic() if deadline is not None else 10
    if left <= 0:
        raise policy.Refused('hold_budget_expired')
    return min(10, left)


def _request(target, key, body):
    import json
    from lib.hal import wire_proxy as w
    url = urlsplit(target.base_url)
    path = url.path.rstrip('/')
    path += '/messages' if path.endswith('/v1') else '/v1/messages'
    data = json.dumps(body, ensure_ascii=True, allow_nan=False).encode()
    auth = 'Authorization: Bearer ' if target.auth == 'bearer' else 'x-api-key: '
    # Build an allowlist from scratch. No inbound credential, cookie, beta,
    # identity or vendor-specific header can cross the provider boundary.
    head = (f'POST {path} HTTP/1.1\r\nHost: {url.netloc}\r\n'
            f'{auth}{key}\r\nContent-Type: application/json\r\n'
            'anthropic-version: 2023-06-01\r\nConnection: close\r\n'
            f'Content-Length: {len(data)}\r\n\r\n').encode('latin-1')
    return w._parse_head(head), data


def _alibaba_reset(message, now):
    """The next UTC occurrence of Alibaba's month/day reset, if well formed."""
    match = _ALIBABA_RESET.search(message) if isinstance(message, str) else None
    if match is None:
        return None
    month, day, hour, minute, second = map(int, match.groups())
    year = datetime.fromtimestamp(now, timezone.utc).year
    for candidate_year in range(year, year + 9):
        try:
            reset = datetime(candidate_year, month, day, hour, minute, second,
                             tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
        if reset > now:
            return int(reset)
    return None


def _retry_after_until(headers, now):
    from lib.hal.account_ledger import read_headers
    seconds = read_headers(headers, now=now).retry_after_s
    return ceil(now + seconds) if seconds is not None and seconds > 0 else None


def _quota_until(headers, body, now) -> tuple[int, str] | None:
    """Return ``(until, source)`` only for a quota-signalled 429."""
    from lib.hal import wire_proxy as w
    retry_present = any(name.lower() == 'retry-after' for name, _ in headers)
    payload = w._json_object(body) if body is not None else None
    quota_code = payload is not None and payload.get('code') == 'Throttling.AllocationQuota'
    if not retry_present and not quota_code:
        return None
    retry = _retry_after_until(headers, now)
    reset = _alibaba_reset(payload.get('message'), now) if quota_code else None
    if retry is not None:
        return retry, 'retry_after'
    if reset is not None:
        return reset, 'stated_reset'
    return int(now) + QUOTA_COOLDOWN_DEFAULT_S, 'default'


def _past_hold_deadline(ex):
    """Whether the hold budget this exchange travels with has run out."""
    import time
    deadline = getattr(ex, 'hold_deadline', None)
    return deadline is not None and time.monotonic() >= deadline


def _refuse_past_hold_deadline(ex):
    """Refuse a move whose hold budget ran out before anything was sent.

    Before the send a refusal is safe, so the reservation is released. After the
    send, ``_within_hold_budget`` ends the wait instead and the send counts as
    uncertain, so the reservation stays (HAL#748 item 16).
    """
    if _past_hold_deadline(ex):
        raise policy.Refused('hold_budget_expired')


def _in_budget(ex, targets):
    """Yield targets while the hold budget lasts.

    Once it runs out no later target gets a signed decision or a connection: a
    decision row signed after the deadline would record a move that cannot happen.
    """
    for target in targets:
        if _past_hold_deadline(ex):
            return
        yield target


def move(proxy, client, ex):
    """Called only at the exhausted floor, with the original answer still unread."""
    from lib.hal import wire_hold
    attempts = getattr(proxy, '_w4_attempts', None)
    turn = wire_hold.exchange_key(ex)
    if attempts is not None and turn and attempts.closed(turn):
        return False
    if _past_hold_deadline(ex):
        return False
    if not wire_hold.exhaustion(proxy, ex.headers)[0]:
        return False
    targets = configured_targets()
    if not targets:
        return False
    try:
        body = eligible(ex)
    except policy.Refused as exc:
        _record(proxy, ex, targets[0].label, exc, 'refused')
        return False
    return _serve(proxy, client, ex, body, targets)


def _after_failure(proxy, ex, target, reason, outcome):
    """Record one failed attempt; ``'stop'`` when no later target may be tried.

    Before any request byte was sent, the failure is safe whatever raised it, and
    ``reason`` is recorded. Once bytes may have left, no exception proves that the
    target did not accept them, whatever its class: the turn is uncertain.
    """
    if not getattr(ex, 'w4_sending', False):
        _record(proxy, ex, target.label, reason, outcome)
        return None
    ex.w4_uncertain = True
    _record(proxy, ex, target.label, 'acceptance_uncertain', outcome)
    return 'stop'


def _attempt(proxy, client, ex, target, key, body, outcome):
    """Send one target's request; return its (conn, answer), or None having recorded why.

    The second item is ``'stop'`` when no later target may be tried either.
    """
    from lib.hal import wire_proxy as w
    conn = None
    ex.w4_sending = False
    try:
        request = _request(target, key, policy.prepare(transform(target, body), target.window))
        _refuse_past_hold_deadline(ex)
        conn = _connect(proxy, target, ex)
        answer, cooldown = _exchange(proxy, client, ex, conn, target, key, request)
    except policy.Refused as exc:
        w._close(conn.sock if conn else None)
        return None, _after_failure(proxy, ex, target, exc, outcome)
    except Exception:
        w._close(conn.sock if conn else None)
        return None, _after_failure(proxy, ex, target, 'before_send', outcome)
    if not 200 <= w._status(answer) < 300:
        w._close(conn.sock)
        _record(proxy, ex, target.label, _refusal(cooldown), outcome)
        return None, None
    return (conn, answer), None


def _exchange(proxy, client, ex, conn, target, key, request):
    """Send the request through the account ledger; return (answer, quota cooldown)."""
    from types import SimpleNamespace
    from lib.hal import spend_guard
    from lib.hal.account_ledger import AccountLedger
    head, data = request
    account = 'w4:' + target.provider + ':' + spend_guard.account_key(key)
    ledger = AccountLedger(resolver=SimpleNamespace(label_for=lambda _: (account, None)))
    with ledger.exchange(head.headers) as watch:
        _refuse_past_hold_deadline(ex)
        with _within_hold_budget(conn, ex):
            ex.w4_sending = True
            conn.sock.sendall(head.raw + data)
            answer = proxy._response_head(client, conn, ex, relay_interim=False)
            if answer is None:
                raise ConnectionError('no response')
            return answer, _note_answer(watch, conn, answer, ex)


@contextmanager
def _within_hold_budget(conn, ex):
    """Cut every wait on the target to what is left of the hold budget.

    A hold promises its client an answer inside its budget. A target that took the
    request and then said nothing would otherwise be waited on for the wire's idle
    timeout, long after that promise. The time left is worked out again before every
    read, so a target that sends one byte at a time cannot stretch the wait either.
    When the budget is gone the wait ends in a timeout, which the caller records as
    an uncertain send. Once the response head is in, the idle timeout applies again:
    from there the client is receiving the answer. After a failure the timeout is
    not put back, because the caller closes the connection. Without a hold deadline
    nothing changes.
    """
    import time
    from lib.hal import wire_proxy as w
    deadline = getattr(ex, 'hold_deadline', None)
    if deadline is None:
        yield
        return
    fill = conn.stream._fill

    def left():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('the hold budget ran out while waiting for the target')
        return min(w._IDLE_TIMEOUT_S, remaining)

    def fill_within_budget():
        conn.sock.settimeout(left())
        return fill()

    conn.stream._fill = fill_within_budget
    try:
        conn.sock.settimeout(left())
        yield
    finally:
        conn.stream._fill = fill
    conn.sock.settimeout(w._IDLE_TIMEOUT_S)


def _note_answer(watch, conn, answer, ex):
    """Feed the target's answer to the ledger; return its quota cooldown, if any."""
    import time
    from lib.hal import wire_proxy as w
    from lib.hal.account_ledger import RESET, STATUS
    code = w._status(answer)
    if code != 429:
        watch.answered(code, answer.headers)
        return None
    refusal = w._peek_body(conn.stream, answer, ex.method)
    cooldown = _quota_until(answer.headers, refusal, time.time())
    if cooldown is None:
        # A bare 429 has no credential-wide quota signal.
        watch.answered(None, answer.headers)
    else:
        # Feed derived, bounded reset evidence through the existing ledger.
        watch.answered(429, ((STATUS, 'rejected'), (RESET, str(cooldown[0]))))
    return cooldown


def _refusal(cooldown):
    """The reason recorded for a non-2xx answer, carrying any quota cooldown."""
    if cooldown is None:
        return 'target_refused'
    until, source = cooldown
    return policy.Refused('target_refused',
                          detail={'cooldown_until': until, 'cooldown_source': source})


def _sign_decision(proxy, ex, target):
    """Sign the decision to move BEFORE anything is sent to ``target``.

    A move hands the conversation to another vendor, so the signed record of
    that choice comes first (ADR-0016 decision 5): no signed row, no move, and
    the honest exhaustion answer stands. ``status`` stays null on this row; the
    outcome row from :func:`_bind_first_move` carries the target's answer.
    """
    from lib.hal.cascade_decision import sign_failover_decision
    from lib.hal.gateway_witness import session_id_from_headers
    if sign_failover_decision(target.provider, target.model, target.billing,
                              session_id_from_headers(ex.headers)):
        return True
    _record(proxy, ex, target.label, 'unsigned', 'refused')
    return False


def _bind_first_move(proxy, ex, target, answer):
    """Stickiness plus the signed outcome row, for a first move only.

    The outcome row names the target it moved to, its declared billing and its
    answer's status; the decision row before it was written by
    :func:`_sign_decision`. Never a token, header value or message text --
    only fields the thread store's allowlist (``infinite_thread.PAYLOAD_FIELDS``)
    already carries.
    """
    from lib.hal import wire_proxy as w
    from lib.hal.gateway_witness import append_thread_row, session_id_from_headers
    session = session_id_from_headers(ex.headers)
    try:
        proxy._w4_sessions.bind(session, target)
    except Exception:
        _record(proxy, ex, target.label, 'binding', 'refused')
        return False
    _record(proxy, ex, target.label, 'fresh_thread', 'moved')
    # billing travels as the keyword: the cost fields overwrite a record's own
    # 'billing' with that keyword, whose default is 'unknown'.
    append_thread_row({'provider': target.provider, 'model': target.model,
                       'status': w._status(answer)},
                      row_class='failover', route='inference', session_id=session,
                      billing=target.billing)
    return True


def _relay_answer(client, ex, conn, answer):
    from lib.hal import wire_proxy as w
    ex.client_started = True
    client.sock.sendall(answer.raw)
    ex.status = w._status(answer)
    ex.framing, length = w._response_framing(answer, ex.method)
    ex.tap = w._UsageTap(answer)
    ex.completed = w._relay(conn.stream, client.sock, (ex.framing, length), ex.tap)
    return True


def _serve(proxy, client, ex, body, targets, *, sticky=False):
    outcome = 'sticky_failure' if sticky else 'refused'
    for target in _in_budget(ex, targets):
        try:
            key = independent(target)
        except policy.Refused as exc:
            _record(proxy, ex, target.label, exc, outcome)
            continue
        if _past_hold_deadline(ex):
            return False
        # One store signs every row, so a refusal here would repeat for every target.
        if not sticky and not _sign_decision(proxy, ex, target):
            return False
        found, stop = _attempt(proxy, client, ex, target, key, body, outcome)
        if found is None:
            if stop:
                return False
            continue
        return _finish_target(proxy, client, ex, target, found, sticky)
    return False


def _finish_target(proxy, client, ex, target, found, sticky):
    from lib.hal import wire_proxy as w
    conn, answer = found
    if not sticky:
        ex.w4_accepted = True
    # A relay failure ends this stream; never restore the original refusal.
    if not sticky and not _bind_first_move(proxy, ex, target, answer):
        w._close(conn.sock)
        return False
    ex.w4_target = target
    try:
        return _relay_answer(client, ex, conn, answer)
    finally:
        w._close(conn.sock)


#: How many turn keys Anthropic answered with a 2xx the record keeps in memory.
#: They feed one count, never a decision (HAL#748 item 4).
ANSWERED_MEMORY = 4096


class Attempts:
    """One turn lifecycle: durable reservations and permanent closures.

    Outstanding Anthropic send counts are memory only. A wire restart forgets
    sends in flight at that moment, while reservations and closures survive.
    After a failed closure write, an operator checks the disk before removing
    UNHEALTHY; the process flag lasts until restart. The last ``ANSWERED_MEMORY``
    turns Anthropic answered with a 2xx are also kept in memory, so an offer of
    one of them can be counted: the "answered turn is not remembered" limit seen
    happening. That memory decides nothing.
    """

    _unhealthy = False

    def __init__(self):
        self._lock = threading.Lock()
        self._outstanding = {}
        self._answered = OrderedDict()

    def _path(self, key):
        if not isinstance(key, str) or re.fullmatch(r'[0-9a-f]{64}', key) is None:
            return None
        return Path(os.environ.get('HAL_HOME') or '~/.hal').expanduser() / 'state/hal/w4-attempts' / key

    def begin(self, mark):
        """Count one send to Anthropic that is on its way. ``mark`` is the turn's key,
        or the session id while the turn's key is not known yet (a streamed body)."""
        if not mark or not isinstance(mark, str):
            return
        with self._lock:
            self._outstanding[mark] = self._outstanding.get(mark, 0) + 1

    def end(self, key, status=None):
        """Settle one send. A 2xx ``status`` also notes the turn as answered."""
        with self._lock:
            count = self._outstanding.get(key, 0)
            if count <= 1:
                self._outstanding.pop(key, None)
            else:
                self._outstanding[key] = count - 1
            if isinstance(status, int) and 200 <= status < 300:
                self._answered.pop(key, None)
                self._answered[key] = True
                while len(self._answered) > ANSWERED_MEMORY:
                    self._answered.popitem(last=False)

    def was_answered(self, key):
        """Whether this process saw Anthropic answer this turn with a 2xx."""
        with self._lock:
            return key in self._answered

    def closed(self, key):
        path = self._path(key)
        return path is not None and path.with_name(key + '.closed').exists()

    def reserve(self, key, session=None):
        """Reserve one turn for a W4 offer; True only when nothing stands in its way.

        The lock covers the two counts and nothing else, because ``begin`` and
        ``end`` run on every turn and must never wait behind this method's file
        work. So the counts and the health flag are read again AFTER the file
        exists. A send that is still on its way then, or a closure write that
        failed meanwhile, withdraws the reservation.

        A send that began AND ended while the file was being made is gone from
        the counts by then. If it ended unanswered, its closure refuses the
        reservation below. If it was answered, that is the known limit that an
        answered turn is not remembered (``docs/w4-provider-cascade.md``, "Known
        limits"). A refused reservation removes the file it made (HAL#748 items 1, 2).
        """
        path = self._path(key)
        if path is None or self._blocked(key, session) or not self._create(path):
            return False
        if (self._blocked(key, session) or (path.parent / 'UNHEALTHY').exists()
                or path.with_name(key + '.closed').exists()):
            self.release(key)
            return False
        return True

    def _blocked(self, key, session):
        """A send of this turn, or of its session, is on its way, or a closure was lost."""
        with self._lock:
            return bool(type(self)._unhealthy or self._outstanding.get(key, 0)
                        or (session and self._outstanding.get(session, 0)))

    @staticmethod
    def _create(path):
        try:
            if (path.parent / 'UNHEALTHY').exists():
                return False
            path.parent.mkdir(parents=True, exist_ok=True)
            os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
            return True
        except OSError:
            return False

    def release(self, key):
        path = self._path(key)
        if path is not None:
            try:
                path.unlink()
            except OSError:
                pass

    def close(self, key):
        path = self._path(key)
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(path.with_name(key + '.closed'), os.O_CREAT | os.O_WRONLY, 0o600)
                os.close(fd)
            except OSError:
                type(self)._unhealthy = True
                try:
                    fd = os.open(path.parent / 'UNHEALTHY', os.O_CREAT | os.O_WRONLY, 0o600)
                    os.close(fd)
                except OSError:
                    pass


class Sessions:
    """Small atomic bindings under HAL_HOME; no credentials or conversation text.

    Bindings and the paused flag both live on disk, under the same root, so a
    wire restart -- or a second wire process mid provider-migration -- reads
    the same state. A binding is cached once read (it never changes once set);
    ``paused`` is read fresh every call, since a sibling process can flip it.
    """
    def __init__(self):
        import os
        import threading
        from pathlib import Path
        self.root = Path(os.environ.get('HAL_HOME', '~/.hal')).expanduser() / 'state/hal/w4-sessions'
        self.bindings = {}
        self.lock = threading.RLock()

    def _path(self, session):
        import hashlib
        return self.root / (hashlib.sha256(session.encode()).hexdigest() + '.json')

    def _paused_path(self, session):
        import hashlib
        return self.root / (hashlib.sha256(session.encode()).hexdigest() + '.paused.json')

    def _atomic_write(self, path, value):
        import json
        import os
        import tempfile
        self.root.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=self.root, prefix='.w4-')
        try:
            with os.fdopen(fd, 'w') as out:
                json.dump(value, out)
                out.flush()
                os.fsync(out.fileno())
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def bound(self, session):
        import json
        if not session:
            return None
        with self.lock:
            if session not in self.bindings:
                try:
                    value = json.loads(self._path(session).read_text())
                    self.bindings[session] = value if isinstance(value, dict) and 'target' in value else {'invalid': True}
                except FileNotFoundError:
                    return None
                except (OSError, ValueError):
                    self.bindings[session] = {'invalid': True}
            return self.bindings[session]

    def bind(self, session, target):
        import json
        from dataclasses import asdict
        value = {'target': json.loads(json.dumps(asdict(target)))}
        with self.lock:
            previous = self.bound(session)
            if previous is not None and previous != value:
                raise policy.Refused('binding')
            self._atomic_write(self._path(session), value)
            self.bindings[session] = value

    def paused(self, session):
        import json
        if not session:
            return False
        try:
            value = json.loads(self._paused_path(session).read_text())
            return bool(value.get('paused', False)) if isinstance(value, dict) else False
        except FileNotFoundError:
            return False
        except (OSError, ValueError):
            return False

    def set_paused(self, session, value):
        if not session:
            return
        value = bool(value)
        with self.lock:
            if self.paused(session) == value:
                return
            self._atomic_write(self._paused_path(session), {'paused': value})


def serve_bound(proxy, client, request):
    """Intercept before any account switching, spend refusal, hold or Anthropic dial."""
    import json
    import time
    from dataclasses import asdict
    from lib.hal import wire_proxy as w, wire_hold
    from lib.hal.gateway_witness import session_id_from_headers
    from lib.hal.wire_classify import classify, request_json, RouteClass
    session = session_id_from_headers(request.headers)
    binding = proxy._w4_sessions.bound(session)
    if not binding or request.parts[0].upper() != 'POST' or w.request_path(request.parts[1]) != '/v1/messages':
        return False
    body, raw = w._Copy(), w._Copy()
    if not w._relay(client, w._RequestTee(w._Discard(), raw), w._request_framing(request), body):
        return True
    route = classify('POST', request.parts[1], request.headers, body.value())
    if route is not RouteClass.INFERENCE and not body.overflow and not raw.overflow:
        client.buf[:0] = raw.value()
        return False
    ex = w._Exchange('POST', request.parts[1], request.headers, time.monotonic(),
                     route=RouteClass.INFERENCE, body=body, wire_body=raw)
    entered = proxy._cap.try_enter()
    try:
        if not entered:
            raise policy.Refused('capacity')
        if body.overflow or raw.overflow:
            raise policy.Refused('overflow')
        payload = request_json(request.headers, body.value())
        if payload is None:
            raise policy.Refused('fields')
        target = next((t for t in configured_targets() if json.loads(json.dumps(asdict(t))) == binding.get('target')), None)
        if target is None:
            raise policy.Refused('binding')
        if _serve(proxy, client, ex, payload, [target], sticky=True):
            return True
    except policy.Refused as exc:
        t = binding.get('target', {})
        _record(proxy, ex, t.get('provider', '') + '/' + t.get('model', ''), exc, 'sticky_failure')
    finally:
        if entered:
            proxy._cap.leave()
        if ex.client_started:
            proxy._record(ex)
    reset = wire_hold.exhaustion(proxy, request.headers)[1]
    client.sock.sendall(wire_hold.exhaustion_response(reset))
    ex.status, ex.completed = 429, True
    proxy._record(ex)
    return True


def status(proxy, session):
    import time
    from lib.hal import account_switch, account_ledger, spend_guard
    available, resets = False, []
    try:
        view = proxy._ledger.view()
        entries = view.accounts()
        for label, token in account_switch._read_pool():
            key = spend_guard.account_key(token)
            until = account_switch.exhausted_until(key) or account_switch.spent_until(key)
            entry = entries.get(label)
            cooling = account_ledger.cooling_of(entry, time.time())
            dead = view.token_health(label, token) == 'dead'
            if until is None and cooling is None and not dead:
                available = True
            reset = until or (cooling.until if cooling else None)
            if reset:
                resets.append(reset)
    except Exception:
        available = False
    fresh_target = None
    for target in configured_targets():
        try:
            independent(target)
            fresh_target = target.label
            break
        except Exception:
            continue
    binding = proxy._w4_sessions.bound(session)
    target = binding.get('target', {}) if binding else {}
    label = target.get('provider', '') + '/' + target.get('model', '') if target else None
    paused = proxy._w4_sessions.paused(session)
    try:
        errors = config_errors()
    except Exception:
        errors = []
    return {'session': session, 'paused': paused, 'on_target': label,
            'fresh_thread_target': fresh_target, 'anthropic_available': available,
            'earliest_reset': min(resets) if resets else None,
            'config_errors': errors}


def observed(proxy, ex):
    from lib.hal.gateway_witness import session_id_from_headers
    from lib.hal.wire_classify import RouteClass
    from lib.hal import wire_hold
    session = session_id_from_headers(ex.headers)
    if not session or ex.route is not RouteClass.INFERENCE:
        return
    paused = bool((ex.hold or {}).get('outcome') == 'spent' or
                  (ex.status == 429 and (proxy._w4_sessions.bound(session) or
                                        wire_hold.exhaustion(proxy, ex.headers)[0])))
    proxy._w4_sessions.set_paused(session, paused)
