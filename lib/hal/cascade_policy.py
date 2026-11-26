"""Shared W4 send gates. No prompt or credential belongs in an audit row."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import socket
import sys
import threading
import time
import urllib.error
from pathlib import Path

from lib.hal import spend_guard

CONFIG = Path(__file__).resolve().parents[2] / 'config' / 'w4_cascade.json'
AUDIT_NAME = re.compile(r'[a-z][a-z0-9_]{0,39}')
_lock = threading.Lock()
_shield = None
#: A failed load is retried after this many seconds, never cached forever --
#: one missing file or import blip must not disable every move for the life
#: of the wire (mirrors ``account_switch._organisation``, HAL#671).
SHIELD_RETRY_SECONDS = 60.0
_shield_retry_at = 0.0


class Refused(ValueError):
    """Closed-vocabulary refusal; only validated detail enters an audit row.

    Detail may contain ``fields``/``types`` (up to 12 safe names each),
    bounded non-negative integer ``unnamed``/``more``, a ``cooldown_until``
    epoch before the year 2100, and, only beside such a time, a
    ``cooldown_source`` (``retry_after``, ``stated_reset``, or ``default``).
    ``record_shape`` drops all other data, and adds any name it drops itself
    to ``unnamed`` (unsafe) or ``more`` (past the twelfth).
    """

    def __init__(self, reason, *, detail=None):
        super().__init__(reason)
        self.detail = {} if detail is None else detail


def load_shield(now=None):
    """Load GRIP once; a success is cached forever, a failure retried after 60s."""
    global _shield, _shield_retry_at
    clock = time.monotonic() if now is None else now
    with _lock:
        if _shield is not None:
            return _shield
        if clock < _shield_retry_at:
            return None
        try:
            directory = (Path(os.environ.get('CLAUDE_DIR', '~/.claude')).expanduser() / 'lib').resolve()
            if not (directory / 'review_redaction.py').is_file():
                raise FileNotFoundError(directory)
            if str(directory) not in sys.path:
                sys.path.append(str(directory))
            module = importlib.import_module('review_redaction')
            if Path(module.__file__).resolve() != directory / 'review_redaction.py':
                raise ImportError('import priority changed')
            _shield = module.shield_payload
            return _shield
        except Exception:
            _shield_retry_at = clock + SHIELD_RETRY_SECONDS
            return None


def shield(body):
    try:
        loader = load_shield()
        if loader is None:
            raise Refused('shield')
        return loader(body)
    except Exception:
        raise Refused('shield') from None


def budget(body):
    limits = [body[k] for k in ('max_tokens', 'max_output_tokens', 'max_completion_tokens') if k in body]
    if any(type(n) is not int or n <= 0 for n in limits):
        return None
    try:
        return len(json.dumps(body, ensure_ascii=True, allow_nan=False).encode()) + 4096 + max(limits or [4096])
    except (ValueError, TypeError):
        return None


def prepare(body, window):
    body = shield(body)
    size = budget(body)
    if size is None or not window or size > window:
        raise Refused('context_unknown_or_small')
    return body


def billing_allowed(provider, billing=None):
    source = spend_guard.OPENAI_API_KEY if provider == 'openai' else provider + '-api-key'
    return billing == 'subscription' or spend_guard.decide_paid(source).allowed


def before_send(exc):
    if isinstance(exc, urllib.error.URLError):
        exc = exc.reason
    return isinstance(exc, (ConnectionRefusedError, socket.gaierror))


#: Bounds for the integers an audit detail may carry. An unbounded integer could
#: encode arbitrary data, so a count or a time outside its plausible range is dropped.
_MAX_AUDIT_COUNT = 10_000
_MAX_AUDIT_EPOCH = 4_102_444_800  # 2100-01-01 00:00:00 UTC


def _safe_names(values):
    """``(kept, unnamed, more)`` for one name list; what is not kept is counted."""
    valid = [name for name in values if isinstance(name, str) and AUDIT_NAME.fullmatch(name)]
    return valid[:12], len(values) - len(valid), max(0, len(valid) - 12)


def _safe_count(value):
    return value if type(value) is int and 0 <= value <= _MAX_AUDIT_COUNT else None


def _safe_cooldown(detail):
    until = detail.get('cooldown_until')
    if type(until) is not int or not 0 < until < _MAX_AUDIT_EPOCH:
        return {}
    safe = {'cooldown_until': until}
    # A source says where that time came from, so it is kept only beside one.
    source = detail.get('cooldown_source')
    if type(source) is str and source in ('retry_after', 'stated_reset', 'default'):
        safe['cooldown_source'] = source
    return safe


def _audit_detail(detail):
    """The detail an audit row may carry: declared keys, checked values, drops counted."""
    if not isinstance(detail, dict):
        return {}
    safe = {}
    dropped = {'unnamed': 0, 'more': 0}
    for key in ('fields', 'types'):
        if isinstance(detail.get(key), list):
            safe[key], unnamed, more = _safe_names(detail[key])
            dropped['unnamed'] += unnamed
            dropped['more'] += more
    for key in ('unnamed', 'more'):
        count = _safe_count(detail.get(key))
        if count is not None or dropped[key]:
            safe[key] = min((count or 0) + dropped[key], _MAX_AUDIT_COUNT)
    safe.update(_safe_cooldown(detail))
    return safe


def record_shape(session, target, reason, outcome, *, detail=None):
    row = {'event': 'w4.cascade', 'session_hash': hashlib.sha256((session or '').encode()).hexdigest(),
           'target': target, 'reason': reason, 'outcome': outcome}
    safe_detail = _audit_detail(detail)
    if safe_detail:
        row['detail'] = safe_detail
    return row


def _config_or_reason():
    """Read and parse the W4 config file: ``(value, reason)``.

    One code path for both callers: ``config()`` keeps only ``value``, and
    ``wire_cascade.config_errors()`` keeps only ``reason``, so the two can
    never check different things. A missing file -- or anything else
    ``read_text``/``json.loads`` raises that is not a parse failure -- is W4
    staying dormant by design, never a reason. Invalid JSON, or JSON whose
    top level is not an object, is.
    """
    try:
        value = json.loads(Path(os.environ.get('HAL_W4_CONFIG', str(CONFIG))).read_text())
    except (OSError, TypeError):
        return {}, None
    except ValueError:
        return {}, 'w4_cascade.json: not valid JSON'
    if not isinstance(value, dict):
        return {}, 'w4_cascade.json: must be a JSON object'
    return value, None


def config():
    return _config_or_reason()[0]
