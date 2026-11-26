"""Bounded, pre-message holds for Claude's streaming inference turns.

Only explicit rejection responses and failures to connect are retryable. Once a
request is sent, silence is waited out on that SAME connection; a reset is never
replayed. The existing in-flight cap bounds held turns. No tokens enter telemetry.

480 seconds leaves 120 seconds below the launcher timeouts; Claude tolerated 660
seconds of pings in tools/wire-matrix/RESULTS.md. A terminal in-band error triggers
Claude's non-streaming fallback. That matching request receives a one-use native
usage-limit 429, carrying the earliest reset only when the ledger knows it.
"""
from __future__ import annotations

import json
import hashlib
import os
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, Optional

from lib.hal import account_ledger, account_switch, spend_guard
from lib.hal.wire_classify import RouteClass, classify, is_account_probe, request_json

BUDGET_S = 480.0
PING_S = 3.0
DECIDE_S = 15.0
RETRY_S = 5.0
THROTTLE_ATTEMPTS = 3
#: How long a terminal answer waits for Claude's matching non-streaming fallback.
FALLBACK_S = 60.0
#: A hold's turn key once worked out and found absent; ``None`` means not worked
#: out yet, so the first send works it out (HAL#748 item 14).
NO_TURN = ""
#: Prefixed to a one-token probe's turn content (``_turn_content``).
PROBE_MARK = "probe\0"
#: The message of a 429 whose reset is the present: a pooled subscription has
#: capacity again, so the client may retry at once (HAL#748 item 18).
RETURNED_MESSAGE = "HAL wire: a pooled account has capacity again; retry now."
PING = b'event: ping\ndata: {"type":"ping"}\n\n'
HEAD = (b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n'
        b'Cache-Control: no-cache\r\nConnection: close\r\n\r\n')


class Fallback(NamedTuple):
    """One pending terminal answer (HAL#748 item 8)."""
    expires: float          # time.monotonic() after which it is gone
    reset: Optional[float]  # the reset the answer names, when one is known
    safe: bool              # the turn is known unsent, so W4 may still take it


class Fallbacks:
    """One-use, short-lived terminal answers, scoped to a credential and turn.

    Only digests are retained. Unrelated non-streaming requests keep their relay.
    Bounded independently of the in-flight cap because clients may never return.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.pending = {}

    def put(self, key, reset, safe=False):
        with self.lock:
            now = time.monotonic()
            self.pending = {k: v for k, v in self.pending.items() if v.expires > now}
            if len(self.pending) >= 128:
                self.pending.pop(next(iter(self.pending)))
            self.pending[key] = Fallback(now + FALLBACK_S, reset, safe)

    def take(self, key):
        with self.lock:
            value = self.pending.pop(key, None)
        return value if value and value.expires > time.monotonic() else None

    def peek(self, key):
        with self.lock:
            value = self.pending.get(key)
        return value if value and value.expires > time.monotonic() else None


def _turn_content(payload, probe):
    # Claude clamps max_tokens and rewrites its billing attribution on fallback.
    # Keep the actual system instructions, messages, tools and session metadata.
    # max_tokens itself stays out, so a turn and its fallback share one key; only an
    # account probe (``probe``: ``wire_classify.is_account_probe``, which honours the
    # gateway hint header) is marked apart from a turn with the same messages, so an
    # unanswered probe never closes that turn's key (HAL#748 item 7).
    body = {k: v for k, v in payload.items() if k not in ("stream", "max_tokens")}
    if isinstance(body.get("system"), list):
        body["system"] = [block for block in body["system"] if not (
            isinstance(block, dict) and isinstance(block.get("text"), str)
            and block["text"].startswith("x-anthropic-billing-header:"))]
    content = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return PROBE_MARK + content if probe else content


def fallback_key(headers, payload):
    credential = account_ledger.credential_of(headers)
    content = _turn_content(payload, is_account_probe(headers, payload))
    return hashlib.sha256((credential + "\0" + content).encode()).hexdigest()


def turn_key(headers, payload):
    """The W4 registry's key for one turn of one session; None without a session id.

    Not ``fallback_key``. That key carries the credential, and the wire may present a
    different pooled credential on a retry of the same turn, so a registry keyed on it
    would not recognise the retry. The session id is what the wire keys W4 bindings on:
    an account switch leaves it alone, a retry or Claude's non-streaming fallback
    carries it unchanged, and two sessions that send the same content stay two turns.
    A turn without one is never eligible for W4, so it needs no key.
    """
    from lib.hal.gateway_witness import session_id_from_headers
    session = session_id_from_headers(headers)
    if not session:
        return None
    content = _turn_content(payload, is_account_probe(headers, payload))
    return hashlib.sha256((session + "\0" + content).encode()).hexdigest()


def close_turn(proxy, ex):
    """Record that something may have accepted this turn: W4 is never offered it again."""
    attempts = getattr(proxy, '_w4_attempts', None)
    if attempts is None:
        return
    key = exchange_key(ex)
    if key:
        attempts.close(key)


def exchange_key(ex):
    """Turn key from a complete decoded body, or its exact wire copy on send failure."""
    for copy in (ex.body, ex.wire_body):
        if copy is None or copy.overflow:
            continue
        payload = request_json(ex.headers, copy.value())
        if payload:
            return turn_key(ex.headers, payload)
    return None


def exhaustion_response(reset, message=None):
    """The native usage-limit shape, with a reset only when actually known.
    ``message`` replaces the default wording, which names the reset."""
    body = json.dumps({"type": "error", "error": {
        "type": "rate_limit_error", "message": message or pause_message(reset)}}).encode()
    headers = (b"HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\n"
               b"anthropic-ratelimit-unified-status: rejected\r\n"
               b"anthropic-ratelimit-unified-representative-claim: five_hour\r\n")
    if reset:
        headers += (f"anthropic-ratelimit-unified-reset: {int(reset)}\r\n"
                    f"anthropic-ratelimit-unified-5h-reset: {int(reset)}\r\n"
                    f"Retry-After: {max(1, int(reset-time.time()))}\r\n").encode()
    return headers + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body


def consume_fallback(proxy, client, request):
    from lib.hal import wire_proxy as w
    if (not enabled() or request.parts[0].upper() != "POST"
            or w.request_path(request.parts[1]) != "/v1/messages"
            or w._is_chunked(request)):
        return False
    length = w._content_length(request)
    body = w._Copy()
    if length is None or length > body.limit:
        return False
    if not w._relay(client, w._Discard(), ("length", length), body):
        return True
    payload = request_json(request.headers, body.value())
    if _spent_stream(proxy, client, request, body, payload):
        return True
    value = (proxy._hold_fallbacks.take(fallback_key(request.headers, payload))
             if payload and payload.get("stream") is not True else None)
    if value:
        terminal(proxy, client, request, body, value.reset,
                 "rejected" if value.safe else "uncertain")
        return True
    client.buf[:0] = body.value()
    return False


def _spent_stream(proxy, client, request, body, payload):
    if not payload or payload.get("stream") is not True:
        return False
    if classify(request.parts[0], request.parts[1], request.headers, body.value()) is not RouteClass.INFERENCE:
        return False
    spent, reset = exhaustion(proxy, request.headers)
    if not spent:
        return False
    terminal(proxy, client, request, body, reset)
    return True


def terminal(proxy, client, request, body, reset, evidence="unsent"):
    from lib.hal import wire_proxy as w
    ex = w._Exchange("POST", request.parts[1], request.headers, time.monotonic(),
                     route=RouteClass.INFERENCE, body=body, wire_body=body, status=429)
    try:
        if _offer_under_cap(proxy, client, ex, evidence):
            return
        client.sock.sendall(exhaustion_response(reset))
        ex.completed = True
    finally:
        proxy._record(ex)


def _offer_under_cap(proxy, client, ex, evidence):
    """Offer W4 from a path that holds no in-flight slot yet, inside one.

    Every other request on the messages rail runs under the cap. Without it, each
    request answered here could hold a connection to a target for as long as that
    target stays silent, and nothing would bound how many. With no slot free, no
    offer is made and the caller sends the exhaustion answer it owes its client.
    """
    if not proxy._cap.try_enter():
        return False
    try:
        return offer_cascade(proxy, client, ex, evidence)
    finally:
        proxy._cap.leave()


def offer_cascade(proxy, client, ex, evidence):
    """Offer only a bounded turn with proof that Anthropic did not accept it."""
    if not _offer_ready(ex, evidence):
        return False
    payload = request_json(ex.headers, ex.body.value())
    turn = turn_key(ex.headers, payload) if payload else None
    if turn is None:
        return False
    fallbacks = getattr(proxy, "_hold_fallbacks", None)
    stored = fallbacks.peek(fallback_key(ex.headers, payload)) if fallbacks else None
    if stored and not stored.safe:
        return False
    from lib.hal import cascade_policy
    from lib.hal.wire_cascade import eligible, move
    try:
        eligible(ex)
    except cascade_policy.Refused:
        return False
    from lib.hal.gateway_witness import session_id_from_headers
    attempts = getattr(proxy, '_w4_attempts', None)
    if attempts is None or not attempts.reserve(turn, session_id_from_headers(ex.headers)):
        return False
    if attempts.was_answered(turn):
        # The "answered turn is not remembered" limit, seen happening. Counted so a
        # design can be chosen on numbers; never a reason to refuse (HAL#748 item 4).
        telemetry = getattr(proxy, '_telemetry', None)
        if telemetry is not None:
            telemetry.record({"event": "w4.limit", "limit": "answered_turn_again"})
    served = move(proxy, client, ex)
    if not served and not ex.w4_uncertain and not ex.w4_accepted and not ex.client_started:
        attempts.release(turn)
    return served


def _offer_ready(ex, evidence):
    return (evidence in ('unsent', 'rejected') and not ex.client_started
            and ex.body is not None and ex.wire_body is not None
            and not ex.body.overflow and not ex.wire_body.overflow
            and not ex.w4_uncertain
            and (ex.hold_deadline is None or time.monotonic() < ex.hold_deadline))


def enabled():
    marker = Path(os.environ.get("HAL_HOME") or os.path.expanduser("~/.hal")) / "state/hal/wire-hold.off"
    try:
        os.stat(marker)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


def exhaustion(proxy, headers):
    """(all spent, earliest known reset), including the request's own account.

    Missing observations mean unknown capacity, never exhaustion. Match persisted
    ledger observations to the credential digest so rotated credentials stay new.
    """
    own = account_ledger.credential_of(headers)
    tokens = {token for _, token in account_switch._read_pool()}
    if own:
        tokens.add(own)
    if not tokens:
        return False, None
    by_digest = _ledger_entries(proxy)
    resets = []
    all_spent = True
    for token in tokens:
        key = spend_guard.account_key(token)
        until = account_switch.exhausted_until(key) or account_switch.spent_until(key)
        entry = by_digest.get(account_ledger.credential_digest(token))
        cooling = account_ledger.cooling_of(entry, time.time())
        if until is None and not (cooling and cooling.reason in ("spent", "paid-overage")):
            all_spent = False
        reset = until or (cooling.until if cooling else None)
        if reset:
            resets.append(reset)
    return all_spent, min(resets) if resets else None


def dead_token_pause(proxy, headers):
    """The honest exhaustion 429 to answer instead of relaying a dead presented
    token's own 401, or None to relay it unchanged. The caller scopes this to
    401 only -- a 403 can be a permission or enforcement answer the operator
    must see, and pausing or switching over it would read as using another
    account after a ban, so it is never offered this answer.

    Honest means at least one OTHER pooled account is merely spent and known
    to come back: its own usage limit reached, its paid overage used up, or
    the ledger's own cooling for one of those two reasons -- the same marks
    :func:`exhaustion` reads. None of those is invented: each already answers
    None once its reset has passed, so no known future reset leaves the 401
    the right answer.
    """
    reset = pooled_reset(proxy, headers)
    return exhaustion_response(reset) if reset else None


def pooled_reset(proxy, headers):
    """The earliest future reset among pooled accounts other than the one
    ``headers`` presents, that are cooling for being spent or for paid
    overage used up. None when no other pooled account is known to come back.

    A pool account the ledger already holds dead is skipped even if it also
    carries a stale cooling reset from before it went dead: a dead token
    needs a re-login, not a timer, so its old reset is never read as a
    promise it will come back on its own.
    """
    own = account_ledger.credential_of(headers)
    by_digest = _ledger_entries(proxy)
    now = time.time()
    resets = []
    for _, token in account_switch._read_pool():
        if token == own:
            continue
        entry = by_digest.get(account_ledger.credential_digest(token))
        if account_ledger.token_health_of(entry, token) == "dead":
            continue
        _, until = _spent_mark(token, entry, now)
        if until is not None:
            resets.append(until)
    return min(resets) if resets else None


def carries_spent_mark(proxy, token):
    """True while ``token``, or a pooled token of its known organisation,
    carries a spent mark (:func:`_spent_mark`).

    The account switcher reads only this process's marks, so an account another
    wire has seen spent -- recorded as the ledger's cooling -- can still be its
    choice; this reads the ledger too (HAL#748 item 18). Accounts in one
    organisation share limits, so a cooling on a pooled sibling counts as well
    (Codex on HAL#770). A sibling outside the pool is not read: its token, and
    so its ledger entry, is unknown here. Nor is a pooled token whose
    organisation this process has never looked up, because the caller's check
    never looks one up (AGORA on HAL#770).
    """
    entries, now = _ledger_entries(proxy), time.time()
    siblings = set(account_switch._org_siblings(spend_guard.account_key(token)))
    tokens = [token] + [other for _, other in account_switch._read_pool()
                        if other != token and spend_guard.account_key(other) in siblings]
    return any(_spent_mark(each, entries.get(account_ledger.credential_digest(each)), now)[0]
               for each in tokens)


def _ledger_entries(proxy):
    """The account ledger's entries, keyed by credential digest. An entry whose
    ``token`` is not an object names no digest, so it is left out rather than
    raised on: a damaged ledger must not cost the client its answer."""
    entries = proxy._ledger.view().accounts().values()
    return {e["token"].get("digest"): e for e in entries if isinstance(e.get("token"), dict)}


def _spent_mark(token, entry, now):
    """``(spent, reset)`` for one credential: whether it carries a spent mark --
    its own usage limit reached, its paid overage used up, or the ledger's
    cooling for one of those two reasons, as :func:`exhaustion` reads them, its
    organisation's siblings included -- and that mark's future reset when one is
    known. A cooling with no known reset is still a mark."""
    key = spend_guard.account_key(token)
    until = account_switch.exhausted_until(key) or account_switch.spent_until(key)
    if until is not None:
        return True, until
    cooling = account_ledger.cooling_of(entry, now)
    if cooling is not None and cooling.reason in ("spent", "paid-overage"):
        return True, cooling.until
    return False, None


def pause_message(reset=None):
    message = "HAL wire: temporarily paused; retry when upstream capacity returns."
    if reset:
        when = datetime.fromtimestamp(reset, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        message = f"HAL wire: capacity exhausted; earliest known reset is {when}."
    return message


def prepare(proxy, client, request, original_headers=None):
    """Handle a bounded Claude streaming body, or restore it for the normal relay.

    Claude sends Content-Length. Unbounded/chunked uploads retain the original
    relay rather than consuming unbounded memory before an upstream is opened.
    """
    from lib.hal import wire_proxy as w
    if (not enabled() or request.parts[0].upper() != "POST"
            or w.request_path(request.parts[1]) != "/v1/messages"):
        return False
    length = w._content_length(request)
    if w._is_chunked(request) or length is None or length > w._MAX_COPY:
        return False
    body = w._Copy()
    if length > body.limit:
        return False
    if not w._relay(client, w._Discard(), ("length", length), body):
        return True
    payload = request_json(request.headers, body.value())
    route = classify(request.parts[0], request.parts[1], request.headers, body.value())
    if not payload or payload.get("stream") is not True or route is not RouteClass.INFERENCE:
        client.buf[:0] = body.value()
        return False
    if not proxy._cap.try_enter():
        client.sock.sendall(proxy._busy)
        return True
    ex = w._Exchange("POST", request.parts[1], request.headers, time.monotonic(),
                     route=route, body=body, wire_body=body)
    try:
        Hold(proxy, client, request, ex,
             fallback_key(original_headers or request.headers, payload),
             turn=turn_key(request.headers, payload) or NO_TURN).run()
    finally:
        proxy._record(ex)
        proxy._cap.leave()
    return True


class BudgetExpired(Exception):
    pass


class Hold:
    def __init__(self, proxy, client, request, ex, fallback, turn=None):
        from lib.hal import wire_proxy as w
        self.w, self.proxy, self.client, self.request, self.ex = w, proxy, client, request, ex
        # The turn's registry key, worked out once for the whole hold: every attempt
        # sends the same turn, and the body is megabytes on a long conversation.
        self.turn = turn
        self.original_request = request
        self.deadline = ex.started + BUDGET_S
        self.ex.hold_deadline = self.deadline
        self.started = None
        self.next_ping = 0.0
        self.attempts = 0
        self.signal = ""
        self.outcome = "interrupted"
        self.conn = None
        self.throttles = 0
        self.quota_replayed = False
        self.fallback = fallback
        self.message_started = False
        self.send_state = "unsent"
        self.send_key = None

    def begin(self, signal):
        if self.started is None:
            self.started = time.monotonic()
            self.signal = signal
            self.ex.client_started = True
            self.client.sock.sendall(HEAD)
        self.tick()

    def tick(self):
        now = time.monotonic()
        if now >= self.deadline:
            raise BudgetExpired
        if self.started is not None and now >= self.next_ping:
            self.client.sock.sendall(PING)
            self.next_ping = now + PING_S

    def wait(self, seconds):
        end = min(time.monotonic() + seconds, self.deadline)
        while time.monotonic() < end:
            self.tick()
            time.sleep(min(PING_S, max(0, end - time.monotonic())))
        self.tick()

    def read_head(self):
        # read_head retains partial headers across socket timeouts.
        while True:
            self.tick()
            self.conn.sock.settimeout(min(PING_S, max(0.001, self.deadline - time.monotonic())))
            try:
                raw = self.conn.stream.read_head()
            except socket.timeout:
                if time.monotonic() - self.ex.started >= DECIDE_S:
                    self.begin("headers_wait")
                continue
            if raw is None:
                raise ConnectionError("accepted request lost its connection")
            head = self.w._parse_head(raw)
            if self.w._status(head) >= 200:
                return head
            # Interim responses are upstream metadata, not message content.

    def fail(self, outcome, reset=None):
        self.outcome = outcome
        if outcome in ("budget", "spent"):
            safe = (outcome == "spent" and self.started is None and not self.ex.client_started
                    and not self.ex.w4_uncertain and time.monotonic() < self.deadline
                    and self.send_state in ("unsent", "rejected"))
            self.proxy._hold_fallbacks.put(self.fallback, reset, safe=safe)
        if self.started is None:
            self.client.sock.sendall(exhaustion_response(reset) if outcome in ("budget", "spent")
                                     else self.w._NO_ANSWER)
            self.ex.status = 429 if outcome in ("budget", "spent") else 503
            self.ex.completed = True
        else:
            error = {"type": "error", "error": {"type": "api_error", "message": pause_message(reset)}}
            self.client.sock.sendall(b"event: error\ndata: " + json.dumps(error).encode() + b"\n\n")

    def run(self):
        try:
            self.tick()
            spent, reset = exhaustion(self.proxy, self.ex.headers)
            if spent:
                if not offer_cascade(self.proxy, self.client, self.ex, "unsent"):
                    self.client.sock.sendall(exhaustion_response(reset))
                    self.ex.status = 429
                    self.ex.completed = True
                return
            self._loop()
        except BudgetExpired:
            self.fail("budget", exhaustion(self.proxy, self.ex.headers)[1])
        except (OSError, ValueError, self.w._WireProtocolError):
            # The request might have been accepted. NEVER retry this path.
            if self.ex.client_started and self.started is None:
                return  # the normal relay already sent headers; never append HTTP
            self.fail("budget" if not self.message_started and time.monotonic() >= self.deadline
                      else "uncertain_acceptance")
        finally:
            if self.send_key:
                # Closed first, then unmarked: a send that got no answer must not
                # leave its mark behind for the life of the process.
                close_turn(self.proxy, self.ex)
                if getattr(self.proxy, '_w4_attempts', None):
                    self.proxy._w4_attempts.end(self.send_key)
                self.send_key = None
            self.w._close(self.conn.sock if self.conn else None)
            if self.started is not None:
                self.ex.hold = {"signal": self.signal,
                    "attempts": self.attempts, "seconds": round(time.monotonic()-self.started, 3),
                    "outcome": self.outcome}
                self.proxy._telemetry.record({"event": "wire.hold", **self.ex.hold})

    def _loop(self):
        while True:
            self.tick()
            spent, reset = exhaustion(self.proxy, self.ex.headers)
            if spent:
                if self.started is None and offer_cascade(
                        self.proxy, self.client, self.ex, self.send_state):
                    return
                self.fail("spent", reset)
                return
            self.attempts += 1
            head = self._answer()
            if head is None:
                self.begin("connect")
                self.wait(RETRY_S)
                continue
            if self._handle_answer(head):
                return

    def _handle_answer(self, head):
        status = self.w._status(head)
        if status in (401, 403) and self.request is not self.original_request:
            self._back_to_original(status)
            return False
        exhausted = account_switch.observe(status, head.headers, self.ex.headers)
        if exhausted is not None or (status in (401, 403) and self.started is None):
            return self._handle_rejection(head, exhausted)
        quota = any(k.lower().startswith("anthropic-ratelimit-unified-") for k, _ in head.headers)
        holdable = (status == 429 and not quota) or 500 <= status <= 599
        if holdable and self._context_error(head):
            holdable = False
        if not holdable:
            self.serve(head)
            return True
        self._wait_retryable(head, status)
        return False

    def _handle_rejection(self, head, exhausted):
        if self.started is None:
            if not self.proxy._replay_rejected(self.client, self.request, self.ex, head):
                self.proxy._relay_response(self.client, self.conn, self.ex, head)
            return True
        if not self.quota_replayed and self._switch():
            self.quota_replayed = True
            self.w._close(self.conn.sock)
            self.conn = None
            return False
        self.fail("spent", exhaustion(self.proxy, self.ex.headers)[1] or exhausted)
        return True

    def _wait_retryable(self, head, status):
        self.begin(str(status))
        self.w._close(self.conn.sock)
        self.conn = None
        reading = account_ledger.read_headers(head.headers)
        self.wait(max(RETRY_S if reading.retry_after_s is None else reading.retry_after_s, 0.01))
        self.throttles = self.throttles + 1 if status == 429 else 0
        if self.throttles >= THROTTLE_ATTEMPTS and self._switch(after_throttle=True):
            self.throttles = 0

    def _answer(self):
        """Send the turn on a fresh connection and read its answer's head; None when
        no connection could be opened. A 401 or 403 is judged from its body, which
        goes back unread (``WireProxy._judge``): an enforcement signal stops all
        account switching before this hold could switch. ``ex.auth`` only ever
        holds the judgement of the head this attempt read."""
        self.ex.auth = None
        try:
            self.conn = self.proxy.pool.take(timeout=min(PING_S, self.deadline-time.monotonic()))
        except (OSError, ValueError):
            return None
        self.conn.sock.settimeout(min(PING_S, self.deadline-time.monotonic()))
        # A send or read failure after this point is ambiguous: run() ends it.
        self.send_state = "uncertain"
        if self.turn is None:
            self.turn = exchange_key(self.ex) or NO_TURN
        self.send_key = self.turn or None
        if self.send_key and getattr(self.proxy, '_w4_attempts', None):
            self.proxy._w4_attempts.begin(self.send_key)
        self.conn.sock.sendall(self.request.raw + self.ex.wire_body.value())
        with self.proxy._ledger.exchange(self.ex.headers) as watch:
            head = self.read_head()
            status = self.w._status(head)
            if self.send_key and getattr(self.proxy, '_w4_attempts', None):
                self.proxy._w4_attempts.end(self.send_key, status)
            self.send_key = None
            watch.answered(status, head.headers)
        self.ex.auth = self.proxy._judge(self.conn.stream, head, self.ex.headers, self.ex.method)
        return head

    def _back_to_original(self, status):
        """A replacement answered 401 or 403: return to the turn's own account,
        which is no switch, and try again after a pause."""
        self.proxy._telemetry.record({"event": "wire.dead_token",
            "account": self.proxy._ledger._label_for(self.ex.headers)[0],
            "switched": False, "outcome": "retry_original", "status": status,
            "signal": getattr(self.ex.auth, "kind", None)})
        self.w._close(self.conn.sock)
        self.conn = None
        self.request = self.original_request
        self.ex.headers = self.request.headers
        self.wait(RETRY_S)

    def _switch(self, after_throttle=False):
        def probe(token):
            self.tick()
            return self.proxy._probe_org(token)
        choice = account_switch.choose(self.ex.method, self.ex.target, self.ex.headers,
            probe_org=probe, after_throttle=after_throttle)
        self.tick()
        if choice:
            replacement = self.w._with_bearer(self.request, choice[1])
            if not spend_guard.check_anthropic_request(self.ex.method, self.ex.target, replacement.headers):
                self.request = replacement
                self.ex.headers = replacement.headers
                return True
        return False

    def _context_error(self, head):
        """Inspect a bounded rejection body; request errors never enter retry."""
        raw, body = self.w._Copy(limit=65536), self.w._Copy(limit=65536)
        self.tick()
        fill = self.conn.stream._fill

        def polling_fill():
            while True:
                self.tick()
                self.conn.sock.settimeout(min(PING_S, max(.001, self.deadline-time.monotonic())))
                try:
                    return fill()
                except socket.timeout:
                    if time.monotonic()-self.ex.started >= DECIDE_S:
                        self.begin("rejection_body_wait")

        self.conn.stream._fill = polling_fill

        class Capture:
            def sendall(_self, data):
                raw(data)
                if raw.overflow:
                    raise self.w._WireProtocolError("oversized rejection")

        try:
            if not self.w._relay(self.conn.stream, Capture(),
                                 self.w._response_framing(head, self.ex.method), body):
                raise self.w._WireProtocolError("incomplete rejection")
        finally:
            self.conn.stream._fill = fill
        self.conn.stream.buf[:0] = raw.value()
        decoded = self.w.decoded_body(head.headers, body.value()) or b""
        error = (self.w._json_object(decoded) or {}).get("error", {})
        message = str(error).lower()
        return any(text in message for text in (
            "context_length_exceeded", "context too long", "prompt is too long",
            "context window", "request_too_large"))

    def serve(self, head):
        status = self.w._status(head)
        self.conn.sock.settimeout(self.w._IDLE_TIMEOUT_S)
        if self.started is None:
            self.proxy._relay_response(self.client, self.conn, self.ex, head)
            return
        if status != 200 or "text/event-stream" not in head.header("content-type").lower():
            self.fail("upstream_error")
            return
        self.ex.status = status
        spend_guard.observe_anthropic_response(status, head.headers, self.ex.headers)
        self.ex.tap = self.w._UsageTap(head)
        self.ex.framing = "close"
        first = bytearray()
        message_started = False
        fill = self.conn.stream._fill

        def polling_fill():
            while True:
                if not message_started:
                    self.tick()
                    self.conn.sock.settimeout(min(PING_S, max(0.001, self.deadline-time.monotonic())))
                try:
                    return fill()
                except socket.timeout:
                    if message_started:
                        raise

        self.conn.stream._fill = polling_fill

        def payload(data):
            nonlocal message_started
            if not message_started:
                first.extend(data)
                if len(first) > self.w._MAX_EVENT:
                    raise self.w._WireProtocolError("missing message_start")
                while True:
                    marker = b"\r\n\r\n" if b"\r\n\r\n" in first else b"\n\n"
                    end = first.find(marker)
                    if end < 0:
                        return
                    block = bytes(first[:end + len(marker)])
                    del first[:end + len(marker)]
                    event = next((line[6:].strip() for line in block.splitlines()
                                  if line.startswith(b"event:")), b"")
                    if event == b"message_start":
                        message_started = True
                        self.message_started = True
                        self.conn.sock.settimeout(self.w._IDLE_TIMEOUT_S)
                        data = block + bytes(first)
                        break
                    if event not in (b"", b"ping"):
                        raise self.w._WireProtocolError("unexpected event before message_start")
            self.client.sock.sendall(data)
            self.ex.tap(data)

        # The committed stream is close-delimited. Decode upstream chunk framing;
        # the relay tap gets payload only, so HTTP chunk sizes never enter the SSE.
        self.ex.completed = self.w._relay(self.conn.stream, self.w._Discard(),
            self.w._response_framing(head, self.ex.method), payload)
        if not message_started:
            self.fail("missing_message_start")
            self.ex.completed = False
            return
        self.outcome = "served" if self.ex.completed else "interrupted"
