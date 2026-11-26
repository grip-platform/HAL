"""Replay a rejected turn over real loopback TLS, with fake accounts only."""
import json

import pytest

from lib.hal import account_switch, spend_guard, wire_proxy
from tests import test_wire_proxy as helpers
from tests.test_wire_proxy import home, pki, make_wire, strict_spend

# pytest resolves these fixtures by name in this module; `make_wire` needs `pki`.
FIXTURES = (home, pki, make_wire, strict_spend)


@pytest.fixture(autouse=True)
def _healthy_attempts(monkeypatch):
    from lib.hal.wire_cascade import Attempts
    monkeypatch.setattr(Attempts, '_unhealthy', False, raising=False)

A, B, C =("fake-account-A-token", "fake-account-B-token", "fake-account-C-token")


def reply(status, body, headers=b""):
    return (f"HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\n".encode()
            + headers + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)


LIMIT = reply(429, b'{"error":{"message":"original limit"}}',
              b"anthropic-ratelimit-unified-status: rejected\r\n")
OK = reply(200, b'{"content":[{"type":"text","text":"replacement"}]}')


class Upstream(helpers.FakeUpstream):
    answers = None

    def _one(self, tls, reader, number):
        head = reader.until(b"\r\n\r\n")
        if not head:
            return False
        if b"Transfer-Encoding: chunked" in head:
            body = reader.until(b"0\r\nX-Trailer: kept\r\n\r\n")
        else:
            body = reader.exact(helpers._header_int(head, b"content-length"))
        self.requests.append((number, head + body))
        token = helpers._bearer_of(head)
        if head.startswith(b"GET /v1/models"):
            tls.sendall(reply(200, b"{}", b"anthropic-organization-id: fake-org-"
                               + token.encode() + b"\r\n"))
        else:
            tls.sendall(self.answers.get(token, OK))
        return True


@pytest.fixture
def replay_wire(make_wire, monkeypatch):
    monkeypatch.setattr(helpers, "FakeUpstream", Upstream)
    monkeypatch.setattr(Upstream, "answers", {A: LIMIT})
    monkeypatch.setattr(spend_guard, "_OAUTH_PREFIX", "fake-account-")
    monkeypatch.setattr(account_switch, "_ORGS", {})
    monkeypatch.setattr(account_switch, "_ORG_RETRY", {})
    monkeypatch.setattr(account_switch, "_EXHAUSTED", {})
    monkeypatch.setattr(account_switch, "_read_pool", lambda: [("B", B), ("C", C)])
    return make_wire(pool_size=0)


def request(path="/v1/messages", payload=None, chunked=False):
    body = json.dumps(helpers.TURN if payload is None else payload).encode()
    framing = (b"Transfer-Encoding: chunked\r\n" if chunked else
               f"Content-Length: {len(body)}\r\n".encode())
    if chunked:
        body = f"{len(body):x};keep=yes\r\n".encode() + body + b"\r\n0\r\nX-Trailer: kept\r\n\r\n"
    return (f"POST {path} HTTP/1.1\r\nHost: {helpers.API}\r\nAuthorization: Bearer {A}\r\n"
            "Content-Type: application/json\r\nX-Unchanged:  spacing\r\n".encode()
            + framing + b"\r\n" + body)


def posts(wire):
    return [raw for _, raw in wire.fake.requests if raw.startswith(b"POST")]


def replay_row(wire):
    rows = helpers._wait_for(lambda: helpers._witness_rows(wire.home))
    return rows[-1].get("account_replay")


@pytest.mark.parametrize("signal", [
    b"anthropic-ratelimit-unified-status: rejected\r\n",
    b"anthropic-ratelimit-unified-representative-claim: five_hour\r\n",
    b"anthropic-ratelimit-unified-overage-status: rejected\r\n",
])
@pytest.mark.parametrize("chunked", [False, True])
def test_limit_replayed_exactly_and_recorded(replay_wire, signal, chunked):
    wire = replay_wire
    wire.fake.answers[A] = reply(429, b"original", signal)
    raw = request(chunked=chunked)
    assert helpers._exchange(wire, raw) == OK
    assert posts(wire) == [raw, raw.replace(A.encode(), B.encode())]
    row = replay_row(wire)
    assert row == {"from": spend_guard.account_key(A), "to": spend_guard.account_key(B),
                   "reason": "limit_reached", "rank": "subscription", "status": 200,
                   "outcome": "relayed"}
    assert all(token not in json.dumps(helpers._rows(wire.home)) for token in (A, B, C))


def test_no_replacement_keeps_original(replay_wire, monkeypatch):
    monkeypatch.setattr(account_switch, "_read_pool", lambda: [])
    assert helpers._exchange(replay_wire, request()) == LIMIT
    assert len(posts(replay_wire)) == 1


def _limit_until(reset):
    return reply(429, b'{"error":{"message":"own limit"}}',
                 b"anthropic-ratelimit-unified-status: rejected\r\n"
                 + f"anthropic-ratelimit-unified-reset: {reset}\r\n".encode())


def _refused_move(wire, monkeypatch, stream, own_answer, replacement_answer,
                  pool=(("B", B),), during_move=None, mark=None):
    """One turn on A whose replacement B answers too, with a fallback target configured
    that refuses the move. ``during_move`` runs while the move is being attempted, and
    ``mark`` names an exchange flag the refused attempt leaves set (an uncertain send).
    Returns the client's answer, split into head and body."""
    from lib.hal import wire_cascade
    wire.fake.answers.update({A: own_answer, B: replacement_answer})
    monkeypatch.setattr(account_switch, "_read_pool", lambda: list(pool))
    monkeypatch.setattr(wire_cascade, "configured_targets", lambda: [object()])
    moves = []

    def refused(proxy, client, ex):
        moves.append(1)
        if during_move:
            during_move()
        if mark:
            setattr(ex, mark, True)
        return False

    monkeypatch.setattr(wire_cascade, "move", refused)
    answer = helpers._exchange(wire, _session_turn(stream))
    assert moves == [1] and len(posts(wire)) == 2
    return answer.split(b"\r\n\r\n", 1)


def _session_turn(stream):
    """A first turn on A, with a session id so a fallback target may be offered it."""
    turn = dict(helpers.TURN, stream=True) if stream else helpers.TURN
    return request(payload=turn).replace(
        b"X-Unchanged:", b"x-claude-code-session-id: fake-session\r\nX-Unchanged:")


def _resets():
    import time
    now = int(time.time())
    return now + 120, now + 600


def _last_row(wire):
    return helpers._wait_for(lambda: helpers._witness_rows(wire.home))[-1]


@pytest.mark.parametrize("stream", [False, True])
def test_a_refused_move_after_a_spent_replacement_names_the_earliest_reset(
        replay_wire, monkeypatch, stream):
    """HAL#676's weekly real-terminal run (S8, every account spent, through the wire):
    account A's limit reset ten minutes out, B's two minutes out, and a fallback
    target was configured but refused the move. The wire relayed A's own refusal,
    so Claude Code paused until A's reset and missed B's by eight minutes. The
    contract promises one pause notice naming the earliest reset."""
    early, late = _resets()
    head, body = _refused_move(replay_wire, monkeypatch, stream,
                               _limit_until(late), _limit_until(early))
    assert head.startswith(b"HTTP/1.1 429")
    assert f"anthropic-ratelimit-unified-reset: {early}".encode() in head
    assert str(late).encode() not in head and b"own limit" not in body
    assert b"earliest known reset" in body
    row = _last_row(replay_wire)
    assert (row["status"], row["served"]) == (429, True)
    # The limit branch keeps the replay's record and marks it paused.
    assert row["account_replay"]["reason"] == "limit_reached"
    assert row["account_replay"]["outcome"] == "paused_until_reset"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("replacement", ["later", "same"])
def test_a_refused_move_keeps_the_own_refusal_that_already_names_the_earliest_reset(
        replay_wire, monkeypatch, stream, replacement):
    """The original refusal is relayed byte for byte when it is already the honest
    answer: A comes back first, or with B, so nothing is synthesised."""
    early, late = _resets()
    original = _limit_until(early)
    other = _limit_until(late if replacement == "later" else early)
    head, body = _refused_move(replay_wire, monkeypatch, stream, original, other)
    assert head + b"\r\n\r\n" + body == original


@pytest.mark.parametrize("stream", [False, True])
def test_a_refused_move_after_a_dead_token_pauses_until_the_replacement_resets(
        replay_wire, monkeypatch, stream):
    """The presented token is dead (a plain 401), its replacement B is spent, and the
    fallback refuses the move. The dead token's 401 would send Claude Code to a
    re-login while B comes back by itself in two minutes, so the wire answers as it
    does for a dead token with no account to carry the turn: the honest pause."""
    early, _ = _resets()
    head, body = _refused_move(replay_wire, monkeypatch, stream,
                               reply(401, b"dead original"), _limit_until(early))
    assert head.startswith(b"HTTP/1.1 429")
    assert f"anthropic-ratelimit-unified-reset: {early}".encode() in head
    assert b"dead original" not in body and b"earliest known reset" in body
    assert _last_row(replay_wire)["account_replay"] == {"outcome": "paused_until_reset"}


@pytest.mark.parametrize("stream", [False, True])
def test_a_limit_that_passes_during_the_move_is_not_taken_for_a_dead_token(
        replay_wire, monkeypatch, stream):
    """Codex on HAL#750: A's limit can pass while a slow fallback attempt runs. A
    missing mark then read as a dead token, so the pause named B's later reset and
    left out A, which was usable again. A's own refusal now goes back, and it names
    a reset that is already over."""
    early, late = _resets()
    original = _limit_until(early)
    own = spend_guard.account_key(A)
    head, body = _refused_move(replay_wire, monkeypatch, stream, original, _limit_until(late),
                               during_move=lambda: account_switch._EXHAUSTED.pop(own))
    assert head + b"\r\n\r\n" + body == original


@pytest.mark.parametrize("stream", [False, True])
def test_a_refused_move_never_names_the_stale_reset_of_a_dead_pool_token(
        replay_wire, monkeypatch, stream):
    """Codex on HAL#750: pool token C is held dead but still carries an earlier mark
    from before its 401. That reset is no promise C will come back, so the pause
    names B's, as the dead-token pause would."""
    from lib.hal import account_ledger
    early, late = _resets()
    account_switch._EXHAUSTED[spend_guard.account_key(C)] = early - 60
    real = account_ledger.token_health_of
    monkeypatch.setattr(account_ledger, "token_health_of",
                        lambda entry, token: "dead" if token == C else real(entry, token))
    head, _ = _refused_move(replay_wire, monkeypatch, stream, _limit_until(late),
                            _limit_until(early), pool=(("B", B), ("C", C)))
    assert f"anthropic-ratelimit-unified-reset: {early}".encode() in head
    assert str(early - 60).encode() not in head


@pytest.mark.parametrize("stream", [False, True])
def test_a_refused_move_with_no_other_known_return_relays_the_own_refusal(
        replay_wire, monkeypatch, stream):
    """AGORA's falsifier on HAL#750: with no other pooled account known to come back
    (every other one dead, say), the wire invents no reset and raises nothing; A's
    own refusal goes back unchanged."""
    from lib.hal import wire_hold
    early, late = _resets()
    original = _limit_until(late)
    monkeypatch.setattr(wire_hold, "pooled_reset", lambda *args: None)
    head, body = _refused_move(replay_wire, monkeypatch, stream, original, _limit_until(early))
    assert head + b"\r\n\r\n" + body == original


def _limit_passes(monkeypatch, *tokens):
    """A ``during_move`` in which each token's limit passes: its mark in this process
    goes, and the ledger's cooling for it, written when it answered, reads as over."""
    from lib.hal import account_ledger
    digests = {account_ledger.credential_digest(token) for token in tokens}
    real = account_ledger.cooling_of

    def cooling_of(entry, now):
        if (entry or {}).get("token", {}).get("digest") in digests:
            return None
        return real(entry, now)

    def passes():
        for token in tokens:
            account_switch._EXHAUSTED.pop(spend_guard.account_key(token), None)
        monkeypatch.setattr(account_ledger, "cooling_of", cooling_of)
    return passes


def _reset_named(head):
    for line in head.split(b"\r\n"):
        name, _, value = line.partition(b":")
        if name.strip().lower() == b"anthropic-ratelimit-unified-reset":
            return int(value)
    return None


@pytest.mark.parametrize("stream", [False, True])
def test_a_replacement_that_comes_back_during_the_move_is_retried_at_once(
        replay_wire, monkeypatch, stream):
    """HAL#748 item 18: B's limit passes while a slow fallback attempt runs, and the
    fallback refuses the turn. A's own refusal would keep Claude Code waiting for A's
    reset although B is free, so the 429 names the present, and the retry Claude Code
    then sends is switched to B."""
    import time
    from lib.hal import wire_hold
    early, late = _resets()
    before = int(time.time())
    head, body = _refused_move(replay_wire, monkeypatch, stream, _limit_until(late),
                               _limit_until(early), during_move=_limit_passes(monkeypatch, B))
    assert head.startswith(b"HTTP/1.1 429")
    assert before <= _reset_named(head) <= int(time.time())
    assert b"Retry-After: 1\r\n" in head
    assert wire_hold.RETURNED_MESSAGE.encode() in body and b"own limit" not in body
    assert _last_row(replay_wire)["account_replay"]["outcome"] == "retry_now"
    replay_wire.fake.answers[B] = OK
    retry = _session_turn(stream)
    assert helpers._exchange(replay_wire, retry) == OK
    assert posts(replay_wire)[-1] == retry.replace(A.encode(), B.encode())


@pytest.mark.parametrize("stream", [False, True])
def test_when_the_own_limit_passes_too_its_own_refusal_goes_back(
        replay_wire, monkeypatch, stream):
    """Astra on HAL#748 item 18: when A's own limit has also passed, A's refusal already
    names a reset that is over, so it goes back byte for byte; nothing is synthesised."""
    early, late = _resets()
    original = _limit_until(late)
    head, body = _refused_move(replay_wire, monkeypatch, stream, original, _limit_until(early),
                               during_move=_limit_passes(monkeypatch, A, B))
    assert head + b"\r\n\r\n" + body == original


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("still", ["sibling_limit", "spend_mark", "ledger_cooling",
                                   "dead", "switching_off"])
def test_a_replacement_the_switcher_would_not_use_never_reads_as_returned(
        replay_wire, monkeypatch, stream, still):
    """Astra on HAL#748 item 18: B's own limit passes, but one reason to pass over B
    remains, so the client is never told to retry at once. The ledger case keeps only
    B's cooling, the one spent mark the switcher does not read; it then gets the
    existing pause naming B's reset. Every other case relays A's own refusal."""
    from lib.hal import account_ledger, wire_hold
    early, late = _resets()
    key = spend_guard.account_key(B)
    real_health = account_ledger.token_health_of

    def remains():
        if still == "sibling_limit":
            account_switch._ORGS["fake-sibling-of-B"] = account_switch._ORGS[key]
            account_switch._EXHAUSTED["fake-sibling-of-B"] = late
        elif still == "spend_mark":
            monkeypatch.setitem(spend_guard._SPENT, key, late)
        elif still == "dead":
            monkeypatch.setattr(account_ledger, "token_health_of", lambda entry, token:
                                "dead" if token == B else real_health(entry, token))
        elif still == "switching_off":
            monkeypatch.setattr(account_switch, "switching_on", lambda: False)

    passes = (lambda: account_switch._EXHAUSTED.pop(key)) if still == "ledger_cooling" \
        else _limit_passes(monkeypatch, B)
    original = _limit_until(late)
    head, body = _refused_move(replay_wire, monkeypatch, stream, original, _limit_until(early),
                               during_move=lambda: (passes(), remains()))
    assert wire_hold.RETURNED_MESSAGE.encode() not in body
    if still == "ledger_cooling":
        assert _reset_named(head) == early
        assert _last_row(replay_wire)["account_replay"]["outcome"] == "paused_until_reset"
    else:
        assert head + b"\r\n\r\n" + body == original


@pytest.mark.parametrize("stream", [False, True])
def test_an_account_offered_only_for_paid_overage_never_reads_as_returned(
        replay_wire, monkeypatch, strict_spend, stream):
    """With paid overage opted in, the switcher offers B for overage once B's own
    limit has passed while its subscription stays spent. Overage is not capacity
    coming back, so the client is not told to retry at once; B's spend mark and A's
    limit name the same reset, so A's own refusal goes back."""
    from lib.hal import wire_hold
    strict_spend(opt_in=["anthropic-overage"])
    early, late = _resets()
    passes, offered = _limit_passes(monkeypatch, B), []

    def overage_only():
        passes()
        monkeypatch.setitem(spend_guard._SPENT, spend_guard.account_key(B), late)
        offered.append(account_switch.choose(
            "POST", "/v1/messages", [("Authorization", "Bearer " + A)]))

    original = _limit_until(late)
    head, body = _refused_move(replay_wire, monkeypatch, stream, original, _limit_until(early),
                               during_move=overage_only)
    assert offered == [("B", B, account_switch.OVERAGE_RANK)]
    assert wire_hold.RETURNED_MESSAGE.encode() not in body
    assert head + b"\r\n\r\n" + body == original


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("organisation", ["known", "never_looked_up"])
def test_a_pool_account_added_during_the_move(replay_wire, monkeypatch, stream, organisation):
    """Astra on HAL#748 item 18: a fresh credential C joins the pool while the fallback
    is tried, and B stays spent. When C's organisation is already known, and is its
    own, the switcher would pick C, so the client retries at once. When it was never
    looked up, the check does not look it up (Codex on HAL#770: this answer never
    waits on the network), C does not count, and the pause names B's earlier reset
    as before. Nothing is recorded that would hold back the retry's own lookup."""
    from lib.hal import wire_hold
    early, late = _resets()
    key = spend_guard.account_key(C)

    def joins():
        monkeypatch.setattr(account_switch, "_read_pool", lambda: [("B", B), ("C", C)])
        if organisation == "known":
            account_switch._ORGS[key] = "fake-org-of-C"

    head, body = _refused_move(replay_wire, monkeypatch, stream, _limit_until(late),
                               _limit_until(early), during_move=joins)
    returned = wire_hold.RETURNED_MESSAGE.encode() in body
    assert returned is (organisation == "known")
    assert not [raw for _, raw in replay_wire.fake.requests
                if raw.startswith(b"GET /v1/models") and C.encode() in raw]
    if not returned:
        assert _reset_named(head) == early
        assert key not in account_switch._ORG_RETRY


@pytest.mark.parametrize("stream", [False, True])
def test_a_ledger_cooling_on_a_pooled_sibling_keeps_the_account_spent(
        replay_wire, monkeypatch, stream):
    """Codex on HAL#770: B's own limit passes during the move, but another wire has
    recorded B2, a pooled token of B's organisation, as spent in the ledger. Accounts
    in one organisation share limits, so B has not come back; B2's reset is A's, so
    A's own refusal goes back."""
    from lib.hal import account_ledger, wire_hold
    early, late = _resets()
    other = "fake-account-B2-token"
    for token in (B, other):
        account_switch._ORGS[spend_guard.account_key(token)] = "fake-org-of-B"
    real_accounts = account_ledger.LedgerView.accounts

    def accounts(view):
        found = real_accounts(view)
        found["B2-seen-by-another-wire"] = {
            "token": {"digest": account_ledger.credential_digest(other)},
            "cooling": {"until": late, "reason": "spent"}}
        return found

    monkeypatch.setattr(account_ledger.LedgerView, "accounts", accounts)
    original = _limit_until(late)
    head, body = _refused_move(replay_wire, monkeypatch, stream, original, _limit_until(early),
                               pool=(("B", B), ("B2", other)),
                               during_move=_limit_passes(monkeypatch, B))
    assert wire_hold.RETURNED_MESSAGE.encode() not in body
    assert head + b"\r\n\r\n" + body == original


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mark", ["w4_uncertain", "w4_accepted"])
def test_no_retry_at_once_after_a_fallback_send_that_may_have_been_accepted(
        replay_wire, monkeypatch, stream, mark):
    """Astra on HAL#748 item 18: once request bytes may have reached a target, a quick
    retry could repeat a turn the target is answering, so B's return is not announced
    and A's own refusal goes back."""
    early, late = _resets()
    original = _limit_until(late)
    head, body = _refused_move(replay_wire, monkeypatch, stream, original, _limit_until(early),
                               during_move=_limit_passes(monkeypatch, B), mark=mark)
    assert head + b"\r\n\r\n" + body == original


@pytest.mark.parametrize("other", ["free", "spent"])
def test_a_replacement_refused_at_the_send_is_never_followed_by_another_choice(
        replay_wire, monkeypatch, other):
    """HAL#748 item 19, left as it is (Astra: option (a)). The spend guard refuses B
    between the switcher's choice and the send, the fallback refuses the turn, and A's
    own refusal goes back unchanged, whether C is free or spent. B is never presented,
    and nothing chooses again after the refusal."""
    from lib.hal import wire_cascade
    monkeypatch.setattr(account_switch, "_read_pool", lambda: [("B", B), ("C", C)])
    if other == "spent":
        account_switch._EXHAUSTED[spend_guard.account_key(C)] = _resets()[1]
    monkeypatch.setattr(wire_cascade, "configured_targets", lambda: [object()])
    monkeypatch.setattr(wire_cascade, "move", lambda *args: False)
    refused, chosen_after = [], []
    real_choose = account_switch.choose

    def choose(*args, **kwargs):
        if refused:
            chosen_after.append(1)
        return real_choose(*args, **kwargs)

    def guard(method, target, headers):
        if spend_guard.anthropic_credential(headers)[1] == spend_guard.account_key(B):
            refused.append(1)
            return spend_guard.Decision(False, spend_guard.OVERAGE, "spent in between",
                                        status=429)
        return None

    monkeypatch.setattr(account_switch, "choose", choose)
    monkeypatch.setattr(spend_guard, "check_anthropic_request", guard)
    raw = _session_turn(False)
    assert helpers._exchange(replay_wire, raw) == LIMIT
    assert posts(replay_wire) == [raw]
    assert refused == [1] and chosen_after == []


@pytest.mark.parametrize("stream", [False, True])
def test_a_limit_on_a_credential_that_is_no_subscription_is_relayed_unchanged(
        replay_wire, monkeypatch, stream):
    """AGORA's falsifier on HAL#750 (second review): a credential the spend guard does
    not know as a subscription leaves no limit mark, so the turn is never replayed or
    moved, nothing raises, and its own refusal goes back unchanged."""
    from lib.hal import wire_cascade
    early, late = _resets()
    unknown = "unknown-credential-token"
    original = _limit_until(late)
    replay_wire.fake.answers.update({unknown: original, B: _limit_until(early)})
    monkeypatch.setattr(account_switch, "_read_pool", lambda: [("B", B)])
    monkeypatch.setattr(wire_cascade, "configured_targets", lambda: [object()])
    monkeypatch.setattr(wire_cascade, "move", lambda *args: pytest.fail("moved"))
    turn = dict(helpers.TURN, stream=True) if stream else helpers.TURN
    raw = request(payload=turn).replace(A.encode(), unknown.encode()).replace(
        b"X-Unchanged:", b"x-claude-code-session-id: fake-session\r\nX-Unchanged:")
    assert helpers._exchange(replay_wire, raw) == original
    assert len(posts(replay_wire)) == 1


@pytest.mark.parametrize("path,payload", [
    ("/v1/code/sessions", helpers.TURN),
    ("/v1/messages/count_tokens", helpers.TURN),
    ("/v1/messages", helpers.QUOTA),
])
def test_non_turn_never_replayed(replay_wire, path, payload):
    assert helpers._exchange(replay_wire, request(path, payload)) == LIMIT
    assert len(posts(replay_wire)) == 1


def test_interim_bytes_prevent_replay(replay_wire):
    interim = b"HTTP/1.1 103 Early Hints\r\n\r\n"
    replay_wire.fake.answers[A] = interim + LIMIT
    tls = helpers._tunnel(replay_wire.port, helpers.API, replay_wire.client)
    try:
        tls.sendall(request())
        reader = helpers._Reader(tls)
        assert helpers._read_response(reader) == interim
        assert helpers._read_response(reader) == LIMIT
    finally:
        tls.close()
    assert len(posts(replay_wire)) == 1


@pytest.mark.parametrize("message,expected", [
    ("Invalid signature in thinking block", "original"),
    ("max_tokens is invalid", "replacement"),
])
def test_replay_400_thinking_rule(replay_wire, message, expected):
    bad = reply(400, json.dumps({"error": {"message": message}}).encode())
    replay_wire.fake.answers[B] = bad
    payload = dict(helpers.TURN, messages=[
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "keep this", "signature": "fake-signature"},
            {"type": "tool_use", "id": "fake-tool", "name": "lookup", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "fake-tool",
                                       "content": "done"}]}])
    raw = request(payload=payload)
    assert helpers._exchange(replay_wire, raw) == (LIMIT if expected == "original" else bad)
    assert posts(replay_wire) == [raw, raw.replace(A.encode(), B.encode())]
    assert replay_row(replay_wire)["outcome"] == (
        "thinking_signature_rejected" if expected == "original" else "relayed")


def test_off_switch_stops_replay(replay_wire):
    account_switch.set_switching(False)
    assert helpers._exchange(replay_wire, request()) == LIMIT
    assert len(posts(replay_wire)) == 1


def test_only_one_replay_even_with_more_capacity(replay_wire):
    other_limit = LIMIT.replace(b"original limit", b"another! limit")
    replay_wire.fake.answers[B] = other_limit
    assert helpers._exchange(replay_wire, request()) == other_limit
    assert [helpers._bearer_of(raw) for raw in posts(replay_wire)] == [A, B]
    assert account_switch.exhausted_until(spend_guard.account_key(B)) is not None


def test_headerless_429_not_replayed(replay_wire):
    plain = reply(429, b"slow down", b"Retry-After: 5\r\n")
    replay_wire.fake.answers[A] = plain
    assert helpers._exchange(replay_wire, request()) == plain
    assert len(posts(replay_wire)) == 1


def test_copy_overflow_not_replayed(replay_wire, monkeypatch):
    original = wire_proxy._Copy
    monkeypatch.setattr(wire_proxy, "_Copy", lambda: original(limit=32))
    assert helpers._exchange(replay_wire, request()) == LIMIT
    assert len(posts(replay_wire)) == 1


def test_a_replacement_the_spend_guard_refuses_is_never_presented(replay_wire, monkeypatch):
    # Defence in depth: choose() should never offer an account the spend guard
    # refuses, but if it ever does, the replay must not present that bearer.
    # Only the first choose() call (before sending) keeps A; the second (after
    # the 429) offers B, which the guard refuses.
    offers = iter([None, ("B", B, account_switch.SUBSCRIPTION_RANK)])
    monkeypatch.setattr(account_switch, "choose", lambda *args, **kwargs: next(offers, None))

    def guard(method, target, headers):
        if spend_guard.anthropic_credential(headers)[1] == spend_guard.account_key(B):
            return spend_guard.Decision(False, spend_guard.OVERAGE, "test refusal", status=429)
        return None

    monkeypatch.setattr(spend_guard, "check_anthropic_request", guard)
    raw = request()
    assert helpers._exchange(replay_wire, raw) == LIMIT
    assert posts(replay_wire) == [raw]


def _cascade_replay_head():
    raw = request().split(b'\r\n\r\n')[0]
    return wire_proxy._parse_head(raw + b'\r\nx-claude-code-session-id: fake-session\r\n\r\n')


def test_replay_send_blocks_w4_until_answer_head(monkeypatch):
    import time
    from types import SimpleNamespace
    from lib.hal import wire_cascade, wire_hold
    request_head = _cascade_replay_head()
    body = wire_proxy._Copy()
    body(json.dumps(helpers.TURN).encode())
    ex = wire_proxy._Exchange('POST', '/v1/messages', request_head.headers,
        time.monotonic(), route=wire_proxy.RouteClass.INFERENCE, body=body, wire_body=body)
    replay = wire_proxy._Exchange('POST', '/v1/messages', request_head.headers,
        ex.started, route=wire_proxy.RouteClass.INFERENCE)
    moves = []
    def send(*args):
        assert not wire_hold.offer_cascade(proxy, None, ex, 'unsent')
    conn = SimpleNamespace(sock=SimpleNamespace(sendall=send, close=lambda: None,
        shutdown=lambda *_: None))
    head = wire_proxy._parse_head(b'HTTP/1.1 429 Limit\r\nContent-Length: 0\r\n\r\n')
    proxy = SimpleNamespace(pool=SimpleNamespace(take=lambda: conn),
        _answer=lambda *args, **kwargs: head,
        _replay_response_decision=lambda *args: 'original',
        _w4_attempts=wire_cascade.Attempts())
    monkeypatch.setattr(wire_cascade, 'move', lambda *args: moves.append(1) or False)
    assert not wire_proxy.WireProxy._replay_sent(proxy, None, request_head,
        replay, ex, ('B', B, account_switch.SUBSCRIPTION_RANK))
    assert not wire_hold.offer_cascade(proxy, None, ex, 'unsent')
    assert moves == [1]


@pytest.mark.parametrize('failure', ['no_head', 'raises'])
def test_a_replay_that_got_no_answer_is_closed_then_unmarked(monkeypatch, failure):
    """Codex on HAL#747: a replay that got no answer left its turn's mark behind for the
    life of the process. The mark must still stand while the closure is written."""
    import time
    from types import SimpleNamespace
    from lib.hal import wire_cascade, wire_hold
    request_head = _cascade_replay_head()
    body = wire_proxy._Copy()
    body(json.dumps(helpers.TURN).encode())
    ex = wire_proxy._Exchange('POST', '/v1/messages', request_head.headers,
        time.monotonic(), route=wire_proxy.RouteClass.INFERENCE, body=body, wire_body=body)
    replay = wire_proxy._Exchange('POST', '/v1/messages', request_head.headers,
        ex.started, route=wire_proxy.RouteClass.INFERENCE)
    key = wire_hold.turn_key(ex.headers, helpers.TURN)
    attempts, at_closure, real_close = wire_cascade.Attempts(), [], wire_cascade.Attempts.close

    def watched_close(self, closing):
        at_closure.append(dict(self._outstanding))
        real_close(self, closing)

    def answer(*args, **kwargs):
        if failure == 'raises':
            raise ConnectionResetError('acceptance unknown')

    monkeypatch.setattr(wire_cascade.Attempts, 'close', watched_close)
    conn = SimpleNamespace(sock=SimpleNamespace(sendall=lambda *_: None, close=lambda: None,
        shutdown=lambda *_: None))
    proxy = SimpleNamespace(pool=SimpleNamespace(take=lambda: conn), _answer=answer,
                            _w4_attempts=attempts)
    assert not wire_proxy.WireProxy._replay_sent(proxy, None, request_head,
        replay, ex, ('B', B, account_switch.SUBSCRIPTION_RANK))
    assert at_closure == [{key: 1}]
    assert attempts._outstanding == {} and attempts.closed(key)


@pytest.mark.parametrize('served', [False, True])
def test_locally_refused_replacement_offers_cascade(monkeypatch, served):
    from types import SimpleNamespace
    from lib.hal import wire_cascade, wire_hold
    from tests.test_wire_proxy import _CascadeSocket
    request_head = _cascade_replay_head()
    body = wire_proxy._Copy()
    body(json.dumps(helpers.TURN).encode())
    ex = wire_proxy._Exchange('POST', '/v1/messages', request_head.headers, 0,
                              route=wire_proxy.RouteClass.INFERENCE, body=body, wire_body=body)
    socket = _CascadeSocket()
    calls = []
    refusal = spend_guard.Decision(False, spend_guard.OVERAGE, 'spent', status=429)
    monkeypatch.setattr(spend_guard, 'check_anthropic_request', lambda *args: refusal)
    monkeypatch.setattr(wire_hold, 'exhaustion', lambda *args: (True, None))

    def fake_move(proxy, client, exchange):
        calls.append(exchange)
        if served:
            client.sock.sendall(b'TARGET')
        return served

    monkeypatch.setattr(wire_cascade, 'move', fake_move)
    proxy = SimpleNamespace(pool=SimpleNamespace(take=lambda: pytest.fail('replacement sent')),
                            _w4_attempts=wire_cascade.Attempts())
    choice = ('B', B, account_switch.SUBSCRIPTION_RANK)
    assert wire_proxy.WireProxy._replay_with(proxy, SimpleNamespace(sock=socket), request_head,
                                             ex, False, choice) is served
    assert calls == [ex]
    assert socket.sent == ([b'TARGET'] if served else [])


def test_locally_refused_replacement_keeps_original_once_without_target(monkeypatch):
    from types import SimpleNamespace
    from lib.hal import wire_cascade
    from tests.test_wire_proxy import _CascadeSocket
    request_head = _cascade_replay_head()
    socket, rows, moves = _CascadeSocket(), [], []
    refusal = spend_guard.Decision(False, spend_guard.OVERAGE, 'spent', status=429)
    monkeypatch.setattr(spend_guard, 'check_anthropic_request', lambda *args: refusal)
    monkeypatch.setattr(wire_cascade, 'move', lambda *args: moves.append(1) or False)
    response = wire_proxy._parse_head(LIMIT.split(b'\r\n\r\n')[0] + b'\r\n\r\n')

    def answer(client, conn, request, ex):
        ex.route = wire_proxy.RouteClass.INFERENCE
        ex.body(json.dumps(helpers.TURN).encode())
        ex.wire_body(json.dumps(helpers.TURN).encode())
        return response

    def replay(client, request, ex, response):
        choice = ('B', B, account_switch.SUBSCRIPTION_RANK)
        return wire_proxy.WireProxy._replay_with(proxy, client, request, ex, False, choice)

    def relay(client, conn, ex, response):
        client.sock.sendall(LIMIT)
        ex.completed = True
        return False

    proxy = SimpleNamespace(_send_for_answer=answer, _replay_rejected=replay,
                            _relay_response=relay, _record=rows.append,
                            _w4_attempts=wire_cascade.Attempts())
    wire_proxy.WireProxy._relay_exchange(proxy, wire_proxy._Stream(socket), None, request_head)
    assert moves == [1] and socket.sent == [LIMIT] and len(rows) == 1


def test_locally_forbidden_replacement_never_offers_cascade(monkeypatch):
    from types import SimpleNamespace
    from lib.hal import wire_cascade
    from tests.test_wire_proxy import _CascadeSocket
    request_head = _cascade_replay_head()
    body = wire_proxy._Copy()
    body(json.dumps(helpers.TURN).encode())
    ex = wire_proxy._Exchange('POST', '/v1/messages', request_head.headers, 0,
                              route=wire_proxy.RouteClass.INFERENCE, body=body, wire_body=body)
    refusal = spend_guard.Decision(False, spend_guard.OVERAGE, 'forbidden', status=403)
    monkeypatch.setattr(spend_guard, 'check_anthropic_request', lambda *args: refusal)
    monkeypatch.setattr(wire_cascade, 'move', lambda *args: pytest.fail('403 moved'))
    proxy = SimpleNamespace(pool=SimpleNamespace(take=lambda: pytest.fail('replacement sent')))
    client = SimpleNamespace(sock=_CascadeSocket())
    choice = ('B', B, account_switch.SUBSCRIPTION_RANK)
    assert wire_proxy.WireProxy._replay_with(proxy, client, request_head,
                                             ex, False, choice) is False
    assert client.sock.sent == []


def test_ineligible_locally_refused_replacement_never_moves(monkeypatch):
    from types import SimpleNamespace
    from lib.hal import wire_cascade
    from tests.test_wire_proxy import _CascadeSocket
    request_head = _cascade_replay_head()
    body = wire_proxy._Copy()
    turn = dict(helpers.TURN, messages=[
        {'role': 'assistant', 'content': 'earlier'}, *helpers.TURN['messages']])
    body(json.dumps(turn).encode())
    ex = wire_proxy._Exchange('POST', '/v1/messages', request_head.headers, 0,
                              route=wire_proxy.RouteClass.INFERENCE, body=body, wire_body=body)
    refusal = spend_guard.Decision(False, spend_guard.OVERAGE, 'spent', status=429)
    monkeypatch.setattr(spend_guard, 'check_anthropic_request', lambda *args: refusal)
    monkeypatch.setattr(wire_cascade, 'move', lambda *args: pytest.fail('ineligible move'))
    proxy = SimpleNamespace(pool=SimpleNamespace(take=lambda: pytest.fail('replacement sent')))
    client = SimpleNamespace(sock=_CascadeSocket())
    choice = ('B', B, account_switch.SUBSCRIPTION_RANK)
    assert wire_proxy.WireProxy._replay_with(proxy, client, request_head,
                                             ex, False, choice) is False
    assert client.sock.sent == []


@pytest.mark.parametrize('phase', ['before_send', 'after_send'])
def test_replacement_timeout_never_offers_cascade(monkeypatch, phase):
    import socket
    from types import SimpleNamespace
    from lib.hal import wire_cascade, wire_hold
    request_head = _cascade_replay_head()
    body = wire_proxy._Copy()
    body(json.dumps(helpers.TURN).encode())
    ex = wire_proxy._Exchange('POST', '/v1/messages', request_head.headers, 0,
                              route=wire_proxy.RouteClass.INFERENCE, body=body, wire_body=body)
    monkeypatch.setattr(spend_guard, 'check_anthropic_request', lambda *args: None)
    monkeypatch.setattr(wire_cascade, 'move', lambda *args: pytest.fail('ambiguous send moved'))

    sent = []

    def timeout(*_args, **_kwargs):
        raise socket.timeout('acceptance unknown')

    connection = SimpleNamespace(sock=SimpleNamespace(sendall=sent.append,
        shutdown=lambda *_: None, close=lambda: None))
    take = timeout if phase == 'before_send' else lambda: connection
    proxy = SimpleNamespace(pool=SimpleNamespace(take=take), _answer=timeout,
                            _w4_attempts=wire_cascade.Attempts())
    choice = ('B', B, account_switch.SUBSCRIPTION_RANK)
    assert wire_proxy.WireProxy._replay_with(proxy, None, request_head,
                                             ex, False, choice) is False
    assert bool(sent) is (phase == 'after_send')
    key = wire_hold.turn_key(ex.headers, json.loads(ex.body.value()))
    assert key is not None and proxy._w4_attempts.reserve(key) is (phase == 'before_send')


def _replay_original(client, conn, request, ex):
    ex.route = wire_proxy.RouteClass.INFERENCE
    ex.body(json.dumps(helpers.TURN).encode())
    ex.wire_body(json.dumps(helpers.TURN).encode())
    return wire_proxy._parse_head(LIMIT.split(b'\r\n\r\n')[0] + b'\r\n\r\n')


def test_target_body_failure_never_appends_original_refusal(monkeypatch):
    from types import SimpleNamespace
    from lib.hal import wire_cascade, wire_hold
    from tests.test_wire_proxy import _CascadeSocket
    request_head = _cascade_replay_head()
    socket, rows = _CascadeSocket(), []
    original = wire_proxy._parse_head(LIMIT.split(b'\r\n\r\n')[0] + b'\r\n\r\n')
    replacement = wire_proxy._parse_head(OK.split(b'\r\n\r\n')[0] + b'\r\n\r\n')
    connection = SimpleNamespace(sock=SimpleNamespace(sendall=lambda *_: None,
        close=lambda: None, shutdown=lambda *_: None))
    def move(proxy, client, ex):
        client.sock.sendall(replacement.raw)
        client.sock.sendall(b'partial')
        ex.client_started = True
        raise OSError('target body broke')
    monkeypatch.setattr(wire_cascade, 'move', move)
    monkeypatch.setattr(wire_hold, 'exhaustion', lambda *args: (True, None))
    monkeypatch.setattr(spend_guard, 'check_anthropic_request', lambda *args: None)
    proxy = SimpleNamespace(pool=SimpleNamespace(take=lambda: connection),
        _send_for_answer=_replay_original, _answer=lambda *args, **kwargs: replacement,
        _replay_response_decision=lambda *args: 'move', _record=rows.append,
        _relay_response=lambda client, conn, ex, head: client.sock.sendall(LIMIT),
        _w4_attempts=wire_cascade.Attempts())
    proxy._replay_rejected = lambda client, request, ex, head: wire_proxy.WireProxy._replay_with(
        proxy, client, request, ex, False, ('B', B, account_switch.SUBSCRIPTION_RANK))
    with pytest.raises(OSError, match='target body broke'):
        wire_proxy.WireProxy._relay_exchange(proxy, wire_proxy._Stream(socket), None, request_head)
    assert socket.sent == [replacement.raw, b'partial'] and len(rows) == 1


def test_a_pause_marks_its_exchange_only_once_the_answer_has_gone():
    """HAL#748 item 21: no later code reads ``client_started`` after a pause, so this
    pins the state the pause leaves behind. Once its bytes have gone the exchange is
    started, answered 429 and complete; if they never went, none of that is claimed."""
    import time
    from types import SimpleNamespace
    sent = []
    ex = wire_proxy._Exchange('POST', '/v1/messages', (), time.monotonic())
    wire_proxy.WireProxy._answer_paused(SimpleNamespace(sock=SimpleNamespace(sendall=sent.append)),
                                        ex, b'paused')
    assert sent == [b'paused']
    assert (ex.client_started, ex.status, ex.completed) == (True, 429, True)

    def gone(data):
        raise OSError('the client went away')

    lost = wire_proxy._Exchange('POST', '/v1/messages', (), time.monotonic())
    with pytest.raises(OSError):
        wire_proxy.WireProxy._answer_paused(SimpleNamespace(sock=SimpleNamespace(sendall=gone)),
                                            lost, b'paused')
    assert (lost.client_started, lost.status, lost.completed) == (False, None, False)
