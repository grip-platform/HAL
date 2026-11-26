"""HAL's own Anthropic calls on an API key, with prompt caching, behind a daily cap.

Covers ``lib/hal/anthropic_key_route.py``, the daily cap in ``lib/hal/spend_guard.py``,
the native provider's one API-key send, and the two seams that ask the route
before signing with CCH: ``CCHSubscription.chat`` and Happi's ``chat``.

Nothing here can reach Anthropic or read a real key. Every test that could send
replaces the two shared objects a real call crosses: ``urllib.request.urlopen``
(the only way the native provider sends) and ``subprocess.run`` (the only way the
keychain is probed or read). tests/conftest.py's permissive policy is replaced by
a strict one and the guard's clock is frozen.

The ``lib.hal`` modules are fetched when each test runs, never at import: some
test modules evict and re-import ``lib.hal`` at collection
(``tests/test_cost_router.py``), and a copy imported here earlier would not be the
copy the code under test uses. Credentials are obvious fakes.
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

API_KEY = "anthropic-api-key"
OVERAGE = "anthropic-overage"
KEY_ITEM = "hal-anthropic"
MOMENT = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
LAST_SECOND = datetime(2026, 10, 8, 23, 59, 59, tzinfo=timezone.utc)
NEXT_DAY = datetime(2026, 10, 9, 0, 0, 0, tzinfo=timezone.utc)
MODEL = "claude-opus-5-5"
UNPRICED = "claude-unpriced-test-model"
FAKE_KEY = "sk-ant-api03-fake-route-test-key"
FAKE_OAUTH = "sk-ant-oat01-fake-route-test-token"
#: claude-opus-5-5 in config/model_prices.yaml, US dollars per million tokens:
#: input 4, output 20, cache read 0.20, cache write 5. So this block costs
#: 1000*4 + 500*20 + 10000*0.20 + 2000*5 = 26000 millionths of a dollar.
USAGE = {"input_tokens": 1000, "output_tokens": 500,
         "cache_read_input_tokens": 10000, "cache_creation_input_tokens": 2000}
COST = Decimal("0.02600000")
SYSTEM = "You are HAL's own reviewer."
TOOLS = [{"name": "Read", "description": "read a file", "input_schema": {"type": "object"}}]
EPHEMERAL = {"type": "ephemeral"}
_MODULES = {"route": "lib.hal.anthropic_key_route", "guard": "lib.hal.spend_guard",
            "native": "lib.hal.providers.anthropic_native", "subscription": "lib.hal.subscription",
            "happi": "lib.hal.happi.providers.anthropic", "envelope": "lib.hal.happi.envelope",
            "prompt_cache": "lib.hal.prompt_cache", "types": "lib.hal.types",
            "tool_cli": "lib.hal.tool_cli_integration"}


# --------------------------------------------------------------------------- #
# Fakes and fixtures.
# --------------------------------------------------------------------------- #
@pytest.fixture
def hal():
    """The ``lib.hal`` modules the code under test will use, fetched now."""
    return SimpleNamespace(**{key: importlib.import_module(name) for key, name in _MODULES.items()})


@pytest.fixture
def clock(hal, monkeypatch):
    """Freeze the guard's clock (every day and price date follows it)."""
    def at(moment: datetime) -> None:
        monkeypatch.setattr(hal.guard, "_clock", lambda: moment)
    at(MOMENT)
    return at


@pytest.fixture
def policy(hal, tmp_path, monkeypatch, clock):
    """A strict policy the test writes: by default opted in with NO cap (closed)."""
    path = tmp_path / "spend_guard.json"

    def write(opt_in=(API_KEY,), caps=None) -> None:
        body = {"paid_opt_in": list(opt_in)}
        if caps is not None:
            body["daily_caps_usd"] = caps
        path.write_text(json.dumps(body), encoding="utf-8")

    monkeypatch.setenv(hal.guard.CONFIG_ENV, str(path))
    write()
    return write


class Keychain:
    """Stands in for ``security``: answers the probe by name, hands out a fake key."""

    def __init__(self) -> None:
        self.has_item = True
        self.calls: list = []

    def run(self, argv, **_kwargs):
        argv = list(argv)
        self.calls.append(argv)
        if argv[:2] != ["security", "find-generic-password"]:
            raise AssertionError(f"unexpected subprocess in a route test: {argv[:2]}")
        found = self.has_item and argv[-1] == KEY_ITEM
        stdout = FAKE_KEY if found and "-w" in argv else ""
        return subprocess.CompletedProcess(argv, 0 if found else 44, stdout=stdout, stderr="")

    def reads(self) -> int:
        """How many times the key itself was read (``-w``)."""
        return sum("-w" in argv for argv in self.calls)


@pytest.fixture
def keychain(monkeypatch):
    """A fake ``security`` holding the item, in place of conftest's empty one."""
    fake = Keychain()
    monkeypatch.setattr(subprocess, "run", fake.run)
    return fake


class _Reply:
    headers = {"content-type": "application/json"}

    def __init__(self, payload: dict) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class FakeAnthropic:
    """Stands in for api.anthropic.com behind ``urllib.request.urlopen``."""

    def __init__(self) -> None:
        self.requests: list = []  # (url, {lower-case header: value}, body)

    def urlopen(self, request, timeout=None):
        headers = {k.lower(): v for k, v in request.header_items()}
        body = json.loads(request.data.decode("utf-8"))
        self.requests.append((request.full_url, headers, body))
        return _Reply({"id": "msg_fake", "type": "message", "role": "assistant",
                       "model": body["model"], "stop_reason": "end_turn",
                       "content": [{"type": "text", "text": "from the key"}],
                       "usage": dict(USAGE)})


@pytest.fixture
def anthropic(monkeypatch):
    fake = FakeAnthropic()
    monkeypatch.setattr(urllib.request, "urlopen", fake.urlopen)
    return fake


@pytest.fixture
def route_open(policy, keychain, anthropic):
    """Every condition of the selection rule holds: opt-in, a $1 cap, the item."""
    policy(caps={API_KEY: "1"})
    return SimpleNamespace(keychain=keychain, anthropic=anthropic)


class FakeSigner:
    """Stands in for CCHSigner: records each call and answers from 'CCH'."""

    def __init__(self, hal) -> None:
        self._types = hal.types
        self.calls: list = []

    def call(self, prompt="", **kwargs):
        self.calls.append(dict(kwargs, prompt=prompt))
        return self._types.ChatResponse(text="from CCH", model=kwargs["model"], provider="anthropic",
                                        usage=self._types.TokenUsage(prompt_tokens=1,
                                                                     completion_tokens=1))


def _cch(hal, signer: FakeSigner):
    """A real CCHSubscription with its signer replaced. CCHSigner's module reads
    HAL_HOME when imported, so it is never constructed here."""
    cls = hal.subscription.CCHSubscription
    sub = cls.__new__(cls)
    sub._token = FAKE_OAUTH
    sub._signer = signer
    return sub


def _audit_rows(hal) -> list:
    path = hal.guard.audit_path()
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _today(hal) -> dict:
    days = json.loads(hal.guard.daily_ledger_path().read_text(encoding="utf-8"))["days"]
    return days["2026-10-08"][API_KEY]


# --------------------------------------------------------------------------- #
# The selection rule.
# --------------------------------------------------------------------------- #
def test_the_committed_policy_keeps_the_route_closed(hal, monkeypatch, keychain, clock):
    """Shipped off: not opted in, and a $0 cap even if it were. The only other
    capped source is TypeSafe's prepaid decision key, at $0.10 (2026-10-17,
    lowered from the $1 set on 2026-10-10)."""
    monkeypatch.delenv(hal.guard.CONFIG_ENV, raising=False)
    committed = hal.guard.load_policy()
    typesafe = hal.guard.TYPESAFE_API_KEY
    assert committed.daily_caps_usd == {API_KEY: Decimal(0), typesafe: Decimal("0.1")}
    assert committed.paid_opt_in == frozenset({OVERAGE, typesafe})
    verdict = hal.route.select(MODEL)
    assert verdict.allowed is False
    assert "paid_opt_in" in verdict.reason
    assert keychain.calls == [], "a closed route does not even probe the keychain"


def test_the_route_opens_only_when_every_condition_holds(hal, policy, keychain):
    policy(caps={API_KEY: "1"})
    verdict = hal.route.select(MODEL)
    assert verdict.allowed is True, verdict.reason
    assert keychain.calls == [["security", "find-generic-password", "-s", KEY_ITEM]]
    assert keychain.reads() == 0, "deciding read the key"


CLOSED_BY_POLICY = {
    "not opted in": {"opt_in": (), "caps": {API_KEY: "1"}},
    "no cap": {"caps": None},
    "a $0 cap": {"caps": {API_KEY: 0}},
    "a malformed cap": {"caps": {API_KEY: "five"}},
    "a negative cap": {"caps": {API_KEY: -1}},
    "a cap for another source only": {"caps": {OVERAGE: "1"}},
}


@pytest.mark.parametrize("setup", CLOSED_BY_POLICY.values(), ids=CLOSED_BY_POLICY.keys())
def test_any_failing_policy_condition_closes_the_route(hal, policy, keychain, setup):
    policy(**setup)
    verdict = hal.route.select(MODEL)
    assert verdict.allowed is False
    assert "spend guard" in verdict.reason


def test_a_missing_keychain_item_closes_the_route(hal, policy, keychain):
    policy(caps={API_KEY: "1"})
    keychain.has_item = False
    verdict = hal.route.select(MODEL)
    assert verdict.allowed is False
    assert KEY_ITEM in verdict.reason


def test_a_keychain_probe_that_times_out_counts_as_no_item(hal, policy, monkeypatch):
    """AGORA on HAL#704 asked whether a probe timeout escapes the probe.
    ``TimeoutExpired`` is a ``SubprocessError``, so the probe answers "no item"
    and the route stays closed instead of raising into the caller."""
    policy(caps={API_KEY: "1"})

    def hangs(argv, *_args, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs.get("timeout"))

    monkeypatch.setattr(subprocess, "run", hangs)
    verdict = hal.route.select(MODEL)
    assert verdict.allowed is False
    assert KEY_ITEM in verdict.reason


def test_a_model_without_a_price_closes_the_route(hal, policy, keychain):
    policy(caps={API_KEY: "1"})
    verdict = hal.route.select(UNPRICED)
    assert verdict.allowed is False
    assert "model_prices.yaml" in verdict.reason


def test_spend_at_the_cap_closes_the_route(hal, policy, keychain):
    policy(caps={API_KEY: "0.02"})
    assert hal.route.select(MODEL).allowed is True       # the control: room left
    hal.guard.record_spend(API_KEY, Decimal("0.02"))
    assert hal.route.select(MODEL).allowed is False


# --------------------------------------------------------------------------- #
# On the key when open; CCH, exactly as before, when not.
# --------------------------------------------------------------------------- #
def test_an_open_route_sends_hal_s_own_call_on_the_key_not_cch(hal, route_open):
    signer = FakeSigner(hal)
    response = _cch(hal, signer).chat("hi", model=MODEL)
    assert response.text == "from the key"
    assert signer.calls == [], "CCH signed a call while the route was open"
    [(url, headers, body)] = route_open.anthropic.requests
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["x-api-key"] == FAKE_KEY
    assert "authorization" not in headers
    assert body["model"] == MODEL
    assert route_open.keychain.reads() == 1


@pytest.mark.parametrize("closed_by", ["opt-in", "cap", "keychain", "price"])
def test_a_closed_route_signs_with_cch_exactly_as_before(hal, policy, keychain, anthropic,
                                                         closed_by):
    """Any one condition failing sends the call to CCH untouched: nothing on the
    key, the key never read, and CCH given exactly the caller's arguments."""
    policy(opt_in=() if closed_by == "opt-in" else (API_KEY,),
           caps={API_KEY: 0 if closed_by == "cap" else "1"})
    keychain.has_item = closed_by != "keychain"
    model = UNPRICED if closed_by == "price" else MODEL
    signer = FakeSigner(hal)
    tools = [dict(tool) for tool in TOOLS]
    response = _cch(hal, signer).chat("hi", model=model, system=SYSTEM, tools=tools, max_tokens=64)
    assert response.text == "from CCH"
    assert anthropic.requests == []
    assert keychain.reads() == 0
    assert signer.calls == [{"prompt": "hi", "oauth_token": FAKE_OAUTH, "model": model,
                             "messages": None, "system": SYSTEM, "max_tokens": 64,
                             "tools": TOOLS}]


def test_a_send_refused_after_selection_falls_back_to_cch(hal, policy, keychain, anthropic,
                                                           monkeypatch):
    """Another process can spend the last of the cap between the decision and the
    send. The send's own check refuses before anything leaves HAL, and the call
    is signed with CCH instead."""
    policy(caps={API_KEY: "0.02"})
    hal.guard.record_spend(API_KEY, Decimal("0.02"))
    monkeypatch.setattr(hal.route, "select",
                        lambda *_a, **_k: hal.guard.Decision(True, API_KEY, "stale"))
    signer = FakeSigner(hal)
    response = _cch(hal, signer).chat("hi", model=MODEL)
    assert response.text == "from CCH"
    assert anthropic.requests == []
    assert keychain.reads() == 0


def test_the_send_itself_refuses_a_caller_that_skips_the_rule(hal, policy, keychain, anthropic):
    """The native provider's one API-key send asks the guard again, so no caller
    can send past the cap by going around the selection rule."""
    policy(caps={API_KEY: 0})
    with pytest.raises(hal.native.AnthropicSpendRefused, match="spend guard"):
        hal.route.api_provider().chat("hi", model=MODEL, prefer="api_key")
    assert anthropic.requests == []
    assert keychain.reads() == 0


# --------------------------------------------------------------------------- #
# Prompt caching: every API-key request, never the CCH path.
# --------------------------------------------------------------------------- #
def test_every_api_key_request_carries_the_cache_markers(hal, route_open):
    tools = [dict(tool) for tool in TOOLS]
    _cch(hal, FakeSigner(hal)).chat("hi", model=MODEL, system=SYSTEM, tools=tools)
    [(_url, _headers, body)] = route_open.anthropic.requests
    assert hal.prompt_cache.has_cache_markers(body)
    assert body["system"] == [{"type": "text", "text": SYSTEM, "cache_control": EPHEMERAL}]
    assert body["tools"][-1]["cache_control"] == EPHEMERAL
    assert tools == TOOLS, "the caller's tool schemas were marked in place"


def test_the_cch_path_never_gets_cache_markers(hal, policy, keychain, anthropic):
    policy(caps={API_KEY: 0})
    signer = FakeSigner(hal)
    _cch(hal, signer).chat("hi", model=MODEL, system=SYSTEM, tools=[dict(t) for t in TOOLS])
    [call] = signer.calls
    assert call["system"] == SYSTEM
    assert all("cache_control" not in tool for tool in call["tools"])


# --------------------------------------------------------------------------- #
# The daily cap.
# --------------------------------------------------------------------------- #
def test_the_call_that_crosses_the_cap_is_the_last_one_on_the_key(hal, policy, keychain,
                                                                  anthropic):
    """Each call costs $0.026. A $0.05 cap allows two (0 -> 0.026 -> 0.052); the
    third and fourth are signed with CCH, and the day gets exactly one audit row."""
    policy(caps={API_KEY: "0.05"})
    sub = _cch(hal, FakeSigner(hal))
    texts = [sub.chat("hi", model=MODEL).text for _ in range(4)]
    assert texts == ["from the key", "from the key", "from CCH", "from CCH"]
    assert len(anthropic.requests) == 2
    assert _today(hal)["spent_usd"] == "0.05200000"
    [row] = _audit_rows(hal)
    assert set(row) == {"ts", "event", "source", "day", "spent_usd", "cap_usd", "calls"}
    assert (row["event"], row["source"], row["day"]) == ("daily_cap_reached", API_KEY, "2026-10-08")
    assert (row["spent_usd"], row["cap_usd"], row["calls"]) == ("0.05200000", "0.05000000", 2)


def test_spend_accumulates_per_utc_day_and_resets_at_midnight(hal, policy, clock):
    policy(caps={API_KEY: "0.05"})
    clock(LAST_SECOND)
    assert hal.guard.record_spend(API_KEY, COST) == COST
    assert hal.guard.record_spend(API_KEY, COST) == Decimal("0.05200000")
    assert hal.guard.decide_within_cap(API_KEY).allowed is False
    clock(NEXT_DAY)
    verdict = hal.guard.decide_within_cap(API_KEY)
    assert verdict.allowed is True, verdict.reason
    assert "$0.00000000 of $0.05 on 2026-10-09" in verdict.reason
    assert _today(hal)["spent_usd"] == "0.05200000"      # yesterday is kept, not counted


@pytest.mark.parametrize("usage", [None, {"output_tokens": 3}], ids=["no-usage", "no-input-count"])
def test_a_response_that_cannot_be_priced_refuses_the_rest_of_the_day(hal, policy, keychain,
                                                                      usage):
    policy(caps={API_KEY: "1"})
    assert hal.route.settle(MODEL, usage) is None
    verdict = hal.route.select(MODEL)
    assert verdict.allowed is False
    assert "could not be priced" in verdict.reason
    [row] = _audit_rows(hal)
    assert row["event"] == "spend_unknown"


def test_an_unreadable_spend_ledger_refuses_the_key_and_nothing_else(hal, policy):
    policy(opt_in=(API_KEY, OVERAGE), caps={API_KEY: "1"})
    ledger = hal.guard.daily_ledger_path()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("{not json", encoding="utf-8")
    assert hal.guard.decide_within_cap(API_KEY).allowed is False
    assert hal.guard.decide_paid(API_KEY).allowed is True
    assert hal.guard.decide_paid(OVERAGE).allowed is True


def test_a_malformed_cap_refuses_only_the_capped_decision(hal, policy):
    policy(opt_in=(API_KEY, OVERAGE), caps={API_KEY: "five"})
    verdict = hal.guard.decide_within_cap(API_KEY)
    assert verdict.allowed is False and "daily_caps_usd" in verdict.reason
    assert hal.guard.decide_paid(OVERAGE).allowed is True


#: A child process: load the guard by its path, as GRIP does, and be refused 25 times.
#: The module is registered before it runs, as loading by path requires: its
#: dataclasses look their own module up in sys.modules (on main as well).
_REFUSED_CHILD = (
    "import importlib.util, sys\n"
    "spec = importlib.util.spec_from_file_location('spend_guard_child', sys.argv[1])\n"
    "guard = importlib.util.module_from_spec(spec)\n"
    "sys.modules[spec.name] = guard\n"
    "spec.loader.exec_module(guard)\n"
    "for _ in range(25):\n"
    "    assert not guard.decide_within_cap('anthropic-api-key').allowed\n"
)


def test_processes_refused_at_the_cap_together_write_one_audit_row(hal, tmp_path):
    """AGORA on HAL#704 suspected duplicate audit rows when processes reach the cap
    together. The "already noted" check and its write happen under the ledger's
    cross-process lock, so four processes refused 25 times each write one row.
    Parent and children run on the real clock, so they count the same UTC day."""
    path = tmp_path / "spend_guard.json"
    path.write_text(json.dumps({"paid_opt_in": [API_KEY], "daily_caps_usd": {API_KEY: "0.01"}}),
                    encoding="utf-8")
    hal.guard.record_spend(API_KEY, Decimal("0.02"))
    env = dict(os.environ, **{hal.guard.CONFIG_ENV: str(path)})
    children = [subprocess.Popen([sys.executable, "-c", _REFUSED_CHILD, hal.guard.__file__], env=env)
                for _ in range(4)]
    assert [child.wait(timeout=60) for child in children] == [0, 0, 0, 0]
    reached = [row for row in _audit_rows(hal) if row["event"] == "daily_cap_reached"]
    assert len(reached) == 1, reached


def test_the_wire_s_guard_never_reads_the_cap(hal, policy):
    """The wire and the gateway relay a session's own sign-in through
    check_anthropic_request. The cap is HAL's own and must not change that answer."""
    policy(caps={API_KEY: 0})
    assert hal.guard.check_anthropic_request(
        "POST", "/v1/messages", [("x-api-key", FAKE_KEY)]) is None


# --------------------------------------------------------------------------- #
# Pricing, from HAL's own price table.
# --------------------------------------------------------------------------- #
def test_a_known_usage_block_prices_to_a_known_figure(hal, policy):
    policy(caps={API_KEY: "1"})
    assert hal.route.settle(MODEL, USAGE) == COST
    entry = _today(hal)
    assert entry["spent_usd"] == "0.02600000"
    assert (entry["input_tokens"], entry["output_tokens"], entry["cache_read_tokens"],
            entry["cache_write_tokens"], entry["calls"]) == (1000, 500, 10000, 2000, 1)


def test_cache_reads_are_priced_apart_from_input(hal, policy):
    policy(caps={API_KEY: "1"})
    as_input = hal.route.settle(MODEL, {"input_tokens": 10000, "output_tokens": 0})
    as_cache_read = hal.route.settle(MODEL, {"input_tokens": 0, "output_tokens": 0,
                                             "cache_read_input_tokens": 10000})
    assert as_input == Decimal("0.04000000")        # 10,000 at $4 per million
    assert as_cache_read == Decimal("0.00200000")   # 10,000 at $0.20 per million


# --------------------------------------------------------------------------- #
# The other seams: Happi's chat and the hal REPL's tool mode.
# --------------------------------------------------------------------------- #
def _envelope(hal, **auth):
    payload = {"v": "happi/1.0", "id": "req-route", "cmd": "chat", "args": ["hello"],
               "flags": {"provider": MODEL, "max_tokens": 64, "system": SYSTEM}}
    if auth:
        payload["auth"] = auth
    return hal.envelope.HappiEnvelope.parse(payload)


@pytest.fixture
def cch_stream(hal, monkeypatch):
    """Records what Happi hands to CCH; the real signer is never reached."""
    seen: list = []

    def fake(envelope, token):
        seen.append((envelope.id, token))
        yield hal.happi.emit_completed(envelope.id, usage={})

    monkeypatch.setattr(hal.happi, "_stream_via_cch", fake)
    return seen


def test_happi_sends_hal_s_own_chat_on_the_key_when_the_route_is_open(hal, route_open, cch_stream,
                                                                       monkeypatch):
    monkeypatch.setenv(hal.happi.SUBSCRIPTION_ENV, FAKE_OAUTH)
    events = [json.loads(line) for line in hal.happi.stream_anthropic(_envelope(hal))]
    assert [event["type"] for event in events] == ["delta", "completed"]
    assert events[0]["text"] == "from the key"
    assert cch_stream == []
    [(_url, headers, body)] = route_open.anthropic.requests
    assert headers["x-api-key"] == FAKE_KEY
    assert body["system"][-1]["cache_control"] == EPHEMERAL


@pytest.mark.parametrize("auth, cap", [
    ({}, 0),                                          # HAL chose the credential; closed
    ({"scheme": "apikey", "token": FAKE_OAUTH}, "1"),  # open, but the caller chose it
], ids=["route-closed", "caller-chose-the-credential"])
def test_happi_keeps_cch_otherwise(hal, policy, keychain, anthropic, cch_stream, monkeypatch,
                                   auth, cap):
    monkeypatch.setenv(hal.happi.SUBSCRIPTION_ENV, FAKE_OAUTH)
    policy(caps={API_KEY: cap})
    list(hal.happi.stream_anthropic(_envelope(hal, **auth)))
    assert cch_stream == [("req-route", FAKE_OAUTH)]
    assert anthropic.requests == []
    assert keychain.reads() == 0


def test_the_hal_repl_without_cch_uses_the_key_only_through_the_route(hal, route_open,
                                                                      monkeypatch):
    monkeypatch.setattr(hal.subscription.CCHSubscription, "from_env",
                        classmethod(lambda cls, **_k: None))
    chat_fn, label = hal.tool_cli.detect_tool_chat_fn(MODEL)
    assert label == "api_key"
    raw = chat_fn(messages=[{"role": "user", "content": "hi"}], tools=[dict(t) for t in TOOLS])
    assert raw["content"][0]["text"] == "from the key"
    [(_url, headers, body)] = route_open.anthropic.requests
    assert headers["x-api-key"] == FAKE_KEY
    assert hal.prompt_cache.has_cache_markers(body)


# --------------------------------------------------------------------------- #
# Status: python -m lib.hal.spend_guard caps
# --------------------------------------------------------------------------- #
def test_caps_prints_each_source_s_cap_spend_and_opt_in(hal, policy, capsys):
    policy(caps={API_KEY: "2.50"})
    hal.guard.record_spend(API_KEY, COST, usage={"cache_read_tokens": 10000})
    assert hal.guard.main(["caps"]) == 0
    lines = capsys.readouterr().out.splitlines()
    key_line = next(line for line in lines if line.startswith(API_KEY))
    assert key_line.split() == [API_KEY, "yes", "$2.50", "$0.0260", "1", "10000"]
    overage = next(line for line in lines if line.startswith(OVERAGE))
    assert overage.split()[1:3] == ["no", "none"] and "not metered by HAL" in overage


def test_caps_on_the_committed_policy_shows_the_key_off_at_zero(hal, monkeypatch, capsys, clock):
    monkeypatch.delenv(hal.guard.CONFIG_ENV, raising=False)
    assert hal.guard.main(["caps"]) == 0
    key_line = next(line for line in capsys.readouterr().out.splitlines()
                    if line.startswith(API_KEY))
    assert key_line.split()[:5] == [API_KEY, "no", "$0", "$0.0000", "0"]


def test_caps_says_a_cap_on_a_source_hal_does_not_meter_is_not_enforced(hal, policy, capsys):
    """AGORA on HAL#704: a cap written for a source HAL does not meter (overage is
    billed on the wire, where HAL counts nothing) must not read as if it applied."""
    policy(opt_in=(API_KEY, OVERAGE), caps={API_KEY: "1", OVERAGE: "10", "made-up-source": "3"})
    assert hal.guard.main(["caps"]) == 0
    lines = capsys.readouterr().out.splitlines()
    for name, cap in ((OVERAGE, "$10"), ("made-up-source", "$3")):
        line = next(line for line in lines if line.startswith(name))
        assert cap in line and line.endswith("not metered by HAL; this cap is not enforced"), line
    key_line = next(line for line in lines if line.startswith(API_KEY))
    assert "not enforced" not in key_line


def test_caps_refuses_anything_else(hal, capsys):
    assert hal.guard.main([]) == 2
    assert hal.guard.main(["caps", "extra"]) == 2
    assert "usage" in capsys.readouterr().err
