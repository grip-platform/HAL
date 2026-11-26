"""Tests for lib.hal.cache_loss -- the wire-side cache-loss ledger.

Part A: hashing, usage shaping, row building, capped append, and the one
fail-open glue function the wire calls. Part B: the "what changed" classifier
(:func:`find_drops`) and the per-model utilization fit (:func:`fit_utilization`),
each with synthetic rows carrying known, hand-computed answers.
"""
from __future__ import annotations

import json

import pytest

from lib.hal import cache_loss
from lib.hal.account_ledger import Reading, Window

FAKE_SECRET = "sk-ant-FAKE-WIRE-TEST-SECRET-0000-DO-NOT-USE"


# --------------------------------------------------------------------------- #
# Part A.1 -- hashing: a change detector, never a content store.
# --------------------------------------------------------------------------- #
def test_hash_hex_is_none_for_none():
    assert cache_loss.hash_hex(None) is None


def test_hash_hex_is_order_independent_and_content_sensitive():
    same_a = cache_loss.hash_hex({"b": 1, "a": 2})
    same_b = cache_loss.hash_hex({"a": 2, "b": 1})
    different = cache_loss.hash_hex({"a": 2, "b": 3})
    assert same_a == same_b
    assert same_a != different


def _body(**fields) -> bytes:
    return json.dumps(fields).encode("utf-8")


def test_request_hashes_reads_system_tools_and_first_message():
    body = _body(system="you are grip", tools=[{"name": "Read"}],
                messages=[{"role": "user", "content": "hi"},
                          {"role": "assistant", "content": "hello"}])
    hashes = cache_loss.request_hashes((), body)
    assert hashes["system_hash"] == cache_loss.hash_hex("you are grip")
    assert hashes["tools_hash"] == cache_loss.hash_hex([{"name": "Read"}])
    assert hashes["first_message_hash"] == cache_loss.hash_hex(
        {"role": "user", "content": "hi"})


def test_request_hashes_are_none_when_fields_are_absent():
    hashes = cache_loss.request_hashes((), _body(messages=[]))
    assert hashes == {"system_hash": None, "tools_hash": None, "first_message_hash": None}


def test_request_hashes_change_only_for_the_field_that_changed():
    one = cache_loss.request_hashes((), _body(system="a", messages=[{"role": "user", "content": "x"}]))
    two = cache_loss.request_hashes((), _body(system="b", messages=[{"role": "user", "content": "x"}]))
    assert one["system_hash"] != two["system_hash"]
    assert one["first_message_hash"] == two["first_message_hash"]


# --------------------------------------------------------------------------- #
# Part A.2 -- usage shaping and window reads.
# --------------------------------------------------------------------------- #
def test_usage_fields_sums_the_two_cache_write_lifetimes():
    raw = {"input_tokens": 10, "output_tokens": 2, "cache_read_tokens": 100,
          "cache_write_tokens": 5, "cache_write_1h_tokens": 3}
    assert cache_loss.usage_fields(raw) == {
        "input_tokens": 10, "output_tokens": 2, "cache_read_tokens": 100,
        "cache_creation_tokens": 8}


@pytest.mark.parametrize("raw", [None, {}])
def test_usage_fields_is_none_when_nothing_was_measured(raw):
    assert cache_loss.usage_fields(raw) is None


def test_window_utilization_reads_the_named_window_only():
    reading = Reading(windows=(("five_hour", Window(utilization=42.5)),))
    assert cache_loss.window_utilization(reading, "five_hour") == 42.5
    assert cache_loss.window_utilization(reading, "seven_day") is None


# --------------------------------------------------------------------------- #
# Part A.3 -- building and appending one row.
# --------------------------------------------------------------------------- #
def test_build_row_carries_exactly_the_declared_fields():
    row = cache_loss.build_row(
        ts="t", session_id="s", account_label="a", org_id="o", model="m",
        hashes={"system_hash": "sh", "tools_hash": "th", "first_message_hash": "fh"},
        usage={"input_tokens": 1, "output_tokens": 2, "cache_read_tokens": 3,
              "cache_creation_tokens": 4},
        five_hour_utilization=1.0, seven_day_utilization=2.0,
        continues_thread=True, account_switched=False, prompt_id="p")
    assert set(row) == {"ts", "session_id", "account_label", "org_id", "model",
                        "system_hash", "tools_hash", "first_message_hash", "usage",
                        "five_hour_utilization", "seven_day_utilization",
                        "continues_thread", "account_switched", "prompt_id"}
    assert (row["ts"], row["session_id"], row["account_label"]) == ("t", "s", "a")
    assert (row["continues_thread"], row["account_switched"]) == (True, False)
    assert row["prompt_id"] == "p"


def test_continues_thread_reads_only_the_thread_continuation_id():
    """HAL#709: a delta names a server-held thread; a full request does not.
    A first request carries the key with a null value, which is not a
    continuation; a body that is not a JSON object answers None, never a guess."""
    full = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).encode()
    delta = json.dumps({"thread": {"type": "conversation", "previous_message_id": "msg_fake"},
                        "messages": [{"role": "user", "content": [{"type": "tool_result"}]}]}).encode()
    first = json.dumps({"thread": {"type": "conversation", "previous_message_id": None}}).encode()
    assert cache_loss.continues_thread(full) is False
    assert cache_loss.continues_thread(delta) is True
    assert cache_loss.continues_thread(first) is False
    assert cache_loss.continues_thread(None) is None
    assert cache_loss.continues_thread(b"") is None
    assert cache_loss.continues_thread(b"not json") is None
    assert cache_loss.continues_thread(b"[1, 2]") is None


def test_append_row_writes_one_canonical_json_line_per_call(tmp_path):
    home = tmp_path / "home"
    cache_loss.append_row({"a": 1}, home=home)
    cache_loss.append_row({"a": 2}, home=home)
    lines = cache_loss.log_path(home).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [{"a": 1}, {"a": 2}]


def test_append_row_rotates_to_one_prior_file_past_the_cap(tmp_path):
    home = tmp_path / "home"
    row = {"a": "x" * 60}
    for _ in range(4):
        cache_loss.append_row(row, home=home, max_bytes=100)
    path = cache_loss.log_path(home)
    rotated = path.with_name(path.name + ".1")
    assert rotated.exists()
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[-1]) == row


def test_append_row_never_raises_on_unserialisable_content(tmp_path):
    class NotSerialisable:
        pass

    # Must not raise: a disk/encoding failure here must never break the
    # request whose response it describes.
    cache_loss.append_row({"a": NotSerialisable()}, home=tmp_path / "home")


def _reply_headers(*, five_hour=None, seven_day=None, org=None):
    headers = []
    if org is not None:
        headers.append(("anthropic-organization-id", org))
    if five_hour is not None:
        headers.append(("anthropic-ratelimit-unified-5h-utilization", str(five_hour)))
    if seven_day is not None:
        headers.append(("anthropic-ratelimit-unified-7d-utilization", str(seven_day)))
    return tuple(headers)


def test_record_response_writes_a_complete_row(tmp_path):
    home = tmp_path / "home"
    body = _body(system="sys", tools=[], messages=[{"role": "user", "content": "hi"}])
    cache_loss.record_response(
        session_id="sess-1", account_label="bob", model="claude-opus-5-5",
        request_headers=(), request_body=body,
        response_headers=_reply_headers(five_hour=12.34, seven_day=5.0, org="org_abc"),
        usage={"input_tokens": 10, "output_tokens": 2, "cache_read_tokens": 100,
              "cache_write_tokens": 3},
        home=home, ts="2026-10-08T00:00:00+00:00")
    rows = [json.loads(line) for line in cache_loss.log_path(home).read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert (row["session_id"], row["account_label"], row["model"]) == (
        "sess-1", "bob", "claude-opus-5-5")
    assert row["org_id"] == "org_abc"
    assert (row["five_hour_utilization"], row["seven_day_utilization"]) == (12.34, 5.0)
    assert row["usage"] == {"input_tokens": 10, "output_tokens": 2,
                            "cache_read_tokens": 100, "cache_creation_tokens": 3}
    assert row["system_hash"] == cache_loss.hash_hex("sys")


def test_record_response_is_fail_open_on_a_broken_usage_argument(tmp_path):
    home = tmp_path / "home"
    # usage.get would raise AttributeError on a non-dict; record_response must
    # swallow it rather than letting it reach the wire's request path.
    cache_loss.record_response(
        session_id=None, account_label="unpooled", model=None,
        request_headers=(), request_body=b"not json", response_headers=(),
        usage="not a dict", home=home)
    assert not cache_loss.log_path(home).exists()


def test_no_secret_or_body_reaches_the_log(tmp_path):
    """The brief's own test: feed a body containing a fake key and prove it
    is absent from the written log, along with every other header value."""
    home = tmp_path / "home"
    body = _body(system=f"remember this key: {FAKE_SECRET}",
                tools=[{"name": "Shell", "description": FAKE_SECRET}],
                messages=[{"role": "user", "content": FAKE_SECRET}])
    cache_loss.record_response(
        session_id="sess-1", account_label="bob", model="m",
        request_headers=(("authorization", f"Bearer {FAKE_SECRET}"),),
        request_body=body,
        response_headers=(("anthropic-organization-id", "org_1"),
                          ("x-request-id", FAKE_SECRET)),
        usage={"input_tokens": 1, "output_tokens": 1, "cache_read_tokens": 0,
              "cache_write_tokens": 0},
        home=home, ts="2026-10-08T00:00:00+00:00")
    text = cache_loss.log_path(home).read_text(encoding="utf-8")
    assert FAKE_SECRET not in text
    row = json.loads(text.strip())
    assert set(row) == {"ts", "session_id", "account_label", "org_id", "model",
                        "system_hash", "tools_hash", "first_message_hash", "usage",
                        "five_hour_utilization", "seven_day_utilization",
                        "continues_thread", "account_switched", "prompt_id"}
    # The two flags are derived or passed, never copied from the body.
    assert (row["continues_thread"], row["account_switched"]) == (False, None)
    assert row["prompt_id"] is None  # no prompt-id header was sent


@pytest.mark.parametrize("header, expected", [
    ((("x-claude-code-prompt-id", "9c2f0a2e-1111-4000-8000-000000000042"),),
     "9c2f0a2e-1111-4000-8000-000000000042"),
    ((), None),
    ((("x-claude-code-prompt-id", "has spaces in it"),), None),
    ((("x-claude-code-prompt-id", "x" * 129),), None),
])
def test_record_response_keeps_the_prompt_id_only_in_the_safe_set(tmp_path, header, expected):
    """Claude Code 2.1.283's x-claude-code-prompt-id groups the requests that
    serve one user prompt. It is copied when it fits the session-id safe set
    (128 characters of [A-Za-z0-9._:-]) and is null otherwise, so a malformed
    value is dropped, never written."""
    home = tmp_path / "home"
    cache_loss.record_response(
        session_id="sess-1", account_label="bob", model="m",
        request_headers=header, request_body=_body(), response_headers=(),
        usage=None, home=home, ts="2026-10-17T00:00:00+00:00")
    row = json.loads(cache_loss.log_path(home).read_text(encoding="utf-8").strip())
    assert row["prompt_id"] == expected


def test_record_response_keeps_a_passed_account_switched_flag(tmp_path):
    home = tmp_path / "home"
    cache_loss.record_response(
        session_id="sess-1", account_label="other", model="m",
        request_headers=(), request_body=_body(), response_headers=(),
        usage=None, home=home, ts="2026-10-10T00:00:00+00:00", account_switched=True)
    row = json.loads(cache_loss.log_path(home).read_text(encoding="utf-8").strip())
    assert row["account_switched"] is True
    assert row["continues_thread"] is False


def test_record_response_still_writes_a_row_for_a_body_that_is_not_an_object(tmp_path):
    """The council's falsifier on HAL#713: a syntactically valid body that is
    not a JSON object must not cost the row; the flag is None and the rest of
    the row is written as before."""
    home = tmp_path / "home"
    cache_loss.record_response(
        session_id="sess-1", account_label="bob", model="m",
        request_headers=(), request_body=b"[1, 2]", response_headers=(),
        usage=None, home=home, ts="2026-10-10T00:00:00+00:00")
    rows = cache_loss.log_path(home).read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["continues_thread"] is None
    assert row["account_switched"] is None
    assert row["session_id"] == "sess-1"


# --------------------------------------------------------------------------- #
# Part B.1 -- what changed before a sharp cache-read fall.
# --------------------------------------------------------------------------- #
def _row(session_id, ts, *, input_tokens, cache_read, cache_creation=0, output=1,
        system_hash="s1", tools_hash="t1", first_message_hash="m1", org_id="o1",
        model="model-a"):
    return {"session_id": session_id, "ts": ts, "system_hash": system_hash,
           "tools_hash": tools_hash, "first_message_hash": first_message_hash,
           "org_id": org_id, "model": model,
           "usage": {"input_tokens": input_tokens, "output_tokens": output,
                    "cache_read_tokens": cache_read,
                    "cache_creation_tokens": cache_creation}}


#: A prompt holding at ~10,000 tokens, cache fraction collapsing from 0.9 to 0.05.
_PREV = _row("s", "2026-10-08T00:00:00+00:00", input_tokens=1000, cache_read=9000)
_NOW = _row("s", "2026-10-08T00:01:00+00:00", input_tokens=500, cache_read=500,
           cache_creation=9000)


def test_find_drops_flags_a_sharp_fall_at_a_similar_context_size():
    drops = cache_loss.find_drops([_PREV, _NOW])
    assert len(drops) == 1
    assert drops[0].session_id == "s"
    assert (drops[0].prev_cache_read_tokens, drops[0].cache_read_tokens) == (9000, 500)


@pytest.mark.parametrize("field,expected_cause", [
    ("system_hash", "system"),
    ("tools_hash", "tools"),
    ("first_message_hash", "first_message"),
    ("org_id", "organisation"),
    ("model", "model"),
])
def test_find_drops_names_each_changed_field(field, expected_cause):
    now = dict(_NOW, **{field: "CHANGED"})
    now["usage"] = dict(_NOW["usage"])  # the dict() merge above shares the nested dict
    drops = cache_loss.find_drops([_PREV, now])
    assert drops[0].changed == (expected_cause,)
    assert drops[0].cause == expected_cause


def test_find_drops_priority_order_when_several_fields_changed():
    """system is named first per the brief's own ordering, even when tools and
    the model also changed on the same turn."""
    now = dict(_NOW, system_hash="CHANGED", tools_hash="CHANGED", model="CHANGED")
    drops = cache_loss.find_drops([_PREV, now])
    assert drops[0].changed == ("system", "tools", "model")
    assert drops[0].cause == "system, tools, model"


def test_find_drops_cause_is_server_side_eviction_when_nothing_changed():
    drops = cache_loss.find_drops([_PREV, _NOW])
    assert drops[0].changed == ()
    assert "server-side eviction" in drops[0].cause


def test_find_drops_ignores_a_different_session():
    now = dict(_NOW, session_id="other-session")
    assert cache_loss.find_drops([_PREV, now]) == []


def test_find_drops_ignores_a_gap_longer_than_five_minutes():
    now = dict(_NOW, ts="2026-10-08T00:10:01+00:00")  # 601s after _PREV
    assert cache_loss.find_drops([_PREV, now]) == []


def test_find_drops_ignores_a_resized_prompt():
    shrunk = _row("s", "2026-10-08T00:01:00+00:00", input_tokens=500, cache_read=500)
    assert cache_loss.find_drops([_PREV, shrunk]) == []  # ratio ~0.1, outside the band


def test_find_drops_ignores_a_turn_whose_cache_was_already_low():
    low_cache_prev = _row("s", "2026-10-08T00:00:00+00:00", input_tokens=9000, cache_read=1000)
    low_cache_now = _row("s", "2026-10-08T00:01:00+00:00", input_tokens=9500, cache_read=500)
    assert cache_loss.find_drops([low_cache_prev, low_cache_now]) == []


def test_find_drops_sorts_rows_itself():
    """Fed in reverse order, the real previous turn is still found."""
    assert cache_loss.find_drops([_NOW, _PREV]) == cache_loss.find_drops([_PREV, _NOW])


def test_find_drops_looks_past_an_error_response_with_no_usage():
    """A 429 or 529 between two turns carries no usage; the later turn must
    still be compared with the last turn that measured a prompt (R0 of an AGORA
    concern on HAL#702)."""
    error = dict(_row("s", "2026-10-08T00:00:30+00:00", input_tokens=0, cache_read=0),
                 usage=None)
    drops = cache_loss.find_drops([_PREV, error, _NOW])
    assert [drop.prev_ts for drop in drops] == [_PREV["ts"]]


def test_report_renders_plain_text():
    assert "no sharp cache-read" in cache_loss.report([])
    text = cache_loss.report([_PREV, _NOW])
    assert "1 sharp cache-read fall" in text
    assert "session=s" in text
    assert "9000->500" in text


# --------------------------------------------------------------------------- #
# Part B.2 -- the utilization-step fit.
# --------------------------------------------------------------------------- #
def _util_row(account, model, minute, usage, five_hour):
    return {"account_label": account, "model": model,
           "ts": f"2026-10-08T00:{minute:02d}:00+00:00",
           "usage": usage, "five_hour_utilization": round(five_hour, 6)}


_TRUE_WEIGHTS = {"input_tokens": 0.0010, "output_tokens": 0.0100,
                 "cache_read_tokens": 0.0002, "cache_creation_tokens": 0.0015}
_USAGES = [
    {"input_tokens": 100, "output_tokens": 10, "cache_read_tokens": 500, "cache_creation_tokens": 50},
    {"input_tokens": 200, "output_tokens": 5, "cache_read_tokens": 1000, "cache_creation_tokens": 0},
    {"input_tokens": 50, "output_tokens": 20, "cache_read_tokens": 0, "cache_creation_tokens": 200},
    {"input_tokens": 300, "output_tokens": 0, "cache_read_tokens": 300, "cache_creation_tokens": 100},
    {"input_tokens": 10, "output_tokens": 50, "cache_read_tokens": 2000, "cache_creation_tokens": 10},
    {"input_tokens": 500, "output_tokens": 1, "cache_read_tokens": 10, "cache_creation_tokens": 500},
]


def _true_delta(usage: dict) -> float:
    return sum(_TRUE_WEIGHTS[name] * usage[name] for name in _TRUE_WEIGHTS)


def _series(account: str, model: str, usages, start: float = 10.0):
    """Rows whose five_hour_utilization is the exact running sum of each
    usage's TRUE linear cost -- unrounded, so the fit's recovery can be
    checked to high precision."""
    rows, util = [], start
    for minute, usage in enumerate(usages, start=1):
        util += _true_delta(usage)
        rows.append(_util_row(account, model, minute, usage, util))
    return rows


def test_fit_utilization_recovers_known_weights_with_near_zero_residual():
    rows = _series("acct-1", "model-x", _USAGES)
    fits = cache_loss.fit_utilization(rows)
    fit = fits["model-x"]
    assert fit.weights is not None, fit.note
    for name, true_value in _TRUE_WEIGHTS.items():
        assert fit.weights[name] == pytest.approx(true_value, abs=1e-6)
    assert fit.residual_rms < 1e-6
    assert fit.n_points == len(_USAGES) - 1  # the first row only sets the baseline


def test_fit_utilization_pools_across_accounts_for_the_same_model():
    rows = _series("acct-1", "model-x", _USAGES, start=10.0)
    rows += _series("acct-2", "model-x", _USAGES, start=70.0)
    fits = cache_loss.fit_utilization(rows, min_points=2 * (len(_USAGES) - 1))
    fit = fits["model-x"]
    assert fit.n_points == 2 * (len(_USAGES) - 1)
    assert fit.weights is not None, fit.note
    for name, true_value in _TRUE_WEIGHTS.items():
        assert fit.weights[name] == pytest.approx(true_value, abs=1e-6)


def test_fit_utilization_reports_too_few_points_plainly():
    rows = _series("acct-1", "model-x", _USAGES[:2])  # 1 step, floor is 5
    fits = cache_loss.fit_utilization(rows)
    fit = fits["model-x"]
    assert fit.weights is None
    assert "too few data points" in fit.note
    assert fit.n_points == 1


def test_fit_utilization_reports_degenerate_when_a_field_never_varies():
    usages = [{**u, "cache_creation_tokens": 0} for u in _USAGES]  # always zero
    rows = _series("acct-1", "model-y", usages)
    fits = cache_loss.fit_utilization(rows)
    fit = fits["model-y"]
    assert fit.weights is None
    assert "degenerate" in fit.note


def test_step_observations_sums_a_flat_stretch_into_one_step():
    """Several rows reporting the SAME utilization (below the header's
    two-decimal resolution) are summed into one step once it finally ticks."""
    flat_usage = {"input_tokens": 1, "output_tokens": 0, "cache_read_tokens": 0,
                 "cache_creation_tokens": 0}
    rows = [_util_row("a", "m", 1, flat_usage, 10.0),
           _util_row("a", "m", 2, flat_usage, 10.0),
           _util_row("a", "m", 3, flat_usage, 10.0),
           _util_row("a", "m", 4, flat_usage, 10.5)]
    steps = cache_loss.step_observations(rows)
    # Row 1 only sets the baseline (10.0); rows 2-4's usage sums into the step
    # that reveals the next tick.
    assert steps["m"] == [((3, 0, 0, 0), pytest.approx(0.5))]


def test_step_observations_discards_the_open_stretch_on_a_window_rollover():
    usage = {"input_tokens": 1, "output_tokens": 0, "cache_read_tokens": 0,
            "cache_creation_tokens": 0}
    rows = [_util_row("a", "m", 1, usage, 95.0),   # baseline; row 1's usage is discarded
           _util_row("a", "m", 2, usage, 97.0),   # a step: (usage*1, 2.0)
           _util_row("a", "m", 3, usage, 1.0),    # the window rolled over; discarded
           _util_row("a", "m", 4, usage, 3.0)]    # a fresh step: (usage*1, 2.0)
    steps = cache_loss.step_observations(rows)
    assert steps["m"] == [((1, 0, 0, 0), pytest.approx(2.0)),
                         ((1, 0, 0, 0), pytest.approx(2.0))]


def test_step_observations_never_carries_usage_across_an_unattributed_step():
    """A step that shows before any row has named a model cannot be
    attributed, so its usage is discarded with it, never folded into the next
    step (R0 of an AGORA concern on HAL#702)."""
    usage = {"input_tokens": 1, "output_tokens": 0, "cache_read_tokens": 0,
            "cache_creation_tokens": 0}
    rows = [_util_row("a", None, 1, usage, 10.0),   # baseline
           _util_row("a", None, 2, usage, 11.0),   # a step, no model yet: discarded
           _util_row("a", "m", 3, usage, 12.0)]    # a step for "m": only its own usage
    steps = cache_loss.step_observations(rows)
    assert steps["m"] == [((1, 0, 0, 0), pytest.approx(1.0))]


def test_format_fit_renders_a_refusal_and_a_result():
    refused = cache_loss.UtilizationFit(model="m", n_points=1, weights=None,
                                        residual_rms=None, note="too few data points (1 < 5)")
    assert cache_loss.format_fit(refused) == "m: too few data points (1 < 5)"
    fitted = cache_loss.UtilizationFit(model="m", n_points=10,
                                       weights={"input_tokens": 0.001}, residual_rms=0.0)
    assert cache_loss.format_fit(fitted) == "m: n=10 weights[input_tokens=0.001000] residual_rms=0.0000"


# --------------------------------------------------------------------------- #
# The command line.
# --------------------------------------------------------------------------- #
def _write_log(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_main_report_reads_a_named_log(tmp_path, capsys):
    log = tmp_path / "ledger.jsonl"
    _write_log(log, [_PREV, _NOW])
    assert cache_loss.main(["report", "--log", str(log)]) == 0
    out = capsys.readouterr().out
    assert "1 sharp cache-read fall" in out


def test_main_fit_reads_a_named_log(tmp_path, capsys):
    log = tmp_path / "ledger.jsonl"
    _write_log(log, _series("acct-1", "model-x", _USAGES))
    assert cache_loss.main(["fit", "--log", str(log)]) == 0
    out = capsys.readouterr().out
    assert "model-x: n=" in out


def test_main_report_on_an_empty_log(tmp_path, capsys):
    assert cache_loss.main(["report", "--log", str(tmp_path / "missing.jsonl")]) == 0
    assert "no sharp cache-read" in capsys.readouterr().out
