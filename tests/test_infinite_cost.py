"""InfiniteMachine v2, Step 4 -- cost truth (``lib/hal/infinite_cost.py``).

What these tests pin, and how each can fail:

* Usage is read from a non-streamed Messages body and from an SSE stream
  (``message_start``, then each ``message_delta``), however the stream is cut
  into chunks, and the gateway's relay is byte-for-byte unchanged while it
  watches.
* A list price comes only from ``config/model_prices.yaml``: a model with no
  row, a fast or priority request, a prompt above the priced tier, or a rate
  the vendor does not publish all give ``list_price_usd: null`` and a warning,
  never a guess.
* ``actual_spend_usd`` is null for subscription and unknown billing, never 0;
  missing usage is null, never 0.
* ``reconcile`` sums list prices per model and per day and counts what it
  could not price.

The bodies are real-shaped: the golden stream HAL's happi tests already use
(``tests/happi_golden/anthropic_hello.sse``) and the usage block Claude Code
2.1.289 reports (``tools/wire-matrix/fixtures/claude/S0/stdout.txt``). Rows are
written with a stand-in signer: these tests are about cost, and signing is
pinned in ``tests/test_infinite_thread.py``. Nothing reaches a provider.
"""
from __future__ import annotations

import io
import json
import logging
import random
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from tests._public_tree import require

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

yaml = pytest.importorskip("yaml")

from lib.hal import infinite_cost, infinite_thread  # noqa: E402
from lib.hal.gateway import UpstreamResponse, handle_proxy_request  # noqa: E402
from lib.hal.infinite_cost import (  # noqa: E402
    SseUsage,
    billing_from_headers,
    cost_fields,
    list_price,
    reconcile,
    usage_from_body,
    usage_from_sse,
)

DAY = date(2026, 11, 24)  # the newest row's effective date (Kimi K3.1); every row must be on or before it
GOLDEN = _REPO / "tests" / "happi_golden" / "anthropic_hello.sse"
OAUTH = "Bearer sk-ant-oat01-TESTONLY"
API_KEY = "sk-ant-api03-TESTONLY"
COLUMNS = ("input", "output", "cache_read", "cache_write", "cache_write_1h")

USAGE_BLOCK = {"input_tokens": 12, "cache_creation_input_tokens": 2048,
               "cache_read_input_tokens": 40960,
               "cache_creation": {"ephemeral_5m_input_tokens": 0,
                                  "ephemeral_1h_input_tokens": 2048},
               "output_tokens": 2, "service_tier": "standard"}
EXPECTED = {"input_tokens": 12, "output_tokens": 321, "cache_read_tokens": 40960,
            "cache_write_tokens": 0, "cache_write_1h_tokens": 2048}


def _event(kind: str, data: dict) -> bytes:
    return f"event: {kind}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


def _stream(*, between: bytes = b"", newline: bytes = b"\n") -> bytes:
    start = {"type": "message_start", "message": {
        "id": "msg_02", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": [], "stop_reason": None, "stop_sequence": None, "usage": USAGE_BLOCK}}
    text = {"type": "content_block_delta", "index": 0, "delta": {
        "type": "text_delta", "text": "the words message_delta and message_start in prose"}}
    delta = {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
             "usage": {"output_tokens": 321}}
    body = (_event("message_start", start) + _event("ping", {"type": "ping"}) + between
            + _event("content_block_delta", text) + _event("message_delta", delta)
            + _event("message_stop", {"type": "message_stop"}))
    return body.replace(b"\n", newline)


def _message_body() -> bytes:
    return json.dumps({"id": "msg_01", "type": "message", "role": "assistant",
                       "model": "claude-opus-5-5", "content": [{"type": "text", "text": "hi"}],
                       "stop_reason": "end_turn",
                       "usage": dict(USAGE_BLOCK, output_tokens=321)}).encode("utf-8")


def _fed_in_steps(data: bytes, rng: random.Random, largest: int):
    parser, pos = SseUsage(), 0
    while pos < len(data):
        step = rng.randint(1, largest)
        parser.feed(data[pos:pos + step])
        pos += step
    return parser.usage()


# --- usage --------------------------------------------------------------------------

def test_a_non_streamed_body_gives_every_count():
    assert usage_from_body(_message_body()) == EXPECTED


def test_the_golden_stream_gives_its_counts():
    usage = usage_from_sse(GOLDEN.read_bytes())
    assert (usage["input_tokens"], usage["output_tokens"]) == (3, 4)


def test_a_stream_cut_anywhere_gives_the_same_usage():
    rng = random.Random(5005)
    results = [_fed_in_steps(_stream(), rng, 23) for _trial in range(60)]
    assert results == [EXPECTED] * 60


def test_a_crlf_stream_parses_too():
    assert usage_from_sse(_stream(newline=b"\r\n")) == EXPECTED


def test_a_huge_content_line_is_skipped_without_losing_usage():
    huge = _event("content_block_delta", {"type": "content_block_delta",
                                          "delta": {"text": "x" * (3 << 20)}})
    assert _fed_in_steps(_stream(between=huge), random.Random(1), 65536) == EXPECTED


def test_one_chunk_holding_a_huge_line_then_the_delta_keeps_the_delta():
    """The reviewer's falsifier on HAL#646, kept as a forward anchor."""
    huge = _event("content_block_delta", {"type": "content_block_delta",
                                          "delta": {"text": "x" * (2 << 20)}})
    parser = SseUsage()
    parser.feed(_stream(between=huge))  # one chunk: line past the cap, then the delta
    assert parser.usage() == EXPECTED


def test_a_delta_with_cumulative_input_counts_replaces_them():
    delta = {"type": "message_delta", "delta": {},
             "usage": {"input_tokens": 99, "output_tokens": 7}}
    usage = usage_from_sse(_stream() + _event("message_delta", delta))
    assert (usage["input_tokens"], usage["output_tokens"]) == (99, 7)


def test_a_stream_without_message_start_has_unknown_usage():
    delta = {"type": "message_delta", "delta": {}, "usage": {"output_tokens": 7}}
    assert usage_from_sse(_event("message_delta", delta)) is None
    assert usage_from_sse(b"event: error\ndata: {\"type\":\"error\"}\n\n") is None


def test_without_the_lifetime_split_every_write_is_a_five_minute_write():
    raw = {"input_tokens": 1, "output_tokens": 1, "cache_creation_input_tokens": 500}
    usage = infinite_cost.anthropic_usage(raw)
    assert (usage["cache_write_tokens"], usage["cache_write_1h_tokens"]) == (500, 0)


def test_claude_codes_own_usage_block_is_standard():
    fixture = require("tools/wire-matrix/fixtures/claude/S0/stdout.txt")
    usage = infinite_cost.anthropic_usage(json.loads(fixture.read_text().splitlines()[-1])["usage"])
    assert usage is not None and "variant" not in usage


# --- list price --------------------------------------------------------------------

def test_list_price_matches_a_hand_computation():
    usage = {"input_tokens": 1000, "output_tokens": 500, "cache_read_tokens": 100000,
             "cache_write_tokens": 2000, "cache_write_1h_tokens": 3000}
    # $4 x 1000 + $20 x 500 + $0.20 x 100000 + $5 x 2000 + $8 x 3000, per million
    assert list_price("claude-opus-5-5", usage, DAY) == ("0.06800000", "claude-opus-5-5@1")


@pytest.mark.parametrize("model", ["anthropic/claude-opus-5-5", "claude-opus-5-5-20260923",
                                   "claude-opus-5-5[1m]"])
def test_a_prefixed_or_dated_id_shares_its_alias_row(model):
    assert list_price(model, EXPECTED, DAY)[1] == "claude-opus-5-5@1"


@pytest.fixture
def warnings_once(monkeypatch, caplog):
    monkeypatch.setattr(infinite_thread, "_WARNED", set())
    caplog.set_level(logging.WARNING, logger="lib.hal.infinite_cost")
    return caplog


def _warned(caplog, text: str) -> int:
    return sum(text in record.getMessage() for record in caplog.records)


def test_sonnet_5_5_prices_from_its_own_row(warnings_once):
    usage = {"input_tokens": 1000, "output_tokens": 500, "cache_read_tokens": 100000,
             "cache_write_tokens": 2000, "cache_write_1h_tokens": 3000}
    # $2 x 1000 + $10 x 500 + $0.10 x 100000 + $2.50 x 2000 + $4 x 3000, per million
    assert list_price("claude-sonnet-5-5", usage, DAY) == ("0.03400000", "claude-sonnet-5-5@1")
    # The row is effective from the day its figures were read, never before.
    assert list_price("claude-sonnet-5-5", usage, date(2026, 10, 11)) == (None, None)


# claude-sonnet-5-5 stood here as the rowless example until its row landed on
# 2026-10-12; claude-sonnet-9-9 is unreleased and inherits nothing.
@pytest.mark.parametrize("model", ["claude-opus-5-5-fast", "claude-sonnet-9-9", "mystery-model"])
def test_a_model_without_a_row_is_unknown_and_logged_once(model, warnings_once):
    assert list_price(model, EXPECTED, DAY) == (None, None)
    assert list_price(model, EXPECTED, DAY) == (None, None)
    assert _warned(warnings_once, f"no list price for {model!r}") == 1


def test_a_row_is_not_used_before_its_effective_date(warnings_once):
    assert list_price("claude-opus-5-5", EXPECTED, date(2026, 10, 4)) == (None, None)


def test_a_fast_request_is_marked_and_not_priced(warnings_once):
    usage = infinite_cost.anthropic_usage(dict(USAGE_BLOCK, speed="fast"))
    assert usage["variant"] == "speed=fast"
    assert list_price("claude-opus-5-5", usage, DAY) == (None, None)
    assert _warned(warnings_once, "speed=fast") == 1


def test_haiku_5_5_prices_its_short_prompt_tier_only(warnings_once):
    usage = {"input_tokens": 1000, "output_tokens": 500, "cache_read_tokens": 50_000,
             "cache_write_tokens": 2000, "cache_write_1h_tokens": 3000}
    # $0.10 x 1000 + $0.50 x 500 + $0.01 x 50000 + $0.125 x 2000 + $0.20 x 3000, per million
    assert list_price("claude-haiku-5-5", usage, date(2026, 10, 9)) == ("0.00170000", "claude-haiku-5-5@1")
    over = {"input_tokens": 1000, "output_tokens": 10, "cache_read_tokens": 90_000,
            "cache_write_tokens": 5000, "cache_write_1h_tokens": 4001}
    assert list_price("claude-haiku-5-5", over, date(2026, 10, 9)) == (None, None)
    assert _warned(warnings_once, "a 100001-token prompt is above the priced tier") == 1
    assert list_price("claude-haiku-5-5", usage, date(2026, 10, 8)) == (None, None)


def test_a_prompt_above_the_priced_tier_is_unknown(warnings_once):
    usage = {"input_tokens": 250_000, "output_tokens": 10}
    assert list_price("gemini-3.1-pro", usage, DAY) == (None, None)
    assert list_price("gemini-3.1-pro", {"input_tokens": 1000, "output_tokens": 10}, DAY)[0]


def test_an_unpublished_rate_is_unknown_only_when_it_is_used(warnings_once):
    used = {"input_tokens": 10, "output_tokens": 10, "cache_write_tokens": 5}
    assert list_price("deepseek-v4-pro", used, DAY) == (None, None)
    assert _warned(warnings_once, "no published cache_write rate") == 1
    assert list_price("deepseek-v4-pro", dict(used, cache_write_tokens=0), DAY)[0]


# --- the three figures -----------------------------------------------------------

@pytest.mark.parametrize("billing", ["subscription", "unknown"])
def test_subscription_and_unknown_billing_never_spend_zero(billing):
    fields = cost_fields("claude-opus-5-5", EXPECTED, billing=billing, day=DAY)
    assert fields["actual_spend_usd"] is None and fields["list_price_usd"]
    assert fields["billing"] == billing


def test_a_metered_key_spends_exactly_the_list_price():
    fields = cost_fields("claude-opus-5-5", EXPECTED, billing="metered", day=DAY)
    assert fields["actual_spend_usd"] == fields["list_price_usd"] is not None


def test_missing_usage_is_null_never_zero():
    fields = cost_fields("claude-opus-5-5", None, billing="metered", day=DAY)
    assert fields == {"usage": None, "list_price_usd": None, "price_version": None,
                      "billing": "metered", "actual_spend_usd": None}


@pytest.mark.parametrize("headers, billing", [
    ([("Authorization", OAUTH)], "subscription"),
    ([("x-api-key", API_KEY)], "metered"),
    ([("authorization", "Bearer " + API_KEY)], "metered"),
    ([("authorization", "Bearer something-else")], "unknown"),
    ([], "unknown"),
])
def test_billing_reads_only_the_credential_kind(headers, billing):
    assert billing_from_headers(headers) == billing


# --- the table ------------------------------------------------------------------

def _raw_rows():
    models = yaml.safe_load(infinite_cost.PRICES_PATH.read_text())["models"]
    rows = []
    for model, entries in models.items():
        rows.extend((model, entry) for entry in entries)
    return rows


def _row_is_sound(row: dict) -> bool:
    rates_are_text = all(row.get(c) is None or isinstance(row[c], str) for c in COLUMNS)
    return (type(row["version"]) is int and bool(row["source"]) and rates_are_text
            and date.fromisoformat(row["effective"]) <= DAY)


def test_every_row_is_versioned_dated_sourced_and_float_free():
    rows = _raw_rows()
    assert len(rows) >= 30
    assert [model for model, row in rows if not _row_is_sound(row)] == []


def _disagreements(model: str, row: dict, tracked) -> list:
    pairs = zip(COLUMNS[:4], tracked)
    return [(model, column) for column, value in pairs
            if row[column] is not None and row[column] != Decimal(str(value))]


def test_the_table_agrees_with_cost_tracker():
    """The ledger row in force today prices what lib/hal/cost_tracker bills today.
    cost_tracker applies dated vendor changes through _PRICE_SCHEDULE (Gemini 4
    Argon doubles on 2026-12-01) and this ledger admits no future-dated row, so on
    the day such a change arrives this fails until the ledger gains its row."""
    from lib.hal.cost_tracker import _PRICING, _resolve_pricing, _today
    table = infinite_cost.price_table()
    assert set(table) <= set(_PRICING)  # every seeded model has a cost_tracker row
    today = date.fromisoformat(_today())
    problems = []
    for model in table:
        row = infinite_cost.price_row(table, model, today)
        assert row is not None, f"{model} has no ledger row in force on {today}"
        problems.extend(_disagreements(model, row, _resolve_pricing(model)))
    assert problems == []


def test_a_prompt_limit_in_the_table_is_the_cost_trackers_first_tier_line():
    """Both files carry a vendor's prompt-length cutoff as a literal (the YAML
    row's max_prompt_tokens, the first _REQUEST_TIERS threshold); this is what
    keeps the two from drifting apart (AGORA on HAL#707)."""
    from lib.hal.cost_tracker import _REQUEST_TIERS
    compared = [(model, row["max_prompt_tokens"], _REQUEST_TIERS[model][0][0])
                for model, row in _raw_rows()
                if row.get("max_prompt_tokens") is not None and model in _REQUEST_TIERS]
    assert {"claude-haiku-5-5", "gemini-3.1-pro"} <= {model for model, _, _ in compared}
    assert [(model, limit, first) for model, limit, first in compared if limit != first] == []


@pytest.mark.parametrize("text", [
    "models: [unclosed",
    "models:\n  m:\n    - {version: 1, effective: '2026-10-05', input: 4.0}\n",
])
def test_an_unreadable_or_float_priced_table_prices_nothing(tmp_path, text, warnings_once):
    path = tmp_path / "prices.yaml"
    path.write_text(text)
    assert infinite_cost.price_table(path) == {}
    assert _warned(warnings_once, "cannot be read") == 1


# --- reconciliation -------------------------------------------------------------

def test_reconcile_sums_per_model_and_day_and_counts_what_it_does_not_cover(tmp_path):
    rows = [
        {"ts": "2026-10-05T10:00:00.000Z", "model": "claude-opus-5-5", "billing": "subscription",
         "usage": EXPECTED, "list_price_usd": "0.01000000", "actual_spend_usd": None},
        {"ts": "2026-10-05T11:00:00.000Z", "model": "claude-opus-5-5", "billing": "metered",
         "usage": EXPECTED, "list_price_usd": "0.02000000", "actual_spend_usd": "0.02000000"},
        {"ts": "2026-10-06T09:00:00.000Z", "model": "deepseek-v4-pro", "billing": "unknown",
         "usage": EXPECTED, "list_price_usd": "0.00300000", "actual_spend_usd": None},
        {"ts": "2026-10-06T09:01:00.000Z", "model": "mystery", "usage": EXPECTED,
         "list_price_usd": None},
        {"ts": "2026-10-06T09:02:00.000Z", "model": "claude-opus-5-5", "usage": None},
    ]
    text = "".join(json.dumps(row) + "\n" for row in rows) + "not a row\n"
    (tmp_path / "abc.jsonl").write_text(text)
    report = reconcile(tmp_path)
    counts = (report["rows"], report["unknown_usage"], report["unpriced"], report["unreadable"])
    assert counts == (5, 1, 1, 1)
    assert report["by_model"]["claude-opus-5-5"] == {"rows": 2, "list_price_usd": "0.03000000"}
    assert report["by_day"]["2026-10-06"] == {"rows": 1, "list_price_usd": "0.00300000"}
    assert (report["list_price_usd"], report["actual_spend_usd"]) == ("0.03300000", "0.02000000")


@pytest.mark.parametrize("price", ["abc", "NaN", "Infinity", 0.5])
def test_reconcile_counts_a_row_with_no_finite_price_as_unpriced(tmp_path, price):
    row = {"ts": "2026-10-05T10:00:00.000Z", "model": "m", "usage": EXPECTED,
           "list_price_usd": price}
    (tmp_path / "t.jsonl").write_text(json.dumps(row) + "\n")
    report = reconcile(tmp_path)
    assert (report["unpriced"], report["list_price_usd"]) == (1, "0.00000000")


def test_a_tiny_price_is_written_in_plain_digits():
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 1}
    assert list_price("deepseek-v4-pro", usage, DAY)[0] == "0.00000004"


# --- the gateway: usage read from the relayed reply ---------------------------

def _stand_in_signer(_root, _host):
    return lambda message: {"scheme": "test-stand-in", "length": len(message)}


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setenv("HAL_HOST_LABEL", "h-cost")
    monkeypatch.setattr(infinite_thread, "default_signer", _stand_in_signer)
    return infinite_thread.state_root() / "threads"


class _Trickle:
    """An upstream body that arrives in small, uneven pieces."""

    def __init__(self, data: bytes):
        self._data, self._rng = data, random.Random(7007)

    def read(self, size: int = -1) -> bytes:
        step = min(size if size > 0 else len(self._data), self._rng.randint(1, 9))
        piece, self._data = self._data[:step], self._data[step:]
        return piece


class _Proxy:
    def __init__(self, body: bytes, content_type: str):
        self._body, self._type = body, content_type

    def forward(self, path, body, headers, *, method="POST", max_retries=None):
        return UpstreamResponse(status=200, headers=[("content-type", self._type)],
                                _reader=_Trickle(self._body))


class _Headers(dict):
    """Header lookup is case-insensitive, as on a real request."""

    def get(self, key, default=None):
        return next((v for k, v in self.items() if k.lower() == key.lower()), default)


class _Handler:
    def __init__(self, headers):
        self._body = json.dumps({"model": "claude-opus-5-5", "max_tokens": 512,
                                 "stream": True, "messages": [
                                     {"role": "user", "content": "hello"}]}).encode()
        self.path, self.command, self.status = "/v1/messages", "POST", None
        self.headers = _Headers({"content-length": str(len(self._body)), **headers})
        self.wfile = io.BytesIO()

    @property
    def rfile(self):
        return io.BytesIO(self._body)

    def send_response(self, code):
        self.status = code

    def send_header(self, *_a):
        pass

    def end_headers(self):
        pass


def _rows(threads: Path) -> list:
    rows = []
    for path in threads.glob("*.jsonl"):
        rows.extend(json.loads(line) for line in path.read_bytes().splitlines())
    return rows


def _relay(headers, body: bytes, content_type: str) -> _Handler:
    handler = _Handler(headers)
    handle_proxy_request(handler, _Proxy(body, content_type), "/v1/messages")
    return handler


def _row_day(row: dict) -> date:
    return date.fromisoformat(row["ts"][:10])


def test_a_streamed_turn_is_relayed_unchanged_and_priced(store):
    handler = _relay({"authorization": OAUTH}, _stream(), "text/event-stream")
    assert handler.wfile.getvalue() == _stream()
    (row,) = _rows(store)
    assert row["usage"] == EXPECTED and row["billing"] == "subscription"
    assert row["list_price_usd"] == list_price("claude-opus-5-5", EXPECTED, _row_day(row))[0]
    assert row["list_price_usd"] is not None and row["actual_spend_usd"] is None


def test_a_buffered_reply_on_a_metered_key_is_priced_and_spent(store):
    _relay({"x-api-key": API_KEY}, _message_body(), "application/json")
    (row,) = _rows(store)
    assert row["usage"] == EXPECTED and row["billing"] == "metered"
    assert row["actual_spend_usd"] == row["list_price_usd"] is not None


def test_an_error_reply_records_unknown_usage_not_zero(store):
    error = b'{"type":"error","error":{"type":"rate_limit_error","message":"limited"}}'
    _relay({"authorization": OAUTH}, error, "application/json")
    (row,) = _rows(store)
    assert row["usage"] is None and row["list_price_usd"] is None
    assert row["actual_spend_usd"] is None


def test_a_failing_observer_never_touches_the_relay(store, monkeypatch):
    def broken(self, chunk):
        raise RuntimeError("observer bug")

    monkeypatch.setattr(infinite_cost.UsageObserver, "_take", broken)
    handler = _relay({"authorization": OAUTH}, _stream(), "text/event-stream")
    assert handler.wfile.getvalue() == _stream()
    (row,) = _rows(store)
    assert row["usage"] is None  # unknown, because the observer gave up


def test_a_failing_witness_still_serves_and_says_so_once(store, monkeypatch, caplog):
    from lib.hal import gateway_witness

    def broken(**_kw):
        raise RuntimeError("witness bug")

    monkeypatch.setattr(infinite_thread, "_WARNED", set())
    monkeypatch.setattr(gateway_witness, "witness_chat", broken)
    caplog.set_level(logging.WARNING, logger="lib.hal.infinite_thread")
    handler = _relay({"authorization": OAUTH}, _stream(), "text/event-stream")
    assert handler.wfile.getvalue() == _stream() and _rows(store) == []
    assert _warned(caplog, "could not build a thread row: RuntimeError") == 1


class _HangUp(io.BytesIO):
    """A client that disconnects on the first byte the gateway sends it."""

    def write(self, data):
        raise BrokenPipeError("client went away")


def test_a_client_that_hangs_up_still_leaves_its_row(store):
    handler = _Handler({"authorization": OAUTH})
    handler.wfile = _HangUp()
    handle_proxy_request(handler, _Proxy(_stream(), "text/event-stream"), "/v1/messages")
    (row,) = _rows(store)
    assert row["class"] == "served" and row["usage"] is None  # nothing complete had passed


@pytest.mark.parametrize("bad", [{"input_tokens": 1, "authorization": "x"},
                                 {"input_tokens": "1"}, {"input_tokens": -1},
                                 {"input_tokens": True}])
def test_usage_carries_only_counts(bad):
    with pytest.raises(ValueError):
        infinite_thread.checked_usage(bad)
