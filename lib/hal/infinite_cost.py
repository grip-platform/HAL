"""infinite_cost -- cost truth for the provable thread (InfiniteMachine v2, Step 4).

Each signed thread row (``lib/hal/infinite_thread.py``) carries three figures,
kept apart on purpose:

``usage``             what the response itself reported: token counts read from
                      a Messages body, or from an SSE stream's ``message_start``
                      and ``message_delta`` events. ``None`` when nothing was
                      measured -- never zeros, which would read as "free".
``list_price_usd``    what those tokens cost at the vendor's LIST price, from the
                      versioned table ``config/model_prices.yaml``. ``None`` when
                      the usage is unknown, the model has no row, or the request
                      used something the row does not price (a fast or priority
                      request, a larger prompt tier, an unpublished rate). Every
                      such case is logged once; none is ever guessed.
``actual_spend_usd``  what was actually spent on this call. ``None`` for
                      subscription traffic (``billing: "subscription"``): its
                      spend is the plan's flat fee, not a per-call figure, and
                      writing 0 would claim it was free. ``None`` too when the
                      billing is unknown. Only ``billing: "metered"`` -- a
                      pay-per-token Anthropic API key -- equals the list price,
                      because that key is billed at exactly these rates.

Prices are ``Decimal`` from quoted strings and written as fixed eight-place
strings, so no binary float enters a row (``canonical`` refuses floats).

``reconcile`` sums list-price equivalents per model and per day over a threads
directory and counts the rows it could not price, so a reader sees what the
total does not cover. It reads rows; it does not verify them -- run
``verify_thread`` first.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from lib.hal.infinite_thread import warn_once

log = logging.getLogger(__name__)

PRICES_PATH = Path(__file__).resolve().parents[2] / "config" / "model_prices.yaml"
_MILLION = Decimal(1_000_000)
_PLACES = Decimal("0.00000001")
#: (usage count, price column) pairs that make up a list price.
_RATED = (("input_tokens", "input"), ("output_tokens", "output"),
          ("cache_read_tokens", "cache_read"), ("cache_write_tokens", "cache_write"),
          ("cache_write_1h_tokens", "cache_write_1h"))
_RATE_NAMES = tuple(column for _count, column in _RATED)
_PROMPT = ("input_tokens", "cache_read_tokens", "cache_write_tokens", "cache_write_1h_tokens")
#: A usage field that names a differently priced request, and its standard values.
_STANDARD = {"service_tier": {"standard"}, "speed": {"standard"},
             "inference_geo": {"", "global"}}
_SNAPSHOT = re.compile(r"-\d{8}$")
_LINE_CAP = 1 << 20  # an SSE line longer than this cannot be a usage event
_BODY_CAP = 16 << 20  # a buffered body longer than this is not parsed


# --- usage --------------------------------------------------------------------------

def _count(value: Any) -> Optional[int]:
    return value if type(value) is int and value >= 0 else None


def anthropic_usage(raw: Any) -> Optional[dict]:
    """Anthropic's ``usage`` block as a row's usage, or None when it is not one.

    Input and output counts must both be present. Cache writes are split by
    lifetime when the block says so (``cache_creation``); without that split
    every write is a 5-minute write, the API's default lifetime.
    """
    if not isinstance(raw, dict):
        return None
    counts = [_count(raw.get(name)) for name in ("input_tokens", "output_tokens")]
    if None in counts:
        return None
    write_5m, write_1h = _cache_writes(raw)
    usage = {"input_tokens": counts[0], "output_tokens": counts[1],
             "cache_read_tokens": _count(raw.get("cache_read_input_tokens")) or 0,
             "cache_write_tokens": write_5m, "cache_write_1h_tokens": write_1h}
    variant = _variant(raw)
    if variant:
        usage["variant"] = variant
    return usage


def _cache_writes(raw: dict) -> Tuple[int, int]:
    detail = raw.get("cache_creation")
    if isinstance(detail, dict) and ({"ephemeral_5m_input_tokens",
                                      "ephemeral_1h_input_tokens"} & set(detail)):
        return (_count(detail.get("ephemeral_5m_input_tokens")) or 0,
                _count(detail.get("ephemeral_1h_input_tokens")) or 0)
    return _count(raw.get("cache_creation_input_tokens")) or 0, 0


def _variant(raw: dict) -> Optional[str]:
    """The first field naming a non-standard (differently priced) request."""
    for name, standard in _STANDARD.items():
        value = raw.get(name)
        if value is not None and str(value).lower() not in standard:
            return f"{name}={value}"[:60]
    return None


def crosswire_usage(normalised: Any) -> Optional[dict]:
    """The gateway's cross-wire usage (``gateway_crosswire.normalise_provider_usage``)."""
    if not isinstance(normalised, dict):
        return None
    usage = {name: _count(normalised.get(name)) or 0
             for name in ("input_tokens", "output_tokens", "cache_read_tokens",
                          "cache_write_tokens", "reasoning_tokens")}
    usage["cache_write_1h_tokens"] = 0
    return usage


def usage_from_body(body: Any) -> Optional[dict]:
    """Usage from a non-streamed Messages body (bytes, text or a parsed dict)."""
    data = body if isinstance(body, dict) else _json_object(body)
    return anthropic_usage(data.get("usage")) if data else None


def _json_object(raw: Any) -> Optional[dict]:
    try:
        data = json.loads(raw)
    except (TypeError, ValueError, RecursionError):
        return None
    return data if isinstance(data, dict) else None


class SseUsage:
    """Usage from an Anthropic SSE stream, fed in chunks of any size.

    ``message_start`` carries the input counts and a first output count; each
    ``message_delta`` carries cumulative counts that replace them. Only those
    two events are parsed. Content is never kept: a line too long to be a usage
    event is dropped as it streams past.
    """

    def __init__(self) -> None:
        self._partial = b""
        self._skipping = False
        self._raw: Optional[dict] = None

    def feed(self, chunk: bytes) -> None:
        if self._skipping:
            cut = chunk.find(b"\n")
            if cut < 0:
                return
            chunk, self._skipping = chunk[cut + 1:], False
        *lines, self._partial = (self._partial + chunk).split(b"\n")
        for line in lines:
            self._line(line.rstrip(b"\r"))
        if len(self._partial) > _LINE_CAP:
            self._partial, self._skipping = b"", True

    def usage(self) -> Optional[dict]:
        if self._partial and not self._skipping:
            self._line(self._partial.rstrip(b"\r"))  # a stream that ended mid-line
            self._partial = b""
        return anthropic_usage(self._raw)

    def _line(self, line: bytes) -> None:
        if not line.startswith(b"data:") or b"message_" not in line:
            return
        event = _json_object(line[5:])
        kind = event.get("type") if event else None
        if kind == "message_start":
            message = event.get("message")
            usage = message.get("usage") if isinstance(message, dict) else None
            self._raw = dict(usage) if isinstance(usage, dict) else self._raw
        elif kind == "message_delta" and isinstance(event.get("usage"), dict):
            self._raw = dict(self._raw or {}, **event["usage"])


def usage_from_sse(data: bytes) -> Optional[dict]:
    """Usage from a whole SSE body."""
    parser = SseUsage()
    parser.feed(data)
    return parser.usage()


class UsageObserver:
    """Watches one relayed response body and keeps only its usage.

    ``feed`` never raises: if anything goes wrong the observer stops watching
    and reports no usage, so the relay it sits in is never disturbed.
    """

    def __init__(self, content_type: str) -> None:
        self._sse = SseUsage() if "event-stream" in (content_type or "").lower() else None
        self._body = bytearray()
        self._broken = False

    def feed(self, chunk: bytes) -> None:
        try:
            self._take(chunk)
        except Exception:  # noqa: BLE001 - observation must never break the relay
            self._broken = True

    def usage(self) -> Optional[dict]:
        if self._broken:
            return None
        if self._sse is not None:
            return self._sse.usage()
        return usage_from_body(bytes(self._body)) if self._body else None

    def _take(self, chunk: bytes) -> None:
        if self._broken:
            return
        if self._sse is not None:
            self._sse.feed(chunk)
        elif len(self._body) + len(chunk) > _BODY_CAP:
            self._broken = True  # too big to have been measured: unknown, not zero
        else:
            self._body.extend(chunk)


# --- billing ------------------------------------------------------------------------

def billing_from_headers(headers: Any) -> str:
    """How an Anthropic request is billed, read from its credential's kind only.

    ``subscription`` for an OAuth token (``sk-ant-oat``), ``metered`` for an API
    key (``sk-ant-api``, sent as ``x-api-key`` or as a Bearer the gateway turns
    into one), ``unknown`` otherwise. Only the prefix is looked at; nothing is
    kept.
    """
    from lib.hal.wire_classify import header_value  # lazy: keeps imports light

    key = header_value(headers, "x-api-key").strip()
    auth = header_value(headers, "authorization").strip()
    token = key or (auth[7:].strip() if auth.lower().startswith("bearer ") else "")
    if token.startswith("sk-ant-oat"):
        return "subscription"
    if token.startswith("sk-ant-api"):
        return "metered"
    return "unknown"


# --- the price table ------------------------------------------------------------

_TABLE_LOCK = threading.Lock()
_TABLE_CACHE: Dict[Tuple[str, int], Dict[str, list]] = {}


def price_table(path: Optional[Path] = None) -> Dict[str, list]:
    """``{model: [row, ...]}`` newest first; ``{}`` (logged once) when unreadable.

    Re-read whenever the file changes, so a price added on disk is used without
    a restart.
    """
    path = Path(path) if path is not None else PRICES_PATH
    try:
        key = (str(path), path.stat().st_mtime_ns)
        with _TABLE_LOCK:
            if key not in _TABLE_CACHE:
                _TABLE_CACHE.clear()
                _TABLE_CACHE[key] = _load_table(path)
            return _TABLE_CACHE[key]
    except Exception as exc:  # noqa: BLE001 - an unreadable table prices nothing
        warn_once(f"infinite_cost: the price table {path.name} cannot be read "
                  f"({type(exc).__name__}); every list price is unknown", logger=log)
        return {}


def _load_table(path: Path) -> Dict[str, list]:
    import yaml  # PyYAML: a HAL runtime dependency (lib/hal/cli.py)

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, dict):
        raise ValueError("the table has no models mapping")
    return {model: _model_rows(model, rows) for model, rows in models.items()}


def _model_rows(model: str, rows: Any) -> list:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{model}: rows must be a non-empty list")
    parsed = [_price_row(model, row) for row in rows]
    return sorted(parsed, key=lambda row: row["effective"], reverse=True)


def _price_row(model: str, raw: Any) -> dict:
    if not isinstance(raw, dict) or type(raw.get("version")) is not int:
        raise ValueError(f"{model}: every row needs an integer version")
    row = {name: _rate(model, raw.get(name)) for name in _RATE_NAMES}
    limit = raw.get("max_prompt_tokens")
    if limit is not None and _count(limit) is None:
        raise ValueError(f"{model}: max_prompt_tokens must be a whole number")
    row.update(model=model, version=raw["version"], max_prompt_tokens=limit,
               effective=date.fromisoformat(str(raw.get("effective"))))
    return row


def _rate(model: str, value: Any) -> Optional[Decimal]:
    """A quoted decimal string, or null for a rate the vendor does not publish."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{model}: write rates as quoted strings, never floats")
    try:
        rate = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{model}: {value!r} is not a decimal") from exc
    if not rate.is_finite() or rate < 0:
        raise ValueError(f"{model}: {value!r} is not a price")
    return rate


def price_id(model: str) -> str:
    """The table's name for ``model``: no provider prefix, no ``[1m]``, no date snapshot."""
    name = (model or "").strip()
    if name.endswith("[1m]"):
        name = name[:-len("[1m]")]
    if "/" in name:
        name = name.split("/", 1)[1]
    return _SNAPSHOT.sub("", name)


def price_row(table: Dict[str, list], model: str, day: date) -> Optional[dict]:
    """The newest row for ``model`` whose effective date is on or before ``day``."""
    rows = table.get(price_id(model)) or []
    return next((row for row in rows if row["effective"] <= day), None)


# --- the three figures ------------------------------------------------------------

def list_price(model: str, usage: Optional[dict], day: date,
               table: Optional[Dict[str, list]] = None) -> Tuple[Optional[str], Optional[str]]:
    """``(list price in USD, the price version used)``, or ``(None, None)``.

    ``(None, None)`` when the usage is unknown, and -- logged once per model and
    cause -- when the table cannot price this request. Never a guess.
    """
    if usage is None:
        return None, None
    row, problem = _priced_row(model, usage, day, table if table is not None else price_table())
    if problem is not None:
        warn_once(f"infinite_cost: no list price for {model!r}: {problem}; "
                  "it is recorded as unknown", logger=log)
        return None, None
    total = sum((Decimal(usage.get(count, 0)) * row[column]
                 for count, column in _RATED if usage.get(count)), Decimal(0))
    return _text(total / _MILLION), f"{row['model']}@{row['version']}"


def _priced_row(model: str, usage: dict, day: date,
                table: Dict[str, list]) -> Tuple[Optional[dict], Optional[str]]:
    if usage.get("variant"):
        return None, f"a {usage['variant']} request is priced differently"
    row = price_row(table, model, day)
    if row is None:
        return None, f"config/model_prices.yaml has no row for it on {day.isoformat()}"
    prompt = sum(usage.get(name, 0) for name in _PROMPT)
    if row["max_prompt_tokens"] is not None and prompt > row["max_prompt_tokens"]:
        return None, f"a {prompt}-token prompt is above the priced tier"
    missing = [column for count, column in _RATED if usage.get(count) and row[column] is None]
    if missing:
        return None, f"no published {missing[0]} rate"
    return row, None


def cost_fields(model: str, usage: Optional[dict], *, billing: str = "unknown",
                day: Optional[date] = None,
                table: Optional[Dict[str, list]] = None) -> dict:
    """The row's cost fields: observed usage, list price and actual spend, kept apart."""
    list_usd, version = list_price(model, usage, day or utc_today(), table)
    return {"usage": usage, "list_price_usd": list_usd, "price_version": version,
            "billing": billing,
            "actual_spend_usd": list_usd if billing == "metered" else None}


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


def _text(value: Decimal) -> str:
    """A dollar figure as a fixed eight-place decimal string.

    ``format(..., "f")``, not ``str``: ``str`` writes a figure under a millionth
    of a dollar, and zero itself, in exponent form (``1E-8``, ``0E-8``).
    """
    return format(value.quantize(_PLACES), "f")


# --- reconciliation -----------------------------------------------------------------

def reconcile(thread_dir: Path) -> dict:
    """List-price equivalents per model and per day, and what they do not cover.

    ``unknown_usage`` counts rows that measured nothing; ``unpriced`` rows whose
    usage is known but had no list price; ``unreadable`` lines that are not
    rows. ``actual_spend_usd`` sums only the rows that state one. Figures are
    decimal strings.
    """
    totals = {"rows": 0, "unknown_usage": 0, "unpriced": 0, "unreadable": 0,
              "by_model": {}, "by_day": {}, "by_billing": {},
              "list_price_usd": Decimal(0), "actual_spend_usd": Decimal(0)}
    for path in sorted(Path(thread_dir).glob("*.jsonl")):
        _tally_lines(totals, path.read_bytes().splitlines())
    return _report(totals)


def _tally_lines(totals: dict, lines: Iterable[bytes]) -> None:
    for line in lines:
        row = _json_object(line)
        if row is None:
            totals["unreadable"] += 1
            continue
        _tally(totals, row)


def _tally(totals: dict, row: dict) -> None:
    totals["rows"] += 1
    billing = str(row.get("billing", "unknown"))
    totals["by_billing"][billing] = totals["by_billing"].get(billing, 0) + 1
    if row.get("usage") is None:
        totals["unknown_usage"] += 1
        return
    price = _decimal(row.get("list_price_usd"))
    if price is None:
        totals["unpriced"] += 1
        return
    _add(totals["by_model"], str(row.get("model") or "unknown"), price)
    _add(totals["by_day"], str(row.get("ts", ""))[:10], price)
    totals["list_price_usd"] += price
    totals["actual_spend_usd"] += _decimal(row.get("actual_spend_usd")) or Decimal(0)


def _decimal(value: Any) -> Optional[Decimal]:
    """A stored dollar figure, or None for anything that is not a finite one.

    ``Decimal("NaN")`` parses, so finiteness is checked too: one such row would
    otherwise turn every total it joins into NaN.
    """
    try:
        number = Decimal(value) if isinstance(value, str) else None
    except InvalidOperation:
        return None
    return number if number is not None and number.is_finite() else None


def _add(bucket: dict, key: str, price: Decimal) -> None:
    entry = bucket.setdefault(key, {"rows": 0, "list_price_usd": Decimal(0)})
    entry["rows"] += 1
    entry["list_price_usd"] += price


def _report(totals: dict) -> dict:
    """The totals with every Decimal written as a fixed eight-place string."""
    _texts(totals["by_model"])
    _texts(totals["by_day"])
    totals["list_price_usd"] = _text(totals["list_price_usd"])
    totals["actual_spend_usd"] = _text(totals["actual_spend_usd"])
    return totals


def _texts(bucket: dict) -> None:
    for entry in bucket.values():
        entry["list_price_usd"] = _text(entry["list_price_usd"])


__all__ = [
    "PRICES_PATH",
    "SseUsage",
    "UsageObserver",
    "anthropic_usage",
    "billing_from_headers",
    "cost_fields",
    "crosswire_usage",
    "list_price",
    "price_id",
    "price_row",
    "price_table",
    "reconcile",
    "usage_from_body",
    "usage_from_sse",
    "utc_today",
]
