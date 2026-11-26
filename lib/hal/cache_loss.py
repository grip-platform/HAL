"""cache_loss -- why Claude Code's prompt cache gets rewritten from scratch,
and what the subscription limit actually charges for it.

A Claude Fable 5.1 research pass over three days of GRIP's own transcripts
(2026-10-08) found 189 turns whose cache read fell under 50k tokens within five
minutes of the previous turn, at about the same context size (ratio 0.8-1.2) --
roughly 100.2M tokens of cache WRITES, about 38% of every main-session cache
write. Only 2 of the 189 followed a ``/compact``; only 18 followed a model
switch. A transcript cannot say why the other 169 happened, because Claude Code
never tells the user what it sent -- only the HAL wire sees both the request
that went out and the response that came back.

**Part A** (:func:`record_response`, called once per ``/v1/messages`` response
from ``lib.hal.wire_proxy``) appends ONE JSON line to a capped log: a UTC
timestamp, the session id, the account label, the response's organisation id,
the served model, sha256 of the request's ``system`` and ``tools`` and of its
first message (so a changed prefix is visible without ever storing the prefix),
the response's usage, the account's 5-hour/7-day unified utilization, the
prompt id Claude Code sends (``x-claude-code-prompt-id``, added in 2.1.283 so a
gateway can group the requests that serve one user prompt; null when absent or
outside the session-id safe set), and two flags that serve the HAL#709 question (whether a delta, the request shape
measured under HAL#706, is served on another account): ``continues_thread``
(the body's ``thread.previous_message_id`` was set; null when there was no JSON
object to read) and ``account_switched`` (the wire sent the request on a pool
account's bearer, by the up-front swap or by a replay; null when the caller did
not say). The row holds those named fields and nothing else -- never a body, a
prompt, a token, or any header value beyond the organisation id, the
utilization, the session id and the prompt id named here.

The hash is sha256 of the field's canonical (sorted-key) JSON text. It is a
*change detector*, not a provenance seal: it is not RFC 8785 and makes no claim
beyond "the same bytes hash the same way on this interpreter". That is all this
ledger needs -- whether a turn's cache-relevant prefix is byte-identical to the
one before it.

**Part B** answers two questions over that log:

* :func:`find_drops` / :func:`report` -- for each turn whose cache read fell
  sharply at a similar context size (this module's own operational version of
  the measurement above, using the hashes rather than guessing from a
  transcript), name what changed since the turn before it in the same
  session: ``system``, ``tools``, ``first_message``, ``organisation``,
  ``model`` -- in that priority order -- or nothing, which points at
  server-side eviction rather than anything Claude Code sent differently.
* :func:`fit_utilization` -- the unified counter's header has two-decimal
  resolution, so most consecutive answers repeat the same percentage until
  enough tokens tip it over to the next representable step. Summing usage
  across each such flat stretch, then regressing the visible step size
  against the four summed usage counts, estimates how many percentage points
  of the 5-hour window an input, a cache read, a cache write and an output
  token each cost. Reported per model, pooling every account's own steps
  (each account's counter is its own monotonic series, so steps are read one
  account at a time and then pooled) -- and reported honestly as "too few
  points" rather than guessed when there are not enough of them.

No code here starts, stops, restarts, or talks to the wire; it only reads and
writes files under ``$HAL_HOME``, exactly like every other sink on this path
(``lib.hal.gateway_failover.GatewayTelemetry``, ``lib.hal.gateway_witness``).
Turning Part A on for live traffic needs the wire to be restarted to pick up
this module's new call site -- deliberately left to the operator, never done
by this module or by the PR that ships it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

LOG_NAME = "cache-loss-ledger.jsonl"
#: Rotate past this many bytes, keeping exactly one prior file (``.1``). A row
#: is a few hundred bytes, so this holds tens of thousands of turns -- weeks of
#: a single account's traffic -- before the oldest half is dropped.
MAX_LOG_BYTES = 10 * 1024 * 1024

#: The four usage counts this ledger ever sums or fits. Fixed order: the fit's
#: weight vector and the report's column order both follow it.
USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_creation_tokens")

#: What a row's ``usage`` sums to "how big was this turn's prompt" -- the three
#: fields that partition one turn's prompt tokens into fresh, newly-cached, and
#: served-from-cache. Output tokens are not part of the prompt and are excluded.
_PROMPT_FIELDS = ("input_tokens", "cache_read_tokens", "cache_creation_tokens")

#: The row fields that explain a cache-read fall, in the priority order the
#: brief names them: a changed prefix explains it before a changed account or
#: model does, because a prefix change is sufficient on its own.
_CAUSE_FIELDS = (("system_hash", "system"), ("tools_hash", "tools"),
                 ("first_message_hash", "first_message"),
                 ("org_id", "organisation"), ("model", "model"))


# --------------------------------------------------------------------------- #
# Part A -- building one row (pure), and appending it (the only I/O).
# --------------------------------------------------------------------------- #
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_hex(value: Any) -> Optional[str]:
    """sha256 hex of ``value``'s canonical JSON text, or None when absent.

    Sorted keys so the hash does not depend on key order; ``default=str`` so an
    unexpected type (never seen in a parsed Anthropic request body, but this
    must not raise into a live request) still hashes rather than failing.
    """
    if value is None:
        return None
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def request_hashes(headers: Any, body_bytes: Optional[bytes]) -> Dict[str, Optional[str]]:
    """``{"system_hash", "tools_hash", "first_message_hash"}`` for one request.

    Reuses :func:`lib.hal.wire_classify.request_json`, the same parse the wire's
    own route classifier runs (it already handles ``Content-Encoding``), so this
    never re-implements body decoding. Each hash is None when its source field
    is absent; nothing here ever returns the field's content.
    """
    from lib.hal.wire_classify import request_json  # lazy: wire imports stay light

    payload = request_json(headers, body_bytes) or {}
    messages = payload.get("messages")
    first = messages[0] if isinstance(messages, list) and messages else None
    return {"system_hash": hash_hex(payload.get("system")),
            "tools_hash": hash_hex(payload.get("tools")),
            "first_message_hash": hash_hex(first)}


def usage_fields(raw_usage: Optional[dict]) -> Optional[Dict[str, int]]:
    """The row's ``usage`` from :func:`lib.hal.infinite_cost.anthropic_usage`'s
    shape, or None when nothing was measured.

    ``cache_creation_tokens`` sums the 5-minute and 1-hour cache-write
    lifetimes: this ledger asks whether the cache was reused, not which
    lifetime wrote it.
    """
    if not raw_usage:
        return None
    return {"input_tokens": int(raw_usage.get("input_tokens") or 0),
            "output_tokens": int(raw_usage.get("output_tokens") or 0),
            "cache_read_tokens": int(raw_usage.get("cache_read_tokens") or 0),
            "cache_creation_tokens": (int(raw_usage.get("cache_write_tokens") or 0)
                                      + int(raw_usage.get("cache_write_1h_tokens") or 0))}


def window_utilization(reading: Any, name: str) -> Optional[float]:
    """One named window's utilization (0-100) from an
    :class:`lib.hal.account_ledger.Reading`, or None when that window was not
    in the answer's headers at all."""
    window = reading.window(name)
    return window.utilization if window is not None else None


def _org_id(response_headers: Any) -> Optional[str]:
    from lib.hal.wire_classify import header_value  # lazy: wire imports stay light

    value = header_value(response_headers, "anthropic-organization-id").strip()
    return value or None


def continues_thread(request_body: Optional[bytes]) -> Optional[bool]:
    """Whether the request continues a server-held thread.

    Claude Code 2.1.295 sends only a thread's first request in full; every
    later one is a delta that carries ``thread.previous_message_id`` and relies
    on the server holding the rest under the bearer that opened the thread
    (HAL#706). Three answers: True when ``thread.previous_message_id`` is set
    (a delta); False when the body is a JSON object without it, which includes
    a thread's first request, whose ``thread`` object holds only ``type`` (the
    first request's ``previous_message_id: null`` sits under ``diagnostics``,
    measured 2026-10-09 on an instrumented wire, and is not read here); None
    when there is no body, or the body is not a JSON object, so the question
    could not be read. Read from the body the wire already copies for this
    ledger; nothing of it is kept.
    """
    if not request_body:
        return None
    try:
        payload = json.loads(request_body)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    thread = payload.get("thread")
    return isinstance(thread, dict) and bool(thread.get("previous_message_id"))


def build_row(*, ts: str, session_id: Optional[str], account_label: str,
             org_id: Optional[str], model: Optional[str],
             hashes: Dict[str, Optional[str]], usage: Optional[Dict[str, int]],
             five_hour_utilization: Optional[float],
             seven_day_utilization: Optional[float],
             continues_thread: Optional[bool] = None,
             account_switched: Optional[bool] = None,
             prompt_id: Optional[str] = None) -> dict:
    """The one JSON-safe dict this ledger ever writes. Hashes and numbers only --
    never a body, a prompt, a token, or any header value this module does not
    name in its own docstring.

    Two flags, both observe-only, let a reader tell a delta (True) from a full
    request (False) where the body could be read, and see whether the request
    went out on an account other than the one the client presented (HAL#709,
    whose outcome is unmeasured): ``continues_thread`` from
    :func:`continues_thread`, which is None for a body that is not a JSON
    object, and ``account_switched``, True when the wire sent the request on a
    pool account's bearer by the up-front swap or by a replay, None when the
    caller did not say. ``prompt_id`` groups the rows that served one user
    prompt (Claude Code's ``x-claude-code-prompt-id``); None when the request
    carried none in the safe set.
    """
    return {"ts": ts, "session_id": session_id, "account_label": account_label,
            "org_id": org_id, "model": model,
            "system_hash": hashes.get("system_hash"),
            "tools_hash": hashes.get("tools_hash"),
            "first_message_hash": hashes.get("first_message_hash"),
            "usage": usage,
            "five_hour_utilization": five_hour_utilization,
            "seven_day_utilization": seven_day_utilization,
            "continues_thread": continues_thread,
            "account_switched": account_switched,
            "prompt_id": prompt_id}


def _prompt_id(request_headers: Any) -> Optional[str]:
    """The request's ``x-claude-code-prompt-id`` in the safe set, else None."""
    from lib.hal.gateway_witness import prompt_id_from_headers  # lazy: wire imports stay light

    return prompt_id_from_headers(request_headers)


def log_path(home: Optional[Path] = None) -> Path:
    base = home if home is not None else Path(os.environ.get("HAL_HOME", str(Path.home() / ".hal")))
    return Path(base) / "logs" / LOG_NAME


def _rotate_if_big(path: Path, max_bytes: int) -> None:
    """Move ``path`` to ``path.1`` (replacing any older one) once it reaches
    ``max_bytes``. Not locked, same as ``gateway_failover._default_sink``: this
    is a diagnostic log, and losing one rotation race under concurrent writers
    costs a little history, never correctness of the line being written."""
    try:
        if path.stat().st_size < max_bytes:
            return
    except FileNotFoundError:
        return
    os.replace(path, path.with_name(path.name + ".1"))


def append_row(row: dict, *, home: Optional[Path] = None,
               max_bytes: int = MAX_LOG_BYTES) -> None:
    """Append one canonical JSON line to the capped log. Fail-open: never
    raises -- a disk or permissions failure here must not break the request
    whose response it describes, exactly like every other sink on this path."""
    try:
        path = log_path(home)
        path.parent.mkdir(parents=True, exist_ok=True)
        _rotate_if_big(path, max_bytes)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    except Exception:  # noqa: BLE001 -- observability never subtracts availability
        pass


def record_response(*, session_id: Optional[str], account_label: str,
                    model: Optional[str], request_headers: Any,
                    request_body: Optional[bytes], response_headers: Any,
                    usage: Optional[dict], home: Optional[Path] = None,
                    max_bytes: int = MAX_LOG_BYTES, ts: Optional[str] = None,
                    account_switched: Optional[bool] = None) -> None:
    """Build and append one row from what the wire already holds for this
    exchange. The sole entry point ``lib.hal.wire_proxy`` calls; fail-open on
    its own, so a caller never needs its own try/except around this.
    ``continues_thread`` is derived here from the body; only ``account_switched``
    is the wire's to say, because only the wire knows which bearer it sent.
    """
    try:
        from lib.hal.account_ledger import read_headers  # lazy: wire imports stay light

        reading = read_headers(response_headers)
        row = build_row(
            ts=ts or now_iso(), session_id=session_id, account_label=account_label,
            org_id=_org_id(response_headers), model=model or None,
            hashes=request_hashes(request_headers, request_body),
            usage=usage_fields(usage),
            five_hour_utilization=window_utilization(reading, "five_hour"),
            seven_day_utilization=window_utilization(reading, "seven_day"),
            continues_thread=continues_thread(request_body),
            account_switched=account_switched,
            prompt_id=_prompt_id(request_headers))
        append_row(row, home=home, max_bytes=max_bytes)
    except Exception:  # noqa: BLE001 -- observability never subtracts availability
        pass


# --------------------------------------------------------------------------- #
# Part B.1 -- what changed just before a sharp cache-read fall.
# --------------------------------------------------------------------------- #
def context_size(usage: Optional[dict]) -> Optional[int]:
    """Roughly how big this turn's prompt was: fresh + newly-cached +
    served-from-cache tokens. None when usage was not measured."""
    if not usage:
        return None
    return sum(int(usage.get(name) or 0) for name in _PROMPT_FIELDS)


def cache_fraction(usage: Optional[dict]) -> Optional[float]:
    """How much of this turn's prompt the cache actually served, 0..1."""
    size = context_size(usage)
    if not size:
        return None
    return (usage.get("cache_read_tokens") or 0) / size


def changed_between(prev: dict, row: dict) -> Tuple[str, ...]:
    """Every row field that differs from ``prev``, in :data:`_CAUSE_FIELDS`
    priority order."""
    return tuple(label for key, label in _CAUSE_FIELDS if row.get(key) != prev.get(key))


def _gap_seconds(earlier: Any, later: Any) -> Optional[float]:
    try:
        return (datetime.fromisoformat(later) - datetime.fromisoformat(earlier)).total_seconds()
    except (TypeError, ValueError):
        return None


def _prompt_shrank_sharply(prev_usage: Optional[dict], row_usage: Optional[dict],
                           size_ratio_band: Tuple[float, float], fallen_below: float,
                           risen_above: float) -> bool:
    """True when the prompt stayed about the same size (``size_ratio_band``)
    but the cache's share of it collapsed, from above ``risen_above`` to below
    ``fallen_below``."""
    size_now, size_prev = context_size(row_usage), context_size(prev_usage)
    if not size_now or not size_prev:
        return False
    ratio = size_now / size_prev
    if not (size_ratio_band[0] <= ratio <= size_ratio_band[1]):
        return False
    frac_now, frac_prev = cache_fraction(row_usage), cache_fraction(prev_usage)
    if frac_now is None or frac_prev is None:
        return False
    return frac_now < fallen_below and frac_prev > risen_above


@dataclass(frozen=True)
class CacheDrop:
    """One turn whose cache read fell sharply at about the same prompt size as
    the turn before it, in the same session."""

    session_id: str
    ts: str
    prev_ts: str
    context_size: int
    prev_context_size: int
    cache_read_tokens: int
    prev_cache_read_tokens: int
    changed: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def cause(self) -> str:
        return ", ".join(self.changed) if self.changed else "nothing changed (server-side eviction?)"


def find_drops(rows: Iterable[dict], *, size_ratio_band: Tuple[float, float] = (0.8, 1.2),
               fallen_below: float = 0.2, risen_above: float = 0.5,
               max_gap_s: float = 300.0) -> List[CacheDrop]:
    """Every turn, across all sessions in ``rows``, whose cache utilization
    collapsed while the prompt stayed about the same size, within
    ``max_gap_s`` seconds of the turn before it in the same session.

    This generalises the transcript-level measurement's absolute "cache read
    under 50k tokens" into a ratio that means the same thing at any context
    size, now that the wire-level log carries exact usage for every turn.
    ``rows`` need not be sorted; this sorts by ``ts`` itself.

    Only a row with a measured prompt becomes the turn a later row is compared
    with: an error response in between (a 429 or 529 carries no usage) would
    otherwise hide a real drop behind it, and errors and account switches are
    plausible causes of exactly these rewrites (R0 of an AGORA concern on
    HAL#702).
    """
    drops: List[CacheDrop] = []
    previous: Dict[str, dict] = {}
    for row in sorted(rows, key=lambda r: r.get("ts") or ""):
        session_id = row.get("session_id")
        prev = previous.get(session_id) if session_id else None
        if session_id and context_size(row.get("usage")):
            previous[session_id] = row
        if prev is None:
            continue
        gap = _gap_seconds(prev.get("ts"), row.get("ts"))
        if gap is None or gap > max_gap_s:
            continue
        if not _prompt_shrank_sharply(prev.get("usage"), row.get("usage"),
                                      size_ratio_band, fallen_below, risen_above):
            continue
        drops.append(_drop(session_id, prev, row))
    return drops


def _drop(session_id: str, prev: dict, row: dict) -> CacheDrop:
    """The :class:`CacheDrop` for one flagged pair of turns."""
    return CacheDrop(
        session_id=session_id, ts=row.get("ts", ""), prev_ts=prev.get("ts", ""),
        context_size=context_size(row.get("usage")) or 0,
        prev_context_size=context_size(prev.get("usage")) or 0,
        cache_read_tokens=(row.get("usage") or {}).get("cache_read_tokens") or 0,
        prev_cache_read_tokens=(prev.get("usage") or {}).get("cache_read_tokens") or 0,
        changed=changed_between(prev, row))


def format_drop(drop: CacheDrop) -> str:
    return (f"{drop.ts}  session={drop.session_id}  "
            f"cache_read {drop.prev_cache_read_tokens}->{drop.cache_read_tokens} tokens "
            f"of a ~{drop.prev_context_size}->{drop.context_size}-token prompt  "
            f"cause: {drop.cause}")


def report(rows: Sequence[dict]) -> str:
    """Part B.1 as plain text: one line per sharp cache-read fall, or one line
    saying none were found."""
    drops = find_drops(rows)
    if not drops:
        return "no sharp cache-read falls found in this log."
    lines = [f"{len(drops)} sharp cache-read fall(s):"]
    lines.extend(f"  {format_drop(drop)}" for drop in drops)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Part B.2 -- pricing the 5-hour (or 7-day) utilization step, per model.
# --------------------------------------------------------------------------- #
def _window_value(row: dict, window: str) -> Optional[float]:
    return row.get(f"{window}_utilization")


def _zero_totals() -> Dict[str, int]:
    return {name: 0 for name in USAGE_FIELDS}


def _add_usage(totals: Dict[str, int], usage: dict) -> Dict[str, int]:
    """New totals with ``usage``'s fields added in. Pure: never mutates ``totals``."""
    return {name: totals[name] + int(usage.get(name) or 0) for name in USAGE_FIELDS}


def _is_new_step(last_value: Optional[float], value: float) -> bool:
    return last_value is not None and value > last_value


def _is_rollover(last_value: Optional[float], value: float) -> bool:
    return last_value is not None and value < last_value


def _account_steps(rows: Sequence[dict],
                   window: str) -> Dict[str, List[Tuple[Tuple[int, ...], float]]]:
    """One account's rows, time-ordered: for each visible step of its unified
    counter, the usage summed since the previous visible value, paired with
    how far the counter moved -- keyed by whichever model was served in that
    stretch. A value lower than the one before it is the window rolling over,
    not a fall: the open stretch is discarded rather than attributed.

    The very first reading this account ever shows is a baseline, not a step:
    its own usage already explains how the counter reached that reading (a
    quantity this function never observed arriving), so it is discarded here
    rather than folded into whichever step comes next. Proved wrong once by a
    synthetic fit with known weights before this discard was added: without
    it, the first recorded step's usage silently included the baseline row's
    usage while its delta did not, overstating every weight in the fit.

    A step that shows before any row has named a model cannot be attributed,
    so its stretch is discarded too. Before that, such a step moved the
    baseline but kept its usage, which then inflated the next step the same
    way (found by R0-testing an AGORA concern on HAL#702).
    """
    ordered = sorted(rows, key=lambda r: r.get("ts") or "")
    steps: Dict[str, List[Tuple[Tuple[int, ...], float]]] = defaultdict(list)
    totals = _zero_totals()
    last_value: Optional[float] = None
    model_in_stretch: Optional[str] = None
    for row in ordered:
        if row.get("model"):
            model_in_stretch = row["model"]
        totals = _add_usage(totals, row.get("usage") or {})
        value = _window_value(row, window)
        if value is None:
            continue
        if last_value is None:
            last_value, totals = value, _zero_totals()
        elif _is_new_step(last_value, value):
            if model_in_stretch:
                usage_tuple = tuple(totals[name] for name in USAGE_FIELDS)
                steps[model_in_stretch].append((usage_tuple, value - last_value))
            last_value, totals = value, _zero_totals()
        elif _is_rollover(last_value, value):
            last_value, totals = value, _zero_totals()
        else:
            last_value = value
    return steps


def _all_model_steps(by_account: Dict[str, List[dict]],
                     window: str) -> Iterator[Tuple[str, List[Tuple[Tuple[int, ...], float]]]]:
    for account_rows in by_account.values():
        yield from _account_steps(account_rows, window).items()


def step_observations(rows: Sequence[dict], *,
                      window: str = "five_hour") -> Dict[str, List[Tuple[Tuple[int, ...], float]]]:
    """Every model's step observations, pooled across accounts.

    Each account's counter is read as its own monotonic time series first (so
    one account's reset never looks like a step in another's), then every
    account's per-model steps are pooled, since the cost of a token against the
    unified window should not depend on whose window it is.
    """
    by_account: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_account[row.get("account_label") or ""].append(row)
    pooled: Dict[str, List[Tuple[Tuple[int, ...], float]]] = defaultdict(list)
    for model, steps in _all_model_steps(by_account, window):
        pooled[model].extend(steps)
    return pooled


def _eliminated_row(row: List[float], pivot: List[float], pivot_col: int) -> List[float]:
    factor = row[pivot_col]
    return [a - factor * b for a, b in zip(row, pivot)]


def _eliminate(augmented: List[List[float]], pivot_col: int) -> List[List[float]]:
    """Every row but the pivot's loses its ``pivot_col``-th column's multiple of
    the (already-normalised) pivot row. Pure: returns a new matrix."""
    pivot = augmented[pivot_col]
    return [row if i == pivot_col else _eliminated_row(row, pivot, pivot_col)
            for i, row in enumerate(augmented)]


def _solve(matrix: List[List[float]], vector: List[float]) -> Optional[List[float]]:
    """``matrix @ x = vector`` by Gauss-Jordan elimination with partial
    pivoting. None when ``matrix`` is singular (within a small tolerance) --
    pure Python, no numpy: this stays in HAL's zero-dependency style for
    ``lib/hal``, and the system here is at most 4x4 (one unknown per usage
    field), never scaling with the number of data points."""
    n = len(vector)
    augmented = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(augmented[r][col]))
        if abs(augmented[pivot_row][col]) < 1e-9:
            return None
        augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]
        pivot = augmented[col][col]
        augmented[col] = [value / pivot for value in augmented[col]]
        augmented = _eliminate(augmented, col)
    return [row[-1] for row in augmented]


def _normal_equations(points: Sequence[Tuple[Tuple[float, ...], float]]
                      ) -> Tuple[List[List[float]], List[float]]:
    """``(X^T X, X^T y)`` for an intercept-free linear fit, built with one real
    loop over the data points; the per-point accumulation is comprehension-based
    so it stays flat regardless of how many usage fields there are."""
    width = len(points[0][0])
    xtx = [[0.0] * width for _ in range(width)]
    xty = [0.0] * width
    for x, y in points:
        xty = [total + xi * y for total, xi in zip(xty, x)]
        xtx = [[cell + xi * xj for cell, xj in zip(row, x)] for row, xi in zip(xtx, x)]
    return xtx, xty


def _residual_rms(points: Sequence[Tuple[Tuple[float, ...], float]],
                  weights: Sequence[float]) -> float:
    sse = sum((sum(w * xi for w, xi in zip(weights, x)) - y) ** 2 for x, y in points)
    return (sse / len(points)) ** 0.5


def _least_squares(points: Sequence[Tuple[Tuple[float, ...], float]]
                   ) -> Tuple[Optional[List[float]], Optional[float]]:
    """Ordinary least squares with no intercept: ``y = sum(w_i * x_i)``.
    Returns ``(weights, residual_rms)``, or ``(None, None)`` when the system is
    singular (e.g. a usage field that never varied across every observation)."""
    xtx, xty = _normal_equations(points)
    weights = _solve(xtx, xty)
    if weights is None:
        return None, None
    return weights, _residual_rms(points, weights)


@dataclass(frozen=True)
class UtilizationFit:
    """One model's fitted weights, or why it could not be fitted."""

    model: str
    n_points: int
    weights: Optional[Dict[str, float]]
    residual_rms: Optional[float]
    note: Optional[str] = None


def _fit_one_model(model: str, points: List[Tuple[Tuple[int, ...], float]],
                   floor: int) -> UtilizationFit:
    if len(points) < floor:
        return UtilizationFit(model=model, n_points=len(points), weights=None, residual_rms=None,
                              note=f"too few data points to fit ({len(points)} < {floor})")
    weights, residual = _least_squares(points)
    if weights is None:
        return UtilizationFit(model=model, n_points=len(points), weights=None, residual_rms=None,
                              note="usage did not vary independently enough to fit (degenerate)")
    return UtilizationFit(model=model, n_points=len(points),
                          weights=dict(zip(USAGE_FIELDS, weights)), residual_rms=residual)


def fit_utilization(rows: Sequence[dict], *, window: str = "five_hour",
                    min_points: Optional[int] = None) -> Dict[str, UtilizationFit]:
    """Part B.2: per model, the weight of each usage field on one percentage
    point of ``window``'s utilization, fitted over every visible step pooled
    across accounts. ``min_points`` (default: one more than the four usage
    fields) is the smallest number of steps this will fit rather than refuse;
    below it, or when the fields never varied independently enough to solve,
    the result says so in plain words instead of guessing.
    """
    floor = min_points if min_points is not None else len(USAGE_FIELDS) + 1
    pooled = step_observations(rows, window=window)
    return {model: _fit_one_model(model, points, floor) for model, points in pooled.items()}


def format_fit(fit: UtilizationFit) -> str:
    if fit.weights is None:
        return f"{fit.model}: {fit.note}"
    weights = ", ".join(f"{name}={value:.6f}" for name, value in fit.weights.items())
    return f"{fit.model}: n={fit.n_points} weights[{weights}] residual_rms={fit.residual_rms:.4f}"


# --------------------------------------------------------------------------- #
# The command line.
# --------------------------------------------------------------------------- #
def _read_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _cmd_report(rows: List[dict]) -> int:
    print(report(rows))
    return 0


def _cmd_fit(rows: List[dict], window: str) -> int:
    fits = fit_utilization(rows, window=window)
    if not fits:
        print(f"no {window} utilization readings in this log.")
        return 0
    for model in sorted(fits):
        print(format_fit(fits[model]))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log", help=f"the ledger file (default: $HAL_HOME/logs/{LOG_NAME})")
    parser = argparse.ArgumentParser(
        prog="python -m lib.hal.cache_loss",
        description="Explain and price the Claude Code prompt-cache ledger (Part A/B).")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("report", parents=[common],
                   help="name what changed before each sharp cache-read fall")
    fit_parser = sub.add_parser(
        "fit", parents=[common],
        help="fit the utilization step against summed usage, per model")
    fit_parser.add_argument("--window", choices=("five_hour", "seven_day"), default="five_hour")
    args = parser.parse_args(argv)
    rows = _read_rows(Path(args.log) if args.log else log_path())
    if args.command == "report":
        return _cmd_report(rows)
    return _cmd_fit(rows, args.window)


__all__ = [
    "CacheDrop",
    "LOG_NAME",
    "MAX_LOG_BYTES",
    "USAGE_FIELDS",
    "UtilizationFit",
    "append_row",
    "build_row",
    "cache_fraction",
    "changed_between",
    "context_size",
    "find_drops",
    "fit_utilization",
    "format_drop",
    "format_fit",
    "hash_hex",
    "log_path",
    "now_iso",
    "record_response",
    "report",
    "request_hashes",
    "step_observations",
    "usage_fields",
    "window_utilization",
]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
