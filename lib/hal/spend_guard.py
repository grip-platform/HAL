"""spend_guard -- the one place HAL decides whether a paid billing source may be used.

The InfiniteMachine contract (``docs/infinite-machine-contract.md``, "The spend
guard") says that HAL alone chooses a billing source, that the guard is
fail-closed and lives in one place, that HAL refuses every paid source without
explicit opt-in, that a paid key never counts as capacity, and that overage showing on a 2xx counts as spent. This
module is that guard. These billing paths ask it before a paid credential goes
upstream:

* ``GatewayProxy.forward`` (``lib/hal/gateway.py``), which every Anthropic-bound
  gateway request crosses, on both rails;
* the wire (``lib/hal/wire_proxy.py``), before it relays a decrypted request;
* ``AnthropicProvider._request`` (``lib/hal/providers/anthropic_native.py``),
  the native provider's API-key path, which hal-server took by itself whenever
  the subscription looked unhealthy;
* GRIP's ``bin/grip-auth-failover --engage`` and GRIP's relay, through
  ``lib/hal_spend_guard.py``, which loads this file by its path.

OpenAI HTTP and realtime transports also ask this guard at their resolved
host, through :func:`require_openai_destination`, which every OpenAI send path
calls before it presents a credential (HAL's tripwire test lists the callers).
Only ``GRIP_CODEX_ALLOW_PAID=1`` opts ``openai-api-key`` in; neither a
policy source entry nor ``*`` substitutes for it. A defective policy still
refuses it. Models served by other hosts are billed by those hosts.

Two rules, deliberately asymmetric.

**A paid source is fail-closed.** Paid means billed per use: an Anthropic API
key (``x-api-key``, or an ``sk-ant-`` bearer that is neither an OAuth token nor
Claude Code's session-ingress token), and
subscription traffic once a 2xx response from that account has shown paid extra
usage (overage). :func:`decide_paid` allows one only when the policy file reads
cleanly and the source is opted in. Any error at all -- a missing, unreadable or
malformed file, a bug -- refuses.

There is no date rule. The 2026-10-11 no-spend date was removed on 2026-10-06 at
the operator's order, after it refused an account whose extra usage he had just
paid for; a policy file that still carries ``no_spend_until`` is read without it.

**Subscription traffic is refused only on evidence.** That evidence is a 2xx
from the same account carrying the overage headers, and it holds until the reset
those headers named. A spent-ledger that cannot be read is not evidence: refusing
on it would end every session on the account without saving anything, and the
next response shows overage again if it is still in use.

What this guard does not cover, so that nobody infers coverage it lacks: a model
a caller asks for by name on a third-party provider (gemini, deepseek and the
rest) is served with that provider's own key and does not cross this guard, and
neither do Anthropic paths that send a key the caller configured explicitly
(``lib/hal/happi/providers/anthropic.py``, the Anthropic side of
``lib/hal/tool_runtime.py``). OpenAI destinations are guarded even with an
explicitly configured key. For Anthropic it
covers every path where HAL or GRIP chose the paid source on the operator's
behalf.

An account is named by a digest of its OAuth token, never the token. A refreshed
token therefore starts unmarked, and costs at most one more overage request
before its own response marks it; the per-account ledger of Step 5b will name
accounts properly.

**A source HAL meters itself also needs a daily cap.** Those sources are
``anthropic-api-key`` when HAL spends it for its own purposes
(``lib/hal/anthropic_key_route.py``), and ``typesafe-api-key``, TypeSafe's
prepaid decision API, which HAL prices from each reply's input tokens
(``lib/hal/decide/typesafe.py``). :func:`decide_within_cap` allows one only
when :func:`decide_paid` does AND today's (UTC) spend is under the source's
``daily_caps_usd`` entry. The policy ships ``anthropic-api-key`` at 0, so opting
it in alone spends nothing, and ``typesafe-api-key`` opted in at $0.10, the
operator's choice of 2026-10-17 (lowered from the $1 set on 2026-10-10). No cap, a malformed cap, a spend ledger that cannot be read, or a
day on which a response could not be priced all refuse. The cap is checked
before a call and the call's cost is added after it, so the call that crosses
the cap is the last one allowed that day; calls already in flight when it is
crossed still finish. The first call refused each day because spend reached a
cap above $0 writes one audit row (``$HAL_HOME/logs/spend-guard-audit.jsonl``), and
so does each response that could not be priced; a row is never a prompt, a body
or a key. :func:`decide_paid`, :func:`check_anthropic_request` and the wire never
read the cap, so a session's own sign-in is unaffected by it.
``python -m lib.hal.spend_guard caps`` prints each source's cap, today's spend
and whether it is opted in.

Standard library only, and nothing imported from ``lib.hal``: GRIP loads this file
by its path, in a process whose own ``lib/hal`` package is a different one.
``HAL_HOME`` and the policy path are read at call time, so a test's isolated home
is honoured (``tests/conftest.py``).
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import logging
import os
import sys
import threading
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Tuple
from urllib.parse import urlsplit

try:  # POSIX: serialise the daily ledger across processes (_ledger_lock)
    import fcntl
except ImportError:  # Windows: the per-process lock is all there is
    fcntl = None

_log = logging.getLogger("hal.spend_guard")

#: Overrides the policy file's path. A missing or malformed file there refuses
#: every paid source, exactly as the default path does.
CONFIG_ENV = "HAL_SPEND_GUARD_CONFIG"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "spend_guard.json"

#: Paid source names; OpenAI uses its literal environment switch below.
API_KEY = "anthropic-api-key"
OVERAGE = "anthropic-overage"
OPENAI_API_KEY = "openai-api-key"
#: TypeSafe's decision API (lib/hal/decide/typesafe.py): prepaid credits, metered here.
TYPESAFE_API_KEY = "typesafe-api-key"
OPENAI_PAID_ENV = "GRIP_CODEX_ALLOW_PAID"
#: In ``paid_opt_in``: every policy-controlled source (not OpenAI).
ALL_PAID = "*"

#: Credential kinds :func:`anthropic_credential` reports.
PAID = "paid"
SUBSCRIPTION = "subscription"

#: How long an account stays spent when the overage response named no reset:
#: the shorter of the two subscription windows.
DEFAULT_SPENT_SECONDS = 5 * 3600

_OAUTH_PREFIX = "sk-ant-oat"
_ANTHROPIC_PREFIX = "sk-ant-"
# Claude Code's session-ingress token, the bearer Remote Control's bridge sends:
# this prefix, then a JWT. It authenticates a session channel, not model usage.
# Read 2026-10-06 in Claude Code 2.1.289's own parser, which strips the prefix,
# needs split(".") to give three parts with a non-empty payload, and decodes that
# payload as base64url JSON. All three tests are mirrored here. Its later check
# for a session_id claim is not, so a renamed claim cannot refuse the bridge again.
_SESSION_INGRESS_PREFIX = "sk-ant-si-"
_BILLABLE_PATHS = frozenset({"/v1/messages", "/v1/complete"})
_BATCHES_PREFIX = "/v1/messages/batches"
# Verified 2026-10-05 in Claude Code 2.1.289's own response-header parser:
#   isUsingOverage: status === "rejected" && (overage === "allowed" || overage === "allowed_warning")
#   overageInUse:   headers.get("anthropic-ratelimit-unified-overage-in-use") === "true"
# The reset it pairs with the status is anthropic-ratelimit-unified-reset, in Unix seconds.
_STATUS = "anthropic-ratelimit-unified-status"
_OVERAGE_STATUS = "anthropic-ratelimit-unified-overage-status"
_OVERAGE_IN_USE = "anthropic-ratelimit-unified-overage-in-use"
_RESET = "anthropic-ratelimit-unified-reset"
_LEDGER_NAME = "spend-guard-spent.json"

_SPENT_LOCK = threading.Lock()
#: account key -> Unix seconds until which it is spent, held by this process.
_SPENT: dict = {}

#: Sources HAL prices and records itself, so a daily cap can bound them.
METERED = frozenset({API_KEY, TYPESAFE_API_KEY})
_DAILY_NAME = "spend-guard-daily.json"
_AUDIT_NAME = "spend-guard-audit.jsonl"
#: How many UTC days the spend ledger keeps; only today's decides anything.
_KEEP_DAYS = 31
_PLACES = Decimal("0.00000001")
#: Token counts kept per source and day, as ``lib/hal/infinite_cost.py`` names them.
_COUNTS = ("input_tokens", "output_tokens", "cache_read_tokens",
           "cache_write_tokens", "cache_write_1h_tokens")
_DAILY_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# The policy.
# --------------------------------------------------------------------------- #
class PolicyError(ValueError):
    """The policy file is missing, unreadable or malformed."""


@dataclass(frozen=True)
class Policy:
    """Which paid sources the operator opted in, and the daily caps set for them.

    ``caps_error`` holds why ``daily_caps_usd`` could not be read. It refuses only
    the capped decisions (:func:`decide_within_cap`), never :func:`decide_paid`,
    so a typo in a cap cannot end overage traffic on the wire.
    """

    paid_opt_in: frozenset
    daily_caps_usd: dict = field(default_factory=dict, hash=False)
    caps_error: str = ""

    def opted_in(self, name: str) -> bool:
        return name in self.paid_opt_in or ALL_PAID in self.paid_opt_in

    def cap(self, name: str) -> Optional[Decimal]:
        """``name``'s daily cap in US dollars, or None when none is set. Raises
        :class:`PolicyError` when ``daily_caps_usd`` itself is malformed."""
        if self.caps_error:
            raise PolicyError(self.caps_error)
        return self.daily_caps_usd.get(name)


def config_path() -> Path:
    """``$HAL_SPEND_GUARD_CONFIG``, else the repository's ``config/spend_guard.json``."""
    override = os.environ.get(CONFIG_ENV, "").strip()
    return Path(override).expanduser() if override else DEFAULT_CONFIG


def load_policy(path: Optional[Path] = None) -> Policy:
    """The policy at ``path`` (default :func:`config_path`). Raises :class:`PolicyError`."""
    target = path if path is not None else config_path()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise PolicyError(f"cannot read {target} ({type(exc).__name__})") from exc
    if not isinstance(raw, dict):
        raise PolicyError(f"{target} is not a JSON object")
    # A legacy "no_spend_until" key is ignored: the date rule is gone (module docstring).
    caps, caps_error = _parse_caps(raw.get("daily_caps_usd"), target)
    return Policy(_parse_opt_in(raw.get("paid_opt_in"), target), caps, caps_error)


def _parse_opt_in(value: Any, target: Path) -> frozenset:
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise PolicyError(f"{target}: paid_opt_in must be a list of source names")
    return frozenset(v.strip() for v in value)


def _parse_caps(value: Any, target: Path) -> Tuple[dict, str]:
    """``daily_caps_usd`` as ``({source: Decimal}, "")``, or ``({}, why)``.

    Never raises: a malformed cap must refuse the capped sources, not every paid
    source (see :class:`Policy`). Absent means no cap is set anywhere.
    """
    if value is None:
        return {}, ""
    if not isinstance(value, dict):
        return {}, f"{target}: daily_caps_usd must map source names to US dollars"
    caps = {}
    for name, amount in value.items():
        try:
            caps[_source_name(name)] = usd(amount)
        except ValueError as exc:
            return {}, f"{target}: daily_caps_usd: {exc}"
    return caps, ""


def _source_name(name: Any) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{name!r} is not a source name")
    return name.strip()


def usd(value: Any) -> Decimal:
    """A US dollar amount: a JSON number or a decimal string, finite and not negative.

    A float goes through ``str`` first, so ``5.1`` is exactly ``Decimal("5.1")``.
    Raises ``ValueError`` for anything else, ``True`` and ``NaN`` included.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise ValueError(f"{value!r} is not a US dollar amount")
    try:
        amount = Decimal(str(value).strip())
    except InvalidOperation as exc:
        raise ValueError(f"{value!r} is not a US dollar amount") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError(f"{value!r} is not a US dollar amount")
    return amount


def usd_text(amount: Decimal) -> str:
    """A dollar figure as a fixed eight-place string, the form the ledger keeps."""
    return format(amount.quantize(_PLACES), "f")


# --------------------------------------------------------------------------- #
# Decisions.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Decision:
    """The guard's answer. ``status`` and ``retry_after`` tell a server how to
    refuse: 403 for a paid key, 429 with the reset for a spent subscription."""

    allowed: bool
    source: str
    reason: str
    status: int = 403
    retry_after: Optional[int] = None


class SpendRefused(Exception):
    """A billing path asked, and the guard said no. Carries the :class:`Decision`."""

    def __init__(self, decision: Decision):
        super().__init__(decision.reason)
        self.decision = decision


def _clock() -> datetime:
    """The real clock. A module function, so a test can freeze time; nothing a
    caller sets in its environment can move it."""
    return datetime.now(timezone.utc)


def _utc_now(now: Optional[datetime]) -> datetime:
    """``now`` in UTC; :func:`_clock` when None. A naive datetime is read as UTC."""
    current = now if now is not None else _clock()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc)


def _refused(name: str, why: str) -> Decision:
    return Decision(False, name, f"HAL spend guard refused '{name}': {why}")


def decide_paid(name: str, *, now: Optional[datetime] = None,
                policy: Optional[Policy] = None) -> Decision:
    """May the paid source ``name`` be used now? Fail-closed on every error."""
    try:
        pol = policy if policy is not None else load_policy()
        # OpenAI shares GRIP's literal switch. A policy wildcard never opts it
        # in, but the policy must still load cleanly before the switch is read.
        if name == OPENAI_API_KEY:
            if os.environ.get(OPENAI_PAID_ENV) != "1":
                return _refused(name, f"set {OPENAI_PAID_ENV}=1 to permit per-token billing")
            return Decision(True, name, f"'{name}' is opted in by {OPENAI_PAID_ENV}=1")
        if not pol.opted_in(name):
            return _refused(name, f"it is not in paid_opt_in ({config_path()})")
        return Decision(True, name, f"'{name}' is opted in ({config_path()})")
    except PolicyError as exc:
        return _refused(name, f"the policy cannot be used ({exc}), so every paid source is refused")
    except Exception as exc:  # noqa: BLE001 -- fail-closed is the contract
        return _refused(name, f"the guard failed ({type(exc).__name__}), so paid use is refused")


def require_paid(name: str, *, now: Optional[datetime] = None,
                 policy: Optional[Policy] = None) -> Decision:
    """:func:`decide_paid`, raising :class:`SpendRefused` on a refusal."""
    verdict = decide_paid(name, now=now, policy=policy)
    if not verdict.allowed:
        raise SpendRefused(verdict)
    return verdict


def require_openai_destination(url: str) -> None:
    """Guard the resolved billing host, regardless of provider alias or model maker.

    These transports use API keys, never ChatGPT subscription credentials.
    Call before a credential is presented to HTTP or WebSocket transports.
    """
    if (urlsplit(url).hostname or "").rstrip(".") == "api.openai.com":
        require_paid(OPENAI_API_KEY)


# --------------------------------------------------------------------------- #
# Anthropic credentials and requests.
# --------------------------------------------------------------------------- #
def _pairs(headers: Any) -> Iterable[Tuple[Any, Any]]:
    items = headers.items() if callable(getattr(headers, "items", None)) else (headers or ())
    for item in items:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            yield item[0], item[1]


def _header(headers: Any, name: str) -> str:
    """One header's values joined by ``, `` (the name is case-insensitive).

    A twin of ``lib.hal.wire_classify.header_value``, kept here because this
    module imports nothing from ``lib.hal`` (module docstring).
    """
    wanted = name.lower()
    return ", ".join(str(value) for key, value in _pairs(headers) if str(key).lower() == wanted)


def account_key(token: str) -> str:
    """A stable, non-secret name for the account behind ``token``: 16 hex
    characters of its SHA-256. The token itself is never kept."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def anthropic_credential(headers: Any) -> Tuple[str, str]:
    """How an outgoing api.anthropic.com request is billed, as ``(kind, account)``.

    ``x-api-key`` is Anthropic's API-key scheme, so any value there is
    ``("paid", "")``. A bearer is the subscription when it is an OAuth token
    (``sk-ant-oat``), paid when it is any other ``sk-ant-`` token, and otherwise
    not a credential Anthropic bills at all (a session token, or a placeholder
    Anthropic rejects): ``("", "")``. Claude Code's session-ingress token carries
    ``sk-ant-`` too, so it is recognised first and is not billed.
    """
    if _header(headers, "x-api-key").strip():
        return PAID, ""
    auth = _header(headers, "authorization").strip()
    token = auth[7:].strip() if auth[:7].lower() == "bearer " else ""
    if token.startswith(_OAUTH_PREFIX):
        return SUBSCRIPTION, account_key(token)
    if _session_ingress(token):
        return "", ""
    if token.startswith(_ANTHROPIC_PREFIX):
        return PAID, ""
    return "", ""


def _session_ingress(token: str) -> bool:
    """Is ``token`` Claude Code's session-ingress token, by Claude Code's own tests?

    The prefix, then three dot-separated parts whose payload decodes to a JSON
    object. Any other ``sk-ant-`` token stays paid, so an unknown or malformed one,
    an undecodable payload included, still fails closed.
    """
    if not token.startswith(_SESSION_INGRESS_PREFIX):
        return False
    parts = token[len(_SESSION_INGRESS_PREFIX):].split(".")
    if len(parts) != 3 or not parts[1]:
        return False
    try:
        claims = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
    except ValueError:  # bad base64, UTF-8 and JSON all raise a ValueError
        return False
    return isinstance(claims, dict)


def billable(method: str, path: str) -> bool:
    """Does this request spend model usage? A POST to messages, complete or batches."""
    if str(method).upper() != "POST":
        return False
    target = str(path)
    bare = urlsplit(target).path if "://" in target else target.split("?", 1)[0]
    return bare in _BILLABLE_PATHS or bare.startswith(_BATCHES_PREFIX)


def check_anthropic_request(method: str, path: str, headers: Any, *,
                            now: Optional[datetime] = None,
                            policy: Optional[Policy] = None) -> Optional[Decision]:
    """The refusal for one outgoing api.anthropic.com request, or None to send it.

    A paid credential is refused unless :func:`decide_paid` allows it. A
    subscription request that spends usage is refused while its account is
    spent (an earlier 2xx showed overage), unless overage is allowed. Anything
    else is sent: no credential, or one Anthropic does not bill.
    """
    try:
        kind, account = anthropic_credential(headers)
    except Exception as exc:  # noqa: BLE001 -- unreadable means it may be paid
        return _refused(API_KEY, f"the request's credential could not be read "
                                 f"({type(exc).__name__}), so it is treated as paid")
    if kind == PAID:
        verdict = decide_paid(API_KEY, now=now, policy=policy)
        return None if verdict.allowed else verdict
    if kind == SUBSCRIPTION and billable(method, path):
        return _check_spent(account, now, policy)
    return None


def _check_spent(account: str, now: Optional[datetime],
                 policy: Optional[Policy]) -> Optional[Decision]:
    current = _utc_now(now).timestamp()
    until = spent_until(account, now=now)
    if until is None:
        return None
    verdict = decide_paid(OVERAGE, now=now, policy=policy)
    if verdict.allowed:
        return None
    reset = datetime.fromtimestamp(until, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return replace(verdict, status=429, retry_after=max(1, int(until - current)),
                   reason=(f"{verdict.reason}. This account's subscription limit is spent "
                           f"until {reset}, and until then its traffic bills paid extra "
                           f"usage (overage)"))


# --------------------------------------------------------------------------- #
# Overage, and the accounts it has spent.
# --------------------------------------------------------------------------- #
def overage_in_use(headers: Any) -> bool:
    """Did this response bill paid extra usage? (f22, verified: see the constants.)"""
    if _header(headers, _OVERAGE_IN_USE).strip().lower() == "true":
        return True
    status = _header(headers, _STATUS).strip().lower()
    overage = _header(headers, _OVERAGE_STATUS).strip().lower()
    return status == "rejected" and overage in ("allowed", "allowed_warning")


def _reset_epoch(headers: Any, current: float) -> float:
    """The reset the response named, or ``DEFAULT_SPENT_SECONDS`` from now."""
    try:
        reset = float(_header(headers, _RESET).strip())
    except ValueError:
        reset = 0.0
    return reset if reset > current else current + DEFAULT_SPENT_SECONDS


def observe_anthropic_response(status: Any, response_headers: Any, request_headers: Any, *,
                               now: Optional[datetime] = None) -> Optional[float]:
    """Mark the request's account spent when a 2xx shows overage.

    Returns the Unix second until which the account is spent, or None. Never
    raises: an observation must never break the relay that called it.
    """
    try:
        if not 200 <= int(status) < 300 or not overage_in_use(response_headers):
            return None
        kind, account = anthropic_credential(request_headers)
        if kind != SUBSCRIPTION:
            return None
        until = _reset_epoch(response_headers, _utc_now(now).timestamp())
        mark_spent(account, until)
        _log.warning("a 2xx showed paid extra usage (overage); the account is spent until %s",
                     datetime.fromtimestamp(until, timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
        return until
    except Exception:  # noqa: BLE001 -- see the docstring
        return None


def _hal_home() -> Path:
    """``$HAL_HOME``, else ``~/.hal``, read at call time (module docstring)."""
    return Path(os.environ.get("HAL_HOME") or os.path.expanduser("~/.hal"))


def ledger_path() -> Path:
    """``$HAL_HOME/state/hal/spend-guard-spent.json``, resolved at call time."""
    return _hal_home() / "state" / "hal" / _LEDGER_NAME


def _read_ledger(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): float(v) for k, v in raw.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _persist(account: str, until: float) -> None:
    """Merge one mark into the ledger file, dropping marks already past."""
    path = ledger_path()
    try:
        current = _utc_now(None).timestamp()
        merged = {k: v for k, v in _read_ledger(path).items() if v > current}
        merged[account] = max(until, merged.get(account, 0.0))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(merged, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        _log.warning("the spend guard could not write its ledger; the mark holds in this process")


def mark_spent(account: str, until: float) -> None:
    """Record that ``account`` bills overage until ``until`` (Unix seconds)."""
    with _SPENT_LOCK:
        _SPENT[account] = max(until, _SPENT.get(account, 0.0))
        _persist(account, _SPENT[account])


def spent_until(account: str, *, now: Optional[datetime] = None) -> Optional[float]:
    """Until when ``account`` is spent, or None. Reads this process's marks and the
    ledger file; a ledger that cannot be read counts as holding no marks."""
    current = _utc_now(now).timestamp()
    with _SPENT_LOCK:
        held = _SPENT.get(account, 0.0)
    until = max(held, _read_ledger(ledger_path()).get(account, 0.0))
    return until if until > current else None


# --------------------------------------------------------------------------- #
# The daily cap, for the sources HAL meters itself (module docstring).
# --------------------------------------------------------------------------- #
#: (source, day) whose spend this process could not write down: refused that day.
_UNRECORDED: set = set()


def today_utc(now: Optional[datetime] = None) -> date:
    """The UTC date of ``now`` (default the guard's clock): the day a cap counts."""
    return _utc_now(now).date()


def daily_ledger_path() -> Path:
    """``$HAL_HOME/state/hal/spend-guard-daily.json``, resolved at call time."""
    return _hal_home() / "state" / "hal" / _DAILY_NAME


def audit_path() -> Path:
    """``$HAL_HOME/logs/spend-guard-audit.jsonl``, resolved at call time."""
    return _hal_home() / "logs" / _AUDIT_NAME


def _blank_entry() -> dict:
    entry = {"spent_usd": usd_text(Decimal(0)), "calls": 0, "unknown": 0, "cap_noted": False}
    entry.update(dict.fromkeys(_COUNTS, 0))
    return entry


def _read_days(path: Path) -> Optional[dict]:
    """The ledger's days: ``{}`` when there is no ledger yet, None when it cannot be read."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    days = raw.get("days") if isinstance(raw, dict) else None
    return days if isinstance(days, dict) else None


def _entry(days: dict, day: str, name: str) -> Optional[dict]:
    """``name``'s entry for ``day`` with every field checked; None when it is malformed."""
    rows = days.get(day, {})
    stored = rows.get(name, {}) if isinstance(rows, dict) else None
    if not isinstance(stored, dict):
        return None
    entry = {**_blank_entry(), **stored}
    if not all(type(entry[key]) is int and entry[key] >= 0 for key in ("calls", "unknown") + _COUNTS):
        return None
    try:
        entry["spent_usd"] = usd_text(usd(entry["spent_usd"]))
    except ValueError:
        return None
    return entry


def _today_entry(name: str, day: str) -> Optional[dict]:
    days = _read_days(daily_ledger_path())
    return None if days is None else _entry(days, day, name)


@contextlib.contextmanager
def _ledger_lock(path: Path) -> Iterator[None]:
    """This process's lock and, on POSIX, an exclusive lock on a sibling file, so
    hal-server, a Happi process and a CLI never lose each other's spend."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _DAILY_LOCK, open(path.with_name(path.name + ".lock"), "a", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def _write_days(path: Path, days: dict) -> None:
    kept = dict(sorted(days.items())[-_KEEP_DAYS:])
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps({"days": kept}, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _update_entry(name: str, day: str, change: Callable[[dict], None]) -> Optional[dict]:
    """Apply ``change`` to ``name``'s entry for ``day`` under the lock; the new entry.

    An entry that cannot be read is replaced by one marking the day's spend
    unknown, which refuses ``name`` until midnight UTC. When the ledger cannot be
    written at all, this process remembers it and refuses ``name`` for the day.
    """
    path = daily_ledger_path()
    try:
        with _ledger_lock(path):
            days = _read_days(path)
            entry = None if days is None else _entry(days, day, name)
            if entry is None:
                days, entry = days or {}, {**_blank_entry(), "unknown": 1}
            change(entry)
            rows = days.get(day)
            days[day] = {**(rows if isinstance(rows, dict) else {}), name: entry}
            _write_days(path, days)
            return entry
    except OSError:
        _UNRECORDED.add((name, day))
        _log.warning("the spend guard could not write %s; '%s' is refused for %s", path, name, day)
        return None


def _token_count(usage: Optional[dict], key: str) -> int:
    value = usage.get(key) if isinstance(usage, dict) else None
    return value if type(value) is int and value >= 0 else 0


def record_spend(name: str, amount: Any, *, usage: Optional[dict] = None,
                 now: Optional[datetime] = None) -> Optional[Decimal]:
    """Add one priced response to today's spend on ``name``.

    ``usage`` is a row usage dict (``lib/hal/infinite_cost.py``); its token counts
    are kept beside the dollars. Returns today's new total, or None when it could
    not be written down (``name`` is then refused for the rest of the day).
    """
    cost = usd(amount)

    def change(entry: dict) -> None:
        entry["spent_usd"] = usd_text(usd(entry["spent_usd"]) + cost)
        entry["calls"] += 1
        for key in _COUNTS:
            entry[key] += _token_count(usage, key)

    entry = _update_entry(name, today_utc(now).isoformat(), change)
    return None if entry is None else usd(entry["spent_usd"])


def record_unknown(name: str, why: str, *, now: Optional[datetime] = None) -> None:
    """A response on ``name`` whose cost cannot be counted. Today's spend is then
    not known, so ``name`` is refused until midnight UTC; one audit row says why."""
    day = today_utc(now).isoformat()

    def change(entry: dict) -> None:
        entry["calls"] += 1
        entry["unknown"] += 1

    _update_entry(name, day, change)
    _audit({"event": "spend_unknown", "source": name, "day": day, "reason": why})


def decide_within_cap(name: str, *, now: Optional[datetime] = None,
                      policy: Optional[Policy] = None) -> Decision:
    """:func:`decide_paid`, and today's (UTC) spend on ``name`` under its daily cap.

    For the sources HAL meters itself. Fail-closed on every error, like the opt-in.
    """
    verdict = decide_paid(name, now=now, policy=policy)
    if not verdict.allowed:
        return verdict
    try:
        pol = policy if policy is not None else load_policy()
        return _cap_verdict(name, pol, today_utc(now).isoformat())
    except PolicyError as exc:
        return _refused(name, f"its daily cap cannot be used ({exc}), so it is refused")
    except Exception as exc:  # noqa: BLE001 -- fail-closed is the contract
        return _refused(name, f"the daily cap could not be checked ({type(exc).__name__}), "
                              "so paid use is refused")


def _cap_verdict(name: str, pol: Policy, day: str) -> Decision:
    cap = pol.cap(name)
    if cap is None:
        return _refused(name, f"it has no daily cap in daily_caps_usd ({config_path()})")
    if cap == 0:
        return _refused(name, f"its daily cap is $0 in daily_caps_usd ({config_path()}), "
                              "so nothing may be spent on it")
    spent, unknown = _spent_on(name, day)
    if unknown:
        return _refused(name, unknown)
    if spent >= cap:
        _note_cap_reached(name, day, spent, cap)
        return _refused(name, f"its daily cap of ${cap} is reached (${usd_text(spent)} spent on "
                              f"{day} UTC), so it is refused until 00:00 UTC")
    return Decision(True, name, f"'{name}' is opted in and under its daily cap "
                                f"(${usd_text(spent)} of ${cap} on {day} UTC)")


def _spent_on(name: str, day: str) -> Tuple[Decimal, str]:
    """``(spent, "")`` for ``name`` on ``day``, or ``(0, why it is not known)``."""
    if (name, day) in _UNRECORDED:
        return Decimal(0), ("this process could not write down today's spend, so it is "
                            "refused until 00:00 UTC")
    entry = _today_entry(name, day)
    if entry is None:
        return Decimal(0), f"today's spend cannot be read ({daily_ledger_path()}), so it is refused"
    if entry["unknown"]:
        return Decimal(0), ("a response today could not be priced, so today's spend is not "
                            "known and it is refused until 00:00 UTC")
    return usd(entry["spent_usd"]), ""


def _note_cap_reached(name: str, day: str, spent: Decimal, cap: Decimal) -> None:
    """One audit row per source and UTC day, on the first call the cap refuses."""
    first: list = []

    def change(entry: dict) -> None:
        if not entry["cap_noted"]:
            entry["cap_noted"] = True
            first.append(entry["calls"])

    _update_entry(name, day, change)
    if first:
        _audit({"event": "daily_cap_reached", "source": name, "day": day,
                "spent_usd": usd_text(spent), "cap_usd": usd_text(cap), "calls": first[0]})


def _audit(row: dict) -> None:
    """Append one row to the audit log. Callers pass names, days, dollar figures and
    counts only: never a prompt, a body or a key. Never raises."""
    path = audit_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        stamped = {"ts": _utc_now(None).strftime("%Y-%m-%dT%H:%M:%SZ"), **row}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(stamped, sort_keys=True) + "\n")
    except OSError:
        _log.warning("the spend guard could not write its audit row to %s", path)


# --------------------------------------------------------------------------- #
# Status: python -m lib.hal.spend_guard caps
# --------------------------------------------------------------------------- #
def caps_report(*, now: Optional[datetime] = None) -> dict:
    """Each paid source: opted in or not, its daily cap, and today's spend on it.

    Raises :class:`PolicyError` when the policy cannot be used.
    """
    pol = load_policy()
    day = today_utc(now).isoformat()
    names = sorted({API_KEY, OVERAGE, OPENAI_API_KEY, TYPESAFE_API_KEY}
                   | (set(pol.paid_opt_in) - {ALL_PAID})
                   | set(pol.daily_caps_usd))
    return {"day": day, "policy": str(config_path()), "ledger": str(daily_ledger_path()),
            "caps_error": pol.caps_error,
            "sources": [_report_row(name, pol, day, now) for name in names]}


def _report_row(name: str, pol: Policy, day: str, now: Optional[datetime]) -> dict:
    cap = None if pol.caps_error else pol.daily_caps_usd.get(name)
    return {"source": name, "opted_in": decide_paid(name, now=now, policy=pol).allowed,
            "cap_usd": None if cap is None else str(cap), "metered": name in METERED,
            "today": _today_entry(name, day) if name in METERED else None}


def render_caps(report: dict) -> str:
    """The report as plain text, one line per source."""
    lines = [f"HAL spend guard, {report['day']} (UTC)",
             f"policy:       {report['policy']}",
             f"spend ledger: {report['ledger']}"]
    if report["caps_error"]:
        lines.append(f"daily_caps_usd cannot be used, so every capped source is refused: "
                     f"{report['caps_error']}")
    lines += ["", f"{'source':<20}{'opted in':<10}{'daily cap':<12}{'spent today':<15}"
                  f"{'calls':<7}cache-read tokens"]
    lines += [_render_row(row) for row in report["sources"]]
    lines += ["", "openai-api-key is opted in only by GRIP_CODEX_ALLOW_PAID=1 in the "
                  "calling process's environment, as read here."]
    return "\n".join(lines)


def _render_row(row: dict) -> str:
    cap = "none" if row["cap_usd"] is None else f"${row['cap_usd']}"
    head = f"{row['source']:<20}{'yes' if row['opted_in'] else 'no':<10}{cap:<12}"
    today = row["today"]
    if not row["metered"]:
        # A cap only means something where HAL counts the spend (METERED).
        unenforced = "" if row["cap_usd"] is None else "; this cap is not enforced"
        return head + "not metered by HAL" + unenforced
    if today is None:
        return head + "the spend ledger cannot be read"
    spent = "unknown" if today["unknown"] else f"${usd(today['spent_usd']):.4f}"
    return head + f"{spent:<15}{today['calls']:<7}{today['cache_read_tokens']}"


def main(argv: Optional[list] = None) -> int:
    """``python -m lib.hal.spend_guard caps``: 0 on success, 1 when the policy cannot
    be used, 2 on a usage error."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args != ["caps"]:
        print("usage: python -m lib.hal.spend_guard caps", file=sys.stderr)
        return 2
    try:
        report = caps_report()
    except PolicyError as exc:
        print(f"HAL spend guard: the policy cannot be used ({exc}), so every paid source "
              "is refused", file=sys.stderr)
        return 1
    print(render_caps(report))
    return 0


__all__ = [
    "ALL_PAID", "API_KEY", "CONFIG_ENV", "DEFAULT_CONFIG", "DEFAULT_SPENT_SECONDS",
    "Decision", "METERED", "OVERAGE", "PAID", "Policy", "PolicyError", "SUBSCRIPTION",
    "SpendRefused", "account_key", "anthropic_credential", "audit_path", "billable",
    "caps_report", "check_anthropic_request", "config_path", "daily_ledger_path",
    "decide_paid", "decide_within_cap", "ledger_path", "load_policy", "main",
    "mark_spent", "observe_anthropic_response", "overage_in_use", "record_spend",
    "record_unknown", "render_caps", "require_paid", "spent_until", "TYPESAFE_API_KEY",
    "today_utc", "usd", "usd_text",
]


if __name__ == "__main__":
    sys.exit(main())
