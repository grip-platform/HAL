"""HAL Gateway — chat-rail WITNESS + SIGN (the VeriSign-of-AI moat foundation).

The ``/v1/chat/completions`` rail — the one EVERY OpenAI-wire harness (Codex /
Aider / opencode via the grip-harness adapter) hits — was the gateway's single
unwitnessed surface. Its two handler paths in ``gateway_openai`` both miss the
failover-layer audit (``FailoverProxy._record``):

  * **cross/same-wire** routes through :func:`gateway_egress.forward_same_wire`
    / ``relay_response`` and NEVER calls ``proxy.forward`` → zero audit rows.
    The SERVED non-Anthropic provider was completely invisible.
  * **verbatim Anthropic** calls ``proxy.forward("/v1/messages", ...)`` → a row
    IS written, but its ``path`` is ``/v1/messages``, so the chat rail is
    mislabelled, never witnessed AS chat.

So HAL did not witness its most-used rail (live evidence: 1× ``/v1/chat/
completions`` vs 285× ``/v1/messages`` in ``logs/gateway-audit.jsonl``).
You cannot anchor what you do not witness — this module closes that hole.

Design (witness → sign, the two load-bearing links of witness→sign→anchor):

* **WITNESS** — :func:`witness_chat` emits ONE audit row at the chat-handler
  boundary, decoupled from ``proxy.forward`` (the cross-wire path never calls
  it). The row is the SAME shape as ``FailoverProxy._record``
  (``event/path/upstream/status/served/failed_over/latency_ms`` + the ``ts``
  ``GatewayTelemetry.record`` prepends) PLUS the two fields the moat needs:
  ``provider`` (the SERVED provider — the real non-Anthropic name on the
  cross-wire path, else ``"anthropic"``) and ``model`` (the resolved served
  model id). It routes through the existing :class:`GatewayTelemetry` sink, so
  the unsigned row lands in the same ``gateway-audit.jsonl`` as every other
  rail — now correctly labelled ``path=/v1/chat/completions``.

* **SIGN** — the SAME record is appended to an HMAC-chained, append-only,
  tamper-evident log via HAL's own zero-dependency signer
  (``lib/hal/voice/audit.py::AuditLogger``; ``cryptography``-based Ed25519/
  ML-DSA from happi-anchor-pilot is deliberately NOT used — it is absent from
  HAL's venv). Each signed record chains to the previous via ``prev_hmac``, so
  any after-the-fact edit breaks ``verify_chain`` from that point on. This is
  the witness→sign link; the full Merkle-batch + offline-verify + Bitcoin
  anchor remains happi-anchor-pilot's job — this leaves an *anchorable signed
  record* per chat call.

  **DEPRECATED (InfiniteMachine v2, Step 4).** This HMAC chain's key sits
  beside its log, so anyone who can write the file can re-sign it. The signed
  record is now the thread row below; this chain and its key stay in place
  until the operator removes them in a later change.

* **SIGNED THREAD ROW (Step 4)** -- the same exchange also becomes one row in
  :mod:`lib.hal.infinite_thread`: signed with this host's PRIVATE key by GRIP's
  Ed25519 + ML-DSA-65 signer, chained per thread, and checkable with the
  PUBLIC key alone. A turn is a ``served`` row and a non-turn on the messages
  rail (a quota check, a token count, a title call) an ``aux`` row. HAL sees no
  conversation key today, so the thread is keyed by Claude Code's
  ``x-claude-code-session-id`` header when it is present and well-formed, and
  otherwise by this gateway process's launch id (``LAUNCH_SESSION_ID``) -- so a
  chat-rail harness that sends no session id shares one thread per gateway
  launch. Only model, provider, status, timing, the route class and the three
  cost figures (observed usage, list price, actual spend:
  :mod:`lib.hal.infinite_cost`) are put on the row; headers never reach it.
  On the messages rail the row is written after the relay, so a streamed
  reply's usage is in it. When GRIP's signer cannot run here, no row is
  written and the reason is logged once.

* **Fail-open (monotone-toward-safe, mirrors the whole gateway).** Neither the
  witness nor the signing may EVER break a request: both are wrapped so any
  telemetry / disk / key failure is swallowed. Observability never subtracts
  availability. The witness lives entirely outside the forward path.

Back-compat: no existing caller's shape changes. The unsigned audit row gains
two NEW keys (``provider``/``model``) that the failover-layer rows never carry —
purely additive. Tests in ``tests/test_gateway_witness.py`` are Goodhart-proof:
a chat call MUST produce a signed row whose model verifies; removing the emit
makes them fail (mutation-sensitive); the signature is never weakened.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from lib.hal.gateway_failover import GatewayTelemetry

# The signed, HMAC-chained chat-witness log + its key live beside the unsigned
# gateway-audit.jsonl under $HAL_HOME/logs. Separate file so the signed chain is
# self-contained (verify_chain reads one chain) and never interleaves with the
# multi-rail unsigned audit.
_SIGNED_LOG_NAME = "gateway-chat-signed.jsonl"
_SIGNED_KEY_NAME = "gateway-chat-audit.key"
_WITNESS_EVENT = "gateway.chat.witness"

# The default Anthropic-native served provider for the verbatim path. The
# cross-wire path passes the real provider name explicitly.
_ANTHROPIC = "anthropic"


def _hal_home() -> Path:
    return Path(os.environ.get("HAL_HOME", str(Path.home() / ".hal")))


def signed_log_path() -> Path:
    """Path of the HMAC-chained, append-only chat-witness log.

    DEPRECATED (Step 4): see the module docstring; the signed record is the
    thread row in :mod:`lib.hal.infinite_thread`.
    """
    return _hal_home() / "logs" / _SIGNED_LOG_NAME


def signed_key_path() -> Path:
    """Path of the chat-witness HMAC key (auto-generated on first sign).

    DEPRECATED (Step 4): this key sits beside the log it signs, so it cannot
    show who wrote a row. Kept until the operator removes it.
    """
    return _hal_home() / "logs" / _SIGNED_KEY_NAME


#: The normalised token fields copied onto a witness row, in report order.
_USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_tokens",
               "cache_write_tokens", "reasoning_tokens")

#: Keys a caller's ``extra`` may never set, even where the record lacks them.
#: The measurement fields are absent exactly when nothing was measured, so an
#: ``extra`` supplying them would turn "not measured" into an invented number;
#: and the audit sink (``GatewayTelemetry.record``) stamps ``ts`` itself, where
#: a caller's value would win.
_RESERVED_EXTRA = frozenset({"ts", "cost_usd", *_USAGE_KEYS})


#: The thread key for traffic that names no session: one per gateway process.
LAUNCH_SESSION_ID = "launch-" + uuid.uuid4().hex[:16]

#: Claude Code's per-session header (seen in the wire-matrix recordings,
#: ``tools/wire-matrix/fixtures/claude/S0/stub.jsonl``, header names only).
SESSION_HEADER = "x-claude-code-session-id"


def session_id_from_headers(headers: Any) -> Optional[str]:
    """Claude Code's session id, when the request carries one the store accepts."""
    from lib.hal.infinite_thread import is_session_id  # lazy: keeps imports light
    from lib.hal.wire_classify import header_value

    value = header_value(headers, SESSION_HEADER).strip()
    return value if is_session_id(value) else None


#: Claude Code's per-prompt header. The 2.1.283 release note: "Added
#: `x-claude-code-prompt-id` to the gateway hint headers so LLM gateways can
#: group the requests that serve one user prompt". One prompt is several
#: requests (the turn, each tool-result round, a title or a token count), so
#: this is the key that joins a wire row to the turn it served, and later an
#: in-process witness of that turn to the network record of the same requests
#: (GRIP's PR 7 of the 2.1.282-2.1.296 absorption). OPT-IN on the client,
#: measured 2026-10-17 on Claude Code 2.1.301 against a loopback stub: with
#: ``CLAUDE_CODE_GATEWAY_HINT_HEADERS`` unset the client sends only the session
#: id; set to ``1`` it also sends this header (a UUID) and
#: ``x-claude-code-request-class``. A row without ``prompt_id`` therefore means
#: the session did not opt in, not that the wire dropped it. Seen on the wire in
#: the wire-matrix recordings of 2026-10-07
#: (``tools/wire-matrix/fixtures/holds-off/claude/S3/stub.jsonl``, header names only).
PROMPT_HEADER = "x-claude-code-prompt-id"


def prompt_id_from_headers(headers: Any) -> Optional[str]:
    """Claude Code's prompt id, when the request carries one in the store's safe set.

    The same acceptance rule as a session id (up to 128 characters of a safe
    set): the value rides on a signed row and in the cache-loss ledger, so
    anything else is dropped rather than written.
    """
    from lib.hal.infinite_thread import is_session_id  # lazy: keeps imports light
    from lib.hal.wire_classify import header_value

    value = header_value(headers, PROMPT_HEADER).strip()
    return value if is_session_id(value) else None


def append_thread_row(record: dict, *, row_class: str, route: str,
                      session_id: Optional[str] = None, usage: Optional[dict] = None,
                      billing: str = "unknown"):
    """Write the signed thread row for one witnessed exchange; never raises.

    The row takes the record's fields the store allows (so a field the store
    adds is never silently dropped here), the route class, and the three cost
    figures from :func:`lib.hal.infinite_cost.cost_fields`: ``usage`` as
    observed (``None`` when nothing was measured), its list price, and the
    actual spend for ``billing``. Returns the store's ``Appended`` result, or
    None if the store could not be reached at all. A refusal (no GRIP signer
    here) is in the result and in the log, once; the request it describes is
    never touched.
    """
    try:
        from lib.hal import infinite_cost, infinite_thread

        fields = {name: value for name, value in record.items()
                  if name in infinite_thread.PAYLOAD_FIELDS}
        fields["route"] = route
        fields.update(infinite_cost.cost_fields(
            str(record.get("model", "")), usage, billing=billing))
        return infinite_thread.default_store().append(
            row_class, fields, session_id=session_id or LAUNCH_SESSION_ID)
    except Exception as exc:  # noqa: BLE001 - the witness never breaks a request
        note_thread_failure(exc)
        return None


@dataclass
class PendingRow:
    """A thread row that waits for its response's usage, then writes itself.

    The messages rail witnesses before it relays, but the usage of a streamed
    reply is known only once the stream has passed, so the row is written
    after the relay (:func:`lib.hal.gateway._relay_witnessed`).
    """

    record: dict
    row_class: str
    route: str
    session_id: Optional[str]
    billing: str

    def write(self, usage: Optional[dict]):
        return append_thread_row(self.record, row_class=self.row_class,
                                 route=self.route, session_id=self.session_id,
                                 usage=usage, billing=self.billing)


def note_thread_failure(exc: Exception) -> None:
    """Say once, in the log, that a thread row could not even be built.

    The gateway calls it too, when its own witnessing step fails, so no row is
    ever dropped without a line in the log.
    """
    try:
        from lib.hal.infinite_thread import warn_once

        warn_once(f"the witness could not build a thread row: {type(exc).__name__}")
    except Exception:  # noqa: BLE001 - logging must not break a request either
        pass


def build_chat_record(
    *,
    provider: str,
    model: str,
    status: Optional[int],
    served: bool,
    failed_over: bool,
    latency_ms: int,
    path: str,
    usage: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Build the chat-rail witness record (pure — no I/O, fully testable).

    Same shape as ``FailoverProxy._record`` (event/path/upstream/status/served/
    failed_over/latency_ms) PLUS the moat fields ``provider`` (SERVED provider)
    and ``model`` (resolved served model id). ``upstream`` mirrors the served
    provider so the row is self-describing alongside the failover-layer rows.

    ``usage`` — normalised token counts from
    ``gateway_crosswire.normalise_provider_usage`` — adds
    ``input_tokens`` / ``output_tokens`` / ``cache_read_tokens`` /
    ``cache_write_tokens`` / ``reasoning_tokens`` / ``cost_usd``. Without them
    the row proves a request was served but says nothing about what it cost, so
    a cross-wire session could not be priced at all. The fields are OMITTED
    rather than zero-filled when no usage was parsed (the verbatim relay never
    reads the body): a row with no numbers means "not measured", which a zero
    would misreport as "free".

    ``extra`` adds fields a caller needs to tell its rows apart (the HAL wire
    adds ``rail`` and ``route_class``). It can only add. Every key the record
    always has (``event``, ``path``, ``upstream``, ``provider``, ``model``,
    ``status``, ``served``, ``failed_over``, ``latency_ms``) keeps the record's
    value, and the keys in ``_RESERVED_EXTRA`` are dropped outright: the token
    fields and ``cost_usd``, which only a measurement may fill, and ``ts``,
    which the audit sink stamps later.
    """
    record = {
        "event": _WITNESS_EVENT,
        "path": path,
        "upstream": provider,
        "provider": provider,
        "model": model,
        "status": status,
        "served": served,
        "failed_over": failed_over,
        "latency_ms": latency_ms,
    }
    if usage:
        record.update({k: int(usage.get(k, 0) or 0) for k in _USAGE_KEYS})
        record["cost_usd"] = _cost_usd(model, record)
    for key, value in (extra or {}).items():
        if key not in _RESERVED_EXTRA:
            record.setdefault(key, value)
    return record


def _cost_usd(model: str, row: dict) -> float:
    """Price one witnessed call in USD (fail-open: 0.0 if pricing is absent).

    ``input_tokens`` is already FRESH input — ``normalise_provider_usage``
    subtracted the cached portion — which is exactly what ``calculate_cost``
    expects, since it prices cached reads separately and adds them.
    ``cost_tracker`` imports only ``threading`` and ``lib.hal.types``, so
    calling it here introduces no cycle.
    """
    try:
        from lib.hal.cost_tracker import calculate_cost
        return round(calculate_cost(
            model,
            input_tokens=row.get("input_tokens", 0),
            output_tokens=row.get("output_tokens", 0),
            cache_read_tokens=row.get("cache_read_tokens", 0),
            cache_creation_tokens=row.get("cache_write_tokens", 0),
        ), 8)
    except Exception:
        return 0.0


def sign_record(
    record: dict,
    *,
    log_path: Optional[Path] = None,
    key_path: Optional[Path] = None,
) -> Optional[str]:
    """Append ``record`` to the HMAC-chained signed log; return its ``audit_id``.

    DEPRECATED (Step 4): anyone who can write this log can also read the key
    beside it and re-sign it. The public-key signed record is
    :func:`append_thread_row`; this stays until the operator removes it.

    Reuses HAL's in-repo zero-dependency signer (``voice.audit.AuditLogger``):
    each record's ``hmac`` chains to the previous via ``prev_hmac``, so any
    later edit breaks ``verify_chain`` from that point. Fail-open: returns
    ``None`` on any error (a signing failure must never break a request).

    ``set_append_only=False`` keeps the privileged ``chflags`` off this path —
    the gateway is not voice.listen; the tamper-evidence is the HMAC chain, and
    the OS append-only flag would only obstruct rotation in a server process.
    """
    try:
        from lib.hal.voice.audit import AuditLogger

        logger = AuditLogger(
            path=log_path or signed_log_path(),
            key_path=key_path or signed_key_path(),
            set_append_only=False,
        )
        return logger.append(event=_WITNESS_EVENT, payload=record)
    except Exception:
        return None


def witness_chat(
    *,
    provider: str,
    model: str,
    status: Optional[int],
    served: bool,
    started: float,
    path: str,
    failed_over: bool = False,
    usage: Optional[dict] = None,
    latency_ms: Optional[int] = None,
    telemetry: Optional[GatewayTelemetry] = None,
    log_path: Optional[Path] = None,
    key_path: Optional[Path] = None,
    extra: Optional[dict] = None,
    session_id: Optional[str] = None,
    row_class: Optional[str] = "served",
    route: str = "unclassified",
    billing: str = "unknown",
    thread_usage: Optional[dict] = None,
) -> dict:
    """Witness ONE chat-rail request, then SIGN that same record.

    The record (carrying the SERVED ``provider`` + ``model``) goes to the
    unsigned :class:`GatewayTelemetry` audit sink, to the deprecated HMAC
    chain, and -- unless ``row_class`` is None -- into its public-key signed
    thread row (:func:`append_thread_row`), keyed by ``session_id`` or this
    gateway's launch id. ``route`` defaults to ``unclassified``: only a caller
    that ran the route classifier may claim ``inference`` or ``aux``.
    ``usage`` is the cross-wire count dict when the forwarder parsed one; a
    caller holding Anthropic's own usage block passes ``thread_usage``
    (:func:`lib.hal.infinite_cost.anthropic_usage`), which keeps the cache
    lifetimes and a priced variant the cross-wire form cannot carry.

    ``started`` is a ``time.monotonic()`` reading taken at handler entry; the
    latency is computed here so the clock is owned in one place. A caller that
    witnesses only after relaying the body passes ``latency_ms`` measured when
    the upstream answered, so the field keeps meaning time to response rather
    than growing to cover however long the client took to read it. Fail-open
    throughout: no leg may raise into the request path. Returns the record.
    """
    import time

    if latency_ms is None:
        latency_ms = round((time.monotonic() - started) * 1000)
    record = build_chat_record(
        provider=provider, model=model, status=status, served=served,
        failed_over=failed_over, latency_ms=latency_ms,
        path=path, usage=usage, extra=extra)
    # (1) unsigned audit row — same sink as every other rail (fail-open inside).
    (telemetry or GatewayTelemetry()).record(dict(record))
    # (2) HMAC-chained row: deprecated, kept until the operator removes it.
    sign_record(record, log_path=log_path, key_path=key_path)
    # (3) public-key signed thread row (fail-open inside; refuses, never weakens).
    if row_class is not None:
        observed = thread_usage if thread_usage is not None else _thread_usage(usage)
        append_thread_row(record, row_class=row_class, route=route,
                          session_id=session_id, usage=observed, billing=billing)
    return record


def _thread_usage(crosswire: Optional[dict]) -> Optional[dict]:
    """A forwarder's cross-wire counts as a thread row's usage; None if not measured."""
    if not crosswire:
        return None
    try:
        from lib.hal.infinite_cost import crosswire_usage

        return crosswire_usage(crosswire)
    except Exception:  # noqa: BLE001 - an unreadable count is an unknown one
        return None


__all__ = [
    "LAUNCH_SESSION_ID",
    "PendingRow",
    "SESSION_HEADER",
    "append_thread_row",
    "build_chat_record",
    "note_thread_failure",
    "session_id_from_headers",
    "sign_record",
    "signed_key_path",
    "signed_log_path",
    "witness_chat",
]
