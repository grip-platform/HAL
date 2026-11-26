"""HAL Gateway Proxy (P1) — local Anthropic-compat ``/v1/messages`` reverse proxy.

Closes GRIP issue #3274. The Claude Code harness drops/reorders the
``signature`` field on prior-assistant ``thinking`` / ``redacted_thinking``
blocks when it reconstructs message history on a retry/compaction, producing
the recurring ``400 ... thinking or redacted_thinking blocks ... cannot be
modified`` error. There is no GRIP code path between the harness and
``api.anthropic.com`` today — so a local gateway is the *only* place GRIP can
intercept.

The fix is correct-by-construction, not clever: this proxy forwards the
request body **verbatim** (opaque bytes — it never deserialises → mutates →
reserialises the messages array) and, on a retryable transport failure,
retries with the **same byte object**. Because the bytes are never touched,
every ``thinking.signature`` and ``redacted_thinking.data`` field survives
byte-identical across the proxy and across retries. The proxy owning the
retry means the harness sends once (correctly) and never gets the chance to
reconstruct-and-drop.

Design contract (syscall-doctrine + llm-augmentation monotone-toward-safe):

* **127.0.0.1 only.** Never bind a routable interface — a harness's own
  upstream credentials must never traverse a network hop.
* **Custody.** The upstream API key / OAuth token is relayed to Anthropic
  but NEVER logged and NEVER echoed in a response. Any error string the
  gateway emits is passed through :func:`redact` first.
* **Fail-open.** The signature-preservation value-add (:func:`extract_thinking_signatures`)
  lives *outside* the forward path — it is telemetry only and can never block
  a request. Transport exhaustion returns a clean Anthropic-shaped error
  envelope (the operator can always revert ``ANTHROPIC_BASE_URL`` to direct).
  The gateway can only ever *add* success, never subtract it.
* **One deliberate exception: paid use is fail-closed.** Before a request
  leaves for Anthropic, :class:`GatewayProxy` asks the spend guard
  (``lib/hal/spend_guard.py``) about its credential, and a refused one is
  answered locally (:func:`send_spend_refusal`) without a byte going upstream.
  The InfiniteMachine contract requires it: HAL never uses a paid key silently.
* **Verbatim stream.** SSE / JSON response bytes are relayed unmodified.

Scope: P1 is the Anthropic-compat path only. OpenAI-compat
``/v1/chat/completions`` (P2), Fern SDK (P4), folded kernel (P5) are later.

Falsifiable hypothesis: **H001** (HAL hypothesis engine,
``learning/hypotheses.jsonl``, deadline 2026-07-27) — forwarding the body
verbatim + retrying byte-identical drives the 400 "thinking blocks cannot be
modified" rate to ~0 over >=20 thinking-bearing post-compaction sessions.
Exogenous anchor: the Anthropic API returning 200 (a process GRIP cannot
please-to-pass), never a self-score.
"""
from __future__ import annotations

import argparse
import errno
import http.client
import json
import math
import os
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Iterable, Iterator, List, Optional, Tuple

from lib.hal import account_switch, spend_guard
from lib.hal.http_refusal import _discard_unread_body
from lib.hal.gateway_guard import enforce_gateway_access
from lib.hal.port_reclaim import reclaim_port

DEFAULT_UPSTREAM = "https://api.anthropic.com"
DEFAULT_GATEWAY_PORT = 4010
_ANTHROPIC_VERSION = "2023-06-01"  # added when translating an API-key Bearer to x-api-key

# Statuses worth a transport-level retry (the body is re-sent byte-identical).
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504, 529})
_DEFAULT_MAX_RETRIES = 2
_INITIAL_BACKOFF_S = 0.5
_BACKOFF_MULTIPLIER = 2.0
_THROTTLE_WAIT_BUDGET_S = 10.0
_STREAM_CHUNK = 65536

# Upstream timeouts (InfiniteMachine v2, Step 5a; fleet finding f83). One 600 s
# socket timeout used to cover every wait, so a dead network took ten minutes to
# notice and a quick call waited as long as a long answer. The default opener
# now splits it. Seconds; each is read from the environment when an opener is
# built, and a value that is not a positive number keeps the default:
#
#   HAL_GATEWAY_CONNECT_TIMEOUT_S      10  TCP connect, any proxy CONNECT and the
#                                          TLS handshake
#   HAL_GATEWAY_TOTAL_TIMEOUT_S       600  the whole exchange, from sending the
#                                          request to the last byte of the answer;
#                                          a streamed answer leaves it once its
#                                          headers have arrived
#   HAL_GATEWAY_STREAM_IDLE_TIMEOUT_S 600  a streamed answer (text/event-stream):
#                                          the longest silence between two reads,
#                                          however long the stream runs
#   HAL_GATEWAY_SHORT_TIMEOUT_S        60  the whole exchange for a GET or a token
#                                          count
#
# The stream idle limit stays at the old per-read 600 s on purpose, so no stream
# that passed before is cut now; Step 5d tunes it against the measured hold
# budgets (H-HOLD-1).
_TIMEOUT_ENV = {
    "connect": "HAL_GATEWAY_CONNECT_TIMEOUT_S",
    "total": "HAL_GATEWAY_TOTAL_TIMEOUT_S",
    "stream_idle": "HAL_GATEWAY_STREAM_IDLE_TIMEOUT_S",
    "short_total": "HAL_GATEWAY_SHORT_TIMEOUT_S",
}


def _positive_number(raw, default: float) -> float:
    """``raw`` as a positive, finite number, else ``default``."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 and math.isfinite(value) else default


@dataclass(frozen=True)
class UpstreamTimeouts:
    """How long the gateway waits on an upstream, in seconds (see the table above)."""
    connect: float = 10.0
    total: float = 600.0
    stream_idle: float = 600.0
    short_total: float = 60.0

    @classmethod
    def from_env(cls, environ=None) -> "UpstreamTimeouts":
        """The timeouts the environment sets, each falling back to its default."""
        env = os.environ if environ is None else environ
        defaults = cls()
        return cls(**{name: _positive_number(env.get(var), getattr(defaults, name))
                      for name, var in _TIMEOUT_ENV.items()})

# Credential shapes redacted from every log line / error string the gateway
# emits. Covers classic keys, OAuth tokens (sk-ant-oat...), and bearer headers.
_KEY_PATTERN = re.compile(
    r"(sk-ant-oat[\w.\-]+|sk-ant-[\w.\-]+|sk-[\w.\-]{6,}|Bearer\s+[\w.\-]+)",
    re.IGNORECASE,
)

# Request headers the gateway must NOT forward upstream: hop-by-hop (RFC 7230
# §6.1), plus Host (urllib recomputes from the upstream URL), Content-Length
# (recomputed from the body), and Accept-Encoding (forced identity so the
# streamed response is never gzip-framed and can be relayed raw).
_STRIP_REQUEST_HEADERS = frozenset({
    "host", "content-length", "connection", "keep-alive", "proxy-connection",
    "proxy-authenticate", "proxy-authorization", "te", "trailer", "trailers",
    "transfer-encoding", "upgrade", "accept-encoding",
})

# Response headers the gateway manages itself (it streams to EOF + closes).
_STRIP_RESPONSE_HEADERS = frozenset({
    "connection", "keep-alive", "transfer-encoding", "content-encoding",
    "content-length", "te", "trailer", "trailers", "upgrade",
    # defensive: never echo an auth header back to the client
    "x-api-key", "authorization",
})

# Transport-level failures (connection refused, DNS, timeout) — retryable and,
# when exhausted, surfaced as a clean GatewayUpstreamError (never a crash).
_CONN_ERRORS = (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError)


class GatewayUpstreamError(Exception):
    """Upstream could not be reached after retries. Message is pre-redacted."""


class GatewayTransportError(GatewayUpstreamError):
    """The connection to the upstream failed (connect, DNS, TLS, a reset or a
    timeout) on every attempt. A client is sent 503 for it, never 400 or 502:
    the fault is the network's, and trying again later may well work."""


def is_transport_error(exc: BaseException) -> bool:
    """Whether ``exc`` means the connection to an upstream failed: connect, DNS,
    TLS, a reset, a timeout, or an answer cut off before its end.

    An HTTP error answer is not one (it is a reply), and neither is a
    ``GatewayUpstreamError`` the gateway raises for its own reasons (an
    unroutable provider), which stays a 502.
    """
    if isinstance(exc, urllib.error.HTTPError):
        return False
    if isinstance(exc, GatewayUpstreamError):
        return isinstance(exc, GatewayTransportError)
    return isinstance(exc, (OSError, http.client.HTTPException))


def failure_reply(exc: BaseException, *, other_status: int = 502,
                  other_label: str = "gateway internal error") -> Tuple[int, str]:
    """The status and redacted message for a request that failed before any byte
    of an answer was sent (fleet findings f48, f68).

    A transport fault talking upstream is a 503: the OpenAI-wire harnesses never
    retry a 400, so the old "translation error" 400 ended a Codex, OpenCode or
    Hermes turn on a connection reset. A failure the gateway names itself is a
    502, and anything else gets ``other_status``.
    """
    if is_transport_error(exc):
        if isinstance(exc, GatewayUpstreamError):
            return 503, redact(str(exc))
        return 503, f"upstream connection failed: {redact(str(exc))}"
    if isinstance(exc, GatewayUpstreamError):
        return 502, redact(str(exc))
    return other_status, f"{other_label}: {redact(str(exc))}"


def redact(text: str) -> str:
    """Replace any credential-shaped substring with ``[REDACTED]``."""
    return _KEY_PATTERN.sub("[REDACTED]", text)


def _block_signature(block) -> Optional[str]:
    """The preserved field of one content block, or ``None`` if it has none.

    ``thinking`` blocks carry ``signature``; ``redacted_thinking`` blocks
    carry an opaque ``data`` blob. Both must survive a round-trip verbatim.
    """
    if not isinstance(block, dict):
        return None
    btype = block.get("type")
    if btype == "thinking":
        return block.get("signature")
    if btype == "redacted_thinking":
        return block.get("data")
    return None


def _message_signatures(message: dict) -> List[str]:
    """Preserved signatures of one message's content blocks, in order."""
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [sig for sig in map(_block_signature, content) if sig is not None]


def extract_thinking_signatures(raw_body: bytes) -> List[str]:
    """Read-only audit: every thinking/redacted_thinking signature, in order.

    Returns the ``signature`` of each ``thinking`` block and the ``data`` of
    each ``redacted_thinking`` block, in document order, across all messages.
    Linear in total block count. Telemetry + test anchor ONLY — never called
    on the forward path, so its failure can never affect a request (fail-open
    by construction). Raises on malformed JSON; callers that want a non-fatal
    probe use :meth:`GatewayProxy.audit_signatures`.
    """
    obj = json.loads(raw_body)
    out: List[str] = []
    for message in obj.get("messages", []):
        out.extend(_message_signatures(message))
    return out


# --- response normalisation -------------------------------------------------

@dataclass
class UpstreamResponse:
    """Normalised upstream response — status + headers + a streamable body."""
    status: int
    headers: List[Tuple[str, str]]
    _reader: object = field(repr=False, default=None)
    #: Normalised token counts, when the forwarder already parsed the provider
    #: body (``gateway_crosswire.normalise_provider_usage`` shape). The witness
    #: runs BEFORE the relay and the reader is single-pass, so re-reading the
    #: body there would drain it and send the client an empty response -- this
    #: slot carries the numbers forward instead. ``None`` on a verbatim relay,
    #: where nothing has parsed the body and no count is available.
    usage: Optional[dict] = field(repr=False, default=None)

    def iter_chunks(self, size: int = _STREAM_CHUNK) -> Iterator[bytes]:
        """Yield body bytes until EOF, each as soon as it arrives. Works for both
        SSE and buffered JSON.

        Reads with ``read1`` where the reader has one: ``read(size)`` on an HTTP
        response waits until ``size`` bytes or the end, so a 3-second stream used
        to reach the client as one 64 KiB block at its end (fleet finding f79).
        """
        if self._reader is None:
            return
        read = read_some(self._reader)
        while True:
            chunk = read(size)
            if not chunk:
                break
            yield chunk

    def read_all(self) -> bytes:
        return b"".join(self.iter_chunks())

    def close(self) -> None:
        """Release the underlying upstream socket. Idempotent + never raises.

        Relay paths break out of the chunk loop on client disconnect; without
        this the upstream urllib response would stay open and leak the socket.
        """
        closer = getattr(self._reader, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass


def _header_items(headers) -> Iterator[Tuple[str, str]]:
    """Yield (name, value) from a urllib message OR a plain (k, v) iterable."""
    items = getattr(headers, "items", None)
    if callable(items):
        yield from items()
    else:
        yield from headers


def read_some(reader) -> Callable[[int], bytes]:
    """The reader's ``read1`` when its class has one (it returns as soon as any
    bytes are there), else its ``read``. Looked up on the class, so a stand-in
    that answers every attribute (a mock) is not taken for a stream."""
    read1 = getattr(type(reader), "read1", None)
    return reader.read1 if callable(read1) else reader.read


class _UpstreamClock:
    """The time limits of one upstream exchange (see ``UpstreamTimeouts``).

    Until the answer's headers arrive, and for the whole of an answer that is
    not a stream, each wait may last only what is left of the total. Once a
    text/event-stream answer starts, each read may wait ``stream_idle`` and the
    total no longer applies: a stream may run for as long as it keeps talking.
    """

    def __init__(self, timeouts: UpstreamTimeouts, *, short: bool):
        budget = timeouts.short_total if short else timeouts.total
        self._deadline = time.monotonic() + budget
        self._idle = timeouts.stream_idle
        self._streaming = False

    def answer_started(self, content_type: str) -> None:
        self._streaming = "text/event-stream" in (content_type or "").lower()

    def next_wait(self) -> float:
        if self._streaming:
            return self._idle
        left = self._deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("the upstream exchange ran past HAL's total timeout")
        return left


class _TimedResponse(http.client.HTTPResponse):
    """An HTTP response that sets its socket's timeout from the clock before
    every read, so no read can wait past what the clock allows."""

    _hal_clock: Optional[_UpstreamClock] = None
    _hal_sock = None

    def hal_start(self, clock: _UpstreamClock, sock) -> None:
        self._hal_clock, self._hal_sock = clock, sock
        clock.answer_started(self.getheader("content-type") or "")

    def _hal_arm(self) -> None:
        # Not once the answer is done: the read that takes a Content-Length
        # body's last byte closes the connection (``self.fp`` goes to None), and
        # arming its socket after that raised EBADF on the read that finds EOF.
        if self.fp is not None and self._hal_clock is not None and self._hal_sock is not None:
            self._hal_sock.settimeout(self._hal_clock.next_wait())

    def read(self, amt=None):
        self._hal_arm()
        return super().read(amt)

    def read1(self, n=-1):
        self._hal_arm()
        return super().read1(n)

    def readinto(self, b):
        self._hal_arm()
        return super().readinto(b)

    def readline(self, limit=-1):
        self._hal_arm()
        return super().readline(limit)

    def peek(self, n=-1):
        self._hal_arm()
        return super().peek(n)


class _TimedConnectionMixin:
    """Connects (TCP, any proxy CONNECT, TLS) under the connect timeout, then
    hands every later wait to the exchange's clock."""

    response_class = _TimedResponse

    def __init__(self, *args, hal_clock: _UpstreamClock, **kwargs):
        super().__init__(*args, **kwargs)
        self._hal_clock = hal_clock

    def connect(self):
        super().connect()
        self.sock.settimeout(self._hal_clock.next_wait())

    def getresponse(self):
        # Kept before the call: http.client drops self.sock once the answer
        # says it will close, and the answer still reads through that socket.
        sock = self.sock
        response = super().getresponse()
        response.hal_start(self._hal_clock, sock)
        return response


class _TimedHTTPConnection(_TimedConnectionMixin, http.client.HTTPConnection):
    pass


class _TimedHTTPSConnection(_TimedConnectionMixin, http.client.HTTPSConnection):
    pass


class _TimedHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, clock: _UpstreamClock):
        super().__init__()
        self._hal_clock = clock

    def http_open(self, req):
        return self.do_open(_TimedHTTPConnection, req, hal_clock=self._hal_clock)


class _TimedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, clock: _UpstreamClock):
        super().__init__()
        self._hal_clock = clock

    def https_open(self, req):
        return self.do_open(_TimedHTTPSConnection, req, context=self._context,
                            hal_clock=self._hal_clock)


def _is_short_exchange(method: str, url: str) -> bool:
    """A GET or a token count: a small answer that gets the short total."""
    path = urllib.parse.urlsplit(url).path
    return method.upper() == "GET" or path.endswith("/count_tokens")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _open_timed(req: urllib.request.Request, timeouts: UpstreamTimeouts, *, short: bool,
                subscription_bound: bool = False):
    """``urlopen`` with the split timeouts. Every other handler (proxies from the
    environment, redirects, error processing) is urllib's own, as before."""
    clock = _UpstreamClock(timeouts, short=short)
    handlers = [_TimedHTTPHandler(clock), _TimedHTTPSHandler(clock)]
    if subscription_bound:
        handlers.append(_NoRedirect())
    opener = urllib.request.build_opener(*handlers)
    return opener.open(req, timeout=timeouts.connect)


def build_default_opener(
    timeout: Optional[float] = None,
    *,
    timeouts: Optional[UpstreamTimeouts] = None,
    subscription_bound: bool = False,
) -> Callable[[str, str, List[Tuple[str, str]], Optional[bytes]], object]:
    """Build the urllib opener, with redirect refusal for bound subscriptions."""
    split = None if timeout is not None else (timeouts or UpstreamTimeouts.from_env())

    def _opener(method: str, url: str, headers: List[Tuple[str, str]],
                body: Optional[bytes]):
        req = urllib.request.Request(url, data=body, method=method)
        for name, value in headers:
            req.add_header(name, value)
        try:
            if split is None and not subscription_bound:
                return urllib.request.urlopen(req, timeout=timeout)
            active = split or UpstreamTimeouts.from_env()
            return _open_timed(req, active, short=_is_short_exchange(method, url),
                               subscription_bound=subscription_bound)
        except urllib.error.HTTPError as http_err:
            return http_err
    return _opener


def _normalise_anthropic_auth(headers: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Translate an OpenAI-style ``Authorization: Bearer <api-key>`` to
    Anthropic's ``x-api-key`` (+ ``anthropic-version``) so an OpenAI client
    (e.g. Codex) works on the RAW relay paths (``GET /v1/models``,
    ``POST /v1/messages``), not only the /v1/responses + /v1/chat handlers
    (which already translate via gateway_openai.anthropic_upstream_headers).

    Anthropic authenticates API keys via ``x-api-key``, NOT ``Authorization:
    Bearer`` — so Codex's ``GET /v1/models`` probe (which carries Bearer) 401s
    "Invalid bearer token" through the verbatim relay. Returned UNCHANGED:
      * an explicit ``x-api-key`` (the Anthropic-native harness path) — it wins;
      * an OAuth Bearer (``sk-ant-oat...``) — Anthropic's OAuth/CCH scheme uses
        Bearer natively, so converting it would BREAK the subscription path;
      * any request with no Bearer.
    """
    lower = {n.lower(): v for n, v in headers}
    if "x-api-key" in lower:
        return headers
    auth = lower.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return headers
    token = auth[7:].strip()
    if not token or token.startswith("sk-ant-oat"):
        return headers  # OAuth Bearer stays Bearer (Anthropic's OAuth scheme)
    out = [(n, v) for n, v in headers if n.lower() != "authorization"]
    out.append(("x-api-key", token))
    if "anthropic-version" not in lower:
        out.append(("anthropic-version", _ANTHROPIC_VERSION))
    return out


class GatewayProxy:
    """Verbatim Anthropic reverse proxy with byte-identical retry.

    ``opener`` is injected (DIP) so tests exercise the retry / preservation /
    fail-open logic with zero network. The production opener is
    :func:`build_default_opener`.
    """

    #: The upstream bills the credentials the spend guard reads: Anthropic's
    #: ``x-api-key`` and OAuth bearers. A subclass pointed at another vendor,
    #: with that vendor's own key, sets this False.
    checks_anthropic_spend = True

    def __init__(
        self,
        *,
        upstream_base: str = DEFAULT_UPSTREAM,
        opener: Optional[Callable] = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.upstream_base = upstream_base.rstrip("/")
        self._opener = opener or build_default_opener()
        self._max_retries = max_retries
        self._sleep = sleep

    def upstream_headers(self, client_headers: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Filter client request headers for forwarding upstream.

        Strips hop-by-hop / managed headers; keeps anthropic-* verbatim. An
        OpenAI-style ``Authorization: Bearer <api-key>`` is translated to
        ``x-api-key`` so OpenAI clients work on the raw relay (OAuth Bearer and
        an explicit x-api-key are left untouched — see _normalise_anthropic_auth).
        Host is dropped (urllib recomputes it from the upstream URL).
        """
        filtered = [
            (name, value)
            for name, value in client_headers
            if name.lower() not in _STRIP_REQUEST_HEADERS
        ]
        return _normalise_anthropic_auth(filtered)

    def forward(self, path, body, client_headers, *, method="POST", max_retries=None):
        # All Anthropic-bound inference rails arrive here fully buffered, before
        # a handler sends headers. Returned readers never re-enter the switch.
        from lib.hal.gateway_accounts import forward
        return forward(self, path, body, list(client_headers), method=method,
                       max_retries=max_retries)

    def _forward(
        self,
        path: str,
        body: Optional[bytes],
        client_headers: Iterable[Tuple[str, str]],
        *,
        method: str = "POST",
        max_retries: Optional[int] = None,
    ) -> UpstreamResponse:
        """Forward ``body`` verbatim to ``upstream_base + path`` with retry.

        The SAME ``body`` object is re-sent on every attempt — it is never
        deserialised or reserialised, so thinking-block signatures are
        preserved byte-identical across retries (the #3274 fix). Upstream
        4xx/5xx are relayed unchanged; only transport failures retry.

        The spend guard sees the request before any attempt: a refused paid
        credential raises :class:`~lib.hal.spend_guard.SpendRefused` and nothing
        is sent. A 2xx that shows paid overage marks its account spent.
        """
        url = self.upstream_base + path
        headers = self.upstream_headers(client_headers)
        if self.checks_anthropic_spend:
            refusal = spend_guard.check_anthropic_request(method, path, headers)
            if refusal is not None:
                raise spend_guard.SpendRefused(refusal)
        retries = self._max_retries if max_retries is None else max_retries
        backoff = _INITIAL_BACKOFF_S
        throttle_wait = 0.0

        for attempt in range(retries + 1):
            try:
                resp = self._opener(method, url, headers, body)
            except urllib.error.HTTPError as http_err:
                # An opener that RAISES (rather than returns) an HTTPError is
                # still a valid upstream answer (it carries .code/.headers/
                # .read). Treat it as a response so the status-based retry/
                # relay below handles raised and returned HTTPErrors
                # identically — a non-retryable raised 400 is relayed, not
                # mis-retried into a 502. (HTTPError subclasses URLError, so
                # this clause MUST precede _CONN_ERRORS. ``.status`` is a
                # read-only property on 3.9+; never assign it — _resp_status
                # falls back to ``.code``.)
                resp = http_err
            except _CONN_ERRORS as exc:
                if attempt < retries:
                    self._sleep(backoff)
                    backoff *= _BACKOFF_MULTIPLIER
                    continue
                raise GatewayTransportError(
                    f"upstream unreachable after {attempt + 1} attempt(s): "
                    f"{redact(str(exc))}"
                ) from exc

            status = _resp_status(resp)
            response_headers = list(_header_items(resp.headers))
            retry = status in _RETRYABLE_STATUS and attempt < retries
            delay = backoff
            if status == 429:
                from lib.hal.gateway_accounts import retry_after
                delay = retry_after(response_headers, default=backoff)
                retry = (retry and not account_switch.limit_reached(status, response_headers)
                         and throttle_wait + delay <= _THROTTLE_WAIT_BUDGET_S)
            if retry:
                _drain(resp)
                self._sleep(delay)
                if status == 429:
                    throttle_wait += delay
                backoff *= _BACKOFF_MULTIPLIER
                continue
            if self.checks_anthropic_spend:
                spend_guard.observe_anthropic_response(status, response_headers, headers)
            return UpstreamResponse(
                status=status,
                headers=response_headers,
                _reader=resp,
            )
        # Unreachable: the loop returns or raises on the final attempt.
        raise GatewayUpstreamError("retry loop exhausted without a response")

    def audit_signatures(self, body: Optional[bytes]) -> Optional[List[str]]:
        """Non-fatal probe of thinking-block signatures. ``None`` on any error."""
        if not body:
            return None
        try:
            return extract_thinking_signatures(body)
        except Exception:
            return None


def _resp_status(resp) -> int:
    """HTTP status of a response-like object.

    Prefers ``.status`` (urllib responses, FakeResponse) and falls back to
    ``.code`` (``urllib.error.HTTPError`` — whose ``.status`` is a read-only
    property on 3.9+ that aliases ``.code``, but which older runtimes lack).
    """
    return int(getattr(resp, "status", None) or getattr(resp, "code", 0) or 0)


def _drain(resp) -> None:
    """Best-effort discard of a retryable response body before retrying."""
    try:
        resp.read()
    except Exception:
        pass


def response_headers(headers: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Filter upstream response headers for relaying to the client."""
    return [
        (name, value)
        for name, value in headers
        if name.lower() not in _STRIP_RESPONSE_HEADERS
    ]


#: The upstream headers that tell a client when to try again. Every relay keeps
#: them, including the ones that rebuild an error in the client's own dialect
#: (fleet finding f73).
RETRY_HEADERS = frozenset({"retry-after", "retry-after-ms"})


def retry_headers(headers: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """The ``Retry-After`` and ``retry-after-ms`` headers among ``headers``."""
    return [(name, value) for name, value in headers if name.lower() in RETRY_HEADERS]


def openai_wire_status(status: int) -> int:
    """The status an OpenAI-wire client (``/v1/responses``, ``/v1/chat/completions``)
    is sent for an upstream ``status``: a 429 goes out as a 503.

    One 429 ends a Codex turn, even with ``Retry-After`` set and four retries
    configured (fleet finding f26, measured on Codex 0.156.1; the wire-matrix S2
    run again), while a 503 is retried. ``Retry-After`` travels unchanged. Step
    5c's hold and account switch will absorb the 429 instead of renaming it.
    """
    return 503 if status == 429 else status


def openai_wire_response(resp: UpstreamResponse) -> UpstreamResponse:
    """``resp`` with its status as an OpenAI-wire client is sent it (see
    :func:`openai_wire_status`); headers and body are unchanged."""
    status = openai_wire_status(resp.status)
    return resp if status == resp.status else replace(resp, status=status)


def is_event_stream(headers: Iterable[Tuple[str, str]]) -> bool:
    """Whether ``headers`` declare a server-sent-event body."""
    return any(name.lower() == "content-type" and "text/event-stream" in str(value).lower()
               for name, value in headers)


def lost_upstream_message(exc: BaseException) -> str:
    """What a client is told when the upstream failed mid-stream. The exception's
    type is named, never its text, and the wording is one the harnesses read as
    retryable ("connection lost", "retry your request")."""
    return (f"HAL gateway: upstream connection lost mid-stream "
            f"({type(exc).__name__}); retry your request")


def anthropic_stream_error(message: str, error_type: str = "api_error") -> bytes:
    """One in-band error event in Anthropic's stream dialect, shaped as Anthropic
    sends it mid-stream. Claude Code re-requests on it (wire-matrix S7)."""
    body = json.dumps({"type": "error", "error": {"type": error_type,
                                                  "message": redact(message)}})
    return f"event: error\ndata: {body}\n\n".encode("utf-8")


# --- BaseHTTPRequestHandler integration (used by server.py + standalone) -----

def read_request_body(handler: BaseHTTPRequestHandler) -> bytes:
    """Read exactly Content-Length raw bytes — NO json parse (verbatim)."""
    length = int(handler.headers.get("Content-Length", 0) or 0)
    body = handler.rfile.read(length) if length > 0 else b""
    if len(body) != length:
        raise ValueError("incomplete request body")
    return body


def relay_response(handler: BaseHTTPRequestHandler, resp: UpstreamResponse, *,
                   stream_error: Optional[Callable[[str], bytes]] = None) -> None:
    """Write an upstream response back to the client, streaming the body.

    Once the status line has gone out no other can follow, so if the upstream
    fails mid-body (a reset, a timeout, an answer cut short) a server-sent-event
    answer ends with one error event in the client's dialect (``stream_error``;
    Anthropic's by default): the client sees an interruption, never an answer
    that silently stopped (fleet finding f72). Any other answer just ends.
    """
    try:
        handler.send_response(resp.status)
        for name, value in response_headers(resp.headers):
            handler.send_header(name, value)
        handler.send_header("Connection", "close")
        handler.end_headers()
    except (BrokenPipeError, ConnectionResetError):
        resp.close()
        return
    try:
        for chunk in resp.iter_chunks():
            try:
                handler.wfile.write(chunk)
                handler.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return  # client hung up mid-stream — nothing to recover
    except Exception as exc:  # the upstream failed mid-body
        if is_event_stream(resp.headers):
            _write_quietly(handler, (stream_error or anthropic_stream_error)(
                lost_upstream_message(exc)))
    finally:
        resp.close()  # release the upstream socket even on client disconnect


def _write_quietly(handler: BaseHTTPRequestHandler, data: bytes) -> None:
    """Write and flush ``data``; a client that has gone away is not an error."""
    try:
        handler.wfile.write(data)
        handler.wfile.flush()
    except OSError:
        pass


def send_gateway_error(handler: BaseHTTPRequestHandler, code: int, detail: str,
                       headers: Iterable[Tuple[str, str]] = ()) -> None:
    """Emit an Anthropic-shaped error envelope (message pre-redacted), plus any
    ``headers`` (a ``Retry-After``, say)."""
    payload = json.dumps({
        "type": "error",
        "error": {"type": "gateway_error", "message": redact(detail)},
    }).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    for name, value in headers:
        handler.send_header(name, value)
    handler.send_header("Connection", "close")
    handler.end_headers()
    try:
        handler.wfile.write(payload)
    except (BrokenPipeError, ConnectionResetError):
        pass


def send_spend_refusal(handler: BaseHTTPRequestHandler, decision) -> None:
    """Answer a request the spend guard refused, Anthropic-shaped, sending nothing upstream.

    A refused paid key is a 403 ``permission_error``; a spent subscription is a
    429 ``rate_limit_error`` with ``Retry-After`` set to the seconds until its
    reset, so a harness waits for it rather than retrying at once.
    """
    error_type = "rate_limit_error" if decision.status == 429 else "permission_error"
    payload = json.dumps({
        "type": "error",
        "error": {"type": error_type, "message": redact(decision.reason)},
    }).encode("utf-8")
    handler.send_response(decision.status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    if decision.retry_after:
        handler.send_header("Retry-After", str(decision.retry_after))
    handler.send_header("Connection", "close")
    handler.end_headers()
    try:
        handler.wfile.write(payload)
    except (BrokenPipeError, ConnectionResetError):
        pass


def _json_dict(body: Optional[bytes]) -> Optional[dict]:
    """``body`` parsed as a JSON object, or ``None`` for anything else."""
    try:
        payload = json.loads(body)
    except (ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _maybe_cross_wire(
    path: str,
    command: str,
    body: Optional[bytes],
    client_headers: Iterable[Tuple[str, str]] = (),
) -> Optional[UpstreamResponse]:
    """A re-pointed response for a non-Anthropic model, or ``None`` to use the
    verbatim Anthropic relay.

    Fires ONLY for: ``HAL_GATEWAY_MULTIPROVIDER`` enabled AND a ``POST
    /v1/messages`` (Anthropic-wire) request AND a model that routes to a known
    non-Anthropic provider. EVERY other case returns ``None`` and falls through
    to the byte-identical verbatim relay. The flag is checked FIRST, so when
    multi-provider is OFF this returns immediately WITHOUT reading or parsing
    the body — the #3274 verbatim path is untouched, zero added overhead. With
    the flag ON, ``claude-*`` and unknown models STILL return ``None``.

    Two re-pointed shapes, in priority order:

    * **Anthropic same-wire** — the provider serves the Anthropic Messages wire
      itself (``gateway_egress.ANTHROPIC_COMPAT_ENDPOINTS``; Moonshot today), so
      the client's bytes go there VERBATIM with the provider's own key —
      thinking, tool_use and cache fields intact. ``client_headers`` exist for
      this branch: ``anthropic-version`` / ``anthropic-beta`` are forwarded, the
      client's credential is not (``AnthropicCompatProxy``).
    * **Cross-wire** — every other known provider, translated to OpenAI Chat
      and back (``cross_wire_forward``), which drops those fields.

    ``POST /v1/messages/count_tokens`` is gated the same way for the same-wire
    providers only (Moonshot's Anthropic surface has no count_tokens, so it is
    answered from the vendor tokenizer, ``anthropic_count_tokens``); for every
    other model it stays on the verbatim relay, as it always has.

    The single exception to family routing is the exact id ``hal-fanout`` —
    the ensemble pseudo-model (``gateway_fanout``) — checked before the
    provider gate because it intentionally matches no family.
    """
    from lib.hal.gateway_egress import multiprovider_enabled
    if not multiprovider_enabled():
        return None
    bare = path.split("?", 1)[0]
    if command != "POST" or bare not in (_MESSAGES_PATH, _COUNT_TOKENS_PATH) or not body:
        return None
    payload = _json_dict(body)
    if payload is None:
        return None  # unparseable / non-object -> let the verbatim relay handle it
    # A model picked from Claude Code's /model picker may arrive under its
    # discovery-visible alias. Resolve it to the id that routes BEFORE any
    # branch reads the model, and re-serialise so the same-wire branch — which
    # forwards the client's bytes verbatim — carries the real id too. Identity
    # for every non-alias request, which stays byte-for-byte as it was.
    from lib.hal.gateway_picker_alias import canonical_for
    _canonical = canonical_for(payload.get("model", ""))
    if _canonical != payload.get("model"):
        payload["model"] = _canonical
        body = json.dumps(payload).encode("utf-8")
    # hal-fanout: the ensemble pseudo-model — EXACT-id opt-in, checked before
    # family routing (it matches no family, so without this it would fall to
    # the verbatim relay and 404 at Anthropic). Lazy imports break the
    # gateway <- egress cycle throughout.
    if bare == _MESSAGES_PATH and payload.get("model") == "hal-fanout":
        from lib.hal.gateway_fanout import fanout_response
        return fanout_response(payload, want_stream=bool(payload.get("stream")))
    from lib.hal.gateway_egress import (
        anthropic_compat_provider, anthropic_count_tokens,
        anthropic_same_wire_forward, cross_wire_forward, same_wire_provider)
    model = payload.get("model", "")
    compat = anthropic_compat_provider(model)
    if compat and bare == _COUNT_TOKENS_PATH:
        return anthropic_count_tokens(payload, provider=compat)
    if compat:
        return anthropic_same_wire_forward(compat, path, body, client_headers)
    if bare == _COUNT_TOKENS_PATH:
        return None  # no cross-wire count_tokens exists: verbatim relay, as before
    # same_wire_provider IS the provider-routing gate (flag + known
    # non-Anthropic provider); claude-*/unknown -> None -> verbatim (#3274).
    provider = same_wire_provider(model)
    if not provider:
        return None
    return cross_wire_forward(
        payload, provider=provider, want_stream=bool(payload.get("stream")))


_MESSAGES_PATH = "/v1/messages"
_COUNT_TOKENS_PATH = "/v1/messages/count_tokens"
_MESSAGES_RAIL_PATH = _MESSAGES_PATH


def _thread_row_class(method: str, path: str, headers, body: Optional[bytes]) -> Tuple[Optional[str], str]:
    """The signed thread row a request earns, and its route class.

    A conversation turn is a ``served`` row; a non-turn on the messages rail (a
    quota check, a token count, a title call) an ``aux`` row. Account-bound and
    unknown calls earn none: they are not exchanges the thread records.
    """
    from lib.hal.wire_classify import RouteClass, classify
    route = classify(method, path, headers, body)
    rows = {RouteClass.INFERENCE: "served", RouteClass.AUX: "aux"}
    return rows.get(route), route.value


def _witnessed_model(body: Optional[bytes]) -> str:
    """The model the caller asked for, with a picker alias resolved.

    This rail re-reads the CALLER's body, which still carries the picker alias
    the routing layer already resolved. Without resolving it the row names the
    wrong provider for every aliased request -- a witness that disagrees with
    what actually ran.
    """
    from lib.hal.gateway_picker_alias import canonical_for
    model = ""
    try:
        if body:
            model = (json.loads(body) or {}).get("model", "") or ""
    except (ValueError, TypeError):
        model = ""
    return canonical_for(model)


def _witness_messages_rail(body: Optional[bytes], resp: "UpstreamResponse", started: float,
                           *, method: str = "POST", path: str = _MESSAGES_PATH,
                           headers=None):
    """Witness+sign one /v1/messages decision (fail-open; never raises).

    Mirrors gateway_openai._witness_chat: lazy import breaks the cycle,
    try/except swallows every failure so observability can never subtract a
    served request.  The witness lives OUTSIDE the forward path and is additive.
    Returns the signed thread row still to be written (a ``PendingRow``, which
    :func:`_relay_witnessed` writes once the relay has shown the reply's
    usage), or None when the route earns no row. ``method``, ``path`` and
    ``headers`` decide that row, its session and its billing; no header is
    ever recorded.
    """
    try:
        from lib.hal.gateway_witness import PendingRow, session_id_from_headers, witness_chat
        from lib.hal.gateway_egress import same_wire_provider
        thread_class, route = _thread_row_class(method, path, headers, body)
        model = _witnessed_model(body)
        provider = same_wire_provider(model) or "anthropic"
        status = getattr(resp, "status", None)
        record = witness_chat(
            provider=provider, model=model, status=status,
            # Never a failover here: ``provider`` is the one the caller ASKED
            # for, and serving it is routing; FailoverProxy records its own.
            served=status is not None, failed_over=False,
            started=started, path=_MESSAGES_RAIL_PATH,
            # Counts the forwarder already parsed (None on the verbatim relay).
            # Reading the body here would drain the single-pass reader.
            usage=getattr(resp, "usage", None),
            row_class=None,  # not now: the thread row is written after the relay
        )
        if thread_class is None:
            return None
        return PendingRow(record, thread_class, route, session_id_from_headers(headers),
                          _witness_billing(provider, headers))
    except Exception as exc:  # the request is served regardless; the log says why
        _note_witness_failure(exc)
        return None


def _note_witness_failure(exc: Exception) -> None:
    """One log line when witnessing itself failed, so no row is dropped unseen."""
    try:
        from lib.hal.gateway_witness import note_thread_failure
        note_thread_failure(exc)
    except Exception:
        pass


def _witness_billing(provider: str, headers) -> str:
    """How the relayed request is billed: read from the client's credential on
    the Anthropic path, unknown on a cross-wire one (HAL's own provider key)."""
    if provider != "anthropic":
        return "unknown"
    from lib.hal.infinite_cost import billing_from_headers
    return billing_from_headers(headers)


class _TeeReader:
    """A body reader that also shows each chunk to an observer.

    The relay reads exactly the bytes it would have read, and the observer's
    ``feed`` never raises (``infinite_cost.UsageObserver``).
    """

    def __init__(self, reader, observer) -> None:
        self._reader = reader
        self._observer = observer

    def read(self, size: int = -1) -> bytes:
        return self._seen(self._reader.read(size))

    def read1(self, size: int = -1) -> bytes:
        """As soon as any bytes are there (``read_some``), so a tee never makes a
        stream wait for a full block (fleet finding f79)."""
        return self._seen(read_some(self._reader)(size))

    def _seen(self, data: bytes) -> bytes:
        if data:
            self._observer.feed(data)
        return data

    def close(self) -> None:
        closer = getattr(self._reader, "close", None)
        if callable(closer):
            closer()


def _relay_witnessed(handler: BaseHTTPRequestHandler, resp: "UpstreamResponse", pending) -> None:
    """Relay ``resp``; then write its pending thread row with the usage it showed.

    The row is written even when the client hangs up mid-stream, with whatever
    usage had passed by then (or none).
    """
    observer = _usage_observer(resp) if pending is not None else None
    if observer is not None:
        # replace() keeps every other field, including any added later.
        resp = replace(resp, _reader=_TeeReader(resp._reader, observer))
    try:
        relay_response(handler, resp)
    finally:
        if pending is not None:
            pending.write(_observed_usage(resp, observer))


def _usage_observer(resp: "UpstreamResponse"):
    """An observer for ``resp``'s body, or None when there is nothing to watch."""
    if resp._reader is None or resp.usage:
        return None  # no body, or the forwarder already counted it
    try:
        from lib.hal.infinite_cost import UsageObserver
        from lib.hal.wire_classify import header_value
        return UsageObserver(header_value(resp.headers, "content-type"))
    except Exception:
        return None


def _observed_usage(resp: "UpstreamResponse", observer) -> Optional[dict]:
    """The reply's usage: the forwarder's own count, else what the observer saw."""
    try:
        if resp.usage:
            from lib.hal.infinite_cost import crosswire_usage
            return crosswire_usage(resp.usage)
        return observer.usage() if observer is not None else None
    except Exception:
        return None


def handle_proxy_request(
    handler: BaseHTTPRequestHandler,
    proxy: GatewayProxy,
    path: str,
) -> None:
    """End-to-end: read body → forward verbatim (retry) → relay. Fail-open.

    A transport fault talking upstream is a clean 503 (fleet findings f48,
    f68); any other gateway-internal exception degrades to a clean redacted
    502 — the gateway never crashes the request, so the operator can always
    fall back to direct ``api.anthropic.com``.
    """
    started = time.monotonic()
    body = read_request_body(handler) if handler.command != "GET" else None
    client_headers = list(handler.headers.items())
    try:
        # Cross-wire (translated) OR verbatim — both yield an UpstreamResponse,
        # relayed identically below. Building it stays INSIDE the try so a
        # translator/empty-key failure maps to a clean 502; the relay itself is
        # OUTSIDE the try (a mid-stream relay error must not try to re-send a
        # 502 over an already-started response — same contract as the verbatim
        # path has always had).
        resp = _maybe_cross_wire(path, handler.command, body, client_headers)
        if resp is None:
            resp = proxy.forward(path, body, client_headers, method=handler.command)
    except spend_guard.SpendRefused as exc:
        send_spend_refusal(handler, exc.decision)
        return
    except Exception as exc:  # fail-open backstop — never propagate a crash
        send_gateway_error(handler, *failure_reply(exc))
        return
    pending = _witness_messages_rail(body, resp, started, method=handler.command,
                                     path=path, headers=client_headers)
    _relay_witnessed(handler, resp, pending)


def _inject_models_list(raw: bytes) -> bytes:
    """Add a top-level ``models`` array alongside Anthropic's ``data`` list.

    Anthropic's ``GET /v1/models`` answers ``{"data": [...], "has_more": ...}``,
    but Codex 0.136's ``models_manager`` deserialises the response into a struct
    with a REQUIRED ``models`` field and fails ("missing field `models`") on the
    Anthropic shape. This injects ``models = [{"id": ...}, ...]`` derived from
    ``data`` so a Codex client is satisfied while ``data`` is preserved verbatim
    for Anthropic-shape clients (additive, not a rewrite).

    Fail-open by construction: a non-dict body, an error envelope (no ``data``
    list), an already-present ``models`` key, or unparseable JSON all return
    ``raw`` UNCHANGED — never worse than the pre-existing verbatim relay.
    """
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return raw
    if not isinstance(obj, dict) or "models" in obj:
        return raw
    data = obj.get("data")
    if not isinstance(data, list):
        return raw
    models = [
        {"id": item["id"]}
        for item in data
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    if not models:
        return raw
    obj["models"] = models
    return json.dumps(obj).encode("utf-8")


def declared_model_entries(registry=None) -> List[dict]:
    """OpenAI-list-shaped entries for every model ``config/providers.yaml``
    declares, in HAL's canonical ``<provider>/<model>`` address form — which
    the gateway routes by explicit prefix, so a harness that picks an id from
    this list reaches the right provider. Offline: the registry reads the YAML
    and resolves no key. Fail-open: any registry failure is an empty list,
    which hands ``GET /v1/models`` back to the verbatim passthrough.
    """
    try:
        if registry is None:
            from lib.hal.registry import get_registry  # lazy: keeps gateway import light
            registry = get_registry()
        infos = registry.list_providers()
    except Exception:
        return []
    canonical = [{"id": f"{info.name}/{model}", "object": "model", "owned_by": info.name}
                 for info in infos for model in info.models]
    # Claude Code's /model picker keeps only ids containing "claude" or
    # "anthropic", so almost none of the above can be SELECTED there even
    # though all of them route. Opt-in aliases make them selectable; the
    # canonical ids stay first and unchanged for every other client.
    from lib.hal.gateway_picker_alias import alias_entries, picker_aliases_enabled
    if not picker_aliases_enabled():
        return canonical
    return canonical + alias_entries(canonical)


def _has_upstream_credential(headers: Iterable[Tuple[str, str]]) -> bool:
    """Whether the client sent something the Anthropic passthrough could use."""
    return any(name.lower() in ("x-api-key", "authorization") and str(value).strip()
               for name, value in headers)


def _fetch_upstream_models(proxy, path: str, client_headers) -> Optional[bytes]:
    """Anthropic's ``GET /v1/models`` body on a 2xx, else ``None``. Never raises:
    a failed passthrough hides nothing the gateway itself knows."""
    try:
        resp = proxy.forward(path, None, client_headers, method="GET")
    except Exception:
        return None
    try:
        raw = resp.read_all()
    except Exception:  # the upstream failed mid-body
        return None
    finally:
        resp.close()
    return raw if 200 <= resp.status < 300 else None


def _merge_models(upstream_raw: Optional[bytes], declared: List[dict]) -> bytes:
    """Anthropic's list (when present) followed by the declared entries,
    de-duplicated by id, with the Codex ``models`` array injected."""
    obj = _json_dict(upstream_raw) if upstream_raw else None
    if obj is None or not isinstance(obj.get("data"), list):
        obj = {"object": "list", "data": []}
    seen = {item.get("id") for item in obj["data"] if isinstance(item, dict)}
    obj["data"].extend(entry for entry in declared if entry["id"] not in seen)
    return _inject_models_list(json.dumps(obj).encode("utf-8"))


def _send_buffered(handler: BaseHTTPRequestHandler, status: int,
                   headers: Iterable[Tuple[str, str]], body: bytes) -> None:
    """Write one buffered response with a correct ``Content-Length``."""
    handler.send_response(status)
    for name, value in headers:
        handler.send_header(name, value)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Connection", "close")
    handler.end_headers()
    try:
        handler.wfile.write(body)
    except (BrokenPipeError, ConnectionResetError):
        pass


def _relay_upstream_models(handler, proxy, path: str, client_headers) -> None:
    """The pre-discovery ``GET /v1/models``: a verbatim Anthropic GET with the
    Codex ``models`` array injected; a transport fault is a redacted 503, any
    other failure a redacted 502."""
    try:
        resp = proxy.forward(path, None, client_headers, method="GET")
    except spend_guard.SpendRefused as exc:
        send_spend_refusal(handler, exc.decision)
        return
    except Exception as exc:  # fail-open backstop — never propagate a crash
        send_gateway_error(handler, *failure_reply(exc))
        return
    try:
        body = _inject_models_list(resp.read_all())
    except Exception as exc:  # the upstream failed mid-body; nothing was sent yet
        send_gateway_error(handler, *failure_reply(exc))
        return
    finally:
        resp.close()  # release the upstream socket
    _send_buffered(handler, resp.status, response_headers(resp.headers), body)


def handle_models_request(
    handler: BaseHTTPRequestHandler, proxy: GatewayProxy, path: str,
    *, declared: Optional[List[dict]] = None,
) -> None:
    """``GET /v1/models``: what this gateway can actually route.

    The ids declared in ``config/providers.yaml`` are served OFFLINE from the
    provider registry (``declared``, injectable for tests), so an OpenAI-wire
    harness that enumerates models sees ``moonshot/kimi-k3`` and friends with
    no Anthropic credential at all — before this, the route was a verbatim
    Anthropic GET and such a client got Anthropic's 401. Anthropic's own list
    is merged in ahead of the declared entries ONLY when the client sent a
    credential (``x-api-key`` or ``Authorization``); a failed or unreachable
    passthrough still yields the declared list as a 200. The list is small and
    finite, so the body is buffered and carries a correct ``Content-Length``,
    plus the Codex-0.136 ``models`` array. With nothing declared (no
    providers.yaml) the pre-discovery verbatim passthrough runs unchanged.
    """
    client_headers = list(handler.headers.items())
    if declared is None:
        declared = declared_model_entries()
    if not declared:
        _relay_upstream_models(handler, proxy, path, client_headers)
        return
    upstream_raw = None
    if _has_upstream_credential(client_headers):
        upstream_raw = _fetch_upstream_models(proxy, path, client_headers)
    _send_buffered(handler, 200, [("Content-Type", "application/json")],
                   _merge_models(upstream_raw, declared))


# --- standalone dedicated-port gateway --------------------------------------

def handle_health_request(handler) -> None:
    """Local liveness identity for launch probes (ADR-0002 W0).

    Answers from the gateway itself: no upstream call, no auth, no body
    read — a probe must be free, instant, and side-effect-less. GRIP's
    GRIP_INFINITE=1 launch seam requires ``ok`` + ``service`` to match
    before wiring a session onto the rails, so a foreign process squatting
    the port can never capture claude traffic. Also reports the
    InfiniteMachine brake state (lib/hal/gateway_brake.py) so kill-switch
    engagement is verifiable in one request.
    """
    from lib.hal.gateway_brake import brake_state  # lazy: handler-import style
    body = json.dumps(
        {"ok": True, "service": "hal-gateway", "brake": brake_state()}
    ).encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class InflightCap:
    """At most ``limit`` requests served at once; past it, a request is turned
    away at once with a 503 and ``Retry-After``, never queued and never a 429
    (InfiniteMachine v2, Step 5a; fleet findings f82, f121).

    The gateway serves each request on its own thread, and a fan-out (12 agents
    lost in 140 s on 2026-09-01) could otherwise take every thread and socket
    the process has; holding requests (Step 5c) makes that likelier still. Read
    from the environment when a server is built, ``<prefix>`` being
    ``HAL_GATEWAY`` for the gateway and ``HAL_WIRE`` for the wire:

      <prefix>_MAX_INFLIGHT        64  requests served at once
      <prefix>_BUSY_RETRY_AFTER_S   5  the Retry-After sent past the cap
    """

    def __init__(self, limit: int = 64, retry_after_s: int = 5):
        self.limit = limit
        self.retry_after_s = retry_after_s
        self._lock = threading.Lock()
        self._inflight = 0

    @classmethod
    def from_env(cls, prefix: str = "HAL_GATEWAY", environ=None) -> "InflightCap":
        env = os.environ if environ is None else environ
        limit = _positive_number(env.get(f"{prefix}_MAX_INFLIGHT"), 64)
        retry = _positive_number(env.get(f"{prefix}_BUSY_RETRY_AFTER_S"), 5)
        return cls(max(1, int(limit)), max(1, math.ceil(retry)))

    @property
    def inflight(self) -> int:
        with self._lock:
            return self._inflight

    def try_enter(self) -> bool:
        """Take a slot; False, at once, when every slot is taken."""
        with self._lock:
            if self._inflight >= self.limit:
                return False
            self._inflight += 1
            return True

    def leave(self) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)

    def busy_message(self) -> str:
        return (f"HAL is serving its limit of {self.limit} requests at once; "
                f"retry after {self.retry_after_s} s")

    def busy_headers(self) -> List[Tuple[str, str]]:
        return [("Retry-After", str(self.retry_after_s))]


#: The OpenAI-wire routes: Codex speaks /v1/responses, OpenCode and Hermes the
#: chat route. Every other /v1/* path speaks Anthropic's dialect.
_OPENAI_WIRE_PATHS = frozenset({"/v1/chat/completions", "/v1/responses"})


def _turn_away(handler: BaseHTTPRequestHandler, path: str, cap: InflightCap) -> None:
    """Answer a request past the cap: 503 with Retry-After, in the client's dialect."""
    if handler.command != "GET":
        read_request_body(handler)  # off the connection first, so the answer is read
    if path in _OPENAI_WIRE_PATHS:
        from lib.hal.gateway_openai import _send_openai_error  # lazy: import cycle
        _send_openai_error(handler, 503, cap.busy_message(), cap.busy_headers())
    else:
        send_gateway_error(handler, 503, cap.busy_message(), cap.busy_headers())


def make_gateway_handler(proxy: GatewayProxy, cap: InflightCap):
    """Build a BaseHTTPRequestHandler class bound to ``proxy``. Every ``/v1/*``
    request holds one slot of ``cap`` while it is served; ``/health`` never
    does. ``cap`` is explicit: the standalone server passes
    ``InflightCap.from_env()`` (read once, at server build), and tests pass a
    fixed ``InflightCap(limit, retry)``, so the environment is never read as a
    hidden default."""

    class _GatewayHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):  # noqa: N802 (stdlib naming)
            path = self.path.split("?", 1)[0]
            if not path.startswith("/v1/"):
                _discard_unread_body(self)
                self.send_error(404)
                return
            # Access guard first: default-deny remote /v1/* (open-relay
            # protection). Only an allowed request reaches the proxy.
            if not enforce_gateway_access(self):
                _discard_unread_body(self)
                return
            self._capped(path, self._post)

        def _post(self, path: str) -> None:
            # Lazy imports break the gateway <- gateway_openai/responses cycle.
            if path == "/v1/chat/completions":
                from lib.hal.gateway_openai import handle_chat_completions
                handle_chat_completions(self, proxy)
            elif path == "/v1/responses":
                from lib.hal.gateway_responses import handle_responses
                handle_responses(self, proxy)
            else:
                handle_proxy_request(self, proxy, self.path)

        def do_GET(self):  # noqa: N802
            # Sibling GETs behind the access guard (default-deny remote /v1/*).
            # /v1/models is transformed to add a Codex-0.136-compatible top-level
            # ``models`` array; every other GET relays verbatim.
            path = self.path.split("?", 1)[0]
            if path == "/health":
                handle_health_request(self)
                return
            if path.startswith("/v1/"):
                if not enforce_gateway_access(self):
                    return
                self._capped(path, self._get)
            else:
                self.send_error(404)

        def _get(self, path: str) -> None:
            if path == "/v1/models":
                handle_models_request(self, proxy, self.path)
            else:
                handle_proxy_request(self, proxy, self.path)

        def _capped(self, path: str, serve) -> None:
            """Serve inside one slot of the cap, or turn the request away."""
            if not cap.try_enter():
                _turn_away(self, path, cap)
                return
            try:
                serve(path)
            finally:
                cap.leave()

        def log_message(self, fmt, *args):
            pass  # custody: never log request lines by default

    return _GatewayHandler


class GatewayHTTPServer(ThreadingHTTPServer):
    """Shared gateway/wire listener; backlog must be set before listen()."""

    request_queue_size = 128

    def server_bind(self) -> None:
        """Bind with strict Windows ownership and unchanged POSIX reuse.

        Windows' SO_REUSEADDR can bind over an active listener. Using
        SO_EXCLUSIVEADDRUSE there preserves the EADDRINUSE signal required by
        the gateway's holder detection and reclaim path. super().server_bind()
        still runs on every platform (http.server.HTTPServer.server_bind sets
        server_name/server_port, which socketserver.TCPServer.server_bind
        alone does not) -- only the socket option differs; POSIX behaviour is
        otherwise byte-identical to the stdlib's.

        Measured on the real Windows 11 VM (tests/test_gateway_windows_rebind_
        measurement.py): a listener can rebind its port immediately, with no
        retry, right after the previous listener on that port closed, even
        while one of its accepted connections -- closed by the server, so the
        server's side is the one left in TIME_WAIT -- is still in that state.
        SO_EXCLUSIVEADDRUSE therefore needs no bind-retry loop in
        _bind_gateway: the idempotent-relaunch path this class exists for
        already works first try on Windows, exactly as it does on POSIX.
        """
        if sys.platform != "win32":
            super().server_bind()
            return
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        reuse, self.allow_reuse_address = self.allow_reuse_address, False
        try:
            super().server_bind()
        finally:
            self.allow_reuse_address = reuse


def _http_port_holder(port: int) -> Optional[str]:
    """Identify an HTTP responder without proxies, redirects or unbounded reads.

    Any HTTP response protects the holder, even an error or an unrecognised
    health body. Only a listener that gives no HTTP response may be reclaimed.
    Never include the holder's arbitrary response content in diagnostics.
    """
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1.0)
    try:
        try:
            connection.request("GET", "/health")
            response = connection.getresponse()
        except (OSError, http.client.HTTPException):
            return None
        try:
            body = json.loads(response.read(4096))
            if (response.status == 200 and isinstance(body, dict)
                    and body.get("service") == "hal-gateway"):
                return "healthy hal-gateway"
        except (OSError, http.client.HTTPException, ValueError):
            pass
        return "responding HTTP service"
    finally:
        connection.close()


def _bind_gateway(
    port: int,
    handler: type,
    *,
    reclaim: bool = True,
    attempts: int = 4,
) -> ThreadingHTTPServer:
    """Bind on loopback, reclaiming only holders that do not answer HTTP.

    A healthy gateway or any other HTTP responder is left alone. Strict mode
    (``reclaim=False``) and errors other than EADDRINUSE propagate immediately.
    """
    last_exc: Optional[OSError] = None
    for _attempt in range(max(1, attempts)):
        # allow_reuse_address on THIS instance (before bind) rather than
        # mutating the stdlib ThreadingHTTPServer class globally — a global
        # mutation would persist to any unrelated server created later.
        server = GatewayHTTPServer(
            ("127.0.0.1", port), handler, bind_and_activate=False  # 127.0.0.1 ONLY
        )
        server.allow_reuse_address = True
        try:
            server.server_bind()
            server.server_activate()
            return server
        except OSError as exc:
            server.server_close()
            last_exc = exc
            if exc.errno != errno.EADDRINUSE or not reclaim:
                raise
            holder = _http_port_holder(port)
            if holder is not None:
                raise OSError(
                    errno.EADDRINUSE,
                    f"port {port} held by {holder}; refusing reclaim",
                ) from exc
            killed = reclaim_port(port)
            sys.stderr.write(
                f"  HAL Gateway: port {port} in use"
                f"{f' (killed pid {killed})' if killed else ''}"
                f" — reclaimed, rebinding\n"
            )
            time.sleep(0.25)  # let the OS release the socket before retrying
    assert last_exc is not None  # loop runs >=1x; EADDRINUSE path sets last_exc
    raise last_exc


def serve(
    port: int = DEFAULT_GATEWAY_PORT,
    upstream_base: str = DEFAULT_UPSTREAM,
    *,
    reclaim: bool = True,
) -> None:
    """Run the standalone gateway on 127.0.0.1:port (blocking).

    Uses the P3 failover-aware proxy with one upstream, ``HAL_GATEWAY_UPSTREAM``
    (the env fallback rung was removed with the spend guard, Step 5h). Lazy
    import breaks the gateway <- gateway_failover cycle. With one upstream this
    is behaviourally identical to a single-upstream GatewayProxy.

    If ``port`` is held, only a listener that does not answer HTTP is
    reclaimed (see :func:`_bind_gateway`). HTTP responders are left alone.
    Pass ``reclaim=False`` (or set ``HAL_GATEWAY_NO_RECLAIM=1``)
    to fail fast on ``EADDRINUSE`` instead.
    """
    from lib.hal.gateway_failover import build_failover_proxy
    proxy = build_failover_proxy(upstream_base=upstream_base)
    handler = make_gateway_handler(proxy, InflightCap.from_env())
    server = _bind_gateway(port, handler, reclaim=reclaim)
    sys.stderr.write(
        f"\n  HAL Gateway — http://127.0.0.1:{port}  ->  {upstream_base}\n"
        f"  point a harness at it:  export ANTHROPIC_BASE_URL=http://127.0.0.1:{port}\n"
        f"  Ctrl+C to stop\n\n"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


def _reclaim_enabled() -> bool:
    """Whether idempotent port-reclaim is on (default yes; env opt-out)."""
    return os.environ.get("HAL_GATEWAY_NO_RECLAIM", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    )


def _upstream_url(value: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(value)
        valid = parsed.scheme in ("http", "https") and bool(parsed.hostname)
    except ValueError:
        valid = False
    if not valid:
        raise argparse.ArgumentTypeError("upstream must be an http(s) URL with a host")
    return value


def main(argv: Optional[List[str]] = None) -> int:
    """Parse all arguments before constructing a proxy or opening a socket."""
    parser = argparse.ArgumentParser(
        description="HAL loopback gateway", allow_abbrev=False)
    parser.add_argument("positional_port", nargs="?", type=int,
                        help="legacy port number")
    parser.add_argument("--port", type=int,
                        help=f"listen port (default {DEFAULT_GATEWAY_PORT})")
    parser.add_argument("--upstream", type=_upstream_url, default=os.environ.get(
        "HAL_GATEWAY_UPSTREAM", DEFAULT_UPSTREAM))
    parser.add_argument("--no-reclaim", action="store_true",
                        help="never reclaim an occupied port")
    args = parser.parse_args(argv)
    if args.port is not None and args.positional_port is not None:
        parser.error("use either --port or a positional port, not both")
    port = args.port if args.port is not None else args.positional_port
    if port is None:
        port = DEFAULT_GATEWAY_PORT
    if not 0 <= port <= 65535:
        parser.error("port must be between 0 and 65535")
    try:
        serve(port=port, upstream_base=args.upstream,
              reclaim=not args.no_reclaim and _reclaim_enabled())
    except OSError as exc:
        sys.stderr.write(f"HAL Gateway: {exc.strerror}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
