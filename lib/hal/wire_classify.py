"""wire_classify -- sort one decrypted api.anthropic.com request into a route class.

InfiniteMachine v2, Step 2 (``docs/infinite-machine-contract.md``, "Route
classes"). The HAL wire (``lib/hal/wire_proxy.py``) decrypts Claude Code's
traffic to ``api.anthropic.com`` and asks this module, once per request, what
the request is. The answer decides what HAL may do with it:

  inference      a real conversation turn on ``POST /v1/messages`` -- witnessed,
                 priced, and the only class failover (Step 5) may ever act on.
  aux            a call on the same path family that is not a turn -- witnessed,
                 never swapped to another account.
  account-bound  a call bound to the signed-in account -- byte-exact pass-through.
  unknown        anything else -- pass-through plus an audit row (method, path).

``classify`` is pure: no I/O, no logging, no network. Headers are inputs and are
never logged; this module writes nothing anywhere.

How each class is recognised, and the evidence (read from the Claude Code
2.1.289 binary on 2026-10-05 and from the 2026-10-04 intercept probe,
GRIP ``tests/fixtures/wire_probe/``):

* **The quota check.** Claude Code's debug log labels it ``source=quota_check``,
  but that label never reaches the wire: the fetch wrapper prints it and sets
  only ``x-client-request-id`` and four other unrelated headers. The request
  itself is built as ``{model, max_tokens: 1, messages: [{role: "user",
  content: "quota"}], metadata}`` -- it asks for one token because it only
  wants the ``anthropic-ratelimit-unified-*`` response headers. So the body is
  distinguishable, and the smallest honest signal is ``max_tokens == 1``: one
  token cannot carry a turn's answer. The word "quota" is incidental text and is
  deliberately NOT a signal (a user may type it). The same rule also catches the
  API-key check (``max_tokens: 1``, content "test"), which likewise reads the
  account rather than holding a conversation. JSON ``true`` is not the integer 1.
* **Token counts** are ``POST /v1/messages/count_tokens``. **Cache keep-alives**
  (``POST /v1/messages/cache_touch``, Claude Code's "anthropic-cache-keepalive",
  every 10-120 s) are aux too: a non-turn on the same path family, per account.
* **Title calls.** Claude Code's session-title helpers ask for structured output
  whose JSON schema has exactly the properties ``{title}`` or ``{title,
  branch}``; on the wire that is ``output_config.format`` (the SDK rewrites the
  deprecated ``output_format`` there, so both are read). A title helper running
  on a model without structured-output support sends no schema and is
  classified ``inference`` -- the harmless direction. The rename helper
  (schema ``{name}``) forks the whole conversation, so it stays ``inference``.
* **Gzip.** Claude Code 2.1.289 can gzip request bodies (its
  ``api_gzip_request_body`` rollout sets ``Content-Encoding: gzip``). Without
  decoding, every turn would read as opaque and fall to ``unknown``, out of
  failover's reach. So ``Content-Encoding`` is read; gzip is decoded under a
  size cap; any other coding leaves the body opaque (``unknown``).
* **The gateway hint header.** With ``CLAUDE_CODE_GATEWAY_HINT_HEADERS`` set,
  Claude Code sends ``x-claude-code-request-class`` (``main``, ``subagent``,
  ``workflow``, ``compaction`` or ``auxiliary``). It is used in one direction
  only: a request it names a turn is never put in ``aux`` by a body heuristic.
  ``auxiliary`` is NOT used to widen ``aux``: it also covers side questions,
  the auto-mode classifier and web-fetch summaries, which the user sees, and
  the contract's promise covers.

The asymmetry behind every choice above: mistaking a turn for ``aux`` exempts it
from failover, so a rate-limit error could reach the user -- the bug the
contract forbids. Mistaking an aux call for ``inference`` costs at most a
needless switch of a call nobody sees. So when in doubt, a ``messages`` request
is ``inference``.

``account-bound`` is the contract's list (``/v1/code/*``, ``/v1/environments*``,
``/api/oauth/*``, ``/api/ws/speech_to_text/*``), the OAuth token endpoint
``/v1/oauth/*`` (the sign-in's own credential), and the settings and telemetry
endpoints the probe saw: ``/api/claude_code*`` (grove, penguin mode,
notification preferences, policy limits), ``/api/claude_cli*`` (bootstrap,
feedback), ``/api/event_logging/*`` and ``/api/eval/*`` (feature flags). Other
endpoints the probe saw, such as ``GET /v1/ultrareview/quota``, are neither, so
they stay ``unknown`` and each use leaves an audit row until someone classifies
them on evidence.
"""
from __future__ import annotations

import gzip
import io
import json
import zlib
from enum import Enum
from typing import Any, Iterable, Optional

MESSAGES_PATH = "/v1/messages"

#: Non-turn calls on the /v1/messages path family, recognised by path alone.
AUX_PATHS = frozenset({"/v1/messages/count_tokens", "/v1/messages/cache_touch"})

#: Roots matched as a path segment: the root itself or anything below it.
ACCOUNT_BOUND_TREES = (
    "/v1/code", "/v1/oauth", "/api/oauth", "/api/ws/speech_to_text",
    "/api/event_logging", "/api/eval",
)

#: Roots matched as a plain string prefix (``/v1/environments*`` in the contract).
ACCOUNT_BOUND_PREFIXES = ("/v1/environments", "/api/claude_code", "/api/claude_cli")

#: A request asking for exactly this many tokens is a probe of the account.
PROBE_MAX_TOKENS = 1

#: Property sets of the structured-output schemas Claude Code's title helpers use.
TITLE_SCHEMAS = (frozenset({"title"}), frozenset({"title", "branch"}))

#: The gateway hint header, and the values that name a request a real turn.
HINT_HEADER = "x-claude-code-request-class"
HINTED_TURNS = frozenset({"main", "subagent", "workflow", "compaction"})

#: Largest decoded body HAL will parse (the Messages API caps requests at 32 MB).
MAX_DECODED_BODY = 64 * 1024 * 1024


class RouteClass(str, Enum):
    """The contract's four route classes."""

    INFERENCE = "inference"
    AUX = "aux"
    ACCOUNT_BOUND = "account-bound"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value


def classify(method: str, path: str, headers: Any,
             body_bytes: Optional[bytes]) -> RouteClass:
    """The route class of one decrypted request. Total: never raises.

    ``path`` is the request target (``/v1/messages?beta=true``); ``headers`` is a
    mapping, a ``(name, value)`` iterable or an ``http.client.HTTPMessage``;
    ``body_bytes`` is the raw request body as it arrived.
    """
    route = request_path(path)
    if is_account_bound(route):
        return RouteClass.ACCOUNT_BOUND
    if (method or "").upper() != "POST":
        return RouteClass.UNKNOWN
    if route in AUX_PATHS:
        return RouteClass.AUX
    if route != MESSAGES_PATH:
        return RouteClass.UNKNOWN
    return _classify_messages(request_json(headers, body_bytes), headers)


def _classify_messages(payload: Optional[dict], headers: Any) -> RouteClass:
    """A ``POST /v1/messages`` body: a turn, an aux call, or unreadable."""
    if payload is None or not _has_messages(payload):
        return RouteClass.UNKNOWN
    if _hinted_turn(headers):
        return RouteClass.INFERENCE
    if is_probe(payload) or is_title_call(payload):
        return RouteClass.AUX
    return RouteClass.INFERENCE


def request_path(target: str) -> str:
    """The path of a request target, without its query string."""
    return (target or "").split("?", 1)[0]


def is_account_bound(route: str) -> bool:
    """Whether a path belongs to the signed-in account (byte-exact pass-through)."""
    for root in ACCOUNT_BOUND_TREES:
        if route == root or route.startswith(root + "/"):
            return True
    return route.startswith(ACCOUNT_BOUND_PREFIXES)


def is_probe(payload: dict) -> bool:
    """A one-token request: an account probe (quota or key check), never a turn."""
    max_tokens = payload.get("max_tokens")
    return type(max_tokens) is int and max_tokens == PROBE_MAX_TOKENS


def is_account_probe(headers: Any, payload: dict) -> bool:
    """A one-token request this module puts in ``aux``. The gateway hint header can
    name such a request a turn, and then it is a turn whatever its ``max_tokens``:
    the same rule ``_classify_messages`` applies (HAL#765, Codex P1)."""
    return is_probe(payload) and not _hinted_turn(headers)


def is_title_call(payload: dict) -> bool:
    """Structured output whose schema is exactly a title helper's."""
    fmt = _output_format(payload)
    if not isinstance(fmt, dict) or fmt.get("type") != "json_schema":
        return False
    schema = fmt.get("schema")
    props = schema.get("properties") if isinstance(schema, dict) else None
    return isinstance(props, dict) and frozenset(props) in TITLE_SCHEMAS


def _output_format(payload: dict) -> Any:
    config = payload.get("output_config")
    if isinstance(config, dict) and "format" in config:
        return config.get("format")
    return payload.get("output_format")


def _has_messages(payload: dict) -> bool:
    messages = payload.get("messages")
    return isinstance(messages, list) and len(messages) > 0


def _hinted_turn(headers: Any) -> bool:
    hint = header_value(headers, HINT_HEADER).strip().lower()
    return hint in HINTED_TURNS


def request_json(headers: Any, body_bytes: Optional[bytes]) -> Optional[dict]:
    """The request body as a JSON object, or None when it cannot be read as one."""
    raw = decoded_body(headers, body_bytes)
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except (ValueError, RecursionError):
        return None
    return payload if isinstance(payload, dict) else None


#: Every Content-Encoding HAL reads, whole (:func:`decoded_body`) or as a body
#: streams by (:class:`StreamDecoder`): none at all, and gzip. Measured
#: 2026-10-08 through the wire: offered gzip, br or zstd alone, api.anthropic.com
#: compressed a streamed reply for gzip only and sent the other two uncompressed.
_PLAIN_CODINGS = ("", "identity")
_GZIP_CODINGS = ("gzip", "x-gzip")


def decoded_body(headers: Any, body_bytes: Optional[bytes]) -> Optional[bytes]:
    """The body with its ``Content-Encoding`` removed, or None if HAL can't decode it."""
    coding = header_value(headers, "content-encoding").strip().lower()
    if coding in _PLAIN_CODINGS:
        return bytes(body_bytes or b"")
    if coding in _GZIP_CODINGS:
        return _gunzip(bytes(body_bytes or b""))
    return None


def _gunzip(data: bytes) -> Optional[bytes]:
    """Decode gzip (all members) up to MAX_DECODED_BODY; None if bad or too big."""
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
            out = stream.read(MAX_DECODED_BODY + 1)
    except (OSError, EOFError, zlib.error):
        return None
    return out if len(out) <= MAX_DECODED_BODY else None


class StreamDecoder:
    """A body's ``Content-Encoding`` removed piece by piece, as the body streams by.

    It reads what :func:`decoded_body` reads, identity and gzip (every member),
    and hands each piece back as blocks of at most ``block`` bytes, so one small
    compressed piece never becomes one huge buffer. It never raises: for a coding
    it does not know, and from the first piece it cannot decode, it yields nothing.
    """

    def __init__(self, headers: Any, block: int):
        coding = header_value(headers, "content-encoding").strip().lower()
        self.block = block
        self.gzip = coding in _GZIP_CODINGS
        self.dead = not self.gzip and coding not in _PLAIN_CODINGS
        self._inflater = zlib.decompressobj(wbits=31)

    def __call__(self, piece: bytes) -> Iterable[bytes]:
        if self.dead:
            return
        if not self.gzip:
            yield piece
            return
        try:
            yield from self._gunzip(piece)
        except zlib.error:
            self.dead = True

    def _gunzip(self, data: bytes) -> Iterable[bytes]:
        while data:
            out = self._inflater.decompress(data, self.block)
            data = self._inflater.unconsumed_tail
            if self._inflater.eof:  # a gzip member ended; another may follow it
                data = self._inflater.unused_data
                self._inflater = zlib.decompressobj(wbits=31)
            if out:
                yield out


def header_value(headers: Any, name: str) -> str:
    """Every value of one header, joined by ``, `` (names are case-insensitive).

    Entries that are not ``(name, value)`` pairs are skipped, so a malformed
    header list can never make ``classify`` raise.
    """
    wanted = name.lower()
    found = [str(item[1]) for item in _header_items(headers)
             if isinstance(item, (tuple, list)) and len(item) == 2
             and str(item[0]).lower() == wanted]
    return ", ".join(found)


def _header_items(headers: Any) -> Iterable:
    if headers is None:
        return ()
    items = getattr(headers, "items", None)
    if callable(items):
        return items()
    try:
        return iter(headers)
    except TypeError:
        return ()


__all__ = [
    "RouteClass",
    "classify",
    "decoded_body",
    "header_value",
    "is_account_bound",
    "is_account_probe",
    "is_probe",
    "is_title_call",
    "request_json",
    "request_path",
]
