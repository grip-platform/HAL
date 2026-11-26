"""POST /api/decide on hal-server: typed decisions over HTTP (docs/decide.md).

The body is ``{"state": ..., "questions": {...}, "model": optional}`` and the
reply is a decision result. Fail-safe like /api/infer: every failure is HTTP 200
with ``ok`` false and a ``reason`` from a closed set, never a 5xx, so each caller
falls back to its own method.

The route rides on a subclass of hal-server's handler rather than in its route
table. ``_make_handler`` in lib/hal/server.py already has a cyclomatic
complexity of 101, over the gate's 30, so the gate refuses every edit inside it
until it is decomposed. When it is, this route moves into the table and this
wrapper goes. Remote access follows the table's rule for every non-public /api
route: open on loopback, a gateway token otherwise.
"""
from __future__ import annotations

from typing import Any

from lib.hal.gateway_guard import enforce_gateway_access
from lib.hal.http_refusal import _discard_unread_body

PATH = "/api/decide"


def decide_response(body: Any) -> dict:
    """The /api/decide reply for a parsed request body. Never raises."""
    body = body if isinstance(body, dict) else {}
    try:
        from lib.hal.decide import DEFAULT_MODEL, decide
        result = decide(body.get("state"), body.get("questions"),
                        model=body.get("model") or DEFAULT_MODEL)
    except Exception as exc:  # noqa: BLE001 -- fail-safe: an answer or a reason, never a 5xx
        return {"ok": False, "status": "unavailable", "reason": "provider_error",
                "detail": type(exc).__name__, "answers": {}}
    return result.to_dict()


def with_decide_route(base: type) -> type:
    """``base`` (a hal-server handler class) with POST /api/decide added; every other request unchanged."""

    class Handler(base):
        def do_POST(self):
            if self.path.split("?", 1)[0] != PATH:
                return super().do_POST()
            if not enforce_gateway_access(self):
                _discard_unread_body(self)
                return None
            self._json(decide_response(self._read_body()))
            return None

    Handler.__name__ = Handler.__qualname__ = base.__name__
    return Handler
