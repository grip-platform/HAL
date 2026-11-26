"""HAPPI envelope — stdin request shape.

Canonical wire form: one JSON object per invocation, consumed from stdin.
Spec: ``happi.md`` (the polyglot file IS the specification — axiom 4).

**The accepted set is ``SUPPORTED_VERSIONS`` below; do not restate it here.**
A prose copy of a constant is a second source of truth that no test can fail
on, and this docstring carried a stale one for two releases.

Every accepted version parses identically. Versions are additive vocabulary,
not parser branches: the implementation reads ``v`` exactly once, as a
set-membership test, so no version changes how an envelope is parsed — only
which cmds and event types are meaningful.

    {
      "v": "happi/1.0",              # protocol version; see SUPPORTED_VERSIONS
      "id": "req-01HX...",           # caller-chosen correlation id, required
      "cmd": "brief",                # command namespace, required
      "args": ["me about my day"],   # positional args
      "flags": {"stream": true},     # named flags
      "auth": {"scheme": "...",      # optional; α-1 does not consume
               "token": "..."},
      "ctx":  {"cwd": "...",         # optional caller context
               "timezone": "..."},
      "capabilities": ["read:stdin"] # declared capabilities (α-1 advisory only)
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


SUPPORTED_VERSIONS = frozenset({"happi/1.0", "happi/1.1", "happi/1.2",
                                "happi/1.3", "happi/1.4", "happi/1.5",
                                "happi/1.6"})


class EnvelopeError(ValueError):
    """Envelope parse or validation failure."""


def _decode_envelope(raw: str | bytes | dict[str, Any]) -> dict[str, Any]:
    """Concern 1 — deserialise raw input to a dict.

    Accepts str/bytes (JSON) or a pre-decoded dict.
    Raises EnvelopeError on JSON decode failure or non-dict result.
    CC: 4 (base 1 + isinstance branch + except + not-isinstance guard)
    """
    if isinstance(raw, (str, bytes)):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EnvelopeError(f"invalid JSON: {exc.msg}") from exc
    else:
        data = raw

    if not isinstance(data, dict):
        raise EnvelopeError("envelope must be a JSON object")

    return data


def _validate_envelope(data: dict[str, Any]) -> None:
    """Concern 2 — field validation; raises EnvelopeError on any violation.

    Validates version (closed set), non-empty id, non-empty cmd,
    list-type args, and list-type capabilities.
    CC: 6 (base 1 + version check + id isinstance + id empty + cmd isinstance
            + cmd empty + args check + caps check = 8 decision points, but
            each compound `if A or B` counts as 2; re-counted accurately below)

    McCabe: base 1
      + if version not in SUPPORTED_VERSIONS       → +1
      + if not isinstance(env_id, str)             → +1
      + or not env_id                              → +1
      + if not isinstance(cmd, str)               → +1
      + or not cmd                                → +1
      + if not isinstance(args_raw, list)         → +1
      + if not isinstance(caps_raw, list)         → +1
    = CC 8  (all guards; no branching beyond raises — grade B)
    """
    version = data.get("v")
    if version not in SUPPORTED_VERSIONS:
        raise EnvelopeError(f"unsupported protocol version: {version!r}")

    env_id = data.get("id")
    if not isinstance(env_id, str) or not env_id:
        raise EnvelopeError("envelope missing non-empty 'id'")

    cmd = data.get("cmd")
    if not isinstance(cmd, str) or not cmd:
        raise EnvelopeError("envelope missing non-empty 'cmd'")

    args_raw = data.get("args", [])
    if not isinstance(args_raw, list):
        raise EnvelopeError("'args' must be a list")

    caps_raw = data.get("capabilities", [])
    if not isinstance(caps_raw, list):
        raise EnvelopeError("'capabilities' must be a list")


def _build_envelope_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Concern 3 — construct kwargs for the HappiEnvelope dataclass.

    All optional fields use null-coalescing defaults; args/capabilities
    are stringified to tuple. Assumes _validate_envelope has already run.
    CC: 5 (base 1 + 2 generators + 4 `or` operators = 7; but generators
            count as 1 each and `or` in assignments count as 1 each)

    McCabe: base 1
      + tuple(str(a) for a in ...)   → +1
      + data.get("flags", {}) or {}  → +1
      + data.get("auth", {}) or {}   → +1
      + data.get("ctx", {}) or {}    → +1
      + str(data.get("content") or "") → +1
      + tuple(str(c) for c in ...)   → +1
    = CC 7  (grade B)
    """
    args_raw = data.get("args", [])
    caps_raw = data.get("capabilities", [])
    return {
        "v": data.get("v"),
        "id": data.get("id"),
        "cmd": data.get("cmd"),
        "args": tuple(str(a) for a in args_raw),
        "flags": dict(data.get("flags", {}) or {}),
        "auth": dict(data.get("auth", {}) or {}),
        "ctx": dict(data.get("ctx", {}) or {}),
        "capabilities": tuple(str(c) for c in caps_raw),
        "content": str(data.get("content") or ""),
    }


@dataclass
class HappiEnvelope:
    """Parsed HAPPI/1.0 request envelope.

    Deliberately NOT frozen — downstream code may attach derived fields
    (resolved auth, precheck verdict) without reconstructing the whole
    record. Callers that need immutability should snapshot via `to_dict`.
    """

    v: str
    id: str
    cmd: str
    args: tuple[str, ...] = ()
    flags: dict[str, Any] = field(default_factory=dict)
    auth: dict[str, Any] = field(default_factory=dict)
    ctx: dict[str, Any] = field(default_factory=dict)
    capabilities: tuple[str, ...] = ()
    content: str = ""

    @classmethod
    def parse(cls, raw: str | bytes | dict[str, Any]) -> HappiEnvelope:
        """Parse a raw envelope (JSON string, bytes, or pre-decoded dict).

        Raises EnvelopeError on any structural or version violation so callers
        can emit a single `error` event and exit without partial dispatch.

        Coordinator (CC: 2 — base 1 + 0 branches; delegates all logic to
        module-private helpers):
          _decode_envelope  → concern 1: deserialise + dict guard
          _validate_envelope → concern 2: field validation + EnvelopeError
          _build_envelope_fields → concern 3: kwargs construction
        """
        data = _decode_envelope(raw)
        _validate_envelope(data)
        return cls(**_build_envelope_fields(data))

    def user_message(self) -> str:
        """Resolve the user-facing prompt from this envelope.

        Canonical precedence (PARAMOUNT — every `chat` provider adapter MUST
        use this; do not re-implement it per adapter):

            1. ``content``        — the primary prompt-carrying field. This is
                                     what GRIP's council dispatch
                                     (lib/hal_council_dispatch.py `_build_envelope`)
                                     and AGORA PR review populate.
            2. ``flags.message``  — legacy single-field convenience.
            3. ``" ".join(args)`` — α-1 echo-style positional fallback.

        Why this exists: the three provider adapters historically resolved the
        message independently. `anthropic.py` honoured ``content``;
        `openai_compat.py` and `gemini.py` only looked at ``flags.message`` and
        ``args``. An envelope that carried the prompt solely in ``content``
        (the documented primary field — see this module's docstring and the
        ``content`` attribute) therefore reached Anthropic with the real
        prompt but reached every OpenAI-compat / Gemini provider with an
        EMPTY user message. Those models then free-associated ("Please provide
        the code you would like me to review"), which AGORA's validity gate
        correctly rejected as filler — surfacing as "AGORA Council Review
        FAILED" on real PRs (e.g. CodeTonight-SA/GRIP#2697). Centralising the
        precedence here makes the envelope contract honoured uniformly and
        kills the drift class at its source (syscall-doctrine: one shim, one
        contract, every caller goes through it).

        Returns the empty string only when the envelope genuinely carries no
        prompt in any of the three fields — callers/providers keep their own
        empty-input handling, and AGORA's empty-Speaker discriminator still
        fails loud rather than hallucinating.
        """
        if self.content:
            return self.content
        flag_message = self.flags.get("message")
        if flag_message:
            return str(flag_message)
        return " ".join(self.args)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "v": self.v,
            "id": self.id,
            "cmd": self.cmd,
            "args": list(self.args),
            "flags": dict(self.flags),
            "auth": dict(self.auth),
            "ctx": dict(self.ctx),
            "capabilities": list(self.capabilities),
        }
        if self.content:
            d["content"] = self.content
        return d
