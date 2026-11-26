"""HAL runtime gates — default rule set + yolo-mode activation.

HAL treats "gate" and "guard" as synonyms for the same idea: a mechanical
check that runs before a potentially-dangerous action and either allows,
denies, or asks the operator. :class:`~lib.hal.guard.GuardEngine` is the
engine; this module is the *policy* — the default rule set HAL ships
with, the small coordinator that wires the engine into the runtime,
plus the autonomous-loop bypass marker (Phase B / H155).

Three collaborators:

* :mod:`lib.hal.guard` — engine (ALLOW / DENY / ASK, strictest-wins).
* :mod:`lib.hal.secrets_guard` — content scanner, reused inside the gates.
* :mod:`lib.hal.hook_dispatch` — optional file-based extension point.

Two entry points are wired in PR #125 follow-up:

* :class:`~lib.hal.pipeline.GuardStage` in ``Conversation._build_pipeline``
  now receives :func:`default_engine`, so every chat turn passes through
  the guard.
* :class:`~lib.hal.tool_dispatch.ToolDispatcher` consults the engine for
  every tool call before running the hook dispatcher.

Default policy (conservative — can be overridden per-session by the caller):

    Read / Glob / Grep / CodeLookup / CodeSearch : ALLOW
    Bash                                         : ASK
    Write / Edit                                 : ASK
    chat (pipeline pseudo-tool)                  : evaluated via content,
                                                   denied on secret match

History: γ-2c (this PR) merged the standalone ``lib/hal/gates.py`` module
into this package ``__init__.py`` because Phase B (PR #251) introduced
``lib/hal/gates/`` as a directory which shadowed ``gates.py`` — Python
resolves the package before the module, breaking
``from lib.hal.gates import default_engine``. Fix: promote the content
to ``__init__.py`` so both surfaces coexist.

Usage::

    from lib.hal.gates import default_engine  # guard engine factory
    from lib.hal.gates.yolo_mode import activate_yolo  # autonomous-loop marker
"""
from __future__ import annotations

import os
import shlex
from typing import Optional

from lib.hal.guard import GuardDecision, GuardEngine, GuardRule, GuardVerdict
from lib.hal.secrets_guard import has_secrets

__all__ = [
    "default_engine",
    "evaluate_chat",
    "evaluate_tool",
]


_DANGEROUS_ARGV0 = frozenset({
    "rm", "dd", "mkfs", "mkfs.ext2", "mkfs.ext3", "mkfs.ext4", "mkfs.xfs",
    "shred", "fdisk", "wipefs", "zpool",
})

_SHELL_WRAPPERS = frozenset({"sh", "bash", "zsh", "ash", "dash"})


def _classify_bash(command: str, _depth: int = 0) -> str:
    """Return ``safe`` | ``dangerous`` | ``unknown``.

    Tokenises via shlex, recursively unwraps ``sh -c <inner>`` /
    ``bash -c <inner>`` (up to a shallow depth to keep the classifier
    linear on adversarial input). Unknown argv[0] returns ``unknown``
    so caller policy decides ASK vs DENY.
    """
    if _depth > 4:
        return "unknown"
    try:
        tokens = shlex.split(command)
    except ValueError:
        return "unknown"
    if not tokens:
        return "safe"
    argv0 = os.path.basename(tokens[0])
    if argv0 in _SHELL_WRAPPERS and len(tokens) >= 3 and tokens[1] == "-c":
        return _classify_bash(tokens[2], _depth=_depth + 1)
    if argv0 in _DANGEROUS_ARGV0:
        return "dangerous"
    return "unknown"


def _read_rules() -> list:
    return [
        GuardRule(tool=t, decision=GuardDecision.ALLOW, source="hal-core",
                  reason=f"{t} is read-only")
        for t in ("Read", "Glob", "Grep", "CodeLookup", "CodeSearch")
    ]


def _write_rules() -> list:
    return [
        GuardRule(tool=t, decision=GuardDecision.ASK, source="hal-core",
                  reason=f"{t} mutates the filesystem")
        for t in ("Write", "Edit")
    ]


def _bash_rules() -> list:
    return [
        GuardRule(tool="Bash", decision=GuardDecision.ASK, source="hal-core",
                  reason="Shell commands require approval"),
        GuardRule(tool="Bash", decision=GuardDecision.DENY, source="hal-core",
                  reason="Refusing `rm -rf /` — destructive root wipe",
                  pattern="rm -rf /"),
        GuardRule(tool="Bash", decision=GuardDecision.DENY, source="hal-core",
                  reason="Refusing `rm -rf ~` — destructive home wipe",
                  pattern="rm -rf ~"),
        GuardRule(tool="Bash", decision=GuardDecision.DENY, source="hal-core",
                  reason="Refusing `mkfs` — filesystem format",
                  pattern="mkfs"),
        GuardRule(tool="Bash", decision=GuardDecision.DENY, source="hal-core",
                  reason="Refusing `dd if=` to raw devices",
                  pattern="dd if="),
        GuardRule(tool="Bash", decision=GuardDecision.DENY, source="hal-core",
                  reason="Refusing force-push to main/master",
                  pattern="push --force origin main"),
        GuardRule(tool="Bash", decision=GuardDecision.DENY, source="hal-core",
                  reason="Refusing force-push to main/master",
                  pattern="push -f origin main"),
    ]


def _chat_rules() -> list:
    return [
        GuardRule(tool="chat", decision=GuardDecision.ALLOW, source="hal-core",
                  reason="Chat allowed; content scanned by pipeline"),
    ]


def _core_rules() -> list:
    return [
        *_read_rules(),
        *_write_rules(),
        *_bash_rules(),
        *_chat_rules(),
    ]


def default_engine(extra_rules: Optional[list] = None) -> GuardEngine:
    """Return a fresh :class:`GuardEngine` populated with HAL default rules.

    Callers may append session-specific rules via ``extra_rules``; they
    are appended after the defaults so they win strictest-wins ties.
    """
    rules = _core_rules()
    if extra_rules:
        rules.extend(extra_rules)
    return GuardEngine(rules=rules)


def evaluate_chat(engine: GuardEngine, message: str, model: str) -> GuardVerdict:
    """Evaluate a chat-turn content payload.

    Combines the rule-based verdict with a content scan: if either the
    rule engine DENIES or :func:`lib.hal.secrets_guard.has_secrets` finds
    a credential, the verdict is DENY.
    """
    verdict = engine.evaluate("chat", {"message": message, "model": model})
    if verdict.denied:
        return verdict
    if has_secrets(message):
        return GuardVerdict(
            decision=GuardDecision.DENY,
            reason="Message contains a credential-shaped token — refusing to transmit",
            rule_source="hal-core:secrets_guard",
        )
    return verdict


def evaluate_tool(
    engine: GuardEngine, tool_name: str, tool_input: dict,
) -> GuardVerdict:
    """Evaluate a tool invocation. Thin wrapper for decorators/stubs."""
    return engine.evaluate(tool_name, tool_input)
