"""HAL Hook Dispatch Engine — in-process Python hook runner.

Runs GRIP's enforcement hooks without CC's subprocess invocation.
HARD hooks run synchronously (deny blocks action) and FAIL CLOSED: a crash,
a timeout or unparseable output from a HARD hook is a deny, never an allow.
SOFT hooks fail-open.

A hook may answer in any of the three shapes Claude Code accepts, and all
three are read (HAL#621 — only the first used to be):
1. top-level ``{"decision": "deny"|"block"|"allow", "reason": ...}``
2. ``{"hookSpecificOutput": {"permissionDecision": "deny"|"ask"|"allow",
   "permissionDecisionReason": ...}}`` — the shape GRIP's gates emit
3. exit code 2 (stdout is the reason)
``permissionDecision: "ask"`` is a deny here: a headless dispatcher has nobody
to ask, so a hook that wants a human cannot be auto-approved.

Two entry modes:
1. In-process: hooks expose run(event: dict) -> dict (HAL-native, zero-latency)
2. Subprocess: falls back to stdin/stdout JSON protocol (CC-compatible)

Usage:
    from lib.hal.hook_dispatch import HookDispatcher

    dispatcher = HookDispatcher()
    result = dispatcher.dispatch("PreToolUse", {"tool_name": "Bash", "tool_input": {...}})
    if result.blocked:
        print(f"Blocked: {result.reason}")
"""
import ast
import importlib.util
import json
import os
import subprocess
import sys
from typing import Optional

from lib.hal.home import HAL_HOME as _HAL_HOME
_HOOKS_DIR = os.path.join(_HAL_HOME, "hooks")
_CLASSIFICATION_PATH = os.path.join(_HAL_HOME, "config", "hook-classification.json")

# Hook event type constants — extended for CC 2.1.76
PRE_TOOL_USE = "PreToolUse"
POST_TOOL_USE = "PostToolUse"
PRE_COMPACT = "PreCompact"
POST_COMPACT = "PostCompact"          # CC 2.1.76: fires after context compaction
ELICITATION = "Elicitation"           # CC 2.1.76: intercept before sending response
ELICITATION_RESULT = "ElicitationResult"  # CC 2.1.76: override response result


class HookResult:
    """Result of dispatching hooks for an event."""

    def __init__(
        self,
        blocked: bool = False,
        reason: str = "",
        hook_name: str = "",
        results: Optional[list[dict]] = None,
    ):
        self.blocked = blocked
        self.reason = reason
        self.hook_name = hook_name
        self.results = results or []


class HookDispatcher:
    """Dispatches hooks by event type with HARD/SOFT classification."""

    def __init__(self, hooks_dir: Optional[str] = None):
        self._hooks_dir = hooks_dir or _HOOKS_DIR
        self._classification = _load_classification()
        self._hard_hooks = self._classification.get("hard", {}).get("hooks", {})
        self._soft_hooks = self._classification.get("soft", {}).get("hooks", {})

    def dispatch(self, event: str, data: dict) -> HookResult:
        """Run all hooks matching an event. HARD hooks block on deny."""
        results = []

        # HARD hooks first — any deny blocks the action
        for hook_name, config in self._hard_hooks.items():
            if not _matches_event(config, event):
                continue
            result = _normalise(self._run_hook(hook_name, event, data, hard=True), hard=True)
            results.append(result)
            if _denies(result):
                return HookResult(
                    blocked=True,
                    reason=_reason(result, hook_name),
                    hook_name=hook_name,
                    results=results,
                )

        # SOFT hooks — run all, never block, log failures
        for hook_name, config in self._soft_hooks.items():
            if not _matches_event(config, event):
                continue
            result = _normalise(self._run_hook(hook_name, event, data, hard=False))
            results.append(result)

        return HookResult(blocked=False, results=results)

    def dispatch_with_environment(self, event: str, data: dict, environment=None) -> HookResult:
        """Dispatch hooks with environment constraints checked first.

        When an environment is provided, tool and provider constraints are
        evaluated before standard hooks. Air-gap mode blocks cloud providers.
        """
        if event == PRE_TOOL_USE and environment:
            from lib.hal.environment import check_tool_allowed, check_provider_allowed
            tool_name = data.get("tool_name", "")
            if tool_name:
                allowed, reason = check_tool_allowed(environment, tool_name)
                if not allowed:
                    return HookResult(blocked=True, reason=reason, hook_name="environment_constraint")
            provider = data.get("provider", "")
            if provider:
                allowed, reason = check_provider_allowed(environment, provider)
                if not allowed:
                    return HookResult(blocked=True, reason=reason, hook_name="environment_constraint")
        return self.dispatch(event, data)

    def list_hooks(self, event: Optional[str] = None) -> dict[str, list[str]]:
        """List hooks grouped by classification, optionally filtered by event."""
        hard = [
            name for name, cfg in self._hard_hooks.items()
            if event is None or _matches_event(cfg, event)
        ]
        soft = [
            name for name, cfg in self._soft_hooks.items()
            if event is None or _matches_event(cfg, event)
        ]
        return {"hard": hard, "soft": soft}

    def _run_hook(self, hook_name: str, event: str, data: dict, hard: bool = False) -> dict:
        """Run a single hook, preferring in-process over subprocess.

        ``hard`` selects the failure policy: a HARD hook that cannot answer
        (missing, crashed, timed out, unparseable) is a deny.
        """
        hook_path = os.path.join(self._hooks_dir, hook_name)

        if not os.path.exists(hook_path):
            return {"decision": _failure_decision(hard), "error": f"{hook_name} not found"}

        try:
            # In-process when the hook defines run(); otherwise the subprocess
            # protocol (CC-compatible). Nothing raised here may escape: an
            # unanswered HARD hook is a deny, never a crash the caller reads as allow.
            result = _try_in_process(hook_path, event, data, hard=hard)
            if result is not None:
                return result
            return _run_subprocess(hook_path, event, data, hard=hard)
        except Exception as e:  # defence in depth; both callees already catch
            return {"decision": _failure_decision(hard), "error": f"{hook_name}: {type(e).__name__}: {str(e)[:160]}"}


#: Verdicts that stop the action. ``ask`` is included: a headless dispatcher
#: has no human to prompt, so a hook that wants one cannot be auto-approved.
_DENYING = frozenset({"deny", "block", "ask"})

#: Subprocess budget per hook, in seconds. Module-level so a test can shorten it.
SUBPROCESS_TIMEOUT_S = 30


def _failure_decision(hard: bool) -> str:
    """What a hook that could not answer counts as: HARD fails closed."""
    return "deny" if hard else "allow"


def _normalise(result: object, hard: bool = False) -> dict:
    """Map every accepted response shape onto a top-level ``decision``.

    The ``hookSpecificOutput.permissionDecision`` shape (what GRIP's gates
    print) wins over a top-level ``decision`` when both are present, and its
    own ``permissionDecisionReason`` travels with it, so a mixed payload can
    never deny with the other shape's reason. Claude Code's ``continue:
    false`` is a stop as well. A result that is not an object answered
    nothing and gets the failure policy. The original fields are kept beside
    the derived ones so nothing a hook said is lost.
    """
    if not isinstance(result, dict):
        return {"decision": _failure_decision(hard), "error": f"non-object hook result: {type(result).__name__}"}
    out = dict(result)
    specific = result.get("hookSpecificOutput")
    if isinstance(specific, dict) and specific.get("permissionDecision"):
        out["decision"] = str(specific["permissionDecision"]).lower()
        reason = specific.get("permissionDecisionReason") or result.get("reason")
        if reason:
            out["reason"] = str(reason)
    elif result.get("continue") is False:
        out["decision"] = "deny"
        if result.get("stopReason") and not out.get("reason"):
            out["reason"] = str(result["stopReason"])
    elif "decision" in result:
        out["decision"] = str(result["decision"]).lower()
    return out


def _denies(result: dict) -> bool:
    return result.get("decision") in _DENYING


def _reason(result: dict, hook_name: str) -> str:
    return str(result.get("reason") or result.get("error") or f"Blocked by {hook_name}")


def _matches_event(config: dict, event: str) -> bool:
    """Check if a hook config matches the given event."""
    hook_event = config.get("event", "")
    if isinstance(hook_event, list):
        return event in hook_event
    return hook_event == event


def _defines_run(hook_path: str) -> bool:
    """True when the hook's source defines a module-level ``run`` function.

    Decided from the AST, never from a substring, so a ``# def run(`` in a
    comment or string cannot make a script-style hook look module-style.
    Unreadable or unparseable source reads as "no run()" and goes to the
    subprocess path, where its real exit code and output are judged.
    """
    try:
        with open(hook_path, "r", encoding="utf-8", errors="replace") as fh:
            tree = ast.parse(fh.read())
    except (OSError, SyntaxError, ValueError):
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        for node in tree.body
    )


def _try_in_process(hook_path: str, event: str, data: dict, hard: bool = False) -> Optional[dict]:
    """Try to run hook in-process via its run() function.

    Returns None only when the hook has no module-level ``run()`` -- the
    subprocess path then judges it. A script-style hook (GRIP's gates read
    stdin and print at top level) is therefore never executed here, where it
    would run once against the wrong stdin and again as the subprocess.

    A module-style hook that fails to load or whose ``run()`` raises has
    answered nothing, so it gets the failure policy: a deny for a HARD hook,
    an allow for a SOFT one. It is never retried as a subprocess, because a
    module-style file run as a script exits 0 with no output and would read
    as an allow.
    """
    if not _defines_run(hook_path):
        return None
    try:
        spec = importlib.util.spec_from_file_location("_hal_hook", hook_path)
        if spec is None or spec.loader is None:
            return {"decision": _failure_decision(hard), "error": f"{hook_path}: cannot load"}

        module = importlib.util.module_from_spec(spec)

        # Add necessary paths before loading
        if _HOOKS_DIR not in sys.path:
            sys.path.insert(0, _HOOKS_DIR)
        lib_dir = os.path.join(_HAL_HOME, "lib")
        if lib_dir not in sys.path:
            sys.path.insert(0, lib_dir)

        spec.loader.exec_module(module)
        result = module.run({"event": event, **data})
        if isinstance(result, dict):
            return result
        return {"decision": _failure_decision(hard), "error": f"{hook_path}: run() returned non-object"}
    except (Exception, SystemExit) as e:
        return {"decision": _failure_decision(hard), "error": f"{hook_path}: in-process {type(e).__name__}: {str(e)[:160]}"}


def _run_subprocess(hook_path: str, event: str, data: dict, hard: bool = False) -> dict:
    """Run hook as subprocess with JSON stdin/stdout protocol.

    Exit 2 is a block; JSON on stdout is returned for :func:`_normalise`;
    a HARD hook that crashes, times out or prints something unparseable is
    a deny, a SOFT one an allow.
    """
    payload = json.dumps({"event": event, **data})

    try:
        result = subprocess.run(
            [sys.executable, hook_path],
            input=payload,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return {"decision": _failure_decision(hard), "error": f"{hook_path}: timeout ({SUBPROCESS_TIMEOUT_S}s)"}
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        # ValueError covers UnicodeDecodeError from text=True on non-UTF-8 output.
        return {"decision": _failure_decision(hard), "error": str(e)[:200]}

    stdout = result.stdout.strip()
    if result.returncode == 2:
        return {"decision": "block", "reason": stdout[:500]}

    # The exit code is read BEFORE stdout: a hook that printed an answer and
    # then crashed (exit 1) did not give a trustworthy answer.
    if result.returncode != 0:
        return {"decision": _failure_decision(hard), "error": f"{hook_path}: exit {result.returncode}"}

    if not stdout:
        return {"decision": "allow"}
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        return {"decision": _failure_decision(hard), "error": f"{hook_path}: unparseable output"}
    if not isinstance(parsed, dict):
        return {"decision": _failure_decision(hard), "error": f"{hook_path}: non-object output"}
    return parsed


def _load_classification() -> dict:
    """Load hook classification from config."""
    if not os.path.exists(_CLASSIFICATION_PATH):
        return {"hard": {"hooks": {}}, "soft": {"hooks": {}}}
    try:
        with open(_CLASSIFICATION_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"hard": {"hooks": {}}, "soft": {"hooks": {}}}
