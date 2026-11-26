"""HAL Native Tool Dispatch — execute tools without any external harness.

Absorbed from TF++ src/tools/. HAL runs Bash, Read, Write, Edit, Glob, Grep
natively with sandbox enforcement via the hook dispatch engine.

Usage:
    from lib.hal.tool_dispatch import ToolDispatcher

    dispatcher = ToolDispatcher()
    result = dispatcher.execute("Read", {"file_path": "/tmp/test.py"})
    result = dispatcher.execute("Bash", {"command": "ls -la"})
"""
import glob as glob_mod
import importlib.util
import os
import re
import subprocess
import sys
from typing import Optional

# Optional AST-based code symbol indexer. HAL does not ship one by default —
# external harnesses (e.g. GRIP) may provide a code_index.py with a CodeIndex
# class exposing lookup(symbol, project) and search_fts(query, project) plus
# format_lookup(results). Point HAL_CODE_INDEX_PATH at that file to enable
# the CodeLookup/CodeSearch tools. Missing file → tools return a friendly error.
_CODE_INDEX_PATH = os.environ.get(
    "HAL_CODE_INDEX_PATH",
    os.path.join(os.path.expanduser(
        os.environ.get("HAL_HOME", "~/.hal")), "lib", "code_index.py"),
)

from lib.hal.gates import default_engine as _default_guard_engine
from lib.hal.guard import GuardEngine
from lib.hal.hook_dispatch import HookDispatcher
from lib.hal.secrets_guard import has_secrets, redact_text

_TOOL_REGISTRY: dict[str, "callable"] = {}


def _input_contains_secret(tool_input: dict) -> bool:
    """Return True if any string leaf in ``tool_input`` matches a secret pattern.

    Only string values are scanned — nested dicts/lists are walked but the
    scan itself is shallow-per-leaf. Short-circuits at the first hit.
    """
    stack: list = [tool_input]
    while stack:
        node = stack.pop()
        if isinstance(node, str):
            if has_secrets(node):
                return True
        elif isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend(node)
    return False


class ToolResult:
    """Result of a tool execution."""

    def __init__(self, output: str = "", error: str = "", blocked: bool = False, reason: str = ""):
        self.output = output
        self.error = error
        self.blocked = blocked
        self.reason = reason


class ToolDispatcher:
    """Dispatches tool calls with guard + hook enforcement.

    Three enforcement layers run per call in this order:

    1. :mod:`lib.hal.secrets_guard` — reject inputs containing credential-
       shaped tokens outright.
    2. :class:`~lib.hal.guard.GuardEngine` via :mod:`lib.hal.gates` — the
       rule engine that decides ALLOW / DENY / ASK per tool, with HAL's
       default destructive-command rules and any extras the caller passes.
    3. :class:`~lib.hal.hook_dispatch.HookDispatcher` — optional file-based
       hooks from ``$HAL_HOME/hooks/`` for users who want to extend the
       policy without touching code. Empty by default.

    Outputs are scrubbed through :func:`~lib.hal.secrets_guard.redact_text`
    before being forwarded to PostToolUse hooks.
    """

    def __init__(
        self,
        hooks: Optional[HookDispatcher] = None,
        *,
        guard: Optional[GuardEngine] = None,
    ):
        self._hooks = hooks or HookDispatcher()
        self._guard = guard or _default_guard_engine()

    def execute(self, tool_name: str, tool_input: dict) -> ToolResult:
        """Execute a tool through all three enforcement layers."""
        if _input_contains_secret(tool_input):
            return ToolResult(
                blocked=True,
                reason="Tool input blocked: contains a credential-shaped token",
            )

        if tool_name == "Bash":
            from lib.hal.gates import _classify_bash
            classification = _classify_bash(tool_input.get("command", ""))
            if classification == "dangerous":
                return ToolResult(
                    blocked=True,
                    reason="Bash classified as dangerous (tokenised pattern match)",
                )

        verdict = self._guard.evaluate(tool_name, tool_input)
        if verdict.denied:
            return ToolResult(
                blocked=True,
                reason=f"Guard denied {tool_name}: {verdict.reason}",
            )

        pre = self._hooks.dispatch("PreToolUse", {
            "tool_name": tool_name, "tool_input": tool_input,
        })
        if pre.blocked:
            return ToolResult(blocked=True, reason=pre.reason)

        handler = _TOOL_REGISTRY.get(tool_name)
        if not handler:
            return ToolResult(error=f"Unknown tool: {tool_name}")

        result = handler(tool_input)
        safe_output = redact_text(result.output) if result.output else ""

        self._hooks.dispatch("PostToolUse", {
            "tool_name": tool_name, "tool_input": tool_input,
            "tool_output": safe_output[:1000],
        })

        if safe_output != result.output:
            return ToolResult(
                output=safe_output, error=result.error,
                blocked=result.blocked, reason=result.reason,
            )
        return result

    def available_tools(self) -> list[str]:
        """List registered tool names."""
        return sorted(_TOOL_REGISTRY.keys())


# --- Tool Implementations (absorbed from TF++ src/tools/) ---

def _tool_bash(args: dict) -> ToolResult:
    """Execute a shell command. Absorbed from TF++ BashTool."""
    command = args.get("command", "")
    timeout = min(args.get("timeout", 120000), 600000) / 1000

    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=args.get("cwd"),
        )
        output = proc.stdout
        if proc.stderr:
            output += f"\n{proc.stderr}" if output else proc.stderr
        if proc.returncode != 0:
            return ToolResult(output=output, error=f"Exit code {proc.returncode}")
        return ToolResult(output=output)
    except subprocess.TimeoutExpired:
        return ToolResult(error=f"Timeout after {timeout}s")
    except OSError as e:
        return ToolResult(error=str(e))


def _tool_read(args: dict) -> ToolResult:
    """Read a file. Absorbed from TF++ FileReadTool."""
    path = args.get("file_path", "")
    offset = args.get("offset", 0)
    limit = args.get("limit", 2000)

    if not os.path.isfile(path):
        return ToolResult(error=f"File not found: {path}")

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        selected = lines[offset:offset + limit]
        numbered = "".join(
            f"{i + offset + 1}\t{line}" for i, line in enumerate(selected)
        )
        return ToolResult(output=numbered)
    except OSError as e:
        return ToolResult(error=str(e))


def _tool_write(args: dict) -> ToolResult:
    """Write a file. Absorbed from TF++ FileWriteTool."""
    path = args.get("file_path", "")
    content = args.get("content", "")

    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return ToolResult(output=f"Written to {path}")
    except OSError as e:
        return ToolResult(error=str(e))


def _tool_edit(args: dict) -> ToolResult:
    """Edit a file via string replacement. Absorbed from TF++ FileEditTool."""
    path = args.get("file_path", "")
    old = args.get("old_string", "")
    new = args.get("new_string", "")
    replace_all = args.get("replace_all", False)

    if not os.path.isfile(path):
        return ToolResult(error=f"File not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        if old not in content:
            return ToolResult(error="old_string not found in file")

        if replace_all:
            updated = content.replace(old, new)
        else:
            updated = content.replace(old, new, 1)

        with open(path, "w", encoding="utf-8") as f:
            f.write(updated)
        return ToolResult(output=f"Updated {path}")
    except OSError as e:
        return ToolResult(error=str(e))


def _tool_glob(args: dict) -> ToolResult:
    """Find files by pattern. Absorbed from TF++ GlobTool."""
    pattern = args.get("pattern", "")
    path = args.get("path", ".")

    full_pattern = os.path.join(path, pattern)
    matches = sorted(glob_mod.glob(full_pattern, recursive=True))
    return ToolResult(output="\n".join(matches))


def _tool_grep(args: dict) -> ToolResult:
    """Search file contents. Absorbed from TF++ GrepTool."""
    pattern = args.get("pattern", "")
    path = args.get("path", ".")
    file_type = args.get("type", "")

    cmd = ["rg", "--no-heading", pattern, path]
    if file_type:
        cmd.extend(["--type", file_type])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return ToolResult(output=proc.stdout[:10000])
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # Fallback to Python regex if rg not available
        return _grep_fallback(pattern, path)


def _grep_fallback(pattern: str, path: str) -> ToolResult:
    """Pure Python grep fallback when rg is not installed."""
    # Collect files first (O(n)), then search each (O(m)) — total O(n+m)
    files = _collect_files(path, max_files=5000)
    regex = re.compile(pattern)
    matches = []
    for fpath in files:
        _grep_single_file(fpath, regex, matches)
        if len(matches) >= 250:
            break
    return ToolResult(output="\n".join(matches))


def _collect_files(path: str, max_files: int) -> list[str]:
    """Collect file paths up to a maximum. O(n) single pass."""
    result = []
    for root, _, names in os.walk(path):
        result.extend(os.path.join(root, n) for n in names)
        if len(result) >= max_files:
            return result[:max_files]
    return result


def _grep_single_file(fpath: str, regex: "re.Pattern", matches: list) -> None:
    """Search one file for regex matches. Appends to matches list."""
    try:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f, 1):
                if regex.search(line):
                    matches.append(f"{fpath}:{i}:{line.rstrip()}")
                    if len(matches) >= 250:
                        return
    except OSError:
        pass


def _load_code_index():
    """Load the GRIP code_index module on demand. Returns module or None.

    Registers the module in sys.modules before exec so that @dataclass
    decorators can resolve their module namespace (required by CPython).
    """
    if not os.path.isfile(_CODE_INDEX_PATH):
        return None
    if "code_index" in sys.modules:
        return sys.modules["code_index"]
    spec = importlib.util.spec_from_file_location("code_index", _CODE_INDEX_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["code_index"] = mod  # Must precede exec_module for @dataclass resolution
    try:
        spec.loader.exec_module(mod)
    except Exception:
        del sys.modules["code_index"]
        raise
    return mod


def _run_code_index_op(op: str, term: str, project: str) -> ToolResult:
    """Shared implementation for code index operations.

    op: 'lookup' or 'search_fts'
    term: symbol name or FTS query string
    project: optional project alias filter
    """
    mod = _load_code_index()
    if mod is None:
        return ToolResult(error=f"code_index not found at {_CODE_INDEX_PATH}")
    try:
        idx = mod.CodeIndex()
        try:
            results = getattr(idx, op)(term, project)
        finally:
            idx.close()
        empty_msg = "No symbols found." if op == "lookup" else "No results found."
        return ToolResult(output=mod.format_lookup(results) if results else empty_msg)
    except Exception as e:
        return ToolResult(error=str(e))


def _tool_code_lookup(args: dict) -> ToolResult:
    """Look up a symbol by name in the GRIP AST code index.

    args:
        symbol (str): Symbol name or pattern (e.g. "ProviderRegistry")
        project (str, optional): Filter by project alias
    """
    symbol = args.get("symbol", "")
    if not symbol:
        return ToolResult(error="'symbol' argument required")
    return _run_code_index_op("lookup", symbol, args.get("project", ""))


def _tool_code_search(args: dict) -> ToolResult:
    """Full-text search the GRIP AST code index.

    args:
        query (str): FTS search query (e.g. "webhook delivery")
        project (str, optional): Filter by project alias
    """
    query = args.get("query", "")
    if not query:
        return ToolResult(error="'query' argument required")

    return _run_code_index_op("search_fts", query, args.get("project", ""))


_OSA_AUTOMINT_ENV = "HAL_OSA_TOOL_CHAIN_AUTOMINT"


def _osa_resolve_script_or_error(name: str):
    """Look up ``name`` in the osa registry. Return ``(script, None)`` on hit,
    ``(None, ToolResult)`` on miss. Factored out to keep ``_tool_osa_run`` short."""
    from lib.hal.osa import ScriptNotRegistered, get_script
    try:
        return get_script(name), None
    except ScriptNotRegistered as exc:
        return None, ToolResult(error=str(exc))


def _osa_mint_tier2_token_or_error(name: str, tier: int):
    """Resolve the capability token for a tier-2+ dispatch.

    Returns ``(token, None)`` when the operator has consented and a
    secret is available, otherwise ``(None, ToolResult)`` carrying an
    instructional error. The caller reads the token back and passes it
    to :func:`lib.hal.osa.osa` or short-circuits on the error.
    """
    from lib.hal.osa import MissingSecret, default_secret, mint_token
    if os.environ.get(_OSA_AUTOMINT_ENV) != "1":
        return None, ToolResult(
            error=(
                f"tier-{tier} script {name!r} requires operator consent: "
                f"set {_OSA_AUTOMINT_ENV}=1 to auto-mint tokens for "
                f"tool-chain dispatches, or pre-mint via `osa mint`."
            ),
        )
    try:
        return mint_token(name, secret=default_secret()), None
    except MissingSecret as exc:
        return None, ToolResult(error=f"auth setup incomplete — {exc}")


def _osa_dispatch_and_translate(name: str, params: dict, timeout_s, token):
    """Call :func:`lib.hal.osa.osa` and translate every exception into a
    :class:`ToolResult`. Keeps the main handler free of nested try/except
    chains and makes every error path greppable by name."""
    from lib.hal.osa import (
        MissingSecret, OsaError, ParamValidationError, ProvenanceType,
        ScriptNotRegistered, TokenExpired, TokenInvalid, TokenReused, osa,
    )
    try:
        result = osa(
            name, params, timeout_s=timeout_s,
            provenance=ProvenanceType.TOOL_CHAIN,
            capability_token=token,
        )
    except (ScriptNotRegistered, ParamValidationError) as exc:
        return ToolResult(error=str(exc))
    except (MissingSecret, TokenInvalid, TokenExpired, TokenReused) as exc:
        return ToolResult(error=f"auth rejected — {exc}")
    except OsaError as exc:
        return ToolResult(error=f"osa kernel failure — {exc}")
    return ToolResult(
        output=result.stdout,
        error="" if result.ok else (result.stderr or "non-zero exit"),
    )


def _tool_osa_run(args: dict) -> ToolResult:
    """Dispatch a registered osa script on behalf of an LLM tool call.

    Tier-1 scripts run without authorisation. Tier-2+ scripts require
    operator consent via ``HAL_OSA_TOOL_CHAIN_AUTOMINT=1`` so the handler
    can mint a single-use capability token. Every dispatch is audit-logged
    with ``provenance=tool_chain``.

    args:
        name (str): registry script name (see ``osa list``)
        params (dict[str, str], optional): per-script parameters
        timeout_s (int, optional): override the registry default
    """
    name = args.get("name", "")
    if not name:
        return ToolResult(error="'name' argument required (see `osa list`)")
    script, err = _osa_resolve_script_or_error(name)
    if err is not None:
        return err
    params = args.get("params") or {}
    if not isinstance(params, dict):
        return ToolResult(error="'params' must be an object")
    token = None
    if script.tier >= 2:
        token, err = _osa_mint_tier2_token_or_error(name, script.tier)
        if err is not None:
            return err
    return _osa_dispatch_and_translate(name, params, args.get("timeout_s"), token)


# Register all tools
_TOOL_REGISTRY.update({
    "Bash": _tool_bash,
    "Read": _tool_read,
    "Write": _tool_write,
    "Edit": _tool_edit,
    "Glob": _tool_glob,
    "Grep": _tool_grep,
    "CodeLookup": _tool_code_lookup,
    "CodeSearch": _tool_code_search,
    "osa_run": _tool_osa_run,
})
