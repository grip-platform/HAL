"""HAL welcome — spoken context-aware greeting after reboot.

Entry point for the operator-facing ``hal welcome`` command. Assembles
the operator's recent session state (git HEAD, last few merged PRs,
session-context notes, active-identity signature), hands the result
to a HAPPI chat envelope, and streams the LLM response through the
β-1 streaming-TTS primitive so HAL speaks the greeting in real time.

Usage:
    hal welcome                           # default Anthropic, Jamie Premium
    hal welcome --provider gemini         # Gemini via HAPPI
    hal welcome --voice Samantha          # different macOS voice
    hal welcome --rate 200                # faster WPM
    hal welcome --text "Welcome back."    # bypass LLM, speak fixed text
    hal welcome --dry-run                 # print envelope + context, no TTS

The command is intentionally zero-config: no secrets required beyond
what HAPPI's runtime already resolves (env var / Keychain). If no
provider keys are available, --text is the always-works fallback.

Stdlib-only. macOS-only for default TTS backend (``say``). Tests inject
a mock spawn factory so the suite runs on any platform.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Callable

from lib.hal.happi.envelope import HappiEnvelope
from lib.hal.voice.stream_speak import (
    DEFAULT_RATE_WPM,
    DEFAULT_VOICE,
    StreamSpeakResult,
    stream_say,
)


HAL_HOME = Path.home() / ".hal"
SESSION_CONTEXT_PATH = HAL_HOME / "state" / "session-context.md"
ACTIVE_IDENTITY_PATH = Path.home() / ".claude" / "state" / "session" / "active-identity"
DEFAULT_PROVIDER = ""  # empty = HAPPI runtime's default (anthropic)
DEFAULT_MAX_TOKENS = 400
RECENT_COMMIT_COUNT = 6

# launchd auto-install constants.
LAUNCH_AGENT_LABEL = "com.codetonight.hal.welcome"
LAUNCH_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"
LAUNCH_AGENT_PLIST = LAUNCH_AGENTS_DIR / f"{LAUNCH_AGENT_LABEL}.plist"
LAUNCH_AGENT_LOG = Path.home() / "Library" / "Logs" / "hal-welcome.log"
LAUNCH_AGENT_ERR = Path.home() / "Library" / "Logs" / "hal-welcome.err"


SYSTEM_PROMPT_TEMPLATE = """\
You are HAL — the voice of the Harness Abstraction Layer. Your operator\
 has just logged in after a reboot.

Greet them with warmth and precision. Summarise the state of things in\
 one or two short sentences. End with exactly one suggested next action.\
 You have roughly 15 seconds of speech. Keep it under 40 words. No\
 filler like "I hope you're well" or "I'm here to help". No preamble.\
 Start with "Welcome back".

Facts for this greeting:
- Operator signature: {signature}
- Current branch: {branch}
- Last commit: {head_oid_short} — {head_subject}
- Recent merged PRs:
{recent_prs}
- Session notes (most recent):
{session_notes}

Speak directly. End with the single next action.\
"""


@dataclass(frozen=True)
class WelcomeContext:
    """Structured operator + repo state used to build the greeting prompt."""
    signature: str
    branch: str
    head_oid_short: str
    head_subject: str
    recent_prs: list[str]
    session_notes: str


def _read_trim(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip() or default
    except FileNotFoundError:
        return default
    except OSError:
        return default


def _git(args: list[str], cwd: Path = HAL_HOME) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True, text=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _recent_merged_prs(limit: int = RECENT_COMMIT_COUNT) -> list[str]:
    """Return the last N squash-merge PR subjects (one line each)."""
    log = _git([
        "log", "--oneline", "--no-decorate", f"-n{limit}", "HEAD",
    ])
    if not log:
        return []
    return [line.strip() for line in log.splitlines() if line.strip()]


def _session_notes_tail(max_lines: int = 12) -> str:
    """Return the last N non-empty lines of session-context.md."""
    raw = _read_trim(SESSION_CONTEXT_PATH)
    if not raw:
        return "(no session notes on file)"
    lines = [line for line in raw.splitlines() if line.strip()]
    tail = lines[-max_lines:]
    return "\n".join(tail)


def gather_context() -> WelcomeContext:
    """Snapshot current operator + repo state for greeting assembly."""
    signature = _read_trim(ACTIVE_IDENTITY_PATH, default="operator")
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"
    head_oid_short = _git(["rev-parse", "--short", "HEAD"]) or "unknown"
    head_subject = _git(["log", "-1", "--pretty=%s"]) or "no commits"
    recent_prs = _recent_merged_prs()
    session_notes = _session_notes_tail()
    return WelcomeContext(
        signature=signature,
        branch=branch,
        head_oid_short=head_oid_short,
        head_subject=head_subject,
        recent_prs=recent_prs,
        session_notes=session_notes,
    )


def render_system_prompt(context: WelcomeContext) -> str:
    recent_prs_block = "\n".join(f"    - {line}" for line in context.recent_prs) \
        or "    - (no recent merges visible)"
    return SYSTEM_PROMPT_TEMPLATE.format(
        signature=context.signature,
        branch=context.branch,
        head_oid_short=context.head_oid_short,
        head_subject=context.head_subject,
        recent_prs=recent_prs_block,
        session_notes=context.session_notes,
    )


def build_welcome_envelope(
    context: WelcomeContext,
    *,
    provider: str = DEFAULT_PROVIDER,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> HappiEnvelope:
    system = render_system_prompt(context)
    return HappiEnvelope.parse({
        "v": "happi/1.0",
        "id": f"welcome-{context.head_oid_short}",
        "cmd": "chat",
        "args": [],
        "flags": {
            "provider": provider,
            "system": system,
            "message": "Greet me now.",
            "max_tokens": max_tokens,
        },
    })


def _fixed_text_stream(text: str, envelope_id: str = "welcome-text"):
    """Yield a minimal NDJSON stream for stream_say when --text is passed.

    Matches the shape stream_say expects: started → delta → completed.
    """
    from lib.hal.happi.events import emit_completed, emit_delta, emit_started
    yield emit_started(envelope_id)
    if text:
        yield emit_delta(envelope_id, text)
    yield emit_completed(envelope_id, usage={})


def welcome(
    *,
    provider: str = DEFAULT_PROVIDER,
    voice: str = DEFAULT_VOICE,
    rate_wpm: int | None = DEFAULT_RATE_WPM,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    fixed_text: str | None = None,
    context_builder: Callable[[], WelcomeContext] = gather_context,
    spawn: Callable = subprocess.Popen,
) -> StreamSpeakResult:
    """Assemble the greeting envelope and stream it through TTS.

    When ``fixed_text`` is provided, the LLM step is bypassed entirely
    and that string is spoken verbatim — useful for smoke-testing the
    audio path or for reboot scenarios where no provider is reachable.
    """
    if fixed_text is not None:
        source = _fixed_text_stream(fixed_text)
    else:
        envelope = build_welcome_envelope(
            context_builder(), provider=provider, max_tokens=max_tokens,
        )
        source = envelope

    return stream_say(source, voice=voice, rate_wpm=rate_wpm, spawn=spawn)


DEFAULT_FALLBACK_TEXT = (
    "Welcome back. All providers are unreachable, so HAL is greeting "
    "you from the fallback text. Type hal welcome dash dash check to "
    "diagnose."
)


def welcome_with_chain(
    *,
    providers: list[str],
    fallback_text: str = DEFAULT_FALLBACK_TEXT,
    voice: str = DEFAULT_VOICE,
    rate_wpm: int | None = DEFAULT_RATE_WPM,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    context_builder: Callable[[], WelcomeContext] = gather_context,
    spawn: Callable = subprocess.Popen,
) -> StreamSpeakResult:
    """Try providers in order; fall back to fixed text if all silent.

    A provider "fails silently" when its result has ``delta_count == 0``
    AND ``error`` is set — i.e., the LLM never produced an audible token.
    Typical cause: ``missing_credentials``, ``provider_http_401/403``, or
    ``provider_network``. In that case move to the next provider without
    the operator hearing anything.

    If a provider starts speaking (``delta_count > 0``) and then errors
    mid-stream we accept the partial result rather than restarting the
    greeting with a different voice — partial audio beats silent
    disorientation.

    If every provider fails silently the function speaks
    ``fallback_text`` so the operator still hears something on reboot.
    """
    context = context_builder()
    last_result: StreamSpeakResult | None = None
    for provider in providers:
        envelope = build_welcome_envelope(
            context, provider=provider, max_tokens=max_tokens,
        )
        result = stream_say(
            envelope, voice=voice, rate_wpm=rate_wpm, spawn=spawn,
        )
        last_result = result
        if result.delta_count > 0:
            return result
        # Silent failure: try the next provider.
    # Every provider failed without producing audio — speak the fallback.
    fallback_stream = _fixed_text_stream(fallback_text, envelope_id="welcome-fallback")
    return stream_say(
        fallback_stream, voice=voice, rate_wpm=rate_wpm, spawn=spawn,
    )


def _parse_provider_chain(raw: str) -> list[str]:
    """Split a comma-separated provider spec into a clean list.

    Empty entries and whitespace-only entries are dropped so that
    operators can write ``"anthropic, gemini, groq/llama-3.1-70b"``
    with spaces around commas. Trailing / leading commas are tolerated.
    """
    return [p.strip() for p in raw.split(",") if p.strip()]


def tail_launch_agent_logs(
    *,
    log_path: Path = LAUNCH_AGENT_LOG,
    err_path: Path = LAUNCH_AGENT_ERR,
    lines: int = 40,
) -> str:
    """Return the last ``lines`` of both stdout + stderr launchd logs.

    If either file is missing the block reports "(no log)" rather than
    raising — the operator might run --log-tail before ever running the
    LaunchAgent, and an exception would be noise.
    """
    def _read_tail(path: Path) -> str:
        if not path.is_file():
            return f"(no log at {path})"
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"(read failed: {exc})"
        tail_lines = content.splitlines()[-lines:]
        return "\n".join(tail_lines) if tail_lines else "(log is empty)"

    stdout_section = _read_tail(log_path)
    stderr_section = _read_tail(err_path)
    return (
        f"--- stdout tail ({log_path}) ---\n"
        f"{stdout_section}\n"
        f"\n"
        f"--- stderr tail ({err_path}) ---\n"
        f"{stderr_section}\n"
    )


LAUNCH_AGENT_PLIST_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>

    <key>ProgramArguments</key>
    <array>
{program_args_xml}
    </array>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <false/>

    <key>StandardOutPath</key>
    <string>{log_path}</string>

    <key>StandardErrorPath</key>
    <string>{err_path}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{path_env}</string>
    </dict>
</dict>
</plist>
"""


@dataclass(frozen=True)
class InstallResult:
    plist_path: Path
    wrote_plist: bool
    loaded: bool
    message: str


def _render_program_args(hal_bin: Path, extra_args: list[str]) -> str:
    """Render the <array> body inside <key>ProgramArguments</key>.

    Each argument becomes one indented <string>...</string> line. XML
    special chars in user-supplied extra_args are escaped defensively
    even though the stdlib's xml.sax.saxutils is overkill here — we only
    need &amp; &lt; &gt; handling for argv strings.
    """
    def _esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    lines = [f"        <string>{_esc(str(hal_bin))}</string>",
             "        <string>welcome</string>"]
    for arg in extra_args:
        lines.append(f"        <string>{_esc(arg)}</string>")
    return "\n".join(lines)


def render_launch_agent_plist(
    hal_bin: Path,
    *,
    extra_args: list[str] | None = None,
    log_path: Path = LAUNCH_AGENT_LOG,
    err_path: Path = LAUNCH_AGENT_ERR,
    path_env: str = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
    label: str = LAUNCH_AGENT_LABEL,
) -> str:
    """Produce the plist XML for an operator-specific launchd agent."""
    return LAUNCH_AGENT_PLIST_TEMPLATE.format(
        label=label,
        program_args_xml=_render_program_args(hal_bin, extra_args or []),
        log_path=str(log_path),
        err_path=str(err_path),
        path_env=path_env,
    )


def _launchctl(action: str, plist_path: Path) -> tuple[int, str]:
    """Run `launchctl <action> <plist>`. Returns (rc, combined_output)."""
    try:
        result = subprocess.run(
            ["launchctl", action, str(plist_path)],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return (1, f"launchctl {action} unavailable: {exc}")
    output = (result.stdout + result.stderr).strip()
    return (result.returncode, output)


def install_launch_agent(
    *,
    hal_bin: Path | None = None,
    extra_args: list[str] | None = None,
    force: bool = False,
    plist_path: Path = LAUNCH_AGENT_PLIST,
    launch_agents_dir: Path = LAUNCH_AGENTS_DIR,
    launchctl_runner: Callable = _launchctl,
) -> InstallResult:
    """Write the LaunchAgent plist and activate it via `launchctl load`.

    Idempotent: if the plist already exists and `force` is False, the
    current file is left untouched and the caller is told to pass
    --force or uninstall first. Returns an InstallResult describing
    what happened so `main()` can surface a human message.
    """
    hal_bin = hal_bin or (HAL_HOME / "bin" / "hal")
    launch_agents_dir.mkdir(parents=True, exist_ok=True)

    if plist_path.exists() and not force:
        return InstallResult(
            plist_path=plist_path,
            wrote_plist=False,
            loaded=False,
            message=(
                f"plist already exists at {plist_path}; "
                f"pass --force to overwrite or run `hal welcome --uninstall` first"
            ),
        )

    plist_xml = render_launch_agent_plist(
        hal_bin, extra_args=extra_args or [],
    )
    plist_path.write_text(plist_xml, encoding="utf-8")

    # Best-effort: if the agent was previously loaded, unload first so
    # `load` sees a clean state. Suppress failures — unload is defensive.
    launchctl_runner("unload", plist_path)

    rc, output = launchctl_runner("load", plist_path)
    if rc != 0:
        return InstallResult(
            plist_path=plist_path,
            wrote_plist=True,
            loaded=False,
            message=(
                f"wrote {plist_path}; launchctl load returned {rc}: {output}. "
                f"Run manually: launchctl load {plist_path}"
            ),
        )
    return InstallResult(
        plist_path=plist_path,
        wrote_plist=True,
        loaded=True,
        message=(
            f"LaunchAgent installed at {plist_path}. Will fire on next login. "
            f"Logs: {LAUNCH_AGENT_LOG}"
        ),
    )


def uninstall_launch_agent(
    *,
    plist_path: Path = LAUNCH_AGENT_PLIST,
    launchctl_runner: Callable = _launchctl,
) -> InstallResult:
    """Remove the LaunchAgent plist and unload it from launchd."""
    if not plist_path.exists():
        return InstallResult(
            plist_path=plist_path,
            wrote_plist=False,
            loaded=False,
            message=f"no LaunchAgent plist found at {plist_path}; nothing to uninstall",
        )
    rc, output = launchctl_runner("unload", plist_path)
    plist_path.unlink()
    return InstallResult(
        plist_path=plist_path,
        wrote_plist=False,
        loaded=(rc == 0),
        message=(
            f"LaunchAgent uninstalled ({plist_path} removed; "
            f"launchctl unload rc={rc})"
        ),
    )


# ========================================================================
# Pre-flight diagnostics: `hal welcome --check`.
# ========================================================================


@dataclass(frozen=True)
class CheckResult:
    """One diagnostic check outcome. `critical` flags whether a failure
    should make the whole --check command exit non-zero."""
    name: str
    ok: bool
    detail: str
    remedy: str
    critical: bool


CHECK_ANTHROPIC_ENV = "CLAUDE_CODE_OAUTH_TOKEN"
CHECK_ANTHROPIC_KEYCHAIN = "grip-claude-oauth"


def _soft(name: str, ok: bool, detail: str, remedy: str = "") -> CheckResult:
    """Helper: build a non-blocking (critical=False) check result."""
    return CheckResult(name=name, ok=ok, detail=detail, remedy=remedy, critical=False)


def _critical(name: str, ok: bool, detail: str, remedy: str) -> CheckResult:
    """Helper: build a blocking (critical=True) check result."""
    return CheckResult(name=name, ok=ok, detail=detail, remedy=remedy, critical=True)


def _which(binary: str) -> str | None:
    """Locate a binary on PATH without shelling out; returns abs path or None."""
    import shutil
    return shutil.which(binary)


def _keychain_has(service: str) -> bool:
    """Probe macOS Keychain for a generic password service. Non-blocking."""
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", service],
            capture_output=True, text=True, timeout=3,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return result.returncode == 0


def _check_hal_bin(hal_bin: Path) -> CheckResult:
    import os
    exists = hal_bin.is_file()
    executable = exists and os.access(hal_bin, os.X_OK)
    return _critical(
        "hal binary",
        executable,
        f"{hal_bin} ({'executable' if executable else 'missing / not executable'})",
        f"chmod +x {hal_bin} — or reinstall HAL",
    )


def _check_git_accessible() -> CheckResult:
    try:
        result = subprocess.run(
            ["git", "--version"], capture_output=True, text=True, timeout=5,
        )
        ok = result.returncode == 0
        detail = result.stdout.strip() or "git runs"
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        ok, detail = False, f"git not found: {exc}"
    return _critical(
        "git accessible", ok, detail,
        "install git (xcode-select --install on macOS)",
    )


def _check_say_binary() -> CheckResult:
    path = _which("say")
    return _critical(
        "say binary (macOS TTS)",
        path is not None,
        path or "not found on PATH",
        "hal welcome --text works without say; real TTS is macOS-only",
    )


def _check_say_voice(voice: str = DEFAULT_VOICE) -> CheckResult:
    name = f"say voice {voice!r}"
    if not _which("say"):
        return _soft(name, False, "say binary missing", "(see previous check)")
    try:
        result = subprocess.run(
            ["say", "-v", "?"], capture_output=True, text=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return _soft(
            name, False, "say -v ? failed",
            "system voices: open System Settings > Accessibility > Spoken Content",
        )
    # `say -v ?` lists voices as "Jamie (Premium)  en_GB  # Hello, ..."
    available = voice in result.stdout
    return _soft(
        name, available,
        f"{voice} {'available' if available else 'not installed'}",
        "download via System Settings > Accessibility > Spoken Content > System Voice > Manage Voices",
    )


def _check_active_identity(path: Path = ACTIVE_IDENTITY_PATH) -> CheckResult:
    exists = path.is_file()
    return _soft(
        "active identity",
        exists,
        f"{path} ({'readable' if exists else 'missing'})",
        f"echo 'V>>' > {path} — greeting falls back to 'operator' otherwise",
    )


def _check_anthropic_token() -> CheckResult:
    import os
    have_env = bool(os.environ.get(CHECK_ANTHROPIC_ENV))
    have_keychain = _keychain_has(CHECK_ANTHROPIC_KEYCHAIN)
    parts = []
    if have_env:
        parts.append(f"env:{CHECK_ANTHROPIC_ENV}")
    if have_keychain:
        parts.append(f"keychain:{CHECK_ANTHROPIC_KEYCHAIN}")
    detail = ", ".join(parts) if parts else "no token source found"
    return _soft(
        "anthropic token",
        have_env or have_keychain,
        detail,
        f"security add-generic-password -s {CHECK_ANTHROPIC_KEYCHAIN} -a V -w <token> "
        "— or fall back to hal welcome --text",
    )


def _check_launch_agent(plist_path: Path = LAUNCH_AGENT_PLIST) -> CheckResult:
    if not plist_path.is_file():
        return _soft(
            "LaunchAgent",
            True,
            f"not installed ({plist_path} absent) — use `hal welcome --install` to auto-run on login",
        )
    try:
        result = subprocess.run(
            ["launchctl", "list"], capture_output=True, text=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return _soft(
            "LaunchAgent", False,
            "launchctl unavailable; can't verify loaded state",
            "unexpected on macOS; reinstall via `hal welcome --install --force`",
        )
    loaded = LAUNCH_AGENT_LABEL in result.stdout
    return _soft(
        "LaunchAgent",
        loaded,
        f"plist exists at {plist_path}; "
        f"{'loaded' if loaded else 'NOT loaded — will not fire on next login'}",
        f"launchctl load {plist_path}  — or `hal welcome --install --force`",
    )


def run_checks(
    *,
    hal_bin: Path | None = None,
    voice: str = DEFAULT_VOICE,
) -> list[CheckResult]:
    """Run all diagnostic checks in a stable order."""
    hal_bin = hal_bin or (HAL_HOME / "bin" / "hal")
    return [
        _check_hal_bin(hal_bin),
        _check_git_accessible(),
        _check_say_binary(),
        _check_say_voice(voice),
        _check_active_identity(),
        _check_anthropic_token(),
        _check_launch_agent(),
    ]


def render_check_report(checks: list[CheckResult]) -> str:
    """Format check results as a plain-text table (no emoji per style)."""
    lines: list[str] = ["--- hal welcome --check ---"]
    width = max(len(c.name) for c in checks)
    for c in checks:
        mark = "[ok]  " if c.ok else ("[FAIL]" if c.critical else "[warn]")
        lines.append(f"{mark} {c.name.ljust(width)}  {c.detail}")
        if not c.ok and c.remedy:
            lines.append(f"       {' ' * width}  remedy: {c.remedy}")
    critical_failures = sum(1 for c in checks if not c.ok and c.critical)
    soft_failures = sum(1 for c in checks if not c.ok and not c.critical)
    lines.append("")
    if critical_failures == 0 and soft_failures == 0:
        lines.append("All checks passed. Tomorrow's welcome should work cleanly.")
    elif critical_failures == 0:
        lines.append(
            f"{soft_failures} soft warning(s); no critical failures. "
            "Welcome will run; consider the remedies above."
        )
    else:
        lines.append(
            f"{critical_failures} CRITICAL failure(s). "
            "Welcome will not speak until those are resolved."
        )
    return "\n".join(lines)


def _add_speak_args(parser: argparse.ArgumentParser) -> None:
    """Flags controlling the speak / LLM dispatch path."""
    parser.add_argument(
        "--provider", default=DEFAULT_PROVIDER,
        help="HAPPI provider spec (e.g. 'gemini'). Empty = runtime default.",
    )
    parser.add_argument(
        "--voice", default=DEFAULT_VOICE,
        help=f"macOS `say` voice name. Default: {DEFAULT_VOICE!r}.",
    )
    parser.add_argument("--rate", type=int, default=None, help="words-per-minute.")
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"Upper bound on greeting length. Default: {DEFAULT_MAX_TOKENS}.",
    )
    parser.add_argument("--text", default=None, help="Speak a fixed string; skip LLM.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print envelope + context to stdout; no TTS.",
    )
    parser.add_argument(
        "--provider-chain", default="",
        help=(
            "Comma-separated provider specs tried in order. If the first "
            "yields no audio (e.g. missing_credentials) the next one fires. "
            "If all fail silently, --fallback-text is spoken. "
            "Example: 'anthropic,gemini,groq/llama-3.1-70b-versatile'."
        ),
    )
    parser.add_argument(
        "--fallback-text", default=DEFAULT_FALLBACK_TEXT,
        help="Text spoken when every provider in --provider-chain fails silently.",
    )


def _add_check_args(parser: argparse.ArgumentParser) -> None:
    """Flags for diagnostic / log-tail paths."""
    parser.add_argument(
        "--check", action="store_true",
        help=(
            "Run pre-flight diagnostics (git, say, identity, tokens, "
            "LaunchAgent loaded state). Exit 0 if all critical checks pass."
        ),
    )
    parser.add_argument(
        "--log-tail", action="store_true",
        help=(
            "Print the last N lines of the LaunchAgent stdout + stderr logs "
            "(default N=40). Useful after a silent morning reboot to see what "
            "HAL actually said or why it errored."
        ),
    )
    parser.add_argument(
        "--log-tail-lines", type=int, default=40,
        help="Line count for --log-tail. Default: 40.",
    )


def _add_install_args(parser: argparse.ArgumentParser) -> None:
    """Flags for the launchd install / uninstall path."""
    parser.add_argument(
        "--install", action="store_true",
        help="Install launchd LaunchAgent to auto-fire on login.",
    )
    parser.add_argument("--uninstall", action="store_true",
                        help="Remove the LaunchAgent + plist.")
    parser.add_argument("--force", action="store_true",
                        help="With --install, overwrite any existing plist.")
    parser.add_argument(
        "--agent-args", default="",
        help="Extra shell-quoted args baked into the LaunchAgent invocation.",
    )


def _parse_argv(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hal welcome",
        description=(
            "Speak a context-aware greeting through streaming TTS, "
            "or install/uninstall a launchd auto-run hook."
        ),
    )
    _add_speak_args(parser)
    _add_check_args(parser)
    _add_install_args(parser)
    return parser.parse_args(argv)


def _dry_run_output(
    args: argparse.Namespace, stream_out: IO[str]
) -> int:
    context = gather_context()
    payload = {
        "context": {
            "signature": context.signature,
            "branch": context.branch,
            "head_oid_short": context.head_oid_short,
            "head_subject": context.head_subject,
            "recent_prs": context.recent_prs,
            "session_notes_preview": context.session_notes[:200],
        },
        "envelope_preview": {
            "provider": args.provider,
            "voice": args.voice,
            "rate_wpm": args.rate,
            "max_tokens": args.max_tokens,
            "text": args.text,
            "system_prompt_len": len(render_system_prompt(context)),
        },
    }
    stream_out.write(json.dumps(payload, indent=2) + "\n")
    stream_out.flush()
    return 0


def _split_agent_args(raw: str) -> list[str]:
    """Split a shell-like string into argv tokens using stdlib shlex.

    Empty input returns []. Unbalanced quotes raise a ValueError that
    propagates so the operator sees what they typed wrong.
    """
    import shlex
    return shlex.split(raw) if raw else []


def _run_check(args: argparse.Namespace, stream_out: IO[str]) -> int:
    checks = run_checks(voice=args.voice)
    stream_out.write(render_check_report(checks) + "\n")
    stream_out.flush()
    critical_failures = sum(1 for c in checks if not c.ok and c.critical)
    return 0 if critical_failures == 0 else 1


def _run_log_tail(args: argparse.Namespace, stream_out: IO[str]) -> int:
    # Explicit paths so monkeypatched module constants take effect in tests;
    # the function-default capture at definition time would otherwise win.
    stream_out.write(tail_launch_agent_logs(
        log_path=LAUNCH_AGENT_LOG,
        err_path=LAUNCH_AGENT_ERR,
        lines=args.log_tail_lines,
    ))
    stream_out.flush()
    return 0


def _run_install(args: argparse.Namespace, stream_out: IO[str]) -> int:
    extra = _split_agent_args(args.agent_args)
    result = install_launch_agent(extra_args=extra, force=args.force)
    stream_out.write(f"[hal welcome] {result.message}\n")
    stream_out.flush()
    return 0 if result.loaded else 1


def _run_uninstall(args: argparse.Namespace, stream_out: IO[str]) -> int:
    result = uninstall_launch_agent()
    stream_out.write(f"[hal welcome] {result.message}\n")
    stream_out.flush()
    return 0


def _run_speak(args: argparse.Namespace, stream_out: IO[str]) -> int:
    # Provider-chain takes precedence over --provider; --text trumps chain.
    chain = _parse_provider_chain(args.provider_chain)
    if chain and args.text is None:
        result = welcome_with_chain(
            providers=chain, fallback_text=args.fallback_text,
            voice=args.voice, rate_wpm=args.rate, max_tokens=args.max_tokens,
        )
    else:
        result = welcome(
            provider=args.provider, voice=args.voice, rate_wpm=args.rate,
            max_tokens=args.max_tokens, fixed_text=args.text,
        )
    # Partial audio is success (user heard something); silent error is failure.
    if result.error and result.delta_count == 0:
        stream_out.write(f"[hal welcome] {result.error}\n")
        stream_out.flush()
        return 1
    return 0


def main(argv: list[str] | None = None, stream_out: IO[str] = sys.stdout) -> int:
    args = _parse_argv(argv if argv is not None else sys.argv[1:])
    if args.install and args.uninstall:
        stream_out.write(
            "[hal welcome] --install and --uninstall are mutually exclusive\n"
        )
        return 2
    if args.check:
        return _run_check(args, stream_out)
    if args.log_tail:
        return _run_log_tail(args, stream_out)
    if args.install:
        return _run_install(args, stream_out)
    if args.uninstall:
        return _run_uninstall(args, stream_out)
    if args.dry_run:
        return _dry_run_output(args, stream_out)
    return _run_speak(args, stream_out)


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "CheckResult",
    "DEFAULT_FALLBACK_TEXT",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_PROVIDER",
    "HAL_HOME",
    "InstallResult",
    "LAUNCH_AGENT_LABEL",
    "LAUNCH_AGENT_LOG",
    "LAUNCH_AGENT_ERR",
    "LAUNCH_AGENT_PLIST",
    "RECENT_COMMIT_COUNT",
    "SESSION_CONTEXT_PATH",
    "SYSTEM_PROMPT_TEMPLATE",
    "WelcomeContext",
    "build_welcome_envelope",
    "gather_context",
    "install_launch_agent",
    "main",
    "render_check_report",
    "render_launch_agent_plist",
    "render_system_prompt",
    "run_checks",
    "tail_launch_agent_logs",
    "uninstall_launch_agent",
    "welcome",
    "welcome_with_chain",
]
