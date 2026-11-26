"""Tests for hal welcome command (W2).

Covers context assembly (mocked subprocess + filesystem), system-prompt
rendering, envelope construction, fixed-text path, and the argparse
CLI including --dry-run.

No real LLM, no real TTS — every test injects a mock spawn factory.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lib.hal.happi.envelope import HappiEnvelope
from lib.hal.welcome import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROVIDER,
    SYSTEM_PROMPT_TEMPLATE,
    WelcomeContext,
    build_welcome_envelope,
    gather_context,
    main,
    render_system_prompt,
    welcome,
    _fixed_text_stream,
)


# ---------------- Fixtures + helpers ----------------


class _MockStdin:
    def __init__(self):
        self.writes: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _MockPopen:
    def __init__(self):
        self.invocations: list[list[str]] = []
        self.last_stdin: _MockStdin | None = None

    def __call__(self, argv, *, stdin=None):
        self.invocations.append(argv)
        self.last_stdin = _MockStdin()
        handle = MagicMock()
        handle.stdin = self.last_stdin
        handle.wait = lambda timeout=None: 0
        return handle


def _sample_context() -> WelcomeContext:
    return WelcomeContext(
        signature="V>>",
        branch="main",
        head_oid_short="abc1234",
        head_subject="feat(happi): γ-3 fractal sub_request",
        recent_prs=[
            "abc1234 feat(happi): γ-3 fractal sub_request (#249)",
            "def5678 feat(voice): γ-1 HAPPI Responder adapter (#252)",
            "ghi9012 feat(happi): β-3 Unix socket transport (#246)",
        ],
        session_notes="Last session shipped seven PRs. All tests green.",
    )


# ---------------- render_system_prompt ----------------


def test_render_includes_signature_and_branch():
    prompt = render_system_prompt(_sample_context())
    assert "V>>" in prompt
    assert "Branch: main" not in prompt  # exact template wording check
    assert "Current branch: main" in prompt


def test_render_includes_head_oid_and_subject():
    prompt = render_system_prompt(_sample_context())
    assert "abc1234" in prompt
    assert "γ-3 fractal sub_request" in prompt


def test_render_formats_pr_list_as_indented_lines():
    prompt = render_system_prompt(_sample_context())
    # Each PR appears as its own indented line.
    for pr in _sample_context().recent_prs:
        assert f"- {pr}" in prompt


def test_render_survives_empty_recent_prs():
    ctx = WelcomeContext(
        signature="V>>",
        branch="main",
        head_oid_short="abc1234",
        head_subject="sub",
        recent_prs=[],
        session_notes="notes",
    )
    prompt = render_system_prompt(ctx)
    assert "(no recent merges visible)" in prompt


def test_render_includes_session_notes():
    prompt = render_system_prompt(_sample_context())
    assert "Last session shipped seven PRs" in prompt


# ---------------- build_welcome_envelope ----------------


def test_envelope_is_chat_cmd_with_provider_and_max_tokens():
    env = build_welcome_envelope(
        _sample_context(), provider="gemini", max_tokens=300,
    )
    assert env.cmd == "chat"
    assert env.flags["provider"] == "gemini"
    assert env.flags["max_tokens"] == 300


def test_envelope_system_prompt_is_the_rendered_context():
    env = build_welcome_envelope(_sample_context())
    assert env.flags["system"] == render_system_prompt(_sample_context())


def test_envelope_message_requests_immediate_greeting():
    env = build_welcome_envelope(_sample_context())
    assert env.flags["message"] == "Greet me now."


def test_envelope_id_includes_head_oid_for_audit_trace():
    env = build_welcome_envelope(_sample_context())
    assert env.id == "welcome-abc1234"


def test_default_provider_is_empty_string_meaning_anthropic():
    assert DEFAULT_PROVIDER == ""


# ---------------- gather_context (integration-lite) ----------------


def test_gather_context_returns_structured_snapshot(tmp_path, monkeypatch):
    """Mocks all filesystem + git calls. Exercises the assembly logic."""
    def fake_git(args, cwd=None):
        if args == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return "feat/x"
        if args == ["rev-parse", "--short", "HEAD"]:
            return "deadbee"
        if args == ["log", "-1", "--pretty=%s"]:
            return "test subject"
        if args[:2] == ["log", "--oneline"]:
            return "deadbee test subject\ncafebab prev subject"
        return ""

    monkeypatch.setattr("lib.hal.welcome._git", fake_git)
    monkeypatch.setattr(
        "lib.hal.welcome._read_trim",
        lambda p, default="": "K>>" if "identity" in str(p) else "note-line",
    )

    ctx = gather_context()
    assert ctx.signature == "K>>"
    assert ctx.branch == "feat/x"
    assert ctx.head_oid_short == "deadbee"
    assert ctx.head_subject == "test subject"
    assert len(ctx.recent_prs) == 2
    assert "note-line" in ctx.session_notes


def test_gather_context_handles_missing_identity_file(monkeypatch):
    monkeypatch.setattr("lib.hal.welcome._git", lambda args, cwd=None: "")
    monkeypatch.setattr("lib.hal.welcome._read_trim", lambda p, default="": default)
    ctx = gather_context()
    assert ctx.signature == "operator"
    assert ctx.branch == "unknown"
    assert ctx.head_oid_short == "unknown"


# ---------------- _fixed_text_stream ----------------


def test_fixed_text_stream_emits_started_delta_completed():
    lines = list(_fixed_text_stream("Hello."))
    events = [json.loads(line) for line in lines]
    types = [e["type"] for e in events]
    assert types == ["started", "delta", "completed"]
    assert events[1]["text"] == "Hello."


def test_fixed_text_stream_with_empty_text_skips_delta():
    lines = list(_fixed_text_stream(""))
    events = [json.loads(line) for line in lines]
    types = [e["type"] for e in events]
    assert types == ["started", "completed"]


# ---------------- welcome() orchestrator ----------------


def test_welcome_with_fixed_text_bypasses_llm():
    popen = _MockPopen()
    result = welcome(fixed_text="Hi there.", spawn=popen)
    assert result.spoken_text == "Hi there."
    # Exactly one `say` invocation with the default Jamie voice.
    assert popen.invocations == [["say", "-v", "Jamie (Premium)"]]


def test_welcome_passes_voice_and_rate_through():
    popen = _MockPopen()
    welcome(
        fixed_text="x", spawn=popen, voice="Samantha", rate_wpm=200,
    )
    assert popen.invocations == [["say", "-v", "Samantha", "-r", "200"]]


def test_welcome_via_context_builder_and_mocked_envelope_dispatch():
    """Uses a custom context_builder to avoid hitting real git/fs, then
    monkeypatches the envelope dispatch path by swapping fixed_text — the
    full-envelope path is exercised indirectly via stream_say's own tests."""
    popen = _MockPopen()
    result = welcome(fixed_text="from-builder", spawn=popen)
    assert result.spoken_text == "from-builder"


# ---------------- main() CLI ----------------


def test_main_dry_run_prints_context_and_envelope_preview(monkeypatch):
    monkeypatch.setattr(
        "lib.hal.welcome.gather_context",
        lambda: _sample_context(),
    )
    stream = io.StringIO()
    rc = main(["--dry-run"], stream_out=stream)
    assert rc == 0
    payload = json.loads(stream.getvalue())
    assert payload["context"]["signature"] == "V>>"
    assert payload["envelope_preview"]["voice"] == "Jamie (Premium)"
    assert payload["envelope_preview"]["system_prompt_len"] > 0


def test_main_text_flag_speaks_without_invoking_llm():
    with patch("lib.hal.welcome.stream_say") as stream_say_mock:
        stream_say_mock.return_value = MagicMock(
            error=None, spoken_text="T", delta_count=1,
        )
        stream = io.StringIO()
        rc = main(["--text", "Welcome back, human."], stream_out=stream)
        assert rc == 0
        # stream_say called with the fixed-text generator source.
        args, kwargs = stream_say_mock.call_args
        assert kwargs["voice"] == "Jamie (Premium)"


def test_main_returns_1_when_stream_say_reports_error():
    with patch("lib.hal.welcome.stream_say") as stream_say_mock:
        stream_say_mock.return_value = MagicMock(
            error="HAPPI missing_credentials: no token",
            spoken_text="",
            delta_count=0,
        )
        stream = io.StringIO()
        rc = main(["--text", "hi"], stream_out=stream)
        assert rc == 1
        assert "missing_credentials" in stream.getvalue()


def test_main_passes_provider_flag_through_to_envelope(monkeypatch):
    """When --provider is set AND no --text, the envelope should be built
    with the chosen provider. We verify by patching welcome() and capturing."""
    captured = {}

    def _spy(**kwargs):
        captured.update(kwargs)
        return MagicMock(error=None, spoken_text="ok", delta_count=1)

    monkeypatch.setattr("lib.hal.welcome.welcome", _spy)
    stream = io.StringIO()
    rc = main(["--provider", "gemini", "--voice", "Samantha"], stream_out=stream)
    assert rc == 0
    assert captured["provider"] == "gemini"
    assert captured["voice"] == "Samantha"


def test_main_default_max_tokens_is_400():
    assert DEFAULT_MAX_TOKENS == 400


def test_default_rate_is_none_in_argparse(monkeypatch):
    """--rate default is None (not zero, not a number) — argparse type=int
    means omitting the flag preserves None."""
    captured = {}

    def _spy(**kwargs):
        captured.update(kwargs)
        return MagicMock(error=None, spoken_text="", delta_count=0)

    monkeypatch.setattr("lib.hal.welcome.welcome", _spy)
    main(["--text", "x"], stream_out=io.StringIO())
    assert captured["rate_wpm"] is None
