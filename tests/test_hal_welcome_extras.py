"""Tests for hal welcome provider-chain + log-tail (W6)."""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lib.hal.voice.stream_speak import StreamSpeakResult
from lib.hal.welcome import (
    DEFAULT_FALLBACK_TEXT,
    _parse_provider_chain,
    main,
    tail_launch_agent_logs,
    welcome_with_chain,
)


# ---------------- _parse_provider_chain ----------------


def test_parse_empty_string_returns_empty_list():
    assert _parse_provider_chain("") == []


def test_parse_single_provider():
    assert _parse_provider_chain("anthropic") == ["anthropic"]


def test_parse_comma_separated_list():
    assert _parse_provider_chain("anthropic,gemini,groq") == [
        "anthropic", "gemini", "groq",
    ]


def test_parse_tolerates_whitespace_around_commas():
    assert _parse_provider_chain("anthropic , gemini , groq") == [
        "anthropic", "gemini", "groq",
    ]


def test_parse_drops_empty_entries():
    assert _parse_provider_chain("anthropic,,gemini,") == [
        "anthropic", "gemini",
    ]


def test_parse_preserves_slash_separated_model_specs():
    assert _parse_provider_chain(
        "gemini/gemini-2.0-flash,groq/llama-3.1-70b-versatile"
    ) == [
        "gemini/gemini-2.0-flash",
        "groq/llama-3.1-70b-versatile",
    ]


# ---------------- welcome_with_chain ----------------


def _fake_ctx_builder():
    from lib.hal.welcome import WelcomeContext
    return lambda: WelcomeContext(
        signature="V>>",
        branch="main",
        head_oid_short="abc1234",
        head_subject="test",
        recent_prs=["abc1234 test"],
        session_notes="notes",
    )


def test_chain_returns_first_provider_that_speaks(monkeypatch):
    calls = []

    def _fake_stream_say(source, *, voice, rate_wpm, spawn):
        calls.append(source)
        if len(calls) == 1:
            return StreamSpeakResult(
                spoken_text="", delta_count=0,
                error="HAPPI missing_credentials: no token",
            )
        if len(calls) == 2:
            return StreamSpeakResult(
                spoken_text="hello", delta_count=3, error=None,
            )
        return StreamSpeakResult(spoken_text="UNEXPECTED", delta_count=99, error=None)

    monkeypatch.setattr("lib.hal.welcome.stream_say", _fake_stream_say)

    result = welcome_with_chain(
        providers=["anthropic", "gemini", "groq"],
        context_builder=_fake_ctx_builder(),
    )
    assert result.spoken_text == "hello"
    assert result.delta_count == 3
    assert len(calls) == 2


def test_chain_falls_through_to_fixed_text_when_all_silent(monkeypatch):
    silent = StreamSpeakResult(
        spoken_text="", delta_count=0,
        error="HAPPI missing_credentials: no token",
    )
    fallback = StreamSpeakResult(
        spoken_text=DEFAULT_FALLBACK_TEXT, delta_count=1, error=None,
    )
    calls = []

    def _fake_stream_say(source, *, voice, rate_wpm, spawn):
        calls.append(source)
        return silent if len(calls) <= 3 else fallback

    monkeypatch.setattr("lib.hal.welcome.stream_say", _fake_stream_say)

    result = welcome_with_chain(
        providers=["anthropic", "gemini", "groq"],
        context_builder=_fake_ctx_builder(),
    )
    assert len(calls) == 4
    assert result.spoken_text == DEFAULT_FALLBACK_TEXT


def test_chain_partial_audio_accepts_without_retry(monkeypatch):
    partial = StreamSpeakResult(
        spoken_text="Welcome back, V", delta_count=4,
        error="HAPPI provider_http_429: rate limited",
    )
    calls = []

    def _fake_stream_say(source, *, voice, rate_wpm, spawn):
        calls.append(source)
        return partial

    monkeypatch.setattr("lib.hal.welcome.stream_say", _fake_stream_say)

    result = welcome_with_chain(
        providers=["anthropic", "gemini"],
        context_builder=_fake_ctx_builder(),
    )
    assert len(calls) == 1
    assert result.delta_count == 4
    assert "rate limited" in result.error


def test_chain_empty_provider_list_goes_straight_to_fallback(monkeypatch):
    fallback = StreamSpeakResult(
        spoken_text="fallback", delta_count=1, error=None,
    )

    def _fake_stream_say(source, *, voice, rate_wpm, spawn):
        return fallback

    monkeypatch.setattr("lib.hal.welcome.stream_say", _fake_stream_say)
    result = welcome_with_chain(
        providers=[], context_builder=_fake_ctx_builder(),
    )
    assert result.spoken_text == "fallback"


# ---------------- tail_launch_agent_logs ----------------


def test_tail_reports_missing_logs_cleanly(tmp_path):
    out = tail_launch_agent_logs(
        log_path=tmp_path / "absent.log",
        err_path=tmp_path / "absent.err",
    )
    assert "(no log at" in out
    assert "stdout tail" in out
    assert "stderr tail" in out


def test_tail_returns_last_n_lines(tmp_path):
    log = tmp_path / "hal.log"
    log.write_text(
        "\n".join(f"line-{i}" for i in range(1, 101)) + "\n",
        encoding="utf-8",
    )
    err = tmp_path / "hal.err"
    err.write_text("only-err-line\n", encoding="utf-8")

    out = tail_launch_agent_logs(log_path=log, err_path=err, lines=5)
    assert "line-96" in out
    assert "line-100" in out
    assert "line-95" not in out
    assert "only-err-line" in out


def test_tail_handles_empty_log_file(tmp_path):
    log = tmp_path / "empty.log"
    log.write_text("", encoding="utf-8")
    err = tmp_path / "empty.err"
    err.write_text("", encoding="utf-8")
    out = tail_launch_agent_logs(log_path=log, err_path=err)
    assert "(log is empty)" in out


def test_tail_separates_stdout_and_stderr_sections(tmp_path):
    log = tmp_path / "hal.log"
    log.write_text("stdout-entry\n", encoding="utf-8")
    err = tmp_path / "hal.err"
    err.write_text("stderr-entry\n", encoding="utf-8")
    out = tail_launch_agent_logs(log_path=log, err_path=err)
    stdout_idx = out.index("--- stdout tail")
    stderr_idx = out.index("--- stderr tail")
    assert stdout_idx < stderr_idx


# ---------------- main() --log-tail + --provider-chain CLI ----------------


def test_main_log_tail_prints_both_sections(monkeypatch, tmp_path):
    log = tmp_path / "hal.log"
    log.write_text("one\ntwo\nthree\n", encoding="utf-8")
    err = tmp_path / "hal.err"
    err.write_text("err-one\n", encoding="utf-8")

    monkeypatch.setattr("lib.hal.welcome.LAUNCH_AGENT_LOG", log)
    monkeypatch.setattr("lib.hal.welcome.LAUNCH_AGENT_ERR", err)

    stream = io.StringIO()
    rc = main(["--log-tail"], stream_out=stream)
    assert rc == 0
    output = stream.getvalue()
    assert "stdout tail" in output
    assert "stderr tail" in output
    assert "three" in output
    assert "err-one" in output


def test_main_log_tail_lines_flag_caps_output(monkeypatch, tmp_path):
    log = tmp_path / "hal.log"
    log.write_text("\n".join(f"l{i}" for i in range(50)) + "\n", encoding="utf-8")
    err = tmp_path / "hal.err"
    err.write_text("", encoding="utf-8")

    monkeypatch.setattr("lib.hal.welcome.LAUNCH_AGENT_LOG", log)
    monkeypatch.setattr("lib.hal.welcome.LAUNCH_AGENT_ERR", err)

    stream = io.StringIO()
    rc = main(["--log-tail", "--log-tail-lines", "3"], stream_out=stream)
    assert rc == 0
    output = stream.getvalue()
    assert "l49" in output
    assert "l47" in output
    assert "l46" not in output


def test_main_provider_chain_flag_invokes_welcome_with_chain(monkeypatch):
    captured = {}

    def _spy(**kwargs):
        captured.update(kwargs)
        return StreamSpeakResult(spoken_text="ok", delta_count=1, error=None)

    monkeypatch.setattr("lib.hal.welcome.welcome_with_chain", _spy)
    stream = io.StringIO()
    rc = main(
        ["--provider-chain", "anthropic,gemini", "--voice", "Samantha"],
        stream_out=stream,
    )
    assert rc == 0
    assert captured["providers"] == ["anthropic", "gemini"]
    assert captured["voice"] == "Samantha"


def test_main_text_flag_takes_precedence_over_provider_chain(monkeypatch):
    chain_called = {"n": 0}
    text_called = {"n": 0}

    def _chain_spy(**_):
        chain_called["n"] += 1
        return StreamSpeakResult(spoken_text="chain", delta_count=1, error=None)

    def _welcome_spy(*, fixed_text=None, **_):
        if fixed_text is not None:
            text_called["n"] += 1
        return StreamSpeakResult(spoken_text="text", delta_count=1, error=None)

    monkeypatch.setattr("lib.hal.welcome.welcome_with_chain", _chain_spy)
    monkeypatch.setattr("lib.hal.welcome.welcome", _welcome_spy)
    stream = io.StringIO()
    rc = main(
        ["--provider-chain", "anthropic,gemini", "--text", "Hello."],
        stream_out=stream,
    )
    assert rc == 0
    assert text_called["n"] == 1
    assert chain_called["n"] == 0


def test_main_partial_audio_returns_zero_not_one(monkeypatch):
    partial = StreamSpeakResult(
        spoken_text="partial", delta_count=5,
        error="HAPPI provider_http_429",
    )
    monkeypatch.setattr("lib.hal.welcome.welcome", lambda **_: partial)
    stream = io.StringIO()
    rc = main(["--text", "irrelevant"], stream_out=stream)
    assert rc == 0


def test_main_silent_failure_returns_one(monkeypatch):
    silent = StreamSpeakResult(
        spoken_text="", delta_count=0,
        error="HAPPI missing_credentials: no token",
    )
    monkeypatch.setattr("lib.hal.welcome.welcome", lambda **_: silent)
    stream = io.StringIO()
    rc = main(["--text", "x"], stream_out=stream)
    assert rc == 1
    assert "missing_credentials" in stream.getvalue()
