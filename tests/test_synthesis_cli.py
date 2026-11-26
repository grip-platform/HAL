"""The ladder over JSON: what GRIP's AGORA sends HAL and what comes back.

Hermetic: every chat is a fake that records what it was asked, so nothing here
spends money or needs a network. The one subprocess test sends a request that
fails validation, which exits before any seat could be asked.
"""
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lib.hal.synthesis.cli import happi_chat, main
from lib.hal.synthesis.rungs import ChatReply

REPO = Path(__file__).resolve().parent.parent
VERDICTS = ["MERGE", "DEFER_CHANGES", "REQUEST_CHANGES"]


def _seat(model, role, vendor="google"):
    return {"model": model, "vendor": vendor, "role": role, "billing": "postpaid",
            "system": f"You are the {role}.", "max_tokens": 512}


def _request(**overrides):
    request = {
        "task": {"question": "Should this PR merge?", "evidence": ["review A", "review B"],
                 "verdicts": VERDICTS, "excluded_vendors": ["deepseek", "anthropic"]},
        "rungs": [
            {"effort": "HIGH", "seats": [_seat("gemini/gemini-2.5-flash-lite", r)
                                         for r in ("explorer", "verifier", "devil")]},
            {"effort": "XHIGH", "seats": [_seat("gemini/gemini-3.1-pro-preview", "synthesiser")]},
        ],
        "floor": "HIGH", "ceiling": "XHIGH", "budget_usd": 1.0, "agreement": 1.0,
    }
    request.update(overrides)
    return request


class FakeChat:
    """Answers by role, from a script; records every call it receives."""

    def __init__(self, verdicts_by_role):
        self.verdicts_by_role = verdicts_by_role
        self.calls = []

    def __call__(self, address, system, prompt, max_tokens):
        self.calls.append((address, system, prompt, max_tokens))
        role = system.removeprefix("You are the ").rstrip(".")
        return ChatReply(text=f"VERDICT: {self.verdicts_by_role[role]}\nbecause.",
                         tokens_in=1_000, tokens_out=100)


def _run(request, chat):
    out = io.StringIO()
    code = main(chat=chat, stdin=io.StringIO(json.dumps(request)), stdout=out)
    return code, (json.loads(out.getvalue()) if out.getvalue() else None)


def test_a_unanimous_council_settles_at_high_and_reports_every_seat():
    chat = FakeChat({"explorer": "MERGE", "verifier": "MERGE", "devil": "MERGE"})
    code, result = _run(_request(), chat)
    assert code == 0
    assert (result["effort"], result["decided"], result["verdict"]) == ("HIGH", True, "MERGE")
    assert [s["role"] for s in result["seats"]] == ["explorer", "verifier", "devil"]
    assert all(s["billing"] == "postpaid" and s["cost_usd"] > 0 for s in result["seats"])
    assert result["cost_usd"] == sum(s["cost_usd"] for s in result["seats"])
    assert len(chat.calls) == 3


def test_a_split_council_escalates_and_the_synthesiser_sees_what_it_said():
    chat = FakeChat({"explorer": "MERGE", "verifier": "MERGE", "devil": "REQUEST_CHANGES",
                     "synthesiser": "DEFER_CHANGES"})
    code, result = _run(_request(), chat)
    assert code == 0
    assert (result["effort"], result["verdict"], result["decided"]) == ("XHIGH", "DEFER_CHANGES", True)
    synthesiser_prompt = chat.calls[-1][2]
    assert "the previous rung (HIGH) said" in synthesiser_prompt
    assert len(result["seats"]) == 4


def test_no_budget_means_no_seat_is_asked():
    chat = FakeChat({})
    code, result = _run(_request(budget_usd=0.0), chat)
    assert code == 0
    assert result["budget_capped"] is True and result["decided"] is False
    assert chat.calls == []


def test_an_excluded_family_never_sits():
    chat = FakeChat({"explorer": "MERGE", "verifier": "MERGE", "devil": "MERGE"})
    request = _request()
    request["task"]["excluded_vendors"] = ["google"]
    code, result = _run(request, chat)
    assert code == 0
    assert chat.calls == []
    assert result["decided"] is False


def test_a_failing_provider_is_a_missing_vote_not_a_crash():
    def broken(address, system, prompt, max_tokens):
        raise ConnectionError("provider down")
    code, result = _run(_request(), broken)
    assert code == 0
    assert result["decided"] is False
    assert all(not s["ok"] and "provider down" in s["note"] for s in result["seats"])


def test_a_provider_error_returned_not_raised_is_also_a_missing_vote():
    # AGORA's review of HAL#624: the failure test above raises; HAPPI returns
    # its errors instead, and those must count as votes not cast too.
    def billing_refused(address, system, prompt, max_tokens):
        return ChatReply(error="provider_error:403: billing")
    code, result = _run(_request(), billing_refused)
    assert code == 0
    assert result["decided"] is False
    assert all(not s["ok"] and "403" in s["note"] for s in result["seats"])


@pytest.mark.parametrize("field, value", [
    ("evidence", "one block"),        # would have become ('o', 'n', 'e', ...)
    ("verdicts", "MERGE"),            # would have become five one-letter verdicts
    ("excluded_vendors", "google"),   # would have excluded no one
    ("verdicts", ["MERGE", 7]),
])
def test_a_list_field_given_as_anything_else_is_refused(capsys, field, value):
    request = _request()
    request["task"][field] = value
    code, _ = _run(request, FakeChat({}))
    assert code == 2
    assert field in capsys.readouterr().err


@pytest.mark.parametrize("seats, word", [
    ({"model": "x"}, "expected a JSON list"),
    (None, "expected a JSON list"),
    ([], "at least one seat"),
])
def test_a_rung_must_list_its_seats(capsys, seats, word):
    request = _request()
    request["rungs"][0]["seats"] = seats
    code, _ = _run(request, FakeChat({}))
    assert code == 2
    assert word in capsys.readouterr().err


@pytest.mark.parametrize("overrides", [
    {"budget_usd": -1.0}, {"agreement": 1.5}, {"agreement": "most"},
])
def test_ladder_settings_out_of_range_are_refused(capsys, overrides):
    code, _ = _run(_request(**overrides), FakeChat({}))
    assert code == 2
    assert "ladder settings" in capsys.readouterr().err


def test_the_budget_is_required(capsys):
    request = _request()
    del request["budget_usd"]
    code, _ = _run(request, FakeChat({}))
    assert code == 2
    assert "budget_usd" in capsys.readouterr().err


def test_a_missing_task_says_so_plainly(capsys):
    code, _ = _run(_request(task=None), FakeChat({}))
    assert code == 2
    assert "task: expected a JSON object" in capsys.readouterr().err


def test_garbage_from_happi_is_a_missing_vote_not_a_crash(monkeypatch):
    # The falsifier AGORA's technical reviewer named: a blank line, then junk,
    # then usage that is not an object. The run must still exit 0.
    def garbage(raw):
        yield "\n"
        yield "not json"
        yield json.dumps({"type": "completed", "usage": "lots"})
    monkeypatch.setattr("lib.hal.happi.runner.run_envelope_line", garbage)
    code, result = _run(_request(), happi_chat)
    assert code == 0
    assert result["decided"] is False
    assert all(not s["ok"] and "unreadable" in s["note"] for s in result["seats"])


def test_a_stray_unreadable_line_does_not_sink_an_answer(monkeypatch):
    def mostly_fine(raw):
        yield "not json"
        yield json.dumps({"type": "delta", "text": "VERDICT: MERGE"})
        yield json.dumps({"type": "completed", "usage": {"in": 10, "out": 5}})
    monkeypatch.setattr("lib.hal.happi.runner.run_envelope_line", mostly_fine)
    reply = happi_chat("gemini/x", "s", "p", 10)
    assert (reply.text, reply.tokens_in, reply.tokens_out, reply.error) == (
        "VERDICT: MERGE", 10, 5, "")


# The edges AGORA's MERGE review of HAL#624 named, each R0-tested and found to hold.

def _seat_with(**changes):
    request = _request()
    request["rungs"][0]["seats"][0].update(changes)
    return request


@pytest.mark.parametrize("request_, word", [
    (_request(rungs=[]), "at least one rung"),
    (_request(floor="XHIGH", ceiling="HIGH"), "above ceiling"),
    (_seat_with(max_tokens=0), "max_tokens"),
    (_seat_with(max_tokens=True), "max_tokens"),
    (_seat_with(max_tokens="big"), "max_tokens"),
    (_seat_with(model=""), "model must be non-empty text"),
    # AGORA on HAL#625: a list vendor became "['google']" and slipped past an
    # exclusion naming "google". A field that was sent but is the wrong type is
    # named as such, not as missing (AGORA on HAL#627).
    (_seat_with(vendor=["google"]), "vendor must be non-empty text"),
    (_seat_with(model=7), "model must be non-empty text"),
    (_seat_with(system=["be strict"]), "system must be text"),
    # Falsy non-text was coerced to "" before the type check (AGORA on HAL#627).
    (_seat_with(system=[]), "system must be text"),
    (_seat_with(system=0), "system must be text"),
    (_seat_with(system=False), "system must be text"),
])
def test_edge_requests_are_refused_with_a_plain_reason(capsys, request_, word):
    code, _ = _run(request_, FakeChat({}))
    assert code == 2
    assert word in capsys.readouterr().err


@pytest.mark.parametrize("seat_vendor, excluded", [
    (" google", "google"), ("google\n", "google"), ("Google", "google"), ("google", " GOOGLE "),
])
def test_a_padded_or_capitalised_family_is_still_excluded(seat_vendor, excluded):
    # AGORA on HAL#627: "google " is non-empty text, but it did not equal an
    # exclusion of "google", so the family could still sit.
    chat = FakeChat({"explorer": "MERGE", "verifier": "MERGE", "devil": "MERGE"})
    request = _request()
    request["task"]["excluded_vendors"] = [excluded]
    for seat in request["rungs"][0]["seats"]:
        seat["vendor"] = seat_vendor
    request["rungs"] = request["rungs"][:1]
    code, result = _run(request, chat)
    assert code == 0
    assert chat.calls == []
    assert result["decided"] is False


@pytest.mark.parametrize("entry", [["google"], 1, None])
def test_an_exclusion_entry_that_is_not_text_is_refused_not_crashed(capsys, entry):
    # AGORA's second review of HAL#627 expected an AttributeError here. _strings
    # refuses a non-text entry before _family ever sees it; this pins that.
    request = _request()
    request["task"]["excluded_vendors"] = ["deepseek", entry]
    code, _ = _run(request, FakeChat({}))
    assert code == 2
    assert "excluded_vendors" in capsys.readouterr().err


def test_a_padded_or_upper_case_billing_mode_is_read_like_a_vendor():
    chat = FakeChat({"explorer": "MERGE", "verifier": "MERGE", "devil": "MERGE"})
    code, result = _run(_seat_with(billing=" POSTPAID "), chat)
    assert code == 0
    assert result["seats"][0]["billing"] == "postpaid"


def test_a_null_max_tokens_takes_the_default():
    chat = FakeChat({"explorer": "MERGE", "verifier": "MERGE", "devil": "MERGE"})
    code, result = _run(_seat_with(max_tokens=None), chat)
    assert code == 0
    assert result["verdict"] == "MERGE"
    # The seat whose max_tokens was null is the first one asked; it got 2048.
    assert chat.calls[0][3] == 2_048


def test_a_seat_that_is_not_an_object_is_refused(capsys):
    request = _request()
    request["rungs"][0]["seats"] = ["gemini"]
    code, _ = _run(request, FakeChat({}))
    assert code == 2
    assert "seat: expected a JSON object" in capsys.readouterr().err


@pytest.mark.parametrize("question", [["q"], 7, "", "   "])
def test_the_question_must_be_text(capsys, question):
    request = _request()
    request["task"]["question"] = question
    code, _ = _run(request, FakeChat({}))
    assert code == 2
    assert "question" in capsys.readouterr().err


def test_a_null_floor_or_ceiling_takes_the_default():
    chat = FakeChat({"explorer": "MERGE", "verifier": "MERGE", "devil": "MERGE"})
    code, result = _run(_request(floor=None, ceiling=None), chat)
    assert code == 0
    assert result["verdict"] == "MERGE"


def test_stdin_that_is_not_utf8_exits_2(capsys):
    stdin = io.TextIOWrapper(io.BytesIO(b"\xff\xfe{"), encoding="utf-8")
    assert main(chat=FakeChat({}), stdin=stdin, stdout=io.StringIO()) == 2
    assert "not JSON" in capsys.readouterr().err


def test_a_null_delta_adds_nothing_and_an_error_drops_the_partial_answer(monkeypatch):
    def null_then_text(raw):
        yield json.dumps({"type": "delta", "text": None})
        yield json.dumps({"type": "delta", "text": "VERDICT: MERGE"})
    monkeypatch.setattr("lib.hal.happi.runner.run_envelope_line", null_then_text)
    assert happi_chat("gemini/x", "s", "p", 10).text == "VERDICT: MERGE"

    def cut_off(raw):
        yield json.dumps({"type": "delta", "text": "VERDICT: MERGE"})
        yield json.dumps({"type": "error", "code": "provider_error:500", "message": "stream cut"})
    monkeypatch.setattr("lib.hal.happi.runner.run_envelope_line", cut_off)
    reply = happi_chat("gemini/x", "s", "p", 10)
    assert (reply.text, reply.error) == ("", "provider_error:500: stream cut")


def test_malformed_requests_exit_2_and_name_the_problem(capsys):
    for request, word in [({"rungs": []}, "task"),
                          (_request(rungs="HIGH"), "rungs"),
                          (_request(floor="HIGHEST"), "HIGHEST")]:
        code, _ = _run(request, FakeChat({}))
        assert code == 2
        assert word in capsys.readouterr().err
    bad_billing = _request()
    bad_billing["rungs"][0]["seats"][0]["billing"] = "free_lunch"
    code, _ = _run(bad_billing, FakeChat({}))
    assert code == 2
    assert "seat" in capsys.readouterr().err


def test_the_module_runs_as_a_command():
    env = {**os.environ, "PYTHONPATH": str(REPO), "HAL_HOME": str(REPO)}
    proc = subprocess.run([sys.executable, "-m", "lib.hal.synthesis"], input="{}",
                          capture_output=True, text=True, cwd=REPO, env=env, timeout=60)
    assert proc.returncode == 2, proc.stderr
    assert "task" in proc.stderr


def test_happi_chat_sends_one_chat_envelope_and_folds_the_events(monkeypatch):
    sent = []

    def fake_run(raw):
        sent.append(json.loads(raw))
        yield json.dumps({"type": "started"})
        yield json.dumps({"type": "delta", "text": "VERDICT: MERGE\n"})
        yield json.dumps({"type": "delta", "text": "fine."})
        yield json.dumps({"type": "completed", "usage": {"in": 1200, "out": 340}})

    monkeypatch.setattr("lib.hal.happi.runner.run_envelope_line", fake_run)
    reply = happi_chat("gemini/gemini-2.5-flash-lite", "You judge.", "Should it merge?", 512)
    assert (reply.text, reply.tokens_in, reply.tokens_out, reply.error) == (
        "VERDICT: MERGE\nfine.", 1200, 340, "")
    assert sent[0]["cmd"] == "chat" and sent[0]["content"] == "Should it merge?"
    assert sent[0]["flags"] == {"provider": "gemini/gemini-2.5-flash-lite", "system": "You judge.",
                                "max_tokens": 512, "temperature": 0}


def test_happi_chat_reports_a_provider_error_instead_of_raising(monkeypatch):
    def fake_run(raw):
        yield json.dumps({"type": "error", "code": "provider_error:403", "message": "billing"})

    monkeypatch.setattr("lib.hal.happi.runner.run_envelope_line", fake_run)
    reply = happi_chat("gemini/x", "s", "p", 10)
    assert reply.error == "provider_error:403: billing" and reply.text == ""


def test_the_envelope_happi_chat_builds_is_one_happi_accepts(monkeypatch):
    # An envelope HAPPI rejected would make every seat fail as a parse error.
    from lib.hal.happi.envelope import HappiEnvelope
    captured = []
    monkeypatch.setattr("lib.hal.happi.runner.run_envelope_line",
                        lambda raw: captured.append(raw) or iter(()))
    happi_chat("gemini/gemini-2.5-flash-lite", "s", "the prompt", 64)
    parsed = HappiEnvelope.parse(captured[0])
    assert parsed.flags["provider"] == "gemini/gemini-2.5-flash-lite"
    assert parsed.user_message() == "the prompt"
