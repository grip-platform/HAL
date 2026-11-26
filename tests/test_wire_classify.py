"""The HAL wire's route classifier (InfiniteMachine v2, Step 2).

Every request shape here is real: the paths are the ones the 2026-10-04 intercept
probe saw Claude Code 2.1.288 send (GRIP ``tests/fixtures/wire_probe/``), and the
quota check, API-key check and title calls are built as Claude Code 2.1.289's
own code builds them (read from the binary on 2026-10-05; see the
``lib/hal/wire_classify.py`` docstring). Identifiers are replaced with
placeholders.
"""
from __future__ import annotations

import gzip
import json
import logging
from email.message import Message

import pytest

from lib.hal import wire_classify
from lib.hal.wire_classify import RouteClass, classify

INF, AUX = RouteClass.INFERENCE, RouteClass.AUX
ACCT, UNK = RouteClass.ACCOUNT_BOUND, RouteClass.UNKNOWN

ORG = "00000000-0000-4000-8000-000000000000"
SESSION = "cse_01EXAMPLE0000"
METADATA = {"user_id": "user_example_account_example_session_example"}

# Claude Code 2.1.289's quota check: {model, max_tokens: 1, messages, metadata}.
QUOTA_CHECK = {"model": "claude-fable-5-1", "max_tokens": 1,
               "messages": [{"role": "user", "content": "quota"}],
               "metadata": METADATA}
# Its API-key check: one token, content "test", temperature 1.
API_KEY_CHECK = {"model": "claude-haiku-4-5", "max_tokens": 1,
                 "messages": [{"role": "user", "content": "test"}],
                 "temperature": 1, "metadata": METADATA}
TURN = {
    "model": "claude-opus-5-5", "max_tokens": 64000, "stream": True,
    "system": [{"type": "text", "text": "You are Claude Code, Anthropic's official CLI."}],
    "messages": [{"role": "user",
                  "content": [{"type": "text", "text": "fix the failing test"}]}],
    "tools": [{"name": "Bash", "description": "Run a shell command",
               "input_schema": {"type": "object",
                                "properties": {"command": {"type": "string"}},
                                "required": ["command"]}}],
    "thinking": {"type": "adaptive"}, "metadata": METADATA,
}
TURN_SAYING_QUOTA = {**TURN, "messages": [{"role": "user", "content": "quota"}]}


def _schema_call(properties):
    """A helper call asking for structured output with these schema properties."""
    schema = {"type": "object",
              "properties": {name: {"type": "string"} for name in properties},
              "required": list(properties), "additionalProperties": False}
    return {"model": "claude-haiku-4-5", "max_tokens": 32000,
            "system": [{"type": "text", "text": "Generate a short title."}],
            "messages": [{"role": "user", "content": "<conversation>...</conversation>"}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
            "metadata": METADATA}


SESSION_TITLE = _schema_call(["title"])
TELEPORT_TITLE = _schema_call(["title", "branch"])
RENAME_NAME = _schema_call(["name"])
LEGACY_TITLE = {**{k: v for k, v in SESSION_TITLE.items() if k != "output_config"},
                "output_format": SESSION_TITLE["output_config"]["format"]}

JSON = {"content-type": "application/json"}


def _body(payload):
    return payload if isinstance(payload, bytes) else json.dumps(payload).encode()


# (label, method, target, headers, body, expected)
CASES = [
    # inference: real turns, however they are worded
    ("turn", "POST", "/v1/messages?beta=true", JSON, TURN, INF),
    ("turn whose text is 'quota'", "POST", "/v1/messages?beta=true", JSON,
     TURN_SAYING_QUOTA, INF),
    ("quota body asking for 2 tokens", "POST", "/v1/messages", JSON,
     {**QUOTA_CHECK, "max_tokens": 2}, INF),
    ("max_tokens true is not 1", "POST", "/v1/messages", JSON,
     {**QUOTA_CHECK, "max_tokens": True}, INF),
    ("rename helper forks the conversation", "POST", "/v1/messages", JSON,
     RENAME_NAME, INF),
    ("lower-case method", "post", "/v1/messages", JSON, TURN, INF),
    # aux: same path family, not a turn
    ("quota check", "POST", "/v1/messages?beta=true", JSON, QUOTA_CHECK, AUX),
    ("api-key check", "POST", "/v1/messages?beta=true", JSON, API_KEY_CHECK, AUX),
    ("session title", "POST", "/v1/messages?beta=true", JSON, SESSION_TITLE, AUX),
    ("teleport title", "POST", "/v1/messages?beta=true", JSON, TELEPORT_TITLE, AUX),
    ("title via deprecated output_format", "POST", "/v1/messages", JSON,
     LEGACY_TITLE, AUX),
    ("count tokens", "POST", "/v1/messages/count_tokens?beta=true", JSON, TURN, AUX),
    ("cache keep-alive", "POST", "/v1/messages/cache_touch?beta=true", JSON,
     {"model": "claude-opus-5-5"}, AUX),
    # account-bound: every non-model path the 2026-10-04 probe saw, plus voice
    ("feature flags", "POST", "/api/eval/sdk-EXAMPLEkey0000", JSON, {}, ACCT),
    ("org plugins", "GET", f"/api/oauth/organizations/{ORG}/plugins/", {}, b"", ACCT),
    ("org marketplaces", "GET", f"/api/oauth/organizations/{ORG}/marketplaces",
     {}, b"", ACCT),
    ("account settings", "GET", "/api/oauth/account/settings", {}, b"", ACCT),
    ("grove", "GET", "/api/claude_code_grove", {}, b"", ACCT),
    ("bootstrap", "GET", "/api/claude_cli/bootstrap", {}, b"", ACCT),
    ("penguin mode", "GET", "/api/claude_code_penguin_mode", {}, b"", ACCT),
    ("create code session", "POST", "/v1/code/sessions", JSON, {}, ACCT),
    ("telemetry batch", "POST", "/api/event_logging/v2/batch", JSON, {}, ACCT),
    ("bridge", "POST", f"/v1/code/sessions/{SESSION}/bridge", JSON, {}, ACCT),
    ("worker events", "POST", f"/v1/code/sessions/{SESSION}/worker/events", JSON,
     {}, ACCT),
    ("worker get", "GET", f"/v1/code/sessions/{SESSION}/worker", {}, b"", ACCT),
    ("worker put", "PUT", f"/v1/code/sessions/{SESSION}/worker", JSON, {}, ACCT),
    ("notification preferences", "GET", "/api/claude_code/notification/preferences",
     {}, b"", ACCT),
    ("environments", "GET", "/v1/environments", {}, b"", ACCT),
    ("environment bridge", "POST", "/v1/environments/bridge/env_1", JSON, {}, ACCT),
    ("oauth token refresh", "POST", "/v1/oauth/token", JSON, {}, ACCT),
    ("voice stream", "GET", "/api/ws/speech_to_text/voice_stream?language=en",
     {"upgrade": "websocket", "connection": "Upgrade"}, b"", ACCT),
    ("a turn body on a code path stays account-bound", "POST",
     f"/v1/code/sessions/{SESSION}/bridge", JSON, TURN, ACCT),
    # unknown: everything else
    ("ultrareview quota", "GET", "/v1/ultrareview/quota", {}, b"", UNK),
    ("models list", "GET", "/v1/models", {}, b"", UNK),
    ("batches", "POST", "/v1/messages/batches", JSON, TURN, UNK),
    ("GET on messages", "GET", "/v1/messages", {}, b"", UNK),
    ("trailing slash is another path", "POST", "/v1/messages/", JSON, TURN, UNK),
    ("not under /v1/code", "POST", "/v1/codex", JSON, TURN, UNK),
    ("not under /api/oauth", "GET", "/api/oauthx/account", {}, b"", UNK),
    ("not under /api/eval", "POST", "/api/evaluate", JSON, {}, UNK),
    ("no messages", "POST", "/v1/messages", JSON, {"model": "claude-opus-5-5"}, UNK),
    ("empty messages", "POST", "/v1/messages", JSON, {**TURN, "messages": []}, UNK),
    ("not JSON", "POST", "/v1/messages", JSON, b"\x00\xffnot json", UNK),
    ("JSON array", "POST", "/v1/messages", JSON, b"[1, 2]", UNK),
    ("empty body", "POST", "/v1/messages", JSON, b"", UNK),
    ("absolute-form target", "POST", "https://api.anthropic.com/v1/messages", JSON,
     TURN, UNK),
]


@pytest.mark.parametrize("label,method,target,headers,body,expected", CASES,
                         ids=[case[0] for case in CASES])
def test_classifier_table(label, method, target, headers, body, expected):
    assert classify(method, target, headers, _body(body)) is expected


def test_the_four_classes_carry_the_contracts_names():
    assert {str(item) for item in RouteClass} == {
        "inference", "aux", "account-bound", "unknown"}
    assert RouteClass.ACCOUNT_BOUND == "account-bound"


def test_a_gzipped_turn_is_read_through_its_content_encoding():
    """Claude Code 2.1.289 can gzip request bodies; without the header it is opaque."""
    packed = gzip.compress(_body(TURN))
    assert classify("POST", "/v1/messages", {"Content-Encoding": "gzip"}, packed) is INF
    assert classify("POST", "/v1/messages", {}, packed) is UNK


def test_a_gzipped_quota_check_is_still_aux():
    packed = gzip.compress(_body(QUOTA_CHECK))
    assert classify("POST", "/v1/messages", [("content-encoding", "gzip")], packed) is AUX


def test_a_multi_member_gzip_body_is_read_whole():
    raw = _body(TURN)
    packed = gzip.compress(raw[:40]) + gzip.compress(raw[40:])
    assert classify("POST", "/v1/messages", {"content-encoding": "gzip"}, packed) is INF


@pytest.mark.parametrize("coding", ["br", "zstd", "deflate", "gzip, gzip"])
def test_an_encoding_hal_cannot_decode_leaves_the_body_opaque(coding):
    assert classify("POST", "/v1/messages", {"content-encoding": coding},
                    gzip.compress(_body(TURN))) is UNK


def test_a_body_past_the_decode_cap_is_opaque(monkeypatch):
    monkeypatch.setattr(wire_classify, "MAX_DECODED_BODY", 256)
    small, large = _body(QUOTA_CHECK), _body(TURN)
    assert len(small) <= 256 < len(large)
    headers = {"content-encoding": "gzip"}
    assert classify("POST", "/v1/messages", headers, gzip.compress(small)) is AUX
    assert classify("POST", "/v1/messages", headers, gzip.compress(large)) is UNK


def test_a_corrupt_gzip_body_is_opaque():
    broken = gzip.compress(_body(TURN))[:-12]
    assert classify("POST", "/v1/messages", {"content-encoding": "gzip"}, broken) is UNK


@pytest.mark.parametrize("hint", ["main", "subagent", "workflow", "compaction", "MAIN"])
def test_a_hint_naming_a_turn_vetoes_the_body_heuristics(hint):
    headers = {"x-claude-code-request-class": hint}
    assert classify("POST", "/v1/messages", headers, _body(SESSION_TITLE)) is INF
    assert classify("POST", "/v1/messages", headers, _body(QUOTA_CHECK)) is INF


def test_an_auxiliary_hint_does_not_widen_aux():
    """'auxiliary' also marks side questions the user sees; they stay inference."""
    headers = {"x-claude-code-request-class": "auxiliary"}
    assert classify("POST", "/v1/messages", headers, _body(TURN)) is INF
    assert classify("POST", "/v1/messages", headers, _body(SESSION_TITLE)) is AUX


@pytest.mark.parametrize("hint,probe", [(None, True), ("auxiliary", True), ("main", False),
                                        ("subagent", False), ("COMPACTION", False)])
def test_an_account_probe_is_what_classify_puts_in_aux(hint, probe):
    """HAL#765: the wire's probe mark follows classify's own rule, hint header included."""
    from lib.hal.wire_classify import is_account_probe
    headers = {"x-claude-code-request-class": hint} if hint else {}
    assert is_account_probe(headers, QUOTA_CHECK) is probe
    assert (classify("POST", "/v1/messages", headers, _body(QUOTA_CHECK)) is AUX) is probe
    assert is_account_probe(headers, TURN) is False


def _http_message(pairs):
    message = Message()
    for name, value in pairs:
        message[name] = value
    return message


@pytest.mark.parametrize("headers", [
    {"Content-Encoding": "gzip"},
    [("Content-Encoding", "gzip")],
    (["content-encoding", "gzip"],),
    _http_message([("Content-Encoding", "gzip")]),
], ids=["dict", "tuples", "lists", "http-message"])
def test_every_header_shape_is_read(headers):
    packed = gzip.compress(_body(TURN))
    assert classify("POST", "/v1/messages", headers, packed) is INF


@pytest.mark.parametrize("headers", [None, 42, [("only-one",)], ["junk"], object()])
def test_malformed_headers_never_raise(headers):
    assert classify("POST", "/v1/messages", headers, _body(TURN)) is INF


@pytest.mark.parametrize("method,target,body", [
    (None, None, None), ("", "", b""), ("POST", "/v1/messages", b"[" * 100000),
    ("POST", "/v1/messages", b"\xff\xfe\x00"), ("POST", "?", b"{}"),
])
def test_garbage_never_raises(method, target, body):
    assert classify(method, target, {}, body) in set(RouteClass)


def test_classify_writes_nothing_to_any_log(caplog):
    caplog.set_level(logging.DEBUG)
    secret = {"authorization": "Bearer sk-ant-oat01-NOT-A-REAL-TOKEN",
              "x-api-key": "sk-ant-api03-NOT-A-REAL-KEY"}
    for method, target, _headers, body, _ in (case[1:] for case in CASES):
        classify(method, target, secret, _body(body))
    assert caplog.records == []


def test_request_json_returns_the_decoded_object():
    payload = wire_classify.request_json({"content-encoding": "gzip"},
                                         gzip.compress(_body(TURN)))
    assert payload["model"] == "claude-opus-5-5"
    assert wire_classify.request_json({}, b"[]") is None
