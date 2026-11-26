"""HAPPI tool loop in the OpenAI-compat adapter (W1b of the Kimi plan).

Contract under test, in the order it was built:

1. The ``tool_call`` / ``tool_result`` encoders carry the fields the spec
   table names (``happi.md`` "Events (stdout)": ``name``, ``input``,
   ``call_id`` / ``name``, ``output``, ``call_id``). Before W1b the library
   emitted ``{tool, args}`` / ``{tool, ok, result}`` with no ``call_id`` --
   the same divergence class HAL#523 closed for ``sub_request``. The spec
   table is read from the file and used as the oracle, so a future drift in
   either direction fails here rather than in a consumer.

Every fixture that reproduces provider traffic is seeded from chunks
observed live against ``api.moonshot.ai`` (2026-09-07), never from an
invented shape.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from lib.hal.happi.events import emit_tool_call, emit_tool_result


_HAPPI_MD = Path(__file__).resolve().parents[1] / "happi.md"
_ENVELOPE_KEYS = {"v", "id", "type", "ts"}


def _spec_required_fields(event_type: str) -> set[str]:
    """Parse the ``Events (stdout)`` table row for ``event_type`` and return
    the set of additional required field names it lists."""
    text = _HAPPI_MD.read_text(encoding="utf-8")
    row = re.search(
        rf"^\|\s*`{re.escape(event_type)}`\s*\|\s*(.+?)\s*\|", text, flags=re.M
    )
    assert row, f"no spec table row for {event_type!r} in happi.md"
    return set(re.findall(r"`([a-z_]+)`", row.group(1)))


# --- 1. encoders match the spec table ---------------------------------------

def test_emit_tool_call_carries_name_input_call_id():
    evt = json.loads(emit_tool_call("req-1", "get_time", {"tz": "UTC"}, "call_1"))
    assert evt["type"] == "tool_call"
    assert evt["name"] == "get_time"
    assert evt["input"] == {"tz": "UTC"}
    assert evt["call_id"] == "call_1"
    # The pre-W1b shape must be gone, not merely supplemented.
    assert "tool" not in evt and "args" not in evt


def test_emit_tool_result_carries_name_output_call_id():
    evt = json.loads(emit_tool_result("req-1", "get_time", "2026-09-07T21:00Z", "call_1"))
    assert evt["type"] == "tool_result"
    assert evt["name"] == "get_time"
    assert evt["output"] == "2026-09-07T21:00Z"
    assert evt["call_id"] == "call_1"
    assert "ok" not in evt and "result" not in evt


@pytest.mark.parametrize("event_type, line", [
    ("tool_call", emit_tool_call("req-1", "n", {}, "c")),
    ("tool_result", emit_tool_result("req-1", "n", "", "c")),
])
def test_tool_event_extra_keys_equal_the_spec_table_exactly(event_type, line):
    """The spec table is the contract (Axiom 4). The event's non-envelope
    keys must be exactly the row's required fields -- no more, no fewer."""
    extra = set(json.loads(line)) - _ENVELOPE_KEYS
    assert extra == _spec_required_fields(event_type)


def test_tool_call_input_is_passed_through_untouched():
    """``input`` is whatever the adapter decoded; a raw string survives when
    the provider's ``arguments`` was not valid JSON (truncated at the token
    budget), so nothing is silently lost."""
    evt = json.loads(emit_tool_call("req-1", "f", '{"city": "Bei', "call_9"))
    assert evt["input"] == '{"city": "Bei'


# --- 2. envelope flags reach the body; refusals happen before dispatch ------
#
# New chat flags (no prior convention existed in lib/hal/happi/):
#   flags.tools        OpenAI tool schemas, sent verbatim as `tools`
#   flags.tool_choice  "auto" | "none" | "required" | {type:function,...}
#   flags.messages     prior turns, echoed VERBATIM between flags.system and
#                      the current user turn (vendor: "pass the complete
#                      assistant message returned by the API back to
#                      `messages` as-is (including `reasoning_content`)")

from unittest.mock import patch  # noqa: E402

from lib.hal.happi.envelope import HappiEnvelope  # noqa: E402
from lib.hal.happi.providers import openai_compat as provider  # noqa: E402

GET_TIME = {"type": "function", "function": {
    "name": "get_time",
    "description": "Return the current time in UTC as an ISO-8601 string.",
    "parameters": {"type": "object", "properties": {}}}}
PROMPT = "What time is it? Use the tool."
# The assistant message kimi-k3 returned live on 2026-09-07 (assembled per
# the vendor's streaming recipe) and the tool message sent back on turn 2.
K3_ASSISTANT = {"role": "assistant", "content": "",
                "reasoning_content": "The user explicitly asks to use the tool for current time. Need call get_time.",
                "tool_calls": [{"id": "get_time_0", "type": "function",
                                "function": {"name": "get_time", "arguments": "{}"}}]}
TOOL_MSG = {"role": "tool", "tool_call_id": "get_time_0", "name": "get_time",
            "content": "2026-09-07T21:05:00Z"}


def _env(args=(PROMPT,), **flags) -> HappiEnvelope:
    return HappiEnvelope.parse({
        "v": "happi/1.1", "id": "req-loop", "cmd": "chat", "args": list(args),
        "flags": {"provider": "moonshot/kimi-k3", **flags},
        "auth": {"scheme": "apikey", "token": "sk-fake-key-for-tests-0123456789abcdef"},
    })


def _body(args=(PROMPT,), model="kimi-k3", **flags) -> dict:
    return provider._build_request_body(model, _env(args, **flags))


def _events(env) -> list[dict]:
    return [json.loads(line) for line in provider.stream_openai_compat(env)]


class _FakeSSEResponse:
    def __init__(self, lines):
        self._lines = [l.encode("utf-8") for l in lines]
        self.closed = False

    def __iter__(self):
        return iter(self._lines)

    def close(self):
        self.closed = True


def test_flags_tools_reach_the_body_verbatim():
    assert _body(tools=[GET_TIME])["tools"] == [GET_TIME]


def test_flags_tool_choice_reaches_the_body():
    assert _body(tools=[GET_TIME], tool_choice="auto")["tool_choice"] == "auto"


def test_required_tool_choice_is_sent_to_k3():
    assert _body(tools=[GET_TIME], tool_choice="required")["tool_choice"] == "required"


def test_body_without_tool_flags_has_no_tool_keys():
    body = _body()
    assert not {"tools", "tool_choice"} & set(body)
    assert body["messages"] == [{"role": "user", "content": PROMPT}]


def test_flags_messages_are_echoed_verbatim_after_the_system_prompt():
    history = [{"role": "user", "content": PROMPT}, K3_ASSISTANT, TOOL_MSG]
    body = _body(args=(), system="Be brief.", messages=history, tools=[GET_TIME])
    assert body["messages"] == [{"role": "system", "content": "Be brief."}, *history]
    # reasoning_content and tool_calls rode through untouched
    assert body["messages"][2]["reasoning_content"] == K3_ASSISTANT["reasoning_content"]
    assert body["messages"][2]["tool_calls"] == K3_ASSISTANT["tool_calls"]


def test_flags_messages_plus_a_new_user_turn_appends_it_last():
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    body = _body(args=("and now?",), messages=history)
    assert body["messages"] == [*history, {"role": "user", "content": "and now?"}]


def test_no_history_and_no_prompt_still_sends_the_empty_user_turn():
    # byte-identical to the pre-W1b body for a prompt-less envelope
    assert _body(args=())["messages"] == [{"role": "user", "content": ""}]


def test_flags_messages_are_not_mutated_by_body_fixes():
    # Deep snapshot, not dict(): a shallow copy shares the nested tool_calls
    # list with K3_ASSISTANT, so an in-place mutation of that list would change
    # both sides of the old `history == [K3_ASSISTANT]` and stay invisible
    # (exogenous review of HAL#549, seal 0e9e254dce7fb442; proved by mutating
    # the adapter to append to tool_calls in place: old form green, this red).
    history = [copy.deepcopy(K3_ASSISTANT)]
    before = copy.deepcopy(history)
    _body(args=(), messages=history)
    assert history == before
    assert history[0]["tool_calls"] == K3_ASSISTANT["tool_calls"]


def _refusal(env) -> list[dict]:
    with patch.object(provider, "_open_stream",
                      side_effect=AssertionError("dispatched despite refusal")):
        return _events(env)


def test_required_tool_choice_on_k26_is_refused_before_dispatch():
    env = _env(provider="moonshot/kimi-k2.6", tools=[GET_TIME], tool_choice="required")
    events = _refusal(env)
    assert [e["type"] for e in events] == ["error", "completed"]
    assert events[0]["code"] == "invalid_tool_choice"
    msg = events[0]["message"]
    assert "kimi-k2.6" in msg and "required" in msg and "kimi-k3" in msg


def test_k26_required_fixture_is_the_vendor_error_the_refusal_quotes():
    """The captured HTTP 400 for kimi-k2.6 + tool_choice=required was shipped
    as a fixture nothing read (same review, concern 2). It now anchors the
    adapter's refusal: the message must quote the vendor's own sentence, so a
    vendor wording change re-captured into the fixture turns this red."""
    captured = (_FIXTURES / "k26_required.err").read_text(encoding="utf-8").strip()
    assert captured.startswith("HTTP 400:")
    vendor_sentence = json.loads(captured.split(":", 1)[1])["error"]["message"]
    assert vendor_sentence == "tool_choice 'required' is incompatible with thinking enabled"
    events = _refusal(_env(provider="moonshot/kimi-k2.6", tools=[GET_TIME], tool_choice="required"))
    assert vendor_sentence in events[0]["message"]


def test_required_tool_choice_on_k27_code_is_refused_before_dispatch():
    env = _env(provider="moonshot/kimi-k2.7-code", tools=[GET_TIME], tool_choice="required")
    assert _refusal(env)[0]["code"] == "invalid_tool_choice"


def test_malformed_tools_flag_is_refused_before_dispatch():
    events = _refusal(_env(tools="get_time"))
    assert events[0]["code"] == "invalid_tools"


def test_malformed_messages_flag_is_refused_before_dispatch():
    events = _refusal(_env(messages=[{"content": "no role here"}]))
    assert events[0]["code"] == "invalid_messages"


_ECHO_STREAM = [
    'data: {"id":"c","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}',
    "",
    'data: {"id":"c","choices":[{"index":0,"delta":{},"finish_reason":"stop",'
    '"usage":{"prompt_tokens":1,"completion_tokens":1}}]}',
    "",
    "data: [DONE]",
]


def test_trailing_tool_messages_are_echoed_as_tool_result_events():
    history = [{"role": "user", "content": PROMPT}, K3_ASSISTANT, TOOL_MSG]
    with patch.object(provider, "_open_stream", return_value=_FakeSSEResponse(_ECHO_STREAM)):
        events = _events(_env(args=(), messages=history, tools=[GET_TIME]))
    assert [e["type"] for e in events] == ["tool_result", "delta", "completed"]
    assert {k: events[0][k] for k in ("name", "output", "call_id")} == {
        "name": "get_time", "output": "2026-09-07T21:05:00Z", "call_id": "get_time_0"}


def test_tool_result_name_is_recovered_from_the_assistant_tool_calls():
    nameless = {k: v for k, v in TOOL_MSG.items() if k != "name"}
    history = [{"role": "user", "content": PROMPT}, K3_ASSISTANT, nameless]
    with patch.object(provider, "_open_stream", return_value=_FakeSSEResponse(_ECHO_STREAM)):
        events = _events(_env(args=(), messages=history, tools=[GET_TIME]))
    assert events[0]["type"] == "tool_result" and events[0]["name"] == "get_time"


def test_only_this_turns_tool_messages_are_echoed():
    """Older results (before a later assistant turn) belong to a previous
    envelope's stream; re-emitting them every turn would be noise."""
    history = [{"role": "user", "content": PROMPT}, K3_ASSISTANT, TOOL_MSG,
               {"role": "assistant", "content": "It is 21:05 UTC."},
               {"role": "user", "content": "thanks"}]
    with patch.object(provider, "_open_stream", return_value=_FakeSSEResponse(_ECHO_STREAM)):
        events = _events(_env(args=(), messages=history))
    assert [e["type"] for e in events] == ["delta", "completed"]


# --- 3. streaming: reasoning -> thinking, tool_calls -> tool_call, completed
#        carries the material the next turn must echo -------------------------
#
# Fixtures are the UNMODIFIED SSE bytes api.moonshot.ai returned on
# 2026-09-07 (tests/fixtures/happi_moonshot_kimi_k3/):
#   turn1.sse        kimi-k3, get_time tool, "What time is it? Use the tool."
#                    -> 34 reasoning tokens, no content, one tool call
#   turn2.sse        the assembled assistant message + a tool result echoed
#                    -> reasoning, then the answer text, finish "stop"
#   k3_required.sse  same prompt with tool_choice "required" -> reasoning,
#                    answer text AND a tool call under finish "tool_calls"

import io  # noqa: E402

from lib.hal.happi.events import emit_completed, emit_thinking  # noqa: E402
from lib.hal.happi.runtime import run as happi_run  # noqa: E402

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "happi_moonshot_kimi_k3"
TURN1_REASONING = ("The user explicitly asks to use the tool for current time. "
                   "Need call get_time. Then answer with time. Ensure not overdo. "
                   "Use UTC ISO from tool.")
TURN2_ANSWER = "It’s **2026-09-07 21:05:00 UTC**."
K3_CALL = {"id": "get_time_0", "type": "function",
           "function": {"name": "get_time", "arguments": "{}"}}


def _sse(name: str) -> list[str]:
    return (_FIXTURES / name).read_text(encoding="utf-8").splitlines()


def _replay(name: str, **flags) -> list[dict]:
    flags.setdefault("tools", [GET_TIME])
    with patch.object(provider, "_open_stream", return_value=_FakeSSEResponse(_sse(name))):
        return _events(_env(**flags))


def _texts(events, kind) -> str:
    return "".join(e["text"] for e in events if e["type"] == kind)


def test_turn1_replay_emits_thinking_then_one_tool_call_then_completed():
    events = _replay("turn1.sse")
    types = [e["type"] for e in events]
    assert types[0] == "thinking" and types[-2:] == ["tool_call", "completed"]
    assert "delta" not in types  # content was "" on this turn
    assert types.count("tool_call") == 1
    call = events[-2]
    assert {k: call[k] for k in ("name", "input", "call_id")} == {
        "name": "get_time", "input": {}, "call_id": "get_time_0"}
    assert _texts(events, "thinking") == TURN1_REASONING


def test_turn1_completed_carries_stop_reason_and_the_verbatim_assistant_message():
    done = _replay("turn1.sse")[-1]
    assert done["usage"] == {"in": 171, "out": 70}
    assert done["stop_reason"] == "tool_calls"
    assert done["message"] == {"role": "assistant", "content": "",
                               "reasoning_content": TURN1_REASONING,
                               "tool_calls": [K3_CALL]}


def test_turn2_replay_streams_reasoning_strictly_before_the_answer():
    history = [{"role": "user", "content": PROMPT}, K3_ASSISTANT, TOOL_MSG]
    events = _replay("turn2.sse", args=(), messages=history)
    types = [e["type"] for e in events]
    assert types[0] == "tool_result"  # this turn's result, echoed
    first_delta = types.index("delta")
    assert set(types[1:first_delta]) == {"thinking"}
    assert "thinking" not in types[first_delta:]  # vendor: reasoning always precedes content
    assert _texts(events, "delta") == TURN2_ANSWER
    done = events[-1]
    assert done["stop_reason"] == "stop" and done["usage"] == {"in": 281, "out": 41}
    assert done["message"] == {"role": "assistant", "content": TURN2_ANSWER,
                               "reasoning_content": "Need final answer concise. Mention UTC."}


def test_reasoning_never_leaks_into_delta_text():
    events = _replay("turn2.sse")
    reasoning, answer = _texts(events, "thinking"), _texts(events, "delta")
    # The two exact equalities ARE the no-leak proof; a trailing
    # `reasoning not in answer` was constant-true once both held (same review).
    assert reasoning == "Need final answer concise. Mention UTC."
    assert answer == TURN2_ANSWER


def test_required_replay_carries_both_answer_text_and_the_forced_call():
    events = _replay("k3_required.sse", tool_choice="required")
    types = [e["type"] for e in events]
    assert types.count("tool_call") == 1 and types[-2:] == ["tool_call", "completed"]
    assert types.index("delta") > types.index("thinking")
    assert types.index("tool_call") > max(i for i, t in enumerate(types) if t == "delta")
    done = events[-1]
    assert done["stop_reason"] == "tool_calls" and done["usage"] == {"in": 207, "out": 136}
    assert done["message"]["content"] == "The current UTC time is **2025-12-19T18:42:35Z**."
    assert done["message"]["tool_calls"] == [K3_CALL]


_PLAIN_STREAM = [
    'data: {"id":"c1","choices":[{"delta":{"content":"Hel"},"index":0}]}',
    "",
    'data: {"id":"c1","choices":[{"delta":{"content":"lo"},"index":0}]}',
    "",
    'data: {"id":"c1","choices":[{"delta":{},"finish_reason":"stop"}],'
    '"usage":{"prompt_tokens":5,"completion_tokens":2}}',
    "",
    "data: [DONE]",
]


def test_plain_text_completed_shape_is_unchanged_for_every_other_provider():
    """No reasoning, no tool calls: completed keeps exactly the pre-W1b keys."""
    with patch.object(provider, "_open_stream", return_value=_FakeSSEResponse(_PLAIN_STREAM)):
        events = _events(_env(provider="deepseek/deepseek-chat"))
    assert [e["type"] for e in events] == ["delta", "delta", "completed"]
    assert set(events[-1]) == {"v", "id", "type", "ts", "usage"}


def test_parallel_tool_call_fragments_accumulate_by_index():
    """Fragment shape per api_chat.md ChoiceDelta.tool_calls and the vendor
    guide ("we will use an additional field called index"): id + name in the
    first fragment, arguments concatenated across the rest."""
    calls: dict = {}
    for fragment in [
        {"index": 0, "id": "a", "type": "function", "function": {"name": "f", "arguments": ""}},
        {"index": 1, "id": "b", "type": "function", "function": {"name": "g", "arguments": ""}},
        {"index": 0, "function": {"arguments": '{"x":'}},
        {"index": 1, "function": {"arguments": "{}"}},
        {"index": 0, "function": {"arguments": "1}"}},
    ]:
        provider._merge_tool_call_fragment(calls, fragment)
    assert [(c["id"], c["name"], c["arguments"]) for _, c in sorted(calls.items())] == [
        ("a", "f", '{"x":1}'), ("b", "g", "{}")]


def test_truncated_arguments_survive_as_a_raw_string_and_no_tool_call_is_emitted():
    """finish_reason "length" (api_chat.md enum) with the arguments cut mid-JSON:
    no tool_call event (the model did not finish asking), but nothing is lost
    -- completed.message carries the partial string and stop_reason says why."""
    first = next(l for l in _sse("turn1.sse") if '"tool_calls"' in l and '"id":"get_time_0"' in l)
    lines = [first, "",
             'data: {"id":"x","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
             '"function":{"arguments":"{\\"city\\": \\"Bei"}}]},"finish_reason":null}]}', "",
             'data: {"id":"x","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}', "",
             "data: [DONE]"]
    with patch.object(provider, "_open_stream", return_value=_FakeSSEResponse(lines)):
        events = _events(_env(tools=[GET_TIME]))
    assert [e["type"] for e in events] == ["completed"]
    assert events[0]["stop_reason"] == "length"
    assert events[0]["message"]["tool_calls"][0]["function"]["arguments"] == '{"city": "Bei'
    assert provider._decode_arguments('{"city": "Bei') == '{"city": "Bei'
    assert provider._decode_arguments("") == {}


def test_emit_thinking_matches_the_spec_row_and_the_openapi_enum():
    evt = json.loads(emit_thinking("req-1", "Need"))
    assert evt["type"] == "thinking" and evt["text"] == "Need"
    assert set(evt) - _ENVELOPE_KEYS == _spec_required_fields("thinking") == {"text"}
    enum = re.search(r"type: \{enum: \[([^\]]+)\]\}", _HAPPI_MD.read_text(encoding="utf-8"))
    assert "thinking" in [t.strip() for t in enum.group(1).split(",")]


def test_emit_completed_extras_ride_beside_usage():
    evt = json.loads(emit_completed("req-1", usage={"in": 1, "out": 2},
                                    stop_reason="stop", message={"role": "assistant"}))
    assert evt["usage"] == {"in": 1, "out": 2}
    assert evt["stop_reason"] == "stop" and evt["message"] == {"role": "assistant"}
    assert "usage" not in json.loads(emit_completed("req-1"))


def test_runtime_passes_the_whole_loop_turn_through():
    envelope = json.dumps({
        "v": "happi/1.1", "id": "rt-loop", "cmd": "chat", "args": [PROMPT],
        "flags": {"provider": "moonshot/kimi-k3", "tools": [GET_TIME]},
        "auth": {"scheme": "apikey", "token": "sk-fake-key-for-tests-0123456789abcdef"}})
    out = io.StringIO()
    with patch.object(provider, "_open_stream", return_value=_FakeSSEResponse(_sse("turn1.sse"))):
        rc = happi_run(io.StringIO(envelope), out)
    types = [json.loads(l)["type"] for l in out.getvalue().splitlines()]
    assert rc == 0
    assert types[:2] == ["started", "thinking"] and types[-2:] == ["tool_call", "completed"]


# --- exogenous review of HAL#549 (HAL#551; seals 598f20477bfcde01, d2b6ab8b2b8794b5) ---

def test_completed_extras_cannot_overwrite_the_envelope_identity():
    """Concern A3: ``emit_completed(**extra)`` rode into ``emit_event(**payload)``
    after the envelope keys, so ``type="error"`` or ``id="X"`` in an extra
    silently rewrote the event's identity (measured before the fix:
    ``{"id":"X","type":"error",...}``). Reserved keys are now refused."""
    import pytest
    from lib.hal.happi.events import emit_event
    for key in ("v", "id", "type", "ts"):
        with pytest.raises(ValueError, match=key):
            emit_completed("req-1", usage={"in": 1, "out": 1}, **{key: "x"})
        with pytest.raises(ValueError, match=key):
            emit_event("req-1", "completed", **{key: "x"})
    # the documented extras still ride
    evt = json.loads(emit_completed("req-1", stop_reason="stop", message={"role": "assistant"}))
    assert evt["type"] == "completed" and evt["id"] == "req-1" and evt["stop_reason"] == "stop"


def test_malformed_tool_calls_in_history_are_refused_not_crashed():
    """Concern B2: ``_is_message_list`` checked only role, so a history message
    whose ``tool_calls`` was a string reached ``_echo_tool_results`` and raised
    an uncaught AttributeError ('str' object has no attribute 'get'), measured
    before the fix. The documented refusal is ``invalid_messages``."""
    for bad in ("oops", [{"id": "c1"}, "not-a-dict"], [{"id": "c1", "function": "not-a-dict"}]):
        history = [{"role": "user", "content": PROMPT},
                   {"role": "assistant", "content": "", "tool_calls": bad}, TOOL_MSG]
        events = _refusal(_env(tools=[GET_TIME], messages=history))
        assert [e["type"] for e in events] == ["error", "completed"], bad
        assert events[0]["code"] == "invalid_messages", bad
        assert "tool_calls" in events[0]["message"], bad


def test_an_empty_tools_list_is_not_sent():
    """Concern B1: the docstring said ``flags.tools`` is sent verbatim; an empty
    list was (and is) dropped. That is the intended behaviour -- OpenAI rejects
    ``tools: []`` with HTTP 400 ("Invalid 'tools': empty array. Expected an
    array with minimum length 1") -- so the test pins the decision and the
    docstring now states it, instead of claiming a verbatim pass-through."""
    assert "tools" not in _body(args=(), tools=[])
    assert _body(args=(), tools=[GET_TIME])["tools"] == [GET_TIME]
