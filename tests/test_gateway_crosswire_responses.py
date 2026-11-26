"""Anthropic Messages -> OpenAI Responses cross-wire (gpt-6 / Astra).

Why this exists: OpenAI answers a chat/completions request for gpt-6-astra
that carries function tools with HTTP 400 ("Function tools with
reasoning_effort are not supported for gpt-6-astra in /v1/chat/completions.
To use function tools, use /v1/responses ..."). Claude Code always sends its
tool definitions, so before this translator a Claude Code session pointed at
the HAL gateway with ``--model gpt-6-astra`` failed on its first message
(measured live 2026-09-10 through a private gateway: witness row provider
openai, status 400). The family whose tools must travel on /v1/responses is
named once, in ``model_quirks.tools_require_responses_api``; the forwarder
branches on that predicate and nowhere else.

Hermetic: opener and key resolver injected, zero network.
"""
from __future__ import annotations

import json
import unittest

from lib.hal.gateway_crosswire_responses import (
    anthropic_to_responses_request,
    responses_to_anthropic_response,
)
from lib.hal.gateway_egress import cross_wire_forward


class _FakeResp:
    def __init__(self, status=200, headers=None, body=b"{}"):
        self.status = status
        self.headers = headers or [("Content-Type", "application/json")]
        self._body = body
        self._read = False

    def read(self, size=-1):
        if self._read:
            return b""
        self._read = True
        return self._body

    def close(self):
        pass


def _capture_opener(resp):
    captured = {}

    def opener(method, url, headers, body):
        captured.update(method=method, url=url, headers=headers, body=body)
        return resp

    return opener, captured


_TOOL = {"name": "Read", "description": "Read a file",
         "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}

_RESPONSES_TEXT_OK = json.dumps({
    "id": "resp_1", "object": "response", "status": "completed", "model": "gpt-6-astra",
    "output": [
        {"type": "reasoning", "id": "rs_1", "summary": []},
        {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": "PONG", "annotations": []}]},
    ],
    "usage": {"input_tokens": 11, "output_tokens": 2, "total_tokens": 13},
}).encode("utf-8")


class TestRequestTranslation(unittest.TestCase):

    def test_system_user_and_limits_map_to_responses_fields(self):
        out = anthropic_to_responses_request({
            "model": "gpt-6-astra", "system": "be terse", "max_tokens": 64,
            "stream": True, "messages": [{"role": "user", "content": "hi"}],
        })
        self.assertEqual(out["model"], "gpt-6-astra")
        self.assertEqual(out["instructions"], "be terse")
        self.assertEqual(out["max_output_tokens"], 64)
        self.assertEqual(out["input"], [
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}])
        self.assertIs(out["stream"], False)   # buffered first version: never stream upstream
        self.assertIs(out["store"], False)    # the gateway keeps custody; OpenAI stores nothing
        self.assertNotIn("messages", out)
        self.assertNotIn("max_tokens", out)

    def test_tools_are_flat_function_tools_and_tool_choice_maps(self):
        out = anthropic_to_responses_request({
            "model": "gpt-6-astra", "messages": [{"role": "user", "content": "hi"}],
            "tools": [_TOOL], "tool_choice": {"type": "tool", "name": "Read"},
        })
        self.assertEqual(out["tools"], [{
            "type": "function", "name": "Read", "description": "Read a file",
            "parameters": _TOOL["input_schema"]}])
        self.assertEqual(out["tool_choice"], {"type": "function", "name": "Read"})
        any_choice = anthropic_to_responses_request({
            "model": "gpt-6-astra", "messages": [], "tools": [_TOOL],
            "tool_choice": {"type": "any"}})
        self.assertEqual(any_choice["tool_choice"], "required")

    def test_tool_history_becomes_function_call_and_output_items(self):
        out = anthropic_to_responses_request({
            "model": "gpt-6-astra",
            "messages": [
                {"role": "user", "content": "read it"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Reading."},
                    {"type": "tool_use", "id": "toolu_1", "name": "Read",
                     "input": {"path": "/a"}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1",
                     "content": [{"type": "text", "text": "file body"}]}]},
            ],
        })
        self.assertEqual(out["input"], [
            {"role": "user", "content": [{"type": "input_text", "text": "read it"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "Reading."}]},
            {"type": "function_call", "call_id": "toolu_1", "name": "Read",
             "arguments": json.dumps({"path": "/a"})},
            {"type": "function_call_output", "call_id": "toolu_1", "output": "file body"},
        ])

    def test_effort_is_sent_only_when_the_model_documents_the_level(self):
        base = {"model": "gpt-6-astra", "messages": []}
        high = anthropic_to_responses_request({**base, "output_config": {"effort": "high"}})
        self.assertEqual(high["reasoning"], {"effort": "high"})
        none = anthropic_to_responses_request({**base, "output_config": {"effort": "none"}})
        self.assertNotIn("reasoning", none)   # Astra refuses "none": omit, vendor default applies
        absent = anthropic_to_responses_request(base)
        self.assertNotIn("reasoning", absent)

    def test_base64_image_becomes_input_image(self):
        out = anthropic_to_responses_request({
            "model": "gpt-6-astra",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": "AAAA"}}]}],
        })
        self.assertEqual(out["input"], [{"role": "user", "content": [
            {"type": "input_text", "text": "what is this"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}]}])


class TestResponseTranslation(unittest.TestCase):

    def test_output_text_becomes_a_text_block(self):
        msg = responses_to_anthropic_response(json.loads(_RESPONSES_TEXT_OK), model="x")
        self.assertEqual(msg["type"], "message")
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["id"], "resp_1")
        self.assertEqual(msg["model"], "gpt-6-astra")
        self.assertEqual(msg["content"], [{"type": "text", "text": "PONG"}])
        self.assertEqual(msg["stop_reason"], "end_turn")
        self.assertEqual(msg["usage"], {"input_tokens": 11, "output_tokens": 2})

    def test_function_call_becomes_tool_use_with_parsed_input(self):
        msg = responses_to_anthropic_response({
            "id": "resp_2", "status": "completed", "model": "gpt-6-astra",
            "output": [{"type": "function_call", "id": "fc_1", "call_id": "call_9",
                        "name": "Read", "arguments": "{\"path\": \"/a\"}"}],
            "usage": {"input_tokens": 3, "output_tokens": 4},
        }, model="gpt-6-astra")
        self.assertEqual(msg["content"], [
            {"type": "tool_use", "id": "call_9", "name": "Read", "input": {"path": "/a"}}])
        self.assertEqual(msg["stop_reason"], "tool_use")

    def test_incomplete_on_max_output_tokens_maps_to_max_tokens(self):
        msg = responses_to_anthropic_response({
            "id": "r", "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": "trunc"}]}],
            "usage": {},
        }, model="gpt-6-astra")
        self.assertEqual(msg["stop_reason"], "max_tokens")
        self.assertEqual(msg["usage"], {"input_tokens": 0, "output_tokens": 0})

    def test_malformed_arguments_drop_the_block_instead_of_faking_empty_input(self):
        """Inverted 2026-09-10. This test previously ASSERTED the defect.

        It required `input == {}` for an unparseable argument string, which is
        the fabrication itself: the client receives a syntactically valid call
        and runs the tool with arguments nobody wrote. The original intent --
        malformed arguments must not raise -- is kept; the expectation is
        corrected to "no block is emitted".
        """
        msg = responses_to_anthropic_response({
            "id": "r", "output": [{"type": "function_call", "call_id": "c",
                                   "name": "Read", "arguments": "{not json"}],
        }, model="m")
        self.assertEqual(msg["content"], [])
        self.assertEqual(msg["stop_reason"], "max_tokens")

    def test_truncated_call_reports_max_tokens_not_tool_use(self):
        """C6, measured on merged main 2026-09-10 before the fix.

        A response that is `incomplete` on `max_output_tokens` and carries a
        `function_call` whose arguments end mid-token returned
        `stop_reason: tool_use` with `input: {}` -- so a truncated
        `Bash{command: "rm -r ..."}` reached the client as `Bash{}`, a call it
        would execute. Truncation is checked first now, and the partial block
        is not emitted at all.
        """
        msg = responses_to_anthropic_response({
            "id": "r", "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{"type": "function_call", "call_id": "c", "name": "Bash",
                        "arguments": '{"command": "rm -r'}],
            "usage": {"input_tokens": 9, "output_tokens": 4},
        }, model="gpt-6-astra")
        self.assertEqual(msg["stop_reason"], "max_tokens")
        self.assertEqual([b for b in msg["content"] if b["type"] == "tool_use"], [])

    def test_truncation_outranks_a_complete_call_in_the_same_response(self):
        """A complete call is kept, but the turn is still reported as cut off.

        The model did not choose to stop and hand over; it ran out of room, and
        more calls may have been coming. `tool_use` would tell the client the
        turn is finished.
        """
        msg = responses_to_anthropic_response({
            "id": "r", "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{"type": "function_call", "call_id": "ok", "name": "Read",
                        "arguments": '{"path": "/a"}'}],
            "usage": {},
        }, model="gpt-6-astra")
        self.assertEqual(msg["stop_reason"], "max_tokens")
        self.assertEqual(msg["content"],
                         [{"type": "tool_use", "id": "ok", "name": "Read",
                           "input": {"path": "/a"}}])

    def test_a_genuinely_zero_argument_call_still_survives(self):
        """The carve-out that keeps the fix from over-reaching.

        `arguments` absent, empty, or `"{}"` is a real zero-argument call and
        must still be emitted -- `_safe_json_obj` renders all three as `{}`,
        and only `_arguments_are_corrupt` tells them apart from a cut-off one.
        """
        for raw in (None, "", "{}", {}):
            msg = responses_to_anthropic_response({
                "id": "r", "status": "completed",
                "output": [{"type": "function_call", "call_id": "c",
                            "name": "ListDir", "arguments": raw}],
            }, model="m")
            self.assertEqual(msg["stop_reason"], "tool_use", raw)
            self.assertEqual(msg["content"],
                             [{"type": "tool_use", "id": "c", "name": "ListDir",
                               "input": {}}], raw)


class TestForwarderBranch(unittest.TestCase):

    def test_gpt6_with_tools_posts_a_responses_body_to_v1_responses(self):
        opener, captured = _capture_opener(_FakeResp(body=_RESPONSES_TEXT_OK))
        resp = cross_wire_forward(
            {"model": "gpt-6-astra", "system": "be terse", "max_tokens": 32, "stream": True,
             "messages": [{"role": "user", "content": "ping"}], "tools": [_TOOL]},
            provider="openai", want_stream=False,
            opener=opener, key_resolver=lambda s: "sk-PROVIDER")
        self.assertEqual(captured["url"], "https://api.openai.com/v1/responses")
        self.assertEqual(dict(captured["headers"])["Authorization"], "Bearer sk-PROVIDER")
        sent = json.loads(captured["body"])
        self.assertNotIn("messages", sent)
        self.assertEqual(sent["instructions"], "be terse")
        self.assertEqual(sent["tools"][0]["name"], "Read")
        self.assertNotIn("function", sent["tools"][0])
        self.assertIs(sent["stream"], False)
        self.assertNotIn("reasoning_effort", sent)  # the chat-only knob never rides this wire
        self.assertEqual(resp.status, 200)
        msg = json.loads(resp.read_all())
        self.assertEqual(msg["content"], [{"type": "text", "text": "PONG"}])

    def test_gpt6_streaming_client_gets_one_anthropic_sse_burst(self):
        opener, _ = _capture_opener(_FakeResp(body=_RESPONSES_TEXT_OK))
        resp = cross_wire_forward(
            {"model": "openai/gpt-6-astra", "stream": True,
             "messages": [{"role": "user", "content": "ping"}]},
            provider="openai", want_stream=True,
            opener=opener, key_resolver=lambda s: "k")
        self.assertEqual(resp.status, 200)
        self.assertEqual(dict(resp.headers)["Content-Type"], "text/event-stream")
        stream = resp.read_all().decode("utf-8")
        self.assertIn("event: message_start", stream)
        self.assertIn("PONG", stream)
        self.assertIn("event: message_stop", stream)

    def test_other_openai_models_still_use_chat_completions(self):
        chat_ok = json.dumps({"id": "c", "model": "gpt-4o", "choices": [
            {"index": 0, "finish_reason": "stop",
             "message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode("utf-8")
        opener, captured = _capture_opener(_FakeResp(body=chat_ok))
        cross_wire_forward(
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            provider="openai", want_stream=False, opener=opener, key_resolver=lambda s: "k")
        self.assertEqual(captured["url"], "https://api.openai.com/v1/chat/completions")

    def test_provider_error_is_relayed_as_anthropic_envelope_with_real_status(self):
        body = json.dumps({"error": {"message": "nope", "type": "invalid_request_error"}}).encode()
        opener, _ = _capture_opener(_FakeResp(status=400, body=body))
        resp = cross_wire_forward(
            {"model": "gpt-6-astra", "messages": [], "stream": True},
            provider="openai", want_stream=True, opener=opener, key_resolver=lambda s: "k")
        self.assertEqual(resp.status, 400)
        env = json.loads(resp.read_all())
        self.assertEqual(env["type"], "error")
        self.assertIn("nope", json.dumps(env))




class TestMalformedToolCallIsSurvivable(unittest.TestCase):
    """Forward anchor for an AGORA concern on PR #574.

    A council speaker observed that ``_assistant_items`` guards ``call`` being
    a dict but not ``call["function"]``, so a non-dict ``function`` would raise
    AttributeError. Tested against the code: NOT reachable today, because the
    only producer of this structure is ``gateway_crosswire`` line 95, which
    builds ``function`` as a dict literal, and ``_chat_message_to_items`` is
    the sole caller. The guard is added anyway so that a future producer, or a
    caller wired straight from client bytes, degrades to an empty name rather
    than 500-ing a live request.
    """

    def test_a_non_dict_function_does_not_raise(self):
        from lib.hal.gateway_crosswire_responses import _chat_message_to_items
        for bad in ("not-a-dict", ["a"], 7, None):
            msg = {"role": "assistant", "content": None,
                   "tool_calls": [{"id": "call_1", "function": bad}]}
            items = _chat_message_to_items(msg)
            self.assertEqual(len(items), 1, bad)
            self.assertEqual(items[0]["type"], "function_call")
            self.assertEqual(items[0]["call_id"], "call_1")
            self.assertEqual(items[0]["name"], "")

    def test_a_well_formed_tool_call_is_unchanged(self):
        from lib.hal.gateway_crosswire_responses import _chat_message_to_items
        msg = {"role": "assistant", "content": None,
               "tool_calls": [{"id": "call_2",
                               "function": {"name": "read", "arguments": '{"p":1}'}}]}
        (item,) = _chat_message_to_items(msg)
        self.assertEqual(item["name"], "read")
        self.assertEqual(item["arguments"], '{"p":1}')

if __name__ == "__main__":
    unittest.main()
