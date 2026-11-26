"""Tests for cross-wire Anthropic->OpenAI request translation (Phase 0 wave 4).

Goodhart-proof: asserts the actual field mappings (system, tool_use->tool_calls,
tool_result->tool message, image, tool_choice) AND round-trips against the
forward converter (gateway_openai.openai_to_anthropic_request) so the two
directions cannot silently diverge.
"""
import json
import unittest

from lib.hal.gateway_crosswire import (
    anthropic_to_openai_request,
    openai_to_anthropic_response,
)
from lib.hal.gateway_openai import openai_to_anthropic_request


class TestBasics(unittest.TestCase):

    def test_model_passthrough_not_aliased(self):
        out = anthropic_to_openai_request({"model": "gpt-5.5", "messages": []})
        self.assertEqual(out["model"], "gpt-5.5")  # NOT resolved to claude

    def test_system_becomes_system_message(self):
        out = anthropic_to_openai_request({
            "model": "gpt-5.5", "system": "be terse",
            "messages": [{"role": "user", "content": "hi"}],
        })
        self.assertEqual(out["messages"][0], {"role": "system", "content": "be terse"})
        self.assertEqual(out["messages"][1], {"role": "user", "content": "hi"})

    def test_system_block_list(self):
        out = anthropic_to_openai_request({
            "model": "gpt-5.5",
            "system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            "messages": [],
        })
        self.assertEqual(out["messages"][0]["content"], "a\n\nb")

    def test_sampling_and_stop(self):
        out = anthropic_to_openai_request({
            "model": "gpt-5.5", "messages": [], "max_tokens": 256,
            "temperature": 0.3, "stop_sequences": ["X", "Y"],
        })
        self.assertEqual(out["max_tokens"], 256)
        self.assertEqual(out["temperature"], 0.3)
        self.assertEqual(out["stop"], ["X", "Y"])


class TestToolRoundTrip(unittest.TestCase):
    """tool_use -> tool_calls and tool_result -> tool message (the hard mapping)."""

    def test_assistant_tool_use_to_tool_calls(self):
        out = anthropic_to_openai_request({
            "model": "gpt-5.5",
            "messages": [{"role": "assistant", "content": [
                {"type": "text", "text": "calling"},
                {"type": "tool_use", "id": "tu_1", "name": "search",
                 "input": {"q": "cats"}},
            ]}],
        })
        msg = out["messages"][0]
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["content"], "calling")
        call = msg["tool_calls"][0]
        self.assertEqual(call["id"], "tu_1")
        self.assertEqual(call["type"], "function")
        self.assertEqual(call["function"]["name"], "search")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"q": "cats"})

    def test_tool_result_to_tool_message(self):
        out = anthropic_to_openai_request({
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1",
                 "content": [{"type": "text", "text": "42 results"}]},
            ]}],
        })
        msg = out["messages"][0]
        self.assertEqual(msg["role"], "tool")
        self.assertEqual(msg["tool_call_id"], "tu_1")
        self.assertEqual(msg["content"], "42 results")

    def test_tools_input_schema_to_parameters(self):
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        out = anthropic_to_openai_request({
            "model": "gpt-5.5", "messages": [],
            "tools": [{"name": "search", "description": "find", "input_schema": schema}],
            "tool_choice": {"type": "tool", "name": "search"},
        })
        tool = out["tools"][0]
        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["function"]["name"], "search")
        self.assertEqual(tool["function"]["parameters"], schema)
        self.assertEqual(out["tool_choice"], {"type": "function", "function": {"name": "search"}})

    def test_tool_choice_any_to_required(self):
        out = anthropic_to_openai_request({
            "model": "gpt-5.5", "messages": [],
            "tools": [{"name": "t", "input_schema": {}}],
            "tool_choice": {"type": "any"},
        })
        self.assertEqual(out["tool_choice"], "required")


class TestImage(unittest.TestCase):

    def test_base64_image_to_data_uri(self):
        out = anthropic_to_openai_request({
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "image", "source": {"type": "base64",
                                             "media_type": "image/jpeg", "data": "QUJD"}},
            ]}],
        })
        parts = out["messages"][0]["content"]
        self.assertEqual(parts[0], {"type": "text", "text": "look"})
        self.assertEqual(parts[1]["type"], "image_url")
        self.assertEqual(parts[1]["image_url"]["url"], "data:image/jpeg;base64,QUJD")


class TestForwardRoundTrip(unittest.TestCase):
    """A request that started as OpenAI, forward-translated to Anthropic, must
    cross-translate back with its load-bearing fields intact."""

    def test_openai_to_anthropic_to_openai_preserves_core(self):
        original = {
            "model": "gpt-5.5",
            "messages": [
                {"role": "system", "content": "be helpful"},
                {"role": "user", "content": "hello"},
            ],
            "tools": [{"type": "function", "function": {
                "name": "lookup", "description": "d",
                "parameters": {"type": "object", "properties": {}}}}],
            "max_tokens": 100,
        }
        anthropic = openai_to_anthropic_request(original)
        # forward converter resolves the model alias; restore the real id as the
        # gateway would (route_model keeps gpt-5.5) so the cross-translate is faithful
        anthropic["model"] = "gpt-5.5"
        back = anthropic_to_openai_request(anthropic)
        self.assertEqual(back["model"], "gpt-5.5")
        self.assertEqual(back["messages"][0], {"role": "system", "content": "be helpful"})
        self.assertEqual(back["messages"][1], {"role": "user", "content": "hello"})
        self.assertEqual(back["tools"][0]["function"]["name"], "lookup")


class TestResponseTranslation(unittest.TestCase):
    """OpenAI ChatCompletion response -> Anthropic Messages response."""

    def test_text_response(self):
        out = openai_to_anthropic_response({
            "id": "cc_1", "model": "gpt-5.5",
            "choices": [{"message": {"role": "assistant", "content": "hello there"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        })
        self.assertEqual(out["type"], "message")
        self.assertEqual(out["role"], "assistant")
        self.assertEqual(out["content"], [{"type": "text", "text": "hello there"}])
        self.assertEqual(out["stop_reason"], "end_turn")
        self.assertEqual(out["usage"], {"input_tokens": 5, "output_tokens": 2})

    def test_tool_calls_to_tool_use(self):
        out = openai_to_anthropic_response({
            "id": "cc_2", "model": "gpt-5.5",
            "choices": [{"message": {"role": "assistant", "content": None,
                "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": "search", "arguments": '{"q":"cats"}'}}]},
                "finish_reason": "tool_calls"}],
        })
        block = out["content"][0]
        self.assertEqual(block["type"], "tool_use")
        self.assertEqual(block["id"], "call_1")
        self.assertEqual(block["name"], "search")
        self.assertEqual(block["input"], {"q": "cats"})
        self.assertEqual(out["stop_reason"], "tool_use")

    def test_finish_reason_length_maps_to_max_tokens(self):
        out = openai_to_anthropic_response({
            "choices": [{"message": {"content": "x"}, "finish_reason": "length"}]})
        self.assertEqual(out["stop_reason"], "max_tokens")

    def test_empty_choices_safe(self):
        out = openai_to_anthropic_response({"choices": []})
        self.assertEqual(out["content"], [])
        self.assertEqual(out["stop_reason"], "end_turn")


class TestTruncatedToolCallOnTheChatWire(unittest.TestCase):
    """The chat wire shares the fabricated-empty-arguments half of C6.

    It gets `stop_reason` right for free -- the provider sends `finish_reason:
    length` -- but it still rendered a cut-off argument string as `input: {}`
    and emitted the block, handing the client a call to run with arguments
    nobody wrote. Same drop as the Responses wire, from the same predicate.
    """

    @staticmethod
    def _reply(arguments, finish="tool_calls"):
        from lib.hal.gateway_crosswire import openai_to_anthropic_response
        return openai_to_anthropic_response({
            "id": "x", "model": "kimi", "choices": [{
                "finish_reason": finish,
                "message": {"content": None, "tool_calls": [
                    {"id": "c1", "function": {"name": "Bash", "arguments": arguments}}]},
            }], "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        })

    def test_truncated_arguments_drop_the_block_and_force_max_tokens(self):
        msg = self._reply('{"command": "rm -r')
        self.assertEqual(msg["content"], [])
        self.assertEqual(msg["stop_reason"], "max_tokens")

    def test_a_zero_argument_call_is_still_emitted(self):
        for raw in (None, "", "{}"):
            msg = self._reply(raw)
            self.assertEqual(msg["stop_reason"], "tool_use", raw)
            self.assertEqual(msg["content"], [
                {"type": "tool_use", "id": "c1", "name": "Bash", "input": {}}], raw)

    def test_a_complete_call_is_untouched(self):
        msg = self._reply('{"command": "ls"}')
        self.assertEqual(msg["stop_reason"], "tool_use")
        self.assertEqual(msg["content"], [
            {"type": "tool_use", "id": "c1", "name": "Bash", "input": {"command": "ls"}}])


class TestArgumentsAreCorrupt(unittest.TestCase):
    """The predicate the whole fix rests on: absent/empty is NOT corrupt."""

    def test_absent_or_empty_or_object_is_not_corrupt(self):
        from lib.hal.gateway_crosswire import _arguments_are_corrupt
        for raw in (None, "", "{}", '{"a": 1}', {}, {"a": 1}):
            self.assertFalse(_arguments_are_corrupt(raw), raw)

    def test_cut_off_or_non_object_is_corrupt(self):
        from lib.hal.gateway_crosswire import _arguments_are_corrupt
        for raw in ('{"command": "rm -r', "{not json", "[1,2]", '"a string"', "7", ["a"], 7):
            self.assertTrue(_arguments_are_corrupt(raw), raw)


if __name__ == "__main__":
    unittest.main()
