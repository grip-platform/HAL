"""Fixture-driven tests for _normalize_openai_to_anthropic (W2 + strict-errors).

The normaliser is the single translation seam between OpenAI-compat
responses and the Anthropic-shape dict run_tool_loop consumes. Covers:

- tool-call only response (finish_reason=tool_calls -> stop_reason=tool_use)
- mixed text + tool_call (rare but valid on some providers)
- malformed function.arguments JSON (surface error block, don't crash)
- length finish_reason maps to max_tokens
- unknown finish_reason maps to end_turn (conservative)

Strict-errors follow-up (PR #173, post-HAL-Council review 2026-04-17)
reverses the earlier "safe empty defaults" posture for structurally
invalid inputs. Silent degradation masked provider errors as legitimate
end-of-turn, which drove run_tool_loop into infinite retries. The
normaliser now raises NormalizerError on non-dict input, surfaced-in-body
provider errors, missing/empty choices, non-dict first choice, and
first choice with neither message nor finish_reason. A legitimately
empty response (model chose to say nothing with finish_reason set) is
NOT an error and returns empty content.

These tests import the private normaliser directly because it is the
translation contract — a change in its behaviour is a change in what
the tool loop sees, and the loop must be protected by an explicit
fixture suite rather than inferred from the chat_fn wrapper tests.
"""
import os
import sys
import unittest

_HAL_HOME = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
if _HAL_HOME not in sys.path:
    sys.path.insert(0, _HAL_HOME)

from lib.hal.tool_runtime import (  # noqa: E402
    NormalizerError,
    _EMPTY_MESSAGE,
    _extract_message_dict,
    _normalize_openai_to_anthropic,
    make_openai_compat_chat_fn,
)


def _tool_call(call_id: str, name: str, arguments: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _make_response(
    *,
    content=None,
    tool_calls=None,
    finish_reason="stop",
    usage=None,
    model="m",
    resp_id="chatcmpl-1",
) -> dict:
    message: dict = {"role": "assistant"}
    if content is not None:
        message["content"] = content
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": resp_id,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0},
    }


class TestTextOnly(unittest.TestCase):
    def test_single_text_block_preserved(self):
        result = _normalize_openai_to_anthropic(_make_response(content="hello"))
        self.assertEqual(result["stop_reason"], "end_turn")
        self.assertEqual(len(result["content"]), 1)
        self.assertEqual(result["content"][0], {"type": "text", "text": "hello"})

    def test_empty_text_omitted(self):
        result = _normalize_openai_to_anthropic(_make_response(content=""))
        self.assertEqual(result["content"], [])

    def test_null_content_omitted(self):
        result = _normalize_openai_to_anthropic(_make_response(content=None))
        self.assertEqual([b for b in result["content"] if b["type"] == "text"], [])


class TestToolCallOnly(unittest.TestCase):
    def test_tool_call_becomes_anthropic_tool_use(self):
        result = _normalize_openai_to_anthropic(_make_response(
            content=None,
            tool_calls=[_tool_call("call_1", "Read", '{"path": "/tmp/x"}')],
            finish_reason="tool_calls",
        ))
        self.assertEqual(result["stop_reason"], "tool_use")
        self.assertEqual(len(result["content"]), 1)
        self.assertEqual(result["content"][0], {
            "type": "tool_use",
            "id": "call_1",
            "name": "Read",
            "input": {"path": "/tmp/x"},
        })

    def test_multiple_tool_calls_produce_multiple_blocks(self):
        result = _normalize_openai_to_anthropic(_make_response(
            tool_calls=[
                _tool_call("c1", "Read", '{"path": "a"}'),
                _tool_call("c2", "Write", '{"path": "b", "content": "x"}'),
            ],
            finish_reason="tool_calls",
        ))
        self.assertEqual(len(result["content"]), 2)
        self.assertEqual(result["content"][0]["name"], "Read")
        self.assertEqual(result["content"][1]["name"], "Write")
        self.assertEqual(result["content"][1]["input"]["content"], "x")


class TestMixedTextAndToolCall(unittest.TestCase):
    def test_text_and_tool_call_both_preserved(self):
        result = _normalize_openai_to_anthropic(_make_response(
            content="I'll read the file first.",
            tool_calls=[_tool_call("c1", "Read", '{"path": "/x"}')],
            finish_reason="tool_calls",
        ))
        self.assertEqual(len(result["content"]), 2)
        self.assertEqual(result["content"][0]["type"], "text")
        self.assertEqual(result["content"][1]["type"], "tool_use")
        self.assertEqual(result["stop_reason"], "tool_use")


class TestMalformedArguments(unittest.TestCase):
    def test_malformed_json_surfaces_as_parse_error_block(self):
        result = _normalize_openai_to_anthropic(_make_response(
            tool_calls=[_tool_call("c1", "Read", '{not json')],
            finish_reason="tool_calls",
        ))
        block = result["content"][0]
        self.assertEqual(block["type"], "tool_use")
        self.assertEqual(block["input"], {})
        self.assertIn("_parse_error", block)
        self.assertIn("JSONDecodeError", block["_parse_error"])

    def test_empty_arguments_string_produces_empty_input(self):
        result = _normalize_openai_to_anthropic(_make_response(
            tool_calls=[_tool_call("c1", "Noop", "")],
            finish_reason="tool_calls",
        ))
        self.assertEqual(result["content"][0]["input"], {})
        self.assertNotIn("_parse_error", result["content"][0])

    def test_arguments_already_dict_passed_through(self):
        result = _normalize_openai_to_anthropic(_make_response(
            tool_calls=[{
                "id": "c1", "type": "function",
                "function": {"name": "Read", "arguments": {"path": "/x"}},
            }],
            finish_reason="tool_calls",
        ))
        self.assertEqual(result["content"][0]["input"], {"path": "/x"})

    def test_non_dict_tool_call_entry_dropped(self):
        result = _normalize_openai_to_anthropic(_make_response(
            tool_calls=[
                "not a dict",
                _tool_call("c1", "Read", '{"path": "/x"}'),
            ],
            finish_reason="tool_calls",
        ))
        self.assertEqual(len(result["content"]), 1)
        self.assertEqual(result["content"][0]["name"], "Read")


class TestFinishReasonMapping(unittest.TestCase):
    def test_stop_maps_to_end_turn(self):
        result = _normalize_openai_to_anthropic(_make_response(
            content="done", finish_reason="stop",
        ))
        self.assertEqual(result["stop_reason"], "end_turn")

    def test_tool_calls_maps_to_tool_use(self):
        result = _normalize_openai_to_anthropic(_make_response(
            tool_calls=[_tool_call("c1", "N", "{}")], finish_reason="tool_calls",
        ))
        self.assertEqual(result["stop_reason"], "tool_use")

    def test_length_maps_to_max_tokens(self):
        result = _normalize_openai_to_anthropic(_make_response(
            content="truncated", finish_reason="length",
        ))
        self.assertEqual(result["stop_reason"], "max_tokens")

    def test_unknown_finish_reason_defaults_to_end_turn(self):
        result = _normalize_openai_to_anthropic(_make_response(
            content="x", finish_reason="content_filter",
        ))
        self.assertEqual(result["stop_reason"], "end_turn")

    def test_null_finish_reason_defaults_to_end_turn(self):
        result = _normalize_openai_to_anthropic(_make_response(
            content="x", finish_reason=None,
        ))
        self.assertEqual(result["stop_reason"], "end_turn")


class TestUsageMapping(unittest.TestCase):
    def test_prompt_and_completion_tokens_rename(self):
        result = _normalize_openai_to_anthropic(_make_response(
            content="x",
            usage={"prompt_tokens": 42, "completion_tokens": 13},
        ))
        self.assertEqual(result["usage"]["input_tokens"], 42)
        self.assertEqual(result["usage"]["output_tokens"], 13)

    def test_missing_usage_defaults_to_zero(self):
        result = _normalize_openai_to_anthropic({
            "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
        })
        self.assertEqual(result["usage"], {"input_tokens": 0, "output_tokens": 0})


class TestMetadataPreservation(unittest.TestCase):
    def test_id_and_model_preserved(self):
        result = _normalize_openai_to_anthropic(_make_response(
            content="x", model="gemma4:e4b", resp_id="chatcmpl-42",
        ))
        self.assertEqual(result["id"], "chatcmpl-42")
        self.assertEqual(result["model"], "gemma4:e4b")


class TestStrictErrors(unittest.TestCase):
    """Normaliser RAISES on structurally invalid inputs (post-Council fix 2026-04-17).

    Previously the normaliser silently degraded to
    ``{"content": [], "stop_reason": "end_turn"}`` on malformed inputs. HAL
    Council identified this as a tool-loop footgun: silent degradation masks
    real errors (rate limits, provider errors, malformed payloads) as
    legitimate end-of-turn, driving run_tool_loop into infinite retries and
    silent context drops. The strict posture propagates the error to the
    caller so the loop halts cleanly.

    Each failure mode is its own named test so a regression can't be
    papered over with a catch-all.
    """

    def test_none_input_raises(self):
        with self.assertRaises(NormalizerError) as ctx:
            _normalize_openai_to_anthropic(None)
        self.assertIn("NoneType", str(ctx.exception))

    def test_string_input_raises(self):
        with self.assertRaises(NormalizerError) as ctx:
            _normalize_openai_to_anthropic("not a dict")
        self.assertIn("str", str(ctx.exception))

    def test_empty_dict_raises_missing_choices(self):
        with self.assertRaises(NormalizerError) as ctx:
            _normalize_openai_to_anthropic({})
        self.assertIn("choices", str(ctx.exception))

    def test_missing_choices_raises(self):
        with self.assertRaises(NormalizerError):
            _normalize_openai_to_anthropic({"id": "x", "model": "m"})

    def test_empty_choices_list_raises(self):
        with self.assertRaises(NormalizerError):
            _normalize_openai_to_anthropic({"choices": []})

    def test_non_dict_first_choice_raises(self):
        with self.assertRaises(NormalizerError) as ctx:
            _normalize_openai_to_anthropic({"choices": ["junk"]})
        # Error names the offending type so the caller can debug
        self.assertIn("str", str(ctx.exception))

    def test_provider_error_dict_raises_with_message(self):
        # Some providers return HTTP 200 with an error in the body
        bad = {"error": {"message": "rate_limit_exceeded", "type": "rate_limit"}}
        with self.assertRaises(NormalizerError) as ctx:
            _normalize_openai_to_anthropic(bad)
        self.assertIn("rate_limit_exceeded", str(ctx.exception))

    def test_provider_error_string_raises(self):
        with self.assertRaises(NormalizerError) as ctx:
            _normalize_openai_to_anthropic({"error": "something went wrong"})
        self.assertIn("something went wrong", str(ctx.exception))

    def test_choice_without_message_or_finish_raises(self):
        bad = {"choices": [{"index": 0}]}
        with self.assertRaises(NormalizerError) as ctx:
            _normalize_openai_to_anthropic(bad)
        self.assertIn("finish_reason", str(ctx.exception))

    def test_provider_error_with_choices_does_not_raise(self):
        # If choices is present, trust them — error key may be non-fatal noise
        ok = {
            "error": {"message": "nonfatal"},
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        }
        result = _normalize_openai_to_anthropic(ok)
        self.assertEqual(result["content"][0]["text"], "hi")


class TestLegitimateEmptyResponse(unittest.TestCase):
    """Valid-but-empty paths that must NOT raise (model chose silence).

    A response where finish_reason is set but the message is missing or
    has null content is a legitimate "model said nothing" turn, distinct
    from malformed input. run_tool_loop handles this as an ordinary
    terminating turn.
    """

    def test_empty_message_with_finish_reason_is_ok(self):
        result = _normalize_openai_to_anthropic(
            {"choices": [{"finish_reason": "stop"}]},
        )
        self.assertEqual(result["content"], [])
        self.assertEqual(result["stop_reason"], "end_turn")

    def test_null_content_with_finish_reason_is_ok(self):
        result = _normalize_openai_to_anthropic(
            {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]},
        )
        self.assertEqual(result["content"], [])

    def test_empty_message_with_length_finish_maps_to_max_tokens(self):
        result = _normalize_openai_to_anthropic(
            {"choices": [{"finish_reason": "length"}]},
        )
        self.assertEqual(result["stop_reason"], "max_tokens")


class TestW2CouncilRefinements(unittest.TestCase):
    """PR #173 council re-review refinements (#1236).

    W2a — NormalizerError base widened from RuntimeError to ValueError so
    Python-convention catchers (``except ValueError``) pick it up and the
    exception type matches its semantic role (structural input validation
    failure, not a runtime code-path fault).

    W2b — docstring ``Notes for callers`` paragraph on the public
    normalisation seams explains NormalizerError's contract and gives a
    minimal circuit-breaker sketch so downstream code doesn't have to
    reverse-engineer the exception hierarchy.

    W2c — module-level ``_EMPTY_MESSAGE`` sentinel replaces the fresh
    ``{}`` literal in the "model said nothing" path so the silent-turn
    branch is visibly identifiable in a debugger.

    Each sub-change has its own test so a regression on any one of the
    three cannot be masked by the others.
    """

    def test_w2a_normalizer_error_is_value_error_subclass(self):
        """W2a: NormalizerError now inherits from ValueError.

        A failure here means the base class was reverted or a subclass
        chain was broken; either way, Python-convention catchers would
        silently miss the error.
        """
        self.assertTrue(issubclass(NormalizerError, ValueError))
        # Regression guard for the specific widening: RuntimeError must
        # NOT be on the MRO or we've inadvertently kept a dual base.
        self.assertFalse(issubclass(NormalizerError, RuntimeError))
        # Instance test covers the real catch-site idiom.
        try:
            _normalize_openai_to_anthropic("not a dict")
        except ValueError as err:
            self.assertIsInstance(err, NormalizerError)
        else:
            self.fail("expected ValueError (NormalizerError) to propagate")

    def test_w2b_docstring_mentions_normalizer_error_contract(self):
        """W2b: public docstrings document the NormalizerError contract.

        The docstring contract is what downstream HAL developers read; if
        the ``Notes for callers`` paragraph is missing, catch-point
        guidance disappears from the API surface and the refinement is
        gone.
        """
        normalise_doc = _normalize_openai_to_anthropic.__doc__ or ""
        factory_doc = make_openai_compat_chat_fn.__doc__ or ""
        for phrase in ("Notes for callers", "NormalizerError"):
            self.assertIn(phrase, normalise_doc)
            self.assertIn(phrase, factory_doc)

    def test_w2c_empty_message_returns_sentinel_identity(self):
        """W2c: the silent-turn path returns the shared _EMPTY_MESSAGE.

        Identity (``is``) matters — a fresh literal would pass equality
        but the whole point of the sentinel is that a debugger inspecting
        the returned dict can see the shared module-level object and
        distinguish it from any other empty dict in the pipeline.
        """
        result = _extract_message_dict({"finish_reason": "stop"})
        self.assertIs(result, _EMPTY_MESSAGE)
        # Negative control: a real message dict must NOT return the sentinel.
        real_message = {"role": "assistant", "content": "hi"}
        self.assertIsNot(
            _extract_message_dict({
                "message": real_message, "finish_reason": "stop",
            }),
            _EMPTY_MESSAGE,
        )


if __name__ == "__main__":
    unittest.main()
