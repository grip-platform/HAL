"""HAL Gateway — Codex-interop translation tests (Bug 1 image, Bug 2 model).

Goodhart-proof: each test asserts a concrete translated value, never a call
count, and fails under the obvious mutation:

  - input_image inside a tool_result not converted    -> test_responses_tool_result_input_image_becomes_image
  - data-URI image_url not converted (responses)       -> test_responses_message_image_url_data_uri
  - http URL input_image not converted                 -> test_translate_http_url_image
  - data-URI base64 source shape wrong                 -> test_translate_data_uri_image
  - input_image survives anywhere in the payload       -> test_no_input_image_tag_survives_full_payload
  - gpt-5.5 not mapped to a backend (responses)        -> test_responses_model_gpt55_resolves
  - gpt-5.5 not mapped to a backend (chat completions) -> test_chat_model_gpt55_resolves
  - Codex alias maps to wrong backend                  -> test_resolve_model_aliases
  - an Anthropic model gets rewritten                  -> test_resolve_model_passthrough
  - image_url in a chat tool message not converted     -> test_chat_tool_message_image_url_becomes_image
"""
import json
import unittest

from lib.hal.gateway_translate import (
    resolve_model,
    translate_content_blocks,
    translate_image_block,
)
from lib.hal.gateway_openai import openai_to_anthropic_request
from lib.hal.gateway_responses import responses_to_anthropic_request

_DATA_URI = "data:image/png;base64,iVBORw0KGgoAAAANS"
_HTTP_URL = "https://example.com/screenshot.png"
_BACKEND = "anthropic/claude-opus-4-8"


def _find_blocks(obj, btype):
    """Every dict in ``obj`` (recursively) whose ``type`` == ``btype``."""
    found = []
    if isinstance(obj, dict):
        if obj.get("type") == btype:
            found.append(obj)
        for v in obj.values():
            found.extend(_find_blocks(v, btype))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(_find_blocks(v, btype))
    return found


# --- Bug 2: model alias resolution ------------------------------------------

class TestModelResolution(unittest.TestCase):
    def test_resolve_model_aliases(self):
        for alias in ("gpt-5.5", "GPT-5.5", "gpt-5.5-codex", "gpt-5-codex", "gpt-5"):
            self.assertEqual(resolve_model(alias), _BACKEND, alias)

    def test_resolve_model_passthrough(self):
        # Already-valid / unknown models are NOT rewritten (fail-open).
        for model in ("claude-opus-4-8", "anthropic/claude-sonnet-4-5", "qwen/qwen3"):
            self.assertEqual(resolve_model(model), model)

    def test_responses_model_gpt55_resolves(self):
        out = responses_to_anthropic_request({"model": "gpt-5.5", "input": "hi"})
        self.assertEqual(out["model"], _BACKEND)
        self.assertNotEqual(out["model"], "gpt-5.5")  # the 404 cause is gone

    def test_chat_model_gpt55_resolves(self):
        out = openai_to_anthropic_request(
            {"model": "gpt-5.5", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(out["model"], _BACKEND)


# --- Bug 1: image-block translation -----------------------------------------

class TestImageTranslation(unittest.TestCase):
    def test_translate_data_uri_image(self):
        block = translate_image_block({"type": "input_image", "image_url": _DATA_URI})
        self.assertEqual(block, {"type": "image", "source": {
            "type": "base64", "media_type": "image/png",
            "data": "iVBORw0KGgoAAAANS"}})

    def test_translate_http_url_image(self):
        block = translate_image_block(
            {"type": "input_image", "image_url": {"url": _HTTP_URL}})
        self.assertEqual(block, {"type": "image", "source": {
            "type": "url", "url": _HTTP_URL}})

    def test_responses_tool_result_input_image_becomes_image(self):
        """The MAIN bug: input_image nested in a tool_result's content array."""
        payload = {
            "model": "gpt-5.5",
            "input": [{
                "type": "function_call_output",
                "call_id": "call_1",
                "output": [
                    {"type": "input_text", "text": "see screenshot"},
                    {"type": "input_image", "image_url": _DATA_URI},
                ],
            }],
        }
        out = responses_to_anthropic_request(payload)
        tool_results = _find_blocks(out["messages"], "tool_result")
        self.assertEqual(len(tool_results), 1)
        inner = tool_results[0]["content"]
        images = [b for b in inner if b.get("type") == "image"]
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]["source"]["type"], "base64")
        self.assertEqual(images[0]["source"]["media_type"], "image/png")
        # And critically: NO input_image tag anywhere in the translated payload.
        self.assertEqual(_find_blocks(out, "input_image"), [])

    def test_responses_message_image_url_data_uri(self):
        payload = {
            "model": "gpt-5.5",
            "input": [{
                "type": "message", "role": "user",
                "content": [
                    {"type": "input_text", "text": "what is this"},
                    {"type": "input_image", "image_url": {"url": _DATA_URI}},
                ],
            }],
        }
        out = responses_to_anthropic_request(payload)
        images = _find_blocks(out["messages"], "image")
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]["source"]["data"], "iVBORw0KGgoAAAANS")

    def test_no_input_image_tag_survives_full_payload(self):
        """End-to-end guard: input_image at message level AND nested in a
        tool_result both vanish; the only image tag is Anthropic's `image`."""
        payload = {
            "model": "gpt-5.5",
            "input": [
                {"type": "message", "role": "user",
                 "content": [{"type": "input_image", "image_url": _HTTP_URL}]},
                {"type": "function_call_output", "call_id": "c2",
                 "output": [{"type": "input_image", "image_url": _DATA_URI}]},
            ],
        }
        out = responses_to_anthropic_request(payload)
        self.assertEqual(_find_blocks(out, "input_image"), [])
        self.assertEqual(_find_blocks(out, "image_url"), [])
        self.assertEqual(len(_find_blocks(out, "image")), 2)

    def test_chat_tool_message_image_url_becomes_image(self):
        """Chat-completions path: image nested in a tool message's content."""
        payload = {
            "model": "gpt-5.5",
            "messages": [{
                "role": "tool", "tool_call_id": "call_9",
                "content": [
                    {"type": "text", "text": "tool output"},
                    {"type": "image_url", "image_url": {"url": _DATA_URI}},
                ],
            }],
        }
        out = openai_to_anthropic_request(payload)
        tool_results = _find_blocks(out["messages"], "tool_result")
        self.assertEqual(len(tool_results), 1)
        images = [b for b in tool_results[0]["content"] if b.get("type") == "image"]
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]["source"]["type"], "base64")
        self.assertEqual(_find_blocks(out, "image_url"), [])

    def test_translate_content_blocks_string_passthrough(self):
        # A plain-string tool content is unchanged (the common no-image case).
        self.assertEqual(translate_content_blocks("plain output"), "plain output")

    def test_translate_content_blocks_preserves_anthropic_blocks(self):
        # An already-Anthropic block passes through verbatim (fail-open).
        already = [{"type": "text", "text": "hi"}]
        self.assertEqual(translate_content_blocks(already), already)


if __name__ == "__main__":
    unittest.main()
