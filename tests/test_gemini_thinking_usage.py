"""Gemini thinking tokens are billable output, on both Gemini paths.

Google's API reference counts thoughts apart from the answer --
"totalTokenCount ... (prompt + thoughts + response candidates)" -- and its
pricing page charges output "including thinking tokens". Both HAL adapters read
candidatesTokenCount alone, so every thinking model's output was under-counted:
AGORA's judge, gemini-3.1-pro-preview, thinks for roughly 800-1,450 tokens a
call against a visible answer of a few hundred.
"""
from lib.hal.happi.providers import gemini as happi_gemini
from lib.providers.gemini import _parse_generate_response


def test_happi_usage_bills_thinking_tokens_as_output():
    meta = {"promptTokenCount": 100, "candidatesTokenCount": 20,
            "thoughtsTokenCount": 80, "totalTokenCount": 200}
    usage = happi_gemini._extract_usage({"usageMetadata": meta}, {})
    assert usage == {"in": 100, "out": 100, "thinking": 80}
    assert usage["in"] + usage["out"] == meta["totalTokenCount"]


def test_happi_usage_does_not_count_streamed_thoughts_twice():
    # Counts are cumulative per response: a thinking-only chunk, then the
    # final chunk with the answer. Re-adding thoughts to a carried-forward
    # "out" would report 150 here instead of 100.
    first = happi_gemini._extract_usage(
        {"usageMetadata": {"promptTokenCount": 100, "thoughtsTokenCount": 50}}, {})
    assert first == {"in": 100, "out": 50, "thinking": 50}
    final = happi_gemini._extract_usage(
        {"usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20,
                           "thoughtsTokenCount": 80}}, first)
    assert final == {"in": 100, "out": 100, "thinking": 80}


def test_happi_usage_keeps_the_answer_when_a_later_chunk_omits_it():
    prev = {"in": 100, "out": 100, "thinking": 80}
    later = happi_gemini._extract_usage(
        {"usageMetadata": {"promptTokenCount": 100, "thoughtsTokenCount": 90}}, prev)
    assert later == {"in": 100, "out": 110, "thinking": 90}


def test_happi_usage_ignores_a_zero_that_would_lower_a_cumulative_count():
    # AGORA's review of HAL#619: a chunk that SENDS a zero (rather than omitting
    # the field) wiped the counts already seen, because dict.get falls back only
    # on a missing key. The counts are cumulative, so none of them can fall.
    prev = {"in": 100, "out": 100, "thinking": 80}
    later = happi_gemini._extract_usage(
        {"usageMetadata": {"promptTokenCount": 0, "candidatesTokenCount": 0,
                           "thoughtsTokenCount": 0}}, prev)
    assert later == {"in": 100, "out": 100, "thinking": 80}


def test_happi_usage_without_thinking_keeps_its_old_shape():
    meta = {"promptTokenCount": 4, "candidatesTokenCount": 3, "totalTokenCount": 7}
    assert happi_gemini._extract_usage({"usageMetadata": meta}, {}) == {"in": 4, "out": 3}


def test_legacy_provider_bills_thinking_tokens_as_completion():
    body = {"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20,
                              "thoughtsTokenCount": 80, "totalTokenCount": 200}}
    usage = _parse_generate_response(body, "gemini-3.1-pro-preview").usage
    assert usage["completion_tokens"] == 100
    assert usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"]
