"""Tests for _OPENAI_COMPAT_ROUTES and the routing helpers in tool_cli_integration.

Verifies:
- All 14 route entries are present with correct prefix, base_url, and key source
- _match_openai_compat_route strips the prefix and returns the right base_url
- Unknown prefixes return None (fall through to Anthropic path)
- Literal bearer sources resolve to the literal value
- Keychain bearer sources return None when the keychain has no entry
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

# H343: worktree-portable path resolution. The previous form
# `_HAL_HOME = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))`
# silently shadowed worktree-local changes — running this test from a
# /tmp/auto-* worktree resolved imports to the main ~/.hal tree, hiding
# changes the test was supposed to verify. Using __file__ ties the test
# to its own checkout, which is correct for any worktree.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.hal.tool_cli_integration import (
    _OPENAI_COMPAT_ROUTES,
    _match_openai_compat_route,
    _resolve_bearer,
)

# Expected routing table: (prefix, base_url_fragment, key_source)
_EXPECTED_ROUTES = [
    ("ollama/",     "localhost:11434",                              "literal:ollama"),
    ("groq/",       "api.groq.com",                                "keychain:grip-groq"),
    ("gemini/",     "generativelanguage.googleapis.com",           "keychain:grip-gemini"),
    ("openai/",     "api.openai.com",                              "keychain:hal-openai"),
    ("mlx_vlm/",    "localhost:8080",                              "literal:mlx_vlm"),
    ("vllm/",       "localhost:8000",                              "literal:vllm"),
    ("lmstudio/",   "localhost:1234",                              "literal:lmstudio"),
    ("lemonade/",   "localhost:7860",                              "literal:lemonade"),
    ("openrouter/", "openrouter.ai",                               "keychain:grip-openrouter"),
    ("qwen/",       "dashscope-intl.aliyuncs.com",                 "keychain:hal-dashscope"),
    ("qwen-token-plan/", "token-plan.ap-southeast-1.maas.aliyuncs.com", "keychain:grip-qwen-token-plan"),
    ("deepseek/",   "api.deepseek.com",                            "keychain:hal-deepseek"),
    ("moonshot/",   "api.moonshot.ai",                             "keychain:grip-moonshot"),
    ("kimi-coding/", "api.kimi.com",                               "keychain:grip-kimi-coding"),
    ("zhipu/",      "api.z.ai",                                    "keychain:grip-zhipu"),
    ("mistral/",    "api.mistral.ai",                              "keychain:grip-mistral"),
    ("xai/",        "api.x.ai",                                    "keychain:hal-xai"),
]


class TestRoutingTableCompleteness(unittest.TestCase):
    def test_route_count_is_17(self):
        # 17 names the real set, not a coincidence: ollama, groq, gemini,
        # openai, mlx_vlm, vllm, lmstudio, lemonade, openrouter, qwen,
        # qwen-token-plan, deepseek, moonshot, kimi-coding, zhipu, mistral,
        # xai. Was 18 until the arcee route (trinity-large-thinking, HTTP
        # 404 since 2026-11-24) was removed the same day the provider was
        # dropped — a bare count would have turned green for the wrong
        # reason here, since qwen-token-plan + kimi-coding (HAL#813,
        # unrelated) were already unlisted extras. The expected-prefix
        # tests below now pin the actual 17-member set.
        self.assertEqual(len(_OPENAI_COMPAT_ROUTES), 17)

    def test_all_expected_prefixes_present(self):
        actual_prefixes = {r[0] for r in _OPENAI_COMPAT_ROUTES}
        for prefix, _, _ in _EXPECTED_ROUTES:
            with self.subTest(prefix=prefix):
                self.assertIn(prefix, actual_prefixes)

    def test_base_urls_contain_expected_host(self):
        url_map = {r[0]: r[1] for r in _OPENAI_COMPAT_ROUTES}
        for prefix, host_fragment, _ in _EXPECTED_ROUTES:
            with self.subTest(prefix=prefix):
                self.assertIn(host_fragment, url_map[prefix])

    def test_key_sources_match(self):
        key_map = {r[0]: r[2] for r in _OPENAI_COMPAT_ROUTES}
        for prefix, _, expected_source in _EXPECTED_ROUTES:
            with self.subTest(prefix=prefix):
                self.assertEqual(key_map[prefix], expected_source)


class TestMatchRoute(unittest.TestCase):
    def test_known_prefix_strips_prefix_from_model(self):
        result = _match_openai_compat_route("ollama/gemma4:e4b")
        self.assertIsNotNone(result)
        stripped_model, _, _ = result
        self.assertEqual(stripped_model, "gemma4:e4b")

    def test_known_prefix_returns_correct_base_url(self):
        _, base_url, _ = _match_openai_compat_route("groq/llama-3.3-70b")
        self.assertIn("groq.com", base_url)

    def test_new_local_prefixes_match(self):
        for prefix in ("mlx_vlm/", "vllm/", "lmstudio/", "lemonade/"):
            with self.subTest(prefix=prefix):
                result = _match_openai_compat_route(f"{prefix}some-model")
                self.assertIsNotNone(result)

    def test_new_cloud_prefixes_match(self):
        for prefix in ("openrouter/", "qwen/", "deepseek/", "moonshot/", "zhipu/", "mistral/"):
            with self.subTest(prefix=prefix):
                result = _match_openai_compat_route(f"{prefix}some-model")
                self.assertIsNotNone(result)

    def test_unknown_prefix_returns_none(self):
        self.assertIsNone(_match_openai_compat_route("claude-opus-4-7"))

    def test_unprefixed_model_returns_none(self):
        self.assertIsNone(_match_openai_compat_route("gpt-4o"))

    def test_partial_prefix_does_not_match(self):
        # "groq" without trailing slash must not match "groq/"
        self.assertIsNone(_match_openai_compat_route("groqllama"))


class TestResolveBearer(unittest.TestCase):
    def test_literal_returns_value_verbatim(self):
        self.assertEqual(_resolve_bearer("literal:ollama"), "ollama")

    def test_literal_mlx_vlm(self):
        self.assertEqual(_resolve_bearer("literal:mlx_vlm"), "mlx_vlm")

    def test_keychain_miss_returns_none(self):
        with mock.patch(
            "lib.hal.providers.anthropic_native._fetch_keychain_key",
            return_value=None,
        ):
            result = _resolve_bearer("keychain:hal-deepseek")
        self.assertIsNone(result)

    def test_keychain_hit_returns_stripped_value(self):
        with mock.patch(
            "lib.hal.providers.anthropic_native._fetch_keychain_key",
            return_value="  sk-ds-test  ",
        ):
            result = _resolve_bearer("keychain:hal-deepseek")
        self.assertEqual(result, "sk-ds-test")

    def test_unknown_spec_returns_none(self):
        self.assertIsNone(_resolve_bearer("something-unknown"))


if __name__ == "__main__":
    unittest.main()
