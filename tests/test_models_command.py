"""Tests for HAL's /models command + subscribe wizard.

Covers:
- catalogue parsing (real config/model_catalog.yaml + synthetic fixtures)
- annotation against a stub registry (every status branch)
- filter_entries (address, provider, tier, tag, substring, no-match)
- render_models (empty, detail, grouped, filter title)
- handle_models_command (orchestration smoke)
- subscribe_wizard (every exit code branch)
"""
import os
import sys
import tempfile
import unittest
from typing import Optional
from unittest import mock

# Make sure the repo root is on sys.path even when pytest is invoked from elsewhere.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.hal.models_command import (  # noqa: E402
    AnnotatedEntry,
    ModelEntry,
    annotate_availability,
    filter_entries,
    handle_models_command,
    load_catalogue,
    render_models,
)
from lib.hal.subscribe_wizard import (  # noqa: E402
    EXIT_KEYCHAIN_FAILED,
    EXIT_OK,
    EXIT_TOKEN_INVALID,
    EXIT_UNSUPPORTED_PLATFORM,
    EXIT_USER_CANCELLED,
    EXIT_VERIFY_FAILED,
    _validate_token,
    run_wizard,
)


# ────────────────────────────────────────────────────────────────────
# Test fixtures
# ────────────────────────────────────────────────────────────────────

_FIXTURE_YAML = """\
# Synthetic catalogue used only by tests.
models:

  - address: alpha/model-one
    tier: frontier
    paid: true
    modality: text
    context_window: 128000
    tags: [reasoning, fast]
    strengths:
      - First strength
      - Second strength
    description: >
      Alpha model description spans
      multiple lines for the test.

  - address: beta/model-two
    tier: cheap
    paid: true
    modality: text+vision
    context_window: 32000
    tags: [vision, cheap]
    strengths:
      - Vision capability
    description: Single-line beta description.

  - address: ollama/local-three
    tier: local
    paid: false
    modality: text
    context_window: 8000
    tags: [local]
    strengths:
      - Runs offline
    description: Local model.

  - address: anthropic/sub-model
    tier: subscription
    paid: true
    modality: text
    context_window: 200000
    tags: [reasoning]
    strengths:
      - Subscription-only access
    description: Subscription test entry.
"""


def _write_fixture() -> str:
    """Persist the fixture YAML to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="hal-models-fixture-")
    with os.fdopen(fd, "w") as f:
        f.write(_FIXTURE_YAML)
    return path


# ────────────────────────────────────────────────────────────────────
# Stub registry
# ────────────────────────────────────────────────────────────────────

class _StubProvider:
    def __init__(self, name: str, healthy: bool = True):
        from lib.hal.types import ProviderType
        self.name = name
        self.type = ProviderType.CLOUD if name != "ollama" else ProviderType.LOCAL
        self.base_url = ""
        self.healthy = healthy
        self.models = ()


class _StubRegistry:
    def __init__(self, providers: Optional[list] = None):
        self._providers = providers or []

    def list_providers(self):
        return list(self._providers)


# ────────────────────────────────────────────────────────────────────
# Catalogue parsing
# ────────────────────────────────────────────────────────────────────

class TestCatalogueParse(unittest.TestCase):
    def test_real_catalogue_loads_with_required_fields(self):
        entries = load_catalogue()
        self.assertGreaterEqual(len(entries), 30,
                                "Real catalogue should have at least 30 entries")
        for e in entries:
            self.assertTrue(e.address, "Every entry needs an address")
            self.assertIn(e.tier, {"subscription", "frontier", "premium",
                                   "mid", "cheap", "local", "free"})
            self.assertGreater(e.context_window, 0,
                               f"{e.address} missing context_window")

    def test_real_catalogue_covers_three_frontier_labs(self):
        entries = load_catalogue()
        providers = {e.address.split("/", 1)[0] for e in entries}
        # Anthropic via subscription, Google via gemini, OpenAI direct.
        for required in ("anthropic", "gemini", "openai"):
            self.assertIn(required, providers,
                          f"Frontier provider {required} missing from catalogue")

    def test_fixture_yaml_round_trips(self):
        path = _write_fixture()
        try:
            entries = load_catalogue(path)
            self.assertEqual(len(entries), 4)
            alpha = entries[0]
            self.assertEqual(alpha.address, "alpha/model-one")
            self.assertEqual(alpha.tier, "frontier")
            self.assertEqual(alpha.modality, "text")
            self.assertEqual(alpha.context_window, 128000)
            self.assertEqual(alpha.tags, ("reasoning", "fast"))
            self.assertEqual(len(alpha.strengths), 2)
            self.assertIn("Alpha model description", alpha.description)
            self.assertIn("multiple lines", alpha.description)
        finally:
            os.unlink(path)

    def test_missing_file_returns_empty(self):
        self.assertEqual(load_catalogue("/no/such/path.yaml"), [])

    def test_inline_array_parses(self):
        path = _write_fixture()
        try:
            entries = load_catalogue(path)
            beta = entries[1]
            self.assertEqual(beta.tags, ("vision", "cheap"))
        finally:
            os.unlink(path)


# ────────────────────────────────────────────────────────────────────
# Annotation
# ────────────────────────────────────────────────────────────────────

def _entry(address: str, tier: str = "cheap") -> ModelEntry:
    return ModelEntry(
        address=address, tier=tier, paid=True, modality="text",
        context_window=128000, tags=(), strengths=(),
        description="",
    )


class TestAnnotation(unittest.TestCase):
    def test_subscription_active_when_detected(self):
        entries = [_entry("anthropic/opus", "subscription")]
        registry = _StubRegistry()
        subs = {"anthropic": object()}  # any truthy stub
        annotated = annotate_availability(entries, registry, subs)
        self.assertEqual(annotated[0].status, "active")

    def test_subscription_setup_needed_when_missing(self):
        entries = [_entry("anthropic/opus", "subscription")]
        annotated = annotate_availability(entries, _StubRegistry(), {})
        self.assertEqual(annotated[0].status, "setup_needed")
        self.assertIn("hal --subscribe", annotated[0].note)

    def test_cloud_needs_key_when_provider_unregistered(self):
        entries = [_entry("openai/gpt-4o", "frontier")]
        annotated = annotate_availability(entries, _StubRegistry(), {})
        self.assertEqual(annotated[0].status, "needs_key")
        self.assertIn("openai", annotated[0].note)

    def test_local_not_running_when_provider_unregistered(self):
        entries = [_entry("ollama/gemma4:26b", "local")]
        annotated = annotate_availability(entries, _StubRegistry(), {})
        self.assertEqual(annotated[0].status, "not_running")

    def test_ready_when_provider_healthy(self):
        entries = [_entry("groq/llama", "cheap")]
        registry = _StubRegistry([_StubProvider("groq", healthy=True)])
        annotated = annotate_availability(entries, registry, {})
        self.assertEqual(annotated[0].status, "ready")

    def test_offline_when_provider_unhealthy(self):
        entries = [_entry("groq/llama", "cheap")]
        registry = _StubRegistry([_StubProvider("groq", healthy=False)])
        annotated = annotate_availability(entries, registry, {})
        self.assertEqual(annotated[0].status, "offline")


# ────────────────────────────────────────────────────────────────────
# Filtering
# ────────────────────────────────────────────────────────────────────

def _annot(address: str, tier: str = "cheap", tags=()):
    e = ModelEntry(address=address, tier=tier, paid=True, modality="text",
                   context_window=128000, tags=tuple(tags), strengths=(),
                   description="")
    return AnnotatedEntry(e, "ready", "")


class TestFilter(unittest.TestCase):
    def setUp(self):
        self.entries = [
            _annot("groq/llama-3.3", "cheap", ("fast", "open")),
            _annot("ollama/gemma4:26b", "local", ("local", "vision")),
            _annot("openai/gpt-4o", "frontier", ("frontier", "vision")),
            _annot("anthropic/claude-opus-4-7", "subscription", ("frontier",)),
        ]

    def test_empty_returns_all(self):
        self.assertEqual(len(filter_entries(self.entries, "")), 4)

    def test_exact_address(self):
        out = filter_entries(self.entries, "openai/gpt-4o")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].entry.address, "openai/gpt-4o")

    def test_provider_prefix(self):
        out = filter_entries(self.entries, "ollama")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].entry.address, "ollama/gemma4:26b")

    def test_tier_alias(self):
        out = filter_entries(self.entries, "frontier")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].entry.tier, "frontier")

    def test_tag_substring(self):
        out = filter_entries(self.entries, "vision")
        addresses = {a.entry.address for a in out}
        self.assertIn("ollama/gemma4:26b", addresses)
        self.assertIn("openai/gpt-4o", addresses)

    def test_no_match_returns_empty(self):
        self.assertEqual(filter_entries(self.entries, "nonexistent-xyz"), [])

    def test_subscription_alias(self):
        out = filter_entries(self.entries, "subscription")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].entry.address, "anthropic/claude-opus-4-7")


# ────────────────────────────────────────────────────────────────────
# Rendering
# ────────────────────────────────────────────────────────────────────

class TestRender(unittest.TestCase):
    def test_empty_message(self):
        out = render_models([], filter_term="frontier")
        self.assertIn("No models match", out)
        self.assertIn("frontier", out)

    def test_detail_mode_for_single_match(self):
        a = _annot("openai/gpt-4o", "frontier", ("vision",))
        out = render_models([a], detail=True)
        self.assertIn("openai/gpt-4o", out)
        self.assertIn("/model openai/gpt-4o", out)  # switch hint

    def test_grouped_render_includes_tier_labels(self):
        entries = [
            _annot("openai/gpt-4o", "frontier"),
            _annot("groq/llama", "cheap"),
            _annot("ollama/gemma4:26b", "local"),
        ]
        out = render_models(entries)
        self.assertIn("Frontier", out)
        self.assertIn("Cheap", out)
        self.assertIn("Local", out)
        for e in entries:
            self.assertIn(e.entry.address, out)

    def test_filter_title_shown(self):
        entries = [_annot("openai/gpt-4o", "frontier")]
        out = render_models(entries, filter_term="frontier")
        self.assertIn("matching `frontier`", out)


# ────────────────────────────────────────────────────────────────────
# Orchestration
# ────────────────────────────────────────────────────────────────────

class TestHandleCommand(unittest.TestCase):
    def test_handle_returns_string_with_sections(self):
        path = _write_fixture()
        try:
            out = handle_models_command(_StubRegistry(), "",
                                        catalogue_path=path, subscriptions={})
            self.assertIn("HAL Models", out)
            self.assertIn("alpha/model-one", out)
            self.assertIn("ollama/local-three", out)
        finally:
            os.unlink(path)

    def test_handle_with_filter(self):
        path = _write_fixture()
        try:
            out = handle_models_command(_StubRegistry(), "frontier",
                                        catalogue_path=path, subscriptions={})
            self.assertIn("alpha/model-one", out)
            self.assertNotIn("beta/model-two", out)
        finally:
            os.unlink(path)

    def test_handle_with_missing_catalogue(self):
        out = handle_models_command(_StubRegistry(), "",
                                    catalogue_path="/no/such/file",
                                    subscriptions={})
        self.assertIn("Catalogue not found", out)


# ────────────────────────────────────────────────────────────────────
# Subscribe wizard
# ────────────────────────────────────────────────────────────────────

class TestValidateToken(unittest.TestCase):
    def test_empty_rejected(self):
        self.assertIsNone(_validate_token(""))
        self.assertIsNone(_validate_token("   "))

    def test_too_short_rejected(self):
        self.assertIsNone(_validate_token("short"))

    def test_strip_whitespace(self):
        self.assertEqual(_validate_token("  " + "x" * 30 + "  "), "x" * 30)

    def test_strip_bearer_prefix(self):
        self.assertEqual(_validate_token("Bearer " + "x" * 30), "x" * 30)


class TestWizard(unittest.TestCase):
    def _printer(self):
        out = []
        return out, lambda *args, **kwargs: out.append(" ".join(str(a) for a in args))

    def test_unsupported_platform(self):
        out, p = self._printer()
        # _print_unsupported_hint uses the real `print`, not the injected printer,
        # so we don't assert on `out` here — the exit code is the contract.
        code = run_wizard(printer=p, platform_check=lambda: False)
        self.assertEqual(code, EXIT_UNSUPPORTED_PLATFORM)

    def test_user_cancellation(self):
        out, p = self._printer()
        def reader(_prompt):
            raise KeyboardInterrupt()
        code = run_wizard(reader=reader, printer=p,
                          platform_check=lambda: True)
        self.assertEqual(code, EXIT_USER_CANCELLED)

    def test_invalid_token(self):
        out, p = self._printer()
        code = run_wizard(reader=lambda _: "short", printer=p,
                          platform_check=lambda: True,
                          _active_loader=lambda: None)
        self.assertEqual(code, EXIT_TOKEN_INVALID)

    def test_keychain_failure(self):
        out, p = self._printer()
        token = "x" * 40
        def writer(t, **_):
            return False, "permission denied"
        code = run_wizard(reader=lambda _: token, writer=writer,
                          printer=p, platform_check=lambda: True,
                          _active_loader=lambda: None)
        self.assertEqual(code, EXIT_KEYCHAIN_FAILED)
        self.assertIn("permission denied", "\n".join(out))

    def test_verify_failure(self):
        out, p = self._printer()
        token = "x" * 40
        code = run_wizard(reader=lambda _: token,
                          writer=lambda t, **_: (True, "ok"),
                          verifier=lambda: False,
                          printer=p, platform_check=lambda: True,
                          _active_loader=lambda: None)
        self.assertEqual(code, EXIT_VERIFY_FAILED)

    def test_success(self):
        out, p = self._printer()
        token = "x" * 40
        code = run_wizard(reader=lambda _: token,
                          writer=lambda t, **_: (True, "ok"),
                          verifier=lambda: True,
                          printer=p, platform_check=lambda: True)
        self.assertEqual(code, EXIT_OK)
        # Wizard now labels the provider explicitly ("<Label> active.")
        self.assertIn("active", "\n".join(out))


# ────────────────────────────────────────────────────────────────────
# Subscription transports (Kimi Code, Qwen) + wizard multi-provider routing
# ────────────────────────────────────────────────────────────────────

from unittest import mock  # noqa: E402

from lib.hal.subscription import (  # noqa: E402
    KimiCodingSubscription,
    OpenAICompatSubscription,
    QwenSubscription,
    _fetch_token,
    auto_detect_subscriptions,
)
from lib.hal.subscribe_wizard import (  # noqa: E402
    DEFAULT_PROVIDER,
    EXIT_KEYCHAIN_FAILED,
    EXIT_UNKNOWN_PROVIDER,
    _PROVIDER_CONFIG,
)


class TestSubscriptionConfig(unittest.TestCase):
    def test_kimi_config(self):
        # Kimi Code (the membership product, Pool D) is NOT the pay-per-token
        # Moonshot platform: its own host, key names and provider id. Full
        # contract in tests/test_kimi_coding_subscription.py.
        self.assertEqual(KimiCodingSubscription.base_url,
                         "https://api.kimi.com/coding/v1")
        self.assertIsInstance(KimiCodingSubscription.env_vars, tuple)
        self.assertEqual(KimiCodingSubscription.env_vars, ("KIMI_CODING_API_KEY",))
        self.assertEqual(KimiCodingSubscription.keychain_service,
                         "grip-kimi-coding")
        self.assertEqual(KimiCodingSubscription.provider_id, "kimi-coding")

    def test_qwen_config(self):
        self.assertEqual(
            QwenSubscription.base_url,
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        self.assertIsInstance(QwenSubscription.env_vars, tuple)
        self.assertIn("DASHSCOPE_SUBSCRIPTION_TOKEN", QwenSubscription.env_vars)
        self.assertEqual(QwenSubscription.keychain_service,
                         "hal-dashscope-subscription")
        self.assertEqual(QwenSubscription.provider_id, "alibaba")

    def test_subclasses_inherit_base(self):
        self.assertTrue(issubclass(KimiCodingSubscription, OpenAICompatSubscription))
        self.assertTrue(issubclass(QwenSubscription, OpenAICompatSubscription))


class TestTupleBugFix(unittest.TestCase):
    """_fetch_token used to be called with a bare string — that iterates chars."""

    def test_from_env_with_empty_environment_returns_none(self):
        for v in KimiCodingSubscription.env_vars:
            os.environ.pop(v, None)
        for v in QwenSubscription.env_vars:
            os.environ.pop(v, None)
        # Keychain stubbed absent (not merely assumed absent on this box) → None.
        with mock.patch("lib.hal.subscription._fetch_keychain_secret", return_value=""):
            self.assertIsNone(KimiCodingSubscription.from_env())
            self.assertIsNone(QwenSubscription.from_env())

    def test_from_env_honours_first_env_var(self):
        for v in KimiCodingSubscription.env_vars:
            os.environ.pop(v, None)
        with mock.patch.dict(os.environ,
                             {"KIMI_CODING_API_KEY": "x" * 40}):
            sub = KimiCodingSubscription.from_env()
            self.assertIsNotNone(sub)
            self.assertEqual(sub._token, "x" * 40)

    def test_from_env_falls_back_to_secondary_env_var(self):
        # Qwen keeps two env names, so it still exercises the tuple walk.
        os.environ.pop("DASHSCOPE_SUBSCRIPTION_TOKEN", None)
        with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "y" * 40}):
            sub = QwenSubscription.from_env()
            self.assertIsNotNone(sub)
            self.assertEqual(sub._token, "y" * 40)

    def test_fetch_token_iterates_tuple_not_chars(self):
        with mock.patch.dict(os.environ, {"HAL_FAKE_ENV_42": "correct"},
                             clear=False):
            result = _fetch_token(("HAL_FAKE_ENV_42",), "no-such-service")
        self.assertEqual(result, "correct")


class TestOpenAICompatSubscription(unittest.TestCase):
    def test_no_token_health_check_false(self):
        sub = KimiCodingSubscription("")
        self.assertFalse(sub.health_check())

    def test_chat_no_longer_raises_not_implemented(self):
        sub = KimiCodingSubscription("x" * 40)
        fake_provider = mock.Mock()
        fake_provider.chat.return_value = "fake-response"
        sub._provider = fake_provider
        result = sub.chat("hello", model="k3-256k")
        self.assertEqual(result, "fake-response")
        # The Kimi locked-temperature quirk lives at the ADAPTER seam now
        # (lib/hal/model_quirks.py — the provider OMITS temperature for the
        # whole locked lineage), so the subscription layer passes kwargs
        # through untouched and the delegate sees no forced temperature.
        fake_provider.chat.assert_called_once_with("hello", model="k3-256k")

    def test_chat_delegates_with_all_kwargs(self):
        sub = QwenSubscription("y" * 40)
        fake_provider = mock.Mock()
        sub._provider = fake_provider
        sub.chat("hi", model="qwen3.5-plus", system="s",
                 temperature=0.1, max_tokens=256)
        fake_provider.chat.assert_called_once_with(
            "hi", model="qwen3.5-plus", system="s",
            temperature=0.1, max_tokens=256,
        )

    def test_health_check_delegates(self):
        sub = KimiCodingSubscription("x" * 40)
        fake_provider = mock.Mock()
        fake_provider.health_check.return_value = True
        sub._provider = fake_provider
        self.assertTrue(sub.health_check())
        fake_provider.health_check.assert_called_once()


class TestAutoDetectShape(unittest.TestCase):
    def test_returns_dict_keyed_by_provider_id(self):
        """Hermetic since 2026-10-17. The real ``from_env`` reads the keychain, and
        on a Mac holding a ``grip-claude-oauth`` item it went on to import
        ``cch.py`` under this test's HAL_HOME, which has no configs/models.toml,
        so the test failed for a reason unrelated to the shape it checks (it was
        red on main since at least d0ca3417). Each provider's discovery is stood
        in for; one is found, so the key really is the provider id."""
        from lib.hal import subscription as sub

        class _Found:
            provider_id = "anthropic"

            def health_check(self):
                return True

        with mock.patch.object(sub.CCHSubscription, "from_env", return_value=_Found()), \
                mock.patch.object(sub.KimiCodingSubscription, "from_env", return_value=None), \
                mock.patch.object(sub.QwenSubscription, "from_env", return_value=None):
            result = auto_detect_subscriptions()
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["anthropic"])


class TestWizardRouting(unittest.TestCase):
    def _printer(self):
        out = []
        return out, lambda *a, **k: out.append(" ".join(str(x) for x in a))

    def test_default_is_anthropic(self):
        self.assertEqual(DEFAULT_PROVIDER, "anthropic")

    def test_provider_config_has_all_three(self):
        self.assertIn("anthropic", _PROVIDER_CONFIG)
        self.assertIn("kimi-coding", _PROVIDER_CONFIG)
        self.assertIn("qwen", _PROVIDER_CONFIG)
        for cfg in _PROVIDER_CONFIG.values():
            self.assertIn("label", cfg)
            self.assertIn("service", cfg)
            self.assertIn("loader", cfg)
            self.assertTrue(callable(cfg["loader"]))

    def test_unknown_provider_exits_with_code(self):
        out, p = self._printer()
        code = run_wizard("bogus", reader=lambda _: "x" * 40, printer=p,
                          platform_check=lambda: True)
        self.assertEqual(code, EXIT_UNKNOWN_PROVIDER)
        self.assertIn("Unknown", "\n".join(out))

    def test_kimi_coding_routing(self):
        captured: dict = {}

        def fake_writer(token, **kwargs):
            captured.update(kwargs)
            return True, "ok"

        _, p = self._printer()
        run_wizard("kimi-coding", reader=lambda _: "x" * 40,
                   writer=fake_writer, verifier=lambda: True,
                   printer=p, platform_check=lambda: True)
        self.assertEqual(captured.get("service"), "grip-kimi-coding")

    def test_qwen_routing(self):
        captured: dict = {}

        def fake_writer(token, **kwargs):
            captured.update(kwargs)
            return True, "ok"

        _, p = self._printer()
        run_wizard("qwen", reader=lambda _: "x" * 40,
                   writer=fake_writer, verifier=lambda: True,
                   printer=p, platform_check=lambda: True)
        self.assertEqual(captured.get("service"), "hal-dashscope-subscription")

    def test_anthropic_routing(self):
        captured: dict = {}

        def fake_writer(token, **kwargs):
            captured.update(kwargs)
            return True, "ok"

        _, p = self._printer()
        run_wizard("anthropic", reader=lambda _: "x" * 40,
                   writer=fake_writer, verifier=lambda: True,
                   printer=p, platform_check=lambda: True,
                   _active_loader=lambda: None)
        self.assertEqual(captured.get("service"), "grip-claude-oauth")

    def test_banner_labels_each_provider(self):
        out, p = self._printer()
        run_wizard("kimi-coding", reader=lambda _: "x" * 40,
                   writer=lambda t, **_: (True, "ok"),
                   verifier=lambda: True, printer=p,
                   platform_check=lambda: True)
        self.assertIn("Kimi Code", "\n".join(out))

    def test_keychain_failure_propagates(self):
        _, p = self._printer()
        code = run_wizard("qwen", reader=lambda _: "x" * 40,
                          writer=lambda t, **_: (False, "permission denied"),
                          printer=p, platform_check=lambda: True)
        self.assertEqual(code, EXIT_KEYCHAIN_FAILED)


class TestCliSubscribeParsing(unittest.TestCase):
    def _parse(self, argv):
        from lib.hal.cli import _parse_args
        original = sys.argv
        sys.argv = ["hal"] + argv
        try:
            return _parse_args()
        finally:
            sys.argv = original

    def test_subscribe_with_provider(self):
        args = self._parse(["--subscribe", "moonshot"])
        self.assertEqual(args["subscribe"], "moonshot")

    def test_subscribe_without_arg_uses_sentinel(self):
        args = self._parse(["--subscribe"])
        self.assertEqual(args["subscribe"], "*")

    def test_subscribe_absent(self):
        args = self._parse([])
        self.assertEqual(args["subscribe"], "")

    def test_subscribe_then_other_flag(self):
        args = self._parse(["--subscribe", "--status"])
        self.assertEqual(args["subscribe"], "*")
        self.assertTrue(args["status"])


if __name__ == "__main__":
    unittest.main()
