"""Kimi Code subscription transport -- provider ``kimi-coding``, GRIP Pool D.

Kimi Code is a SEPARATE product from Moonshot's pay-per-token API platform:
"the Kimi Code API is independent from the API service on this platform"
(platform.kimi.ai/docs/guide/product-plans). Different host
(``api.kimi.com/coding``), different credential (a Kimi Code Console API key),
different model ids (``k3``, ``k3-256k``, ``kimi-for-coding``,
``kimi-for-coding-highspeed``) and different billing (a membership quota that
"refreshes automatically every 7 days", with "a rolling 5-hour rate window").

The sovereign council of 2026-09-07 (seal ``e49c00c214c500be``) ruled this a
new Pool D, the metered Kimi API stays Pool B, and neither is Pool C -- and
that the seam be "entirely unarmed and absent from normal routing until the
dedicated subscription credential exists". These tests pin exactly that:

* the transport resolves ONLY ``KIMI_CODING_API_KEY`` / keychain
  ``grip-kimi-coding`` -- the names the GRIP side uses
  (``lib/provider_secret.resolve_provider_key("kimi-coding")``), so both repos
  arm off one credential;
* nothing is detected or registered while neither exists;
* the mis-modelled ``KimiK2Subscription`` (keychain
  ``grip-moonshot-subscription`` pointed at the pay-per-token host) is gone.

Hermetic: the keychain reader is patched, the environment is pinned, and no
network call is made. UNVERIFIED live -- no Kimi Code credential exists.
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

import lib.hal.subscription as subscription
from lib.hal.subscription import (
    KimiCodingSubscription,
    OpenAICompatSubscription,
    SubscriptionAuth,
    auto_detect_subscriptions,
)
from lib.hal.subscribe_wizard import (
    EXIT_UNKNOWN_PROVIDER,
    _PROVIDER_CONFIG,
    run_wizard,
)

# Obviously fake; long enough to pass any length sanity check. Never a real key.
_FAKE_TOKEN = "sk-kimi-FAKE-0000000000000000000000000000"
_KEYCHAIN_PATH = "lib.hal.subscription._fetch_keychain_secret"


def _no_env():
    """Context manager: the Kimi Code env var (and the retired names) absent."""
    cleared = {k: v for k, v in os.environ.items()
               if k not in ("KIMI_CODING_API_KEY", "MOONSHOT_SUBSCRIPTION_TOKEN",
                            "MOONSHOT_API_KEY", "MOONSHOT_BASE_URL")}
    return mock.patch.dict(os.environ, cleared, clear=True)


class TestTransportConfig(unittest.TestCase):
    """The class attributes ARE the seam contract -- one credential name per repo."""

    def test_provider_identity_and_hosts(self):
        self.assertEqual(KimiCodingSubscription.provider_id, "kimi-coding")
        self.assertEqual(KimiCodingSubscription.name, "kimi-coding-subscription")
        # Vendor table (kimi.com/code/docs/en/kimi-code/models): "OpenAI
        # compatible https://api.kimi.com/coding/v1 | Anthropic compatible
        # https://api.kimi.com/coding/".
        self.assertEqual(KimiCodingSubscription.base_url, "https://api.kimi.com/coding/v1")
        self.assertEqual(KimiCodingSubscription.anthropic_base_url, "https://api.kimi.com/coding/")

    def test_credential_names_match_the_grip_side_exactly(self):
        # GRIP#5336: resolve_provider_key("kimi-coding") -> keychain
        # grip-kimi-coding, env fallback KIMI_CODING_API_KEY. A single env name,
        # on purpose: the retired MOONSHOT_* names belong to the metered product.
        self.assertEqual(KimiCodingSubscription.env_vars, ("KIMI_CODING_API_KEY",))
        self.assertEqual(KimiCodingSubscription.keychain_service, "grip-kimi-coding")

    def test_pool_and_quota_descriptor(self):
        self.assertEqual(KimiCodingSubscription.pool, "D")
        # Membership page: "refreshes automatically every 7 days ... a rolling
        # 5-hour rate window".
        self.assertEqual(dict(KimiCodingSubscription.quota),
                         {"window_days": 7, "rolling_hours": 5})
        with self.assertRaises(TypeError):
            KimiCodingSubscription.quota["window_days"] = 30  # read-only descriptor

    def test_is_an_openai_compat_subscription_transport(self):
        self.assertTrue(issubclass(KimiCodingSubscription, OpenAICompatSubscription))
        self.assertIsInstance(KimiCodingSubscription(_FAKE_TOKEN), SubscriptionAuth)

    def test_underlying_provider_targets_the_coding_host(self):
        provider = KimiCodingSubscription(_FAKE_TOKEN)._get_provider()
        self.assertEqual(provider.base_url, "https://api.kimi.com/coding/v1")
        self.assertEqual(provider.name, "kimi-coding-subscription")

    def test_old_transport_is_gone(self):
        # The old class modelled a product that does not exist (a
        # "subscription" on the pay-per-token host). No alias: nothing in HAL
        # or GRIP imports the old name any more.
        self.assertFalse(hasattr(subscription, "KimiK2Subscription"))


class TestUnarmedByConstruction(unittest.TestCase):
    """No credential anywhere => None, and only the NEW keychain service is asked."""

    def test_from_env_is_none_without_env_or_keychain(self):
        asked: list[str] = []

        def fake_keychain(service: str) -> str:
            asked.append(service)
            return ""

        with _no_env(), mock.patch(_KEYCHAIN_PATH, side_effect=fake_keychain):
            self.assertIsNone(KimiCodingSubscription.from_env())
        self.assertEqual(asked, ["grip-kimi-coding"])
        self.assertNotIn("grip-moonshot-subscription", asked)

    def test_retired_moonshot_env_names_do_not_arm_it(self):
        with _no_env(), mock.patch.dict(os.environ, {
            "MOONSHOT_SUBSCRIPTION_TOKEN": _FAKE_TOKEN,
            "MOONSHOT_API_KEY": _FAKE_TOKEN,
        }), mock.patch(_KEYCHAIN_PATH, return_value=""):
            self.assertIsNone(KimiCodingSubscription.from_env())

    def test_auto_detect_omits_kimi_coding_when_unresolved(self):
        with mock.patch.object(subscription.CCHSubscription, "from_env", return_value=None), \
             mock.patch.object(subscription.QwenSubscription, "from_env", return_value=None), \
             mock.patch.object(KimiCodingSubscription, "from_env", return_value=None):
            detected = auto_detect_subscriptions()
        self.assertEqual(detected, {})


class TestArmedWhenTheCredentialExists(unittest.TestCase):
    """The same seam resolves the moment the operator stores the key."""

    def test_from_env_prefers_the_env_var(self):
        with _no_env(), mock.patch.dict(os.environ, {"KIMI_CODING_API_KEY": _FAKE_TOKEN}), \
             mock.patch(_KEYCHAIN_PATH, return_value="") as keychain:
            sub = KimiCodingSubscription.from_env()
        self.assertIsNotNone(sub)
        self.assertEqual(sub._token, _FAKE_TOKEN)
        keychain.assert_not_called()

    def test_from_env_falls_back_to_the_keychain_service(self):
        def fake_keychain(service: str) -> str:
            return _FAKE_TOKEN if service == "grip-kimi-coding" else ""

        with _no_env(), mock.patch(_KEYCHAIN_PATH, side_effect=fake_keychain):
            sub = KimiCodingSubscription.from_env()
        self.assertIsNotNone(sub)
        self.assertEqual(sub._token, _FAKE_TOKEN)

    def test_auto_detect_registers_under_provider_id_kimi_coding(self):
        armed = KimiCodingSubscription(_FAKE_TOKEN)
        armed._provider = mock.Mock(**{"health_check.return_value": True})
        with mock.patch.object(subscription.CCHSubscription, "from_env", return_value=None), \
             mock.patch.object(subscription.QwenSubscription, "from_env", return_value=None), \
             mock.patch.object(KimiCodingSubscription, "from_env", return_value=armed):
            detected = auto_detect_subscriptions()
        self.assertIs(detected.get("kimi-coding"), armed)
        # The metered provider id is NOT claimed by this transport -- Pool B
        # (moonshot, keychain grip-moonshot) and Pool D are different bills.
        self.assertNotIn("moonshot", detected)

    def test_chat_delegates_with_the_bare_kimi_code_model_id(self):
        sub = KimiCodingSubscription(_FAKE_TOKEN)
        fake_provider = mock.Mock()
        fake_provider.chat.return_value = "fake-response"
        sub._provider = fake_provider
        self.assertEqual(sub.chat("hello", model="k3-256k"), "fake-response")
        fake_provider.chat.assert_called_once_with("hello", model="k3-256k")


class TestWizardRoutesKimiCoding(unittest.TestCase):
    """``hal --subscribe kimi-coding`` writes keychain ``grip-kimi-coding``."""

    def test_wizard_entry(self):
        cfg = _PROVIDER_CONFIG["kimi-coding"]
        self.assertEqual(cfg["service"], "grip-kimi-coding")
        # A classmethod access mints a fresh bound method each time, so identity
        # can never hold; bound methods compare EQUAL by (__self__, __func__).
        self.assertEqual(cfg["loader"], KimiCodingSubscription.from_env)
        self.assertIs(cfg["loader"].__self__, KimiCodingSubscription)
        self.assertIn("Kimi Code", cfg["label"])
        self.assertIn("Pool D", cfg["label"])

    def test_moonshot_is_no_longer_a_subscription_key(self):
        # A Moonshot platform key is metered (Pool B); offering it as a
        # "subscription" was the mis-model this change removes.
        self.assertNotIn("moonshot", _PROVIDER_CONFIG)
        code = run_wizard("moonshot", reader=lambda _: _FAKE_TOKEN,
                          printer=lambda *a, **k: None, platform_check=lambda: True)
        self.assertEqual(code, EXIT_UNKNOWN_PROVIDER)

    def test_wizard_writes_the_new_service(self):
        captured: dict = {}

        def fake_writer(token, **kwargs):
            captured.update(kwargs)
            return True, "ok"

        run_wizard("kimi-coding", reader=lambda _: _FAKE_TOKEN, writer=fake_writer,
                   verifier=lambda: True, printer=lambda *a, **k: None,
                   platform_check=lambda: True)
        self.assertEqual(captured.get("service"), "grip-kimi-coding")


if __name__ == "__main__":
    unittest.main()
