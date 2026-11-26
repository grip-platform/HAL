"""Cross-platform sovereignty: a 'keychain:service' api_key_source resolves via
GRIP_<PROVIDER>_API_KEY env when the macOS keychain is unavailable (Linux/CI).

This is what lets ONE committed providers.yaml ('keychain:grip-<provider>')
work on macOS (keychain hit) AND the Linux VPS (env fallback) — removing the
per-host 'env:GRIP_*' providers.yaml divergence.

Mutation-killing: delete the GRIP_-prefixed fallback in
lib/hal/providers/openai_compat._resolve_api_key and
test_keychain_miss_falls_back_to_grip_env fails. It mocks the keychain to miss
(simulating Linux) so it never touches the host keychain.
"""
import os
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import lib.hal.providers.openai_compat as oc
from lib.hal.providers.openai_compat import _resolve_api_key


@contextmanager
def _env(**kv):
    prev = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    try:
        yield
    finally:
        for k, v in prev.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


@contextmanager
def _no_keychain():
    # Simulate a non-macOS host: keychain lookup always misses.
    with mock.patch.object(oc, "_fetch_keychain_key", return_value=None):
        yield


class TestOpenAICompatKeyResolution(unittest.TestCase):

    def test_env_source_unchanged(self):
        with _env(FOO_KEY="v1"):
            self.assertEqual(_resolve_api_key("env:FOO_KEY"), "v1")

    def test_keychain_hit_unchanged(self):
        with mock.patch.object(oc, "_fetch_keychain_key", return_value="sk-kc"):
            self.assertEqual(_resolve_api_key("keychain:hal-deepseek"), "sk-kc")

    def test_keychain_miss_falls_back_to_grip_env(self):
        # THE Linux sovereignty path: keychain absent -> GRIP_DEEPSEEK_API_KEY.
        with _no_keychain(), _env(DEEPSEEK_API_KEY=None,
                                  GRIP_DEEPSEEK_API_KEY="sk-grip"):
            self.assertEqual(_resolve_api_key("keychain:hal-deepseek"), "sk-grip")

    def test_keychain_miss_bare_env_takes_precedence(self):
        with _no_keychain(), _env(DEEPSEEK_API_KEY="sk-bare",
                                  GRIP_DEEPSEEK_API_KEY="sk-grip"):
            self.assertEqual(_resolve_api_key("keychain:hal-deepseek"), "sk-bare")

    def test_the_public_slot_name_resolves_like_the_private_one(self):
        # The public build renames grip- slots to hal- (HAL#859): same env key.
        with _no_keychain(), _env(DEEPSEEK_API_KEY="sk-bare", GRIP_DEEPSEEK_API_KEY=None):
            self.assertEqual(_resolve_api_key("keychain:hal-deepseek"), "sk-bare")

    def test_keychain_miss_no_env_returns_none(self):
        with _no_keychain(), _env(DEEPSEEK_API_KEY=None,
                                  GRIP_DEEPSEEK_API_KEY=None):
            self.assertIsNone(_resolve_api_key("keychain:hal-deepseek"))

    def test_literal_unchanged(self):
        self.assertEqual(_resolve_api_key("sk-literal"), "sk-literal")


if __name__ == "__main__":
    unittest.main()
