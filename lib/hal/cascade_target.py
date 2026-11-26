"""Validated product bindings for the OpenAI W4 cascade."""
from __future__ import annotations

from dataclasses import dataclass

from lib.hal.models_command import load_catalogue
from lib.hal.cascade_policy import CONFIG


@dataclass(frozen=True)
class Target:
    provider: str
    product: str
    model: str
    base_url: str
    credential: str
    auth: str
    billing: str
    transport: str
    window: int

    @property
    def label(self):
        return self.provider + '/' + self.model


PRODUCT = 'token-plan-personal'
PROVIDER = 'qwen-token-plan'
MODELS = frozenset({'qwen3.8-max', 'deepseek-v4-pro'})
FIELDS = frozenset({'provider', 'product', 'model', 'base_url', 'credential',
                    'auth', 'billing', 'transport'})
BASE_URL = 'https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1'
SELECTOR = 'grip-qwen-token-plan'


def validate(entry):
    """Accept only the exact catalogued Token Plan product tuple."""
    if not isinstance(entry, dict) or set(entry) != FIELDS:
        return None
    from lib.hal.gateway_egress import PROVIDER_ENDPOINTS
    if PROVIDER_ENDPOINTS.get(PROVIDER) != (BASE_URL, SELECTOR):
        return None
    expected = {'provider': PROVIDER, 'product': PRODUCT, 'base_url': BASE_URL,
                'credential': SELECTOR, 'auth': 'bearer',
                'billing': 'subscription', 'transport': 'openai-chat'}
    if any(type(entry[k]) is not str or entry[k] != value for k, value in expected.items()):
        return None
    if type(entry['model']) is not str or entry['model'] not in MODELS:
        return None
    windows = {model.address: model.context_window for model in
               load_catalogue(str(CONFIG.parent / 'model_catalog.yaml'))}
    window = windows.get(PROVIDER + '/' + entry['model'])
    if not isinstance(window, int) or window <= 0:
        return None
    return Target(**entry, window=window)


def is_valid(target):
    return isinstance(target, Target) and validate({k: getattr(target, k) for k in FIELDS}) == target
