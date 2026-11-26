"""Jurisdiction guard MUST NOT drop locally-hosted providers.

Bug: the jurisdiction guard in lib/hal/router.py strips China-origin
providers (DeepSeek, Qwen) from the fallback chain on sensitive content.
That is correct for a Chinese *cloud API* (data leaves the machine to a
third party in another jurisdiction). It is WRONG for a *locally-hosted*
model (Ollama/vLLM/LM Studio on localhost) even when the weights are
Chinese-origin: the data never leaves the firm's own infrastructure, so
the origin jurisdiction is irrelevant. Dropping local models sabotages the
"self-host, privileged content never leaves" guarantee.

Goodhart-proofing: each test builds a real ProviderRegistry from a real
temp providers.yaml and asserts the *resulting chain contents* (behaviour),
never a call count or a mocked internal. Each test fails iff the routing
contract regresses — e.g. reverting the fix makes the first test fail
because the local Chinese-origin provider gets stripped again, and a
blanket-disable would make the second test fail because the remote Chinese
API would no longer be dropped.
"""
from __future__ import annotations

import textwrap

import pytest

from lib.hal.registry import ProviderRegistry
from lib.hal.router import _filter_sensitive_providers, _local_provider_names

_SENSITIVE = "Please review this confidential litigation contract for GDPR compliance."
_BENIGN = "Explain the SOLID principles in one paragraph."

# Real providers.yaml fixture: a locally-hosted Chinese-origin model
# (ollama serving a Qwen weight) and a remote Chinese cloud API (qwen) and a
# remote Western provider (groq). The local entry deliberately serves a
# "qwen"-named model so the test proves the decision keys off provider TYPE,
# not the model/weight name.
_YAML = textwrap.dedent(
    """\
    providers:
      - name: groq
        type: cloud
        base_url: https://api.groq.com/openai/v1
        api_key_source: keychain:grip-groq
        models: [llama-3.3-70b-versatile]
        default: true
      - name: qwen
        type: cloud
        base_url: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
        api_key_source: keychain:hal-dashscope
        models: [qwen3.5-flash]
      - name: ollama
        type: local
        base_url: http://localhost:11434/v1
        models: [qwen2.5:7b]

    routing:
      default: fallback
      default_model: groq/llama-3.3-70b-versatile
      fallback_chain: [groq/llama-3.3-70b-versatile, qwen/qwen3.5-flash, ollama/qwen2.5:7b]
      tier_chains:
        free: [ollama/qwen2.5:7b]
        cheap: [qwen/qwen3.5-flash, groq/llama-3.3-70b-versatile]
    """
)


@pytest.fixture()
def registry(tmp_path) -> ProviderRegistry:
    yaml_path = tmp_path / "providers.yaml"
    yaml_path.write_text(_YAML)
    return ProviderRegistry(config_path=str(yaml_path))


def test_local_chinese_origin_provider_retained_on_sensitive(registry):
    """A locally-hosted Qwen/DeepSeek-style provider + sensitive content →
    provider RETAINED. Fails if the old strip-it behaviour returns (revert
    of the fix re-drops `ollama/qwen2.5:7b` here)."""
    locals_ = _local_provider_names(registry)
    assert "ollama" in locals_, "ollama (type: local) must be detected as local"

    chain = list(registry.routing.fallback_chain)
    filtered = _filter_sensitive_providers(chain, _SENSITIVE, locals_)

    assert "ollama/qwen2.5:7b" in filtered, (
        "Locally-hosted Qwen weight must NOT be dropped on sensitive content "
        "— data never leaves the firm's own infrastructure."
    )


def test_remote_chinese_cloud_provider_still_dropped_on_sensitive(registry):
    """A remote/cloud China-jurisdiction provider + same sensitive content →
    still DROPPED. Proves the real guard is preserved, not blanket-disabled.
    A blanket "never filter" regression makes this assertion fail."""
    locals_ = _local_provider_names(registry)
    chain = list(registry.routing.fallback_chain)
    filtered = _filter_sensitive_providers(chain, _SENSITIVE, locals_)

    assert "qwen/qwen3.5-flash" not in filtered, (
        "Remote Chinese cloud API must still be dropped for sensitive content "
        "— the legitimate jurisdiction guard must remain intact."
    )
    # The Western remote provider is unaffected.
    assert "groq/llama-3.3-70b-versatile" in filtered


def test_non_sensitive_content_chain_unchanged(registry):
    """Non-sensitive content → chain unchanged either way (no regression).
    Local and remote Chinese providers both survive a benign prompt."""
    locals_ = _local_provider_names(registry)
    chain = list(registry.routing.fallback_chain)
    filtered = _filter_sensitive_providers(chain, _BENIGN, locals_)

    assert filtered == chain, (
        "Benign content must not trigger any jurisdiction filtering."
    )


def test_tier_chain_local_provider_retained_on_sensitive(registry):
    """The tier-chain code path must apply the same exemption: a local
    provider in a tier chain is retained on sensitive content while the
    remote Chinese provider in the same tier is dropped."""
    locals_ = _local_provider_names(registry)

    free = _filter_sensitive_providers(
        list(registry.routing.tier_chains["free"]), _SENSITIVE, locals_
    )
    assert "ollama/qwen2.5:7b" in free

    cheap = _filter_sensitive_providers(
        list(registry.routing.tier_chains["cheap"]), _SENSITIVE, locals_
    )
    assert "qwen/qwen3.5-flash" not in cheap
    assert "groq/llama-3.3-70b-versatile" in cheap
