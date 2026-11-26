"""Goodhart anchor: config/providers.yaml MUST load via HAL's REAL loader.

Origin: 2026-06-23. A malformed ``ollama-local`` block was appended as a YAML
sequence item (``- name:``) under the ``routing:`` mapping (commit 4e04fec5).
HAL does NOT use PyYAML for this file -- its loader is the hand-rolled
``lib.hal.registry._load_yaml_config`` (``_parse_simple_yaml``), which is
lenient: instead of rejecting the bad shape it mis-parses ``routing`` into a
list, and ``_parse_routing_entry`` then raises ``AttributeError('list' object
has no attribute 'get')``. ``_load_config`` had no try/except around it, so the
HAL server/daemon crashed at boot on every node that pulled main (reverted in
6589f3d7).

PyYAML-valid is NOT registry-valid, so these tests run the REAL loader. The
negative test reproduces the break against the ACTUAL config file (not a
hand-written stand-in) so it always exercises the real parse path, and asserts
the SPECIFIC error message so a different failure mode cannot satisfy it.
"""
from pathlib import Path

import pytest

from lib.hal.registry import _load_yaml_config

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PROVIDERS = _REPO_ROOT / "config" / "providers.yaml"

# Skip (not error) if the config is absent from this tree -- keeps collection
# green on a partial checkout instead of failing at run time.
pytestmark = pytest.mark.skipif(
    not _PROVIDERS.exists(),
    reason="config/providers.yaml not present in this tree",
)


def test_canonical_providers_yaml_loads_via_real_loader():
    """Loader correctness: the canonical config parses via HAL's own loader and
    yields a usable routing config.

    Content-agnostic on purpose -- it asserts the loader produces *a* non-empty
    provider set and *a* usable routing/fallback chain, with no coupling to the
    exact provider names or count, so it tests the LOADER (not the config's
    contents) and does not false-fail when a provider is legitimately added,
    removed, or renamed.
    """
    configs, routing = _load_yaml_config(str(_PROVIDERS))

    assert configs, "providers.yaml produced zero providers (loader returned nothing)"
    assert routing is not None, "routing section failed to parse"
    assert routing.fallback_chain, "routing.fallback_chain is empty"


def test_malformed_sequence_under_routing_is_rejected(tmp_path):
    """Mutation guard reproducing the exact 2026-06-23 break against the REAL file.

    Appends a ``- name:`` sequence item after the ``routing:`` mapping -- exactly
    what commit 4e04fec5 did. HAL's parser mis-parses routing into a list and
    ``_parse_routing_entry`` raises ``AttributeError: 'list' object has no
    attribute 'get'``. Asserting the SPECIFIC message pins the exact failure
    path: a different ``AttributeError`` (e.g. a provider missing ``name``) or an
    earlier parse-stage rejection would not match, so a loader change that alters
    this failure mode fails here.
    """
    broken = _PROVIDERS.read_text() + (
        "  - name: ollama-local\n"
        "    type: openai_compat\n"
        "    base_url: http://localhost:11434/v1\n"
    )
    bad = tmp_path / "providers.yaml"
    bad.write_text(broken)

    with pytest.raises(AttributeError, match="has no attribute 'get'"):
        _load_yaml_config(str(bad))
