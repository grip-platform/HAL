"""Registry model listings enriched from the static catalogue (issue #509, item 5).

Provider /models endpoints return only ids, so registry.list_models() used to
emit ModelInfo entries with context_window=0 and costs 0.0 even for models the
static catalogue (lib/hal/model_catalog.py) fully describes — three surfaces,
one model, no cross-check. Listings now fill zero-valued fields from the
catalogue on id match; provider-supplied non-zero values are never overwritten.

Deterministic — no network (fake provider object).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lib.hal.registry import ProviderRegistry, _enrich_from_catalog  # noqa: E402
from lib.hal.types import ModelInfo  # noqa: E402


def test_known_model_gains_catalog_metadata():
    out = _enrich_from_catalog(ModelInfo(id="kimi-k3", provider="moonshot"))
    assert out.context_window == 1_048_576
    assert out.max_output_tokens == 1_048_576
    assert out.cost_per_mtok_input == 3.0
    assert out.cost_per_mtok_output == 15.0


def test_unknown_model_stays_untouched():
    out = _enrich_from_catalog(ModelInfo(id="mystery-model-9", provider="wat"))
    assert out.context_window == 0
    assert out.cost_per_mtok_input == 0.0 and out.cost_per_mtok_output == 0.0


def test_provider_supplied_values_never_overwritten():
    out = _enrich_from_catalog(ModelInfo(
        id="kimi-k3", provider="moonshot",
        context_window=4096, cost_per_mtok_input=9.9))
    assert out.context_window == 4096, "provider truth must win over catalogue"
    assert out.cost_per_mtok_input == 9.9
    # Zero-valued fields still fill from the catalogue.
    assert out.cost_per_mtok_output == 15.0


class _FakeProvider:
    def list_models(self):
        return [ModelInfo(id="kimi-k3", provider="moonshot"),
                ModelInfo(id="mystery-model-9", provider="moonshot")]


def test_list_models_applies_enrichment_per_provider_and_globally():
    reg = ProviderRegistry()
    # Hermetic: replace the WHOLE provider map — the bare constructor
    # auto-loads real providers whose list_models() would hit the network.
    reg._providers = {"moonshot": _FakeProvider()}

    scoped = {m.id: m for m in reg.list_models("moonshot")}
    assert scoped["kimi-k3"].context_window == 1_048_576
    assert scoped["mystery-model-9"].context_window == 0

    everything = {m.id: m for m in reg.list_models()}
    assert everything["kimi-k3"].cost_per_mtok_output == 15.0
