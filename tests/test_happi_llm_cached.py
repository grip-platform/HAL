"""Goodhart-proof tests for the happi ``llm.cached`` cmd (prompt-fold cache).

Driven through the REAL ``runtime.run`` (StringIO in/out) -- behaviour, not
mocks of the runtime. HERMETIC: the provider is faked via the documented
``make_grip_sentinel`` dependency-injection seam in ``cmds_infer`` (no
network, no keychain), the feature flag state is reset per test against a
tmp features.json, and the cache lives in a per-test tmp dir via the
``GRIP_PROMPT_FOLD_CACHE_DIR`` env var.

Verifies:
  * flag OFF (default) -> pure passthrough: provider called every dispatch,
    ZERO cache files created, usage carries cache_route="disabled" only
  * flag ON first run  -> cache_route="fresh", provider called once, stored
  * flag ON second identical prompt -> cache_route="exact_hit", provider NOT
    called again, no new record appended
  * invalid risk_class -> plain-language parse_error event, never a traceback
  * provider raises on a miss -> the normal infer error surfaces unchanged
    and nothing is stored
  * shadow-safety: loading the GRIP seam module leaves HAL's own
    ``lib.hal.happi.envelope`` namespace fully intact
"""
import io
import importlib
import json
import sys
from pathlib import Path

import pytest

import lib.hal.features as features
import lib.hal.happi.cmds_cache as cmds_cache
import lib.hal.happi.cmds_infer as cmds_infer
from lib.hal.happi import runtime
from lib.hal.happi.envelope import HappiEnvelope

_MODEL = "openai/fake-model"
_FLAG_ON = '{"prompt_fold_cache": true}'

# These four tests exercise the GRIP seam: cmds_cache loads PromptFoldCache from
# <grip>/lib/prompt_fold_cache.py at runtime. On a machine with no GRIP install
# — a bare CI runner, for instance — that file cannot exist, so the tests cannot
# run and their failure would say nothing about HAL. The guard reuses
# cmds_cache._grip_dir() rather than re-deriving the path, so it can never drift
# from the resolver it is guarding.
_GRIP_FOLD_CACHE = cmds_cache._grip_dir() / "lib" / "prompt_fold_cache.py"
requires_grip_fold_cache = pytest.mark.skipif(
    not _GRIP_FOLD_CACHE.exists(),
    reason=("needs GRIP's prompt_fold_cache.py at %s — set CLAUDE_DIR to a GRIP "
            "install to run it" % _GRIP_FOLD_CACHE),
)


class _FakeResponse:
    def __init__(self, text):
        self.text = text
        self.provider = "openai"
        self.model = "fake-model"


class _FakeSentinel:
    """Records every prompt it serves; optionally raises (provider outage)."""

    def __init__(self, calls, *, text="cached-answer", raise_exc=None):
        self._calls = calls
        self._text = text
        self._raise = raise_exc

    def dispatch_with_history(self, model_address, prompt):
        self._calls.append(prompt)
        if self._raise is not None:
            raise self._raise
        return _FakeResponse(self._text)


@pytest.fixture
def cache_env(tmp_path, monkeypatch):
    """Hermetic flag + cache state: features reset against a tmp features.json
    (no persisted machine state can leak in), env override cleared, and the
    prompt-fold cache pointed at a per-test tmp dir."""
    monkeypatch.setattr(features, "_FEATURES_PATH",
                        str(tmp_path / "features.json"))
    monkeypatch.setattr(features, "_cache", {})
    monkeypatch.setattr(features, "_overrides", {})
    monkeypatch.setattr(features, "_loaded", False)
    monkeypatch.delenv("HAL_FEATURE_OVERRIDES", raising=False)
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("GRIP_PROMPT_FOLD_CACHE_DIR", str(cache_dir))
    return cache_dir


def _mock_provider(monkeypatch, calls, **kwargs):
    monkeypatch.setattr(cmds_infer, "make_grip_sentinel",
                        lambda: _FakeSentinel(calls, **kwargs))


def _run(envelope: dict) -> list[dict]:
    out = io.StringIO()
    rc = runtime.run(io.StringIO(json.dumps(envelope)), out)
    assert rc == 0
    return [json.loads(l) for l in out.getvalue().splitlines() if l.strip()]


def _envelope(prompt="what is the answer?", **flags) -> dict:
    return {
        "v": "happi/1.1", "id": "lc1", "cmd": "llm.cached",
        "flags": {"model": _MODEL, "prompt": prompt, **flags},
    }


def _records(cache_dir: Path) -> list[str]:
    path = cache_dir / "records.jsonl"
    if not path.exists():
        return []
    return [l for l in path.read_text().splitlines() if l.strip()]


def test_flag_off_default_is_pure_passthrough(cache_env, monkeypatch):
    """(a) Default OFF: provider called on EVERY dispatch, zero cache files,
    and the stream is a plain infer plus the additive disabled marker."""
    calls: list[str] = []
    _mock_provider(monkeypatch, calls)

    for _ in range(2):
        events = _run(_envelope())
        assert [e["type"] for e in events] == ["started", "delta", "completed"]
        assert events[1]["text"] == "cached-answer"
        usage = events[2]["usage"]
        assert usage["cache_route"] == "disabled"
        assert usage["provider"] == "openai"
        assert usage["requested"] == _MODEL
        assert "exact_key" not in usage

    assert len(calls) == 2                       # no caching whatsoever
    assert not cache_env.exists()                # zero cache reads OR writes


@requires_grip_fold_cache
def test_flag_on_first_run_is_fresh(cache_env, monkeypatch):
    """(b) ON via HAL_FEATURE_OVERRIDES: first dispatch is a fresh route,
    provider called exactly once, decision stored."""
    monkeypatch.setenv("HAL_FEATURE_OVERRIDES", _FLAG_ON)
    calls: list[str] = []
    _mock_provider(monkeypatch, calls)

    assert "llm.cached" in runtime.SUPPORTED_COMMANDS
    events = _run(_envelope())
    assert [e["type"] for e in events] == ["started", "delta", "completed"]
    assert events[1]["text"] == "cached-answer"
    usage = events[2]["usage"]
    assert usage["cache_route"] == "fresh"
    assert usage["provider"] == "openai"
    assert len(usage["exact_key"]) == 64
    assert int(usage["exact_key"], 16) >= 0      # hex sha256
    assert len(calls) == 1
    assert len(_records(cache_env)) == 1         # one stored record


@requires_grip_fold_cache
def test_flag_on_second_identical_prompt_is_exact_hit(cache_env, monkeypatch):
    """(c) ON: identical prompt -> exact_hit served from the store; the
    provider mock is NOT called again and no new record is appended."""
    monkeypatch.setenv("HAL_FEATURE_OVERRIDES", _FLAG_ON)
    calls: list[str] = []
    _mock_provider(monkeypatch, calls)

    first = _run(_envelope())
    second = _run(_envelope())

    assert first[2]["usage"]["cache_route"] == "fresh"
    usage = second[2]["usage"]
    assert usage["cache_route"] == "exact_hit"
    assert usage["exact_key"] == first[2]["usage"]["exact_key"]
    assert second[1]["text"] == "cached-answer"  # bytes round-trip intact
    assert len(calls) == 1                       # provider NOT called on hit
    assert len(_records(cache_env)) == 1         # hits never append


def test_invalid_risk_class_is_plain_language_error(cache_env, monkeypatch):
    """(d) Invalid risk_class -> a plain-language parse_error event (the run
    completes rc=0, so no traceback escaped) and the cache is untouched."""
    monkeypatch.setenv("HAL_FEATURE_OVERRIDES", _FLAG_ON)
    calls: list[str] = []
    _mock_provider(monkeypatch, calls)

    events = _run(_envelope(risk_class="extreme"))
    err = [e for e in events if e["type"] == "error"]
    assert err and err[0]["code"] == "parse_error"
    assert "risk_class" in err[0]["message"]
    assert "'extreme'" in err[0]["message"]
    assert "low" in err[0]["message"]            # names the valid options
    assert not any(e["type"] in ("delta", "completed") for e in events)
    assert calls == []                           # provider never reached
    assert not cache_env.exists()


@requires_grip_fold_cache
def test_provider_raises_on_miss_error_surfaces_unchanged(cache_env,
                                                          monkeypatch):
    """(e) A provider failure on a cache miss surfaces as the NORMAL infer
    error (same message shape, same code) and nothing is stored."""
    monkeypatch.setenv("HAL_FEATURE_OVERRIDES", _FLAG_ON)
    calls: list[str] = []
    _mock_provider(monkeypatch, calls,
                   raise_exc=RuntimeError("provider down"))

    events = _run(_envelope())
    err = [e for e in events if e["type"] == "error"]
    assert err and err[0]["code"] == "runtime_error"
    assert "infer: dispatch failed" in err[0]["message"]
    assert "provider down" in err[0]["message"]
    assert not any(e["type"] in ("delta", "completed") for e in events)
    assert len(calls) == 1
    assert _records(cache_env) == []             # failed dispatch never stored


@requires_grip_fold_cache
def test_shadow_safety_hal_namespace_intact_after_seam_load():
    """(f) Loading the GRIP seam module must leave HAL's own ``lib`` package
    untouched: ``lib.hal.happi.envelope`` still resolves to HAL's class, the
    ``lib`` package still lives under the HAL tree, and any sys.path entry
    the GRIP module added for its own imports has been removed again."""
    grip_lib = str((cmds_cache._grip_dir() / "lib"
                    / "prompt_fold_cache.py").resolve().parent)
    was_on_path = grip_lib in sys.path

    module = cmds_cache._load_prompt_fold_cache()
    assert callable(module.run_or_get)
    assert callable(module.PromptCacheRequest)
    assert cmds_cache._load_prompt_fold_cache() is module   # cached seam

    envelope_mod = importlib.import_module("lib.hal.happi.envelope")
    assert envelope_mod.HappiEnvelope is HappiEnvelope      # HAL's, unshadowed

    hal_lib = Path(list(importlib.import_module("lib").__path__)[0]).resolve()
    assert hal_lib == Path(runtime.__file__).resolve().parents[2]
    if not was_on_path:
        assert grip_lib not in sys.path                     # no lingering path
