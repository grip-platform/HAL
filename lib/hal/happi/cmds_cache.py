"""HAPPI ``llm.cached`` cmd -- prompt-fold-cached inference (config-gated).

Wires GRIP's PromptFoldCache (``<grip>/lib/prompt_fold_cache.py``) in front of
the SAME sentinel machinery ``infer`` uses. The cache module folds the raw
prompt (Banach fold canonicalisation), looks up an exact SHA-256 key, applies
the policy-gated semantic near-match, and only then dispatches fresh -- every
decision replayable from its record.

Feature gate: ``features.is_enabled("prompt_fold_cache", False)``. Absent key
or false -> the handler is a PURE PASSTHROUGH to the fresh inference path with
NO cache reads or writes -- byte-identical behaviour to a plain ``infer``,
plus one additive ``cache_route: "disabled"`` field on the completed event's
usage. When ON, the completed usage additionally carries ``cache_route``
(fresh / exact_hit / semantic_hit) and ``exact_key`` -- additive fields old
clients can ignore.

Import seam: GRIP's module bare-imports its siblings (``fold_compress``,
``claude_dir``) after prepending its own lib/ to ``sys.path``. HAL has its own
top-level ``lib`` package, so the GRIP module is loaded via
``importlib.util.spec_from_file_location`` under an isolated module name, and
any sys.path entry the module added for its own imports is removed afterwards
(shadow-safety: HAL's ``lib.hal.*`` namespace is never shadowed -- proven by
``tests/test_happi_llm_cached.py``).

Provider logic is NOT duplicated: the fresh path delegates to
``cmds_infer._resolve_response`` (which builds the sentinel via the documented
``make_grip_sentinel`` dependency-injection seam -- tests monkeypatch it).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator

from lib.hal import features
from lib.hal.happi.cmds_infer import _extract_prompt, _resolve_response
from lib.hal.happi.envelope import HappiEnvelope
from lib.hal.happi.events import emit_completed, emit_delta, emit_error
from lib.hal.raillm_ladder import GRIP_LADDER_ADDRESSES
from lib.hal.registry import parse_model_address

FEATURE_FLAG = "prompt_fold_cache"
_SEAM_MODULE_NAME = "grip_prompt_fold_cache"

# Wire-contract mirror of prompt_fold_cache.RISK_CLASSES, kept local so the
# flag-OFF passthrough never has to load the GRIP module just to validate.
# Drift backstop: run_or_get's own ValueError is caught and surfaced as a
# plain-language error event (never a traceback), so a divergence between the
# two sets can only ever tighten, not crash.
_VALID_RISK_CLASSES = frozenset({"low", "medium", "high"})


class _FreshDispatchError(Exception):
    """Carries the (message, code) pair from a failed fresh dispatch out of
    ``run_or_get`` so the normal ``infer`` error surfaces unchanged and
    nothing is stored in the cache."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


def _grip_dir() -> Path:
    """GRIP install dir: ``$CLAUDE_DIR`` when set, else ``~/.claude``."""
    raw = os.environ.get("CLAUDE_DIR", "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".claude"


def _load_prompt_fold_cache():
    """Load GRIP's prompt_fold_cache module via an isolated importlib seam.

    ``spec_from_file_location`` keeps HAL's own top-level ``lib`` package out
    of the module-name space. The GRIP module prepends its lib/ dir to
    ``sys.path`` at exec time for its OWN sibling imports (``fold_compress``,
    ``claude_dir``); that entry is removed again here (unless it pre-existed)
    so no GRIP module can ever shadow a later HAL or stdlib import. The
    siblings stay resolvable because exec left them in ``sys.modules``.
    """
    cached = sys.modules.get(_SEAM_MODULE_NAME)
    if cached is not None:
        return cached
    path = (_grip_dir() / "lib" / "prompt_fold_cache.py").resolve()
    spec = importlib.util.spec_from_file_location(_SEAM_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"llm.cached: cannot load GRIP prompt-fold cache from {path} -- "
            "set CLAUDE_DIR to the GRIP install directory"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_SEAM_MODULE_NAME] = module
    grip_lib = str(path.parent)
    was_on_path = grip_lib in sys.path
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_SEAM_MODULE_NAME, None)
        raise
    finally:
        if not was_on_path:
            while grip_lib in sys.path:
                sys.path.remove(grip_lib)
    return module


def _envelope_sha256(envelope: HappiEnvelope) -> str:
    """Deterministic provenance hash of the envelope's canonical dict form."""
    canonical = json.dumps(envelope.to_dict(), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _fresh_usage(model_address: str, response) -> dict[str, Any]:
    """Usage dict for a live provider response -- mirrors the ``infer``
    emission pattern (provider/model/requested + failover derived from the
    provider that actually answered vs the one requested)."""
    usage: dict[str, Any] = {
        "provider": response.provider,
        "model": response.model,
        "requested": model_address,
    }
    requested_provider, _ = parse_model_address(model_address)
    if response.provider and response.provider != requested_provider:
        usage["failover"] = {
            "requested": requested_provider,
            "served_by": response.provider,
        }
    return usage


def _dispatch_disabled(envelope: HappiEnvelope, model_address: str,
                       prompt: str) -> Iterator[str]:
    """Flag OFF: pure passthrough to the fresh path. No cache module load,
    no cache reads, no cache writes -- byte-identical to a plain ``infer``
    plus the additive ``cache_route: "disabled"`` marker."""
    response, error = _resolve_response(model_address, prompt, None)
    if error is not None:
        yield emit_error(envelope.id, error[0], code=error[1])
        return
    usage = _fresh_usage(model_address, response)
    usage["cache_route"] = "disabled"
    yield emit_delta(envelope.id, response.text)
    yield emit_completed(envelope.id, usage=usage)


def _emit_decision(envelope: HappiEnvelope, model_address: str, decision,
                   response) -> Iterator[str]:
    """Stream a cache decision: delta + completed with additive cache fields.

    A fresh route has a live ``response`` (provider/model/failover known); a
    cache hit has only the stored output bytes, so usage honestly carries just
    ``requested`` -- never a fabricated provider identity.
    """
    text = (decision.output or b"").decode("utf-8", errors="replace")
    if response is not None:
        usage = _fresh_usage(model_address, response)
    else:
        usage = {"requested": model_address}
    usage["cache_route"] = decision.route
    usage["exact_key"] = decision.exact_key
    yield emit_delta(envelope.id, text)
    yield emit_completed(envelope.id, usage=usage)


def _dispatch_enabled(envelope: HappiEnvelope, model_address: str, prompt: str,
                      risk_class: str, flags: dict[str, Any]) -> Iterator[str]:
    """Flag ON: fold -> exact -> gated semantic -> fresh dispatch + store.

    ``fresh_runner`` receives the FOLDED prompt bytes (the cache module's
    canonical form) and delegates to the same sentinel machinery ``infer``
    uses via ``_resolve_response``. A fresh-dispatch failure raises out of
    ``run_or_get`` (nothing stored) and surfaces as the unchanged ``infer``
    error. ``model_fingerprint`` is the resolved model address -- flags.model
    when the caller named one, else the ladder-primary identity.
    """
    try:
        cache = _load_prompt_fold_cache()
    except (OSError, RuntimeError, ImportError) as exc:
        yield emit_error(
            envelope.id,
            f"llm.cached: prompt-fold cache unavailable -- {exc}",
            code="runtime_error",
        )
        return
    request = cache.PromptCacheRequest(
        raw_prompt=prompt.encode("utf-8"),
        task_type=str(flags.get("task_type") or "general"),
        risk_class=risk_class,
        allow_semantic=flags.get("allow_semantic") is True,
    )
    captured: dict[str, Any] = {}

    def _fresh_runner(folded: bytes) -> bytes:
        folded_prompt = folded.decode("utf-8", errors="replace")
        response, error = _resolve_response(model_address, folded_prompt, None)
        if error is not None:
            raise _FreshDispatchError(error[0], error[1])
        captured["response"] = response
        return str(response.text or "").encode("utf-8")

    try:
        decision = cache.run_or_get(
            request,
            _fresh_runner,
            model_fingerprint=model_address,
            happi_envelope_sha256=_envelope_sha256(envelope),
        )
    except _FreshDispatchError as exc:
        yield emit_error(envelope.id, exc.message, code=exc.code)
        return
    except ValueError as exc:
        yield emit_error(envelope.id, f"llm.cached: {exc}", code="parse_error")
        return
    yield from _emit_decision(
        envelope, model_address, decision, captured.get("response"))


def dispatch_llm_cached(envelope: HappiEnvelope) -> Iterator[str]:
    """Dispatch one inference through the prompt-fold cache (config-gated).

    Coordinator: validate flags -> feature gate -> disabled passthrough or
    cached dispatch. Flags: ``prompt`` (required, or args[0]); ``task_type``
    (default "general"); ``risk_class`` (default "low"); ``allow_semantic``
    (default false, literal boolean); ``model`` (RAILLM address, default
    ladder primary). Yields only delta/completed/error -- the runtime emits
    ``started`` and (when ``flags.audit`` is true) the ``idr`` terminator.
    """
    prompt = _extract_prompt(envelope)
    if not prompt:
        yield emit_error(
            envelope.id,
            "llm.cached: requires a prompt (args[0] or flags.prompt)",
            code="parse_error",
        )
        return
    flags = envelope.flags or {}
    risk_class = str(flags.get("risk_class") or "low")
    if risk_class not in _VALID_RISK_CLASSES:
        yield emit_error(
            envelope.id,
            f"llm.cached: risk_class {risk_class!r} is not recognised -- use "
            f"one of: {', '.join(sorted(_VALID_RISK_CLASSES))}. The risk "
            "class decides whether a cached answer may be reused (low and "
            "medium allow exact reuse; high always runs fresh).",
            code="parse_error",
        )
        return
    model_address = str(flags.get("model") or GRIP_LADDER_ADDRESSES[0])
    if not features.is_enabled(FEATURE_FLAG, False):
        yield from _dispatch_disabled(envelope, model_address, prompt)
        return
    yield from _dispatch_enabled(envelope, model_address, prompt,
                                 risk_class, flags)


# Extension dispatch table -- merged by runtime._dispatch_core via the same
# .get() lookup pattern as V11_HANDLERS / INFER_HANDLERS (no if/elif chain).
# Kept OUT of cmds_v11.py so that module stays byte-parity with the canonical
# polyglot happi.md runtime (which has no `llm.cached` cmd).
CACHE_HANDLERS = {"llm.cached": dispatch_llm_cached}
CACHE_COMMANDS = frozenset(CACHE_HANDLERS)
