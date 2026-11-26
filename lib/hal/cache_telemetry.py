"""HAL Cache Telemetry — per-call prompt-cache JSONL logger.

Appends one JSON line per LLM call to ~/.hal/state/hal/cache_calls.jsonl.
Used by cache_idle_report.py to analyse whether cached prefixes actually
idle beyond the ~5 min Anthropic TTL and resume cold (cache_creation > 0
after a gap), which determines if a prompt-cache keepalive is worth wiring.

Hypothesis H793: if >=10 % of cached-prefix resumes are cold-after-idle,
a keepalive is worth the complexity; <10 % => negligible.

Design constraints:
- Pure stdlib — no external dependencies.
- Fail-soft: every I/O path is wrapped in try/except; a telemetry failure
  MUST NOT raise into the inference path.
- Opt-out: set HAL_DISABLE_CACHE_TELEMETRY=1 (or true/yes) to suppress all
  writes.
- Thread-safe: file is opened in append mode per call; GIL + O_APPEND are
  sufficient on POSIX.  No lock needed for append-only writes.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path


_TELEMETRY_DIR = Path.home() / ".hal" / "state" / "hal"
_TELEMETRY_FILE = _TELEMETRY_DIR / "cache_calls.jsonl"


def _is_disabled() -> bool:
    return os.environ.get("HAL_DISABLE_CACHE_TELEMETRY", "").lower() in ("1", "true", "yes")


def log_cache_call(
    *,
    model: str,
    provider: str,
    prefix_hash: str,
    cache_creation: int,
    cache_read: int,
    input_tokens: int,
    output_tokens: int,
    ts: float | None = None,
) -> None:
    """Append one telemetry record to cache_calls.jsonl.

    All parameters are keyword-only.  ``ts`` defaults to ``time.time()``
    when omitted.  The function is fail-soft: it returns silently on any
    I/O or serialisation error.

    Args:
        model:          Model ID as returned by the provider (e.g. "claude-opus-4-7").
        provider:       Provider name (e.g. "anthropic", "groq").
        prefix_hash:    Caller-supplied 16-char hex digest identifying the cached
                        prefix (sha256 of system prompt + tool defs).
        cache_creation: Tokens written into the cache this call (Anthropic
                        usage.cache_creation_input_tokens).
        cache_read:     Tokens read from cache this call
                        (usage.cache_read_input_tokens).
        input_tokens:   Total input tokens (including cached).
        output_tokens:  Output / completion tokens.
        ts:             Unix timestamp; defaults to time.time().
    """
    if _is_disabled():
        return
    try:
        record = {
            "ts": ts if ts is not None else time.time(),
            "model": model,
            "provider": provider,
            "prefix_hash": prefix_hash,
            "cache_creation": cache_creation,
            "cache_read": cache_read,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        _TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
        with _TELEMETRY_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    except Exception:  # noqa: BLE001 — intentional broad catch; telemetry must never raise
        pass


def compute_prefix_hash(system_prompt: str, tools_repr: str = "") -> str:
    """Return a 16-char hex prefix hash from system_prompt + tools_repr.

    Stable across calls as long as the system prompt and tool definitions
    do not change — which is the condition under which Anthropic's cache
    TTL applies.
    """
    raw = (system_prompt + tools_repr).encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()[:16]


__all__ = ["log_cache_call", "compute_prefix_hash"]
