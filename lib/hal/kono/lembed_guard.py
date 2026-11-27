"""Input guard for sqlite-lembed — prevents SIGSEGV on pathological inputs.

Background
----------
HAL issue #199 surfaced one SIGSEGV class: NULL-deref in ``embed_single`` when
``temp.lembed_models`` is unprimed. That was hardened by PR #203 (per-
connection model registration). Subsequent W3 benchmark harness (investigation
on 2026-04-18) reproduced a *distinct* SIGSEGV class on pathological inputs:

    * ``"a" * 1100`` → SIGSEGV
    * ``"Lorem ipsum..." * 100`` (5700 chars) → SIGSEGV

Both are low-entropy or overly long inputs. Matches upstream issues
`sqlite-lembed #7 <https://github.com/asg017/sqlite-lembed/issues/7>`_ and
`#20 <https://github.com/asg017/sqlite-lembed/issues/20>`_ (both unresolved
since 2024). PR #203's registration guard does NOT cover this class.

This module ships a multi-signal input guard. It is the Phase-0a hardening
shipped by the apsw-lembed-stack investigation sprint (W5). Phase 1 — full
replacement of sqlite-lembed with an Ollama/llama-cpp-python embedder — is
tracked separately and may ultimately delete this module.

Design
------
* Pure-Python, no C deps, no network.
* Complexity-class: O(n) in the worst case with small constants.
* One public function: :func:`is_likely_crash_input`.
* Three independent signals; any one triggers the guard.

The guard is intentionally conservative. A caller that trips it should
skip the lembed call and do without that embedding, rather than attempting
it. (The semantic response cache once fell back to a bag-of-words embedder
here; since HAL#965 it no longer embeds through this path at all.)
"""
from __future__ import annotations

# Thresholds — tuned from W3 crash reproduction.
# See investigation/w3-benchmark-results.md for the empirical data.
#
# Observed unique-char ratios:
#   "a" * 1100            -> 0.0009  (CRASH)
#   "Lorem..." * 100      -> 0.0033  (CRASH, 5700 chars)
#   English * 6 (1062 ch) -> 0.0386  (SAFE)
#
# Threshold 0.02 separates CRASH from SAFE with ~5x margin on either side.
# Max repeat run:
#   "a" * 1100 -> 1100 (CRASH)
#   English    ->    2 (SAFE)
_LEN_THRESHOLD = 900
_UNIQUE_RATIO_THRESHOLD = 0.02
_MAX_REPEAT_RUN = 200


def is_likely_crash_input(text: str) -> bool:
    """Return True if this text is likely to SIGSEGV sqlite-lembed's embed_single.

    Rejects short inputs below the length threshold outright (they are never
    crash candidates in the W3 reproduction). For longer inputs, returns True
    if either the unique-character ratio is below the entropy threshold or the
    longest single-character run exceeds the repeat threshold.

    Args:
        text: Input string destined for ``SELECT lembed(...)``.

    Returns:
        True if the guard should reject this input; False if it's safe.
    """
    if len(text) < _LEN_THRESHOLD:
        return False

    # Signal 1: Low entropy — few unique characters across a long body.
    # Catches ``"a" * N`` and ``"abc" * N/3`` style inputs.
    if (len(set(text)) / len(text)) < _UNIQUE_RATIO_THRESHOLD:
        return True

    # Signal 2: Long single-character run — catches "aaaa...aaaa" embedded in
    # otherwise-varied text.
    if _max_repeat_run(text) > _MAX_REPEAT_RUN:
        return True

    return False


def _max_repeat_run(text: str) -> int:
    """Longest run of the same character. O(n) single pass."""
    if not text:
        return 0
    best = current = 1
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            current += 1
            if current > best:
                best = current
        else:
            current = 1
    return best


class LembedInputTooRisky(ValueError):
    """Raised when input trips the guard. Callers should skip the lembed call
    and do without that embedding, or use another embedder backend.
    """
