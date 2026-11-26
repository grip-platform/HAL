"""HAL Adaptive num_ctx wrapper (Phase γ W6, H-CTX).

Pure formula for per-dispatch context-window sizing:
    num_ctx = min(prompt_tokens + expected_response + safety_margin, model_max)

Motivates a 3–10× speedup on local models at small prompt sizes, since
most Ollama/MLX decoders allocate KV cache at num_ctx on every call —
shrinking it when not needed is free speed.

PR #174 (``num_ctx_policy`` module) landed the base policy; this
thin wrapper is the integration point that ALL local dispatches call.

Hypothesis H-CTX: 4K-ctx Gemma runs 3× faster than 32K-ctx on small
queries; no accuracy regression on known prompts.

Public API:
    compute_num_ctx(prompt_tokens, expected_response, *, model_max,
                    safety_margin=512, floor=2048) -> int
"""
from __future__ import annotations


DEFAULT_SAFETY_MARGIN = 512
DEFAULT_FLOOR = 2048  # don't allocate below this — rounding to a sane minimum


def compute_num_ctx(
    prompt_tokens: int,
    expected_response: int,
    *,
    model_max: int,
    safety_margin: int = DEFAULT_SAFETY_MARGIN,
    floor: int = DEFAULT_FLOOR,
) -> int:
    """Return the per-dispatch num_ctx for a local model.

    Guarantees:
      * Never exceeds ``model_max``.
      * Never dips below ``floor`` (avoids KV cache thrashing on tiny prompts).
      * Always >= prompt_tokens + expected_response + safety_margin when
        that sum is within bounds.
      * Invalid inputs (negatives) are clamped to 0 before summing so a
        caller typo can't produce nonsense.
    """
    if model_max <= 0:
        raise ValueError("model_max must be positive")
    pt = max(0, int(prompt_tokens))
    er = max(0, int(expected_response))
    sm = max(0, int(safety_margin))
    ideal = pt + er + sm
    if ideal <= floor:
        return min(floor, model_max)
    return min(ideal, model_max)


__all__ = ["DEFAULT_FLOOR", "DEFAULT_SAFETY_MARGIN", "compute_num_ctx"]
