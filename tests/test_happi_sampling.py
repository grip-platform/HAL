"""Tests for the shared sampling-parameter resolver (lib/hal/happi/sampling.py).

The resolver is the single contract three provider adapters share for optional
`temperature` / `top_p` / `seed` flags. The load-bearing properties:

- ABSENT flags are OMITTED (backwards-compatible: provider default preserved).
- `temperature: 0.0` SURVIVES — it is the determinism value AGORA needs, and
  `0.0` is falsy, so a truthiness guard would wrongly drop it. This is the
  Goodhart anchor: the test fails if anyone re-introduces `if temperature:`.
- `seed` is gated by `allow_seed` (Anthropic Messages API has no seed field).
"""
from __future__ import annotations

from lib.hal.happi.sampling import resolve_sampling


# --- Absence / backwards-compatibility -------------------------------------

def test_empty_flags_yield_empty_dict():
    assert resolve_sampling({}) == {}


def test_unrelated_flags_ignored():
    assert resolve_sampling({"provider": "gemini", "max_tokens": 256}) == {}


def test_explicit_null_is_omitted():
    # A caller passing JSON null must behave like absence, not crash/coerce.
    assert resolve_sampling({"temperature": None, "top_p": None, "seed": None}) == {}


# --- temperature: the 0.0 Goodhart anchor ----------------------------------

def test_temperature_zero_is_preserved_not_dropped():
    # 0.0 is falsy; a truthiness guard would drop the one value that matters.
    out = resolve_sampling({"temperature": 0.0})
    assert out == {"temperature": 0.0}


def test_temperature_nonzero_float():
    assert resolve_sampling({"temperature": 0.7}) == {"temperature": 0.7}


def test_temperature_int_coerced_to_float():
    out = resolve_sampling({"temperature": 1})
    assert out["temperature"] == 1.0
    assert isinstance(out["temperature"], float)


# --- top_p ------------------------------------------------------------------

def test_top_p_resolved_as_float():
    out = resolve_sampling({"top_p": 0.9})
    assert out == {"top_p": 0.9}
    assert isinstance(out["top_p"], float)


# --- seed gating ------------------------------------------------------------

def test_seed_included_when_allowed():
    out = resolve_sampling({"seed": 42}, allow_seed=True)
    assert out == {"seed": 42}
    assert isinstance(out["seed"], int)


def test_seed_excluded_when_not_allowed():
    # Anthropic path: seed must never reach the body (API rejects it).
    out = resolve_sampling({"seed": 42}, allow_seed=False)
    assert "seed" not in out
    assert out == {}


def test_seed_string_coerced_to_int():
    out = resolve_sampling({"seed": "7"}, allow_seed=True)
    assert out == {"seed": 7}


# --- Malformed input is fail-safe (omitted, never raised) ------------------

def test_non_numeric_temperature_omitted_not_raised():
    # A garbage temperature must behave like absence (provider default), never
    # crash the inference with an uncaught ValueError mid-stream.
    assert resolve_sampling({"temperature": "hot"}) == {}


def test_non_numeric_top_p_omitted():
    assert resolve_sampling({"top_p": "wide"}) == {}


def test_non_numeric_seed_omitted():
    assert resolve_sampling({"seed": "not-an-int"}, allow_seed=True) == {}


def test_container_value_omitted_not_raised():
    # A list/dict (TypeError on float()/int()) is dropped, not propagated.
    assert resolve_sampling({"temperature": [1, 2], "seed": {"x": 1}}) == {}


def test_malformed_dropped_but_valid_siblings_kept():
    # One bad knob does not poison the good ones in the same envelope.
    out = resolve_sampling({"temperature": 0.0, "top_p": "bad", "seed": 7})
    assert out == {"temperature": 0.0, "seed": 7}


# --- Seed must be an exact integer (no silent truncation) ------------------

def test_fractional_float_seed_omitted_not_truncated():
    # 3.7 must NOT silently become 3 — a fractional seed is a caller error.
    assert resolve_sampling({"seed": 3.7}, allow_seed=True) == {}


def test_fractional_string_seed_omitted():
    assert resolve_sampling({"seed": "3.7"}, allow_seed=True) == {}


def test_integer_valued_float_seed_accepted():
    # 7.0 is integer-valued, so it resolves to int 7 (no information lost).
    out = resolve_sampling({"seed": 7.0}, allow_seed=True)
    assert out == {"seed": 7}
    assert isinstance(out["seed"], int)


def test_bool_seed_rejected():
    # bool is an int subclass; a boolean seed is nonsensical, not seed=1.
    assert resolve_sampling({"seed": True}, allow_seed=True) == {}


# --- combination ------------------------------------------------------------

def test_all_three_together_with_seed():
    out = resolve_sampling(
        {"temperature": 0.0, "top_p": 0.95, "seed": 7}, allow_seed=True
    )
    assert out == {"temperature": 0.0, "top_p": 0.95, "seed": 7}


def test_combination_drops_only_seed_when_disallowed():
    out = resolve_sampling(
        {"temperature": 0.0, "top_p": 0.95, "seed": 7}, allow_seed=False
    )
    assert out == {"temperature": 0.0, "top_p": 0.95}
