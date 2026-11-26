"""Two-sided Goodhart-proof tests for lib.hal.voice.usage."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.hal.voice.usage import (  # noqa: E402
    DEFAULT_USAGE_PATH,
    FREE_TIER_LIMITS,
    UsageCounter,
    UsageRow,
    WARN_THRESHOLD_PCT,
)


@pytest.fixture
def usage_file(tmp_path: Path) -> Path:
    return tmp_path / "usage.jsonl"


@pytest.fixture
def counter(usage_file: Path) -> UsageCounter:
    return UsageCounter(path=usage_file)


def test_default_usage_path_under_hal_state() -> None:
    assert str(DEFAULT_USAGE_PATH).endswith(".hal/state/voice/usage.jsonl")


def test_warn_threshold_is_80_percent() -> None:
    assert WARN_THRESHOLD_PCT == 80.0


def test_free_tier_limits_contains_tier3_providers() -> None:
    assert FREE_TIER_LIMITS["google_cloud_tts_free"] == 1_000_000
    assert FREE_TIER_LIMITS["gemini_tts_free"] is None
    assert FREE_TIER_LIMITS["piper"] is None
    assert FREE_TIER_LIMITS["macos_say"] is None


def test_record_round_trip(counter: UsageCounter) -> None:
    counter.record("google_cloud_tts_free", 250)
    assert counter.chars_this_month("google_cloud_tts_free") == 250


def test_record_sums_across_multiple_calls(counter: UsageCounter) -> None:
    counter.record("google_cloud_tts_free", 100)
    counter.record("google_cloud_tts_free", 250)
    counter.record("google_cloud_tts_free", 50)
    assert counter.chars_this_month("google_cloud_tts_free") == 400


def test_chars_this_month_zero_for_unknown_provider(counter: UsageCounter) -> None:
    assert counter.chars_this_month("never-used") == 0


def test_chars_this_month_ignores_other_provider(counter: UsageCounter) -> None:
    counter.record("google_cloud_tts_free", 500)
    counter.record("gemini_tts_free", 1000)
    assert counter.chars_this_month("google_cloud_tts_free") == 500
    assert counter.chars_this_month("gemini_tts_free") == 1000


def test_chars_this_month_ignores_other_months(usage_file: Path) -> None:
    usage_file.parent.mkdir(parents=True, exist_ok=True)
    prior = json.dumps(
        {
            "ts": "2026-01-15T00:00:00Z",
            "ym": "2026-01",
            "provider": "google_cloud_tts_free",
            "chars": 999_000,
            "profile": "default",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    usage_file.write_text(prior + "\n")
    counter = UsageCounter(path=usage_file)
    assert counter.chars_this_month("google_cloud_tts_free") == 0


def test_chars_this_month_targeted_by_ym(counter: UsageCounter) -> None:
    counter.path.parent.mkdir(parents=True, exist_ok=True)
    prior = json.dumps(
        {
            "ts": "2026-01-15T00:00:00Z",
            "ym": "2026-01",
            "provider": "google_cloud_tts_free",
            "chars": 750_000,
            "profile": "default",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    with counter.path.open("a") as f:
        f.write(prior + "\n")
    assert counter.chars_this_month("google_cloud_tts_free", ym="2026-01") == 750_000


def test_malformed_line_tolerated(counter: UsageCounter) -> None:
    counter.path.parent.mkdir(parents=True, exist_ok=True)
    counter.path.write_text("garbage\n")
    counter.record("google_cloud_tts_free", 100)
    assert counter.chars_this_month("google_cloud_tts_free") == 100


def test_record_persists_profile_field(counter: UsageCounter) -> None:
    counter.record("google_cloud_tts_free", 42, profile="arnold")
    lines = counter.path.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["profile"] == "arnold"
    assert record["chars"] == 42
    assert record["provider"] == "google_cloud_tts_free"


def test_summary_contains_row_per_known_provider(counter: UsageCounter) -> None:
    rows = counter.summary()
    names = {r.provider for r in rows}
    assert {"google_cloud_tts_free", "gemini_tts_free", "piper", "macos_say"}.issubset(names)


def test_summary_zero_chars_when_provider_unused(counter: UsageCounter) -> None:
    rows = counter.summary()
    gct = next(r for r in rows if r.provider == "google_cloud_tts_free")
    assert gct.chars_this_month == 0
    assert gct.percent_used == 0.0


def test_summary_percent_used_calculation(counter: UsageCounter) -> None:
    counter.record("google_cloud_tts_free", 500_000)
    rows = counter.summary()
    gct = next(r for r in rows if r.provider == "google_cloud_tts_free")
    assert abs(gct.percent_used - 50.0) < 1e-9


def test_summary_warn_threshold_flag(counter: UsageCounter) -> None:
    counter.record("google_cloud_tts_free", 799_000)
    rows = counter.summary()
    assert not next(r for r in rows if r.provider == "google_cloud_tts_free").over_warn_threshold

    counter.record("google_cloud_tts_free", 2_000)  # cumulative 801_000
    rows = counter.summary()
    assert next(r for r in rows if r.provider == "google_cloud_tts_free").over_warn_threshold


def test_summary_provider_with_no_limit_has_none_percent(counter: UsageCounter) -> None:
    rows = counter.summary()
    piper = next(r for r in rows if r.provider == "piper")
    assert piper.free_tier_limit is None
    assert piper.percent_used is None
    assert piper.over_warn_threshold is False


def test_summary_includes_unknown_providers_as_extras(counter: UsageCounter) -> None:
    counter.record("mystery_provider", 500)
    rows = counter.summary()
    extras = [r for r in rows if r.provider == "mystery_provider"]
    assert len(extras) == 1
    assert extras[0].chars_this_month == 500
    assert extras[0].free_tier_limit is None


def test_summary_custom_limits_argument(counter: UsageCounter) -> None:
    counter.record("piper", 999)
    rows = counter.summary(limits={"piper": 1000})
    piper = next(r for r in rows if r.provider == "piper")
    assert piper.free_tier_limit == 1000
    assert abs(piper.percent_used - 99.9) < 1e-6
    assert piper.over_warn_threshold is True


def test_usage_row_is_frozen() -> None:
    row = UsageRow(provider="p", chars_this_month=10, free_tier_limit=100, percent_used=10.0)
    with pytest.raises(Exception):
        row.chars_this_month = 999  # type: ignore[misc]


def test_usage_row_over_warn_threshold_boundaries() -> None:
    below = UsageRow(provider="x", chars_this_month=79, free_tier_limit=100, percent_used=79.9)
    at = UsageRow(provider="x", chars_this_month=80, free_tier_limit=100, percent_used=80.0)
    above = UsageRow(provider="x", chars_this_month=99, free_tier_limit=100, percent_used=99.0)
    assert below.over_warn_threshold is False
    assert at.over_warn_threshold is True
    assert above.over_warn_threshold is True


def test_usage_row_none_percent_never_over_threshold() -> None:
    row = UsageRow(provider="piper", chars_this_month=10**9, free_tier_limit=None, percent_used=None)
    assert row.over_warn_threshold is False
