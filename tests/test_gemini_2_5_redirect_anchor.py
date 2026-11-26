"""Regression anchor: gemini-2.5-flash / gemini-2.5-pro never reappear in a
live call path this PR fixed.

Origin: GRIP#9389 (2026-11-24). ai.google.dev/gemini-api/docs/deprecations,
read 2026-11-24: "effective December 1, 2026, API requests to
`gemini-2.5-flash` will automatically route to [`gemini-3.5-flash-lite`]...
and be billed at [Gemini 3.5 Flash-Lite rates]"; the same sentence for
`gemini-2.5-pro` -> `gemini-3.8-flash`. Every live HAL call site that named
either old id was repointed to its redirect target the same day.

The regex excludes a DIFFERENT model that merely shares the "gemini-2.5-flash"
/ "gemini-2.5-pro" prefix (gemini-2.5-flash-lite, gemini-2.5-pro-preview-tts,
...) via a negative lookahead for a following hyphen -- those ids are out of
this fix's scope and still legitimately appear in config/model_prices.yaml,
lib/hal/cost_tracker.py's historical price rows (billing already-made calls
correctly) and elsewhere.

This is a regression ANCHOR over the specific files this PR touched, not a
repo-wide sweep: a sweep would also flag deliberately-untouched illustrative
docstrings (lib/hal/demo.py, lib/hal/model_compare.py, lib/hal/council_agent.py)
and documentation (skills/*.md) that describe the OLD id as history/example,
never call it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# A bare gemini-2.5-flash or gemini-2.5-pro, not part of a longer id
# (gemini-2.5-flash-lite, gemini-2.5-pro-preview-tts, ... are different models).
_FORBIDDEN = re.compile(r"gemini-2\.5-(?:flash|pro)(?!-)")

# Every live call path this PR repointed from gemini-2.5-flash/-pro to
# gemini-3.5-flash-lite/gemini-3.8-flash (GRIP#9389, 2026-11-24).
_LIVE_CALL_PATH_FILES = [
    "config/providers.yaml",
    "config/providers.windows.yaml",
    "config/model_catalog.yaml",
    "bin/hal",
    "lib/providers/gemini.py",
    "lib/hal_council_dispatch.py",
    "lib/hal/learned_lattice.py",
    "lib/hal/rsi.py",
    "lib/hal/free_tier_cascade.py",
    "lib/hal/code_gen.py",
    "lib/hal/raillm_ladder.py",
    "lib/hal/model_catalog.py",
    "lib/hal/gateway_fanout.py",
    "lib/hal/raillm.py",
]


def _live_lines(text: str) -> list[str]:
    """Every line with any `#`-comment stripped (YAML, Python and bash all
    use `#`, which covers every file this anchor scans) -- so the sunset
    notes this same PR writes, which quote the retired id as prose in a
    whole-line or trailing comment, do not trip their own anchor. A `#`
    inside a quoted string would be mishandled; none of the scanned files
    use one, and this is a regression anchor, not a general-purpose linter."""
    return [ln.split("#", 1)[0] for ln in text.splitlines()]


@pytest.mark.parametrize("relpath", _LIVE_CALL_PATH_FILES)
def test_old_gemini_id_absent_from_fixed_call_path(relpath):
    text = (REPO / relpath).read_text(encoding="utf-8")
    hits = [ln for ln in _live_lines(text) if _FORBIDDEN.search(ln)]
    assert not hits, (
        f"{relpath} names a retired Gemini 2.5 id in a live call path "
        f"(found {hits!r}) -- Google auto-routes gemini-2.5-flash to "
        "gemini-3.5-flash-lite and gemini-2.5-pro to gemini-3.8-flash from "
        "2026-12-01; name the model that actually answers.")


def test_the_pattern_still_matches_the_retired_bare_ids():
    """Goodhart guard: prove the regex can still fire, so a typo'd pattern
    that matches nothing can't make every test above pass vacuously."""
    assert _FORBIDDEN.search("gemini/gemini-2.5-flash")
    assert _FORBIDDEN.search('"gemini-2.5-pro"')


def test_the_pattern_excludes_the_different_sibling_models():
    """gemini-2.5-flash-lite and gemini-2.5-pro-preview-tts are different,
    still-legitimate models -- the anchor must not flag them."""
    assert not _FORBIDDEN.search("gemini-2.5-flash-lite")
    assert not _FORBIDDEN.search("gemini-2.5-pro-preview-tts")
