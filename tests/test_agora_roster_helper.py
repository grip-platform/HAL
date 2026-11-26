"""The seatable-roster test helper does exactly what the shape tests rely on.

``tests/_agora_roster.py:seated_roster`` gives each ABSENT rung (an empty auth
mapping, as ``_apikey_auth_or_absent`` returns on a keyless runner) a fake key,
and leaves every rung that already has auth untouched (HAL#885, AGORA on HAL#886).
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._agora_roster import FAKE_AUTH, seated_roster  # noqa: E402

_REAL = {"scheme": "apikey", "token": "env:GROQ_API_KEY"}
_ROSTER = [
    ("absent", "groq/model-a", {}, "an absent rung"),
    ("present", "xai/model-b", _REAL, "a rung with auth"),
]


def test_absent_rung_gets_the_fake_key():
    seated = seated_roster(_ROSTER)
    assert seated[0][2] == FAKE_AUTH
    assert seated[0][:2] == ("absent", "groq/model-a")
    assert seated[0][3] == "an absent rung"


def test_present_rung_keeps_its_own_auth():
    seated = seated_roster(_ROSTER)
    assert seated[1] == _ROSTER[1]
    assert seated[1][2] is _REAL


def test_each_absent_rung_gets_its_own_copy():
    roster = [("a", "groq/x", {}, ""), ("b", "xai/y", {}, "")]
    seated = seated_roster(roster)
    seated[0][2]["token"] = "changed"
    assert seated[1][2] == FAKE_AUTH
    assert FAKE_AUTH["token"] == "env:HAL_TEST_FAKE_KEY"


def test_live_roster_is_fully_seated():
    assert seated_roster(), "the live AGORA roster is empty"
    assert all(rung[2] for rung in seated_roster())


def test_converge_registry_is_this_tests_own(tmp_path):
    """tests/conftest.py keeps the kernel's fixed-point registry out of ~/.claude."""
    import lib.improve_kernel as kernel

    live = Path.home() / ".claude"
    for path in (kernel._DEFAULT_FIXED_POINTS, kernel._DEFAULT_TELEMETRY):
        assert tmp_path in path.parents, f"{path} is outside this test's tmp_path"
        assert live not in path.parents, f"{path} is under the live {live}"
