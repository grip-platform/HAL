"""Presence gating for AGORA ladder rungs (ported from GRIP fb198113 / GRIP#5382).

A rung whose credential does not exist must read ABSENT so the round-robin
chain filter in ``agora_pr_review`` drops it, rather than dispatching it and
burning a runtime auth failure on the role.

The defect these anchor: GitHub Actions sets an environment variable to the
EMPTY STRING when the named secret does not exist, so a membership test on the
variable NAME reads "declared but empty" as present. Asserts BEHAVIOUR (which
path the value takes), not spelling.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lib.hal_council_dispatch import (  # noqa: E402
    _apikey_auth_or_absent,
    _env_present,
)

_VAR = "HAL_TEST_AGORA_FAKE_CREDENTIAL"
_SVC = "hal-test-fake-service"


@pytest.mark.parametrize("value", ["", "   ", "\t", "\n"])
def test_blank_value_is_absent_not_present(monkeypatch, value):
    """A declared-but-empty secret takes the UNSET path.

    This is the GRIP#5382 defect: `NAME in os.environ` is True for an empty
    string, which marked the rung present carrying an empty token.
    """
    monkeypatch.setenv(_VAR, value)
    assert _env_present(_VAR) is False


def test_real_value_is_present(monkeypatch):
    monkeypatch.setenv(_VAR, "sk-not-a-real-key")
    assert _env_present(_VAR) is True


def test_unset_is_absent(monkeypatch):
    monkeypatch.delenv(_VAR, raising=False)
    assert _env_present(_VAR) is False


def test_blank_secret_in_ci_yields_absent_auth(monkeypatch):
    """The load-bearing case: in CI a blank secret must produce a FALSY auth
    mapping, because that emptiness is what the chain filter drops on."""
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv(_VAR, "")
    assert _apikey_auth_or_absent(_VAR, _SVC) == {}


def test_present_secret_in_ci_uses_env_reference(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv(_VAR, "sk-not-a-real-key")
    auth = _apikey_auth_or_absent(_VAR, _SVC)
    assert auth == {"scheme": "apikey", "token": f"env:{_VAR}"}


def test_local_blank_secret_falls_back_to_keychain(monkeypatch):
    """Off CI, behaviour is unchanged from _token: keychain fallback, never
    absent — so a developer's local run keeps every rung."""
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv(_VAR, raising=False)
    auth = _apikey_auth_or_absent(_VAR, _SVC)
    assert auth == {"scheme": "apikey", "token": f"keychain:{_SVC}"}


def test_absent_auth_is_dropped_by_the_chain_filter(monkeypatch):
    """End-to-end property: an absent rung must not reach any role's chain.

    Asserts the CONSEQUENCE rather than the predicate — a rung that reads
    absent but still gets dispatched would pass every test above.
    """
    from lib.agora_pr_review import _build_role_chains

    roster = [
        ("live-a", "groq/openai/gpt-oss-120b", {"scheme": "apikey", "token": "env:X"}, ""),
        ("dead", "qwen/qwen3.5-flash", {}, ""),
        ("live-b", "xai/grok-3", {"scheme": "apikey", "token": "env:Y"}, ""),
    ]
    chains = _build_role_chains(roster)
    for role, chain in chains.items():
        seated = [pid for pid, _prov, _auth in chain]
        assert "dead" not in seated, (
            f"role {role!r} chain seats the absent rung: {seated!r}"
        )
        assert len(chain) == 2, f"role {role!r} expected 2 live rungs, got {seated!r}"


def test_a_seated_speaker_is_logged_with_its_provider_and_rung(caplog):
    """A successful seat must be visible in the workflow log.

    HAL logged per speaker only on FAILURE, so a reader could not tell which
    provider spoke for a role, nor whether it did so at depth 0 or as a rescue
    several rungs down — a degraded council read exactly like a healthy one.
    This asserts the success line exists and carries both facts.

    Inverted 2026-09-10 (tests-must-be-able-to-fail shape 4). It used to call
    ``_converge_step`` with ``depth=0``, a value the kernel never passes:
    ``InnerspeakerGo`` is 1-based, so the first real step arrives as
    ``depth=1``. With a one-rung chain that meant the test demonstrated
    "rung=0" through a call shape production never makes — under the real
    kernel this role would have been skipped as exhausted. The assertion was
    right all along; the INPUT specified the off-by-one. Driving it at the
    kernel's actual first depth keeps the assertion and makes it true for the
    right reason.
    """
    import logging as _logging

    import lib.agora_pr_review as apr

    chains = {
        "plain_language": [("a", "groq/openai/gpt-oss-120b", {"scheme": "apikey"})],
        "quality": [("b", "xai/grok-3", {"scheme": "apikey"})],
        "technical": [("c", "deepseek/deepseek-v4-pro", {"scheme": "apikey"})],
    }
    real = apr._dispatch_speaker
    apr._dispatch_speaker = lambda role, prompt, prov, auth, **kw: f"text for {role}"
    try:
        with caplog.at_level(_logging.INFO, logger=apr.logger.name):
            apr._converge_step(
                {}, 1, pr_title="t", diff_excerpt="d", role_chains=chains,
            )
    finally:
        apr._dispatch_speaker = real

    answered = [r.getMessage() for r in caplog.records if "answered" in r.getMessage()]
    assert len(answered) == 3, f"expected one line per seat, got {answered!r}"
    joined = "\n".join(answered)
    for provider in ("groq/openai/gpt-oss-120b", "xai/grok-3", "deepseek/deepseek-v4-pro"):
        assert provider in joined, f"{provider!r} missing from:\n{joined}"
    assert "rung=0" in joined, f"rung/depth missing from:\n{joined}"


def test_an_empty_seat_is_not_logged_as_answered(caplog):
    """The line must track the SEAT, not the attempt — a rung that returned
    nothing has not answered, and logging it would restore exactly the
    ambiguity this line exists to remove.

    Depth corrected from 0 to 1 for the same reason as the test above: the
    kernel's first step is depth 1, and depth 0 is now rejected outright.
    """
    import logging as _logging

    import lib.agora_pr_review as apr

    chains = {"quality": [("b", "xai/grok-3", {"scheme": "apikey"})]}
    real = apr._dispatch_speaker
    apr._dispatch_speaker = lambda role, prompt, prov, auth, **kw: ""
    try:
        with caplog.at_level(_logging.INFO, logger=apr.logger.name):
            apr._converge_step({}, 1, pr_title="t", diff_excerpt="d", role_chains=chains)
    finally:
        apr._dispatch_speaker = real

    assert not [r for r in caplog.records if "answered" in r.getMessage()], (
        "an empty rung was logged as having answered"
    )
