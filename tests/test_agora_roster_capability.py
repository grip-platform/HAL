"""Forward-anchor tests for the AGORA + Council speaker rosters (issue #510).

Guards the two dead-leg classes proven in workflow run 30183216589
(2026-07-26, PR #508 review):

1. Retired DeepSeek ids — the provider retired ``deepseek-chat`` and
   ``deepseek-reasoner`` (HTTP 400 as of 2026-07-26), then renamed
   ``deepseek-v4-flash`` and ``deepseek-v4-flash-vision-exp`` to
   ``deepseek-flash`` (2026-10-26, now text+vision); ``GET /models`` now
   serves only ``deepseek-flash`` and ``deepseek-v4-pro``. Both rosters
   pinned a retired id at some point.
2. CI-incapable auth — an AGORA speaker with scheme "subscription" can never
   respond inside GitHub Actions (the workflow env carries no Anthropic
   auth), so its role returned empty on every run and AGORA refused to
   render a review three consecutive times.

Asserts CAPABILITY, not spelling (a test that pins a spelling stays green
while the thing is dead — GRIP PR #4140 lesson; pattern mirrors GRIP
tests/test_sovereign_council_roster.py). Deterministic — no network calls.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lib.hal_council_dispatch import (  # noqa: E402
    _AGORA_SPEAKER_ROSTER,
    _PROVIDER_ROSTER,
)

# Model ids the DeepSeek API retired (HTTP 400): chat/reasoner on 2026-07-26,
# then v4-flash and v4-flash-vision-exp renamed to deepseek-flash on 2026-10-26.
_DEAD_DEEPSEEK_MODELS = {
    "deepseek-chat",
    "deepseek-reasoner",
    "deepseek-v4-flash",
    "deepseek-v4-flash-vision-exp",
}


def _model_of(happi_provider: str) -> str:
    return happi_provider.split("/")[-1]


def test_no_retired_deepseek_ids_in_any_roster():
    """Neither roster may pin a DeepSeek model id the provider retired."""
    for roster_name, roster in (
        ("_AGORA_SPEAKER_ROSTER", _AGORA_SPEAKER_ROSTER),
        ("_PROVIDER_ROSTER", _PROVIDER_ROSTER),
    ):
        for pid, happi_provider, _auth, _desc in roster:
            if happi_provider.startswith("deepseek/"):
                assert _model_of(happi_provider) not in _DEAD_DEEPSEEK_MODELS, (
                    f"{roster_name} entry {pid!r} pins retired DeepSeek model "
                    f"{happi_provider!r}; use deepseek-v4-pro (reasoning) or "
                    "deepseek-flash (fast)"
                )


def test_agora_deepseek_speaker_is_reasoning_flagship():
    """The AGORA deepseek leg is the cheap deepseek-flash, and the technical seat is
    non-reasoning (empty_completion fix, 2026-10-02).

    v4-pro — the reasoning flagship this test used to pin — returned
    empty_completion on every technical-speaker run: its always-on reasoning
    shares the max_tokens budget, so on a heavy diff it exhausted the whole
    budget on reasoning_content and emitted zero visible content, which the
    parser discards (thinking events are dropped, only delta is kept). Every
    reasoning token was paid for and thrown away. The technical seat now rides
    a non-reasoning model (grok-4.20-0309-non-reasoning), and the deepseek leg
    is deepseek-flash — cheaper, same family, same key.
    """
    deepseek_legs = [
        happi for _pid, happi, _auth, _desc in _AGORA_SPEAKER_ROSTER
        if happi.startswith("deepseek/")
    ]
    assert deepseek_legs == ["deepseek/deepseek-flash"], (
        f"expected one deepseek/deepseek-flash AGORA leg, got {deepseek_legs!r}"
    )
    # The technical seat must be a NON-REASONING model — that was the bug. The
    # assertion uses the same predicate dispatch uses, so it pins the property
    # (no reasoning budget to exhaust) rather than any one provider family.
    from lib.agora_pr_review import _AGORA_ROLES, _build_role_chains, _is_reasoning_rung
    from tests._agora_roster import seated_roster
    chains = _build_role_chains(seated_roster())
    _pid, technical_happi, _auth = chains["technical"][0]
    assert not _is_reasoning_rung(technical_happi), (
        f"technical seat is a reasoning model {technical_happi!r} — "
        "empty_completion will recur"
    )


def test_every_agora_speaker_is_ci_capable():
    """No AGORA speaker may use the subscription auth scheme.

    AGORA runs ONLY in GitHub Actions, whose env provides API keys
    (GROQ/DEEPSEEK/XAI/GEMINI) but no Anthropic subscription auth — a
    subscription-scheme speaker is structurally dead there and its role
    renders empty on every review (run 30183216589 evidence: the
    claude-sonnet-4-6 leg returned empty for both roles it was tried on).
    """
    for pid, _happi, auth, _desc in _AGORA_SPEAKER_ROSTER:
        assert auth.get("scheme") != "subscription", (
            f"AGORA speaker {pid!r} uses subscription auth — dead in CI; "
            "use an apikey-scheme provider whose key is in the workflow env"
        )


def test_agora_depth0_seats_are_three_distinct_nongoogle_providers():
    """Panel diversity at the seats that actually speak.

    The roster became a LADDER rather than a fixed three-speaker binding, so
    the original form of this test — len(roster) == 3 — pinned the mechanism
    instead of the property, and a correct ladder turned it red. The intent it
    was written to protect is unchanged and asserted here: the three roles are
    seated by three DISTINCT provider families, none of them the judge's.

    _build_role_chains gives role i the chain starting at rung i, so the first
    three rungs are the depth-0 seats. Fallback rungs below them are covered by
    the next test, which is strictly stronger than the old one.
    """
    from lib.agora_pr_review import _AGORA_ROLES, _build_role_chains
    from tests._agora_roster import seated_roster

    chains = _build_role_chains(seated_roster())
    assert len(chains) == 3, f"expected 3 roles, got {sorted(chains)!r}"

    seats = {}
    for role in _AGORA_ROLES:
        chain = chains[role]
        assert chain, f"role {role!r} has no rung to seat at depth 0"
        _pid, happi, _auth = chain[0]
        seats[role] = happi.split("/")[0] if "/" in happi else "anthropic"

    families = list(seats.values())
    assert len(set(families)) == 3, (
        f"depth-0 seats must be 3 distinct families, got {seats!r}"
    )


def test_every_ladder_rung_is_exogenous_to_the_judge():
    """No rung, at ANY depth, may share the judge's provider family.

    Stronger than the three-speaker check it replaces: a fallback rung is
    reached precisely when a seat fails, which is exactly when a Google speaker
    under the Google judge would silently break producer != verifier
    (rules/exogenous-anchor-law.md). Checking only the seated three would leave
    the degraded path — the one that runs during an incident — unguarded.
    """
    for pid, happi, _auth, _desc in _AGORA_SPEAKER_ROSTER:
        family = happi.split("/")[0] if "/" in happi else "anthropic"
        assert family not in ("gemini", "google"), (
            f"ladder rung {pid!r} (family {family!r}) collides with the "
            "Google judge — the verdict would become self-fulfilling whenever "
            "this rung is reached"
        )


def test_every_seat_has_at_least_one_fallback():
    """Each role's chain must offer more than the rung it starts on.

    Deliberately asserted over the CHAIN, not the roster length. Chains are
    round-robin over every live rung, so three rungs already give each of the
    three roles two providers to fall through to. A fourth rung was tried and
    removed — see _build_agora_speaker_roster — because with round-robin it
    displaced a working provider at a seat rather than extending the tail.
    """
    from lib.agora_pr_review import _build_role_chains
    from tests._agora_roster import seated_roster

    for role, chain in _build_role_chains(seated_roster()).items():
        assert len(chain) >= 2, (
            f"role {role!r} has no fallback: {[c[1] for c in chain]!r}"
        )


def test_reasoning_rungs_get_token_headroom():
    """A thinking model's reasoning tokens share the max_tokens budget with
    its visible answer; at the 2048 default it returns NOTHING on a real
    speaker prompt (measured on GRIP 2026-07-26: reasoning_tokens=2048,
    finish_reason=length, 0 chars — mirrors GRIP 5d90d74a)."""
    from lib.agora_pr_review import (
        _JUDGE_MAX_TOKENS,
        _REASONING_SPEAKER_MAX_TOKENS,
        _is_reasoning_rung,
    )

    assert _is_reasoning_rung("deepseek/deepseek-v4-pro") is True
    assert _is_reasoning_rung("deepseek/deepseek-v4-flash") is True
    assert _is_reasoning_rung("groq/llama-3.3-70b-versatile") is False
    assert _is_reasoning_rung("xai/grok-3") is False
    assert _is_reasoning_rung("") is False
    # Same headroom, same stated reason as the judge — variable reasoning
    # length spikes on complex diffs.
    assert _REASONING_SPEAKER_MAX_TOKENS == _JUDGE_MAX_TOKENS == 8192


def test_agora_reasoning_speaker_is_covered_by_headroom_predicate():
    """The roster's actual reasoning leg must satisfy the predicate — a roster
    swap to an uncovered reasoning family would silently starve again."""
    from lib.agora_pr_review import _is_reasoning_rung

    reasoning_legs = [
        happi for _pid, happi, _auth, _desc in _AGORA_SPEAKER_ROSTER
        if happi.startswith("deepseek/")
    ]
    assert reasoning_legs, "roster lost its reasoning leg entirely"
    for happi in reasoning_legs:
        assert _is_reasoning_rung(happi) is True


def test_every_ladder_rung_is_routable_by_hal_config():
    """Every rung address must be DECLARED in HAL's own provider config.

    The trap this anchors, learned in GRIP: an address can ANSWER at the
    provider yet be absent from the config the dispatch path routes through,
    and the CLI then hard-errors ("'xai/grok-3' is not in the model
    catalogue") — so provider liveness alone is not sufficient to seat a rung.
    The two repositories disagree about which xAI ids they carry, which makes
    copying a roster across them unsafe: GRIP seats xai/grok-4.3, and that
    address appears nowhere in HAL's config.

    Checked STRUCTURALLY — provider name plus model id parsed out of the YAML,
    never a substring scan of the file text. A substring check would pass on a
    retired id merely MENTIONED in a comment, which is exactly how this file's
    grok rows read today.
    """
    import pathlib as _pathlib

    import yaml

    cfg = _pathlib.Path(__file__).resolve().parent.parent / "config" / "providers.yaml"
    declared: set[str] = set()
    for prov in yaml.safe_load(cfg.read_text(encoding="utf-8"))["providers"]:
        name = prov.get("name", "")
        for model in prov.get("models") or []:
            declared.add(f"{name}/{model}")

    for pid, happi, _auth, _desc in _AGORA_SPEAKER_ROSTER:
        assert happi in declared, (
            f"ladder rung {pid!r} routes to {happi!r}, which config/"
            f"providers.yaml does not declare — the dispatch path would "
            f"hard-error on it and the seat would be lost"
        )
