"""True parallel multi-LLM PR council dispatch via HAL HAPPI.

Dispatches the same PR review prompt to N providers simultaneously via
ThreadPoolExecutor. Each provider gives an independent binary verdict:
PASS or BLOCK, with specific file:line issues only.

Anti-Goodhart guarantees:
    - Continuous quality scores are ignored; only file:line issues count
    - BLOCKER requires >= QUORUM providers to agree (except security issues)
    - Security issues (credentials, eval, SQL injection) block on ANY single report
    - The producing model NEVER scores its own output

Broly MESH deliberation roles:
    Analyst:          collects all unique issues from all providers
    Devil's Advocate: challenges issues found by only one provider
    Pragmatist:       filters out issues without valid file:line references
    Integrator:       applies quorum rule, produces final PASS/BLOCK

Usage:
    from lib.hal_council_dispatch import run_council_review, dispatch_parallel, mesh_deliberate

    verdict = run_council_review(diff_text="...", pr_title="fix: auth bug")
    if not verdict.is_pass:
        for issue in verdict.confirmed_blockers:
            print(f"BLOCKER: {issue.file_line} — {issue.description}")
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
import uuid
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed, Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


HAL_HOME = Path(os.environ.get("HAL_HOME", str(Path.home() / ".hal")))
_HAL_BIN_DEFAULT = str(HAL_HOME / "bin" / "hal")
# A fast-model budget. The AGORA technical speaker was a reasoning model
# (deepseek-v4-pro) whose always-on reasoning shared the max_tokens budget and
# returned empty on heavy diffs; its rungs included a rate-limited Groq seat
# and another reasoner (grok-4.6) that exceeded 90s on a CI runner, so the
# speaker returned empty and AGORA refused every review as a validity failure —
# blocking all merges in the repo (HAL#599). 180s stays, but the technical seat
# is now non-reasoning (HAL#636) so the budget cannot be exhausted.
_DEFAULT_TIMEOUT_S = 180

_MAX_DIFF_CHARS = 32_000

# Minimum providers that must agree on a BLOCKER before it blocks the PR.
QUORUM = 2


def _env_int(name: str, default: int) -> int:
    """Parse a non-negative int env override; fall back to default on garbage."""
    try:
        v = int(os.environ.get(name, "").strip())
        return v if v >= 0 else default
    except (ValueError, TypeError):
        return default


# #2725 graceful-degradation policy.
#
# Minimum number of providers that must return a non-ERROR vote for the
# council verdict to be a VALID PASS/BLOCK. Below this floor the council
# cannot decide the PR's code quality — the surviving providers' blockers
# are low-confidence (issue #2725 evidence: under 3/5 quorum #2723 B4
# emitted a disproven "all IDR hash corruption" false blocker). Rather than
# let a degraded reviewer false-BLOCK on noise, the council emits a DISTINCT
# non-code verdict (CouncilVerdict.quorum_failure=True) so loops/operators
# can tell reviewer-infrastructure failure from a real code block.
#
# Default 3 (a 3/5 majority of the 5-provider roster). Configurable via
# HAL_COUNCIL_MIN_QUORUM. With >= this many responders the verdict is valid
# and decided ONLY by the responders — never forced BLOCK by max-iterations.
MIN_QUORUM_RESPONDERS = _env_int("HAL_COUNCIL_MIN_QUORUM", 3)

# 429 backoff+retry (issue #2725 option 3). A provider is only counted as
# non-responding AFTER these retries are exhausted, so a transient
# rate-limit does not silently drop a provider from quorum.
_RATELIMIT_MAX_RETRIES = _env_int("HAL_COUNCIL_RATELIMIT_RETRIES", 2)
_RATELIMIT_BASE_BACKOFF_S = 2.0  # exponential: 2s, 4s, 8s … plus jitter
_RATELIMIT_MAX_BACKOFF_S = 30.0

_RATELIMIT_MARKERS = ("rate_limit", "429", "too many requests", "ratelimit")


def _is_rate_limit_error(reason: Optional[str]) -> bool:
    """True when a happi error reason indicates provider rate-limiting (HTTP 429)."""
    if not reason:
        return False
    low = reason.lower()
    return any(m in low for m in _RATELIMIT_MARKERS)


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with full jitter, capped. attempt is 0-indexed."""
    ceiling = min(_RATELIMIT_BASE_BACKOFF_S * (2 ** attempt), _RATELIMIT_MAX_BACKOFF_S)
    return random.uniform(0.0, ceiling)


def _token(env_var: str, keychain_service: str) -> str:
    """Return `env:VAR` if the env var is already set, else `keychain:SERVICE`.

    HAL's _resolve_apikey_source handles both prefix formats natively.
    This makes the roster work in CI (env vars from GitHub Secrets) and
    locally (macOS Keychain) without any code change between environments.
    Evaluated at import time — set env vars before import when needed.
    """
    return f"env:{env_var}" if os.environ.get(env_var) else f"keychain:{keychain_service}"


def _in_ci() -> bool:
    """True inside a GitHub Actions runner (GITHUB_ACTIONS=true, set by GitHub)."""
    return os.environ.get("GITHUB_ACTIONS", "").strip().lower() == "true"


def _env_present(env_var: str) -> bool:
    """True iff ``env_var`` is set to a NON-BLANK value.

    GitHub Actions sets an environment variable to the EMPTY STRING when the
    named secret does not exist. So ``NAME in os.environ`` — a membership test
    on the NAME — cannot tell "set" from "declared but empty", and a workflow
    that exports a secret nobody has created marks the rung PRESENT carrying an
    empty token. That is strictly WORSE than absent: instead of falling through
    to the next rung on the ladder, the role burns on a runtime auth failure,
    and the council sits a thinner bench with no signal that it has.

    Ported from GRIP fb198113 (GRIP#5382), which fixed exactly this defect.
    Only the EMPTINESS of the value is inspected here. It is never returned,
    logged, or bound to a name, so this module's standing convention — the
    secret VALUE never enters the roster — is preserved.
    """
    return bool(os.environ.get(env_var, "").strip())


def _apikey_auth_or_absent(env_var: str, keychain_service: str) -> dict:
    """Presence-gated apikey auth for an AGORA ladder rung.

    The difference from ``_token`` above is the ABSENT case, and it is the
    whole point. ``_token`` always returns a non-empty reference, so the auth
    mapping it builds is always truthy and the ``if not auth`` filter in
    ``agora_pr_review._build_role_chains`` can never fire. That filter has been
    present and structurally dead: a rung whose credential does not exist was
    still dispatched, burning a runtime failure on the role instead of falling
    through to the next rung.

    Env var set to a non-blank value → ``env:VAR`` (CI and local). Unset,
    empty, or whitespace-only — one state, see ``_env_present``:
      - in CI there is no macOS keychain, so return ``{}``. The empty dict
        marks the rung ABSENT and the chain filter drops it.
      - locally, fall back to ``keychain:SERVICE`` exactly like ``_token``,
        so existing developer behaviour is unchanged.
    """
    if _env_present(env_var):
        return {"scheme": "apikey", "token": f"env:{env_var}"}
    if _in_ci():
        return {}
    return {"scheme": "apikey", "token": f"keychain:{keychain_service}"}

_SECURITY_KEYWORDS = frozenset({
    "credential", "secret", "api key", "apikey", "token", "password",
    "eval(", "sql injection", "path traversal", "ssrf", "hardcoded",
    "subprocess.call", "os.system", "pickle.load",
})

_BLOCKER_RE = re.compile(r"^BLOCKER:\s*(\S+)\s*[—\-–]+\s*(.+)$", re.MULTILINE)
_WARNING_RE = re.compile(r"^WARNING:\s*(\S+)\s*[—\-–]+\s*(.+)$", re.MULTILINE)
_VERDICT_RE = re.compile(r"^VERDICT:\s*(PASS|BLOCK)", re.MULTILINE)

_REVIEW_PROMPT_TEMPLATE = """\
You are an independent code reviewer on a multi-model council. Your job is
to find REAL bugs, security issues, and breaking changes in this PR diff.

RULES (follow exactly):
1. Only report issues you are >90% confident will cause production failures
   or security vulnerabilities.
2. Every issue MUST cite a specific file and line: "lib/foo.py:42"
3. Do NOT give quality scores, ratings, or subjective feedback.
4. Do NOT praise or summarise the code.
5. Security issues (hardcoded secrets, eval(), SQL injection, path traversal,
   SSRF, unsafe deserialization) are always BLOCKER.
6. Logic bugs that will cause production failures are BLOCKER.
7. All other issues are WARNING.

RESPONSE FORMAT (strict — one issue per line):
  BLOCKER: lib/foo.py:42 — why this will break production
  WARNING: lib/bar.py:17 — what should be improved

Finish with exactly one of these on its own line:
  VERDICT: PASS
  VERDICT: BLOCK

If you find no issues, write only:
  VERDICT: PASS

PR TITLE: {pr_title}

PR DIFF:
{diff_text}
"""

# Provider roster: (id, happi_provider_string, auth_dict, human_description)
# "subscription" reads CLAUDE_CODE_OAUTH_TOKEN (Claude Code OAuth token).
# "keychain:X" reads macOS Keychain service X at dispatch time.
_PROVIDER_ROSTER: list[tuple[str, str, dict, str]] = [
    (
        "claude-opus-4-7",
        "claude-opus-4-7",
        {"scheme": "subscription"},
        "Anthropic Claude Opus 4.7",
    ),
    (
        # Repointed from gemini-2.5-flash on 2026-11-24 (GRIP#9389): Google
        # auto-routes that id to gemini-3.5-flash-lite from 2026-12-01.
        "gemini-3.5-flash-lite",
        "gemini/gemini-3.5-flash-lite",
        {"scheme": "apikey", "token": _token("GEMINI_API_KEY", "grip-gemini")},
        "Google Gemini 3.5 Flash-Lite",
    ),
    (
        # llama-3.3-70b-versatile retired at Groq (2026-08-30) —
        # gpt-oss-120b is the current fast/free-tier Groq default.
        "gpt-oss-120b",
        "groq/openai/gpt-oss-120b",
        {"scheme": "apikey", "token": _token("GROQ_API_KEY", "grip-groq")},
        "GPT-OSS 120B via Groq",
    ),
    (
        # deepseek-chat retired by the provider (HTTP 400, 2026-07-26) —
        # v4-pro is the current reasoning flagship (mirrors GRIP #4137).
        "deepseek-v4-pro",
        "deepseek/deepseek-v4-pro",
        {"scheme": "apikey", "token": _token("DEEPSEEK_API_KEY", "hal-deepseek")},
        "DeepSeek V4 Pro",
    ),
    (
        "grok-4.6",
        "xai/grok-4.6",
        {"scheme": "apikey", "token": _token("XAI_API_KEY", "hal-xai")},
        "xAI Grok 4.6",
    ),
]


# AGORA-specific speaker roster (3 non-Google providers).
# V>>-ratified 2026-05-21; ported from GRIP lib/hal_council_dispatch.py:204.
# DISTINCT from _PROVIDER_ROSTER above so HAL Council (binary PASS/BLOCK) keeps
# its 5-voter quorum semantics while AGORA uses the slimmer 3-speaker panel.
#
# Exogenous-anchor-law (rules/exogenous-anchor-law.md): the 3 speakers MUST be
# non-Google because the judge (_JUDGE_PROVIDER below) is Google Gemini. The
# judge synthesises the 3 speaker outputs into a verdict — if any speaker were
# Google, the judge could predict-to-please its own family, violating the
# producer != verifier invariant (anchor #3) and turning the verdict into a
# silent integrity failure (NOT a crash). HAL's own _PROVIDER_ROSTER includes
# gemini-3.5-flash-lite (formerly gemini-2.5-flash) as a council speaker — a naive "reuse the roster + add a
# Gemini judge" would make the judge one of HAL's own speakers. This dedicated
# non-Google panel is the resolution. The import-time assertion below pins the
# property mechanically; test_agora_verdict.py asserts it as a Goodhart-proof
# regression anchor.
# Roster repaired 2026-07-26 (issue #510; workflow run 30183216589 evidence):
# the subscription-scheme Claude leg can never respond in CI — the AGORA
# workflow env carries only GROQ/DEEPSEEK/XAI/GEMINI keys, no Anthropic auth,
# so that role rendered empty on every run — and DeepSeek retired
# deepseek-reasoner (HTTP 400: "The supported API model names are
# deepseek-v4-pro or deepseek-v4-flash"). Speaker A is now GPT-OSS 120B on
# Groq (key already in the workflow env, free tier — llama-3.3 retired
# 2026-08-30); Speaker C is the v4-pro reasoning
# flagship (mirrors GRIP #4137). Speaker B is grok-4.6: grok-3 was kept here on
# the strength of a same-run responsiveness log, but a live GET
# api.x.ai/v1/models on 2026-09-15 shows grok-3 is no longer served at all, so
# the seat was silently dead. Capability anchors: tests/test_agora_roster_capability.py.
def _build_agora_speaker_roster() -> list[tuple[str, str, dict, str]]:
    """Construct the AGORA speaker roster with presence-gated credentials.

    Three addresses, three non-Google families, judge-exogenous — the same
    three HAL has always used, deliberately unchanged. What changes is how
    their credentials resolve: a rung whose secret does not exist now carries
    an empty auth mapping and is DROPPED by ``_build_role_chains``, instead of
    being dispatched and burning a runtime authentication failure on the role.
    That filter has existed for a long time and could never fire, because
    ``_token`` always returns a reference.

    NO FOURTH RUNG, and the reason is measured rather than assumed. A qwen
    rung was added here, seated, and REMOVED again after its own council run
    (PR #567, workflow run 34423854570) showed it made the bench thinner, not
    wider:

      - ``qwen/qwen3.5-flash`` answers in 3.2 s through this dispatch path
        from a developer machine, but TIMED OUT at 45 s from a GitHub Actions
        runner. It is not usable from CI today.
      - Worse, adding it CHANGED THE SEATING. Chains are round-robin, so with
        four rungs qwen lands at the technical role's first dispatched
        position, displacing a provider that would otherwise have been tried.
        The technical seat then emptied and AGORA correctly refused to render.

    So the fourth rung cost a seat rather than saving one. The DashScope
    secret and its workflow export are kept anyway, staged for the day the
    rung is viable, exactly as HAL#566 argued: an export added at the same
    time as the rung would authenticate against an empty variable on the
    first run.

    A second, larger finding came out of that run and is NOT fixed here
    because it belongs to the kernel rather than the roster. ``InnerspeakerGo``
    calls ``converge(state, depth=depth + 1)``
    (``lib/improve_kernel.py``), so the FIRST dispatch happens at chain index
    1 and ``chain[0]`` is never reached. The rung this file lists first has
    therefore never spoken. Every description of this roster — including
    HAL#566's, which reasoned it out from the code — has named the wrong three
    models. The new per-seat log line is what exposed it, and it is left
    visible rather than quietly corrected, because changing the kernel's depth
    sequencing affects every caller of the improve kernel and deserves its own
    change.

    The xAI seat is grok-4.6. It was grok-3 until 2026-09-15, when a live GET
    ``api.x.ai/v1/models`` showed grok-3 is no longer served at all — the seat
    had been dead while the roster still named it. The probe note below is kept
    as the record of HOW a seat was chosen, not as a claim about the current id.

    Both surviving addresses were probed LIVE through this module's own
    dispatch path on 2026-09-10 (``_call_happi_ex`` against
    ``hal --happi-api``): groq/openai/gpt-oss-120b 0.5 s, the xAI seat 3.1 s,
    deepseek/deepseek-v4-pro seated and answering in CI the same night.

    Provider liveness alone is not sufficient to seat a rung, which is why the
    routability anchor exists: GRIP seats ``xai/grok-4.3``, and that address
    appears in neither ``config/providers.yaml`` nor
    ``config/model_catalog.yaml`` here, so importing GRIP's roster wholesale
    would hard-error on the dispatch path and lose the seat. HAL carries
    grok-4.6 instead. The two repositories disagree about which xAI ids they
    carry, in both directions.

    Families stay {groq, xai, deepseek} — all non-Google, so the Gemini judge
    remains exogenous to every rung (rules/exogenous-anchor-law.md, producer
    != verifier). Gemini is deliberately excluded.

    Separated into a function so tests can rebuild the roster under a
    controlled environment; the module-level snapshot below is taken at import
    time, when the CI env is already fixed.
    """
    _api = _apikey_auth_or_absent
    return [
        # --- depth-0 seats: the panel HAL seats today, order preserved ---
        (
            "gpt-oss-120b",
            "groq/openai/gpt-oss-120b",
            _api("GROQ_API_KEY", "grip-groq"),
            "GPT-OSS 120B via Groq (Speaker A — plain language + ELI5)",
        ),
        (
            "deepseek-flash",
            "deepseek/deepseek-flash",
            _api("DEEPSEEK_API_KEY", "hal-deepseek"),
            "DeepSeek Flash (Speaker B — quality + alignment)",
        ),
        (
            "grok-4.20-0309-non-reasoning",
            "xai/grok-4.20-0309-non-reasoning",
            _api("XAI_API_KEY", "hal-xai"),
            "xAI Grok 4.20 non-reasoning (Speaker C — technical concerns)",
        ),
        # --- fallback rungs: reached only when a seat above is unavailable ---
    ]


_AGORA_SPEAKER_ROSTER: list[tuple[str, str, dict, str]] = _build_agora_speaker_roster()


# LLM-AS-JUDGE provider (Google Gemini — exogenous to all speakers above).
# V>>-ratified 2026-05-21. Tuple shape matches roster:
# (id, happi_provider_string, auth_dict, human_description).
#
# Fail-closed contract: if the judge timeouts, errors, returns invalid verdict
# format, or returns auth failure, AGORA renders a "Judge unavailable" banner
# and DOES NOT fabricate a verdict. Per rules/exogenous-anchor-law.md: a
# self-fulfilling judge from a speaker provider is worse than no judge.
#
# Model: gemini-3.1-pro-preview (upgraded from 2.5-flash on 2026-05-29). The
# Flash judge produced false-positive REQUEST_CHANGES verdicts on correct PRs
# (HAL #387/#389/#390) that contradicted its own speaker synthesis — e.g. a
# CPython-inapplicable "memory-model race" verdict on #390's double-checked
# lock. 3.1 Pro is config/providers.yaml's 'complex'/'premium'-tier Gemini
# model (also in config/model_catalog.yaml). It reuses the SAME Gemini family
# and the SAME grip-gemini keychain auth as before (see the auth dict below),
# so the existing workflow credential unblocks it with no new setup, and it
# stays exogenous to the non-Google speakers (the import-time guard holds).
_JUDGE_PROVIDER: tuple[str, str, dict, str] = (
    "gemini-3.1-pro-judge",
    "gemini/gemini-3.1-pro-preview",
    {"scheme": "apikey", "token": _token("GEMINI_API_KEY", "grip-gemini")},
    "Google Gemini 3.1 Pro (judge — exogenous to speakers)",
)


# LOAD-BEARING exogeneity guard (rules/exogenous-anchor-law.md, anchor #3).
# The judge MUST NOT be one of the AGORA speakers — otherwise the verifier
# shares a provider family with a producer it judges, and the verdict becomes
# a self-fulfilling prophecy. Asserted at import time so a future edit that
# (e.g.) adds Gemini to _AGORA_SPEAKER_ROSTER fails LOUDLY at module load
# rather than silently corrupting verdicts in production.
assert _JUDGE_PROVIDER[0] not in {s[0] for s in _AGORA_SPEAKER_ROSTER}, (
    "exogeneity violated: judge provider id "
    f"{_JUDGE_PROVIDER[0]!r} is also an AGORA speaker — see "
    "rules/exogenous-anchor-law.md (producer != verifier)"
)


@dataclass
class Issue:
    """A specific code issue with a file:line reference."""
    file_line: str  # e.g. "lib/foo.py:42"
    description: str
    is_security: bool = False


@dataclass
class ProviderVote:
    """One provider's independent verdict on a PR diff."""
    provider: str
    verdict: str  # "PASS", "BLOCK", or "ERROR"
    blockers: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    raw_text: str = ""
    error: str = ""

    @property
    def adversarial_no_error(self) -> bool:
        """True if no blockers found — consumed by pipeline step_adversarial."""
        return self.verdict == "PASS"

    @property
    def score(self) -> float:
        """Numeric proxy for pipeline step_final_score outsider-multiplier."""
        return 0.0 if self.verdict == "BLOCK" else (50.0 if self.verdict == "ERROR" else 100.0)

    @property
    def is_outsider(self) -> bool:
        return False


@dataclass
class CouncilVerdict:
    """Final MESH-deliberated verdict from N independent providers.

    #2725: ``quorum_failure`` is True when fewer than MIN_QUORUM_RESPONDERS
    providers returned a non-ERROR vote. It is a DISTINCT verdict from a
    real code block — ``is_pass`` is False (the PR is not waved through on a
    broken reviewer) but the cause is reviewer-infrastructure (provider 429),
    NOT confirmed code defects. ``confirmed_blockers`` is empty in this state
    because the surviving providers are below the confidence floor. Defaults
    False so existing constructors/tests remain valid.
    """
    is_pass: bool
    confirmed_blockers: list[Issue]
    all_warnings: list[Issue]
    votes: list[ProviderVote]
    providers_responded: int
    providers_total: int
    quorum_failure: bool = False

    @property
    def adversarial_no_error(self) -> bool:
        return self.is_pass

    @property
    def score(self) -> float:
        return 100.0 if self.is_pass else 0.0

    @property
    def is_outsider(self) -> bool:
        return False


def _is_security_issue(description: str) -> bool:
    desc_lower = description.lower()
    return any(kw in desc_lower for kw in _SECURITY_KEYWORDS)


def _has_valid_file_line(file_line: str) -> bool:
    """Pragmatist gate: reject references without a colon or with placeholder text."""
    if ":" not in file_line:
        return False
    low = file_line.lower()
    return not any(low.startswith(p) for p in ("n/a", "none", "—", "-", "unknown"))


def _normalise_key(file_line: str) -> str:
    """Return the file portion only for quorum grouping.

    Different providers may flag the same bug at neighbouring lines. Grouping
    by file (not file:line) prevents quorum fragmentation across line offsets.
    """
    return file_line.split(":")[0] if ":" in file_line else file_line


def _parse_vote(provider: str, raw_text: str) -> ProviderVote:
    """Parse a provider's raw text response into a ProviderVote."""
    if not raw_text.strip():
        return ProviderVote(provider=provider, verdict="ERROR", error="empty_response")

    blockers: list[Issue] = []
    for m in _BLOCKER_RE.finditer(raw_text):
        file_line = m.group(1).strip()
        desc = m.group(2).strip()
        if _has_valid_file_line(file_line):
            blockers.append(Issue(
                file_line=file_line,
                description=desc,
                is_security=_is_security_issue(desc),
            ))

    warnings: list[Issue] = []
    for m in _WARNING_RE.finditer(raw_text):
        file_line = m.group(1).strip()
        desc = m.group(2).strip()
        if _has_valid_file_line(file_line):
            warnings.append(Issue(file_line=file_line, description=desc))

    verdict_m = _VERDICT_RE.search(raw_text)
    verdict = verdict_m.group(1) if verdict_m else ("BLOCK" if blockers else "PASS")

    return ProviderVote(
        provider=provider,
        verdict=verdict,
        blockers=blockers,
        warnings=warnings,
        raw_text=raw_text,
    )


_DEFAULT_SYSTEM_PROMPT = (
    "You are a code reviewer. Be precise. "
    "Cite file:line for every issue. "
    "Follow the exact response format in the user message."
)


def _build_envelope(
    provider_id: str,
    happi_provider: str,
    auth: dict,
    prompt: str,
    system_override: Optional[str] = None,
) -> dict:
    """Build a HAPPI envelope for a single-shot council/AGORA dispatch.

    ``system_override`` lets callers replace the default HAL Council system
    prompt with a role-specific one. AGORA Speakers (plain-language, quality,
    technical) need their own system prompt because the default
    "Cite file:line for every issue" instruction conflicts with AGORA's
    plain-language Speaker A role (which is narrative, not citation-heavy).

    Origin: 2026-05-13. PRs #2670, #2671, #2674 all produced filler-output
    AGORA reviews ("Please provide the code for review") because the hardcoded
    HAL Council system prompt overrode AGORA's user-prompt role assignment.
    Models resolved the conflict by treating the diff as inadequate and
    asking for code, or by hallucinating generic reviews ("main.py:44 is_prime")
    on stock code. Falsified if AGORA filler-rate stays >=5% after the fix.
    """
    # HAL envelope schema asymmetry (2026-05-13 root cause for PRs #2670/#2671/#2674):
    #   - anthropic.py reads envelope.content first, then flags.message, then args
    #   - gemini.py and openai_compat.py read ONLY flags.message (or args)
    # HAL Council had been setting only ``content`` — so 4/5 providers received
    # an EMPTY user message and hallucinated generic stock reviews. Setting
    # BOTH ``content`` AND ``flags.message`` is the cross-provider-compat fix
    # until HAL is patched to make all adapters read ``content`` (filed as
    # follow-up HAL issue). Sentinel: this dual-write is safe because Anthropic
    # already prefers content over message.
    system_prompt = system_override if system_override is not None else _DEFAULT_SYSTEM_PROMPT
    return {
        "v": "happi/1.0",
        "id": f"council-{provider_id[:8]}-{uuid.uuid4().hex[:8]}",
        "cmd": "chat",
        "content": prompt,
        "flags": {
            "provider": happi_provider,
            "max_tokens": 2048,
            "system": system_prompt,
            "message": prompt,
        },
        "auth": auth,
    }


def _run_hal_subprocess(
    payload: str, hal_bin: str, timeout: int,
) -> tuple[Optional[subprocess.CompletedProcess], Optional[str]]:
    """Run `hal --happi-api`; return ``(proc, None)`` or ``(None, reason)``.

    ``reason`` is ``"timeout"`` or ``"missing_binary"``. The two were previously
    merged into one string, ``timeout_or_missing_binary``, and that conflation is
    what sent HAL#599 down a missing-binary path when the real cause was a slow
    reasoning model exceeding the timeout. They have different remedies — a
    longer budget versus an install — so they are reported apart.
    """
    try:
        return subprocess.run(
            [hal_bin, "--happi-api"],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
        ), None
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except FileNotFoundError:
        return None, "missing_binary"


def _parse_happi_events(stdout: str) -> tuple[list[str], Optional[str]]:
    """Parse happi JSONL stdout into (delta_parts, error_reason).

    error_reason is non-None iff at least one ``type:error`` event was seen.
    Malformed JSON lines are silently skipped (matches happi's tolerant shape).
    """
    parts: list[str] = []
    error_reason: Optional[str] = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = event.get("type", "")
        if t == "delta":
            parts.append(event.get("text", ""))
        elif t == "error":
            error_reason = event.get("error") or event.get("message") or json.dumps(event)
    return parts, error_reason


@dataclass(frozen=True)
class _DispatchReq:
    """One provider dispatch request — the params threaded through the
    attempt → retry → fallback chain. A single param object replaces the
    repeated 6-arg signature (DRY)."""
    pid: str
    happi_provider: str
    auth: dict
    prompt: str
    hal_bin: str
    timeout: int

    def with_provider(self, pid: str, happi_provider: str, auth: dict) -> "_DispatchReq":
        """Same prompt/bin/timeout, different provider — for fallback substitution."""
        return _DispatchReq(pid, happi_provider, auth, self.prompt, self.hal_bin, self.timeout)


def _log_nonzero_exit(proc, envelope_id: str, provider: str) -> Optional[str]:
    """Log a non-zero happi exit; return the error-reason string.

    The caller decides whether to surface it as a hard error (no parts) or
    keep partial output (#2725: return-but-warn — a SIGKILL'd provider that
    emitted partial deltas must never be counted as a clean complete vote
    silently).
    """
    logger.warning(
        "happi non-zero exit=%d (envelope=%s, provider=%s); stderr: %s",
        proc.returncode, envelope_id, provider, (proc.stderr or "")[:500],
    )
    return f"nonzero_exit_{proc.returncode}"


def _call_happi_ex(envelope: dict, hal_bin: str, timeout: int) -> tuple[str, Optional[str]]:
    """Call `hal --happi-api`; return (delta_text, error_reason).

    ``error_reason`` is non-None when the call failed (timeout, missing
    binary, happi error event, or non-zero exit with no parts). #2725:
    callers inspect it to detect HTTP 429 rate-limiting and retry/fall back
    BEFORE counting a provider as non-responding. Origin: 2026-05-06.
    """
    envelope_id = envelope.get("id", "<no-id>")
    provider = envelope.get("flags", {}).get("provider", "<unknown>")
    proc, failure = _run_hal_subprocess(json.dumps(envelope), hal_bin, timeout)
    if proc is None:
        logger.warning(
            "happi subprocess failed (envelope=%s, provider=%s, timeout=%ds): %s",
            envelope_id, provider, timeout, failure,
        )
        return "", failure

    parts, error_reason = _parse_happi_events(proc.stdout)
    if error_reason:
        logger.warning("happi error event (envelope=%s, provider=%s): %s",
                        envelope_id, provider, error_reason)
        return "", error_reason

    if proc.returncode != 0:
        reason = _log_nonzero_exit(proc, envelope_id, provider)
        if not parts:
            return "", reason
        # #2725: partial deltas arrived before a non-zero exit (e.g. the
        # provider was SIGKILL'd mid-stream, rc=137). Return the partial
        # text — but the warning above is the audit trail; never silent.

    text = "".join(parts)
    if not text.strip():
        logger.warning("happi empty completion (envelope=%s, provider=%s, exit=%d)",
                        envelope_id, provider, proc.returncode)
        return text, "empty_completion"
    return text, None


def _call_happi(envelope: dict, hal_bin: str, timeout: int) -> str:
    """Backward-compatible thin wrapper: return only the delta text.

    Retained so existing callers (e.g. AGORA adapters) are unaffected by the
    #2725 (text, reason) tuple split.
    """
    text, _reason = _call_happi_ex(envelope, hal_bin, timeout)
    return text


def _dispatch_attempt(req: _DispatchReq) -> tuple[ProviderVote, Optional[str]]:
    """One single HAPPI call. Returns (vote, error_reason).

    error_reason is non-None on any failure (so the caller can detect 429
    and decide whether to retry / fall back). On success error_reason is
    None and the vote is the parsed verdict.
    """
    try:
        envelope = _build_envelope(req.pid, req.happi_provider, req.auth, req.prompt)
        raw, reason = _call_happi_ex(envelope, req.hal_bin, req.timeout)
        if reason is not None or not raw.strip():
            err = reason or "no_response"
            return ProviderVote(provider=req.pid, verdict="ERROR", error=err), err
        return _parse_vote(req.pid, raw), None
    except Exception as exc:  # noqa: BLE001
        return ProviderVote(provider=req.pid, verdict="ERROR", error=str(exc)), str(exc)


def _dispatch_with_retry(req: _DispatchReq) -> tuple[ProviderVote, Optional[str]]:
    """#2725 option 3: dispatch with jittered exponential backoff on 429.

    Retries ONLY on rate-limit errors (not on timeouts/other failures —
    those are not transient and retrying wastes the council's wall-clock
    budget). Returns the final (vote, error_reason) after retries exhausted.
    """
    vote, reason = _dispatch_attempt(req)
    attempt = 0
    while _is_rate_limit_error(reason) and attempt < _RATELIMIT_MAX_RETRIES:
        sleep_s = _backoff_seconds(attempt)
        logger.warning(
            "provider %s rate-limited (429); backoff %.1fs then retry %d/%d",
            req.pid, sleep_s, attempt + 1, _RATELIMIT_MAX_RETRIES,
        )
        time.sleep(sleep_s)
        vote, reason = _dispatch_attempt(req)
        attempt += 1
    return vote, reason


def _dispatch_one(
    pid: str,
    happi_provider: str,
    auth: dict,
    prompt: str,
    hal_bin: str,
    timeout: int,
    *,
    fallbacks: Optional[list[tuple[str, str, dict, str]]] = None,
) -> ProviderVote:
    """Dispatch one provider; always returns a ProviderVote.

    #2725 option 2+3: backoff+retry on 429, then if still rate-limited
    substitute the next provider down the HAL ladder (``fallbacks``, in
    cloud→cheap→local order) so a full set of N independent responses is
    still gathered. The vote is tagged with its actual responding provider
    (``provider=fallback_pid``) so the verdict is honest about which model
    voted. Falls back ONLY on rate-limit (a substitute cannot fix a code
    timeout or a malformed envelope — those are not provider-specific).

    Signature kept positional for ``council_converge._request_fixes``
    backward-compat; internally adapted to ``_DispatchReq``.
    """
    req = _DispatchReq(pid, happi_provider, auth, prompt, hal_bin, timeout)
    vote, reason = _dispatch_with_retry(req)
    if not _is_rate_limit_error(reason):
        return vote

    for fb_pid, fb_provider, fb_auth, _desc in (fallbacks or []):
        logger.warning(
            "provider %s still 429 after %d retries; falling back to %s",
            pid, _RATELIMIT_MAX_RETRIES, fb_pid,
        )
        fb_vote, fb_reason = _dispatch_with_retry(
            req.with_provider(fb_pid, fb_provider, fb_auth),
        )
        if not _is_rate_limit_error(fb_reason) and fb_vote.verdict != "ERROR":
            return fb_vote
    return vote  # all fallbacks exhausted/also-429 — surface the original ERROR


def _dedupe_active(votes: list[ProviderVote]) -> list[ProviderVote]:
    """Non-ERROR votes, deduped by responding-provider id (keep first).

    #2725: provider fallback can substitute a 429'd primary with another
    model. Deduping by responding-provider id guarantees a substitute can
    never double-count one model toward quorum — quorum independence is
    preserved even under degradation.
    """
    seen: set[str] = set()
    active: list[ProviderVote] = []
    for v in votes:
        if v.verdict == "ERROR" or v.provider in seen:
            continue
        seen.add(v.provider)
        active.append(v)
    return active


def _quorum_failure_verdict(
    votes: list[ProviderVote], responded: int,
) -> CouncilVerdict:
    """#2725 option 1+4: the distinct non-code verdict for a degraded reviewer.

    Fewer than MIN_QUORUM_RESPONDERS providers returned a usable vote (after
    backoff+retry and provider fallback). The council cannot judge the PR's
    code: ``is_pass=False`` (a broken reviewer must not wave a PR through —
    this is the council false-NEGATIVE guard, issue #2725 falsification
    clause), ``quorum_failure=True`` and ``confirmed_blockers=[]`` so the
    verdict is unambiguously "reviewer-infrastructure failure, NOT a
    confirmed code block". This is NOT a forced-BLOCK-by-max-iterations:
    it is a first-class verdict the loop/operator can act on differently
    (escalate the reviewer, do not loop it).
    """
    return CouncilVerdict(
        is_pass=False,
        confirmed_blockers=[],
        all_warnings=[],
        votes=votes,
        providers_responded=responded,
        providers_total=len(votes),
        quorum_failure=True,
    )


def mesh_deliberate(
    votes: list[ProviderVote],
    quorum: int = QUORUM,
    min_responders: int = MIN_QUORUM_RESPONDERS,
) -> CouncilVerdict:
    """Apply Broly MESH deliberation to produce a final binary verdict.

    Analyst:          collect all unique file-level issues from all providers
    Devil's Advocate: challenge issues found by only one provider
    Pragmatist:       filter issues lacking valid file:line references
    Integrator:       apply quorum rule → final PASS/BLOCK

    #2725: an explicit quorum POLICY gates the Integrator. With at least
    ``min_responders`` non-ERROR votes the verdict is VALID and decided
    only by the responders (never forced BLOCK by a max-iterations
    timeout). Below the floor the council emits the distinct
    quorum-failure verdict instead of letting low-confidence surviving
    providers false-BLOCK on noise (the #2723 B4 failure mode).
    """
    active = _dedupe_active(votes)

    if len(active) < max(min_responders, 1):
        # Degraded reviewer (provider 429s, all-fail, etc.) — distinct,
        # honest verdict. Subsumes the old all-fail branch.
        return _quorum_failure_verdict(votes, len(active))

    # Analyst: flatten all blocker/warning pairs first, then single-pass group.
    # Flatten is O(V*I) but avoids double-loop iteration in the grouping step.
    flat_blockers = [
        (issue, vote.provider)
        for vote in active
        for issue in vote.blockers
    ]
    flat_warnings = [
        (issue, vote.provider)
        for vote in active
        for issue in vote.warnings
    ]

    blocker_groups: dict[str, list[tuple[Issue, str]]] = {}
    for issue, provider in flat_blockers:
        blocker_groups.setdefault(_normalise_key(issue.file_line), []).append((issue, provider))

    # Devil's Advocate + Integrator: quorum check in one pass over groups.
    # Security issues: ANY single provider can confirm.
    # Other issues: require >= quorum providers to agree.
    confirmed_blockers: list[Issue] = []
    for reporter_pairs in blocker_groups.values():
        is_security = any(i.is_security for i, _ in reporter_pairs)
        if is_security or len(reporter_pairs) >= quorum:
            best = max(reporter_pairs, key=lambda p: len(p[0].description))[0]
            confirmed_blockers.append(best)

    # Collect unique warnings (union, deduplicated by file) — one pass, O(W).
    seen_warning_keys: set[str] = set()
    all_warnings: list[Issue] = []
    for w, _ in flat_warnings:
        key = _normalise_key(w.file_line)
        if key not in seen_warning_keys:
            seen_warning_keys.add(key)
            all_warnings.append(w)

    return CouncilVerdict(
        is_pass=len(confirmed_blockers) == 0,
        confirmed_blockers=confirmed_blockers,
        all_warnings=all_warnings[:10],
        votes=votes,
        providers_responded=len(active),
        providers_total=len(votes),
    )


def _build_fallback_chains(
    roster: list[tuple],
) -> dict[str, list[tuple[str, str, dict, str]]]:
    """Map each provider id → ordered fallback list (all OTHER roster entries).

    #2725 option 2. HAL-ladder order is preserved (roster is declared
    cloud→cheap→local), so a 429'd cloud primary substitutes the next
    cheaper model first. _dedupe_active prevents a model that also responds
    as a primary from being double-counted when consumed as a fallback.
    """
    return {
        pid: [entry for entry in roster if entry[0] != pid]
        for pid, *_rest in roster
    }


_FUTURE_DEADLINE_SLACK_S = 10  # Extra seconds beyond per-call timeout for future.result().


def dispatch_parallel(
    diff_text: str,
    pr_title: str = "",
    providers: Optional[list[tuple]] = None,
    hal_bin: Optional[str] = None,
    timeout: int = _DEFAULT_TIMEOUT_S,
    max_workers: int = 4,
    future_deadline: Optional[int] = None,
) -> list[ProviderVote]:
    """Dispatch the PR review prompt to N providers simultaneously.

    Returns one ProviderVote per provider. Failed providers return
    ProviderVote(verdict="ERROR") — never raises.

    Args:
        future_deadline: Hard wall-clock deadline (seconds) for each
            future.result() call. Defaults to timeout + _FUTURE_DEADLINE_SLACK_S.
            When a future exceeds this deadline the provider is cancelled and
            a tombstone ProviderVote(verdict="ERROR", error="deadline_exceeded")
            is appended — the polling loop always sees a terminal state.
    """
    roster = providers or _PROVIDER_ROSTER
    bin_path = hal_bin or _HAL_BIN_DEFAULT
    deadline = future_deadline if future_deadline is not None else timeout + _FUTURE_DEADLINE_SLACK_S

    diff_truncated = diff_text[:_MAX_DIFF_CHARS]
    if len(diff_text) > _MAX_DIFF_CHARS:
        diff_truncated += "\n\n[diff truncated — showing first 32k chars]"

    prompt = _REVIEW_PROMPT_TEMPLATE.format(
        pr_title=pr_title or "(untitled)",
        diff_text=diff_truncated,
    )

    # #2725 option 2: each primary provider gets the OTHER roster providers
    # as an ordered fallback chain (HAL ladder order). If the primary 429s
    # after backoff+retry, _dispatch_one substitutes the first non-429
    # fallback so a full set of N independent responses is still gathered.
    fb_chains = _build_fallback_chains(roster)

    votes: list[ProviderVote] = []
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures: dict[Future, str] = {
        executor.submit(
            _dispatch_one, pid, hprov, auth, prompt, bin_path, timeout,
            fallbacks=fb_chains.get(pid),
        ): pid
        for pid, hprov, auth, _ in roster
    }
    try:
        for future in as_completed(futures, timeout=deadline):
            pid = futures[future]
            try:
                votes.append(future.result())
            except Exception as exc:  # noqa: BLE001
                votes.append(ProviderVote(provider=pid, verdict="ERROR", error=str(exc)))
    except FutureTimeoutError:
        # as_completed timed out — tombstone any providers not yet collected.
        collected_pids = {v.provider for v in votes}
        for future, pid in futures.items():
            if pid not in collected_pids:
                future.cancel()
                logger.warning(
                    "broly dispatch deadline exceeded for provider %s after %ds",
                    pid, deadline,
                )
                votes.append(ProviderVote(
                    provider=pid,
                    verdict="ERROR",
                    error="deadline_exceeded",
                ))
    finally:
        # Do not wait for hanging threads — they hold no shared state.
        executor.shutdown(wait=False)

    return votes


def run_council_review(
    diff_text: str,
    pr_title: str = "",
    hal_bin: Optional[str] = None,
    quorum: int = QUORUM,
    min_responders: int = MIN_QUORUM_RESPONDERS,
) -> CouncilVerdict:
    """Convenience entry: dispatch + MESH deliberate → final verdict."""
    votes = dispatch_parallel(diff_text, pr_title=pr_title, hal_bin=hal_bin)
    return mesh_deliberate(votes, quorum=quorum, min_responders=min_responders)


__all__ = [
    "Issue",
    "ProviderVote",
    "CouncilVerdict",
    "dispatch_parallel",
    "mesh_deliberate",
    "run_council_review",
    "_parse_vote",
    "_normalise_key",
    "_has_valid_file_line",
    "QUORUM",
    "MIN_QUORUM_RESPONDERS",
    "_PROVIDER_ROSTER",
    "_AGORA_SPEAKER_ROSTER",
    "_JUDGE_PROVIDER",
    "_is_rate_limit_error",
    "_build_fallback_chains",
    "_dedupe_active",
]
