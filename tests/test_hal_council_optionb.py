"""Goodhart-resistant tests for HAL Council Option B (#334).

Three test classes:
  1. ImportSmoke        — each copied engine module imports cleanly + known symbol present.
  2. Structural         — the ADOPTED workflow YAML parses, has on.pull_request,
                          lacks cross-repo HAL checkout, references the local engine.
  3. AdoptionInvariant  — the AGORA workflow lives at .github/workflows/ (it was
                          git-mv'd on main post-merge, the end-state the original
                          Gate-N staging test anticipated) and no stale staged
                          duplicate lingers under ci/staged-workflows/.

History (2026-07-26): sections 2-3 originally asserted the PRE-adoption layout
(YAMLs staged in ci/staged-workflows/, forbidden under .github/workflows/).
After adoption that inverted reality — 1 failure + 10 setup errors on clean
main (issue #509). hal-council-review.yml was never adopted and no longer
exists anywhere, so structural checks cover the adopted AGORA workflow.
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest
import yaml

from tests._public_tree import require

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

WORKTREE = Path(__file__).resolve().parents[1]  # /tmp/sprint-334b
STAGED = WORKTREE / "ci" / "staged-workflows"
WORKFLOWS_DIR = WORKTREE / ".github" / "workflows"
LIB = WORKTREE / "lib"
SCRIPTS = WORKTREE / ".github" / "scripts"

# Ensure the local lib is importable (mirrors what PYTHONPATH does in CI)
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))


# ---------------------------------------------------------------------------
# 1. Import-smoke tests
# ---------------------------------------------------------------------------

class TestImportSmoke:
    """Each copied engine module imports cleanly and exposes a known public symbol.

    Fails if a transitive dep is missing — real import, not a compile-only check.
    """

    def _import(self, module_name: str) -> object:
        # Fresh-import smoke check WITHOUT leaking module identity: later test
        # files hold symbol references from the ORIGINAL module object, so a
        # reload left in sys.modules desyncs them (order-dependent failure:
        # test_agora_verdict's judge test fails when run after this file).
        # Import fresh for the assertion, then restore the original object.
        original = sys.modules.get(module_name)
        if module_name in sys.modules:
            del sys.modules[module_name]
        try:
            return importlib.import_module(module_name)
        finally:
            if original is not None:
                sys.modules[module_name] = original

    def test_hal_council_dispatch_imports(self):
        mod = self._import("lib.hal_council_dispatch")
        assert hasattr(mod, "dispatch_parallel"), "dispatch_parallel missing"
        assert hasattr(mod, "mesh_deliberate"), "mesh_deliberate missing"
        assert hasattr(mod, "CouncilVerdict"), "CouncilVerdict missing"

    def test_hal_council_pipeline_imports(self):
        mod = self._import("lib.hal_council_pipeline")
        assert hasattr(mod, "run_pipeline"), "run_pipeline missing"
        assert hasattr(mod, "PipelineContext"), "PipelineContext missing"

    def test_agora_pr_review_imports(self):
        mod = self._import("lib.agora_pr_review")
        assert hasattr(mod, "review_pr"), "review_pr missing"
        assert hasattr(mod, "AgoraReview"), "AgoraReview missing"

    def test_improve_kernel_imports(self):
        mod = self._import("lib.improve_kernel")
        assert hasattr(mod, "InnerspeakerGo"), "InnerspeakerGo missing"
        assert hasattr(mod, "fingerprint"), "fingerprint missing"

    def test_pr_diff_imports(self):
        mod = self._import("lib.pr_diff")
        assert hasattr(mod, "robust_pr_diff"), "robust_pr_diff missing"

    def test_council_converge_script_exists(self):
        script = require(".github/scripts/council_converge.py")
        assert script.exists(), f"council_converge.py missing at {script}"
        # Verify it defines the converge() function (source-level check)
        source = script.read_text(encoding="utf-8")
        assert "def converge(" in source, "converge() function not found in script"


# ---------------------------------------------------------------------------
# 2. Structural tests
# ---------------------------------------------------------------------------

def _load_yaml(name: str) -> dict:
    require(".github/workflows/" + name)
    path = WORKFLOWS_DIR / name
    assert path.exists(), f"Adopted workflow missing: {path}"
    with path.open() as fh:
        return yaml.safe_load(fh)


class TestStructural:
    """The adopted workflow YAML has correct structure for HAL Option B."""

    @pytest.fixture(params=["agora-pr-review.yml"])
    def workflow(self, request):
        return _load_yaml(request.param), request.param

    def test_parses_as_yaml(self, workflow):
        data, name = workflow
        assert isinstance(data, dict), f"{name}: not a valid YAML mapping"

    def test_has_pull_request_trigger(self, workflow):
        data, name = workflow
        # PyYAML (YAML 1.1) coerces the bare key `on:` to boolean True.
        on_block = data.get("on") or data.get(True) or {}
        assert "pull_request" in on_block, \
            f"{name}: missing on.pull_request trigger"

    def test_no_cross_repo_hal_checkout(self, workflow):
        """Option B invariant: no step checks out CodeTonight-SA/HAL as a separate repo."""
        data, name = workflow
        jobs = data.get("jobs", {})
        for job_name, job in jobs.items():
            for step in job.get("steps", []):
                uses = step.get("uses", "")
                step_with = step.get("with", {})
                repository = step_with.get("repository", "")
                assert repository != "CodeTonight-SA/HAL", (
                    f"{name} job={job_name}: found cross-repo HAL checkout step "
                    f"(repository: CodeTonight-SA/HAL) — Option B must not need it"
                )

    def test_references_local_engine(self, workflow):
        """The workflow must reference engine scripts from the local workspace."""
        data, name = workflow
        raw = (WORKFLOWS_DIR / name).read_text(encoding="utf-8")
        # At least one of the local engine entrypoints must be referenced
        local_refs = [
            "lib/agora_pr_review.py",
            ".github/scripts/council_converge.py",
            "council_converge.py",
            "agora_pr_review.py",
        ]
        assert any(ref in raw for ref in local_refs), (
            f"{name}: no reference to local engine entrypoints found — "
            f"expected one of {local_refs}"
        )

    def test_hal_home_points_to_workspace_not_subdirectory(self, workflow):
        """HAL_HOME must be ${{ github.workspace }} not ${{ github.workspace }}/.hal."""
        data, name = workflow
        raw = (WORKFLOWS_DIR / name).read_text(encoding="utf-8")
        # Option B: HAL IS the repo, so HAL_HOME = workspace root
        assert "github.workspace }}/.hal" not in raw, (
            f"{name}: HAL_HOME still points to .hal subdirectory — "
            "Option B uses HAL_HOME = ${{ github.workspace }} (root)"
        )


# ---------------------------------------------------------------------------
# 3. Gate-N invariant
# ---------------------------------------------------------------------------

class TestAdoptionInvariant:
    """The AGORA workflow is ADOPTED: it lives at .github/workflows/ on main
    (the end-state the original Gate-N staging test anticipated — "V>> git-mv's
    them on main post-merge"), and no stale staged duplicate lingers that could
    drift from the adopted copy.
    """

    def test_agora_workflow_is_adopted(self):
        adopted = require(".github/workflows/agora-pr-review.yml")
        assert adopted.exists(), (
            "agora-pr-review.yml missing from .github/workflows/ — the AGORA "
            "reviewer would silently stop running on PRs"
        )
        with adopted.open() as fh:
            data = yaml.safe_load(fh)
        assert isinstance(data, dict), "adopted AGORA workflow is not a YAML mapping"

    def test_no_stale_staged_duplicates(self):
        # A lingering ci/staged-workflows/ copy would silently drift from the
        # adopted workflow. Absent dir = fully adopted = pass.
        if not STAGED.exists():
            return
        stale = {p.name for p in STAGED.iterdir() if p.is_file()} & {
            "hal-council-review.yml", "agora-pr-review.yml",
        }
        assert not stale, (
            f"stale staged duplicate(s) {sorted(stale)} under ci/staged-workflows/ "
            "— the adopted copy in .github/workflows/ is canonical; delete these"
        )


# ---------------------------------------------------------------------------
# 4. Speaker-credential coverage
# ---------------------------------------------------------------------------

_DISPATCH_SOURCE = LIB / "hal_council_dispatch.py"

# Matches BOTH credential resolvers the rosters use:
#   _token("GROQ_API_KEY", "grip-groq")                   — HAL Council roster
#   _apikey_auth_or_absent("DASHSCOPE_API_KEY", "grip-…")  — AGORA ladder
#   _api("DASHSCOPE_API_KEY", "hal-dashscope")            — its local alias
#
# Both are matched deliberately. A regex naming only _token would have gone on
# passing while covering nothing but the older roster the moment the AGORA
# ladder moved to the presence-gated resolver — a green test holding open
# exactly the gap it was written to close.
#
# The NAME is read out of the SOURCE rather than an imported roster because
# both resolvers return "env:VAR" or "keychain:SVC" depending on the
# environment they are imported in, so an imported roster forgets the variable
# name on any machine where that variable happens to be unset.
_TOKEN_CALL_RE = re.compile(
    r'(?:_token|_apikey_auth_or_absent|_api)\(\s*"([A-Z0-9_]+)"'
)


def _exported_env(data: dict) -> set[str]:
    """Every environment variable name the workflow exports, at any level.

    GitHub Actions resolves `env:` at workflow, job and step level and merges
    all three, so a check that reads only one level calls a credential missing
    the moment someone hoists the block. That is a false alarm on correct code,
    which is the failure most likely to get a test deleted rather than fixed.

    Raised by Speaker B on PR #566, confirmed against this function, fixed here.
    """
    names: set[str] = set(data.get("env") or {})
    for job in data.get("jobs", {}).values():
        names.update(job.get("env") or {})
        for step in job.get("steps", []):
            names.update(step.get("env") or {})
    return names


class TestSpeakerCredentialCoverage:
    """Every credential a roster reads is exported by the workflow.

    The failure this pins is silent by construction: a rung whose credential is
    absent does not crash the workflow, it drops that speaker and the council
    reviews the PR with a thinner panel than its own roster claims.
    """

    @pytest.fixture(params=["agora-pr-review.yml"])
    def workflow(self, request):
        return _load_yaml(request.param), request.param

    def test_every_roster_credential_is_exported(self, workflow):
        data, name = workflow
        needed = set(_TOKEN_CALL_RE.findall(
            _DISPATCH_SOURCE.read_text(encoding="utf-8")))
        assert needed, (
            f"no credential resolvers found in {_DISPATCH_SOURCE.name} — the "
            "rosters moved or changed shape, so this test is no longer reading "
            "them and would pass on an empty set"
        )
        missing = sorted(needed - _exported_env(data))
        assert not missing, (
            f"{name}: a roster reads {missing} but the workflow does not export "
            "them — those rungs drop out of the council silently"
        )

    def test_the_agora_resolver_is_actually_covered(self):
        """The regex must match the resolver the AGORA roster really uses.

        Without this, a rename of that resolver would leave the coverage test
        green while it silently checked only the older five-voter roster —
        passing, and blind to every rung it exists to protect.
        """
        source = _DISPATCH_SOURCE.read_text(encoding="utf-8")
        found = set(_TOKEN_CALL_RE.findall(source))
        assert "_apikey_auth_or_absent" in source or "_api(" in source, (
            "the AGORA roster no longer uses the presence-gated resolver"
        )
        assert {"GROQ_API_KEY", "XAI_API_KEY", "DEEPSEEK_API_KEY"} <= found, (
            f"the coverage regex no longer reaches the AGORA roster's "
            f"credentials — it found only {sorted(found)}"
        )

    def test_dashscope_credential_is_exported(self, workflow):
        """Pinned by name because no rung reads it yet (HAL#566's reasoning).

        A qwen rung was seated and removed again after CI showed DashScope
        timing out from GitHub Actions runners. The export is kept staged, so
        that whoever re-seats the rung does not have it authenticate against
        an empty variable on its first run — which is how an exported-but-
        missing secret behaves.
        """
        data, name = workflow
        assert "DASHSCOPE_API_KEY" in _exported_env(data), (
            f"{name}: the DashScope export was removed — a qwen rung added "
            "later would authenticate with an empty token"
        )

    def test_env_is_read_from_every_level(self):
        """Speaker C's falsifier from PR #566, written as a forward anchor.

        A credential defined only at job level, or only at workflow level, is
        exported just as surely as one defined on the step. Reading a single
        level would report both as missing.
        """
        job_level = {"jobs": {"agora": {
            "env": {"HOISTED_TO_JOB": "x"},
            "steps": [{"run": "echo", "env": {"ON_THE_STEP": "x"}}],
        }}}
        assert _exported_env(job_level) == {"HOISTED_TO_JOB", "ON_THE_STEP"}

        workflow_level = {"env": {"HOISTED_TO_WORKFLOW": "x"}, "jobs": {}}
        assert _exported_env(workflow_level) == {"HOISTED_TO_WORKFLOW"}
