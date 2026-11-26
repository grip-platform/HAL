"""HAL Git Operations — git and GitHub abstraction for standalone operation.

Wraps git CLI and gh CLI calls in a clean interface. Used by MCP apps,
context assembler, and session management. No external harness needed.

Usage:
    from lib.hal.git_ops import GitOps

    git = GitOps()
    print(git.branch())
    print(git.log(limit=5))
    print(git.status())
    issues = git.gh_issues(state="open", limit=10)
"""
import json
import re
import subprocess
from typing import Optional


class GitRepo:
    """Local git repository read operations (7 methods)."""

    def __init__(self, cwd: Optional[str] = None):
        self._cwd = cwd

    def branch(self) -> str:
        return _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], self._cwd)

    def status(self, short: bool = True) -> str:
        args = ["git", "status"] + (["--short"] if short else [])
        return _run(args, self._cwd)

    def log(self, limit: int = 5) -> str:
        return _run(["git", "log", f"-{limit}", "--oneline"], self._cwd)

    def diff(self, staged: bool = False) -> str:
        args = ["git", "diff"] + (["--cached"] if staged else [])
        return _run(args, self._cwd)

    def changed_files(self, base: str = "HEAD~1") -> list[str]:
        raw = _run(["git", "diff", "--name-only", base], self._cwd)
        return [f for f in raw.split("\n") if f.strip()]

    def is_repo(self) -> bool:
        return bool(_run(["git", "rev-parse", "--git-dir"], self._cwd))

    def remote_url(self) -> str:
        return _run(["git", "remote", "get-url", "origin"], self._cwd)


class GitMutator:
    """Git write operations — branch, stage, commit (3 methods)."""

    def __init__(self, cwd: Optional[str] = None):
        self._cwd = cwd
        self._repo = GitRepo(cwd)

    def create_branch(self, name: str) -> str:
        """Create and checkout a new branch. Returns branch name or empty on failure."""
        _run(["git", "checkout", "-b", name], self._cwd)
        return name if self._repo.branch() == name else ""

    def add_and_commit(self, files: list[str], message: str) -> str:
        """Stage specific files and commit. Returns commit hash or empty on failure."""
        if not files:
            return ""
        _run(["git", "add", "--"] + files, self._cwd)
        _run(["git", "commit", "-m", message], self._cwd)
        return _run(["git", "rev-parse", "--short", "HEAD"], self._cwd)

    def push(self, branch: str, remote: str = "origin") -> bool:
        """Push branch to remote. Returns True on success."""
        return bool(_run(["git", "push", "-u", remote, branch], self._cwd))


class GitHubOps:
    """GitHub CLI operations (4 methods)."""

    def __init__(self, cwd: Optional[str] = None):
        self._cwd = cwd

    def issues(self, state: str = "open", limit: int = 10, labels: str = "") -> list[dict]:
        args = ["gh", "issue", "list", "--state", state, "--limit", str(limit),
                "--json", "number,title,state,labels,assignees"]
        if labels:
            args.extend(["--label", labels])
        return _run_json(args, self._cwd)

    def prs(self, state: str = "open", limit: int = 10) -> list[dict]:
        return _run_json(["gh", "pr", "list", "--state", state,
                          "--limit", str(limit),
                          "--json", "number,title,state,author,headRefName"], self._cwd)

    def create_pr(self, title: str, body: str, base: str = "main") -> str:
        return _run(["gh", "pr", "create", "--title", title, "--body", body, "--base", base], self._cwd)

    def create_issue(self, title: str, body: str, labels: str = "") -> str:
        args = ["gh", "issue", "create", "--title", title, "--body", body]
        if labels:
            args.extend(["--label", labels])
        return _run(args, self._cwd)

    def pr_number_from_url(self, url: str) -> int:
        """Extract PR number from a GitHub PR URL. Returns 0 if not found."""
        m = re.search(r"/pull/(\d+)", url)
        return int(m.group(1)) if m else 0

    def get_pr_status(self, pr_number: int) -> dict:
        """Return CI and merge status for a PR.

        Returns: {state, mergeable: bool, ci_passing: bool, checks: list}
        NOTE: uses _run() + json.loads() — gh pr view returns a JSON object,
        not an array, so _run_json() (which assumes list) cannot be used here.
        """
        raw = _run(
            ["gh", "pr", "view", str(pr_number), "--json",
             "state,mergeStateStatus,statusCheckRollup,mergeable"],
            self._cwd,
        )
        if not raw:
            return {"state": "", "mergeable": False, "ci_passing": False, "checks": []}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {"state": "", "mergeable": False, "ci_passing": False, "checks": []}

        checks = data.get("statusCheckRollup") or []
        # ci_passing: at least one SUCCESS and all non-null conclusions are SUCCESS/SKIPPED
        conclusions = [c.get("conclusion") for c in checks if c.get("conclusion")]
        ci_passing = bool(conclusions) and all(
            c in ("SUCCESS", "SKIPPED") for c in conclusions
        )
        return {
            "state": data.get("state", ""),
            "mergeable": data.get("mergeable") == "MERGEABLE",
            "ci_passing": ci_passing,
            "checks": checks,
        }

    def merge_pr(self, pr_number: int, strategy: str = "squash") -> bool:
        """Merge a PR via gh CLI. Returns True on success."""
        flag = f"--{strategy}" if strategy in ("squash", "merge", "rebase") else "--squash"
        try:
            result = subprocess.run(
                ["gh", "pr", "merge", str(pr_number), flag, "--admin", "--delete-branch"],
                capture_output=True, text=True, timeout=60, cwd=self._cwd,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False


def _run_json(cmd: list[str], cwd: Optional[str] = None) -> list[dict]:
    """Run command and parse JSON output."""
    raw = _run(cmd, cwd)
    if not raw:
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


def _run(cmd: list[str], cwd: Optional[str] = None) -> str:
    """Run command, return stdout. Empty string on failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, cwd=cwd,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""
