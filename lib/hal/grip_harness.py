"""HAL GRIP Harness — auto-absorb GRIP innovations into HAL standalone runtime.

The recursive loop: GRIP innovates via RSI sprints → HAL detects new/changed
components → scores portability → absorbs portable ones → HAL standalone grows.

This is the missing link. AutoHarness scans external repos. GRIPHarness scans
the local GRIP installation (~/.claude) and syncs portable components to HAL.

Usage:
    from lib.hal.grip_harness import GRIPHarness

    harness = GRIPHarness()
    report = harness.scan()          # Detect drift between GRIP and HAL
    harness.absorb(report)           # Sync portable components
    harness.absorb(report, dry=True) # Preview without writing
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

from lib.hal.home import HAL_HOME as _HAL_HOME

_GRIP_HOME = os.path.expanduser("~/.claude")
_REPORTS_DIR = os.path.join(_HAL_HOME, "state", "hal", "grip-harness")

# Patterns that indicate hard Claude Code dependency (not portable standalone).
# PreToolUse/PostToolUse are HAL-native events (hook_dispatch.py) — NOT CC-dependent.
# .claude/settings.json is bridgeable via HAL_HOME resolution.
_CC_MARKERS = re.compile(
    r"CLAUDE_CODE_SESSION_ID|claude\s+-p|--dangerously-skip-permissions|"
    r"cc_rate_limiter|cc_task_metrics|cc_identity_resolver|cc_feature_scanner",
    re.IGNORECASE,
)

# Components that are always GRIP-internal (never sync to HAL).
# Note: PreToolUse/PostToolUse guards ARE HAL-native (hook_dispatch.py).
# Only skip components that depend on CC's process model, not its event model.
_SKIP_COMPONENTS = frozenset({
    "_archived",
    "cc-auto-harness",      # CC release scanner — CC-specific by definition
    "batch-edit-enforcer",  # CC MultiEdit tool — no HAL equivalent yet
})


@dataclass
class ComponentDiff:
    """A single component that differs between GRIP and HAL."""
    name: str
    component_type: str  # "skill", "agent", "hook"
    status: str          # "new", "modified", "removed"
    portable: bool       # True if no CC-specific dependencies
    cc_markers: list = field(default_factory=list)
    grip_path: str = ""
    hal_path: str = ""
    lines_changed: int = 0


@dataclass
class GRIPHarnessReport:
    """Scan report showing drift between GRIP and HAL."""
    scan_time: float
    grip_skills: int
    hal_skills: int
    grip_agents: int
    hal_agents: int
    grip_hooks: int
    hal_hooks: int
    diffs: list  # list[ComponentDiff]
    summary: str

    @property
    def absorbable(self) -> list:
        return [d for d in self.diffs if d.portable and d.status != "removed"]

    @property
    def cc_dependent(self) -> list:
        return [d for d in self.diffs if not d.portable]

    def to_dict(self) -> dict:
        return {
            "scan_time": self.scan_time,
            "counts": {
                "grip": {"skills": self.grip_skills, "agents": self.grip_agents, "hooks": self.grip_hooks},
                "hal": {"skills": self.hal_skills, "agents": self.hal_agents, "hooks": self.hal_hooks},
            },
            "diffs": [
                {"name": d.name, "type": d.component_type, "status": d.status,
                 "portable": d.portable, "cc_markers": d.cc_markers}
                for d in self.diffs
            ],
            "summary": self.summary,
        }


class GRIPHarness:
    """Detect GRIP innovations and absorb portable ones into HAL.

    Phase C.a PR 1 (H169) parameterised the constructor so the same
    harness can target alternative HAL installs — ``HAL_HOME=/tmp/fresh
    hal`` can absorb into a clean tree without needing the default
    ``~/.hal`` to exist. ``grip_home`` is kwarg for symmetry but the
    default ``~/.claude`` remains the only supported source tree.
    """

    def __init__(
        self,
        *,
        hal_home: str = _HAL_HOME,
        grip_home: str = _GRIP_HOME,
    ) -> None:
        self.hal_home = hal_home
        self.grip_home = grip_home

    @property
    def _reports_dir(self) -> str:
        """Derive reports dir from the instance's hal_home."""
        return os.path.join(self.hal_home, "state", "hal", "grip-harness")

    def scan(self) -> GRIPHarnessReport:
        """Compare GRIP and HAL component sets. Returns drift report."""
        start = time.time()
        diffs: list[ComponentDiff] = []

        grip_skills = _list_components(self.grip_home, "skills")
        hal_skills = _list_components(self.hal_home, "skills")
        diffs.extend(_diff_components(grip_skills, hal_skills, "skill", self.grip_home, self.hal_home))

        grip_agents = _list_components(self.grip_home, "agents", ext=".md")
        hal_agents = _list_components(self.hal_home, "agents", ext=".md")
        diffs.extend(_diff_components(grip_agents, hal_agents, "agent", self.grip_home, self.hal_home))

        grip_hooks = _list_components(self.grip_home, "hooks", ext=".py")
        hal_hooks = _list_components(self.hal_home, "hooks", ext=".py")
        diffs.extend(_diff_components(grip_hooks, hal_hooks, "hook", self.grip_home, self.hal_home))

        # Score portability for each diff
        for d in diffs:
            _score_portability(d)

        absorbable = [d for d in diffs if d.portable and d.status != "removed"]
        cc_dep = [d for d in diffs if not d.portable]
        new_count = sum(1 for d in diffs if d.status == "new")
        mod_count = sum(1 for d in diffs if d.status == "modified")

        summary = (
            f"GRIP->HAL drift: {len(diffs)} differences "
            f"({new_count} new, {mod_count} modified, "
            f"{len(absorbable)} absorbable, {len(cc_dep)} CC-dependent)"
        )

        report = GRIPHarnessReport(
            scan_time=time.time() - start,
            grip_skills=len(grip_skills),
            hal_skills=len(hal_skills),
            grip_agents=len(grip_agents),
            hal_agents=len(hal_agents),
            grip_hooks=len(grip_hooks),
            hal_hooks=len(hal_hooks),
            diffs=diffs,
            summary=summary,
        )

        _save_report(report, self._reports_dir)
        return report

    def absorb(self, report: GRIPHarnessReport, dry: bool = False) -> list[str]:
        """Sync absorbable components from GRIP to HAL.

        Args:
            report: Scan report from scan()
            dry: If True, return actions without writing

        Returns:
            List of actions taken (or would-be-taken if dry)
        """
        actions: list[str] = []

        for diff in report.absorbable:
            action = self._absorb_one(diff, dry)
            if action:
                actions.append(action)

        if not dry and actions:
            _save_absorb_log(actions, self._reports_dir)

        return actions

    def _absorb_one(self, diff: ComponentDiff, dry: bool) -> str:
        """Absorb a single component with path-aware, idempotent semantics.

        Replaces the prior rmtree-then-copytree brute force (H145 audit,
        Bucket 2). Behaviour contract:

        * Idempotent — unchanged content produces zero mtime drift
        * chflags-safe — destinations with BSD uappnd/schg/uchg flags are
          skipped with an explicit action note; never silently overwritten
        * Symlink-preserving — links copy as links (broken links preserved)
        * Mode-preserving — executable bits survive the copy (``shutil.copy2``)
        * Stale-file-aware — within a skill directory, files present in the
          destination but absent in the source are removed
        * Dry-run — ``dry=True`` never touches the filesystem

        Caller supplies the :class:`ComponentDiff`; this method decides copy
        vs skip vs dry-run based on path state, not on ``diff.status``.
        """
        src = diff.grip_path
        dst = diff.hal_path

        if not src or not os.path.exists(src):
            return ""

        if os.path.exists(dst) and _has_immutable_flag(dst):
            return (
                f"skipped {diff.component_type} {diff.name} "
                f"(chflags-locked, status={diff.status})"
            )

        if dry:
            return f"would sync {diff.component_type} {diff.name} ({diff.status})"

        if diff.component_type == "skill":
            _sync_tree(src, dst)
        else:
            _sync_file(src, dst)

        return f"synced {diff.component_type} {diff.name} ({diff.status})"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_IMMUTABLE_FLAGS = ("uappnd", "schg", "uchg")
_HASH_CHUNK = 65536


def _has_immutable_flag(path: str) -> bool:
    """Return True if ``path`` has a BSD append-only / immutable flag set.

    The absorber checks this on the destination before overwriting; a
    chflags-locked target is treated as operator-protected and the
    absorb is skipped with an explicit audit note. ``ls -ldO`` is used
    because macOS ``stat`` does not expose flags in a portable column
    format. Non-Darwin platforms return False (no equivalent flag).
    """
    if sys.platform != "darwin":
        return False
    try:
        result = subprocess.run(
            ["ls", "-ldO", path],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return any(flag in result.stdout for flag in _IMMUTABLE_FLAGS)


def _file_sha256(path: str) -> str:
    """Return the hex sha256 digest of ``path`` (regular file only)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _files_identical(src: str, dst: str) -> bool:
    """Return True if ``src`` and ``dst`` have identical content.

    Symlinks compare by target (never by resolved content), so a link
    pointing to a different file is not silently accepted. Regular-file
    comparison short-circuits on size mismatch.
    """
    if not (os.path.lexists(src) and os.path.lexists(dst)):
        return False
    if os.path.islink(src) or os.path.islink(dst):
        if os.path.islink(src) != os.path.islink(dst):
            return False
        return os.readlink(src) == os.readlink(dst)
    try:
        if os.stat(src).st_size != os.stat(dst).st_size:
            return False
    except OSError:
        return False
    return _file_sha256(src) == _file_sha256(dst)


def _sync_file(src: str, dst: str) -> None:
    """Copy ``src`` to ``dst`` only if contents differ. Preserves mode/mtime.

    Parent dirs are created on demand. ``shutil.copy2`` copies file mode,
    mtime, and extended metadata — the 0o755 bit on an executable survives
    the round-trip. Idempotent: a second call with unchanged ``src`` is
    a no-op at the filesystem level (no open, no write).
    """
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if _files_identical(src, dst):
        return
    if os.path.lexists(dst):
        os.unlink(dst)
    if os.path.islink(src):
        os.symlink(os.readlink(src), dst)
    else:
        shutil.copy2(src, dst)


def _sync_symlink(src: str, dst: str) -> None:
    """Mirror ``src``'s symlink target at ``dst`` (never follows the link)."""
    target = os.readlink(src)
    if os.path.islink(dst):
        if os.readlink(dst) == target:
            return
        os.unlink(dst)
    elif os.path.lexists(dst):
        os.unlink(dst)
    os.symlink(target, dst)


def _sync_one_src_entry(
    src_path: str, dst_root: str, name: str, src: str, src_rel: set[str]
) -> None:
    """Copy one src entry into dst, recording its relative path."""
    dst_path = os.path.join(dst_root, name)
    src_rel.add(os.path.relpath(src_path, src))
    if os.path.islink(src_path):
        _sync_symlink(src_path, dst_path)
    elif os.path.isfile(src_path):
        _sync_file(src_path, dst_path)


def _collect_src_entries(src: str, dst: str) -> set[str]:
    """Walk ``src`` once, mirror every entry into ``dst``, return rel paths."""
    src_rel: set[str] = set()
    walker = os.walk(src, followlinks=False)
    for root, dirs, files in walker:
        rel = os.path.relpath(root, src)
        dst_root = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(dst_root, exist_ok=True)
        if rel != ".":
            src_rel.add(rel)
        entries = tuple(dirs) + tuple(files)
        _apply_entries(entries, root, dst_root, src, src_rel)
    return src_rel


def _apply_entries(
    entries: tuple,
    root: str,
    dst_root: str,
    src: str,
    src_rel: set[str],
) -> None:
    """Single-pass apply of the current root's entries via map-like iteration."""
    list(map(
        lambda name: _sync_one_src_entry(
            os.path.join(root, name), dst_root, name, src, src_rel,
        ),
        entries,
    ))


def _remove_stale(dst: str, src_rel: set[str]) -> None:
    """Bottom-up delete of entries absent from ``src_rel``."""
    walker = os.walk(dst, followlinks=False, topdown=False)
    for root, dirs, files in walker:
        _remove_stale_batch(root, files, dst, src_rel, is_dir=False)
        _remove_stale_batch(root, dirs, dst, src_rel, is_dir=True)


def _remove_stale_batch(
    root: str, names: list, dst: str, src_rel: set[str], *, is_dir: bool
) -> None:
    """Single-pass stale removal for a flat batch of names under ``root``."""
    list(map(
        lambda name: _remove_one_stale(
            os.path.join(root, name), dst, src_rel, is_dir=is_dir,
        ),
        names,
    ))


def _remove_one_stale(
    dst_path: str, dst: str, src_rel: set[str], *, is_dir: bool
) -> None:
    """Delete a single stale entry (file, link, or empty dir) or skip."""
    if os.path.relpath(dst_path, dst) in src_rel:
        return
    if is_dir and not os.path.islink(dst_path):
        try:
            os.rmdir(dst_path)
        except OSError:
            pass
        return
    try:
        os.unlink(dst_path)
    except OSError:
        pass


def _sync_tree(src: str, dst: str) -> None:
    """Rsync-like directory sync.

    Two sequential passes, each O(total entries). First pass mirrors ``src``
    into ``dst`` (creating nested dirs, copying files, preserving symlinks)
    and records every source-side relative path. Second pass walks ``dst``
    bottom-up and deletes anything whose rel path is not in that set.
    ``followlinks=False`` keeps symlinked subdirs as symlinks.
    """
    os.makedirs(dst, exist_ok=True)
    src_rel = _collect_src_entries(src, dst)
    _remove_stale(dst, src_rel)


def _list_components(base: str, subdir: str, ext: str = "") -> set[str]:
    """List component names in a directory."""
    path = os.path.join(base, subdir)
    if not os.path.isdir(path):
        return set()

    names = set()
    for entry in os.listdir(path):
        if entry.startswith("."):
            continue
        if ext:
            if entry.endswith(ext):
                names.add(entry.removesuffix(ext) if ext else entry)
        else:
            # Directory-based components (skills)
            if os.path.isdir(os.path.join(path, entry)):
                names.add(entry)
    return names


def _diff_components(
    grip_set: set[str],
    hal_set: set[str],
    component_type: str,
    grip_base: str,
    hal_base: str,
) -> list[ComponentDiff]:
    """Find new, modified, and removed components."""
    diffs: list[ComponentDiff] = []
    subdir = {"skill": "skills", "agent": "agents", "hook": "hooks"}[component_type]
    ext = {"skill": "", "agent": ".md", "hook": ".py"}[component_type]

    # New in GRIP (not in HAL)
    for name in grip_set - hal_set:
        if name in _SKIP_COMPONENTS:
            continue
        grip_path = os.path.join(grip_base, subdir, name + ext if ext else name)
        hal_path = os.path.join(hal_base, subdir, name + ext if ext else name)
        diffs.append(ComponentDiff(
            name=name, component_type=component_type, status="new",
            portable=True, grip_path=grip_path, hal_path=hal_path,
        ))

    # Modified (exists in both, check content diff)
    for name in grip_set & hal_set:
        if name in _SKIP_COMPONENTS:
            continue
        grip_path = os.path.join(grip_base, subdir, name + ext if ext else name)
        hal_path = os.path.join(hal_base, subdir, name + ext if ext else name)
        changed = _content_differs(grip_path, hal_path, component_type)
        if changed > 0:
            diffs.append(ComponentDiff(
                name=name, component_type=component_type, status="modified",
                portable=True, grip_path=grip_path, hal_path=hal_path,
                lines_changed=changed,
            ))

    # Removed from GRIP (still in HAL) — informational only
    for name in hal_set - grip_set:
        if name in _SKIP_COMPONENTS:
            continue
        hal_path = os.path.join(hal_base, subdir, name + ext if ext else name)
        diffs.append(ComponentDiff(
            name=name, component_type=component_type, status="removed",
            portable=True, hal_path=hal_path,
        ))

    return diffs


def _content_differs(grip_path: str, hal_path: str, component_type: str) -> int:
    """Check if content differs. Returns approximate line count of changes, 0 if same."""
    if component_type == "skill":
        # Compare SKILL.md files
        grip_file = os.path.join(grip_path, "SKILL.md")
        hal_file = os.path.join(hal_path, "SKILL.md")
    else:
        grip_file = grip_path
        hal_file = hal_path

    try:
        with open(grip_file, "r", errors="replace", encoding="utf-8") as f:
            grip_content = f.read()
        with open(hal_file, "r", errors="replace", encoding="utf-8") as f:
            hal_content = f.read()
        if grip_content == hal_content:
            return 0
        # Rough line diff count
        grip_lines = set(grip_content.splitlines())
        hal_lines = set(hal_content.splitlines())
        return len(grip_lines.symmetric_difference(hal_lines))
    except OSError:
        return 0


def _score_portability(diff: ComponentDiff) -> None:
    """Check if a component depends on Claude Code specifics."""
    path = diff.grip_path
    if not path or not os.path.exists(path):
        return

    content = ""
    if os.path.isdir(path):
        # Read SKILL.md for skills
        skill_file = os.path.join(path, "SKILL.md")
        if os.path.exists(skill_file):
            with open(skill_file, "r", errors="replace", encoding="utf-8") as f:
                content = f.read()
    else:
        with open(path, "r", errors="replace", encoding="utf-8") as f:
            content = f.read()

    markers = _CC_MARKERS.findall(content)
    if markers:
        diff.portable = False
        diff.cc_markers = list(set(markers))


def _save_report(report: GRIPHarnessReport, reports_dir: str = _REPORTS_DIR) -> None:
    """Persist scan report under ``reports_dir`` (defaults to ``~/.hal`` tree)."""
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, "latest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)


def _save_absorb_log(actions: list[str], reports_dir: str = _REPORTS_DIR) -> None:
    """Append absorption actions to ``reports_dir/absorb.log``."""
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, "absorb.log")
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n--- {timestamp} ---\n")
        for action in actions:
            f.write(f"  {action}\n")
