"""Tests for path-aware grip_harness absorber (H148).

15-case matrix covering the H145-audit-mandated invariants:

* Idempotency (unchanged content → zero mtime drift)
* chflags-locked destinations skipped with explicit note
* Symlink preservation (intact + broken)
* Executable-mode preservation
* Hidden-file inclusion within a skill
* Stale-file removal on re-absorb
* Symlink target change detection
* Dry-run no-side-effect contract
* Agent-file parity (new / unchanged / modified / chflags-locked)

Some tests are macOS-specific (``chflags`` is BSD) and skip on other
platforms. The rest are cross-platform.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from lib.hal.grip_harness import (
    ComponentDiff,
    GRIPHarness,
    _files_identical,
    _has_immutable_flag,
    _sync_file,
    _sync_tree,
)


class AbsorbOneMatrix(unittest.TestCase):
    """15-case path-aware absorber test matrix."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="hal-absorb-")
        self.src = os.path.join(self.tmpdir, "src")
        self.dst = os.path.join(self.tmpdir, "dst")
        os.makedirs(self.src)
        os.makedirs(self.dst)
        self.harness = GRIPHarness()

    def tearDown(self) -> None:
        if sys.platform == "darwin":
            subprocess.run(
                ["chflags", "-R", "nouappnd,nouchg,noschg", self.tmpdir],
                capture_output=True, check=False,
            )
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _diff(
        self, ctype: str, name: str, status: str = "new",
    ) -> ComponentDiff:
        return ComponentDiff(
            name=name,
            component_type=ctype,
            status=status,
            portable=True,
            grip_path=os.path.join(self.src, name),
            hal_path=os.path.join(self.dst, name),
        )

    def test_01_absorb_new_skill_creates_destination(self) -> None:
        skill_src = os.path.join(self.src, "new_skill")
        os.makedirs(skill_src)
        (Path(skill_src) / "SKILL.md").write_text("hello")
        result = self.harness._absorb_one(self._diff("skill", "new_skill"), dry=False)
        self.assertIn("new_skill", result)
        self.assertTrue(
            os.path.exists(os.path.join(self.dst, "new_skill", "SKILL.md"))
        )

    def test_02_unchanged_skill_is_idempotent(self) -> None:
        skill_src = os.path.join(self.src, "stable")
        os.makedirs(skill_src)
        (Path(skill_src) / "SKILL.md").write_text("stable")
        self.harness._absorb_one(self._diff("skill", "stable"), dry=False)
        target = os.path.join(self.dst, "stable", "SKILL.md")
        before = os.stat(target).st_mtime_ns
        time.sleep(0.05)
        self.harness._absorb_one(
            self._diff("skill", "stable", status="modified"), dry=False,
        )
        after = os.stat(target).st_mtime_ns
        self.assertEqual(before, after)

    def test_03_modified_skill_updates_content(self) -> None:
        skill_src = os.path.join(self.src, "changed")
        os.makedirs(skill_src)
        (Path(skill_src) / "SKILL.md").write_text("v1")
        self.harness._absorb_one(self._diff("skill", "changed"), dry=False)
        (Path(skill_src) / "SKILL.md").write_text("v2")
        self.harness._absorb_one(
            self._diff("skill", "changed", status="modified"), dry=False,
        )
        dst_skill_file = os.path.join(self.dst, "changed", "SKILL.md")
        self.assertEqual(Path(dst_skill_file).read_text(), "v2")

    @unittest.skipIf(sys.platform != "darwin", "chflags is BSD-only")
    def test_04_chflags_locked_skill_destination_is_skipped(self) -> None:
        skill_src = os.path.join(self.src, "locked")
        skill_dst = os.path.join(self.dst, "locked")
        os.makedirs(skill_src)
        os.makedirs(skill_dst)
        (Path(skill_src) / "SKILL.md").write_text("incoming")
        (Path(skill_dst) / "SKILL.md").write_text("protected")
        subprocess.run(["chflags", "uappnd", skill_dst], check=True)
        try:
            result = self.harness._absorb_one(
                self._diff("skill", "locked"), dry=False,
            )
            self.assertIn("chflags-locked", result)
            self.assertEqual(
                Path(skill_dst, "SKILL.md").read_text(), "protected",
            )
        finally:
            subprocess.run(["chflags", "nouappnd", skill_dst], check=False)

    def test_05_symlink_preserved_as_symlink(self) -> None:
        skill_src = os.path.join(self.src, "linked")
        os.makedirs(skill_src)
        (Path(skill_src) / "real.md").write_text("content")
        os.symlink("real.md", os.path.join(skill_src, "link.md"))
        self.harness._absorb_one(self._diff("skill", "linked"), dry=False)
        dst_link = os.path.join(self.dst, "linked", "link.md")
        self.assertTrue(os.path.islink(dst_link))
        self.assertEqual(os.readlink(dst_link), "real.md")

    def test_06_broken_symlink_preserved_intact(self) -> None:
        skill_src = os.path.join(self.src, "broken")
        os.makedirs(skill_src)
        os.symlink("/does/not/exist", os.path.join(skill_src, "dangling.md"))
        self.harness._absorb_one(self._diff("skill", "broken"), dry=False)
        dst_link = os.path.join(self.dst, "broken", "dangling.md")
        self.assertTrue(os.path.islink(dst_link))
        self.assertFalse(os.path.exists(dst_link))
        self.assertEqual(os.readlink(dst_link), "/does/not/exist")

    def test_07_executable_mode_preserved(self) -> None:
        skill_src = os.path.join(self.src, "withbin")
        os.makedirs(skill_src)
        bin_path = os.path.join(skill_src, "run.sh")
        Path(bin_path).write_text("#!/bin/bash\necho hi\n")
        os.chmod(bin_path, 0o755)
        self.harness._absorb_one(self._diff("skill", "withbin"), dry=False)
        dst_bin = os.path.join(self.dst, "withbin", "run.sh")
        self.assertEqual(os.stat(dst_bin).st_mode & 0o777, 0o755)

    def test_08_new_agent_file_created(self) -> None:
        agent_src = os.path.join(self.src, "new_agent.md")
        Path(agent_src).write_text("---\nname: new_agent\n---")
        self.harness._absorb_one(self._diff("agent", "new_agent.md"), dry=False)
        self.assertTrue(os.path.exists(os.path.join(self.dst, "new_agent.md")))

    def test_09_unchanged_agent_is_idempotent(self) -> None:
        name = "stable.md"
        Path(os.path.join(self.src, name)).write_text("stable")
        self.harness._absorb_one(self._diff("agent", name), dry=False)
        dst_path = os.path.join(self.dst, name)
        before = os.stat(dst_path).st_mtime_ns
        time.sleep(0.05)
        self.harness._absorb_one(
            self._diff("agent", name, status="modified"), dry=False,
        )
        after = os.stat(dst_path).st_mtime_ns
        self.assertEqual(before, after)

    def test_10_modified_agent_updates_content(self) -> None:
        name = "agent.md"
        Path(os.path.join(self.src, name)).write_text("v1")
        self.harness._absorb_one(self._diff("agent", name), dry=False)
        Path(os.path.join(self.src, name)).write_text("v2")
        self.harness._absorb_one(
            self._diff("agent", name, status="modified"), dry=False,
        )
        self.assertEqual(Path(self.dst, name).read_text(), "v2")

    @unittest.skipIf(sys.platform != "darwin", "chflags is BSD-only")
    def test_11_chflags_locked_agent_file_is_skipped(self) -> None:
        name = "locked.md"
        src_path = os.path.join(self.src, name)
        dst_path = os.path.join(self.dst, name)
        Path(src_path).write_text("incoming")
        Path(dst_path).write_text("protected")
        subprocess.run(["chflags", "uappnd", dst_path], check=True)
        try:
            result = self.harness._absorb_one(
                self._diff("agent", name, status="modified"), dry=False,
            )
            self.assertIn("chflags-locked", result)
            self.assertEqual(Path(dst_path).read_text(), "protected")
        finally:
            subprocess.run(["chflags", "nouappnd", dst_path], check=False)

    def test_12_hidden_files_within_skill_are_included(self) -> None:
        skill_src = os.path.join(self.src, "withhidden")
        os.makedirs(skill_src)
        (Path(skill_src) / "SKILL.md").write_text("main")
        (Path(skill_src) / ".envrc").write_text("export X=1")
        self.harness._absorb_one(self._diff("skill", "withhidden"), dry=False)
        self.assertTrue(
            os.path.exists(os.path.join(self.dst, "withhidden", ".envrc"))
        )

    def test_13_stale_file_in_dst_removed_on_re_absorb(self) -> None:
        skill_src = os.path.join(self.src, "cleanup")
        skill_dst = os.path.join(self.dst, "cleanup")
        os.makedirs(skill_src)
        (Path(skill_src) / "keep.md").write_text("keep")
        os.makedirs(skill_dst)
        (Path(skill_dst) / "keep.md").write_text("keep")
        (Path(skill_dst) / "stale.md").write_text("should be removed")
        self.harness._absorb_one(
            self._diff("skill", "cleanup", status="modified"), dry=False,
        )
        self.assertTrue(os.path.exists(os.path.join(skill_dst, "keep.md")))
        self.assertFalse(os.path.exists(os.path.join(skill_dst, "stale.md")))

    def test_14_symlink_target_change_updates_destination(self) -> None:
        skill_src = os.path.join(self.src, "retarget")
        skill_dst = os.path.join(self.dst, "retarget")
        os.makedirs(skill_src)
        os.symlink("targetA", os.path.join(skill_src, "link"))
        self.harness._absorb_one(self._diff("skill", "retarget"), dry=False)
        self.assertEqual(
            os.readlink(os.path.join(skill_dst, "link")), "targetA",
        )
        os.unlink(os.path.join(skill_src, "link"))
        os.symlink("targetB", os.path.join(skill_src, "link"))
        self.harness._absorb_one(
            self._diff("skill", "retarget", status="modified"), dry=False,
        )
        self.assertEqual(
            os.readlink(os.path.join(skill_dst, "link")), "targetB",
        )

    def test_15_dry_run_has_no_side_effects(self) -> None:
        skill_src = os.path.join(self.src, "dry_check")
        os.makedirs(skill_src)
        (Path(skill_src) / "SKILL.md").write_text("content")
        result = self.harness._absorb_one(
            self._diff("skill", "dry_check"), dry=True,
        )
        self.assertIn("would", result)
        self.assertFalse(os.path.exists(os.path.join(self.dst, "dry_check")))


class HelperFunctions(unittest.TestCase):
    """Focused unit tests for the absorber's internal helpers."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="hal-absorb-h-")

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_files_identical_detects_size_mismatch(self) -> None:
        a = os.path.join(self.tmpdir, "a")
        b = os.path.join(self.tmpdir, "b")
        Path(a).write_text("1")
        Path(b).write_text("12")
        self.assertFalse(_files_identical(a, b))

    def test_files_identical_returns_true_for_matching(self) -> None:
        a = os.path.join(self.tmpdir, "a")
        b = os.path.join(self.tmpdir, "b")
        Path(a).write_text("same")
        Path(b).write_text("same")
        self.assertTrue(_files_identical(a, b))

    def test_has_immutable_flag_false_for_plain_file(self) -> None:
        plain = os.path.join(self.tmpdir, "plain")
        Path(plain).write_text("unlocked")
        self.assertFalse(_has_immutable_flag(plain))

    @unittest.skipIf(sys.platform != "darwin", "chflags is BSD-only")
    def test_has_immutable_flag_true_after_chflags(self) -> None:
        locked = os.path.join(self.tmpdir, "locked")
        Path(locked).write_text("will be locked")
        subprocess.run(["chflags", "uappnd", locked], check=True)
        try:
            self.assertTrue(_has_immutable_flag(locked))
        finally:
            subprocess.run(["chflags", "nouappnd", locked], check=False)


class ParameterisedHalHome(unittest.TestCase):
    """H169 — GRIPHarness(hal_home=, grip_home=) absorbs into an empty tree.

    Phase C.a PR 1 of 5 per docs/hal-grip-independence-audit.md r3.
    Proves substitutability: a fresh HAL_HOME=/tmp/fresh with no prior
    install can be populated from scratch by one scan+absorb call.
    """

    def setUp(self) -> None:
        self.workdir = tempfile.mkdtemp(prefix="hal-freshhome-")
        self.grip_src = os.path.join(self.workdir, "grip-src")
        self.hal_fresh = os.path.join(self.workdir, "hal-fresh")
        self._seed_grip_src()
        os.makedirs(self.hal_fresh, exist_ok=True)

    def tearDown(self) -> None:
        if sys.platform == "darwin":
            subprocess.run(
                ["chflags", "-R", "nouappnd,nouchg,noschg", self.workdir],
                capture_output=True, check=False,
            )
        import shutil
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _seed_grip_src(self) -> None:
        """Build a minimal GRIP-shaped source tree with one of each component."""
        skills = Path(self.grip_src) / "skills" / "demo-skill"
        skills.mkdir(parents=True)
        (skills / "SKILL.md").write_text("# demo skill\n")

        agents = Path(self.grip_src) / "agents"
        agents.mkdir(parents=True)
        (agents / "demo-agent.md").write_text("---\nname: demo\n---\n")

        hooks = Path(self.grip_src) / "hooks"
        hooks.mkdir(parents=True)
        (hooks / "demo-hook.py").write_text("# hook\n")

    def test_fresh_hal_home_receives_all_three_component_types(self) -> None:
        harness = GRIPHarness(hal_home=self.hal_fresh, grip_home=self.grip_src)
        report = harness.scan()
        actions = harness.absorb(report)

        self.assertGreaterEqual(len(actions), 3)
        self.assertTrue(
            (Path(self.hal_fresh) / "skills" / "demo-skill" / "SKILL.md").exists()
        )
        self.assertTrue(
            (Path(self.hal_fresh) / "agents" / "demo-agent.md").exists()
        )
        self.assertTrue(
            (Path(self.hal_fresh) / "hooks" / "demo-hook.py").exists()
        )

    def test_reports_dir_lands_under_instance_hal_home(self) -> None:
        harness = GRIPHarness(hal_home=self.hal_fresh, grip_home=self.grip_src)
        harness.scan()
        expected_report = (
            Path(self.hal_fresh) / "state" / "hal" / "grip-harness" / "latest.json"
        )
        self.assertTrue(expected_report.exists())

    def test_default_constructor_preserves_module_level_defaults(self) -> None:
        """Backwards compat — existing call sites using GRIPHarness()."""
        from lib.hal.grip_harness import _HAL_HOME as module_hal
        from lib.hal.grip_harness import _GRIP_HOME as module_grip
        harness = GRIPHarness()
        self.assertEqual(harness.hal_home, module_hal)
        self.assertEqual(harness.grip_home, module_grip)

    def test_idempotent_second_absorb_is_no_op_on_mtime(self) -> None:
        harness = GRIPHarness(hal_home=self.hal_fresh, grip_home=self.grip_src)
        harness.absorb(harness.scan())
        skill_file = Path(self.hal_fresh) / "skills" / "demo-skill" / "SKILL.md"
        before = skill_file.stat().st_mtime_ns

        time.sleep(0.05)
        harness.absorb(harness.scan())
        after = skill_file.stat().st_mtime_ns
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
