"""Tests for ``lib.hal.improve_kernel`` — InnerspeakerGo DFI-FIB loop."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from lib.hal.improve_kernel import (
    ImproveResult,
    InnerspeakerGo,
    default_fixed_points_path,
    default_telemetry_path,
    fingerprint,
)


class Fingerprint(unittest.TestCase):
    """Deterministic across types and Python restarts."""

    def test_str_input_is_stable(self) -> None:
        self.assertEqual(fingerprint("hello"), fingerprint("hello"))

    def test_different_strings_hash_differently(self) -> None:
        self.assertNotEqual(fingerprint("a"), fingerprint("b"))

    def test_bytes_and_str_hash_same_when_equal(self) -> None:
        self.assertEqual(fingerprint("abc"), fingerprint(b"abc"))

    def test_dict_serialised_stably(self) -> None:
        a = fingerprint({"x": 1, "y": 2})
        b = fingerprint({"y": 2, "x": 1})
        self.assertEqual(a, b)

    def test_output_is_16_hex_chars(self) -> None:
        fp = fingerprint("anything")
        self.assertEqual(len(fp), 16)
        self.assertTrue(all(c in "0123456789abcdef" for c in fp))


class ImproveResultThreshold(unittest.TestCase):
    """``passes`` honours ge / le ops."""

    def test_ge_threshold_pass(self) -> None:
        r = ImproveResult(state="x", metrics={"q": 0.95}, depth=1, fingerprint="fp")
        self.assertTrue(r.passes({"q": ("ge", 0.9)}))

    def test_ge_threshold_fail(self) -> None:
        r = ImproveResult(state="x", metrics={"q": 0.5}, depth=1, fingerprint="fp")
        self.assertFalse(r.passes({"q": ("ge", 0.9)}))

    def test_le_threshold_pass(self) -> None:
        r = ImproveResult(state="x", metrics={"err": 0.01}, depth=1, fingerprint="fp")
        self.assertTrue(r.passes({"err": ("le", 0.05)}))

    def test_missing_metric_treated_as_zero(self) -> None:
        r = ImproveResult(state="x", metrics={}, depth=1, fingerprint="fp")
        self.assertFalse(r.passes({"q": ("ge", 0.1)}))


class KernelLoop(unittest.TestCase):
    """End-to-end: converge → threshold pass → cache → fixed point → depth cap."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="hal-kernel-")
        os.environ["HAL_HOME"] = self._tmp
        self.fp_path = Path(self._tmp) / "state" / "data" / "converge-fixed-points.jsonl"
        self.tel_path = Path(self._tmp) / "logs" / "improve-kernel.jsonl"

    def tearDown(self) -> None:
        os.environ.pop("HAL_HOME", None)
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_paths_root_under_hal_home(self) -> None:
        self.assertEqual(default_fixed_points_path(), self.fp_path)
        self.assertEqual(default_telemetry_path(), self.tel_path)

    def test_threshold_pass_returns_immediately(self) -> None:
        def always_good(state: str, depth: int) -> ImproveResult[str]:
            return ImproveResult(
                state=state + "!", metrics={"q": 1.0}, depth=depth,
                fingerprint=fingerprint(state + "!"),
            )
        kernel = InnerspeakerGo(
            converge=always_good,
            threshold={"q": ("ge", 0.9)},
            fixed_points_path=self.fp_path,
            telemetry_path=self.tel_path,
        )
        result = kernel("start")
        self.assertTrue(result.passes(kernel.threshold))
        self.assertEqual(result.state, "start!")

    def test_fixed_point_detected_when_state_unchanged(self) -> None:
        calls = {"n": 0}

        def no_op(state: str, depth: int) -> ImproveResult[str]:
            calls["n"] += 1
            return ImproveResult(
                state=state, metrics={"q": 0.0}, depth=depth,
                fingerprint=fingerprint(state),
            )
        kernel = InnerspeakerGo(
            converge=no_op,
            threshold={"q": ("ge", 0.9)},
            fixed_points_path=self.fp_path,
            telemetry_path=self.tel_path,
        )
        result = kernel("immovable")
        self.assertEqual(result.state, "immovable")
        self.assertEqual(calls["n"], 1)  # fixed-point detected after one call

    def test_fixed_point_persisted_to_jsonl(self) -> None:
        def no_op(state: str, depth: int) -> ImproveResult[str]:
            return ImproveResult(
                state=state, metrics={"q": 0.0}, depth=depth,
                fingerprint=fingerprint(state),
            )
        kernel = InnerspeakerGo(
            converge=no_op,
            threshold={"q": ("ge", 0.9)},
            fixed_points_path=self.fp_path,
            telemetry_path=self.tel_path,
        )
        kernel("locked")
        self.assertTrue(self.fp_path.exists())
        records = [
            json.loads(line) for line in self.fp_path.read_text().splitlines() if line
        ]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["fingerprint"], fingerprint("locked"))

    def test_depth_cap_halts_runaway_loop(self) -> None:
        def never_passes(state: str, depth: int) -> ImproveResult[str]:
            new = state + "x"
            return ImproveResult(
                state=new, metrics={"q": 0.0}, depth=depth,
                fingerprint=fingerprint(new),
            )
        kernel = InnerspeakerGo(
            converge=never_passes,
            threshold={"q": ("ge", 0.9)},
            max_depth=5,
            fixed_points_path=self.fp_path,
            telemetry_path=self.tel_path,
        )
        result = kernel("start")
        self.assertLessEqual(result.depth, 5)

    def test_cache_short_circuits_second_call_same_fingerprint(self) -> None:
        calls = {"n": 0}

        def improving(state: str, depth: int) -> ImproveResult[str]:
            calls["n"] += 1
            return ImproveResult(
                state=state + "+", metrics={"q": 1.0}, depth=depth,
                fingerprint=fingerprint(state + "+"),
            )
        kernel = InnerspeakerGo(
            converge=improving,
            threshold={"q": ("ge", 0.9)},
            fixed_points_path=self.fp_path,
            telemetry_path=self.tel_path,
        )
        kernel("x")  # call 1 — cache populated, passes threshold
        before = calls["n"]
        kernel("x")  # call 2 — should hit cache
        self.assertEqual(calls["n"], before)

    def test_telemetry_line_per_converge_call(self) -> None:
        def improving(state: str, depth: int) -> ImproveResult[str]:
            return ImproveResult(
                state=state + ".", metrics={"q": 1.0}, depth=depth,
                fingerprint=fingerprint(state + "."),
            )
        kernel = InnerspeakerGo(
            converge=improving,
            threshold={"q": ("ge", 0.9)},
            fixed_points_path=self.fp_path,
            telemetry_path=self.tel_path,
        )
        kernel("t")
        self.assertTrue(self.tel_path.exists())
        lines = [
            json.loads(l) for l in self.tel_path.read_text().splitlines() if l
        ]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["outcome"], "threshold_pass")

    def test_preloaded_fixed_point_short_circuits_convergence(self) -> None:
        """A fingerprint recorded on disk must short-circuit without
        calling converge even once."""
        self.fp_path.parent.mkdir(parents=True, exist_ok=True)
        fp_value = fingerprint("already-done")
        self.fp_path.write_text(json.dumps({"fingerprint": fp_value}) + "\n")

        calls = {"n": 0}

        def would_improve(state: str, depth: int) -> ImproveResult[str]:
            calls["n"] += 1
            return ImproveResult(
                state=state + "!", metrics={"q": 1.0}, depth=depth,
                fingerprint=fingerprint(state + "!"),
            )
        kernel = InnerspeakerGo(
            converge=would_improve,
            threshold={"q": ("ge", 0.9)},
            fixed_points_path=self.fp_path,
            telemetry_path=self.tel_path,
        )
        result = kernel("already-done")
        self.assertEqual(result.fingerprint, fp_value)
        self.assertEqual(calls["n"], 0)


if __name__ == "__main__":
    unittest.main()
