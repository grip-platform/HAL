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
        """A fingerprint recorded on disk, in this caller's namespace, must
        short-circuit without calling converge even once (GRIP#10030: a line
        from another caller, or with no namespace, is not honoured)."""
        self.fp_path.parent.mkdir(parents=True, exist_ok=True)
        fp_value = fingerprint("already-done")
        self.fp_path.write_text(json.dumps({"fingerprint": fp_value, "namespace": "tests.preloaded"}) + "\n")

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
            namespace="tests.preloaded",
        )
        result = kernel("already-done")
        self.assertEqual(result.fingerprint, fp_value)
        self.assertEqual(calls["n"], 0)




# ---------------------------------------------------------------------------
# GRIP#10030: one caller's fixed point never stops another caller's run. Both
# HAL kernel copies carry the namespace; lib/improve_kernel.py shares GRIP's file.
# ---------------------------------------------------------------------------

import importlib  # noqa: E402

import pytest  # noqa: E402

_KERNELS = ("lib.hal.improve_kernel", "lib.improve_kernel")


def _stuck(mod):
    def stuck(state, depth):
        return mod.ImproveResult(state=state, metrics={"q": 0.0}, depth=depth,
                                 fingerprint=mod.fingerprint(state))
    return stuck


def _other(mod, calls):
    def other(state, depth):
        calls.append(depth)
        done = {"done": True}
        return mod.ImproveResult(state=done, metrics={"q": 1.0}, depth=depth,
                                 fingerprint=mod.fingerprint(done))
    return other


@pytest.mark.parametrize("module", _KERNELS)
def test_another_callers_fixed_point_never_stops_this_caller(module, tmp_path):
    mod = importlib.import_module(module)
    fp_path = tmp_path / "fp.jsonl"
    mod.InnerspeakerGo(converge=_stuck(mod), threshold={"q": ("ge", 0.9)},
                       fixed_points_path=fp_path, telemetry_path=tmp_path / "t1.jsonl")({})
    assert mod.fingerprint({}) in fp_path.read_text()
    calls = []
    result = mod.InnerspeakerGo(converge=_other(mod, calls), threshold={"q": ("ge", 0.5)},
                                fixed_points_path=fp_path, telemetry_path=tmp_path / "t2.jsonl")({})
    assert calls == [1] and result.state == {"done": True}


@pytest.mark.parametrize("module", _KERNELS)
def test_a_line_with_no_namespace_is_never_honoured(module, tmp_path):
    """The exact poison of GRIP#9966, written before namespaces existed."""
    mod = importlib.import_module(module)
    fp_path = tmp_path / "fp.jsonl"
    fp_path.write_text(json.dumps({"fingerprint": mod.fingerprint({}), "ts": "2026-11-25T21:30:22Z"}) + "\n")
    calls = []
    mod.InnerspeakerGo(converge=_other(mod, calls), threshold={"q": ("ge", 0.5)},
                       fixed_points_path=fp_path, telemetry_path=tmp_path / "t.jsonl")({})
    assert calls == [1]


@pytest.mark.parametrize("module", _KERNELS)
def test_a_null_namespace_is_treated_like_a_missing_one(module, tmp_path):
    """HAL#900: a line whose namespace is JSON null is skipped like a line with none.

    The valid line after it must still load, so a loader that dropped the whole
    file on a null namespace fails here as well.
    """
    mod = importlib.import_module(module)
    fp_path = tmp_path / "fp.jsonl"
    lines = [json.dumps({"fingerprint": "nnnn", "namespace": None}),
             json.dumps({"fingerprint": "aaaa", "namespace": "caller.a"})]
    fp_path.write_text("\n".join(lines) + "\n")
    kernel = mod.InnerspeakerGo(converge=_stuck(mod), threshold={"q": ("ge", 0.9)},
                                fixed_points_path=fp_path, telemetry_path=tmp_path / "t.jsonl",
                                namespace="caller.a")
    assert kernel._fixed_points == {"aaaa"}


@pytest.mark.parametrize("module", _KERNELS)
def test_an_empty_namespace_is_a_real_one_and_its_lines_load(module, tmp_path):
    """Unlike null, an explicit "" is a namespace: a kernel created with it loads its lines."""
    mod = importlib.import_module(module)
    fp_path = tmp_path / "fp.jsonl"
    lines = [json.dumps({"fingerprint": "nnnn", "namespace": None}),
             json.dumps({"fingerprint": "eeee", "namespace": ""})]
    fp_path.write_text("\n".join(lines) + "\n")
    kernel = mod.InnerspeakerGo(converge=_stuck(mod), threshold={"q": ("ge", 0.9)},
                                fixed_points_path=fp_path, telemetry_path=tmp_path / "t.jsonl",
                                namespace="")
    assert kernel._fixed_points == {"eeee"}


@pytest.mark.parametrize("module", _KERNELS)
def test_one_namespace_shares_its_fixed_points(module, tmp_path):
    """A caller that passes the same namespace keeps its fixed points across runs."""
    mod = importlib.import_module(module)
    fp_path = tmp_path / "fp.jsonl"
    kwargs = dict(threshold={"q": ("ge", 0.9)}, fixed_points_path=fp_path, namespace="tests.shared")
    mod.InnerspeakerGo(converge=_stuck(mod), telemetry_path=tmp_path / "t1.jsonl", **kwargs)("locked")
    calls = []
    mod.InnerspeakerGo(converge=_other(mod, calls), telemetry_path=tmp_path / "t2.jsonl", **kwargs)("locked")
    assert calls == []
    assert json.loads(fp_path.read_text().splitlines()[0])["namespace"] == "tests.shared"


@pytest.mark.parametrize("module", _KERNELS)
def test_the_default_namespace_is_the_same_in_a_fresh_process(module):
    """Fixed points outlive the process, so the default must not depend on it."""
    import subprocess
    import sys
    mod = importlib.import_module(module)
    here = mod._default_namespace(mod.fingerprint)
    repo = str(Path(__file__).resolve().parent.parent)
    code = ("import importlib, sys; sys.path.insert(0, sys.argv[1]); "
            "k = importlib.import_module(sys.argv[2]); print(k._default_namespace(k.fingerprint))")
    out = subprocess.run([sys.executable, "-c", code, repo, module], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == here == f"{module}.fingerprint"


@pytest.mark.parametrize("module", _KERNELS)
def test_partials_of_different_functions_get_different_namespaces(module):
    import functools
    mod = importlib.import_module(module)

    def one(state, depth, extra=None):
        return state

    def two(state, depth, extra=None):
        return state

    a = mod._default_namespace(functools.partial(one, extra=1))
    b = mod._default_namespace(functools.partial(two, extra=1))
    assert a != b and a.endswith("one") and b.endswith("two")


@pytest.mark.parametrize("module", _KERNELS)
def test_a_kernel_loads_only_its_own_lines_and_skips_non_objects(module, tmp_path):
    mod = importlib.import_module(module)
    fp_path = tmp_path / "fp.jsonl"
    lines = ["null", "[1, 2]", "7", json.dumps({"fingerprint": "aaaa", "namespace": "caller.a"}),
             json.dumps({"fingerprint": "bbbb", "namespace": "caller.b"})]
    fp_path.write_text("\n".join(lines) + "\n")
    kernel = mod.InnerspeakerGo(converge=_stuck(mod), threshold={"q": ("ge", 0.9)},
                                fixed_points_path=fp_path, telemetry_path=tmp_path / "t.jsonl",
                                namespace="caller.a")
    assert kernel._fixed_points == {"aaaa"}



@pytest.mark.parametrize("module", _KERNELS)
def test_an_unhashable_fingerprint_in_this_namespace_is_skipped(module, tmp_path):
    """A corrupt line in the caller's own namespace must not stop the kernel from starting."""
    mod = importlib.import_module(module)
    fp_path = tmp_path / "fp.jsonl"
    lines = [{"fingerprint": ["not", "hashable"], "namespace": "caller.a"},
             {"fingerprint": "aaaa", "namespace": "caller.a"}]
    fp_path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    kernel = mod.InnerspeakerGo(converge=_stuck(mod), threshold={"q": ("ge", 0.9)},
                                fixed_points_path=fp_path, telemetry_path=tmp_path / "t.jsonl",
                                namespace="caller.a")
    assert kernel._fixed_points == {"aaaa"}


@pytest.mark.parametrize("module", _KERNELS)
def test_an_explicit_empty_namespace_is_honoured(module, tmp_path):
    mod = importlib.import_module(module)
    kernel = mod.InnerspeakerGo(converge=_stuck(mod), threshold={"q": ("ge", 0.9)},
                                fixed_points_path=tmp_path / "fp.jsonl",
                                telemetry_path=tmp_path / "t.jsonl", namespace="")
    assert kernel._namespace == ""


@pytest.mark.parametrize("module", _KERNELS)
def test_partials_of_one_function_share_a_namespace(module):
    """The documented limit: arguments are not part of the name, so pass namespace= when they matter."""
    import functools
    mod = importlib.import_module(module)

    def step(state, depth, mode=None):
        return state

    a = mod._default_namespace(functools.partial(step, mode="a"))
    b = mod._default_namespace(functools.partial(step, mode="b"))
    assert a == b and a.endswith("step")


if __name__ == "__main__":
    unittest.main()
