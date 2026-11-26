"""Tests for HAL workflow engine."""
import os
import sys
import tempfile
import unittest

from lib.hal.workflow import (
    Step,
    StepResult,
    Workflow,
    WorkflowResult,
    _resolve_variables,
    load_workflow,
)


class TestStep(unittest.TestCase):

    def test_prompt_step(self):
        step = Step("review", "prompt", prompt="Review this code")
        self.assertEqual(step.step_type, "prompt")

    def test_tool_step(self):
        step = Step("read", "tool", tool_name="Read", params={"file": "test.py"})
        self.assertEqual(step.tool_name, "Read")


class TestResolveVariables(unittest.TestCase):

    def test_replaces_output_ref(self):
        outputs = {"step1": "hello world"}
        result = _resolve_variables("Input: {step1.output}", outputs)
        self.assertEqual(result, "Input: hello world")

    def test_missing_ref(self):
        result = _resolve_variables("{missing.output}", {})
        self.assertIn("missing", result)

    def test_no_refs(self):
        result = _resolve_variables("plain text", {})
        self.assertEqual(result, "plain text")


class TestWorkflow(unittest.TestCase):

    def test_add_steps(self):
        wf = Workflow("test")
        wf.add_step(Step("a", "tool", tool_name="Echo", params={"text": "hello"}))
        self.assertEqual(len(wf.steps), 1)

    def test_execute_echo_tool(self):
        wf = Workflow("echo_test")
        wf.add_step(Step("greet", "tool", tool_name="Echo", params={"text": "hello"}))
        result = wf.execute()
        self.assertTrue(result.success)
        self.assertEqual(result.final_output, "hello")

    def test_execute_chained_steps(self):
        wf = Workflow("chain")
        wf.add_step(Step("first", "tool", tool_name="Echo", params={"text": "data"}))
        wf.add_step(Step("second", "tool", tool_name="Echo", params={"text": "got: {first.output}"}))
        result = wf.execute()
        self.assertTrue(result.success)
        self.assertEqual(result.final_output, "got: data")

    def test_execute_read_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("file content here")
            path = f.name
        try:
            wf = Workflow("read_test")
            wf.add_step(Step("read", "tool", tool_name="Read", params={"file": path}))
            result = wf.execute()
            self.assertTrue(result.success)
            self.assertIn("file content", result.final_output)
        finally:
            os.unlink(path)

    def test_execute_stops_on_failure(self):
        wf = Workflow("fail_test")
        wf.add_step(Step("bad", "tool", tool_name="Read", params={"file": "/nonexistent/file.txt"}))
        wf.add_step(Step("never", "tool", tool_name="Echo", params={"text": "should not run"}))
        result = wf.execute()
        self.assertFalse(result.success)
        self.assertEqual(len(result.steps), 1)

    def test_prompt_step_no_registry(self):
        wf = Workflow("no_registry")
        wf.add_step(Step("ask", "prompt", prompt="Hello"))
        result = wf.execute()
        self.assertFalse(result.success)


class TestWorkflowResult(unittest.TestCase):

    def test_summary(self):
        result = WorkflowResult("test", [
            StepResult("a", "ok", True, 100),
            StepResult("b", "ok", True, 200),
        ], total_duration_ms=300)
        self.assertIn("2/2", result.summary)

    def test_to_dict(self):
        result = WorkflowResult("test", [StepResult("a", "out", True)])
        d = result.to_dict()
        self.assertEqual(d["workflow"], "test")
        self.assertTrue(d["success"])


class TestWorkflowPersistence(unittest.TestCase):

    def test_save_and_load(self):
        wf = Workflow("persist_test")
        wf.add_step(Step("echo", "tool", tool_name="Echo", params={"text": "hi"}))

        import lib.hal.workflow as mod
        original = mod._WORKFLOWS_DIR
        with tempfile.TemporaryDirectory() as tmpdir:
            mod._WORKFLOWS_DIR = tmpdir
            try:
                wf.save()
                loaded = load_workflow("persist_test")
                self.assertIsNotNone(loaded)
                self.assertEqual(loaded.name, "persist_test")
                self.assertEqual(len(loaded.steps), 1)
            finally:
                mod._WORKFLOWS_DIR = original

    def test_load_nonexistent(self):
        self.assertIsNone(load_workflow("nonexistent_workflow_12345"))


if __name__ == "__main__":
    unittest.main()
