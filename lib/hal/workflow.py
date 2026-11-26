"""HAL Workflow Engine — multi-step automated task execution.

Defines workflows as sequences of steps that execute in order. Each step
can be a prompt (sent to an LLM), a tool call, or a conditional branch.
Steps can reference outputs from previous steps via {step_name.output}.

Usage:
    from lib.hal.workflow import Workflow, Step

    wf = Workflow("code_review_pipeline")
    wf.add_step(Step("read", "tool", tool_name="Read", params={"file": "main.py"}))
    wf.add_step(Step("review", "prompt", prompt="Review this code: {read.output}"))
    wf.add_step(Step("summary", "prompt", prompt="Summarise the review: {review.output}"))

    results = wf.execute(registry)
"""
import json
import os
import re
import time
from dataclasses import dataclass, field

from lib.hal.home import HAL_HOME as _HAL_HOME

_WORKFLOWS_DIR = os.path.join(_HAL_HOME, "config", "workflows")


@dataclass
class Step:
    """A single workflow step."""
    name: str
    step_type: str  # "prompt", "tool", "conditional"
    prompt: str = ""
    tool_name: str = ""
    params: dict = field(default_factory=dict)
    condition: str = ""  # For conditional steps
    model: str = ""  # Override model for this step


@dataclass
class StepResult:
    """Result from executing a single step."""
    name: str
    output: str
    success: bool
    duration_ms: int = 0
    error: str = ""


@dataclass
class WorkflowResult:
    """Result from executing a complete workflow."""
    workflow_name: str
    steps: list[StepResult] = field(default_factory=list)
    total_duration_ms: int = 0

    @property
    def success(self) -> bool:
        return all(s.success for s in self.steps)

    @property
    def final_output(self) -> str:
        return self.steps[-1].output if self.steps else ""

    @property
    def summary(self) -> str:
        ok = sum(1 for s in self.steps if s.success)
        return f"Workflow '{self.workflow_name}': {ok}/{len(self.steps)} steps ok, {self.total_duration_ms}ms"

    def to_dict(self) -> dict:
        return {
            "workflow": self.workflow_name,
            "success": self.success,
            "steps": [
                {"name": s.name, "success": s.success, "output": s.output[:500], "duration_ms": s.duration_ms}
                for s in self.steps
            ],
            "total_duration_ms": self.total_duration_ms,
        }


_VAR_RE = re.compile(r"\{(\w+)\.output\}")


class Workflow:
    """Multi-step automated workflow."""

    def __init__(self, name: str):
        self.name = name
        self._steps: list[Step] = []

    def add_step(self, step: Step) -> None:
        """Add a step to the workflow."""
        self._steps.append(step)

    @property
    def steps(self) -> list[Step]:
        return list(self._steps)

    def execute(self, registry=None) -> WorkflowResult:
        """Execute all steps in sequence, passing outputs forward."""
        start = time.time()
        outputs: dict[str, str] = {}
        results: list[StepResult] = []

        for step in self._steps:
            step_start = time.time()
            result = _execute_step(step, outputs, registry)
            result.duration_ms = int((time.time() - step_start) * 1000)
            results.append(result)
            outputs[step.name] = result.output

            if not result.success:
                break  # Stop on first failure

        return WorkflowResult(
            workflow_name=self.name,
            steps=results,
            total_duration_ms=int((time.time() - start) * 1000),
        )

    def save(self) -> str:
        """Save workflow definition to disk."""
        os.makedirs(_WORKFLOWS_DIR, exist_ok=True)
        path = os.path.join(_WORKFLOWS_DIR, f"{self.name}.json")
        data = {
            "name": self.name,
            "steps": [
                {"name": s.name, "type": s.step_type, "prompt": s.prompt,
                 "tool_name": s.tool_name, "params": s.params, "model": s.model}
                for s in self._steps
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path


def load_workflow(name: str) -> Workflow | None:
    """Load a saved workflow definition."""
    path = os.path.join(_WORKFLOWS_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        wf = Workflow(data["name"])
        for s in data.get("steps", []):
            wf.add_step(Step(
                name=s["name"], step_type=s["type"],
                prompt=s.get("prompt", ""), tool_name=s.get("tool_name", ""),
                params=s.get("params", {}), model=s.get("model", ""),
            ))
        return wf
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def _execute_step(step: Step, outputs: dict[str, str], registry) -> StepResult:
    """Execute a single workflow step."""
    if step.step_type == "prompt":
        return _execute_prompt_step(step, outputs, registry)
    if step.step_type == "tool":
        return _execute_tool_step(step, outputs)
    return StepResult(name=step.name, output="", success=False, error=f"Unknown step type: {step.step_type}")


def _resolve_variables(text: str, outputs: dict[str, str]) -> str:
    """Replace {step_name.output} references with actual outputs."""
    def _replace(match: re.Match) -> str:
        return outputs.get(match.group(1), f"[missing: {match.group(1)}]")
    return _VAR_RE.sub(_replace, text)


def _execute_prompt_step(step: Step, outputs: dict[str, str], registry) -> StepResult:
    """Execute a prompt step via the LLM."""
    if not registry:
        return StepResult(name=step.name, output="", success=False, error="No registry available")

    prompt = _resolve_variables(step.prompt, outputs)
    model = step.model
    if not model:
        providers = registry.list_providers()
        for p in providers:
            if p.models:
                model = f"{p.name}/{p.models[0]}"
                break

    if not model:
        return StepResult(name=step.name, output="", success=False, error="No model available")

    try:
        response = registry.chat(model, prompt, max_tokens=1000)
        return StepResult(name=step.name, output=response.text, success=True)
    except Exception as e:
        return StepResult(name=step.name, output="", success=False, error=str(e)[:200])


def _execute_tool_step(step: Step, outputs: dict[str, str]) -> StepResult:
    """Execute a tool step (file read, command, etc.)."""
    resolved_params = {
        k: _resolve_variables(str(v), outputs) for k, v in step.params.items()
    }

    if step.tool_name == "Read":
        return _tool_read(step.name, resolved_params)
    if step.tool_name == "Echo":
        return StepResult(name=step.name, output=resolved_params.get("text", ""), success=True)

    return StepResult(name=step.name, output="", success=False, error=f"Unknown tool: {step.tool_name}")


def _tool_read(name: str, params: dict) -> StepResult:
    """Read a file."""
    path = params.get("file", "")
    if not path or not os.path.exists(path):
        return StepResult(name=name, output="", success=False, error=f"File not found: {path}")
    try:
        with open(path, "r", errors="replace") as f:
            content = f.read(10000)
        return StepResult(name=name, output=content, success=True)
    except OSError as e:
        return StepResult(name=name, output="", success=False, error=str(e))
