"""Tests for HAL request processing pipeline."""
import os
import sys
import unittest
from unittest.mock import MagicMock

from lib.hal.pipeline import (
    Pipeline, PipelineContext, PipelineStage,
    MediaStage, IntentStage, GuardStage, RouteStage,
    ContextStage, DispatchStage, RecordStage,
)


class _AppendStage(PipelineStage):
    """Test stage that appends its name to metadata['order']."""
    def __init__(self, label):
        self._label = label
    @property
    def name(self):
        return self._label
    def process(self, ctx):
        ctx.metadata.setdefault("order", []).append(self._label)
        return ctx


class _FailStage(PipelineStage):
    """Test stage that raises an error."""
    def process(self, ctx):
        raise ValueError("intentional failure")


class TestPipelineOrdering(unittest.TestCase):

    def test_stages_run_in_order(self):
        pipe = Pipeline([_AppendStage("A"), _AppendStage("B"), _AppendStage("C")])
        ctx = pipe.run(PipelineContext(message="hello"))
        self.assertEqual(ctx.metadata["order"], ["A", "B", "C"])

    def test_error_stops_pipeline(self):
        pipe = Pipeline([_AppendStage("A"), _FailStage(), _AppendStage("C")])
        ctx = pipe.run(PipelineContext(message="hello"))
        self.assertEqual(ctx.metadata["order"], ["A"])
        self.assertIn("intentional failure", ctx.error)

    def test_stage_timings_recorded(self):
        pipe = Pipeline([_AppendStage("A"), _AppendStage("B")])
        ctx = pipe.run(PipelineContext(message="hello"))
        self.assertIn("A", ctx.stage_timings)
        self.assertIn("B", ctx.stage_timings)
        self.assertIsInstance(ctx.stage_timings["A"], float)

    def test_context_carries_state(self):
        """State set by stage A is visible to stage B."""
        class SetStage(PipelineStage):
            def process(self, ctx):
                ctx.model = "groq/llama"
                return ctx
        class CheckStage(PipelineStage):
            def process(self, ctx):
                ctx.metadata["saw_model"] = ctx.model
                return ctx
        pipe = Pipeline([SetStage(), CheckStage()])
        ctx = pipe.run(PipelineContext(message="test"))
        self.assertEqual(ctx.metadata["saw_model"], "groq/llama")

    def test_empty_pipeline_is_noop(self):
        pipe = Pipeline([])
        ctx = pipe.run(PipelineContext(message="test"))
        self.assertEqual(ctx.message, "test")
        self.assertEqual(ctx.error, "")

    def test_stages_property_returns_copy(self):
        stages = [_AppendStage("A")]
        pipe = Pipeline(stages)
        returned = pipe.stages
        returned.append(_AppendStage("B"))
        self.assertEqual(len(pipe.stages), 1)


class TestConcreteStages(unittest.TestCase):

    def test_media_stage_calls_prepare(self):
        prepare = MagicMock(return_value=([{"type": "image"}], ["/tmp/img.png"]))
        stage = MediaStage(prepare_fn=prepare)
        ctx = stage.process(PipelineContext(message="look at img.png"))
        prepare.assert_called_once_with("look at img.png")
        self.assertEqual(ctx.media_paths, ["/tmp/img.png"])

    def test_media_stage_noop_without_fn(self):
        stage = MediaStage()
        ctx = stage.process(PipelineContext(message="hello"))
        self.assertIsNone(ctx.multimodal)

    def test_intent_stage_classifies(self):
        mock_detector = MagicMock()
        mock_intent = MagicMock()
        mock_intent.category = "debug"
        mock_intent.confidence = 0.9
        mock_intent.weights = MagicMock()
        mock_intent.weights.to_dict.return_value = {"debug": 0.9}
        mock_detector.classify.return_value = mock_intent
        stage = IntentStage(detector=mock_detector)
        ctx = stage.process(PipelineContext(message="fix the bug"))
        self.assertEqual(ctx.metadata["intent"], "debug")
        self.assertEqual(ctx.metadata["intent_weights"], {"debug": 0.9})

    def test_guard_stage_passes(self):
        stage = GuardStage()
        ctx = stage.process(PipelineContext(message="test"))
        self.assertTrue(ctx.metadata["guard_passed"])

    def test_route_stage_selects_model(self):
        select = MagicMock(return_value="groq/llama-3.3-70b")
        stage = RouteStage(select_fn=select)
        ctx = stage.process(PipelineContext(message="hello"))
        self.assertEqual(ctx.model, "groq/llama-3.3-70b")

    def test_context_stage_builds_system(self):
        build = MagicMock(return_value="You are HAL.")
        stage = ContextStage(build_fn=build)
        ctx = stage.process(PipelineContext(message="test"))
        self.assertEqual(ctx.system_prompt, "You are HAL.")

    def test_dispatch_stage_calls_sentinel(self):
        mock_sentinel = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello!"
        mock_sentinel.dispatch.return_value = mock_response
        stage = DispatchStage(sentinel=mock_sentinel)
        ctx = PipelineContext(message="hi", model="groq/llama", system_prompt="sys")
        ctx = stage.process(ctx)
        mock_sentinel.dispatch.assert_called_once()
        self.assertEqual(ctx.response.text, "Hello!")

    def test_dispatch_stage_no_sentinel_sets_error(self):
        stage = DispatchStage()
        ctx = stage.process(PipelineContext(message="hi"))
        self.assertEqual(ctx.error, "No sentinel configured")

    def test_record_stage_tracks_cost(self):
        mock_cost = MagicMock()
        mock_response = MagicMock()
        mock_response.model = "llama"
        mock_response.provider = "groq"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
        stage = RecordStage(cost_tracker=mock_cost)
        ctx = PipelineContext(message="test")
        ctx.response = mock_response
        ctx = stage.process(ctx)
        mock_cost.record_response.assert_called_once_with("llama", "groq", mock_response.usage)

    def test_record_stage_emits_event(self):
        mock_emitter = MagicMock()
        mock_response = MagicMock()
        mock_response.model = "llama"
        mock_response.provider = "groq"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
        stage = RecordStage(emitter=mock_emitter)
        ctx = PipelineContext(message="test")
        ctx.response = mock_response
        ctx = stage.process(ctx)
        mock_emitter.emit.assert_called_once()
        call_args = mock_emitter.emit.call_args
        self.assertEqual(call_args[0][0], "span.model_request_end")


class TestFullPipeline(unittest.TestCase):
    """Test the 7-stage pipeline end-to-end with mocks."""

    def test_seven_stage_pipeline(self):
        mock_detector = MagicMock()
        mock_intent = MagicMock(category="chat", confidence=0.5, weights=None)
        mock_detector.classify.return_value = mock_intent

        mock_sentinel = MagicMock()
        mock_response = MagicMock(
            text="reply", model="groq/llama", provider="groq",
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        )
        mock_sentinel.dispatch.return_value = mock_response

        mock_cost = MagicMock()

        pipe = Pipeline([
            MediaStage(),
            IntentStage(detector=mock_detector),
            GuardStage(),
            RouteStage(select_fn=lambda msg, model: "groq/llama"),
            ContextStage(build_fn=lambda msg: "You are HAL."),
            DispatchStage(sentinel=mock_sentinel),
            RecordStage(cost_tracker=mock_cost),
        ])

        ctx = pipe.run(PipelineContext(message="hello world"))
        self.assertEqual(ctx.response.text, "reply")
        self.assertEqual(ctx.model, "groq/llama")
        self.assertEqual(ctx.system_prompt, "You are HAL.")
        self.assertEqual(ctx.metadata["intent"], "chat")
        self.assertTrue(ctx.metadata["guard_passed"])
        self.assertEqual(len(ctx.stage_timings), 7)
        self.assertEqual(ctx.error, "")
        mock_cost.record_response.assert_called_once()


if __name__ == "__main__":
    unittest.main()
