"""Integration tests for Conversation's hooks to Session Replay (#34),
Input Attribution (#35), and User Adaptation (#36).

These exercise the conversation.py surface without hitting a real provider.
Everything the user-facing contract promises — opt-in attribution, automatic
feedback inference, recorder wiring — is covered here.
"""
import os
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.hal.attribution import AttributionReport, AttributionScore
from lib.hal.conversation import Conversation


@pytest.fixture
def stub_registry():
    reg = mock.MagicMock()
    reg.routing.default_model = "test/model"
    reg.list_providers.return_value = []
    return reg


@pytest.fixture
def conversation(stub_registry):
    """Build a Conversation without triggering the full pipeline."""
    with mock.patch("lib.hal.conversation.ContextMonitor"), \
         mock.patch("lib.hal.conversation.SessionManager"), \
         mock.patch("lib.hal.conversation.LimitSentinel"), \
         mock.patch("lib.hal.conversation.CostTracker"), \
         mock.patch("lib.hal.conversation.SessionRecorder"), \
         mock.patch("lib.hal.conversation.IntentDetector"), \
         mock.patch("lib.hal.conversation.ContextAssembler"):
        conv = Conversation(
            model="test/model",
            registry=stub_registry,
            system_prompt="test system",
        )
        yield conv


class TestAttributionOptIn35:
    """#35: InputAttributor must be opt-in; default disabled."""

    def test_attributor_disabled_by_default(self, conversation):
        assert conversation._attributor is None

    def test_attribute_last_returns_none_when_disabled(self, conversation):
        result = conversation.attribute_last({"instructions": "do X"})
        assert result is None

    def test_attribute_last_returns_none_before_any_response(self, stub_registry):
        with mock.patch("lib.hal.conversation.ContextMonitor"), \
             mock.patch("lib.hal.conversation.SessionManager"), \
             mock.patch("lib.hal.conversation.LimitSentinel"), \
             mock.patch("lib.hal.conversation.CostTracker"), \
             mock.patch("lib.hal.conversation.SessionRecorder"), \
             mock.patch("lib.hal.conversation.IntentDetector"), \
             mock.patch("lib.hal.conversation.ContextAssembler"):
            conv = Conversation(
                model="test/model",
                registry=stub_registry,
                system_prompt="sys",
                enable_attribution=True,
            )
        assert conv._attributor is not None
        assert conv.attribute_last({"a": "b"}) is None  # no response captured yet

    def test_attribute_last_forwards_to_attributor(self, stub_registry):
        """With enable_attribution=True + captured response, forwards to InputAttributor."""
        with mock.patch("lib.hal.conversation.ContextMonitor"), \
             mock.patch("lib.hal.conversation.SessionManager"), \
             mock.patch("lib.hal.conversation.LimitSentinel"), \
             mock.patch("lib.hal.conversation.CostTracker"), \
             mock.patch("lib.hal.conversation.SessionRecorder"), \
             mock.patch("lib.hal.conversation.IntentDetector"), \
             mock.patch("lib.hal.conversation.ContextAssembler"):
            conv = Conversation(
                model="test/model",
                registry=stub_registry,
                system_prompt="sys",
                enable_attribution=True,
            )
        fake_report = AttributionReport(
            scores=(AttributionScore(section="instructions", importance=1.0),),
            method="rating",
            total_sections=1,
        )
        conv._last_response_text = "response text"
        conv._attributor = mock.MagicMock()
        conv._attributor.attribute_sections.return_value = fake_report

        got = conv.attribute_last(
            {"instructions": "do X"}, method="ablation", model="m"
        )
        assert got is fake_report
        conv._attributor.attribute_sections.assert_called_once_with(
            sections={"instructions": "do X"},
            response="response text",
            method="ablation",
            model="m",
        )


class TestUserAdaptFeedbackInference36:
    """#36: send() must pass each user message through observe_from_feedback."""

    def test_observe_from_feedback_is_invoked(self, conversation):
        conversation._user_adapter = mock.MagicMock()
        conversation._pipeline = mock.MagicMock()
        fake_response = mock.MagicMock()
        fake_response.text = "reply"
        fake_ctx = mock.MagicMock()
        fake_ctx.error = None
        fake_ctx.response = fake_response
        conversation._pipeline.run.return_value = fake_ctx
        conversation._prepare_media = mock.MagicMock(return_value=(None, []))
        conversation._context_monitor = mock.MagicMock()

        conversation.send("please be shorter — too long")

        conversation._user_adapter.observe_from_feedback.assert_called_once_with(
            "please be shorter — too long"
        )

    def test_last_response_captured_for_attribution(self, conversation):
        conversation._pipeline = mock.MagicMock()
        fake_response = mock.MagicMock()
        fake_response.text = "the answer"
        fake_ctx = mock.MagicMock()
        fake_ctx.error = None
        fake_ctx.response = fake_response
        conversation._pipeline.run.return_value = fake_ctx
        conversation._prepare_media = mock.MagicMock(return_value=(None, []))
        conversation._context_monitor = mock.MagicMock()

        conversation.send("hi")
        assert conversation._last_response_text == "the answer"


class TestSessionRecorderWired34:
    """#34: SessionRecorder wired into __init__ and send() already."""

    def test_recorder_attached(self, conversation):
        assert conversation._recorder is not None

    def test_recorder_called_on_turn(self, conversation):
        conversation._recorder = mock.MagicMock()
        conversation._pipeline = mock.MagicMock()
        fake_response = mock.MagicMock()
        fake_response.text = "a"
        fake_ctx = mock.MagicMock()
        fake_ctx.error = None
        fake_ctx.response = fake_response
        conversation._pipeline.run.return_value = fake_ctx
        conversation._prepare_media = mock.MagicMock(return_value=(None, []))
        conversation._context_monitor = mock.MagicMock()

        conversation.send("hello")
        # One user record, one assistant record
        calls = conversation._recorder.record.call_args_list
        assert len(calls) == 2
        assert calls[0].kwargs["role"] == "user"
        assert calls[1].kwargs["role"] == "assistant"
