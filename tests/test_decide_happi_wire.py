"""The HAPPI-layer wire contract for `decide` (happi/1.6): envelope in,
`delta`-per-question-in-order out, then `completed.usage`, or the existing
`error` event with a reason from `lib.hal.decide.contract.REASONS`.

Nothing here reaches the network: every provider is a stand-in, injected
through `dispatch_decide`'s own `provider=` keyword -- the hermetic seam the
module docstring describes. `catalog.resolve()` still runs for real where it
is reached (a pure YAML read, never a network call); only the provider's own
`decide()` is replaced. Each test names the mutation that turns it red.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.hal.decide.contract import DecisionResult
from lib.hal.happi.cmds_decide import DECIDE_COMMANDS, DECIDE_HANDLERS, dispatch_decide
from lib.hal.happi.envelope import HappiEnvelope, SUPPORTED_VERSIONS
from lib.hal.happi.events import PROTOCOL_VERSION_DEFAULT
from lib.hal.happi.runtime import SUPPORTED_COMMANDS, _dispatch_core
from tests._canonical_version import canonical_runtime_version

REPO = Path(__file__).resolve().parents[1]

QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Is this urgent?"},
    "team": {"type": "choice", "instructions": "Which team?",
             "criteria": {"billing": "Payments", "technical": None}},
}
ANSWERS = {
    "urgent": {"type": "noul", "noul": 0.95},
    "team": {"type": "choice", "choice": "billing",
             "probabilities": {"billing": 0.8, "technical": 0.2},
             "confidence": 0.6},
}


class FakeProvider:
    """A DecisionProvider stand-in -- same shape as test_decide_surfaces.py's."""

    name = "fake"

    def __init__(self, result=None, raises=None):
        self.calls, self.result, self.raises = [], result, raises

    def decide(self, model, state, questions):
        self.calls.append((model, state, dict(questions)))
        if self.raises is not None:
            raise self.raises
        return self.result or DecisionResult(ok=True, answers=dict(ANSWERS))


def _ok_result(answers=None):
    return DecisionResult(
        ok=True, answers=dict(answers or ANSWERS), model="jev-1.13.0",
        pinned="jev-1.13.0", usage={"input_tokens": 287, "output_tokens": 0},
        cost_usd="0.000012", calibrated=False,
    )


def _envelope(cmd="decide", **flags) -> HappiEnvelope:
    return HappiEnvelope(v="happi/1.6", id="d-test", cmd=cmd, flags=flags)


def _events(gen):
    return [json.loads(line) for line in gen]


def test_decide_is_wired_into_the_runtime_dispatch_chain():
    """Mutation: drop `| DECIDE_COMMANDS` from `SUPPORTED_COMMANDS` in
    `lib/hal/happi/runtime.py` -- `decide` falls out of the supported set."""
    assert "decide" in SUPPORTED_COMMANDS
    assert DECIDE_HANDLERS["decide"] is dispatch_decide
    assert DECIDE_COMMANDS == frozenset({"decide"})


def test_full_dispatch_core_round_trip_on_invalid_request():
    """The REAL coordinator (`runtime._dispatch_core`, no provider override)
    routes `cmd: decide` to `dispatch_decide`, which pre-validates and refuses
    before any provider is touched -- so this exercises the whole envelope ->
    started -> handler -> error chain with zero network risk.

    Mutation: in `runtime._dispatch_core`'s handler lookup, change
    `DECIDE_HANDLERS.get(envelope.cmd)` to only match a different cmd name --
    `decide` would then fall through to `unsupported_cmd` instead of
    `invalid_request`.
    """
    env = _envelope(state="s", questions={})
    events = _events(_dispatch_core(env))
    assert [e["type"] for e in events] == ["started", "error"]
    assert events[1]["code"] == "invalid_request"


def test_success_round_trip_yields_deltas_then_completed():
    """Mutation: in `dispatch_decide`, change `provider=provider` to
    `provider=None` in the call to `decide(...)` -- the injected fake would be
    ignored, the real (hermetically-unreachable) catalogue provider would be
    tried instead, and the stream would carry `error`, not two deltas and a
    clean `completed`.
    """
    fake = FakeProvider(_ok_result())
    events = _events(dispatch_decide(_envelope(state="s", questions=QUESTIONS), provider=fake))
    assert [e["type"] for e in events] == ["delta", "delta", "completed"]
    assert len(fake.calls) == 1


def test_one_delta_per_question_in_question_order():
    """Mutation: in `dispatch_decide`, replace `questions.items()` with
    `reversed(list(questions.items()))` -- "team" would precede "urgent",
    flipping the order assertion below.
    """
    fake = FakeProvider(_ok_result())
    events = _events(dispatch_decide(_envelope(state="s", questions=QUESTIONS), provider=fake))
    deltas = [e for e in events if e["type"] == "delta"]
    assert [d["qid"] for d in deltas] == ["urgent", "team"]
    assert deltas[0]["answer"] == ANSWERS["urgent"]
    assert deltas[1]["answer"] == ANSWERS["team"]
    assert all(isinstance(d["text"], str) and d["text"] for d in deltas)


def test_completed_usage_shape():
    """Mutation: in `cmds_decide._usage_record`, delete the
    `usage["calibrated"] = result.calibrated` line -- `calibrated` would be
    absent from `completed.usage` below.
    """
    fake = FakeProvider(_ok_result())
    events = _events(dispatch_decide(_envelope(state="s", questions=QUESTIONS), provider=fake))
    usage = events[-1]["usage"]
    assert usage == {
        "input_tokens": 287, "output_tokens": 0,
        "cost_usd": "0.000012", "model": "jev-1.13.0", "calibrated": False,
    }


@pytest.mark.parametrize("flags", [
    {"state": "s", "questions": {}},
    {"state": None, "questions": QUESTIONS},
    {"state": "s", "questions": {"q": {"type": "nope"}}},
])
def test_invalid_input_becomes_error_with_a_decide_reason(flags):
    """A request the format rejects never reaches the provider.

    Mutation: in `dispatch_decide`, remove the `try/except InvalidDecision`
    around `validate(state, questions)` -- every case here would then raise
    out of the generator instead of yielding an `error` event.
    """
    fake = FakeProvider(_ok_result())
    events = _events(dispatch_decide(_envelope(**flags), provider=fake))
    assert [e["type"] for e in events] == ["error"]
    assert events[0]["code"] == "invalid_request"
    assert not fake.calls


def test_provider_unavailable_result_becomes_error_with_its_own_reason():
    """A valid request the provider still can't answer reports ITS reason, not
    `invalid_request`.

    Mutation: in `dispatch_decide`, hardcode `code="invalid_request"` in the
    `if not result.ok` branch instead of `code=result.reason` -- the
    `unknown_model` reason below would be masked.
    """
    bad = FakeProvider(DecisionResult.unavailable("unknown_model", "no such model"))
    events = _events(dispatch_decide(_envelope(state="s", questions=QUESTIONS), provider=bad))
    assert [e["type"] for e in events] == ["error"]
    assert events[0]["code"] == "unknown_model"
    assert "no such model" in events[0]["message"]


@pytest.mark.parametrize("reason", ["shield_refused", "shield_unavailable"])
def test_shield_refusal_becomes_an_error_with_the_shield_reason_code(reason):
    """AGORA review (HAL#728): does an untrusted `state` that the redaction
    shield refuses ever reach the network unshielded? It does not -- the
    shield runs inside `lib/hal/decide/typesafe.py`'s real provider, one
    layer below this module (see the module docstring), and a refusal there
    is a `DecisionResult.unavailable(...)` this handler never distinguishes
    from any other unavailable reason. This pins that the caller-visible
    result of a shield refusal is a typed, correctly-coded `error` event,
    never a silent `ok` and never a generic one.

    Mutation: in `dispatch_decide`, hardcode `code="invalid_request"` in the
    `if not result.ok` branch instead of `code=result.reason` -- both
    `shield_refused` and `shield_unavailable` below would be masked, exactly
    as `unknown_model` is in the test above.
    """
    shielded = FakeProvider(DecisionResult.unavailable(reason, "redaction shield refused the body"))
    events = _events(dispatch_decide(_envelope(state="s", questions=QUESTIONS), provider=shielded))
    assert [e["type"] for e in events] == ["error"]
    assert events[0]["code"] == reason
    assert "redaction shield refused the body" in events[0]["message"]


def test_unknown_command_still_errors_as_before():
    """Wiring in `decide` narrows what counts as unknown; it does not change
    how an unknown cmd is reported.

    Mutation: in `runtime._dispatch_core`'s handler lookup, change
    `DECIDE_HANDLERS.get(envelope.cmd)` to
    `DECIDE_HANDLERS.get(envelope.cmd, DECIDE_HANDLERS["decide"])` -- a
    genuinely unknown cmd would then be swallowed by `dispatch_decide` (and
    error on the missing `questions` flag) instead of reaching
    `unsupported_cmd`.
    """
    events = _events(_dispatch_core(_envelope(cmd="not-a-real-cmd")))
    assert [e["type"] for e in events] == ["started", "error", "completed"]
    assert events[1]["code"] == "unsupported_cmd"


def test_version_1_6_throughout():
    """The canonical polyglot runtime, the library's default protocol
    version (H428), and the library's accepted envelope-version set all
    agree on happi/1.6 -- and a live `version` dispatch actually emits it.

    Mutation: in `lib/hal/happi/events.py`, revert `PROTOCOL_VERSION_DEFAULT`
    to `"happi/1.5"` -- `canonical_runtime_version()` (read from the OTHER
    declaration, in `happi-sources/04-python-runtime.py`) still returns
    `"happi/1.6"`, so the cross-file equality assertion fails.
    """
    assert canonical_runtime_version() == "happi/1.6"
    assert PROTOCOL_VERSION_DEFAULT == "happi/1.6"
    assert "happi/1.6" in SUPPORTED_VERSIONS

    events = _events(_dispatch_core(HappiEnvelope(v="happi/1.0", id="v-1", cmd="version")))
    assert [e["type"] for e in events] == ["started", "delta", "completed"]
    delta = events[1]
    assert delta["v"] == "happi/1.6"
    assert delta["text"] == "happi/1.6"
