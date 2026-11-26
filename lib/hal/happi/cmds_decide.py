"""HAPPI ``decide`` cmd -- typed decisions from a decision model (v1.6).

Where ``chat``/``infer`` route a PROMPT to a model that answers in text,
``decide`` routes a ``state`` plus a map of typed questions to a DECISION
model that answers each one with a checked Noul/Choice/Score -- never free
text. This is "happi.md as the shim" for HAL's ``lib/hal/decide/`` package,
following the same division as ``infer`` (``lib/hal/happi/cmds_infer.py``):
the embedded polyglot runtime in ``happi.md`` stays stdlib-only and does not
implement ``decide`` -- reaching a decision model needs a network call, a
spend guard (``config/spend_guard.json``) and GRIP's redaction shield, none
of which belong in that runtime. This module is the reference LIBRARY
implementation, wired into ``lib/hal/happi/runtime.py`` beside
``INFER_HANDLERS``.

**Where the shield actually runs -- not here.** This module validates the
envelope and streams whatever ``dispatch.decide`` returns; it does not touch
the network, the spend guard or the shield itself. Those three run one layer
down, inside ``lib/hal/decide/typesafe.py``'s ``TypeSafeDecisionProvider.decide()``
-- the circuit breaker, then the spend guard, then the key, then GRIP's
redaction shield, in that order, all before anything is sent (``docs/decide.md``,
"What every request goes through"). A shield that cannot scrub the request, or
that is missing entirely, never reaches the network: the provider returns
``DecisionResult.unavailable("shield_refused", ...)`` or
``unavailable("shield_unavailable", ...)``, and this module's own
``if not result.ok`` branch below turns that straight into a HAPPI ``error``
event carrying that exact code -- it does not need its own shield check,
because an unshielded request never produces an ``ok`` result to stream.

Wire contract (``happi.md``, "decide -- typed decisions (v1.6)"):
    inputs:  ``flags.state`` (str|dict|list), ``flags.questions`` (map),
             ``flags.model`` (str, optional)
    success: one ``delta`` per question, IN QUESTION ORDER, each carrying
             ``{qid, answer}``; then one ``completed`` whose ``usage`` carries
             ``input_tokens``, ``output_tokens``, ``cost_usd``, ``model``, and
             ``calibrated`` (always ``False`` today -- see ``docs/decide.md``).
    failure: the EXISTING ``error`` event, ``code`` taken verbatim from
             ``lib.hal.decide.contract.REASONS`` -- no new event type, ever.

The ``provider`` keyword is the hermetic test seam. Production dispatch (the
handler table in ``runtime.py``) calls this with one positional argument, so
``provider`` always defaults to ``None`` there and ``dispatch.decide``
resolves the real catalogue + network provider. A test calls
``dispatch_decide(envelope, provider=FakeProvider())`` directly to exercise
the full wire contract with no network, keychain or spend-guard access --
the same seam ``tests/test_decide_contract.py`` and friends already use on
``dispatch.decide`` itself, threaded one layer up.
"""
from __future__ import annotations

from typing import Iterator, Mapping, Optional

from lib.hal.decide.contract import DecisionProvider, InvalidDecision, validate
from lib.hal.decide.dispatch import DEFAULT_MODEL, decide
from lib.hal.happi.envelope import HappiEnvelope
from lib.hal.happi.events import emit_completed, emit_error, emit_event


def _render_answer(qid: str, answer: Mapping) -> str:
    """A short human-readable render of one typed answer, for ``delta.text`` --
    the same convenience ``cite.verify``'s ``"<id> <status>"`` deltas give a
    reader watching the NDJSON stream. The full typed answer rides alongside
    as ``answer``; this text is never the only copy of it.
    """
    kind = answer.get("type")
    if kind == "noul":
        return f"{qid} noul={answer.get('noul')}"
    if kind == "choice":
        return f"{qid} choice={answer.get('choice')}"
    if kind == "score":
        return f"{qid} score={answer.get('score')}"
    return f"{qid} {kind}"


def _usage_record(result) -> dict:
    """``completed.usage`` exactly as happi.md's decide section documents it:
    the provider's own token usage, plus cost, the model that actually
    answered, and ``calibrated`` -- always ``False`` today (no labelled-set
    measurement is on file for any question; see ``docs/decide.md``)."""
    usage = dict(result.usage)
    usage["cost_usd"] = result.cost_usd
    usage["model"] = result.model
    usage["calibrated"] = result.calibrated
    return usage


def dispatch_decide(envelope: HappiEnvelope, *,
                     provider: Optional[DecisionProvider] = None) -> Iterator[str]:
    """Dispatch one ``decide`` envelope through HAL's decision-model package.

    Coordinator: pre-validate -> dispatch -> stream answers -> complete. Never
    raises -- every failure, including a malformed request, becomes an
    ``error`` event carrying a code from ``REASONS``, matching every other
    HAPPI handler's never-raise contract. Pre-validating with
    ``contract.validate`` (rather than relying solely on ``dispatch.decide``'s
    own internal check) means a malformed request never reaches the catalogue
    or the network at all -- the handler's own first concern, visible in its
    own control flow rather than inherited silently from the callee.
    """
    flags = envelope.flags or {}
    state = flags.get("state")
    questions = flags.get("questions")
    model = str(flags.get("model") or DEFAULT_MODEL)

    try:
        validate(state, questions)
    except InvalidDecision as exc:
        yield emit_error(envelope.id, str(exc), code="invalid_request")
        return

    result = decide(state, questions, model=model, provider=provider)
    if not result.ok:
        yield emit_error(envelope.id, result.detail or result.reason, code=result.reason)
        return

    # `questions` is a Mapping once `validate` has passed, and a JSON object's
    # key order survives `json.loads` in CPython -- so iterating it directly
    # is iterating in the order the caller asked, with no separate ordering
    # concept to keep in sync.
    for qid, question in questions.items():
        answer = result.answers[qid]
        yield emit_event(envelope.id, "delta", text=_render_answer(qid, answer),
                          qid=qid, answer=answer)

    yield emit_completed(envelope.id, usage=_usage_record(result))


# Extension dispatch table -- merged by runtime._dispatch_core via the same
# .get() lookup pattern as INFER_HANDLERS (no if/elif chain). Kept OUT of
# cmds_v11.py so that module stays byte-parity with the canonical polyglot
# happi.md runtime (which has no `decide` cmd -- see this module's docstring).
DECIDE_HANDLERS = {"decide": dispatch_decide}
DECIDE_COMMANDS = frozenset(DECIDE_HANDLERS)
