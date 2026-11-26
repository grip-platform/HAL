"""The shipped W4 config arms Alibaba Token Plan, and each of its targets accepts
a request with the field set Claude Code 2.1.289 really sends, in its empty forms.

HAL#710 armed a second Token Plan model (deepseek-v4-pro beside qwen3.8-max).
This file still demanded exactly one Token Plan target, so its fixture errored
on main from then on, unseen because no workflow ran the file. Every check now
runs against each shipped target, and the set itself is pinned.

Measured 2026-10-08: a real Claude Code 2.1.289 fresh-thread request, captured
on a loopback fake upstream, carried exactly these top-level fields --
context_management, max_tokens, messages, metadata, model, output_config,
safeguards, stream, system, thinking, tools -- plus a system-role message inside
messages. The first draft of this target's `drop` list left `safeguards` out,
and transform() refused that real request ("fields"). `metadata` was never the
cause: the converter exempts it from that check and strips it unconditionally,
regardless of what `drop` names (transform() in lib/hal/wire_cascade.py), so
listing it in `drop` was redundant and has been removed from the shipped
config. The body below is synthetic with the same field set; no captured
content is committed.

A declared drop is value-sensitive. `context_management`, `safeguards` and
`output_config` can carry a requirement the target would never see, so they
are dropped only in their empty forms (plus an effort-only `output_config`);
anything else refuses the move with "fields", naming the field and never its
value. `thinking` and `diagnostics` are dropped whatever they hold: that is the
downgrade the operator accepted when these targets were armed.

The VALUES in the body below are synthetic, and a real request does not carry
those empty forms. Probed 2026-10-12: five `claude -p` turns of Claude Code
2.1.296 against a loopback endpoint (API-key auth, an isolated config folder,
model claude-opus-5-5; only field shapes were recorded). Every one carried one
`safeguards` entry with the keys `type` and `classifier_context`. The last
probe also recorded its type, `dangerous_tool_use`, and the key names inside
`classifier_context`, among them `home_dir` and `user_identity` (no value was
read). Treat it as naming the user and the machine: it must never be
forwarded to another vendor. With thinking on, `context_management` was
`{"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}`; with
`--thinking disabled` the field was absent. `output_config` held only `effort`.
So under this rule neither shipped target takes that request: its
`safeguards` entry refuses it (test_the_request_claude_code_2_1_296_sends_is_refused).

Probed 2026-10-15 for HAL#676: Claude Code 2.1.296 in a real terminal, through
the wire with a subscription bearer, sends that same edit on every turn with
its thinking setting on or off, with and without `--safe-mode`, and no
`safeguards` field. "keep": "all" keeps every thinking block (Anthropic's
context-editing page), so the edit clears nothing and is dropped like an empty
form (test_claude_codes_keep_all_thinking_edit_is_dropped). Any other edit
still refuses.
"""
import copy
import dataclasses
import hashlib
import json
from collections import UserDict
from types import SimpleNamespace

import pytest

from lib.hal import cascade_policy as policy
from lib.hal import wire_cascade as wc

CC_2_1_289_FIELDS = {
    "context_management", "max_tokens", "messages", "metadata", "model",
    "output_config", "safeguards", "stream", "system", "thinking", "tools",
}

SENTINEL = "SECRET_SENTINEL"
NESTED_KEY = SENTINEL + "_KEY"

# One constraint per row, in a field the shipped targets drop. Every row but the
# two near misses of Claude Code's own edit plants the sentinel in a value AND in
# a nested key: neither may reach a refusal or an audit row.
CONSTRAINTS = [
    ("output_config", {"format": {"type": "json_schema", "schema": {NESTED_KEY: SENTINEL}}}),
    ("context_management", {"edits": [{NESTED_KEY: SENTINEL}]}),
    # Near misses of KEEP_ALL_THINKING: an edit that clears blocks, and the
    # keep-all edit beside another one. Each is a requirement, so each refuses.
    ("context_management", {"edits": [{"type": "clear_thinking_20251015",
                                       "keep": {"type": "thinking_turns", "value": 1}}]}),
    ("context_management", {"edits": [{"type": "clear_thinking_20251015", "keep": "all"},
                                      {"type": "clear_tool_uses_20250919"}]}),
    ("safeguards", [{NESTED_KEY: SENTINEL}]),
    ("output_config", {"effort": "max", NESTED_KEY: SENTINEL}),
    ("context_management", {"edits": [], NESTED_KEY: SENTINEL}),
]


def _body():
    return {
        "model": "claude-opus-5-5", "max_tokens": 32000, "stream": True,
        "system": [{"type": "text", "text": "system prompt"}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]},
                     {"role": "system", "content": "a reminder"}],
        "tools": [{"name": "Read", "description": "Read a file.",
                   "input_schema": {"type": "object", "properties": {}}}],
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "max"},
        "context_management": {"edits": []},
        "metadata": {"user_id": "user_0000_synthetic"},
        "safeguards": [],
    }


def _expected(target):
    """What transform() makes of _body(): the dropped fields and metadata gone,
    the system-role message folded into system, nothing else changed."""
    return {
        "model": target.model, "max_tokens": 32000, "stream": True,
        "system": [{"type": "text", "text": "system prompt"},
                   {"type": "text", "text": "a reminder"}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        "tools": [{"name": "Read", "description": "Read a file.",
                   "input_schema": {"type": "object", "properties": {}}}],
    }


TOKEN_PLAN_MODELS = ("qwen3.8-max", "deepseek-v4-pro")


@pytest.fixture(params=TOKEN_PLAN_MODELS)
def token_plan(monkeypatch, request):
    monkeypatch.delenv("HAL_W4_CONFIG", raising=False)
    found = [t for t in wc.configured_targets()
             if t.provider == "qwen-token-plan" and t.model == request.param]
    assert len(found) == 1, wc.config_errors()
    return found[0]


def test_shipped_config_has_no_errors(monkeypatch):
    monkeypatch.delenv("HAL_W4_CONFIG", raising=False)
    assert wc.config_errors() == []


def test_shipped_token_plan_targets_are_exactly_these(monkeypatch):
    # Adding or removing a Token Plan target must be a deliberate edit here,
    # so that every shipped target stays covered by the checks below.
    monkeypatch.delenv("HAL_W4_CONFIG", raising=False)
    found = [t.model for t in wc.configured_targets() if t.provider == "qwen-token-plan"]
    assert sorted(found) == sorted(TOKEN_PLAN_MODELS), wc.config_errors()


def test_token_plan_target_shape(token_plan):
    assert (token_plan.auth, token_plan.billing,
            token_plan.system_messages) == ("bearer", "subscription", "fold")
    assert token_plan.window == 983616


def test_fixture_carries_the_measured_field_set():
    assert set(_body()) == CC_2_1_289_FIELDS


def test_token_plan_accepts_a_claude_code_2_1_289_request(token_plan):
    # The measured field SET with synthetic, empty settings (see the docstring).
    assert wc.transform(token_plan, _body()) == _expected(token_plan)


def test_the_request_claude_code_2_1_296_sends_is_refused(token_plan):
    # The headless shape probed on 2026-10-12 (see the docstring). The
    # classifier_context here is synthetic: only the real one's key names were
    # recorded. The keep-all edit is dropped; the safeguards entry still refuses,
    # and whoever decides that it may be dropped changes this test on purpose.
    body = _body()
    body["thinking"] = {"type": "adaptive", "display": "omitted"}
    body["context_management"] = {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}
    body["safeguards"] = [{"type": "dangerous_tool_use", "classifier_context": {"v": 0}}]
    with pytest.raises(policy.Refused) as raised:
        wc.transform(token_plan, body)
    assert raised.value.detail == {"fields": ["safeguards"], "unnamed": 0}
    # With thinking disabled the edit list is absent, and the entry still refuses.
    del body["thinking"], body["context_management"]
    with pytest.raises(policy.Refused) as raised:
        wc.transform(token_plan, body)
    assert raised.value.detail == {"fields": ["safeguards"], "unnamed": 0}


def test_claude_codes_keep_all_thinking_edit_is_dropped(token_plan):
    # The interactive shape probed on 2026-10-15 for HAL#676: the keep-all edit
    # and no safeguards entry. The edit clears nothing, so it goes like an empty
    # form and the target gets the same request as from the empty forms.
    body = _body()
    body["context_management"] = {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}
    del body["safeguards"]
    assert wc.transform(token_plan, body) == _expected(token_plan)


@pytest.mark.parametrize("field,value", CONSTRAINTS)
def test_constraint_in_declared_drop_refuses_without_value(token_plan, field, value):
    body = _body()
    body[field] = value
    original = copy.deepcopy(body)
    with pytest.raises(policy.Refused) as raised:
        wc.transform(token_plan, body)
    assert str(raised.value) == "fields"
    assert raised.value.args == ("fields",)
    assert raised.value.detail == {"fields": [field], "unnamed": 0}
    assert SENTINEL not in repr(raised.value.detail)
    assert body == original


def _one_attempt(monkeypatch, target, body):
    """Run the real _attempt() with no socket. Returns its result, the audit
    rows it wrote through _record() and policy.record_shape(), every dial it
    made, and every payload it handed to the shield."""
    seen = SimpleNamespace(rows=[], dialled=[], shielded=[])

    def dial(_proxy, dialled_target, _ex):
        seen.dialled.append(dialled_target.label)
        raise ConnectionRefusedError

    def shield(payload):
        seen.shielded.append(sorted(payload))
        return payload

    monkeypatch.setattr(policy, "load_shield", lambda: shield)
    monkeypatch.setattr(wc, "_connect", dial)
    proxy = SimpleNamespace(_telemetry=SimpleNamespace(record=seen.rows.append))
    ex = SimpleNamespace(headers=(("x-claude-code-session-id", "fake-session"),))
    seen.result = wc._attempt(proxy, None, ex, target, "fake-key", body, "refused")
    return seen


@pytest.mark.parametrize("field,value", CONSTRAINTS)
def test_constraint_refusal_reaches_the_audit_as_a_field_name_only(
        monkeypatch, token_plan, field, value):
    body = _body()
    body[field] = value
    seen = _one_attempt(monkeypatch, token_plan, body)
    assert seen.result == (None, None)  # refused; a later target may still be tried
    # Refused first: the request is not prepared for sending, and nothing is dialled.
    assert seen.shielded == []
    assert seen.dialled == []
    # The whole row, so nothing beside the field name can ride along.
    assert seen.rows == [{
        "event": "w4.cascade",
        "session_hash": hashlib.sha256(b"fake-session").hexdigest(),
        "target": token_plan.label, "reason": "fields", "outcome": "refused",
        "detail": {"fields": [field], "unnamed": 0},
    }]
    assert SENTINEL not in json.dumps(seen.rows)


def test_the_measured_request_gets_as_far_as_the_dial(monkeypatch, token_plan):
    # The control for the test above: the same harness does shield the request
    # and reach the target's dial when nothing refuses, so the two empty lists
    # there are not vacuous.
    seen = _one_attempt(monkeypatch, token_plan, _body())
    assert seen.result == (None, None)
    assert seen.shielded == [sorted(_expected(token_plan))]
    assert seen.dialled == [token_plan.label]
    assert [row["reason"] for row in seen.rows] == ["before_send"]


def test_shipped_drop_lists_hold_only_fields_with_a_decided_rule(token_plan):
    # _droppable_field() drops any other declared name whatever it holds. A new
    # name in a shipped drop list must therefore be a deliberate edit here,
    # made together with a decision about what that field may lose.
    assert set(token_plan.drop) <= {
        "context_management", "diagnostics", "output_config", "safeguards", "thinking"}


@pytest.mark.parametrize("field,value", [
    ("output_config", SENTINEL),
    ("context_management", SENTINEL),
    ("safeguards", SENTINEL),
    # Exact shapes, never truthiness: an empty-looking value of another type
    # is not one of the approved empty forms.
    ("safeguards", {}),
    ("safeguards", None),
    ("output_config", None),
    ("output_config", []),
    ("context_management", None),
    ("context_management", []),
    ("context_management", {"edits": None}),
    ("context_management", {"edits": {}}),
])
def test_wrong_drop_field_type_refuses(token_plan, field, value):
    body = _body()
    body[field] = value
    with pytest.raises(policy.Refused) as raised:
        wc.transform(token_plan, body)
    assert str(raised.value) == "fields"
    assert raised.value.detail == {"fields": [field], "unnamed": 0}
    assert SENTINEL not in repr(raised.value.detail)


class _Dict(dict):
    """A dict that is not a plain dict: not one of the approved empty forms."""


class _List(list):
    """A list that is not a plain list: not one of the approved empty forms."""


@pytest.mark.parametrize("field,value", [
    ("context_management", _Dict()),
    ("context_management", _Dict(edits=[])),
    ("context_management", UserDict()),
    ("safeguards", _List()),
    ("output_config", _Dict()),
    ("output_config", _Dict(effort="max")),
])
def test_an_empty_form_in_a_container_that_is_not_plain_refuses(token_plan, field, value):
    # The council's falsifier on HAL#738: a container subclass that passes the
    # shape check and then changes the caller's body under the deep copy. It
    # cannot pass: only a plain dict or list is an approved empty form (type(),
    # not isinstance()), and a wire body comes from json.loads, which makes
    # plain ones. The refusal leaves the body, and the container's type, as
    # they were.
    body = _body()
    body[field] = value
    original = copy.deepcopy(body)
    with pytest.raises(policy.Refused) as raised:
        wc.transform(token_plan, body)
    assert raised.value.detail == {"fields": [field], "unnamed": 0}
    assert body == original
    assert type(body[field]) is type(value)


@pytest.mark.parametrize("field,value", [
    ("context_management", {}),
    ("context_management", {"edits": []}),
    ("safeguards", []),
    ("output_config", {}),
    ("output_config", {"effort": "max"}),
    ("output_config", {"effort": "low"}),
    # The accepted downgrade: thinking and diagnostics go whatever they hold.
    ("thinking", {"type": "enabled", "budget_tokens": 31999}),
    ("diagnostics", {"trace": SENTINEL}),
])
def test_approved_empty_or_downgrade_shape_transforms_like_the_measured_request(
        token_plan, field, value):
    body = _body()
    body[field] = value
    original = copy.deepcopy(body)
    assert wc.transform(token_plan, body) == _expected(token_plan)
    assert body == original


@pytest.mark.parametrize("field", ["context_management", "safeguards", "output_config", "thinking"])
def test_an_absent_drop_field_is_not_a_constraint(token_plan, field):
    body = _body()
    del body[field]
    assert wc.transform(token_plan, body) == _expected(token_plan)


def test_every_blocked_field_is_named_once(token_plan):
    # Two constraints at once, and a config that lists one drop name twice.
    doubled = dataclasses.replace(token_plan, drop=token_plan.drop + ("safeguards",))
    body = _body()
    body["safeguards"] = [SENTINEL]
    body["output_config"] = {"format": {"type": "json_schema", "schema": {NESTED_KEY: SENTINEL}}}
    with pytest.raises(policy.Refused) as raised:
        wc.transform(doubled, body)
    assert raised.value.detail == {"fields": ["output_config", "safeguards"], "unnamed": 0}


def test_a_target_that_accepts_the_field_keeps_it_whole(token_plan):
    # The rule guards a DROP only: a target that takes the field gets all of it.
    accepting = dataclasses.replace(
        token_plan, accept=("output_config",),
        drop=tuple(name for name in token_plan.drop if name != "output_config"))
    body = _body()
    body["output_config"] = {"effort": "max",
                             "format": {"type": "json_schema", "schema": {"type": "object"}}}
    out = wc.transform(accepting, body)
    assert out == dict(_expected(token_plan), output_config=body["output_config"])


def test_transform_keeps_callers_nested_body(token_plan):
    body = _body()
    body["tools"][0]["defer_loading"] = True
    original = copy.deepcopy(body)
    out = wc.transform(token_plan, body)
    assert body == original
    assert "defer_loading" not in out["tools"][0]


def test_no_account_identifier_leaves(token_plan):
    # metadata is not in the shipped target's drop list at all -- this proves
    # the converter's OWN unconditional strip (in transform()), not a config
    # choice.
    out = wc.transform(token_plan, _body())
    assert "metadata" not in out
    assert "user_0000_synthetic" not in json.dumps(out)


@pytest.mark.parametrize("field", ["safeguards"])
def test_each_newly_measured_field_must_be_dropped(token_plan, field):
    narrowed = dataclasses.replace(
        token_plan, drop=tuple(name for name in token_plan.drop if name != field))
    with pytest.raises(policy.Refused):
        wc.transform(narrowed, _body())
