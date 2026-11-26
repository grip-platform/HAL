"""W4: request-boundary provider selection after the account pool is spent.

Only policy is shared with RAILLM. Conversation transport and translation stay
in the gateway. No request headers, prompt text or credentials enter the audit.
The default target list is empty: a vendor offering a free tier does not prove
that the gateway's credential uses that tier. See docs/w4-provider-cascade.md.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from lib.hal import cascade_policy as policy
from lib.hal import spend_guard
from lib.hal.account_ledger import AccountLedger, LedgerView
from lib.hal.cascade_decision import sign_failover_decision
from lib.hal.cascade_target import Target, is_valid, validate
from lib.hal.gateway_failover import AccountsSpent
from lib.hal.model_catalog import ModelCatalog
from lib.hal.raillm_ladder import build_grip_ladder, ladder_cascade_order

CONFIG = Path(__file__).resolve().parents[2] / 'config' / 'w4_cascade.json'
# These gateway transports support ordinary function definitions. Native hosted
# tools (search, computer, MCP, custom grammars, etc.) are deliberately refused.
FUNCTION_PROVIDERS = frozenset({'moonshot', 'groq', 'gemini', 'deepseek'})
REPLAY_FIELDS = frozenset({'previous_response_id', 'conversation', 'encrypted_content',
                          'reasoning', 'reasoning_content', 'reasoning_details',
                          'thought_signature', 'signature', 'extra_content',
                          'cache_control', 'store', 'include'})
REPLAY_TYPES = frozenset({'reasoning', 'thinking', 'redacted_thinking'})
DECLARED_BILLING = 'unknown'  # OpenAI target addresses declare no credential tier.


def configured_targets() -> list[str | Target]:
    """Select exact product bindings or known legacy ladder addresses."""
    try:
        path = Path(os.environ.get('HAL_W4_CONFIG', str(CONFIG)))
        entries = json.loads(path.read_text(encoding="utf-8"))['targets']
        if not isinstance(entries, list):
            return []
        legacy = [entry for entry in entries if isinstance(entry, str)]
        ordered = [a for a in ladder_cascade_order() if a in legacy]
        return ordered + [target for entry in entries if (target := validate(entry))]
    except (OSError, ValueError, KeyError, TypeError):
        return []


def normal_id(value: str, prefix: str = 'call_') -> str:
    return prefix + hashlib.sha256(value.encode()).hexdigest()[:32]


def clean_request(payload: dict) -> dict:
    """Copy protocol fields, preserving user text and opaque tool schemas/data.

    IDs are mapped deterministically, so a tool result still names its call.
    Never recurse into a JSON schema or tool arguments: keys there are user data.
    """
    def clean(value):
        if isinstance(value, list):
            return [clean(v) for v in value if not (
                isinstance(v, dict) and v.get('type') in REPLAY_TYPES)]
        if not isinstance(value, dict):
            return value
        out = {}
        for key, val in value.items():
            if key in REPLAY_FIELDS:
                continue
            if key in {'parameters', 'input_schema', 'arguments', 'input', 'output'} and (
                    key != 'input' or value is not payload):
                out[key] = val
            elif key in {'id', 'call_id', 'tool_call_id'} and isinstance(val, str):
                out[key] = normal_id(val, 'resp_' if val.startswith('resp_') else 'call_')
            else:
                out[key] = clean(val)
        return out
    return clean(payload)


def request_budget(payload: dict, path: str) -> int | None:
    """Conservative text-only upper bound, including output and framing.

    One token per serialized UTF-8 byte bounds byte-BPE and byte-fallback
    tokenizers; ensure_ascii only inflates that bound. Add 4096 framing tokens. Images/audio/files/reference-only
    histories have no byte-derived token bound and are refused, never dropped.
    """
    # A server-side continuation hides history from both the size bound and
    # the translator. Stripping its ID cannot reconstruct that missing context.
    if payload.get('previous_response_id') or payload.get('conversation'):
        return None
    history = payload.get('input' if path == '/v1/responses' else 'messages')
    if not history or not isinstance(history, (str, list)):
        return None
    allowed_types = {'message', 'text', 'input_text', 'output_text', 'function_call',
                     'function_call_output', 'function'} | REPLAY_TYPES
    def text_only(value):
        if isinstance(value, list):
            return all(text_only(v) for v in value)
        if not isinstance(value, dict):
            return isinstance(value, (str, int, float, bool, type(None)))
        if value.get('type') == 'function_call_output' and not isinstance(value.get('output'), str):
            return False
        if value.get('type', 'message') not in allowed_types:
            return False
        return all(text_only(v) for k, v in value.items()
                   if k not in {'arguments', 'parameters', 'output'} | REPLAY_FIELDS)
    if not text_only(history):
        return None
    return policy.budget(payload)


def refusal(address: str | Target, payload: dict, path: str) -> str | None:
    """All mandatory gates, prior to credentials or network access."""
    target = address if isinstance(address, Target) else None
    if target is not None and not is_valid(target):
        return 'spend_guard'
    provider = target.provider if target else address.split('/', 1)[0]
    if not policy.billing_allowed(provider, target.billing if target else None):
        return 'spend_guard'
    if not supported_tools(payload, path):
        return 'unsupported_tool'
    if _context_refused(address, target, payload, path):
        return 'context_unknown_or_small'
    if target is not None and not supported_constraints(payload, path):
        return 'unsupported_tool'
    if not _transport_supported(provider, target):
        return 'unsupported_transport'
    return None


def _context_refused(address, target, payload, path):
    model = target.model if target else address.split('/', 1)[1]
    info = ModelCatalog().get(model) if target is None else None
    windows = ([target.window] if target else
               [tier.context_window for tier in build_grip_ladder().values()
                if address in tier.members])
    budget = request_budget(payload, path)
    return not windows or budget is None or budget > min(
        windows + ([info.context_window] if info else []))


def _transport_supported(provider, target):
    return target.transport == 'openai-chat' if target else provider in FUNCTION_PROVIDERS


def supported_constraints(payload: dict, path: str) -> bool:
    """Only fields preserved by both halves of the Anthropic pivot may pass."""
    common = {'model', 'stream', 'temperature', 'top_p', 'tools', 'tool_choice'}
    allowed = (common | {'messages', 'max_tokens', 'stop'} if path == '/v1/chat/completions'
               else common | {'input', 'instructions', 'max_output_tokens'})
    return not (set(payload) - allowed)


def supported_tools(payload: dict, path: str) -> bool:
    """Never permit a declaration the existing translators silently discard."""
    history = payload.get('input', [])
    if isinstance(history, list) and any(isinstance(item, dict) and
            item.get('type') == 'additional_tools' for item in history):
        return False
    if 'additional_tools' in payload:
        return False
    if not _function_tools_supported(payload.get('tools', []), path):
        return False
    # Responses' pivot cannot preserve a forced tool choice. Neither pivot
    # preserves parallel-call constraints, so fail closed on those as well.
    choice = payload.get('tool_choice', 'auto')
    if path == '/v1/responses' and choice != 'auto':
        return False
    if path == '/v1/chat/completions' and choice == 'none':
        return False
    return 'parallel_tool_calls' not in payload


def _function_tools_supported(tools, path):
    if not isinstance(tools, list):
        return False
    for tool in tools:
        if not isinstance(tool, dict) or tool.get('type') != 'function':
            return False
        definition = tool if path == '/v1/responses' else tool.get('function')
        if not isinstance(definition, dict) or not definition.get('name'):
            return False
        if set(definition) - {'type', 'name', 'description', 'parameters'}:
            return False
    return True


_before_send = policy.before_send


def _record(payload, path, exhausted, declined, *, provider='none', model='', resp=None,
            session_id=None, target=None):
    from lib.hal.gateway_witness import witness_chat
    witness_chat(provider=provider, model=model or payload.get('model', ''),
                 status=resp.status if resp is not None else exhausted.decision.status,
                 served=resp is not None, failed_over=True, started=time.monotonic(),
                 path=path, route='inference', row_class='failover' if resp else 'pause',
                 billing=target.billing if target else DECLARED_BILLING, session_id=session_id,
                 usage=getattr(resp, 'usage', None), extra={
                     **policy.record_shape(session_id, provider + '/' + model, 'all_configured_accounts_spent',
                                          'served' if resp is not None else 'refused'),
                     'cascade_reason': 'all_configured_accounts_spent',
                     **({'product': target.product, 'billing': target.billing} if target else {}),
                     'cascade_refusals': declined,
                     'previous_retry_after': exhausted.decision.retry_after})


def _prepare_candidate(payload, path, address):
    from lib.hal import gateway_egress as egress
    from lib.hal.gateway_openai import openai_to_anthropic_request
    from lib.hal.gateway_responses import responses_to_anthropic_request

    target = address if isinstance(address, Target) else None
    provider, model = (target.provider, target.model) if target else address.split('/', 1)
    service = target.credential if target else egress.PROVIDER_ENDPOINTS[provider][1]
    key = egress._default_key_resolver(service)
    if not key:
        return None, 'no_credential'
    account = 'w4:' + provider + ':' + spend_guard.account_key(key)
    if LedgerView().cooling(account) is not None or spend_guard.spent_until(account):
        return None, 'cooldown'
    translate = (responses_to_anthropic_request if path == '/v1/responses'
                 else openai_to_anthropic_request)
    request = translate(clean_request(payload))
    request['model'] = model
    request['max_tokens'] = max(payload.get(k, 0) for k in (
        'max_tokens', 'max_output_tokens', 'max_completion_tokens')) or 4096
    ledger = AccountLedger(resolver=SimpleNamespace(label_for=lambda _: (account, None)))
    raw = egress.build_default_opener(subscription_bound=bool(target))
    raw_opener = _quota_observer(ledger, raw) if target else ledger.observing(raw)
    window = target.window if target else min(ModelCatalog().get(model).context_window, *(
        tier.context_window for tier in build_grip_ladder().values() if address in tier.members))
    def opener(method, url, headers, body):
        final = policy.prepare(json.loads(body), window)
        return raw_opener(method, url, headers, json.dumps(final).encode())
    return (request, key, opener, target), None


def _quota_observer(ledger, raw_opener):
    """Record product-wide reset evidence from a quota body before translating it."""
    from lib.hal.account_ledger import RESET, STATUS
    from lib.hal.wire_cascade import _quota_until
    def observed(method, url, headers, body):
        with ledger.exchange(headers) as watch:
            answer = raw_opener(method, url, headers, body)
            code = getattr(answer, 'status', None)
            if code != 429:
                watch.answered(code, answer.headers)
                return answer
            content = answer.read()
            pairs = list(answer.headers.items()) if hasattr(answer.headers, 'items') else list(answer.headers)
            cooldown = _quota_until(pairs, content, time.time())
            if cooldown:
                watch.answered(429, ((STATUS, 'rejected'), (RESET, str(cooldown[0]))))
            else:
                watch.answered(None, pairs)
            replay = io.BytesIO(content)
            replay.status, replay.headers = code, answer.headers
            return replay
    return observed


def _forward_candidate(prepared, payload, provider):
    from lib.hal import gateway_egress as egress
    request, key, opener, target = prepared
    try:
        resp = egress.cross_wire_forward(request, provider=provider,
                want_stream=bool(payload.get('stream')), key_resolver=lambda _: key,
                opener=opener, binding=target)
    except policy.Refused as exc:
        return None, str(exc), False
    except Exception as exc:
        safe = _before_send(exc)
        return None, 'before_send' if safe else 'acceptance_uncertain', not safe
    return resp, None, False


def _send_candidate(prepared, payload, path, exhausted, declined, provider, model,
                    session_id):
    target = prepared[3]
    if not sign_failover_decision(provider, model, target.billing if target else DECLARED_BILLING, session_id):
        declined.append({'provider': provider, 'reason': 'unsigned'})
        return None, True
    resp, reason, stop = _forward_candidate(prepared, payload, provider)
    if reason:
        declined.append({'provider': provider, 'reason': reason})
        return None, stop
    if not 200 <= resp.status < 300:
        declined.append({'provider': provider, 'reason': 'target_refused', 'status': resp.status})
        resp.close()
        return None, False
    _record(payload, path, exhausted, declined, provider=provider, model=model,
            resp=resp, session_id=session_id, target=target)
    return (_NormalizedReply(resp), model), True


def try_cascade(payload: dict, path: str, exhausted: spend_guard.SpendRefused,
                session_id: str | None = None):
    """Try after exhaustion; uncertain acceptance ends retries before streaming."""
    if not isinstance(exhausted, AccountsSpent):
        return None
    declined = []
    for address in configured_targets():
        provider, model = ((address.provider, address.model) if isinstance(address, Target)
                           else address.split('/', 1))
        why = refusal(address, payload, path)
        if why:
            declined.append({'provider': provider, 'reason': why})
            continue
        try:
            prepared, reason = _prepare_candidate(payload, path, address)
        except Exception:
            declined.append({'provider': provider, 'reason': 'before_send'})
            continue
        if reason:
            declined.append({'provider': provider, 'reason': reason})
            continue
        selected, stop = _send_candidate(prepared, payload, path, exhausted, declined,
                                         provider, model, session_id)
        if selected is not None:
            return selected
        if stop:
            break
    _record(payload, path, exhausted, declined, session_id=session_id)
    return None


class _NormalizedReply:
    """Normalize fresh provider IDs as well as the carried request's IDs.

    Replies remain lazy. No retry or candidate selection exists in this adapter;
    the normal gateway relays still own stream failure reporting and closure.
    """
    def __init__(self, source):
        self.source = source
        self.status, self.headers = source.status, source.headers
        self.usage = getattr(source, 'usage', None)

    def read_all(self):
        return json.dumps(clean_request(json.loads(self.source.read_all()))).encode()

    def iter_chunks(self):
        from lib.hal.gateway_openai import _iter_sse_events
        for event in _iter_sse_events(self.source.iter_chunks()):
            yield ('data: ' + json.dumps(clean_request(event)) + '\n\n').encode()

    def close(self):
        self.source.close()


def serve_cascade(handler, payload: dict, path: str, exhausted) -> bool:
    """Handle a selected response before ingress commits headers; no stream retry."""
    from lib.hal import gateway_openai as chat, gateway_responses as responses
    from lib.hal.gateway_witness import session_id_from_headers
    rail = responses if path == '/v1/responses' else chat
    session_id = session_id_from_headers(handler.headers)
    try:
        selected = try_cascade(payload, path, exhausted, session_id=session_id)
    except Exception:
        # Includes an unexpected policy/translation failure. The existing final
        # notice is the only fallback, never a 5xx that elicits a client replay.
        if isinstance(exhausted, AccountsSpent):
            _record(payload, path, exhausted, [{'provider': 'none', 'reason': 'cascade_failed'}],
                    session_id=session_id)
        return False
    if selected is None:
        return False
    resp, model = selected
    args = {'model': model}
    if rail is responses:
        args['resp_id'] = responses._resp_id()
    else:
        args.update(completion_id=chat._completion_id(), created=chat._now())
    relay = rail._relay_stream if payload.get('stream') else rail._relay_nonstream
    relay(handler, resp, **args)
    return True
