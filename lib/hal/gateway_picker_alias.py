"""Discovery-visible aliases so HAL's models reach Claude Code's /model picker.

THE MEASUREMENT (2026-09-10, private gateway on 127.0.0.1:49841, Claude Code
2.1.267). With ``CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`` and
``ANTHROPIC_BASE_URL`` pointed at the gateway, Claude Code fetched ``GET
/v1/models?limit=1000``, received 68 ids, and wrote ONE of them to its own
picker cache (``$CLAUDE_CONFIG_DIR/cache/gateway-models.json``)::

    {"baseUrl":"http://127.0.0.1:49841","fetchedAt":1789009134799,
     "models":[{"id":"soruxgpt/claude-3-5-sonnet-20241022"}]}

67 were dropped, including both models this gateway is most used for --
``openai/gpt-6-astra`` and ``moonshot/kimi-k3``.

THE RULE, quoted from Anthropic's gateway-protocol reference
(https://code.claude.com/docs/en/llm-gateway-protocol): "Claude Code keeps an
entry when its ``id`` contains ``claude`` or ``anthropic`` anywhere in the
string, matched case-insensitively, and ignores the rest. Provider-prefixed IDs
such as ``vertex_ai/claude-sonnet-4-6`` or ``bedrock/anthropic.claude-sonnet-4-5``
pass the filter; an ID that contains neither substring doesn't." HAL addresses
its models ``<provider>/<model>``, so almost none of them contain either
substring and almost none can be picked -- although every one of them ROUTES
when its id is typed or pinned with ``--model`` (measured the same night:
astra answered PONG, witness provider openai status 200; kimi answered PONG,
provider moonshot status 200). Selecting and routing are separate mechanisms
and only the first was broken.

WHY AN ALIAS IS THE HONEST FIX, not a way round the filter. The substring is a
proxy for "this model speaks the Anthropic Messages wire". For this gateway the
proxy is wrong rather than protective: the cross-wire translates every one of
these models onto the Anthropic Messages format, which is the only reason
Claude Code can talk to them at all. The alias supplies the fact the substring
stands in for and says so plainly in the id -- ``anthropic-wire/<canonical>``
-- while ``display_name`` keeps the row the human reads honest about who
actually serves the model. Anthropic's own doc points the same way for ids that
miss the filter: "developers can add those aliases manually with the model
configuration variables"; this just moves that work to the gateway that already
knows the answer.

OFF BY DEFAULT. Turning it on roughly doubles a list other harnesses consume
(Codex reads the injected ``models`` array off the same route), so the operator
opts in with ``HAL_GATEWAY_PICKER_ALIASES=1``. Note this is the inverse default
of ``HAL_GATEWAY_MULTIPROVIDER``, which is on-by-default because it only ever
changes where an already-named model goes; this one changes what a shared list
CONTAINS, which is a wider blast radius.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Mapping, Optional

# The prefix an alias carries. Contains "anthropic" (so Claude Code keeps the
# entry) and reads as what it is: the Anthropic-wire address of a model HAL
# serves. Trailing slash makes the canonical id a clean suffix.
ALIAS_PREFIX = "anthropic-wire/"

_FLAG = "HAL_GATEWAY_PICKER_ALIASES"
_TRUTHY = ("1", "true", "yes", "on")

# Claude Code's documented discovery filter, as substrings.
_DISCOVERY_SUBSTRINGS = ("claude", "anthropic")


def picker_aliases_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """Whether to emit discovery-visible aliases. DEFAULT OFF.

    ``env`` is injectable so the predicate is testable without mutating the
    process environment.
    """
    source = os.environ if env is None else env
    return str(source.get(_FLAG, "")).strip().lower() in _TRUTHY


def is_discoverable(model_id: str) -> bool:
    """Whether Claude Code's ``/v1/models`` filter would KEEP this id.

    Reimplements the documented rule: the id contains ``claude`` or
    ``anthropic`` anywhere, case-insensitively. Kept as its own named predicate
    because it is the one fact the whole feature turns on -- if Anthropic
    changes the rule, this is the single line that has to move.
    """
    lowered = str(model_id).lower()
    return any(s in lowered for s in _DISCOVERY_SUBSTRINGS)


def alias_for(model_id: str) -> str:
    """The discovery-visible address of ``model_id``."""
    return ALIAS_PREFIX + str(model_id)


def is_alias(model_id: str) -> bool:
    """Whether this id is one of ours to rewrite.

    Exists so a caller can ASK before it mutates, instead of inferring it from
    ``canonical_for(x) != x``. That inference is subtly wrong at a callsite
    that defaults a missing id: ``""`` compares unequal to ``None`` and the
    caller rewrites a request that named no model at all. Raised by the AGORA
    judge on PR #575 against exactly that shape on the Responses rail.
    """
    return str(model_id).startswith(ALIAS_PREFIX)


def canonical_for(model_id: str) -> str:
    """The model that actually routes, given an id that may be an alias.

    Identity for anything that is not an alias, because this runs on the
    routing hot path for EVERY request. Strips exactly one prefix so a model
    whose own name contains the prefix is not over-stripped.
    """
    text = str(model_id)
    if text.startswith(ALIAS_PREFIX):
        return text[len(ALIAS_PREFIX):]
    return text


def _alias_entry(declared: Dict) -> Optional[Dict]:
    """One picker row for one declared model, or ``None`` to skip it.

    Skips a row with no usable id and a row that already survives the filter
    (aliasing it would put the same model in the picker twice).
    """
    if not isinstance(declared, dict):
        return None
    model_id = declared.get("id")
    if not isinstance(model_id, str) or not model_id:
        return None
    if is_discoverable(model_id):
        return None
    owner = declared.get("owned_by")
    if not isinstance(owner, str) or not owner:
        owner = model_id.split("/", 1)[0]
    bare = model_id.split("/", 1)[1] if "/" in model_id else model_id
    return {
        "id": alias_for(model_id),
        "object": "model",
        "owned_by": owner,
        # The picker shows display_name when it differs from the id. Name the
        # real provider and the real model; never imply Claude or Anthropic.
        "display_name": f"{bare} ({owner})",
        "description": f"Served by {owner} through the HAL gateway, translated onto the Anthropic wire.",
    }


def alias_entries(declared: Iterable) -> List[Dict]:
    """Discovery-visible rows for every declared model the picker would drop.

    Fail-open by construction: a malformed row is skipped rather than raised,
    because this feeds a live ``GET /v1/models`` that must never 500.
    """
    out: List[Dict] = []
    for row in declared or []:
        entry = _alias_entry(row)
        if entry is not None:
            out.append(entry)
    return out
