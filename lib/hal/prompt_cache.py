"""HAL Prompt Cache — ephemeral cache_control shim for CCH (Phase γ W3, H-PCACHE).

Anthropic's prompt caching drops cached-input cost to ~10% of normal
rate. This module augments HAL's CCH body with the ``cache_control:
{type: "ephemeral"}`` markers on the two blocks that actually benefit
(long-lived system prompt + tool definitions) and leaves the volatile
user-message tail uncached. It's a pure builder — the transport layer
calls ``apply_cache_control(body)`` right before the HTTP POST.

Hypothesis H-PCACHE: on a 4K-token system + 2K-token tool definitions,
turn 2+ of a multi-turn session saves ~$0.011 per call at Opus rates.

Public API:
    CacheControlConfig — which blocks to mark, env override hooks
    apply_cache_control(body, *, config=None) -> body
    has_cache_markers(body) -> bool (test-friendly introspection)
"""
from __future__ import annotations

import os
from dataclasses import dataclass


_EPHEMERAL = {"type": "ephemeral"}


@dataclass(frozen=True)
class CacheControlConfig:
    mark_system: bool = True
    mark_tools: bool = True
    disabled: bool = False


def _env_disabled() -> bool:
    return os.environ.get("HAL_DISABLE_PROMPT_CACHE", "").lower() in ("1", "true", "yes")


def _from_env(base: CacheControlConfig | None) -> CacheControlConfig:
    cfg = base or CacheControlConfig()
    if _env_disabled():
        return CacheControlConfig(mark_system=False, mark_tools=False, disabled=True)
    return cfg


def _mark_system_block(body: dict) -> None:
    """Add cache_control to the system prompt, supporting the 3 canonical shapes.

    Anthropic accepts system as either:
        1. top-level "system": "string"           -> upgrade to list-of-blocks
        2. top-level "system": [ {type:text,...} ] -> mark the LAST block
        3. absent                                   -> no-op
    """
    system = body.get("system")
    if system is None:
        return
    if isinstance(system, str):
        body["system"] = [{"type": "text", "text": system, "cache_control": _EPHEMERAL}]
        return
    if isinstance(system, list) and system:
        last = system[-1]
        if isinstance(last, dict):
            last["cache_control"] = _EPHEMERAL


def _mark_tools_block(body: dict) -> None:
    """Add cache_control to the LAST tool entry (the whole tools block caches as a unit)."""
    tools = body.get("tools")
    if not isinstance(tools, list) or not tools:
        return
    last = tools[-1]
    if isinstance(last, dict):
        last["cache_control"] = _EPHEMERAL


def apply_cache_control(body: dict, *, config: CacheControlConfig | None = None) -> dict:
    """Return ``body`` with cache_control markers applied in place and returned.

    Fail-closed on ``HAL_DISABLE_PROMPT_CACHE=1``: no markers applied.
    Mutates the dict and returns it for chaining — callers don't need
    to worry about aliasing.
    """
    cfg = _from_env(config)
    if cfg.disabled:
        return body
    if cfg.mark_system:
        _mark_system_block(body)
    if cfg.mark_tools:
        _mark_tools_block(body)
    return body


def has_cache_markers(body: dict) -> bool:
    """True iff any cache_control marker is present in system/tools blocks."""
    system = body.get("system")
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get("cache_control") == _EPHEMERAL:
                return True
    tools = body.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict) and tool.get("cache_control") == _EPHEMERAL:
                return True
    return False


__all__ = [
    "CacheControlConfig",
    "apply_cache_control",
    "has_cache_markers",
]
