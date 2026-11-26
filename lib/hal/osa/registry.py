"""osa script registry — name → Script resolution + param validation.

Phase α-3a of the osascript control layer plan. Loads ``registry.json`` at
the package root into a cached mapping of script name → :class:`Script`.
Each script declares its tier (1/2/3), its body (or path to a script file),
its parameter schema, and its timeout.

Public API::

    from lib.hal.osa.registry import (
        Script, ParamSpec, load_registry, get_script, validate_params,
        ScriptNotRegistered, ParamValidationError,
    )

Kept to stdlib only (json, re) — no PyYAML dependency. Intentional
deviation from the parent plan's ``registry.yaml`` for α-3a; a YAML loader
can be added later without breaking the public API.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from lib.hal.osa.allowlist import (
    AllowlistNotFound,
    AllowlistRejection,
    check_allowed,
)
from lib.hal.osa.rate_limit import RateLimit

__all__ = [
    "Script",
    "ParamSpec",
    "ScriptNotRegistered",
    "ParamValidationError",
    "load_registry",
    "get_script",
    "validate_params",
    "DEFAULT_REGISTRY_PATH",
]


_OSA_DIR = Path(__file__).parent
DEFAULT_REGISTRY_PATH = _OSA_DIR / "registry.json"


class ScriptNotRegistered(KeyError):
    """Script name is not present in the active registry."""


class ParamValidationError(ValueError):
    """A supplied parameter fails its declared schema."""


@dataclass(frozen=True)
class ParamSpec:
    """Declarative schema for one script parameter."""
    name: str
    type: str = "str"
    required: bool = True
    pattern: Optional[str] = None
    max_length: Optional[int] = None
    allowlist: Optional[str] = None
    default: Optional[str] = None


@dataclass(frozen=True)
class Script:
    """Registry entry for one named osa script."""
    name: str
    tier: int
    description: str = ""
    body: Optional[str] = None
    path: Optional[str] = None
    params: Tuple[ParamSpec, ...] = ()
    timeout_s: int = 10
    language: str = "AppleScript"
    rate_limit: Optional[RateLimit] = None


_REGISTRY_CACHE: Optional[Dict[str, Script]] = None


def load_registry(path: Optional[Path] = None) -> Dict[str, Script]:
    """Load the registry at ``path`` into a cached dict of name → Script.

    The default-path case is cached across calls within one process so
    the ~5 ms JSON parse doesn't run on every ``osa()`` dispatch. Non-
    default paths (tests, custom registries) are never cached.
    """
    global _REGISTRY_CACHE
    effective = path or DEFAULT_REGISTRY_PATH
    if path is None and _REGISTRY_CACHE is not None:
        return _REGISTRY_CACHE
    data = json.loads(effective.read_text(encoding="utf-8"))
    scripts: Dict[str, Script] = {}
    for name, entry in data.get("scripts", {}).items():
        param_specs = tuple(ParamSpec(**p) for p in entry.get("params", []))
        rate_cfg = entry.get("rate_limit")
        rate = RateLimit(
            window_s=int(rate_cfg["window_s"]),
            max_calls=int(rate_cfg["max_calls"]),
        ) if rate_cfg else None
        scripts[name] = Script(
            name=name,
            tier=int(entry["tier"]),
            description=entry.get("description", ""),
            body=entry.get("body"),
            path=entry.get("path"),
            params=param_specs,
            timeout_s=int(entry.get("timeout_s", 10)),
            language=entry.get("language", "AppleScript"),
            rate_limit=rate,
        )
    if path is None:
        _REGISTRY_CACHE = scripts
    return scripts


def get_script(
    name: str, registry: Optional[Dict[str, Script]] = None
) -> Script:
    """Return the :class:`Script` registered under ``name`` or raise."""
    reg = registry if registry is not None else load_registry()
    if name not in reg:
        raise ScriptNotRegistered(f"osa script {name!r} not in registry")
    return reg[name]


def validate_params(script: Script, supplied: Dict[str, object]) -> None:
    """Validate ``supplied`` against ``script.params`` — raise on first violation.

    Enforces (in declaration order): required presence, type, max_length,
    regex pattern. Intentionally conservative — unknown keys in ``supplied``
    are ignored (forward compatibility with registry additions). The caller
    supplies exactly what the schema asks for, no more.
    """
    for spec in script.params:
        if spec.name not in supplied:
            if spec.required:
                raise ParamValidationError(
                    f"missing required param {spec.name!r}"
                )
            continue
        val = supplied[spec.name]
        _check_param(spec, val)


def _check_param(spec: ParamSpec, val: object) -> None:
    if spec.type == "str" and not isinstance(val, str):
        raise ParamValidationError(
            f"param {spec.name!r} must be str, got {type(val).__name__}"
        )
    if spec.type == "int" and not isinstance(val, int):
        raise ParamValidationError(
            f"param {spec.name!r} must be int, got {type(val).__name__}"
        )
    if spec.max_length is not None and len(str(val)) > spec.max_length:
        raise ParamValidationError(
            f"param {spec.name!r} exceeds max_length {spec.max_length}"
        )
    if spec.pattern is not None and not re.match(spec.pattern, str(val)):
        raise ParamValidationError(
            f"param {spec.name!r} fails pattern {spec.pattern!r}"
        )
    if spec.allowlist is not None:
        try:
            check_allowed(spec.allowlist, str(val))
        except AllowlistNotFound as exc:
            raise ParamValidationError(
                f"param {spec.name!r} declares allowlist {spec.allowlist!r} "
                f"but no backing file exists: {exc}"
            ) from exc
        except AllowlistRejection as exc:
            raise ParamValidationError(
                f"param {spec.name!r} rejected by allowlist "
                f"{spec.allowlist!r}: {exc}"
            ) from exc


def _reset_cache_for_tests() -> None:
    """Test-only hook to clear the registry cache between loads."""
    global _REGISTRY_CACHE
    _REGISTRY_CACHE = None
