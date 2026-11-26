"""HAL Feature Flags — simple JSON-based feature management.

Toggle features without code changes. Persists to config/features.json.
Runtime overrides: HAL_FEATURE_OVERRIDES='{"flag": true}' hal ...

Usage:
    from lib.hal.features import is_enabled, enable, disable

    if is_enabled("sse_streaming"):
        start_sse_endpoint()
"""
import json
import logging
import os

from lib.hal.home import HAL_HOME as _HAL_HOME

_FEATURES_PATH = os.path.join(_HAL_HOME, "config", "features.json")
_cache: dict[str, bool] = {}
_overrides: dict[str, bool] = {}
_loaded = False
_log = logging.getLogger(__name__)


def _load_overrides() -> dict[str, bool]:
    raw = os.environ.get("HAL_FEATURE_OVERRIDES", "")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("expected JSON object")
        return {k: bool(v) for k, v in parsed.items()}
    except (json.JSONDecodeError, ValueError) as exc:
        _log.warning("HAL_FEATURE_OVERRIDES ignored — invalid JSON: %s", exc)
        return {}


def is_enabled(feature: str, default: bool = False) -> bool:
    _ensure_loaded()
    if feature in _overrides:
        return _overrides[feature]
    return _cache.get(feature, default)


def enable(feature: str) -> None:
    _ensure_loaded()
    _cache[feature] = True
    _save()


def disable(feature: str) -> None:
    _ensure_loaded()
    _cache[feature] = False
    _save()


def list_features() -> dict[str, bool]:
    _ensure_loaded()
    return {**_cache, **_overrides}


def _ensure_loaded() -> None:
    global _loaded, _overrides
    if _loaded:
        return
    if os.path.exists(_FEATURES_PATH):
        try:
            with open(_FEATURES_PATH, "r", encoding="utf-8") as f:
                _cache.update(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    _overrides = _load_overrides()
    if _overrides:
        _log.debug("HAL_FEATURE_OVERRIDES active: %s", list(_overrides.keys()))
    _loaded = True


def _save() -> None:
    os.makedirs(os.path.dirname(_FEATURES_PATH), exist_ok=True)
    with open(_FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(_cache, f, indent=2)
