"""HAL Config Manager — unified configuration management.

Loads, merges, and validates HAL configuration from multiple sources:
1. Defaults (built-in)
2. config/hal.json (user overrides)
3. Environment variables (HAL_* prefix)

Usage:
    from lib.hal.config_manager import ConfigManager

    config = ConfigManager()
    port = config.get("server.port", 3850)
    config.set("server.port", 8080)
    config.save()
"""
import json
import os

from lib.hal.home import HAL_HOME as _HAL_HOME

_CONFIG_PATH = os.path.join(_HAL_HOME, "config", "hal.json")

# Default configuration
_DEFAULTS: dict[str, object] = {
    "server.port": 3850,
    "server.host": "127.0.0.1",
    "provider.default": "groq/llama-3.3-70b-versatile",
    "provider.timeout_s": 30,
    "council.min_models": 2,
    "council.max_models": 5,
    "budget.max_usd": 5.0,
    "cache.ttl_seconds": 300,
    "cache.max_entries": 100,
    "context.max_tokens": 200_000,
    "context.warn_80": True,
    "context.warn_85": True,
    "context.warn_95": True,
    "retry.max_attempts": 3,
    "retry.base_delay_s": 1.0,
    "telemetry.enabled": False,
    "telemetry.error_reporting": False,
    "daemon.auto_warmup": True,
    "daemon.mesh_enabled": True,
    "safety.trap_detection": True,
    "safety.verify_edits": True,
}


class ConfigManager:
    """Unified configuration with defaults, file, and env overrides."""

    def __init__(self, config_path: str = ""):
        self._path = config_path or _CONFIG_PATH
        self._config: dict[str, object] = dict(_DEFAULTS)
        self._load_file()
        self._load_env()

    def get(self, key: str, default: object = None) -> object:
        """Get a config value by dotted key."""
        return self._config.get(key, default)

    def set(self, key: str, value: object) -> None:
        """Set a config value."""
        self._config[key] = value

    def all(self) -> dict[str, object]:
        """Get all config values."""
        return dict(self._config)

    def save(self) -> None:
        """Persist non-default values to config file."""
        overrides = {
            k: v for k, v in self._config.items()
            if k not in _DEFAULTS or _DEFAULTS[k] != v
        }
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(overrides, f, indent=2)

    def reset(self, key: str) -> None:
        """Reset a key to its default value."""
        if key in _DEFAULTS:
            self._config[key] = _DEFAULTS[key]

    def _load_file(self) -> None:
        """Load overrides from config file."""
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path) as f:
                overrides = json.load(f)
            if isinstance(overrides, dict):
                self._config.update(overrides)
        except (json.JSONDecodeError, OSError):
            pass

    def _load_env(self) -> None:
        """Load overrides from HAL_* environment variables."""
        for env_key, env_val in os.environ.items():
            if not env_key.startswith("HAL_CONFIG_"):
                continue
            config_key = env_key[len("HAL_CONFIG_"):].lower().replace("__", ".")
            self._config[config_key] = _parse_env_value(env_val)


def _parse_env_value(value: str) -> object:
    """Parse env var value to appropriate Python type."""
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
