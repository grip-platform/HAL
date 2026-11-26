"""HAL Init — bootstraps the complete HAL runtime.

Single entry point that initialises all HAL subsystems in the correct
order. Used by cli.py and daemon.py.

Usage:
    from lib.hal.hal_init import init_hal
    runtime = init_hal()
"""
from dataclasses import dataclass
from lib.hal.system_info import HAL_VERSION, get_version_string


@dataclass
class HALRuntime:
    """Initialised HAL runtime with all subsystems."""
    version: str = HAL_VERSION
    registry: object = None
    conversation: object = None
    event_bus: object = None
    config: object = None

    @property
    def ready(self) -> bool:
        return self.registry is not None


def init_hal(config_path: str = "") -> HALRuntime:
    """Bootstrap HAL runtime. Initialises subsystems in dependency order."""
    from lib.hal.config_manager import ConfigManager
    from lib.hal.event_bus import EventBus
    from lib.hal.registry import get_registry
    from lib.hal.discovery import discover_local_providers
    from lib.hal.conversation import Conversation

    config = ConfigManager(config_path)
    registry = get_registry()
    discover_local_providers(registry)
    event_bus = EventBus()
    conversation = Conversation(registry=registry)

    event_bus.emit("hal_init", {"version": HAL_VERSION})

    return HALRuntime(
        version=HAL_VERSION,
        registry=registry,
        conversation=conversation,
        event_bus=event_bus,
        config=config,
    )
