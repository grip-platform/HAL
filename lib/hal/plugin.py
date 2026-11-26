"""HAL Provider Plugin System — extend HAL with custom providers.

Enables users to add new LLM providers by placing plugin files in
~/.hal/plugins/providers/. Each plugin exports a ProviderPlugin class.

Usage:
    from lib.hal.plugin import PluginLoader
    loader = PluginLoader()
    plugins = loader.discover()
    loader.register_all(registry)
"""
import importlib.util
import os
from dataclasses import dataclass, field

from lib.hal.home import HAL_HOME as _HAL_HOME

_PLUGINS_DIR = os.path.join(_HAL_HOME, "plugins", "providers")


@dataclass
class PluginInfo:
    """Information about a discovered plugin."""
    name: str
    path: str
    loaded: bool = False
    error: str = ""
    version: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name, "path": self.path,
            "loaded": self.loaded, "error": self.error,
            "version": self.version,
        }


@dataclass
class PluginReport:
    """Report from plugin discovery."""
    plugins: list[PluginInfo] = field(default_factory=list)

    @property
    def loaded_count(self) -> int:
        return sum(1 for p in self.plugins if p.loaded)

    @property
    def summary(self) -> str:
        return f"Plugins: {self.loaded_count}/{len(self.plugins)} loaded"


class PluginLoader:
    """Discovers and loads provider plugins."""

    def __init__(self, plugins_dir: str = ""):
        self._dir = plugins_dir or _PLUGINS_DIR
        self._plugins: dict[str, object] = {}

    def discover(self) -> PluginReport:
        """Scan plugins directory and attempt to load each plugin."""
        report = PluginReport()
        if not os.path.isdir(self._dir):
            return report

        for filename in sorted(os.listdir(self._dir)):
            if not filename.endswith(".py") or filename.startswith("_"):
                continue
            info = self._load_plugin(filename, os.path.join(self._dir, filename))
            report.plugins.append(info)

        return report

    def get_plugin(self, name: str) -> object | None:
        """Get a loaded plugin by name."""
        return self._plugins.get(name)

    def list_plugins(self) -> list[str]:
        """List names of loaded plugins."""
        return sorted(self._plugins.keys())

    def register_all(self, registry) -> int:
        """Register all loaded plugins with the provider registry."""
        registered = 0
        for plugin in self._plugins.values():
            if hasattr(plugin, "connect"):
                try:
                    plugin.connect(registry)
                    registered += 1
                except Exception:
                    pass
        return registered

    def _load_plugin(self, filename: str, path: str) -> PluginInfo:
        """Load a single plugin module."""
        module_name = filename[:-3]
        info = PluginInfo(name=module_name, path=path)

        try:
            spec = importlib.util.spec_from_file_location(f"hal_plugin_{module_name}", path)
            if spec is None or spec.loader is None:
                info.error = "Invalid module spec"
                return info

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            plugin_class = getattr(module, "ProviderPlugin", None)
            if plugin_class is None:
                info.error = "No ProviderPlugin class found"
                return info

            instance = plugin_class()
            self._plugins[module_name] = instance
            info.loaded = True
            info.version = getattr(instance, "version", "")
            info.name = getattr(instance, "name", module_name)

        except Exception as e:
            info.error = str(e)[:200]

        return info
