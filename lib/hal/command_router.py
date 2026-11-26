"""HAL Command Router — route slash commands to handlers.

Provides a registry for slash commands (/status, /council, /help, etc.)
with argument parsing and help text generation.

Usage:
    from lib.hal.command_router import CommandRouter

    router = CommandRouter()
    router.register("status", handler_fn, "Show system status")
    result = router.execute("/status --verbose")
"""
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CommandDef:
    """Definition of a slash command."""
    name: str
    description: str
    usage: str = ""


@dataclass(frozen=True)
class CommandResult:
    """Result from executing a command."""
    command: str
    output: str
    success: bool
    error: str = ""


_COMMAND_RE = re.compile(r"^/(\w+)\s*(.*)?$")


class CommandRouter:
    """Routes slash commands to handler functions."""

    def __init__(self):
        self._commands: dict[str, tuple[callable, CommandDef]] = {}
        self._register_builtins()

    def register(self, name: str, handler: callable, description: str, usage: str = "") -> None:
        """Register a command handler."""
        self._commands[name] = (handler, CommandDef(name, description, usage))

    def execute(self, input_text: str) -> CommandResult:
        """Parse and execute a slash command."""
        match = _COMMAND_RE.match(input_text.strip())
        if not match:
            return CommandResult("", "", False, "Not a slash command")

        name = match.group(1).lower()
        args = (match.group(2) or "").strip()

        entry = self._commands.get(name)
        if not entry:
            return CommandResult(name, "", False, f"Unknown command: /{name}")

        handler, _cmd_def = entry
        try:
            output = handler(args)
            return CommandResult(name, str(output), True)
        except Exception as e:
            return CommandResult(name, "", False, str(e)[:200])

    def is_command(self, text: str) -> bool:
        """Check if text starts with a slash command."""
        return bool(_COMMAND_RE.match(text.strip()))

    def list_commands(self) -> list[CommandDef]:
        """List all registered commands."""
        return sorted([cmd_def for _, cmd_def in self._commands.values()], key=lambda c: c.name)

    def help_text(self) -> str:
        """Generate help text for all commands."""
        lines = ["Available commands:"]
        for cmd_def in self.list_commands():
            usage = f" {cmd_def.usage}" if cmd_def.usage else ""
            lines.append(f"  /{cmd_def.name}{usage} — {cmd_def.description}")
        return "\n".join(lines)

    def _register_builtins(self) -> None:
        """Register built-in commands."""
        self.register("help", lambda _: self.help_text(), "Show available commands")
