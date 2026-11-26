"""Tests for HAL command router."""
import os, sys, unittest
from lib.hal.command_router import CommandRouter, CommandResult

class TestCommandRouter(unittest.TestCase):
    def test_register_and_execute(self):
        router = CommandRouter()
        router.register("greet", lambda args: f"Hello {args}", "Greet someone")
        result = router.execute("/greet World")
        self.assertTrue(result.success)
        self.assertIn("World", result.output)

    def test_unknown_command(self):
        result = CommandRouter().execute("/nonexistent")
        self.assertFalse(result.success)

    def test_not_a_command(self):
        result = CommandRouter().execute("plain text")
        self.assertFalse(result.success)

    def test_is_command(self):
        router = CommandRouter()
        self.assertTrue(router.is_command("/help"))
        self.assertFalse(router.is_command("hello"))

    def test_list_commands(self):
        router = CommandRouter()
        cmds = router.list_commands()
        self.assertGreater(len(cmds), 0)

    def test_help_builtin(self):
        result = CommandRouter().execute("/help")
        self.assertTrue(result.success)
        self.assertIn("help", result.output)

    def test_help_text(self):
        router = CommandRouter()
        router.register("test", lambda a: "ok", "A test command", "<arg>")
        text = router.help_text()
        self.assertIn("/test", text)
        self.assertIn("<arg>", text)

    def test_handler_error(self):
        router = CommandRouter()
        router.register("fail", lambda a: 1/0, "Always fails")
        result = router.execute("/fail")
        self.assertFalse(result.success)

if __name__ == "__main__":
    unittest.main()
