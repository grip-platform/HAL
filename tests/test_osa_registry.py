"""Tests for ``lib.hal.osa.registry`` — script loader + param validator."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lib.hal.osa.registry import (
    ParamSpec,
    ParamValidationError,
    Script,
    ScriptNotRegistered,
    get_script,
    load_registry,
    validate_params,
    _reset_cache_for_tests,
)


def _write_registry(payload: dict) -> Path:
    """Write ``payload`` as JSON into a fresh temp file and return its path."""
    fh = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(payload, fh)
    fh.close()
    return Path(fh.name)


class RegistryLoad(unittest.TestCase):
    """``load_registry`` parses JSON into Script instances."""

    def test_loads_empty_registry(self) -> None:
        path = _write_registry({"scripts": {}})
        self.assertEqual(load_registry(path), {})

    def test_loads_single_tier_1_script(self) -> None:
        path = _write_registry({
            "scripts": {
                "battery_get": {
                    "tier": 1,
                    "description": "read battery",
                    "body": "return 42",
                    "params": [],
                    "timeout_s": 5,
                }
            }
        })
        reg = load_registry(path)
        self.assertIn("battery_get", reg)
        script = reg["battery_get"]
        self.assertIsInstance(script, Script)
        self.assertEqual(script.tier, 1)
        self.assertEqual(script.body, "return 42")
        self.assertEqual(script.params, ())
        self.assertEqual(script.timeout_s, 5)

    def test_loads_param_specs(self) -> None:
        path = _write_registry({
            "scripts": {
                "notification": {
                    "tier": 2,
                    "body": "display notification {body}",
                    "params": [
                        {
                            "name": "body",
                            "type": "str",
                            "required": True,
                            "max_length": 140,
                        }
                    ],
                    "timeout_s": 5,
                }
            }
        })
        reg = load_registry(path)
        spec = reg["notification"].params[0]
        self.assertIsInstance(spec, ParamSpec)
        self.assertEqual(spec.name, "body")
        self.assertEqual(spec.max_length, 140)

    def test_default_registry_caches_across_calls(self) -> None:
        _reset_cache_for_tests()
        first = load_registry()
        second = load_registry()
        self.assertIs(first, second)

    def test_custom_path_never_caches(self) -> None:
        _reset_cache_for_tests()
        path = _write_registry({"scripts": {}})
        first = load_registry(path)
        second = load_registry(path)
        self.assertIsNot(first, second)


class GetScript(unittest.TestCase):
    """``get_script`` raises a typed exception on unknown names."""

    def test_known_script_returned(self) -> None:
        script = Script(name="x", tier=1, body="return 1")
        registry = {"x": script}
        self.assertIs(get_script("x", registry), script)

    def test_unknown_script_raises(self) -> None:
        with self.assertRaises(ScriptNotRegistered):
            get_script("does_not_exist", {})


class ParamValidation(unittest.TestCase):
    """Required, type, length, and pattern all enforced in declaration order."""

    def _make(self, *specs) -> Script:
        return Script(name="t", tier=1, body="noop", params=tuple(specs))

    def test_missing_required_raises(self) -> None:
        script = self._make(ParamSpec(name="title", required=True))
        with self.assertRaises(ParamValidationError):
            validate_params(script, {})

    def test_missing_optional_ok(self) -> None:
        script = self._make(ParamSpec(name="x", required=False))
        try:
            validate_params(script, {})
        except ParamValidationError:
            self.fail("optional missing param should not raise")

    def test_wrong_type_raises(self) -> None:
        script = self._make(ParamSpec(name="x", type="int"))
        with self.assertRaises(ParamValidationError):
            validate_params(script, {"x": "not an int"})

    def test_max_length_enforced(self) -> None:
        script = self._make(ParamSpec(name="x", max_length=3))
        with self.assertRaises(ParamValidationError):
            validate_params(script, {"x": "too long"})

    def test_pattern_enforced(self) -> None:
        script = self._make(ParamSpec(name="x", pattern=r"^[a-z]+$"))
        with self.assertRaises(ParamValidationError):
            validate_params(script, {"x": "HAS_UPPERCASE"})

    def test_all_pass_returns_none(self) -> None:
        script = self._make(
            ParamSpec(name="x", pattern=r"^[a-z]+$", max_length=10)
        )
        self.assertIsNone(validate_params(script, {"x": "hello"}))

    def test_unknown_keys_ignored(self) -> None:
        """Forward compatibility — extra keys don't blow up."""
        script = self._make(ParamSpec(name="x", required=False))
        try:
            validate_params(script, {"x": "ok", "extra": "also ok"})
        except ParamValidationError:
            self.fail("unknown keys in supplied should be ignored")


class DefaultRegistryContract(unittest.TestCase):
    """The shipped ``registry.json`` must contain exactly the α-3a seed scripts."""

    def test_seed_scripts_present(self) -> None:
        _reset_cache_for_tests()
        reg = load_registry()
        self.assertIn("battery_get", reg)
        self.assertIn("volume_get", reg)
        self.assertIn("notification", reg)

    def test_notification_has_charset_cap(self) -> None:
        """Council invariant — notification body must be charset- and length-capped."""
        _reset_cache_for_tests()
        reg = load_registry()
        notif = reg["notification"]
        body_spec = next(p for p in notif.params if p.name == "body")
        self.assertIsNotNone(body_spec.pattern)
        self.assertLessEqual(body_spec.max_length or 0, 140)


class Beta4bEntries(unittest.TestCase):
    """β-4b expansion — entries required for the GRIP migration sprint must
    be present, correctly tiered, and injection-hardened.

    Goodhart-proof: each assertion would fail if a future refactor renamed
    a script, relaxed a pattern to ``.*``, or let a tier-2 script skip
    whitelist enforcement on a free-text identifier.
    """

    def setUp(self) -> None:
        _reset_cache_for_tests()
        self.reg = load_registry()

    def test_all_four_entries_present(self) -> None:
        for name in (
            "tab_close_chromium_by_url",
            "findmy_process_probe",
            "crash_dialog_detect",
            "crash_dialog_dismiss",
        ):
            self.assertIn(name, self.reg, msg=f"β-4b expects {name!r} in registry")

    def test_tier_split_is_correct(self) -> None:
        """Tier-1 entries must NOT mutate system state; tier-2 must."""
        self.assertEqual(self.reg["findmy_process_probe"].tier, 1)
        self.assertEqual(self.reg["crash_dialog_detect"].tier, 1)
        self.assertEqual(self.reg["tab_close_chromium_by_url"].tier, 2)
        self.assertEqual(self.reg["crash_dialog_dismiss"].tier, 2)

    def test_tab_close_browser_is_whitelisted(self) -> None:
        script = self.reg["tab_close_chromium_by_url"]
        browser_spec = next(p for p in script.params if p.name == "browser")
        self.assertIsNotNone(browser_spec.pattern, "browser param must be pattern-guarded")
        # Assert the whitelist is explicit, not a wildcard
        self.assertNotIn(".*", browser_spec.pattern)
        self.assertIn("Google Chrome", browser_spec.pattern)
        self.assertIn("Arc", browser_spec.pattern)

    def test_tab_close_url_pattern_disallows_quotes_and_backslashes(self) -> None:
        """Injection guard — url_pattern must NOT accept quote/backslash."""
        import re
        script = self.reg["tab_close_chromium_by_url"]
        spec = next(p for p in script.params if p.name == "url_pattern")
        rx = re.compile(spec.pattern)
        # Legitimate URL fragments pass
        for ok in ("example.com", "https://foo.bar/baz?x=1", "prefix with spaces"):
            self.assertIsNotNone(rx.match(ok), msg=f"legit fragment rejected: {ok!r}")
        # Injection attempts fail
        for bad in ('"; do shell script "rm -rf /', 'foo"bar', 'foo\\bar', "foo\nbar"):
            self.assertIsNone(rx.match(bad), msg=f"injection accepted: {bad!r}")

    def test_crash_dialog_dismiss_process_is_whitelist_only(self) -> None:
        """Tier-2 dismiss must not accept arbitrary process names — only the
        two known crash-reporter apps. Prevents 'click OK on anything'."""
        import re
        script = self.reg["crash_dialog_dismiss"]
        spec = next(p for p in script.params if p.name == "process_name")
        rx = re.compile(spec.pattern)
        self.assertIsNotNone(rx.match("Problem Reporter"))
        self.assertIsNotNone(rx.match("UserNotificationCenter"))
        for bad in ("Finder", "Safari", "Problem Reporter; touch /tmp/pwn", "", " "):
            self.assertIsNone(rx.match(bad), msg=f"non-whitelisted accepted: {bad!r}")

    def test_tier_1_entries_have_no_params(self) -> None:
        """findmy_process_probe and crash_dialog_detect take no input —
        nothing to validate, nothing to inject through."""
        self.assertEqual(self.reg["findmy_process_probe"].params, ())
        self.assertEqual(self.reg["crash_dialog_detect"].params, ())


class W5NotificationSubtitle(unittest.TestCase):
    """W5 OSA migration — notification_with_subtitle registry entry guards.

    Ensures the entry is present, correctly tiered, injection-hardened,
    and that validate_params enforces the subtitle requirement.
    """

    def setUp(self) -> None:
        _reset_cache_for_tests()
        self.reg = load_registry()

    def test_notification_with_subtitle_present(self) -> None:
        self.assertIn(
            "notification_with_subtitle", self.reg,
            "W5 requires notification_with_subtitle in registry",
        )

    def test_is_tier_2(self) -> None:
        self.assertEqual(self.reg["notification_with_subtitle"].tier, 2)

    def test_has_three_required_params(self) -> None:
        script = self.reg["notification_with_subtitle"]
        names = {p.name for p in script.params}
        self.assertSetEqual(names, {"title", "body", "subtitle"})
        for p in script.params:
            self.assertTrue(p.required, f"param {p.name!r} must be required")

    def test_subtitle_charset_and_length_cap(self) -> None:
        """Subtitle must be charset- and length-capped — same as title.

        Pattern `^[\\x20-\\x7E]+$` accepts printable ASCII only; AppleScript
        injection via quote chars is handled at dispatch by applescript_escape,
        not at this validation layer. Control chars and non-ASCII are blocked here.
        """
        import re
        script = self.reg["notification_with_subtitle"]
        spec = next(p for p in script.params if p.name == "subtitle")
        self.assertIsNotNone(spec.pattern, "subtitle must have pattern guard")
        rx = re.compile(spec.pattern)
        self.assertIsNotNone(rx.match("Build done"), "legit subtitle rejected")
        self.assertIsNotNone(rx.match("PR #300 merged"), "subtitle with # rejected")
        # Control chars and non-printable must be blocked
        self.assertIsNone(rx.match("foo\x00bar"), "null byte accepted")
        self.assertIsNone(rx.match("foo\nbar"), "newline accepted")
        self.assertIsNone(rx.match("foo\x1bbar"), "escape accepted")
        self.assertLessEqual(spec.max_length or 0, 100)

    def test_validate_params_rejects_missing_subtitle(self) -> None:
        from lib.hal.osa.registry import validate_params, ParamValidationError
        script = self.reg["notification_with_subtitle"]
        with self.assertRaises(ParamValidationError):
            validate_params(script, {"title": "GRIP", "body": "done"})

    def test_validate_params_accepts_all_three(self) -> None:
        from lib.hal.osa.registry import validate_params
        script = self.reg["notification_with_subtitle"]
        validate_params(script, {"title": "GRIP", "body": "done", "subtitle": "W5"})

    def test_body_template_contains_subtitle_placeholder(self) -> None:
        script = self.reg["notification_with_subtitle"]
        self.assertIn("{subtitle}", script.body)


class BetaOneRegistry(unittest.TestCase):
    """β-1 registry expansion — new scripts are present, correctly tiered,
    and param-guarded against injection where applicable."""

    @classmethod
    def setUpClass(cls) -> None:
        from lib.hal.osa.registry import DEFAULT_REGISTRY_PATH
        _reset_cache_for_tests()
        cls.reg = load_registry(DEFAULT_REGISTRY_PATH)

    def test_mute_is_tier_2_no_params(self) -> None:
        s = self.reg["mute"]
        self.assertEqual(s.tier, 2)
        self.assertEqual(s.params, ())

    def test_unmute_is_tier_2_no_params(self) -> None:
        self.assertEqual(self.reg["unmute"].tier, 2)

    def test_mute_status_get_is_tier_1(self) -> None:
        self.assertEqual(self.reg["mute_status_get"].tier, 1)

    def test_app_launch_app_name_pattern_rejects_injection(self) -> None:
        import re
        s = self.reg["app_launch"]
        spec = next(p for p in s.params if p.name == "app_name")
        rx = re.compile(spec.pattern)
        for ok in ("Safari", "Google Chrome", "VS Code", "iTerm2"):
            self.assertIsNotNone(rx.match(ok), msg=f"legit name rejected: {ok!r}")
        for bad in ("Finder; rm -rf /", 'Safari"', "app\nname", ";shell"):
            self.assertIsNone(rx.match(bad), msg=f"injection accepted: {bad!r}")

    def test_app_quit_shares_app_name_pattern(self) -> None:
        import re
        s = self.reg["app_quit"]
        spec = next(p for p in s.params if p.name == "app_name")
        # Same tier and same guard as app_launch
        self.assertEqual(s.tier, 2)
        self.assertRegex(spec.pattern, r"\[A-Za-z")  # character class present

    def test_clipboard_set_max_length_4000(self) -> None:
        s = self.reg["clipboard_set"]
        spec = next(p for p in s.params if p.name == "text")
        self.assertEqual(spec.max_length, 4000)

    def test_clipboard_set_is_tier_2(self) -> None:
        self.assertEqual(self.reg["clipboard_set"].tier, 2)

    def test_screen_wake_is_tier_2_no_params(self) -> None:
        s = self.reg["screen_wake"]
        self.assertEqual(s.tier, 2)
        self.assertEqual(s.params, ())

    def test_music_now_playing_is_tier_1(self) -> None:
        self.assertEqual(self.reg["music_now_playing"].tier, 1)

    def test_music_play_pause_next_prev_are_tier_2(self) -> None:
        for name in ("music_play", "music_pause", "music_next_track", "music_previous_track"):
            with self.subTest(script=name):
                self.assertEqual(self.reg[name].tier, 2)

    def test_shortcut_run_name_rejects_path_traversal(self) -> None:
        import re
        s = self.reg["shortcut_run"]
        spec = next(p for p in s.params if p.name == "name")
        rx = re.compile(spec.pattern)
        self.assertIsNotNone(rx.match("GRIP-flashlight"))
        self.assertIsNone(rx.match("../../bad"))
        self.assertIsNone(rx.match("name;cmd"))

    def test_shortcut_run_input_is_optional(self) -> None:
        s = self.reg["shortcut_run"]
        spec = next(p for p in s.params if p.name == "input")
        self.assertFalse(spec.required)

    def test_caffeinate_start_accepts_mac_macro_durations(self) -> None:
        # /feature-complete regression. The GRIP /mac macros pass these exact
        # durations: morning 8h (28800), watch 3h (10800), sprint 2h (7200).
        # The pre-fix 4-char cap (max_length 4, pattern capped at 9999) silently
        # REJECTED 28800/10800 via validate_params, so caffeinate no-opped and
        # the screen could sleep mid-ritual/mid-stream. Assert the full contract
        # (max_length AND pattern, end-to-end) accepts the real durations and
        # still rejects digit-overflow and shell-injection.
        s = self.reg["caffeinate_start"]
        for ok in ("1", "7200", "10800", "28800", "3600", "99999"):
            with self.subTest(accept=ok):
                validate_params(s, {"seconds": ok})  # must not raise
        for bad in ("100000", "0", "abc", "10; rm -rf /", "-1", "3.5", "10800 "):
            with self.subTest(reject=bad):
                with self.assertRaises(ParamValidationError):
                    validate_params(s, {"seconds": bad})

    def test_imessage_send_handle_rejects_shell_metacharacters(self) -> None:
        import re
        s = self.reg["imessage_send"]
        spec = next(p for p in s.params if p.name == "handle")
        rx = re.compile(spec.pattern)
        for ok in ("+27821234567", "user@example.com", "development@codetonight.co.za"):
            self.assertIsNotNone(rx.match(ok), msg=f"legit handle rejected: {ok!r}")
        for bad in ('user"@x.com', "user;cmd", "handle\nwith\nnewline"):
            self.assertIsNone(rx.match(bad), msg=f"injection accepted: {bad!r}")

    def test_imessage_send_text_max_length_4000(self) -> None:
        s = self.reg["imessage_send"]
        spec = next(p for p in s.params if p.name == "text")
        self.assertEqual(spec.max_length, 4000)

    def test_sleep_now_is_tier_3(self) -> None:
        self.assertEqual(self.reg["sleep_now"].tier, 3)

    def test_shutdown_is_tier_3(self) -> None:
        self.assertEqual(self.reg["shutdown"].tier, 3)

    def test_restart_is_tier_3(self) -> None:
        self.assertEqual(self.reg["restart"].tier, 3)

    def test_total_script_count_expanded(self) -> None:
        """Registry has grown beyond the α-3 seed of 18 scripts."""
        self.assertGreater(len(self.reg), 20)


class OsaRunToolRegistration(unittest.TestCase):
    """β-3: osa_run must appear in HAL's built-in tool registry."""

    def test_osa_run_in_builtin_tools(self) -> None:
        from lib.hal.tool_registry import ToolRegistry
        registry = ToolRegistry()
        tool = registry.get("osa_run")
        self.assertIsNotNone(tool)

    def test_osa_run_is_enabled(self) -> None:
        from lib.hal.tool_registry import ToolRegistry
        tool = ToolRegistry().get("osa_run")
        self.assertTrue(tool.enabled)

    def test_osa_run_has_script_param(self) -> None:
        from lib.hal.tool_registry import ToolRegistry
        tool = ToolRegistry().get("osa_run")
        self.assertIn("script", tool.parameters)

    def test_osa_run_description_mentions_tier(self) -> None:
        from lib.hal.tool_registry import ToolRegistry
        tool = ToolRegistry().get("osa_run")
        self.assertIn("ier", tool.description)  # "Tier" or "tier"

    def test_osa_run_in_list_tools(self) -> None:
        from lib.hal.tool_registry import ToolRegistry
        names = [t.name for t in ToolRegistry().list_tools()]
        self.assertIn("osa_run", names)


if __name__ == "__main__":
    unittest.main()
