"""Tests for ``lib.hal.osa.allowlist`` — γ-2c file-backed identity gate.

Covers:

* Parsing contract — one entry per line, comments + blank lines ignored,
  inline annotations (text after first whitespace) discarded
* Missing-file raises :class:`AllowlistNotFound`
* Unknown value raises :class:`AllowlistRejection`
* Cache invalidation on mtime change
* ``ParamSpec.allowlist`` wiring through :func:`validate_params`
* Path validation rejects traversal attempts
"""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from lib.hal.osa.allowlist import (
    AllowlistNotFound,
    AllowlistRejection,
    _reset_cache_for_tests,
    allowlist_path,
    check_allowed,
    load_allowlist,
)
from lib.hal.osa.registry import (
    ParamSpec,
    ParamValidationError,
    Script,
    validate_params,
)


class AllowlistPath(unittest.TestCase):
    """Path resolves under ``$HAL_HOME/config/osa-allowlists``."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="hal-al-p-")
        os.environ["HAL_HOME"] = self._tmp
        _reset_cache_for_tests()

    def tearDown(self) -> None:
        os.environ.pop("HAL_HOME", None)
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_path_under_hal_home_config(self) -> None:
        self.assertEqual(
            allowlist_path("imessage"),
            Path(self._tmp) / "config" / "osa-allowlists" / "imessage.txt",
        )

    def test_slash_in_kind_rejected(self) -> None:
        with self.assertRaises(ValueError):
            allowlist_path("../etc/passwd")

    def test_empty_kind_rejected(self) -> None:
        with self.assertRaises(ValueError):
            allowlist_path("")

    def test_dot_dot_in_kind_rejected(self) -> None:
        with self.assertRaises(ValueError):
            allowlist_path("foo..bar")


class LoadAllowlistParsing(unittest.TestCase):
    """Line parsing: comments, blanks, annotations."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="hal-al-l-")
        os.environ["HAL_HOME"] = self._tmp
        self.file = allowlist_path("imessage")
        self.file.parent.mkdir(parents=True, exist_ok=True)
        _reset_cache_for_tests()

    def tearDown(self) -> None:
        os.environ.pop("HAL_HOME", None)
        _reset_cache_for_tests()
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(AllowlistNotFound):
            load_allowlist("imessage")

    def test_single_entry_parsed(self) -> None:
        self.file.write_text("+27123456789\n")
        self.assertEqual(load_allowlist("imessage"), frozenset({"+27123456789"}))

    def test_annotation_after_whitespace_dropped(self) -> None:
        self.file.write_text("+27123456789   Grandma\n")
        self.assertEqual(load_allowlist("imessage"), frozenset({"+27123456789"}))

    def test_blank_lines_ignored(self) -> None:
        self.file.write_text("\n\n+27111\n\n+27222\n\n")
        self.assertEqual(
            load_allowlist("imessage"), frozenset({"+27111", "+27222"}),
        )

    def test_hash_comment_lines_ignored(self) -> None:
        self.file.write_text("# notes\n+27111\n# more notes\n+27222\n")
        self.assertEqual(
            load_allowlist("imessage"), frozenset({"+27111", "+27222"}),
        )

    def test_leading_whitespace_tolerated(self) -> None:
        self.file.write_text("   +27111\n")
        self.assertEqual(load_allowlist("imessage"), frozenset({"+27111"}))


class CheckAllowed(unittest.TestCase):
    """``check_allowed`` raises for missing values, passes for matches."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="hal-al-c-")
        os.environ["HAL_HOME"] = self._tmp
        self.file = allowlist_path("shortcuts")
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.file.write_text("approved_shortcut\nother_shortcut\n")
        _reset_cache_for_tests()

    def tearDown(self) -> None:
        os.environ.pop("HAL_HOME", None)
        _reset_cache_for_tests()
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_allowed_value_passes(self) -> None:
        check_allowed("shortcuts", "approved_shortcut")

    def test_unknown_value_rejected(self) -> None:
        with self.assertRaises(AllowlistRejection):
            check_allowed("shortcuts", "rm -rf /")

    def test_error_names_kind_and_count(self) -> None:
        with self.assertRaises(AllowlistRejection) as ctx:
            check_allowed("shortcuts", "impostor")
        msg = str(ctx.exception)
        self.assertIn("shortcuts", msg)
        self.assertIn("2 entries", msg)


class CacheInvalidation(unittest.TestCase):
    """Cache invalidates on mtime change; unchanged file avoids re-read."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="hal-al-cache-")
        os.environ["HAL_HOME"] = self._tmp
        self.file = allowlist_path("imessage")
        self.file.parent.mkdir(parents=True, exist_ok=True)
        _reset_cache_for_tests()

    def tearDown(self) -> None:
        os.environ.pop("HAL_HOME", None)
        _reset_cache_for_tests()
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_cache_hit_after_initial_load(self) -> None:
        self.file.write_text("+27111\n")
        first = load_allowlist("imessage")
        second = load_allowlist("imessage")
        self.assertIs(first, second)  # frozen set identity — same object

    def test_edit_invalidates_cache(self) -> None:
        self.file.write_text("+27111\n")
        load_allowlist("imessage")
        time.sleep(0.01)
        # Rewrite with explicit later mtime
        self.file.write_text("+27111\n+27222\n")
        new_mtime = time.time() + 1
        os.utime(self.file, (new_mtime, new_mtime))
        reloaded = load_allowlist("imessage")
        self.assertEqual(reloaded, frozenset({"+27111", "+27222"}))


class ValidateParamsIntegration(unittest.TestCase):
    """``ParamSpec.allowlist`` drives ``validate_params`` via ``check_allowed``."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="hal-al-v-")
        os.environ["HAL_HOME"] = self._tmp
        self.file = allowlist_path("imessage")
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.file.write_text("+27111   Grandma\n")
        _reset_cache_for_tests()
        self.script = Script(
            name="imessage_send", tier=3, body="",
            params=(ParamSpec(name="recipient", allowlist="imessage"),),
            timeout_s=5,
        )

    def tearDown(self) -> None:
        os.environ.pop("HAL_HOME", None)
        _reset_cache_for_tests()
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_allowed_recipient_passes_validation(self) -> None:
        validate_params(self.script, {"recipient": "+27111"})

    def test_unknown_recipient_raises_param_validation_error(self) -> None:
        with self.assertRaises(ParamValidationError) as ctx:
            validate_params(self.script, {"recipient": "+27999"})
        self.assertIn("imessage", str(ctx.exception))

    def test_missing_allowlist_file_raises_param_validation_error(self) -> None:
        missing = Script(
            name="s", tier=3, body="",
            params=(ParamSpec(name="x", allowlist="nonexistent_kind"),),
            timeout_s=5,
        )
        with self.assertRaises(ParamValidationError) as ctx:
            validate_params(missing, {"x": "anything"})
        self.assertIn("nonexistent_kind", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
