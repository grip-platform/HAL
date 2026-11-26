"""Tests for ``bin/osa`` — the HAL osascript dispatch CLI.

The CLI is a thin adapter — we verify it wires arguments through to
:func:`lib.hal.osa.dispatch.osa` correctly and returns the documented exit
codes. Behaviour that belongs to the library (auth, validation, audit)
is tested where it lives.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest.mock import patch


_REPO_ROOT = Path(__file__).resolve().parent.parent
_BIN_OSA = _REPO_ROOT / "bin" / "osa"
_FIXTURE_SECRET = bytes(range(32))  # deterministic, not a real secret


def _load_cli():
    """Import ``bin/osa`` as a module despite its extensionless filename.

    ``spec_from_file_location`` defaults to extension-based loader
    inference, which skips files without a ``.py`` suffix. Passing
    :class:`SourceFileLoader` explicitly forces source-file treatment.
    """
    if "hal_osa_cli" in sys.modules:
        return sys.modules["hal_osa_cli"]
    loader = SourceFileLoader("hal_osa_cli", str(_BIN_OSA))
    spec = importlib.util.spec_from_loader("hal_osa_cli", loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hal_osa_cli"] = mod
    loader.exec_module(mod)
    return mod


class ListCommand(unittest.TestCase):
    """``osa list`` prints every registered script with its tier."""

    def test_list_prints_every_script(self) -> None:
        cli = _load_cli()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli.main(["list"])
        self.assertEqual(rc, 0)
        output = buf.getvalue()
        self.assertIn("battery_get", output)
        self.assertIn("volume_get", output)
        self.assertIn("notification", output)
        self.assertIn("frontmost_app_get", output)
        self.assertIn("screen_lock", output)
        self.assertIn("speak", output)

    def test_list_is_tier_annotated(self) -> None:
        cli = _load_cli()
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.main(["list"])
        for line in buf.getvalue().splitlines():
            self.assertTrue(line.startswith("tier-"), f"row {line!r}")


class InfoCommand(unittest.TestCase):
    """``osa info`` prints the script schema as JSON."""

    def test_known_script_returns_valid_json(self) -> None:
        cli = _load_cli()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli.main(["info", "battery_get"])
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["name"], "battery_get")
        self.assertEqual(payload["tier"], 1)
        self.assertEqual(payload["params"], [])

    def test_unknown_script_exits_2(self) -> None:
        cli = _load_cli()
        err = io.StringIO()
        with redirect_stderr(err):
            rc = cli.main(["info", "does_not_exist"])
        self.assertEqual(rc, 2)
        self.assertIn("does_not_exist", err.getvalue())


class MintCommand(unittest.TestCase):
    """``osa mint`` requires a secret and emits a CapabilityToken JSON."""

    def test_no_secret_returns_exit_1(self) -> None:
        cli = _load_cli()
        err = io.StringIO()
        with patch("hal_osa_cli.default_secret", side_effect=_raise_missing):
            with redirect_stderr(err):
                rc = cli.main(["mint", "notification"])
        self.assertEqual(rc, 1)
        self.assertIn("secret", err.getvalue().lower())

    def test_with_secret_prints_token_json(self) -> None:
        cli = _load_cli()
        buf = io.StringIO()
        with patch("hal_osa_cli.default_secret", return_value=_FIXTURE_SECRET):
            with redirect_stdout(buf):
                rc = cli.main(["mint", "notification", "--ttl", "30"])
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        for field in ("script_name", "issued_at", "expires_at", "nonce", "mac"):
            self.assertIn(field, payload)
        self.assertEqual(payload["script_name"], "notification")
        self.assertAlmostEqual(
            payload["expires_at"] - payload["issued_at"], 30, delta=0.5
        )


def _raise_missing() -> bytes:
    from lib.hal.osa.auth import MissingSecret
    raise MissingSecret("HAL_OSA_CAPABILITY_SECRET is not set")


class RunParams(unittest.TestCase):
    """``_parse_params`` and ``_load_token`` are the CLI's only real logic."""

    def test_parse_single_pair(self) -> None:
        cli = _load_cli()
        self.assertEqual(cli._parse_params(["k=v"]), {"k": "v"})

    def test_parse_multiple_pairs(self) -> None:
        cli = _load_cli()
        self.assertEqual(
            cli._parse_params(["a=1", "b=two"]), {"a": "1", "b": "two"}
        )

    def test_value_with_equals_keeps_right_split(self) -> None:
        cli = _load_cli()
        self.assertEqual(cli._parse_params(["url=a=b"]), {"url": "a=b"})

    def test_malformed_pair_exits_3(self) -> None:
        cli = _load_cli()
        with self.assertRaises(SystemExit) as ctx:
            cli._parse_params(["no_equals"])
        self.assertEqual(ctx.exception.code, 3)

    def test_load_token_from_inline_json(self) -> None:
        cli = _load_cli()
        payload = {
            "script_name": "x",
            "issued_at": 1.0,
            "expires_at": 2.0,
            "nonce": "abc",
            "mac": "deadbeef",
        }
        tok = cli._load_token(json.dumps(payload))
        self.assertEqual(tok.script_name, "x")
        self.assertEqual(tok.mac, "deadbeef")

    def test_load_token_from_file(self) -> None:
        cli = _load_cli()
        payload = {
            "script_name": "y",
            "issued_at": 1.0,
            "expires_at": 2.0,
            "nonce": "xyz",
            "mac": "cafef00d",
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as fh:
            json.dump(payload, fh)
            path = fh.name
        try:
            tok = cli._load_token(path)
        finally:
            os.unlink(path)
        self.assertEqual(tok.script_name, "y")


if __name__ == "__main__":
    unittest.main()
