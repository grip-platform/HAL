"""HAL Gateway /v1/messages-rail WITNESS+SIGN tests — Goodhart-proof.

Four mutation-sensitive tests:

1. A cross-wire /v1/messages request (model routes to deepseek, multiprovider ON)
   MUST produce a signed row whose served provider+model verifies via verify_chain
   → (True, None).  [the cross-wire case on the messages rail]

2. Tampering one byte of a fixture signed log → verify_chain returns (False, N);
   grip-verify-spine exits non-zero.

3. Removing _witness_messages_rail (or its emit) makes test 1 FAIL — proves the
   wire is load-bearing (the signed row appears ONLY because the wire fired).

4. That same cross-wire request is witnessed with failed_over FALSE, in both the
   unsigned audit row and the signed one. Serving the model the caller asked for
   is routing; this rail never knows whether a failover happened.

HAL_HOME is isolated to a fresh temp dir for every test.  Zero network.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Ensure the repo root is on sys.path so lib.hal.* resolves from any cwd.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lib.hal.gateway import UpstreamResponse, handle_proxy_request
from lib.hal.gateway_picker_alias import alias_for
from lib.hal.gateway_witness import signed_key_path, signed_log_path
from lib.hal.voice.audit import verify_chain

_MESSAGES_PATH = "/v1/messages"
_DEEPSEEK_MODEL = "deepseek-chat"


# ---------------------------------------------------------------------------
# Test doubles (mirrors test_gateway_witness.py style)
# ---------------------------------------------------------------------------

class _FakeHeaders(dict):
    def get(self, key, default=None):
        for k, v in self.items():
            if k.lower() == key.lower():
                return v
        return default

    def items(self):
        return list(super().items())


class _FakeHandler:
    """Minimal BaseHTTPRequestHandler stand-in for handle_proxy_request."""

    def __init__(self, body: bytes, path: str = _MESSAGES_PATH, command: str = "POST"):
        self._body = body
        self.path = path
        self.command = command
        self.headers = _FakeHeaders({"content-length": str(len(body))})
        self.wfile = io.BytesIO()
        self._status = None

    @property
    def rfile(self):
        return io.BytesIO(self._body)

    def send_response(self, code):
        self._status = code

    def send_header(self, *_a):
        pass

    def end_headers(self):
        pass


class _FakeProxy:
    """Returns a scripted UpstreamResponse for the verbatim relay path."""

    def __init__(self, status: int = 200,
                 body: bytes = b'{"type":"message","content":[{"type":"text","text":"ok"}]}'):
        self._status = status
        self._body = body

    def forward(self, path, body, headers, *, method="POST", max_retries=None):
        return UpstreamResponse(
            status=self._status,
            headers=[("content-type", "application/json")],
            _reader=io.BytesIO(self._body),
        )


class _HalHome:
    """Context manager: isolate HAL_HOME to a fresh temp dir, restore after."""

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = os.environ.get("HAL_HOME")
        os.environ["HAL_HOME"] = self._tmp.name
        return Path(self._tmp.name)

    def __exit__(self, *exc):
        if self._saved is not None:
            os.environ["HAL_HOME"] = self._saved
        else:
            os.environ.pop("HAL_HOME", None)
        self._tmp.cleanup()


def _read_signed_rows():
    p = signed_log_path()
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


# ---------------------------------------------------------------------------
# Test 1 — cross-wire /v1/messages → signed row verifies
# ---------------------------------------------------------------------------

class TestMessagesRailWiresFired(unittest.TestCase):
    """Test 1: multiprovider ON, deepseek model → signed row (True, None)."""

    def _run_cross_wire_request(self, home: Path):
        """Drive handle_proxy_request with multiprovider ON and a deepseek model.

        _maybe_cross_wire routes deepseek-chat to the deepseek provider when
        HAL_GATEWAY_MULTIPROVIDER=1, so we mock cross_wire_forward to return a
        scripted response rather than touching the network.
        """
        body = json.dumps({
            "model": _DEEPSEEK_MODEL,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "PONG"}],
        }).encode()
        handler = _FakeHandler(body)
        proxy = _FakeProxy()

        fake_resp = UpstreamResponse(
            status=200,
            headers=[("content-type", "application/json")],
            _reader=io.BytesIO(b'{"id":"x","choices":[{"message":{"content":"PONG"}}]}'),
        )
        with mock.patch.dict(os.environ, {"HAL_GATEWAY_MULTIPROVIDER": "1"}):
            with mock.patch(
                "lib.hal.gateway_egress.cross_wire_forward",
                return_value=fake_resp,
            ):
                handle_proxy_request(handler, proxy, _MESSAGES_PATH)

    def test_cross_wire_messages_produces_signed_row_that_verifies(self):
        with _HalHome() as home:
            self._run_cross_wire_request(home)
            rows = _read_signed_rows()
            self.assertEqual(len(rows), 1,
                             "cross-wire /v1/messages must emit exactly one signed row")
            payload = rows[0]["payload"]
            self.assertEqual(payload["path"], _MESSAGES_PATH)
            # deepseek-chat routes to the deepseek provider.
            self.assertEqual(payload["provider"], "deepseek")
            self.assertIn("deepseek", payload.get("model", "").lower(),
                          "model field must carry the requested model name")
            # Chain integrity — the load-bearing moat guarantee.
            key = signed_key_path().read_bytes()
            ok, bad_line = verify_chain(signed_log_path(), key)
            self.assertTrue(ok, f"signed chain must verify; bad_line={bad_line}")
            self.assertIsNone(bad_line)


# ---------------------------------------------------------------------------
# Test 2 — tampered log → verify_chain FAIL + grip-verify-spine exits non-zero
# ---------------------------------------------------------------------------

class TestTamperedLogFails(unittest.TestCase):
    """Test 2: editing one byte after signing → chain broken at that line."""

    def test_tampered_row_fails_verify_and_spine_exits_nonzero(self):
        from lib.hal.gateway_witness import build_chat_record, sign_record

        with _HalHome() as home:
            # Build and sign a fixture row (reuse build_chat_record for brevity).
            sign_record(build_chat_record(
                provider="deepseek", model=_DEEPSEEK_MODEL, status=200,
                served=True, failed_over=True, latency_ms=42,
                path=_MESSAGES_PATH,
            ))
            p = signed_log_path()
            row = json.loads(p.read_text().splitlines()[0])
            row["payload"]["model"] = "forged-model"  # tamper
            p.write_text(json.dumps(row, separators=(",", ":")) + "\n")

            key = signed_key_path().read_bytes()
            ok, bad = verify_chain(p, key)
            self.assertFalse(ok, "tampered row MUST fail verify_chain")
            self.assertEqual(bad, 1)

            # grip-verify-spine must also exit non-zero on the same tampered log.
            spine = str(_REPO / "bin" / "grip-verify-spine")
            result = subprocess.run(
                [sys.executable, spine],
                env={**os.environ, "HAL_HOME": home},
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0,
                                "grip-verify-spine must exit non-zero on tampered log")


# ---------------------------------------------------------------------------
# Test 3 — wire is load-bearing: patching it out → no signed row
# ---------------------------------------------------------------------------

class TestWireIsLoadBearing(unittest.TestCase):
    """Test 3: if _witness_messages_rail is suppressed, no signed row appears."""

    def test_no_witness_call_means_no_signed_row(self):
        body = json.dumps({
            "model": _DEEPSEEK_MODEL,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "PONG"}],
        }).encode()
        handler = _FakeHandler(body)
        proxy = _FakeProxy()
        fake_resp = UpstreamResponse(
            status=200,
            headers=[("content-type", "application/json")],
            _reader=io.BytesIO(b'{"id":"x","choices":[]}'),
        )

        with _HalHome():
            with mock.patch.dict(os.environ, {"HAL_GATEWAY_MULTIPROVIDER": "1"}):
                with mock.patch(
                    "lib.hal.gateway_egress.cross_wire_forward",
                    return_value=fake_resp,
                ):
                    # Patch _witness_messages_rail to a no-op — simulates the
                    # wire being removed.  The signed log must remain empty.
                    with mock.patch(
                        "lib.hal.gateway._witness_messages_rail",
                        return_value=None,
                    ):
                        handle_proxy_request(handler, proxy, _MESSAGES_PATH)

            rows = _read_signed_rows()
            self.assertEqual(rows, [],
                             "signed log must be empty when the wire is suppressed; "
                             "a non-empty log means the wire is NOT load-bearing")

    def test_call_site_invokes_witness_exactly_once(self):
        """Direct measurement (not inference): handle_proxy_request MUST call
        _witness_messages_rail exactly once per served request, passing the
        response whose status is witnessed. Proves the CALL SITE is load-bearing,
        not merely that the function writes when called (forward-anchor for the
        AGORA #451 concern-4 instinct: measure the call, don't infer it)."""
        body = json.dumps({
            "model": _DEEPSEEK_MODEL,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "PONG"}],
        }).encode()
        handler = _FakeHandler(body)
        proxy = _FakeProxy()
        fake_resp = UpstreamResponse(
            status=200,
            headers=[("content-type", "application/json")],
            _reader=io.BytesIO(b'{"id":"x","choices":[]}'),
        )
        with _HalHome():
            with mock.patch.dict(os.environ, {"HAL_GATEWAY_MULTIPROVIDER": "1"}):
                with mock.patch(
                    "lib.hal.gateway_egress.cross_wire_forward",
                    return_value=fake_resp,
                ):
                    with mock.patch(
                        "lib.hal.gateway._witness_messages_rail",
                    ) as spy:
                        handle_proxy_request(handler, proxy, _MESSAGES_PATH)

        self.assertEqual(
            spy.call_count, 1,
            "the /v1/messages handler MUST call _witness_messages_rail exactly "
            "once per request — the call site, not just the function, is load-bearing")
        passed_resp = (spy.call_args.args[1] if len(spy.call_args.args) > 1
                       else spy.call_args.kwargs.get("resp"))
        self.assertEqual(getattr(passed_resp, "status", None), 200,
                         "the witnessed response (its status) must reach the wire")


# ---------------------------------------------------------------------------
# Test 4 — a cross-wire request is routing, never a failover
# ---------------------------------------------------------------------------

class TestCrossWireIsNotFailover(unittest.TestCase):
    """Test 4: the rail must not call a requested non-Anthropic model a failover.

    It used to set ``failed_over = provider != "anthropic"``, where the provider
    comes from the MODEL the caller asked for, so every cross-wire request was
    witnessed as a failover (50 such rows in the live log on 2026-10-05).
    Failover means "served by a different upstream than the one first tried",
    which only ``FailoverProxy`` knows, and it records its own rows for that.
    """

    def _rows_for(self, model):
        """Drive one request; return (unsigned audit rows, signed rows)."""
        body = json.dumps({
            "model": model,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "PONG"}],
        }).encode()
        fake_resp = UpstreamResponse(
            status=200,
            headers=[("content-type", "application/json")],
            _reader=io.BytesIO(b'{"id":"x","choices":[]}'),
        )
        with _HalHome() as home:
            with mock.patch.dict(os.environ, {"HAL_GATEWAY_MULTIPROVIDER": "1"}):
                with mock.patch("lib.hal.gateway_egress.cross_wire_forward",
                                return_value=fake_resp):
                    handle_proxy_request(_FakeHandler(body), _FakeProxy(), _MESSAGES_PATH)
            log = home / "logs" / "gateway-audit.jsonl"
            audit = [json.loads(ln) for ln in log.read_text().splitlines()]
            return audit, _read_signed_rows()

    def test_a_cross_wire_model_is_witnessed_without_failed_over(self):
        # The plain id, and the picker alias that canonicalises to it.
        for model in (_DEEPSEEK_MODEL, alias_for(_DEEPSEEK_MODEL)):
            with self.subTest(model=model):
                audit, signed = self._rows_for(model)
                (row,) = audit
                # Really the cross-wire case, so the assertion below is not vacuous.
                self.assertEqual(row["provider"], "deepseek")
                self.assertIs(row["failed_over"], False)
                self.assertIs(signed[0]["payload"]["failed_over"], False)


if __name__ == "__main__":
    unittest.main()
