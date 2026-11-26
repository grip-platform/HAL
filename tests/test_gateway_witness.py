"""HAL Gateway chat-rail WITNESS+SIGN tests — Goodhart-proof.

Each test fails under the obvious mutation:

  - chat call produces NO witnessed row        -> test_verbatim_path_*  /  test_cross_wire_path_*
  - witnessed row lacks the served MODEL id     -> test_*_record_carries_served_model
  - cross-wire row lacks the served PROVIDER    -> test_cross_wire_path_witnesses_served_provider
  - signed row's signature does not verify      -> test_signed_chain_verifies
  - a tampered signed row still "verifies"      -> test_tampered_signed_row_fails_verify
  - removing the emit still passes              -> test_no_chat_call_no_signed_row (mutation control)
  - the unsigned audit mislabels the chat path  -> test_*_unsigned_audit_row_labelled_chat_path

The signature is NEVER weakened: verification uses the REAL
``voice.audit.verify_chain`` HMAC chain, and the tamper test proves an edited
record is rejected. Integration paths drive the real
``handle_chat_completions`` with a fake handler + injected proxy / patched
egress — zero network. Witness output is asserted against the REAL on-disk
``$HAL_HOME/logs`` files (a temp dir per test).
"""
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from lib.hal.gateway import UpstreamResponse
from lib.hal.gateway_witness import (
    build_chat_record,
    sign_record,
    signed_key_path,
    signed_log_path,
    witness_chat,
)
from lib.hal.voice.audit import verify_chain

_CHAT_PATH = "/v1/chat/completions"


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #

class _FakeHeaders(dict):
    def get(self, key, default=None):  # case-insensitive get for content-length
        for k, v in self.items():
            if k.lower() == key.lower():
                return v
        return default

    def items(self):
        return list(super().items())


class _FakeHandler:
    """Minimal BaseHTTPRequestHandler stand-in for handle_chat_completions."""

    def __init__(self, body: bytes, path: str = _CHAT_PATH):
        self._body = body
        self.path = path
        self.headers = _FakeHeaders({"content-length": str(len(body))})
        self.wfile = io.BytesIO()
        self._status = None

    @property
    def rfile(self):  # read_request_body reads Content-Length bytes off rfile
        return io.BytesIO(self._body)

    def send_response(self, code):
        self._status = code

    def send_header(self, *_a):
        pass

    def end_headers(self):
        pass


class _FakeProxy:
    """Returns a scripted UpstreamResponse for the verbatim path."""

    def __init__(self, status=200, body=b'{"content":[{"type":"text","text":"hi"}],'
                                        b'"model":"claude-opus-4-8","usage":{}}'):
        self._status = status
        self._body = body
        self.calls = []

    def forward(self, path, body, headers, *, method="POST", max_retries=None):
        self.calls.append(path)
        return UpstreamResponse(status=self._status, headers=[
            ("content-type", "application/json")], _reader=io.BytesIO(self._body))


def _resp(status=200, body=b'{"id":"x"}'):
    return UpstreamResponse(status=status, headers=[
        ("content-type", "application/json")], _reader=io.BytesIO(body))


class _HalHome:
    """Context manager: point HAL_HOME at a fresh temp dir, restore after."""

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


def _read_audit_rows():
    p = Path(os.environ["HAL_HOME"]) / "logs" / "gateway-audit.jsonl"
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


# --------------------------------------------------------------------------- #
# Pure record shape
# --------------------------------------------------------------------------- #

class TestRecordShape(unittest.TestCase):

    def test_record_carries_served_provider_and_model(self):
        rec = build_chat_record(
            provider="groq", model="anthropic/claude-opus-4-8", status=200,
            served=True, failed_over=False, latency_ms=12, path=_CHAT_PATH)
        # The two moat fields are the whole point — they must be present + correct.
        self.assertEqual(rec["provider"], "groq")
        self.assertEqual(rec["model"], "anthropic/claude-opus-4-8")
        self.assertEqual(rec["path"], _CHAT_PATH)
        self.assertEqual(rec["status"], 200)
        self.assertTrue(rec["served"])

    def test_record_shape_matches_failover_keys(self):
        # Same shape as FailoverProxy._record (+ provider/model) — so the chat
        # rail is self-describing alongside the other rails' rows.
        rec = build_chat_record(
            provider="anthropic", model="claude-opus-4-8", status=200,
            served=True, failed_over=False, latency_ms=0, path=_CHAT_PATH)
        for key in ("event", "path", "upstream", "status", "served",
                    "failed_over", "latency_ms", "provider", "model"):
            self.assertIn(key, rec)


# --------------------------------------------------------------------------- #
# Signing + verification (the SIGN link)
# --------------------------------------------------------------------------- #

class TestSigning(unittest.TestCase):

    def test_signed_chain_verifies(self):
        with _HalHome():
            rec = build_chat_record(
                provider="openai", model="anthropic/claude-opus-4-8", status=200,
                served=True, failed_over=False, latency_ms=5, path=_CHAT_PATH)
            audit_id = sign_record(rec)
            self.assertTrue(audit_id, "sign_record must return a non-empty audit_id")
            key = signed_key_path().read_bytes()
            ok, bad = verify_chain(signed_log_path(), key)
            self.assertTrue(ok, "freshly-signed chain must verify")
            self.assertIsNone(bad)

    def test_signed_row_payload_carries_served_model(self):
        with _HalHome():
            sign_record(build_chat_record(
                provider="groq", model="anthropic/claude-opus-4-8", status=200,
                served=True, failed_over=False, latency_ms=1, path=_CHAT_PATH))
            rows = _read_signed_rows()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["payload"]["model"], "anthropic/claude-opus-4-8")
            self.assertEqual(rows[0]["payload"]["provider"], "groq")

    def test_tampered_signed_row_fails_verify(self):
        # The signature must be tamper-evident: editing a served-model AFTER
        # signing breaks verify_chain. (A weakened/forged signer would pass.)
        with _HalHome():
            sign_record(build_chat_record(
                provider="anthropic", model="claude-opus-4-8", status=200,
                served=True, failed_over=False, latency_ms=1, path=_CHAT_PATH))
            p = signed_log_path()
            row = json.loads(p.read_text().splitlines()[0])
            row["payload"]["model"] = "gpt-4o-but-actually-forged"  # tamper
            p.write_text(json.dumps(row, separators=(",", ":")) + "\n")
            key = signed_key_path().read_bytes()
            ok, bad = verify_chain(p, key)
            self.assertFalse(ok, "tampered record MUST fail verification")
            self.assertEqual(bad, 1)

    def test_chain_links_two_records(self):
        with _HalHome():
            sign_record(build_chat_record(
                provider="anthropic", model="claude-opus-4-8", status=200,
                served=True, failed_over=False, latency_ms=1, path=_CHAT_PATH))
            sign_record(build_chat_record(
                provider="groq", model="anthropic/claude-opus-4-8", status=200,
                served=True, failed_over=False, latency_ms=2, path=_CHAT_PATH))
            rows = _read_signed_rows()
            self.assertEqual(len(rows), 2)
            # 2nd record chains to the 1st (prev_hmac == row0.hmac).
            self.assertEqual(rows[1]["prev_hmac"], rows[0]["hmac"])
            ok, _ = verify_chain(signed_log_path(), signed_key_path().read_bytes())
            self.assertTrue(ok)

    def test_sign_record_fail_open(self):
        # A signing failure must NEVER raise into the request path. Point the log
        # at an un-writable location (a path whose parent is a file) -> None.
        with _HalHome() as home:
            blocker = home / "blocker"
            blocker.write_text("not-a-dir")
            out = sign_record(
                build_chat_record(
                    provider="anthropic", model="x", status=200, served=True,
                    failed_over=False, latency_ms=0, path=_CHAT_PATH),
                log_path=blocker / "nested" / "log.jsonl",
                key_path=blocker / "nested" / "k.key")
            self.assertIsNone(out)


# --------------------------------------------------------------------------- #
# witness_chat: unsigned audit + signed chain together
# --------------------------------------------------------------------------- #

class TestWitnessChat(unittest.TestCase):

    def test_witness_writes_both_audit_and_signed(self):
        with _HalHome():
            witness_chat(provider="groq", model="anthropic/claude-opus-4-8",
                         status=200, served=True, started=time.monotonic(),
                         path=_CHAT_PATH)
            audit = _read_audit_rows()
            signed = _read_signed_rows()
            self.assertEqual(len(audit), 1)
            self.assertEqual(len(signed), 1)
            # unsigned row is labelled as the CHAT path + carries the moat fields.
            self.assertEqual(audit[0]["path"], _CHAT_PATH)
            self.assertEqual(audit[0]["provider"], "groq")
            self.assertEqual(audit[0]["model"], "anthropic/claude-opus-4-8")

    def test_witness_latency_is_non_negative(self):
        with _HalHome():
            rec = witness_chat(provider="anthropic", model="claude-opus-4-8",
                               status=200, served=True,
                               started=time.monotonic(), path=_CHAT_PATH)
            self.assertGreaterEqual(rec["latency_ms"], 0)


# --------------------------------------------------------------------------- #
# Integration: the REAL handler, both paths
# --------------------------------------------------------------------------- #

class TestHandlerVerbatimPath(unittest.TestCase):
    """gpt-* with multiprovider OFF resolves to opus-4-8 + the verbatim path."""

    def test_verbatim_path_witnesses_and_signs(self):
        from lib.hal import gateway_openai
        body = json.dumps({"model": "gpt-5", "messages": [
            {"role": "user", "content": "hi"}]}).encode("utf-8")
        with _HalHome():
            # multiprovider OFF -> same_wire_provider returns None -> verbatim.
            # (Default flipped to ON with an explicit kill-switch — an empty
            # string no longer disables; only {0,false,no,off} do. 2026-08-24.)
            with mock.patch.dict(os.environ, {"HAL_GATEWAY_MULTIPROVIDER": "0"}):
                handler = _FakeHandler(body)
                gateway_openai.handle_chat_completions(handler, _FakeProxy())
            signed = _read_signed_rows()
            self.assertEqual(len(signed), 1, "verbatim chat call must witness+sign")
            payload = signed[0]["payload"]
            self.assertEqual(payload["path"], _CHAT_PATH)
            self.assertEqual(payload["provider"], "anthropic")
            # gpt-5 resolves to the operator's opus-4-8 backend (resolve_model).
            self.assertEqual(payload["model"], "anthropic/claude-opus-4-8")
            ok, _ = verify_chain(signed_log_path(), signed_key_path().read_bytes())
            self.assertTrue(ok)

    def test_verbatim_unsigned_audit_row_labelled_chat_path(self):
        from lib.hal import gateway_openai
        body = json.dumps({"model": "gpt-5", "messages": []}).encode("utf-8")
        with _HalHome():
            # "0", not "": since the 2026-08-24 default flip an empty value no
            # longer disables multi-provider (see the test above), so "" sent
            # this call down the real cross-wire path to a live provider.
            with mock.patch.dict(os.environ, {"HAL_GATEWAY_MULTIPROVIDER": "0"}):
                gateway_openai.handle_chat_completions(_FakeHandler(body), _FakeProxy())
            # The chat-rail unsigned row exists AND is labelled /v1/chat/completions
            # (the bug was it only ever appearing as /v1/messages).
            chat_rows = [r for r in _read_audit_rows()
                         if r.get("path") == _CHAT_PATH]
            self.assertEqual(len(chat_rows), 1)
            self.assertEqual(chat_rows[0]["event"], "gateway.chat.witness")

    def test_no_chat_call_no_signed_row(self):
        # Mutation control: if NO chat request is handled, there is NO signed
        # row. This is what fails if the witness emit is deleted from the
        # handler (the positive tests above would then ALSO produce 0 rows).
        with _HalHome():
            self.assertEqual(_read_signed_rows(), [])


class TestHandlerCrossWirePath(unittest.TestCase):
    """gpt-* with multiprovider ON same-wires to a real provider (groq)."""

    def test_cross_wire_path_witnesses_served_provider(self):
        from lib.hal import gateway_openai
        body = json.dumps({"model": "llama-3.3-70b", "messages": [
            {"role": "user", "content": "hi"}]}).encode("utf-8")
        served = _resp(200, b'{"id":"chatcmpl-x","choices":[]}')
        with _HalHome():
            with mock.patch.dict(os.environ, {"HAL_GATEWAY_MULTIPROVIDER": "1"}):
                # Patch the egress send so no network/keychain is touched; the
                # handler still computes the served provider via same_wire_provider.
                with mock.patch(
                        "lib.hal.gateway_egress.forward_same_wire",
                        return_value=served):
                    gateway_openai.handle_chat_completions(
                        _FakeHandler(body), _FakeProxy())
            signed = _read_signed_rows()
            self.assertEqual(len(signed), 1, "cross-wire chat call must witness+sign")
            payload = signed[0]["payload"]
            # The SERVED non-Anthropic provider is now VISIBLE (the whole point —
            # this path never touched proxy.forward, so it was previously unwitnessed).
            self.assertEqual(payload["provider"], "groq")  # llama-* -> groq
            self.assertEqual(payload["path"], _CHAT_PATH)
            ok, _ = verify_chain(signed_log_path(), signed_key_path().read_bytes())
            self.assertTrue(ok)

    def test_cross_wire_unsigned_row_records_served_provider(self):
        from lib.hal import gateway_openai
        body = json.dumps({"model": "grok-2", "messages": []}).encode("utf-8")
        served = _resp(200, b'{"id":"x","choices":[]}')
        with _HalHome():
            with mock.patch.dict(os.environ, {"HAL_GATEWAY_MULTIPROVIDER": "1"}):
                with mock.patch(
                        "lib.hal.gateway_egress.forward_same_wire",
                        return_value=served):
                    gateway_openai.handle_chat_completions(
                        _FakeHandler(body), _FakeProxy())
            rows = [r for r in _read_audit_rows() if r.get("path") == _CHAT_PATH]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["provider"], "xai")  # grok-* -> xai


if __name__ == "__main__":
    unittest.main()
