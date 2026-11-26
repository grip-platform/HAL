"""A loopback HTTP upstream for gateway tests: scripted answers, recorded requests.

Shared by ``tests/test_gateway_one_credential.py`` and
``tests/test_account_ledger_hooks.py``. Binds 127.0.0.1 on a port the kernel
picks; nothing reaches a real provider.
"""
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROXY_VARIABLES = ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY",
                   "all_proxy")


class HttpUpstream:
    """Answers scripted ``(status, headers)`` pairs in order; keeps every request's headers."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.requests: list = []
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 (stdlib naming)
                self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
                upstream.requests.append(list(self.headers.items()))
                status, headers = upstream.answers.pop(0)
                payload = b'{"type": "message"}'
                self.send_response(status)
                for name, value in headers:
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, fmt, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def clear_proxies(monkeypatch) -> None:
    """Send this test's HTTP straight to loopback, whatever proxy the shell exports."""
    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "*")


def credentials(headers) -> list:
    """The credential headers in ``headers``, lower-cased, in order."""
    return [(name.lower(), value) for name, value in headers
            if name.lower() in ("authorization", "x-api-key")]
