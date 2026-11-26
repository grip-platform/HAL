"""Durable, fail-open brain sync worker. No gateway imports this module.

Enqueue performs local disk I/O only. flush() belongs in a separate worker; it
never changes a LedgerView's file or thread store. A timeout keeps queued work
for retry, and the last shared-state cache lives separately from local truth.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sqlite3
import time
import urllib.parse
import urllib.request
from pathlib import Path

from . import infinite_thread
from .account_ledger import LedgerView
from .brain import MAX_BODY, cooldown, head_body, sign_head


class BrainClient:
    def __init__(self, url: str, queue: Path, *, timeout: float = 2):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("brain URL requires http(s), without credentials")
        if parsed.query or parsed.fragment or parsed.path not in ("", "/"):
            raise ValueError("brain URL must name the service root")
        self.url, self.path, self.timeout = url.rstrip("/"), Path(queue), timeout
        # Direct tailnet/loopback transport, never ambient inference proxies.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    @contextlib.contextmanager
    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=0.05)
        try:
            with db:
                db.execute("CREATE TABLE IF NOT EXISTS queue (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                           "route TEXT NOT NULL, body TEXT NOT NULL, UNIQUE(route, body))")
                db.execute("CREATE TABLE IF NOT EXISTS cache (id INTEGER PRIMARY KEY, body TEXT NOT NULL)")
                yield db
        finally:
            db.close()

    def enqueue(self, route: str, value: dict) -> bool:
        """Persist only; False means local queue storage failed. Never contacts VPS."""
        try:
            if route == "/v1/cooldowns":
                cooldown(value)
            elif route == "/v1/heads":
                head_body(value)
            else:
                return False
            body = infinite_thread.canonical(value).decode()
            if len(body.encode()) > MAX_BODY:
                return False
            with self.connect() as db:
                db.execute("INSERT OR IGNORE INTO queue(route, body) VALUES (?, ?)", (route, body))
            return True
        except Exception:
            return False

    def collect(self, ledger: LedgerView, root: Path, host: str, *, signer=None) -> bool:
        """Read local metadata, sign head envelopes, queue them; no local writes.

        Public account aliases and random host labels should be provisioned by
        the operator. No token health, token digest, headers or row content leave.
        """
        try:
            ok = True
            for account, entry in ledger.accounts().items():
                quota = entry.get("quota") or {}
                cooling = entry.get("cooling") or {}
                resets = [v for v in (quota.get("reset"), cooling.get("until")) if type(v) is int]
                value = {"account": account, "reset": max(resets) if resets else None,
                         "status": "rejected" if cooling else quota.get("status", "unknown")}
                ok = self.enqueue("/v1/cooldowns", value) and ok
            marks = infinite_thread.heads_snapshot(root)
            if marks:
                if signer is None:
                    infinite_thread.host_public_key(root, host)  # refuse missing keys; never provision
                sign = signer or infinite_thread.default_signer(root, host, provision=False)
                for thread, mark in marks.items():
                    ok = self.enqueue("/v1/heads", sign_head(host, thread, mark, sign)) and ok
            return ok
        except Exception:
            return False

    def _request(self, route, body=None):
        req = urllib.request.Request(self.url + route,
                                     data=body.encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
        with self.opener.open(req, timeout=self.timeout) as response:
            data = response.read(MAX_BODY + 1)
            if len(data) > MAX_BODY:
                raise ValueError("response too large")
            return json.loads(data)

    def flush(self) -> bool:
        """One bounded batch, at-least-once delivery. Network failures keep work."""
        try:
            with self.connect() as db:
                pending = db.execute("SELECT id, route, body FROM queue ORDER BY id LIMIT 100").fetchall()
            for ident, route, body in pending:
                answer = self._request(route, body)
                if answer.get("stored") is not True:
                    return False
                with self.connect() as db:
                    db.execute("DELETE FROM queue WHERE id=?", (ident,))
            answer = self._request("/v1/cooldowns")
            accounts = answer["accounts"]
            if not isinstance(accounts, dict):
                return False
            for label, value in accounts.items():
                if cooldown(value)["account"] != label:
                    return False
            with self.connect() as db:
                db.execute("INSERT OR REPLACE INTO cache VALUES (1, ?)", (json.dumps(accounts),))
            return True
        except Exception:
            return False

    def shared_cooldowns(self) -> dict:
        try:
            with self.connect() as db:
                row = db.execute("SELECT body FROM cache WHERE id=1").fetchone()
            return json.loads(row[0]) if row else {}
        except Exception:
            return {}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--threads", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if args.interval < 1:
        parser.error("interval must be at least one second")
    client = BrainClient(args.url, args.queue)
    while True:
        collected = client.collect(LedgerView(args.ledger), args.threads, args.host)
        synced = client.flush()
        if args.once:
            return 0 if collected and synced else 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
