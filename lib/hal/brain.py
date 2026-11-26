"""Shared cooldown and signed-head service; never called by inference code.

Only the socket peer is trusted for access control. Pins are provisioned out of
band; an HTTP request cannot enrol a host or replace a key.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import re
import sqlite3
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import gateway_guard, infinite_thread
from .http_refusal import _discard_unread_body

logger = logging.getLogger(__name__)


class InvalidUpdate(ValueError):
    """A request violates the brain wire schema or update rules."""


DEFAULT_PORT = 7416
MAX_BODY = 1024 * 1024
HEAD_VERSION = "hal-brain-head/1"
# Unknown is more cautious than a warning, but explicit rejection is strongest.
STATUS_ORDER = {"allowed": 0, "allowed_warning": 1, "unknown": 2, "rejected": 3}
_LABEL = re.compile(r"[a-zA-Z0-9_-]{1,64}\Z")
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_THREAD = re.compile(r"[0-9a-f]{16}\Z")


def cooldown(value: dict) -> dict:
    """Closed wire schema: no credentials, token fingerprints or header values."""
    if not isinstance(value, dict) or set(value) != {"account", "reset", "status"}:
        raise InvalidUpdate("invalid cooldown")
    if not isinstance(value["account"], str) or not _LABEL.fullmatch(value["account"]):
        raise InvalidUpdate("invalid account label")
    reset = value["reset"]
    if reset is not None and (type(reset) is not int or not 0 <= reset <= 100_000_000_000):
        raise InvalidUpdate("invalid reset")
    if not isinstance(value["status"], str) or value["status"] not in STATUS_ORDER:
        raise InvalidUpdate("invalid status")
    return dict(value)


def merge_cooldown(old: dict, new: dict) -> dict:
    old, new = cooldown(old), cooldown(new)
    if old["account"] != new["account"]:
        raise InvalidUpdate("different accounts")
    resets = [v for v in (old["reset"], new["reset"]) if v is not None]
    return {"account": old["account"], "reset": max(resets) if resets else None,
            "status": max((old["status"], new["status"]), key=STATUS_ORDER.__getitem__)}


def head_body(head: dict) -> dict:
    fields = {"v", "host", "thread", "seq", "sha256", "sig"}
    if not isinstance(head, dict) or set(head) != fields or head["v"] != HEAD_VERSION:
        raise InvalidUpdate("invalid head")
    for field, pattern in (("host", _LABEL), ("thread", _THREAD), ("sha256", _HEX)):
        if not isinstance(head[field], str) or not pattern.fullmatch(head[field]):
            raise InvalidUpdate("invalid head field")
    if type(head["seq"]) is not int or not 1 <= head["seq"] < 2**63:
        raise InvalidUpdate("invalid sequence")
    if not isinstance(head["sig"], dict):
        raise InvalidUpdate("invalid signature")
    return {k: v for k, v in head.items() if k != "sig"}


def sign_head(host: str, thread: str, mark: dict, signer) -> dict:
    head = {"v": HEAD_VERSION, "host": host, "thread": thread,
            "seq": mark["seq"], "sha256": mark["sha256"], "sig": {}}
    head["sig"] = signer(infinite_thread.canonical(head_body(head)))
    return head


def verify_head(head: dict, pins: dict, verifier=None) -> None:
    body = head_body(head)
    key = pins.get(body["host"])
    if key is None:
        raise InvalidUpdate("unknown host")
    if verifier is None and key.mldsa65 is not None and not infinite_thread.mldsa_ready():
        raise InvalidUpdate("pinned post-quantum key cannot be checked")
    try:
        verdict, _ = (verifier or infinite_thread.grip_verifier)(
            infinite_thread.canonical(body), head["sig"], key)
    except Exception:
        raise InvalidUpdate("head verification failed") from None
    if verdict != "verified":
        raise InvalidUpdate("head verification failed")


class BrainStore:
    """SQLite transactions serialize updates across HTTP threads and anchor jobs."""

    def __init__(self, path: Path, pins: dict, *, verifier=None):
        self.path, self.pins, self.verifier = Path(path), dict(pins), verifier
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS cooldowns (account TEXT PRIMARY KEY, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS heads (
                    host TEXT, thread TEXT, seq INTEGER NOT NULL, body TEXT NOT NULL,
                    PRIMARY KEY(host, thread));
            """)

    @contextlib.contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=2)
        try:
            with db:
                yield db
        finally:
            db.close()

    def merge(self, update: dict) -> dict:
        update = cooldown(update)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT body FROM cooldowns WHERE account=?", (update["account"],)).fetchone()
            merged = merge_cooldown(json.loads(row[0]), update) if row else update
            db.execute("INSERT OR REPLACE INTO cooldowns VALUES (?, ?)",
                       (merged["account"], json.dumps(merged)))
        return merged

    def cooldowns(self) -> dict:
        with self.connect() as db:
            return {account: json.loads(body) for account, body in
                    db.execute("SELECT account, body FROM cooldowns ORDER BY account")}

    def put_head(self, head: dict) -> None:
        verify_head(head, self.pins, self.verifier)
        body = infinite_thread.canonical(head).decode()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT seq, body FROM heads WHERE host=? AND thread=?",
                             (head["host"], head["thread"])).fetchone()
            if old and head["seq"] < old[0]:
                return  # authenticated delayed retry; never roll back the checkpoint
            if old and head["seq"] == old[0]:
                if json.loads(old[1])["sha256"] != head["sha256"]:
                    raise InvalidUpdate("conflicting head")
                return
            db.execute("INSERT OR REPLACE INTO heads VALUES (?, ?, ?, ?)",
                       (head["host"], head["thread"], head["seq"], body))

    def heads(self) -> list[dict]:
        with self.connect() as db:
            heads = [json.loads(row[0]) for row in db.execute(
                "SELECT body FROM heads ORDER BY host, thread")]
        for head in heads:
            verify_head(head, self.pins, self.verifier)
        return heads


def load_pins(path: Path) -> dict:
    """Explicit host labels -> raw public keys encoded as hex, never private keys."""
    values = json.loads(path.read_bytes())
    pins = {}
    for host, value in values.items():
        if not _LABEL.fullmatch(host) or set(value) - {"ed25519", "mldsa65"}:
            raise ValueError("invalid pins")
        ed = bytes.fromhex(value["ed25519"])
        pq = bytes.fromhex(value["mldsa65"]) if value.get("mldsa65") else None
        if len(ed) != 32 or (pq is not None and len(pq) != 1952):
            raise ValueError("invalid public key length")
        pins[host] = infinite_thread.PinnedKey(ed, pq)
    return pins


def tailscale_whois(address: str):
    result = subprocess.run(["tailscale", "whois", "--json", address],
                            capture_output=True, timeout=3, check=True)
    return json.loads(result.stdout)


class BrainServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128

    def __init__(self, address, store: BrainStore, *, whois=tailscale_whois):
        self.store, self.whois = store, whois
        super().__init__(address, BrainHandler)

    def handle_error(self, request, client_address):
        pass  # never dump request data, peer identity or headers


class BrainHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, *_args):
        pass

    def _reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def _allowed(self):
        allowed = gateway_guard.gateway_access_allowed(
            self.headers, handler=self, whois_func=self.server.whois)
        if not allowed:
            if self.command == "POST":
                _discard_unread_body(self)
            self._reply(403, {"error": "access denied"})
        return allowed

    def do_GET(self):
        if not self._allowed():
            return
        if self.path == "/health":
            self._reply(200, {"service": "hal-brain"})
        elif self.path == "/v1/cooldowns":
            self._reply(200, {"accounts": self.server.store.cooldowns()})
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self):
        if not self._allowed():
            return
        if self.path not in ("/v1/cooldowns", "/v1/heads"):
            _discard_unread_body(self)
            self._reply(404, {"error": "not found"})
            return
        body_read = False
        try:
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or self.headers.get("Transfer-Encoding"):
                raise InvalidUpdate("invalid framing")
            try:
                size = int(lengths[0])
            except ValueError:
                raise InvalidUpdate("invalid length") from None
            if not 0 < size <= MAX_BODY:
                raise InvalidUpdate("invalid size")
            raw = self.rfile.read(size)
            body_read = True
            try:
                body = json.loads(raw)
            except ValueError:
                # Includes invalid encoding, JSON syntax and oversized integers.
                raise InvalidUpdate("invalid JSON") from None
            if self.path == "/v1/heads":
                head_body(body)
                self.server.store.put_head(body)
            else:
                cooldown(body)
                self.server.store.merge(body)
        except (InvalidUpdate, json.JSONDecodeError, RecursionError):
            if not body_read:
                _discard_unread_body(self)
            self._reply(400, {"error": "invalid update"})
            return
        except (OSError, sqlite3.Error):
            self._reply(503, {"error": "state unavailable"})
            return
        except Exception as exc:
            # Exception text/tracebacks can contain request data or credentials.
            logger.error("brain update failed (%s)", type(exc).__name__)
            self._reply(500, {"error": "internal error"})
            return
        self._reply(200, {"stored": True})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--pins", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.port in (4010, 4011, 4012):
        parser.error("reserved service port")
    store = BrainStore(args.state, load_pins(args.pins))
    with BrainServer((args.bind, args.port), store) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
