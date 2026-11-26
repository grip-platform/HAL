"""Step 6 brain: isolated sockets, ephemeral keys, local TSA, no live services."""
from __future__ import annotations

import base64
import copy
import hashlib
import http.client
import itertools
import json
import socket
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from lib.hal import brain, brain_anchor, brain_client, infinite_thread
from lib.hal.account_ledger import LedgerView


@pytest.fixture
def signing():
    private = Ed25519PrivateKey.generate()
    key = infinite_thread.PinnedKey(private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw))

    def sign(message):
        return {"scheme": "test-ed25519", "signature": private.sign(message).hex()}

    def verify(message, sig, pinned):
        try:
            Ed25519PublicKey.from_public_bytes(pinned.ed25519).verify(bytes.fromhex(sig["signature"]), message)
        except Exception:
            return "broken", "ed25519 test backend"
        return "verified", "ed25519 test backend"

    return key, sign, verify


@pytest.fixture
def store(tmp_path, signing):
    key, _, verify = signing
    return brain.BrainStore(tmp_path / "brain.db", {"host-test": key}, verifier=verify)


def head(signing, seq=1, digest="a" * 64, thread="b" * 16):
    return brain.sign_head("host-test", thread, {"seq": seq, "sha256": digest}, signing[1])


@contextmanager
def serving(store, **kwargs):
    server = brain.BrainServer(("127.0.0.1", 0), store, **kwargs)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)
        assert not worker.is_alive()


def request(server, route, body=None, headers=None):
    conn = http.client.HTTPConnection(*server.server_address, timeout=3)
    try:
        conn.request("POST" if body is not None else "GET", route,
                     json.dumps(body) if body is not None else None, headers or {})
        response = conn.getresponse()
        return response.status, json.loads(response.read())
    finally:
        conn.close()


def test_cooldown_merge_latest_and_most_restrictive(store):
    values = [{"account": "account-test", "reset": 800, "status": "allowed"},
              {"account": "account-test", "reset": 500, "status": "rejected"},
              {"account": "account-test", "reset": None, "status": "unknown"}]
    expected = {"account": "account-test", "reset": 800, "status": "rejected"}
    for order in itertools.permutations(values):
        merged = order[0]
        for value in order[1:]:
            merged = brain.merge_cooldown(merged, value)
        assert merged == expected
        assert brain.merge_cooldown(merged, merged) == expected
    for value in values:
        store.merge(value)
    assert store.cooldowns() == {"account-test": expected}
    for first, second in itertools.combinations(brain.STATUS_ORDER, 2):
        a = {"account": "account-test", "reset": None, "status": first}
        b = dict(a, status=second)
        assert brain.merge_cooldown(a, b) == b


def test_valid_head_persists_and_never_regresses(store, signing):
    first, later = head(signing), head(signing, 2, "c" * 64)
    store.put_head(first)
    store.put_head(later)
    store.put_head(first)
    store.put_head(later)
    reopened = brain.BrainStore(store.path, store.pins, verifier=signing[2])
    assert reopened.heads() == [later]
    with pytest.raises(ValueError, match="conflicting"):
        store.put_head(head(signing, 2, "d" * 64))
    assert reopened.heads() == [later]


@pytest.mark.parametrize("attack", ["forged", "unknown-host", "wrong-key", "changed-thread", "changed-sequence"])
def test_bad_heads_refused_over_http(store, signing, attack):
    update = head(signing)
    if attack == "forged":
        update["sha256"] = "e" * 64
    elif attack == "unknown-host":
        update = brain.sign_head("unknown-test", "b" * 16, {"seq": 1, "sha256": "a" * 64}, signing[1])
    elif attack == "wrong-key":
        other = Ed25519PrivateKey.generate()
        update["sig"]["signature"] = other.sign(infinite_thread.canonical(brain.head_body(update))).hex()
    elif attack == "changed-thread":
        update["thread"] = "c" * 16
    else:
        update["seq"] = 2
    with serving(store) as server:
        assert request(server, "/v1/heads", update)[0] == 400
    assert store.heads() == []


def test_stored_head_is_reverified_before_anchoring(store, signing, tmp_path):
    store.put_head(head(signing))
    with store.connect() as db:
        forged = head(signing)
        forged["sha256"] = "e" * 64
        db.execute("UPDATE heads SET body=?", (json.dumps(forged),))
    with pytest.raises(ValueError, match="verification"):
        brain_anchor.anchor_once(store, tmp_path / "anchors", tmp_path / "ca.pem")
    assert not (tmp_path / "anchors").exists()


def test_merkle_root_reproducible_and_binds_every_head(signing):
    heads = [head(signing, thread=char * 16) for char in "bcd"]
    leaves = [hashlib.sha256(b"\x00" + infinite_thread.canonical(h)).digest() for h in heads]
    left = hashlib.sha256(b"\x01" + leaves[0] + leaves[1]).digest()
    right = hashlib.sha256(b"\x01" + leaves[2] + leaves[2]).digest()
    expected = hashlib.sha256(b"\x01" + left + right).hexdigest()
    for order in itertools.permutations(heads):
        assert brain_anchor.merkle_root(list(order)) == expected
    assert brain_anchor.merkle_root([]) == hashlib.sha256(b"").hexdigest()
    assert brain_anchor.merkle_root(heads[:1]) == leaves[0].hex()
    for index in range(3):
        changed = copy.deepcopy(heads)
        changed[index]["sha256"] = "e" * 64
        assert brain_anchor.merkle_root(changed) != expected
    with pytest.raises(ValueError, match="duplicate"):
        brain_anchor.merkle_root([heads[0], heads[0]])


@pytest.mark.parametrize("authenticated,forwarded,expected", [
    (False, False, 403), (True, False, 200), (True, True, 403)])
def test_non_loopback_requires_authenticated_tag(store, monkeypatch, authenticated, forwarded, expected):
    # Real accepted socket and HTTP handler, with only the kernel peer address
    # substituted: portable on hosts where no non-loopback test bind exists.
    original = brain.BrainServer.get_request

    def remote(server):
        conn, peer = original(server)
        return conn, ("192.0.2.10", peer[1])

    monkeypatch.setattr(brain.BrainServer, "get_request", remote)
    monkeypatch.setenv("HAL_GATEWAY_TS_ACL_TAG", "tag:brain-test")
    monkeypatch.delenv("HAL_GATEWAY_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("HAL_GATEWAY_REQUIRE_CLIENT_CERT", raising=False)
    seen = []

    def whois(address):
        seen.append(address)
        return {"Node": {"Tags": ["tag:brain-test"] if authenticated else []}}

    headers = {"X-Forwarded-For": "127.0.0.1"} if forwarded else {}
    with serving(store, whois=whois) as server:
        for route, body in (("/health", None), ("/v1/cooldowns", None),
                            ("/v1/cooldowns", {"account": "account-test", "reset": 5, "status": "rejected"})):
            assert request(server, route, body, headers)[0] == expected
    if not forwarded:
        assert seen == ["192.0.2.10"] * 3


@pytest.mark.parametrize("bad", [
    {"account": "account-test", "reset": True, "status": "allowed"},
    {"account": "account-test", "reset": -1, "status": "allowed"},
    {"account": "account-test", "reset": 1, "status": "made-up"},
    {"account": "../account", "reset": 1, "status": "allowed"},
    {"account": "account-test", "reset": 1, "status": "allowed", "token": "fake-account-A-token"},
])
def test_cooldown_schema_refuses_untrusted_fields(store, bad):
    with serving(store) as server:
        assert request(server, "/v1/cooldowns", bad)[0] == 400
    assert store.cooldowns() == {}


def test_client_outage_queues_then_syncs_without_touching_local_state(store, signing, tmp_path):
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps({"accounts": {"account-test": {
        "quota": {"status": "allowed", "reset": 500},
        "cooling": {"until": 800}, "token": {"digest": "fake-fingerprint"}}}}))
    thread_root = tmp_path / "threads"
    local = infinite_thread.ThreadStore(root=thread_root, host="host-test", signer=signing[1])
    assert local.append("served", {"status": 200}, session_id="session-test").written
    before = {p: p.read_bytes() for p in thread_root.rglob("*") if p.is_file()}
    ledger_before = ledger_path.read_bytes()
    # Bound but not listening: deterministic connection refusal without a race
    # for a port that another process could take. Never touches core ports.
    with socket.socket() as unavailable:
        unavailable.bind(("127.0.0.1", 0))
        url = "http://127.0.0.1:" + str(unavailable.getsockname()[1])
        client = brain_client.BrainClient(url, tmp_path / "queue.db", timeout=0.1)
        assert client.collect(LedgerView(ledger_path), thread_root, "host-test", signer=signing[1])
        assert client.flush() is False
        with client.connect() as db:
            assert db.execute("SELECT COUNT(*) FROM queue").fetchone()[0] == 2
            assert "fake-fingerprint" not in str(db.execute("SELECT body FROM queue").fetchall())
        assert client.shared_cooldowns() == {}
        assert ledger_path.read_bytes() == ledger_before
        assert all(path.read_bytes() == data for path, data in before.items())
        assert local.append("served", {"status": 200}, session_id="session-test").written
        assert client.collect(LedgerView(ledger_path), thread_root, "host-test", signer=signing[1])
    with serving(store) as server:
        restarted = brain_client.BrainClient("http://127.0.0.1:" + str(server.server_port), client.path)
        assert restarted.flush() is True
        assert restarted.shared_cooldowns()["account-test"] == {
            "account": "account-test", "status": "rejected", "reset": 800}
        with restarted.connect() as db:
            assert db.execute("SELECT COUNT(*) FROM queue").fetchone()[0] == 0
        assert store.heads()[0]["seq"] == 2
        assert ledger_path.read_bytes() == ledger_before


def test_client_queue_storage_failure_is_fail_open(tmp_path):
    blocking = tmp_path / "file"
    blocking.write_text("not a directory")
    client = brain_client.BrainClient("http://127.0.0.1:1", blocking / "queue.db")
    assert client.enqueue("/v1/cooldowns", {"account": "account-test", "reset": 5, "status": "rejected"}) is False
    assert client.flush() is False
    assert client.shared_cooldowns() == {}


@pytest.mark.parametrize("private_present", [False, True])
def test_collection_never_provisions_local_keys(tmp_path, signing, monkeypatch, private_present):
    root = tmp_path / "threads"
    private = root / "keys" / "host-test" / "signing.key"
    private.parent.mkdir(parents=True)
    if private_present:
        private.write_text("fake-test-private-key-placeholder")
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    provisioned = []

    def provision(base, host):
        provisioned.append(host)
        private.write_text("fake-replacement-private-key-placeholder")

    def sign(message, host, *, base):
        private.read_bytes()  # missing private material must fail without provisioning
        return signing[1](message)

    monkeypatch.setattr(infinite_thread, "_ensure_keys", provision)
    monkeypatch.setattr(infinite_thread, "_pq_module", lambda: SimpleNamespace(dual_sign=sign))
    monkeypatch.setattr(infinite_thread, "mldsa_ready", lambda: True)
    monkeypatch.setattr(infinite_thread, "host_public_key", lambda *_: signing[0])
    monkeypatch.setattr(infinite_thread, "heads_snapshot", lambda *_: {
        "b" * 16: {"seq": 1, "sha256": "a" * 64}})
    client = brain_client.BrainClient("http://127.0.0.1:1", tmp_path / "queue.db")
    assert client.collect(LedgerView(tmp_path / "absent-ledger"), root, "host-test") is private_present
    assert provisioned == []
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
    with client.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM queue").fetchone()[0] == int(private_present)


def openssl(*args):
    result = subprocess.run(["openssl", *map(str, args)], capture_output=True, timeout=15)
    assert result.returncode == 0, "local OpenSSL fixture failed (output withheld)"
    return result.stdout


@pytest.fixture
def tsa(tmp_path):
    ca, key = tmp_path / "tsa.pem", tmp_path / "tsa-private.pem"
    openssl("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", key,
            "-out", ca, "-days", "1", "-subj", "/CN=Fake Test TSA",
            "-addext", "extendedKeyUsage=critical,timeStamping",
            "-addext", "keyUsage=critical,digitalSignature", "-addext", "basicConstraints=critical,CA:FALSE")
    serial = tmp_path / "serial"
    serial.write_text("01")
    config = tmp_path / "tsa.conf"
    config.write_text(f"""[tsa]
default_tsa = test
[test]
serial = {serial}
signer_cert = {ca}
signer_key = {key}
signer_digest = sha256
default_policy = 1.2.3.4.1
digests = sha256
accuracy = secs:1
ordering = yes
tsa_name = yes
ess_cert_id_chain = no
ess_cert_id_alg = sha256
""")

    def issue(digest):
        query = tmp_path / "request.der"
        token = tmp_path / "token.der"
        openssl("ts", "-query", "-digest", digest.hex(), "-sha256", "-cert", "-out", query)
        openssl("ts", "-reply", "-config", config, "-queryfile", query, "-token_out", "-out", token)
        return {"status": "anchored", "token_b64": base64.b64encode(token.read_bytes()).decode()}

    return ca, issue


def test_anchor_stores_real_verifiable_timestamp(store, signing, tsa, tmp_path, monkeypatch):
    store.put_head(head(signing))
    ca, issue = tsa
    calls = []

    def anchor(digest):
        calls.append(digest)
        return issue(digest)

    monkeypatch.setattr(infinite_thread, "grip_anchor", anchor)
    result = brain_anchor.anchor_once(store, tmp_path / "anchors", ca)
    assert result["status"] == "anchored"
    manifest = json.loads((Path(result["path"]) / "root.json").read_bytes())
    assert calls == [bytes.fromhex(manifest["root"])]
    assert manifest["heads"] == store.heads()
    assert manifest["root"] == brain_anchor.merkle_root(manifest["heads"])
    token = Path(result["path"]) / "timestamp.der"
    openssl("ts", "-verify", "-digest", result["root"], "-token_in", "-in", token, "-CAfile", ca)
    assert not brain_anchor.verify_timestamp("f" * 64, token, ca)


@pytest.mark.parametrize("failure", ["wrong-imprint", "bad-signature", "disabled", "down"])
def test_anchor_failures_never_claim_anchored(store, signing, tsa, tmp_path, failure):
    store.put_head(head(signing))
    ca, issue = tsa

    def anchor(digest):
        if failure == "down":
            raise OSError("fake-account-A-token")
        if failure == "disabled":
            return {"status": "disabled"}
        result = issue(b"\x00" * 32 if failure == "wrong-imprint" else digest)
        if failure == "bad-signature":
            from asn1crypto import cms, tsp  # noqa: F401 - register timestamp OIDs before CMS parsing
            content = cms.ContentInfo.load(base64.b64decode(result["token_b64"]))
            signer = content["content"]["signer_infos"][0]
            sig = signer["signature"].native
            signer["signature"] = bytes([sig[0] ^ 1]) + sig[1:]
            result["token_b64"] = base64.b64encode(content.dump()).decode()
        return result

    result = brain_anchor.anchor_once(store, tmp_path / "anchors", ca, anchor=anchor)
    assert result["status"] == ("disabled" if failure == "disabled" else "unavailable")
    manifest = (Path(result["path"]) / "root.json").read_text()
    assert "fake-account-A-token" not in manifest
    assert json.loads(manifest)["heads"] == store.heads()


def test_pins_loaded_only_from_explicit_public_key_file(tmp_path, signing):
    path = tmp_path / "pins.json"
    path.write_text(json.dumps({"host-test": {"ed25519": signing[0].ed25519.hex()}}))
    assert brain.load_pins(path) == {"host-test": signing[0]}
    path.write_text(json.dumps({"host-test": {"ed25519": "00"}}))
    with pytest.raises(ValueError, match="length"):
        brain.load_pins(path)


def test_tailscale_whois_uses_socket_address_without_shell(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, b'{"Node":{"Tags":["tag:brain-test"]}}')

    monkeypatch.setattr(brain.subprocess, "run", run)
    assert brain.tailscale_whois("192.0.2.10") == {"Node": {"Tags": ["tag:brain-test"]}}
    assert calls == [(["tailscale", "whois", "--json", "192.0.2.10"],
                      {"capture_output": True, "timeout": 3, "check": True})]


def test_pinned_post_quantum_key_cannot_silently_degrade(signing, monkeypatch):
    key = infinite_thread.PinnedKey(signing[0].ed25519, b"fake-public-key")
    monkeypatch.setattr(infinite_thread, "mldsa_ready", lambda: False)
    monkeypatch.setattr(infinite_thread, "grip_verifier", lambda *_: ("verified", "ed25519 only"))
    with pytest.raises(ValueError, match="post-quantum"):
        brain.verify_head(head(signing), {"host-test": key})


def memory_handler(store, route, raw, headers=None):
    import io
    h = brain.BrainHandler.__new__(brain.BrainHandler)
    h.path, h.command = route, "POST"
    h.request_version = "HTTP/1.0"
    h.requestline = f"POST {route} HTTP/1.0"
    framing = headers or f"Content-Length: {len(raw)}\r\n"
    h.headers = http.client.parse_headers(io.BytesIO((framing + "\r\n").encode()))
    h.rfile, h.wfile = io.BytesIO(raw), io.BytesIO()
    h.client_address = ("127.0.0.1", 12345)
    h.server = SimpleNamespace(store=store, whois=lambda _: {})
    h.close_connection = True
    return h


@pytest.mark.parametrize("route", ["/v1/cooldowns", "/v1/heads"])
@pytest.mark.parametrize("case", ["list", "string", "number", "null", "missing", "types",
                                   "unknown", "deep", "json", "encoding", "huge-number"])
def test_malformed_updates_leave_store_unchanged(store, signing, route, case):
    store.merge({"account": "existing", "reset": 10, "status": "rejected"})
    store.put_head(head(signing))
    before = (store.cooldowns(), store.heads())
    values = {"list": [], "string": "bad", "number": 4, "null": None, "missing": {},
              "types": dict(head(signing), seq=[]) if route.endswith("heads") else
              {"account": [], "reset": "bad", "status": {}},
              "unknown": dict(head(signing), host="unknown-test")}
    raw = {"deep": b"[" * 2000 + b"0" + b"]" * 2000,
           "json": b"{", "encoding": b"\xff", "huge-number": b"9" * 5000}.get(case)
    if raw is None:
        raw = json.dumps(values[case]).encode()
    h = memory_handler(store, route, raw)
    h.do_POST()
    assert h.wfile.getvalue().startswith(b"HTTP/1.0 400 ")
    assert (store.cooldowns(), store.heads()) == before


@pytest.mark.parametrize("route,method", [("/v1/cooldowns", "merge"), ("/v1/heads", "put_head")])
@pytest.mark.parametrize("error", [KeyError, TypeError, ValueError])
def test_store_bugs_are_logged_500(store, signing, monkeypatch, caplog, route, method, error):
    def broken(_):
        raise error("fake-account-A-token")
    monkeypatch.setattr(store, method, broken)
    body = head(signing) if method == "put_head" else {
        "account": "account-test", "reset": 5, "status": "allowed"}
    with serving(store) as server:
        status, response = request(server, route, body)
    assert status == 500
    assert response == {"error": "internal error"}
    assert f"brain update failed ({error.__name__})" in caplog.text
    assert "fake-account-A-token" not in caplog.text


@pytest.mark.parametrize("case,status", [("access", 403), ("route", 404), ("framing", 400)])
def test_brain_refusals_drain_body(store, monkeypatch, case, status):
    monkeypatch.setattr(brain.gateway_guard, "gateway_access_allowed", lambda *a, **k: case != "access")
    raw = b'{"test":true}'
    headers = f"Content-Length: {len(raw)}\r\nTransfer-Encoding: chunked\r\n" if case == "framing" else None
    h = memory_handler(store, "/missing" if case == "route" else "/v1/heads", raw, headers)
    h.do_POST()
    assert h.wfile.getvalue().startswith(f"HTTP/1.0 {status} ".encode())
    assert h.rfile.tell() == len(raw)


def test_brain_listen_uses_backlog_128(store, monkeypatch):
    calls = []
    original = socket.socket.listen
    def listen(sock, backlog):
        calls.append(backlog)
        return original(sock, backlog)
    monkeypatch.setattr(socket.socket, "listen", listen)
    with brain.BrainServer(("127.0.0.1", 0), store) as server:
        assert server.server_port not in {4010, 4011, 4012, 7416}
        assert calls == [128]
