"""The HAL wire's per-host CA (InfiniteMachine v2, Step 2).

Every test runs with ``HAL_HOME`` at a fresh directory under ``tmp_path`` and the
key forced into the 0600 file (``HAL_WIRE_CA_STORE=file``), so nothing touches
the operator's real Keychain or ``~/.hal``. The Keychain and libsecret adapters
are tested against stand-in ``security`` and ``secret-tool`` executables that
record what they were given.
"""
from __future__ import annotations

import base64
import ipaddress
import json
import os
import shutil
import socket
import ssl
import stat
import subprocess
import sys
import threading
from datetime import timedelta
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from cryptography import x509  # noqa: E402
from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402

from lib.hal import wire_ca  # noqa: E402
from lib.hal.wire_ca import CONSTRAINED_HOST, WireCAError  # noqa: E402

PERMITTED_VIOLATION = 47  # X509_V_ERR_PERMITTED_VIOLATION
EXCLUDED_VIOLATION = 48   # X509_V_ERR_EXCLUDED_VIOLATION


@pytest.fixture
def home(tmp_path, monkeypatch):
    path = tmp_path / "hal-home"
    monkeypatch.setenv("HAL_HOME", str(path))
    monkeypatch.setenv(wire_ca.STORE_ENV, "file")
    return path


# --------------------------------------------------------------------------- #
# A one-shot TLS server and client.
# --------------------------------------------------------------------------- #
def _serve_once(context: ssl.SSLContext) -> int:
    listener = socket.create_server(("127.0.0.1", 0))

    def run():
        conn, _ = listener.accept()
        try:
            with context.wrap_socket(conn, server_side=True) as tls:
                tls.sendall(b"ok")
        except (ssl.SSLError, OSError):
            pass
        finally:
            listener.close()

    threading.Thread(target=run, daemon=True).start()
    return listener.getsockname()[1]


def _client(ca_pem: bytes | None) -> ssl.SSLContext:
    """A client trusting ``ca_pem`` and nothing else (no system store)."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.set_alpn_protocols(["http/1.1"])
    if ca_pem:
        context.load_verify_locations(cadata=ca_pem.decode("ascii"))
    return context


def _handshake(server: ssl.SSLContext, client: ssl.SSLContext, host: str) -> dict:
    port = _serve_once(server)
    with socket.create_connection(("127.0.0.1", port), timeout=5) as raw:
        with client.wrap_socket(raw, server_hostname=host) as tls:
            return {"alpn": tls.selected_alpn_protocol(), "data": tls.recv(2),
                    "peer": tls.getpeercert()}


def _ca_pem(ca) -> bytes:
    return ca.cert.public_bytes(serialization.Encoding.PEM)


# --------------------------------------------------------------------------- #
# The CA itself.
# --------------------------------------------------------------------------- #
def test_init_creates_a_ca_limited_to_one_host(home):
    ca = wire_ca.load_or_create()
    assert wire_ca.ca_cert_path() == home / "state" / "wire-ca" / "hal-wire-ca.pem"
    exts = ca.cert.extensions
    basic = exts.get_extension_for_class(x509.BasicConstraints)
    names = exts.get_extension_for_class(x509.NameConstraints)
    assert basic.critical and basic.value.ca and basic.value.path_length == 0
    assert names.critical
    assert [n.value for n in names.value.permitted_subtrees] == [CONSTRAINED_HOST]
    assert {n.value for n in names.value.excluded_subtrees} == {
        ipaddress.ip_network("0.0.0.0/0"), ipaddress.ip_network("::/0")}
    assert isinstance(ca.cert.public_key().curve, ec.SECP256R1)
    life = ca.cert.not_valid_after_utc - ca.cert.not_valid_before_utc
    assert timedelta(days=365) <= life <= timedelta(days=366)


def test_the_key_file_is_0600_inside_a_0700_directory(home):
    wire_ca.load_or_create()
    assert stat.S_IMODE(os.stat(wire_ca.ca_key_path()).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(wire_ca.ca_dir()).st_mode) == 0o700


def test_load_or_create_never_replaces_an_existing_ca(home):
    first = wire_ca.load_or_create().fingerprint
    assert wire_ca.load_or_create().fingerprint == first
    assert wire_ca.load().fingerprint == first


def test_a_certificate_without_its_key_fails_closed(home):
    first = wire_ca.load_or_create().fingerprint
    wire_ca.ca_key_path().unlink()
    with pytest.raises(WireCAError, match="private key"):
        wire_ca.load_or_create()
    assert wire_ca.fingerprint(x509.load_pem_x509_certificate(
        wire_ca.ca_cert_path().read_bytes())) == first


def test_a_certificate_without_the_constraint_is_refused(home):
    wire_ca.load_or_create()
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(x509.oid.NameOID.COMMON_NAME, "open CA")])
    now = wire_ca._utcnow()
    open_ca = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
               .public_key(key.public_key()).serial_number(1)
               .not_valid_before(now).not_valid_after(now + timedelta(days=1))
               .add_extension(x509.BasicConstraints(ca=True, path_length=0), True)
               .sign(key, hashes.SHA256()))
    wire_ca.ca_cert_path().write_bytes(open_ca.public_bytes(serialization.Encoding.PEM))
    wire_ca.ca_key_path().write_bytes(wire_ca._key_pem(key))
    with pytest.raises(WireCAError, match="name constraint"):
        wire_ca.load()


def test_an_expired_ca_is_refused(home, monkeypatch):
    real_now = wire_ca._utcnow()
    monkeypatch.setattr(wire_ca, "_utcnow", lambda: real_now - timedelta(days=400))
    wire_ca.load_or_create()
    monkeypatch.setattr(wire_ca, "_utcnow", lambda: real_now)
    with pytest.raises(WireCAError, match="expired"):
        wire_ca.load()


def test_an_unknown_store_setting_is_an_error(home, monkeypatch):
    monkeypatch.setenv(wire_ca.STORE_ENV, "keychain")
    with pytest.raises(WireCAError, match=wire_ca.STORE_ENV):
        wire_ca.load_or_create()


# --------------------------------------------------------------------------- #
# Leaves.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("host", [
    "example.com", "api.anthropic.com.example.com", "evil-api.anthropic.com",
    "sub.api.anthropic.com", "anthropic.com", "", "127.0.0.1",
])
def test_a_leaf_for_any_other_host_cannot_be_minted(home, host):
    ca = wire_ca.load_or_create()
    with pytest.raises(WireCAError, match="refusing to mint"):
        ca.mint_leaf(host)


def test_the_constrained_host_is_matched_case_and_dot_insensitively(home):
    leaf = wire_ca.load_or_create().mint_leaf("API.Anthropic.com.")
    assert leaf.hostname == CONSTRAINED_HOST


def test_a_leaf_verifies_with_the_ca_as_the_only_trust_anchor(home):
    ca = wire_ca.load_or_create()
    leaf = ca.mint_leaf(CONSTRAINED_HOST)
    result = _handshake(wire_ca.server_context(leaf), _client(_ca_pem(ca)),
                        CONSTRAINED_HOST)
    assert result["data"] == b"ok" and result["alpn"] == "http/1.1"
    assert ("DNS", CONSTRAINED_HOST) in result["peer"]["subjectAltName"]


def test_the_same_leaf_fails_without_the_ca(home):
    leaf = wire_ca.load_or_create().mint_leaf(CONSTRAINED_HOST)
    with pytest.raises(ssl.SSLCertVerificationError):
        _handshake(wire_ca.server_context(leaf), _client(None), CONSTRAINED_HOST)


def test_the_name_constraint_holds_even_against_the_cas_own_key(home):
    """A leaf for another host, signed with the CA key, still fails verification."""
    ca = wire_ca.load_or_create()
    forged = wire_ca._sign_leaf(ca, "example.com")
    with pytest.raises(ssl.SSLCertVerificationError) as caught:
        _handshake(wire_ca.server_context(forged), _client(_ca_pem(ca)), "example.com")
    assert caught.value.verify_code == PERMITTED_VIOLATION


def _forge_leaf(ca, names: list) -> "wire_ca.Leaf":
    """A leaf naming ``names``, signed with the CA's own key (no mint_leaf check)."""
    key = ec.generate_private_key(ec.SECP256R1())
    now = wire_ca._utcnow()
    cert = (x509.CertificateBuilder()
            .subject_name(wire_ca._name(CONSTRAINED_HOST)).issuer_name(ca.cert.subject)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(now + timedelta(hours=1))
            .add_extension(x509.SubjectAlternativeName(names), False)
            .sign(ca.key, hashes.SHA256()))
    return wire_ca.Leaf(hostname=CONSTRAINED_HOST, cert_pem=wire_ca._cert_pem(cert),
                        key_pem=wire_ca._key_pem(key), not_after=now + timedelta(hours=1))


@pytest.mark.parametrize("host", [CONSTRAINED_HOST, "127.0.0.1"])
def test_an_ip_name_fails_even_beside_the_permitted_dns_name(home, host):
    """AGORA's falsifier on HAL#643: a leaf forged with the CA's own key that names
    api.anthropic.com AND 127.0.0.1 is rejected, because the CA excludes every IP."""
    ca = wire_ca.load_or_create()
    forged = _forge_leaf(ca, [x509.DNSName(CONSTRAINED_HOST),
                              x509.IPAddress(ipaddress.ip_address("127.0.0.1"))])
    with pytest.raises(ssl.SSLCertVerificationError) as caught:
        _handshake(wire_ca.server_context(forged), _client(_ca_pem(ca)), host)
    assert caught.value.verify_code == EXCLUDED_VIOLATION


def test_a_leaf_is_short_lived(home):
    leaf = wire_ca.load_or_create().mint_leaf()
    now = wire_ca._utcnow()
    assert leaf.expires_within(25 * 3600, now)
    assert not leaf.expires_within(23 * 3600, now)


def test_minting_writes_nothing_to_disk(home):
    ca = wire_ca.load_or_create()
    before = sorted(p.name for p in home.rglob("*"))
    wire_ca.server_context(ca.mint_leaf())
    assert sorted(p.name for p in home.rglob("*")) == before


# --------------------------------------------------------------------------- #
# Rotation and revocation.
# --------------------------------------------------------------------------- #
def test_rotation_records_the_old_fingerprint_and_refuses_it_after(home):
    first = wire_ca.load_or_create()
    old_pem = wire_ca.ca_cert_path().read_bytes()
    old_key = wire_ca.ca_key_path().read_bytes()
    second = wire_ca.rotate()
    assert second.fingerprint != first.fingerprint
    rows = json.loads(wire_ca.revoked_path().read_text())["revoked"]
    assert [row["fingerprint_sha256"] for row in rows] == [first.fingerprint]
    assert rows[0]["reason"] == "rotated"
    assert wire_ca.load().fingerprint == second.fingerprint
    wire_ca.ca_cert_path().write_bytes(old_pem)       # an old CA restored ...
    wire_ca.ca_key_path().write_bytes(old_key)
    with pytest.raises(WireCAError, match="revoked"):  # ... is refused
        wire_ca.load()


def test_rotating_twice_keeps_every_revocation(home):
    fingerprints = [wire_ca.load_or_create().fingerprint]
    fingerprints.append(wire_ca.rotate().fingerprint)
    wire_ca.rotate()
    assert sorted(wire_ca.revoked_fingerprints()) == sorted(fingerprints)


def test_a_corrupt_revocation_list_fails_closed(home):
    wire_ca.load_or_create()
    wire_ca.revoked_path().write_text("{not json")
    with pytest.raises(WireCAError, match="revocation list"):
        wire_ca.load()
    with pytest.raises(WireCAError, match="revocation list"):
        wire_ca.rotate()


# --------------------------------------------------------------------------- #
# Key custody: the key is in its one file and nowhere else.
# --------------------------------------------------------------------------- #
def _key_forms(key) -> list:
    der = key.private_bytes(serialization.Encoding.DER,
                            serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption())
    pem = wire_ca._key_pem(key)
    body = b"".join(pem.splitlines()[1:-1])
    scalar = key.private_numbers().private_value.to_bytes(32, "big")
    return [pem, body, base64.b64encode(pem), der, der.hex().encode(), scalar,
            scalar.hex().encode()]


def _files_holding(root: Path, forms: list) -> set:
    return {path.name for path in root.rglob("*") if path.is_file()
            and any(form in path.read_bytes() for form in forms)}


def test_the_key_appears_only_in_the_key_file_and_never_in_output(home, capsys):
    assert wire_ca.main(["init"]) == 0
    old_forms = _key_forms(wire_ca.load().key)
    assert wire_ca.main(["rotate"]) == 0
    assert wire_ca.main(["show"]) == 0
    new_ca = wire_ca.load()
    new_forms = _key_forms(new_ca.key)
    printed = capsys.readouterr()
    for form in old_forms + new_forms:
        assert form not in (printed.out + printed.err).encode()
    assert _files_holding(home, new_forms) == {wire_ca.CA_KEY_NAME}
    assert _files_holding(home, old_forms) == set()
    for text in (repr(new_ca), repr(new_ca.mint_leaf())):
        assert "PRIVATE KEY" not in text and "key=" not in text.replace("key_", "")


def test_show_reports_paths_and_fingerprints(home, capsys):
    assert wire_ca.main(["init"]) == 0
    capsys.readouterr()
    assert wire_ca.main(["show"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] and report["ca_cert"] == str(wire_ca.ca_cert_path())
    assert report["fingerprint_sha256"] == wire_ca.load().fingerprint
    assert report["key_custody"] == "file" and report["name_constraint"] == CONSTRAINED_HOST


def test_show_without_a_ca_says_so(home, capsys):
    assert wire_ca.main(["show"]) == 1
    assert "init" in json.loads(capsys.readouterr().out)["error"]


# --------------------------------------------------------------------------- #
# The OS secret-store adapters, against recording stand-ins.
# --------------------------------------------------------------------------- #
_STAND_IN = r'''#!{python}
import json, os, sys
log = os.environ["STORE_LOG"]
data = sys.stdin.buffer.read() if not sys.stdin.isatty() else b""
with open(log, "a") as handle:
    handle.write(json.dumps({{"argv": sys.argv[1:], "stdin": data.decode()}}) + "\n")
if os.environ.get("STORE_FAIL"):
    sys.exit(1)
vault = log + ".vault"
words = data.decode().split()
if "-w" in words:                              # security -i add-generic-password
    open(vault, "w").write(words[words.index("-w") + 1])
elif sys.argv[1:2] == ["store"]:               # secret-tool store
    open(vault, "w").write(data.decode())
elif sys.argv[1:2] in (["find-generic-password"], ["lookup"]):
    if not os.path.exists(vault):
        sys.exit(44)
    sys.stdout.write(open(vault).read())
'''


@pytest.fixture
def stand_in(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("security", "secret-tool"):
        tool = bin_dir / name
        tool.write_text(_STAND_IN.format(python=sys.executable))
        tool.chmod(0o755)
    log = tmp_path / "store.log"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("STORE_LOG", str(log))
    return log


def _calls(log: Path) -> list:
    return [json.loads(line) for line in log.read_text().splitlines()]


@pytest.mark.parametrize("store_class", [wire_ca._KeychainStore, wire_ca._SecretToolStore])
def test_the_secret_goes_to_the_store_on_stdin_never_on_the_command_line(
        stand_in, store_class):
    secret = wire_ca._key_pem(ec.generate_private_key(ec.SECP256R1()))
    store = store_class("hal-home-0123456789abcdef")
    store.put(secret)
    assert store.get() == secret
    token = base64.b64encode(secret).decode()
    calls = _calls(stand_in)
    assert any(token in call["stdin"] for call in calls)
    assert not any(token in " ".join(call["argv"]) for call in calls)


def test_a_failing_store_falls_back_to_the_file_with_a_warning(
        home, stand_in, monkeypatch, caplog):
    monkeypatch.setenv(wire_ca.STORE_ENV, "auto")
    monkeypatch.setenv("STORE_FAIL", "1")
    monkeypatch.setattr(wire_ca, "_preferred_store",
                        lambda: wire_ca._KeychainStore("hal-home-0123456789abcdef"))
    ca = wire_ca.load_or_create()
    assert ca.custody == "file"
    assert "0600 file" in caplog.text and "did not keep the key" in caplog.text
    assert wire_ca.load().fingerprint == ca.fingerprint


def test_a_working_store_leaves_no_key_file(home, stand_in, monkeypatch):
    monkeypatch.setenv(wire_ca.STORE_ENV, "auto")
    monkeypatch.setattr(wire_ca, "_preferred_store",
                        lambda: wire_ca._KeychainStore("hal-home-0123456789abcdef"))
    ca = wire_ca.load_or_create()
    assert ca.custody == "macos-keychain"
    assert not wire_ca.ca_key_path().exists()
    assert wire_ca.load().fingerprint == ca.fingerprint


def test_the_store_account_does_not_spell_out_the_home_path(home):
    account = wire_ca._account()
    assert account.startswith("hal-home-") and str(home) not in account
    assert len(account) == len("hal-home-") + 16


# --------------------------------------------------------------------------- #
# The same checks in a JavaScript runtime that reads NODE_EXTRA_CA_CERTS.
# --------------------------------------------------------------------------- #
_NODE_CLIENT = r"""
const tls = require('tls');
const [port, servername] = process.argv.slice(1);
const say = (verdict) => { console.log(JSON.stringify(verdict)); process.exit(0); };
const s = tls.connect({host: '127.0.0.1', port: Number(port), servername},
                      () => say({ok: s.authorized}));
s.on('error', (e) => say({ok: false, code: e.code}));
"""


def _node(port: int, servername: str, ca_path: Path | None) -> dict:
    """Node's first verdict on one handshake (it exits as soon as it has one)."""
    env = {key: value for key, value in os.environ.items()
           if key != "NODE_EXTRA_CA_CERTS"}
    if ca_path:
        env["NODE_EXTRA_CA_CERTS"] = str(ca_path)
    done = subprocess.run(["node", "-e", _NODE_CLIENT, str(port), servername],
                          capture_output=True, text=True, timeout=20, env=env)
    lines = done.stdout.strip().splitlines()
    assert lines, f"node gave no verdict (exit {done.returncode}): {done.stderr[-400:]}"
    return json.loads(lines[0])


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_node_trusts_the_ca_for_the_constrained_host_only(home):
    """Claude Code runs on Bun, which this suite cannot drive; Node reads the same
    NODE_EXTRA_CA_CERTS. The live check is GRIP's ``grip-wire-probe --arm intercept``."""
    ca = wire_ca.load_or_create()
    good = wire_ca.server_context(ca.mint_leaf())
    assert _node(_serve_once(good), CONSTRAINED_HOST, wire_ca.ca_cert_path())["ok"]
    good = wire_ca.server_context(ca.mint_leaf())
    assert not _node(_serve_once(good), CONSTRAINED_HOST, None)["ok"]
    forged = wire_ca.server_context(wire_ca._sign_leaf(ca, "example.com"))
    assert not _node(_serve_once(forged), "example.com", wire_ca.ca_cert_path())["ok"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_node_refuses_an_ip_name_beside_the_permitted_one(home):
    """AGORA's falsifier on HAL#644: the IP exclusion holds under NODE_EXTRA_CA_CERTS."""
    ca = wire_ca.load_or_create()
    forged = wire_ca.server_context(_forge_leaf(
        ca, [x509.DNSName(CONSTRAINED_HOST), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]))
    verdict = _node(_serve_once(forged), CONSTRAINED_HOST, wire_ca.ca_cert_path())
    assert not verdict["ok"], verdict
