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


@pytest.mark.skipif(sys.platform == "win32",
                    reason="Windows ACLs, not POSIX mode bits, control file access")
def test_the_key_file_is_0600_inside_a_0700_directory(home):
    wire_ca.load_or_create()
    assert stat.S_IMODE(os.stat(wire_ca.ca_key_path()).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(wire_ca.ca_dir()).st_mode) == 0o700


def test_the_atomic_write_works_where_os_has_no_fchmod(tmp_path, monkeypatch):
    # Windows has os.fchmod only from Python 3.13; the public check runs 3.12.
    monkeypatch.delattr(wire_ca.os, "fchmod", raising=False)
    target = tmp_path / "key.pem"
    wire_ca._atomic_write(target, b"secret", 0o600)
    assert target.read_bytes() == b"secret"
    assert not list(tmp_path.glob(".key.pem.*.tmp"))


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


@pytest.mark.skipif(os.name != "nt", reason="real Windows named pipes exist only on Windows")
def test_on_windows_a_real_handshake_loads_its_leaf_through_real_named_pipes(home, monkeypatch):
    # No stand-in: the real Win32 pipes and a real ssl.SSLContext, observed only.
    calls = []
    real_loader = wire_ca._load_chain_from_windows_pipes

    def observed(context, cert_pem, key_pem):
        calls.append(type(context))
        return real_loader(context, cert_pem, key_pem)

    monkeypatch.setattr(wire_ca, "_load_chain_from_windows_pipes", observed)
    ca = wire_ca.load_or_create()
    leaf = ca.mint_leaf(CONSTRAINED_HOST)
    result = _handshake(wire_ca.server_context(leaf), _client(_ca_pem(ca)),
                        CONSTRAINED_HOST)
    assert calls == [ssl.SSLContext]
    assert result["data"] == b"ok"
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


class _StandInWinAPI:
    def __init__(self, written=None):
        self.create_args = None
        self.closed = []
        self.written = written
        self.parts = []
        self.created = 0

    def CreateNamedPipe(self, *args):
        self.create_args = args
        self.created += 1
        return self.created

    def ConnectNamedPipe(self, handle, overlapped):
        return None

    def WriteFile(self, handle, data, overlapped):
        count = len(data) if self.written is None else min(self.written, len(data))
        self.parts.append(data[:count])
        return count, 0

    def CloseHandle(self, handle):
        self.closed.append(handle)


@pytest.fixture
def native_pipe_api(monkeypatch):
    """Exercise the production ctypes adapter with recording DLL entry points."""
    import ctypes as c
    from ctypes import wintypes as w
    from types import SimpleNamespace

    records = SimpleNamespace(sddl=None, security=None, freed=[], created=0,
                              protected=[], fail_descriptor=False,
                              null_descriptor=False, fail_create=False)
    descriptor = c.create_string_buffer(b"restricted descriptor")

    def convert(sddl, revision, output, size):
        records.sddl = sddl
        assert revision == 1
        if records.fail_descriptor:
            raise OSError("SDDL conversion denied")
        if records.null_descriptor:
            return 1
        c.cast(output, c.POINTER(c.c_void_p))[0] = c.addressof(descriptor)
        return 1

    def create(*args):
        records.created += 1
        assert args[-1], "default security attributes must never reach CreateNamedPipeW"
        attributes = c.cast(args[-1], c.POINTER(api.SecurityAttributes)).contents
        assert attributes.nLength == c.sizeof(api.SecurityAttributes)
        assert attributes.lpSecurityDescriptor == c.addressof(descriptor)
        assert not attributes.bInheritHandle
        records.security = records.sddl
        if records.fail_create:
            raise OSError("pipe creation failed")
        return 42

    def set_file_security(path, information, security):
        records.protected.append((path, information, records.sddl))
        return 1

    class Function:
        def __init__(self, fn):
            self.fn = fn

        def __call__(self, *args):
            return self.fn(*args)

    class DLL:
        def __getattr__(self, name):
            return Function({
                "ConvertStringSecurityDescriptorToSecurityDescriptorW": convert,
                "CreateNamedPipeW": create,
                "SetFileSecurityW": set_file_security,
                "LocalFree": lambda ptr: records.freed.append(ptr.value),
            }.get(name, lambda *args: 1))

    monkeypatch.setattr(c, "WinDLL", lambda *a, **kw: DLL(), raising=False)
    api = wire_ca._WindowsPipeAPI()
    monkeypatch.setattr(api, "_read_user_sid",
                        lambda: "S-1-5-21-123-456-789-1001")
    return api, records


def test_the_windows_pipe_receives_only_user_and_system_security(native_pipe_api):
    api, records = native_pipe_api
    server = wire_ca._WindowsPEMPipe(api, b"pem")
    try:
        sid = "S-1-5-21-123-456-789-1001"
        assert records.security == f"O:{sid}D:P(A;;FA;;;SY)(A;;FA;;;{sid})"
        assert records.created == 1
        assert len(records.freed) == 1
    finally:
        server.finish()


def test_windows_descriptor_failure_never_creates_a_default_pipe(native_pipe_api):
    api, records = native_pipe_api
    records.fail_descriptor = True
    with pytest.raises(WireCAError, match="cannot restrict.*SDDL conversion denied"):
        wire_ca._WindowsPEMPipe(api, b"pem")
    assert records.created == 0


def test_windows_null_descriptor_fails_closed(native_pipe_api):
    api, records = native_pipe_api
    records.null_descriptor = True
    with pytest.raises(WireCAError, match="NULL descriptor"):
        wire_ca._WindowsPEMPipe(api, b"pem")
    assert records.created == 0


def test_windows_descriptor_is_freed_if_creation_fails(native_pipe_api):
    api, records = native_pipe_api
    records.fail_create = True
    with pytest.raises(OSError, match="pipe creation failed"):
        wire_ca._WindowsPEMPipe(api, b"pem")
    assert len(records.freed) == 1


def test_windows_user_sid_is_cached_on_the_api(native_pipe_api, monkeypatch):
    api, _ = native_pipe_api
    reads = []
    monkeypatch.setattr(api, "_read_user_sid", lambda: reads.append(1) or "S-1-5-21-1")
    assert api._user_sid() == api._user_sid() == "S-1-5-21-1"
    assert reads == [1]


def test_windows_folder_and_key_receive_owner_only_security(home, monkeypatch):
    events = []
    sid = "S-1-5-21-123-456-789-1001"
    sddl = f"O:{sid}D:P(A;OICI;FA;;;SY)(A;OICI;FA;;;{sid})"

    class API:
        def ProtectPath(self, path):
            events.append(("protect", Path(path), sddl))

    real_atomic_write = wire_ca._atomic_write

    def record_write(path, data, mode):
        events.append(("write", Path(path)))
        real_atomic_write(path, data, mode)

    monkeypatch.setattr(wire_ca, "_is_windows", lambda: True)
    monkeypatch.setattr(wire_ca, "_WindowsPipeAPI", API)
    monkeypatch.setattr(wire_ca, "_atomic_write", record_write)
    wire_ca._FileStore(wire_ca.ca_key_path()).put(b"private")
    assert events == [
        ("protect", wire_ca.ca_dir(), sddl),
        ("write", wire_ca.ca_key_path()),
        ("protect", wire_ca.ca_key_path(), sddl),
    ]
    assert wire_ca.ca_key_path().read_bytes() == b"private"


def test_windows_folder_security_failure_refuses_key_write(home, monkeypatch):
    class API:
        def ProtectPath(self, path):
            raise OSError("ACL denied")

    monkeypatch.setattr(wire_ca, "_is_windows", lambda: True)
    monkeypatch.setattr(wire_ca, "_WindowsPipeAPI", API)
    with pytest.raises(WireCAError, match="cannot restrict.*ACL denied"):
        wire_ca._FileStore(wire_ca.ca_key_path()).put(b"private")
    assert not wire_ca.ca_key_path().exists()


def test_windows_key_security_failure_removes_written_key(home, monkeypatch):
    calls = []

    def protect(path, purpose):
        calls.append(purpose)
        if purpose == "wire CA private key":
            raise WireCAError("cannot restrict Windows key security: ACL denied")

    monkeypatch.setattr(wire_ca, "_is_windows", lambda: True)
    monkeypatch.setattr(wire_ca, "_protect_windows_path", protect)
    with pytest.raises(WireCAError, match="ACL denied"):
        wire_ca._FileStore(wire_ca.ca_key_path()).put(b"private")
    assert calls == ["wire CA folder", "wire CA private key"]
    assert not wire_ca.ca_key_path().exists()


@pytest.mark.skipif(sys.platform != "win32", reason="requires real Windows ACL APIs")
def test_windows_key_dacl_contains_only_system_and_current_user(home):
    import ctypes as c
    from ctypes import wintypes as w

    wire_ca.load_or_create()
    api = wire_ca._WindowsPipeAPI()
    advapi = c.WinDLL("advapi32", use_last_error=True)
    kernel = c.WinDLL("kernel32", use_last_error=True)
    get_security = advapi.GetFileSecurityW
    get_security.argtypes = [w.LPCWSTR, w.DWORD, c.c_void_p, w.DWORD,
                             c.POINTER(w.DWORD)]
    get_security.restype = w.BOOL
    to_string = advapi.ConvertSecurityDescriptorToStringSecurityDescriptorW
    to_string.argtypes = [c.c_void_p, w.DWORD, w.DWORD, c.POINTER(w.LPWSTR),
                          c.POINTER(w.DWORD)]
    to_string.restype = w.BOOL
    needed = w.DWORD()
    information = (wire_ca._OWNER_SECURITY_INFORMATION
                   | wire_ca._DACL_SECURITY_INFORMATION)
    get_security(str(wire_ca.ca_key_path()), information, None, 0, c.byref(needed))
    descriptor = c.create_string_buffer(needed.value)
    api._check(get_security(str(wire_ca.ca_key_path()), information, descriptor,
                            needed, c.byref(needed)))
    text = w.LPWSTR()
    api._check(to_string(descriptor, 1, information, c.byref(text), None))
    try:
        sddl = text.value
    finally:
        kernel.LocalFree(c.cast(text, c.c_void_p))
    sid = api._user_sid()
    assert sddl.startswith(f"O:{sid}D:P")
    assert sddl.count("(A;") == 2
    assert ";;;SY)" in sddl and f";;;{sid})" in sddl


def test_a_short_windows_pipe_write_sends_the_remaining_bytes():
    api = _StandInWinAPI(written=2)
    server = wire_ca._WindowsPEMPipe(api, b"pem")
    server.start()
    server.finish()
    assert api.parts == [b"pe", b"m"]
    assert api.closed == [1]


def test_a_stalled_windows_pipe_write_times_out(monkeypatch):
    api = _StandInWinAPI(written=0)
    monkeypatch.setattr(wire_ca, "_PIPE_TIMEOUT_S", 0.02)
    server = wire_ca._WindowsPEMPipe(api, b"pem")
    server.start()
    with pytest.raises(WireCAError, match="timed out while writing"):
        server.finish()
    assert api.closed == [1]


@pytest.mark.parametrize("oversized", [False, True])
def test_first_windows_pipe_is_closed_when_second_creation_fails(monkeypatch, oversized):
    api = _StandInWinAPI()
    original = api.CreateNamedPipe

    def create(*args):
        if api.created:
            raise OSError("second pipe failed")
        return original(*args)

    api.CreateNamedPipe = create
    monkeypatch.setattr(wire_ca, "_WindowsPipeAPI", lambda: api)
    with pytest.raises((WireCAError, OSError), match="larger|second pipe failed"):
        wire_ca._load_chain_from_windows_pipes(
            object(), b"cert", b"x" * (wire_ca._PIPE_LIMIT + 1) if oversized else b"key")
    assert api.closed == [1]


def test_non_windows_without_dev_fd_fails_clearly(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(wire_ca, "os", SimpleNamespace(
        name="posix", path=SimpleNamespace(isdir=lambda path: False)))
    with pytest.raises(WireCAError, match="needs /dev/fd on POSIX"):
        wire_ca._load_chain_in_memory(object(), b"cert", b"key")


def test_windows_dispatch_does_not_depend_on_dev_fd(monkeypatch):
    from types import SimpleNamespace
    calls = []
    monkeypatch.setattr(wire_ca, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(wire_ca, "_load_chain_from_windows_pipes",
                        lambda *args: calls.append(args))
    context = object()
    wire_ca._load_chain_in_memory(context, b"cert", b"key")
    assert calls == [(context, b"cert", b"key")]


def test_every_windows_pipe_is_finished_after_an_earlier_finish_error(monkeypatch):
    finished = []

    class StandInServer:
        def __init__(self, winapi, data):
            self.path = data.decode()

        def start(self):
            pass

        def finish(self):
            finished.append(self.path)
            raise WireCAError(f"finish {self.path}")

    class StandInContext:
        def load_cert_chain(self, cert, key):
            pass

    monkeypatch.setattr(wire_ca, "_WindowsPipeAPI", lambda: object())
    monkeypatch.setattr(wire_ca, "_WindowsPEMPipe", StandInServer)
    with pytest.raises(WireCAError, match="finish cert"):
        wire_ca._load_chain_from_windows_pipes(StandInContext(), b"cert", b"key")
    assert finished == ["cert", "key"]


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


@pytest.mark.parametrize("store_class", [
    pytest.param(wire_ca._KeychainStore,
                 marks=pytest.mark.skipif(os.name == "nt",
                                          reason="the stand-in executable requires POSIX")),
    pytest.param(wire_ca._SecretToolStore,
                 marks=pytest.mark.skipif(os.name == "nt",
                                          reason="the stand-in executable requires POSIX")),
])
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
    protection = "Windows owner-only protected DACL" if os.name == "nt" else "0600 file"
    assert protection in caplog.text and "did not keep the key" in caplog.text
    assert wire_ca.load().fingerprint == ca.fingerprint


@pytest.mark.skipif(os.name == "nt",
                    reason="the stand-in executable requires POSIX")
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
const about = {node: process.version, openssl: process.versions.openssl};
const say = (verdict) => { console.log(JSON.stringify({...verdict, ...about})); process.exit(0); };
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
    # Each assertion shows Node's verdict, its refusal code and its Node and OpenSSL
    # versions, so a failure on a CI runner says why from its one-line summary.
    good = wire_ca.server_context(ca.mint_leaf())
    verdict = _node(_serve_once(good), CONSTRAINED_HOST, wire_ca.ca_cert_path())
    assert verdict["ok"], verdict
    good = wire_ca.server_context(ca.mint_leaf())
    verdict = _node(_serve_once(good), CONSTRAINED_HOST, None)
    assert not verdict["ok"], verdict
    forged = wire_ca.server_context(wire_ca._sign_leaf(ca, "example.com"))
    verdict = _node(_serve_once(forged), "example.com", wire_ca.ca_cert_path())
    assert not verdict["ok"], verdict


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_node_refuses_an_ip_name_beside_the_permitted_one(home):
    """AGORA's falsifier on HAL#644: the IP exclusion holds under NODE_EXTRA_CA_CERTS."""
    ca = wire_ca.load_or_create()
    forged = wire_ca.server_context(_forge_leaf(
        ca, [x509.DNSName(CONSTRAINED_HOST), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]))
    verdict = _node(_serve_once(forged), CONSTRAINED_HOST, wire_ca.ca_cert_path())
    assert not verdict["ok"], verdict
