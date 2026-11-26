"""wire_ca -- the HAL wire's per-host certificate authority.

InfiniteMachine v2, Step 2. Operator decision 1 (``docs/infinite-machine-contract.md``):
Claude Code is wired by ``HTTPS_PROXY`` "plus a per-host CA name-constrained to
``api.anthropic.com``", so HAL can read the inference calls while Claude Code keeps
``api.anthropic.com`` as its address (which Remote Control requires). This module
is that CA. The proxy itself is ``lib/hal/wire_proxy.py``.

The CA, as proven by the 2026-10-04 intercept probe (GRIP ``lib/wire_probe.py``):
ECDSA P-256; ``BasicConstraints(ca, path_length=0)``; ``NameConstraints``
permitting only the DNS name ``api.anthropic.com`` and excluding every IPv4 and
IPv6 address (a DNS-only constraint says nothing about IP names, so a leaf naming
an address would otherwise pass); valid one year. Even with the key, it cannot
vouch for any other host in a client that checks name constraints.

Files, under ``$HAL_HOME/state/wire-ca/`` (``HAL_HOME`` defaults to ``~/.hal``):

  hal-wire-ca.pem   the CA's public certificate. GRIP's launcher points
                    ``NODE_EXTRA_CA_CERTS`` at it. Public.
  revoked.json      the fingerprints of CAs that ``rotate`` replaced. A revoked CA
                    is refused at load, so a restored old certificate cannot
                    come back into use.
  hal-wire-ca.key   ONLY when no OS secret store is usable: the private key, mode
                    0600, with a logged warning.

Key custody: the macOS Keychain through ``security`` (service ``hal-wire-ca``),
Linux libsecret through ``secret-tool`` when it is installed, otherwise the 0600
file. The account is a digest of the resolved ``HAL_HOME``, so two HAL homes on
one host never share a key and the username is not written into the store. The
secret reaches ``security`` and ``secret-tool`` on stdin, never on the command
line, which any local process can read through ``ps``. Their error output is
never logged. ``HAL_WIRE_CA_STORE=file`` forces the file (tests, and hosts with no
usable store); ``auto`` is the default.

Leaves: minted in memory, for ``api.anthropic.com`` only (stricter than the
constraint, which RFC 5280 extends to subdomains), valid 24 hours, never
written to disk. ``server_context`` loads one into an ``ssl.SSLContext`` through
two anonymous pipes opened as ``/dev/fd/N``: the standard library loads a
certificate chain only from a path, and a pipe's bytes never reach a filesystem.

Failure is closed. A certificate that exists while its key cannot be read, a
revoked or expired CA, or a certificate that lost its constraint raises
``WireCAError``; ``load_or_create`` never replaces an existing CA by itself,
because new sessions would then trust a different CA than running ones. With
no wire, GRIP's launcher sends the session direct with its "unwitnessed" badge.

CLI: ``python -m lib.hal.wire_ca {init,show,rotate}`` prints paths, fingerprints,
validity and custody as JSON. It never prints key material.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import logging
import os
import shutil
import ssl
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

CONSTRAINED_HOST = "api.anthropic.com"
SERVICE = "hal-wire-ca"
CA_CERT_NAME = "hal-wire-ca.pem"
CA_KEY_NAME = "hal-wire-ca.key"
REVOKED_NAME = "revoked.json"
STORE_ENV = "HAL_WIRE_CA_STORE"
STORE_CHOICES = ("auto", "file")

CA_LIFETIME = timedelta(days=365)
LEAF_LIFETIME = timedelta(hours=24)
CLOCK_SKEW = timedelta(minutes=5)

_PIPE_LIMIT = 4096        # a P-256 leaf PEM is ~0.7 KB; every pipe buffer holds 4 KB
_STORE_TIMEOUT_S = 15     # a store that prompts or hangs counts as unusable
_CA_COMMON_NAME = "HAL wire CA (api.anthropic.com only)"

_log = logging.getLogger("hal.wire_ca")


class WireCAError(RuntimeError):
    """The wire CA cannot be used. The message names the fix; it holds no secret."""


class _StoreUnavailable(Exception):
    """An OS secret store could not hold or return the key (fixed text only)."""


# --------------------------------------------------------------------------- #
# Paths.
# --------------------------------------------------------------------------- #
def hal_home() -> Path:
    """``$HAL_HOME``, else ``~/.hal`` (resolved at call time, never at import)."""
    return Path(os.environ.get("HAL_HOME") or (Path.home() / ".hal")).expanduser()


def ca_dir() -> Path:
    return hal_home() / "state" / "wire-ca"


def ca_cert_path() -> Path:
    """The public CA certificate GRIP's launcher hands to ``NODE_EXTRA_CA_CERTS``."""
    return ca_dir() / CA_CERT_NAME


def ca_key_path() -> Path:
    """The 0600 fallback key file (used only when no OS secret store is usable)."""
    return ca_dir() / CA_KEY_NAME


def revoked_path() -> Path:
    return ca_dir() / REVOKED_NAME


def _ensure_dir() -> Path:
    directory = ca_dir()
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    return directory


def _atomic_write(path: Path, data: bytes, mode: int) -> None:
    """Write ``data`` to ``path`` with ``mode`` (umask-independent), atomically."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        os.fchmod(fd, mode)
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


@contextlib.contextmanager
def _locked():
    """Serialise create and rotate across processes (the launcher and the proxy)."""
    fd = os.open(_ensure_dir() / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        _flock(fd)
        yield
    finally:
        os.close(fd)


def _flock(fd: int) -> None:
    try:
        import fcntl
    except ImportError:  # no advisory locks (Windows): one writer is assumed
        return
    fcntl.flock(fd, fcntl.LOCK_EX)


# --------------------------------------------------------------------------- #
# Key custody: OS secret store, else a 0600 file.
# --------------------------------------------------------------------------- #
def _run(argv: list, stdin: Optional[bytes] = None) -> Optional[bytes]:
    """A store command's stdout, or None on any failure. Its stderr is discarded."""
    try:
        done = subprocess.run(argv, input=stdin, capture_output=True,
                              timeout=_STORE_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def _decode_token(out: Optional[bytes]) -> Optional[bytes]:
    if not out:
        return None
    try:
        return base64.b64decode(out.strip(), validate=True)
    except ValueError:
        return None


class _FileStore:
    kind = "file"

    def __init__(self, path: Path):
        self.path = path

    def get(self) -> Optional[bytes]:
        try:
            return self.path.read_bytes()
        except OSError:
            return None

    def put(self, key_pem: bytes) -> None:
        _atomic_write(self.path, key_pem, 0o600)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)

    def where(self) -> str:
        return f"file {self.path} (mode 0600)"


class _KeychainStore:
    """macOS Keychain via ``security``; the secret goes in on stdin (``security -i``)."""

    kind = "macos-keychain"

    def __init__(self, account: str):
        self.account = account

    def get(self) -> Optional[bytes]:
        return _decode_token(_run(["security", "find-generic-password",
                                   "-a", self.account, "-s", SERVICE, "-w"]))

    def put(self, key_pem: bytes) -> None:
        token = base64.b64encode(key_pem).decode("ascii")
        line = (f'add-generic-password -U -a {self.account} -s {SERVICE} '
                f'-l "HAL wire CA key" -w {token}\n')
        _run(["security", "-i"], stdin=line.encode("ascii"))
        if self.get() != key_pem:
            raise _StoreUnavailable("the macOS Keychain did not keep the key")

    def clear(self) -> None:
        _run(["security", "delete-generic-password", "-a", self.account, "-s", SERVICE])

    def where(self) -> str:
        return f"macOS Keychain, service {SERVICE}, account {self.account}"


class _SecretToolStore:
    """Linux libsecret via ``secret-tool``; ``store`` reads the secret from stdin."""

    kind = "secret-tool"

    def __init__(self, account: str):
        self.account = account

    def _attrs(self) -> list:
        return ["service", SERVICE, "account", self.account]

    def get(self) -> Optional[bytes]:
        return _decode_token(_run(["secret-tool", "lookup", *self._attrs()]))

    def put(self, key_pem: bytes) -> None:
        _run(["secret-tool", "store", "--label=HAL wire CA key", *self._attrs()],
             stdin=base64.b64encode(key_pem))
        if self.get() != key_pem:
            raise _StoreUnavailable("libsecret (secret-tool) did not keep the key")

    def clear(self) -> None:
        _run(["secret-tool", "clear", *self._attrs()])

    def where(self) -> str:
        return f"libsecret (secret-tool), service {SERVICE}, account {self.account}"


def _account() -> str:
    """A per-HAL-home store account that does not spell out the home path."""
    digest = hashlib.sha256(str(hal_home().resolve()).encode("utf-8")).hexdigest()
    return f"hal-home-{digest[:16]}"


def _store_mode() -> str:
    mode = (os.environ.get(STORE_ENV) or "auto").strip().lower()
    if mode not in STORE_CHOICES:
        raise WireCAError(f"{STORE_ENV} must be one of {', '.join(STORE_CHOICES)}")
    return mode


def _preferred_store():
    """The OS secret store for this platform, or the file when none is usable."""
    if _store_mode() == "file":
        return _FileStore(ca_key_path())
    if sys.platform == "darwin" and shutil.which("security"):
        return _KeychainStore(_account())
    if sys.platform.startswith("linux") and shutil.which("secret-tool"):
        return _SecretToolStore(_account())
    return _FileStore(ca_key_path())


def _store_of_kind(kind: str):
    stores = {"file": lambda: _FileStore(ca_key_path()),
              "macos-keychain": lambda: _KeychainStore(_account()),
              "secret-tool": lambda: _SecretToolStore(_account())}
    return stores[kind]()


def _store_key(key_pem: bytes) -> str:
    """Put the CA key in the preferred store, else the 0600 file. Returns its custody."""
    store = _preferred_store()
    file_store = _FileStore(ca_key_path())
    if store.kind == "file":
        reason = (f"{STORE_ENV}=file" if _store_mode() == "file"
                  else "no OS secret store was found")
        return _fall_back(file_store, key_pem, reason)
    try:
        store.put(key_pem)
    except _StoreUnavailable as exc:
        return _fall_back(file_store, key_pem, str(exc))
    file_store.clear()  # once a secret store holds the key, no file may keep a copy
    return store.kind


def _fall_back(file_store: _FileStore, key_pem: bytes, reason: str) -> str:
    file_store.put(key_pem)
    _log.warning("wire CA private key kept in a 0600 file (%s): %s",
                 file_store.path, reason)
    return file_store.kind


# --------------------------------------------------------------------------- #
# Certificates.
# --------------------------------------------------------------------------- #
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _name(common_name: str):
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])


def _key_usage(**granted: bool):
    from cryptography import x509
    names = ("digital_signature", "content_commitment", "key_encipherment",
             "data_encipherment", "key_agreement", "key_cert_sign", "crl_sign",
             "encipher_only", "decipher_only")
    return x509.KeyUsage(**{name: granted.get(name, False) for name in names})


def _ca_extensions(key) -> list:
    import ipaddress

    from cryptography import x509
    every_ip = [x509.IPAddress(ipaddress.ip_network("0.0.0.0/0")),
                x509.IPAddress(ipaddress.ip_network("::/0"))]
    return [
        (x509.BasicConstraints(ca=True, path_length=0), True),
        (_key_usage(key_cert_sign=True, crl_sign=True), True),
        (x509.NameConstraints(permitted_subtrees=[x509.DNSName(CONSTRAINED_HOST)],
                              excluded_subtrees=every_ip), True),
        (x509.SubjectKeyIdentifier.from_public_key(key.public_key()), False),
    ]


def _new_ca() -> tuple:
    """A fresh CA key and self-signed, name-constrained certificate."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    now = _utcnow()
    key = ec.generate_private_key(ec.SECP256R1())
    builder = (x509.CertificateBuilder()
               .subject_name(_name(_CA_COMMON_NAME)).issuer_name(_name(_CA_COMMON_NAME))
               .public_key(key.public_key()).serial_number(x509.random_serial_number())
               .not_valid_before(now - CLOCK_SKEW).not_valid_after(now + CA_LIFETIME))
    for extension, critical in _ca_extensions(key):
        builder = builder.add_extension(extension, critical)
    return key, builder.sign(key, hashes.SHA256())


def _key_pem(key) -> bytes:
    from cryptography.hazmat.primitives import serialization
    return key.private_bytes(serialization.Encoding.PEM,
                             serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())


def _cert_pem(cert) -> bytes:
    from cryptography.hazmat.primitives import serialization
    return cert.public_bytes(serialization.Encoding.PEM)


def fingerprint(cert) -> str:
    """SHA-256 of the certificate's DER encoding, lower-case hex."""
    from cryptography.hazmat.primitives import hashes
    return cert.fingerprint(hashes.SHA256()).hex()


def _common_name(cert) -> str:
    from cryptography.x509.oid import NameOID
    names = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    return str(names[0].value) if names else ""


def _read_cert(path: Path):
    from cryptography import x509
    try:
        return x509.load_pem_x509_certificate(path.read_bytes())
    except FileNotFoundError as exc:
        raise WireCAError(f"no wire CA at {path}; run `python -m lib.hal.wire_ca init`") from exc
    except (OSError, ValueError) as exc:
        raise WireCAError(f"the wire CA at {path} cannot be read; run "
                          "`python -m lib.hal.wire_ca rotate`") from exc


def _matching_key(key_pem: Optional[bytes], cert):
    """The private key in ``key_pem`` if it is the certificate's key, else None."""
    from cryptography.hazmat.primitives import serialization
    if not key_pem:
        return None
    try:
        key = serialization.load_pem_private_key(key_pem, password=None)
    except (ValueError, TypeError):
        return None
    spki = serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    if key.public_key().public_bytes(*spki) != cert.public_key().public_bytes(*spki):
        return None
    return key


def _assert_constrained(cert) -> None:
    """Refuse a certificate that is not exactly this module's constrained CA."""
    from cryptography import x509
    try:
        basic = cert.extensions.get_extension_for_class(x509.BasicConstraints).value
        names = cert.extensions.get_extension_for_class(x509.NameConstraints).value
    except x509.ExtensionNotFound as exc:
        raise WireCAError("the wire CA certificate has no name constraint; run "
                          "`python -m lib.hal.wire_ca rotate`") from exc
    permitted = [entry.value for entry in (names.permitted_subtrees or [])]
    if not basic.ca or basic.path_length != 0 or permitted != [CONSTRAINED_HOST]:
        raise WireCAError(f"the wire CA certificate is not limited to {CONSTRAINED_HOST} "
                          "with path length 0; run `python -m lib.hal.wire_ca rotate`")


def _assert_usable(cert) -> None:
    _assert_constrained(cert)
    digest = fingerprint(cert)
    if digest in revoked_fingerprints():
        raise WireCAError(f"the wire CA {digest[:16]} was revoked; run "
                          "`python -m lib.hal.wire_ca rotate`")
    if cert.not_valid_after_utc <= _utcnow():
        raise WireCAError("the wire CA has expired; run `python -m lib.hal.wire_ca rotate`")


# --------------------------------------------------------------------------- #
# Revocation.
# --------------------------------------------------------------------------- #
def revoked() -> list:
    """The revocation records. A file that exists but cannot be read is an error."""
    path = revoked_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as exc:
        raise WireCAError(f"the revocation list {path} cannot be read; repair it "
                          "before using the wire CA") from exc
    rows = data.get("revoked") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise WireCAError(f"the revocation list {path} has no 'revoked' list")
    return [row for row in rows if isinstance(row, dict)]


def revoked_fingerprints() -> set:
    return {str(row.get("fingerprint_sha256", "")) for row in revoked()}


def _record_revoked(cert, reason: str) -> None:
    rows = revoked()
    digest = fingerprint(cert)
    if digest not in {row.get("fingerprint_sha256") for row in rows}:
        rows.append({"fingerprint_sha256": digest, "subject": _common_name(cert),
                     "not_after": cert.not_valid_after_utc.isoformat(),
                     "revoked_at": _utcnow().isoformat(), "reason": reason})
    _atomic_write(revoked_path(), json.dumps({"revoked": rows}, indent=2).encode("utf-8"),
                  0o644)


# --------------------------------------------------------------------------- #
# The CA and its leaves.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Leaf:
    """A server certificate for one connection lifetime. Lives in memory only."""

    hostname: str
    cert_pem: bytes
    key_pem: bytes = field(repr=False)
    not_after: datetime

    def expires_within(self, seconds: float, now: Optional[datetime] = None) -> bool:
        """Whether the leaf expires within ``seconds`` (re-mint before then)."""
        return self.not_after - (now or _utcnow()) <= timedelta(seconds=seconds)


@dataclass(frozen=True)
class WireCA:
    """A loaded wire CA. ``key`` never appears in ``repr``."""

    cert: Any
    key: Any = field(repr=False)
    custody: str = "file"

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.cert)

    def mint_leaf(self, hostname: str = CONSTRAINED_HOST) -> Leaf:
        """A fresh in-memory leaf for ``hostname``; any other host is refused."""
        return _sign_leaf(self, _constrained_host(hostname))


def _constrained_host(hostname: str) -> str:
    host = str(hostname or "").strip().rstrip(".").lower()
    if host != CONSTRAINED_HOST:
        raise WireCAError(f"refusing to mint a leaf for {host!r}: the wire CA "
                          f"vouches for {CONSTRAINED_HOST} only")
    return host


def _leaf_extensions(host: str, key, ca: WireCA) -> list:
    from cryptography import x509
    from cryptography.x509.oid import ExtendedKeyUsageOID
    return [
        (x509.SubjectAlternativeName([x509.DNSName(host)]), False),
        (x509.BasicConstraints(ca=False, path_length=None), True),
        (_key_usage(digital_signature=True), True),
        (x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), False),
        (x509.SubjectKeyIdentifier.from_public_key(key.public_key()), False),
        (x509.AuthorityKeyIdentifier.from_issuer_public_key(ca.cert.public_key()), False),
    ]


def _sign_leaf(ca: WireCA, host: str) -> Leaf:
    """Sign a leaf for ``host`` with no host check (``mint_leaf`` does the check)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    now = _utcnow()
    key = ec.generate_private_key(ec.SECP256R1())
    not_after = min(now + LEAF_LIFETIME, ca.cert.not_valid_after_utc)
    builder = (x509.CertificateBuilder()
               .subject_name(_name(host)).issuer_name(ca.cert.subject)
               .public_key(key.public_key()).serial_number(x509.random_serial_number())
               .not_valid_before(now - CLOCK_SKEW).not_valid_after(not_after))
    for extension, critical in _leaf_extensions(host, key, ca):
        builder = builder.add_extension(extension, critical)
    cert = builder.sign(ca.key, hashes.SHA256())
    return Leaf(hostname=host, cert_pem=_cert_pem(cert), key_pem=_key_pem(key),
                not_after=not_after)


def load() -> WireCA:
    """The existing CA, checked: constrained, not revoked, not expired, key present."""
    cert = _read_cert(ca_cert_path())
    _assert_usable(cert)
    key, custody = _load_key(cert)
    return WireCA(cert=cert, key=key, custody=custody)


def _load_key(cert) -> tuple:
    """(key, custody) from the preferred store, then the fallback file."""
    stores = [_preferred_store()]
    if stores[0].kind != "file":
        stores.append(_FileStore(ca_key_path()))
    for store in stores:
        key = _matching_key(store.get(), cert)
        if key is not None:
            return key, store.kind
    raise WireCAError(
        f"the wire CA's private key is not readable from {stores[0].where()} or "
        f"{ca_key_path()}, or it does not match {ca_cert_path()}; unlock the store, "
        "or run `python -m lib.hal.wire_ca rotate` to replace the CA")


def _create() -> WireCA:
    """A new CA: the key goes into custody first, then the certificate is published."""
    key, cert = _new_ca()
    custody = _store_key(_key_pem(key))
    _atomic_write(ca_cert_path(), _cert_pem(cert), 0o644)
    return WireCA(cert=cert, key=key, custody=custody)


def load_or_create() -> WireCA:
    """The existing CA, or a new one when there is none. Never replaces one by itself."""
    with _locked():
        if ca_cert_path().exists():
            return load()
        return _create()


def rotate() -> WireCA:
    """Replace the CA: revoke the current one (if any), then create a new one.

    Each step leaves a state ``load`` refuses until the next completes (a revoked
    certificate, or a key that does not match it), so a crash mid-rotation fails
    closed, and running ``rotate`` again finishes the job.
    """
    with _locked():
        old = _current_cert_or_none()
        if old is not None:
            _record_revoked(old, reason="rotated")
        return _create()


def _current_cert_or_none():
    """The certificate being replaced, or None when there is none or it is unreadable."""
    if not ca_cert_path().exists():
        return None
    try:
        return _read_cert(ca_cert_path())
    except WireCAError:
        _log.warning("the old wire CA certificate could not be read, so its "
                     "fingerprint was not recorded as revoked")
        return None


def mint_leaf(hostname: str = CONSTRAINED_HOST, ca: Optional[WireCA] = None) -> Leaf:
    """A fresh in-memory leaf for ``hostname`` from ``ca`` (default: the loaded CA)."""
    return (ca or load()).mint_leaf(hostname)


# --------------------------------------------------------------------------- #
# Loading a leaf into an SSLContext without touching disk.
# --------------------------------------------------------------------------- #
def server_context(leaf: Leaf) -> ssl.SSLContext:
    """A TLS 1.2+ server context serving ``leaf``, ALPN ``http/1.1`` only."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.set_alpn_protocols(["http/1.1"])
    _load_chain_in_memory(context, leaf.cert_pem, leaf.key_pem)
    return context


def _load_chain_in_memory(context: ssl.SSLContext, cert_pem: bytes, key_pem: bytes) -> None:
    if not os.path.isdir("/dev/fd"):
        raise WireCAError("loading a leaf without writing it to disk needs /dev/fd "
                          "(macOS or Linux); this platform has none")
    cert_fd = _filled_pipe(cert_pem)
    try:
        key_fd = _filled_pipe(key_pem)
        try:
            context.load_cert_chain(f"/dev/fd/{cert_fd}", f"/dev/fd/{key_fd}")
        finally:
            os.close(key_fd)
    finally:
        os.close(cert_fd)


def _filled_pipe(data: bytes) -> int:
    """The read end of a pipe already holding ``data``, its write end closed."""
    if len(data) > _PIPE_LIMIT:
        raise WireCAError("a PEM block is larger than one pipe buffer")
    read_fd, write_fd = os.pipe()
    try:
        written = os.write(write_fd, data)
    except OSError:
        os.close(read_fd)
        raise
    finally:
        os.close(write_fd)
    if written != len(data):
        os.close(read_fd)
        raise WireCAError("a PEM block did not fit in one pipe write")
    return read_fd


# --------------------------------------------------------------------------- #
# Reporting + CLI.
# --------------------------------------------------------------------------- #
def describe(ca: WireCA) -> dict:
    """Paths, fingerprints, validity and custody. Never key material."""
    not_after = ca.cert.not_valid_after_utc
    return {
        "ca_cert": str(ca_cert_path()),
        "fingerprint_sha256": ca.fingerprint,
        "subject": _common_name(ca.cert),
        "not_before": ca.cert.not_valid_before_utc.isoformat(),
        "not_after": not_after.isoformat(),
        "days_remaining": (not_after - _utcnow()).days,
        "name_constraint": CONSTRAINED_HOST,
        "path_length": 0,
        "key_custody": ca.custody,
        "key_location": _store_of_kind(ca.custody).where(),
        "revoked_fingerprints": sorted(revoked_fingerprints()),
    }


def show() -> dict:
    return describe(load())


_COMMANDS: dict = {"init": load_or_create, "show": load, "rotate": rotate}


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lib.hal.wire_ca",
        description="The HAL wire's per-host CA (name-constrained to api.anthropic.com).")
    parser.add_argument("command", choices=sorted(_COMMANDS),
                        help="init: create if absent; show: report; rotate: replace")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="wire_ca: %(message)s")
    command: Callable[[], WireCA] = _COMMANDS[args.command]
    try:
        report = describe(command())
    except WireCAError as exc:
        print(json.dumps({"ok": False, "command": args.command, "error": str(exc)}))
        return 1
    print(json.dumps({"ok": True, "command": args.command, **report}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
