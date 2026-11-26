"""deploy_render -- fill HAL's service templates (``deploy/``) for one host.

InfiniteMachine v2, Step 6 (GRIP plan ``ULTRAPLAN-2026-10-04-infinite-machine-v2.md``,
"Step 6 -- Everywhere"; the operator's Decision 4: a wire on every trusted host,
any OS). The gateway (127.0.0.1:4010) and the wire (127.0.0.1:4011) each have
one template per service manager under ``deploy/``. The brain is a Linux host
service and has a systemd user unit only::

    launchd (macOS)          com.hal.gateway.plist    com.hal.wire.plist
    systemd --user (Linux)   hal-gateway.service      hal-wire.service
    Task Scheduler (Windows) hal-gateway.task.xml     hal-wire.task.xml
    systemd brain            hal-brain.service

This module is the one place that fills them. GRIP's installer
(``bin/grip-hal-services``) calls the CLI below, then hands the text to GRIP's
``bin/grip-service-reload``, which installs it and proves it with ``/health``.

Placeholders (``__NAME__``) and what fills them:

    PYTHON         the interpreter: an absolute path inside a virtual environment
                   (its ``pyvenv.cfg`` must exist). A system interpreter is
                   refused: Homebrew's python3 has no ``cryptography``, which the
                   wire CA and the signed thread store need.
    HAL_HOME       the HAL checkout the service runs from.
    LOG_DIR        where the service's stdout and stderr are appended.
    PORT           the port (default 4010 for the gateway, 4011 for the wire, both on
                   loopback; 7416 for the brain, on its BIND address). For the
                   brain 4010, 4011 and 4012 are refused: the brain exits on them.
    WIRE_CA_STORE  wire only: ``auto`` (the OS secret store) or ``file`` (0600).
    USER           Task Scheduler only: the account whose logon starts the task.
    BIND           brain only: an IPv4 address, loopback (default 127.0.0.1) or
                   this host's tailnet address (100.64.0.0/10). ``0.0.0.0``, any
                   IPv6 address (``::1`` too) and any hostname are refused.
    STATE_DB       brain only: required absolute SQLite path.
    PINS_FILE      brain only: required absolute public pins path.
    TAILSCALE_TAG  brain only: ``tag:name``, required for a tailnet bind; may be
                   empty on loopback, where the brain admits no remote peer.

Values are escaped for their format (XML for the plist and the task; systemd
needs none for what is allowed), and a value that cannot be made safe is refused
rather than mangled: a control character anywhere; ``"``, ``\\``, ``$`` or ``%``
in a systemd unit (quoting, variable and specifier expansion); ``"`` or ``%`` on
a Windows command line (cmd.exe quoting and variable expansion). Every
placeholder must be filled and every value must have a slot, so a template that
gains or loses a placeholder fails loudly instead of rendering half-filled. A
filled definition that still holds placeholder-shaped text is refused as well.

No secret passes through here: the templates hold none, and no parameter can
carry one.

CLI::

    python -m lib.hal.deploy_render list
    python -m lib.hal.deploy_render render --platform launchd --service wire \\
        --python <venv>/bin/python --hal-home <hal> --log-dir <dir> \\
        [--port N] [--wire-ca-store auto|file] [--user DOMAIN\\name] [--no-check-paths]
    python -m lib.hal.deploy_render render --platform systemd --service brain \\
        --python <venv>/bin/python --hal-home <hal> --log-dir <dir> \\
        --state-db <absolute-path> --pins-file <absolute-path> \\
        [--bind <loopback-or-tailnet-IPv4>] [--tailscale-tag tag:name]

``list`` prints the services, their names on each platform, ports and ``/health``
identities as JSON. ``render`` writes the filled template to stdout (UTF-8) and
nothing else, or exits 2 with the reason on stderr. ``--no-check-paths`` skips
the checks that need the paths to exist on this machine (rendering for another
host); the syntax checks always run.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Dict, Mapping, Optional
from xml.sax.saxutils import escape as xml_escape

DEPLOY_DIR = Path(__file__).resolve().parents[2] / "deploy"
PLATFORMS = ("launchd", "systemd", "schtasks")
WIRE_CA_STORES = ("auto", "file")
DEFAULT_WIRE_CA_STORE = "auto"

_PLACEHOLDER = re.compile(r"__([A-Z][A-Z_]*[A-Z])__")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_UNSAFE = {"launchd": "", "systemd": '"\\$%', "schtasks": '"%'}
_XML_PLATFORMS = ("launchd", "schtasks")
_PATH_VALUES = ("PYTHON", "HAL_HOME", "LOG_DIR")
_TAILNET = ipaddress.ip_network("100.64.0.0/10")
#: Tailscale's own rule (``tailcfg.CheckTag``): ``tag:``, one ASCII letter, then
#: ASCII letters, digits or dashes. HAL's guard compares the tag exactly, case
#: included, so an upper-case tag must pass through here unchanged.
_TAILSCALE_TAG = re.compile(r"tag:[A-Za-z][A-Za-z0-9-]*\Z")
_BIND_RULE = ("bind must be an IPv4 address on loopback (127.0.0.0/8) or in the tailnet "
              "range 100.64.0.0/10; a hostname or an IPv6 address is refused, because "
              "the brain listens on IPv4 only")
#: The ports ``lib.hal.brain`` exits on at start. A unit carrying one would be
#: restarted for ever, so it is refused here first. A test holds the two in step.
_BRAIN_REFUSED_PORTS = (4010, 4011, 4012)


class RenderError(ValueError):
    """A template could not be filled safely. The message says why."""


@dataclass(frozen=True)
class Service:
    """One HAL service: what it runs, where it listens, and its name per platform."""

    key: str
    module: str
    port: int
    health: str
    names: Mapping[str, str]
    templates: Mapping[str, str]


SERVICES: Dict[str, Service] = {
    "brain": Service(
        key="brain", module="lib.hal.brain", port=7416, health="hal-brain",
        names={"systemd": "hal-brain.service"},
        templates={"systemd": "hal-brain.service"}),
    "gateway": Service(
        key="gateway", module="lib.hal.gateway", port=4010, health="hal-gateway",
        names={"launchd": "com.hal.gateway", "systemd": "hal-gateway.service",
               "schtasks": "GRIP HAL Gateway"},
        templates={"launchd": "com.hal.gateway.plist", "systemd": "hal-gateway.service",
                   "schtasks": "hal-gateway.task.xml"}),
    "wire": Service(
        key="wire", module="lib.hal.wire_proxy", port=4011, health="hal-wire",
        names={"launchd": "com.hal.wire", "systemd": "hal-wire.service",
               "schtasks": "GRIP HAL Wire"},
        templates={"launchd": "com.hal.wire.plist", "systemd": "hal-wire.service",
                   "schtasks": "hal-wire.task.xml"}),
}


# --------------------------------------------------------------------------- #
# Checks on the values.
# --------------------------------------------------------------------------- #
def _service(key: str) -> Service:
    if key not in SERVICES:
        raise RenderError(f"unknown service {key!r}; known: {', '.join(sorted(SERVICES))}")
    return SERVICES[key]


def _check_platform(platform: str) -> None:
    if platform not in PLATFORMS:
        raise RenderError(f"unknown platform {platform!r}; known: {', '.join(PLATFORMS)}")


def _check_service_platform(platform: str, service: Service) -> None:
    if platform not in service.templates:
        if service.key == "brain":
            raise RenderError("the brain is a systemd (Linux host) service by design")
        raise RenderError(f"{service.key} is unsupported on {platform}")


def _check_text(platform: str, name: str, value: object) -> str:
    """A non-empty string with nothing the target format cannot carry safely."""
    if not isinstance(value, str) or not value:
        raise RenderError(f"{name} is empty")
    if _CONTROL.search(value):
        raise RenderError(f"{name} holds a control character")
    unsafe = sorted(set(value) & set(_UNSAFE[platform]))
    if unsafe:
        raise RenderError(f"{name} holds {''.join(unsafe)!r}, which a {platform} "
                          "definition cannot carry safely")
    return value


def _check_absolute(platform: str, name: str, value: str) -> None:
    pure = PureWindowsPath if platform == "schtasks" else PurePosixPath
    if not pure(value).is_absolute():
        raise RenderError(f"{name} must be an absolute path, not {value!r}")


def _port(port: Optional[int], service: Service) -> str:
    chosen = service.port if port is None else port
    if isinstance(chosen, bool) or not isinstance(chosen, int) or not 1 <= chosen <= 65535:
        raise RenderError(f"port must be an integer from 1 to 65535, not {chosen!r}")
    if service.key == "brain" and chosen in _BRAIN_REFUSED_PORTS:
        raise RenderError(f"port {chosen} is reserved for another HAL service: the brain "
                          "refuses to start on it")
    return str(chosen)


def _wire_ca_store(service: Service, store: Optional[str]) -> Dict[str, str]:
    if service.key != "wire":
        if store is not None:
            raise RenderError("the wire CA store applies to the wire only")
        return {}
    chosen = DEFAULT_WIRE_CA_STORE if store is None else store
    if chosen not in WIRE_CA_STORES:
        raise RenderError(f"wire CA store must be one of {', '.join(WIRE_CA_STORES)}")
    return {"WIRE_CA_STORE": chosen}


def _brain_bind(bind: Optional[str]) -> tuple[str, bool]:
    """The brain's bind address, and whether it is loopback.

    ``lib.hal.brain`` serves on an IPv4 socket, so ``::1`` is refused with every
    other IPv6 address. A hostname is refused too: what ``localhost`` or a
    tailnet name resolves to on the target host cannot be checked from here.
    """
    chosen = "127.0.0.1" if bind is None else bind
    try:
        address = ipaddress.IPv4Address(chosen) if isinstance(chosen, str) else None
    except ValueError:
        address = None
    if address is None or not (address.is_loopback or address in _TAILNET):
        raise RenderError(f"{_BIND_RULE}; got {chosen!r}")
    return str(address), address.is_loopback


def _brain_path(platform: str, name: str, value: Optional[str]) -> str:
    if value is None:
        flag = "--" + name.replace("_", "-")
        raise RenderError(f"the brain needs {name} ({flag}): an absolute path, with no "
                          "safe default")
    _check_absolute(platform, name, _check_text(platform, name, value))
    return value


def _brain_tag(tailscale_tag: Optional[str], loopback: bool) -> str:
    """The ACL tag HAL's guard asks of a remote peer. Empty admits no remote peer."""
    tag = "" if tailscale_tag is None else tailscale_tag
    if not isinstance(tag, str) or (tag and not _TAILSCALE_TAG.fullmatch(tag)):
        raise RenderError("tailscale tag must match tag:name (a letter, then letters, "
                          "digits or dashes)")
    if not loopback and not tag:
        raise RenderError("tailscale tag is required for a tailnet bind: without one "
                          "the brain refuses every remote peer")
    return tag


def _brain_values(platform: str, service: Service, bind: Optional[str],
                  state_db: Optional[str], pins_file: Optional[str],
                  tailscale_tag: Optional[str]) -> Dict[str, str]:
    if service.key != "brain":
        if any(value is not None for value in (bind, state_db, pins_file, tailscale_tag)):
            raise RenderError("bind, state DB, pins file and Tailscale tag are brain only")
        return {}
    address, loopback = _brain_bind(bind)
    return {"BIND": address,
            "STATE_DB": _brain_path(platform, "state_db", state_db),
            "PINS_FILE": _brain_path(platform, "pins_file", pins_file),
            "TAILSCALE_TAG": _brain_tag(tailscale_tag, loopback)}


def _user(platform: str, user: Optional[str]) -> Dict[str, str]:
    if platform != "schtasks":
        if user is not None:
            raise RenderError("the logon user applies to Task Scheduler only")
        return {}
    if user is None:
        raise RenderError("Task Scheduler needs the logon user (DOMAIN\\name)")
    return {"USER": _check_text(platform, "USER", user)}


def _check_on_disk(service: Service, python: str, hal_home: str) -> None:
    """The interpreter is inside a venv and HAL_HOME holds the service's module."""
    interpreter = Path(python)
    if not interpreter.is_file():
        raise RenderError(f"the interpreter {python} does not exist")
    if not (interpreter.parent.parent / "pyvenv.cfg").is_file():
        raise RenderError(f"{python} is not inside a virtual environment (no pyvenv.cfg "
                          "beside its bin directory); use GRIP's venv, never a system python")
    module = Path(hal_home, *service.module.split(".")).with_suffix(".py")
    if not module.is_file():
        raise RenderError(f"{hal_home} is not a HAL checkout: {module} is missing")


def values_for(platform: str, service: Service, *, python: str, hal_home: str,
               log_dir: str, port: Optional[int], wire_ca_store: Optional[str],
               user: Optional[str], bind: Optional[str], state_db: Optional[str],
               pins_file: Optional[str], tailscale_tag: Optional[str]) -> Dict[str, str]:
    """Every placeholder's value for ``service`` on ``platform``, checked."""
    values = {"PYTHON": python, "HAL_HOME": hal_home, "LOG_DIR": log_dir}
    for name in _PATH_VALUES:
        _check_absolute(platform, name, _check_text(platform, name, values[name]))
    values["PORT"] = _port(port, service)
    values.update(_wire_ca_store(service, wire_ca_store))
    values.update(_user(platform, user))
    values.update(_brain_values(platform, service, bind, state_db, pins_file, tailscale_tag))
    return values


# --------------------------------------------------------------------------- #
# Filling a template.
# --------------------------------------------------------------------------- #
def template_text(platform: str, service: str) -> str:
    """The raw template for ``service`` on ``platform``."""
    _check_platform(platform)
    svc = _service(service)
    _check_service_platform(platform, svc)
    return (DEPLOY_DIR / svc.templates[platform]).read_text(encoding="utf-8")


def placeholders(text: str) -> set:
    """The placeholder names a template carries."""
    return set(_PLACEHOLDER.findall(text))


def fill(text: str, values: Mapping[str, str], platform: str) -> str:
    """``text`` with every placeholder replaced, escaped for ``platform``."""
    wanted = placeholders(text)
    missing = sorted(wanted - set(values))
    unused = sorted(set(values) - wanted)
    if missing:
        raise RenderError(f"the template has placeholders nothing fills: {missing}")
    if unused:
        raise RenderError(f"the template has no slot for: {unused}")
    escape = xml_escape if platform in _XML_PLATFORMS else str
    rendered = _PLACEHOLDER.sub(lambda match: escape(values[match.group(1)]), text)
    left = sorted(placeholders(rendered))
    if left:
        raise RenderError(f"a placeholder survived rendering (a value is shaped like one): {left}")
    return rendered


def render(platform: str, service: str, *, python: str, hal_home: str, log_dir: str,
           port: Optional[int] = None, wire_ca_store: Optional[str] = None,
           user: Optional[str] = None, check_paths: bool = True, bind: Optional[str] = None,
           state_db: Optional[str] = None, pins_file: Optional[str] = None,
           tailscale_tag: Optional[str] = None) -> str:
    """The filled template for ``service`` on ``platform``. Raises RenderError."""
    _check_platform(platform)
    svc = _service(service)
    _check_service_platform(platform, svc)
    values = values_for(platform, svc, python=python, hal_home=hal_home, log_dir=log_dir,
                        port=port, wire_ca_store=wire_ca_store, user=user, bind=bind,
                        state_db=state_db, pins_file=pins_file, tailscale_tag=tailscale_tag)
    if check_paths:
        _check_on_disk(svc, python, hal_home)
    return fill(template_text(platform, service), values, platform)


def catalogue() -> dict:
    """What ``list`` prints: services, names per platform, ports, health identities."""
    return {
        "platforms": list(PLATFORMS),
        "wire_ca_stores": list(WIRE_CA_STORES),
        "services": {key: {"module": svc.module, "port": svc.port, "health": svc.health,
                           "names": dict(svc.names), "templates": dict(svc.templates)}
                     for key, svc in SERVICES.items()},
    }


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lib.hal.deploy_render",
        description="Fill HAL's service templates (deploy/) for this host.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="services, names per platform, ports, /health identities")
    render_cmd = sub.add_parser("render", help="print one filled template")
    render_cmd.add_argument("--platform", required=True, choices=PLATFORMS)
    render_cmd.add_argument("--service", required=True, choices=sorted(SERVICES))
    render_cmd.add_argument("--python", required=True, help="a venv interpreter")
    render_cmd.add_argument("--hal-home", required=True)
    render_cmd.add_argument("--log-dir", required=True)
    render_cmd.add_argument("--port", type=int, help="default: the service's own port")
    render_cmd.add_argument("--wire-ca-store", choices=WIRE_CA_STORES,
                            help=f"wire only (default {DEFAULT_WIRE_CA_STORE})")
    render_cmd.add_argument("--user", help="Task Scheduler only: DOMAIN\\name")
    _add_brain_arguments(render_cmd)
    render_cmd.add_argument("--no-check-paths", action="store_true",
                            help="skip the checks that need the paths on this machine")
    return parser


def _add_brain_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bind", help="brain only: IPv4 loopback or tailnet address "
                                       "(default 127.0.0.1)")
    parser.add_argument("--state-db", help="brain only, required: absolute SQLite path")
    parser.add_argument("--pins-file", help="brain only, required: absolute public pins path")
    parser.add_argument("--tailscale-tag", help="brain only: tag:name, required for a "
                                                "tailnet bind")


def main(argv: Optional[list] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "list":
        print(json.dumps(catalogue(), indent=2))
        return 0
    try:
        text = render(args.platform, args.service, python=args.python,
                      hal_home=args.hal_home, log_dir=args.log_dir, port=args.port,
                      wire_ca_store=args.wire_ca_store, user=args.user,
                      bind=args.bind, state_db=args.state_db, pins_file=args.pins_file,
                      tailscale_tag=args.tailscale_tag,
                      check_paths=not args.no_check_paths)
    except RenderError as exc:
        print(f"deploy_render: {exc}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(text.encode("utf-8"))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
