"""Tests for lib/hal/deploy_render.py and the service templates under deploy/.

InfiniteMachine v2, Step 6: the gateway and the wire as a per-user service on
macOS (launchd), Linux (systemd --user) and Windows (Task Scheduler). GRIP's
installer renders these and GRIP's safe reload helper installs them, so a
wrong template reaches every host at once. These tests pin what each definition
must do -- run the service from a venv, keep it alive, log to the log
directory, set no variable beyond an allowlist, carry no secret -- and that the
renderer refuses a value it cannot carry safely instead of mangling it.

The brain (the shared cooldown and signed-head service) is a Linux host
service: one systemd user unit and no other platform. Its section at the end
pins the same promises, plus the one that matters most for it: the bind address
is IPv4 loopback or a tailnet address, never a public interface.

Golden files: tests/fixtures/deploy/<platform>-<service>.golden, rendered from
fixed, obviously fake inputs. After a deliberate template change, regenerate
with HAL_UPDATE_GOLDEN=1 and read the diff before committing.
"""
from __future__ import annotations

import json
import os
import plistlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from lib.hal import deploy_render as dr
from tests._public_tree import require

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "tests" / "fixtures" / "deploy"
TASK = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
POSIX = {"python": "/opt/grip/venv/bin/python", "hal_home": "/opt/hal",
         "log_dir": "/opt/hal-logs"}
WINDOWS = {"python": r"C:\grip\venv\Scripts\python.exe", "hal_home": r"C:\hal",
           "log_dir": r"C:\hal-logs", "user": r"EXAMPLE\jane"}
#: The only variables a template may set. A new one -- a token, say -- fails
#: here until a reviewer decides it belongs.
ENV_ALLOWLIST = {"gateway": {"HAL_HOME", "HAL_GATEWAY_MULTIPROVIDER", "HAL_GATEWAY_NO_RECLAIM",
                             "PATH", "PYTHONUNBUFFERED"},
                 "wire": {"HAL_HOME", "HAL_WIRE_CA_STORE", "PATH", "PYTHONUNBUFFERED"}}
#: Port reclaim is off in every definition: the gateway reads it from its
#: environment, the wire from a flag (HAL main's reclaim kills even a healthy holder).
NO_RECLAIM_ARGS = {"gateway": [], "wire": ["--no-reclaim"]}
SLOTS = {"launchd": {"PYTHON", "HAL_HOME", "LOG_DIR", "PORT"},
         "systemd": {"PYTHON", "HAL_HOME", "LOG_DIR", "PORT"},
         "schtasks": {"PYTHON", "HAL_HOME", "LOG_DIR", "PORT", "USER"}}
SECRET_SHAPES = re.compile(r"(?i)(sk-ant-|sk-[a-z0-9]{20,}|ghp_|github_pat_|AKIA[0-9A-Z]{12}"
                           r"|-----BEGIN|bearer\s+\S|x-api-key|authorization:)")
#: The services that exist on every platform, read from the renderer so that a
#: new one joins every test below (and fails until it has its golden files).
#: The brain is systemd only, so it is not a row of CASES: its tests are the
#: last section of this file, and one of them fails if any other service is
#: ever limited to some platforms without tests of its own.
SERVICES = sorted(key for key, svc in dr.SERVICES.items()
                  if set(svc.templates) == set(dr.PLATFORMS))
CASES = [(platform, service) for platform in dr.PLATFORMS for service in SERVICES]
BRAIN = {"python": "/srv/hal/venv/bin/python", "hal_home": "/srv/hal",
         "log_dir": "/srv/hal/logs", "state_db": "/srv/hal/brain.sqlite",
         "pins_file": "/srv/hal/public-pins.json", "bind": "100.64.0.1",
         "tailscale_tag": "tag:example"}
BRAIN_SLOTS = {"PYTHON", "HAL_HOME", "LOG_DIR", "PORT", "BIND", "STATE_DB", "PINS_FILE",
               "TAILSCALE_TAG"}
BRAIN_ENV = {"HAL_HOME", "PYTHONUNBUFFERED", "HAL_GATEWAY_TS_ACL_TAG"}
#: One value for each brain-only argument, and an empty one: given for another
#: service, each must be refused rather than dropped.
BRAIN_ARGS = [("bind", "127.0.0.1"), ("state_db", "/srv/hal/state.sqlite"),
              ("pins_file", "/srv/hal/pins.json"), ("tailscale_tag", "tag:example"),
              ("tailscale_tag", "")]
BRAIN_CLI = ["--bind", BRAIN["bind"], "--state-db", BRAIN["state_db"],
             "--pins-file", BRAIN["pins_file"], "--tailscale-tag", BRAIN["tailscale_tag"]]


def _inputs(platform: str) -> dict:
    return dict(WINDOWS if platform == "schtasks" else POSIX)


def _render(platform: str, service: str, **overrides) -> str:
    try:
        return dr.render(platform, service, **{**_inputs(platform), "check_paths": False,
                                               **overrides})
    except FileNotFoundError:
        require("deploy")  # no deploy/ on a public checkout: skip; otherwise fail
        raise


def _brain(**overrides) -> str:
    try:
        return dr.render("systemd", "brain", **{**BRAIN, "check_paths": False, **overrides})
    except FileNotFoundError:
        require("deploy")  # no deploy/ on a public checkout: skip; otherwise fail
        raise


def _unit(text: str) -> dict:
    """(section, key) -> every value; unit files repeat keys, so configparser will not do."""
    parsed, section = {}, None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        key, _, value = line.partition("=")
        parsed.setdefault((section, key), []).append(value)
    return parsed


def _venv(tmp_path: Path) -> Path:
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /nowhere\n", encoding="utf-8")
    python = venv / "bin" / "python"
    python.write_text("", encoding="utf-8")
    return python


def _hal_checkout(tmp_path: Path) -> Path:
    home = tmp_path / "hal"
    for svc in dr.SERVICES.values():
        module = Path(home, *svc.module.split(".")).with_suffix(".py")
        module.parent.mkdir(parents=True, exist_ok=True)
        module.write_text("", encoding="utf-8")
    return home


# --------------------------------------------------------------------------- #
# Golden files: one per platform and service.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("platform,service", CASES)
def test_each_rendered_definition_matches_its_golden_file(platform, service):
    text = _render(platform, service)
    golden = GOLDEN / f"{platform}-{service}.golden"
    if os.environ.get("HAL_UPDATE_GOLDEN") == "1":
        golden.write_text(text, encoding="utf-8")
    assert text == golden.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# What each definition must do.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("service", SERVICES)
def test_launchd_runs_the_service_from_the_venv_and_keeps_it_alive(service):
    svc = dr.SERVICES[service]
    plist = plistlib.loads(_render("launchd", service).encode("utf-8"))
    assert plist["Label"] == svc.names["launchd"]
    assert plist["ProgramArguments"] == [POSIX["python"], "-m", svc.module,
                                         "--port", str(svc.port), *NO_RECLAIM_ARGS[service]]
    assert plist["WorkingDirectory"] == POSIX["hal_home"]
    assert plist["EnvironmentVariables"]["HAL_HOME"] == POSIX["hal_home"]
    assert plist["RunAtLoad"] is True and plist["KeepAlive"] is True
    assert plist["ThrottleInterval"] >= 5
    for key in ("StandardOutPath", "StandardErrorPath"):
        assert plist[key].startswith(POSIX["log_dir"] + "/")
    assert set(plist["EnvironmentVariables"]) == ENV_ALLOWLIST[service]


@pytest.mark.parametrize("service", SERVICES)
def test_systemd_unit_restarts_always_and_logs_to_the_log_dir(service):
    svc = dr.SERVICES[service]
    unit = _unit(_render("systemd", service))
    assert unit[("Service", "ExecStart")] == [" ".join(
        [f'"{POSIX["python"]}" -m {svc.module} --port {svc.port}', *NO_RECLAIM_ARGS[service]])]
    assert unit[("Service", "Restart")] == ["always"]
    assert int(unit[("Service", "RestartSec")][0]) >= 5
    assert unit[("Service", "WorkingDirectory")] == [POSIX["hal_home"]]
    assert unit[("Install", "WantedBy")] == ["default.target"]
    for key in ("StandardOutput", "StandardError"):
        assert unit[("Service", key)][0].startswith(f"append:{POSIX['log_dir']}/")
    names = {value.strip('"').split("=", 1)[0] for value in unit[("Service", "Environment")]}
    assert names == ENV_ALLOWLIST[service] - {"PATH"}  # systemd gives a user service its PATH


@pytest.mark.parametrize("service", SERVICES)
def test_task_starts_at_logon_restarts_on_failure_and_has_no_time_limit(service):
    svc = dr.SERVICES[service]
    root = ET.fromstring(_render("schtasks", service))
    assert root.tag == "{%s}Task" % TASK["t"]
    assert root.findtext("t:Triggers/t:LogonTrigger/t:UserId", namespaces=TASK) == WINDOWS["user"]
    assert root.findtext("t:Settings/t:ExecutionTimeLimit", namespaces=TASK) == "PT0S"
    # Microsoft's schema: Count is an unsignedByte of at least 1; Interval at least PT1M.
    count = int(root.findtext("t:Settings/t:RestartOnFailure/t:Count", namespaces=TASK))
    assert 1 <= count <= 255
    assert root.findtext("t:Settings/t:RestartOnFailure/t:Interval", namespaces=TASK) == "PT1M"
    assert root.findtext("t:Actions/t:Exec/t:Command", namespaces=TASK) == "cmd.exe"
    assert root.findtext("t:Actions/t:Exec/t:WorkingDirectory", namespaces=TASK) == WINDOWS["hal_home"]
    args = root.findtext("t:Actions/t:Exec/t:Arguments", namespaces=TASK)
    command = " ".join([f'"{WINDOWS["python"]}" -m {svc.module} --port {svc.port}',
                        *NO_RECLAIM_ARGS[service]])
    assert f"{command} 1>>" in args
    assert args.count(f'"{WINDOWS["log_dir"]}\\') == 2
    assert set(re.findall(r'set "([A-Z_]+)=', args)) == ENV_ALLOWLIST[service] - {"PATH"}


@pytest.mark.parametrize("service", SERVICES)
def test_every_xml_template_is_well_formed_as_written(service):
    """XML forbids two dashes in a row inside a comment, so a comment naming a
    flag such as the wire's no-reclaim breaks the whole template."""
    require("deploy")
    plistlib.loads(dr.template_text("launchd", service).encode("utf-8"))
    assert ET.fromstring(dr.template_text("schtasks", service)).tag.endswith("Task")


@pytest.mark.parametrize("service", SERVICES)
def test_a_task_staged_as_utf16_with_a_byte_order_mark_still_parses(service):
    """GRIP stages a task as UTF-16 with a BOM, the encoding the template declares."""
    text = _render("schtasks", service)
    assert text.startswith('<?xml version="1.0" encoding="UTF-16"?>')
    assert ET.fromstring(text.encode("utf-16")).tag.endswith("Task")


def test_the_port_can_be_chosen():
    plist = plistlib.loads(_render("launchd", "gateway", port=54321).encode("utf-8"))
    assert plist["ProgramArguments"][-2:] == ["--port", "54321"]
    assert "--port 54321 --no-reclaim\n" in _render("systemd", "wire", port=54321)
    assert "--port 54321 1&gt;&gt;" in _render("schtasks", "gateway", port=54321)


@pytest.mark.parametrize("platform,service", CASES)
def test_port_reclaim_is_off_in_every_definition(platform, service):
    """HAL main's reclaim kills whatever holds the port, even a healthy instance
    (it took the live gateway down at 18:36 UTC on 2026-10-05). A managed
    service never needs it: the reload helper starts a new instance only after
    the old one is gone and refuses when another process holds the port."""
    text = _render(platform, service)
    if service == "wire":
        assert "--no-reclaim" in text
    else:
        assert re.search(r"HAL_GATEWAY_NO_RECLAIM(</key>\s*<string>|=)1", text), text


def test_the_gateway_reads_the_variable_that_turns_reclaim_off(monkeypatch):
    """The gateway template's variable is the one HAL's gateway actually reads."""
    from lib.hal import gateway  # importing starts nothing; only __main__ serves
    monkeypatch.setenv("HAL_GATEWAY_NO_RECLAIM", "1")
    assert gateway._reclaim_enabled() is False
    monkeypatch.delenv("HAL_GATEWAY_NO_RECLAIM")
    assert gateway._reclaim_enabled() is True


def test_the_default_ports_are_the_registered_ones():
    assert dr.SERVICES["gateway"].port == 4010
    assert dr.SERVICES["wire"].port == 4011
    assert dr.SERVICES["gateway"].health == "hal-gateway"
    assert dr.SERVICES["wire"].health == "hal-wire"


def test_the_wire_ca_store_reaches_the_wire_definition():
    plist = plistlib.loads(_render("launchd", "wire", wire_ca_store="file").encode("utf-8"))
    assert plist["EnvironmentVariables"]["HAL_WIRE_CA_STORE"] == "file"


# --------------------------------------------------------------------------- #
# Checks on the machine: a venv interpreter and a HAL checkout.
# --------------------------------------------------------------------------- #
def test_a_system_interpreter_is_refused(tmp_path):
    system = tmp_path / "usr" / "bin" / "python3"
    system.parent.mkdir(parents=True)
    system.write_text("", encoding="utf-8")
    with pytest.raises(dr.RenderError, match="not inside a virtual environment"):
        dr.render("launchd", "gateway", python=str(system),
                  hal_home=str(_hal_checkout(tmp_path)), log_dir=str(tmp_path / "logs"))


def test_a_venv_interpreter_and_a_hal_checkout_are_accepted(tmp_path):
    require("deploy")
    python = _venv(tmp_path)
    text = dr.render("systemd", "wire", python=str(python),
                     hal_home=str(_hal_checkout(tmp_path)), log_dir=str(tmp_path / "logs"))
    assert f'ExecStart="{python}"' in text


def test_a_missing_interpreter_is_refused(tmp_path):
    with pytest.raises(dr.RenderError, match="does not exist"):
        dr.render("launchd", "wire", python=str(tmp_path / "venv" / "bin" / "python"),
                  hal_home=str(_hal_checkout(tmp_path)), log_dir=str(tmp_path / "logs"))


def test_a_directory_that_is_not_a_hal_checkout_is_refused(tmp_path):
    with pytest.raises(dr.RenderError, match="not a HAL checkout"):
        dr.render("launchd", "wire", python=str(_venv(tmp_path)),
                  hal_home=str(tmp_path / "empty"), log_dir=str(tmp_path / "logs"))


# --------------------------------------------------------------------------- #
# Values the formats cannot carry.
# --------------------------------------------------------------------------- #
def test_xml_special_characters_survive_a_round_trip_through_the_plist():
    home = "/opt/R&D <hal>"
    plist = plistlib.loads(_render("launchd", "gateway", hal_home=home).encode("utf-8"))
    assert plist["WorkingDirectory"] == home
    assert plist["EnvironmentVariables"]["HAL_HOME"] == home


def test_xml_special_characters_survive_a_round_trip_through_the_task():
    home = r"C:\R&D\hal"
    root = ET.fromstring(_render("schtasks", "gateway", hal_home=home))
    assert root.findtext("t:Actions/t:Exec/t:WorkingDirectory", namespaces=TASK) == home
    assert f'set "HAL_HOME={home}"' in root.findtext("t:Actions/t:Exec/t:Arguments",
                                                     namespaces=TASK)


@pytest.mark.parametrize("platform", dr.PLATFORMS)
@pytest.mark.parametrize("field", ["python", "hal_home", "log_dir"])
def test_a_control_character_is_refused_everywhere(platform, field):
    with pytest.raises(dr.RenderError, match="control character"):
        _render(platform, "gateway", **{field: _inputs(platform)[field] + "\nInjected=1"})


@pytest.mark.parametrize("platform,char", [("systemd", '"'), ("systemd", "\\"),
                                           ("systemd", "$"), ("systemd", "%"),
                                           ("schtasks", '"'), ("schtasks", "%")])
def test_characters_a_format_cannot_carry_are_refused(platform, char):
    with pytest.raises(dr.RenderError, match="cannot carry safely"):
        _render(platform, "wire", hal_home=_inputs(platform)["hal_home"] + char + "x")


@pytest.mark.parametrize("platform,field,value", [("launchd", "log_dir", "logs"),
                                                  ("systemd", "python", "venv/bin/python"),
                                                  ("schtasks", "hal_home", "/opt/hal")])
def test_a_path_that_is_not_absolute_for_its_platform_is_refused(platform, field, value):
    with pytest.raises(dr.RenderError, match="absolute path"):
        _render(platform, "gateway", **{field: value})


@pytest.mark.parametrize("port", [0, 70000, True, "4011"])
def test_a_port_outside_the_tcp_range_is_refused(port):
    with pytest.raises(dr.RenderError, match="port must be"):
        _render("launchd", "wire", port=port)


def test_a_task_needs_its_logon_user():
    with pytest.raises(dr.RenderError, match="logon user"):
        _render("schtasks", "gateway", user=None)


# --------------------------------------------------------------------------- #
# Template integrity: slots, secrets.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("platform,service", CASES)
def test_each_template_carries_exactly_the_slots_the_renderer_fills(platform, service):
    expected = SLOTS[platform] | ({"WIRE_CA_STORE"} if service == "wire" else set())
    require("deploy")
    assert dr.placeholders(dr.template_text(platform, service)) == expected


def test_a_template_slot_nothing_fills_is_refused():
    with pytest.raises(dr.RenderError, match="nothing fills"):
        dr.fill("<a>__PYTHON__ __SECRET_TOKEN__</a>", {"PYTHON": "/x"}, "launchd")


def test_a_value_with_no_slot_is_refused():
    with pytest.raises(dr.RenderError, match="no slot"):
        dr.fill("<a>__PYTHON__</a>", {"PYTHON": "/x", "PORT": "1"}, "launchd")


@pytest.mark.parametrize("platform,service", CASES)
def test_no_template_carries_anything_shaped_like_a_secret(platform, service):
    require("deploy")
    assert not SECRET_SHAPES.search(dr.template_text(platform, service))


def test_the_secret_scan_can_fail():
    """The scan above is only worth its green if it fires on a planted token."""
    assert SECRET_SHAPES.search("<string>sk-ant-" + "x" * 20 + "</string>")


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def _render_argv(platform: str, service: str, *extra: str) -> list:
    inputs = _inputs(platform)
    argv = ["render", "--platform", platform, "--service", service,
            "--python", inputs["python"], "--hal-home", inputs["hal_home"],
            "--log-dir", inputs["log_dir"], "--no-check-paths", *extra]
    if platform == "schtasks":
        argv += ["--user", inputs["user"]]
    return argv


@pytest.mark.parametrize("platform,service", CASES)
def test_cli_render_prints_exactly_the_rendered_text(capsys, platform, service):
    require("deploy")
    assert dr.main(_render_argv(platform, service)) == 0
    assert capsys.readouterr().out == _render(platform, service)


def test_cli_list_names_every_service_on_every_platform(capsys):
    assert dr.main(["list"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert set(data["platforms"]) == {"launchd", "systemd", "schtasks"}
    assert data["services"]["wire"]["names"] == {"launchd": "com.hal.wire",
                                                 "systemd": "hal-wire.service",
                                                 "schtasks": "GRIP HAL Wire"}
    assert data["services"]["gateway"]["health"] == "hal-gateway"
    assert data["services"]["wire"]["port"] == 4011


@pytest.mark.parametrize("extra,message", [(["--wire-ca-store", "file"], "wire only"),
                                           (["--user", r"EXAMPLE\jane"], "Task Scheduler only"),
                                           (["--bind", "127.0.0.1"], "brain only"),
                                           (["--tailscale-tag", ""], "brain only")])
def test_cli_refuses_an_option_the_service_would_drop(capsys, extra, message):
    assert dr.main(_render_argv("launchd", "gateway", *extra)) == 2
    assert message in capsys.readouterr().err


def test_cli_refuses_a_system_interpreter_when_it_checks_paths(capsys, tmp_path):
    system = tmp_path / "python3"
    system.write_text("", encoding="utf-8")
    argv = ["render", "--platform", "launchd", "--service", "wire", "--python", str(system),
            "--hal-home", str(_hal_checkout(tmp_path)), "--log-dir", str(tmp_path)]
    assert dr.main(argv) == 2
    assert "virtual environment" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# The brain: one systemd user unit, bound to loopback or the tailnet only.
# --------------------------------------------------------------------------- #
def _brain_argv(*extra: str, platform: str = "systemd") -> list:
    return ["render", "--platform", platform, "--service", "brain",
            "--python", BRAIN["python"], "--hal-home", BRAIN["hal_home"],
            "--log-dir", BRAIN["log_dir"], "--no-check-paths", *extra]


def test_brain_unit_matches_golden_and_has_no_placeholders():
    text = _brain()
    golden = GOLDEN / "systemd-brain.golden"
    if os.environ.get("HAL_UPDATE_GOLDEN") == "1":
        golden.write_text(text, encoding="utf-8")
    assert text == golden.read_text(encoding="utf-8")
    # Two checks on purpose: the renderer's own pattern, and one that does not use it.
    assert not dr.placeholders(text)
    assert "__" not in text


def test_brain_template_keeps_the_promises_the_other_units_keep():
    require("deploy")
    template = dr.template_text("systemd", "brain")
    assert dr.placeholders(template) == BRAIN_SLOTS
    assert not SECRET_SHAPES.search(template)
    unit = _unit(_brain())
    assert unit[("Service", "ExecStart")] == [
        f'"{BRAIN["python"]}" -m lib.hal.brain --bind {BRAIN["bind"]} --port 7416 '
        f'--state "{BRAIN["state_db"]}" --pins "{BRAIN["pins_file"]}"']
    assert unit[("Service", "Restart")] == ["always"]
    assert unit[("Service", "WorkingDirectory")] == [BRAIN["hal_home"]]
    assert unit[("Install", "WantedBy")] == ["default.target"]
    names = {value.strip('"').split("=", 1)[0] for value in unit[("Service", "Environment")]}
    assert names == BRAIN_ENV


def test_brain_checks_the_interpreter_and_the_checkout_as_the_other_services_do(tmp_path):
    require("deploy")
    python, home = _venv(tmp_path), _hal_checkout(tmp_path)
    inputs = {**BRAIN, "python": str(python), "hal_home": str(home)}
    assert f'ExecStart="{python}" -m lib.hal.brain ' in dr.render("systemd", "brain", **inputs)
    (home / "lib" / "hal" / "brain.py").unlink()
    with pytest.raises(dr.RenderError, match="not a HAL checkout"):
        dr.render("systemd", "brain", **inputs)


@pytest.mark.parametrize("field,changed,old_line,new_line", [
    ("bind", "100.64.0.2", "--bind 100.64.0.1", "--bind 100.64.0.2"),
    ("state_db", "/srv/hal/other.sqlite", '--state "/srv/hal/brain.sqlite"',
     '--state "/srv/hal/other.sqlite"'),
    ("pins_file", "/srv/hal/other-pins.json", '--pins "/srv/hal/public-pins.json"',
     '--pins "/srv/hal/other-pins.json"'),
    ("tailscale_tag", "tag:other", "HAL_GATEWAY_TS_ACL_TAG=tag:example",
     "HAL_GATEWAY_TS_ACL_TAG=tag:other")])
def test_each_brain_argument_changes_only_its_own_line(field, changed, old_line, new_line):
    before, after = _brain().splitlines(), _brain(**{field: changed}).splitlines()
    differing = [(a, b) for a, b in zip(before, after) if a != b]
    assert len(before) == len(after)
    old = next(line for line in before if old_line in line)
    assert differing == [(old, old.replace(old_line, new_line))]


@pytest.mark.parametrize("bind", [
    "0.0.0.0", "192.0.2.1", "203.0.113.7",          # every interface; public addresses
    "10.0.0.1", "172.16.0.1", "192.168.0.1", "169.254.0.1",  # a LAN is not the tailnet
    "100.63.255.255", "100.128.0.0",     # one below and one above 100.64.0.0/10
    "::", "::1", "::ffff:127.0.0.1",     # IPv6, its loopback too: the brain is IPv4 only
    "localhost", "brain.example",        # a hostname cannot be checked from here
    "127.0.0.1/8", " 127.0.0.1", "",     # not an address at all
    2130706433])                         # 127.0.0.1 as an integer: not text, so not accepted
def test_brain_refuses_a_bind_that_is_not_ipv4_loopback_or_tailnet(bind):
    with pytest.raises(dr.RenderError, match="bind must be an IPv4 address on loopback"):
        _brain(bind=bind)


@pytest.mark.parametrize("bind", ["127.0.0.1", "127.255.255.254", "100.64.0.0", "100.127.255.255"])
def test_brain_accepts_loopback_and_the_whole_tailnet_range(bind):
    assert f"--bind {bind} --port 7416 " in _brain(bind=bind)


@pytest.mark.parametrize("field", ["state_db", "pins_file"])
def test_brain_refuses_relative_paths(field):
    with pytest.raises(dr.RenderError, match="absolute path"):
        _brain(**{field: "relative.json"})


@pytest.mark.parametrize("field", ["state_db", "pins_file"])
@pytest.mark.parametrize("value,message", [(None, "the brain needs {field}"),
                                           ("", "{field} is empty")])
def test_brain_requires_state_and_pins(field, value, message):
    with pytest.raises(dr.RenderError, match=message.format(field=field)):
        _brain(**{field: value})


@pytest.mark.parametrize("field", ["state_db", "pins_file", "hal_home", "log_dir"])
def test_no_placeholder_shape_survives_in_a_brain_unit(field):
    """A value shaped like a placeholder would leave a unit that looks unfilled."""
    with pytest.raises(dr.RenderError, match="placeholder survived"):
        _brain(**{field: "/srv/__STATE_DB__"})


def test_brain_allows_an_empty_tag_on_loopback_only():
    """With no tag HAL's guard refuses every remote peer: fine on loopback, a
    brain nobody can reach on the tailnet."""
    assert 'Environment="HAL_GATEWAY_TS_ACL_TAG="\n' in _brain(bind="127.0.0.1", tailscale_tag="")
    for missing in ("", None):
        with pytest.raises(dr.RenderError, match="required for a tailnet bind"):
            _brain(tailscale_tag=missing)


@pytest.mark.parametrize("tag", ["example", "tag:", "tag:1st", "tag:a_b", "tag:a b",
                                 "TAG:example", "tag:ok\nEnvironment=INJECTED=1", 7])
def test_brain_refuses_a_tag_tailscale_itself_would_refuse(tag):
    with pytest.raises(dr.RenderError, match="tag:name"):
        _brain(tailscale_tag=tag)
    with pytest.raises(dr.RenderError, match="tag:name"):
        _brain(bind="127.0.0.1", tailscale_tag=tag)


def test_brain_keeps_the_case_of_a_tag():
    """Tailscale allows upper case in a tag and HAL's guard compares it exactly."""
    text = _brain(tailscale_tag="tag:Brain-2")
    assert 'Environment="HAL_GATEWAY_TS_ACL_TAG=tag:Brain-2"\n' in text


def test_the_brain_unit_sets_the_tag_variable_the_guard_reads():
    """The name says gateway because the brain reuses the gateway's guard. A
    unit that set any other name would leave the guard with no tag, and a brain
    that refuses every tailnet peer."""
    from lib.hal import gateway_guard  # standard library only; importing starts nothing
    assert gateway_guard._REQUIRED_TS_TAG_ENV in BRAIN_ENV
    assert f'Environment="{gateway_guard._REQUIRED_TS_TAG_ENV}=tag:example"\n' in _brain()


def test_brain_defaults_to_loopback_and_an_empty_tag():
    text = _brain(bind=None, tailscale_tag=None)
    assert "--bind 127.0.0.1 --port 7416 " in text
    assert 'Environment="HAL_GATEWAY_TS_ACL_TAG="\n' in text


def test_an_empty_tag_in_the_unit_admits_loopback_and_no_other_peer(monkeypatch):
    """What the unit's line means when no tag is given (the variable set to
    nothing): to the guard it is the same as unset. A tailnet peer is refused,
    even one whose tag list holds an empty string; a loopback caller is served."""
    from types import SimpleNamespace
    from lib.hal import gateway_guard

    def allowed(address: str, tags: list) -> bool:
        return gateway_guard.gateway_access_allowed(
            {}, token="", require_client_cert=False,
            handler=SimpleNamespace(client_address=(address, 40000)),
            whois_func=lambda _address: {"Node": {"Tags": tags}})

    monkeypatch.setenv(gateway_guard._REQUIRED_TS_TAG_ENV, "tag:example")
    assert allowed("100.64.0.2", ["tag:example"]) is True  # the control: a tagged peer gets in
    monkeypatch.setenv(gateway_guard._REQUIRED_TS_TAG_ENV, "")
    assert allowed("100.64.0.2", [""]) is False
    assert allowed("100.64.0.2", ["tag:example"]) is False
    assert allowed("127.0.0.1", []) is True


def test_brain_port_defaults_to_the_registered_one_and_can_be_chosen():
    assert dr.SERVICES["brain"].port == 7416
    assert "--bind 100.64.0.1 --port 7500 --state " in _brain(port=7500)
    with pytest.raises(dr.RenderError, match="port must be"):
        _brain(port=0)


@pytest.mark.parametrize("port", [4010, 4011, 4012])
def test_a_port_the_brain_exits_on_is_refused_before_a_unit_is_written(port, tmp_path):
    """``lib.hal.brain`` exits at once on these ports, and the unit restarts it
    always, so such a unit would loop for ever. The first half holds the
    renderer's list in step with the brain's own: the brain must exit while it
    reads its arguments, before it opens the (absent) pins file or any socket."""
    from lib.hal import brain  # standard library only at import; nothing is served
    with pytest.raises(SystemExit):
        brain.main(["--port", str(port), "--state", str(tmp_path / "absent" / "brain.sqlite"),
                    "--pins", str(tmp_path / "absent" / "pins.json")])
    assert not (tmp_path / "absent").exists()
    with pytest.raises(dr.RenderError, match=f"port {port} is reserved"):
        _brain(port=port)
    assert f"--port {port}" in _render("systemd", "gateway", port=port)  # the brain only


def test_the_brain_is_the_only_service_limited_to_some_platforms():
    """SERVICES takes every service that exists on all platforms, so those are
    tested by construction. Any other service needs tests of its own, and
    this fails until it is named here beside the brain."""
    assert set(dr.SERVICES) - set(SERVICES) == {"brain"}
    assert set(dr.SERVICES["brain"].templates) == {"systemd"}
    assert set(dr.SERVICES["brain"].names) == {"systemd"}


@pytest.mark.parametrize("platform", ["launchd", "schtasks"])
def test_brain_is_a_systemd_service_only(platform):
    with pytest.raises(dr.RenderError, match="systemd.*Linux host"):
        dr.render(platform, "brain", **{**BRAIN, "check_paths": False})
    with pytest.raises(dr.RenderError, match="systemd.*Linux host"):
        dr.template_text(platform, "brain")


@pytest.mark.parametrize("service", SERVICES)
@pytest.mark.parametrize("field,value", BRAIN_ARGS)
def test_gateway_and_wire_refuse_every_brain_argument(service, field, value):
    with pytest.raises(dr.RenderError, match="brain only"):
        _render("systemd", service, **{field: value})


def test_fill_refuses_placeholder_supplied_as_a_value_and_names_it():
    with pytest.raises(dr.RenderError, match="placeholder survived.*UNFILLED"):
        dr.fill("x=__PYTHON__", {"PYTHON": "__UNFILLED__"}, "systemd")


def test_catalogue_preserves_existing_services_and_adds_brain():
    services = dr.catalogue()["services"]
    assert services["gateway"] == {"module": "lib.hal.gateway", "port": 4010,
        "health": "hal-gateway", "names": {"launchd": "com.hal.gateway",
        "systemd": "hal-gateway.service", "schtasks": "GRIP HAL Gateway"},
        "templates": {"launchd": "com.hal.gateway.plist",
        "systemd": "hal-gateway.service", "schtasks": "hal-gateway.task.xml"}}
    assert services["wire"] == {"module": "lib.hal.wire_proxy", "port": 4011,
        "health": "hal-wire", "names": {"launchd": "com.hal.wire",
        "systemd": "hal-wire.service", "schtasks": "GRIP HAL Wire"},
        "templates": {"launchd": "com.hal.wire.plist",
        "systemd": "hal-wire.service", "schtasks": "hal-wire.task.xml"}}
    assert services["brain"] == {"module": "lib.hal.brain", "port": 7416,
        "health": "hal-brain", "names": {"systemd": "hal-brain.service"},
        "templates": {"systemd": "hal-brain.service"}}


def test_cli_renders_brain_with_explicit_arguments(capsys):
    require("deploy")
    assert dr.main(_brain_argv(*BRAIN_CLI)) == 0
    assert capsys.readouterr().out == _brain()


@pytest.mark.parametrize("argv,message", [
    (_brain_argv(*BRAIN_CLI, platform="launchd"), "systemd (Linux host) service"),
    (_brain_argv("--pins-file", BRAIN["pins_file"]), "the brain needs state_db (--state-db)"),
    (_brain_argv("--state-db", BRAIN["state_db"]), "the brain needs pins_file (--pins-file)"),
    (_brain_argv(*BRAIN_CLI, "--bind", "0.0.0.0"), "bind must be an IPv4 address"),
    (_brain_argv(*BRAIN_CLI, "--bind", "::1"), "bind must be an IPv4 address")])
def test_cli_refuses_an_unsafe_or_incomplete_brain_and_prints_no_unit(capsys, argv, message):
    assert dr.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert message in captured.err
