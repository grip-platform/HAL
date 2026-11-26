"""gateway_brake — the hal/infinite operator kill switch (ADR-0002 W0).

A per-request brake for HAL InfiniteMachine features, mirroring GRIP's
``bin/grip-stop`` marker pattern (``state/session/outbound-stop.json``):
presence of the marker file IS the engaged state — deterministic,
file-backed, no daemon, no network. FAIL-CLOSED ON MARKER: a marker that
exists but cannot be read (corrupt, permission-broken) still engages the
brake; only the file's ABSENCE releases it.

Scope, stated precisely so the brake can never be mistaken for a relay
kill: the brake gates InfiniteMachine ABSORPTION/serving features only
(the W3 tee must call :func:`brake_engaged` before every absorb; Mode R
must refuse to serve rebuilt bodies while engaged). The verbatim
``/v1/messages`` relay is deliberately NOT gated — braking the relay would
kill live sessions, and the safe state is "plain witnessed gateway", not
"no gateway". ``GET /health`` reports the brake state so engagement is
verifiable in one request.

Marker path resolution: ``$HAL_INFINITE_BRAKE_MARKER`` when set, else
``${CLAUDE_DIR:-~/.claude}/state/session/infinite-stop.json`` — a sibling
of grip-stop's ``outbound-stop.json``, produced by GRIP-side tooling
(``bin/grip-infinite-stop``). The two markers are distinct files by
design: the WhatsApp outbound brake and the InfiniteMachine brake must
never trip each other.

The marker pattern is one small class, :class:`MarkerBrake`, so another brake
reuses it with a marker of its own. The second is the wire's egress brake
(``lib/hal/wire_proxy.py``): ``$HAL_WIRE_EGRESS_BRAKE_MARKER`` when set, else
``${CLAUDE_DIR:-~/.claude}/state/session/wire-egress-brake.json``, produced by
GRIP's ``bin/grip-wire-brake``. Distinct markers, so no brake trips another.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ENV_MARKER = "HAL_INFINITE_BRAKE_MARKER"
_DEFAULT_REL = ("state", "session", "infinite-stop.json")


def _present(path: Path) -> bool:
    """Whether a marker is present, fail-closed: only "no such file" releases.

    ``Path.exists()`` cannot decide this. For a path the process may not
    inspect, Python 3.12 and 3.13 raise PermissionError but 3.14 answers False
    (measured 2026-10-05), which would release a brake whose state cannot be
    established. ``os.stat`` raises on every Python, so the outcome is the same
    on all of them.
    """
    try:
        os.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError:
        return True
    return True


@dataclass(frozen=True)
class MarkerBrake:
    """A brake whose engaged state is the presence of one marker file.

    ``env_key`` names the variable that overrides the marker's path; otherwise
    the marker is ``rel`` under ``${CLAUDE_DIR:-~/.claude}``. The InfiniteMachine
    brake below is one; the wire's egress brake (``lib/hal/wire_proxy.py``) is
    another, with its own marker, so neither can trip the other.
    """
    env_key: str
    rel: tuple

    def path(self, env: Optional[dict] = None) -> Path:
        """Resolve the marker path (env override → GRIP-sibling default)."""
        env = os.environ if env is None else env
        explicit = (env.get(self.env_key) or "").strip()
        if explicit:
            return Path(explicit).expanduser()
        base = (env.get("CLAUDE_DIR") or "").strip() or str(Path.home() / ".claude")
        return Path(base).joinpath(*self.rel)

    def engaged(self, env: Optional[dict] = None) -> bool:
        """Presence == engaged. Fail-closed on the marker: existence engages the
        brake even when the file is unreadable or corrupt; only absence
        releases. An unstattable path also reads as engaged — when the brake
        state cannot be established, the braked behaviour stays on."""
        return _present(self.path(env))

    def state(self, env: Optional[dict] = None) -> dict:
        """The brake as reportable data for /health: engaged flag, marker path,
        and the marker's JSON body when it parses (None when it does not — a
        corrupt marker still brakes, it just cannot explain itself)."""
        p = self.path(env)
        engaged = _present(p)
        state = {"engaged": engaged, "marker": str(p)}
        if engaged:
            try:
                state["detail"] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                state["detail"] = None
        return state


INFINITE_BRAKE = MarkerBrake(ENV_MARKER, _DEFAULT_REL)


def marker_path(env: Optional[dict] = None) -> Path:
    """The InfiniteMachine brake's marker path."""
    return INFINITE_BRAKE.path(env)


def brake_engaged(env: Optional[dict] = None) -> bool:
    """Whether the InfiniteMachine brake is engaged (see ``MarkerBrake.engaged``)."""
    return INFINITE_BRAKE.engaged(env)


def brake_state(env: Optional[dict] = None) -> dict:
    """The InfiniteMachine brake as reportable data for /health."""
    return INFINITE_BRAKE.state(env)
