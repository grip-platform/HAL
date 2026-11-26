"""Flag-seam tests for the HAL_LEARNED_LATTICE wiring in grip_runtime (HAL #420).

The load-bearing invariant: with the flag UNSET, ``_ladder_addresses()``
returns the exact ``GRIP_LADDER_ADDRESSES`` tuple object (identity), the
fail-safe primary is byte-identical to pre-#420, and ``lib.hal.learned_lattice``
is never imported. With the flag SET, the primary/order come from the learned
lattice and remain a permutation of the input (no rung dropped).
"""
import os
import subprocess
import sys

import lib.hal.grip_runtime as gr
from lib.hal.raillm_ladder import GRIP_LADDER_ADDRESSES


# --- flag OFF: byte-identical default --------------------------------------
def test_flag_off_ladder_addresses_is_identity(monkeypatch):
    """Flag unset => _ladder_addresses() returns the SAME tuple object
    (``is``), not a copy — true byte-identity."""
    monkeypatch.delenv("HAL_LEARNED_LATTICE", raising=False)
    assert gr._ladder_addresses() is GRIP_LADDER_ADDRESSES


def test_flag_off_explicit_zero_is_identity(monkeypatch):
    """Any value other than exactly "1" is treated as off (== "1" gate)."""
    monkeypatch.setenv("HAL_LEARNED_LATTICE", "0")
    assert gr._ladder_addresses() is GRIP_LADDER_ADDRESSES
    monkeypatch.setenv("HAL_LEARNED_LATTICE", "")
    assert gr._ladder_addresses() is GRIP_LADDER_ADDRESSES


def test_flag_off_resolve_primary_unchanged(monkeypatch):
    """Flag unset, no marker => _resolve_primary(None) is the fixed ladder
    primary, byte-identical to pre-#420."""
    monkeypatch.delenv("HAL_LEARNED_LATTICE", raising=False)
    monkeypatch.setattr(gr, "_active_marker", lambda: None)
    assert gr._resolve_primary(None) == GRIP_LADDER_ADDRESSES[0]
    # explicit model still always wins.
    assert gr._resolve_primary("custom/model") == "custom/model"


def test_flag_off_does_not_import_learned_lattice():
    """In a fresh interpreter with the flag UNSET, importing grip_runtime and
    resolving the primary must NOT pull in lib.hal.learned_lattice — the
    flag-off executed path carries zero new import cost (subprocess so the
    sibling test module's import cannot pollute sys.modules)."""
    code = (
        "import sys; "
        "import lib.hal.grip_runtime as gr; "
        "gr._active_marker = lambda: None; "
        "addrs = gr._ladder_addresses(); "
        "prim = gr._resolve_primary(None); "
        "assert 'lib.hal.learned_lattice' not in sys.modules, "
        "'flag-off pulled in learned_lattice'; "
        "from lib.hal.raillm_ladder import GRIP_LADDER_ADDRESSES as G; "
        "assert addrs is G and prim == G[0]; "
        "print('OK')"
    )
    env = dict(os.environ)
    env.pop("HAL_LEARNED_LATTICE", None)
    env["PYTHONPATH"] = os.getcwd()
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=os.getcwd(), env=env)
    assert out.returncode == 0, out.stderr
    assert "OK" in out.stdout


# --- flag ON: learned ordering ---------------------------------------------
def test_flag_on_uses_learned_order(monkeypatch):
    """Flag set => _ladder_addresses() is the learned permutation (every rung
    preserved, primary first)."""
    monkeypatch.setenv("HAL_LEARNED_LATTICE", "1")
    addrs = gr._ladder_addresses()
    assert sorted(addrs) == sorted(GRIP_LADDER_ADDRESSES)
    assert addrs[0] == GRIP_LADDER_ADDRESSES[0]
    assert isinstance(addrs, tuple)


def test_flag_on_resolve_primary_from_lattice(monkeypatch):
    """Flag set, no marker => primary is the learned-ladder head."""
    monkeypatch.setenv("HAL_LEARNED_LATTICE", "1")
    monkeypatch.setattr(gr, "_active_marker", lambda: None)
    from lib.hal.learned_lattice import learned_ladder_order
    assert gr._resolve_primary(None) == learned_ladder_order(GRIP_LADDER_ADDRESSES)[0]


def test_flag_on_imports_learned_lattice(monkeypatch):
    """Flag set => the lattice module IS imported (proves the guarded import
    fires only on the flag-on branch)."""
    monkeypatch.setenv("HAL_LEARNED_LATTICE", "1")
    gr._ladder_addresses()
    assert "lib.hal.learned_lattice" in sys.modules
