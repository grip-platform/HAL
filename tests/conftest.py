"""Suite-wide isolation: no test may write into the operator's live HAL home.

Why this exists, measured on 2026-10-05 rather than assumed.
``lib/hal/gateway_failover.py::_default_sink`` appends every gateway audit event
to ``$HAL_HOME/logs/gateway-audit.jsonl``, and ``HAL_HOME`` defaults to
``~/.hal``, the directory the live gateway (launchd ``com.hal.server``) writes
its own audit log into. The failover tests build ``FailoverProxy`` with
``telemetry=None`` (``tests/test_gateway_failover.py::_fp`` and
``tests/test_gateway_any_harness_failover.py::_make_fp``), so the default sink
ran, and every full-suite run with ``HAL_HOME`` unset (or naming the live home)
appended FIXTURE rows to that live log:

* one full run writes 35 audit rows (7 ``failover``, 4 ``unreachable``,
  12 ``served``, 12 ``gateway.chat.witness``), plus 12 rows on the HMAC-signed
  chat chain and the chain's key file, all under ``$HAL_HOME/logs``;
* the live log held 437 rows naming the tests' fake upstreams ``a``, ``b``,
  ``primary`` and ``secondary``: 133 ``failover``, 76 ``unreachable`` and
  228 ``served``. 437 is exactly 19 runs of 23 such rows, and every one of the
  133 ``failover`` events in that log came from a test, not from traffic.

So a reader of the log (``bin/hal-cost-report``, GRIP's
``grip-infinite-witness``) saw failovers that never happened.

The fix is one autouse fixture: every test gets its own empty HAL home under
pytest's ``tmp_path``, so anything resolving ``HAL_HOME`` at call time writes
there instead. ``tests/test_audit_log_isolation.py`` pins it.

With this fixture in place a full run (6,734 tests, ``HAL_HOME`` and
``HAL_ROOT`` at the checkout as CI sets them, an empty ``HOME``) gives the same
62 failures and 6,629 passes as two runs without it, and writes nothing under
the starting ``HAL_HOME/logs``.

What this does not cover, measured the same day: a module that reads
``HAL_HOME`` once, at import, into a module constant keeps the value it saw when
the test modules were collected. A full run with this fixture still wrote
``state/hal/limits.json`` (``sentinel.py``; also ``key_failover.py``, until the
spend guard removed it, Step 5h),
``state/hal/tasks/`` (``background_executor.py``), ``state/hal/benchmarks/``
(``benchmark.py``) and ``state/hal/vault-index.json`` (``obsidian_vault.py``)
under the ``HAL_HOME`` the run started with. Where that is the live ``~/.hal``,
those writes still reach it; moving them needs those constants resolved at call
time, which is separate work.

No test may read the operator's Claude token store either (added 2026-10-05,
InfiniteMachine v2 Step 5b). The wire and the gateway now put each request on
its account by reading GRIP's token pool (``lib/hal/account_ledger.py``), at
call time, from the path ``GRIP_RELAY_CCH_OAUTH_PATH`` names -- the override GRIP
itself honours. Pointing it at a path that does not exist makes every pool read
in a test come back empty unless that test builds its own store, so no real
token is ever read into a test. ``tests/test_account_ledger.py`` pins it.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

#: GRIP's override of its token store's active file (see the docstring).
TOKEN_STORE_ENV = "GRIP_RELAY_CCH_OAUTH_PATH"


@pytest.fixture(autouse=True)
def _isolated_hal_home(tmp_path, monkeypatch):
    """Point ``HAL_HOME`` at a fresh, empty directory for the length of one test,
    and GRIP's token store at a path that does not exist."""
    home = tmp_path / "hal-home"
    home.mkdir()
    monkeypatch.setenv("HAL_HOME", str(home))
    monkeypatch.setenv(TOKEN_STORE_ENV,
                       str(tmp_path / "no-claude-token-store" / "cch-oauth-token"))
    return home


@pytest.fixture(autouse=True)
def _isolated_converge_registry(tmp_path, monkeypatch):
    """Keep the converge kernel's fixed-point registry and telemetry out of the
    operator's ``~/.claude`` for the length of one test (HAL#885).

    ``lib/improve_kernel.py`` binds both paths from ``Path.home()`` at import, so
    the HAL_HOME fixture above does not reach them. A test that converges with an
    empty AGORA roster leaves the state unchanged, and the kernel then registers
    the empty state's fingerprint as a fixed point. From then on every AGORA review
    on that machine that reads the same registry, live ones included, stops before
    its first speaker. Measured 2026-11-25: one keyless run of the AGORA tests
    silenced in-session AGORA until that registry line was removed.
    """
    import lib.improve_kernel as kernel
    monkeypatch.setattr(kernel, "_DEFAULT_FIXED_POINTS", tmp_path / "converge-fixed-points.jsonl")
    monkeypatch.setattr(kernel, "_DEFAULT_TELEMETRY", tmp_path / "improve-kernel.jsonl")


def _effective_hal_home() -> Path:
    """The HAL home a sink would write into, resolved the same way the sinks do
    (``HAL_HOME``, defaulting to ``~/.hal``)."""
    return Path(os.environ.get("HAL_HOME", str(Path.home() / ".hal"))).resolve()


@pytest.fixture
def isolated_hal_home(tmp_path) -> Path:
    """The effective HAL home, asserted to be this test's own, never the live one.

    Shared by every test that pins the suite-wide isolation (the autouse
    ``_isolated_hal_home`` above), so the check cannot be bypassed by editing
    only one test file. Returns the resolved home.
    """
    home = _effective_hal_home()
    assert home != (Path.home() / ".hal").resolve(), "HAL_HOME is the live home"
    assert tmp_path.resolve() in home.parents, f"HAL_HOME {home} is outside this test's tmp_path"
    return home


#: Allows every paid source: "*" opted in. HAL's own API-key path also needs a
#: daily cap with room left (spend_guard.decide_within_cap), so it gets one far
#: above anything a test records.
PERMISSIVE_SPEND_POLICY = ('{"paid_opt_in": ["*"], '
                           '"daily_caps_usd": {"anthropic-api-key": 1000000}}')


@pytest.fixture(autouse=True)
def _permissive_spend_guard(tmp_path, monkeypatch):
    """Let this suite's made-up credentials through the spend guard.

    Most of the suite drives the gateway, the wire and the native provider with
    invented keys (an ``x-api-key`` of ``sk-ant-xxx`` and the like) to test
    relaying, not billing. Under the real policy (``config/spend_guard.json``)
    the guard refuses every one of them, which is its job, so each test gets a
    policy that allows every paid source and an empty set of spent accounts.
    The guard's own tests (``tests/test_spend_guard*.py``) replace it with a
    strict one and clear the OpenAI opt-in where relevant: they are what goes
    red when a billing path stops asking.
    """
    from lib.hal import spend_guard
    monkeypatch.setenv("GRIP_CODEX_ALLOW_PAID", "1")
    path = tmp_path / "spend-guard-permissive.json"
    path.write_text(PERMISSIVE_SPEND_POLICY, encoding="utf-8")
    monkeypatch.setenv(spend_guard.CONFIG_ENV, str(path))
    monkeypatch.setattr(spend_guard, "_SPENT", {})
    monkeypatch.setattr(spend_guard, "_UNRECORDED", set())
    return path


#: The keychain item holding HAL's own Anthropic API key
#: (``lib/hal/providers/anthropic_native.py`` API_KEY_SERVICE).
HAL_API_KEY_ITEM = "hal-anthropic"
#: The keychain item holding TypeSafe's key (config/decision_models.yaml), fenced
#: the same way: on a machine that holds it, a test must never read it.
TYPESAFE_KEY_ITEM = "grip-typesafe"
_FENCED_ITEMS = frozenset({HAL_API_KEY_ITEM, TYPESAFE_KEY_ITEM})
#: The environment fallback for TypeSafe's key (lib/hal/providers/openai_compat.py).
_FENCED_ENV = ("TYPESAFE_API_KEY", "GRIP_TYPESAFE_API_KEY")


def _asks_for_hal_api_key(args) -> bool:
    """Is this a ``security`` command about a fenced item (:data:`_FENCED_ITEMS`)?"""
    words = [str(word) for word in args] if isinstance(args, (list, tuple)) else str(args).split()
    return bool(words) and os.path.basename(words[0]) == "security" and bool(_FENCED_ITEMS & set(words))


@pytest.fixture(autouse=True)
def _hal_api_key_absent(monkeypatch):
    """In every test, the keychain holds no ``hal-anthropic`` or ``grip-typesafe`` item,
    and the environment carries no TypeSafe key.

    The permissive policy above opts HAL's key in and gives it a cap, so the one
    condition left before HAL's own API-key route opens is that keychain item
    (``lib/hal/anthropic_key_route.py``). A real probe would open the route on a
    machine that holds it, and a test that then reached the native provider would
    read the real key and send it. So ``security`` is answered "not found" (exit
    44) for that item, the probe and the read alike; every other command reaches
    the real ``subprocess.run`` unchanged, live opt-in probes included.

    The fence sits on ``subprocess.run`` itself, the one object every copy of a
    module shares, because some test modules evict and re-import ``lib.hal`` at
    collection (``tests/test_cost_router.py``), and a patch on a module attribute
    can miss the copy the code under test uses. A test opens the route by
    installing its own ``subprocess.run``, as ``tests/test_anthropic_key_route.py``
    does, together with a stub of ``urllib.request.urlopen``.
    """
    real_run = subprocess.run

    def run(args, *rest, **kwargs):
        if not _asks_for_hal_api_key(args):
            return real_run(args, *rest, **kwargs)
        text = kwargs.get("text") or kwargs.get("universal_newlines") or kwargs.get("encoding")
        empty = "" if text else b""
        return subprocess.CompletedProcess(args, 44, stdout=empty, stderr=empty)

    monkeypatch.setattr(subprocess, "run", run)
    for name in _FENCED_ENV:
        monkeypatch.delenv(name, raising=False)
