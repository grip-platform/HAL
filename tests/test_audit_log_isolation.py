"""The suite-wide HAL_HOME isolation in ``tests/conftest.py`` is real.

The failover tests used to append fixture rows to the operator's live
``~/.hal/logs/gateway-audit.jsonl`` (see the conftest docstring for the
measurement). This drives a failover through the DEFAULT telemetry sink, the
path those tests took, and reads back the file the sink actually wrote.

It is guarded so it can never write a real log itself: the home it is about to
write into is checked before the proxy runs, in the shared ``isolated_hal_home``
fixture (``tests/conftest.py``). With the conftest disabled (``--noconftest``)
and ``HAL_HOME`` unset, the fixture's first assertion fails before any write,
which is how this test was shown able to fail.
"""
from __future__ import annotations

import json

from lib.hal.gateway_failover import FailoverProxy, UpstreamSpec


class _Answer:
    """A minimal upstream response: a status and close()."""

    def __init__(self, status):
        self.status = status
        self.closed = False

    def close(self):
        self.closed = True


class _Upstream:
    """A fake same-format upstream that always answers ``status``."""

    def __init__(self, status):
        self.status = status

    def forward(self, path, body, client_headers, *, method="POST", max_retries=None):
        return _Answer(self.status)


def test_default_sink_writes_into_the_per_test_home_not_the_real_one(isolated_hal_home):
    # isolated_hal_home checks, BEFORE anything is written, that the effective
    # HAL home is this test's own and not the live ~/.hal -- so a broken
    # isolation fails here and never touches whatever home the environment names.
    home = isolated_hal_home

    upstreams = {"first": _Upstream(429), "second": _Upstream(200)}
    proxy = FailoverProxy(
        [UpstreamSpec("first"), UpstreamSpec("second")],
        proxy_factory=lambda spec: upstreams[spec.name],
    )  # telemetry=None: the default sink, exactly as the failover tests use it
    assert proxy.forward("/v1/messages", b"{}", []).status == 200

    log = home / "logs" / "gateway-audit.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(r["event"], r["upstream"]) for r in rows] == [
        ("failover", "first"), ("served", "second")]
    assert rows[-1]["failed_over"] is True
