"""osa_fleet — parallel multi-host script dispatch.

γ-3 of the osascript control layer plan. Fans out a plan of
``{host_key: (script_name, params)}`` dispatches across
:class:`~lib.hal.osa.transport.LocalTransport` and
:class:`~lib.hal.osa.transport.SSHTransport` in parallel via
:class:`concurrent.futures.ThreadPoolExecutor`.

Each host result is independent — a failing host never cancels other
in-flight dispatches. Failures are captured as :class:`Exception` values
in the returned dict rather than being re-raised.

Usage::

    from lib.hal.osa.fleet import osa_fleet

    results = osa_fleet({
        "local":          ("mute", {}),
        "100.64.0.10": ("notification", {"title": "Heads up", "body": "Shipping"}),
    })
    for host, outcome in results.items():
        if isinstance(outcome, Exception):
            print(f"{host}: FAILED — {outcome}")
        else:
            print(f"{host}: ok={outcome.ok}, stdout={outcome.stdout!r}")

Host keys:
    ``"local"`` → :class:`LocalTransport` (runs on this Mac).
    Any other string → :class:`SSHTransport` using the key as both the
    logical host name and SSH target. Override with a Tailscale IP to
    route correctly when the logical name is not in SSH ``known_hosts``.
"""
from __future__ import annotations

import concurrent.futures
from typing import Callable, Dict, Optional, Tuple, Union

from lib.hal.osa.auth import CapabilityToken
from lib.hal.osa.dispatch import osa
from lib.hal.osa.osa import OsaResult
from lib.hal.osa.provenance import ProvenanceType
from lib.hal.osa.transport import LocalTransport, SSHTransport, Transport

__all__ = ["osa_fleet", "default_transport_factory", "FleetPlan", "FleetResult"]

FleetPlan = Dict[str, Tuple[str, Dict[str, object]]]
FleetResult = Dict[str, Union[OsaResult, Exception]]
TransportFactory = Callable[[str], Transport]

_LOCAL = "local"


def default_transport_factory(host_key: str) -> Transport:
    """Map a host key to a :class:`Transport`.

    ``"local"`` → :class:`LocalTransport`; anything else →
    :class:`SSHTransport` with the key as IP and logical name.
    Pass a custom factory to :func:`osa_fleet` to override (useful in tests
    or for non-standard SSH configs).
    """
    if host_key == _LOCAL:
        return LocalTransport()
    return SSHTransport(host_name=host_key, ip=host_key)


def _dispatch_one(
    host_key: str,
    script_name: str,
    params: Dict[str, object],
    *,
    transport_factory: TransportFactory,
    provenance: ProvenanceType,
    capability_token: Optional[CapabilityToken],
    timeout_s: Optional[int],
    audit: bool,
) -> OsaResult:
    transport = transport_factory(host_key)
    return osa(
        script_name,
        params,
        transport=transport,
        provenance=provenance,
        capability_token=capability_token,
        timeout_s=timeout_s,
        audit=audit,
    )


def osa_fleet(
    plan: FleetPlan,
    *,
    provenance: ProvenanceType = ProvenanceType.LOCAL,
    capability_token: Optional[CapabilityToken] = None,
    timeout_s: Optional[int] = None,
    audit: bool = True,
    max_workers: Optional[int] = None,
    transport_factory: TransportFactory = default_transport_factory,
) -> FleetResult:
    """Dispatch named scripts to multiple hosts in parallel.

    Args:
        plan: ``{host_key: (script_name, params_dict)}``. Use ``"local"``
            for the local Mac; any other key is routed via SSH.
        provenance: Propagated to each :func:`~lib.hal.osa.dispatch.osa`
            call and recorded in the audit log.
        capability_token: Passed to each dispatch. Required for tier-2+
            local scripts; remote dispatches skip local HMAC verification.
        timeout_s: Per-script timeout override (seconds). ``None`` uses
            each script's registry default.
        audit: Write one audit entry per dispatch when ``True`` (default).
        max_workers: Thread-pool size. Defaults to ``len(plan)`` so all
            dispatches can be in-flight simultaneously. Cap this for large
            fleets to bound concurrency.
        transport_factory: Callable ``(host_key) → Transport``. Defaults to
            :func:`default_transport_factory`. Override in tests or for
            non-standard SSH configs without patching internals.

    Returns:
        Dict mapping each host key to its :class:`~lib.hal.osa.osa.OsaResult`
        on success or the :class:`Exception` it raised on failure. Every
        host in the plan appears in the returned dict exactly once.

    Raises:
        Nothing — all per-host exceptions are captured into the result dict.
    """
    if not plan:
        return {}

    workers = max_workers if max_workers is not None else len(plan)
    results: FleetResult = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _dispatch_one,
                host_key, script_name, params,
                transport_factory=transport_factory,
                provenance=provenance,
                capability_token=capability_token,
                timeout_s=timeout_s,
                audit=audit,
            ): host_key
            for host_key, (script_name, params) in plan.items()
        }
        for future in concurrent.futures.as_completed(futures):
            host_key = futures[future]
            try:
                results[host_key] = future.result()
            except Exception as exc:  # noqa: BLE001 — fleet isolation by design
                results[host_key] = exc

    return results
