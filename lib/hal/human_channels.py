"""HAL Human Channel Gateway — a gateway over human channels.

Person-side mirror of HAL's LLM-provider gateway. One human, many channels;
one API to reach them on the right one; a relay that bridges a message from one
of a person's channels to another (the "Discord -> WhatsApp" case).

Built ON TOP of lib/hal/channels.ChannelRouter (transport plumbing) — it does
NOT replace it. The gateway resolves a (peer, channel) to a MessageEnvelope and
hands it to the existing ChannelRouter.send.

Contract, routing semantics, and safety rails: docs/human-channel-gateway.md.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from lib.hal.channels import MessageEnvelope
from lib.hal.channel_format import ChannelFormatTranscoder


# --- path resolution ---------------------------------------------------------

def hal_home() -> Path:
    """Repo root that ships this code, resolved from this file (NOT $HAL_HOME).

    config/ and state/ travel with the code, so a worktree or alternate checkout
    resolves to its OWN tree instead of a globally-pinned $HAL_HOME pointing at a
    different tree (the bug a worktree dry-run surfaced).
    """
    # <root>/lib/hal/human_channels.py -> parents[2] == <root>
    return Path(__file__).resolve().parents[2]


def default_peers_path() -> Path:
    """Peer registry path: $HAL_PEERS_CONFIG override (purpose-specific), else
    config/peers.json beside the shipping code."""
    env = os.environ.get("HAL_PEERS_CONFIG")
    return Path(env) if env else hal_home() / "config" / "peers.json"


# --- errors (all fail-closed) ------------------------------------------------

class HumanChannelError(Exception):
    """Base class for gateway errors."""


class PeerNotFound(HumanChannelError, KeyError):
    """No peer with that id/alias is configured."""


class ChannelNotConfigured(HumanChannelError, KeyError):
    """The peer has no endpoint on the requested platform."""


class AmbiguousChannel(HumanChannelError):
    """'auto' could not resolve to a single channel and we refuse to guess."""


# --- peer + registry ---------------------------------------------------------

@dataclass(frozen=True)
class Peer:
    """A person addressable across one or more channel endpoints.

    Every `address` is an INDIVIDUAL handle (user id / phone JID / DM user id),
    never a group — this is the structural guarantee behind the 1:1 rail.
    """
    id: str
    name: str
    endpoints: Mapping[str, str]
    engagement_protocol: Optional[str] = None
    default_channel: Optional[str] = None
    aliases: tuple = ()

    def channels(self) -> tuple:
        return tuple(sorted(self.endpoints))

    def has(self, platform: str) -> bool:
        return platform in self.endpoints

    def endpoint(self, platform: str) -> str:
        try:
            return self.endpoints[platform]
        except KeyError:
            raise ChannelNotConfigured(
                f"peer {self.id!r} has no {platform!r} endpoint "
                f"(configured: {', '.join(self.channels()) or 'none'})"
            )

    @classmethod
    def from_dict(cls, peer_id: str, d: Mapping) -> "Peer":
        endpoints = dict(d.get("endpoints") or {})
        if not endpoints:
            raise HumanChannelError(f"peer {peer_id!r} has no endpoints")
        return cls(
            id=peer_id,
            name=d.get("name", peer_id),
            endpoints=endpoints,
            engagement_protocol=d.get("engagement_protocol"),
            default_channel=d.get("default_channel"),
            aliases=tuple(d.get("aliases") or ()),
        )


class PeerRegistry:
    """Resolves a peer id or alias to a Peer. Adding a person is config-only."""

    def __init__(self, peers):
        peers = tuple(peers)
        self._by_id: dict[str, Peer] = {p.id: p for p in peers}
        # One pass; each peer contributes its alias->id pairs via a single update
        # (O(total aliases), no nested loop).
        self._alias: dict[str, str] = {}
        for p in peers:
            self._alias.update(dict.fromkeys(p.aliases, p.id))

    @classmethod
    def from_dict(cls, data: Mapping) -> "PeerRegistry":
        peers = [Peer.from_dict(pid, pd) for pid, pd in (data.get("peers") or {}).items()]
        return cls(peers)

    @classmethod
    def from_config(cls, path=None) -> "PeerRegistry":
        path = Path(path) if path else default_peers_path()
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def get(self, key: str) -> Peer:
        if key in self._by_id:
            return self._by_id[key]
        if key in self._alias:
            return self._by_id[self._alias[key]]
        raise PeerNotFound(
            f"no peer configured for {key!r} "
            f"(known: {', '.join(sorted(self._by_id)) or 'none'})"
        )

    def peers(self) -> tuple:
        return tuple(self._by_id[k] for k in sorted(self._by_id))


# --- presence (which channel the peer was last active on) --------------------

class PresenceStore:
    """Interface: where was a peer last active? A hint for 'auto' routing."""

    def last_active(self, peer_id: str) -> Optional[str]:
        return None

    def record(self, peer_id: str, platform: str, ts: float) -> None:
        pass


class InMemoryPresence(PresenceStore):
    def __init__(self):
        self._d: dict[str, str] = {}

    def last_active(self, peer_id: str) -> Optional[str]:
        return self._d.get(peer_id)

    def record(self, peer_id: str, platform: str, ts: float) -> None:
        self._d[peer_id] = platform


class JsonPresence(PresenceStore):
    """Persistent presence in a JSON file. Fail-open: presence is a hint, never
    a blocker — any IO error degrades to 'no signal', never raises."""

    def __init__(self, path):
        self._path = Path(path)

    def _read(self) -> dict:
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    def last_active(self, peer_id: str) -> Optional[str]:
        rec = self._read().get(peer_id)
        return rec.get("platform") if isinstance(rec, dict) else None

    def record(self, peer_id: str, platform: str, ts: float) -> None:
        try:
            data = self._read()
            data[peer_id] = {"platform": platform, "ts": ts}
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, sort_keys=True)
        except Exception:
            pass  # presence is best-effort; never break a send


# --- receipt -----------------------------------------------------------------

@dataclass(frozen=True)
class Receipt:
    peer_id: str
    platform: str
    address: str
    ok: bool
    dry_run: bool = False
    detail: str = ""
    transport_result: Any = None
    ts: float = field(default_factory=time.time)


# --- the gateway -------------------------------------------------------------

def _coerce_inbound(inbound) -> tuple:
    """Normalise a relay input to (from_platform, text)."""
    if isinstance(inbound, MessageEnvelope):
        return inbound.transport, inbound.body
    if isinstance(inbound, tuple) and len(inbound) == 2:
        return inbound[0], inbound[1]
    if isinstance(inbound, Mapping):
        platform = inbound.get("from_platform") or inbound.get("transport")
        text = inbound.get("text", inbound.get("body", ""))
        return platform, text
    raise HumanChannelError(
        "inbound must be a MessageEnvelope, a (platform, text) tuple, or a mapping"
    )


class HumanChannelGateway:
    """Address a person; pick the channel; transcode; send or relay.

    All resolution is fail-closed — an unknown peer, an absent endpoint, or an
    ambiguous 'auto' raises rather than guessing or sending to a wrong place.
    """

    def __init__(self, registry: PeerRegistry, router, *,
                 presence: PresenceStore = None,
                 transcoder: ChannelFormatTranscoder = None,
                 now: Callable[[], float] = None):
        self._registry = registry
        self._router = router
        self._presence = presence if presence is not None else InMemoryPresence()
        self._transcoder = transcoder or ChannelFormatTranscoder()
        self._now = now or time.time

    # -- channel resolution --------------------------------------------------

    def _resolve_channel(self, peer: Peer, channel: str) -> str:
        if channel and channel != "auto":
            if not peer.has(channel):
                raise ChannelNotConfigured(
                    f"peer {peer.id!r} has no {channel!r} endpoint "
                    f"(configured: {', '.join(peer.channels())})"
                )
            return channel
        last = self._presence.last_active(peer.id)
        if last and peer.has(last):
            return last
        if peer.default_channel and peer.has(peer.default_channel):
            return peer.default_channel
        chans = peer.channels()
        if len(chans) == 1:
            return chans[0]
        raise AmbiguousChannel(
            f"channel='auto' for peer {peer.id!r} is ambiguous across "
            f"{', '.join(chans)} (no presence, no default_channel); "
            f"pass channel= explicitly"
        )

    def _resolve_relay_target(self, peer: Peer, from_platform, to: str) -> str:
        if to and to != "auto":
            if not peer.has(to):
                raise ChannelNotConfigured(
                    f"peer {peer.id!r} has no {to!r} endpoint "
                    f"(configured: {', '.join(peer.channels())})"
                )
            return to
        others = [c for c in peer.channels() if c != from_platform]
        if len(others) == 1:
            return others[0]
        raise AmbiguousChannel(
            f"relay target 'auto' for peer {peer.id!r} from {from_platform!r} is "
            f"ambiguous across {', '.join(others) or 'none'}; pass to= explicitly "
            f"(we never fan a private message out to several channels)"
        )

    # -- core send -----------------------------------------------------------

    def _emit(self, peer: Peer, platform: str, text: str, *,
              source_platform, dry_run: bool) -> Receipt:
        address = peer.endpoint(platform)  # fail-closed
        body = self._transcoder.transcode(
            text, from_platform=source_platform, to_platform=platform
        )
        if dry_run:
            return Receipt(
                peer.id, platform, address, ok=True, dry_run=True,
                detail=f"dry-run: would send to {peer.name} via {platform} ({address})",
            )
        meta = {"peer": peer.id, "engagement_protocol": peer.engagement_protocol}
        if source_platform is not None:  # only a relay carries a source channel
            meta["source_platform"] = source_platform
        envelope = MessageEnvelope(
            transport=platform,
            external_id=peer.id,
            thread_id=address,
            body=body,
            direction="outbound",
            meta=meta,
        )
        result = self._router.send(envelope)
        return Receipt(
            peer.id, platform, address, ok=True, dry_run=False,
            detail=f"sent to {peer.name} via {platform}", transport_result=result,
        )

    def send(self, peer_id: str, text: str, *, channel: str = "auto",
             dry_run: bool = False) -> Receipt:
        peer = self._registry.get(peer_id)          # fail-closed
        platform = self._resolve_channel(peer, channel)
        return self._emit(peer, platform, text, source_platform=None, dry_run=dry_run)

    def relay(self, peer_id: str, inbound, *, to: str = "auto",
              dry_run: bool = False) -> Receipt:
        peer = self._registry.get(peer_id)
        from_platform, text = _coerce_inbound(inbound)
        if from_platform:
            try:
                self._presence.record(peer.id, from_platform, self._now())
            except Exception:
                pass
        to_platform = self._resolve_relay_target(peer, from_platform, to)
        return self._emit(peer, to_platform, text,
                          source_platform=from_platform, dry_run=dry_run)

    def record_presence(self, peer_id: str, platform: str) -> None:
        peer = self._registry.get(peer_id)
        if not peer.has(platform):
            raise ChannelNotConfigured(
                f"peer {peer.id!r} has no {platform!r} endpoint"
            )
        self._presence.record(peer.id, platform, self._now())


# --- CLI ---------------------------------------------------------------------

def _format_peer(p: Peer) -> str:
    """One peer's id/name (+ protocol tag) and its indented endpoint lines."""
    tag = f"  [{p.engagement_protocol}]" if p.engagement_protocol else ""
    lines = [f"{p.id}: {p.name}{tag}"]
    lines.extend(f"    {plat}: {p.endpoint(plat)}" for plat in p.channels())
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse
    from lib.hal.channel_backends import build_gateway  # lazy: avoids import cycle

    ap = argparse.ArgumentParser(prog="hal-channel",
                                 description="Human Channel Gateway")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("send", help="send a message to a peer")
    s.add_argument("peer")
    s.add_argument("text")
    s.add_argument("--channel", default="auto",
                   help="auto | whatsapp | discord | slack")
    s.add_argument("--dry-run", action="store_true")

    r = sub.add_parser("relay", help="relay a peer's message to their other channel")
    r.add_argument("peer")
    r.add_argument("--from", dest="from_platform", required=True)
    r.add_argument("--text", required=True)
    r.add_argument("--to", default="auto")
    r.add_argument("--dry-run", action="store_true")

    sub.add_parser("list", help="list configured peers and endpoints")

    args = ap.parse_args(argv)

    if args.cmd == "list":
        for p in PeerRegistry.from_config().peers():
            print(_format_peer(p))
        return 0

    gw = build_gateway()
    if args.cmd == "send":
        rcpt = gw.send(args.peer, args.text, channel=args.channel, dry_run=args.dry_run)
    else:
        rcpt = gw.relay(args.peer, (args.from_platform, args.text),
                        to=args.to, dry_run=args.dry_run)
    print(rcpt.detail)
    return 0 if rcpt.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
