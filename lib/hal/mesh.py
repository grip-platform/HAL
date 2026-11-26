"""HAL Inter-Instance Mesh — file-based peer discovery and messaging.

Multiple HAL instances discover each other via heartbeat files and exchange
messages through the filesystem. KISS design: no HTTP, no mDNS, just files.

Each instance writes: ~/.hal/state/mesh/{pid}.json (heartbeat)
Messages via: ~/.hal/state/mesh/messages/{from}-{to}-{ts}.json

Usage:
    from lib.hal.mesh import MeshRegistry

    mesh = MeshRegistry()
    mesh.announce()                    # Write heartbeat
    peers = mesh.discover_peers()      # Find other instances
    mesh.send_to(peer_pid, "hello")   # Send message
    messages = mesh.receive()          # Read incoming messages
"""
import json
import os
import time

from lib.hal.home import HAL_HOME as _HAL_HOME

_MESH_DIR = os.path.join(_HAL_HOME, "state", "mesh")
_MSG_DIR = os.path.join(_MESH_DIR, "messages")
_STALE_THRESHOLD_S = 120  # Peers with no heartbeat for 2 min are stale


class PeerInfo:
    """Information about a discovered HAL instance."""

    def __init__(self, pid: int, model: str = "", status: str = "active",
                 start_time: float = 0.0, last_heartbeat: float = 0.0):
        self.pid = pid
        self.model = model
        self.status = status
        self.start_time = start_time
        self.last_heartbeat = last_heartbeat

    @property
    def is_stale(self) -> bool:
        return (time.time() - self.last_heartbeat) > _STALE_THRESHOLD_S


class MeshRegistry:
    """File-based peer discovery and messaging for HAL instances."""

    def __init__(self):
        self._pid = os.getpid()
        os.makedirs(_MESH_DIR, exist_ok=True)
        os.makedirs(_MSG_DIR, exist_ok=True)

    def announce(self, model: str = "", status: str = "active") -> None:
        """Write heartbeat file for this instance."""
        heartbeat = {
            "pid": self._pid,
            "model": model,
            "status": status,
            "start_time": getattr(self, "_start_time", time.time()),
            "last_heartbeat": time.time(),
        }
        path = os.path.join(_MESH_DIR, f"{self._pid}.json")
        with open(path, "w") as f:
            json.dump(heartbeat, f)

    def discover_peers(self) -> list[PeerInfo]:
        """Find all active HAL instances (excluding self)."""
        peers: list[PeerInfo] = []
        for fname in os.listdir(_MESH_DIR):
            if not fname.endswith(".json") or fname == f"{self._pid}.json":
                continue
            path = os.path.join(_MESH_DIR, fname)
            peer = _read_peer(path)
            if peer and not peer.is_stale:
                peers.append(peer)
        return peers

    def send_to(self, peer_pid: int, content: str) -> None:
        """Send a message to a specific peer."""
        msg = {
            "from_pid": self._pid,
            "to_pid": peer_pid,
            "content": content,
            "timestamp": time.time(),
        }
        fname = f"{self._pid}-{peer_pid}-{int(time.time() * 1000)}.json"
        path = os.path.join(_MSG_DIR, fname)
        with open(path, "w") as f:
            json.dump(msg, f)

    def receive(self) -> list[dict]:
        """Read and consume all messages addressed to this instance."""
        messages: list[dict] = []
        for fname in sorted(os.listdir(_MSG_DIR)):
            if not fname.endswith(".json"):
                continue
            parts = fname.replace(".json", "").split("-")
            if len(parts) >= 3 and parts[1] == str(self._pid):
                path = os.path.join(_MSG_DIR, fname)
                msg = _read_json(path)
                if msg:
                    messages.append(msg)
                    os.remove(path)  # Consume
        return messages

    def cleanup(self) -> None:
        """Remove this instance's heartbeat and prune stale peers."""
        # Remove own heartbeat
        path = os.path.join(_MESH_DIR, f"{self._pid}.json")
        if os.path.exists(path):
            os.remove(path)
        # Prune stale peers
        for fname in os.listdir(_MESH_DIR):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(_MESH_DIR, fname)
            peer = _read_peer(path)
            if peer and peer.is_stale:
                os.remove(path)


def _read_peer(path: str) -> PeerInfo:
    """Read a peer heartbeat file."""
    data = _read_json(path)
    if not data:
        return None
    return PeerInfo(
        pid=data.get("pid", 0),
        model=data.get("model", ""),
        status=data.get("status", ""),
        start_time=data.get("start_time", 0),
        last_heartbeat=data.get("last_heartbeat", 0),
    )


def _read_json(path: str) -> dict:
    """Read a JSON file, return empty dict on failure."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
