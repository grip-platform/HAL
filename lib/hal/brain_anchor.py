"""Merkle snapshots with RFC 3161 tokens, outside every host's inference path.

Reuses infinite_thread.grip_anchor -> GRIP lib/rfc3161_anchor.py (2026-09-17).
Only a SHA-256 digest reaches the TSA. Unlike that helper, this job also checks
its returned token against the configured TSA trust certificate before success.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path

from . import infinite_thread
from .brain import BrainStore, load_pins


def merkle_root(heads: list[dict]) -> str:
    """SHA256(00 || canonical head); sorted by host/thread; nodes use 01.

    Odd levels duplicate their last node. Empty trees use SHA256(empty bytes).
    The full signed envelope is committed, including host and thread identity.
    """
    ordered = sorted(heads, key=lambda head: (head["host"], head["thread"]))
    if len({(h["host"], h["thread"]) for h in ordered}) != len(ordered):
        raise ValueError("duplicate head")
    nodes = [hashlib.sha256(b"\x00" + infinite_thread.canonical(h)).digest() for h in ordered]
    if not nodes:
        return hashlib.sha256(b"").hexdigest()
    while len(nodes) > 1:
        if len(nodes) % 2:
            nodes.append(nodes[-1])
        nodes = [hashlib.sha256(b"\x01" + nodes[i] + nodes[i + 1]).digest()
                 for i in range(0, len(nodes), 2)]
    return nodes[0].hex()


def verify_timestamp(root: str, token: Path, ca_file: Path) -> bool:
    """OpenSSL checks both the message imprint and the TSA signature/chain/EKU."""
    try:
        result = subprocess.run(
            ["openssl", "ts", "-verify", "-digest", root, "-token_in",
             "-in", str(token), "-CAfile", str(ca_file)],
            capture_output=True, timeout=15)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def anchor_once(store: BrainStore, destination: Path, ca_file: Path, *, anchor=None) -> dict:
    heads = store.heads()  # reverify persisted heads against current pins
    root = merkle_root(heads)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    name = uuid.uuid4().hex
    pending = destination / ("." + name)
    pending.mkdir(mode=0o700)
    status = "unavailable"
    try:
        answer = (anchor or infinite_thread.grip_anchor)(bytes.fromhex(root))
        if answer.get("status") == "disabled":
            status = "disabled"
        elif answer.get("status") == "anchored":
            token = base64.b64decode(answer["token_b64"], validate=True)
            token_path = pending / "timestamp.der"
            token_path.write_bytes(token)
            if verify_timestamp(root, token_path, ca_file):
                status = "anchored"
    except Exception:
        # No exception strings: a TSA URL may include credentials.
        status = "unavailable"
    manifest = {"v": "hal-brain-anchor/1", "algorithm": "sha256-domain-separated-duplicate-odd",
                "root": root, "heads": heads, "status": status}
    # Publish a complete directory together. Failed attempts remain explicit,
    # and never overwrite any earlier successful timestamp for the same root.
    with (pending / "root.json").open("wb") as out:
        out.write(infinite_thread.canonical(manifest))
        out.flush()
        os.fsync(out.fileno())
    if (pending / "timestamp.der").exists():
        with (pending / "timestamp.der").open("rb") as token_file:
            os.fsync(token_file.fileno())
    pending.rename(destination / name)
    return {"root": root, "status": status, "path": str(destination / name)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tsa-ca", type=Path, required=True)
    args = parser.parse_args(argv)
    result = anchor_once(BrainStore(args.state, load_pins(args.pins)), args.output, args.tsa_ca)
    print(json.dumps({"root": result["root"], "status": result["status"]}))
    return 0 if result["status"] == "anchored" else 1


if __name__ == "__main__":
    raise SystemExit(main())
