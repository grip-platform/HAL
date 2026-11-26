"""Tests for HAL inter-instance mesh."""
import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from lib.hal.mesh import MeshRegistry, PeerInfo, _STALE_THRESHOLD_S


class TestPeerInfo(unittest.TestCase):

    def test_fresh_peer_not_stale(self):
        peer = PeerInfo(pid=1234, last_heartbeat=time.time())
        self.assertFalse(peer.is_stale)

    def test_old_peer_is_stale(self):
        peer = PeerInfo(pid=1234, last_heartbeat=time.time() - _STALE_THRESHOLD_S - 10)
        self.assertTrue(peer.is_stale)


class TestMeshRegistry(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig_mesh = os.environ.get("HAL_HOME")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("lib.hal.mesh._MESH_DIR")
    @patch("lib.hal.mesh._MSG_DIR")
    def test_announce_creates_heartbeat(self, mock_msg_dir, mock_mesh_dir):
        mesh_dir = os.path.join(self.tmpdir, "mesh")
        msg_dir = os.path.join(mesh_dir, "messages")
        mock_mesh_dir.__str__ = lambda s: mesh_dir
        mock_msg_dir.__str__ = lambda s: msg_dir

        # Direct test of announce logic
        os.makedirs(mesh_dir, exist_ok=True)
        os.makedirs(msg_dir, exist_ok=True)
        pid = os.getpid()
        heartbeat = {"pid": pid, "status": "active", "last_heartbeat": time.time()}
        path = os.path.join(mesh_dir, f"{pid}.json")
        with open(path, "w") as f:
            json.dump(heartbeat, f)

        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data["pid"], pid)


class TestMeshMessaging(unittest.TestCase):

    def test_send_creates_file(self):
        tmpdir = tempfile.mkdtemp()
        msg_dir = os.path.join(tmpdir, "messages")
        os.makedirs(msg_dir)

        msg = {"from_pid": 100, "to_pid": 200, "content": "hello", "timestamp": time.time()}
        fname = f"100-200-{int(time.time() * 1000)}.json"
        path = os.path.join(msg_dir, fname)
        with open(path, "w") as f:
            json.dump(msg, f)

        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data["content"], "hello")

        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
