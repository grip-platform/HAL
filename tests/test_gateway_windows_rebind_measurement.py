"""Measure Windows rebind behaviour after a server-side close enters TIME_WAIT."""
from __future__ import annotations

import socket

import pytest


@pytest.mark.skipif(not hasattr(socket, "SO_EXCLUSIVEADDRUSE"),
                    reason="Windows-only socket option")
def test_exclusive_listener_immediate_rebind_after_server_side_close():
    """Mutation: omit the accepted connection and measure only a bare listener.
    Survives by completing a real connection whose server side sends FIN first,
    then reporting either immediate-rebind outcome without prescribing one."""
    address = "127.0.0.1"
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    listener.bind((address, 0))
    port = listener.getsockname()[1]
    listener.listen(1)

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    accepted = None
    replacement = None
    try:
        client.connect((address, port))
        accepted, _ = listener.accept()
        accepted.shutdown(socket.SHUT_WR)
        assert client.recv(1) == b""
        accepted.close()
        accepted = None
        client.close()
        listener.close()

        replacement = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        replacement.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            replacement.bind((address, port))
        except OSError as exc:
            print(f"PASS/MEASURED: immediate rebind refused: {exc!r}")
        else:
            print("PASS/MEASURED: immediate rebind succeeded")
    finally:
        if accepted is not None:
            accepted.close()
        client.close()
        listener.close()
        if replacement is not None:
            replacement.close()
