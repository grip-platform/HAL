"""Bounded body draining for HTTP refusals shared by HAL servers."""

#: The most an early refusal reads and drops before the connection closes.
_DISCARD_CAP = 1 << 20
#: How long an early refusal waits for a body the client is holding back.
_DISCARD_WAIT_S = 2.0


def _discard_unread_body(handler) -> None:
    """Read and drop the body of a request refused before reading it.

    Closing a socket that still holds unread request bytes makes Linux send a
    reset, so the client sees "connection reset" instead of the 403, 404 or
    429 it was just sent. Large bodies are the realistic case (a refused
    ``/v1/messages`` turn runs to hundreds of kilobytes); a small body usually
    arrives with the headers and is already buffered. Call it only on a refusal
    that answers before the body is read. The
    read is bounded twice: a body over ``_DISCARD_CAP`` bytes is left alone,
    and a client that holds its body back is waited for at most
    ``_DISCARD_WAIT_S`` seconds, so a refusal cannot pin a server thread. The
    connection closes next either way. Only the expected failures are
    absorbed -- a client that holds its body back (``TimeoutError``) or goes
    away (``ConnectionError``); anything else reaches the server's error log.
    """
    try:
        length = int(handler.headers.get("Content-Length", 0) or 0)
    except (TypeError, ValueError):
        return
    if not 0 < length <= _DISCARD_CAP:
        return
    conn = getattr(handler, "connection", None)
    try:
        if conn is not None:
            conn.settimeout(_DISCARD_WAIT_S)
        handler.rfile.read(length)
    except (TimeoutError, ConnectionError):
        pass
