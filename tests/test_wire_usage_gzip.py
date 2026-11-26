"""The wire reads usage from a gzipped stream.

Anthropic gzips a streamed reply whenever the client offers gzip, and Claude
Code does. The wire's usage tap scanned the compressed bytes for events and
found none, so every row the wire wrote for a real turn -- the signed thread
row and the cache-loss ledger -- carried no usage. Measured through the live
wire on 2026-10-08: the same one-word Haiku turn recorded no usage when gzip
was offered, and 4,038 input tokens when it was not.

Each gzip case here fails on the tap as it was, which scanned every relayed
piece as it arrived; each plain case is the control that passes either way.
"""
from __future__ import annotations

import gzip
import json
import zlib

import pytest

from lib.hal import wire_proxy
from lib.hal.wire_classify import StreamDecoder
from tests import test_wire_proxy as h
from tests.test_wire_proxy import (  # noqa: F401 -- fixtures pytest finds by name here:
    home, pki, make_wire, wire,
)

GZIP = (("content-encoding", "gzip"),)
#: The usage in h.SSE_EVENTS, as a thread row holds it and as the ledger holds it.
THREAD_USAGE = {"input_tokens": 21, "output_tokens": 9, "cache_read_tokens": 5,
                "cache_write_tokens": 3, "cache_write_1h_tokens": 0}
LEDGER_USAGE = {"input_tokens": 21, "output_tokens": 9, "cache_read_tokens": 5,
                "cache_creation_tokens": 3}


def _event_bytes() -> list:
    return [f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
            for event in h.SSE_EVENTS]


def _gzip_parts(parts: list) -> list:
    """Each part compressed and flushed on its own, as a streaming server sends it."""
    stream = zlib.compressobj(wbits=31)
    out = [stream.compress(part) + stream.flush(zlib.Z_SYNC_FLUSH) for part in parts]
    return out + [stream.flush()]


def _pieces(data: bytes, size: int) -> list:
    return [data[i:i + size] for i in range(0, len(data), size)]


def _decode_all(decoder: StreamDecoder, pieces: list) -> bytes:
    out = bytearray()
    for piece in pieces:
        out += b"".join(decoder(piece))
    return bytes(out)


# --------------------------------------------------------------------------- #
# StreamDecoder on its own.
# --------------------------------------------------------------------------- #
def test_an_unencoded_body_passes_through_untouched():
    assert _decode_all(StreamDecoder((), 64), [b"ab", b"", b"cd"]) == b"abcd"


@pytest.mark.parametrize("coding", ["gzip", "x-gzip", "GZip"])
def test_a_gzip_stream_cut_anywhere_decodes_to_what_was_sent(coding):
    sent = b"".join(_event_bytes())
    stream = b"".join(_gzip_parts(_pieces(sent, 50)))
    decoder = StreamDecoder((("content-encoding", coding),), 64)
    assert _decode_all(decoder, _pieces(stream, 7)) == sent


def test_every_gzip_member_is_read():
    stream = gzip.compress(b"first member\n") + gzip.compress(b"second member\n")
    decoded = _decode_all(StreamDecoder(GZIP, 64), _pieces(stream, 5))
    assert decoded == b"first member\nsecond member\n"


def test_no_block_is_longer_than_asked_for():
    sent = b"x" * 5_000_000
    blocks = list(StreamDecoder(GZIP, 65536)(gzip.compress(sent)))
    assert max(len(block) for block in blocks) <= 65536
    assert b"".join(blocks) == sent


def test_a_corrupt_stream_yields_nothing_more_and_never_raises():
    decoder = StreamDecoder(GZIP, 64)
    assert _decode_all(decoder, [b"this was never gzip"]) == b""
    assert decoder.dead
    assert _decode_all(decoder, [gzip.compress(b"too late")]) == b""


@pytest.mark.parametrize("coding", ["br", "zstd", "deflate", "gzip, br"])
def test_a_coding_it_does_not_read_yields_nothing(coding):
    decoder = StreamDecoder((("content-encoding", coding),), 64)
    assert _decode_all(decoder, [b"event: message_start\n\n"]) == b""


# --------------------------------------------------------------------------- #
# The usage tap, fed a plain stream and a gzipped one.
# --------------------------------------------------------------------------- #
def _tap(coding: str) -> wire_proxy._UsageTap:
    head = (b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
            + (f"content-encoding: {coding}\r\n".encode() if coding else b""))
    return wire_proxy._UsageTap(wire_proxy._parse_head(head))


def _cut_in_the_last_usage_event(sent: bytes, where: str) -> int:
    """An offset in ``sent`` at ``where`` in the message_delta event, the one that
    carries the turn's final output count."""
    start = sent.index(b'"type": "message_delta"')
    end = sent.index(b"\n\n", start)
    return {"inside its data line": start + 5, "between its two newlines": end + 1,
            "right after it": end + 2}[where]


@pytest.mark.parametrize("where", ["inside its data line", "between its two newlines",
                                   "right after it"])
def test_usage_survives_a_gzip_member_ending_in_the_last_usage_event(where):
    """The falsifier AGORA named on HAL#705: one event cut across two gzip members
    must still be read whole, or the row loses the final output count."""
    sent = b"".join(_event_bytes())
    cut = _cut_in_the_last_usage_event(sent, where)
    tap = _tap("gzip")
    for piece in _pieces(gzip.compress(sent[:cut]) + gzip.compress(sent[cut:]), 7):
        tap(piece)
    assert tap.thread_usage() == THREAD_USAGE


def test_a_stream_labelled_with_two_codings_is_not_read_as_gzip():
    """Two Content-Encoding lines join as "gzip, br", which HAL cannot undo, so
    the tap reads nothing rather than misread a stream it was told is br too."""
    head = (b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
            b"content-encoding: gzip\r\ncontent-encoding: br\r\n")
    tap = wire_proxy._UsageTap(wire_proxy._parse_head(head))
    tap(gzip.compress(b"".join(_event_bytes())))
    assert tap.thread_usage() is None


@pytest.mark.parametrize("coding", ["", "gzip"])
def test_the_tap_reads_usage_from_a_plain_or_gzipped_stream(coding):
    parts = _event_bytes()
    tap = _tap(coding)
    for piece in _pieces(b"".join(_gzip_parts(parts) if coding else parts), 7):
        tap(piece)
    assert tap.thread_usage() == THREAD_USAGE
    assert tap.result()[0] == "claude-opus-5-5"


# --------------------------------------------------------------------------- #
# End to end: one streamed turn through a running wire.
# --------------------------------------------------------------------------- #
def _streaming_reply(coding: str, sent: list):
    """Stands in for FakeUpstream._sse, gzipping when asked, recording every frame."""
    def reply(tls) -> bool:
        parts = _event_bytes()
        sent.append(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
                    + (b"content-encoding: gzip\r\n" if coding else b"")
                    + b"transfer-encoding: chunked\r\n\r\n")
        sent.extend(f"{len(part):x}\r\n".encode() + part + b"\r\n"
                    for part in (_gzip_parts(parts) if coding else parts) if part)
        sent.append(b"0\r\n\r\n")
        for frame in sent:
            tls.sendall(frame)
        return True
    return reply


@pytest.mark.parametrize("coding", ["", "gzip"])
def test_a_streamed_turn_puts_its_usage_on_both_wire_rows(wire, monkeypatch, coding):
    thread_rows, sent = [], []
    monkeypatch.setattr("lib.hal.gateway_witness.append_thread_row",
                        lambda record, **kwargs: thread_rows.append(kwargs))
    wire.fake._sse = _streaming_reply(coding, sent)
    turn = h._post("/v1/messages?beta=true", {**h.TURN, "stream": True},
                   extra=f"x-claude-code-session-id: {h.SESSION_ID}\r\n")
    received = h._exchange(wire, turn)
    assert received == b"".join(sent)  # relayed as sent: only the tap decodes
    assert h._wait_for(lambda: thread_rows and h._cache_loss_rows(wire.home))
    assert thread_rows[-1]["usage"] == THREAD_USAGE
    assert h._cache_loss_rows(wire.home)[-1]["usage"] == LEDGER_USAGE
