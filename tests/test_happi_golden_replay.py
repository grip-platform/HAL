"""HAPPI α-4 golden-suite SSE replay harness.

Loads hand-curated SSE response fixtures from `tests/happi_golden/`,
patches each adapter's `_open_stream` to replay the fixture bytes, and
asserts that the resulting HAPPI event sequence matches the canonical
shape:

    [delta("Hello"), delta(", "), delta("HAPPI!"), completed(usage!={})]

Regression coverage: a refactor to any adapter's `_translate_and_terminate`
or `_iter_sse_*` that produces the wrong delta text, wrong event types,
or a missing usage block will fail at least one parametrised test here.

All fixtures MUST produce the same concatenated text ("Hello, HAPPI!")
so the harness assertions are provider-agnostic. See `happi_golden/README.md`.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Iterable
from unittest.mock import patch

import pytest

from lib.hal.happi.envelope import HappiEnvelope


FIXTURE_DIR = Path(__file__).parent / "happi_golden"
CANONICAL_TEXT = "Hello, HAPPI!"
CANONICAL_TYPES = ["delta", "delta", "delta", "completed"]


class _FakeSSEResponse:
    """Minimal response emulating what urlopen returns.

    Iterates over newline-terminated byte lines from a fixture. Closes are
    no-ops apart from flipping a flag so tests can assert the adapter
    closed the stream.
    """

    def __init__(self, raw_bytes: bytes):
        # Preserve trailing-newline semantics so SSE boundary detection
        # (empty-line = event separator) works identically to urllib's
        # HTTPResponse iterator.
        lines = raw_bytes.split(b"\n")
        self._lines: list[bytes] = [line + b"\n" for line in lines]
        self.closed = False

    def __iter__(self) -> Iterable[bytes]:
        return iter(self._lines)

    def close(self) -> None:
        self.closed = True


FIXTURES: dict[str, dict[str, Any]] = {
    "anthropic": {
        "fixture": "anthropic_hello.sse",
        "module": "lib.hal.happi.providers.anthropic",
        "stream_fn": "stream_anthropic",
        "envelope_flags": {"provider": "claude-opus-4-7", "max_tokens": 100},
        # The token SHAPE is load-bearing, not decorative. stream_anthropic
        # branches on _is_oauth_token (providers/anthropic.py) and routes
        # subscription-OAuth-prefixed tokens through _stream_via_cch, returning
        # before _open_stream — the function this suite patches — is ever
        # called. A fixture carrying that prefix is therefore never replayed at
        # all, and every assertion here fails while saying nothing about the
        # adapter. This value deliberately matches no real credential format:
        # it only has to miss the OAuth prefix to take the SSE path.
        "envelope_auth": {
            "scheme": "apikey",
            "token": "golden-fixture-token-not-a-real-credential",
        },
    },
    "openai_compat": {
        "fixture": "openai_compat_hello.sse",
        "module": "lib.hal.happi.providers.openai_compat",
        "stream_fn": "stream_openai_compat",
        "envelope_flags": {"provider": "openai/gpt-4o-mini", "max_tokens": 100},
        "envelope_auth": {
            "scheme": "apikey",
            "token": "sk-proj-goldentest0123456789abcdef",
        },
    },
    "gemini": {
        "fixture": "gemini_hello.sse",
        "module": "lib.hal.happi.providers.gemini",
        "stream_fn": "stream_gemini",
        "envelope_flags": {"provider": "gemini/gemini-2.0-flash", "max_tokens": 100},
        "envelope_auth": {
            "scheme": "apikey",
            "token": "AIzaSyFakeGoldenTestKey0123456789",
        },
    },
}


def _load_fixture(name: str) -> bytes:
    return (FIXTURE_DIR / name).read_bytes()


def _build_envelope(flags: dict, auth: dict) -> HappiEnvelope:
    return HappiEnvelope.parse({
        "v": "happi/1.0",
        "id": "golden-replay",
        "cmd": "chat",
        "args": ["unused — fixture replays"],
        "flags": flags,
        "auth": auth,
    })


def _collect_events(stream_iter) -> list[dict]:
    return [json.loads(line) for line in stream_iter]


@pytest.fixture
def fixtures_exist():
    missing = [spec["fixture"] for spec in FIXTURES.values()
               if not (FIXTURE_DIR / spec["fixture"]).exists()]
    if missing:
        pytest.skip(f"fixture files missing: {missing}")


@pytest.mark.parametrize("family", list(FIXTURES.keys()))
def test_golden_replay_yields_canonical_event_types(family, fixtures_exist):
    spec = FIXTURES[family]
    module = importlib.import_module(spec["module"])
    stream_fn = getattr(module, spec["stream_fn"])

    fixture_bytes = _load_fixture(spec["fixture"])
    fake_response = _FakeSSEResponse(fixture_bytes)
    envelope = _build_envelope(spec["envelope_flags"], spec["envelope_auth"])

    with patch.object(module, "_open_stream", return_value=fake_response):
        events = _collect_events(stream_fn(envelope))

    types = [e["type"] for e in events]
    assert types == CANONICAL_TYPES, (
        f"{family}: expected {CANONICAL_TYPES}, got {types}"
    )


@pytest.mark.parametrize("family", list(FIXTURES.keys()))
def test_golden_replay_delta_text_matches_canonical(family, fixtures_exist):
    spec = FIXTURES[family]
    module = importlib.import_module(spec["module"])
    stream_fn = getattr(module, spec["stream_fn"])

    fixture_bytes = _load_fixture(spec["fixture"])
    fake_response = _FakeSSEResponse(fixture_bytes)
    envelope = _build_envelope(spec["envelope_flags"], spec["envelope_auth"])

    with patch.object(module, "_open_stream", return_value=fake_response):
        events = _collect_events(stream_fn(envelope))

    delta_text = "".join(e["text"] for e in events if e["type"] == "delta")
    assert delta_text == CANONICAL_TEXT, (
        f"{family}: expected {CANONICAL_TEXT!r}, got {delta_text!r}"
    )


@pytest.mark.parametrize("family", list(FIXTURES.keys()))
def test_golden_replay_reports_non_empty_usage(family, fixtures_exist):
    spec = FIXTURES[family]
    module = importlib.import_module(spec["module"])
    stream_fn = getattr(module, spec["stream_fn"])

    fixture_bytes = _load_fixture(spec["fixture"])
    fake_response = _FakeSSEResponse(fixture_bytes)
    envelope = _build_envelope(spec["envelope_flags"], spec["envelope_auth"])

    with patch.object(module, "_open_stream", return_value=fake_response):
        events = _collect_events(stream_fn(envelope))

    completed = [e for e in events if e["type"] == "completed"]
    assert completed, f"{family}: no completed event"
    usage = completed[0].get("usage") or {}
    assert usage, f"{family}: completed.usage was empty"
    # Canonical fixtures all claim 3 input / 4 output tokens.
    assert usage.get("in") == 3, f"{family}: usage.in expected 3, got {usage.get('in')}"
    assert usage.get("out") == 4, f"{family}: usage.out expected 4, got {usage.get('out')}"


@pytest.mark.parametrize("family", list(FIXTURES.keys()))
def test_golden_replay_closes_response(family, fixtures_exist):
    spec = FIXTURES[family]
    module = importlib.import_module(spec["module"])
    stream_fn = getattr(module, spec["stream_fn"])

    fixture_bytes = _load_fixture(spec["fixture"])
    fake_response = _FakeSSEResponse(fixture_bytes)
    envelope = _build_envelope(spec["envelope_flags"], spec["envelope_auth"])

    with patch.object(module, "_open_stream", return_value=fake_response):
        list(stream_fn(envelope))

    assert fake_response.closed is True, f"{family}: adapter did not close the response"


def test_all_fixtures_registered():
    """Each .sse file in happi_golden/ must have a FIXTURES entry."""
    sse_files = {p.name for p in FIXTURE_DIR.glob("*.sse")}
    registered = {spec["fixture"] for spec in FIXTURES.values()}
    assert sse_files == registered, (
        f"fixture registry drift: sse files={sse_files}, registered={registered}"
    )
