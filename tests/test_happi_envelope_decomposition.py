"""Regression anchor for HappiEnvelope.parse decomposition (issue #321).

Goodhart-proof: asserts concrete field values and specific EnvelopeError
message substrings — not call counts or mock state. Every assertion would
fail under an obvious mutation (e.g. dropping a field, swapping error text,
or silently coercing a value).

Covers:
- All three input branches of _decode_envelope: str, bytes, pre-decoded dict
- All six contracted EnvelopeError message substrings
- Value-preservation: flags/auth/ctx verbatim, args/caps -> tuple[str,...],
  v echoed not coerced, content defaults to ""
- Round-trip identity: parse -> to_dict field parity
"""
from __future__ import annotations

import pytest

from lib.hal.happi.envelope import (
    SUPPORTED_VERSIONS,
    EnvelopeError,
    HappiEnvelope,
    _build_envelope_fields,
    _decode_envelope,
    _validate_envelope,
)


# ---------------------------------------------------------------------------
# Input-branch coverage: str, bytes, pre-decoded dict
# ---------------------------------------------------------------------------

_CANONICAL_RAW = (
    '{"v":"happi/1.1","id":"r-decomp","cmd":"echo",'
    '"args":["x","y"],'
    '"flags":{"audit":true,"stream":false},'
    '"auth":{"scheme":"subscription"},'
    '"ctx":{"cwd":"/tmp"},'
    '"capabilities":["read:stdin"],'
    '"content":"hello world"}'
)


def test_parse_accepts_str_input():
    env = HappiEnvelope.parse(_CANONICAL_RAW)
    assert env.v == "happi/1.1"
    assert env.id == "r-decomp"
    assert env.cmd == "echo"
    assert env.args == ("x", "y")
    assert env.flags == {"audit": True, "stream": False}
    assert env.auth == {"scheme": "subscription"}
    assert env.ctx == {"cwd": "/tmp"}
    assert env.capabilities == ("read:stdin",)
    assert env.content == "hello world"


def test_parse_accepts_bytes_input():
    env = HappiEnvelope.parse(_CANONICAL_RAW.encode())
    assert env.id == "r-decomp"
    assert env.content == "hello world"
    # flags boolean values preserved verbatim — no coercion
    assert env.flags["audit"] is True
    assert env.flags["stream"] is False


def test_parse_accepts_pre_decoded_dict():
    data = {
        "v": "happi/1.0",
        "id": "dict-input",
        "cmd": "version",
        "args": [1, 2],  # ints must be stringified
        "capabilities": [3],
    }
    env = HappiEnvelope.parse(data)
    assert env.v == "happi/1.0"
    assert env.args == ("1", "2")
    assert env.capabilities == ("3",)


def test_parse_accepts_happi_1_2_and_1_3():
    """happi/1.2 and happi/1.3 parse identically to 1.0/1.1 at the envelope layer.

    Regression anchor for task #479(b): the hal binary rejected 1.2/1.3
    ("unsupported protocol version") while V>>'s directive makes 1.3 canonical.
    Widening SUPPORTED_VERSIONS must ACCEPT them and echo the version verbatim.
    Mutation killer: drop "happi/1.3" from SUPPORTED_VERSIONS -> this fails.
    """
    for version in ("happi/1.2", "happi/1.3"):
        env = HappiEnvelope.parse(
            f'{{"v":"{version}","id":"v13","cmd":"brief","args":["hi"]}}'
        )
        assert env.v == version
        assert env.id == "v13"
        assert env.cmd == "brief"
        assert env.args == ("hi",)
    assert {"happi/1.2", "happi/1.3"} <= SUPPORTED_VERSIONS


def test_validate_envelope_accepts_happi_1_2_and_1_3():
    """AGORA #494 Speaker C: exercise _validate_envelope DIRECTLY (not only via
    parse) for the two new versions — it must NOT raise on 1.2/1.3, proving the
    version gate itself accepts them.

    Mutation killer: drop 1.3 from SUPPORTED_VERSIONS -> _validate_envelope
    raises "unsupported protocol version" -> this fails.
    """
    for version in ("happi/1.2", "happi/1.3"):
        # No exception == accepted at the validation layer.
        _validate_envelope({"v": version, "id": "x", "cmd": "c"})


def test_build_envelope_fields_preserves_new_versions_verbatim():
    """AGORA #494 Speaker C: exercise _build_envelope_fields DIRECTLY for
    1.2/1.3 — it echoes the version verbatim and builds the identical field
    set, proving "parse identically" at the field-construction layer, not just
    the parse() happy path.
    """
    for version in ("happi/1.2", "happi/1.3"):
        fields = _build_envelope_fields(
            {"v": version, "id": "v13b", "cmd": "chat", "args": ["a"], "capabilities": ["c"]}
        )
        assert fields["v"] == version
        assert set(fields.keys()) == {
            "v", "id", "cmd", "args", "flags", "auth", "ctx", "capabilities", "content",
        }
        assert fields["args"] == ("a",)
        assert fields["capabilities"] == ("c",)
        assert fields["content"] == ""


# ---------------------------------------------------------------------------
# Value-preservation invariants
# ---------------------------------------------------------------------------

def test_v_is_echoed_not_coerced():
    """v field is echoed from the wire; no normalisation to v1.0 or v1.1."""
    for version in ("happi/1.0", "happi/1.1", "happi/1.2", "happi/1.3"):
        env = HappiEnvelope.parse(f'{{"v":"{version}","id":"x","cmd":"c"}}')
        assert env.v == version


def test_content_defaults_to_empty_string():
    env = HappiEnvelope.parse('{"v":"happi/1.1","id":"x","cmd":"c"}')
    assert env.content == ""
    assert isinstance(env.content, str)


def test_flags_auth_ctx_default_to_empty_dict():
    env = HappiEnvelope.parse('{"v":"happi/1.1","id":"x","cmd":"c"}')
    assert env.flags == {}
    assert env.auth == {}
    assert env.ctx == {}


def test_args_and_capabilities_become_tuple_of_str():
    raw = '{"v":"happi/1.1","id":"x","cmd":"c","args":[1,true,3.5],"capabilities":[42]}'
    env = HappiEnvelope.parse(raw)
    assert env.args == ("1", "True", "3.5")
    assert env.capabilities == ("42",)
    assert isinstance(env.args, tuple)
    assert isinstance(env.capabilities, tuple)


# ---------------------------------------------------------------------------
# EnvelopeError message-substring contract (all six contracted strings)
# ---------------------------------------------------------------------------

def test_error_on_invalid_json():
    with pytest.raises(EnvelopeError) as exc:
        HappiEnvelope.parse("{bad json")
    assert "invalid JSON" in str(exc.value)


def test_error_on_non_dict_json():
    with pytest.raises(EnvelopeError) as exc:
        HappiEnvelope.parse("[1,2,3]")
    assert "envelope must be a JSON object" in str(exc.value)


def test_error_on_unsupported_version():
    # happi/1.2 and 1.3 are now supported; a version above the ceiling is still rejected.
    with pytest.raises(EnvelopeError) as exc:
        HappiEnvelope.parse('{"v":"happi/2.0","id":"x","cmd":"c"}')
    assert "unsupported protocol version" in str(exc.value)
    assert "happi/2.0" in str(exc.value)


def test_error_on_missing_id():
    with pytest.raises(EnvelopeError) as exc:
        HappiEnvelope.parse('{"v":"happi/1.1","cmd":"c"}')
    assert "envelope missing non-empty 'id'" in str(exc.value)


def test_error_on_empty_id():
    with pytest.raises(EnvelopeError) as exc:
        HappiEnvelope.parse('{"v":"happi/1.1","id":"","cmd":"c"}')
    assert "envelope missing non-empty 'id'" in str(exc.value)


def test_error_on_missing_cmd():
    with pytest.raises(EnvelopeError) as exc:
        HappiEnvelope.parse('{"v":"happi/1.1","id":"x"}')
    assert "envelope missing non-empty 'cmd'" in str(exc.value)


def test_error_on_args_not_list():
    with pytest.raises(EnvelopeError) as exc:
        HappiEnvelope.parse('{"v":"happi/1.1","id":"x","cmd":"c","args":"bad"}')
    assert "'args' must be a list" in str(exc.value)


def test_error_on_capabilities_not_list():
    with pytest.raises(EnvelopeError) as exc:
        HappiEnvelope.parse('{"v":"happi/1.1","id":"x","cmd":"c","capabilities":"bad"}')
    assert "'capabilities' must be a list" in str(exc.value)


# ---------------------------------------------------------------------------
# Round-trip identity
# ---------------------------------------------------------------------------

def test_round_trip_parse_to_dict():
    raw = {
        "v": "happi/1.1",
        "id": "rt-1",
        "cmd": "echo",
        "args": ["a"],
        "flags": {"k": True},
        "auth": {"scheme": "sub"},
        "ctx": {"cwd": "/"},
        "capabilities": ["read:stdin"],
        "content": "prompt text",
    }
    env = HappiEnvelope.parse(raw)
    d = env.to_dict()
    assert d["v"] == raw["v"]
    assert d["id"] == raw["id"]
    assert d["cmd"] == raw["cmd"]
    assert d["args"] == raw["args"]
    assert d["flags"] == raw["flags"]
    assert d["auth"] == raw["auth"]
    assert d["ctx"] == raw["ctx"]
    assert d["capabilities"] == raw["capabilities"]
    assert d["content"] == raw["content"]


# ---------------------------------------------------------------------------
# Mutation-resistance: helpers are importable and behave as specified
# ---------------------------------------------------------------------------

def test_decode_envelope_rejects_non_dict_after_parse():
    """_decode_envelope raises if JSON is valid but not an object."""
    with pytest.raises(EnvelopeError, match="envelope must be a JSON object"):
        _decode_envelope('"just a string"')


def test_validate_envelope_raises_on_bad_version():
    with pytest.raises(EnvelopeError, match="unsupported protocol version"):
        _validate_envelope({"v": "happi/9.9", "id": "x", "cmd": "c"})


def test_build_envelope_fields_returns_correct_keys():
    data = {
        "v": "happi/1.1",
        "id": "x",
        "cmd": "c",
        "args": ["a"],
        "capabilities": ["cap"],
    }
    fields = _build_envelope_fields(data)
    assert set(fields.keys()) == {"v", "id", "cmd", "args", "flags", "auth", "ctx", "capabilities", "content"}
    assert fields["args"] == ("a",)
    assert fields["capabilities"] == ("cap",)
    assert fields["content"] == ""
    assert fields["flags"] == {}
