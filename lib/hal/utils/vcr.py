"""VCR — record-and-replay fixture system for CCH API calls.

VCR_RECORD=1  : make real call, write fixture to disk
VCR_RECORD unset + fixture exists : replay fixture (no API call)
VCR_RECORD unset + fixture missing + CI=1 : raise VCRFixtureMissingError
VCR_RECORD unset + fixture missing + not CI : make real call (dev mode)

Fixture path: {CLAUDE_CODE_TEST_FIXTURES_ROOT or ./fixtures}/cch-{hash12}.json
Hash: SHA1 of JSON-serialised dehydrated request body (first 12 chars).
"""
import hashlib
import json
import os
import re
from pathlib import Path


class VCRFixtureMissingError(Exception):
    """Raised in CI when a fixture file does not exist and VCR_RECORD is off."""


def _fixture_dir() -> Path:
    root = os.environ.get("CLAUDE_CODE_TEST_FIXTURES_ROOT")
    return Path(root) if root else Path.cwd() / "fixtures"


def _dehydrate(body: str) -> str:
    cwd = str(Path.cwd())
    home = str(Path.home())
    body = body.replace(cwd, "[CWD]").replace(home, "[CONFIG_HOME]")
    body = re.sub(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        "[UUID]",
        body,
        flags=re.IGNORECASE,
    )
    body = re.sub(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?",
        "[TIMESTAMP]",
        body,
    )
    return body


def _fixture_path(request_body: str) -> Path:
    normalised = _dehydrate(request_body)
    digest = hashlib.sha1(normalised.encode()).hexdigest()[:12]
    return _fixture_dir() / f"cch-{digest}.json"


def _replay(path: Path):
    from lib.hal.types import ChatResponse, TokenUsage

    data = json.loads(path.read_text(encoding="utf-8"))
    usage = data.get("usage", {})
    return ChatResponse(
        text=data.get("text", ""),
        model=data.get("model", ""),
        provider=data.get("provider", "anthropic-cch"),
        finish_reason=data.get("finish_reason", "stop"),
        usage=TokenUsage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            cache_read_tokens=usage.get("cache_read_tokens", 0),
            cache_creation_tokens=usage.get("cache_creation_tokens", 0),
        ),
        latency_ms=data.get("latency_ms", 0.0),
        raw=data.get("raw", {}),
    )


def _record(path: Path, response) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "text": response.text,
        "model": response.model,
        "provider": response.provider,
        "finish_reason": response.finish_reason,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "cache_read_tokens": response.usage.cache_read_tokens,
            "cache_creation_tokens": response.usage.cache_creation_tokens,
        },
        "latency_ms": response.latency_ms,
        "raw": response.raw,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def with_vcr(fn, *, request_body: str):
    """Call fn() with VCR record/replay semantics.

    fn           : zero-argument callable returning a ChatResponse
    request_body : raw JSON request body (used for fixture naming)
    """
    path = _fixture_path(request_body)
    record = os.environ.get("VCR_RECORD") == "1"

    if not record and path.exists():
        return _replay(path)

    if not record and not path.exists() and os.environ.get("CI") == "1":
        raise VCRFixtureMissingError(
            f"VCR fixture missing in CI: {path}. "
            "Run with VCR_RECORD=1 outside CI to record it."
        )

    response = fn()
    if record:
        _record(path, response)
    return response
