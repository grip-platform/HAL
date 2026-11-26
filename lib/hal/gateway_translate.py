"""HAL Gateway — shared OpenAI->Anthropic translation helpers (Codex interop).

Two cross-cutting concerns shared by the ``/v1/chat/completions`` (P2,
``gateway_openai``) and ``/v1/responses`` (P6, ``gateway_responses``) handlers:

1. **Image block translation.** OpenAI wire formats carry images as
   ``image_url`` (ChatCompletions) or ``input_image`` (Responses) content
   blocks — including nested inside a ``tool_result``'s content array when a
   harness (Codex) pastes a screenshot into a tool result. Anthropic's
   ``/v1/messages`` rejects those tags: its ONLY valid content tags are
   ``document``, ``image``, ``search_result``, ``text``, ``tool_reference``.
   :func:`translate_image_block` converts an OpenAI image block into an
   Anthropic ``image`` block (base64 for a data-URI, url-source for an http(s)
   URL); :func:`translate_content_blocks` walks a content array doing the same
   and recursing into nested ``tool_result`` content.

2. **Model alias resolution.** Codex sends its own model aliases (e.g.
   ``gpt-5.5``) that have no Anthropic backend, so a verbatim forward 404s
   ``model: gpt-5.5``. :func:`resolve_model` maps the known Codex aliases onto
   the Anthropic backend the operator runs (``anthropic/claude-opus-4-8``);
   any already-valid / unknown model is returned UNCHANGED (fail-open — an
   Anthropic-native ``claude-*`` request is never rewritten).

Pure dict-in/dict-out — fully testable without a socket. No coupling to the
HAL registry or catalogue (the gateway speaks raw wire JSON, syscall-doctrine).
"""
from __future__ import annotations

from typing import Optional

# --- Bug 2: Codex model-alias resolution ------------------------------------

# The Anthropic backend the operator drives through the gateway.
_OPUS_4_8 = "anthropic/claude-opus-4-8"

# Codex / OpenAI-wire aliases that have no Anthropic backend. Mapped onto the
# operator's opus-4-8 so a verbatim forward stops 404-ing. Explicit dict so a
# wrong entry fails a Goodhart test rather than silently mis-routing. The
# stripped ``anthropic/`` prefix forms are accepted too (a harness may send the
# bare model id) and resolve to the SAME backend.
_MODEL_ALIASES = {
    "gpt-5.5": _OPUS_4_8,
    "gpt-5.5-codex": _OPUS_4_8,
    "gpt-5-codex": _OPUS_4_8,
    "gpt-5": _OPUS_4_8,
}


def resolve_model(model) -> str:
    """Map a Codex alias onto the Anthropic backend; pass anything else through.

    A known Codex alias (case-insensitive) -> ``anthropic/claude-opus-4-8``.
    An already-valid Anthropic model (``claude-*``, ``anthropic/...``) or any
    unknown string is returned UNCHANGED so the gateway never rewrites a
    request that was already addressed correctly (fail-open).
    """
    if not isinstance(model, str) or not model:
        return model if isinstance(model, str) else ""
    return _MODEL_ALIASES.get(model.strip().lower(), model)


# --- Bug 1: OpenAI image block -> Anthropic image block ---------------------

# OpenAI content-block type tags that carry an image. ChatCompletions uses
# ``image_url`` (url under ``image_url.url``); Responses uses ``input_image``
# (url under ``image_url`` as a bare string, or ``image_url.url``).
_IMAGE_BLOCK_TYPES = frozenset({"image_url", "input_image"})


def _image_url_of(block: dict) -> str:
    """Extract the image URL/data-URI from an OpenAI image block.

    Handles both shapes: ``{"image_url": {"url": "..."}}`` (ChatCompletions and
    some Responses producers) and ``{"image_url": "..."}`` (Responses bare
    string). Returns "" when no URL is present.
    """
    raw = block.get("image_url")
    if isinstance(raw, dict):
        return raw.get("url", "") or ""
    if isinstance(raw, str):
        return raw
    return ""


def image_source_from_url(url: str) -> Optional[dict]:
    """Build an Anthropic ``image`` block from a data-URI or http(s) URL.

    ``data:<mime>;base64,<b64>`` -> base64 source; ``http(s)://...`` -> url
    source. Returns ``None`` for an empty/unsupported URL so callers can drop
    the block rather than emit an invalid one.
    """
    if not url:
        return None
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        media_type = header[5:].split(";", 1)[0] or "image/png"
        return {"type": "image", "source": {
            "type": "base64", "media_type": media_type, "data": data}}
    if url.startswith("http://") or url.startswith("https://"):
        return {"type": "image", "source": {"type": "url", "url": url}}
    return None


def translate_image_block(block: dict) -> Optional[dict]:
    """OpenAI image block (``image_url``/``input_image``) -> Anthropic ``image``.

    Returns ``None`` if the block carries no usable URL (caller drops it).
    """
    return image_source_from_url(_image_url_of(block))


def translate_content_blocks(content):
    """Translate an OpenAI content array into Anthropic content blocks.

    - ``text`` / ``input_text`` / ``output_text`` -> Anthropic ``text``.
    - ``image_url`` / ``input_image`` -> Anthropic ``image`` (base64 or url).
    - ``tool_result`` -> recurse into its ``content`` (this is where Codex
      nests a pasted screenshot: ``tool_result.content[0].type == input_image``)
      and translate the inner blocks, so no ``input_image`` survives.
    - any other block is preserved verbatim (already-Anthropic blocks pass
      through untouched — fail-open).

    A bare string is returned unchanged. Pure: no I/O.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content
    out: list = []
    for block in content:
        translated = _translate_one_block(block)
        if translated is not None:
            out.append(translated)
    return out


def _translate_one_block(block):
    """Translate a single content block (helper for translate_content_blocks)."""
    if not isinstance(block, dict):
        return block
    btype = block.get("type")
    if btype in _IMAGE_BLOCK_TYPES:
        return translate_image_block(block)  # may be None -> dropped
    if btype in ("text", "input_text", "output_text"):
        return {"type": "text", "text": block.get("text", "")}
    if btype == "tool_result":
        inner = block.get("content")
        return {**block, "content": translate_content_blocks(inner)}
    return block  # already-Anthropic / unknown block: preserve verbatim
