"""HAL ChannelFormatTranscoder — make a message look right on its destination.

Pure text transform, zero dependencies. Fixes the per-platform formatting that
actually breaks when the same message is sent to different chat apps. This is
NOT a full markdown engine — it normalises the handful of constructs that render
differently across WhatsApp / Slack / Discord (see rules/operational-context.md
"Platform Rendering Quirks"). New platform quirks are added here, in one place.
"""
from __future__ import annotations

import re

WHATSAPP = "whatsapp"
SLACK = "slack"
DISCORD = "discord"

# **bold** or __bold__  ->  *bold*   (WhatsApp + Slack both use single-star bold)
_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
# `inline code` — a single backtick pair, not part of a ``` fence (lookarounds
# exclude an adjacent backtick, and [^`\n] keeps it on one line).
_INLINE_CODE = re.compile(r"(?<!`)`([^`\n]+?)`(?!`)")


def _bold_to_single_star(text: str) -> str:
    return _BOLD.sub(lambda m: "*" + (m.group(1) or m.group(2)) + "*", text)


def _to_whatsapp(text: str) -> str:
    # WhatsApp: **x**/__x__ -> *x*; single-backtick inline shows the backticks
    # literally (they are noise), so strip them. Triple-backtick fences DO render
    # and are left untouched by _INLINE_CODE's lookarounds.
    text = _bold_to_single_star(text)
    return _INLINE_CODE.sub(lambda m: m.group(1), text)


def _to_slack(text: str) -> str:
    # Slack mrkdwn: single-star bold; inline + block code render fine — keep them.
    return _bold_to_single_star(text)


def _to_discord(text: str) -> str:
    # Discord speaks standard markdown — nothing to fix.
    return text


# Dispatch table over per-platform renderers (no if/elif chain).
_RENDERERS = {
    WHATSAPP: _to_whatsapp,
    SLACK: _to_slack,
    DISCORD: _to_discord,
}


class UnsupportedPlatform(KeyError):
    """transcode() was asked for a platform with no rendering rule (fail-closed)."""


class ChannelFormatTranscoder:
    """Rewrite markup so a message renders correctly on the destination platform."""

    def transcode(self, text: str, *, from_platform=None, to_platform: str) -> str:
        """Return `text` rewritten for `to_platform`.

        `from_platform` is accepted for future source-aware rules; the current
        ruleset renders from a common markdown-ish source to the target.

        Fail-closed: an unknown `to_platform` raises rather than silently passing
        un-transcoded text through (a typo like 'whatsap', or a peer endpoint on a
        platform with no rule yet). Adding a platform = add one renderer here, by
        contract (docs/human-channel-gateway.md).
        """
        renderer = _RENDERERS.get(to_platform)
        if renderer is None:
            raise UnsupportedPlatform(
                f"no transcode rule for platform {to_platform!r}; "
                f"known: {', '.join(sorted(_RENDERERS))}"
            )
        return renderer(text)
