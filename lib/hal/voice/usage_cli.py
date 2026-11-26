"""HAL Voice — ``hal voice <subcommand>`` CLI entrypoint.

Per plans/voice-tier3-cloud-fallover.md §4 W5.

Subcommands:

    hal voice say <text> [--provider P] [--ladder L]
    hal voice usage [--month YYYY-MM] [--path FILE]

Exit code 0 on success; 1 when usage exceeds the 80% warning threshold.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .usage import (
    FREE_TIER_LIMITS,
    UsageCounter,
    UsageRow,
    WARN_THRESHOLD_PCT,
)


def _format_row(row: UsageRow) -> str:
    if row.free_tier_limit is None:
        limit_cell = "n/a"
        pct_cell = "n/a"
    else:
        limit_cell = f"{row.free_tier_limit:>10,}"
        pct_cell = f"{(row.percent_used or 0):>5.1f}%"
    warn = "  WARN" if row.over_warn_threshold else ""
    return (
        f"{row.provider:<28}  "
        f"{row.chars_this_month:>10,}  "
        f"{limit_cell}  "
        f"{pct_cell}"
        f"{warn}"
    )


def render(rows: list[UsageRow]) -> str:
    """Format the per-provider usage table as a string."""
    header = (
        f"{'provider':<28}  "
        f"{'chars/mo':>10}  "
        f"{'limit':>10}  "
        f"{'pct':>6}"
    )
    separator = "-" * len(header)
    body_lines = [_format_row(r) for r in rows]
    over = [r for r in rows if r.over_warn_threshold]
    trailer_lines = []
    if over:
        trailer_lines.append(
            f"\nWARNING: {len(over)} provider(s) over "
            f"{WARN_THRESHOLD_PCT:.0f}% of free tier"
        )
    return "\n".join([header, separator, *body_lines, *trailer_lines])


def _cmd_listen(args: argparse.Namespace) -> int:
    from .voice import listen
    from .types import VoiceError, PlatformUnsupported
    print(
        f"Listening for {args.duration:.0f}s — speak now ...",
        file=sys.stderr,
    )
    try:
        result = listen(provider="whisper", max_duration_s=args.duration)
        print(result.text)
    except (VoiceError, PlatformUnsupported) as exc:
        print(f"voice error: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_say(args: argparse.Namespace) -> int:
    from .voice import speak, VoiceError
    text = " ".join(args.text)
    try:
        speak(
            text=text,
            provider=args.provider or None,
            ladder=args.ladder or None,
        )
    except VoiceError as exc:
        print(f"voice error: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_usage(args: argparse.Namespace) -> int:
    path = Path(args.path) if args.path else None
    counter = UsageCounter(path=path)
    rows = counter.summary(ym=args.month)
    print(render(rows))
    over_count = sum(1 for r in rows if r.over_warn_threshold)
    return 1 if over_count else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hal voice",
        description="HAL voice harness CLI — usage telemetry and Tier-3 introspection",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    listen_p = subparsers.add_parser("listen", help="record mic and transcribe via Whisper STT")
    listen_p.add_argument(
        "--duration", type=float, default=10.0,
        help="recording duration in seconds (default: 10)",
    )
    listen_p.set_defaults(func=_cmd_listen)

    say_p = subparsers.add_parser("say", help="synthesise speech via the Tier-3 provider ladder")
    say_p.add_argument("text", nargs="+", help="text to speak")
    say_p.add_argument("--provider", default="", help="force a specific provider (e.g. macos_say)")
    say_p.add_argument("--ladder", default="", help="use a named ladder (e.g. fallover)")
    say_p.set_defaults(func=_cmd_say)
    usage_p = subparsers.add_parser(
        "usage", help="show per-provider monthly usage + free-tier limits"
    )
    usage_p.add_argument(
        "--month",
        default=None,
        help="YYYY-MM bucket to report (default: current UTC month)",
    )
    usage_p.add_argument(
        "--path",
        default=None,
        help="override usage.jsonl path (default: ~/.hal/state/voice/usage.jsonl)",
    )
    usage_p.set_defaults(func=_cmd_usage)

    # realtime full-duplex voice (GA OpenAI Realtime) — the subcommand registers
    # itself; lazy import keeps realtime deps out of the `usage`/`say` paths.
    from .realtime_cli import build_realtime_subparser
    build_realtime_subparser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
