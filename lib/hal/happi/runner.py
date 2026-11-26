"""HAPPI/1.0 γ-2 queue driver.

Reads a polyglot HAPPI plan document, extracts the fenced `ndjson` block
that carries the `happi-run-queue` envelope list, and dispatches each
envelope through the same `runtime._dispatch` generator the stdin/stdout
runtime + β-1 HTTP + β-3 socket transports all use.

Wire behaviour:

  Input:  one polyglot plan file (e.g. `plans/happi-protocol-triphase.md`)
          containing a `ndjson`-labelled fence whose body is one JSON
          envelope per line. Empty lines + comment lines starting with
          `//` are skipped.

  Output: a single NDJSON stream combining all envelopes' events in
          order: envelope N's events appear only after envelope N-1's
          `completed` (or `error`) has been emitted. Parse errors on
          individual envelope lines produce a `parse_error` event
          attributed to a synthetic id and do NOT stop the queue.

Library API:

    from lib.hal.happi.runner import run_queue_file

    for ndjson_line in run_queue_file("plans/happi-protocol-triphase.md"):
        print(ndjson_line)

CLI:

    python -m lib.hal.happi.runner <plan.md>  [--label ndjson]

Writes NDJSON events to stdout, returns 0 on success (queue dispatched
to completion), 1 on document parse failure (file not found, no queue
block).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import IO, Iterator

from lib.hal.happi.envelope import EnvelopeError, HappiEnvelope
from lib.hal.happi.events import emit_error
from lib.hal.happi.parser import parse_happi_file
from lib.hal.happi.runtime import _dispatch


QUEUE_LABEL = "ndjson"


def extract_queue_lines(block_body: str) -> list[str]:
    """Split a fenced-block body into envelope candidate lines.

    Drops empty lines and lines that start with ``//`` (allowing inline
    comments alongside envelope lines inside the NDJSON fence).
    """
    lines: list[str] = []
    for raw in block_body.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("//"):
            continue
        lines.append(stripped)
    return lines


def run_envelope_line(raw: str) -> Iterator[str]:
    """Parse + dispatch one envelope line. Yields NDJSON event strings.

    Parse errors emit a single ``parse_error`` event tagged with a
    synthetic id so downstream consumers can correlate. Never raises —
    the queue continues even if one line is malformed.
    """
    try:
        envelope = HappiEnvelope.parse(raw)
    except EnvelopeError as exc:
        yield emit_error(
            "queue-invalid",
            f"envelope parse failed: {exc}",
            code="parse_error",
        )
        return

    yield from _dispatch(envelope, raw.encode("utf-8"))


def run_queue_lines(lines: list[str]) -> Iterator[str]:
    """Iterate `lines` (each a raw JSON envelope), dispatching each in order."""
    for raw in lines:
        yield from run_envelope_line(raw)


def run_queue_file(
    path: str, label: str = QUEUE_LABEL
) -> Iterator[str]:
    """Open a polyglot plan, extract the queue block, dispatch envelopes.

    If the document has no block with `label`, yields a single
    ``queue_missing`` error event and terminates.
    """
    document = parse_happi_file(path)
    blocks = document.get_all(label)
    if not blocks:
        yield emit_error(
            "queue-missing",
            f"no {label!r}-labelled fenced block in {path!r}",
            code="queue_missing",
        )
        return

    # Use the first block only — multiple queue blocks would be ambiguous
    # for ordering guarantees.
    lines = extract_queue_lines(blocks[0])
    yield from run_queue_lines(lines)


def main(argv: list[str] | None = None, stream_out: IO[str] = sys.stdout) -> int:
    """CLI entry: dispatch a queue from a polyglot plan file to stdout."""
    parser = argparse.ArgumentParser(
        prog="happi-run",
        description="Dispatch the happi-run-queue block of a polyglot HAPPI plan.",
    )
    parser.add_argument(
        "plan",
        help="Path to a polyglot plan file with an `ndjson` fenced block.",
    )
    parser.add_argument(
        "--label",
        default=QUEUE_LABEL,
        help=f"Fence label to extract (default: {QUEUE_LABEL!r}).",
    )
    args = parser.parse_args(argv)

    if not Path(args.plan).exists():
        stream_out.write(
            emit_error(
                "queue-missing",
                f"plan file not found: {args.plan!r}",
                code="queue_missing",
            )
            + "\n"
        )
        stream_out.flush()
        return 1

    for line in run_queue_file(args.plan, label=args.label):
        stream_out.write(line + "\n")
        stream_out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
