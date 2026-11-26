"""``hal decide``: ask a decision model typed questions from the command line.

    hal decide --state "Payouts have failed for 3 days." --noul urgent "Is this urgent?"
    hal decide --request body.json         # a whole System One request body
    hal decide --request - < body.json     # the same, from standard input

Prints the result as JSON. Exits 0 when answered, 3 when no model could answer
(the result's ``reason`` says why), and 2 on a usage error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from lib.hal.decide import DEFAULT_MODEL, decide, noul


def _parse(argv: Optional[list]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="hal decide", description=__doc__.splitlines()[0])
    parser.add_argument("--state", help="the text to judge")
    parser.add_argument("--noul", nargs=2, action="append", default=[], metavar=("ID", "QUESTION"),
                        help="a yes/no question; repeat for more, all go in one call")
    parser.add_argument("--request", metavar="FILE", help="a System One request body ('-' for stdin)")
    parser.add_argument("--model", help=f"a decision model id (default: the body's, else {DEFAULT_MODEL})")
    return parser.parse_args(argv)


def _request(args: argparse.Namespace) -> dict:
    if args.request:
        if args.request == "-":
            return json.loads(sys.stdin.read())
        # bin/hal runs from HAL_HOME and names the caller's directory, so a
        # relative path is read where the caller typed it.
        path = Path(os.environ.get("HAL_CALLER_CWD") or ".") / Path(args.request).expanduser()
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    return {"state": args.state, "questions": {qid: noul(text) for qid, text in args.noul}}


def main(argv: Optional[list] = None) -> int:
    args = _parse(argv)
    if not args.request and (args.state is None or not args.noul):
        print("hal decide: give --state with at least one --noul, or --request FILE", file=sys.stderr)
        return 2
    try:
        body = _request(args)
    except (OSError, ValueError) as exc:
        print(f"hal decide: cannot read the request ({type(exc).__name__})", file=sys.stderr)
        return 2
    if not isinstance(body, dict):
        print("hal decide: the request must be a JSON object", file=sys.stderr)
        return 2
    result = decide(body.get("state"), body.get("questions"),
                    model=args.model or body.get("model") or DEFAULT_MODEL)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.ok else 3


if __name__ == "__main__":
    sys.exit(main())
