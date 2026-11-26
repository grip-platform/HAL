"""Run the Synthesis Ladder from JSON: a task and its rungs in, the synthesis out.

GRIP's AGORA cannot import this package, because both repositories name their
top-level package ``lib`` and one process can hold only one of them. So it calls
HAL the way it already calls ``hal --happi-api``: as a subprocess. That also
keeps the seat roster (models, roles, prompts) with the caller and the
machinery here.

    python -m lib.hal.synthesis < request.json > synthesis.json

A request looks like this:

    {"task": {"question": "...", "evidence": ["..."],
              "verdicts": ["MERGE", "DEFER_CHANGES", "REQUEST_CHANGES"],
              "stakes": "normal", "privacy": "private_cloud",
              "excluded_vendors": ["deepseek", "anthropic"]},
     "rungs": [{"effort": "HIGH",
                "seats": [{"model": "gemini/gemini-2.5-flash-lite",
                           "vendor": "google", "role": "verifier",
                           "billing": "postpaid", "system": "...",
                           "max_tokens": 2048}]}],
     "floor": "HIGH", "ceiling": "XHIGH", "budget_usd": 0.10, "agreement": 1.0}

``budget_usd`` is required. A malformed request exits 2 with the reason on
stderr. A well-formed one always prints a synthesis and exits 0, decided or
not: a provider failure is a seat that did not vote, never a crash. Seats are
asked over HAPPI (``happi_chat``).
"""
from __future__ import annotations

import json
import math
import sys
import uuid
from dataclasses import asdict
from enum import Enum
from typing import Any, Iterable, Optional, Sequence, TextIO

from lib.hal.synthesis.contract import Billing, Effort, Privacy, Stakes, Synthesis, SynthesisTask
from lib.hal.synthesis.rungs import ChatFn, ChatReply, SeatRung, SeatSpec
from lib.hal.synthesis.runner import run_ladder


class RequestError(ValueError):
    """The request cannot be read as a task and its rungs."""


def parse_task(data: Any) -> SynthesisTask:
    """A SynthesisTask from its JSON form; RequestError names what was wrong."""
    if not isinstance(data, dict):
        raise RequestError("task: expected a JSON object")
    question = data.get("question")
    if not isinstance(question, str) or not question.strip():
        raise RequestError("task: question must be a non-empty string")
    lists = {name: _strings(data.get(name), name)
             for name in ("evidence", "verdicts", "excluded_vendors")}
    try:
        return SynthesisTask(
            question=question,
            evidence=lists["evidence"],
            verdicts=lists["verdicts"],
            stakes=Stakes(data.get("stakes", Stakes.NORMAL.value)),
            privacy=Privacy(data.get("privacy", Privacy.PRIVATE_CLOUD.value)),
            excluded_vendors=frozenset(_family(v) for v in lists["excluded_vendors"]),
        )
    except (KeyError, ValueError) as exc:
        raise RequestError(f"task: {exc!r}") from exc


def _strings(value: Any, name: str) -> tuple[str, ...]:
    """A JSON list of strings, or nothing when absent.

    A bare string is refused rather than iterated: iterating it yields its
    characters, so ``"verdicts": "MERGE"`` would become five verdicts and
    ``"excluded_vendors": "google"`` would exclude no one at all.
    """
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RequestError(f"{name}: expected a JSON list of strings")
    return tuple(value)


def parse_seat(data: Any) -> SeatSpec:
    """A SeatSpec from its JSON form; the billing must be a known mode."""
    if not isinstance(data, dict):
        raise RequestError("seat: expected a JSON object")
    # Text, not merely truthy: a vendor of ["google"] would otherwise become
    # "['google']" and slip past an exclusion naming "google".
    bad = [key for key in ("model", "vendor", "role", "billing") if not _text(data.get(key))]
    if bad:
        raise RequestError(f"seat: {', '.join(bad)} must be non-empty text")
    system = data.get("system")
    if system is None:
        system = ""
    if not isinstance(system, str):
        raise RequestError("seat: system must be text")
    max_tokens = data.get("max_tokens")
    if max_tokens is None:            # JSON null means "not given", as for floor and ceiling
        max_tokens = 2_048
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise RequestError("seat: max_tokens must be a whole number, 1 or more")
    try:
        # Trimmed and lower-cased like the vendor, so " POSTPAID" reads as postpaid.
        billing = Billing(data["billing"].strip().lower())
    except ValueError as exc:
        raise RequestError(f"seat: unknown billing {data['billing']!r}") from exc
    return SeatSpec(model=data["model"].strip(), vendor=_family(data["vendor"]),
                    role=data["role"].strip(), billing=billing, system=system,
                    max_tokens=max_tokens)


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _family(name: str) -> str:
    """A model family as exclusion compares it: trimmed and lower-case.

    Applied to both a seat's vendor and the exclusion list, so "Google" or
    "google " cannot sit past an exclusion naming "google".
    """
    return name.strip().lower()


def build_rungs(data: Sequence[Any], chat: ChatFn) -> list[SeatRung]:
    return [_build_rung(rung, chat) for rung in data]


def _build_rung(data: Any, chat: ChatFn) -> SeatRung:
    if not isinstance(data, dict):
        raise RequestError("rung: expected a JSON object")
    effort = _effort(data.get("effort"))
    seats = _as_list(data.get("seats"), f"{effort.name} seats")
    if not seats:
        raise RequestError(f"{effort.name} seats: a rung needs at least one seat")
    return SeatRung(effort, [parse_seat(seat) for seat in seats], chat)


def _effort(name: Any) -> Effort:
    try:
        return Effort[str(name).upper()]
    except KeyError as exc:
        raise RequestError(f"unknown effort {name!r}") from exc


def run_request(request: Any, chat: ChatFn) -> Synthesis:
    """Validate the whole request first, then climb the ladder it describes."""
    if not isinstance(request, dict):
        raise RequestError("the request must be a JSON object")
    task = parse_task(request.get("task"))
    rungs = build_rungs(_as_list(request.get("rungs"), "rungs"), chat)
    if not rungs:
        raise RequestError("rungs: a ladder needs at least one rung")
    budget, agreement = _ladder_settings(request)
    # JSON null means "not given", so the defaults apply to it too.
    floor = _effort(request.get("floor") or "DETERMINISTIC")
    ceiling = _effort(request.get("ceiling") or "XHIGH")
    if floor > ceiling:
        raise RequestError(f"floor {floor.name} is above ceiling {ceiling.name}")
    return run_ladder(task, rungs, floor=floor, ceiling=ceiling,
                      budget_usd=budget, agreement=agreement)


def _ladder_settings(request: dict) -> tuple[float, float]:
    """The budget, which is required so a forgotten key cannot quietly ask no one,
    and the agreement bar, a fraction."""
    if "budget_usd" not in request:
        raise RequestError("budget_usd: required, in dollars (0 asks no paid seat)")
    try:
        budget = float(request["budget_usd"])
        agreement = float(request.get("agreement", 1.0))
    except (TypeError, ValueError) as exc:
        raise RequestError(f"ladder settings: {exc!r}") from exc
    if not (math.isfinite(budget) and budget >= 0 and 0.0 <= agreement <= 1.0):
        raise RequestError("ladder settings: budget_usd must be 0 or more and "
                           "agreement between 0 and 1")
    return budget, agreement


def _as_list(value: Any, name: str) -> list:
    if not isinstance(value, list):
        raise RequestError(f"{name}: expected a JSON list")
    return value


def to_json(result: Synthesis) -> dict:
    """The synthesis as plain JSON: the effort by name, billing modes by value."""
    data = asdict(result)
    data["effort"] = result.effort.name
    for seat in data["seats"]:
        seat["billing"] = Billing(seat["billing"]).value
    return data


def _json_default(value: Any) -> Any:
    """What json cannot encode alone: an enum by name, a set as a sorted list."""
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"cannot encode {type(value).__name__}")


def happi_chat(address: str, system: str, prompt: str, max_tokens: int) -> ChatReply:
    """One seat over HAPPI, in process.

    HAPPI needs only the standard library, and it is the transport AGORA's CI
    already uses for every speaker and judge, so a runner with no third-party
    packages installed can still seat a council. The envelope carries no auth,
    so each adapter finds its key the way it does for AGORA: the vendor's
    environment variable, then the keychain. Temperature 0 keeps a judge's
    verdict repeatable.
    """
    from lib.hal.happi.runner import run_envelope_line
    envelope = {"v": "happi/1.0", "id": f"synthesis-{uuid.uuid4().hex[:12]}", "cmd": "chat",
                "content": prompt, "auth": {},
                "flags": {"provider": address, "system": system,
                          "max_tokens": max_tokens, "temperature": 0}}
    return _reply_from_events(run_envelope_line(json.dumps(envelope)))


def _reply_from_events(lines: Iterable[str]) -> ChatReply:
    """Fold HAPPI's NDJSON events into one reply: deltas joined, usage, any error.

    A line that is not a JSON object is skipped instead of sinking an answer
    that arrived, and usage that is not an object counts as unreported. When
    unreadable lines are all there was, the reply says so as its error. An
    error event means the seat did not answer, so any partial text is dropped
    and cannot be read as a verdict; a delta whose text is not a string adds
    nothing.
    """
    text: list[str] = []
    usage: dict = {}
    error = ""
    unreadable = 0
    for line in lines:
        event = _event(line)
        if event is None:
            unreadable += bool(line.strip())
            continue
        kind = event.get("type")
        if kind == "delta" and isinstance(event.get("text"), str):
            text.append(event["text"])
        elif kind == "completed" and isinstance(event.get("usage"), dict):
            usage = event["usage"]
        elif kind == "error":
            error = f"{event.get('code', 'error')}: {event.get('message', '')}"
    if unreadable and not text and not error:
        error = f"{unreadable} unreadable HAPPI line(s) and no answer"
    return ChatReply(text="" if error else "".join(text), tokens_in=_count(usage.get("in")),
                     tokens_out=_count(usage.get("out")), error=error)


def _event(line: str) -> Optional[dict]:
    try:
        event = json.loads(line)
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def _count(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _refuse(reason: str) -> int:
    print(f"synthesis: {reason}", file=sys.stderr)
    return 2


def main(chat: Optional[ChatFn] = None, stdin: Optional[TextIO] = None,
         stdout: Optional[TextIO] = None) -> int:
    try:
        request = json.load(stdin or sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return _refuse(f"the request is not JSON: {exc}")
    try:
        result = run_request(request, chat or happi_chat)
    except RequestError as exc:
        return _refuse(str(exc))
    out = stdout or sys.stdout
    json.dump(to_json(result), out, default=_json_default)
    out.write("\n")
    return 0
