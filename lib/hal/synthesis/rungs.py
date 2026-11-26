"""The two rung shapes every Synthesis Ladder rung is built from.

``DeterministicRung`` runs checks that cost nothing (AGORA registers its quote,
claim and temporal gates as these). ``SeatRung`` asks one or more model seats --
one for LOW and MEDIUM, three for a HIGH council -- through an injected chat
function, so every rung is testable without a network and the production path
goes through ``registry_chat``.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, replace
from typing import Callable, Optional, Sequence

from lib.hal.cost_tracker import calculate_cost
from lib.hal.synthesis.contract import (
    MONEY_BILLING,
    Billing,
    Effort,
    Privacy,
    Seat,
    Synthesis,
    SynthesisTask,
    undecided,
)

# A check returns (verdict or "", note). A verdict outside the task's tokens is ignored.
Check = Callable[[SynthesisTask], "tuple[str, str]"]

#: The most of the previous rung's answer a prompt carries, so estimate() can
#: bound a prompt it builds before the previous rung has run.
PRIOR_CAP_CHARS = 2_000
#: The most tokens that answer and its header line can add: up to four UTF-8
#: bytes a character, and at most one token a byte.
PRIOR_BOUND_TOKENS = PRIOR_CAP_CHARS * 4 + 128
#: Chat-template framing a provider wraps around the messages.
FRAMING_TOKENS = 256

_VERDICT_RE = re.compile(r"^[\s*_#>]*VERDICT[\s*_]*:[\s*_]*([A-Za-z][A-Za-z _]*)",
                         re.MULTILINE | re.IGNORECASE)


@dataclass(frozen=True)
class ChatReply:
    """What a chat call returned; ``error`` is set instead of raising."""

    text: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    error: str = ""


# (provider/model address, system prompt, user prompt, max output tokens) -> reply
ChatFn = Callable[[str, str, str, int], ChatReply]


@dataclass(frozen=True)
class SeatSpec:
    """A seat a rung may fill: which model, whose family, in what role, paid how.

    ``ready`` is the presence gate -- a key is set, a local model is installed,
    a subscription pool has headroom. It must not spend anything.
    """

    model: str
    vendor: str
    role: str
    billing: Billing
    system: str
    max_tokens: int = 2_048
    ready: Callable[[], bool] = lambda: True


class DeterministicRung:
    """Checks that cost nothing; their notes become evidence for the next rung."""

    effort = Effort.DETERMINISTIC

    def __init__(self, checks: Sequence[Check]):
        self._checks = tuple(checks)

    def available(self, task: SynthesisTask) -> bool:
        return True

    def estimate(self, task: SynthesisTask) -> float:
        return 0.0

    def synthesise(self, task: SynthesisTask, prior: Optional[Synthesis]) -> Synthesis:
        allowed = frozenset(task.verdicts)
        outcomes = [_run_check(check, task) for check in self._checks]
        notes = [note for _verdict, note in outcomes if note]
        verdicts = [v for v, _note in outcomes if v and (not allowed or v in allowed)]
        agreed = set(verdicts)
        decided = len(agreed) == 1
        return Synthesis(effort=self.effort, decided=decided,
                         verdict=verdicts[0] if decided else "",
                         answer="\n".join(notes), agreement=1.0 if decided else 0.0,
                         dissent=tuple(sorted(agreed)) if len(agreed) > 1 else ())


def _run_check(check: Check, task: SynthesisTask) -> tuple[str, str]:
    try:
        verdict, note = check(task)
    except Exception as exc:  # noqa: BLE001 -- a broken check is a note, never a crash
        return "", f"{getattr(check, '__name__', 'check')} raised {type(exc).__name__}: {exc}"
    return str(verdict or ""), str(note or "")


class SeatRung:
    """Asks every eligible seat, then tallies their verdicts."""

    def __init__(self, effort: Effort, seats: Sequence[SeatSpec], chat: ChatFn):
        self.effort = effort
        self._seats = tuple(seats)
        self._chat = chat

    def available(self, task: SynthesisTask) -> bool:
        return bool(self._eligible(task))

    def estimate(self, task: SynthesisTask) -> float:
        prompt_bound = upper_bound_tokens(render_prompt(task, None)) + PRIOR_BOUND_TOKENS
        return sum(seat_money(s, prompt_bound + upper_bound_tokens(s.system) + FRAMING_TOKENS,
                              s.max_tokens)
                   for s in self._eligible(task))

    def synthesise(self, task: SynthesisTask, prior: Optional[Synthesis]) -> Synthesis:
        seats = self._eligible(task)
        if not seats:
            return undecided(self.effort, "no eligible seat (excluded, private, or not ready)")
        prompt = render_prompt(task, prior)
        results = tuple(self._ask(spec, prompt, task) for spec in seats)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return replace(tally(self.effort, task, results), prompt_sha256=digest)

    def _eligible(self, task: SynthesisTask) -> list[SeatSpec]:
        return [s for s in self._seats if _eligible(s, task)]

    def _ask(self, spec: SeatSpec, prompt: str, task: SynthesisTask) -> Seat:
        try:
            reply = self._chat(spec.model, spec.system, prompt, spec.max_tokens)
        except Exception as exc:  # noqa: BLE001 -- a dead seat is a vote not cast
            reply = ChatReply(error=f"{type(exc).__name__}: {exc}")
        return Seat(model=spec.model, vendor=spec.vendor, role=spec.role, billing=spec.billing,
                    verdict=parse_verdict(reply.text, task.verdicts), text=reply.text,
                    tokens_in=reply.tokens_in, tokens_out=reply.tokens_out,
                    cost_usd=seat_money(spec, reply.tokens_in, reply.tokens_out),
                    ok=not reply.error and bool(reply.text.strip()), note=reply.error)


def _eligible(spec: SeatSpec, task: SynthesisTask) -> bool:
    if spec.vendor in task.excluded_vendors or spec.billing is Billing.FREE_TIER:
        return False
    if task.privacy is Privacy.LOCAL_ONLY and spec.billing is not Billing.LOCAL:
        return False
    try:
        return bool(spec.ready())
    except Exception:  # noqa: BLE001 -- a presence probe that fails means not ready
        return False


def seat_money(spec: SeatSpec, tokens_in: int, tokens_out: int) -> float:
    """USD a seat's call adds; zero for local and subscription seats."""
    if spec.billing not in MONEY_BILLING:
        return 0.0
    return calculate_cost(spec.model, input_tokens=tokens_in, output_tokens=tokens_out)


def upper_bound_tokens(text: str) -> int:
    """The most tokens ``text`` can cost: its UTF-8 byte count.

    Byte-level tokenizers emit at most one token a byte. Characters per token
    is not a bound -- CJK text and emoji can cost a token a character or more.
    For English this over-estimates roughly fourfold, which is the price of an
    estimate that is a true upper bound.
    """
    return len(text.encode("utf-8"))


def render_prompt(task: SynthesisTask, prior: Optional[Synthesis]) -> str:
    parts = [task.question]
    parts += [f"--- evidence {i} ---\n{block}" for i, block in enumerate(task.evidence, 1)]
    if prior is not None:
        said = (prior.verdict + "\n" + prior.answer).strip()[:PRIOR_CAP_CHARS]
        parts.append(f"--- the previous rung ({prior.effort.name}) said ---\n{said or '(nothing)'}")
    if task.verdicts:
        parts.append("Begin with one line 'VERDICT: <token>', where <token> is exactly one of: "
                     + " | ".join(task.verdicts) + ". Then give your reasons.")
    return "\n\n".join(parts)


def parse_verdict(text: str, allowed: Sequence[str]) -> str:
    """The allowed token the first usable VERDICT line names, else "".

    Spaces become underscores ("REQUEST CHANGES" -> REQUEST_CHANGES), and words
    after the token are ignored. The tokens are compiled into one alternation,
    longest first, so a short token can never shadow a longer one that shares
    its start, and each line is checked in a single pass.
    """
    if not allowed:
        return ""
    tokens = "|".join(re.escape(t) for t in sorted(allowed, key=len, reverse=True))
    token_re = re.compile(rf"({tokens})(?:_|$)")
    for match in _VERDICT_RE.finditer(text or ""):
        hit = token_re.match(re.sub(r"\s+", "_", match.group(1).strip().upper()))
        if hit:
            return hit.group(1)
    return ""


def tally(effort: Effort, task: SynthesisTask, seats: tuple[Seat, ...]) -> Synthesis:
    """Turn seats into a synthesis. A failed seat counts against agreement."""
    if task.verdicts:
        return _tally_verdicts(effort, seats)
    answers = [s for s in seats if s.ok]
    # Several free-form answers need a synthesiser to reconcile them, so only a
    # single answering seat decides at this shape.
    single = len(seats) == 1 and len(answers) == 1
    return Synthesis(effort=effort, decided=single, agreement=1.0 if single else 0.0,
                     answer=answers[0].text if single else "", seats=seats,
                     cost_usd=sum(s.cost_usd for s in seats),
                     reasons=() if answers else ("no seat answered",))


def _tally_verdicts(effort: Effort, seats: tuple[Seat, ...]) -> Synthesis:
    votes = Counter(s.verdict for s in seats if s.ok and s.verdict)
    if not votes:
        return undecided(effort, "no seat gave a valid verdict", seats=seats)
    verdict, count = votes.most_common(1)[0]
    voice = next(s for s in seats if s.ok and s.verdict == verdict)
    return Synthesis(effort=effort, decided=count * 2 > len(seats), verdict=verdict,
                     answer=voice.text, agreement=count / len(seats), seats=seats,
                     cost_usd=sum(s.cost_usd for s in seats),
                     dissent=tuple(f"{s.model}: {s.verdict or 'no verdict'}"
                                   for s in seats if s.verdict != verdict))


def registry_chat(registry) -> ChatFn:
    """A ChatFn over HAL's ProviderRegistry.

    ``ProviderRegistry.chat(model_address, prompt, **kwargs)`` forwards to
    ``OpenAICompatProvider.chat(prompt, *, model, system=..., max_tokens=...)``.
    HAL's ``CouncilRunner`` called it with ``messages=..., model=...`` instead,
    so against the real registry every seat raised TypeError, which it then
    swallowed -- it has never returned a vote. Temperature 0 keeps a judge's
    verdict repeatable.
    """
    def chat(address: str, system: str, prompt: str, max_tokens: int) -> ChatReply:
        response = registry.chat(address, prompt, system=system,
                                 max_tokens=max_tokens, temperature=0.0)
        usage = response.usage
        return ChatReply(text=response.text or "", tokens_in=usage.prompt_tokens,
                         tokens_out=usage.completion_tokens)
    return chat
