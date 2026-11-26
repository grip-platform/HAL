#!/usr/bin/env python3
"""redaction_shield.py — GRIP's canonical redaction / PII-shield syscall.

ONE shield for every surface that carries free text toward an audience that
must not learn client identities: the public decision chain, rendered
artifacts, channel payloads, and prompts leaving for a cloud LLM.

This is a copy of GRIP's lib/redaction_shield.py for HAL.

Origin (2026-07-02): the public decision-chain ledger
(codetonight-sa.github.io/grip-decision-chain) carried client/external
identities in its free-text fields — the emitter's leak scan covered only
absolute paths and private-key blocks, so client names, personal emails,
Tailscale IPs and WhatsApp JIDs were structurally invisible to it. Third
confirmed instance of the "decorative real-data on an ungated surface" class
(1: rendered video frames, rules/visual-artifact-leak-prevention.md;
2: this ledger; 3: un-shielded LLM egress) — N=3, so the scattered per-surface
regexes mature into this single shim per rules/knowledge-maturation-functor.md.

Influences (absorbed, not copied — GRIP-native rewrite):
  - an earlier in-house PII-shield module — session-stable placeholders,
                                     longest-first substitution, deanonymize
                                     mapping, local-only LLM detector
                                     (fail-closed egress stance)
  - nexus-poc services/redaction.py — single combined alternation scan, span
                                     collect → longest-wins overlap dedupe →
                                     single-pass rebuild, span manifest + counts
  - bin/render-leak-sweep +
    lib/happiverse_proof._LEAK_PATTERNS — the real-data denylist seed

Two profiles:
  public     -- deterministic denylist + generic PII floor; opaque pinned
                pseudonyms ([ORG-1], [EMAIL]). Same text in ⇒ same text out
                (the decision-chain Banach publish contract depends on this).
  llm_egress -- everything public does, PLUS pattern-level PERSON/ORG/CASE and
                street-address discovery and a session mapping, so "the context
                still knows what we are talking about": the LLM sees [PERSON-8]
                used consistently, and deanonymize() restores originals locally.
                One deliberate difference: a ``topic`` entry (subject vocabulary,
                see _PUBLIC_ONLY_CLASSES) is removed on public and kept here,
                because a reviewer cannot read a document whose subject is gone.

Layering (rules/llm-augmentation-pattern.md): this module is the L1
deterministic floor. ``llm_scan`` is the optional L2 semantic recall — it can
only ADD findings (monotone toward redaction), fails open to L1 unless
``strict=True``, and never rewrites text (so the deterministic publish
pipeline stays deterministic; L2 findings feed the denylist instead).

The denylist config (config/redaction-denylist.json) is git-crypt encrypted —
the entity list itself names the identities it protects. ``denylist_state``
distinguishes missing / ciphertext / invalid so publish gates can fail CLOSED
on a locked checkout instead of silently publishing unshielded text.

Complexity: every scan tier is ONE compiled alternation over the text —
O(k · n) for k constant tiers, no nested per-pattern iteration (the same
single-pass design as nexus-poc ``_PII_ALTERNATION`` and the earlier in-house
module's ``_ENTITY_PATTERN``). Dedupe is sort-then-scan, rebuild is one
cursor pass.

Falsification: this shield is wrong if, over its first ~10 governed publishes,
(a) it redacts allowlisted public brands (over-block — fix the allowlist, not
the shield); (b) a denylisted identity still reaches a governed surface
(matcher defect — widen, never abandon); or (c) profile=public output differs
across runs on identical input (determinism broken — that is a bug, the
Banach contract depends on it). Track via tests/test_redaction_shield.py.

Sufficiency (2026-10-01). Measured against a matter's own pseudonymiser on fake
data, the shield alone was NOT a sufficient egress control, and the two had to
be composed. Eight defects, each now pinned by a test in
tests/test_redaction_shield_sufficiency.py: ``canonical`` was never compiled
(an entry whose canonical sat in no alias leaked); aliases were wrapped in
``\\b``, so one ending in ``)`` or ``.`` could never match; no Unicode
normalisation (a decomposed ``ë`` missed a precomposed alias); no 13-digit
identity-number shape and no ``0xx`` local phone shape on the floor; one
single-word allowlist entry disarmed name detection for any name carrying it;
``--config`` took one list; ``verify`` had no strict mode; and ``deanonymize``
restored the first form seen rather than the canonical. ``safe()`` is the
pseudonymiser's discipline absorbed: redact, then RE-READ the output and raise
on anything that survived.

Usage:
    from redaction_shield import Shield, load_denylist
    shield = Shield(load_denylist())                    # public profile
    result = shield.redact("Met Jane Example at Acme Corp")
    # result.text == "Met [PERSON-1] at [ORG-1]"

    session = ShieldSession()
    egress = Shield(load_denylist(), profile="llm_egress", session=session)
    out = egress.redact("Email bob@example.com about Acme Corp")
    restored = session.deanonymize(llm_reply)

CLI:
    venv/bin/python -m lib.redaction_shield redact  [--profile public] [file|-]
    venv/bin/python -m lib.redaction_shield verify  [file|-]     # exit 2 on findings
    venv/bin/python -m lib.redaction_shield verify --strict [file|-]
        # exit 2 on ANY residue: what redact() would replace PLUS, under
        # public, the name-shaped spans blockers() would hold a publish for
    venv/bin/python -m lib.redaction_shield verify --config matter.json --config repo.json -
        # --config is repeatable; lists merge in the order given, and at
        # equal alias length the FIRST list wins (put the matter list first)
    venv/bin/python -m lib.redaction_shield selftest             # idempotence proof
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field, replace
from itertools import chain
from pathlib import Path

_LIB = Path(__file__).resolve().parent
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))
def get_claude_dir() -> Path:
    """HAL root: a non-empty HAL_DIR, otherwise this repository root."""
    configured = os.environ.get("HAL_DIR", "").strip()
    return Path(configured).expanduser() if configured else _LIB.parent


DEFAULT_CONFIG = get_claude_dir() / "config" / "redaction-denylist.json"
# Names the operator's private guardian repository, whose
# config/redaction-denylist.json is merged first (private_overlay_path). Only
# the operator's own machines carry this gitignored file.
GUARDIAN_LOCAL = get_claude_dir() / "config" / "guardian.local.json"
OVERLAY_ENV = "GRIP_REDACTION_OVERLAY"      # a path, or "none" to turn it off
_GITCRYPT_MAGIC = b"\x00GITCRYPT"

PROFILE_PUBLIC = "public"
PROFILE_LLM_EGRESS = "llm_egress"
_PROFILES = (PROFILE_PUBLIC, PROFILE_LLM_EGRESS)


class RedactionConfigError(RuntimeError):
    """The denylist config exists but cannot be used (ciphertext / invalid)."""


class ShieldResidue(RuntimeError):
    """Text the shield re-read still holds something that must not leave.

    Raised by ``Shield.safe`` and by ``Shield.verify(strict=True)``. The
    message names categories and offsets ONLY — never the matched text —
    because an exception message is the one thing that reliably ends up in a
    log, and the text is the thing being kept out of logs."""

    def __init__(self, findings: list) -> None:
        self.findings = list(findings)
        where = ", ".join(f"{s.category}@{s.start}" for s in self.findings)
        super().__init__(f"shield residue: {len(self.findings)} finding(s): {where}")


def _nfc(text: str) -> str:
    """Canonical composition, so one spelling of ``ë`` cannot slip past the other.

    Applied to every alias at compile time and to every text at scan time;
    ``redact()`` therefore returns NFC text and span offsets index into it."""
    return unicodedata.normalize("NFC", text or "")


# ---------------------------------------------------------------------------
# Spans + results (nexus-poc redaction.py influence — manifest, not just text)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Span:
    """One replacement the shield made (or would make, in verify mode)."""

    start: int
    end: int
    original: str = field(repr=False)  # the unredacted text: never in a repr
    category: str        # entity class ("org", "person", …) or floor name ("email")
    replacement: str
    canonical: str = field(default="", repr=False)  # the listed entry's real name; "" otherwise


@dataclass(frozen=True)
class ShieldResult:
    """Redacted text plus the full span manifest — audit-ready by construction."""

    text: str
    spans: tuple
    counts: dict

    @property
    def changed(self) -> bool:
        return bool(self.spans)


# ---------------------------------------------------------------------------
# Denylist config — git-crypted, pinned pseudonyms, three-state loadability
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DenylistEntry:
    canonical: str
    aliases: tuple          # of (text, case_sensitive) pairs
    pseudonym: str          # pinned token, e.g. "[ORG-1]" — determinism anchor
    cls: str                # "org" | "person" | "project" | "host" | "phone" | "matter" | "topic"
    # False once an earlier list owns the canonical (Denylist.merge): the entry
    # still restores to it in deanonymize(), but no longer matches it.
    compile_canonical: bool = True


@dataclass(frozen=True)
class Denylist:
    entries: tuple
    allowlist: tuple        # names that must NEVER be redacted (public brands)

    @classmethod
    def empty(cls) -> "Denylist":
        return cls(entries=(), allowlist=())

    @classmethod
    def merge(cls, *lists: "Denylist") -> "Denylist":
        """One list from several, in the order given.

        Order is load-bearing. A spelling belongs to the FIRST list that has
        it: a later list's entry loses that spelling, and an entry with no
        spelling left is dropped. Without this, a private overlay that files
        a word as ``topic`` was overridden under llm_egress by the same word
        filed as ``matter`` further down, because egress skips the topic entry
        and the matter one then matched. An entry that keeps some aliases but
        whose canonical is owned stops matching that canonical too, or the
        compiler would add it straight back (Codex, P1, on GRIP#7575); it still
        restores to it. Among the surviving aliases the compiler still
        sorts longest-first and breaks ties by this order, which is why a
        matter list goes before the repo list. Allowlists union, first
        occurrence kept."""
        owned, entries = set(), []
        for dl in lists:
            entries.extend(e for e in (_unowned(e, owned) for e in dl.entries) if e)
            owned |= {s for e in dl.entries for s in _spellings(e)}
        allow = tuple(dict.fromkeys(chain.from_iterable(dl.allowlist for dl in lists)))
        return cls(entries=tuple(entries), allowlist=allow)


def _spellings(entry: DenylistEntry) -> set:
    """Every spelling ``entry`` matches -- aliases and canonical -- NFC-lowercased."""
    return {_nfc(text).lower() for text, _ in entry.aliases} | {_nfc(entry.canonical).lower()}


def _unowned(entry: DenylistEntry, owned: set):
    """``entry`` less the spellings an earlier list owns; None when nothing is left."""
    left = tuple(a for a in entry.aliases if _nfc(a[0]).lower() not in owned)
    canonical_owned = _nfc(entry.canonical).lower() in owned
    if not left and canonical_owned:
        return None
    if len(left) == len(entry.aliases) and not canonical_owned:
        return entry
    return replace(entry, aliases=left, compile_canonical=not canonical_owned)


def denylist_state(path: Path | None = None) -> str:
    """Non-raising classification: 'ok' | 'missing' | 'ciphertext' | 'invalid'.

    Publish gates branch on this — a git-crypt-locked checkout reads
    'ciphertext' and must fail CLOSED rather than publish unshielded text.

    Called with no path, the private overlay counts too: one that is
    configured but unusable reads 'overlay-missing', 'overlay-ciphertext' or
    'overlay-invalid', so every gate that already refuses on anything but
    'ok' refuses on that as well. A path given explicitly -- even the default
    one -- is classified alone (Codex, P2, on GRIP#7575)."""
    target = DEFAULT_CONFIG if path is None else Path(path)
    state = _file_state(target)
    if state != "ok" or path is not None:
        return state
    overlay = overlay_state()
    return state if overlay in ("ok", "disabled", "unconfigured") else f"overlay-{overlay}"


def private_overlay_path() -> Path | None:
    """The operator's private denylist, merged ahead of the shared one.

    Personal entries live in the operator's private guardian repository, never
    in GRIP (rules/personal-matters-stay-private.md), so GRIP finds them
    through config/guardian.local.json, a gitignored file only the operator's
    machines carry. $GRIP_REDACTION_OVERLAY names a file directly, and "none"
    turns the overlay off. None when nothing names one."""
    named = os.environ.get(OVERLAY_ENV, "").strip()
    if named:
        return None if named == "none" else Path(named).expanduser()
    if not GUARDIAN_LOCAL.exists():
        return None
    repo = _guardian_repo()
    return Path(repo).expanduser() / "config" / "redaction-denylist.json" if repo else None


def _guardian_repo() -> str:
    """The guardian_repo value from config/guardian.local.json, "" if unreadable."""
    try:
        repo = json.loads(GUARDIAN_LOCAL.read_text()).get("guardian_repo")
    except (OSError, ValueError, AttributeError):
        return ""
    return repo if isinstance(repo, str) else ""


def overlay_state() -> str:
    """'disabled' | 'unconfigured' | 'ok' | 'missing' | 'ciphertext' | 'invalid'.

    'unconfigured' is a machine with no overlay at all (CI, a teammate). A
    guardian config that names no usable repository reads 'invalid': the
    operator meant to have an overlay, so it fails closed rather than quietly
    becoming 'unconfigured'."""
    if os.environ.get(OVERLAY_ENV, "").strip() == "none":
        return "disabled"
    path = private_overlay_path()
    if path is None:
        return "invalid" if GUARDIAN_LOCAL.exists() else "unconfigured"
    return _file_state(path)


def _file_state(path: Path) -> str:
    """The loadability of one denylist file (see ``denylist_state``)."""
    if not path.exists():
        return "missing"
    raw = path.read_bytes()
    if raw.startswith(_GITCRYPT_MAGIC):
        return "ciphertext"
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return "invalid"
    # A JSON array or scalar is invalid too; calling .get on one would raise,
    # breaking the non-raising promise above.
    ok = isinstance(data, dict) and isinstance(data.get("entities"), list)
    return "ok" if ok else "invalid"


def _normalise_aliases(entry: dict) -> tuple:
    """Config aliases are strings (case-insensitive) or {"alias","cs"} dicts."""
    out = []
    for a in entry.get("aliases", []):
        if isinstance(a, dict):
            out.append((str(a["alias"]), bool(a.get("cs", False))))
        else:
            out.append((str(a), False))
    return tuple(out)


def _parse_entry(e: dict) -> DenylistEntry:
    return DenylistEntry(
        canonical=str(e["canonical"]),
        aliases=_normalise_aliases(e),
        pseudonym=str(e["pseudonym"]),
        cls=str(e.get("class", "entity")),
    )


def load_denylist(path: Path | None = None) -> Denylist:
    """Load the canonical denylist. Raises on unusable states so callers can
    never silently run with a weaker list than they believe they have.

    Called with no path, the default list comes with the operator's private
    overlay merged FIRST (load_overlay), so its entries win ties
    (Denylist.merge). A path given explicitly -- even the default one -- loads
    that file alone; a caller that also wants the overlay merges it itself,
    as lib/review_redaction.py does."""
    if path is not None:
        return _load_file(Path(path))
    return Denylist.merge(load_overlay(), _load_file(DEFAULT_CONFIG))


def load_overlay() -> Denylist:
    """The operator's private overlay alone: ``Denylist.empty()`` (never None)
    where there is none, and a raise where one is configured but unusable,
    exactly as the shared list raises -- so no caller silently runs without
    entries it believes it has."""
    overlay = overlay_state()
    if overlay in ("disabled", "unconfigured"):
        return Denylist.empty()
    if overlay != "ok":
        raise RedactionConfigError(
            f"private redaction overlay unusable ({overlay}): "
            f"{private_overlay_path() or GUARDIAN_LOCAL}"
        )
    return _load_file(private_overlay_path())


def _load_file(path: Path) -> Denylist:
    """One denylist file, raising on any unusable state."""
    state = _file_state(path)
    if state == "missing":
        raise FileNotFoundError(f"redaction denylist not found: {path}")
    if state != "ok":
        raise RedactionConfigError(
            f"redaction denylist unusable ({state}): {path} — "
            "on a git-crypt-locked checkout run `git-crypt unlock` first"
        )
    data = json.loads(path.read_text())
    return Denylist(
        entries=tuple(_parse_entry(e) for e in data["entities"]),
        allowlist=tuple(data.get("allowlist", [])),
    )


def load_denylists(paths) -> Denylist:
    """Load several configs and merge them in the order given (see Denylist.merge).

    Each path goes through ``load_denylist``, so an unusable file anywhere in
    the sequence raises rather than quietly narrowing the protection."""
    return Denylist.merge(*(load_denylist(Path(p)) for p in paths))


# A standalone install (a public clone, with no operator config at all) carries
# no entries here by construction -- this is the explicit "generic built-in
# default", never a real name baked into the module. The structural PII floor
# (``_GENERIC_RE``: email/phone/IP/key/JID) is unaffected either way; only the
# named-entity denylist layer is empty.
GENERIC_DEFAULT_DENYLIST: Denylist = Denylist.empty()


def load_denylist_or_default(path: Path | None = None) -> Denylist:
    """Like ``load_denylist``, but a MISSING file degrades to
    ``GENERIC_DEFAULT_DENYLIST`` instead of raising -- the safe entry point for
    a standalone install with no operator denylist configured at all (the
    common case for anyone who is not this project's own operator). A file
    that IS configured but unusable (ciphertext/invalid) still raises: that is
    a real misconfiguration, not an absent one, and deserves to fail loud
    exactly as ``load_denylist`` already does."""
    try:
        return load_denylist(path)
    except FileNotFoundError:
        return GENERIC_DEFAULT_DENYLIST


# ---------------------------------------------------------------------------
# Generic PII floor — ONE combined alternation, both profiles
# ---------------------------------------------------------------------------

# Alternation order is precedence at equal start positions (leftmost-first):
# key > jid > email > id_number > phone > ip. A JID therefore beats the email
# pattern for the same "digits@host" span. Overlaps elsewhere resolve in _dedupe.
#
# jid: a WhatsApp ID in every spelling GRIP writes it. The JID form allows up to
# 30 digits, the most lib/wa_control_state.py accepts for a group JID, joined to
# its suffix by "@" or by the "_" the message mirror puts in its file names
# (state/whatsapp-history/<digits>_s.whatsapp.net.jsonl). The slug form is that
# module's folder name: g_ plus 8-30 digits for a group, lid_ plus 6-20 for a
# LID, matched only as a whole word because slug_for_jid writes nothing after the
# digits. Measured 2026-10-06: the file-name form and both slug forms passed this
# floor untouched, an 18-digit @lid JID did too, and an 18-digit @g.us JID was
# caught only as an email address. A person's slug is bare digits, so it is left
# to the names list, for the reason given under phone below.
#
# id_number: a 13-digit South African identity number, contiguous or broken at
# its field boundaries -- YYMMDD SSSS CA Z. A break may sit at any of those three
# boundaries, as one hyphen or up to two whitespace characters, so the 6-4-2-1
# form the ID book prints is caught along with 6-4-3, 6-7 and the rest. The date
# digits are validated (month 01-12, day 01-31) so a 13-digit millisecond epoch,
# the other 13-digit run common in engineering text, is not swallowed:
# 1790815018000 reads as month 90 and is left alone. Measured 2026-10-01: both
# spaced and contiguous fakes passed the old floor untouched. Measured
# 2026-10-05 (GRIP#7349): the 6-4-2-1 form still did, because a break was
# allowed only at 6-4-3; widening it matched nothing new in 5,979 tracked files.
#
# phone: the existing international form, plus the local 0xx form (0XX XXX
# XXXX, separators optional) — the shape a South African writes a mobile or
# landline in. Anchored on digits at both ends so a longer run (an identity
# number, an account number) is never cut into a "phone". Deliberately NOT the
# pseudonymiser's loose any-9-digit-run rule: that would turn every 10-digit
# epoch in a technical page into [PHONE].
_GENERIC_RE = re.compile(
    r"(?P<key>\b(?:AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|xox[baprs]-[\w-]{10,}"
    r"|sk-[A-Za-z0-9_-]{16,}|gsk_[A-Za-z0-9_-]{16,}|AIza[A-Za-z0-9_-]{12,}))"
    r"|(?P<jid>\b\d{6,30}[@_](?:s\.whatsapp\.net|g\.us|lid|c\.us)\b"
    r"|\b(?:g_\d{8,30}|lid_\d{6,20})\b)"
    r"|(?P<email>\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b)"
    r"|(?P<id_number>(?<!\d)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])"
    r"(?:\s{1,2}|-)?\d{4}(?:\s{1,2}|-)?\d{2}(?:\s{1,2}|-)?\d(?!\d))"
    r"|(?P<phone>\+\d[\d\s().-]{6,16}\d\b"
    r"|(?<!\d)0\d{2}[\s.-]?\d{3}[\s.-]?\d{4}(?!\d))"
    r"|(?P<ip>\b(?:\d{1,3}\.){3}\d{1,3}\b)"
)
# Loopback/wildcard hosts are documentation, not topology (same exclusion as
# the earlier in-house module's own _LOCAL_HOSTS).
_LOCAL_IPS = frozenset({"127.0.0.1", "0.0.0.0"})

# llm_egress discovery (same influence as the earlier in-house PII shield):
# catch entities the denylist has never heard of before they leave for a
# cloud provider. Same single-alternation design; CASE first (most specific),
# then ORG, then PERSON.
_NER_RE = re.compile(
    r"(?P<case>\b[A-Z]{2,8}-\d{4}-\d{2,6}\b)"
    r"|(?P<org>\b[A-Z][A-Za-z\-&.']{0,30}\s+"
    r"(?:Corp|Inc|Ltd|LLC|LLP|LP|PLC|GmbH|AG|NV|SA|Pty|Group|Holdings|"
    r"Partners|Associates|Attorneys|Lawyers|Chambers|Foundation|Trust)\b\.?)"
    r"|(?P<person>\b[A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){1,2}\b)"
)

# A line that opens with one of these words opens an ANSWER -- a form's "Not
# applicable", "Yes", "None" -- and nobody's name runs on into one. Without this
# stop, a label's last capitalised word and the answer under it read as a
# two-word name: "6. Vehicle\nNot applicable" became "6. [PERSON-1] applicable" and
# "Status\nYes" vanished into one token (measured 2026-10-05, GRIP#7349). Only
# line breaks are cut, never a word: cutting the words themselves anywhere in
# the text let an identity containing one through, because the rest was too
# short to match -- "No Holdings" and "John No" (Codex, P1, on GRIP#7506). The
# answer line is scanned on its own, cut from the line below it too, so in a
# stacked form "Not Applicable\nNotes" is not read as a three-word name.
_ANSWER_LINE_RE = re.compile(r"(?m)^[ \t]*(?:Not|No|Yes|None|Nil|Unknown)\b.*$")

# Title-cased form answers that are two capitalised words, so the name pattern
# read each one as a person: "Status: Not Applicable" became "Status: [PERSON-1]"
# (measured 2026-10-05 on the GRIP#7502 review). A closed list matched EXACTLY,
# like a topic phrase: no name is spelled this way, and a longer candidate
# ("Vehicle Not Applicable") is still treated as a name. Lower case, one space.
_ANSWER_PHRASES = frozenset({
    "not applicable", "not known", "not available", "not provided", "not stated",
    "not specified", "not required", "not sure", "not given", "no change",
    "no changes", "no comment", "no answer", "no response", "no data",
    "none recorded", "none known",
})

# A spared phrase that runs on into more of a name is part of that name, so it is
# never spared: "No Change, Inc.", "No Change (Pty) Ltd" and "No Change & Co."
# each came back unredacted from an adversarial check through HAL on GRIP#7506
# (DeepSeek, over two rounds). Main caught them all. Same line only, so a form's
# stacked fields ("Not Applicable" above "Notes") stay readable.
_NAME_CONTINUES_RE = re.compile(r"[ \t]*(?:&|\(|,[ \t]*[A-Z])")

# llm_egress: a street address, as ONE token. The name pattern alone read
# "12 Example Street, Some Town" as two names and left the house number in the
# clear -- "12 [PERSON-1], [PERSON-2]" (measured 2026-10-05, GRIP#7349). Shape:
# a house number, one to three capitalised words and a street word, then up to
# two capitalised place names and a four-digit postal code, each after a comma.
_ADDRESS_RE = re.compile(
    r"\b\d{1,5}[A-Za-z]?(?:\s+[A-Z][a-z]+){1,3}\s+"
    r"(?:Street|Road|Avenue|Drive|Lane|Close|Crescent|Way|Place|Court|Boulevard|"
    r"Terrace|Square|Grove|Rise|Walk|Parade|Highway|Mews|St|Rd|Ave|Dr|Ln|Cres|Blvd)\b\.?"
    r"(?:,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}){0,2}(?:,?\s*\d{4}\b)?"
)


# ---------------------------------------------------------------------------
# Session mapping — "the context still knows what we are talking about"
# ---------------------------------------------------------------------------


@dataclass
class ShieldSession:
    """Session-stable entity↔token registry (same influence as the earlier
    in-house module's PiiSession).

    The same entity maps to the same token for the life of the session, so a
    shielded conversation stays coherent for the LLM; ``deanonymize`` restores
    originals locally. The mapping never leaves the machine. Not thread-safe —
    one session per conversation/stream."""

    _fwd: dict = field(default_factory=dict)   # original -> token
    _rev: dict = field(default_factory=dict)   # token -> original
    _counters: dict = field(default_factory=dict)

    def seed_counters(self, counters: dict) -> None:
        """Start minting above the config-pinned indices so a session token
        can never collide with a pinned pseudonym."""
        for k, v in counters.items():
            self._counters[k] = max(self._counters.get(k, 0), v)

    def register(self, original: str, cls: str, pinned: str | None = None,
                 canonical: str | None = None) -> str:
        """Map ``original`` to a token; ``canonical`` is what the token restores to.

        A listed entity is met in whichever alias the text happened to use —
        "Bob", "B. Example" — and the first one seen used to become what every
        later ``deanonymize`` wrote back. The reverse map now holds the entry's
        canonical, so a reply mentioning the token restores the full name
        regardless of which alias the prompt carried. A discovered (unlisted)
        entity has no canonical and restores to the form it was seen in."""
        if original in self._fwd:
            return self._fwd[original]
        if pinned is None:
            n = self._counters.get(cls, 0) + 1
            self._counters[cls] = n
            pinned = f"[{cls.upper()}-{n}]"
        self._fwd[original] = pinned
        self._rev.setdefault(pinned, canonical or original)
        return pinned

    def deanonymize(self, text: str) -> str:
        """Single-pass token→original restoration (longest token first)."""
        if not self._rev:
            return text
        pattern = re.compile(
            "|".join(re.escape(t) for t in sorted(self._rev, key=len, reverse=True))
        )
        return pattern.sub(lambda m: self._rev[m.group(0)], text)

    def snapshot(self) -> dict:
        """Copy of original→token for local audit — never publish this."""
        return dict(self._fwd)


# ---------------------------------------------------------------------------
# The Shield
# ---------------------------------------------------------------------------


_PIN_INDEX = re.compile(r"^\[([A-Z]+)-(\d+)\]$")


# A `case` span whose text is a PUBLIC identifier, not a private matter
# reference. CVE ids match the case pattern exactly (`[A-Z]{2,8}-\d{4}-\d{2,6}`)
# and are the opposite of sensitive — they are published by definition, and a
# security PR quotes several. Handled as a RULE rather than as allowlist rows
# because new CVE numbers appear constantly; a list would need editing forever
# and would block a genuine security fix in the meantime.
_PUBLIC_IDENTIFIER_PREFIXES = ("CVE-", "CWE-", "GHSA-", "RFC-")


def _is_public_identifier(span) -> bool:
    """True for an identifier that is public by construction."""
    return span.category == "case" and span.original.upper().startswith(
        _PUBLIC_IDENTIFIER_PREFIXES
    )


# A "topic" entry is subject vocabulary: a word that says what a private matter
# is ABOUT, not who is in it. It must stay off a public surface, where it would
# tell a stranger what someone is dealing with. But a reviewer sent that very
# document needs its subject to do the job, and with every subject word replaced
# they are handed a form about nothing (GRIP#7349). So llm_egress keeps topic
# entries and still removes every identity class. "matter" is deliberately NOT
# in this set: there it names a legal matter, which identifies the people in it.
_PUBLIC_ONLY_CLASSES = frozenset({"topic"})


def _entries_for(profile: str, entries: tuple) -> tuple:
    """The denylist entries a profile rewrites: every one on public, and on
    llm_egress every one outside _PUBLIC_ONLY_CLASSES."""
    if profile == PROFILE_PUBLIC:
        return entries
    return tuple(e for e in entries if e.cls not in _PUBLIC_ONLY_CLASSES)


def _topic_phrases(entries: tuple) -> frozenset:
    """Every spelling of every public-only entry, NFC-lowercased with its
    whitespace collapsed: the phrases the egress name pattern must leave alone."""
    return frozenset(
        " ".join(_nfc(text).lower().split())
        for e in entries if e.cls in _PUBLIC_ONLY_CLASSES
        for text in chain((alias for alias, _ in e.aliases), (e.canonical,))
    )


class Shield:
    """Deterministic redactor over a pinned denylist + generic PII floor.

    profile="public": opaque tokens, pure function of (denylist, text).
    profile="llm_egress": adds pattern discovery + a ShieldSession mapping."""

    def __init__(self, denylist: Denylist, profile: str = PROFILE_PUBLIC,
                 session: ShieldSession | None = None):
        if profile not in _PROFILES:
            raise ValueError(f"unknown profile {profile!r} — one of {_PROFILES}")
        self.denylist = denylist
        self.profile = profile
        self.session = session or (
            ShieldSession() if profile == PROFILE_LLM_EGRESS else None
        )
        if self.session is not None:
            self.session.seed_counters(_pinned_counters(denylist))
        # The entries this profile rewrites; _match_spans indexes into this tuple.
        self._entries = _entries_for(profile, denylist.entries)
        # Exact candidates the egress name pattern leaves alone (_egress_spans).
        self._spared = _topic_phrases(denylist.entries) | _ANSWER_PHRASES
        self._ci_re, self._ci_map, self._cs_re, self._cs_map = _compile_denylist(
            self._entries
        )
        self._allow_lower = frozenset(a.lower() for a in denylist.allowlist)

    # -- public API ---------------------------------------------------------

    def redact(self, text: str) -> ShieldResult:
        """Redact ``text``; returns new text + the span manifest. Idempotent:
        redact(redact(x).text) makes no further changes (tokens never re-match).
        The returned text is NFC-normalised (see ``_nfc``)."""
        text = _nfc(text)
        spans = self._final_spans(text)
        if self.session is not None:
            spans = [self._sessionise(s) for s in spans]
        new_text, counts = _rebuild(text, spans)
        return ShieldResult(text=new_text, spans=tuple(spans), counts=counts)

    def verify(self, text: str, strict: bool = False) -> list:
        """Detector mode: the spans redact() WOULD replace, no rewrite, no
        session mutation. Publish gates call this — empty list ⇒ clean.

        ``strict`` widens the question from "what would I rewrite" to "what
        must not leave" (``residue``) and RAISES ``ShieldResidue`` on a
        non-empty answer, so a gate written as ``shield.verify(t, strict=True)``
        cannot forget to check the return value."""
        if not strict:
            return self._final_spans(_nfc(text))
        found = self.residue(text)
        if found:
            raise ShieldResidue(found)
        return found

    def residue(self, text: str) -> list:
        """Everything a strict gate refuses over, in ONE offset frame.

        What ``redact()`` would replace, plus — under ``public`` only — the
        name-shaped spans ``blockers()`` reports, since on that profile they are
        detected but never rewritten. Under ``llm_egress`` the NER spans are
        already inside ``_final_spans``. Overlaps resolve through ``_dedupe``,
        so a listed name is reported once, as its pinned span."""
        text = _nfc(text)
        found = self._final_spans(text)
        if self.profile == PROFILE_PUBLIC:
            found = _dedupe(found + self.blockers(text))
        return found

    def safe(self, text: str) -> ShieldResult:
        """Redact, then RE-READ the output and raise if anything survived.

        Absorbed from ``wa_pseudonym.Pseudonymiser.safe``: a scrubber that
        reports success is making a claim about its own operation, and the
        only check worth having is the one that looks at the text about to
        leave. A substitution can fail for reasons the substituting code
        cannot see — a shape the floor does not model, a name nobody listed —
        and those are observable only in the output."""
        out = self.redact(text)
        left = self.residue(out.text)
        if left:
            raise ShieldResidue(left)
        return out

    def blockers(self, text: str) -> list:
        """Name-shaped spans that must REFUSE a public publish, not be rewritten.

        The asymmetry this closes: ``_ner_spans`` — the only detector that finds
        a person, an organisation or a case reference — runs under ``llm_egress``
        and NOT under ``public``. Measured 2026-09-16 on one sentence, ``public``
        returned 1 finding (an email) where ``llm_egress`` returned 3 (person,
        org, email). So the path to the OPEN INTERNET saw less than the path to
        a vendor API, and the public path's entire name protection was the
        39-entry denylist: a name nobody had listed went straight through.

        Why this is a separate channel rather than adding NER to ``_final_spans``:
        the person pattern matches any two or three consecutive Capitalised
        Words, so REWRITING with it would replace "Google Drive", "Plain English"
        and "Primary Author" with ``[PERSON]`` and destroy an ordinary technical
        page. Detection is right here; rewriting is not. ``redact()`` therefore
        stays exactly as it was — deterministic, denylist-driven — and this
        reports what a human must decide about.

        The decision is binary and cheap: the finding is a real identity, so it
        goes on the DENYLIST and gets redacted; or it is a technical phrase, so
        it goes on the ALLOWLIST and never blocks again. That is the remedy the
        module's own falsification clause already prescribes for over-blocking —
        "fix the allowlist, not the shield".

        Empty under ``llm_egress`` on purpose: there the same spans are already
        REWRITTEN by ``redact()``, so blocking as well would refuse over text the
        shield had itself just made safe.

        Called on ALREADY-SHIELDED text by the publish gate, which is what keeps
        it quiet about denylisted identities — those are ``[ORG-1]`` tokens by
        then and match no name pattern.
        """
        if self.profile != PROFILE_PUBLIC:
            return []
        text = _nfc(text)
        # A match spanning a newline is a FIELD-JOIN ARTEFACT, not a name. The
        # gate concatenates a row's five free-text fields with "\n" and the
        # person pattern's `\s+` happily crosses that boundary, manufacturing
        # findings like "Mac\n\nAdds" out of two unrelated sentences. Dropped
        # here rather than by re-writing the shared _NER_RE, because that regex
        # also drives llm_egress REWRITING and narrowing it there would silently
        # stop redacting a name that genuinely wraps a line.
        return [
            s for s in _dedupe(self._ner_spans(text))
            if "\n" not in s.original and not _is_public_identifier(s)
        ]

    # -- span collection (each collector = ONE alternation pass) -------------

    def _final_spans(self, text: str) -> list:
        spans = self._denylist_spans(text) + _generic_spans(text)
        if self.profile == PROFILE_LLM_EGRESS:
            spans += self._egress_spans(text)
        return _dedupe(spans)

    def _denylist_spans(self, text: str) -> list:
        ci = _match_spans(self._ci_re, self._ci_map, self._entries, text)
        cs = _match_spans(self._cs_re, self._cs_map, self._entries, text)
        return ci + cs

    def _egress_spans(self, text: str) -> list:
        """llm_egress discovery: names, organisations and case references, found
        one answer-delimited piece at a time so none runs on into an answer line
        (_ANSWER_LINE_RE), less any candidate that is exactly a topic phrase or a
        listed answer phrase (_ANSWER_PHRASES), plus street addresses.

        A title-cased topic phrase -- "Warranty Claim" as a heading -- is two
        capitalised words, so the name pattern took it for a person and undid
        the topic class (measured 2026-10-05 on the GRIP#7502 review). Only an
        EXACT match is spared: no name fits inside a phrase made wholly of topic
        or answer words, but one can stand beside such a phrase, so a longer
        candidate such as "Re Warranty Claim" is still treated as a name.

        ``blockers()`` keeps the unsegmented ``_ner_spans`` on purpose: the public
        gate's verdicts stay exactly as they were, and this change touches only
        the profile whose rewriting produced the defect."""
        found = [replace(s, start=s.start + offset, end=s.end + offset)
                 for offset, piece in _answer_pieces(text)
                 for s in self._ner_spans(piece)
                 if not self._is_spared(piece, s)]
        return found + _address_spans(text)

    def _is_spared(self, piece: str, span: Span) -> bool:
        """True when ``span`` is exactly a topic or answer phrase and nothing on
        its line continues it into a longer name (_NAME_CONTINUES_RE)."""
        return (" ".join(span.original.lower().split()) in self._spared
                and not _NAME_CONTINUES_RE.match(piece, span.end))

    def _ner_spans(self, text: str) -> list:
        """llm_egress discovery: PERSON/ORG/CASE shapes not on the allowlist."""
        out = []
        for m in _NER_RE.finditer(text):
            if self._allowed(m.group(0)):
                continue
            name = m.lastgroup or "entity"
            out.append(Span(m.start(), m.end(), m.group(0),
                            name, f"[{name.upper()}]"))
        return out

    def _allowed(self, candidate: str) -> bool:
        """The whole candidate is allowlisted, or EVERY word of it is.

        This used to pass a candidate when ANY word of it was allowlisted, so
        one single-word brand switched name detection off for every name that
        carried it: with "GRIP" allowlisted, "Jane Grip" went through. The
        every-word form keeps "Grip Hal" (two allowlisted brands) quiet while
        a surname beside a brand is still a name. Measured 2026-10-01; the
        repo allowlist carries 20 single-word entries."""
        low = candidate.lower()
        if low in self._allow_lower:
            return True
        words = low.split()
        return bool(words) and all(w in self._allow_lower for w in words)

    def _sessionise(self, span: Span) -> Span:
        """In egress profile every replacement goes through the session so the
        mapping is reversible; denylist pins keep their config token."""
        pinned = span.replacement if _PIN_INDEX.match(span.replacement) else None
        token = self.session.register(span.original, span.category, pinned,
                                      canonical=span.canonical or None)
        return Span(span.start, span.end, span.original, span.category, token,
                    span.canonical)


def _match_spans(regex, group_map: dict, entries: tuple, text: str) -> list:
    """Spans for one compiled denylist alternation — a single finditer pass."""
    if regex is None:
        return []
    out = []
    for m in regex.finditer(text):
        entry = entries[group_map[m.lastgroup]]
        out.append(Span(m.start(), m.end(), m.group(0), entry.cls, entry.pseudonym,
                        entry.canonical))
    return out


# A unified diff marks every added line with "+", so an added line that opens
# with an ISO date ("+2026-10-08: ...") matches the international phone form
# above, and AGORA's council was shown "[PHONE]" where the file holds a date,
# then raised a concern about a placeholder the file does not contain
# (GRIP#7762, 2026-10-08). Exactly "+YYYY-MM-DD" is exempt, wherever it
# appears, because no number can be written that way: it begins 19 or 20, so
# its country code is 1 or 20 (country codes are prefix-free), and both plans
# need at least ten digits after the "+" (the shortest is an Alexandria
# landline, +20-3-xxx-xxxx), while a date carries eight. Month and day are
# validated as the id_number pattern does, so "+2026-13-08" stays redacted.
# A date followed by an hour ("+2026-10-08 12:34") stays redacted too: with the
# hour it reaches ten digits, the length of an Alexandria number, and what
# follows cannot be told apart from an extension (Codex review, GRIP#7770).
_PLUS_ISO_DATE_RE = re.compile(r"\+(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])")


def _is_plus_date(match) -> bool:
    """True when a phone-shaped match is exactly "+YYYY-MM-DD", which no
    numbering plan can dial."""
    return _PLUS_ISO_DATE_RE.fullmatch(match.group(0)) is not None


def _generic_spans(text: str) -> list:
    """Generic PII floor — one alternation pass; loopback IPs and an exact
    "+YYYY-MM-DD" (a diff's "+" before a date) exempted."""
    out = []
    for m in _GENERIC_RE.finditer(text):
        name = m.lastgroup or "pii"
        if name == "ip" and m.group(0) in _LOCAL_IPS:
            continue
        if name == "phone" and _is_plus_date(m):
            continue
        out.append(Span(m.start(), m.end(), m.group(0), name, f"[{name.upper()}]"))
    return out


def _answer_pieces(text: str):
    """(offset, piece) pairs: every answer line as a piece of its own, and the
    text between them, so a name pattern scanned per piece never runs into or
    out of an answer line. Every character stays in exactly one piece, at its
    own offset; nothing is dropped, so an identity on an answer line is still
    scanned whole."""
    start = 0
    for m in _ANSWER_LINE_RE.finditer(text):
        yield start, text[start:m.start()]
        yield m.start(), m.group(0)
        start = m.end()
    yield start, text[start:]


def _address_spans(text: str) -> list:
    """Street addresses (llm_egress only) -- one alternation pass."""
    return [Span(m.start(), m.end(), m.group(0), "address", "[ADDRESS]")
            for m in _ADDRESS_RE.finditer(text)]


def _pinned_counters(denylist: Denylist) -> dict:
    counters: dict = {}
    for e in denylist.entries:
        m = _PIN_INDEX.match(e.pseudonym)
        if not m:
            continue
        cls, n = m.group(1).lower(), int(m.group(2))
        counters[cls] = max(counters.get(cls, 0), n)
        counters[e.cls] = max(counters.get(e.cls, 0), n)
    return counters


def _entry_alias_triples(indexed) -> list:
    """(alias_text, entry_idx, case_sensitive) triples for ONE entry.

    The canonical is compiled too — case-insensitively — whenever no alias
    already spells it. Before this, ``canonical`` was a label and nothing
    more: an entry whose canonical sat in no alias loaded cleanly, reported
    state ``ok`` and matched nothing. Measured 2026-10-01 on a matter list,
    one such entry reached the output; the repo list (41 entries) spells
    every canonical as an alias already, so there this adds no match."""
    idx, entry = indexed
    triples = [(text, idx, case_sensitive) for text, case_sensitive in entry.aliases]
    spelled = {_nfc(text).lower() for text, _ in entry.aliases}
    canonical = entry.canonical if entry.compile_canonical else ""
    if canonical and _nfc(canonical).lower() not in spelled:
        triples.append((canonical, idx, False))
    return triples


def _split_by_case(entries: tuple) -> tuple:
    """Partition (alias_text, entry_idx) pairs into case-insensitive/-sensitive.
    Flat single-pass over the flattened alias triples — per-entry expansion is
    delegated to _entry_alias_triples, then each partition is one filter."""
    triples = list(chain.from_iterable(map(_entry_alias_triples, enumerate(entries))))
    ci = [(text, idx) for text, idx, case_sensitive in triples if not case_sensitive]
    cs = [(text, idx) for text, idx, case_sensitive in triples if case_sensitive]
    return ci, cs


def _compile_denylist(entries: tuple):
    """Two alternations (case-insensitive + case-sensitive), aliases sorted
    longest-first so 'Jane Example' always beats 'Jane' at the same position.
    Returns (ci_re, ci_map, cs_re, cs_map); a regex is None when it has no
    aliases (re.compile('') would match everywhere)."""
    ci, cs = _split_by_case(entries)
    return (*_compile_alternation(ci, re.IGNORECASE),
            *_compile_alternation(cs, 0))


def _compile_alternation(aliases: list, flags: int):
    """One alternation, longest alias first, each alias NFC-normalised and
    fenced by ``(?<!\\w)`` / ``(?!\\w)`` rather than ``\\b``.

    ``\\b`` is a WORD boundary: it demands a word character on exactly one
    side. An alias that ends in ``)``, ``.`` or ``>`` — "Example (Pty)",
    "Acme Corp." — has a non-word character on both sides of its last
    position, so ``\\b`` can never sit there and the alias can never match.
    The lookarounds say the real thing: nothing word-like touches either
    end. For an alias made of word characters the two are identical, which
    is why the defect was invisible on every ordinary name. An all-digit
    alias is fenced differently; see _fence."""
    if not aliases:
        return None, {}
    aliases = sorted(aliases, key=lambda a: len(a[0]), reverse=True)
    parts, group_map = [], {}
    for k, (text, idx) in enumerate(aliases):
        name = f"a{k}"
        parts.append(rf"(?P<{name}>{_fence(text)})")
        group_map[name] = idx
    return re.compile("|".join(parts), flags), group_map


def _fence(text: str) -> str:
    """One alias as a pattern, fenced so it matches only as a whole.

    A word is fenced from word characters. An all-digit alias, such as a phone
    number or an ID, is fenced only from other digits: code writes letters right
    beside a number (``\\b<digits>\\b`` in a regex, ``tel_<digits>`` in a key), and
    a letter beside it does not make it a different number, while one more digit
    does. Measured 2026-10-08: a listed number inside ``r"\\b...\\b"`` passed the
    word fence and would have reached AGORA's outside reviewers."""
    body = re.escape(_nfc(text))
    if text.isascii() and text.isdigit():
        return rf"(?<!\d){body}(?!\d)"
    return rf"(?<!\w){body}(?!\w)"


def _dedupe(spans: list) -> list:
    """Sort-then-scan overlap resolution: longest span wins; on equal spans the
    earlier collector wins (denylist pins beat the generic floor)."""
    ordered = sorted(enumerate(spans), key=lambda p: (p[1].start,
                                                      -(p[1].end - p[1].start),
                                                      p[0]))
    out, last_end = [], -1
    for _, s in ordered:
        if s.start < last_end:
            continue
        out.append(s)
        last_end = s.end
    return out


def _rebuild(text: str, spans: list):
    """Single-pass reconstruction + per-category counts."""
    parts, counts, cursor = [], {}, 0
    for s in spans:
        parts.append(text[cursor:s.start])
        parts.append(s.replacement)
        cursor = s.end
        counts[s.category] = counts.get(s.category, 0) + 1
    parts.append(text[cursor:])
    return "".join(parts), counts


# ---------------------------------------------------------------------------
# L2 — LLM semantic recall (monotone: can only ADD findings; fail-open)
# ---------------------------------------------------------------------------

_L2_PROMPT = """You are a privacy reviewer. List every REAL-WORLD identity in the text below that could identify a company, person, project, host, or legal matter — EXCLUDING these public names: {allow}.
Text:
---
{text}
---
Respond with a strict JSON array of objects: [{{"name": "...", "class": "org|person|project|host|matter"}}]. Empty array if none."""


def llm_scan(text: str, denylist: Denylist, caller=None, strict: bool = False) -> list:
    """L2 semantic pass over ALREADY-L1-SHIELDED text: returns candidate
    identities the deterministic denylist missed, as [{name, class}] dicts.

    Detector only — it never rewrites (deterministic pipelines stay
    deterministic); findings are review input for the denylist. Monotone by
    construction (output can only add redaction candidates). ``caller`` is
    DIP-injected for tests; default is GRIP's canonical AI syscall. Fails open
    to [] on any error unless ``strict`` (then the caller has chosen
    fail-closed, the earlier in-house module's stance for cloud egress)."""
    prompt = _L2_PROMPT.format(
        allow=", ".join(denylist.allowlist) or "(none)", text=text[:6000]
    )
    try:
        if caller is None:
            # lazy — L1 never needs HAL. NOTE: `hal_llm_adapter` exposes no module-level
            # `chat`; it is a METHOD on HALLLMAdapter. Importing it raised ImportError,
            # was swallowed by the except below, and llm_scan silently returned [] --
            # so L2 was dead for every caller that did not inject `caller=` itself.
            # `infer` is what the `getattr(raw, "text", raw)` line below already expects:
            # it returns HALResult, which carries `.text`. audit=False because a
            # redaction scan is not a decision worth an IDR leaf per call.
            from functools import partial
            try:
                from hal_llm_adapter import HALLLMAdapter
            except ImportError:
                return []
            caller = partial(HALLLMAdapter().infer, audit=False)
        raw = caller(prompt)
        raw_text = getattr(raw, "text", raw)
        return _parse_l2(raw_text if isinstance(raw_text, str) else "")
    except Exception as exc:
        if strict:
            raise RedactionConfigError(f"L2 scan unavailable in strict mode: {exc}")
        return []


def _parse_l2(raw: str) -> list:
    """Lenient JSON-array extraction (code fences tolerated); drops non-dicts
    and empty names so a chatty model cannot poison the findings."""
    m = re.search(r"\[.*\]", raw.strip().strip("`"), re.DOTALL)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except ValueError:
        return []
    return [
        {"name": str(i["name"]).strip(), "class": str(i.get("class", "entity"))}
        for i in items
        if isinstance(i, dict) and str(i.get("name", "")).strip()
    ]


# ---------------------------------------------------------------------------
# Selftest — the Banach/idempotence proof (fails loudly)
# ---------------------------------------------------------------------------

_SELFTEST_TEXT = (
    "Met Alpha Example Corp about CASE-2026-001; mail alpha@example.com, "
    "call +41 00 000 00 00 from 10.0.0.9 (not 127.0.0.1)."
)


def selftest_idempotent(denylist: Denylist | None = None) -> bool:
    """Prove shield(shield(x)) == shield(x) for both profiles and that verify()
    of shielded output is empty. Raises AssertionError on drift — mirrors
    decision_chain_publish.selftest_idempotent."""
    dl = denylist if denylist is not None else _selftest_denylist()
    for profile in _PROFILES:
        shield = Shield(dl, profile=profile)
        once = shield.redact(_SELFTEST_TEXT)
        twice = shield.redact(once.text)
        assert twice.text == once.text, (
            f"redaction_shield NOT idempotent ({profile}): "
            f"{once.text!r} -> {twice.text!r}"
        )
        assert Shield(dl, profile=profile).verify(once.text) == [], (
            f"verify() found residue in shielded output ({profile})"
        )
    return True


def _selftest_denylist() -> Denylist:
    """Hermetic fixture — obvious fakes only, per visual-artifact-leak rule."""
    return Denylist(
        entries=(
            DenylistEntry("Alpha Example", (("alpha example", False),),
                          "[ORG-1]", "org"),
        ),
        allowlist=("GRIP", "HAL", "HAPPI"),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read_input(target: str) -> str:
    return sys.stdin.read() if target == "-" else Path(target).read_text()


def _cmd_selftest() -> int:
    try:
        selftest_idempotent()
    except AssertionError as exc:
        print(f"IDEMPOTENCE BROKEN: {exc}", file=sys.stderr)
        return 1
    print("shield fixed point OK — redact(redact(x)) == redact(x)")
    return 0


def _cmd_redact(shield: Shield, text: str) -> int:
    result = shield.redact(text)
    sys.stdout.write(result.text)
    print(f"\n[shield] {len(result.spans)} replacement(s): "
          f"{json.dumps(result.counts)}", file=sys.stderr)
    return 0


def _cmd_verify(shield: Shield, text: str, strict: bool = False) -> int:
    findings = shield.residue(text) if strict else shield.verify(text)
    for s in findings:
        print(f"{s.category}: {s.original!r} at {s.start}", file=sys.stderr)
    print(f"[shield] {len(findings)} finding(s)", file=sys.stderr)
    return 2 if findings else 0


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="GRIP canonical redaction shield")
    sub = ap.add_subparsers(dest="cmd", required=True)
    parsers = {}
    for name in ("redact", "verify"):
        p = parsers[name] = sub.add_parser(name)
        p.add_argument("target", nargs="?", default="-",
                       help="file path or - for stdin")
        p.add_argument("--profile", default=PROFILE_PUBLIC, choices=_PROFILES)
        p.add_argument("--config", type=Path, action="append", default=None,
                       help="denylist JSON (repeatable; lists merge in the order "
                            "given and the FIRST list owns a spelling -- matter list "
                            "first). Named lists load alone. Default: the repo list "
                            "with the private overlay merged first.")
    parsers["verify"].add_argument(
        "--strict", action="store_true",
        help="exit 2 on ANY residue: what redact() would replace plus, under "
             "public, the name-shaped spans blockers() would hold a publish for",
    )
    sub.add_parser("selftest", help="prove idempotence (exit 1 on drift)")
    ns = ap.parse_args(argv)

    if ns.cmd == "selftest":
        return _cmd_selftest()
    # No --config is the default list WITH the private overlay, degrading to the
    # generic default rather than crashing when neither is configured at all
    # (a standalone install); named configs load alone and still raise if missing
    # -- the caller named that file explicitly, so a missing one is their error.
    shield = Shield(load_denylists(ns.config) if ns.config else load_denylist_or_default(),
                    profile=ns.profile)
    text = _read_input(ns.target)
    if ns.cmd == "redact":
        return _cmd_redact(shield, text)
    return _cmd_verify(shield, text, strict=ns.strict)


if __name__ == "__main__":
    sys.exit(main())
