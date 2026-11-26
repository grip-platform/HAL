"""Redact what a review sends to model providers outside HAL.

HAL AGORA (lib/agora_pr_review.py) sends a pull request's diff, file context,
and its own working text to outside providers. Every such text
passes through the redaction shield's public profile first: the denylist's
names and matter terms, then the shield's floor (keys, message IDs, emails,
identity numbers, phone numbers and IP addresses). The public profile is used,
not llm_egress, because llm_egress treats any run of capitalised words as a
person, which would mangle code under review.

Where the denylist comes from, in order:
  1. REVIEW_REDACTION_DENYLIST_B64, base64 of the denylist JSON. CI reads the
     git-crypt-encrypted file as ciphertext, so a runner gets the list this way,
     from a repository secret.
  2. config/redaction-denylist.json in this checkout, when it is decrypted.
With none of these, only the floor runs, and the footer says so, so a verdict
never implies protection that did not happen.

Fails closed: if redaction itself raises, the whole text is replaced by
WITHHELD and nothing of it is sent.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

_LIB = Path(__file__).resolve().parent
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))
# Both modules carry a hal_ prefix on purpose. The wire imports GRIP's review_redaction by
# bare name and refuses when another file answers to it (lib/hal/cascade_policy.py,
# load_shield), and the line above puts this directory first on sys.path, so a HAL module
# named review_redaction or redaction_shield would make the wire refuse every request.
import hal_redaction_shield as rs  # noqa: E402

DENYLIST_ENV = "REVIEW_REDACTION_DENYLIST_B64"
WITHHELD = "[text withheld: the redaction shield failed, so none of it was sent]"
SOURCE_SECRET, SOURCE_CHECKOUT, SOURCE_NONE = "secret", "checkout", "none"
SOURCE_OVERLAY = "overlay"   # no shared list, but the operator's private one supplied names

_state: dict = {}


def reset() -> None:
    """Forget the cached shield and the tally (tests, or a new review run)."""
    _state.clear()
    _state.update(shield=None, source="", counts=Counter(), seen=set(), sends=0)


reset()


def _with_names(denylist: rs.Denylist | None) -> rs.Denylist | None:
    """``denylist`` if it lists anyone, else None.

    A list with no entries redacts no name, so it counts as no list at all.
    Accepting it would let the footer say names were redacted with it.
    """
    return denylist if denylist is not None and denylist.entries else None


def _from_env(env: dict) -> rs.Denylist | None:
    """The denylist carried base64-encoded in the environment, or None."""
    raw = (env.get(DENYLIST_ENV) or "").strip()
    if not raw:
        return None
    try:
        data = base64.b64decode(raw)
    except (binascii.Error, ValueError):
        return None
    with tempfile.NamedTemporaryFile(suffix=".json") as handle:
        handle.write(data)
        handle.flush()
        try:
            return _with_names(rs.load_denylist(Path(handle.name)))
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            return None  # a malformed secret falls through to the next source


def _from_files() -> rs.Denylist | None:
    """This checkout's decrypted denylist file, if it is readable and lists anyone."""
    path = _LIB.parent / "config" / "redaction-denylist.json"
    return _with_names(rs.load_denylist(path)) if rs.denylist_state(path) == "ok" else None


def _shield() -> rs.Shield:
    if _state["shield"] is None:
        denylist, source = _from_env(dict(os.environ)), SOURCE_SECRET
        if denylist is None:
            denylist, source = _from_files(), SOURCE_CHECKOUT
        if denylist is None:
            denylist, source = rs.Denylist.empty(), SOURCE_NONE
        # The operator's private overlay goes first wherever it exists (never on
        # CI). One that is configured but unreadable raises here, and redact()
        # then withholds the whole text rather than send it without its entries.
        # When it is the only list with names, the footer says so (Codex, P2,
        # on GRIP#7579) instead of claiming that no names were redacted.
        overlay = rs.load_overlay()
        if source == SOURCE_NONE and overlay.entries:
            source = SOURCE_OVERLAY
        denylist = rs.Denylist.merge(overlay, denylist)
        _state["shield"] = rs.Shield(denylist, profile=rs.PROFILE_PUBLIC)
        _state["source"] = source
    return _state["shield"]


def redact(text: str) -> str:
    """``text`` as the shield's public profile rewrites it; WITHHELD if that fails.

    Pure: no tally. The quote gate uses this to compare a speaker's quote, which
    was written against redacted text, with the real file line.
    """
    try:
        return _shield().redact(text or "").text
    except Exception:  # noqa: BLE001 -- a failed redaction must never send the original
        return WITHHELD


def redact_outbound(text: str) -> str:
    """``redact(text)``, counted for the verdict footer. Call it on text that leaves.

    The same text sent twice (one prompt to several judge rungs) counts once.
    """
    digest = hashlib.sha256((text or "").encode("utf-8", "surrogatepass")).hexdigest()
    try:
        result = _shield().redact(text or "")
    except Exception:  # noqa: BLE001 -- a failed redaction must never send the original
        _state["counts"]["withheld"] += 1
        return WITHHELD
    if digest not in _state["seen"]:
        _state["seen"].add(digest)
        _state["sends"] += 1
        _state["counts"].update(result.counts)
    return result.text


def summary() -> dict:
    """What the shield did this run: where its names came from, and what it replaced."""
    return {"names_list": _state["source"], "sends": _state["sends"],
            "replaced": dict(_state["counts"])}


def render_footer() -> str:
    """One italic line for the review comment; "" when nothing was sent."""
    if not _state["sends"] and not _state["counts"]:
        return ""
    counts = dict(_state["counts"])
    total = sum(n for k, n in counts.items() if k != "withheld")
    detail = ", ".join(f"{n} {k}" for k, n in sorted(counts.items()) if k != "withheld")
    replaced = f"{total} item{'s' if total != 1 else ''} replaced" + (f" ({detail})" if detail else "")
    where = {SOURCE_SECRET: "the repository secret", SOURCE_CHECKOUT: "this checkout",
             SOURCE_OVERLAY: "the private overlay"}
    if _state["source"] in where:
        names = f"names were redacted using the list from {where[_state['source']]}"
    else:
        names = ("the names list was not available on this runner, so only identifiers "
                 "(keys, message IDs, emails, ID numbers, phone numbers, IPs) were redacted")
    withheld = counts.get("withheld", 0)
    tail = f"; {withheld} text(s) withheld because redaction failed" if withheld else ""
    return f"\n\n*Outbound text passed the redaction shield: {replaced}; {names}{tail}.*\n"
