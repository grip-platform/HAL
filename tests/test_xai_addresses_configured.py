"""Every xAI model address referenced in `lib/` or `config/` must be configured.

A provider config that lists a model the vendor no longer serves fails only at
request time, with a 404, on whatever path happens to reach it — and nothing
notices in between.

On 2026-09-15 a live ``GET api.x.ai/v1/models`` returned 12 ids and showed the
configured xai list was five-sixths stale. The retired ``xai/grok-3`` was still
wired into the fallback chain, the premium tier and a council seat, so three
runtime paths were dead while every test stayed green, because nothing ever
compared a *reference* to the provider's own *list*.

This closes that gap statically and without network access: an address that is
not in the provider's configured ``models`` list is a bug by construction.

Detection is deliberately narrow so prose cannot create a false positive:

* **config** is parsed, then walked as data — a string starting ``xai/`` is a
  reference. A slash-separated prose list such as ``gemini/groq/xai/deepseek``
  is data too, but sitting inside a comment it is never parsed.
* **code** is matched only on *quoted* literals — ``"xai/<id>"`` or
  ``("xai", "<id>")``. The same prose inside a Python comment or docstring has
  no closing quote after the id and is ignored.

Scope: `lib/` and `config/` only. Tests use retired ids as neutral fixture
strings (mocked, never dispatched) and are not scanned. Generalising this to
every cloud provider would surface a wider set of pre-existing references; that
is a separate sweep, not this regression anchor.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

#: A QUOTED address literal: `"xai/grok-4.6"`. The trailing quote is what keeps
#: a docstring's `gemini/groq/xai/deepseek` from reading as an address.
_QUOTED = re.compile(r"""["']xai/([A-Za-z0-9][A-Za-z0-9._-]*)["']""")
#: The explicit pair form used by the fan-out rungs: `("xai", "grok-4.6")`.
_PAIR = re.compile(r"""\(\s*["']xai["']\s*,\s*["']([A-Za-z0-9][A-Za-z0-9._-]*)["']""")


def configured_xai_models() -> set[str]:
    cfg = yaml.safe_load((ROOT / "config" / "providers.yaml").read_text())
    for provider in cfg["providers"]:
        if provider["name"] == "xai":
            return set(provider["models"])
    raise AssertionError("no `xai` provider in config/providers.yaml")


def _strings(obj):
    """Every string anywhere in a parsed YAML document."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _strings(value)


def referenced_xai_models() -> dict[str, set[str]]:
    """Every xAI address referenced under lib/ or config/, keyed by file."""
    found: dict[str, set[str]] = {}

    for path in sorted((ROOT / "lib").rglob("*.py")):
        text = path.read_text(errors="replace")
        ids = set(_QUOTED.findall(text)) | set(_PAIR.findall(text))
        if ids:
            found[str(path.relative_to(ROOT))] = ids

    for path in sorted((ROOT / "config").glob("*.y*ml")):
        data = yaml.safe_load(path.read_text(errors="replace"))
        ids = {s.split("/", 1)[1] for s in _strings(data) if s.startswith("xai/")}
        if ids:
            found[str(path.relative_to(ROOT))] = ids

    return found


def test_every_referenced_xai_model_is_configured():
    """The defect this file exists for: a reference to an unconfigured model."""
    configured = configured_xai_models()
    stray = {path: sorted(ids - configured)
             for path, ids in referenced_xai_models().items()
             if ids - configured}
    assert not stray, f"xAI addresses referenced but not configured: {stray}"


def test_the_retired_grok3_address_is_gone():
    """Regression anchor: the live probe on 2026-09-15 shows grok-3 unserved."""
    assert "grok-3" not in configured_xai_models()
    stragglers = [path for path, ids in referenced_xai_models().items()
                  if "grok-3" in ids]
    assert not stragglers, f"still references retired xai/grok-3: {stragglers}"
