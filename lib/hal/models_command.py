"""HAL `/models` slash command — list and pick from every available model.

Reads ``config/model_catalog.yaml`` (per-model metadata: tier, modality,
context window, strengths, description) and merges it with the live
provider registry + detected subscriptions to produce a grouped,
filtered, copy-paste-friendly listing.

Usage:
    from lib.hal.models_command import handle_models_command

    print(handle_models_command(registry, ""))            # everything
    print(handle_models_command(registry, "frontier"))    # frontier only
    print(handle_models_command(registry, "free"))        # free + local
    print(handle_models_command(registry, "vision"))      # vision-capable
    print(handle_models_command(registry, "ollama"))      # one provider
    print(handle_models_command(registry, "anthropic/claude-opus-4-7"))  # one model

Design:
- Catalogue parsing is intentionally a small dedicated parser (no PyYAML
  dependency; matches the convention of :mod:`lib.hal.registry`).
- Status annotation is pure: takes registry + subscriptions, returns
  enriched entries. Easy to test without touching real providers.
- Rendering is pure ANSI text — no print side effects in the helpers.
"""
import os
import re
from dataclasses import dataclass
from typing import Optional

from lib.hal.home import HAL_HOME
from lib.hal.subscription import auto_detect_subscriptions

_CATALOGUE_PATH = os.path.join(HAL_HOME, "config", "model_catalog.yaml")

# ANSI colour codes (mirrors cli.py palette so output blends in).
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GRAY = "\033[38;5;240m"
_RESET = "\033[0m"

# Tier display order — subscription first (zero per-request cost), then
# frontier down through local. Anything not in this list renders last.
_TIER_ORDER = ("subscription", "frontier", "premium", "mid", "cheap", "local", "free")

_TIER_LABEL = {
    "subscription": "Subscription (flat-fee, zero per-request cost)",
    "frontier": "Frontier (cutting-edge, premium pricing)",
    "premium": "Premium (strong reasoning, mid-high cost)",
    "mid": "Mid (balanced quality/cost workhorses)",
    "cheap": "Cheap (high-volume, low cost per token)",
    "local": "Local (runs on your hardware, no network egress)",
    "free": "Free",
}

# Human-readable context-window formatter.
def _fmt_ctx(tokens: int) -> str:
    if tokens >= 1_000_000:
        return f"{tokens // 1_000_000}M ctx"
    if tokens >= 1000:
        return f"{tokens // 1000}K ctx"
    return f"{tokens} ctx"


# ────────────────────────────────────────────────────────────────────
# Data model
# ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelEntry:
    """One entry parsed from model_catalog.yaml."""
    address: str
    tier: str
    paid: bool
    modality: str
    context_window: int
    tags: tuple[str, ...]
    strengths: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class AnnotatedEntry:
    """A catalogue entry plus runtime status."""
    entry: ModelEntry
    status: str        # "ready" | "offline" | "active" | "setup_needed" | "not_running" | "needs_key"
    note: str          # operator-facing hint when status != ready/active


# ────────────────────────────────────────────────────────────────────
# Catalogue parser — small dedicated subset, no PyYAML dependency
# ────────────────────────────────────────────────────────────────────

_INLINE_LIST_RE = re.compile(r"^\[(.*)\]$")
_BLOCK_TRUE = {"true", "True", "yes"}
_BLOCK_FALSE = {"false", "False", "no"}


def _coerce_scalar(value: str):
    """Convert a YAML scalar string to its Python type."""
    value = value.strip()
    if not value:
        return ""
    if value in _BLOCK_TRUE:
        return True
    if value in _BLOCK_FALSE:
        return False
    inline = _INLINE_LIST_RE.match(value)
    if inline:
        items = [s.strip().strip("'\"") for s in inline.group(1).split(",")]
        return [s for s in items if s]
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _parse_catalogue_yaml(path: str) -> list[dict]:
    """Parse model_catalog.yaml into a list of dicts.

    Handles the catalogue's specific subset:
    - top-level ``models:`` followed by a list of entries
    - each entry is a mapping of ``key: value`` lines
    - inline arrays ``tags: [a, b, c]``
    - nested lists ``strengths:`` followed by ``- "..."`` lines
    - folded block scalars ``description: >`` followed by indented text

    Anything outside this subset is ignored (forward-compatible).
    """
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        lines = f.readlines()

    entries: list[dict] = []
    current: Optional[dict] = None
    pending_list_key: Optional[str] = None
    pending_block_key: Optional[str] = None
    pending_block_indent: int = 0
    pending_block_lines: list[str] = []
    in_models = False

    def _flush_block(target: dict) -> None:
        nonlocal pending_block_key, pending_block_lines, pending_block_indent
        if pending_block_key is not None:
            target[pending_block_key] = " ".join(s.strip() for s in pending_block_lines).strip()
            pending_block_key = None
            pending_block_lines = []
            pending_block_indent = 0

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if not stripped or stripped.startswith("#"):
            continue

        if not in_models:
            if stripped.startswith("models:"):
                in_models = True
            continue

        # Block-scalar continuation (description: > followed by indented text)
        if pending_block_key is not None:
            if indent > pending_block_indent and not stripped.startswith("- "):
                pending_block_lines.append(stripped)
                continue
            _flush_block(current)  # type: ignore[arg-type]

        # New entry: "  - key: value"
        if stripped.startswith("- ") and indent == 2:
            if current is not None:
                entries.append(current)
            current = {}
            pending_list_key = None
            rest = stripped[2:].strip()
            if ":" in rest:
                k, _, v = rest.partition(":")
                key, val = k.strip(), v.strip()
                if val == ">":
                    pending_block_key = key
                    pending_block_indent = indent
                    pending_block_lines = []
                    current[key] = ""
                else:
                    current[key] = _coerce_scalar(val)
            continue

        if current is None:
            continue

        # Nested list bullet within an entry: "      - item"
        if stripped.startswith("- "):
            value = _coerce_scalar(stripped[2:].strip())
            if pending_list_key is not None:
                current.setdefault(pending_list_key, []).append(value)
            continue

        # Plain "key: value" within entry
        if ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()
            if val == "":
                # opens a nested list (e.g. strengths:)
                pending_list_key = key
                current[key] = []
            elif val == ">":
                pending_block_key = key
                pending_block_indent = indent
                pending_block_lines = []
                pending_list_key = None
            else:
                current[key] = _coerce_scalar(val)
                pending_list_key = None

    if pending_block_key is not None and current is not None:
        _flush_block(current)
    if current is not None:
        entries.append(current)
    return entries


def load_catalogue(path: Optional[str] = None) -> list[ModelEntry]:
    """Parse the model catalogue YAML into ModelEntry instances.

    Missing fields default sensibly so a sparsely-populated entry still
    renders. Unknown fields in the YAML are ignored.
    """
    raw_entries = _parse_catalogue_yaml(path or _CATALOGUE_PATH)
    return [_dict_to_entry(d) for d in raw_entries if d.get("address")]


def _dict_to_entry(d: dict) -> ModelEntry:
    """Build a ModelEntry from a parsed dict, applying defaults."""
    return ModelEntry(
        address=str(d.get("address", "")),
        tier=str(d.get("tier", "mid")),
        paid=bool(d.get("paid", True)),
        modality=str(d.get("modality", "text")),
        context_window=int(d.get("context_window", 0) or 0),
        tags=tuple(str(t) for t in (d.get("tags") or [])),
        strengths=tuple(str(s) for s in (d.get("strengths") or [])),
        description=str(d.get("description", "")).strip(),
    )


# ────────────────────────────────────────────────────────────────────
# Annotation — merge catalogue with runtime state
# ────────────────────────────────────────────────────────────────────

def _provider_of(address: str) -> str:
    """Extract the provider prefix from a `provider/model` address."""
    return address.split("/", 1)[0] if "/" in address else address


def annotate_availability(
    entries: list[ModelEntry],
    registry,
    subscriptions: dict,
) -> list[AnnotatedEntry]:
    """Tag each entry with its current runtime status.

    Status values:
        ready          provider registered, healthy, model usable
        offline        provider registered but health check failed
        active         subscription transport detected for this provider
        setup_needed   subscription model but no subscription configured
        not_running    local provider whose server is not discovered
        needs_key      cloud provider that has no API key configured

    The note carries an actionable hint — the renderer surfaces it for
    every non-ready / non-active entry.
    """
    providers_by_name = {p.name: p for p in registry.list_providers()}
    annotated: list[AnnotatedEntry] = []
    for entry in entries:
        provider_name = _provider_of(entry.address)

        if entry.tier == "subscription":
            sub = subscriptions.get(provider_name)
            if sub is not None:
                annotated.append(AnnotatedEntry(entry, "active", ""))
            else:
                annotated.append(AnnotatedEntry(
                    entry, "setup_needed",
                    "Run `hal --subscribe` to activate the subscription transport.",
                ))
            continue

        provider = providers_by_name.get(provider_name)
        if provider is None:
            if entry.tier == "local":
                annotated.append(AnnotatedEntry(
                    entry, "not_running",
                    f"Start the local server (e.g. `{provider_name} serve`) and re-run.",
                ))
            else:
                annotated.append(AnnotatedEntry(
                    entry, "needs_key",
                    f"Add an API key under keychain `grip-{provider_name}` "
                    f"or set the corresponding env var.",
                ))
            continue

        if not provider.healthy:
            annotated.append(AnnotatedEntry(
                entry, "offline", f"Provider `{provider_name}` failed its last health check.",
            ))
            continue

        annotated.append(AnnotatedEntry(entry, "ready", ""))
    return annotated


# ────────────────────────────────────────────────────────────────────
# Filtering
# ────────────────────────────────────────────────────────────────────

# Aliases so common keywords map to the right field.
_TIER_ALIASES = {
    "subscription": "subscription", "sub": "subscription",
    "frontier": "frontier", "top": "frontier", "best": "frontier",
    "premium": "premium",
    "mid": "mid", "balanced": "mid",
    "cheap": "cheap", "low-cost": "cheap", "fast": None,  # fast is a tag, handled below
    "local": "local", "offline": "local",
    "free": "free",
}


def filter_entries(annotated: list[AnnotatedEntry], term: str) -> list[AnnotatedEntry]:
    """Filter annotated entries by tier, tag, provider prefix, or address.

    Empty term returns the input unchanged. Unknown terms try a substring
    match against (address, tier, modality, tags) so the user can pass
    any reasonable keyword.
    """
    term = term.strip().lower()
    if not term:
        return annotated

    # Exact address match — single-entry detail view.
    exact = [a for a in annotated if a.entry.address.lower() == term]
    if exact:
        return exact

    # Provider-prefix match: "ollama" → ollama/*, "anthropic" → anthropic/*
    if term.isalnum() or term in ("mlx_vlm", "openrouter"):
        prefix_matches = [
            a for a in annotated
            if _provider_of(a.entry.address).lower() == term
        ]
        if prefix_matches:
            return prefix_matches

    # Tier match.
    tier_target = _TIER_ALIASES.get(term)
    if tier_target:
        return [a for a in annotated if a.entry.tier == tier_target]

    # Tag or modality substring match.
    out = []
    for a in annotated:
        haystack = " ".join((
            a.entry.address.lower(),
            a.entry.tier.lower(),
            a.entry.modality.lower(),
            " ".join(t.lower() for t in a.entry.tags),
        ))
        if term in haystack:
            out.append(a)
    return out


# ────────────────────────────────────────────────────────────────────
# Rendering
# ────────────────────────────────────────────────────────────────────

_STATUS_BADGE = {
    "ready":        ("ready",   _GREEN),
    "active":       ("active",  _GREEN),
    "offline":      ("offline", _RED),
    "setup_needed": ("setup",   _YELLOW),
    "not_running":  ("offline", _YELLOW),
    "needs_key":    ("no key",  _YELLOW),
}


def _render_badge(status: str) -> str:
    label, colour = _STATUS_BADGE.get(status, (status, _DIM))
    return f"{colour}{label:<8}{_RESET}"


def _first_sentence(text: str) -> str:
    """Return the first sentence of a paragraph (best-effort)."""
    if not text:
        return ""
    for end in (". ", "? ", "! "):
        idx = text.find(end)
        if 0 < idx < 200:
            return text[: idx + 1].strip()
    return (text[:200] + ("…" if len(text) > 200 else "")).strip()


def _render_entry_line(a: AnnotatedEntry) -> str:
    """Render one entry as two lines: header + summary."""
    e = a.entry
    badge = _render_badge(a.status)
    paid_tag = "paid" if e.paid else "free"
    tags_str = ", ".join(e.tags) if e.tags else "-"
    meta = f"{e.modality} · {_fmt_ctx(e.context_window)} · {paid_tag}"
    line1 = f"  {_BOLD}{e.address}{_RESET}  {badge}  {_DIM}{meta}{_RESET}"
    summary = _first_sentence(e.description) or (e.strengths[0] if e.strengths else "")
    line2 = f"      {_DIM}{summary}{_RESET}"
    if a.note:
        line2 += f"\n      {_YELLOW}↳ {a.note}{_RESET}"
    if e.tags:
        line2 += f"\n      {_GRAY}tags: {tags_str}{_RESET}"
    return f"{line1}\n{line2}"


def _render_detail(a: AnnotatedEntry) -> str:
    """Render the full detail view for a single entry (exact-address match)."""
    e = a.entry
    lines = [
        f"\n  {_BOLD}{e.address}{_RESET}  {_render_badge(a.status)}",
        f"  {_DIM}{e.tier} · {e.modality} · {_fmt_ctx(e.context_window)}"
        f" · {'paid' if e.paid else 'free'}{_RESET}",
        "",
    ]
    if e.description:
        lines.append(f"  {e.description}")
        lines.append("")
    if e.strengths:
        lines.append(f"  {_BOLD}Strengths{_RESET}")
        for s in e.strengths:
            lines.append(f"    · {s}")
        lines.append("")
    if e.tags:
        lines.append(f"  {_GRAY}tags: {', '.join(e.tags)}{_RESET}")
    if a.note:
        lines.append(f"  {_YELLOW}↳ {a.note}{_RESET}")
    lines.append("")
    lines.append(f"  {_CYAN}Switch to this model:{_RESET}  {_BOLD}/model {e.address}{_RESET}")
    return "\n".join(lines)


def render_models(
    annotated: list[AnnotatedEntry],
    *,
    detail: bool = False,
    filter_term: str = "",
) -> str:
    """Render annotated entries grouped by tier, with help footer.

    When ``detail=True`` and exactly one entry matches, renders a full
    single-model view. Otherwise renders grouped tier sections.
    """
    if not annotated:
        return f"  {_YELLOW}No models match `{filter_term}`. Try /models for everything.{_RESET}"

    if detail and len(annotated) == 1:
        return _render_detail(annotated[0])

    # Sort-then-scan: O(n log n) sort, then a single linear pass with a
    # tier-change marker. This is the gate-recommended replacement for
    # partition-then-iterate, and it's also clearer to read.
    tier_rank = {t: i for i, t in enumerate(_TIER_ORDER)}
    sorted_entries = sorted(
        annotated,
        key=lambda a: (tier_rank.get(a.entry.tier, len(_TIER_ORDER)), a.entry.address),
    )

    sections = []
    title_suffix = f" matching `{filter_term}`" if filter_term else ""
    sections.append(
        f"\n  {_BOLD}HAL Models{_RESET}"
        f"{_DIM}{title_suffix} — {len(annotated)} entries{_RESET}\n",
    )

    last_tier = None
    for a in sorted_entries:
        if a.entry.tier != last_tier:
            label = _TIER_LABEL.get(a.entry.tier, a.entry.tier.title())
            sections.append(f"  {_BOLD}{label}{_RESET}")
            last_tier = a.entry.tier
        sections.append(_render_entry_line(a))
        sections.append("")

    sections.append(f"  {_DIM}Filter: /models <tier|tag|provider>"
                    f"  · Detail: /models <provider/model>{_RESET}")
    sections.append(f"  {_DIM}Switch: /model <provider/model>"
                    f"  · Subscription: hal --subscribe{_RESET}")
    return "\n".join(sections)


# ────────────────────────────────────────────────────────────────────
# Public entry point
# ────────────────────────────────────────────────────────────────────

def handle_models_command(registry, arg: str = "", *,
                          catalogue_path: Optional[str] = None,
                          subscriptions: Optional[dict] = None) -> str:
    """Top-level handler — read catalogue, annotate, filter, render.

    ``subscriptions`` and ``catalogue_path`` are injected for tests.
    Production callers pass neither; the defaults read from disk + env.
    """
    catalogue = load_catalogue(catalogue_path)
    if not catalogue:
        return (f"  {_YELLOW}Catalogue not found at "
                f"{catalogue_path or _CATALOGUE_PATH}.{_RESET}")

    subs = subscriptions if subscriptions is not None else auto_detect_subscriptions()
    annotated = annotate_availability(catalogue, registry, subs)

    term = arg.strip()
    filtered = filter_entries(annotated, term) if term else annotated
    detail_mode = bool(term) and "/" in term and len(filtered) == 1
    return render_models(filtered, detail=detail_mode, filter_term=term)
