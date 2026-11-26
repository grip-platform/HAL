"""Natural-language and fuzzy model switching for the HAL REPL."""
import difflib
import re
from typing import Optional

_SWITCH_RE = re.compile(
    r'^(?:switch|change|use|try|load|run)\s+(?:to\s+|with\s+)?'
    r'(.+?)(?:\s+model|\s+llm|\s+ai)?$',
    re.I,
)


def detect_model_switch(user_input: str, registry) -> Optional[str]:
    """Return a provider/model address if input is a model-switch request, else None."""
    text = user_input.strip()
    if text.startswith('/') or len(text) > 80:
        return None
    m = _SWITCH_RE.match(text)
    if not m:
        return None
    fragment = m.group(1).strip().lower()
    return _fuzzy_resolve(fragment, registry)


def fuzzy_resolve_model(fragment: str, registry) -> Optional[str]:
    """Public entry point for /model fuzzy matching."""
    return _fuzzy_resolve(fragment.lower().strip(), registry)


def _build_model_list(registry) -> list:
    """Return all provider/model addresses from the registry."""
    addresses = []
    for p in registry.list_providers():
        models = p.models
        if not models:
            addresses.append(p.name)
        else:
            for m in models:
                addresses.append(f"{p.name}/{m}")
    return addresses


def _fuzzy_resolve(fragment: str, registry) -> Optional[str]:
    all_models = _build_model_list(registry)
    lower_models = [a.lower() for a in all_models]

    # 1. Substring match
    for i, addr in enumerate(lower_models):
        if fragment in addr:
            return all_models[i]

    # 2. Provider-name-only match → first model of that provider
    for p in registry.list_providers():
        name = p.name.lower()
        if name == fragment or name.startswith(fragment):
            if p.models:
                return f"{p.name}/{p.models[0]}"
            return p.name

    # 3. Difflib fuzzy for typos
    matches = difflib.get_close_matches(fragment, lower_models, n=1, cutoff=0.55)
    if matches:
        return all_models[lower_models.index(matches[0])]

    return None
