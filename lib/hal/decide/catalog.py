"""The decision models HAL knows, and the providers that serve them.

Read from ``config/decision_models.yaml``; ``$HAL_DECISION_MODELS`` overrides the
path. Decision models are kept out of ``config/model_catalog.yaml`` on purpose:
that catalogue's loader keeps every entry with an address, so a decision model
listed there could be routed as a chat model (``lib/hal/models_command.py``).
"""
from __future__ import annotations

import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional, Tuple

from lib.hal.decide.contract import DecisionModel, ProviderSpec

CONFIG_ENV = "HAL_DECISION_MODELS"
DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "config" / "decision_models.yaml"


class CatalogError(ValueError):
    """config/decision_models.yaml is missing, unreadable or malformed."""


def config_path() -> Path:
    """``$HAL_DECISION_MODELS``, else the repository's ``config/decision_models.yaml``."""
    override = os.environ.get(CONFIG_ENV, "").strip()
    return Path(override).expanduser() if override else DEFAULT_CONFIG


def load(path: Optional[Path] = None) -> Tuple[dict, dict]:
    """``(providers, models)``, each keyed by name. Raises CatalogError."""
    target = path if path is not None else config_path()
    try:
        import yaml  # PyYAML: a HAL runtime dependency (lib/hal/cli.py)
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 -- any read or parse failure is the same refusal
        raise CatalogError(f"cannot read {target} ({type(exc).__name__})") from exc
    if not isinstance(raw, dict):
        raise CatalogError(f"{target} is not a mapping")
    providers = {name: _provider(name, spec, target)
                 for name, spec in (raw.get("providers") or {}).items()}
    models = {mid: _model(mid, spec, providers, target)
              for mid, spec in (raw.get("models") or {}).items()}
    return providers, models


def resolve(model_id: str, path: Optional[Path] = None) -> Tuple[DecisionModel, ProviderSpec]:
    """``model_id`` and the provider serving it. KeyError when unknown; CatalogError when unusable."""
    providers, models = load(path)
    model = models[model_id]
    return model, providers[model.provider]


def _provider(name: str, spec: dict, target: Path) -> ProviderSpec:
    try:
        endpoint = str(spec["endpoint"])
        if not endpoint.startswith("https://"):
            raise ValueError("a decision endpoint must be https")
        return ProviderSpec(
            name=str(name), endpoint=endpoint, key_source=str(spec["key_source"]),
            spend_source=str(spec["spend_source"]), billing=str(spec["billing"]),
            usd_per_million_input=_usd(spec["usd_per_million_input_tokens"]),
            usd_per_million_output=_usd(spec["usd_per_million_output_tokens"]),
            max_request_bytes=int(spec["max_request_bytes"]))
    except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
        raise CatalogError(f"{target}: provider {name!r} is malformed ({exc})") from exc


def _model(mid: str, spec: dict, providers: dict, target: Path) -> DecisionModel:
    if not isinstance(spec, dict) or spec.get("provider") not in providers:
        raise CatalogError(f"{target}: model {mid!r} names no provider declared there")
    return DecisionModel(id=str(mid), provider=spec["provider"], pinned=str(spec.get("pinned") or ""))


def _usd(value) -> Decimal:
    amount = Decimal(str(value))
    if not amount.is_finite() or amount < 0:
        raise ValueError(f"{value!r} is not a dollar amount")
    return amount
