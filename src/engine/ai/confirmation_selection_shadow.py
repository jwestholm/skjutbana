"""Frozen confirmation-based selector used for physical shadow diagnostics.

This module is deliberately diagnostic-only.  It never writes to a scanner or
changes the emitted hit.  The formula is frozen from the accepted development
replay and its hash is stored in every physical trace.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

SELECTOR = "CONFIRMATION_SELECTION_SHADOW"
STATUS = "PHYSICAL_REPLAY_CHALLENGER"
FORMULA = "confirmation_center_abs + confirmation_darkening - confirmation_compact"
WEIGHTS = {
    "v2225_confirm_center_abs": 1.0,
    "v2225_confirm_darkening": 1.0,
    "v2225_confirm_compact": -1.0,
}
CONFIG = {
    "selector": SELECTOR,
    "status": STATUS,
    "formula": FORMULA,
    "weights": WEIGHTS,
    "version": "frozen-development-replay-1",
}


def _canonical_config() -> str:
    return json.dumps(CONFIG, sort_keys=True, separators=(",", ":"), allow_nan=False)


CONFIG_HASH = hashlib.sha256(_canonical_config().encode("utf-8")).hexdigest()


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def score(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Return frozen score components without mutating *candidate*."""
    components = {
        key: (_finite(candidate.get(key)) or 0.0) * weight
        for key, weight in WEIGHTS.items()
    }
    available = any(_finite(candidate.get(key)) is not None for key in WEIGHTS)
    return {"components": components, "score": float(sum(components.values())), "available": available}


def select(candidates: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
    """Select a candidate for diagnostics, using stable deterministic ties."""
    values = [candidate for candidate in (candidates or []) if isinstance(candidate, Mapping)]
    scored = [(index, candidate, score(candidate)) for index, candidate in enumerate(values)]
    usable = [item for item in scored if item[2]["available"]]
    if not usable:
        return {
            "selector": SELECTOR, "status": "UNAVAILABLE", "config_hash": CONFIG_HASH,
            "formula": FORMULA, "candidate_count": len(values), "selected": None,
        }
    index, candidate, details = max(
        usable,
        key=lambda item: (item[2]["score"], _finite(item[1].get("score")) or 0.0, -item[0]),
    )
    return {
        "selector": SELECTOR, "status": STATUS, "config_hash": CONFIG_HASH,
        "formula": FORMULA, "candidate_count": len(values), "selected_index": index,
        "selected": dict(candidate), "score": details["score"],
        "score_components": details["components"], "features_available": details["available"],
    }


def confirmation_candidates(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Collect the latest local-confirmation candidates, preserving provenance."""
    result: list[dict[str, Any]] = []
    for stage in trace.get("stages", []) if isinstance(trace, Mapping) else []:
        confirmation = stage.get("local_confirmation") if isinstance(stage, Mapping) else None
        values = confirmation.get("candidates") if isinstance(confirmation, Mapping) else None
        if isinstance(values, list):
            result.extend(dict(value) for value in values if isinstance(value, Mapping))
    return result


def from_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    values = confirmation_candidates(trace)
    result = select(values)
    result["pool_semantics"] = "local_confirmation_candidates"
    return result
