from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .model import RankModelV223

ROOT = Path("content/ai/training_v223")
MODELS_ROOT = ROOT / "models"
REGISTRY_PATH = MODELS_ROOT / "registry.json"
CHAMPION_PATH = MODELS_ROOT / "champion.json"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def load_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {"schema_version": "2.23.0", "models": [], "updated_at": None}
    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"schema_version": "2.23.0", "models": []}
    except Exception:
        return {"schema_version": "2.23.0", "models": []}


def register_model(
    model: RankModelV223,
    *,
    metrics: dict[str, Any],
    trigger: str,
    provisional: bool,
    trial_id: str,
) -> dict[str, Any]:
    directory = MODELS_ROOT / "challengers" / trial_id
    model.metadata.update({
        "trial_id": trial_id,
        "trigger": trigger,
        "provisional_split": bool(provisional),
        "eligible_for_live_authority": False,
    })
    model.save(directory)
    entry = {
        "trial_id": trial_id,
        "created_at": time.time(),
        "kind": model.kind,
        "path": str(directory),
        "metrics": metrics,
        "trigger": trigger,
        "provisional": bool(provisional),
        "eligible_for_live_authority": False,
    }
    registry = load_registry()
    registry.setdefault("models", []).append(entry)
    registry["updated_at"] = time.time()
    _atomic_json(REGISTRY_PATH, registry)
    return entry


def load_champion_entry() -> dict[str, Any] | None:
    if not CHAMPION_PATH.exists():
        return None
    try:
        data = json.loads(CHAMPION_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def load_champion_model() -> RankModelV223 | None:
    entry = load_champion_entry()
    if not entry:
        return None
    path = entry.get("path")
    if not path:
        return None
    try:
        return RankModelV223.load(Path(path))
    except Exception:
        return None


def _objective(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics.get("conditional_top1_20_rate", 0.0)),
        float(metrics.get("mrr20", 0.0)),
        -float(metrics.get("median_positive_rank") or 1e9),
    )


def maybe_promote_research_champion(entry: dict[str, Any]) -> tuple[bool, str]:
    """Promote only the best *research/shadow* champion. Never live authority."""
    current = load_champion_entry()
    if current is not None and _objective(current.get("metrics", {})) >= _objective(entry.get("metrics", {})):
        return False, "existing_champion_is_equal_or_better"
    champion = dict(entry)
    champion["status"] = "research_shadow_champion"
    champion["promoted_at"] = time.time()
    champion["eligible_for_live_authority"] = False
    _atomic_json(CHAMPION_PATH, champion)
    return True, "better_validation_objective"
