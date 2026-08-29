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

# Research/shadow promotion gates. These do NOT grant live authority; they only
# prevent meaningless plumbing runs from becoming the model used for shadow
# comparisons. Protected holdout remains outside automatic selection entirely.
MIN_VALIDATION_SHOTS = 12
MIN_VALIDATION_ORACLE20 = 5
MIN_VALIDATION_ORACLE20_RATE = 0.20
MIN_DEV_ORACLE20 = 8
MIN_MRR_IMPROVEMENT = 0.005
MIN_CONDITIONAL_TOP1_IMPROVEMENT = 0.01


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def load_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {"schema_version": "2.23.1", "models": [], "updated_at": None}
    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"schema_version": "2.23.1", "models": []}
    except Exception:
        return {"schema_version": "2.23.1", "models": []}


def register_model(
    model: RankModelV223,
    *,
    metrics: dict[str, Any],
    trigger: str,
    provisional: bool,
    trial_id: str,
    baseline_metrics: dict[str, Any] | None = None,
    support: dict[str, Any] | None = None,
) -> dict[str, Any]:
    directory = MODELS_ROOT / "challengers" / trial_id
    model.metadata.update({
        "trial_id": trial_id,
        "trigger": trigger,
        "provisional_split": bool(provisional),
        "eligible_for_live_authority": False,
        "v2231_support": dict(support or {}),
        "v2231_baseline_validation": dict(baseline_metrics or {}),
    })
    model.save(directory)
    entry = {
        "trial_id": trial_id,
        "created_at": time.time(),
        "kind": model.kind,
        "path": str(directory),
        "metrics": metrics,
        "baseline_metrics": dict(baseline_metrics or {}),
        "support": dict(support or {}),
        "trigger": trigger,
        "provisional": bool(provisional),
        "eligible_for_live_authority": False,
        "registry_schema": "2.23.1",
    }
    gate = research_promotion_gate(entry)
    entry["research_promotion_gate"] = gate
    registry = load_registry()
    registry["schema_version"] = "2.23.1"
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


def _metric(metrics: dict[str, Any], name: str) -> float:
    try:
        return float(metrics.get(name, 0.0) or 0.0)
    except Exception:
        return 0.0


def research_promotion_gate(entry: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(entry.get("metrics", {}) or {})
    baseline = dict(entry.get("baseline_metrics", {}) or {})
    support = dict(entry.get("support", {}) or {})

    validation_shots = int(metrics.get("shots", support.get("validation_shots", 0)) or 0)
    validation_oracle20 = int(metrics.get("oracle20", support.get("validation_oracle20", 0)) or 0)
    validation_oracle20_rate = _metric(metrics, "oracle20_rate")
    dev_oracle20 = int(support.get("development_oracle20", 0) or 0)

    checks = {
        "validation_shots_ge_min": validation_shots >= MIN_VALIDATION_SHOTS,
        "validation_oracle20_ge_min": validation_oracle20 >= MIN_VALIDATION_ORACLE20,
        "validation_oracle20_rate_ge_min": validation_oracle20_rate >= MIN_VALIDATION_ORACLE20_RATE,
        "development_oracle20_ge_min": dev_oracle20 >= MIN_DEV_ORACLE20,
    }

    challenger_cond = _metric(metrics, "conditional_top1_20_rate")
    challenger_mrr = _metric(metrics, "mrr20")
    baseline_cond = _metric(baseline, "conditional_top1_20_rate")
    baseline_mrr = _metric(baseline, "mrr20")
    if baseline:
        improved = (
            challenger_cond >= baseline_cond + MIN_CONDITIONAL_TOP1_IMPROVEMENT
            or challenger_mrr >= baseline_mrr + MIN_MRR_IMPROVEMENT
        )
    else:
        # Old V2.23.0 entries have no baseline/support metadata. They must not be
        # silently trusted; at minimum they need actual positive validation rank.
        improved = challenger_cond > 0.0 or challenger_mrr > 0.0
    checks["beats_baseline_or_has_meaningful_rank_signal"] = bool(improved)

    reasons = [name for name, ok in checks.items() if not ok]
    return {
        "passed": not reasons,
        "checks": checks,
        "reasons": reasons,
        "thresholds": {
            "min_validation_shots": MIN_VALIDATION_SHOTS,
            "min_validation_oracle20": MIN_VALIDATION_ORACLE20,
            "min_validation_oracle20_rate": MIN_VALIDATION_ORACLE20_RATE,
            "min_development_oracle20": MIN_DEV_ORACLE20,
            "min_mrr_improvement": MIN_MRR_IMPROVEMENT,
            "min_conditional_top1_improvement": MIN_CONDITIONAL_TOP1_IMPROVEMENT,
        },
    }


def champion_gate_status() -> dict[str, Any]:
    entry = load_champion_entry()
    if not entry:
        return {"exists": False, "usable": False, "reason": "none"}
    gate = research_promotion_gate(entry)
    return {
        "exists": True,
        "usable": bool(gate.get("passed")),
        "trial_id": entry.get("trial_id"),
        "gate": gate,
        "reason": "valid_research_champion" if gate.get("passed") else "quarantined_pre_v2231_or_insufficient_support",
    }


def load_champion_model() -> RankModelV223 | None:
    entry = load_champion_entry()
    if not entry:
        return None
    gate = research_promotion_gate(entry)
    if not bool(gate.get("passed")):
        # V2.23.0 could promote a model with zero positive validation shots. Such
        # plumbing champions remain on disk for audit but are never used in shadow.
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
        _metric(metrics, "conditional_top1_20_rate"),
        _metric(metrics, "mrr20"),
        -float(metrics.get("median_positive_rank") or 1e9),
    )


def maybe_promote_research_champion(entry: dict[str, Any]) -> tuple[bool, str]:
    """Promote only a support-gated *research/shadow* champion. Never live authority."""
    gate = research_promotion_gate(entry)
    if not bool(gate.get("passed")):
        return False, "promotion_gate_failed:" + ",".join(gate.get("reasons", []))

    current = load_champion_entry()
    if current is not None:
        current_gate = research_promotion_gate(current)
        if bool(current_gate.get("passed")) and _objective(current.get("metrics", {})) >= _objective(entry.get("metrics", {})):
            return False, "existing_valid_champion_is_equal_or_better"

    champion = dict(entry)
    champion["status"] = "research_shadow_champion"
    champion["promoted_at"] = time.time()
    champion["eligible_for_live_authority"] = False
    champion["research_promotion_gate"] = gate
    _atomic_json(CHAMPION_PATH, champion)
    return True, "support_gate_passed_and_better_validation_objective"
