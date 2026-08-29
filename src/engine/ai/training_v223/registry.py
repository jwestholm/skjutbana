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
MIN_DOMAIN_SHOTS = 50
MIN_DOMAIN_ORACLE20 = 5
MIN_DOMAIN_ORACLE20_RATE = 0.05
MIN_BASELINE_ELIGIBLE_FRACTION = 0.50


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def load_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {"schema_version": "2.23.2", "models": [], "updated_at": None}
    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"schema_version": "2.23.2", "models": []}
    except Exception:
        return {"schema_version": "2.23.2", "models": []}


def register_model(
    model: RankModelV223,
    *,
    metrics: dict[str, Any],
    trigger: str,
    provisional: bool,
    trial_id: str,
    baseline_metrics: dict[str, Any] | None = None,
    support: dict[str, Any] | None = None,
    domain_metrics: dict[str, Any] | None = None,
    domain_baseline_metrics: dict[str, Any] | None = None,
    domain_support: dict[str, Any] | None = None,
) -> dict[str, Any]:
    directory = MODELS_ROOT / "challengers" / trial_id
    model.metadata.update({
        "trial_id": trial_id,
        "trigger": trigger,
        "provisional_split": bool(provisional),
        "eligible_for_live_authority": False,
        "v2231_support": dict(support or {}),
        "v2231_baseline_validation": dict(baseline_metrics or {}),
        "v2232_domain_metrics": dict(domain_metrics or {}),
        "v2232_domain_baseline": dict(domain_baseline_metrics or {}),
        "v2232_domain_support": dict(domain_support or {}),
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
        "domain_metrics": dict(domain_metrics or {}),
        "domain_baseline_metrics": dict(domain_baseline_metrics or {}),
        "domain_support": dict(domain_support or {}),
        "trigger": trigger,
        "provisional": bool(provisional),
        "eligible_for_live_authority": False,
        "registry_schema": "2.23.2",
    }
    gate = research_promotion_gate(entry)
    entry["research_promotion_gate"] = gate
    registry = load_registry()
    registry["schema_version"] = "2.23.2"
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
    domain = dict(entry.get("domain_metrics", {}) or {})
    domain_baseline = dict(entry.get("domain_baseline_metrics", {}) or {})
    domain_support = dict(entry.get("domain_support", {}) or {})

    validation_shots = int(metrics.get("shots", support.get("validation_shots", 0)) or 0)
    validation_oracle20 = int(metrics.get("oracle20", support.get("validation_oracle20", 0)) or 0)
    validation_oracle20_rate = _metric(metrics, "oracle20_rate")
    dev_oracle20 = int(support.get("development_oracle20", 0) or 0)
    domain_shots = int(domain.get("shots", domain_support.get("shots", 0)) or 0)
    domain_oracle20 = int(domain.get("oracle20", domain_support.get("oracle20", 0)) or 0)
    domain_oracle20_rate = _metric(domain, "oracle20_rate")

    baseline_eligible = int(baseline.get("eligible_ranked_shots", 0) or 0)
    domain_baseline_eligible = int(domain_baseline.get("eligible_ranked_shots", 0) or 0)
    min_base = max(1, int(round(validation_shots * MIN_BASELINE_ELIGIBLE_FRACTION)))
    min_domain_base = max(1, int(round(domain_shots * MIN_BASELINE_ELIGIBLE_FRACTION)))

    checks = {
        "validation_shots_ge_min": validation_shots >= MIN_VALIDATION_SHOTS,
        "validation_oracle20_ge_min": validation_oracle20 >= MIN_VALIDATION_ORACLE20,
        "validation_oracle20_rate_ge_min": validation_oracle20_rate >= MIN_VALIDATION_ORACLE20_RATE,
        "development_oracle20_ge_min": dev_oracle20 >= MIN_DEV_ORACLE20,
        "validation_reference_baseline_coverage": baseline_eligible >= min_base,
        "domain_shots_ge_min": domain_shots >= MIN_DOMAIN_SHOTS,
        "domain_oracle20_ge_min": domain_oracle20 >= MIN_DOMAIN_ORACLE20,
        "domain_oracle20_rate_ge_min": domain_oracle20_rate >= MIN_DOMAIN_ORACLE20_RATE,
        "domain_reference_baseline_coverage": domain_baseline_eligible >= min_domain_base,
    }

    challenger_cond = _metric(metrics, "conditional_top1_20_rate")
    challenger_mrr = _metric(metrics, "mrr20")
    baseline_cond = _metric(baseline, "conditional_top1_20_rate")
    baseline_mrr = _metric(baseline, "mrr20")
    validation_improved = (
        challenger_cond >= baseline_cond + MIN_CONDITIONAL_TOP1_IMPROVEMENT
        or challenger_mrr >= baseline_mrr + MIN_MRR_IMPROVEMENT
    )
    domain_cond = _metric(domain, "conditional_top1_20_rate")
    domain_mrr = _metric(domain, "mrr20")
    domain_base_cond = _metric(domain_baseline, "conditional_top1_20_rate")
    domain_base_mrr = _metric(domain_baseline, "mrr20")
    domain_improved = (
        domain_cond >= domain_base_cond + MIN_CONDITIONAL_TOP1_IMPROVEMENT
        or domain_mrr >= domain_base_mrr + MIN_MRR_IMPROVEMENT
    )
    checks["beats_validation_reference"] = bool(validation_improved)
    checks["beats_fresh_f2_domain_reference"] = bool(domain_improved)

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
            "min_domain_shots": MIN_DOMAIN_SHOTS,
            "min_domain_oracle20": MIN_DOMAIN_ORACLE20,
            "min_domain_oracle20_rate": MIN_DOMAIN_ORACLE20_RATE,
            "min_reference_eligible_fraction": MIN_BASELINE_ELIGIBLE_FRACTION,
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
        "reason": "valid_research_champion" if gate.get("passed") else "quarantined_pre_v2232_or_missing_domain_gate",
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


def _entry_objective(entry: dict[str, Any]) -> tuple[float, float, float, float, float]:
    """Prefer fresh-domain generalisation before ordinary validation."""
    domain = dict(entry.get("domain_metrics", {}) or {})
    validation = dict(entry.get("metrics", {}) or {})
    return (
        _metric(domain, "conditional_top1_20_rate"),
        _metric(domain, "mrr20"),
        _metric(validation, "conditional_top1_20_rate"),
        _metric(validation, "mrr20"),
        -float(validation.get("median_positive_rank") or 1e9),
    )


def maybe_promote_research_champion(entry: dict[str, Any]) -> tuple[bool, str]:
    """Promote only a support-gated *research/shadow* champion. Never live authority."""
    gate = research_promotion_gate(entry)
    if not bool(gate.get("passed")):
        return False, "promotion_gate_failed:" + ",".join(gate.get("reasons", []))

    current = load_champion_entry()
    if current is not None:
        current_gate = research_promotion_gate(current)
        if bool(current_gate.get("passed")) and _entry_objective(current) >= _entry_objective(entry):
            return False, "existing_valid_champion_is_equal_or_better"

    champion = dict(entry)
    champion["status"] = "research_shadow_champion"
    champion["promoted_at"] = time.time()
    champion["eligible_for_live_authority"] = False
    champion["research_promotion_gate"] = gate
    _atomic_json(CHAMPION_PATH, champion)
    return True, "support_gate_passed_and_better_validation_objective"
