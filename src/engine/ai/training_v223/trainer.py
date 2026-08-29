from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Sequence

from .dataset import DatasetV223, compile_dataset
from .domain import select_fresh_f2_domain
from .model import evaluate_baseline, evaluate_model, train_rank_model
from .registry import (
    MIN_DEV_ORACLE20,
    MIN_DOMAIN_ORACLE20,
    MIN_DOMAIN_SHOTS,
    MIN_VALIDATION_ORACLE20,
    MIN_VALIDATION_SHOTS,
    maybe_promote_research_champion,
    register_model,
)

REPORT_ROOT = Path("content/ai/training_v223/reports")
_LOCK = threading.Lock()
_BACKGROUND_THREAD: threading.Thread | None = None


def _safe_write_report(report: dict[str, Any], *, name: str) -> Path:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    path = REPORT_ROOT / name
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (REPORT_ROOT / "latest.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _oracle20_count(records: Sequence[Any]) -> int:
    return sum(int(bool(getattr(record, "oracle20", False))) for record in records)


def _support(split: Any) -> dict[str, Any]:
    return {
        "development_shots": len(split.development),
        "development_oracle20": _oracle20_count(split.development),
        "validation_shots": len(split.validation),
        "validation_oracle20": _oracle20_count(split.validation),
        "protected_holdout_shots": len(split.holdout),
        "protected_holdout_evaluated_for_selection": False,
    }


def _domain_support(records: Sequence[Any], session_id: str | None) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "shots": len(records),
        "oracle20": _oracle20_count(records),
        "proposal_expanded_shots": sum(int(bool(getattr(r, "metadata", {}).get("v2232_proposal_expanded"))) for r in records),
    }


def train_once_v223(
    *,
    trigger: str = "offline",
    quick: bool = False,
    include_legacy: bool = True,
    seed_base: int = 2232,
) -> dict[str, Any]:
    """Train challengers while reserving the freshest substantial F2 session.

    V2.23.2 uses the latest >=50-shot F2/projector session as a domain gate. It
    is excluded from fitting and ordinary validation. Protected holdout is still
    never evaluated for automatic selection.
    """
    with _LOCK:
        dataset = compile_dataset(include_legacy=include_legacy)
        domain_sel = select_fresh_f2_domain(dataset.records, min_shots=MIN_DOMAIN_SHOTS)
        engineering = DatasetV223(records=domain_sel.engineering_records, legacy_report=dataset.legacy_report)
        split = engineering.split()
        summary = dataset.summary()
        support = _support(split)
        domain_support = _domain_support(domain_sel.records, domain_sel.session_id)

        basic_ok = len(split.development) >= 2 and len(split.validation) >= 1
        positive_ok = (
            support["development_oracle20"] >= MIN_DEV_ORACLE20
            and support["validation_shots"] >= MIN_VALIDATION_SHOTS
            and support["validation_oracle20"] >= MIN_VALIDATION_ORACLE20
        )
        domain_ok = (
            domain_support["shots"] >= MIN_DOMAIN_SHOTS
            and domain_support["oracle20"] >= MIN_DOMAIN_ORACLE20
        )
        baseline = evaluate_baseline(split.validation) if split.validation else None
        domain_baseline = evaluate_baseline(domain_sel.records) if domain_sel.records else None

        if not basic_ok or not positive_ok or not domain_ok:
            reasons: list[str] = []
            if len(split.development) < 2: reasons.append("development_shots_lt_2")
            if len(split.validation) < 1: reasons.append("validation_shots_lt_1")
            if support["development_oracle20"] < MIN_DEV_ORACLE20: reasons.append(f"development_oracle20_lt_{MIN_DEV_ORACLE20}")
            if support["validation_shots"] < MIN_VALIDATION_SHOTS: reasons.append(f"validation_shots_lt_{MIN_VALIDATION_SHOTS}")
            if support["validation_oracle20"] < MIN_VALIDATION_ORACLE20: reasons.append(f"validation_oracle20_lt_{MIN_VALIDATION_ORACLE20}")
            if domain_support["shots"] < MIN_DOMAIN_SHOTS: reasons.append(f"fresh_f2_domain_shots_lt_{MIN_DOMAIN_SHOTS}")
            if domain_support["oracle20"] < MIN_DOMAIN_ORACLE20: reasons.append(f"fresh_f2_domain_oracle20_lt_{MIN_DOMAIN_ORACLE20}")
            report = {
                "schema_version": "2.23.2",
                "status": "insufficient_domain_or_positive_support",
                "trigger": trigger,
                "dataset": summary,
                "legacy_import": dataset.legacy_report,
                "support": support,
                "domain": {"selection_reason": domain_sel.reason, "support": domain_support, "baseline": domain_baseline},
                "split": {
                    "development": len(split.development), "validation": len(split.validation),
                    "holdout_protected": len(split.holdout), "provisional": split.provisional,
                    "notes": split.notes,
                },
                "baseline_validation": baseline,
                "reasons": reasons,
                "research_champion_promoted": False,
                "eligible_for_live_authority": False,
                "protected_holdout_evaluated_for_selection": False,
            }
            _safe_write_report(report, name=f"train_{int(time.time())}_insufficient_v2232.json")
            return report

        trials = [
            {"kind": "linear", "hidden": 0, "seed": seed_base, "epochs": 35 if quick else 90, "lr": 0.018, "l2": 0.0015},
            {"kind": "mlp", "hidden": 16, "seed": seed_base + 1, "epochs": 45 if quick else 110, "lr": 0.010, "l2": 0.0015},
            {"kind": "mlp", "hidden": 32, "seed": seed_base + 2, "epochs": 55 if quick else 130, "lr": 0.008, "l2": 0.0020},
        ]
        entries: list[dict[str, Any]] = []
        errors: list[str] = []
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for idx, cfg in enumerate(trials, start=1):
            try:
                model, train_info = train_rank_model(
                    split.development,
                    kind=cfg["kind"], hidden=cfg["hidden"] or 16,
                    epochs=cfg["epochs"], learning_rate=cfg["lr"], l2=cfg["l2"], seed=cfg["seed"],
                    metadata={
                        "schema_version": "2.23.2",
                        "dataset_summary": summary,
                        "v2232_support": support,
                        "v2232_domain_session": domain_sel.session_id,
                        "development_sessions": sorted(set(r.session_id for r in split.development)),
                        "validation_sessions": sorted(set(r.session_id for r in split.validation)),
                        "protected_holdout_sessions": sorted(set(r.session_id for r in split.holdout)),
                        "holdout_evaluated_for_selection": False,
                    },
                )
                metrics = evaluate_model(model, split.validation)
                domain_metrics = evaluate_model(model, domain_sel.records)
                trial_id = f"{stamp}_{idx}_{cfg['kind']}_s{cfg['seed']}"
                entry = register_model(
                    model,
                    metrics=metrics,
                    baseline_metrics=baseline or {},
                    support=support,
                    domain_metrics=domain_metrics,
                    domain_baseline_metrics=domain_baseline or {},
                    domain_support=domain_support,
                    trigger=trigger,
                    provisional=True,  # fresh-domain research gate still does not imply authority
                    trial_id=trial_id,
                )
                entry["train_info"] = {
                    "usable_training_shots": train_info.get("usable_training_shots"),
                    "final_loss": train_info.get("loss_history", [None])[-1],
                }
                entries.append(entry)
            except Exception as exc:
                errors.append(f"{cfg['kind']} seed={cfg['seed']}: {type(exc).__name__}: {exc}")

        def objective(entry: dict[str, Any]) -> tuple[float, float, float, float]:
            dm = entry.get("domain_metrics", {})
            vm = entry.get("metrics", {})
            return (
                float(dm.get("conditional_top1_20_rate", 0.0)),
                float(dm.get("mrr20", 0.0)),
                float(vm.get("conditional_top1_20_rate", 0.0)),
                float(vm.get("mrr20", 0.0)),
            )

        best = max(entries, key=objective) if entries else None
        promoted = False
        promote_reason = "no_successful_challenger"
        if best is not None:
            promoted, promote_reason = maybe_promote_research_champion(best)
        report = {
            "schema_version": "2.23.2",
            "status": "ok" if entries else "failed",
            "trigger": trigger,
            "timestamp": time.time(),
            "dataset": summary,
            "legacy_import": dataset.legacy_report,
            "support": support,
            "domain": {
                "selection_reason": domain_sel.reason,
                "session_id": domain_sel.session_id,
                "support": domain_support,
                "baseline": domain_baseline,
            },
            "split": {
                "development": len(split.development), "validation": len(split.validation),
                "holdout_protected": len(split.holdout), "provisional": True,
                "notes": split.notes + ["Newest substantial F2 session excluded from fitting and used as fresh-domain research gate."],
            },
            "baseline_validation": baseline,
            "trials": entries,
            "errors": errors,
            "best_trial_id": best.get("trial_id") if best else None,
            "best_promotion_gate": best.get("research_promotion_gate") if best else None,
            "research_champion_promoted": promoted,
            "promotion_reason": promote_reason,
            "eligible_for_live_authority": False,
            "protected_holdout_evaluated_for_selection": False,
        }
        _safe_write_report(report, name=f"train_{stamp}.json")
        return report


def schedule_quick_autotrain_v223(*, trigger: str = "f2_completed") -> bool:
    global _BACKGROUND_THREAD
    if _BACKGROUND_THREAD is not None and _BACKGROUND_THREAD.is_alive():
        return False

    def worker() -> None:
        print("[V2.23.2 AUTOTRAIN] proposal expansion + challenger cycle started (shadow only)")
        try:
            # Expand the latest captured F2 session first. The latest session is
            # then reserved by train_once_v223 as fresh domain validation.
            try:
                from .proposal import expand_session
                proposal_report = expand_session("latest")
                print(f"[V2.23.2 AUTOTRAIN] proposal status={proposal_report.get('status')} processed={proposal_report.get('processed',0)} oracle20={proposal_report.get('oracle20',{}).get('union')}")
            except Exception as exc:
                print(f"[V2.23.2 AUTOTRAIN] proposal expansion failed open: {type(exc).__name__}: {exc}")
            report = train_once_v223(trigger=trigger, quick=True)
            print(
                "[V2.23.2 AUTOTRAIN] finished "
                f"status={report.get('status')} shots={report.get('dataset', {}).get('shots', 0)} "
                f"domain={report.get('domain',{}).get('session_id')} best={report.get('best_trial_id')} "
                f"promoted={report.get('research_champion_promoted', False)}"
            )
        except Exception as exc:
            print(f"[V2.23.2 AUTOTRAIN] failed open: {type(exc).__name__}: {exc}")

    _BACKGROUND_THREAD = threading.Thread(target=worker, name="V2232Autotrain", daemon=True)
    _BACKGROUND_THREAD.start()
    return True
