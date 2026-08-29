from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from .dataset import compile_dataset
from .model import evaluate_baseline, evaluate_model, train_rank_model
from .registry import maybe_promote_research_champion, register_model

REPORT_ROOT = Path("content/ai/training_v223/reports")
_LOCK = threading.Lock()
_BACKGROUND_THREAD: threading.Thread | None = None


def _safe_write_report(report: dict[str, Any], *, name: str) -> Path:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    path = REPORT_ROOT / name
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (REPORT_ROOT / "latest.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def train_once_v223(
    *,
    trigger: str = "offline",
    quick: bool = False,
    include_legacy: bool = True,
    seed_base: int = 2230,
) -> dict[str, Any]:
    with _LOCK:
        dataset = compile_dataset(include_legacy=include_legacy)
        split = dataset.split()
        summary = dataset.summary()
        if len(split.development) < 2 or len(split.validation) < 1:
            report = {
                "schema_version": "2.23.0", "status": "insufficient_data", "trigger": trigger,
                "dataset": summary, "split": {
                    "development": len(split.development), "validation": len(split.validation),
                    "holdout_protected": len(split.holdout), "provisional": split.provisional,
                    "notes": split.notes,
                },
                "message": "Need at least two development shots and one validation shot with usable actual candidates.",
            }
            _safe_write_report(report, name=f"train_{int(time.time())}_insufficient.json")
            return report

        baseline = evaluate_baseline(split.validation)
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
                        "dataset_summary": summary,
                        "development_sessions": sorted(set(r.session_id for r in split.development)),
                        "validation_sessions": sorted(set(r.session_id for r in split.validation)),
                        "protected_holdout_sessions": sorted(set(r.session_id for r in split.holdout)),
                        "holdout_evaluated_for_selection": False,
                    },
                )
                metrics = evaluate_model(model, split.validation)
                trial_id = f"{stamp}_{idx}_{cfg['kind']}_s{cfg['seed']}"
                entry = register_model(model, metrics=metrics, trigger=trigger, provisional=split.provisional, trial_id=trial_id)
                entry["train_info"] = {"usable_training_shots": train_info.get("usable_training_shots"), "final_loss": train_info.get("loss_history", [None])[-1]}
                entries.append(entry)
            except Exception as exc:
                errors.append(f"{cfg['kind']} seed={cfg['seed']}: {type(exc).__name__}: {exc}")

        def objective(entry: dict[str, Any]) -> tuple[float, float, float]:
            m = entry.get("metrics", {})
            return (float(m.get("conditional_top1_20_rate", 0.0)), float(m.get("mrr20", 0.0)), -float(m.get("median_positive_rank") or 1e9))

        best = max(entries, key=objective) if entries else None
        promoted = False
        promote_reason = "no_successful_challenger"
        if best is not None:
            promoted, promote_reason = maybe_promote_research_champion(best)
        report = {
            "schema_version": "2.23.0",
            "status": "ok" if entries else "failed",
            "trigger": trigger,
            "timestamp": time.time(),
            "dataset": summary,
            "legacy_import": dataset.legacy_report,
            "split": {
                "development": len(split.development), "validation": len(split.validation),
                "holdout_protected": len(split.holdout), "provisional": split.provisional,
                "notes": split.notes,
            },
            "baseline_validation": baseline,
            "trials": entries,
            "errors": errors,
            "best_trial_id": best.get("trial_id") if best else None,
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
        print("[V2.23 AUTOTRAIN] background challenger training started (shadow only)")
        try:
            report = train_once_v223(trigger=trigger, quick=True)
            print(
                "[V2.23 AUTOTRAIN] finished "
                f"status={report.get('status')} shots={report.get('dataset', {}).get('shots', 0)} "
                f"best={report.get('best_trial_id')} promoted={report.get('research_champion_promoted', False)}"
            )
        except Exception as exc:
            print(f"[V2.23 AUTOTRAIN] failed open: {type(exc).__name__}: {exc}")

    _BACKGROUND_THREAD = threading.Thread(target=worker, name="V223Autotrain", daemon=True)
    _BACKGROUND_THREAD.start()
    return True
