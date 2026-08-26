from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.engine.ai.hole_patch_ai_v214 import HolePatchAIConfigV214
from src.engine.offline.hole_training_v214 import (
    DomainRandomizationConfigV214,
    SamplingConfigV214,
    run_training_experiment_v214,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run V2.14 mild/standard/strong domain profiles. Winner is ranked WITHOUT strict novel/REAL holdouts.")
    p.add_argument("--root", type=Path, default=Path("content/ai/holes"))
    p.add_argument("--out", type=Path, default=Path("content/ai/reports/v214_sweep"))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-assets", type=int, default=72)
    p.add_argument("--max-train-assets", type=int, default=0)
    p.add_argument("--max-eval-assets", type=int, default=0)
    p.add_argument("--seed", type=int, default=21401)
    return p


def main() -> int:
    args = parser().parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, profile in enumerate(("mild", "standard", "strong")):
        print(f"\n=== PROFILE {profile.upper()} ===")
        report = run_training_experiment_v214(
            holes_root=args.root,
            output_dir=args.out / profile,
            model_config=HolePatchAIConfigV214(),
            sampling=SamplingConfigV214(),
            domain=DomainRandomizationConfigV214.from_profile(profile),
            holdout_backgrounds=("black", "checker", "gray", "bubbles"),
            epochs=args.epochs,
            batch_assets=args.batch_assets,
            seed=args.seed + index * 101,
            max_train_assets=args.max_train_assets or None,
            max_eval_assets=args.max_eval_assets or None,
            v213_report_path=Path("content/ai/reports/v213/hole_v213_report.json"),
        )
        selected = next((row for row in report.get("history", []) if row.get("selected_epoch")), None) or {}
        novel = ((report.get("evaluations", {}).get("novel_background_holdout") or {}).get("ai") or {})
        real = ((report.get("evaluations", {}).get("real_holdout") or {}).get("ai") or {})
        rows.append({
            "profile": profile,
            "selection_score": float(selected.get("selection_score") or 0.0),
            "selected_epoch": report.get("selected_epoch"),
            "strict_novel_auc_report_only": novel.get("auc"),
            "real_recall_report_only": real.get("recall"),
            "model_path": report.get("model_path"),
        })

    ranked = sorted(rows, key=lambda row: row["selection_score"], reverse=True)
    payload = {
        "schema_version": "2.14",
        "selection_rule": "rank by clean+procedural-domain validation selection_score only; strict novel and REAL holdouts are report-only",
        "winner": ranked[0] if ranked else None,
        "profiles": ranked,
    }
    report_path = args.out / "sweep_summary.json"
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nV2.14 SWEEP SUMMARY")
    print("===================")
    for row in ranked:
        print(
            f"{row['profile']:9s} select={row['selection_score']:.4f} "
            f"strict_novel_auc={row['strict_novel_auc_report_only']} real_recall={row['real_recall_report_only']}"
        )
    if ranked:
        print(f"Winner by NON-HOLDOUT selection: {ranked[0]['profile']}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
