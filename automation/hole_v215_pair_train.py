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
    p = argparse.ArgumentParser(
        description=(
            "Retrain V2.14 mild + standard on one identical whole-session split. "
            "This fixes the V2.14 sweep comparison where profile seed also changed the split."
        )
    )
    p.add_argument("--root", type=Path, default=Path("content/ai/holes"))
    p.add_argument("--out", type=Path, default=Path("content/ai/reports/v215_pair"))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-assets", type=int, default=72)
    p.add_argument("--split-seed", type=int, default=21501)
    p.add_argument("--model-seed", type=int, default=31501)
    p.add_argument("--max-train-assets", type=int, default=0)
    p.add_argument("--max-eval-assets", type=int, default=0)
    p.add_argument("--holdout-backgrounds", default="black,checker,gray,bubbles")
    return p


def main() -> int:
    args = parser().parse_args()
    holdout = tuple(value.strip() for value in args.holdout_backgrounds.split(",") if value.strip())
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, profile in enumerate(("mild", "standard")):
        training_seed = int(args.model_seed) + index * 1009
        print(f"\n=== V2.15 PAIRED PROFILE {profile.upper()} ===")
        print(f"Shared split seed : {args.split_seed}")
        print(f"Training seed     : {training_seed}")
        report = run_training_experiment_v214(
            holes_root=args.root,
            output_dir=args.out / profile,
            model_config=HolePatchAIConfigV214(),
            sampling=SamplingConfigV214(),
            domain=DomainRandomizationConfigV214.from_profile(profile),
            holdout_backgrounds=holdout,
            epochs=args.epochs,
            batch_assets=args.batch_assets,
            seed=training_seed,
            split_seed=int(args.split_seed),
            max_train_assets=args.max_train_assets or None,
            max_eval_assets=args.max_eval_assets or None,
            v213_report_path=Path("content/ai/reports/v213/hole_v213_report.json"),
        )
        selected = next((row for row in report.get("history", []) if row.get("selected_epoch")), None) or {}
        novel = ((report.get("evaluations", {}).get("novel_background_holdout") or {}).get("ai") or {})
        real = ((report.get("evaluations", {}).get("real_holdout") or {}).get("ai") or {})
        rows.append(
            {
                "profile": profile,
                "training_seed": training_seed,
                "split_seed": int(args.split_seed),
                "selection_score": selected.get("selection_score"),
                "strict_novel_auc_report_only": novel.get("auc"),
                "real_recall_report_only": real.get("recall"),
                "model_path": report.get("model_path"),
                "session_assignment": (report.get("split") or {}).get("session_assignment"),
            }
        )

    assignments = [row.get("session_assignment") for row in rows]
    paired_ok = bool(assignments and all(value == assignments[0] for value in assignments[1:]))
    payload = {
        "schema_version": "2.15",
        "purpose": "paired_mild_standard_training_with_identical_session_split",
        "split_seed": int(args.split_seed),
        "paired_session_assignment_verified": paired_ok,
        "profiles": rows,
        "important_note": (
            "Strict novel-background and real holdouts are report-only. "
            "Do not choose the ensemble weight from these values."
        ),
    }
    path = args.out / "pair_summary.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nV2.15 PAIRED TRAIN SUMMARY")
    print("==========================")
    for row in rows:
        print(
            f"{row['profile']:9s} select={row['selection_score']} "
            f"strict_novel_auc={row['strict_novel_auc_report_only']} "
            f"real_recall={row['real_recall_report_only']}"
        )
    print(f"Shared session split verified: {paired_ok}")
    print(f"Report: {path}")
    if not paired_ok:
        print("ERROR: paired split verification failed; do not run V2.15 ensemble.")
        return 3
    print("Next: python3 -m automation.hole_v215_ensemble")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
