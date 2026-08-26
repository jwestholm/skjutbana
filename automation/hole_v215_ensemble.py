from __future__ import annotations

import argparse
from pathlib import Path

from src.engine.offline.hole_ensemble_v215 import (
    EnsembleSearchConfigV215,
    run_ensemble_experiment_v215,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "V2.15 paired mild+standard Hole-AI experiment. Blend selection uses ONLY "
            "clean/procedural validation; strict holdouts are report-only."
        )
    )
    p.add_argument("--root", type=Path, default=Path("content/ai/holes"))
    p.add_argument("--standard-model", type=Path, default=Path("content/ai/reports/v215_pair/standard/hole_patch_ai_v214.npz"))
    p.add_argument("--mild-model", type=Path, default=Path("content/ai/reports/v215_pair/mild/hole_patch_ai_v214.npz"))
    p.add_argument("--out", type=Path, default=Path("content/ai/reports/v215"))
    p.add_argument("--seed", type=int, default=21501)
    p.add_argument("--max-eval-assets", type=int, default=0)
    p.add_argument("--weight-step", type=float, default=0.025)
    return p


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def main() -> int:
    args = parser().parse_args()
    missing = [path for path in (args.standard_model, args.mild_model) if not path.exists()]
    if missing:
        print("ERROR: V2.14 sweep model(s) missing:")
        for path in missing:
            print(f"  {path}")
        print("Run: python3 -m automation.hole_v215_pair_train")
        return 2

    try:
        report = run_ensemble_experiment_v215(
            holes_root=args.root,
            standard_model_path=args.standard_model,
            mild_model_path=args.mild_model,
            output_dir=args.out,
            seed=args.seed,
            max_eval_assets=args.max_eval_assets or None,
            search=EnsembleSearchConfigV215(weight_step=args.weight_step),
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 3

    winner = (report.get("search") or {}).get("winner") or {}
    print("\nV2.15 PAIRED ENSEMBLE RESULT")
    print("============================")
    print(f"Standard weight : {_fmt(winner.get('standard_weight'))}")
    print(f"Mild weight     : {_fmt(winner.get('mild_weight'))}")
    print(f"Fused threshold : {_fmt(winner.get('threshold'))}")
    print(f"Selection score : {_fmt(winner.get('selection_score'))}")
    print(f"Best pure score : {_fmt(winner.get('best_pure_selection_score'))}")
    print(f"Non-trivial mix : {bool(winner.get('blend_is_nontrivial'))}")

    for name in ("validation", "synthetic_test", "novel_background_holdout", "real_holdout", "off_center_stress", "procedural_domain_stress"):
        row = (report.get("evaluations") or {}).get(name) or {}
        fused = row.get("fused") or {}
        comp = row.get("complementarity") or {}
        if not fused:
            continue
        print(
            f"{name:28s} fused_auc={_fmt(fused.get('auc'))} "
            f"f1={_fmt(fused.get('f1'))} recall={_fmt(fused.get('recall'))} "
            f"either_recall={_fmt(comp.get('positive_oracle_either_recall'))} "
            f"complementary={_fmt(comp.get('positive_complementary_rescue_fraction'))}"
        )

    print("\nNOVEL BACKGROUND BREAKDOWN")
    for background, row in (report.get("novel_per_background") or {}).items():
        fused = row.get("fused") or {}
        print(f"  {background:12s} AUC={_fmt(fused.get('auc'))} recall={_fmt(fused.get('recall'))}")

    print("\nGATE")
    for key, value in (report.get("gate") or {}).items():
        print(f"  {key}: {value}")
    print(f"\nReport : {args.out / 'hole_v215_report.json'}")
    print(f"Config : {args.out / 'hole_v215_ensemble.json'}")
    print("V2.15 remains SHADOW/OFFLINE ONLY. The oracle 'either' value is diagnostic, not live authority.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
