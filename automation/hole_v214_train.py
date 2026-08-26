from __future__ import annotations

import argparse
from pathlib import Path

from src.engine.ai.hole_patch_ai_v214 import HolePatchAIConfigV214
from src.engine.offline.hole_training_v214 import (
    DomainRandomizationConfigV214,
    SamplingConfigV214,
    run_training_experiment_v214,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train V2.14 background-generalising Hole-AI on synthetic camera hole patches.")
    p.add_argument("--root", type=Path, default=Path("content/ai/holes"))
    p.add_argument("--out", type=Path, default=Path("content/ai/reports/v214"))
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-assets", type=int, default=72)
    p.add_argument("--seed", type=int, default=21401)
    p.add_argument("--profile", choices=("mild", "standard", "strong"), default="standard")
    p.add_argument("--holdout-backgrounds", default="black,checker,gray,bubbles")
    p.add_argument("--max-train-assets", type=int, default=0)
    p.add_argument("--max-eval-assets", type=int, default=0)
    p.add_argument("--hidden", type=int, default=80)
    p.add_argument("--input-size", type=int, default=22)
    p.add_argument("--crop-size", type=int, default=64)
    p.add_argument("--positive-jitter", type=float, default=16.0)
    p.add_argument("--negative-min", type=float, default=24.0)
    p.add_argument("--negative-max", type=float, default=30.0)
    p.add_argument("--positives-per-image", type=int, default=1)
    p.add_argument("--negatives-per-image", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=0.0012)
    p.add_argument("--v213-report", type=Path, default=Path("content/ai/reports/v213/hole_v213_report.json"))
    return p


def main() -> int:
    args = parser().parse_args()
    if args.positive_jitter <= 0:
        print("ERROR: centre-only training is forbidden; --positive-jitter must be >0")
        return 2
    if args.negative_min <= args.positive_jitter + 2:
        print("ERROR: --negative-min must remain > --positive-jitter + 2px")
        return 2
    if args.negative_max <= args.negative_min:
        print("ERROR: --negative-max must be greater than --negative-min")
        return 2

    model_cfg = HolePatchAIConfigV214(
        crop_size=args.crop_size,
        input_size=args.input_size,
        hidden_size=args.hidden,
        learning_rate=args.learning_rate,
    )
    sampling = SamplingConfigV214(
        positive_jitter_px=args.positive_jitter,
        negative_min_px=args.negative_min,
        negative_max_px=args.negative_max,
        positives_per_image=args.positives_per_image,
        negatives_per_image=args.negatives_per_image,
    )
    domain = DomainRandomizationConfigV214.from_profile(args.profile)
    holdout = tuple(value.strip() for value in args.holdout_backgrounds.split(",") if value.strip())

    report = run_training_experiment_v214(
        holes_root=args.root,
        output_dir=args.out,
        model_config=model_cfg,
        sampling=sampling,
        domain=domain,
        holdout_backgrounds=holdout,
        epochs=args.epochs,
        batch_assets=args.batch_assets,
        seed=args.seed,
        max_train_assets=args.max_train_assets or None,
        max_eval_assets=args.max_eval_assets or None,
        v213_report_path=args.v213_report if args.v213_report.exists() else None,
    )

    print("\nV2.14 RESULT")
    print("============")
    print(f"Model : {report['model_path']}")
    print(f"Report: {Path(args.out) / 'hole_v214_report.json'}")
    for name, result in report.get("evaluations", {}).items():
        ai = result.get("ai") if isinstance(result, dict) else None
        if isinstance(ai, dict):
            print(f"{name:28s} AUC={ai.get('auc')} F1={ai.get('f1')} recall={ai.get('recall')}")

    comparison = report.get("comparison_to_v213") or {}
    if comparison.get("available"):
        print("\nDELTA VS V2.13")
        for name, row in (comparison.get("metrics") or {}).items():
            print(
                f"{name:28s} AUC {row.get('v213_auc')} -> {row.get('v214_auc')} "
                f"(delta {row.get('auc_delta')})"
            )

    gate = report.get("gate") or {}
    print("\nGATE")
    for key, value in gate.items():
        print(f"  {key}: {value}")
    print("\nV2.14 remains OFFLINE ONLY. A passing gate means 'worth candidate-shadow integration', not live authority.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
