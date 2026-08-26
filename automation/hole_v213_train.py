from __future__ import annotations

import argparse
from pathlib import Path

from src.engine.ai.hole_patch_ai_v213 import HolePatchAIConfig
from src.engine.offline.hole_training_v213 import SamplingConfig, run_training_experiment


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train V2.13 Hole-AI v1 on candidate-centred crops derived from synt_*.png. "
            "Real hole_*.png stays holdout-only. Requires only numpy + OpenCV."
        )
    )
    p.add_argument("--root", type=Path, default=Path("content/ai/holes"))
    p.add_argument("--out", type=Path, default=Path("content/ai/reports/v213"))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-assets", type=int, default=96, help="Source images loaded per mini-batch (each generates positive + negatives)")
    p.add_argument("--seed", type=int, default=21301)
    p.add_argument("--holdout-backgrounds", default="black,checker,gray,bubbles")
    p.add_argument("--max-train-assets", type=int, default=0, help="0 = all; useful for a quick smoke run")
    p.add_argument("--max-eval-assets", type=int, default=0, help="0 = all")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--input-size", type=int, default=24)
    p.add_argument("--crop-size", type=int, default=64)
    p.add_argument("--positive-jitter", type=float, default=14.0)
    p.add_argument("--negative-min", type=float, default=24.0)
    p.add_argument("--negative-max", type=float, default=30.0)
    p.add_argument("--positives-per-image", type=int, default=1)
    p.add_argument("--negatives-per-image", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=0.0015)
    return p


def main() -> int:
    args = parser().parse_args()
    if args.positive_jitter <= 0:
        print("ERROR: V2.13 refuses centre-only training. --positive-jitter must be > 0.")
        return 2
    if args.negative_min <= args.positive_jitter + 2.0:
        print("ERROR: keep an ambiguity gap: --negative-min must be > --positive-jitter + 2px")
        return 2
    if args.negative_max <= args.negative_min:
        print("ERROR: --negative-max must be greater than --negative-min")
        return 2

    model_cfg = HolePatchAIConfig(
        crop_size=args.crop_size,
        input_size=args.input_size,
        hidden_size=args.hidden,
        learning_rate=args.learning_rate,
    )
    sampling = SamplingConfig(
        positive_jitter_px=args.positive_jitter,
        negative_min_px=args.negative_min,
        negative_max_px=args.negative_max,
        positives_per_image=args.positives_per_image,
        negatives_per_image=args.negatives_per_image,
    )
    holdout = tuple(value.strip() for value in args.holdout_backgrounds.split(",") if value.strip())
    report = run_training_experiment(
        holes_root=args.root,
        output_dir=args.out,
        model_config=model_cfg,
        sampling=sampling,
        holdout_backgrounds=holdout,
        epochs=args.epochs,
        batch_assets=args.batch_assets,
        seed=args.seed,
        max_train_assets=args.max_train_assets or None,
        max_eval_assets=args.max_eval_assets or None,
    )

    print("\nV2.13 RESULT")
    print("============")
    print(f"Model : {report['model_path']}")
    print(f"Report: {Path(args.out) / 'hole_v213_report.json'}")
    for name, result in report.get("evaluations", {}).items():
        ai = result.get("ai") if isinstance(result, dict) else None
        baseline = result.get("center_contrast_baseline") if isinstance(result, dict) else None
        if not isinstance(ai, dict):
            continue
        print(
            f"{name:28s} AI auc={ai.get('auc')} f1={ai.get('f1')} recall={ai.get('recall')} "
            f"| baseline f1={baseline.get('f1') if isinstance(baseline, dict) else None}"
        )
    print("\nInterpretation: synthetic/real holdouts are tests of pixel learning, NOT yet final full-frame hit detection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
