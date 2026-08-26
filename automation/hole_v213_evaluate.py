from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.engine.ai.hole_patch_ai_v213 import HolePatchAI
from src.engine.offline.hole_dataset_v213 import build_dataset_split, discover_hole_assets
from src.engine.offline.hole_training_v213 import (
    SamplingConfig,
    evaluate_assets,
    evaluation_seed,
    summarize_evaluation,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Re-evaluate an already trained V2.13 Hole-AI without training it again.")
    p.add_argument("--root", type=Path, default=Path("content/ai/holes"))
    p.add_argument("--model", type=Path, default=Path("content/ai/reports/v213/hole_patch_ai_v213.npz"))
    p.add_argument("--out", type=Path, default=Path("content/ai/reports/v213/hole_v213_reevaluate.json"))
    p.add_argument("--seed", type=int, default=None, help="Override training seed; normally read from model metadata")
    return p


def main() -> int:
    args = parser().parse_args()
    model, metadata = HolePatchAI.load(args.model)
    model_meta = dict(metadata.get("metadata") or {})
    sampling = SamplingConfig(**dict(model_meta.get("sampling") or {}))
    holdout_backgrounds = tuple(model_meta.get("holdout_backgrounds") or ("black", "checker", "gray", "bubbles"))
    ai_threshold = float(model_meta.get("ai_threshold", 0.5))
    baseline_threshold = float(model_meta.get("baseline_threshold", 0.0))
    seed = int(args.seed if args.seed is not None else model_meta.get("seed", 21301))

    assets, summary = discover_hole_assets(args.root, inspect_images=False)
    split = build_dataset_split(assets, holdout_backgrounds=holdout_backgrounds, seed=seed)
    groups = {
        "validation": split.validation,
        "synthetic_test": split.test,
        "novel_background_holdout": split.background_holdout,
        "real_holdout": split.real_holdout,
    }
    evaluations = {}
    for i, (name, group) in enumerate(groups.items()):
        if not group:
            continue
        rows = evaluate_assets(
            model,
            group,
            sampling=sampling,
            seed=evaluation_seed(seed, name),
            positives_per_image=2 if name == "real_holdout" else 1,
            negatives_per_image=4 if name == "real_holdout" else 2,
            augment=False,
        )
        evaluations[name] = summarize_evaluation(rows, ai_threshold=ai_threshold, baseline_threshold=baseline_threshold)

    stress_assets = tuple(split.test[: max(1, min(len(split.test), 1000))]) or tuple(split.validation[:1000])
    if stress_assets:
        stress_rows = evaluate_assets(
            model,
            stress_assets,
            sampling=sampling,
            seed=evaluation_seed(seed, "off_center_stress"),
            positives_per_image=2,
            negatives_per_image=2,
            positive_jitter_px=min(sampling.negative_min_px - 2.0, sampling.positive_jitter_px + 4.0),
            augment=False,
        )
        evaluations["off_center_stress"] = summarize_evaluation(
            stress_rows, ai_threshold=ai_threshold, baseline_threshold=baseline_threshold
        )

    payload = {
        "schema_version": "2.13",
        "model": str(args.model),
        "thresholds": {"hole_ai": ai_threshold, "center_contrast_baseline": baseline_threshold},
        "archive": summary.to_dict(),
        "split": split.to_dict(),
        "evaluations": evaluations,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("V2.13 RE-EVALUATION")
    print("===================")
    for name, result in evaluations.items():
        ai = result["ai"]
        print(f"{name:28s} AUC={ai.get('auc')} F1={ai.get('f1')} recall={ai.get('recall')} specificity={ai.get('specificity')}")
    print(f"Report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
