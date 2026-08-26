from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.engine.ai.hole_patch_ai_v214 import HolePatchAIV214
from src.engine.offline.hole_dataset_v213 import build_dataset_split, discover_hole_assets
from src.engine.offline.hole_training_v213 import evaluation_seed, summarize_evaluation
from src.engine.offline.hole_training_v214 import (
    DomainRandomizationConfigV214,
    SamplingConfigV214,
    evaluate_assets_v214,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Re-evaluate a trained V2.14 model on clean, novel, real and stress holdouts.")
    p.add_argument("--root", type=Path, default=Path("content/ai/holes"))
    p.add_argument("--model", type=Path, default=Path("content/ai/reports/v214/hole_patch_ai_v214.npz"))
    p.add_argument("--out", type=Path, default=Path("content/ai/reports/v214/hole_v214_reevaluate.json"))
    return p


def main() -> int:
    args = parser().parse_args()
    model, metadata = HolePatchAIV214.load(args.model)
    meta = dict(metadata.get("metadata") or {})
    sampling = SamplingConfigV214(**dict(meta.get("sampling") or {}))
    domain = DomainRandomizationConfigV214(**dict(meta.get("domain_randomization") or {}))
    holdout = tuple(meta.get("holdout_backgrounds") or ("black", "checker", "gray", "bubbles"))
    seed = int(meta.get("seed", 21401))
    ai_threshold = float(meta.get("ai_threshold", 0.5))
    baseline_threshold = float(meta.get("baseline_threshold", 0.0))

    assets, summary = discover_hole_assets(args.root, inspect_images=False)
    split = build_dataset_split(assets, holdout_backgrounds=holdout, seed=seed)
    groups = {
        "validation": split.validation,
        "synthetic_test": split.test,
        "novel_background_holdout": split.background_holdout,
        "real_holdout": split.real_holdout,
    }
    evaluations = {}
    for name, group in groups.items():
        if not group:
            continue
        rows = evaluate_assets_v214(
            model, group, sampling=sampling, domain=domain,
            seed=evaluation_seed(seed, name),
            positives_per_image=2 if name == "real_holdout" else 1,
            negatives_per_image=4 if name == "real_holdout" else 2,
        )
        evaluations[name] = summarize_evaluation(rows, ai_threshold=ai_threshold, baseline_threshold=baseline_threshold)

    stress_assets = tuple(split.test[: max(1, min(len(split.test), 1200))]) or tuple(split.validation[:1200])
    if stress_assets:
        off = evaluate_assets_v214(
            model, stress_assets, sampling=sampling, domain=domain,
            seed=evaluation_seed(seed, "off_center_stress"), positives_per_image=2, negatives_per_image=2,
            positive_jitter_px=min(sampling.negative_min_px - 2.0, sampling.positive_jitter_px + 5.0),
        )
        evaluations["off_center_stress"] = summarize_evaluation(off, ai_threshold=ai_threshold, baseline_threshold=baseline_threshold)
        procedural = evaluate_assets_v214(
            model, stress_assets, sampling=sampling, domain=DomainRandomizationConfigV214.from_profile("strong"),
            seed=seed + 19000, positives_per_image=2, negatives_per_image=2, domain_stress=True,
        )
        evaluations["procedural_domain_stress"] = summarize_evaluation(
            procedural, ai_threshold=ai_threshold, baseline_threshold=baseline_threshold
        )

    payload = {
        "schema_version": "2.14",
        "model": str(args.model),
        "archive": summary.to_dict(),
        "split": split.to_dict(),
        "evaluations": evaluations,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("V2.14 RE-EVALUATION")
    print("===================")
    for name, result in evaluations.items():
        ai = result.get("ai") or {}
        print(f"{name:28s} AUC={ai.get('auc')} F1={ai.get('f1')} recall={ai.get('recall')}")
    print(f"Report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
