from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.engine.ai.hole_patch_ai_v213 import HolePatchAI
from src.engine.offline.hole_dataset_v213 import (
    crop_candidate,
    discover_hole_assets,
    image_center_xy,
    read_gray,
    sample_candidate_center,
)
from src.engine.offline.hole_training_v213 import SamplingConfig


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualise exactly what V2.13 Hole-AI sees after candidate jitter (diagnostic only).")
    p.add_argument("--root", type=Path, default=Path("content/ai/holes"))
    p.add_argument("--model", type=Path, default=Path("content/ai/reports/v213/hole_patch_ai_v213.npz"))
    p.add_argument("--kind", choices=("synthetic", "real"), default="synthetic")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--seed", type=int, default=21301)
    p.add_argument("--out-dir", type=Path, default=Path("content/ai/reports/v213/visualize"))
    return p


def _to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray.copy()


def main() -> int:
    args = parser().parse_args()
    model, metadata = HolePatchAI.load(args.model)
    model_meta = dict(metadata.get("metadata") or {})
    sampling = SamplingConfig(**dict(model_meta.get("sampling") or {}))
    assets, _ = discover_hole_assets(args.root, inspect_images=False)
    selected = [asset for asset in assets if asset.kind == args.kind]
    if not selected:
        print(f"ERROR: no {args.kind} assets found")
        return 2
    asset = selected[args.index % len(selected)]
    image = read_gray(asset)
    gt = image_center_xy(image)
    rng = np.random.default_rng(args.seed)

    cases: list[tuple[str, int, tuple[float, float], tuple[float, float], np.ndarray]] = []
    for label, name, count, pos_jitter in (
        (1, "positive", 3, sampling.positive_jitter_px),
        (1, "stress_positive", 2, min(sampling.negative_min_px - 2.0, sampling.positive_jitter_px + 4.0)),
        (0, "negative", 3, sampling.positive_jitter_px),
    ):
        for i in range(count):
            candidate, target_offset = sample_candidate_center(
                image,
                rng=rng,
                label=label,
                crop_size=model.config.crop_size,
                positive_jitter_px=pos_jitter,
                negative_min_px=sampling.negative_min_px,
                negative_max_px=sampling.negative_max_px,
            )
            patch = crop_candidate(image, candidate, model.config.crop_size)
            cases.append((f"{name}_{i+1}", label, candidate, target_offset, patch))

    probabilities, predicted_offsets = model.predict_patches([case[4] for case in cases])
    out_dir = args.out_dir / asset.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    source = _to_bgr(image)
    cv2.drawMarker(source, (int(round(gt[0])), int(round(gt[1]))), (0, 0, 255), cv2.MARKER_CROSS, 13, 1)
    for idx, case in enumerate(cases):
        _, label, candidate, _, _ = case
        color = (0, 220, 0) if label else (0, 180, 255)
        cv2.circle(source, (int(round(candidate[0])), int(round(candidate[1]))), 3, color, 1)
        cv2.putText(source, str(idx + 1), (int(candidate[0]) + 4, int(candidate[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "00_source_and_candidate_centres.png"), source)

    print("V2.13 CANDIDATE-CENTRE VISUALISATION")
    print("===================================")
    print(f"Asset: {asset.image_path.name}  kind={asset.kind} background={asset.background_mode}")
    print("Red cross = source archive hole centre; numbered dots = sampled MODEL candidate centres.")
    for idx, (case, probability, pred_offset) in enumerate(zip(cases, probabilities, predicted_offsets), start=1):
        name, label, candidate, target_offset, patch = case
        canvas = _to_bgr(patch)
        c = model.config.crop_size // 2
        cv2.drawMarker(canvas, (c, c), (255, 255, 0), cv2.MARKER_CROSS, 9, 1)
        target_xy = (int(round(c + target_offset[0])), int(round(c + target_offset[1])))
        pred_xy = (int(round(c + pred_offset[0])), int(round(c + pred_offset[1])))
        if label:
            cv2.circle(canvas, target_xy, 4, (0, 0, 255), 1)
        cv2.circle(canvas, pred_xy, 4, (255, 0, 255), 1)
        cv2.putText(canvas, f"p={float(probability):.3f}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"{idx:02d}_{name}_label{label}.png"), canvas)
        print(
            f"{idx:02d} {name:18s} label={label} p={float(probability):.3f} "
            f"GT offset=({target_offset[0]:+.1f},{target_offset[1]:+.1f}) "
            f"AI offset=({float(pred_offset[0]):+.1f},{float(pred_offset[1]):+.1f})"
        )
    print(f"Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
