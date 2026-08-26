from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.engine.ai.hole_patch_ai_v214 import HolePatchAIV214
from src.engine.offline.hole_dataset_v213 import (
    crop_candidate,
    discover_hole_assets,
    read_gray,
    sample_candidate_center,
)
from src.engine.offline.hole_training_v214 import (
    DomainRandomizationConfigV214,
    SamplingConfigV214,
    domain_randomize_patch,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualise V2.14 candidate jitter, background remix and background-invariant feature maps.")
    p.add_argument("--root", type=Path, default=Path("content/ai/holes"))
    p.add_argument("--model", type=Path, default=Path("content/ai/reports/v214/hole_patch_ai_v214.npz"))
    p.add_argument("--kind", choices=("synthetic", "real"), default="synthetic")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--seed", type=int, default=21401)
    p.add_argument("--profile", choices=("mild", "standard", "strong"), default="standard")
    p.add_argument("--out-dir", type=Path, default=Path("content/ai/reports/v214/visualize"))
    return p


def _u8_feature(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi - lo < 1e-9:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def main() -> int:
    args = parser().parse_args()
    model, metadata = HolePatchAIV214.load(args.model)
    meta = dict(metadata.get("metadata") or {})
    sampling = SamplingConfigV214(**dict(meta.get("sampling") or {}))
    domain = DomainRandomizationConfigV214.from_profile(args.profile)
    assets, _ = discover_hole_assets(args.root, inspect_images=False)
    selected = [asset for asset in assets if asset.kind == args.kind]
    if not selected:
        print(f"ERROR: no {args.kind} assets found")
        return 2
    asset = selected[args.index % len(selected)]
    image = read_gray(asset)
    rng = np.random.default_rng(args.seed)
    out_dir = args.out_dir / asset.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate, offset = sample_candidate_center(
        image,
        rng=rng,
        label=1,
        crop_size=model.config.crop_size,
        positive_jitter_px=sampling.positive_jitter_px,
        negative_min_px=sampling.negative_min_px,
        negative_max_px=sampling.negative_max_px,
    )
    clean = crop_candidate(image, candidate, model.config.crop_size)
    cv2.imwrite(str(out_dir / "00_clean_candidate.png"), clean)

    variants = [clean]
    for i in range(8):
        augmented, _ = domain_randomize_patch(
            clean,
            offset,
            rng=rng,
            config=domain,
            sampling=sampling,
            enabled=True,
            force=True,
        )
        variants.append(augmented)
        cv2.imwrite(str(out_dir / f"1{i+1:02d}_domain_variant.png"), augmented)

    probs, offsets = model.predict_patches(variants)
    maps = model.feature_maps_from_patch(variants[1])
    names = ("local_residual", "dog", "blackhat", "gradient")
    for name, feature in zip(names, maps):
        enlarged = cv2.resize(_u8_feature(feature), (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_dir / f"30_feature_{name}.png"), enlarged)

    print("V2.14 BACKGROUND GENERALISATION VISUALISATION")
    print("============================================")
    print(f"Asset      : {asset.image_path.name} ({asset.background_mode})")
    print(f"Candidate  : ({candidate[0]:.1f}, {candidate[1]:.1f})")
    print(f"GT offset  : ({offset[0]:+.1f}, {offset[1]:+.1f})")
    print(f"Profile    : {args.profile}")
    for i, (prob, pred) in enumerate(zip(probs, offsets)):
        label = "clean" if i == 0 else f"domain_{i}"
        print(f"{label:10s} p={float(prob):.3f} predicted_offset=({float(pred[0]):+.1f},{float(pred[1]):+.1f})")
    print(f"Output     : {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
