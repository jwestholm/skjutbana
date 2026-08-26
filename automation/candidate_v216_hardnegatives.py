from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from src.engine.ai.hole_patch_ensemble_v215 import HolePatchEnsembleV215
from src.engine.offline.candidate_pack_v216 import CandidatePackV216, DEFAULT_DATA_ROOT, discover_candidate_packs
from src.engine.offline.candidate_shadow_analysis_v216 import (
    DEFAULT_ENSEMBLE_CONFIG,
    hard_negative_rows_v216,
    score_pack_v216,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine real detector hard-negative candidate patches from V2.16 captures")
    parser.add_argument("--root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--ensemble", default=str(DEFAULT_ENSEMBLE_CONFIG))
    parser.add_argument("--out", default="content/ai/reports/v216/hard_negatives")
    parser.add_argument("--min-distance", type=float, default=55.0)
    parser.add_argument("--max-per-shot", type=int, default=8)
    parser.add_argument("--max-shots", type=int, default=None)
    parser.add_argument("--export-images", action="store_true")
    args = parser.parse_args()

    paths = discover_candidate_packs(Path(args.root))
    if args.max_shots is not None:
        paths = paths[: max(0, int(args.max_shots))]
    if not paths:
        raise SystemExit("No V2.16 candidate packs found.")
    ensemble = HolePatchEnsembleV215.load(Path(args.ensemble))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "hard_negatives.jsonl"
    written = 0
    with manifest.open("w", encoding="utf-8") as handle:
        for path in paths:
            pack = CandidatePackV216.load(path)
            shot = score_pack_v216(pack, ensemble)
            rows = hard_negative_rows_v216(shot, min_distance_px=args.min_distance, max_per_shot=args.max_per_shot)
            for local_index, row in enumerate(rows):
                capture_index = int(row["capture_index"])
                image_file = None
                if args.export_images and pack.post_patches.ndim == 4 and pack.post_patches.shape[1] > 0:
                    image_file = f"{shot.session_id}_shot{shot.round_id:06d}_neg{local_index:02d}.png"
                    cv2.imwrite(str(out / image_file), pack.post_patches[capture_index, -1])
                payload = {
                    "schema_version": "2.16",
                    "label": 0,
                    "kind": "real_detector_hard_negative",
                    "source_pack": str(path),
                    "session_id": shot.session_id,
                    "round_id": shot.round_id,
                    "capture_index": capture_index,
                    "camera_x": row.get("camera_x"),
                    "camera_y": row.get("camera_y"),
                    "distance_gt_px": row.get("distance_gt_px"),
                    "hardness_v216": row.get("hardness_v216"),
                    "evidence_v216": row.get("evidence_v216"),
                    "image_file": image_file,
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                written += 1
    print("V2.16 HARD-NEGATIVE MINING")
    print("==========================")
    print(f"Shot packs       : {len(paths)}")
    print(f"Negatives written: {written}")
    print(f"Manifest         : {manifest}")
    print(f"PNG export       : {bool(args.export_images)}")
    print("These are training assets for a FUTURE Hole-AI retrain; V2.16 itself stays shadow-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
