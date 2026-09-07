"""Read-only evaluation of saved framepack pools or versioned stage traces."""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from src.engine.offline.evaluation import VERSION, encoded, evaluate, provenance, summary


def framepack_observations(root):
    from src.engine.ai.training_v223.framepack import discover_framepacks, load_framepack
    shots, paths = [], []
    discovered = discover_framepacks(root)
    orphaned = set(root.glob("*/shot_*.json")) - set(discovered)
    if orphaned:
        raise ValueError(f"Missing NPZ for {len(orphaned)} framepacks; refusing silent exclusion")
    for path in discovered:
        meta, pre, posts, timestamps = load_framepack(path)
        if len(timestamps) != len(posts):
            raise ValueError(f"Timestamp/frame count mismatch: {path}")
        gt = meta.get("gt_camera_xy")
        shots.append({"session_id": meta["session_id"], "shot_id": meta["shot_id"],
            "coordinate_space": "camera", "source_kind": meta.get("source_kind", "unknown"),
            "ground_truth": None if gt is None else {"camera_x": gt[0], "camera_y": gt[1]},
            "saved_pool": meta.get("current_candidates"),
            "capture": {"path": str(path), "post_frame_count": len(posts),
                        "post_timestamps": timestamps.tolist(), "frame_shape": list(pre.shape),
                        "shot_timestamp": meta.get("shot_timestamp"), "calibration": meta.get("calibration")}})
        paths.extend([path, path.with_suffix(".npz")])
    return shots, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--framepacks", type=Path)
    source.add_argument("--trace", type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing paths are refused")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; preserve baseline and choose a new directory")
    if args.framepacks:
        shots, paths = framepack_observations(args.framepacks)
        mode = "offline_candidate"
        runtime = {"effective_detector_settings": None, "models": None, "calibration": None,
                   "status": "unavailable_in_historical_capture", "detector_executed": False}
        limitations = ["Historical saved candidate pools only; no current detector execution.",
            "F2/projected labels do not establish physical-shot performance.",
            "Raw/filter/retention/confirmation/selection/emission boundaries and ranking are not recorded.",
            "Historical effective settings, model hashes and calibration are unavailable; producer cannot be reproduced."]
    else:
        payload = json.loads(args.trace.read_text())
        if payload.get("schema_version") != VERSION:
            raise ValueError("Unsupported trace schema")
        mode, shots = payload["mode"], payload["shots"]
        runtime = payload["producer_runtime"]
        for key in ("effective_detector_settings", "models", "calibration"):
            if key not in runtime:
                raise ValueError(f"Missing producer runtime field: {key}; use explicit null if unknown")
        paths = [args.trace]
        limitations = payload.get("limitations", [])
        if mode != "offline_candidate":
            if not payload.get("validation_evidence"):
                raise ValueError("Replay/physical mode requires documented validation_evidence")
            limitations = [*limitations, "Mode/evidence are producer declarations, not certification by this scorecard."]
    if not shots:
        raise ValueError("No shots found; refusing an empty baseline")
    report = evaluate(shots, mode=mode, top_k=args.top_k)
    root = Path(__file__).resolve().parents[1]
    report["provenance"] = provenance(root, paths, args.dataset_id, runtime)
    report["provenance"]["evaluated_shots"] = report["total_shots"]
    report["limitations"] = limitations
    report["observations"] = shots
    report["validation_evidence"] = None if args.framepacks else payload.get("validation_evidence")
    text = summary(report) + "\nLimitations:\n" + "\n".join(f"- {x}" for x in limitations) + "\n"
    args.output.mkdir(parents=True, exist_ok=False)
    # Archive the exact source files named by provenance, including untracked work.
    with tarfile.open(args.output / "source.tar.gz", "w:gz") as archive:
        for item in report["provenance"]["source_manifest"]:
            archive.add(root / item["path"], arcname=item["path"], recursive=False)
    (args.output / "report.json").write_text(encoded(report))
    (args.output / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
