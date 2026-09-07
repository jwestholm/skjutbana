"""Read-only diagnosis of missing candidates in a saved native session.

This tool deliberately does not run the detector.  A training record contains
the candidate pool, but not the PRE/POST pixels needed to explain proposal
generation.  Missing evidence is reported as unavailable rather than inferred.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from src.engine.offline.evaluation import DEFAULT_RADII, digest, encoded


def _distance(candidate, gt):
    return math.hypot(float(candidate["camera_x"]) - gt[0], float(candidate["camera_y"]) - gt[1])


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))]


def diagnose(session_root: Path):
    paths = sorted(Path(session_root).glob("shot_*.json"))
    if not paths:
        raise ValueError(f"No shot records found in {session_root}")
    framepack_root = Path("content/ai/training_v223/framepacks")
    framepacks = {p.stem: p for p in framepack_root.glob("*/shot_*.json")}
    rows = []
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        gt = (float(raw["gt_camera_x"]), float(raw["gt_camera_y"]))
        candidates = [c for c in raw.get("candidates", []) if "camera_x" in c and "camera_y" in c]
        nearest = min(candidates, key=lambda c: _distance(c, gt), default=None)
        distance = None if nearest is None else _distance(nearest, gt)
        rows.append({
            "shot_id": str(raw.get("shot_id", path.stem)),
            "path": str(path),
            "source_kind": raw.get("source_kind", "unknown"),
            "ground_truth": {"camera_x": gt[0], "camera_y": gt[1]},
            "candidate_count": len(candidates),
            "nearest_candidate": None if nearest is None else {"camera_x": float(nearest["camera_x"]), "camera_y": float(nearest["camera_y"]), "distance_px": distance},
            "offset": None if nearest is None else {"dx": float(nearest["camera_x"]) - gt[0], "dy": float(nearest["camera_y"]) - gt[1]},
            "oracle": {str(int(r)): bool(distance is not None and distance <= r) for r in DEFAULT_RADII},
            "evidence": {
                "framepack": None if path.stem not in framepacks else str(framepacks[path.stem]),
                "gt_inside_valid_image_roi": "UNAVAILABLE",
                "coordinate_consistent_with_pixels": "UNAVAILABLE",
                "pre_post_signal_near_gt": "UNAVAILABLE",
                "darkening_change_registered_evidence": "UNAVAILABLE",
                "response_peak_before_filtering": "UNAVAILABLE",
                "threshold_rejection": "UNAVAILABLE",
                "candidate_generated_then_suppressed": "UNAVAILABLE",
                "candidate_limit_or_quota": "UNAVAILABLE",
            },
        })
    misses = [r for r in rows if not r["oracle"]["20"]]
    distances = [r["nearest_candidate"]["distance_px"] for r in misses if r["nearest_candidate"] is not None]
    dx = [r["offset"]["dx"] for r in misses if r["offset"] is not None]
    dy = [r["offset"]["dy"] for r in misses if r["offset"] is not None]
    categories = Counter()
    for row in misses:
        # No DEVELOPMENT framepack has the information needed to distinguish
        # generation, filtering, threshold, or temporal causes.
        categories["TEMPORAL_DATA_INSUFFICIENT"] += 1
    return {
        "schema_version": "candidate-root-cause-1",
        "session": str(session_root),
        "total_shots": len(rows),
        "source_breakdown": dict(Counter(r["source_kind"] for r in rows)),
        "framepack_matches": sum(r["evidence"]["framepack"] is not None for r in rows),
        "oracle_coverage": {str(int(r)): sum(x["oracle"][str(int(r))] for x in rows) for r in DEFAULT_RADII},
        "missing_at_20": len(misses),
        "failure_categories": dict(categories),
        "nearest_distance_px": {
            "evaluated": len(distances), "unavailable": len(misses) - len(distances),
            "median": statistics.median(distances) if distances else None,
            "mean": statistics.mean(distances) if distances else None,
            "p95": _percentile(distances, .95), "max": max(distances) if distances else None,
        },
        "offset_px": {
            "evaluated": len(dx), "unavailable": len(misses) - len(dx),
            "dx_mean": statistics.mean(dx) if dx else None, "dy_mean": statistics.mean(dy) if dy else None,
            "dx_median": statistics.median(dx) if dx else None, "dy_median": statistics.median(dy) if dy else None,
        },
        "interpretation": {
            "candidate_generation_testable": False,
            "dominant_root_cause": "F2_CAPTURE_MISMATCH_OR_MISSING_TEMPORAL_DATA",
            "classification_note": "TEMPORAL_DATA_INSUFFICIENT does not mean the live detector has a temporal failure; it means the stored dataset cannot determine the actual failure cause.",
            "reason": "The DEVELOPMENT records have candidate metadata but no matching PRE/POST framepack pixels, ROI/calibration snapshot, intermediate peaks, suppression reasons, or quota telemetry.",
            "optimization_validity": "INSUFFICIENT_FOR_CANDIDATE_GENERATOR_OPTIMIZATION",
            "next_capture": "Record shot-linked PRE frames, every POST frame through confirmation/rescue timeout, canonical calibration/ROI, raw peaks before filtering, rejection reasons, candidate quotas, and final emissions.",
        },
        "representative_failures": [
            {"shot_id": r["shot_id"], "ground_truth": r["ground_truth"], "nearest_candidate": r["nearest_candidate"], "offset": r["offset"]}
            for r in misses[:5]
        ],
        "shots": rows,
        "input_sha256": [{"path": str(p), "sha256": digest(p)} for p in paths],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new diagnostic directory")
    report = diagnose(args.session)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(encoded(report), encoding="utf-8")
    summary = [f"TOTAL DEVELOPMENT SHOTS: {report['total_shots']}", f"ORACLE: {report['oracle_coverage']}",
               f"MISSING @20 PX: {report['missing_at_20']}", f"FAILURE CATEGORIES: {report['failure_categories']}",
               f"NEAREST DISTANCE: {report['nearest_distance_px']}", f"OFFSETS: {report['offset_px']}",
               "STAGE DIAGNOSTICS: UNAVAILABLE (no matching DEVELOPMENT PRE/POST framepacks)",
               "CONCLUSION: F2 capture mismatch / missing temporal data; do not optimize candidate generation from this dataset."]
    (args.output / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
