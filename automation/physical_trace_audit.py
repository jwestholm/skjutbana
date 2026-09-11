"""Create a read-only coordinate audit and target-space diagnostic images."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from automation.physical_trace_target import camera_to_target, homography_from_calibration, load_calibration, target_evidence_view


def audit(root: Path, evaluation_trace: Path, output: Path, calibration: Path) -> dict:
    payload = json.loads(evaluation_trace.read_text(encoding="utf-8"))
    h = homography_from_calibration(load_calibration(calibration))
    if h is None:
        raise ValueError("No usable calibration homography")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for shot in payload["shots"]:
        gt = shot.get("ground_truth")
        if not gt:
            continue
        xy = (float(gt["camera_x"]), float(gt["camera_y"]))
        retained = shot.get("retained") or []
        nearest = min(retained, key=lambda p: math.hypot(float(p["camera_x"])-xy[0], float(p["camera_y"])-xy[1])) if retained else None
        selected = (shot.get("selected") or [None])[0]
        def dist(value):
            return None if not value else math.hypot(float(value["camera_x"])-xy[0], float(value["camera_y"])-xy[1])
        rows.append({"shot_id": shot["shot_id"], "ground_truth": {"x": xy[0], "y": xy[1], "space": "original_camera"},
                     "selected": None if selected is None else {"x": selected["camera_x"], "y": selected["camera_y"], "space": "original_camera"},
                     "emitted": None if not shot.get("emitted") else {"x": shot["emitted"][0]["camera_x"], "y": shot["emitted"][0]["camera_y"], "space": "original_camera"},
                     "nearest_retained": None if nearest is None else {"x": nearest["camera_x"], "y": nearest["camera_y"], "space": "original_camera"},
                     "distance_gt_selected_px": dist(selected), "distance_gt_emitted_px": dist((shot.get("emitted") or [None])[0]),
                     "distance_gt_nearest_retained_px": dist(nearest)})
        trace = json.loads((root / "shots" / f"shot_{int(shot['shot_id']):08d}" / "trace.json").read_text(encoding="utf-8"))
        directory = root / "shots" / f"shot_{int(shot['shot_id']):08d}"
        posts = [f for f in trace.get("frames", []) if f.get("kind") == "post"]
        pres = [f for f in trace.get("frames", []) if f.get("kind") in {"pre_history", "pre_snapshot"}]
        if not posts or not pres:
            continue
        post, pre = np.load(directory / posts[-1]["path"], allow_pickle=False), np.load(directory / pres[-1]["path"], allow_pickle=False)
        _, image = target_evidence_view(pre, post, h, (960, 720))
        markers = [(xy, (0, 0, 255), "GT approximate"),
                   (None if selected is None else (selected["camera_x"], selected["camera_y"]), (0, 255, 0), "selected/emitted"),
                   (None if nearest is None else (nearest["camera_x"], nearest["camera_y"]), (0, 215, 255), "nearest retained")]
        for point, color, label in markers:
            if point is None:
                continue
            q = tuple(int(round(v)) for v in camera_to_target(point, h))
            cv2.drawMarker(image, q, color, cv2.MARKER_CROSS, 28, 3)
            cv2.putText(image, label, (q[0] + 12, q[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2, cv2.LINE_AA)
        cv2.imwrite(str(output / f"shot_{int(shot['shot_id']):08d}_coordinate_audit.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    report = {"coordinate_space": "original_camera", "target_space": "calibration homography output", "shots": rows}
    (output / "coordinate_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--evaluation-trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--calibration", type=Path, default=Path("content/settings.json"))
    args = p.parse_args()
    if args.output.exists():
        p.error("Output exists; choose a new path")
    report = audit(args.root, args.evaluation_trace, args.output, args.calibration)
    for row in report["shots"]:
        print(row)


if __name__ == "__main__":
    main()
