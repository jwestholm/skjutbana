"""Export self-contained physical shot traces to the evaluation trace schema."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.engine.offline.evaluation import encoded


def export(root: Path, output: Path) -> None:
    shots = []
    for path in sorted((root / "shots").glob("shot_*/trace.json")):
        trace = json.loads(path.read_text(encoding="utf-8"))
        gt_path = path.parent / "ground_truth.json"
        gt = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.exists() else None
        stages = trace.get("stages", [])
        candidates = stages[-1].get("candidates", []) if stages else []
        emitted = trace.get("outcome", {}).get("final_camera_xy")
        if isinstance(emitted, dict) and "camera_x" in emitted:
            emitted = [{"camera_x": emitted["camera_x"], "camera_y": emitted["camera_y"]}]
        else:
            emitted = [] if trace.get("outcome", {}).get("emitted") else None
        shots.append({"session_id": trace.get("session_id", root.name), "shot_id": str(trace["shot_id"]), "source_kind": "physical_trace", "coordinate_space": "camera", "ground_truth": None if gt is None else {"camera_x": gt["camera_x"], "camera_y": gt["camera_y"]}, "raw": None, "filtered": None, "retained": candidates, "confirmed": None, "selected": emitted, "emitted": emitted, "ranked": candidates, "rescue_used": trace.get("outcome", {}).get("rescue_used"), "latency_ms": trace.get("outcome", {}).get("latency_ms")})
    if not shots:
        raise ValueError(f"No traces found in {root}")
    first_path = next((root / "shots").glob("shot_*/trace.json"))
    first = json.loads(first_path.read_text(encoding="utf-8"))
    runtime = dict(first.get("provenance", {}))
    runtime.setdefault("effective_detector_settings", None)
    runtime.setdefault("models", [])
    runtime.setdefault("calibration", {"status": "unavailable"})
    payload = {"schema_version": "1.0", "mode": "live_path_replay", "producer_runtime": runtime, "limitations": ["Stage fields not populated by producer are unavailable; export does not infer them."], "validation_evidence": {"producer": "physical_trace_capture", "complete_runtime_trace": False}, "shots": shots}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded(payload), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new path")
    export(args.root, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
