"""Export self-contained physical shot traces to the evaluation trace schema."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.engine.offline.evaluation import VERSION, encoded
from src.engine.offline.causal_candidates import analyze_trace


def export(root: Path, output: Path) -> None:
    shots = []
    paths = sorted((root / "shots").glob("shot_*/trace.json"))
    traces = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for index, (path, trace) in enumerate(zip(paths, traces)):
        gt_path = path.parent / "ground_truth.json"
        gt = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.exists() else None
        stages = trace.get("stages", [])
        # A physical trace may contain an explicit pipeline snapshot.  Older
        # traces only contain the last observed candidate list; retain that
        # list as an observed retention pool, but never infer raw/filter/
        # confirmation stages from it.
        pipeline = None
        for stage in reversed(stages):
            value = stage.get("pipeline") if isinstance(stage, dict) else None
            if isinstance(value, dict):
                pipeline = value
                break
        observed_candidates = next((s.get("candidates", []) for s in reversed(stages)
                                    if isinstance(s, dict) and isinstance(s.get("candidates"), list)
                                    and s.get("candidates")), [])
        retained = pipeline.get("retained_candidates") if isinstance(pipeline, dict) else None
        if not isinstance(retained, list):
            retained = observed_candidates
        raw = pipeline.get("raw_candidates") if isinstance(pipeline, dict) else None
        filtered = pipeline.get("filtered_candidates") if isinstance(pipeline, dict) else None
        confirmed = pipeline.get("confirmed_candidates") if isinstance(pipeline, dict) else None
        ranked = pipeline.get("ranked_candidates") if isinstance(pipeline, dict) else None
        raw = raw if isinstance(raw, list) else None
        filtered = filtered if isinstance(filtered, list) else None
        confirmed = confirmed if isinstance(confirmed, list) else None
        if confirmed is None:
            confirmation = next((s.get("local_confirmation") for s in reversed(stages)
                                 if isinstance(s, dict) and isinstance(s.get("local_confirmation"), dict)), None)
            if confirmation is not None:
                confirmed = confirmation.get("candidates")
        ranked = ranked if isinstance(ranked, list) else None
        decision = trace.get("decision_input", {}).get("deterministic_selection")
        selected = ([{"camera_x": decision["camera_x"], "camera_y": decision["camera_y"]}]
                    if isinstance(decision, dict) and "camera_x" in decision else None)
        emitted = trace.get("outcome", {}).get("final_camera_xy")
        if isinstance(emitted, dict) and "camera_x" in emitted:
            emitted = [{"camera_x": emitted["camera_x"], "camera_y": emitted["camera_y"]}]
        else:
            emitted = [] if trace.get("outcome", {}).get("emitted") else None
        ground_truth = None
        if gt is not None:
            ground_truth = {"camera_x": gt["camera_x"], "camera_y": gt["camera_y"]}
            # Legacy physical clicks were explicitly approximate in the
            # session notes but predate the quality field.  Preserve the file
            # unchanged and mark that uncertainty as unknown in the export;
            # the evaluator will not call a miss definitive without a radius.
            ground_truth["quality"] = gt.get("quality", "unknown")
            if "uncertainty_radius_px" in gt:
                ground_truth["uncertainty_radius_px"] = gt["uncertainty_radius_px"]
        outcome = trace.get("outcome", {})
        shots.append({"session_id": trace.get("session_id", root.name), "shot_id": str(trace["shot_id"]), "source_kind": "physical_trace", "coordinate_space": "camera", "ground_truth": ground_truth, "raw": raw, "filtered": filtered, "retained": retained, "confirmed": confirmed, "selected": selected, "emitted": emitted, "ranked": ranked, "rescue_used": outcome.get("rescue_used"), "latency_ms": outcome.get("detector_e2e_latency_ms"), "trace_completion_latency_ms": outcome.get("trace_completion_latency_ms"), "latency_semantics": "detector_e2e_decision_or_emission; null when producer did not capture it", "trace_complete": trace.get("completeness", {}).get("trace_complete"), "trace_completeness": trace.get("completeness")})
        causal = analyze_trace(trace, traces[index + 1]['peak_ts'] if index + 1 < len(traces) else None, gt)
        shots[-1]['causal_candidate_audit_v1'] = {k: v for k, v in causal.items() if k not in ('records', 'observations')}
        shots[-1]['causally_available_candidates_v1'] = [r['candidate'] for r in causal['records'] if r['availability'] == 'CAUSALLY_AVAILABLE']
        complete = trace.get('decision_input', {}).get('complete_track_audit')
        if complete and complete.get('complete'):
            from src.engine.offline.track_replay import current_exact_replay
            chosen = current_exact_replay(complete, trace['decision_input']['deterministic_selection'])
            shots[-1]['current_exact_replay_v1'] = {
                'status': 'MATCH', 'track_id': chosen['track_id'] if chosen else None,
                'complete_pool_count': len(complete['tracks']),
                'eligible_count': sum(t['eligible'] for t in complete['tracks']),
            }
        else:
            shots[-1]['current_exact_replay_v1'] = {'status': 'UNAVAILABLE', 'reason': 'complete snapshot absent'}
        shots[-1]['legacy_stage_semantics'] = 'historical last-observed snapshots; may include post-decision evidence; not causal oracle'
    if not shots:
        raise ValueError(f"No traces found in {root}")
    first_path = next((root / "shots").glob("shot_*/trace.json"))
    first = json.loads(first_path.read_text(encoding="utf-8"))
    runtime = dict(first.get("provenance", {}))
    runtime.setdefault("effective_detector_settings", None)
    runtime.setdefault("models", [])
    runtime.setdefault("calibration", {"status": "unavailable"})
    payload = {"schema_version": VERSION, "mode": "live_path_replay", "producer_runtime": runtime, "limitations": ["Stage fields not populated by producer are unavailable; export does not infer them.", "This is a physical trace observation, not a claim of broad physical accuracy."], "validation_evidence": {"producer": "physical_trace_capture", "complete_runtime_trace": True}, "shots": shots}
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
