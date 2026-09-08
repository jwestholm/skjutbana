"""Frozen physical selection audit and one shadow-only selector replay."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    return math.hypot(float(a["camera_x"]) - float(b["camera_x"]), float(a["camera_y"]) - float(b["camera_y"]))


def _candidate_features(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    keys = (
        "camera_x", "camera_y", "score", "timestamp", "provenance", "source", "area", "radius",
        "circularity", "center_darkening", "change_value", "pre_shot_change", "local_contrast_gain",
        "blackhat_value", "v2222_novelty_factor", "v2222_ridge_fresh_preserved", "v2222_stale_known_removed",
        "v26_vault_age_s", "v26_vault_carried", "v26_vault_hits", "v26_vault_seen_frames",
        "candidate_bank_confirmed", "candidate_bank_hits", "candidate_bank_streak", "v2_bank_confirmed",
        "v2_bank_hits", "v2_bank_streak", "detector_v1", "detector_v2", "v2225_local_confirm",
        "v2225_confirm_center_abs", "v2225_confirm_ring_abs", "v2225_confirm_compact",
        "v2225_confirm_peak_abs", "v2225_confirm_darkening", "v2225_confirm_best_dx",
        "v2225_confirm_best_dy", "v2226_same_frame_support", "v2226_frame_unique_observation",
        "v2226_support_score_max", "v253_confirm_score", "v253_authority_ok", "v253_distance_novelty",
        "v253_history_distance_px", "v252_fresh_physical", "v252_confirm_score", "v252_physical_score",
        "v251_confirm_score", "v251_region_evidence", "v251_region_group", "v251_region_group_size",
        "v251_density", "v251_source_diversity", "analysis_crop_x0", "analysis_crop_y0",
        "analysis_geometry_v2221", "v2225_fast_extract", "v2225_rescue_blob", "v2225_rescue_saliency",
        "v2225_rescue_temporal",
    )
    return {key: candidate.get(key) for key in keys if key in candidate}


def _track_features(track: dict[str, Any] | None, peak_ts: float, gt: dict[str, Any]) -> dict[str, Any] | None:
    if track is None:
        return None
    row = dict(track)
    row["onset_dt_s"] = float(track.get("first_seen_ts", peak_ts)) - float(peak_ts)
    row["last_seen_age_s"] = float(track.get("last_seen_ts", peak_ts)) - float(peak_ts)
    row["distance_to_gt_px"] = distance(track, gt)
    row["last_candidate"] = _candidate_features(track.get("last_candidate"))
    return row


def _tracks(trace: dict[str, Any]) -> dict[int, dict[str, Any]]:
    # Keep the latest occurrence of each debug track. This is a trace observation,
    # not an attempt to reconstruct tracks omitted from the producer's top-eight view.
    result: dict[int, dict[str, Any]] = {}
    for stage in trace.get("stages", []):
        for track in stage.get("tracks", []) or []:
            if isinstance(track, dict) and track.get("track_id") is not None:
                result[int(track["track_id"])] = track
    return result


def _local_candidates(trace: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for stage in trace.get("stages", []):
        confirmation = stage.get("local_confirmation")
        if isinstance(confirmation, dict):
            rows.extend(c for c in confirmation.get("candidates", []) or [] if isinstance(c, dict))
    return rows


def _shadow_score(candidate: dict[str, Any]) -> float:
    """One replay hypothesis: confirmation evidence first, score as tie-break.

    This reuses physical PRE->POST evidence already produced by V2.22.5. It is
    deliberately a replay function and is never installed in HitScanner.
    """
    return (
        float(candidate.get("v2225_confirm_center_abs", 0.0) or 0.0)
        + float(candidate.get("v2225_confirm_darkening", 0.0) or 0.0)
        - float(candidate.get("v2225_confirm_compact", 0.0) or 0.0)
    )


def audit(comparison_path: Path, output_path: Path) -> dict[str, Any]:
    comparison = json.loads(comparison_path.read_text())
    root = Path(comparison["session"])
    rows = []
    for summary in comparison["shots"]:
        sid = int(summary["shot_id"])
        trace_path = root / "shots" / f"shot_{sid:08d}" / "trace.json"
        trace = json.loads(trace_path.read_text())
        gt = summary["ground_truth"]
        pool = trace.get("decision_input", {}).get("retained_candidates") or []
        gt_candidate = min(pool, key=lambda c: distance(c, gt), default=None)
        selected_xy = summary.get("emitted")
        selected_candidate = min(pool, key=lambda c: distance(c, selected_xy), default=None) if selected_xy else None
        tracks = _tracks(trace)
        selected_track = min(tracks.values(), key=lambda t: distance(t, selected_xy), default=None) if selected_xy and tracks else None
        gt_track = min(tracks.values(), key=lambda t: distance(t, gt), default=None) if tracks else None
        challenger = summary.get("challenger") or {}
        ai_order = challenger.get("order") or []
        ai_rows = challenger.get("candidates") or []
        ai_row = ai_rows[ai_order[0]] if ai_order and ai_order[0] < len(ai_rows) else None
        local = _local_candidates(trace)
        replay_candidate = max(local, key=lambda c: (_shadow_score(c), float(c.get("score", 0.0) or 0.0)), default=None)
        replay_distance = distance(replay_candidate, gt) if replay_candidate else None
        rows.append({
            "shot_id": sid,
            "peak_ts": trace.get("peak_ts"),
            "ground_truth": {**gt, "candidate_distance_px": distance(gt_candidate, gt) if gt_candidate else None,
                             "retained_position": (pool.index(gt_candidate) + 1 if gt_candidate in pool else None),
                             "candidate": _candidate_features(gt_candidate),
                             "track_nearest_to_gt": _track_features(gt_track, trace.get("peak_ts", 0.0), gt)},
            "deterministic": {"emitted": summary.get("emitted"), "distance_px": summary.get("emitted_distance_px"),
                              "retained_position": (pool.index(selected_candidate) + 1 if selected_candidate in pool else None),
                              "candidate": _candidate_features(selected_candidate),
                              "track": _track_features(selected_track, trace.get("peak_ts", 0.0), gt),
                              "winner_rule": "global HitScanner._best_track_for_event: (abs(onset_dt), -best_score)"},
            "ai_challenger": {"distance_px": summary.get("challenger_distance_px"), "positive_ranks": summary.get("challenger_positive_ranks"),
                              "selected": ai_row, "confidence": challenger.get("confidence"), "status": challenger.get("status")},
            "shadow_replay": {"hypothesis": "confirmation_evidence_first_center_plus_darkening_minus_compact_then_score",
                              "candidate": _candidate_features(replay_candidate), "distance_px": replay_distance,
                              "local_confirmation_candidate_count": len(local), "status": "PHYSICAL_REPLAY_CHALLENGER"},
            "all_visible_tracks": [_track_features(track, trace.get("peak_ts", 0.0), gt) for track in tracks.values()],
            "trace_completeness": trace.get("completeness"),
            "context": trace.get("context"),
        })
    result = {"schema": "selection-audit-1", "comparison": str(comparison_path), "session": str(root),
              "selection_code": {"base": "src/engine/camera/hit_scanner.py:485-506", "emit": "src/engine/camera/hit_scanner.py:508-545",
                                  "track_order": "src/engine/shot_track_v2226.py:169-243", "context_selectors":
                                      ["src/engine/shot_region_proposal_v251.py:805-842", "src/engine/shot_region_freshness_v252.py:579-656", "src/engine/shot_cross_thread_novelty_v253.py:604-678"]},
              "limitations": ["RAW is unavailable in this producer trace.", "Track list is the saved top-eight debug view, not all active tracks.",
                              "Replay uses saved local-confirmation candidates and is not live-path-equivalent."], "shots": rows}
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.comparison, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
