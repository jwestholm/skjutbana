"""Decompose detector scores in a labelled physical session (read-only)."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

COEFFICIENTS = {
    "saliency": 0.16,
    "center_darkening": 0.23,
    "local_contrast_gain": 0.15,
    "blackhat_value": 0.11,
    "zscore_clipped": 0.06,
}
EXPECTED_SHADOW_HASH = "123a2e510f545895adbee1def7c1a29e17e860af8bad050cfad28cfee94c1d9f"


def distance(a, b):
    if not isinstance(a, dict) or not isinstance(b, dict):
        return None
    try:
        return math.hypot(float(a["camera_x"]) - float(b["camera_x"]), float(a["camera_y"]) - float(b["camera_y"]))
    except (KeyError, TypeError, ValueError):
        return None


def source(candidate):
    if not isinstance(candidate, dict):
        return "UNKNOWN"
    if candidate.get("v2225_fast_extract"):
        return "FAST_V2225"
    if candidate.get("v2_rescue_temporal") or candidate.get("v2_rescue_blob"):
        return "V2_RESCUE"
    if candidate.get("v26_vault_carried") or candidate.get("v26_vault_hits") or candidate.get("v26_vault_seen_frames"):
        return "V26_VAULT"
    if candidate.get("candidate_bank_confirmed") or candidate.get("v2_bank_confirmed"):
        return "V2_BANK"
    if candidate.get("detector_v1") and not candidate.get("detector_v2"):
        return "V1"
    if candidate.get("v2_primary_peak"):
        return "V2_PRIMARY"
    return "V2_OTHER"


def decomposition(candidate):
    c = candidate if isinstance(candidate, dict) else {}
    values = {
        "saliency": float(c.get("v2_saliency", 0.0) or 0.0),
        "center_darkening": float(c.get("center_darkening", 0.0) or 0.0),
        "local_contrast_gain": float(c.get("local_contrast_gain", 0.0) or 0.0),
        "blackhat_value": float(c.get("blackhat_value", 0.0) or 0.0),
        "zscore_clipped": min(float(c.get("v2_zscore", 0.0) or 0.0), 25.0),
    }
    contributions = {name: COEFFICIENTS[name] * values[name] for name in values}
    raw = sum(contributions.values())
    clipped = min(35.0, max(3.6, raw))
    final = float(c.get("score", 0.0) or 0.0)
    return {"values": values, "contributions": contributions, "raw_score": raw,
            "clipped_base_score": clipped, "recorded_candidate_score": final,
            "residual_after_clip": final - clipped,
            "source": source(c), "penalty_factor_inferred": final / clipped if clipped else None}


def _nearest(pool, gt):
    pairs = [(distance(c, gt), i, c) for i, c in enumerate(pool or [])]
    pairs = [p for p in pairs if p[0] is not None]
    return min(pairs, default=(None, None, None))


def _candidate_by_xy(pool, selected):
    if not isinstance(selected, dict):
        return None
    matches = [(distance(c, selected), c) for c in pool or []]
    matches = [p for p in matches if p[0] is not None]
    return min(matches, default=(None, None))[1]


def audit(root: Path, comparison: Path):
    comp = json.loads(comparison.read_text())
    rows = []
    for row in comp.get("shots", []):
        sid = int(row["shot_id"])
        trace_path = root / "shots" / f"shot_{sid:08d}" / "trace.json"
        trace = json.loads(trace_path.read_text())
        gt = row.get("ground_truth")
        pool = trace.get("decision_input", {}).get("retained_candidates") or []
        nearest_dist, nearest_index, nearest = _nearest(pool, gt)
        selectors = row.get("selectors", {})
        selected = {name: (selectors.get(name) or {}).get("selected") for name in
                    ("CURRENT_DETERMINISTIC", "CONFIRMATION_SELECTION_SHADOW", "CANONICAL_AI_SHADOW")}
        candidates = {"nearest_physical_gt": nearest}
        for name, value in selected.items():
            candidates[name] = _candidate_by_xy(pool, value) or value
        entries = {}
        tracks = []
        for stage in trace.get("stages", []):
            if isinstance(stage.get("tracks"), list): tracks.extend(c for c in stage["tracks"] if isinstance(c, dict))
        for name, candidate in candidates.items():
            d = distance(candidate, gt)
            matching = [track for track in tracks if distance(track, candidate) is not None and distance(track, candidate) <= 2.0]
            entries[name] = {"distance_px": d, "retained_position": (next((i + 1 for i, c in enumerate(pool) if c is candidate), None)),
                             "candidate": candidate, "decomposition": decomposition(candidate),
                             "track_evidence": [{"track_id": t.get("track_id"), "best_score": t.get("best_score"),
                                                 "hits": t.get("hits"), "first_seen_ts": t.get("first_seen_ts"),
                                                 "last_seen_ts": t.get("last_seen_ts"), "missed_frames": t.get("missed_frames"),
                                                 "unique_frame_hits": t.get("v2226_unique_frame_hits"),
                                                 "same_frame_support": t.get("v2226_same_frame_support"),
                                                 "state": t.get("state")} for t in matching[-5:]]}
        rows.append({"shot_id": sid, "ground_truth": gt, "nearest_retained_distance_px": nearest_dist,
                     "nearest_retained_position": None if nearest_index is None else nearest_index + 1,
                     "candidates": entries, "trace_sha256": row.get("trace_sha256"),
                     "audio_peak_ts": trace.get("peak_ts")})
    distributions = defaultdict(list)
    physical_positive_sources = Counter()
    for row in rows:
        pool = json.loads((root / "shots" / f"shot_{int(row['shot_id']):08d}" / "trace.json").read_text()).get("decision_input", {}).get("retained_candidates") or []
        for candidate in pool:
            distributions[source(candidate)].append(float(candidate.get("score", 0.0) or 0.0))
        nearest_entry = row["candidates"].get("nearest_physical_gt", {})
        nearest_candidate = nearest_entry.get("candidate")
        if nearest_candidate: physical_positive_sources[source(nearest_candidate)] += 1
    def quantiles(values):
        values = sorted(values)
        if not values: return {"count": 0, "median": None, "p90": None, "p99": None, "max": None}
        def q(p): return values[min(len(values) - 1, max(0, math.ceil(p * len(values)) - 1))]
        return {"count": len(values), "median": q(.5), "p90": q(.9), "p99": q(.99), "max": values[-1]}
    distributions_out = {name: {**quantiles(values), "physical_positive_candidates": physical_positive_sources.get(name, 0)} for name, values in sorted(distributions.items())}
    return {"schema": "physical-score-audit-1", "session": str(root), "comparison": str(comparison),
            "score_formula": {"coefficients": COEFFICIENTS, "clip": [3.6, 35.0],
                               "source": "CandidateGeneratorV2._candidate_features"},
            "frozen_shadow_hash_expected": EXPECTED_SHADOW_HASH,
            "shots": rows, "candidate_source_score_distributions": distributions_out,
            "limitations": ["Track best_score may exceed the candidate score through persistence/support aggregation.",
                            "Historical traces do not preserve every raw generator intermediate; source classes are flag-based.",
                            "No ground-truth coordinates are used by decomposition or selection."]}


def write_outputs(report, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    (output / "score_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "candidates.csv").open("w", newline="") as stream:
        fields = ["shot_id", "role", "distance_px", "retained_position", "source", "score", "raw_score", "clipped_base_score", "residual_after_clip", "v2_saliency", "center_darkening", "local_contrast_gain", "blackhat_value", "v2_zscore", "candidate_bank_hits", "v26_vault_hits", "v2226_unique_frame_hits"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for shot in report["shots"]:
            for role, entry in shot["candidates"].items():
                c = entry.get("candidate") or {}; d = entry["decomposition"]
                writer.writerow({"shot_id": shot["shot_id"], "role": role, "distance_px": entry["distance_px"],
                    "retained_position": entry["retained_position"], "source": d["source"], "score": c.get("score"),
                    "raw_score": d["raw_score"], "clipped_base_score": d["clipped_base_score"], "residual_after_clip": d["residual_after_clip"],
                    "v2_saliency": c.get("v2_saliency"), "center_darkening": c.get("center_darkening"), "local_contrast_gain": c.get("local_contrast_gain"), "blackhat_value": c.get("blackhat_value"), "v2_zscore": c.get("v2_zscore"), "candidate_bank_hits": c.get("candidate_bank_hits"), "v26_vault_hits": c.get("v26_vault_hits"), "v2226_unique_frame_hits": c.get("v2226_unique_frame_hits")})
    lines = ["# Physical detector score audit", "", f"Session: `{report['session']}`", "", "## Score path", "", "`0.16*v2_saliency + 0.23*center_darkening + 0.15*local_contrast_gain + 0.11*blackhat_value + 0.06*min(v2_zscore,25)`, clipped to `[3.6,35.0]`. Known-hole penalties multiply the clipped value. Track `best_score` can then aggregate support/history.", "", "## Candidate source distributions", "", "| Source | Count | Median | P90 | P99 | Max |", "|---|---:|---:|---:|---:|---:|"]
    for name, stats in report["candidate_source_score_distributions"].items(): lines.append(f"| {name} | {stats['count']} | {stats['median']} | {stats['p90']} | {stats['p99']} | {stats['max']} |")
    lines += ["", "## Per-shot comparison", "", "| Shot | GT nearest px/pos | Deterministic px | Shadow px | AI px |", "|---:|---:|---:|---:|---:|"]
    for shot in report["shots"]:
        e=shot["candidates"]; lines.append(f"| {shot['shot_id']} | {shot['nearest_retained_distance_px']:.2f} / {shot['nearest_retained_position']} | {e['CURRENT_DETERMINISTIC']['distance_px']:.2f} | {e['CONFIRMATION_SELECTION_SHADOW']['distance_px']:.2f} | {e['CANONICAL_AI_SHADOW']['distance_px']:.2f} |")
    lines += ["", "The JSON and CSV retain all available provenance and score components. Visual image generation is not attempted because these traces do not guarantee a complete, synchronized frame-to-candidate crop mapping."]
    (output / "score_audit.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--comparison", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); write_outputs(audit(args.root, args.comparison), args.output); print(args.output)


if __name__ == "__main__": main()
