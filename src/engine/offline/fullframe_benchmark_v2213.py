from __future__ import annotations

"""V2.21.3 physical full-frame benchmark.

This is a DEVELOPMENT-tuned / protected-evaluation experiment.  A small fixed
profile sweep is scored only on development shots.  The winning profile is
then frozen before confirmation/holdout are reported.

No proposal function receives GT.  GT is used only by this evaluator.
"""

import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .candidate_pack_v216 import CandidatePackV216, discover_candidate_packs
from .direct_proposal_v221 import DirectProposalConfigV221, propose_direct_v221
from .fullframe_benchmark_v2212 import _current_rows, _pack_frames, _union
from .new_hole_training_v217 import _shot_split_keys_v217
from .temporal_local_v2212 import LocalTemporalConfigV2212, propose_local_temporal_v2212
from .temporal_consensus_v2213 import (
    MaskedDirectConfigV2213,
    TemporalConsensusConfigV2213,
    consensus_config_dict_v2213,
    masked_direct_config_dict_v2213,
    propose_masked_direct_v2213,
    propose_temporal_consensus_v2213,
)

RADII = (5.0, 10.0, 20.0, 42.0)


def _dist(candidate: dict[str, Any], gt: tuple[float, float]) -> float:
    return float(math.hypot(float(candidate.get("camera_x", 0.0)) - gt[0], float(candidate.get("camera_y", 0.0)) - gt[1]))


def _nearest(candidates: Sequence[dict[str, Any]], gt: tuple[float, float]) -> float:
    if not candidates:
        return 9999.0
    return min(_dist(row, gt) for row in candidates)


def _hit(candidates: Sequence[dict[str, Any]], gt: tuple[float, float], radius: float) -> bool:
    return any(_dist(row, gt) <= radius for row in candidates)


def _profiles() -> dict[str, TemporalConsensusConfigV2213]:
    base = TemporalConsensusConfigV2213()
    return {
        "tight_core": replace(
            base,
            search_radius_px=48,
            threshold_percentile=90.0,
            components_per_source=6,
            top_per_anchor=2,
        ),
        "balanced": base,
        "recall_core": replace(
            base,
            search_radius_px=60,
            threshold_percentile=88.0,
            components_per_source=8,
            top_per_anchor=3,
            proposal_limit=1600,
        ),
        "blackhat_tophat": replace(
            base,
            search_radius_px=58,
            threshold_percentile=87.0,
            components_per_source=10,
            top_per_anchor=3,
            proposal_limit=1500,
            source_names=("blackhat_gain", "tophat_gain"),
            source_weights={"blackhat_gain": 1.0, "tophat_gain": 0.95},
        ),
        "wide_core": replace(
            base,
            search_radius_px=68,
            threshold_percentile=90.0,
            components_per_source=7,
            top_per_anchor=2,
            proposal_limit=1400,
            distance_prior_weight=0.04,
        ),
    }


def _profile_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    distances = np.asarray([float(r["distance_px"]) for r in rows if float(r["distance_px"]) < 9000], dtype=np.float64)
    result = {
        "shots": len(rows),
        "oracle": {
            str(int(radius)): float(sum(bool(r["recall"][str(int(radius))]) for r in rows) / max(1, len(rows)))
            for radius in RADII
        },
        "mean_candidates": float(np.mean([int(r["candidate_count"]) for r in rows])) if rows else 0.0,
        "median_distance": float(np.median(distances)) if len(distances) else 9999.0,
    }
    return result


def _winner_key(item: tuple[str, dict[str, Any]]) -> tuple[float, float, float, float]:
    _name, summary = item
    oracle = summary["oracle"]
    return (
        float(oracle["20"]),
        float(oracle["10"]),
        float(oracle["42"]),
        -float(summary["mean_candidates"]),
    )


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"shots": len(rows)}
    sources = (
        "current",
        "v2212_local",
        "v2212_union",
        "consensus",
        "current_plus_consensus",
        "masked_direct",
        "final_union",
    )
    for source in sources:
        distances = [float(r["nearest"][source]) for r in rows if float(r["nearest"][source]) < 9000]
        result[source] = {
            "oracle": {
                str(int(radius)): float(sum(bool(r["recall"][source][str(int(radius))]) for r in rows) / max(1, len(rows)))
                for radius in RADII
            },
            "mean_candidates": float(np.mean([int(r["counts"][source]) for r in rows])) if rows else 0.0,
            "median_nearest_px": float(np.median(distances)) if distances else 9999.0,
        }
    result["rescued_at_20"] = {
        source: int(sum((not r["recall"]["current"]["20"]) and r["recall"][source]["20"] for r in rows))
        for source in ("v2212_union", "current_plus_consensus", "masked_direct", "final_union")
    }
    result["mask_fraction_mean"] = float(np.mean([float(r.get("mask_fraction", 0.0)) for r in rows])) if rows else 0.0
    return result


def _debug(path: Path, post: np.ndarray, gt: tuple[float, float], mask: np.ndarray, current, consensus, masked_direct) -> None:
    base = cv2.cvtColor(np.asarray(post, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    gx, gy = int(round(gt[0])), int(round(gt[1]))
    cv2.drawMarker(base, (gx, gy), (0, 255, 0), cv2.MARKER_CROSS, 42, 3)
    for items, color, radius in (
        (current, (255, 160, 0), 5),
        (consensus, (0, 255, 255), 4),
        (masked_direct, (0, 0, 255), 4),
    ):
        for cand in items[:800]:
            cv2.circle(base, (int(round(cand["camera_x"])), int(round(cand["camera_y"]))), radius, color, 1)
    mask_u8 = (np.asarray(mask, dtype=np.uint8) * 255)
    mask_vis = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
    cv2.drawMarker(mask_vis, (gx, gy), (0, 255, 0), cv2.MARKER_CROSS, 42, 3)
    half_w = min(1200, base.shape[1])
    scale = min(1.0, half_w / float(base.shape[1]))
    if scale < 1.0:
        size = (int(base.shape[1] * scale), int(base.shape[0] * scale))
        base = cv2.resize(base, size, interpolation=cv2.INTER_AREA)
        mask_vis = cv2.resize(mask_vis, size, interpolation=cv2.INTER_NEAREST)
    canvas = np.concatenate([base, mask_vis], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def benchmark_fullframe_v2213(
    root: Path,
    *,
    direct_config: DirectProposalConfigV221 | None = None,
    baseline_local_config: LocalTemporalConfigV2212 | None = None,
    masked_direct_config: MaskedDirectConfigV2213 | None = None,
    debug_dir: Path | None = None,
    debug_limit: int = 12,
) -> dict[str, Any]:
    root = Path(root)
    direct_cfg = direct_config or DirectProposalConfigV221()
    baseline_cfg = baseline_local_config or LocalTemporalConfigV2212()
    masked_cfg = masked_direct_config or MaskedDirectConfigV2213()
    profiles = _profiles()

    paths = discover_candidate_packs(root)
    if not paths:
        raise RuntimeError(f"No candidate packs under {root}")
    split_keys, provisional = _shot_split_keys_v217(root)
    lookup = {key: name for name, keys in split_keys.items() for key in keys}

    # -------------------- DEVELOPMENT-ONLY TUNING --------------------
    tuning: dict[str, list[dict[str, Any]]] = {name: [] for name in profiles}
    dev_fullframe = 0
    for number, path in enumerate(paths, 1):
        pack = CandidatePackV216.load(path)
        key = (str(pack.metadata.get("session_id", "unknown")), int(pack.metadata.get("round_id", 0)))
        if lookup.get(key, "unknown") != "development":
            continue
        frames = _pack_frames(pack)
        gt = pack.gt_xy
        if frames is None or gt is None:
            continue
        dev_fullframe += 1
        pre, posts = frames
        direct_result = propose_direct_v221(pre, posts, config=direct_cfg)
        current = _current_rows(pack)
        for name, cfg in profiles.items():
            items = propose_temporal_consensus_v2213(current, direct_result.maps, config=cfg)
            tuning[name].append({
                "recall": {str(int(radius)): _hit(items, gt, radius) for radius in RADII},
                "candidate_count": len(items),
                "distance_px": _nearest(items, gt),
            })
        if dev_fullframe % 6 == 0:
            print(f"V2.21.3 tuning: {dev_fullframe} development full-frame shots")

    tuning_summary = {name: _profile_summary(items) for name, items in tuning.items()}
    if not tuning_summary or max((s["shots"] for s in tuning_summary.values()), default=0) == 0:
        raise RuntimeError("No development full-frame packs available for V2.21.3 tuning")
    winner_name, winner_summary = max(tuning_summary.items(), key=_winner_key)
    winner_cfg = profiles[winner_name]

    # -------------------- FROZEN EVALUATION --------------------
    rows: list[dict[str, Any]] = []
    missing = 0
    debug_written = 0
    for number, path in enumerate(paths, 1):
        pack = CandidatePackV216.load(path)
        gt = pack.gt_xy
        frames = _pack_frames(pack)
        if frames is None or gt is None:
            missing += 1
            continue
        pre, posts = frames
        direct_result = propose_direct_v221(pre, posts, config=direct_cfg)
        current = _current_rows(pack)
        baseline_local = propose_local_temporal_v2212(current, direct_result.maps, direct_result.fused, config=baseline_cfg)
        baseline_union = _union(current, baseline_local)
        consensus = propose_temporal_consensus_v2213(current, direct_result.maps, config=winner_cfg)
        current_plus_consensus = _union(current, consensus)
        masked_direct, mask = propose_masked_direct_v2213(current, direct_result.maps, config=masked_cfg)
        final_union = _union(current, consensus, masked_direct)
        groups = {
            "current": current,
            "v2212_local": baseline_local,
            "v2212_union": baseline_union,
            "consensus": consensus,
            "current_plus_consensus": current_plus_consensus,
            "masked_direct": masked_direct,
            "final_union": final_union,
        }
        recall = {
            name: {str(int(radius)): _hit(items, gt, radius) for radius in RADII}
            for name, items in groups.items()
        }
        nearest = {name: _nearest(items, gt) for name, items in groups.items()}
        key = (str(pack.metadata.get("session_id", "unknown")), int(pack.metadata.get("round_id", 0)))
        row = {
            "json_path": str(path),
            "session_id": key[0],
            "round_id": key[1],
            "split": lookup.get(key, "unknown"),
            "gt": [float(gt[0]), float(gt[1])],
            "counts": {name: len(items) for name, items in groups.items()},
            "recall": recall,
            "nearest": nearest,
            "mask_fraction": float(np.mean(mask)),
            "runtime_ms_direct_maps": float(direct_result.metadata.get("runtime_ms", 0.0)),
        }
        rows.append(row)

        if debug_dir is not None and debug_written < max(0, int(debug_limit)):
            if not recall["final_union"]["20"] or (not recall["current"]["20"] and recall["final_union"]["20"]):
                _debug(
                    Path(debug_dir) / f"{key[0]}_shot_{key[1]:06d}_v2213.png",
                    direct_result.post_reference,
                    gt,
                    mask,
                    current,
                    consensus,
                    masked_direct,
                )
                debug_written += 1
        if number % 10 == 0 or number == len(paths):
            print(f"V2.21.3 frozen benchmark: {number}/{len(paths)} packs")

    splits = {
        name: _aggregate([r for r in rows if r["split"] == name])
        for name in ("development", "confirmation", "holdout")
    }
    all_summary = _aggregate(rows)
    return {
        "schema_version": "2.21.3",
        "root": str(root),
        "packs_discovered": len(paths),
        "packs_benchmarked": len(rows),
        "packs_missing_full_frames": missing,
        "split_is_provisional": bool(provisional),
        "development_tuning_only": True,
        "development_fullframe_shots": int(dev_fullframe),
        "profile_sweep": {
            name: {
                "config": consensus_config_dict_v2213(profiles[name]),
                "development": tuning_summary[name],
            }
            for name in profiles
        },
        "selected_profile": winner_name,
        "selected_config": consensus_config_dict_v2213(winner_cfg),
        "selected_development_summary": winner_summary,
        "masked_direct_config": masked_direct_config_dict_v2213(masked_cfg),
        "all": all_summary,
        "splits": splits,
        "rows": rows,
        "debug_written": debug_written,
        "gate": {
            "development_consensus_beats_v2212_union_oracle20": bool(
                splits["development"]["current_plus_consensus"]["oracle"]["20"]
                > splits["development"]["v2212_union"]["oracle"]["20"]
            ),
            "confirmation_final_union_oracle20_ge_070": bool(splits["confirmation"]["final_union"]["oracle"]["20"] >= 0.70),
            "holdout_final_union_oracle20_ge_070": bool(splits["holdout"]["final_union"]["oracle"]["20"] >= 0.70),
            "eligible_for_live_authority": False,
        },
        "semantic_note": (
            "Profile selection uses DEVELOPMENT only. Confirmation/holdout are evaluated only after the winning "
            "configuration is frozen. Proposal generation receives no GT. This is still provisional one/two-session data."
        ),
    }
