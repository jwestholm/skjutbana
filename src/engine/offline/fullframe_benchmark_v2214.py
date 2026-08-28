from __future__ import annotations

"""V2.21.4 physical dense-model training and frozen benchmark helpers."""

import json
import math
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .candidate_pack_v216 import CandidatePackV216, discover_candidate_packs
from .direct_proposal_v221 import DirectProposalConfigV221, propose_direct_v221
from .fullframe_benchmark_v2212 import _current_rows, _pack_frames, _union
from .new_hole_training_v217 import _shot_split_keys_v217
from .physical_dense_v2214 import (
    DEFAULT_MODEL_PATH,
    DensePoolConfigV2214,
    DenseRankerV2214,
    DenseTrainingConfigV2214,
    build_dense_pool_v2214,
    load_dense_configs_v2214,
    make_shot_training_data_v2214,
    rank_dense_pool_v2214,
    train_dense_ranker_v2214,
)
from .temporal_local_v2212 import LocalTemporalConfigV2212, propose_local_temporal_v2212

RADII = (5.0, 10.0, 20.0, 42.0)


def _dist(candidate: dict[str, Any], gt: tuple[float, float]) -> float:
    return float(math.hypot(float(candidate.get("camera_x", 0.0)) - gt[0], float(candidate.get("camera_y", 0.0)) - gt[1]))


def _nearest(candidates: Sequence[dict[str, Any]], gt: tuple[float, float]) -> float:
    if not candidates:
        return 9999.0
    return min(_dist(row, gt) for row in candidates)


def _hit(candidates: Sequence[dict[str, Any]], gt: tuple[float, float], radius: float) -> bool:
    return any(_dist(row, gt) <= radius for row in candidates)


def _rank20(candidates: Sequence[dict[str, Any]], gt: tuple[float, float]) -> int:
    for index, row in enumerate(candidates, 1):
        if _dist(row, gt) <= 20.0:
            return index
    return 9999


def train_physical_dense_v2214(
    root: Path,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    config_path: Path | None = None,
    direct_config: DirectProposalConfigV221 | None = None,
) -> dict[str, Any]:
    root = Path(root)
    pool_cfg, train_cfg = load_dense_configs_v2214(config_path)
    direct_cfg = direct_config or DirectProposalConfigV221()
    paths = discover_candidate_packs(root)
    if not paths:
        raise RuntimeError(f"No candidate packs under {root}")
    split_keys, provisional = _shot_split_keys_v217(root)
    lookup = {key: name for name, keys in split_keys.items() for key in keys}

    shots = []
    skipped_no_fullframe = 0
    protected_fullframe = {"confirmation": 0, "holdout": 0}
    dev_seen = 0
    for number, path in enumerate(paths, 1):
        pack = CandidatePackV216.load(path)
        key = (str(pack.metadata.get("session_id", "unknown")), int(pack.metadata.get("round_id", 0)))
        split = lookup.get(key, "unknown")
        frames = _pack_frames(pack)
        gt = pack.gt_xy
        if frames is None or gt is None:
            skipped_no_fullframe += 1
            continue
        if split != "development":
            if split in protected_fullframe:
                protected_fullframe[split] += 1
            continue
        pre, posts = frames
        result = propose_direct_v221(pre, posts, config=direct_cfg)
        current = _current_rows(pack)
        shot = make_shot_training_data_v2214(
            f"{key[0]}:{key[1]}",
            current,
            result.maps,
            result.fused,
            gt,
            pool_config=pool_cfg,
            training_config=train_cfg,
        )
        shots.append(shot)
        dev_seen += 1
        print(
            f"V2.21.4 train-data: {dev_seen} development shots | "
            f"pool={len(shot.pool_rows)} pool20={bool(np.any(shot.pool_distances <= 20.0))}"
        )

    if not shots:
        raise RuntimeError("No DEVELOPMENT full-frame shots found for V2.21.4 training")
    model, training_report = train_dense_ranker_v2214(
        shots,
        training_config=train_cfg,
        pool_config=pool_cfg,
    )
    model_path = model.save(model_path)
    report = {
        "schema_version": "2.21.4",
        "root": str(root),
        "model_path": str(model_path),
        "split_is_provisional": bool(provisional),
        "development_only_training": True,
        "development_fullframe_shots_used": len(shots),
        "protected_fullframe_shots_not_used": protected_fullframe,
        "skipped_no_fullframe": int(skipped_no_fullframe),
        "training": training_report,
        "gate": {
            "development_dense_pool_oracle20_ge_070": bool(training_report["development_dense_pool_oracle20"] >= 0.70),
            "model_frozen_before_protected_evaluation": True,
            "eligible_for_live_authority": False,
        },
        "semantic_note": (
            "Only DEVELOPMENT full-frame shots are used to build samples, fit normalisation, mine hard negatives "
            "and fit weights. Confirmation and holdout counts are reported but their frames are not opened for training."
        ),
    }
    return report


def _aggregate(rows: Sequence[dict[str, Any]], top_k_values: Sequence[int], frozen_top_k: int) -> dict[str, Any]:
    result: dict[str, Any] = {"shots": len(rows)}
    fixed_sources = (
        "current",
        "v2212_union",
        "dense_pool",
        f"learned_{frozen_top_k}",
        f"current_plus_learned_{frozen_top_k}",
        f"v2212_plus_learned_{frozen_top_k}",
    )
    dynamic_sources = tuple(f"learned_{int(k)}" for k in top_k_values)
    sources = tuple(dict.fromkeys(fixed_sources + dynamic_sources))
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
    ranks = [int(r["learned_gt_rank20"]) for r in rows if int(r["learned_gt_rank20"]) < 9999]
    result["learned_gt_rank20"] = {
        "shots_with_pool_candidate20": len(ranks),
        "median_rank": float(np.median(ranks)) if ranks else 9999.0,
        "p90_rank": float(np.percentile(ranks, 90)) if ranks else 9999.0,
    }
    final_source = f"v2212_plus_learned_{frozen_top_k}"
    result["rescued_at_20"] = {
        "vs_current": int(sum((not r["recall"]["current"]["20"]) and r["recall"][final_source]["20"] for r in rows)),
        "beyond_v2212": int(sum((not r["recall"]["v2212_union"]["20"]) and r["recall"][final_source]["20"] for r in rows)),
    }
    result["target_mask"] = {
        "mean_image_fraction": float(np.mean([float(r["mask_fraction"]) for r in rows])) if rows else 0.0,
        "gt_coverage_fraction_diagnostic_only": float(np.mean([bool(r["gt_in_target_mask"]) for r in rows])) if rows else 0.0,
    }
    return result


def _debug_image(
    path: Path,
    post: np.ndarray,
    gt: tuple[float, float],
    mask: np.ndarray,
    current: Sequence[dict[str, Any]],
    baseline: Sequence[dict[str, Any]],
    learned: Sequence[dict[str, Any]],
) -> None:
    base = cv2.cvtColor(np.asarray(post, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    gx, gy = int(round(gt[0])), int(round(gt[1]))
    cv2.drawMarker(base, (gx, gy), (0, 255, 0), cv2.MARKER_CROSS, 42, 3)
    for items, color, radius, cap in (
        (current, (255, 160, 0), 4, 450),
        (baseline, (0, 255, 255), 3, 900),
        (learned, (0, 0, 255), 4, 512),
    ):
        for cand in items[:cap]:
            cv2.circle(base, (int(round(cand["camera_x"])), int(round(cand["camera_y"]))), radius, color, 1)
    mask_vis = cv2.cvtColor((np.asarray(mask, dtype=np.uint8) * 255), cv2.COLOR_GRAY2BGR)
    cv2.drawMarker(mask_vis, (gx, gy), (0, 255, 0), cv2.MARKER_CROSS, 42, 3)
    max_w = 1200
    scale = min(1.0, max_w / float(base.shape[1]))
    if scale < 1.0:
        size = (int(base.shape[1] * scale), int(base.shape[0] * scale))
        base = cv2.resize(base, size, interpolation=cv2.INTER_AREA)
        mask_vis = cv2.resize(mask_vis, size, interpolation=cv2.INTER_NEAREST)
    canvas = np.concatenate([base, mask_vis], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def benchmark_physical_dense_v2214(
    root: Path,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    config_path: Path | None = None,
    direct_config: DirectProposalConfigV221 | None = None,
    local_config: LocalTemporalConfigV2212 | None = None,
    debug_dir: Path | None = None,
    debug_limit: int = 12,
) -> dict[str, Any]:
    root = Path(root)
    model = DenseRankerV2214.load(model_path)
    pool_cfg, train_cfg = load_dense_configs_v2214(config_path)
    direct_cfg = direct_config or DirectProposalConfigV221()
    local_cfg = local_config or LocalTemporalConfigV2212()
    top_k_values = tuple(sorted(set(int(k) for k in train_cfg.top_k_values)))
    frozen_top_k = int(train_cfg.frozen_top_k)
    max_top_k = max(max(top_k_values), frozen_top_k)

    paths = discover_candidate_packs(root)
    if not paths:
        raise RuntimeError(f"No candidate packs under {root}")
    split_keys, provisional = _shot_split_keys_v217(root)
    lookup = {key: name for name, keys in split_keys.items() for key in keys}

    # Strong safety check: the model must identify exactly which DEVELOPMENT
    # shots it was fitted on.  Protected evaluation never updates it.
    trained_keys = set(str(x) for x in model.metadata.get("shot_keys", []))
    rows: list[dict[str, Any]] = []
    missing = 0
    debug_written = 0
    for number, path in enumerate(paths, 1):
        pack = CandidatePackV216.load(path)
        frames = _pack_frames(pack)
        gt = pack.gt_xy
        if frames is None or gt is None:
            missing += 1
            continue
        key = (str(pack.metadata.get("session_id", "unknown")), int(pack.metadata.get("round_id", 0)))
        split = lookup.get(key, "unknown")
        shot_key = f"{key[0]}:{key[1]}"
        if split in ("confirmation", "holdout") and shot_key in trained_keys:
            raise RuntimeError(f"Protected shot leaked into V2.21.4 model training: {shot_key}")

        pre, posts = frames
        direct_result = propose_direct_v221(pre, posts, config=direct_cfg)
        current = _current_rows(pack)
        local = propose_local_temporal_v2212(current, direct_result.maps, direct_result.fused, config=local_cfg)
        baseline = _union(current, local)
        pool, target_mask = build_dense_pool_v2214(current, direct_result.maps, direct_result.fused, config=pool_cfg)
        learned_max = rank_dense_pool_v2214(
            pool,
            direct_result.maps,
            direct_result.fused,
            model,
            limit=max_top_k,
            nms_radius_px=4.0,
        )
        learned_groups = {f"learned_{k}": learned_max[:k] for k in top_k_values}
        learned_frozen = learned_max[:frozen_top_k]
        current_plus = _union(current, learned_frozen)
        final_union = _union(baseline, learned_frozen)
        groups: dict[str, Sequence[dict[str, Any]]] = {
            "current": current,
            "v2212_union": baseline,
            "dense_pool": pool,
            f"learned_{frozen_top_k}": learned_frozen,
            f"current_plus_learned_{frozen_top_k}": current_plus,
            f"v2212_plus_learned_{frozen_top_k}": final_union,
            **learned_groups,
        }
        recall = {
            name: {str(int(radius)): _hit(items, gt, radius) for radius in RADII}
            for name, items in groups.items()
        }
        nearest = {name: _nearest(items, gt) for name, items in groups.items()}
        rank20 = _rank20(learned_max, gt)
        gx = min(max(int(round(gt[0])), 0), target_mask.shape[1] - 1)
        gy = min(max(int(round(gt[1])), 0), target_mask.shape[0] - 1)
        row = {
            "json_path": str(path),
            "session_id": key[0],
            "round_id": key[1],
            "split": split,
            "was_training_shot": shot_key in trained_keys,
            "gt": [float(gt[0]), float(gt[1])],
            "counts": {name: len(items) for name, items in groups.items()},
            "recall": recall,
            "nearest": nearest,
            "learned_gt_rank20": int(rank20),
            "mask_fraction": float(np.mean(target_mask)),
            "gt_in_target_mask": bool(target_mask[gy, gx]),
        }
        rows.append(row)

        final_name = f"v2212_plus_learned_{frozen_top_k}"
        if debug_dir is not None and debug_written < max(0, int(debug_limit)):
            interesting = (
                (not recall["v2212_union"]["20"] and recall[final_name]["20"])
                or not recall[final_name]["20"]
            )
            if interesting:
                _debug_image(
                    Path(debug_dir) / f"{key[0]}_shot_{key[1]:06d}_v2214.png",
                    direct_result.post_reference,
                    gt,
                    target_mask,
                    current,
                    baseline,
                    learned_frozen,
                )
                debug_written += 1
        if number % 10 == 0 or number == len(paths):
            print(f"V2.21.4 frozen benchmark: {number}/{len(paths)} packs")

    splits = {
        name: _aggregate([r for r in rows if r["split"] == name], top_k_values, frozen_top_k)
        for name in ("development", "confirmation", "holdout")
    }
    all_summary = _aggregate(rows, top_k_values, frozen_top_k)
    final_name = f"v2212_plus_learned_{frozen_top_k}"
    return {
        "schema_version": "2.21.4",
        "root": str(root),
        "model_path": str(model_path),
        "packs_discovered": len(paths),
        "packs_benchmarked": len(rows),
        "packs_missing_full_frames": missing,
        "split_is_provisional": bool(provisional),
        "development_only_training": True,
        "model_metadata": model.metadata,
        "top_k_values": list(top_k_values),
        "frozen_top_k": frozen_top_k,
        "all": all_summary,
        "splits": splits,
        "rows": rows,
        "debug_written": debug_written,
        "gate": {
            "development_dense_pool_oracle20_ge_090": bool(splits["development"]["dense_pool"]["oracle"]["20"] >= 0.90),
            "development_final_beats_v2212_oracle20": bool(
                splits["development"][final_name]["oracle"]["20"] > splits["development"]["v2212_union"]["oracle"]["20"]
            ),
            "confirmation_final_oracle20_ge_070": bool(splits["confirmation"][final_name]["oracle"]["20"] >= 0.70),
            "holdout_final_oracle20_ge_070": bool(splits["holdout"][final_name]["oracle"]["20"] >= 0.70),
            "protected_shots_not_used_for_training": True,
            "eligible_for_live_authority": False,
        },
        "semantic_note": (
            "The broad dense pool and learned ranking receive no GT during benchmark. GT is used only afterwards for metrics. "
            "The model was previously fitted on DEVELOPMENT full-frame shots only. Confirmation/holdout remain protected, "
            "but the split is still provisional until independent full-frame sessions exist."
        ),
    }
