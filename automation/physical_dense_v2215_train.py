from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from src.engine.offline.candidate_pack_v216 import CandidatePackV216, discover_candidate_packs
from src.engine.offline.direct_proposal_v221 import DirectProposalConfigV221, propose_direct_v221
from src.engine.offline.fullframe_benchmark_v2212 import _current_rows, _pack_frames
from src.engine.offline.new_hole_training_v217 import _shot_split_keys_v217
from src.engine.offline.physical_dense_v2215 import (
    DensePoolConfigV2215,
    ListwiseConfigV2215,
    ListwiseShotV2215,
    candidate_distances_v2215,
    cross_validate_listwise_v2215,
    dense_pool_config_dict_v2215,
    extract_candidate_features_v2215,
    fit_listwise_ranker_v2215,
    listwise_config_dict_v2215,
    propose_dense_pool_v2215,
    shot_key_v2215,
)
from src.engine.offline.temporal_local_v2212 import LocalTemporalConfigV2212, propose_local_temporal_v2212


def _oracle(distances: np.ndarray, radius: float) -> bool:
    return bool(np.any(np.asarray(distances, dtype=np.float32) <= float(radius)))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="V2.21.5 candidate-aligned DEVELOPMENT-only listwise physical dense training"
    )
    parser.add_argument("--root", default="content/ai/candidate_shadow_v216")
    parser.add_argument("--model", default="content/ai/reports/v2215/physical_dense_ranker_v2215.npz")
    parser.add_argument("--out", default="content/ai/reports/v2215/physical_dense_training_v2215.json")
    parser.add_argument("--seed", type=int, default=2215)
    args = parser.parse_args()

    root = Path(args.root)
    paths = discover_candidate_packs(root)
    if not paths:
        raise RuntimeError(f"No candidate packs under {root}")

    split_keys, provisional = _shot_split_keys_v217(root)
    lookup = {key: name for name, keys in split_keys.items() for key in keys}
    pool_cfg = DensePoolConfigV2215()
    list_cfg = ListwiseConfigV2215(seed=int(args.seed))
    direct_cfg = DirectProposalConfigV221()
    local_cfg = LocalTemporalConfigV2212()

    shots: list[ListwiseShotV2215] = []
    shot_keys: list[str] = []
    feature_names: tuple[str, ...] | None = None
    rows: list[dict] = []
    skipped_no_fullframe = 0
    protected_counts = {"confirmation": 0, "holdout": 0}

    for path in paths:
        pack = CandidatePackV216.load(path)
        session_id = str(pack.metadata.get("session_id", "unknown"))
        round_id = int(pack.metadata.get("round_id", 0))
        split = lookup.get((session_id, round_id), "unknown")
        frames = _pack_frames(pack)
        if frames is None or pack.gt_xy is None:
            skipped_no_fullframe += 1
            continue
        if split in protected_counts:
            protected_counts[split] += 1
            # Protected frames are deliberately not opened by the fitting path.
            continue
        if split != "development":
            continue

        pre, posts = frames
        direct = propose_direct_v221(pre, posts, config=direct_cfg)
        current = _current_rows(pack)
        maps = dict(direct.maps)
        maps["fused"] = np.asarray(direct.fused, dtype=np.float32)
        local = propose_local_temporal_v2212(current, direct.maps, direct.fused, config=local_cfg)
        dense = propose_dense_pool_v2215(current, maps, config=pool_cfg)
        features = extract_candidate_features_v2215(
            dense.candidates,
            maps,
            dense.target_mask,
            current_candidates=current,
            local_candidates=local,
        )
        if feature_names is None:
            feature_names = features.feature_names
        elif tuple(feature_names) != tuple(features.feature_names):
            raise RuntimeError("Feature schema changed between development shots")

        gt = (float(pack.gt_xy[0]), float(pack.gt_xy[1]))
        distances = candidate_distances_v2215(dense.candidates, gt)
        key = shot_key_v2215(session_id, round_id)
        dense_scores = np.asarray([float(c.get("dense_score", 0.0)) for c in dense.candidates], dtype=np.float32)
        shots.append(ListwiseShotV2215(key=key, matrix=features.matrix, distances_px=distances, dense_scores=dense_scores))
        shot_keys.append(key)
        row = {
            "key": key,
            "pool_count": len(dense.candidates),
            "pool_oracle5": _oracle(distances, 5.0),
            "pool_oracle10": _oracle(distances, 10.0),
            "pool_oracle20": _oracle(distances, 20.0),
            "pool_oracle42": _oracle(distances, 42.0),
            "nearest_pool_px": float(np.min(distances)) if len(distances) else 9999.0,
            "mask_fraction": float(dense.metadata.get("mask_fraction", 0.0)),
        }
        rows.append(row)
        print(
            f"V2.21.5 train-data: {len(shots)} development shots | "
            f"pool={len(dense.candidates)} pool20={row['pool_oracle20']} nearest={row['nearest_pool_px']:.1f}px"
        )

    if not shots or feature_names is None:
        raise RuntimeError("No DEVELOPMENT full-frame shots available for V2.21.5")

    # Crucial diagnostic: cross-fit only DEVELOPMENT before fitting the final
    # model. Confirmation/holdout are not supplied to this function at all.
    cv = cross_validate_listwise_v2215(shots, feature_names, config=list_cfg)

    model_meta = {
        "development_only_training": True,
        "split_is_provisional": bool(provisional),
        "shot_keys": shot_keys,
        "protected_fullframe_shots_not_used": protected_counts,
        "dense_pool_config": dense_pool_config_dict_v2215(pool_cfg),
        "listwise_config": listwise_config_dict_v2215(list_cfg),
        "development_crossfit": {k: v for k, v in cv.items() if k != "rows"},
        "semantic_note": (
            "Every positive used for fitting is an actual GT-free dense-pool candidate. "
            "No forced GT coordinate, GT jitter candidate, confirmation shot or holdout shot "
            "is used for fitting, standardisation, hard-candidate mining or feature selection."
        ),
    }
    model, fit_report = fit_listwise_ranker_v2215(shots, feature_names, config=list_cfg, metadata=model_meta)
    model.save(Path(args.model))

    pool_summary = {
        "shots": len(rows),
        "oracle5": float(np.mean([float(r["pool_oracle5"]) for r in rows])),
        "oracle10": float(np.mean([float(r["pool_oracle10"]) for r in rows])),
        "oracle20": float(np.mean([float(r["pool_oracle20"]) for r in rows])),
        "oracle42": float(np.mean([float(r["pool_oracle42"]) for r in rows])),
        "median_nearest_px": float(np.median([float(r["nearest_pool_px"]) for r in rows])),
        "mean_candidates": float(np.mean([int(r["pool_count"]) for r in rows])),
        "mean_mask_fraction": float(np.mean([float(r["mask_fraction"]) for r in rows])),
    }
    report = {
        "schema_version": "2.21.5",
        "root": str(root),
        "model_path": str(Path(args.model)),
        "split_is_provisional": bool(provisional),
        "development_only_training": True,
        "development_fullframe_shots_used": len(shots),
        "protected_fullframe_shots_not_used": protected_counts,
        "skipped_no_fullframe": skipped_no_fullframe,
        "candidate_aligned_only": True,
        "forced_positive_jitter_count": 0,
        "development_pool": pool_summary,
        "development_crossfit": cv,
        "fit": fit_report,
        "feature_count": len(feature_names),
        "feature_names": list(feature_names),
        "weight_by_feature": {name: float(weight) for name, weight in zip(model.feature_names, model.weights)},
        "rows": rows,
        "gate": {
            "development_pool_oracle20_ge_080": bool(pool_summary["oracle20"] >= 0.80),
            "development_crossfit_top512_oracle20_ge_050": bool(float(cv.get("top512_oracle20", 0.0)) >= 0.50),
            "forced_positive_jitter_is_zero": True,
            "protected_shots_not_used_for_training": True,
            "eligible_for_live_authority": False,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("V2.21.5 CANDIDATE-ALIGNED LISTWISE TRAINING")
    print("============================================")
    print(f"Development full-frame shots : {len(shots)}")
    print(f"Protected not used           : {protected_counts}")
    print(f"Dense pool DEV oracle20      : {pool_summary['oracle20']:.4f}")
    print(f"Dense pool DEV oracle42      : {pool_summary['oracle42']:.4f}")
    print(f"Dense pool mean candidates   : {pool_summary['mean_candidates']:.1f}")
    print(f"Actual candidate positives   : shots_with_positive20={fit_report['usable_shots']} skipped={fit_report['skipped_no_positive20']}")
    print("Forced GT/jitter positives   : 0")
    print()
    print("DEVELOPMENT-ONLY CROSS-FIT")
    for k in list_cfg.top_k_values:
        print(f"  top{k:<4} oracle20={float(cv.get(f'top{k}_oracle20', 0.0)):.4f}")
    print(f"  pool oracle20={float(cv.get('pool20_oracle', 0.0)):.4f}")
    print(f"  rank20 median when present={float(cv.get('rank20_median_when_present', 9999.0)):.1f}")
    print()
    print(f"Final fit loss stage1        : {fit_report['stage1_loss_head_tail']}")
    print(f"Final fit loss stage2        : {fit_report['stage2_loss_head_tail']}")
    print(f"Final train top512 oracle20  : {fit_report['train_oracle20'].get('512', 0.0):.4f}")
    print(f"Model                        : {args.model}")
    print(f"Report                       : {out}")
    print("NEXT: run automation.physical_dense_v2215_benchmark with this frozen model. Protected data has not been used for fitting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
