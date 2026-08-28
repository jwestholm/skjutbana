from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.engine.offline.candidate_pack_v216 import CandidatePackV216, discover_candidate_packs
from src.engine.offline.direct_proposal_v221 import DirectProposalConfigV221, propose_direct_v221
from src.engine.offline.fullframe_benchmark_v2212 import _current_rows, _pack_frames, _union
from src.engine.offline.new_hole_training_v217 import _shot_split_keys_v217
from src.engine.offline.physical_dense_v2215 import (
    DensePoolConfigV2215,
    ListwiseModelV2215,
    candidate_distances_v2215,
    extract_candidate_features_v2215,
    propose_dense_pool_v2215,
    rank_candidates_v2215,
    shot_key_v2215,
)
from src.engine.offline.temporal_local_v2212 import LocalTemporalConfigV2212, propose_local_temporal_v2212


RADII = (5, 10, 20, 42)
TOP_KS = (64, 128, 256, 512, 1024)


def _distances(rows: Sequence[dict[str, Any]], gt: tuple[float, float]) -> np.ndarray:
    return candidate_distances_v2215(rows, gt)


def _nearest(rows: Sequence[dict[str, Any]], gt: tuple[float, float]) -> float:
    d = _distances(rows, gt)
    return float(np.min(d)) if len(d) else 9999.0


def _recall(rows: Sequence[dict[str, Any]], gt: tuple[float, float]) -> dict[str, bool]:
    d = _distances(rows, gt)
    return {str(r): bool(np.any(d <= float(r))) for r in RADII}


def _summarize(rows: list[dict[str, Any]], source: str) -> dict[str, Any]:
    if not rows:
        return {
            "oracle": {str(r): 0.0 for r in RADII},
            "mean_candidates": 0.0,
            "median_nearest_px": 9999.0,
        }
    return {
        "oracle": {
            str(r): float(np.mean([float(row["recall"][source][str(r)]) for row in rows]))
            for r in RADII
        },
        "mean_candidates": float(np.mean([int(row["counts"][source]) for row in rows])),
        "median_nearest_px": float(np.median([float(row["nearest"][source]) for row in rows])),
    }


def _rank_of_positive(ranked: Sequence[dict[str, Any]], gt: tuple[float, float], radius: float = 20.0) -> int:
    d = _distances(ranked, gt)
    idx = np.flatnonzero(d <= float(radius))
    if len(idx) == 0:
        return 9999
    return int(np.min(idx) + 1)


def _union_sources(*groups: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return _union(*groups, radius=4.0)


def _inside_mask(mask: np.ndarray, gt: tuple[float, float]) -> bool:
    h, w = mask.shape[:2]
    x = int(np.clip(round(float(gt[0])), 0, w - 1))
    y = int(np.clip(round(float(gt[1])), 0, h - 1))
    return bool(mask[y, x])


def main() -> int:
    parser = argparse.ArgumentParser(description="V2.21.5 frozen candidate-aligned physical dense benchmark")
    parser.add_argument("--root", default="content/ai/candidate_shadow_v216")
    parser.add_argument("--model", default="content/ai/reports/v2215/physical_dense_ranker_v2215.npz")
    parser.add_argument("--out", default="content/ai/reports/v2215/fullframe_benchmark_v2215.json")
    args = parser.parse_args()

    root = Path(args.root)
    model = ListwiseModelV2215.load(Path(args.model))
    meta = dict(model.metadata)
    if not bool(meta.get("candidate_aligned_only", False)):
        raise RuntimeError("Refusing benchmark: model is not marked candidate_aligned_only")
    if int(meta.get("forced_positive_jitter_count", -1)) != 0:
        raise RuntimeError("Refusing benchmark: V2.21.5 model metadata reports forced GT/jitter candidates")

    paths = discover_candidate_packs(root)
    if not paths:
        raise RuntimeError(f"No candidate packs under {root}")
    split_keys, provisional = _shot_split_keys_v217(root)
    lookup = {key: name for name, keys in split_keys.items() for key in keys}
    training_keys = set(str(x) for x in meta.get("shot_keys", []))

    # Hard leakage gate before opening protected frames.
    protected_keys = {
        shot_key_v2215(session, rid)
        for split in ("confirmation", "holdout")
        for session, rid in split_keys.get(split, set())
    }
    overlap = sorted(training_keys & protected_keys)
    if overlap:
        raise RuntimeError(f"Protected-shot leakage in frozen model metadata: {overlap[:5]}")

    pool_cfg = DensePoolConfigV2215()
    direct_cfg = DirectProposalConfigV221()
    local_cfg = LocalTemporalConfigV2212()
    output_rows: list[dict[str, Any]] = []
    missing = 0

    for index, path in enumerate(paths, 1):
        pack = CandidatePackV216.load(path)
        frames = _pack_frames(pack)
        if frames is None or pack.gt_xy is None:
            missing += 1
            continue
        pre, posts = frames
        session_id = str(pack.metadata.get("session_id", "unknown"))
        round_id = int(pack.metadata.get("round_id", 0))
        key = shot_key_v2215(session_id, round_id)
        split = lookup.get((session_id, round_id), "unknown")
        gt = (float(pack.gt_xy[0]), float(pack.gt_xy[1]))

        direct = propose_direct_v221(pre, posts, config=direct_cfg)
        current = _current_rows(pack)
        maps = dict(direct.maps)
        maps["fused"] = np.asarray(direct.fused, dtype=np.float32)
        local = propose_local_temporal_v2212(current, direct.maps, direct.fused, config=local_cfg)
        v2212_union = _union_sources(current, local)

        dense = propose_dense_pool_v2215(current, maps, config=pool_cfg)
        features = extract_candidate_features_v2215(
            dense.candidates,
            maps,
            dense.target_mask,
            current_candidates=current,
            local_candidates=local,
        )
        ranked = rank_candidates_v2215(dense.candidates, features, model)

        groups: dict[str, list[dict[str, Any]]] = {
            "current": list(current),
            "v2212_union": list(v2212_union),
            "dense_pool": list(dense.candidates),
        }
        for k in TOP_KS:
            groups[f"learned_{k}"] = list(ranked[:k])
            groups[f"v2212_plus_learned_{k}"] = _union_sources(v2212_union, ranked[:k])

        row = {
            "json_path": str(path),
            "session_id": session_id,
            "round_id": round_id,
            "key": key,
            "split": split,
            "was_training_shot": key in training_keys,
            "gt": [gt[0], gt[1]],
            "counts": {name: len(group) for name, group in groups.items()},
            "recall": {name: _recall(group, gt) for name, group in groups.items()},
            "nearest": {name: _nearest(group, gt) for name, group in groups.items()},
            "learned_gt_rank20": _rank_of_positive(ranked, gt, 20.0),
            "mask_fraction": float(np.mean(dense.target_mask)),
            "gt_in_target_mask": _inside_mask(dense.target_mask, gt),
            "pool_metadata": {
                "raw_points": int(dense.metadata.get("raw_points", 0)),
                "clustered_points": int(dense.metadata.get("clustered_points", 0)),
                "final_points": int(dense.metadata.get("final_points", 0)),
            },
        }
        output_rows.append(row)
        if index % 10 == 0 or index == len(paths):
            print(f"V2.21.5 frozen benchmark: {index}/{len(paths)} packs")

    def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"shots": 0}
        result: dict[str, Any] = {"shots": len(rows)}
        source_names = ["current", "v2212_union", "dense_pool"]
        for k in TOP_KS:
            source_names += [f"learned_{k}", f"v2212_plus_learned_{k}"]
        for source in source_names:
            result[source] = _summarize(rows, source)
        ranks = [int(r["learned_gt_rank20"]) for r in rows if int(r["learned_gt_rank20"]) < 9999]
        result["learned_gt_rank20"] = {
            "shots_with_pool_candidate20": len(ranks),
            "median_rank": float(np.median(ranks)) if ranks else 9999.0,
            "p90_rank": float(np.percentile(ranks, 90)) if ranks else 9999.0,
        }
        result["rescued_at_20"] = {}
        for k in (512, 1024):
            learned_source = f"learned_{k}"
            final_source = f"v2212_plus_learned_{k}"
            result["rescued_at_20"][f"learned_{k}_vs_current"] = int(sum(
                (not r["recall"]["current"]["20"]) and r["recall"][learned_source]["20"] for r in rows
            ))
            result["rescued_at_20"][f"learned_{k}_beyond_v2212"] = int(sum(
                (not r["recall"]["v2212_union"]["20"]) and r["recall"][final_source]["20"] for r in rows
            ))
        result["target_mask"] = {
            "mean_image_fraction": float(np.mean([float(r["mask_fraction"]) for r in rows])),
            "gt_coverage_fraction_diagnostic_only": float(np.mean([float(r["gt_in_target_mask"]) for r in rows])),
        }
        return result

    by_split = {
        split: [r for r in output_rows if r["split"] == split]
        for split in ("development", "confirmation", "holdout")
    }
    all_summary = build_summary(output_rows)
    split_summary = {name: build_summary(rows) for name, rows in by_split.items()}

    cv = dict(meta.get("development_crossfit", {}))
    gate = {
        "candidate_aligned_only": True,
        "forced_positive_jitter_is_zero": True,
        "protected_shots_not_used_for_training": not bool(overlap),
        "development_crossfit_top512_oracle20_ge_050": float(cv.get("top512_oracle20", 0.0)) >= 0.50,
        "confirmation_final512_oracle20_ge_070": float(split_summary.get("confirmation", {}).get("v2212_plus_learned_512", {}).get("oracle", {}).get("20", 0.0)) >= 0.70,
        "holdout_final512_oracle20_ge_070": float(split_summary.get("holdout", {}).get("v2212_plus_learned_512", {}).get("oracle", {}).get("20", 0.0)) >= 0.70,
        "eligible_for_live_authority": False,
    }
    report = {
        "schema_version": "2.21.5",
        "root": str(root),
        "model_path": str(Path(args.model)),
        "packs_discovered": len(paths),
        "packs_benchmarked": len(output_rows),
        "packs_missing_full_frames": missing,
        "split_is_provisional": bool(provisional),
        "candidate_aligned_only": True,
        "forced_positive_jitter_count": 0,
        "model_metadata": meta,
        "top_k_values": list(TOP_KS),
        "all": all_summary,
        "splits": split_summary,
        "gate": gate,
        "rows": output_rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("V2.21.5 CANDIDATE-ALIGNED LISTWISE DENSE BENCHMARK")
    print("===================================================")
    print(f"Packs discovered    : {len(paths)}")
    print(f"Full-frame packs    : {len(output_rows)}")
    print(f"Missing full frames : {missing}")
    print(f"Split provisional   : {bool(provisional)}")
    print("Forced GT/jitter    : 0")
    print()

    def show(name: str, summary: dict[str, Any]) -> None:
        print(f"{name} shots={summary.get('shots', 0)}")
        for source in ["current", "v2212_union", "dense_pool", "learned_64", "learned_128", "learned_256", "learned_512", "learned_1024", "v2212_plus_learned_512", "v2212_plus_learned_1024"]:
            item = summary.get(source)
            if not item:
                continue
            o = item["oracle"]
            print(
                f"  {source:<28} o5={o['5']:.4f} o10={o['10']:.4f} "
                f"o20={o['20']:.4f} o42={o['42']:.4f} "
                f"median={item['median_nearest_px']:.1f}px mean_n={item['mean_candidates']:.1f}"
            )
        rank = summary.get("learned_gt_rank20", {})
        print(
            "  learned GT rank@20: "
            f"n={rank.get('shots_with_pool_candidate20', 0)} "
            f"median={rank.get('median_rank', 9999.0):.1f} p90={rank.get('p90_rank', 9999.0):.1f}"
        )
        print(f"  rescued@20={summary.get('rescued_at_20', {})}")
        print(f"  target-mask={summary.get('target_mask', {})}")
        print()

    show("ALL", all_summary)
    for split in ("development", "confirmation", "holdout"):
        show(split.upper(), split_summary[split])
    print(f"Gate        : {gate}")
    print(f"Report      : {out}")
    print("NEXT: Compare DEVELOPMENT cross-fit and frozen protected ranking. Dense-pool recall is a proposal ceiling; ranking must improve without GT/jitter candidates before any live authority.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
