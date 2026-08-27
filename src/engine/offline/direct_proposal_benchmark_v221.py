from __future__ import annotations

"""Benchmark V2.21 direct full-frame proposals on candidate packs that contain
honest full-frame PRE/POST evidence.
"""

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .candidate_pack_v216 import CandidatePackV216, discover_candidate_packs
from .direct_proposal_v221 import DirectProposalConfigV221, propose_direct_v221
from .new_hole_training_v217 import _shot_split_keys_v217


RADII = (5.0, 10.0, 20.0, 42.0)


def _dist(candidate: dict[str, Any], gt: tuple[float, float]) -> float:
    return float(math.hypot(float(candidate.get("camera_x", 0.0)) - gt[0], float(candidate.get("camera_y", 0.0)) - gt[1]))


def _hit(candidates: Sequence[dict[str, Any]], gt: tuple[float, float], radius: float) -> bool:
    return any(_dist(row, gt) <= radius for row in candidates)


def _current_rows(pack: CandidatePackV216) -> list[dict[str, Any]]:
    out = []
    for row in pack.candidates:
        if bool(row.get("capture_forced_gt_nearest")):
            continue
        if not (bool(row.get("in_ranked_pool")) or bool(row.get("in_raw_pool"))):
            continue
        candidate = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
        out.append({
            "camera_x": float(row.get("camera_x", candidate.get("camera_x", 0.0))),
            "camera_y": float(row.get("camera_y", candidate.get("camera_y", 0.0))),
            "score": float(candidate.get("score", 0.0) or 0.0),
            "evidence_source": "current_v1v2",
            "evidence_sources": ["current_v1v2"],
        })
    return out


def _union(current: Sequence[dict[str, Any]], direct: Sequence[dict[str, Any]], radius: float = 5.0) -> list[dict[str, Any]]:
    merged = [dict(row) for row in current]
    for row in direct:
        x, y = float(row["camera_x"]), float(row["camera_y"])
        found = None
        for index, old in enumerate(merged):
            if math.hypot(x - float(old["camera_x"]), y - float(old["camera_y"])) <= radius:
                found = index; break
        if found is None:
            merged.append(dict(row))
        else:
            sources = list(merged[found].get("evidence_sources") or [])
            for source in row.get("evidence_sources") or []:
                if source not in sources: sources.append(source)
            merged[found]["evidence_sources"] = sources
    return merged


def _pack_frames(pack: CandidatePackV216) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    pre: np.ndarray | None = None
    if isinstance(pack.full_recent_pre_frame, np.ndarray) and pack.full_recent_pre_frame.ndim == 2:
        pre = pack.full_recent_pre_frame
    elif isinstance(pack.full_pre_frame, np.ndarray) and pack.full_pre_frame.ndim == 2:
        pre = pack.full_pre_frame
    if pre is None or not isinstance(pack.full_post_frames, np.ndarray) or not pack.full_post_frames.size:
        return None
    post = pack.full_post_frames
    post_frames = [post] if post.ndim == 2 else [np.asarray(frame) for frame in post]
    return [pre], post_frames


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    result: dict[str, Any] = {"shots": n}
    for source in ("current", "direct", "union"):
        result[source] = {
            "oracle": {
                str(int(radius)): float(sum(bool(row["recall"][source][str(int(radius))]) for row in rows) / max(1, n))
                for radius in RADII
            },
            "mean_candidates": float(np.mean([row["counts"][source] for row in rows])) if rows else 0.0,
        }
    result["rescued_current_miss"] = {
        str(int(radius)): int(sum(
            (not row["recall"]["current"][str(int(radius))]) and row["recall"]["direct"][str(int(radius))]
            for row in rows
        )) for radius in RADII
    }
    result["runtime_ms_mean"] = float(np.mean([row["runtime_ms"] for row in rows])) if rows else 0.0
    return result


def benchmark_direct_proposals_v221(
    root: Path,
    *,
    config: DirectProposalConfigV221 | None = None,
) -> dict[str, Any]:
    root = Path(root); cfg = config or DirectProposalConfigV221()
    paths = discover_candidate_packs(root)
    if not paths:
        raise RuntimeError(f"No candidate packs under {root}")
    split_keys, provisional = _shot_split_keys_v217(root)
    lookup = {key: name for name, keys in split_keys.items() for key in keys}
    rows: list[dict[str, Any]] = []
    missing_full_frames = 0
    for number, path in enumerate(paths, 1):
        pack = CandidatePackV216.load(path); gt = pack.gt_xy
        frames = _pack_frames(pack)
        if gt is None or frames is None:
            missing_full_frames += 1
            continue
        pre, post = frames
        direct_result = propose_direct_v221(pre, post, config=cfg)
        current = _current_rows(pack); direct = direct_result.candidates; union = _union(current, direct)
        recall = {
            source: {str(int(radius)): _hit(items, gt, radius) for radius in RADII}
            for source, items in (("current", current), ("direct", direct), ("union", union))
        }
        key = (str(pack.metadata.get("session_id", "unknown")), int(pack.metadata.get("round_id", 0)))
        rows.append({
            "json_path": str(path), "session_id": key[0], "round_id": key[1],
            "split": lookup.get(key, "unknown"), "gt": [float(gt[0]), float(gt[1])],
            "counts": {"current": len(current), "direct": len(direct), "union": len(union)},
            "recall": recall, "runtime_ms": float(direct_result.metadata["runtime_ms"]),
            "nearest_direct_px": min((_dist(row, gt) for row in direct), default=9999.0),
        })
        if number % 10 == 0 or number == len(paths):
            print(f"V2.21 direct proposal benchmark: {number}/{len(paths)} packs")

    splits = {
        name: _aggregate([row for row in rows if row["split"] == name])
        for name in ("development", "confirmation", "holdout")
    }
    all_summary = _aggregate(rows)
    can_measure = bool(rows)
    return {
        "schema_version": "2.21",
        "root": str(root),
        "split_is_provisional": bool(provisional),
        "packs_discovered": len(paths),
        "packs_benchmarked": len(rows),
        "packs_missing_full_frames": missing_full_frames,
        "can_measure_physical_direct_recall": can_measure,
        "all": all_summary,
        "splits": splits,
        "rows": rows,
        "gate": {
            "confirmation_union_oracle20_ge_070": bool(can_measure and splits["confirmation"]["union"]["oracle"]["20"] >= 0.70),
            "holdout_union_oracle20_ge_070": bool(can_measure and splits["holdout"]["union"]["oracle"]["20"] >= 0.70),
            "eligible_for_live_authority": False,
        },
        "next_requirement": (
            "Collect V2.21 full-frame shadow capture first; old packs are patch-only."
            if not can_measure else
            "Inspect rescued/missed shots; optimize proposal recall before ranker retraining."
        ),
        "semantic_note": "Forced GT-nearest diagnostic candidates are excluded. Ground truth is used only after proposal generation for scoring.",
    }


def write_direct_benchmark_v221(path: Path, report: dict[str, Any]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
