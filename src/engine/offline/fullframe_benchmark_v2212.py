from __future__ import annotations

"""V2.21.2 physical full-frame diagnostic benchmark.

The purpose is to decide what the next model should learn, not to grant live
authority.  Ground truth is used only after proposals/refinements have been
created.
"""

import json
import math
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .candidate_pack_v216 import CandidatePackV216, discover_candidate_packs
from .direct_proposal_v221 import DirectProposalConfigV221, propose_direct_v221
from .new_hole_training_v217 import _shot_split_keys_v217
from .temporal_local_v2212 import LocalTemporalConfigV2212, local_config_dict_v2212, propose_local_temporal_v2212

RADII = (5.0, 10.0, 20.0, 42.0)


def _dist(candidate: dict[str, Any], gt: tuple[float, float]) -> float:
    return float(math.hypot(float(candidate.get("camera_x", 0.0)) - gt[0], float(candidate.get("camera_y", 0.0)) - gt[1]))


def _nearest(candidates: Sequence[dict[str, Any]], gt: tuple[float, float]) -> tuple[float, dict[str, Any] | None]:
    if not candidates:
        return 9999.0, None
    row = min(candidates, key=lambda item: _dist(item, gt))
    return _dist(row, gt), row


def _hit(candidates: Sequence[dict[str, Any]], gt: tuple[float, float], radius: float) -> bool:
    return any(_dist(row, gt) <= radius for row in candidates)


def _current_rows(pack: CandidatePackV216) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
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
            "current_rank": row.get("current_rank"),
            "evidence_source": "current_v1v2",
            "evidence_sources": ["current_v1v2"],
        })
    return out


def _union(*groups: Sequence[dict[str, Any]], radius: float = 4.0) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    r2 = float(radius * radius)
    for group in groups:
        for raw in group:
            row = dict(raw)
            x, y = float(row["camera_x"]), float(row["camera_y"])
            found = None
            for index, old in enumerate(merged):
                if (x - float(old["camera_x"])) ** 2 + (y - float(old["camera_y"])) ** 2 <= r2:
                    found = index
                    break
            if found is None:
                merged.append(row)
            else:
                sources = list(merged[found].get("evidence_sources") or [])
                for source in row.get("evidence_sources") or []:
                    if source not in sources:
                        sources.append(source)
                merged[found]["evidence_sources"] = sources
    return merged


def _pack_frames(pack: CandidatePackV216) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    pre = None
    if isinstance(pack.full_recent_pre_frame, np.ndarray) and pack.full_recent_pre_frame.ndim == 2:
        pre = pack.full_recent_pre_frame
    elif isinstance(pack.full_pre_frame, np.ndarray) and pack.full_pre_frame.ndim == 2:
        pre = pack.full_pre_frame
    post = pack.full_post_frames
    if pre is None or not isinstance(post, np.ndarray) or not post.size:
        return None
    posts = [post] if post.ndim == 2 else [np.asarray(frame) for frame in post]
    return [pre], posts


def _percentile_at(values: np.ndarray, x: float, y: float) -> float:
    arr = np.asarray(values, dtype=np.float32)
    yy = min(max(int(round(y)), 0), arr.shape[0] - 1)
    xx = min(max(int(round(x)), 0), arr.shape[1] - 1)
    value = float(arr[yy, xx])
    flat = arr.reshape(-1)
    stride = max(1, len(flat) // 120_000)
    sample = flat[::stride]
    return float(100.0 * np.mean(sample <= value))


def _best_in_radius(values: np.ndarray, x: float, y: float, radius: int) -> float:
    arr = np.asarray(values, dtype=np.float32)
    h, w = arr.shape[:2]
    x0, x1 = max(0, int(x - radius)), min(w, int(x + radius + 1))
    y0, y1 = max(0, int(y - radius)), min(h, int(y + radius + 1))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(np.max(arr[y0:y1, x0:x1]))


def _distance_percentiles(values: list[float]) -> dict[str, float]:
    finite = np.asarray([v for v in values if v < 9000.0], dtype=np.float64)
    if not len(finite):
        return {}
    qs = (0, 25, 50, 75, 90, 95, 100)
    return {f"p{q}": float(np.percentile(finite, q)) for q in qs}


def _offset_summary(rows: Sequence[dict[str, Any]], source: str, max_distance: float = 42.0) -> dict[str, Any]:
    offsets = []
    for row in rows:
        item = row.get("nearest", {}).get(source)
        if not isinstance(item, dict):
            continue
        distance = float(item.get("distance_px", 9999.0))
        if distance <= max_distance:
            offsets.append((float(item.get("dx", 0.0)), float(item.get("dy", 0.0)), distance))
    if not offsets:
        return {"count": 0}
    arr = np.asarray(offsets, dtype=np.float64)
    return {
        "count": int(len(arr)),
        "median_dx": float(np.median(arr[:, 0])),
        "median_dy": float(np.median(arr[:, 1])),
        "median_distance": float(np.median(arr[:, 2])),
        "mad_dx": float(np.median(np.abs(arr[:, 0] - np.median(arr[:, 0])))),
        "mad_dy": float(np.median(np.abs(arr[:, 1] - np.median(arr[:, 1])))),
    }


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"shots": len(rows)}
    for source in ("current", "direct", "local", "current_plus_local", "all_union"):
        result[source] = {
            "oracle": {
                str(int(radius)): float(sum(bool(r["recall"][source][str(int(radius))]) for r in rows) / max(1, len(rows)))
                for radius in RADII
            },
            "mean_candidates": float(np.mean([r["counts"][source] for r in rows])) if rows else 0.0,
            "nearest_distance_percentiles": _distance_percentiles([float(r["nearest"][source]["distance_px"]) for r in rows]),
        }
    result["rescued_at_20"] = {
        "direct": int(sum((not r["recall"]["current"]["20"]) and r["recall"]["direct"]["20"] for r in rows)),
        "local": int(sum((not r["recall"]["current"]["20"]) and r["recall"]["local"]["20"] for r in rows)),
        "current_plus_local": int(sum((not r["recall"]["current"]["20"]) and r["recall"]["current_plus_local"]["20"] for r in rows)),
        "all_union": int(sum((not r["recall"]["current"]["20"]) and r["recall"]["all_union"]["20"] for r in rows)),
    }
    result["current_offset_within42"] = _offset_summary(rows, "current", 42.0)
    if rows:
        regs = [reg for r in rows for reg in (r.get("registration") or []) if isinstance(reg, dict)]
        result["registration"] = {
            "frames": len(regs),
            "applied_fraction": float(np.mean([float(reg.get("applied", 0.0)) for reg in regs])) if regs else 0.0,
            "median_dx": float(np.median([float(reg.get("dx", 0.0)) for reg in regs])) if regs else 0.0,
            "median_dy": float(np.median([float(reg.get("dy", 0.0)) for reg in regs])) if regs else 0.0,
            "median_response": float(np.median([float(reg.get("response", 0.0)) for reg in regs])) if regs else 0.0,
        }
    return result


def _debug_image(path: Path, row: dict[str, Any], post: np.ndarray, fused: np.ndarray, current, direct, local) -> None:
    base = cv2.cvtColor(np.asarray(post, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    gt = row["gt"]
    gx, gy = int(round(gt[0])), int(round(gt[1]))
    cv2.drawMarker(base, (gx, gy), (0, 255, 0), cv2.MARKER_CROSS, 42, 3)
    for items, color, radius in ((current, (255, 160, 0), 5), (direct, (0, 0, 255), 4), (local, (0, 255, 255), 3)):
        for cand in items[:600]:
            cv2.circle(base, (int(round(cand["camera_x"])), int(round(cand["camera_y"]))), radius, color, 1)
    heat = np.clip(np.asarray(fused, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    cv2.drawMarker(heat, (gx, gy), (255, 255, 255), cv2.MARKER_CROSS, 42, 3)
    max_w = 1200
    scale = min(1.0, max_w / float(base.shape[1]))
    if scale < 1.0:
        size = (int(base.shape[1] * scale), int(base.shape[0] * scale))
        base = cv2.resize(base, size, interpolation=cv2.INTER_AREA)
        heat = cv2.resize(heat, size, interpolation=cv2.INTER_AREA)
    canvas = np.concatenate([base, heat], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def benchmark_fullframe_v2212(
    root: Path,
    *,
    direct_config: DirectProposalConfigV221 | None = None,
    local_config: LocalTemporalConfigV2212 | None = None,
    debug_dir: Path | None = None,
    debug_limit: int = 0,
) -> dict[str, Any]:
    root = Path(root)
    direct_cfg = direct_config or DirectProposalConfigV221()
    local_cfg = local_config or LocalTemporalConfigV2212()
    paths = discover_candidate_packs(root)
    if not paths:
        raise RuntimeError(f"No candidate packs under {root}")
    split_keys, provisional = _shot_split_keys_v217(root)
    lookup = {key: name for name, keys in split_keys.items() for key in keys}
    rows: list[dict[str, Any]] = []
    missing = 0
    debug_written = 0

    for number, path in enumerate(paths, 1):
        pack = CandidatePackV216.load(path)
        gt = pack.gt_xy
        frames = _pack_frames(pack)
        if gt is None or frames is None:
            missing += 1
            continue
        pre, posts = frames
        result = propose_direct_v221(pre, posts, config=direct_cfg)
        current = _current_rows(pack)
        direct = result.candidates
        local = propose_local_temporal_v2212(current, result.maps, result.fused, config=local_cfg)
        current_plus_local = _union(current, local)
        all_union = _union(current, local, direct)
        groups = {
            "current": current,
            "direct": direct,
            "local": local,
            "current_plus_local": current_plus_local,
            "all_union": all_union,
        }
        recall = {
            name: {str(int(radius)): _hit(items, gt, radius) for radius in RADII}
            for name, items in groups.items()
        }
        nearest: dict[str, Any] = {}
        for name, items in groups.items():
            distance, best = _nearest(items, gt)
            nearest[name] = {
                "distance_px": float(distance),
                "dx": None if best is None else float(best["camera_x"] - gt[0]),
                "dy": None if best is None else float(best["camera_y"] - gt[1]),
            }
        map_diag = {}
        for name, values in {**result.maps, "fused": result.fused}.items():
            map_diag[name] = {
                "gt_percentile": _percentile_at(values, gt[0], gt[1]),
                "best_within20": _best_in_radius(values, gt[0], gt[1], 20),
                "best_within42": _best_in_radius(values, gt[0], gt[1], 42),
            }
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
            "maps_at_gt": map_diag,
            "registration": list(result.metadata.get("registration") or []),
            "runtime_ms": float(result.metadata.get("runtime_ms", 0.0)),
        }
        rows.append(row)

        # Debug only the interesting misses first.  GT is intentionally drawn
        # only in this offline artifact, never fed into proposal generation.
        if debug_dir is not None and debug_written < max(0, int(debug_limit)):
            if not recall["current_plus_local"]["20"] or not recall["direct"]["42"]:
                name = f"{key[0]}_shot_{key[1]:06d}_debug.png"
                _debug_image(Path(debug_dir) / name, row, result.post_reference, result.fused, current, direct, local)
                debug_written += 1
        if number % 10 == 0 or number == len(paths):
            print(f"V2.21.2 full-frame benchmark: {number}/{len(paths)} packs")

    splits = {name: _aggregate([r for r in rows if r["split"] == name]) for name in ("development", "confirmation", "holdout")}
    all_summary = _aggregate(rows)
    # Summarise how strongly each map sees GT.  These numbers diagnose whether
    # the V2.21 global proposal failure is threshold/top-N crowding vs absent signal.
    map_names = sorted({name for r in rows for name in r.get("maps_at_gt", {})})
    map_summary = {}
    for name in map_names:
        vals = [r["maps_at_gt"][name]["gt_percentile"] for r in rows if name in r.get("maps_at_gt", {})]
        dev_vals = [r["maps_at_gt"][name]["gt_percentile"] for r in rows if r["split"] == "development" and name in r.get("maps_at_gt", {})]
        map_summary[name] = {
            "gt_percentile_median_all": float(np.median(vals)) if vals else 0.0,
            "gt_percentile_median_development": float(np.median(dev_vals)) if dev_vals else 0.0,
        }
    return {
        "schema_version": "2.21.2",
        "root": str(root),
        "packs_discovered": len(paths),
        "packs_benchmarked": len(rows),
        "packs_missing_full_frames": missing,
        "split_is_provisional": bool(provisional),
        "direct_config": direct_cfg.__dict__,
        "local_config": local_config_dict_v2212(local_cfg),
        "all": all_summary,
        "splits": splits,
        "map_gt_summary": map_summary,
        "rows": rows,
        "debug_written": debug_written,
        "gate": {
            "development_local_improves_oracle20": bool(splits["development"]["current_plus_local"]["oracle"]["20"] > splits["development"]["current"]["oracle"]["20"]),
            "confirmation_current_plus_local_oracle20_ge_070": bool(splits["confirmation"]["current_plus_local"]["oracle"]["20"] >= 0.70),
            "holdout_current_plus_local_oracle20_ge_070": bool(splits["holdout"]["current_plus_local"]["oracle"]["20"] >= 0.70),
            "eligible_for_live_authority": False,
        },
        "semantic_note": "GT is used only for post-hoc scoring/debug drawing. Local proposals are evidence-backed maxima near current candidates; no geometric GT padding is used.",
    }


def write_fullframe_report_v2212(path: Path, report: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
