from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .framepack import FRAMEPACK_ROOT, discover_framepacks, load_framepack
from .schema import CandidateTrainingRow

PROPOSAL_ROOT = Path("content/ai/training_v223/proposals_v2232")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _finite_xy(row: Mapping[str, Any]) -> tuple[float, float] | None:
    try:
        x = float(row.get("camera_x", row.get("x", 0.0)))
        y = float(row.get("camera_y", row.get("y", 0.0)))
    except Exception:
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def _nearest(candidates: Sequence[Mapping[str, Any]], gt: tuple[float, float]) -> float:
    values = []
    for row in candidates:
        xy = _finite_xy(row)
        if xy is not None:
            values.append(math.hypot(xy[0] - gt[0], xy[1] - gt[1]))
    return min(values) if values else float("inf")


def _dedupe_union(named: Sequence[tuple[str, Sequence[Mapping[str, Any]]]], radius_px: float = 1.5) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cell = max(0.5, float(radius_px))
    grid: dict[tuple[int, int], list[int]] = {}
    r2 = radius_px * radius_px
    for source, rows in named:
        for row in rows:
            xy = _finite_xy(row)
            if xy is None:
                continue
            x, y = xy
            gx = int(math.floor(x / cell)); gy = int(math.floor(y / cell))
            found = None
            for yy in range(gy - 1, gy + 2):
                for xx in range(gx - 1, gx + 2):
                    for idx in grid.get((xx, yy), []):
                        old = out[idx]
                        if (x - float(old["camera_x"])) ** 2 + (y - float(old["camera_y"])) ** 2 <= r2:
                            found = idx; break
                    if found is not None: break
                if found is not None: break
            if found is None:
                item = dict(row)
                item["camera_x"] = x; item["camera_y"] = y
                item["provenance"] = sorted(set([source] + list(item.get("provenance", []) if isinstance(item.get("provenance"), list) else [])))
                out.append(item)
                grid.setdefault((gx, gy), []).append(len(out) - 1)
            else:
                existing = out[found]
                prov = existing.get("provenance", [])
                if not isinstance(prov, list):
                    prov = []
                if source not in prov:
                    prov.append(source)
                existing["provenance"] = sorted(set(str(value) for value in prov))

                # Preserve physical evidence when two proposal families nominate
                # essentially the same coordinate.  Current/local rows are added
                # before dense rows, so without this merge the strongest dense
                # evidence would silently disappear on duplicate coordinates.
                incoming_physical = row.get("physical_features")
                if isinstance(incoming_physical, Mapping):
                    physical = dict(existing.get("physical_features", {}) or {})
                    for key, value in incoming_physical.items():
                        try:
                            physical[str(key)] = float(value)
                        except Exception:
                            continue
                    if physical:
                        existing["physical_features"] = physical
                for key, value in row.items():
                    if str(key).startswith("dense_") and key not in existing:
                        existing[key] = value
    return out


def _attach_dense_features(candidates: list[dict[str, Any]], feature_batch: Any) -> None:
    names = tuple(getattr(feature_batch, "feature_names", ()))
    matrix = np.asarray(getattr(feature_batch, "matrix", np.zeros((0, 0))), dtype=np.float32)
    if matrix.shape[0] != len(candidates):
        return
    wanted = {
        "dense_source_support": "dense_source_support",
        "dense_score": "dense_score",
        "map_percentile:max": "dense_map_percentile_max",
        "map_percentile:top3_mean": "dense_map_percentile_top3",
        "map_percentile:mean": "dense_map_percentile_mean",
        "distance:current:clip100": "dense_current_distance_clip100",
        "distance:current:exp24": "dense_current_distance_exp24",
        "distance:current:within20": "dense_current_within20",
        "distance:current:within42": "dense_current_within42",
        "distance:v2212_local:clip100": "dense_local_distance_clip100",
        "distance:v2212_local:exp24": "dense_local_distance_exp24",
        "distance:v2212_local:within20": "dense_local_within20",
        "distance:v2212_local:within42": "dense_local_within42",
        "interaction_pct:max*support": "dense_percentile_support",
    }
    indices = {name: idx for idx, name in enumerate(names)}
    for ridx, candidate in enumerate(candidates):
        physical = dict(candidate.get("physical_features", {}) or {})
        for old, new in wanted.items():
            idx = indices.get(old)
            if idx is not None:
                physical[new] = float(matrix[ridx, idx])
        candidate["physical_features"] = physical


def expand_framepack(path: Path, *, force: bool = False) -> dict[str, Any]:
    path = Path(path)
    meta, pre, posts, _ = load_framepack(path)
    session_id = str(meta.get("session_id", path.parent.name))
    seq = int(meta.get("sequence", int(path.stem.split("_")[-1])))
    out_dir = PROPOSAL_ROOT / session_id
    out_path = out_dir / f"shot_{seq:06d}.json"
    if out_path.exists() and not force:
        try:
            cached = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict):
                cached["cache"] = True
                return cached
        except Exception:
            pass

    # Heavy imports stay offline; installing V2.23.2 never changes live authority.
    from src.engine.offline.direct_proposal_v221 import DirectProposalConfigV221, propose_direct_v221
    from src.engine.offline.temporal_local_v2212 import LocalTemporalConfigV2212, propose_local_temporal_v2212
    from src.engine.offline.physical_dense_v2215 import DensePoolConfigV2215, extract_candidate_features_v2215, propose_dense_pool_v2215

    current = [dict(x) for x in meta.get("current_candidates", []) if isinstance(x, Mapping)]
    t0 = time.perf_counter()
    direct = propose_direct_v221([pre], [np.asarray(p, dtype=np.uint8) for p in posts], config=DirectProposalConfigV221())
    maps = dict(direct.maps)
    maps["fused"] = np.asarray(direct.fused, dtype=np.float32)
    local = propose_local_temporal_v2212(current, direct.maps, direct.fused, config=LocalTemporalConfigV2212())
    dense = propose_dense_pool_v2215(current, maps, config=DensePoolConfigV2215())
    dense_candidates = [dict(x) for x in dense.candidates]
    dense_features = extract_candidate_features_v2215(
        dense_candidates, maps, dense.target_mask, current_candidates=current, local_candidates=local,
    )
    _attach_dense_features(dense_candidates, dense_features)
    union = _dedupe_union((
        ("v2232_current", current),
        ("v2232_local", local),
        ("v2232_dense_v2215", dense_candidates),
    ))
    gt_raw = meta.get("gt_camera_xy", [0.0, 0.0])
    gt = (float(gt_raw[0]), float(gt_raw[1]))
    nearest = {name: _nearest(rows, gt) for name, rows in (
        ("current", current), ("local", local), ("dense", dense_candidates), ("union", union),
    )}
    payload = {
        "schema_version": "2.23.2-proposals-1",
        "session_id": session_id,
        "shot_id": str(meta.get("shot_id", seq)),
        "sequence": seq,
        "source_framepack": str(path),
        "generated_at": time.time(),
        "gt_camera_xy": [gt[0], gt[1]],
        "gt_used_for_proposal_generation": False,
        "counts": {"current": len(current), "local": len(local), "dense": len(dense_candidates), "union": len(union)},
        "nearest": nearest,
        "oracle": {
            str(radius): {name: bool(value <= radius) for name, value in nearest.items()}
            for radius in (5, 10, 20, 42)
        },
        "dense_metadata": dict(dense.metadata),
        "runtime_ms": (time.perf_counter() - t0) * 1000.0,
        "candidates": union,
        "authority": "offline_shadow_only",
    }
    _atomic_json(out_path, payload)
    return payload


def expand_session(session_id: str | None = None, *, force: bool = False, limit: int | None = None) -> dict[str, Any]:
    paths = discover_framepacks()
    if not paths:
        return {"status": "no_framepacks", "processed": 0}
    groups: dict[str, list[Path]] = {}
    for path in paths:
        groups.setdefault(path.parent.name, []).append(path)
    if session_id in (None, "latest"):
        session_id = max(groups, key=lambda sid: max(p.stat().st_mtime for p in groups[sid]))
    selected = groups.get(str(session_id), [])
    if limit is not None:
        selected = selected[: max(0, int(limit))]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for idx, path in enumerate(selected, start=1):
        try:
            result = expand_framepack(path, force=force)
            rows.append(result)
            print(
                f"[V2.23.2 PROPOSAL] {idx}/{len(selected)} shot={result.get('shot_id')} "
                f"dense={result.get('counts',{}).get('dense',0)} union={result.get('counts',{}).get('union',0)} "
                f"near20={result.get('oracle',{}).get('20',{}).get('union',False)} "
                f"nearest={float(result.get('nearest',{}).get('union',9999)):.1f}px"
            )
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            print(f"[V2.23.2 PROPOSAL] failed {path.name}: {type(exc).__name__}: {exc}")
    def rate(radius: str, pool: str) -> float:
        vals = [bool(r.get("oracle", {}).get(radius, {}).get(pool, False)) for r in rows]
        return sum(vals) / len(vals) if vals else 0.0
    summary = {
        "status": "ok" if rows else "failed",
        "session_id": session_id,
        "processed": len(rows),
        "errors": errors,
        "oracle20": {pool: rate("20", pool) for pool in ("current", "local", "dense", "union")},
        "oracle42": {pool: rate("42", pool) for pool in ("current", "local", "dense", "union")},
        "mean_candidates": {
            pool: (float(np.mean([r.get("counts", {}).get(pool, 0) for r in rows])) if rows else 0.0)
            for pool in ("current", "local", "dense", "union")
        },
    }
    report_path = PROPOSAL_ROOT / str(session_id) / "summary.json"
    _atomic_json(report_path, summary)
    return summary
