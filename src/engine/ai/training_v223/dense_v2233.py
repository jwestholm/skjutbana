from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .proposal import PROPOSAL_ROOT
from .rich_v2233 import RICH_FEATURE_NAMES, discover_proposal_sessions, load_rich_matrix

CACHE_ROOT = Path("content/ai/training_v223/reducer_v2233/cache")

BASE_FEATURE_NAMES: tuple[str, ...] = (
    "dense_score",
    "dense_source_support",
    "dense_map_percentile_max",
    "dense_map_percentile_top3",
    "dense_map_percentile_mean",
    "dense_current_distance_clip100",
    "dense_current_distance_exp24",
    "dense_current_within20",
    "dense_current_within42",
    "dense_local_distance_clip100",
    "dense_local_distance_exp24",
    "dense_local_within20",
    "dense_local_within42",
    "dense_percentile_support",
)
REDUCER_FEATURE_NAMES: tuple[str, ...] = BASE_FEATURE_NAMES + RICH_FEATURE_NAMES


@dataclass(frozen=True)
class DenseShotRefV2233:
    session_id: str
    shot_id: str
    sequence: int
    proposal_path: Path
    cache_path: Path


@dataclass
class DenseShotV2233:
    session_id: str
    shot_id: str
    sequence: int
    xy: np.ndarray
    features: np.ndarray
    distances: np.ndarray
    baseline_score: np.ndarray
    gt_xy: tuple[float, float]

    @property
    def oracle20(self) -> bool:
        return bool(np.any(self.distances <= 20.0))

    @property
    def oracle42(self) -> bool:
        return bool(np.any(self.distances <= 42.0))


def _finite(v: Any, default: float = 0.0) -> float:
    try:
        out = float(v)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _candidate_feature(row: Mapping[str, Any], name: str) -> float:
    physical = row.get("physical_features")
    if isinstance(physical, Mapping) and name in physical:
        return _finite(physical.get(name))
    if name in row:
        return _finite(row.get(name))
    features = row.get("features")
    if isinstance(features, Mapping) and name in features:
        return _finite(features.get(name))
    return 0.0


def _candidate_baseline(row: Mapping[str, Any]) -> float:
    # Only a reproducible GT-free reference for mining/diagnostics.
    for key in ("baseline_score", "combined_score", "dense_score", "score"):
        if key in row:
            return _finite(row.get(key), 0.0)
    physical = row.get("physical_features")
    if isinstance(physical, Mapping):
        for key in ("dense_score", "dense_percentile_support", "dense_map_percentile_max"):
            if key in physical:
                return _finite(physical.get(key), 0.0)
    return 0.0


def _signature(path: Path) -> str:
    parts = ["v2233-cache-1", str(path.resolve())]
    for p in (path, path.with_name(path.stem + ".rich_v2233.npz"), path.with_name(path.stem + ".rich_v2233.json")):
        try:
            st = p.stat(); parts.extend([str(st.st_size), str(st.st_mtime_ns)])
        except Exception:
            parts.append("missing")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]


def _cache_path(proposal_path: Path, session_id: str, sequence: int) -> Path:
    sig = _signature(proposal_path)
    return CACHE_ROOT / session_id / f"shot_{sequence:06d}_{sig}.npz"


def compile_dense_shot(proposal_path: Path, *, force: bool = False) -> DenseShotRefV2233:
    proposal_path = Path(proposal_path)
    raw = json.loads(proposal_path.read_text(encoding="utf-8"))
    candidates = [x for x in raw.get("candidates", []) if isinstance(x, Mapping)]
    session_id = str(raw.get("session_id", proposal_path.parent.name))
    shot_id = str(raw.get("shot_id", proposal_path.stem))
    sequence = int(raw.get("sequence", 0) or 0)
    cache_path = _cache_path(proposal_path, session_id, sequence)
    if cache_path.exists() and not force:
        return DenseShotRefV2233(session_id, shot_id, sequence, proposal_path, cache_path)
    rich = load_rich_matrix(proposal_path, expected_count=len(candidates))
    if rich is None:
        raise FileNotFoundError(f"Missing V2.23.3 rich features for {proposal_path}")
    rich_names, rich_matrix = rich
    rich_index = {name: i for i, name in enumerate(rich_names)}
    n = len(candidates)
    xy = np.zeros((n, 2), dtype=np.float32)
    base = np.zeros((n, len(BASE_FEATURE_NAMES)), dtype=np.float32)
    baseline = np.zeros((n,), dtype=np.float32)
    for i, row in enumerate(candidates):
        xy[i, 0] = _finite(row.get("camera_x", row.get("x", 0.0)))
        xy[i, 1] = _finite(row.get("camera_y", row.get("y", 0.0)))
        baseline[i] = _candidate_baseline(row)
        for j, name in enumerate(BASE_FEATURE_NAMES):
            base[i, j] = _candidate_feature(row, name)
    rich_cols = []
    for name in RICH_FEATURE_NAMES:
        idx = rich_index.get(name)
        if idx is None:
            rich_cols.append(np.zeros((n, 1), dtype=np.float32))
        else:
            rich_cols.append(rich_matrix[:, idx:idx+1].astype(np.float32, copy=False))
    features = np.concatenate([base] + rich_cols, axis=1).astype(np.float32, copy=False)
    gt_raw = raw.get("gt_camera_xy", [0.0, 0.0])
    gt = np.asarray([_finite(gt_raw[0]), _finite(gt_raw[1])], dtype=np.float32)
    distances = np.sqrt(np.sum((xy - gt[None, :]) ** 2, axis=1)).astype(np.float32)
    features[~np.isfinite(features)] = 0.0
    baseline[~np.isfinite(baseline)] = 0.0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(
            fh,
            xy=xy,
            features=features,
            distances=distances,
            baseline_score=baseline,
            gt_xy=gt,
        )
    os.replace(tmp, cache_path)
    # Clean stale signatures for this shot.
    for old in cache_path.parent.glob(f"shot_{sequence:06d}_*.npz"):
        if old != cache_path:
            try: old.unlink()
            except Exception: pass
    return DenseShotRefV2233(session_id, shot_id, sequence, proposal_path, cache_path)


def load_dense_shot(ref: DenseShotRefV2233) -> DenseShotV2233:
    with np.load(ref.cache_path, allow_pickle=False) as data:
        xy = np.asarray(data["xy"], dtype=np.float32)
        features = np.asarray(data["features"], dtype=np.float32)
        distances = np.asarray(data["distances"], dtype=np.float32)
        baseline = np.asarray(data["baseline_score"], dtype=np.float32)
        gt = np.asarray(data["gt_xy"], dtype=np.float32)
    if features.shape[1] != len(REDUCER_FEATURE_NAMES):
        raise ValueError(f"Feature width mismatch in {ref.cache_path}: {features.shape[1]} != {len(REDUCER_FEATURE_NAMES)}")
    return DenseShotV2233(ref.session_id, ref.shot_id, ref.sequence, xy, features, distances, baseline, (float(gt[0]), float(gt[1])))


def discover_compilable_sessions(*, min_shots: int = 1) -> dict[str, list[Path]]:
    groups = discover_proposal_sessions(PROPOSAL_ROOT)
    out: dict[str, list[Path]] = {}
    for sid, paths in groups.items():
        ready = []
        for path in paths:
            rich_npz = path.with_name(path.stem + ".rich_v2233.npz")
            rich_json = path.with_name(path.stem + ".rich_v2233.json")
            if rich_npz.exists() and rich_json.exists():
                ready.append(path)
        if len(ready) >= int(min_shots):
            out[sid] = sorted(ready)
    return out


def compile_session(session_id: str | None = "latest", *, force: bool = False, min_shots: int = 1) -> dict[str, Any]:
    groups = discover_compilable_sessions(min_shots=min_shots)
    if not groups:
        return {"status": "no_rich_sessions", "processed": 0}
    if session_id in (None, "latest"):
        session_id = max(groups, key=lambda sid: max(p.stat().st_mtime for p in groups[sid]))
    selected = groups.get(str(session_id), [])
    refs: list[DenseShotRefV2233] = []; errors: list[str] = []
    for idx, path in enumerate(selected, start=1):
        try:
            ref = compile_dense_shot(path, force=force)
            refs.append(ref)
            if idx == 1 or idx == len(selected) or idx % 10 == 0:
                print(f"[V2.23.3 CACHE] {idx}/{len(selected)} shot={ref.shot_id} -> {ref.cache_path.name}")
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            print(f"[V2.23.3 CACHE] failed {path.name}: {type(exc).__name__}: {exc}")
    return {"status": "ok" if refs else "failed", "session_id": session_id, "processed": len(refs), "errors": errors}


def discover_cached_sessions(*, min_shots: int = 1) -> dict[str, list[DenseShotRefV2233]]:
    groups = discover_compilable_sessions(min_shots=min_shots)
    out: dict[str, list[DenseShotRefV2233]] = {}
    for sid, paths in groups.items():
        refs = []
        for path in paths:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                seq = int(raw.get("sequence", 0) or 0)
                shot_id = str(raw.get("shot_id", path.stem))
                cache = _cache_path(path, sid, seq)
                if cache.exists():
                    refs.append(DenseShotRefV2233(sid, shot_id, seq, path, cache))
            except Exception:
                continue
        if len(refs) >= min_shots:
            out[sid] = sorted(refs, key=lambda r: r.sequence)
    return out
