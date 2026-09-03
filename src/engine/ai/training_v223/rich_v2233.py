from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .framepack import load_framepack
from .proposal import PROPOSAL_ROOT

RICH_SCHEMA_VERSION = "2.23.3-rich-1"
RICH_FEATURE_NAMES: tuple[str, ...] = (
    "v2233_abs_r2",
    "v2233_abs_r5",
    "v2233_abs_r10",
    "v2233_abs_peak_r2",
    "v2233_dark_r2",
    "v2233_dark_r5",
    "v2233_bright_r2",
    "v2233_bright_r5",
    "v2233_signed_r2",
    "v2233_persist_r2",
    "v2233_persist_r5",
    "v2233_temporal_std_r2",
    "v2233_pre_std_r5",
    "v2233_post_std_r5",
    "v2233_edge_gain_r3",
    "v2233_center_ring_abs",
    "v2233_center_ring_dark",
    "v2233_small_large_ratio",
    "v2233_dark_fraction",
    "v2233_newhole_heuristic",
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _sidecar_rich_paths(proposal_path: Path) -> tuple[Path, Path]:
    stem = proposal_path.stem
    return proposal_path.with_name(stem + ".rich_v2233.npz"), proposal_path.with_name(stem + ".rich_v2233.json")


def _candidate_xy(candidates: Sequence[Mapping[str, Any]], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    xs = np.empty(len(candidates), dtype=np.int32)
    ys = np.empty(len(candidates), dtype=np.int32)
    for i, row in enumerate(candidates):
        try:
            x = int(round(float(row.get("camera_x", row.get("x", 0.0)))))
            y = int(round(float(row.get("camera_y", row.get("y", 0.0)))))
        except Exception:
            x = y = 0
        xs[i] = max(0, min(w - 1, x))
        ys[i] = max(0, min(h - 1, y))
    return xs, ys


def _box_map(src: np.ndarray, radius: int) -> np.ndarray:
    k = max(1, int(radius) * 2 + 1)
    try:
        import cv2
        return cv2.boxFilter(src.astype(np.float32, copy=False), ddepth=-1, ksize=(k, k), normalize=True, borderType=cv2.BORDER_REFLECT)
    except Exception:
        # Integral-image fallback.  This is slower but keeps the offline tool portable.
        a = src.astype(np.float32, copy=False)
        pad = int(radius)
        p = np.pad(a, ((pad, pad), (pad, pad)), mode="reflect")
        integral = np.pad(np.cumsum(np.cumsum(p, axis=0), axis=1), ((1, 0), (1, 0)))
        y0 = np.arange(a.shape[0]); y1 = y0 + k
        x0 = np.arange(a.shape[1]); x1 = x0 + k
        out = integral[y1[:, None], x1[None, :]] - integral[y0[:, None], x1[None, :]] - integral[y1[:, None], x0[None, :]] + integral[y0[:, None], x0[None, :]]
        return out / float(k * k)


def _local_std_map(src: np.ndarray, radius: int) -> np.ndarray:
    mean = _box_map(src, radius)
    mean2 = _box_map(src.astype(np.float32, copy=False) ** 2, radius)
    return np.sqrt(np.maximum(0.0, mean2 - mean * mean))


def _lap_abs(src: np.ndarray) -> np.ndarray:
    try:
        import cv2
        return np.abs(cv2.Laplacian(src.astype(np.float32, copy=False), cv2.CV_32F, ksize=3))
    except Exception:
        a = src.astype(np.float32, copy=False)
        up = np.roll(a, 1, axis=0); down = np.roll(a, -1, axis=0)
        left = np.roll(a, 1, axis=1); right = np.roll(a, -1, axis=1)
        return np.abs((up + down + left + right) - 4.0 * a)


def compute_rich_feature_matrix(
    pre: np.ndarray,
    posts: Sequence[np.ndarray],
    candidates: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Compute GT-free PRE->POST evidence for many candidate coordinates.

    The implementation is deliberately memory-bounded for 4K frames. It keeps
    only five float32 accumulators, derives dark/bright evidence algebraically,
    and samples each local map before releasing it. Candidate extraction is
    indexed sampling rather than thousands of independent crops.
    """
    pre_u8 = np.asarray(pre, dtype=np.uint8)
    post_list = [np.asarray(p, dtype=np.uint8) for p in posts if np.asarray(p).shape == pre_u8.shape]
    if not post_list:
        raise ValueError("Need at least one POST frame matching PRE")
    h, w = pre_u8.shape[:2]
    xs, ys = _candidate_xy(candidates, (h, w))
    n = len(candidates)
    matrix = np.zeros((n, len(RICH_FEATURE_NAMES)), dtype=np.float32)
    if n == 0:
        return matrix

    pre_f = pre_u8.astype(np.float32)
    count = float(len(post_list))
    abs_mean = np.zeros_like(pre_f)
    abs_peak = np.zeros_like(pre_f)
    signed_mean = np.zeros_like(pre_f)
    temporal_std = np.zeros_like(pre_f)  # accumulates squared signed delta first
    persistence = np.zeros_like(pre_f)

    for post in post_list:
        # One signed float32 temporary; abs is reused immediately.
        d = post.astype(np.float32) - pre_f
        ad = np.abs(d)
        abs_mean += ad
        np.maximum(abs_peak, ad, out=abs_peak)
        signed_mean += d
        temporal_std += d * d
        persistence += (ad >= 2.5)
        del d, ad

    abs_mean /= count
    signed_mean /= count
    temporal_std /= count
    temporal_std -= signed_mean * signed_mean
    np.maximum(temporal_std, 0.0, out=temporal_std)
    np.sqrt(temporal_std, out=temporal_std)
    persistence /= count

    cols: dict[str, np.ndarray] = {}

    def sample_box(name: str, src: np.ndarray, radius: int) -> None:
        local = _box_map(src, radius)
        cols[name] = local[ys, xs].astype(np.float32, copy=True)
        del local

    sample_box("v2233_abs_r2", abs_mean, 2)
    sample_box("v2233_abs_r5", abs_mean, 5)
    sample_box("v2233_abs_r10", abs_mean, 10)
    sample_box("v2233_abs_peak_r2", abs_peak, 2)
    del abs_peak

    # mean(max(-d,0)) == (mean(abs(d)) - mean(d))/2, so separate
    # 4K dark/bright accumulators are unnecessary.
    dark_mean = (abs_mean - signed_mean) * 0.5
    np.maximum(dark_mean, 0.0, out=dark_mean)
    sample_box("v2233_dark_r2", dark_mean, 2)
    sample_box("v2233_dark_r5", dark_mean, 5)
    dark_large = _box_map(dark_mean, 10)[ys, xs].astype(np.float32, copy=True)

    bright_mean = (abs_mean + signed_mean) * 0.5
    np.maximum(bright_mean, 0.0, out=bright_mean)
    sample_box("v2233_bright_r2", bright_mean, 2)
    sample_box("v2233_bright_r5", bright_mean, 5)
    del bright_mean

    sample_box("v2233_signed_r2", signed_mean, 2)
    sample_box("v2233_persist_r2", persistence, 2)
    sample_box("v2233_persist_r5", persistence, 5)
    sample_box("v2233_temporal_std_r2", temporal_std, 2)
    del persistence, temporal_std

    pre_std = _local_std_map(pre_f, 5)
    cols["v2233_pre_std_r5"] = pre_std[ys, xs].astype(np.float32, copy=True)
    del pre_std
    post_mean = pre_f + signed_mean
    post_std = _local_std_map(post_mean, 5)
    cols["v2233_post_std_r5"] = post_std[ys, xs].astype(np.float32, copy=True)
    del post_std

    edge_pre = _box_map(_lap_abs(pre_f), 3)
    edge_post = _box_map(_lap_abs(post_mean), 3)
    cols["v2233_edge_gain_r3"] = (edge_post[ys, xs] - edge_pre[ys, xs]).astype(np.float32, copy=True)
    del edge_pre, edge_post, post_mean, pre_f

    abs_small = cols["v2233_abs_r2"]
    abs_large = cols["v2233_abs_r10"]
    dark_small = cols["v2233_dark_r2"]
    cols["v2233_center_ring_abs"] = (abs_small - abs_large).astype(np.float32)
    cols["v2233_center_ring_dark"] = (dark_small - dark_large).astype(np.float32)
    cols["v2233_small_large_ratio"] = (abs_small / (abs_large + 0.75)).astype(np.float32)
    cols["v2233_dark_fraction"] = (dark_small / (abs_small + 0.75)).astype(np.float32)
    del dark_large, dark_mean, abs_mean, signed_mean

    compact = np.maximum(cols["v2233_center_ring_abs"], 0.0)
    persistent = np.clip(cols["v2233_persist_r2"], 0.0, 1.0)
    sign_pref = np.maximum(cols["v2233_dark_fraction"], 0.15)
    cols["v2233_newhole_heuristic"] = (
        np.log1p(np.maximum(abs_small, 0.0))
        * (0.35 + 0.65 * persistent)
        * (0.5 + 0.5 * np.clip(sign_pref, 0.0, 1.5))
        * (1.0 + 0.12 * np.log1p(compact))
    ).astype(np.float32)

    for j, name in enumerate(RICH_FEATURE_NAMES):
        values = np.asarray(cols[name], dtype=np.float32)
        values[~np.isfinite(values)] = 0.0
        matrix[:, j] = values
    return matrix

def enrich_proposal_sidecar(proposal_path: Path, *, force: bool = False) -> dict[str, Any]:
    proposal_path = Path(proposal_path)
    rich_npz, rich_json = _sidecar_rich_paths(proposal_path)
    if rich_npz.exists() and rich_json.exists() and not force:
        try:
            meta = json.loads(rich_json.read_text(encoding="utf-8"))
            meta["cache"] = True
            return meta
        except Exception:
            pass
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    candidates = [x for x in proposal.get("candidates", []) if isinstance(x, Mapping)]
    framepack_path = Path(str(proposal.get("source_framepack", "")))
    if not framepack_path.exists():
        # Proposal sidecars store repo-relative framepack paths.
        candidate = Path.cwd() / framepack_path
        if candidate.exists():
            framepack_path = candidate
    if not framepack_path.exists():
        raise FileNotFoundError(f"Framepack not found for {proposal_path}: {proposal.get('source_framepack')}")
    _, pre, posts, _ = load_framepack(framepack_path)
    t0 = time.perf_counter()
    matrix = compute_rich_feature_matrix(pre, posts, candidates)
    rich_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp_npz = rich_npz.with_suffix(rich_npz.suffix + ".tmp")
    with tmp_npz.open("wb") as fh:
        np.savez_compressed(fh, features=matrix)
    os.replace(tmp_npz, rich_npz)
    meta = {
        "schema_version": RICH_SCHEMA_VERSION,
        "proposal_path": str(proposal_path),
        "framepack_path": str(framepack_path),
        "session_id": str(proposal.get("session_id", proposal_path.parent.name)),
        "shot_id": str(proposal.get("shot_id", proposal_path.stem)),
        "sequence": int(proposal.get("sequence", 0) or 0),
        "candidate_count": int(len(candidates)),
        "feature_names": list(RICH_FEATURE_NAMES),
        "gt_used_for_feature_generation": False,
        "runtime_ms": (time.perf_counter() - t0) * 1000.0,
        "created_at": time.time(),
    }
    _atomic_json(rich_json, meta)
    return meta


def discover_proposal_sessions(root: Path = PROPOSAL_ROOT) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    if not root.exists():
        return groups
    for path in sorted(root.glob("*/shot_*.json")):
        if path.name.endswith(".rich_v2233.json") or path.name == "summary.json":
            continue
        groups.setdefault(path.parent.name, []).append(path)
    return groups


def enrich_session(session_id: str | None = "latest", *, force: bool = False, limit: int | None = None) -> dict[str, Any]:
    groups = discover_proposal_sessions()
    if not groups:
        return {"status": "no_proposals", "processed": 0}
    if session_id in (None, "latest"):
        session_id = max(groups, key=lambda sid: max(p.stat().st_mtime for p in groups[sid]))
    selected = groups.get(str(session_id), [])
    if limit is not None:
        selected = selected[: max(0, int(limit))]
    ok = 0; cached = 0; errors: list[str] = []; runtimes: list[float] = []
    for idx, path in enumerate(selected, start=1):
        try:
            result = enrich_proposal_sidecar(path, force=force)
            ok += 1
            cached += int(bool(result.get("cache")))
            runtimes.append(float(result.get("runtime_ms", 0.0) or 0.0))
            if idx == 1 or idx == len(selected) or idx % 5 == 0:
                print(
                    f"[V2.23.3 RICH] {idx}/{len(selected)} shot={result.get('shot_id')} "
                    f"candidates={result.get('candidate_count',0)} cache={bool(result.get('cache'))} "
                    f"time={float(result.get('runtime_ms',0.0)):.0f}ms"
                )
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            print(f"[V2.23.3 RICH] failed {path.name}: {type(exc).__name__}: {exc}")
    return {
        "status": "ok" if ok else "failed",
        "session_id": session_id,
        "processed": ok,
        "cached": cached,
        "errors": errors,
        "mean_runtime_ms": float(np.mean(runtimes)) if runtimes else 0.0,
    }


def load_rich_matrix(proposal_path: Path, *, expected_count: int | None = None) -> tuple[tuple[str, ...], np.ndarray] | None:
    rich_npz, rich_json = _sidecar_rich_paths(Path(proposal_path))
    if not rich_npz.exists() or not rich_json.exists():
        return None
    meta = json.loads(rich_json.read_text(encoding="utf-8"))
    names = tuple(str(x) for x in meta.get("feature_names", []))
    with np.load(rich_npz, allow_pickle=False) as data:
        matrix = np.asarray(data["features"], dtype=np.float32)
    if expected_count is not None and matrix.shape[0] != int(expected_count):
        raise ValueError(f"Rich feature count mismatch: {matrix.shape[0]} != {expected_count}")
    if matrix.shape[1] != len(names):
        raise ValueError("Rich feature column/name mismatch")
    return names, matrix
