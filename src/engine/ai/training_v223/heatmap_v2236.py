from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .dense_v2233 import DenseShotRefV2233, discover_cached_sessions, load_dense_shot
from .evidence_patch_v2235 import EVIDENCE_CHANNEL_NAMES, build_registered_evidence_channels
from .framepack import load_framepack

HEATMAP_ROOT = Path("content/ai/training_v223/heatmap_v2236")
HEATMAP_CACHE_ROOT = HEATMAP_ROOT / "cache"
HEATMAP_SCHEMA_VERSION = "2.23.6-heatmap-cache-1"
HEATMAP_STRIDE = 4
HEATMAP_CHANNELS = len(EVIDENCE_CHANNEL_NAMES)


@dataclass(frozen=True)
class HeatmapShotRefV2236:
    session_id: str
    shot_id: str
    sequence: int
    dense_ref: DenseShotRefV2233
    cache_path: Path


@dataclass
class HeatmapShotV2236:
    ref: HeatmapShotRefV2236
    maps: np.ndarray          # uint8 [C,Hc,Wc]
    gt_xy: tuple[float, float]
    full_shape: tuple[int, int]
    dense_xy: np.ndarray
    dense_distances: np.ndarray


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _resolve_framepack(ref: DenseShotRefV2233) -> Path:
    raw = json.loads(ref.proposal_path.read_text(encoding="utf-8"))
    p = Path(str(raw.get("source_framepack", "") or ""))
    if p.exists():
        return p
    p2 = Path.cwd() / p
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Framepack not found for {ref.proposal_path}: {p}")


def _signature(ref: DenseShotRefV2233, framepack: Path) -> str:
    parts = [HEATMAP_SCHEMA_VERSION, str(HEATMAP_STRIDE), str(tuple(EVIDENCE_CHANNEL_NAMES))]
    for p in (ref.proposal_path, ref.cache_path, framepack, framepack.with_suffix(".npz")):
        try:
            st = p.stat()
            parts.extend([str(p.resolve()), str(st.st_size), str(st.st_mtime_ns)])
        except Exception:
            parts.extend([str(p), "missing"])
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]


def downsample_block_mean(channels: np.ndarray, stride: int = HEATMAP_STRIDE) -> np.ndarray:
    arr = np.asarray(channels, dtype=np.uint8)
    if arr.ndim != 3:
        raise ValueError(f"channels must be [C,H,W], got {arr.shape}")
    c, h, w = arr.shape
    s = max(1, int(stride))
    hh = h // s
    ww = w // s
    if hh <= 0 or ww <= 0:
        raise ValueError(f"frame too small for stride {s}: {arr.shape}")
    trimmed = arr[:, : hh * s, : ww * s].astype(np.float32)
    pooled = trimmed.reshape(c, hh, s, ww, s).mean(axis=(2, 4))
    return np.rint(pooled).clip(0, 255).astype(np.uint8)


def coarse_to_camera(x: float, y: float, stride: int = HEATMAP_STRIDE) -> tuple[float, float]:
    s = float(stride)
    offset = (s - 1.0) * 0.5
    return float(x) * s + offset, float(y) * s + offset


def camera_to_coarse(x: float, y: float, stride: int = HEATMAP_STRIDE) -> tuple[float, float]:
    s = float(stride)
    offset = (s - 1.0) * 0.5
    return (float(x) - offset) / s, (float(y) - offset) / s


def compile_heatmap_shot(ref: DenseShotRefV2233, *, force: bool = False) -> HeatmapShotRefV2236:
    framepack = _resolve_framepack(ref)
    sig = _signature(ref, framepack)
    out_dir = HEATMAP_CACHE_ROOT / ref.session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"shot_{ref.sequence:06d}_{sig}.npz"
    meta_path = cache_path.with_suffix(".json")
    if cache_path.exists() and meta_path.exists() and not force:
        return HeatmapShotRefV2236(ref.session_id, ref.shot_id, ref.sequence, ref, cache_path)

    dense = load_dense_shot(ref)
    _, pre, posts, _ = load_framepack(framepack)
    t0 = time.perf_counter()
    channels, channel_meta = build_registered_evidence_channels(pre, posts)
    maps = downsample_block_mean(channels)
    gt = np.asarray(dense.gt_xy, dtype=np.float32)

    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(
            fh,
            maps=maps,
            gt_xy=gt,
            full_shape=np.asarray(pre.shape[:2], dtype=np.int32),
            dense_xy=np.asarray(dense.xy, dtype=np.float32),
            dense_distances=np.asarray(dense.distances, dtype=np.float32),
        )
    os.replace(tmp, cache_path)
    _atomic_json(meta_path, {
        "schema_version": HEATMAP_SCHEMA_VERSION,
        "session_id": ref.session_id,
        "shot_id": ref.shot_id,
        "sequence": ref.sequence,
        "stride": HEATMAP_STRIDE,
        "channel_names": list(EVIDENCE_CHANNEL_NAMES),
        "coarse_shape": [int(maps.shape[1]), int(maps.shape[2])],
        "full_shape": [int(pre.shape[0]), int(pre.shape[1])],
        "source_framepack": str(framepack),
        "dense_cache": str(ref.cache_path),
        "gt_used_for_map_generation": False,
        "gt_used_only_for_training_and_metrics": True,
        "channel_meta": channel_meta,
        "runtime_ms": (time.perf_counter() - t0) * 1000.0,
        "live_authority": False,
    })
    for old in out_dir.glob(f"shot_{ref.sequence:06d}_*.npz"):
        if old != cache_path:
            try:
                old.unlink()
                old.with_suffix(".json").unlink(missing_ok=True)
            except Exception:
                pass
    return HeatmapShotRefV2236(ref.session_id, ref.shot_id, ref.sequence, ref, cache_path)


def load_heatmap_shot(ref: HeatmapShotRefV2236) -> HeatmapShotV2236:
    with np.load(ref.cache_path, allow_pickle=False) as data:
        maps = np.asarray(data["maps"], dtype=np.uint8)
        gt = np.asarray(data["gt_xy"], dtype=np.float32)
        shape = np.asarray(data["full_shape"], dtype=np.int32)
        dense_xy = np.asarray(data["dense_xy"], dtype=np.float32)
        dense_distances = np.asarray(data["dense_distances"], dtype=np.float32)
    if maps.ndim != 3 or maps.shape[0] != HEATMAP_CHANNELS:
        raise ValueError(f"Invalid heatmap shape {maps.shape} in {ref.cache_path}")
    return HeatmapShotV2236(
        ref=ref,
        maps=maps,
        gt_xy=(float(gt[0]), float(gt[1])),
        full_shape=(int(shape[0]), int(shape[1])),
        dense_xy=dense_xy,
        dense_distances=dense_distances,
    )


def discover_heatmap_sessions(*, min_shots: int = 1) -> dict[str, list[HeatmapShotRefV2236]]:
    groups = discover_cached_sessions(min_shots=min_shots)
    out: dict[str, list[HeatmapShotRefV2236]] = {}
    for sid, dense_refs in groups.items():
        refs: list[HeatmapShotRefV2236] = []
        for dense_ref in dense_refs:
            try:
                framepack = _resolve_framepack(dense_ref)
                sig = _signature(dense_ref, framepack)
                path = HEATMAP_CACHE_ROOT / sid / f"shot_{dense_ref.sequence:06d}_{sig}.npz"
                if path.exists() and path.with_suffix(".json").exists():
                    refs.append(HeatmapShotRefV2236(sid, dense_ref.shot_id, dense_ref.sequence, dense_ref, path))
            except Exception:
                continue
        if len(refs) >= int(min_shots):
            out[sid] = sorted(refs, key=lambda r: r.sequence)
    return out


def prepare_heatmap_sessions(*, session: str | None = None, force: bool = False, min_session_shots: int = 50) -> dict[str, Any]:
    groups = discover_cached_sessions(min_shots=min_session_shots)
    if not groups:
        return {"status": "no_substantial_dense_sessions", "sessions": {}}
    if session not in (None, "all"):
        if session == "latest":
            session = max(groups, key=lambda sid: max(r.proposal_path.stat().st_mtime for r in groups[sid]))
        groups = {str(session): groups.get(str(session), [])} if str(session) in groups else {}
    reports: dict[str, Any] = {}
    for sid, refs in groups.items():
        print(f"[V2.23.6 PREP] heatmap session={sid} shots={len(refs)}")
        processed = cached = 0
        errors: list[str] = []
        bytes_total = 0
        for idx, ref in enumerate(refs, start=1):
            try:
                framepack = _resolve_framepack(ref)
                sig = _signature(ref, framepack)
                expected = HEATMAP_CACHE_ROOT / sid / f"shot_{ref.sequence:06d}_{sig}.npz"
                was_cached = expected.exists() and expected.with_suffix(".json").exists() and not force
                h_ref = compile_heatmap_shot(ref, force=force)
                processed += 1
                cached += int(was_cached)
                try: bytes_total += h_ref.cache_path.stat().st_size
                except Exception: pass
                if idx == 1 or idx == len(refs) or idx % 5 == 0:
                    print(
                        f"[V2.23.6 MAP] {idx}/{len(refs)} shot={ref.shot_id} "
                        f"cache={was_cached} file={h_ref.cache_path.stat().st_size/1024/1024:.1f}MB"
                    )
            except Exception as exc:
                errors.append(f"{ref.proposal_path}: {type(exc).__name__}: {exc}")
                print(f"[V2.23.6 MAP] failed shot={ref.shot_id}: {type(exc).__name__}: {exc}")
        reports[sid] = {
            "processed": processed,
            "cached": cached,
            "cache_mb": bytes_total / 1024.0 / 1024.0,
            "errors": errors,
        }
    return {"status": "ok", "sessions": reports}
