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

from .dense_v2233 import DenseShotRefV2233, DenseShotV2233, discover_cached_sessions, load_dense_shot
from .framepack import load_framepack

EVIDENCE_ROOT = Path("content/ai/training_v223/evidence_v2235")
EVIDENCE_CACHE_ROOT = EVIDENCE_ROOT / "cache"
EVIDENCE_SCHEMA_VERSION = "2.23.5-evidence-patch-1"

# These are the same registered/photometrically compensated physical maps that
# feed the V2.21.5 dense proposal engine.  V2.23.5 learns spatial structure in
# these maps instead of asking a tiny network to rediscover registration and
# scene compensation from raw PRE/POST pixels.
EVIDENCE_CHANNEL_NAMES: tuple[str, ...] = (
    "blackhat_gain",
    "tophat_gain",
    "persistent_abs",
    "gradient_gain",
    "persistent_dark",
    "persistent_bright",
    "fused",
    "compact_change",
)
EVIDENCE_CROP_SIZE = 27
EVIDENCE_PATCH_SIZE = 9
EVIDENCE_CHANNELS = len(EVIDENCE_CHANNEL_NAMES)

# Patch-learning label contract.  <=6px means the physical event is actually
# centred enough to be visible as the candidate's local pattern.  The wide
# 6..42px band is neutral, never a negative.  Final localisation metrics remain
# measured at <=20px.
PATCH_POSITIVE_RADIUS_PX = 6.0
PATCH_NEGATIVE_RADIUS_PX = 42.0
GT_ANCHOR_OFFSETS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (-2.0, 0.0), (2.0, 0.0), (0.0, -2.0), (0.0, 2.0),
    (-3.0, -3.0), (-3.0, 3.0), (3.0, -3.0), (3.0, 3.0),
    (-6.0, 0.0), (6.0, 0.0), (0.0, -6.0), (0.0, 6.0),
)


@dataclass(frozen=True)
class EvidenceShotRefV2235:
    session_id: str
    shot_id: str
    sequence: int
    dense_ref: DenseShotRefV2233
    cache_path: Path


@dataclass
class EvidenceShotV2235:
    ref: EvidenceShotRefV2235
    dense: DenseShotV2233
    patches: np.ndarray       # uint8 [N,C,9,9]
    anchors: np.ndarray       # uint8 [A,C,9,9], training-only
    anchor_distances: np.ndarray


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _resolve_framepack(proposal_path: Path, proposal: Mapping[str, Any]) -> Path:
    raw = str(proposal.get("source_framepack", "") or "")
    p = Path(raw)
    if p.exists():
        return p
    p2 = Path.cwd() / p
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Framepack not found for {proposal_path}: {raw}")


def _signature(ref: DenseShotRefV2233, framepack: Path) -> str:
    parts = [EVIDENCE_SCHEMA_VERSION, str(EVIDENCE_CROP_SIZE), str(EVIDENCE_PATCH_SIZE)]
    for p in (ref.proposal_path, ref.cache_path, framepack, framepack.with_suffix(".npz")):
        try:
            st = p.stat()
            parts.extend([str(p.resolve()), str(st.st_size), str(st.st_mtime_ns)])
        except Exception:
            parts.extend([str(p), "missing"])
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]


def _normalise_map(src: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Robust per-shot map normalisation without GT.

    Dense maps are non-negative physical evidence with very different native
    scales.  We retain spatial shape and map-relative peak strength by mapping
    the shot median to 0 and the 99.5th percentile to roughly mid/high range.
    Stronger values may saturate at 2x that span.
    """
    a = np.asarray(src, dtype=np.float32)
    a = np.where(np.isfinite(a), a, 0.0).astype(np.float32, copy=False)
    sample = a[::4, ::4] if a.ndim == 2 else a.reshape(-1)[::16]
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return np.zeros_like(a, dtype=np.uint8), {"q50": 0.0, "q995": 1.0}
    q50 = float(np.percentile(finite, 50.0))
    q995 = float(np.percentile(finite, 99.5))
    span = max(q995 - q50, 1e-6)
    z = np.clip((a - q50) / span, 0.0, 2.0)
    out = np.rint(z * 127.5).clip(0, 255).astype(np.uint8)
    return out, {"q50": q50, "q995": q995}


def build_registered_evidence_channels(pre: np.ndarray, posts: Sequence[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    """Rebuild the GT-free V2.21 registered physical maps for one framepack."""
    from src.engine.offline.direct_proposal_v221 import DirectProposalConfigV221, propose_direct_v221

    post_list = [np.asarray(p, dtype=np.uint8) for p in posts if np.asarray(p).shape == np.asarray(pre).shape]
    if not post_list:
        raise ValueError("Need >=1 POST frame matching PRE")
    direct = propose_direct_v221([np.asarray(pre, dtype=np.uint8)], post_list, config=DirectProposalConfigV221())
    maps = dict(getattr(direct, "maps", {}) or {})
    maps["fused"] = np.asarray(getattr(direct, "fused"), dtype=np.float32)
    shape = np.asarray(pre).shape[:2]
    channels: list[np.ndarray] = []
    norms: dict[str, Any] = {}
    missing: list[str] = []
    for name in EVIDENCE_CHANNEL_NAMES:
        raw = maps.get(name)
        if raw is None:
            channels.append(np.zeros(shape, dtype=np.uint8))
            norms[name] = {"q50": 0.0, "q995": 1.0, "missing": True}
            missing.append(name)
            continue
        arr = np.asarray(raw, dtype=np.float32)
        if arr.shape != shape:
            raise ValueError(f"Evidence map {name} shape {arr.shape} != PRE {shape}")
        norm, meta = _normalise_map(arr)
        channels.append(norm)
        norms[name] = meta
    return np.stack(channels, axis=0), {"normalisation": norms, "missing_channels": missing}


def extract_evidence_patches(channels: np.ndarray, xy: np.ndarray, *, batch_size: int = 512) -> np.ndarray:
    """Extract 27x27 candidate-centred crops and pool 3x -> 9x9."""
    ch = np.asarray(channels, dtype=np.uint8)
    pts = np.asarray(xy, dtype=np.float32)
    if ch.ndim != 3 or ch.shape[0] != EVIDENCE_CHANNELS:
        raise ValueError(f"channels must be [{EVIDENCE_CHANNELS},H,W], got {ch.shape}")
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"xy must be [N,2], got {pts.shape}")
    c, h, w = ch.shape
    n = len(pts)
    out = np.empty((n, c, EVIDENCE_PATCH_SIZE, EVIDENCE_PATCH_SIZE), dtype=np.uint8)
    half = EVIDENCE_CROP_SIZE // 2
    padded = np.pad(ch, ((0, 0), (half, half), (half, half)), mode="reflect")
    offsets = np.arange(-half, half + 1, dtype=np.int32)
    if len(offsets) != EVIDENCE_CROP_SIZE:
        raise AssertionError("crop geometry")
    step = max(1, int(batch_size))
    for start in range(0, n, step):
        stop = min(n, start + step)
        x = np.clip(np.rint(pts[start:stop, 0]).astype(np.int32), 0, w - 1) + half
        y = np.clip(np.rint(pts[start:stop, 1]).astype(np.int32), 0, h - 1) + half
        yy = y[:, None, None] + offsets[None, :, None]
        xx = x[:, None, None] + offsets[None, None, :]
        for ci in range(c):
            crop = padded[ci][yy, xx]
            pooled = crop.reshape(stop-start, EVIDENCE_PATCH_SIZE, 3, EVIDENCE_PATCH_SIZE, 3).mean(axis=(2, 4))
            out[start:stop, ci] = np.rint(pooled).clip(0, 255).astype(np.uint8)
    return out


def compile_evidence_shot(ref: DenseShotRefV2233, *, force: bool = False) -> EvidenceShotRefV2235:
    proposal = json.loads(ref.proposal_path.read_text(encoding="utf-8"))
    framepack = _resolve_framepack(ref.proposal_path, proposal)
    sig = _signature(ref, framepack)
    out_dir = EVIDENCE_CACHE_ROOT / ref.session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"shot_{ref.sequence:06d}_{sig}.npz"
    meta_path = cache_path.with_suffix(".json")
    if cache_path.exists() and meta_path.exists() and not force:
        return EvidenceShotRefV2235(ref.session_id, ref.shot_id, ref.sequence, ref, cache_path)

    dense = load_dense_shot(ref)
    _, pre, posts, _ = load_framepack(framepack)
    t0 = time.perf_counter()
    channels, channel_meta = build_registered_evidence_channels(pre, posts)
    patches = extract_evidence_patches(channels, dense.xy)

    gt = np.asarray(dense.gt_xy, dtype=np.float32)
    anchor_xy = np.asarray([[gt[0] + dx, gt[1] + dy] for dx, dy in GT_ANCHOR_OFFSETS], dtype=np.float32)
    anchors = extract_evidence_patches(channels, anchor_xy, batch_size=len(anchor_xy))
    anchor_distances = np.asarray([math.hypot(dx, dy) for dx, dy in GT_ANCHOR_OFFSETS], dtype=np.float32)

    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, patches=patches, anchors=anchors, anchor_distances=anchor_distances)
    os.replace(tmp, cache_path)
    meta = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "session_id": ref.session_id,
        "shot_id": ref.shot_id,
        "sequence": ref.sequence,
        "candidate_count": int(len(dense.xy)),
        "channel_names": list(EVIDENCE_CHANNEL_NAMES),
        "patch_shape": [EVIDENCE_CHANNELS, EVIDENCE_PATCH_SIZE, EVIDENCE_PATCH_SIZE],
        "crop_size": EVIDENCE_CROP_SIZE,
        "framepack": str(framepack),
        "dense_cache": str(ref.cache_path),
        "candidate_patches_gt_free": True,
        "gt_anchors_training_only": True,
        "positive_radius_px": PATCH_POSITIVE_RADIUS_PX,
        "neutral_band_px": [PATCH_POSITIVE_RADIUS_PX, PATCH_NEGATIVE_RADIUS_PX],
        "negative_radius_gt_px": PATCH_NEGATIVE_RADIUS_PX,
        "channel_meta": channel_meta,
        "runtime_ms": (time.perf_counter() - t0) * 1000.0,
        "created_at": time.time(),
        "live_authority": False,
    }
    _atomic_json(meta_path, meta)
    for old in out_dir.glob(f"shot_{ref.sequence:06d}_*.npz"):
        if old != cache_path:
            try:
                old.unlink()
                old.with_suffix(".json").unlink(missing_ok=True)
            except Exception:
                pass
    return EvidenceShotRefV2235(ref.session_id, ref.shot_id, ref.sequence, ref, cache_path)


def load_evidence_shot(ref: EvidenceShotRefV2235) -> EvidenceShotV2235:
    dense = load_dense_shot(ref.dense_ref)
    with np.load(ref.cache_path, allow_pickle=False) as data:
        patches = np.asarray(data["patches"], dtype=np.uint8)
        anchors = np.asarray(data["anchors"], dtype=np.uint8)
        anchor_distances = np.asarray(data["anchor_distances"], dtype=np.float32)
    expected = (len(dense.xy), EVIDENCE_CHANNELS, EVIDENCE_PATCH_SIZE, EVIDENCE_PATCH_SIZE)
    if patches.shape != expected:
        raise ValueError(f"Evidence patch bank shape mismatch {patches.shape} vs {expected}")
    return EvidenceShotV2235(ref, dense, patches, anchors, anchor_distances)


def discover_evidence_sessions(*, min_shots: int = 1) -> dict[str, list[EvidenceShotRefV2235]]:
    dense_groups = discover_cached_sessions(min_shots=min_shots)
    out: dict[str, list[EvidenceShotRefV2235]] = {}
    for sid, refs in dense_groups.items():
        ready: list[EvidenceShotRefV2235] = []
        for ref in refs:
            try:
                proposal = json.loads(ref.proposal_path.read_text(encoding="utf-8"))
                framepack = _resolve_framepack(ref.proposal_path, proposal)
                sig = _signature(ref, framepack)
                p = EVIDENCE_CACHE_ROOT / sid / f"shot_{ref.sequence:06d}_{sig}.npz"
                if p.exists() and p.with_suffix(".json").exists():
                    ready.append(EvidenceShotRefV2235(sid, ref.shot_id, ref.sequence, ref, p))
            except Exception:
                continue
        if len(ready) >= int(min_shots):
            out[sid] = sorted(ready, key=lambda r: r.sequence)
    return out


def compile_evidence_session(session_id: str | None = "latest", *, force: bool = False, min_shots: int = 1) -> dict[str, Any]:
    groups = discover_cached_sessions(min_shots=min_shots)
    if not groups:
        return {"status": "no_dense_sessions", "processed": 0}
    if session_id in (None, "latest"):
        session_id = max(groups, key=lambda sid: max(r.proposal_path.stat().st_mtime for r in groups[sid]))
    selected = groups.get(str(session_id), [])
    ok = 0
    cached = 0
    errors: list[str] = []
    bytes_total = 0
    for idx, ref in enumerate(selected, start=1):
        try:
            proposal = json.loads(ref.proposal_path.read_text(encoding="utf-8"))
            framepack = _resolve_framepack(ref.proposal_path, proposal)
            sig = _signature(ref, framepack)
            expected = EVIDENCE_CACHE_ROOT / ref.session_id / f"shot_{ref.sequence:06d}_{sig}.npz"
            was_cached = expected.exists() and expected.with_suffix(".json").exists() and not force
            eref = compile_evidence_shot(ref, force=force)
            ok += 1
            cached += int(was_cached)
            try:
                bytes_total += int(eref.cache_path.stat().st_size)
            except Exception:
                pass
            if idx == 1 or idx == len(selected) or idx % 5 == 0:
                print(
                    f"[V2.23.5 EVIDENCE] {idx}/{len(selected)} shot={ref.shot_id} "
                    f"candidates={len(load_dense_shot(ref).xy)} cache={was_cached} file={eref.cache_path.stat().st_size/1024/1024:.1f}MB"
                )
        except Exception as exc:
            errors.append(f"{ref.proposal_path}: {type(exc).__name__}: {exc}")
            print(f"[V2.23.5 EVIDENCE] failed shot={ref.shot_id}: {type(exc).__name__}: {exc}")
    return {
        "status": "ok" if ok else "failed",
        "session_id": session_id,
        "processed": ok,
        "cached": cached,
        "cache_bytes": bytes_total,
        "errors": errors,
    }
