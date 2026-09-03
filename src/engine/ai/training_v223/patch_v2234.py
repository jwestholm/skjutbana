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

PATCH_ROOT = Path("content/ai/training_v223/patch_v2234")
PATCH_CACHE_ROOT = PATCH_ROOT / "cache"
PATCH_SCHEMA_VERSION = "2.23.4-patchbank-1"
PATCH_CHANNEL_NAMES: tuple[str, ...] = (
    "pre_gray",
    "post_mean_gray",
    "absdiff_x8",
    "signed_diff_x4_center128",
    "persistence_0_255",
)
PATCH_CROP_SIZE = 32
PATCH_SIZE = 16
PATCH_CHANNELS = len(PATCH_CHANNEL_NAMES)


@dataclass(frozen=True)
class PatchShotRefV2234:
    session_id: str
    shot_id: str
    sequence: int
    dense_ref: DenseShotRefV2233
    patch_path: Path


@dataclass
class PatchShotV2234:
    ref: PatchShotRefV2234
    dense: DenseShotV2233
    patches: np.ndarray  # uint8 [N,C,H,W]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _signature(ref: DenseShotRefV2233, proposal: Mapping[str, Any], framepack_json: Path) -> str:
    parts = [PATCH_SCHEMA_VERSION, str(PATCH_CROP_SIZE), str(PATCH_SIZE), str(ref.cache_path.resolve())]
    for p in (ref.cache_path, ref.proposal_path, framepack_json, framepack_json.with_suffix(".npz")):
        try:
            st = p.stat(); parts.extend([str(p.resolve()), str(st.st_size), str(st.st_mtime_ns)])
        except Exception:
            parts.extend([str(p), "missing"])
    parts.append(str(proposal.get("source_framepack", "")))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]


def _resolve_framepack(proposal_path: Path, proposal: Mapping[str, Any]) -> Path:
    raw = str(proposal.get("source_framepack", "") or "")
    p = Path(raw)
    if p.exists():
        return p
    p2 = Path.cwd() / p
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Framepack not found for {proposal_path}: {raw}")


def _make_channel_images(pre: np.ndarray, posts: Sequence[np.ndarray]) -> np.ndarray:
    pre_u8 = np.asarray(pre, dtype=np.uint8)
    valid = [np.asarray(p, dtype=np.uint8) for p in posts if np.asarray(p).shape == pre_u8.shape]
    if not valid:
        raise ValueError("Need >=1 POST matching PRE")
    stack = np.stack(valid, axis=0).astype(np.int16)
    pre16 = pre_u8.astype(np.int16)
    signed_each = stack - pre16[None, :, :]
    signed_mean = np.mean(signed_each, axis=0, dtype=np.float32)
    abs_mean = np.mean(np.abs(signed_each), axis=0, dtype=np.float32)
    post_mean = np.mean(stack, axis=0).round().clip(0, 255).astype(np.uint8)
    abs_amp = np.clip(abs_mean * 8.0, 0.0, 255.0).round().astype(np.uint8)
    signed_amp = np.clip(128.0 + signed_mean * 4.0, 0.0, 255.0).round().astype(np.uint8)
    persistence = (np.mean(np.abs(signed_each) >= 2.5, axis=0) * 255.0).round().astype(np.uint8)
    return np.stack([pre_u8, post_mean, abs_amp, signed_amp, persistence], axis=0)


def _extract_patches_from_channels(
    channels: np.ndarray,
    xy: np.ndarray,
    *,
    crop_size: int = PATCH_CROP_SIZE,
    patch_size: int = PATCH_SIZE,
    batch_size: int = 512,
) -> np.ndarray:
    """Candidate-centred multi-channel patch extraction.

    Default 32x32 crops are average-pooled 2x to 16x16. This preserves a
    useful local field while keeping ~10k-candidate shot banks compact enough
    for repeated offline model training.
    """
    ch = np.asarray(channels, dtype=np.uint8)
    pts = np.asarray(xy, dtype=np.float32)
    if ch.ndim != 3:
        raise ValueError(f"channels must be [C,H,W], got {ch.shape}")
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"xy must be [N,2], got {pts.shape}")
    if crop_size != patch_size * 2:
        raise ValueError("Current extractor requires crop_size == patch_size*2")
    c, h, w = ch.shape
    n = len(pts)
    out = np.empty((n, c, patch_size, patch_size), dtype=np.uint8)
    half = crop_size // 2
    padded = np.pad(ch, ((0, 0), (half, half), (half, half)), mode="reflect")
    offsets = np.arange(-half, half, dtype=np.int32)
    for start in range(0, n, max(1, int(batch_size))):
        stop = min(n, start + max(1, int(batch_size)))
        x = np.rint(pts[start:stop, 0]).astype(np.int32)
        y = np.rint(pts[start:stop, 1]).astype(np.int32)
        x = np.clip(x, 0, w - 1) + half
        y = np.clip(y, 0, h - 1) + half
        yy = y[:, None, None] + offsets[None, :, None]
        xx = x[:, None, None] + offsets[None, None, :]
        for ci in range(c):
            crop = padded[ci][yy, xx]  # [B,crop,crop]
            pooled = crop.reshape(stop-start, patch_size, 2, patch_size, 2).mean(axis=(2, 4))
            out[start:stop, ci] = np.clip(np.rint(pooled), 0, 255).astype(np.uint8)
    return out


def extract_patch_tensor(
    pre: np.ndarray,
    posts: Sequence[np.ndarray],
    xy: np.ndarray,
    *,
    batch_size: int = 512,
) -> np.ndarray:
    channels = _make_channel_images(pre, posts)
    return _extract_patches_from_channels(channels, xy, batch_size=batch_size)


def compile_patch_shot(ref: DenseShotRefV2233, *, force: bool = False) -> PatchShotRefV2234:
    proposal = json.loads(ref.proposal_path.read_text(encoding="utf-8"))
    framepack = _resolve_framepack(ref.proposal_path, proposal)
    sig = _signature(ref, proposal, framepack)
    out_dir = PATCH_CACHE_ROOT / ref.session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    patch_path = out_dir / f"shot_{ref.sequence:06d}_{sig}.npz"
    meta_path = patch_path.with_suffix(".json")
    if patch_path.exists() and meta_path.exists() and not force:
        return PatchShotRefV2234(ref.session_id, ref.shot_id, ref.sequence, ref, patch_path)

    dense = load_dense_shot(ref)
    _, pre, posts, _ = load_framepack(framepack)
    t0 = time.perf_counter()
    patches = extract_patch_tensor(pre, posts, dense.xy)
    tmp = patch_path.with_suffix(patch_path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, patches=patches)
    os.replace(tmp, patch_path)
    meta = {
        "schema_version": PATCH_SCHEMA_VERSION,
        "session_id": ref.session_id,
        "shot_id": ref.shot_id,
        "sequence": ref.sequence,
        "candidate_count": int(len(dense.xy)),
        "patch_shape": [PATCH_CHANNELS, PATCH_SIZE, PATCH_SIZE],
        "channel_names": list(PATCH_CHANNEL_NAMES),
        "crop_size": PATCH_CROP_SIZE,
        "framepack": str(framepack),
        "dense_cache": str(ref.cache_path),
        "gt_used_for_candidate_patch_extraction": False,
        "runtime_ms": (time.perf_counter() - t0) * 1000.0,
        "created_at": time.time(),
    }
    _atomic_json(meta_path, meta)
    for old in out_dir.glob(f"shot_{ref.sequence:06d}_*.npz"):
        if old != patch_path:
            try:
                old.unlink()
                old.with_suffix(".json").unlink(missing_ok=True)
            except Exception:
                pass
    return PatchShotRefV2234(ref.session_id, ref.shot_id, ref.sequence, ref, patch_path)


def load_patch_shot(ref: PatchShotRefV2234) -> PatchShotV2234:
    dense = load_dense_shot(ref.dense_ref)
    with np.load(ref.patch_path, allow_pickle=False) as data:
        patches = np.asarray(data["patches"], dtype=np.uint8)
    if patches.shape != (len(dense.xy), PATCH_CHANNELS, PATCH_SIZE, PATCH_SIZE):
        raise ValueError(f"Patch bank shape mismatch {patches.shape} vs N={len(dense.xy)}")
    return PatchShotV2234(ref, dense, patches)


def discover_patch_sessions(*, min_shots: int = 1) -> dict[str, list[PatchShotRefV2234]]:
    dense_groups = discover_cached_sessions(min_shots=min_shots)
    out: dict[str, list[PatchShotRefV2234]] = {}
    for sid, refs in dense_groups.items():
        ready: list[PatchShotRefV2234] = []
        for ref in refs:
            try:
                proposal = json.loads(ref.proposal_path.read_text(encoding="utf-8"))
                framepack = _resolve_framepack(ref.proposal_path, proposal)
                sig = _signature(ref, proposal, framepack)
                p = PATCH_CACHE_ROOT / sid / f"shot_{ref.sequence:06d}_{sig}.npz"
                if p.exists() and p.with_suffix(".json").exists():
                    ready.append(PatchShotRefV2234(sid, ref.shot_id, ref.sequence, ref, p))
            except Exception:
                continue
        if len(ready) >= int(min_shots):
            out[sid] = sorted(ready, key=lambda r: r.sequence)
    return out


def compile_patch_session(session_id: str | None = "latest", *, force: bool = False, min_shots: int = 1) -> dict[str, Any]:
    groups = discover_cached_sessions(min_shots=min_shots)
    if not groups:
        return {"status": "no_dense_sessions", "processed": 0}
    if session_id in (None, "latest"):
        session_id = max(groups, key=lambda sid: max(r.proposal_path.stat().st_mtime for r in groups[sid]))
    selected = groups.get(str(session_id), [])
    ok = 0; cached = 0; errors: list[str] = []; bytes_written = 0
    for idx, ref in enumerate(selected, start=1):
        try:
            before = None
            proposal = json.loads(ref.proposal_path.read_text(encoding="utf-8"))
            fp = _resolve_framepack(ref.proposal_path, proposal)
            sig = _signature(ref, proposal, fp)
            expected = PATCH_CACHE_ROOT / ref.session_id / f"shot_{ref.sequence:06d}_{sig}.npz"
            if expected.exists() and not force:
                before = expected.stat().st_size
            pref = compile_patch_shot(ref, force=force)
            ok += 1
            cached += int(before is not None)
            try: bytes_written += int(pref.patch_path.stat().st_size)
            except Exception: pass
            if idx == 1 or idx == len(selected) or idx % 5 == 0:
                size_mb = pref.patch_path.stat().st_size / (1024*1024)
                print(f"[V2.23.4 PATCH] {idx}/{len(selected)} shot={ref.shot_id} candidates={len(load_dense_shot(ref).xy)} cache={before is not None} file={size_mb:.1f}MB")
        except Exception as exc:
            errors.append(f"{ref.proposal_path}: {type(exc).__name__}: {exc}")
            print(f"[V2.23.4 PATCH] failed {ref.proposal_path.name}: {type(exc).__name__}: {exc}")
    return {
        "status": "ok" if ok else "failed",
        "session_id": session_id,
        "processed": ok,
        "cached": cached,
        "errors": errors,
        "cache_bytes": bytes_written,
    }


def extract_gt_anchor_patches(ref: PatchShotRefV2234, offsets: Sequence[tuple[float, float]] = ((0,0),(-4,0),(4,0),(0,-4),(0,4))) -> tuple[np.ndarray, np.ndarray]:
    """Training-only GT anchors; never inserted into proposal/evaluation pools."""
    dense = load_dense_shot(ref.dense_ref)
    proposal = json.loads(ref.dense_ref.proposal_path.read_text(encoding="utf-8"))
    framepack = _resolve_framepack(ref.dense_ref.proposal_path, proposal)
    _, pre, posts, _ = load_framepack(framepack)
    gt = np.asarray(dense.gt_xy, dtype=np.float32)
    xy = np.asarray([[gt[0] + dx, gt[1] + dy] for dx, dy in offsets], dtype=np.float32)
    patches = extract_patch_tensor(pre, posts, xy, batch_size=len(xy))
    dists = np.asarray([math.hypot(dx, dy) for dx, dy in offsets], dtype=np.float32)
    return patches, dists
