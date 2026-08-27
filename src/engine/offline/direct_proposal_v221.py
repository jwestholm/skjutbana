from __future__ import annotations

"""V2.21 direct full-frame proposal foundation.

This module is intentionally recall-oriented.  It searches PRE -> POST image
change directly and emits candidate points even when the legacy V1/V2 detector
never proposed a point there.  It is offline/shadow-only in V2.21.

No function in this module accepts ground-truth coordinates.
"""

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


DEFAULT_CONFIG_PATH = Path("content/ai/direct_v221.json")


@dataclass(frozen=True)
class DirectProposalConfigV221:
    registration_enabled: bool = True
    registration_max_shift_px: float = 6.0
    registration_min_response: float = 0.03
    photometric_compensation: bool = True
    pre_blur_sigma: float = 0.45
    morphology_radius_px: int = 4
    compact_small_sigma: float = 0.9
    compact_large_sigma: float = 4.2
    local_max_kernel: int = 3
    nms_radius_px: float = 5.0
    proposals_per_source: int = 72
    proposal_limit: int = 180
    source_quantile: float = 99.72
    source_min_score: float = 0.20
    fused_quantile: float = 99.55
    fused_min_score: float = 0.18
    weights: dict[str, float] = field(default_factory=lambda: {
        "persistent_abs": 0.16,
        "persistent_dark": 0.23,
        "persistent_bright": 0.06,
        "blackhat_gain": 0.22,
        "tophat_gain": 0.05,
        "compact_change": 0.18,
        "gradient_gain": 0.10,
    })

    @classmethod
    def from_file(cls, path: Path | None = None) -> "DirectProposalConfigV221":
        path = Path(path or DEFAULT_CONFIG_PATH)
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            section = raw.get("direct_proposal") if isinstance(raw, dict) else None
            if not isinstance(section, dict):
                section = raw if isinstance(raw, dict) else {}
            allowed = {name for name in cls.__dataclass_fields__}
            kwargs = {k: v for k, v in section.items() if k in allowed}
            if isinstance(kwargs.get("weights"), dict):
                kwargs["weights"] = dict(kwargs["weights"])
            return cls(**kwargs)
        except Exception:
            return cls()


@dataclass
class DirectProposalResultV221:
    candidates: list[dict[str, Any]]
    maps: dict[str, np.ndarray]
    fused: np.ndarray
    reference: np.ndarray
    post_reference: np.ndarray
    metadata: dict[str, Any]


def _gray(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        out = arr
    elif arr.ndim == 3 and arr.shape[2] == 3:
        out = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        out = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported frame shape: {arr.shape}")
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)


def _sample(values: np.ndarray, max_values: int = 160_000) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    if len(flat) <= max_values:
        return flat
    stride = max(1, len(flat) // max_values)
    return flat[::stride]


def _unit(values: np.ndarray) -> np.ndarray:
    v = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    s = _sample(v)
    if not len(s):
        return np.zeros_like(v, dtype=np.float32)
    p50, p90, p995 = np.percentile(s, [50.0, 90.0, 99.5])
    low = float(max(0.0, p50))
    high = float(max(low + 1e-5, p995, p90 * 1.35))
    return np.clip((v - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _register(reference: np.ndarray, current: np.ndarray, cfg: DirectProposalConfigV221) -> tuple[np.ndarray, dict[str, float]]:
    if not cfg.registration_enabled:
        return current, {"dx": 0.0, "dy": 0.0, "response": 0.0, "applied": 0.0}
    ref = reference.astype(np.float32)
    cur = current.astype(np.float32)
    h, w = ref.shape
    scale = min(1.0, 800.0 / float(max(h, w)))
    if scale < 1.0:
        size = (max(24, int(round(w * scale))), max(24, int(round(h * scale))))
        ref_small = cv2.resize(ref, size, interpolation=cv2.INTER_AREA)
        cur_small = cv2.resize(cur, size, interpolation=cv2.INTER_AREA)
    else:
        ref_small, cur_small = ref, cur
    try:
        (dxs, dys), response = cv2.phaseCorrelate(ref_small, cur_small)
        dx, dy = float(dxs) / scale, float(dys) / scale
        response = float(response)
    except Exception:
        return current, {"dx": 0.0, "dy": 0.0, "response": 0.0, "applied": 0.0}
    if (
        not math.isfinite(dx) or not math.isfinite(dy)
        or math.hypot(dx, dy) > float(cfg.registration_max_shift_px)
        or response < float(cfg.registration_min_response)
    ):
        return current, {"dx": dx, "dy": dy, "response": response, "applied": 0.0}
    matrix = np.float32([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
    aligned = cv2.warpAffine(current, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return aligned, {"dx": dx, "dy": dy, "response": response, "applied": 1.0}


def _photo_align(reference: np.ndarray, current: np.ndarray, enabled: bool) -> tuple[np.ndarray, float]:
    current_f = current.astype(np.float32)
    if not enabled:
        return current_f, 0.0
    delta = current_f - reference.astype(np.float32)
    offset = float(np.median(_sample(delta)))
    return np.clip(current_f - offset, 0.0, 255.0), offset


def build_direct_maps_v221(
    pre_frames: Sequence[np.ndarray],
    post_frames: Sequence[np.ndarray],
    *,
    config: DirectProposalConfigV221 | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    cfg = config or DirectProposalConfigV221()
    if not pre_frames or not post_frames:
        raise ValueError("Direct proposals require at least one PRE and one POST frame")
    pre = [_gray(frame) for frame in pre_frames]
    post = [_gray(frame) for frame in post_frames]
    shape = pre[0].shape
    if any(frame.shape != shape for frame in pre + post):
        raise ValueError("All direct-proposal frames must have identical dimensions")

    if float(cfg.pre_blur_sigma) > 0.02:
        pre_s = [cv2.GaussianBlur(frame, (0, 0), float(cfg.pre_blur_sigma)) for frame in pre]
        post_s = [cv2.GaussianBlur(frame, (0, 0), float(cfg.pre_blur_sigma)) for frame in post]
    else:
        pre_s, post_s = pre, post

    reference = np.median(np.stack([x.astype(np.float32) for x in pre_s], axis=0), axis=0).astype(np.float32)
    reference_u8 = np.clip(reference, 0, 255).astype(np.uint8)
    aligned_posts: list[np.ndarray] = []
    registration: list[dict[str, float]] = []
    photo_offsets: list[float] = []
    for frame in post_s:
        aligned_u8, reg = _register(reference_u8, frame, cfg)
        aligned, photo = _photo_align(reference, aligned_u8, cfg.photometric_compensation)
        aligned_posts.append(aligned)
        registration.append(reg)
        photo_offsets.append(float(photo))

    post_stack = np.stack(aligned_posts, axis=0).astype(np.float32)
    signed = post_stack - reference[None, ...]
    persistent_abs_raw = np.median(np.abs(signed), axis=0)
    persistent_dark_raw = np.median(np.maximum(-signed, 0.0), axis=0)
    persistent_bright_raw = np.median(np.maximum(signed, 0.0), axis=0)
    post_reference = np.median(post_stack, axis=0).astype(np.float32)

    radius = max(2, int(cfg.morphology_radius_px))
    k = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    ref_bh = cv2.morphologyEx(reference, cv2.MORPH_BLACKHAT, kernel)
    post_bh = cv2.morphologyEx(post_reference, cv2.MORPH_BLACKHAT, kernel)
    blackhat_gain_raw = np.maximum(post_bh - ref_bh, 0.0)

    ref_th = cv2.morphologyEx(reference, cv2.MORPH_TOPHAT, kernel)
    post_th = cv2.morphologyEx(post_reference, cv2.MORPH_TOPHAT, kernel)
    tophat_gain_raw = np.maximum(post_th - ref_th, 0.0)

    small = cv2.GaussianBlur(persistent_abs_raw, (0, 0), max(0.25, float(cfg.compact_small_sigma)))
    large = cv2.GaussianBlur(persistent_abs_raw, (0, 0), max(float(cfg.compact_small_sigma) + 0.5, float(cfg.compact_large_sigma)))
    compact_change_raw = np.maximum(small - large, 0.0)

    ref_gx = cv2.Sobel(reference, cv2.CV_32F, 1, 0, ksize=3)
    ref_gy = cv2.Sobel(reference, cv2.CV_32F, 0, 1, ksize=3)
    post_gx = cv2.Sobel(post_reference, cv2.CV_32F, 1, 0, ksize=3)
    post_gy = cv2.Sobel(post_reference, cv2.CV_32F, 0, 1, ksize=3)
    gradient_gain_raw = np.maximum(cv2.magnitude(post_gx, post_gy) - cv2.magnitude(ref_gx, ref_gy), 0.0)

    raw_maps = {
        "persistent_abs": persistent_abs_raw,
        "persistent_dark": persistent_dark_raw,
        "persistent_bright": persistent_bright_raw,
        "blackhat_gain": blackhat_gain_raw,
        "tophat_gain": tophat_gain_raw,
        "compact_change": compact_change_raw,
        "gradient_gain": gradient_gain_raw,
    }
    maps = {name: _unit(values) for name, values in raw_maps.items()}
    fused = np.zeros(shape, dtype=np.float32)
    weight_sum = 0.0
    for name, values in maps.items():
        weight = max(0.0, float(cfg.weights.get(name, 0.0)))
        if weight:
            fused += weight * values
            weight_sum += weight
    if weight_sum:
        fused /= weight_sum
    fused = np.clip(fused, 0.0, 1.0).astype(np.float32)
    metadata = {
        "pre_frames": len(pre), "post_frames": len(post),
        "registration": registration, "photometric_offsets": photo_offsets,
        "config": asdict(cfg),
        "raw_map_max": {name: float(np.max(values)) for name, values in raw_maps.items()},
    }
    return reference_u8, np.clip(post_reference, 0, 255).astype(np.uint8), maps, fused, metadata


def _threshold(values: np.ndarray, quantile: float, minimum: float) -> float:
    sample = _sample(values)
    if not len(sample):
        return float(minimum)
    q = float(np.percentile(sample, float(np.clip(quantile, 0.0, 100.0))))
    return max(float(minimum), q)


def _maxima_candidates(
    values: np.ndarray,
    *,
    source: str,
    quantile: float,
    minimum: float,
    limit: int,
    kernel_size: int,
) -> list[dict[str, Any]]:
    values = np.asarray(values, dtype=np.float32)
    threshold = _threshold(values, quantile, minimum)
    kernel_size = max(1, int(kernel_size)) | 1
    dilated = cv2.dilate(values, np.ones((kernel_size, kernel_size), dtype=np.uint8))
    mask = (values >= threshold) & (values >= dilated - 1e-7)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return []
    score = values[ys, xs]
    raw_limit = max(limit * 8, 600)
    if len(score) > raw_limit:
        keep = np.argpartition(score, -raw_limit)[-raw_limit:]
        xs, ys, score = xs[keep], ys[keep], score[keep]
    order = np.argsort(score)[::-1][: max(1, int(limit))]
    return [
        {
            "camera_x": float(xs[i]), "camera_y": float(ys[i]),
            "direct_score": float(score[i]),
            "score": float(100.0 * score[i]),
            "evidence_source": f"ai_direct_v221:{source}",
            "evidence_sources": [f"ai_direct_v221:{source}"],
            "ai_direct_v221": 1.0,
        }
        for i in order
    ]


def _merge_nms(candidates: Sequence[dict[str, Any]], maps: dict[str, np.ndarray], fused: np.ndarray, cfg: DirectProposalConfigV221) -> list[dict[str, Any]]:
    # Score all source maxima using a common fused score plus multi-source support.
    enriched: list[dict[str, Any]] = []
    for raw in candidates:
        row = dict(raw)
        x, y = int(round(float(row["camera_x"]))), int(round(float(row["camera_y"])))
        if not (0 <= y < fused.shape[0] and 0 <= x < fused.shape[1]):
            continue
        values = {name: float(v[y, x]) for name, v in maps.items()}
        support = sum(value >= 0.35 for value in values.values())
        row["fused_score"] = float(fused[y, x])
        row["direct_source_support"] = int(support)
        row["direct_components"] = values
        row["score"] = float(100.0 * (0.82 * row["fused_score"] + 0.18 * min(1.0, support / 3.0)))
        enriched.append(row)
    enriched.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)

    radius = max(0.5, float(cfg.nms_radius_px)); r2 = radius * radius
    cell = radius
    grid: dict[tuple[int, int], list[int]] = {}
    selected: list[dict[str, Any]] = []
    for row in enriched:
        x, y = float(row["camera_x"]), float(row["camera_y"])
        gx, gy = int(math.floor(x / cell)), int(math.floor(y / cell))
        merge_index: int | None = None
        for ny in range(gy - 1, gy + 2):
            for nx in range(gx - 1, gx + 2):
                for idx in grid.get((nx, ny), []):
                    old = selected[idx]
                    if (x - float(old["camera_x"])) ** 2 + (y - float(old["camera_y"])) ** 2 <= r2:
                        merge_index = idx
                        break
                if merge_index is not None: break
            if merge_index is not None: break
        if merge_index is not None:
            old = selected[merge_index]
            sources = list(old.get("evidence_sources") or [])
            for source in row.get("evidence_sources") or []:
                if source not in sources: sources.append(source)
            old["evidence_sources"] = sources
            old["direct_source_support"] = max(int(old.get("direct_source_support", 0)), int(row.get("direct_source_support", 0)))
            continue
        idx = len(selected)
        selected.append(row)
        grid.setdefault((gx, gy), []).append(idx)
        if len(selected) >= max(1, int(cfg.proposal_limit)):
            break
    return selected


def propose_direct_v221(
    pre_frames: Sequence[np.ndarray],
    post_frames: Sequence[np.ndarray],
    *,
    config: DirectProposalConfigV221 | None = None,
) -> DirectProposalResultV221:
    cfg = config or DirectProposalConfigV221()
    started = time.perf_counter()
    reference, post_reference, maps, fused, metadata = build_direct_maps_v221(pre_frames, post_frames, config=cfg)
    raw: list[dict[str, Any]] = []
    raw.extend(_maxima_candidates(
        fused, source="fused", quantile=cfg.fused_quantile, minimum=cfg.fused_min_score,
        limit=max(cfg.proposals_per_source, cfg.proposal_limit), kernel_size=cfg.local_max_kernel,
    ))
    for source, values in maps.items():
        raw.extend(_maxima_candidates(
            values, source=source, quantile=cfg.source_quantile, minimum=cfg.source_min_score,
            limit=cfg.proposals_per_source, kernel_size=cfg.local_max_kernel,
        ))
    candidates = _merge_nms(raw, maps, fused, cfg)
    metadata = dict(metadata)
    metadata.update({
        "raw_source_maxima": len(raw),
        "proposals": len(candidates),
        "runtime_ms": 1000.0 * (time.perf_counter() - started),
        "semantic_contract": "full-frame PRE->POST proposal coverage only; no GT input; shadow/offline only",
    })
    return DirectProposalResultV221(candidates, maps, fused, reference, post_reference, metadata)
