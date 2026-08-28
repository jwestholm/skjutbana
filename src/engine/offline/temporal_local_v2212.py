from __future__ import annotations

"""V2.21.2 local temporal proposal/refinement helpers.

The V2.21 global full-frame proposal prototype showed that a global top-N search
was dominated by projector/camera nuisance.  V2.21.2 therefore separates two
jobs:

* GLOBAL direct proposals: still allowed to rescue a shot when V1/V2 has no
  useful candidate at all.
* LOCAL temporal proposals: use a current V1/V2 candidate only as an *anchor*
  and search nearby PRE->POST evidence for a better coordinate.

This module is offline/shadow-only.  It never sees ground truth.
"""

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class LocalTemporalConfigV2212:
    search_radius_px: int = 48
    local_max_kernel: int = 3
    peaks_per_source: int = 2
    nms_radius_px: float = 4.0
    proposal_limit: int = 3000
    minimum_local_contrast: float = 0.025
    source_names: tuple[str, ...] = (
        "persistent_dark",
        "blackhat_gain",
        "compact_change",
        "persistent_abs",
        "fused",
    )
    source_weights: dict[str, float] = field(default_factory=lambda: {
        "persistent_dark": 1.00,
        "blackhat_gain": 0.95,
        "compact_change": 0.90,
        "persistent_abs": 0.70,
        "fused": 1.00,
    })


def _source_map(name: str, maps: Mapping[str, np.ndarray], fused: np.ndarray) -> np.ndarray | None:
    if name == "fused":
        return np.asarray(fused, dtype=np.float32)
    value = maps.get(name)
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32)


def _local_peaks(
    values: np.ndarray,
    *,
    cx: float,
    cy: float,
    radius: int,
    kernel_size: int,
    limit: int,
    minimum_local_contrast: float,
) -> list[tuple[float, float, float, float]]:
    h, w = values.shape[:2]
    x0 = max(0, int(math.floor(cx - radius)))
    y0 = max(0, int(math.floor(cy - radius)))
    x1 = min(w, int(math.ceil(cx + radius + 1)))
    y1 = min(h, int(math.ceil(cy + radius + 1)))
    if x1 <= x0 or y1 <= y0:
        return []
    roi = np.asarray(values[y0:y1, x0:x1], dtype=np.float32)
    if roi.size == 0:
        return []

    # Robust local baseline.  This is deliberately local so broad projector
    # flicker/global gradients cannot win merely because they dominate the
    # whole 4K image.
    baseline = float(np.median(roi))
    upper = float(np.percentile(roi, 99.5))
    if upper - baseline < float(minimum_local_contrast):
        return []

    k = max(1, int(kernel_size)) | 1
    dilated = cv2.dilate(roi, np.ones((k, k), dtype=np.uint8))
    maxima = (roi >= dilated - 1e-7) & (roi >= baseline + float(minimum_local_contrast))
    ys, xs = np.nonzero(maxima)
    if not len(xs):
        return []

    scores = roi[ys, xs]
    contrasts = scores - baseline
    # Prefer contrast, then absolute map support.
    quality = contrasts + 0.20 * scores
    keep_n = min(len(quality), max(1, int(limit)))
    if len(quality) > keep_n:
        keep = np.argpartition(quality, -keep_n)[-keep_n:]
        xs, ys, scores, contrasts, quality = xs[keep], ys[keep], scores[keep], contrasts[keep], quality[keep]
    order = np.argsort(quality)[::-1]
    out: list[tuple[float, float, float, float]] = []
    r2 = float(radius * radius)
    for idx in order:
        x = float(x0 + int(xs[idx]))
        y = float(y0 + int(ys[idx]))
        # Keep the search circular rather than square.
        if (x - cx) ** 2 + (y - cy) ** 2 > r2:
            continue
        out.append((x, y, float(scores[idx]), float(contrasts[idx])))
        if len(out) >= keep_n:
            break
    return out


def _nms(rows: Sequence[dict[str, Any]], radius: float, limit: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda r: float(r.get("score", 0.0)), reverse=True)
    radius = max(0.5, float(radius))
    r2 = radius * radius
    cell = radius
    grid: dict[tuple[int, int], list[int]] = {}
    selected: list[dict[str, Any]] = []
    for row in ordered:
        x, y = float(row["camera_x"]), float(row["camera_y"])
        gx, gy = int(math.floor(x / cell)), int(math.floor(y / cell))
        merged_index: int | None = None
        for ny in range(gy - 1, gy + 2):
            for nx in range(gx - 1, gx + 2):
                for idx in grid.get((nx, ny), []):
                    old = selected[idx]
                    if (x - float(old["camera_x"])) ** 2 + (y - float(old["camera_y"])) ** 2 <= r2:
                        merged_index = idx
                        break
                if merged_index is not None:
                    break
            if merged_index is not None:
                break
        if merged_index is not None:
            old = selected[merged_index]
            sources = list(old.get("evidence_sources") or [])
            for source in row.get("evidence_sources") or []:
                if source not in sources:
                    sources.append(source)
            old["evidence_sources"] = sources
            old["local_source_support"] = max(
                int(old.get("local_source_support", 1)),
                int(row.get("local_source_support", 1)),
            )
            continue
        idx = len(selected)
        selected.append(dict(row))
        grid.setdefault((gx, gy), []).append(idx)
        if len(selected) >= max(1, int(limit)):
            break
    return selected


def propose_local_temporal_v2212(
    current_candidates: Sequence[dict[str, Any]],
    maps: Mapping[str, np.ndarray],
    fused: np.ndarray,
    *,
    config: LocalTemporalConfigV2212 | None = None,
) -> list[dict[str, Any]]:
    """Create temporal coordinates near current candidates without GT.

    Every emitted point must be supported by a local temporal maximum.  Merely
    sprinkling a geometric grid around a current candidate is forbidden because
    that would inflate oracle recall without adding evidence.
    """
    cfg = config or LocalTemporalConfigV2212()
    raw: list[dict[str, Any]] = []
    for parent_index, parent in enumerate(current_candidates):
        cx = float(parent.get("camera_x", 0.0))
        cy = float(parent.get("camera_y", 0.0))
        for source_name in cfg.source_names:
            values = _source_map(source_name, maps, fused)
            if values is None or values.ndim != 2:
                continue
            peaks = _local_peaks(
                values,
                cx=cx,
                cy=cy,
                radius=max(1, int(cfg.search_radius_px)),
                kernel_size=cfg.local_max_kernel,
                limit=max(1, int(cfg.peaks_per_source)),
                minimum_local_contrast=float(cfg.minimum_local_contrast),
            )
            weight = max(0.0, float(cfg.source_weights.get(source_name, 1.0)))
            for x, y, map_score, contrast in peaks:
                shift = float(math.hypot(x - cx, y - cy))
                quality = weight * (0.70 * contrast + 0.30 * map_score)
                raw.append({
                    "camera_x": x,
                    "camera_y": y,
                    "score": float(100.0 * quality),
                    "local_map_score": map_score,
                    "local_contrast": contrast,
                    "local_shift_px": shift,
                    "parent_camera_x": cx,
                    "parent_camera_y": cy,
                    "parent_index": int(parent_index),
                    "evidence_source": f"temporal_local_v2212:{source_name}",
                    "evidence_sources": [f"temporal_local_v2212:{source_name}"],
                    "ai_temporal_local_v2212": 1.0,
                    "local_source_support": 1,
                })
    return _nms(raw, cfg.nms_radius_px, cfg.proposal_limit)


def local_config_dict_v2212(config: LocalTemporalConfigV2212 | None = None) -> dict[str, Any]:
    return asdict(config or LocalTemporalConfigV2212())
