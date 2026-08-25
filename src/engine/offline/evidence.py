from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


DEFAULT_CONFIG_PATH = Path("content/ai/offline_v212.json")


@dataclass(frozen=True)
class EvidenceConfig:
    """Configuration for the first direct-image evidence source.

    All maps are physical image evidence.  Game/context priors intentionally do
    not live here; they will be separate evidence sources so they can never
    silently turn into hard rejection rules.
    """

    registration_enabled: bool = True
    registration_max_shift_px: float = 5.0
    registration_min_response: float = 0.05
    photometric_compensation: bool = True
    blur_kernel: int = 3
    blur_sigma: float = 0.55
    noise_floor: float = 1.5
    local_small: int = 3
    local_large: int = 15
    consensus_change: float = 2.0
    consensus_zscore: float = 1.0
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "temporal_consensus": 0.35,
            "persistent_zscore": 0.27,
            "local_contrast": 0.18,
            "darkening": 0.12,
            "absdiff": 0.08,
        }
    )
    candidate_robust_sigma: float = 2.2
    candidate_min_score: float = 0.24
    candidate_local_max_kernel: int = 3
    candidate_nms_radius_px: float = 4.0
    candidate_limit: int = 320

    @classmethod
    def from_file(cls, path: Path | None = None) -> EvidenceConfig:
        path = Path(path or DEFAULT_CONFIG_PATH)
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return cls()
        if not isinstance(raw, dict):
            return cls()
        section = raw.get("evidence") if isinstance(raw.get("evidence"), dict) else raw
        base = cls()
        kwargs: dict[str, Any] = {}
        for name in (
            "registration_enabled",
            "registration_max_shift_px",
            "registration_min_response",
            "photometric_compensation",
            "blur_kernel",
            "blur_sigma",
            "noise_floor",
            "local_small",
            "local_large",
            "consensus_change",
            "consensus_zscore",
            "candidate_robust_sigma",
            "candidate_min_score",
            "candidate_local_max_kernel",
            "candidate_nms_radius_px",
            "candidate_limit",
        ):
            if name in section:
                kwargs[name] = section[name]
        weights = section.get("weights")
        kwargs["weights"] = dict(weights) if isinstance(weights, dict) else dict(base.weights)
        try:
            return cls(**kwargs)
        except Exception:
            return cls()


@dataclass
class EvidenceOverlay:
    name: str
    values: np.ndarray
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.values, np.ndarray) or self.values.ndim != 2:
            raise ValueError(f"Evidence overlay {self.name!r} must be a 2D numpy array")
        self.values = self.values.astype(np.float32, copy=False)


@dataclass
class EvidenceBundle:
    reference: np.ndarray
    overlays: dict[str, EvidenceOverlay]
    fused: EvidenceOverlay
    metadata: dict[str, Any] = field(default_factory=dict)


def _odd(value: int, minimum: int = 1) -> int:
    value = max(minimum, int(value))
    return value if value % 2 else value + 1


def _as_gray(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Expected non-empty image")
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _ensure_same_shape(images: Sequence[np.ndarray]) -> tuple[int, int]:
    if not images:
        raise ValueError("No images supplied")
    shape = _as_gray(images[0]).shape
    for image in images[1:]:
        if _as_gray(image).shape != shape:
            raise ValueError("All pre/post images must have identical camera dimensions")
    return shape


def _sample_values(values: np.ndarray, max_values: int = 200_000) -> np.ndarray:
    flat = values.reshape(-1)
    if flat.size <= max_values:
        return flat
    stride = max(1, flat.size // max_values)
    return flat[::stride]


def robust_unit(values: np.ndarray, *, floor: float = 0.0) -> np.ndarray:
    """Robustly map non-negative evidence to [0,1] without per-image max domination."""
    values = np.maximum(values.astype(np.float32, copy=False), float(floor))
    sample = _sample_values(values)
    if sample.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    p50, p90, p995 = (float(v) for v in np.percentile(sample, [50.0, 90.0, 99.5]))
    low = max(float(floor), p50)
    high = max(low + 1e-6, p995, p90 * 1.35)
    result = (values - low) / (high - low)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _phase_register(reference: np.ndarray, current: np.ndarray, cfg: EvidenceConfig) -> tuple[np.ndarray, dict[str, float]]:
    if not cfg.registration_enabled:
        return current, {"dx": 0.0, "dy": 0.0, "response": 0.0, "applied": 0.0}
    ref = reference.astype(np.float32)
    cur = current.astype(np.float32)
    h, w = ref.shape
    scale = min(1.0, 720.0 / float(max(h, w)))
    if scale < 1.0:
        size = (max(16, int(round(w * scale))), max(16, int(round(h * scale))))
        ref_small = cv2.resize(ref, size, interpolation=cv2.INTER_AREA)
        cur_small = cv2.resize(cur, size, interpolation=cv2.INTER_AREA)
    else:
        ref_small, cur_small = ref, cur
    try:
        (dx_small, dy_small), response = cv2.phaseCorrelate(ref_small, cur_small)
        dx = float(dx_small) / scale
        dy = float(dy_small) / scale
        response = float(response)
    except Exception:
        return current, {"dx": 0.0, "dy": 0.0, "response": 0.0, "applied": 0.0}
    magnitude = math.hypot(dx, dy)
    if not math.isfinite(magnitude) or magnitude > float(cfg.registration_max_shift_px) or response < float(cfg.registration_min_response):
        return current, {"dx": dx, "dy": dy, "response": response, "applied": 0.0}
    # phaseCorrelate(ref, current) returns displacement of current relative to ref;
    # shift current back by the measured displacement.
    matrix = np.float32([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
    aligned = cv2.warpAffine(
        current,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return aligned, {"dx": dx, "dy": dy, "response": response, "applied": 1.0}


def _photometric_align(reference: np.ndarray, current: np.ndarray, enabled: bool) -> tuple[np.ndarray, float]:
    if not enabled:
        return current.astype(np.float32), 0.0
    # Median offset is deliberately simple and robust.  A bullet hole occupies a
    # negligible image fraction and therefore cannot meaningfully move it.
    delta = current.astype(np.float32) - reference.astype(np.float32)
    offset = float(np.median(_sample_values(delta)))
    aligned = np.clip(current.astype(np.float32) - offset, 0.0, 255.0)
    return aligned, offset


def build_evidence(
    pre_frames: Sequence[np.ndarray],
    post_frames: Sequence[np.ndarray],
    *,
    config: EvidenceConfig | None = None,
) -> EvidenceBundle:
    """Build V2.12 direct-image physical evidence maps.

    The first V2.12 source is *temporal consensus*: a real new hole should be a
    new local change that remains at approximately the same registered location
    in several post-shot frames.  We deliberately keep component maps separate
    so later AI/projector/game evidence can be added without hiding provenance.
    """

    cfg = config or EvidenceConfig()
    if not pre_frames or not post_frames:
        raise ValueError("V2.12 evidence requires at least one pre and one post frame")
    all_frames = list(pre_frames) + list(post_frames)
    _ensure_same_shape(all_frames)
    pre_raw = [_as_gray(frame) for frame in pre_frames]
    post_raw = [_as_gray(frame) for frame in post_frames]
    blur_kernel = _odd(cfg.blur_kernel, 1)
    if blur_kernel > 1 or float(cfg.blur_sigma) > 0.01:
        pre = [cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), float(cfg.blur_sigma)) for frame in pre_raw]
        post = [cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), float(cfg.blur_sigma)) for frame in post_raw]
    else:
        pre, post = pre_raw, post_raw

    pre_stack = np.stack([frame.astype(np.float32) for frame in pre], axis=0)
    reference = np.median(pre_stack, axis=0).astype(np.float32)
    pre_mad = np.median(np.abs(pre_stack - reference[None, ...]), axis=0).astype(np.float32)
    noise = np.maximum(float(cfg.noise_floor), 1.4826 * pre_mad).astype(np.float32)

    abs_maps: list[np.ndarray] = []
    dark_maps: list[np.ndarray] = []
    z_maps: list[np.ndarray] = []
    registration: list[dict[str, float]] = []
    photo_offsets: list[float] = []

    ref_u8 = np.clip(reference, 0, 255).astype(np.uint8)
    for frame in post:
        registered, reg = _phase_register(ref_u8, frame, cfg)
        current, offset = _photometric_align(reference, registered, cfg.photometric_compensation)
        signed = current - reference
        absdiff = np.abs(signed).astype(np.float32)
        darkening = np.maximum(-signed, 0.0).astype(np.float32)
        zscore = (absdiff / np.maximum(noise, 1e-3)).astype(np.float32)
        abs_maps.append(absdiff)
        dark_maps.append(darkening)
        z_maps.append(zscore)
        registration.append(reg)
        photo_offsets.append(offset)

    abs_stack = np.stack(abs_maps, axis=0)
    dark_stack = np.stack(dark_maps, axis=0)
    z_stack = np.stack(z_maps, axis=0)

    # Median/low-percentile behaviour is the key persistence idea: transient
    # noise/motion may be strong in one frame, while a hole remains changed.
    persistent_abs = np.median(abs_stack, axis=0).astype(np.float32)
    persistent_dark = np.median(dark_stack, axis=0).astype(np.float32)
    persistent_z = np.median(z_stack, axis=0).astype(np.float32)

    consensus_hits = (
        (abs_stack >= float(cfg.consensus_change))
        & (z_stack >= float(cfg.consensus_zscore))
    )
    temporal_consensus = np.mean(consensus_hits.astype(np.float32), axis=0)

    small = _odd(cfg.local_small, 1)
    large = _odd(max(cfg.local_large, small + 2), 3)
    local_small = cv2.boxFilter(persistent_abs, cv2.CV_32F, (small, small), normalize=True)
    local_large = cv2.boxFilter(persistent_abs, cv2.CV_32F, (large, large), normalize=True)
    local_contrast = np.maximum(local_small - local_large, 0.0).astype(np.float32)

    raw_maps = {
        "temporal_consensus": temporal_consensus,
        "persistent_zscore": persistent_z,
        "local_contrast": local_contrast,
        "darkening": persistent_dark,
        "absdiff": persistent_abs,
    }
    overlays: dict[str, EvidenceOverlay] = {}
    fused = np.zeros_like(reference, dtype=np.float32)
    total_weight = 0.0
    for name, raw in raw_maps.items():
        if name == "temporal_consensus":
            unit = np.clip(raw, 0.0, 1.0).astype(np.float32)
        else:
            unit = robust_unit(raw)
        weight = max(0.0, float(cfg.weights.get(name, 0.0)))
        overlays[name] = EvidenceOverlay(
            name=name,
            values=unit,
            weight=weight,
            metadata={
                "raw_max": float(np.max(raw)),
                "raw_median": float(np.median(_sample_values(raw))),
            },
        )
        if weight > 0.0:
            fused += weight * unit
            total_weight += weight
    if total_weight > 0.0:
        fused /= total_weight
    fused = np.clip(fused, 0.0, 1.0).astype(np.float32)

    return EvidenceBundle(
        reference=np.clip(reference, 0, 255).astype(np.uint8),
        overlays=overlays,
        fused=EvidenceOverlay("physical_fusion_v212", fused, 1.0),
        metadata={
            "pre_frames": len(pre),
            "post_frames": len(post),
            "registration": registration,
            "photometric_offsets": photo_offsets,
            "pre_noise_median": float(np.median(_sample_values(noise))),
        },
    )


def _robust_threshold(values: np.ndarray, cfg: EvidenceConfig) -> float:
    sample = _sample_values(values)
    median = float(np.median(sample)) if sample.size else 0.0
    mad = float(np.median(np.abs(sample - median))) if sample.size else 0.0
    threshold = median + float(cfg.candidate_robust_sigma) * 1.4826 * mad
    return max(float(cfg.candidate_min_score), float(threshold))


def extract_overlay_candidates(
    overlay: EvidenceOverlay,
    *,
    config: EvidenceConfig | None = None,
    roi_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Convert one evidence heatmap into permissive point proposals."""

    cfg = config or EvidenceConfig()
    values = overlay.values.astype(np.float32, copy=False)
    valid = np.ones(values.shape, dtype=bool)
    if roi_mask is not None:
        if roi_mask.shape != values.shape:
            raise ValueError("ROI mask shape does not match evidence map")
        valid &= roi_mask > 0
    threshold = _robust_threshold(values[valid] if np.any(valid) else values, cfg)
    kernel = _odd(cfg.candidate_local_max_kernel, 1)
    dilated = cv2.dilate(values, np.ones((kernel, kernel), dtype=np.uint8))
    maxima = valid & (values >= threshold) & (values >= dilated - 1e-7)
    ys, xs = np.nonzero(maxima)
    if xs.size == 0:
        return []

    scores = values[ys, xs]
    nms_radius = max(0.5, float(cfg.candidate_nms_radius_px))
    limit = max(1, int(cfg.candidate_limit))

    # Dense projected/game imagery can create tens of thousands of local maxima.
    # Full sorting + O(N*limit) distance checks becomes the bottleneck in offline
    # million-shot work.  Keep a generous score headroom, then use a spatial hash
    # for exact-radius NMS.  This is a recall pool, so the cap is deliberately much
    # wider than the final output limit.
    raw_cap = max(1200, limit * 8)
    if scores.size > raw_cap:
        keep = np.argpartition(scores, -raw_cap)[-raw_cap:]
        xs = xs[keep]
        ys = ys[keep]
        scores = scores[keep]
    order = np.argsort(scores)[::-1]

    cell_size = max(1.0, nms_radius)
    grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
    selected: list[dict[str, Any]] = []
    radius_sq = nms_radius * nms_radius

    for index in order:
        x = float(xs[index])
        y = float(ys[index])
        gx = int(math.floor(x / cell_size))
        gy = int(math.floor(y / cell_size))
        blocked = False
        for ny in range(gy - 1, gy + 2):
            for nx in range(gx - 1, gx + 2):
                for sx, sy in grid.get((nx, ny), ()):
                    dx = x - sx
                    dy = y - sy
                    if dx * dx + dy * dy < radius_sq:
                        blocked = True
                        break
                if blocked:
                    break
            if blocked:
                break
        if blocked:
            continue

        score = float(scores[index])
        selected.append(
            {
                "camera_x": x,
                "camera_y": y,
                "score": max(0.01, 35.0 * score),
                "overlay_score": score,
                "detector_v212_overlay": 1.0,
                "evidence_source": overlay.name,
                "evidence_sources": [overlay.name],
            }
        )
        grid.setdefault((gx, gy), []).append((x, y))
        if len(selected) >= limit:
            break
    return selected


def merge_candidate_sources(
    sources: Sequence[tuple[str, Sequence[dict[str, Any]]]],
    *,
    merge_radius_px: float = 5.0,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Union candidate proposals while preserving evidence provenance.

    This is intentionally a *recall pool*, not an authoritative ranker.  A later
    fusion/ranking stage may learn how much each source should be trusted.
    """

    merged: list[dict[str, Any]] = []
    radius = max(0.5, float(merge_radius_px))
    for source_name, candidates in sources:
        for raw in candidates:
            candidate = dict(raw)
            try:
                x = float(candidate["camera_x"])
                y = float(candidate["camera_y"])
            except Exception:
                continue
            best_index = -1
            best_distance = float("inf")
            for index, existing in enumerate(merged):
                distance = math.hypot(x - float(existing["camera_x"]), y - float(existing["camera_y"]))
                if distance <= radius and distance < best_distance:
                    best_index = index
                    best_distance = distance
            if best_index < 0:
                candidate.setdefault("evidence_sources", [])
                source_list = list(candidate.get("evidence_sources") or [])
                if source_name not in source_list:
                    source_list.append(source_name)
                candidate["evidence_sources"] = source_list
                merged.append(candidate)
                continue

            existing = merged[best_index]
            source_list = list(existing.get("evidence_sources") or [])
            for value in list(candidate.get("evidence_sources") or []) + [source_name]:
                if value not in source_list:
                    source_list.append(value)
            existing["evidence_sources"] = source_list
            existing["evidence_source_count"] = float(len(source_list))
            # Keep strongest score but do not invent a final confidence here.
            existing["score"] = max(float(existing.get("score", 0.0)), float(candidate.get("score", 0.0)))
            existing["overlay_score"] = max(float(existing.get("overlay_score", 0.0)), float(candidate.get("overlay_score", 0.0)))
            existing["detector_v212_overlay"] = max(
                float(existing.get("detector_v212_overlay", 0.0)),
                float(candidate.get("detector_v212_overlay", 0.0)),
            )
    merged.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return merged[: max(1, int(limit))]
