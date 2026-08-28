"""Cheap candidate-cleanup helpers for V2.22.2.

These filters run *after* the existing V2.22.1 perspective-safe detector has
produced camera candidates but *before* candidates reach tracking / AIRuntime /
ShotResolver.

The intent is deliberately conservative:
- old/static hole appearance is demoted using PRE-shot novelty;
- a known old hole is rejected only when there is no fresh PRE->POST evidence;
- long horizontal candidate bands are treated as board/projector motion ridges;
- a strong fresh candidate is preserved even when it lies on such a ridge.

All geometry-aware ridge decisions are made in screen/playfield coordinates via
one vectorised homography transform.  Canonical candidate coordinates remain
full-camera XY throughout the engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

SCHEMA_VERSION = "2.22.2"


@dataclass(frozen=True)
class CandidateCleanupStatsV2222:
    input_count: int
    output_count: int
    stale_known_removed: int = 0
    novelty_demoted: int = 0
    ridge_removed: int = 0
    ridge_groups: int = 0
    ridge_preserved_fresh: int = 0
    pre_shot_informative: bool = False

    def as_dict(self) -> dict[str, float]:
        return {
            "v2222_candidates_input": float(self.input_count),
            "v2222_candidates_output": float(self.output_count),
            "v2222_stale_known_removed": float(self.stale_known_removed),
            "v2222_novelty_demoted": float(self.novelty_demoted),
            "v2222_ridge_removed": float(self.ridge_removed),
            "v2222_ridge_groups": float(self.ridge_groups),
            "v2222_ridge_preserved_fresh": float(self.ridge_preserved_fresh),
            "v2222_pre_shot_informative": 1.0 if self.pre_shot_informative else 0.0,
        }


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else float(default)
    except Exception:
        return float(default)


def _nearest_known_hole(
    candidate: Mapping[str, Any],
    known_holes: Sequence[Mapping[str, Any]],
    radius_px: float,
) -> float | None:
    x = _finite(candidate.get("camera_x"))
    y = _finite(candidate.get("camera_y"))
    best: float | None = None
    limit = max(0.0, float(radius_px))
    for hole in known_holes:
        hx = _finite(hole.get("camera_x"), 1e30)
        hy = _finite(hole.get("camera_y"), 1e30)
        dist = float(np.hypot(hx - x, hy - y))
        if dist <= limit and (best is None or dist < best):
            best = dist
    return best


def _pre_shot_is_informative(candidates: Sequence[Mapping[str, Any]], threshold: float = 1.5) -> bool:
    values = np.asarray([
        max(0.0, _finite(c.get("pre_shot_change", 0.0))) for c in candidates
    ], dtype=np.float32)
    if values.size < 3:
        return bool(values.size and float(np.max(values)) >= threshold)
    return float(np.percentile(values, 90.0)) >= float(threshold)


def apply_novelty_cleanup_v2222(
    candidates: Sequence[Mapping[str, Any]],
    known_holes: Sequence[Mapping[str, Any]],
    *,
    duplicate_radius_px: float = 18.0,
    fresh_rehit_min: float = 5.0,
) -> tuple[list[dict[str, Any]], dict[str, int | bool]]:
    """Demote static appearance and reject stale candidates at known holes.

    Re-hits remain possible.  A candidate close to a registered hole survives
    when PRE->POST evidence is fresh enough; otherwise it is stale by
    construction and is removed before tracking.
    """
    items = [dict(c) for c in candidates]
    informative = _pre_shot_is_informative(items)
    stale_removed = 0
    demoted = 0
    kept: list[dict[str, Any]] = []

    for item in items:
        psc = max(0.0, _finite(item.get("pre_shot_change", 0.0)))
        near = _nearest_known_hole(item, known_holes, duplicate_radius_px)
        if near is not None:
            item["v2222_known_hole_dist"] = float(near)
            if informative and psc < float(fresh_rehit_min):
                stale_removed += 1
                continue
            item["v2222_fresh_rehit"] = 1.0 if psc >= float(fresh_rehit_min) else 0.0

        # Existing detector scores can still be dominated by scene-reference
        # morphology.  PRE-shot novelty is the cheap discriminator for a NEW
        # physical change.  Only demote globally when the shot actually has an
        # informative PRE-shot signal; otherwise fail open.
        factor = 1.0
        if informative:
            if psc < 1.5:
                factor = 0.10
            elif psc < 3.0:
                factor = 0.25
            elif psc < 5.0:
                factor = 0.55
        if factor < 0.999:
            item["score"] = _finite(item.get("score", 0.0)) * factor
            item["v2222_novelty_factor"] = float(factor)
            demoted += 1
        else:
            item["v2222_novelty_factor"] = 1.0
        kept.append(item)

    kept.sort(key=lambda c: _finite(c.get("score", 0.0)), reverse=True)
    return kept, {
        "pre_shot_informative": bool(informative),
        "stale_known_removed": int(stale_removed),
        "novelty_demoted": int(demoted),
    }


def project_camera_candidates_to_screen_v2222(
    candidates: Sequence[Mapping[str, Any]],
    homography_camera_to_screen: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    if not candidates:
        return np.empty((0, 2), dtype=np.float32)
    H = np.asarray(homography_camera_to_screen, dtype=np.float32)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        raise ValueError("homography must be a finite 3x3 matrix")
    pts = np.asarray([
        [_finite(c.get("camera_x")), _finite(c.get("camera_y"))]
        for c in candidates
    ], dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def suppress_horizontal_ridges_v2222(
    candidates: Sequence[Mapping[str, Any]],
    screen_points: np.ndarray,
    *,
    screen_rect_xywh: Sequence[float],
    band_px: float = 7.0,
    min_count: int = 9,
    min_span_fraction: float = 0.35,
    fresh_preserve_min: float = 6.0,
    max_preserve_per_ridge: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Remove long horizontal motion bands in *screen* coordinates.

    The camera can be angled, so a physical horizontal band may be sloped in the
    camera image.  Transforming only the already-reduced candidate list to the
    screen plane lets us detect the ridge correctly without warping any image.
    """
    items = [dict(c) for c in candidates]
    pts = np.asarray(screen_points, dtype=np.float32).reshape(-1, 2)
    if len(items) != len(pts) or len(items) == 0:
        return items, {"ridge_removed": 0, "ridge_groups": 0, "ridge_preserved_fresh": 0}

    sx, sy, sw, sh = (float(v) for v in screen_rect_xywh[:4])
    del sh
    bw = max(2.0, float(band_px))
    groups: dict[int, list[int]] = {}
    for i, (x, y) in enumerate(pts):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        # Bucket relative to playfield origin so identical physical rows stay
        # together regardless of absolute screen placement.
        key = int(np.floor((float(y) - sy) / bw))
        groups.setdefault(key, []).append(i)

    ridge_indices: set[int] = set()
    preserve_indices: set[int] = set()
    ridge_groups = 0
    preserved = 0
    required_span = max(80.0, max(0.05, float(min_span_fraction)) * max(1.0, sw))

    for indices in groups.values():
        if len(indices) < max(3, int(min_count)):
            continue
        xs = [float(pts[i, 0]) for i in indices]
        span = max(xs) - min(xs)
        if span < required_span:
            continue
        ridge_groups += 1
        ridge_indices.update(indices)

        fresh = [
            i for i in indices
            if _finite(items[i].get("pre_shot_change", 0.0)) >= float(fresh_preserve_min)
        ]
        fresh.sort(
            key=lambda i: (
                _finite(items[i].get("pre_shot_change", 0.0)),
                _finite(items[i].get("score", 0.0)),
            ),
            reverse=True,
        )
        for i in fresh[: max(0, int(max_preserve_per_ridge))]:
            preserve_indices.add(i)
            items[i]["v2222_ridge_fresh_preserved"] = 1.0
            preserved += 1

        # Safety valve: never erase an entire physical row.  Preserve the best
        # remaining detector candidate even if PRE-shot novelty is weak.  This
        # keeps a real shot recoverable while still collapsing tens/hundreds of
        # board-motion proposals to one.
        if not any(i in preserve_indices for i in indices):
            fallback = max(indices, key=lambda i: _finite(items[i].get("score", 0.0)))
            preserve_indices.add(fallback)
            items[fallback]["v2222_ridge_fallback_preserved"] = 1.0

    output: list[dict[str, Any]] = []
    removed = 0
    for i, item in enumerate(items):
        if i in ridge_indices and i not in preserve_indices:
            removed += 1
            continue
        output.append(item)
    output.sort(key=lambda c: _finite(c.get("score", 0.0)), reverse=True)
    return output, {
        "ridge_removed": int(removed),
        "ridge_groups": int(ridge_groups),
        "ridge_preserved_fresh": int(preserved),
    }


__all__ = [
    "SCHEMA_VERSION",
    "CandidateCleanupStatsV2222",
    "apply_novelty_cleanup_v2222",
    "project_camera_candidates_to_screen_v2222",
    "suppress_horizontal_ridges_v2222",
]
