"""V2.25.1 balanced per-object physical proposal and confirmation.

V2.25.0 physical testing proved that the shot-id/frozen GameObject contract works,
but it also exposed a detector failure mode: all object HitRegions were unioned into
one search mask, allowing a noisy/old-hole area inside one region to dominate the
candidate pool for shots physically landing in another region.

This patch keeps game semantics out of hit authority.  Every active HitRegion is
processed identically.  The regions only partition the already-calibrated physical
search space so each area gets a fair chance to contribute fresh PRE->POST evidence.
The final XY is always an observed detector coordinate; no candidate is moved or
snapped to an object.

Pipeline (normal object-context shot):

    frozen camera HitRegions
          -> camera->V2.22.1 working-space map (V2.24.4)
          -> individual/overlap-group masks
          -> region-local robust proposal threshold
          -> bounded candidates per region
          -> hybrid/bank output re-balanced per region
          -> V2.22.5 PRE->POST local confirmation
          -> bounded confirmed candidates per region
          -> physical-evidence track selection
          -> unchanged HitScanner emission / HitEvent / GameObject exact collision

V2.22.5 FULL-RESCUE is deliberately bypassed by this patch and remains global.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from src.engine.input.object_hit_v2223 import object_hit_registry_v2223
from src.engine.shot_fast_v2225 import rescue_router_v2225

SCHEMA_VERSION = "2.25.1"
PATCH_REVISION = "r1"
_INSTALLED = False


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else float(default)
    except Exception:
        return float(default)


def _runtime_settings() -> dict[str, Any]:
    try:
        from src.engine.ai.runtime import get_ai_runtime
        value = getattr(get_ai_runtime(), "settings", {})
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _setting_bool(name: str, default: bool) -> bool:
    return bool(_runtime_settings().get(name, default))


def _setting_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(_runtime_settings().get(name, default))
    except Exception:
        value = int(default)
    return max(lo, min(hi, value))


def _setting_float(name: str, default: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, _finite(_runtime_settings().get(name, default), default)))


def _enabled() -> bool:
    return _setting_bool("object_region_proposal_enabled_v251", True)


def _log_enabled() -> bool:
    return _setting_bool("object_region_proposal_log_v251", True)


def _margin_px() -> float:
    # Keep the already physically tested V2.24 margin as the default.
    return _setting_float("object_region_margin_px_v251", 36.0, 0.0, 192.0)


def _per_region_proposals() -> int:
    return _setting_int("object_region_proposals_per_region_v251", 8, 1, 32)


def _per_region_confirmed() -> int:
    return _setting_int("object_region_confirmed_per_region_v251", 2, 1, 8)


def _confirmed_total() -> int:
    return _setting_int("object_region_confirmed_total_v251", 8, 2, 48)


def _group_overlap_ratio() -> float:
    return _setting_float("object_region_group_overlap_v251", 0.80, 0.50, 1.0)


def _shot_id_from_scanner(scanner: Any) -> int:
    if scanner is None:
        return 0
    pending = [
        ev for ev in list(getattr(scanner, "audio_events", []) or [])
        if str(getattr(ev, "state", "")) == "pending"
    ]
    if not pending:
        return 0
    ev = min(pending, key=lambda item: float(getattr(item, "peak_ts", 0.0) or 0.0))
    return int(getattr(ev, "shot_id", 0) or 0)


@dataclass(frozen=True)
class RegionGroupV251:
    group_id: str
    object_ids: tuple[str, ...]
    roles: tuple[str, ...]
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= float(x) <= self.x1 and self.y0 <= float(y) <= self.y1


def _intersection_area(a: RegionGroupV251, b: RegionGroupV251) -> float:
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _same_search_area(a: RegionGroupV251, b: RegionGroupV251) -> bool:
    inter = _intersection_area(a, b)
    denom = max(1e-6, min(a.area, b.area))
    return inter / denom >= _group_overlap_ratio()


def _merge_groups(groups: Sequence[RegionGroupV251]) -> tuple[RegionGroupV251, ...]:
    """Merge only near-identical/strongly-overlapping physical search areas.

    This is intentionally NOT V2.24's transitive broad AABB union.  The purpose
    here is to preserve separate search opportunities while avoiding duplicate
    work for e.g. a glass panel and the target directly behind it.
    """
    work = list(groups)
    changed = True
    while changed:
        changed = False
        out: list[RegionGroupV251] = []
        used = [False] * len(work)
        for i, first in enumerate(work):
            if used[i]:
                continue
            members = [first]
            used[i] = True
            for j in range(i + 1, len(work)):
                if used[j]:
                    continue
                if any(_same_search_area(m, work[j]) for m in members):
                    members.append(work[j])
                    used[j] = True
                    changed = True
            ids = tuple(dict.fromkeys(x for m in members for x in m.object_ids))
            roles = tuple(dict.fromkeys(x for m in members for x in m.roles))
            out.append(RegionGroupV251(
                group_id="+".join(ids),
                object_ids=ids,
                roles=roles,
                x0=min(m.x0 for m in members),
                y0=min(m.y0 for m in members),
                x1=max(m.x1 for m in members),
                y1=max(m.y1 for m in members),
            ))
        work = out
    return tuple(work)


def _camera_regions_to_work_groups(scanner: Any, shot_id: int) -> tuple[RegionGroupV251, ...]:
    snap = object_hit_registry_v2223.snapshot_for_shot(int(shot_id))
    camera_regions = tuple(getattr(snap, "camera_regions", ()) or ()) if snap is not None else ()
    if not camera_regions:
        return ()

    diag = getattr(scanner, "_v244_roi_diag", None)
    if not isinstance(diag, dict):
        return ()
    crop = diag.get("crop")
    scale = diag.get("scale")
    work_shape = diag.get("work_shape")
    if not (isinstance(crop, (tuple, list)) and len(crop) >= 4 and
            isinstance(scale, (tuple, list)) and len(scale) >= 2 and
            isinstance(work_shape, (tuple, list)) and len(work_shape) >= 2):
        return ()

    crop_x0, crop_y0 = _finite(crop[0]), _finite(crop[1])
    sx, sy = _finite(scale[0], 1.0), _finite(scale[1], 1.0)
    work_h, work_w = int(work_shape[0]), int(work_shape[1])
    margin = _margin_px()
    groups: list[RegionGroupV251] = []
    for region in camera_regions:
        try:
            x0 = (_finite(region.x) - margin - crop_x0) * sx
            y0 = (_finite(region.y) - margin - crop_y0) * sy
            x1 = (_finite(region.x + region.width) + margin - crop_x0) * sx
            y1 = (_finite(region.y + region.height) + margin - crop_y0) * sy
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            x0 = max(0.0, min(float(work_w), x0))
            x1 = max(0.0, min(float(work_w), x1))
            y0 = max(0.0, min(float(work_h), y0))
            y1 = max(0.0, min(float(work_h), y1))
            if x1 - x0 < 2.0 or y1 - y0 < 2.0:
                continue
            groups.append(RegionGroupV251(
                group_id=str(getattr(region, "object_id", "region")),
                object_ids=(str(getattr(region, "object_id", "region")),),
                roles=(str(getattr(region, "role", "target")),),
                x0=x0, y0=y0, x1=x1, y1=y1,
            ))
        except Exception:
            continue
    return _merge_groups(groups)


def _bbox_region_mask(
    shape: Sequence[int],
    valid: np.ndarray,
    bbox: tuple[int, int, int, int],
    group: RegionGroupV251,
) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    bx0, by0, _bx1, _by1 = bbox
    x0 = max(0, min(w, int(math.floor(group.x0 - bx0))))
    y0 = max(0, min(h, int(math.floor(group.y0 - by0))))
    x1 = max(0, min(w, int(math.ceil(group.x1 - bx0))))
    y1 = max(0, min(h, int(math.ceil(group.y1 - by0))))
    mask = np.zeros((h, w), dtype=bool)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return np.asarray(valid, dtype=bool) & mask


def _temporal_map(absdiff: np.ndarray, zscore: np.ndarray, dog: np.ndarray) -> np.ndarray:
    return (
        absdiff.astype(np.float32) * (1.0 + 0.55 * np.clip(zscore, 0.0, 6.0))
        + 0.35 * np.maximum(dog.astype(np.float32), 0.0)
    ).astype(np.float32)


def _robust_stats(values: np.ndarray) -> tuple[float, float]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size > 60_000:
        flat = flat[::max(1, flat.size // 60_000)]
    if flat.size == 0:
        return 0.0, 1.0
    med = float(np.median(flat))
    mad = float(np.median(np.abs(flat - med)))
    scale = max(0.35, 1.4826 * mad)
    return med, scale


def _candidate_evidence(
    candidate: dict[str, Any],
    temporal: np.ndarray,
    *,
    bbox: tuple[int, int, int, int],
    med: float,
    scale: float,
) -> tuple[float, float]:
    bx0, by0, _bx1, _by1 = bbox
    px = int(round(_finite(candidate.get("camera_x")) - bx0))
    py = int(round(_finite(candidate.get("camera_y")) - by0))
    if 0 <= py < temporal.shape[0] and 0 <= px < temporal.shape[1]:
        value = float(temporal[py, px])
    else:
        value = 0.0
    sigma = max(0.0, (value - med) / max(0.35, scale))
    # Keep this a physical-evidence score.  No role/object priority appears here.
    abs_change = _finite(candidate.get("v2_absdiff", candidate.get("change_value", 0.0)))
    z = _finite(candidate.get("v2_zscore", 0.0))
    evidence = sigma + 0.10 * min(20.0, abs_change) + 0.08 * min(12.0, z)
    return float(evidence), float(value)


def _dedupe_candidates(candidates: Iterable[dict[str, Any]], radius: float = 3.5) -> list[dict[str, Any]]:
    ordered = sorted(
        (dict(c) for c in candidates),
        key=lambda c: (
            _finite(c.get("v251_region_evidence", 0.0)),
            _finite(c.get("score", 0.0)),
        ),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    r2 = float(radius) ** 2
    for cand in ordered:
        x, y = _finite(cand.get("camera_x")), _finite(cand.get("camera_y"))
        duplicate = None
        for old in kept:
            dx = x - _finite(old.get("camera_x"))
            dy = y - _finite(old.get("camera_y"))
            if dx * dx + dy * dy <= r2:
                duplicate = old
                break
        if duplicate is None:
            kept.append(cand)
            continue
        ids = set(str(duplicate.get("v251_region_objects", "")).split("|"))
        ids.update(str(cand.get("v251_region_objects", "")).split("|"))
        roles = set(str(duplicate.get("v251_region_roles", "")).split("|"))
        roles.update(str(cand.get("v251_region_roles", "")).split("|"))
        duplicate["v251_region_objects"] = "|".join(sorted(x for x in ids if x))
        duplicate["v251_region_roles"] = "|".join(sorted(x for x in roles if x))
    return kept


def _region_for_candidate(groups: Sequence[RegionGroupV251], candidate: dict[str, Any]) -> RegionGroupV251 | None:
    x = _finite(candidate.get("camera_x"))
    y = _finite(candidate.get("camera_y"))
    containing = [g for g in groups if g.contains(x, y)]
    if not containing:
        return None
    # Deterministic only; no gameplay priority.
    containing.sort(key=lambda g: (g.area, g.group_id))
    return containing[0]


def _fallback_evidence(candidate: dict[str, Any]) -> float:
    return (
        0.15 * max(0.0, _finite(candidate.get("score", 0.0)))
        + 0.25 * max(0.0, _finite(candidate.get("pre_shot_change", 0.0)))
        + 0.15 * max(0.0, _finite(candidate.get("change_value", 0.0)))
    )


def _balance_merged_candidates(
    candidates: Sequence[dict[str, Any]],
    groups: Sequence[RegionGroupV251],
    *,
    shot_id: int,
) -> list[dict[str, Any]]:
    per_group: dict[str, list[dict[str, Any]]] = {g.group_id: [] for g in groups}
    lookup = {g.group_id: g for g in groups}
    for original in candidates:
        cand = dict(original)
        group_name = str(cand.get("v251_region_group", ""))
        group = lookup.get(group_name)
        if group is None:
            group = _region_for_candidate(groups, cand)
        if group is None:
            continue
        cand.setdefault("v251_shot_id", int(shot_id))
        cand.setdefault("v251_region_group", group.group_id)
        cand.setdefault("v251_region_objects", "|".join(group.object_ids))
        cand.setdefault("v251_region_roles", "|".join(group.roles))
        cand.setdefault("v251_region_evidence", _fallback_evidence(cand))
        per_group[group.group_id].append(cand)

    chosen: list[dict[str, Any]] = []
    limit = _per_region_proposals()
    for group in groups:
        values = per_group.get(group.group_id, [])
        values.sort(
            key=lambda c: (
                1 if _finite(c.get("v251_registered_region_proposal", 0.0)) > 0.5 else 0,
                _finite(c.get("v251_region_evidence", 0.0)),
                _finite(c.get("score", 0.0)),
            ),
            reverse=True,
        )
        chosen.extend(values[:limit])
    return _dedupe_candidates(chosen)



def _top_sparse_region_peaks(
    score_map: np.ndarray,
    region_valid: np.ndarray,
    *,
    kernel: int,
    limit: int,
) -> list[tuple[float, int, int]]:
    """Local maxima on only the small physical region bounding box."""
    ys, xs = np.nonzero(region_valid)
    if xs.size == 0:
        return []
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    local_score = np.asarray(score_map[y0:y1, x0:x1], dtype=np.float32)
    local_valid = np.asarray(region_valid[y0:y1, x0:x1], dtype=bool)
    k = max(3, int(kernel))
    if k % 2 == 0:
        k += 1
    dilated = cv2.dilate(
        local_score,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
    )
    peak_mask = local_valid & (local_score >= (dilated - 1e-6))
    pys, pxs = np.nonzero(peak_mask)
    if pxs.size == 0:
        return []
    scores = local_score[pys, pxs]
    take = min(max(1, int(limit)), int(scores.size))
    if scores.size > take:
        idx = np.argpartition(scores, -take)[-take:]
        pys, pxs, scores = pys[idx], pxs[idx], scores[idx]
    order = np.argsort(scores)[::-1]
    return [
        (float(scores[i]), int(pxs[i]) + x0, int(pys[i]) + y0)
        for i in order
    ]


def _region_physical_candidates(
    generator: Any,
    scanner: Any,
    group: RegionGroupV251,
    *,
    saliency: np.ndarray,
    absdiff: np.ndarray,
    darkening: np.ndarray,
    dog: np.ndarray,
    zscore: np.ndarray,
    region_valid: np.ndarray,
    bbox: tuple[int, int, int, int],
    frame_ts: float,
    region_threshold: float,
    cfg: dict[str, Any],
    shot_id: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Generate a tiny high-recall physical pool for one region.

    Unlike calling the complete V2.22.5 extractor once per object, this scans
    only each region's small bounding box and computes the shared temporal map
    once in the caller.  That keeps the fairness change from multiplying the
    whole-ROI detector cost by the number of objects.
    """
    temporal = _temporal_map(absdiff, zscore, dog)
    min_change = max(0.0, _finite(cfg.get("min_temporal_change", 1.8), 1.8))
    min_z = max(0.0, _finite(cfg.get("min_zscore", 1.5), 1.5))
    strong_change = max(min_change, _finite(cfg.get("strong_temporal_change", 4.0), 4.0))
    primary_evidence = (absdiff >= strong_change) | ((absdiff >= min_change) & (zscore >= min_z))
    primary_mask = region_valid & primary_evidence & (saliency >= float(region_threshold))

    raw_limit = _setting_int("object_region_raw_peaks_v251", 64, 8, 256)
    primary = _top_sparse_region_peaks(
        saliency, primary_mask,
        kernel=int(cfg.get("local_max_kernel", 3) or 3),
        limit=raw_limit,
    )

    med, scale = _robust_stats(temporal[region_valid])
    temporal_sigma = _setting_float("object_region_temporal_sigma_v251", 1.35, 0.5, 4.0)
    temporal_min = _setting_float("object_region_temporal_min_v251", 3.2, 0.5, 20.0)
    temporal_threshold = max(temporal_min, med + temporal_sigma * scale)
    rescue_change = max(0.0, _finite(cfg.get("rescue_min_temporal_change", 1.8), 1.8))
    rescue_z = max(0.0, _finite(cfg.get("rescue_min_zscore", 1.25), 1.25))
    rescue_strong = max(rescue_change, _finite(cfg.get("rescue_strong_temporal_change", 4.0), 4.0))
    rescue_evidence = (absdiff >= rescue_strong) | ((absdiff >= rescue_change) & (zscore >= rescue_z))
    rescue_mask = region_valid & rescue_evidence & (temporal >= temporal_threshold)
    rescue = _top_sparse_region_peaks(
        temporal, rescue_mask,
        kernel=int(cfg.get("rescue_local_max_kernel", 3) or 3),
        limit=raw_limit,
    )

    merged: dict[tuple[int, int], tuple[float, set[str]]] = {}
    for score, px, py in primary:
        merged[(px, py)] = (float(score), {"primary"})
    for score, px, py in rescue:
        old = merged.get((px, py))
        if old is None:
            merged[(px, py)] = (float(score), {"region_temporal"})
        else:
            merged[(px, py)] = (max(float(score), old[0]), set(old[1]) | {"region_temporal"})

    rough = [(score, px, py, sources) for (px, py), (score, sources) in merged.items()]
    rough.sort(key=lambda item: item[0], reverse=True)
    refine_limit = min(len(rough), max(24, _per_region_proposals() * 6))
    bx0, by0, _bx1, _by1 = bbox
    candidates: list[dict[str, Any]] = []
    for peak_score, px, py, sources in rough[:refine_limit]:
        try:
            rx, ry, shift = generator._refine_peak(
                px=px, py=py, absdiff=absdiff, zscore=zscore,
                dog=dog, valid=region_valid, cfg=cfg,
            )
        except Exception:
            rx, ry, shift = px, py, 0.0
        try:
            features = generator._candidate_features(
                px=int(rx), py=int(ry), saliency=saliency, absdiff=absdiff,
                darkening=darkening, dog=dog, zscore=zscore,
            )
        except Exception:
            continue
        candidate: dict[str, Any] = {
            "camera_x": float(int(rx) + bx0),
            "camera_y": float(int(ry) + by0),
            "area": float(features.get("area", 1.0)),
            "radius": float(features.get("radius", 1.0)),
            "circularity": float(features.get("circularity", 0.75)),
            "score": float(features.get("score", 3.6)),
            "center_darkening": float(features.get("center_change", 0.0)),
            "local_contrast_gain": float(features.get("local_contrast", 0.0)),
            "blackhat_value": float(features.get("dog_value", 0.0)),
            "change_value": float(features.get("center_change", 0.0)),
            "pre_shot_change": float(features.get("center_change", 0.0)),
            "timestamp": float(frame_ts),
            "detector_v2": 1.0,
            "detector_v1": 0.0,
            "v2_saliency": float(saliency[int(ry), int(rx)]),
            "v2_zscore": float(features.get("zscore", 0.0)),
            "v2_absdiff": float(features.get("absdiff", 0.0)),
            "v2_darkening": float(features.get("darkening", 0.0)),
            "v2_dog": float(features.get("dog_value", 0.0)),
            "v2_primary_peak": 1.0 if "primary" in sources else 0.0,
            "v2_rescue_saliency": 0.0,
            "v2_rescue_temporal": 1.0 if "region_temporal" in sources else 0.0,
            "v2_rescue_blob": 0.0,
            "v2_refine_shift_px": float(shift),
            "v2225_fast_extract": 1.0,
            "v251_shot_id": int(shot_id),
            "v251_region_group": group.group_id,
            "v251_region_objects": "|".join(group.object_ids),
            "v251_region_roles": "|".join(group.roles),
            "v251_registered_region_proposal": 1.0,
        }
        try:
            generator._apply_known_hole_penalty(scanner, candidate)
        except Exception:
            pass
        evidence, change = _candidate_evidence(candidate, temporal, bbox=bbox, med=med, scale=scale)
        candidate["v251_region_evidence"] = float(evidence)
        candidate["v251_region_temporal"] = float(change)
        candidates.append(candidate)

    candidates = _dedupe_candidates(candidates)
    candidates.sort(
        key=lambda c: (_finite(c.get("v251_region_evidence")), _finite(c.get("score"))),
        reverse=True,
    )
    candidates = candidates[:_per_region_proposals()]
    return candidates, {
        "primary_raw": float(len(primary)),
        "temporal_raw": float(len(rescue)),
        "best": max((_finite(c.get("v251_region_evidence")) for c in candidates), default=0.0),
        "change": max((_finite(c.get("v251_region_temporal")) for c in candidates), default=0.0),
    }


def _install_settings_defaults() -> None:
    defaults = {
        "object_region_proposal_enabled_v251": True,
        "object_region_proposal_log_v251": True,
        "object_region_margin_px_v251": 36.0,
        "object_region_proposals_per_region_v251": 8,
        "object_region_confirmed_per_region_v251": 2,
        "object_region_confirmed_total_v251": 8,
        "object_region_group_overlap_v251": 0.80,
        "object_region_temporal_sigma_v251": 1.35,
        "object_region_temporal_min_v251": 3.2,
        "object_region_raw_peaks_v251": 64,
    }
    try:
        import src.engine.ai.runtime as runtime_module
        runtime_module.DEFAULT_SETTINGS.update(defaults)
        existing = getattr(runtime_module, "_RUNTIME", None)
        if existing is not None:
            for key, value in defaults.items():
                getattr(existing, "settings", {}).setdefault(key, value)
    except Exception:
        pass


def _install_candidate_region_patch() -> None:
    from src.engine.camera.candidate_generator_v2 import CandidateGeneratorV2

    if getattr(CandidateGeneratorV2, "_v251_region_proposal_patch", False):
        return

    previous_extract = CandidateGeneratorV2._extract_candidates
    previous_generate = CandidateGeneratorV2.generate

    def extract_v251(self, *args, **kwargs):
        scanner = kwargs.get("scanner")
        sid = _shot_id_from_scanner(scanner)
        live_worker = threading.current_thread().name.startswith("shot-cv-v2224")
        if not _enabled() or not live_worker or sid <= 0:
            return previous_extract(self, *args, **kwargs)
        # The explicit V2.22.5 rescue must remain exactly the old global path.
        if rescue_router_v2225.requested(sid):
            return previous_extract(self, *args, **kwargs)

        groups = _camera_regions_to_work_groups(scanner, sid)
        if not groups:
            return previous_extract(self, *args, **kwargs)
        started = time.perf_counter()

        saliency = kwargs.get("saliency")
        absdiff = kwargs.get("absdiff")
        darkening = kwargs.get("darkening")
        dog = kwargs.get("dog")
        zscore = kwargs.get("zscore")
        valid = kwargs.get("valid")
        bbox = kwargs.get("bbox")
        cfg = kwargs.get("cfg") or {}
        if not all(isinstance(x, np.ndarray) for x in (saliency, absdiff, darkening, dog, zscore, valid)):
            return previous_extract(self, *args, **kwargs)
        if not (isinstance(bbox, tuple) and len(bbox) == 4):
            return previous_extract(self, *args, **kwargs)

        all_candidates: list[dict[str, Any]] = []
        total_primary = total_temporal = 0
        region_logs: list[dict[str, Any]] = []

        for group in groups:
            region_valid = _bbox_region_mask(saliency.shape, valid, bbox, group)
            nvalid = int(np.count_nonzero(region_valid))
            if nvalid <= 0:
                region_logs.append({"group": group.group_id, "valid": 0, "candidates": 0, "primary": 0, "temporal": 0, "best": 0.0, "change": 0.0})
                continue
            try:
                region_threshold, _stats = self._robust_threshold(saliency, valid=region_valid, cfg=cfg)
            except Exception:
                region_threshold = _finite(kwargs.get("threshold", 0.0))
            kept, evidence_diag = _region_physical_candidates(
                self, scanner, group,
                saliency=saliency, absdiff=absdiff, darkening=darkening,
                dog=dog, zscore=zscore, region_valid=region_valid,
                bbox=bbox, frame_ts=_finite(kwargs.get("frame_ts", 0.0)),
                region_threshold=float(region_threshold), cfg=cfg, shot_id=sid,
            )
            all_candidates.extend(kept)
            total_primary += int(evidence_diag.get("primary_raw", 0.0))
            total_temporal += int(evidence_diag.get("temporal_raw", 0.0))
            region_logs.append({
                "group": group.group_id,
                "valid": nvalid,
                "candidates": len(kept),
                "primary": int(evidence_diag.get("primary_raw", 0.0)),
                "temporal": int(evidence_diag.get("temporal_raw", 0.0)),
                "best": float(evidence_diag.get("best", 0.0)),
                "change": float(evidence_diag.get("change", 0.0)),
            })

        balanced = _dedupe_candidates(all_candidates)
        try:
            scanner.last_window_debug["v251_region_groups"] = float(len(groups))
            scanner.last_window_debug["v251_region_raw"] = float(len(all_candidates))
            scanner.last_window_debug["v251_region_balanced"] = float(len(balanced))
            scanner.last_window_debug["v251_region_extract_ms"] = float((time.perf_counter() - started) * 1000.0)
            # Preserve V2.22.5 telemetry as aggregate rather than whichever
            # region happened to run last.
            scanner.last_window_debug["v2225_fast_primary"] = float(total_primary)
            scanner.last_window_debug["v2225_fast_rescue"] = float(total_temporal)
            scanner.last_window_debug["v2225_fast_output"] = float(len(balanced))
        except Exception:
            pass

        if _log_enabled():
            for item in region_logs:
                print(
                    f"[V2.25.1 REGION-PROPOSAL] shot={sid} object={item['group']} "
                    f"valid={item['valid']} candidates={item['candidates']} primary={item['primary']} temporal={item.get('temporal', 0)} "
                    f"best_sigma={item['best']:.2f} best_change={item['change']:.2f}"
                )
            print(
                f"[V2.25.1 REGION-POOL] shot={sid} regions={len(groups)} "
                f"raw={len(all_candidates)} balanced={len(balanced)} "
                f"time={(time.perf_counter() - started) * 1000.0:.1f}ms"
            )
        return balanced

    def generate_v251(self, scanner, gray, frame_ts, legacy_candidates):
        sid = _shot_id_from_scanner(scanner)
        live_worker = threading.current_thread().name.startswith("shot-cv-v2224")
        rescue_before = bool(sid > 0 and rescue_router_v2225.requested(sid))
        result = previous_generate(self, scanner, gray, frame_ts, legacy_candidates)
        if not _enabled() or not live_worker or sid <= 0 or rescue_before:
            return result
        groups = _camera_regions_to_work_groups(scanner, sid)
        if not groups:
            return result
        try:
            original_count = len(result.candidates)
            result.candidates = _balance_merged_candidates(result.candidates, groups, shot_id=sid)
            result.telemetry = dict(result.telemetry or {})
            result.telemetry["v251_region_balanced"] = True
            result.telemetry["v251_before_balance"] = original_count
            result.telemetry["v251_after_balance"] = len(result.candidates)
            scanner.last_window_debug["v251_merged_before"] = float(original_count)
            scanner.last_window_debug["v251_merged_after"] = float(len(result.candidates))
        except Exception:
            # Diagnostic/balancing failure must never block the physical detector.
            return result
        return result

    CandidateGeneratorV2._extract_candidates = extract_v251
    CandidateGeneratorV2.generate = generate_v251
    CandidateGeneratorV2._v251_region_proposal_patch = True
    CandidateGeneratorV2._v251_previous_extract = previous_extract
    CandidateGeneratorV2._v251_previous_generate = previous_generate


def _confirm_strength(candidate: dict[str, Any]) -> float:
    return (
        1.15 * max(0.0, _finite(candidate.get("v251_region_evidence", 0.0)))
        + 0.55 * max(0.0, _finite(candidate.get("v2225_confirm_center_abs", 0.0)))
        + 0.90 * max(0.0, _finite(candidate.get("v2225_confirm_compact", 0.0)))
        + 0.08 * max(0.0, _finite(candidate.get("v2225_confirm_peak_abs", 0.0)))
        + 0.25 * max(0.0, _finite(candidate.get("v2225_confirm_darkening", 0.0)))
    )


def _balance_confirmed(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for original in candidates:
        cand = dict(original)
        group = str(cand.get("v251_region_group", ""))
        if not group:
            continue
        cand["v251_confirm_score"] = float(_confirm_strength(cand))
        groups.setdefault(group, []).append(cand)
    if not groups:
        return list(candidates)
    for values in groups.values():
        values.sort(key=lambda c: (_finite(c.get("v251_confirm_score")), _finite(c.get("score"))), reverse=True)

    # Preserve at least one physical survivor per region, then use remaining
    # capacity for the strongest extra confirmations. This prevents one noisy
    # region from monopolising the second-frame evidence.
    selected: list[dict[str, Any]] = []
    extras: list[dict[str, Any]] = []
    per_limit = _per_region_confirmed()
    for group in sorted(groups):
        values = groups[group]
        if values:
            selected.append(values[0])
            extras.extend(values[1:per_limit])
    extras.sort(key=lambda c: (_finite(c.get("v251_confirm_score")), _finite(c.get("score"))), reverse=True)
    total_limit = max(len(selected), _confirmed_total())
    selected.extend(extras[:max(0, total_limit - len(selected))])
    selected.sort(key=lambda c: (_finite(c.get("v251_confirm_score")), _finite(c.get("score"))), reverse=True)
    return _dedupe_candidates(selected)


def _install_confirmation_patch() -> None:
    import src.engine.shot_fast_v2225 as fast

    if getattr(fast, "_v251_region_confirm_patch", False):
        return
    previous_confirm = fast.local_confirm_candidates_v2225

    def confirm_v251(pre_gray, current_gray, candidates, *, frame_ts, config=None):
        confirmed, diag = previous_confirm(
            pre_gray, current_gray, candidates, frame_ts=frame_ts, config=config
        )
        annotated = [c for c in confirmed if int(_finite(c.get("v251_shot_id", 0.0))) > 0]
        if not _enabled() or not annotated:
            return confirmed, diag
        balanced = _balance_confirmed(annotated)
        sid = int(_finite(annotated[0].get("v251_shot_id", 0.0)))
        if _log_enabled():
            best = max((_finite(c.get("v251_confirm_score", 0.0)) for c in balanced), default=0.0)
            print(
                f"[V2.25.1 REGION-CONFIRM] shot={sid} tested={int(diag.get('tested', 0))} "
                f"v2225={len(confirmed)} balanced={len(balanced)} "
                f"groups={len(set(str(c.get('v251_region_group', '')) for c in balanced))} best={best:.2f}"
            )
        new_diag = dict(diag)
        new_diag["v251_before_balance"] = float(len(confirmed))
        new_diag["v251_after_balance"] = float(len(balanced))
        return balanced, new_diag

    fast.local_confirm_candidates_v2225 = confirm_v251
    fast._v251_region_confirm_patch = True
    fast._v251_previous_local_confirm = previous_confirm


def _install_track_selector_patch() -> None:
    from src.engine.camera.hit_scanner import HitScanner

    if getattr(HitScanner, "_v251_physical_track_selector", False):
        return
    previous_best = HitScanner._best_track_for_event

    def best_track_v251(self, event):
        sid = int(getattr(event, "shot_id", 0) or 0)
        snap = object_hit_registry_v2223.snapshot_for_shot(sid) if sid > 0 else None
        if not _enabled() or snap is None or not tuple(getattr(snap, "camera_regions", ()) or ()):
            return previous_best(self, event)

        eligible = []
        for track in getattr(self, "_active_tracks", {}).values():
            onset_dt = float(track.first_seen_ts) - float(event.peak_ts)
            if onset_dt < -float(self.association_lead_s) or onset_dt > float(self.association_lag_s):
                continue
            if track.emitted and event.matched_track_id != track.track_id:
                continue
            cand = getattr(track, "last_candidate", {}) or {}
            if _finite(cand.get("v251_confirm_score", 0.0)) <= 0.0:
                continue
            eligible.append((track, cand, onset_dt))
        if not eligible:
            return previous_best(self, event)

        # All terms are physical detector evidence/timing. No target role,
        # object priority, owner or game score participates in this choice.
        best, cand, onset = max(
            eligible,
            key=lambda item: (
                _finite(item[1].get("v251_confirm_score", 0.0)),
                _finite(item[1].get("v251_region_evidence", 0.0)),
                _finite(item[0].best_score, 0.0),
                -abs(float(item[2])),
            ),
        )
        try:
            self.last_event_debug["v251_selector"] = "confirmed_physical"
            self.last_event_debug["v251_confirm_score"] = _finite(cand.get("v251_confirm_score", 0.0))
            self.last_event_debug["v251_region_evidence"] = _finite(cand.get("v251_region_evidence", 0.0))
        except Exception:
            pass
        return best

    HitScanner._best_track_for_event = best_track_v251
    HitScanner._v251_physical_track_selector = True
    HitScanner._v251_previous_best_track = previous_best


def install_v251_runtime(AppClass: Any) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_settings_defaults()
    _install_candidate_region_patch()
    _install_confirmation_patch()
    _install_track_selector_patch()
    AppClass._v251_region_proposal_patch = True
    _INSTALLED = True
    print(
        f"[V2.25.1] balanced per-object physical proposal/confirmation installed "
        f"(per_region={_per_region_proposals()}, confirm={_per_region_confirmed()}, global rescue preserved)"
    )


__all__ = [
    "SCHEMA_VERSION",
    "PATCH_REVISION",
    "RegionGroupV251",
    "_merge_groups",
    "_bbox_region_mask",
    "_balance_confirmed",
    "install_v251_runtime",
]
