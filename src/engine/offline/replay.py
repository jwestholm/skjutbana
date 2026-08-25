from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .evidence import EvidenceConfig, build_evidence, extract_overlay_candidates, merge_candidate_sources
from .metrics import ReplayMetrics, nearest_distance
from .shot_case import ShotCase


@dataclass
class ReplaySettings:
    use_current_detector: bool = True
    use_overlay: bool = True
    candidate_limit: int = 500
    union_limit: int = 700
    merge_radius_px: float = 5.0
    post_interval_s: float = 1.0 / 30.0
    evidence_config_path: Path | None = None

    # Compatibility for early V2.12 scripts/tests that used `use_v2=`.
    def __init__(
        self,
        use_current_detector: bool = True,
        use_overlay: bool = True,
        candidate_limit: int = 500,
        union_limit: int = 700,
        merge_radius_px: float = 5.0,
        post_interval_s: float = 1.0 / 30.0,
        evidence_config_path: Path | None = None,
        use_v2: bool | None = None,
    ) -> None:
        if use_v2 is not None:
            use_current_detector = bool(use_v2)
        self.use_current_detector = bool(use_current_detector)
        self.use_overlay = bool(use_overlay)
        self.candidate_limit = int(candidate_limit)
        self.union_limit = int(union_limit)
        self.merge_radius_px = float(merge_radius_px)
        self.post_interval_s = float(post_interval_s)
        self.evidence_config_path = evidence_config_path

    @property
    def use_v2(self) -> bool:
        return self.use_current_detector


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError(f"Could not read image: {path}")
    return image


def _load_case_frames(case: ShotCase, root: Path | None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    pre = [load_gray(path) for path in case.resolved_pre_paths(root)]
    post = [load_gray(path) for path in case.resolved_post_paths(root)]
    if not pre or not post:
        raise ValueError(f"Shot {case.shot_id}: missing pre/post frames")
    shape = pre[0].shape
    for frame in pre + post:
        if frame.shape != shape:
            raise ValueError(f"Shot {case.shot_id}: pre/post shape mismatch")
    return pre, post


def _known_holes_from_metadata(metadata: dict[str, Any]) -> list[tuple[float, float]]:
    values = metadata.get("known_holes") or metadata.get("old_holes") or []
    result: list[tuple[float, float]] = []
    if isinstance(values, list):
        for value in values:
            try:
                if isinstance(value, dict):
                    result.append((float(value.get("camera_x", value.get("x"))), float(value.get("camera_y", value.get("y")))))
                elif isinstance(value, (list, tuple)) and len(value) >= 2:
                    result.append((float(value[0]), float(value[1])))
            except Exception:
                pass
    return result


def _build_live_detector():
    from .live_detector_replay import LiveHybridReplayDetector

    return LiveHybridReplayDetector()


def replay_case(
    case: ShotCase,
    *,
    root: Path | None = None,
    settings: ReplaySettings | None = None,
    live_detector: Any | None = None,
) -> dict[str, Any]:
    settings = settings or ReplaySettings()
    pre, post = _load_case_frames(case, root)
    gt_xy = case.ground_truth.as_xy() if case.ground_truth else None

    detector_candidates: list[dict[str, Any]] = []
    detector_v1_candidates: list[dict[str, Any]] = []
    detector_v2_candidates: list[dict[str, Any]] = []
    detector_agreement_candidates: list[dict[str, Any]] = []
    detector_telemetry: dict[str, Any] = {}

    timing_ms: dict[str, float] = {}

    detector = live_detector
    if settings.use_current_detector:
        detector_started = time.perf_counter()
        detector = detector or _build_live_detector()
        detector_result = detector.detect(
            pre_frames=pre,
            post_frames=post,
            known_holes=_known_holes_from_metadata(case.metadata),
            ground_truth=gt_xy,
            candidate_limit=settings.candidate_limit,
            post_interval_s=settings.post_interval_s,
        )
        detector_candidates = [dict(item) for item in detector_result.candidates]
        detector_v1_candidates = [dict(item) for item in detector_result.v1_candidates]
        detector_v2_candidates = [dict(item) for item in detector_result.v2_candidates]
        detector_agreement_candidates = [dict(item) for item in detector_result.agreement_candidates]
        detector_telemetry = dict(detector_result.telemetry)
        timing_ms["current_detector"] = round(1000.0 * (time.perf_counter() - detector_started), 3)

    overlay_candidates: list[dict[str, Any]] = []
    component_candidates: dict[str, list[dict[str, Any]]] = {}
    overlay_meta: dict[str, Any] = {}
    if settings.use_overlay:
        overlay_started = time.perf_counter()
        evidence_cfg = EvidenceConfig.from_file(settings.evidence_config_path)
        bundle = build_evidence(pre, post, config=evidence_cfg)
        overlay_candidates = extract_overlay_candidates(bundle.fused, config=evidence_cfg)
        # Keep each component independently measurable.  This is deliberately
        # offline-only: add/keep/reject one overlay at a time instead of hiding
        # every change behind one fused percentage.
        for name, component in bundle.overlays.items():
            component_candidates[name] = extract_overlay_candidates(component, config=evidence_cfg)
        # Attach all component values to each fused proposal for future
        # ranking/Hole-AI/fusion without recomputing the maps.
        for candidate in overlay_candidates:
            x = int(round(float(candidate["camera_x"])))
            y = int(round(float(candidate["camera_y"])))
            if 0 <= y < bundle.fused.values.shape[0] and 0 <= x < bundle.fused.values.shape[1]:
                for name, overlay in bundle.overlays.items():
                    candidate[f"overlay_{name}"] = float(overlay.values[y, x])
        overlay_meta = dict(bundle.metadata)
        overlay_meta["component_names"] = list(bundle.overlays)
        overlay_meta["fused_max"] = float(np.max(bundle.fused.values))
        timing_ms["overlay"] = round(1000.0 * (time.perf_counter() - overlay_started), 3)

    union_started = time.perf_counter()
    union_candidates = merge_candidate_sources(
        (("current_detector", detector_candidates), ("overlay", overlay_candidates)),
        merge_radius_px=settings.merge_radius_px,
        limit=settings.union_limit,
    )
    timing_ms["union_merge"] = round(1000.0 * (time.perf_counter() - union_started), 3)

    def source_payload(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return {
            "candidate_count": len(candidates),
            "nearest_gt_distance": nearest_distance(candidates, gt_xy) if gt_xy else None,
        }

    sources: dict[str, dict[str, Any]] = {}
    candidate_sets: dict[str, list[dict[str, Any]]] = {}
    if settings.use_current_detector:
        sources.update({
            "current_detector": {**source_payload(detector_candidates), "telemetry": detector_telemetry},
            "current_detector_v1": source_payload(detector_v1_candidates),
            "current_detector_v2": source_payload(detector_v2_candidates),
            "current_detector_agreement": source_payload(detector_agreement_candidates),
        })
        candidate_sets.update({
            "current_detector": detector_candidates,
            "current_detector_v1": detector_v1_candidates,
            "current_detector_v2": detector_v2_candidates,
            "current_detector_agreement": detector_agreement_candidates,
        })
    if settings.use_overlay:
        sources.update({
            **{f"overlay_{name}": source_payload(candidates) for name, candidates in component_candidates.items()},
            "overlay": {**source_payload(overlay_candidates), "telemetry": overlay_meta},
        })
        candidate_sets.update({
            **{f"overlay_{name}": candidates for name, candidates in component_candidates.items()},
            "overlay": overlay_candidates,
        })
    # Union is meaningful whenever at least one proposal source is enabled.
    sources["union"] = source_payload(union_candidates)
    candidate_sets["union"] = union_candidates

    return {
        "schema_version": "2.12",
        "shot_id": case.shot_id,
        "session_id": case.session_id,
        "ground_truth": case.ground_truth.to_dict() if case.ground_truth else None,
        "timing_ms": timing_ms,
        "sources": sources,
        "candidates": candidate_sets,
    }


def benchmark_cases(
    cases: Sequence[ShotCase],
    *,
    root: Path | None = None,
    settings: ReplaySettings | None = None,
    include_candidates: bool = False,
) -> dict[str, Any]:
    settings = settings or ReplaySettings()
    metrics = ReplayMetrics()
    shot_rows: list[dict[str, Any]] = []
    detector = _build_live_detector() if settings.use_current_detector else None
    errors: list[dict[str, str]] = []

    for case in cases:
        try:
            result = replay_case(
                case,
                root=root,
                settings=settings,
                live_detector=detector,
            )
            metrics.add(result)
            if not include_candidates:
                result.pop("candidates", None)
            shot_rows.append(result)
        except Exception as exc:
            errors.append({"shot_id": case.shot_id, "error": f"{type(exc).__name__}: {exc}"})

    return {
        "schema_version": "2.12",
        "purpose": "offline_current_live_detector_replay_and_evidence_complementarity",
        "settings": {
            "use_current_detector": settings.use_current_detector,
            "use_overlay": settings.use_overlay,
            "candidate_limit": settings.candidate_limit,
            "union_limit": settings.union_limit,
            "merge_radius_px": settings.merge_radius_px,
        },
        "summary": metrics.to_dict(),
        "shots": shot_rows,
        "errors": errors,
    }


def write_benchmark(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)
