"""Temporary demo-only OBJECT precision helper for Skjutprylar.

This module is intentionally isolated from the canonical exact-coordinate
hit authority.  It observes the V2.4 PRE/POST image evidence that already
exists inside CandidateGeneratorV2 and asks a simpler question:

    "Which frozen target region received the strongest local physical change?"

It does NOT move or replace the detector's emitted HitEvent, does not change
candidate ranking, and does not alter camera/calibration code.  The Skjutprylar
game may consume the result as a temporary demo gameplay authority.

The extension only activates for GameObjects tagged ``shoot_stuff`` so other
games pay essentially no cost and cannot accidentally consume this demo path.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import threading
import time
from typing import Any, Mapping

import numpy as np

from src.engine.input.object_hit_v2223 import object_hit_registry_v2223


DEMO_TAG = "shoot_stuff"

# Deliberately permissive: this is a demo OBJECT detector, not an accuracy claim.
# A hit still needs real PRE/POST evidence inside one of the frozen object regions.
MIN_PHYSICAL_SIGNAL = 0.16
MIN_FINAL_SCORE = 0.18
MIN_CONFIDENCE = 0.22
MIN_MARGIN = 0.008
LATE_FRAME_AGE_S = 0.075
INNER_INSET_FRACTION = 0.20
RING_EXPAND_FRACTION = 0.35
MAX_SHOTS = 32


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else float(default)
    except Exception:
        return float(default)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _top_mean(values: np.ndarray, *, fraction: float = 0.035, min_count: int = 3, max_count: int = 28) -> float:
    if values.size <= 0:
        return 0.0
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size <= 0:
        return 0.0
    count = int(round(float(flat.size) * float(fraction)))
    count = max(int(min_count), min(int(max_count), count))
    count = min(count, int(flat.size))
    if count >= flat.size:
        return float(np.mean(flat))
    # partition is substantially cheaper than sorting the whole region.
    top = np.partition(flat, flat.size - count)[-count:]
    return float(np.mean(top))


def _region_is_demo_target(region: Any) -> bool:
    try:
        metadata = dict(getattr(region, "metadata", {}) or {})
        snapshot = metadata.get("v250_object_snapshot")
        if not isinstance(snapshot, Mapping):
            return False
        tags = {str(tag) for tag in snapshot.get("tags", ()) or ()}
        return DEMO_TAG in tags
    except Exception:
        return False


def _candidate_quality(candidate: Mapping[str, Any]) -> float:
    prior = _clip01(_safe_float(candidate.get("v24_patch_prior", 0.0)))
    abs_value = _safe_float(
        candidate.get(
            "v2_absdiff",
            candidate.get("v24_patch_core_abs", candidate.get("pre_shot_change", 0.0)),
        )
    )
    z_value = _safe_float(candidate.get("v2_zscore", candidate.get("v24_patch_core_z", 0.0)))
    hits = max(
        _safe_float(candidate.get("shot_accumulator_hits", 0.0)),
        _safe_float(candidate.get("v26_vault_hits", 0.0)),
    )
    agreement = 1.0 if _safe_float(candidate.get("detector_agreement", 0.0)) > 0.5 else 0.0
    return _clip01(
        0.34 * prior
        + 0.25 * math.tanh(max(0.0, abs_value) / 5.0)
        + 0.18 * math.tanh(max(0.0, z_value) / 2.2)
        + 0.13 * math.tanh(max(0.0, hits) / 2.0)
        + 0.10 * agreement
    )


@dataclass(frozen=True, slots=True)
class DemoObjectZoneEvidence:
    object_id: str
    score: float
    physical_signal: float
    best_frame_score: float
    late_best_score: float
    mean_score: float
    late_mean_score: float
    wins: int
    late_wins: int
    frames: int
    late_frames: int
    max_abs: float
    max_dark: float
    max_z: float
    max_candidate_support: float


@dataclass(frozen=True, slots=True)
class DemoObjectZoneResult:
    shot_id: int
    object_id: str | None
    accepted: bool
    confidence: float
    margin: float
    score: float
    physical_signal: float
    frames: int
    late_frames: int
    reason: str
    evidence: tuple[DemoObjectZoneEvidence, ...]
    updated_ts: float


class DemoObjectZoneRegistry:
    """Thread-safe per-shot accumulator for temporary OBJECT-level demo hits."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._shots: dict[int, dict[str, Any]] = {}
        self._results: dict[int, DemoObjectZoneResult] = {}

    def reset(self) -> None:
        with self._lock:
            self._shots.clear()
            self._results.clear()

    def _prune(self) -> None:
        if len(self._shots) <= MAX_SHOTS:
            return
        keep = set(sorted(self._shots)[-MAX_SHOTS:])
        self._shots = {sid: state for sid, state in self._shots.items() if sid in keep}
        self._results = {sid: result for sid, result in self._results.items() if sid in keep}

    def observe_frame(
        self,
        *,
        shot_id: int,
        peak_ts: float,
        frame_ts: float,
        metrics: Mapping[str, Mapping[str, float]],
    ) -> DemoObjectZoneResult | None:
        if int(shot_id) <= 0 or not metrics:
            return None

        with self._lock:
            state = self._shots.setdefault(
                int(shot_id),
                {
                    "shot_id": int(shot_id),
                    "peak_ts": float(peak_ts),
                    "frames": 0,
                    "late_frames": 0,
                    "objects": {},
                    "last_ts": float(frame_ts),
                },
            )
            state["frames"] = int(state.get("frames", 0)) + 1
            state["last_ts"] = float(frame_ts)
            age_s = max(0.0, float(frame_ts) - float(state.get("peak_ts", peak_ts)))
            is_late = age_s >= LATE_FRAME_AGE_S
            if is_late:
                state["late_frames"] = int(state.get("late_frames", 0)) + 1

            raw_scores = [max(0.0, _safe_float(m.get("frame_score", 0.0))) for m in metrics.values()]
            median = float(np.median(np.asarray(raw_scores, dtype=np.float32))) if raw_scores else 0.0
            mad = (
                float(np.median(np.abs(np.asarray(raw_scores, dtype=np.float32) - median)))
                if raw_scores
                else 0.0
            )
            denom = max(0.035, 1.4826 * mad + 0.020)

            frame_fused: dict[str, float] = {}
            for object_id, metric in metrics.items():
                raw = max(0.0, _safe_float(metric.get("frame_score", 0.0)))
                relative = max(0.0, (raw - median) / denom)
                relative01 = _clip01(math.tanh(relative / 2.0))
                # Later frames get a modest bonus because a real hole persists
                # while board-motion artifacts normally decay.
                late_bonus = 0.035 if is_late else 0.0
                fused = _clip01(0.76 * raw + 0.24 * relative01 + late_bonus)
                frame_fused[str(object_id)] = fused

            winner_id = max(frame_fused, key=frame_fused.get)
            ordered_frame = sorted(frame_fused.values(), reverse=True)
            frame_margin = ordered_frame[0] - (ordered_frame[1] if len(ordered_frame) > 1 else 0.0)

            for object_id, metric in metrics.items():
                key = str(object_id)
                obj = state["objects"].setdefault(
                    key,
                    {
                        "frames": 0,
                        "late_frames": 0,
                        "sum_score": 0.0,
                        "late_sum_score": 0.0,
                        "best_score": 0.0,
                        "late_best_score": 0.0,
                        "wins": 0,
                        "late_wins": 0,
                        "max_physical": 0.0,
                        "max_abs": 0.0,
                        "max_dark": 0.0,
                        "max_z": 0.0,
                        "max_candidate": 0.0,
                    },
                )
                fused = frame_fused[key]
                obj["frames"] += 1
                obj["sum_score"] += fused
                obj["best_score"] = max(float(obj["best_score"]), fused)
                obj["max_physical"] = max(
                    float(obj["max_physical"]),
                    _safe_float(metric.get("physical_signal", 0.0)),
                )
                obj["max_abs"] = max(float(obj["max_abs"]), _safe_float(metric.get("abs_top", 0.0)))
                obj["max_dark"] = max(float(obj["max_dark"]), _safe_float(metric.get("dark_top", 0.0)))
                obj["max_z"] = max(float(obj["max_z"]), _safe_float(metric.get("z_top", 0.0)))
                obj["max_candidate"] = max(
                    float(obj["max_candidate"]),
                    _safe_float(metric.get("candidate_support", 0.0)),
                )
                if is_late:
                    obj["late_frames"] += 1
                    obj["late_sum_score"] += fused
                    obj["late_best_score"] = max(float(obj["late_best_score"]), fused)
                if key == winner_id and frame_margin >= 0.004:
                    obj["wins"] += 1
                    if is_late:
                        obj["late_wins"] += 1

            result = self._build_result(state)
            self._results[int(shot_id)] = result
            self._prune()
            return result

    def _build_result(self, state: Mapping[str, Any]) -> DemoObjectZoneResult:
        total_frames = max(1, int(state.get("frames", 0)))
        total_late = max(0, int(state.get("late_frames", 0)))
        evidence: list[DemoObjectZoneEvidence] = []

        for object_id, obj in dict(state.get("objects", {})).items():
            frames = max(1, int(obj.get("frames", 0)))
            late_frames = max(0, int(obj.get("late_frames", 0)))
            mean_score = float(obj.get("sum_score", 0.0)) / float(frames)
            late_mean = (
                float(obj.get("late_sum_score", 0.0)) / float(late_frames)
                if late_frames > 0
                else mean_score
            )
            best_score = float(obj.get("best_score", 0.0))
            late_best = float(obj.get("late_best_score", 0.0)) if late_frames > 0 else best_score
            wins = int(obj.get("wins", 0))
            late_wins = int(obj.get("late_wins", 0))
            win_ratio = wins / float(total_frames)
            late_win_ratio = late_wins / float(max(1, total_late))
            candidate = _clip01(float(obj.get("max_candidate", 0.0)))

            # OBJECT demo score intentionally favours persistent region dominance
            # over one globally high candidate.
            score = _clip01(
                0.26 * best_score
                + 0.28 * late_best
                + 0.18 * mean_score
                + 0.12 * late_mean
                + 0.09 * win_ratio
                + 0.05 * late_win_ratio
                + 0.02 * candidate
            )
            evidence.append(
                DemoObjectZoneEvidence(
                    object_id=str(object_id),
                    score=score,
                    physical_signal=_clip01(float(obj.get("max_physical", 0.0))),
                    best_frame_score=best_score,
                    late_best_score=late_best,
                    mean_score=mean_score,
                    late_mean_score=late_mean,
                    wins=wins,
                    late_wins=late_wins,
                    frames=frames,
                    late_frames=late_frames,
                    max_abs=float(obj.get("max_abs", 0.0)),
                    max_dark=float(obj.get("max_dark", 0.0)),
                    max_z=float(obj.get("max_z", 0.0)),
                    max_candidate_support=candidate,
                )
            )

        evidence.sort(key=lambda item: (item.score, item.physical_signal, item.late_wins, item.wins), reverse=True)
        if not evidence:
            return DemoObjectZoneResult(
                shot_id=int(state.get("shot_id", 0)),
                object_id=None,
                accepted=False,
                confidence=0.0,
                margin=0.0,
                score=0.0,
                physical_signal=0.0,
                frames=total_frames,
                late_frames=total_late,
                reason="no_regions",
                evidence=(),
                updated_ts=float(state.get("last_ts", time.time())),
            )

        best = evidence[0]
        second_score = evidence[1].score if len(evidence) > 1 else 0.0
        margin = max(0.0, best.score - second_score)
        uniqueness = _clip01(math.tanh(margin / 0.055))
        persistence = _clip01(
            0.55 * (best.wins / float(total_frames))
            + 0.45 * (best.late_wins / float(max(1, total_late)))
        )
        confidence = _clip01(
            0.50 * best.score
            + 0.22 * best.physical_signal
            + 0.18 * uniqueness
            + 0.10 * persistence
        )

        physical_ok = best.physical_signal >= MIN_PHYSICAL_SIGNAL
        score_ok = best.score >= MIN_FINAL_SCORE
        # With several frames, repeated region wins can compensate for a tiny
        # margin. With only one frame, require a visibly stronger physical signal.
        uniqueness_ok = (
            margin >= MIN_MARGIN
            or best.wins >= 2
            or best.late_wins >= 1
            or (total_frames <= 1 and best.physical_signal >= 0.34)
        )
        accepted = bool(
            physical_ok
            and score_ok
            and uniqueness_ok
            and confidence >= MIN_CONFIDENCE
        )
        if not physical_ok:
            reason = "no_physical_signal"
        elif not score_ok:
            reason = "weak_region_score"
        elif not uniqueness_ok:
            reason = "ambiguous_regions"
        elif confidence < MIN_CONFIDENCE:
            reason = "low_confidence"
        else:
            reason = "accepted_demo_object_hit"

        return DemoObjectZoneResult(
            shot_id=int(state.get("shot_id", 0)),
            object_id=best.object_id if accepted else None,
            accepted=accepted,
            confidence=confidence,
            margin=margin,
            score=best.score,
            physical_signal=best.physical_signal,
            frames=total_frames,
            late_frames=total_late,
            reason=reason,
            evidence=tuple(evidence),
            updated_ts=float(state.get("last_ts", time.time())),
        )

    def result_for_shot(self, shot_id: int | None) -> DemoObjectZoneResult | None:
        if shot_id is None:
            return None
        with self._lock:
            return self._results.get(int(shot_id))


demo_object_zone_registry = DemoObjectZoneRegistry()


def _array_metric(
    array: np.ndarray,
    valid: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> float:
    if x1 <= x0 or y1 <= y0:
        return 0.0
    patch = array[y0:y1, x0:x1]
    mask = np.asarray(valid[y0:y1, x0:x1], dtype=bool)
    if patch.size <= 0 or mask.size <= 0 or not np.any(mask):
        return 0.0
    return _top_mean(patch[mask])


def _ring_metric(
    array: np.ndarray,
    valid: np.ndarray,
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    ex0: int,
    ey0: int,
    ex1: int,
    ey1: int,
) -> float:
    if ex1 <= ex0 or ey1 <= ey0:
        return 0.0
    patch = array[ey0:ey1, ex0:ex1]
    mask = np.asarray(valid[ey0:ey1, ex0:ex1], dtype=bool).copy()
    if patch.size <= 0 or mask.size <= 0:
        return 0.0
    ix0 = max(0, x0 - ex0)
    iy0 = max(0, y0 - ey0)
    ix1 = min(mask.shape[1], x1 - ex0)
    iy1 = min(mask.shape[0], y1 - ey0)
    if ix1 > ix0 and iy1 > iy0:
        mask[iy0:iy1, ix0:ix1] = False
    if not np.any(mask):
        return 0.0
    return _top_mean(patch[mask])


def _metrics_for_regions(
    *,
    regions: tuple[Any, ...],
    context: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    # Regions, candidates and bbox must all be in the generator's working plane.
    bbox = context.get("bbox")
    absdiff = context.get("absdiff")
    darkening = context.get("darkening")
    zscore = context.get("zscore")
    saliency = context.get("saliency")
    valid = context.get("valid")
    if (
        not isinstance(bbox, tuple)
        or len(bbox) != 4
        or not isinstance(absdiff, np.ndarray)
        or not isinstance(darkening, np.ndarray)
        or not isinstance(zscore, np.ndarray)
        or not isinstance(valid, np.ndarray)
    ):
        return {}
    if not isinstance(saliency, np.ndarray):
        saliency = absdiff

    full_x0, full_y0, _full_x1, _full_y1 = (int(v) for v in bbox)
    h, w = absdiff.shape[:2]
    metrics: dict[str, dict[str, float]] = {}

    for region in regions:
        if not _region_is_demo_target(region):
            continue
        object_id = str(getattr(region, "object_id", ""))
        if not object_id:
            continue

        rx0 = int(math.floor(float(getattr(region, "x", 0.0)) - full_x0))
        ry0 = int(math.floor(float(getattr(region, "y", 0.0)) - full_y0))
        rx1 = int(math.ceil(float(getattr(region, "x", 0.0)) + float(getattr(region, "width", 0.0)) - full_x0))
        ry1 = int(math.ceil(float(getattr(region, "y", 0.0)) + float(getattr(region, "height", 0.0)) - full_y0))
        rx0, ry0 = max(0, rx0), max(0, ry0)
        rx1, ry1 = min(w, rx1), min(h, ry1)
        if rx1 <= rx0 or ry1 <= ry0:
            continue

        rw = rx1 - rx0
        rh = ry1 - ry0
        inset_x = int(round(rw * INNER_INSET_FRACTION)) if rw >= 9 else 0
        inset_y = int(round(rh * INNER_INSET_FRACTION)) if rh >= 9 else 0
        ix0, iy0 = rx0 + inset_x, ry0 + inset_y
        ix1, iy1 = rx1 - inset_x, ry1 - inset_y
        if ix1 - ix0 < 2 or iy1 - iy0 < 2:
            ix0, iy0, ix1, iy1 = rx0, ry0, rx1, ry1

        expand_x = max(2, int(round(rw * RING_EXPAND_FRACTION)))
        expand_y = max(2, int(round(rh * RING_EXPAND_FRACTION)))
        ex0, ey0 = max(0, rx0 - expand_x), max(0, ry0 - expand_y)
        ex1, ey1 = min(w, rx1 + expand_x), min(h, ry1 + expand_y)

        # Central interior is intentionally favoured for the demo. It is less
        # contaminated by projected-object edge motion, while shooters naturally
        # aim near target centres. Full-region values remain a fallback for edge hits.
        inner_abs = _array_metric(absdiff, valid, ix0, iy0, ix1, iy1)
        full_abs = _array_metric(absdiff, valid, rx0, ry0, rx1, ry1)
        inner_dark = _array_metric(darkening, valid, ix0, iy0, ix1, iy1)
        full_dark = _array_metric(darkening, valid, rx0, ry0, rx1, ry1)
        inner_z = _array_metric(zscore, valid, ix0, iy0, ix1, iy1)
        full_z = _array_metric(zscore, valid, rx0, ry0, rx1, ry1)
        inner_sal = _array_metric(saliency, valid, ix0, iy0, ix1, iy1)

        abs_top = max(inner_abs, 0.86 * full_abs)
        dark_top = max(inner_dark, 0.86 * full_dark)
        z_top = max(inner_z, 0.86 * full_z)

        ring_abs = _ring_metric(
            absdiff, valid,
            x0=rx0, y0=ry0, x1=rx1, y1=ry1,
            ex0=ex0, ey0=ey0, ex1=ex1, ey1=ey1,
        )
        ring_dark = _ring_metric(
            darkening, valid,
            x0=rx0, y0=ry0, x1=rx1, y1=ry1,
            ex0=ex0, ey0=ey0, ex1=ex1, ey1=ey1,
        )

        abs_n = _clip01(math.tanh(max(0.0, abs_top) / 5.0))
        dark_n = _clip01(math.tanh(max(0.0, dark_top) / 3.5))
        z_n = _clip01(math.tanh(max(0.0, z_top) / 2.2))
        sal_n = _clip01(math.tanh(max(0.0, inner_sal) / 6.0))
        ring_n = _clip01(
            0.56 * math.tanh(max(0.0, ring_abs) / 5.5)
            + 0.44 * math.tanh(max(0.0, ring_dark) / 4.0)
        )

        candidate_support = 0.0
        cam_x0 = float(getattr(region, "x", 0.0))
        cam_y0 = float(getattr(region, "y", 0.0))
        cam_x1 = cam_x0 + float(getattr(region, "width", 0.0))
        cam_y1 = cam_y0 + float(getattr(region, "height", 0.0))
        for candidate in candidates:
            cx = _safe_float(candidate.get("camera_x"), -1e9)
            cy = _safe_float(candidate.get("camera_y"), -1e9)
            if cam_x0 <= cx <= cam_x1 and cam_y0 <= cy <= cam_y1:
                candidate_support = max(candidate_support, _candidate_quality(candidate))

        # Hole-like physical signal: darkening and compact local change matter
        # most. Candidate evidence is only a soft tie-breaker.
        physical_signal = _clip01(
            0.43 * dark_n
            + 0.34 * abs_n
            + 0.18 * z_n
            + 0.05 * sal_n
            - 0.16 * ring_n
        )
        frame_score = _clip01(
            0.34 * dark_n
            + 0.27 * abs_n
            + 0.16 * z_n
            + 0.07 * sal_n
            + 0.16 * candidate_support
            - 0.18 * ring_n
        )

        metrics[object_id] = {
            "frame_score": frame_score,
            "physical_signal": physical_signal,
            "abs_top": float(abs_top),
            "dark_top": float(dark_top),
            "z_top": float(z_top),
            "candidate_support": candidate_support,
            "ring": ring_n,
        }

    return metrics


def _regions_in_work_space(regions, scanner, shape):
    """Reuse V2.24.4's camera/crop/scale map for frozen camera HitRegions."""
    from src.engine.shot_object_local_v244 import map_camera_windows_to_work_v244
    mapping, _ = map_camera_windows_to_work_v244(scanner, shape, ())
    output = []
    for region in regions:
        x0, y0 = mapping.camera_to_work(region.x, region.y)
        x1, y1 = mapping.camera_to_work(region.x + region.width, region.y + region.height)
        output.append(replace(region, x=x0, y=y0, width=x1-x0, height=y1-y0))
    return tuple(output)


def patch_candidate_generator_for_demo_object_zone(cls: type) -> None:
    if bool(getattr(cls, "_demo_object_zone_extension_installed", False)):
        return

    original_generate = cls.generate

    def generate_wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_generate(self, *args, **kwargs)
        try:
            telemetry = dict(getattr(result, "telemetry", {}) or {})
            shot_id = int(_safe_float(telemetry.get("shot_id", 0), 0.0))
            if shot_id <= 0:
                return result

            snapshot = object_hit_registry_v2223.snapshot_for_shot(shot_id)
            if snapshot is None:
                return result
            regions = tuple(
                region
                for region in tuple(getattr(snapshot, "camera_regions", ()) or ())
                if _region_is_demo_target(region)
            )
            if not regions:
                return result

            context = getattr(self, "_v24_last_context", None)
            if not isinstance(context, Mapping):
                return result
            frame_ts = _safe_float(
                telemetry.get("frame_ts", context.get("frame_ts", kwargs.get("frame_ts", time.time()))),
                time.time(),
            )
            context_ts = _safe_float(context.get("frame_ts", frame_ts), frame_ts)
            if abs(context_ts - frame_ts) > 0.050:
                return result

            candidates = [dict(candidate) for candidate in list(getattr(result, "candidates", ()) or ())]
            scanner = kwargs.get("scanner", args[0] if args else None)
            gray = kwargs.get("gray", args[1] if len(args) > 1 else None)
            if scanner is None or not isinstance(gray, np.ndarray):
                return result
            # V2 maps/candidates are still crop-local here. The outer analysis
            # wrapper converts emitted candidates back to camera space later.
            # Subtracting only context.bbox from a full-camera region is wrong
            # whenever the analysis crop has an offset (or is downsampled).
            work_regions = _regions_in_work_space(regions, scanner, gray.shape)
            metrics = _metrics_for_regions(
                regions=work_regions,
                context=context,
                candidates=candidates,
            )
            if not metrics:
                return result

            zone_result = demo_object_zone_registry.observe_frame(
                shot_id=shot_id,
                peak_ts=float(getattr(snapshot, "peak_ts", frame_ts)),
                frame_ts=frame_ts,
                metrics=metrics,
            )
            if zone_result is not None and zone_result.evidence:
                # Debug only. Keep output terse enough for a live demo log.
                top = zone_result.evidence[0]
                print(
                    "[DEMO-OBJECT-ZONE] "
                    f"shot={shot_id} top={top.object_id} "
                    f"score={top.score:.3f} physical={top.physical_signal:.3f} "
                    f"conf={zone_result.confidence:.3f} margin={zone_result.margin:.3f} "
                    f"frames={zone_result.frames}/{zone_result.late_frames} "
                    f"accepted={zone_result.accepted} reason={zone_result.reason}"
                )
        except Exception as exc:
            # Absolute rule for the demo extension: never break canonical detection.
            print(f"[DEMO-OBJECT-ZONE] fallback: {exc}")
        return result

    cls.generate = generate_wrapped
    cls._demo_object_zone_extension_installed = True


def apply_demo_object_zone_extension() -> None:
    from src.engine.camera.candidate_generator_v2 import CandidateGeneratorV2

    patch_candidate_generator_for_demo_object_zone(CandidateGeneratorV2)
    print("[DEMO-OBJECT-ZONE] temporary shoot_stuff OBJECT detector installed")


__all__ = [
    "DemoObjectZoneEvidence",
    "DemoObjectZoneRegistry",
    "DemoObjectZoneResult",
    "apply_demo_object_zone_extension",
    "demo_object_zone_registry",
    "patch_candidate_generator_for_demo_object_zone",
]
